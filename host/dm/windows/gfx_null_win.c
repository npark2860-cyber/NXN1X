// Windows port, stage 2: a null graphics backend behind the replay interface of nvn_mtl.h (the Metal backend's functions), taken from the Android port's
// gfx_null.c and gfx_common.c and brought up to the current interface. Nothing is drawn. Everything the CPU can observe of the GPU's work is honoured:
//   · a submission completes at once: events are written / added, counters get {value, timestamp} (NVN's 614.4 MHz ticks from the guest clock, nvn_timestamp.h),
//     FenceSync objects take the submission's fence and every fence counts as passed;
//   · buffer copies, fills and inline uniform updates are done in order with plain memory operations (pool memory is the guest's own memory);
//   · samples-passed counters report many samples (everything visible) rather than 0 (everything culled).
// Presents are counted, not shown; there is no display link, so the host's own pacing applies (hle_nvn.c).
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include "nvn_capture.h"
#include "nvn_timestamp.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t guest_ns(void);
int nvn_game_fps(void);
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; } return nvn_timestamp_from_ns(g); }

// ---------------- memory pools: GPU address = CPU address ----------------
typedef struct { uint64_t lo, hi; } pool_ent;
static pool_ent *g_pools; static int g_npool, g_poolcap;
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
void mtl_pool_add(const void *mem, uint64_t size) {
    pthread_rwlock_wrlock(&g_pool_mu);
    if (g_npool == g_poolcap) { g_poolcap = g_poolcap ? 2 * g_poolcap : 256; g_pools = realloc(g_pools, g_poolcap * sizeof *g_pools); }
    g_pools[g_npool++] = (pool_ent){(uint64_t)(uintptr_t)mem, (uint64_t)(uintptr_t)mem + size};
    pthread_rwlock_unlock(&g_pool_mu);
}
void mtl_pool_del(const void *mem) {
    pthread_rwlock_wrlock(&g_pool_mu);
    for (int i = 0; i < g_npool; i++) if (g_pools[i].lo == (uint64_t)(uintptr_t)mem) { g_pools[i] = g_pools[--g_npool]; break; }
    pthread_rwlock_unlock(&g_pool_mu);
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int ok = 0;
    pthread_rwlock_rdlock(&g_pool_mu);
    for (int i = 0; i < g_npool && !ok; i++) if (addr >= g_pools[i].lo && addr + len <= g_pools[i].hi) { if (buf) *buf = (void *)(uintptr_t)g_pools[i].lo; if (off) *off = addr - g_pools[i].lo; ok = 1; }
    pthread_rwlock_unlock(&g_pool_mu);
    return ok;
}
void mtl_alias_drop_pool(const void *pool) { (void)pool; }
// r48 (rebase onto the newer upstream): the backend calls hle_nvn.c / hle_hid.c added since. This backend keeps its own GPU-write tracking (vk_gpu_wrote) and has no
// sparse tile heap, residency set or frame-step GPU clock ramp, so these keep upstream's contract without acting on it.
void mtl_pool_gpu_written(const void *pool) { pool_t *p = (pool_t *)pool; if (p) atomic_store_explicit(&p->gpu_written, 1, memory_order_relaxed); }   // the flag, as nvn_mtl.m sets it
void mtl_sp_phys_gone(const void *mem, uint64_t size) { (void)mem; (void)size; }
int mtl_probe_ramp_on(void) { const char *e = getenv("DM_FSTEP_PROBE_RAMP"); return e ? atoi(e) != 0 : 1; }   // nvn_mtl.m's default: on
void dm_gfx_restop(void) { LOG("DM_CTL restop: this backend has no residency set"); }

// ---------------- submissions ----------------
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // kind 1 event, 2 counter
static post_op *g_post; static int g_npost, g_postcap;
static _Atomic uint64_t g_fence;
static void post(post_op o) { if (g_npost == g_postcap) { g_postcap = g_postcap ? 2 * g_postcap : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); } g_post[g_npost++] = o; }

void mtl_init(void) {}
void mtl_prepare(void) {}
void mtl_sub_begin(void) { g_npost = 0; }
void mtl_queue_activate(uint64_t serial) { (void)serial; }
void mtl_queue_forget(uint64_t serial) { (void)serial; }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) { if (n) memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, n); }
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) { uint8_t *d = (uint8_t *)(uintptr_t)dst; for (uint64_t i = 0; i + 4 <= n; i += 4) memcpy(d + i, &value, 4); }
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type}); }

void mtl_exec(const uint8_t *p) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const uint8_t *a = p + 8;
        switch (h.op) {
        case OP_END: return;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); mtl_exec((const uint8_t *)(uintptr_t)t); break; }
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); mtl_sub_event(val, m, v); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); mtl_sub_counter(ad, (uint32_t)t); break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = atomic_load(&g_fence) + 1; s->fenced = 1; break; }
        case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); mtl_sub_copy(s_, d, n); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8); memcpy((void *)(uintptr_t)d, a + 16, n); break; }
        default: break;   // drawing, binding, texture copies and clears: no visible effect without a GPU
        }
        p += h.size;
    }
}
uint64_t mtl_sub_commit(void) {
    uint64_t now = gpu_ticks();
    for (int i = 0; i < g_npost; i++) {
        post_op *o = &g_post[i];
        if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
        else { uint64_t rec[2] = {o->type == 0 || o->type == 0x10 ? now : o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? (1u << 20) : 0, now}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }
    }
    g_npost = 0;
    return atomic_fetch_add(&g_fence, 1) + 1;
}
uint64_t mtl_last_fence(void) { return atomic_load(&g_fence); }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) { (void)fence; (void)timeout_ns; return 1; }
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { (void)fence; (void)timeout_ns; return 1; }
void mtl_finish(void) {}
void mtl_drain(void) {}

// ---------------- presentation: frames are counted, not shown ----------------
static _Atomic uint64_t g_presents; static double g_pres_t[64];
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    (void)t; (void)period; (void)crop;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t n = atomic_load(&g_presents) + 1; g_pres_t[n % 64] = ts.tv_sec + ts.tv_nsec / 1e9; atomic_store(&g_presents, n);
    return n;
}
void mtl_present_wait(uint64_t tk) { (void)tk; }
double mtl_presented_time(uint64_t n) { uint64_t p = atomic_load(&g_presents); return n && n <= p && p - n < 64 ? g_pres_t[n % 64] : 0; }
uint32_t mtl_present_qos(void) { return 0; }
int mtl_link_active(void) { return 0; }
int mtl_link_stalled(void) { return 0; }
int mtl_link_rate(void) { return 60; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
int mtl_snap_on(void) { return 0; }
uint64_t mtl_snap_acquire_wait(void) { return 0; }
int mtl_has_window(void) { return 0; }
int mtl_display_hz(void) { return 0; }   // as nvn_mtl.m without a window
int mtl_fps_target(void) {   // as nvn_mtl.m: DM_FPS=30/60 pins it, otherwise the game's own rate
    static int env = -2; if (env == -2) { const char *e = getenv("DM_FPS"); env = e ? atoi(e) : -1; if (env != 30 && env != 60) env = -1; }
    return env > 0 ? env : nvn_game_fps() == 30 ? 30 : 60;
}
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_set_frame(uint64_t f) { (void)f; }
void mtl_set_depth_zero(int on) { (void)on; }
void mtl_set_origin_lower_left(int on) { (void)on; }
void mtl_set_queue_origin_lower_left(int on) { (void)on; }

// ---------------- resources ----------------
void mtl_tex_init(const tex_t *t) { (void)t; }
void mtl_tex_release(tex_t *t) { (void)t; }
void mtl_tex_cpu_written(const tex_t *t) { (void)t; }
unsigned mtl_cpu_tex_count(void) { return 0; }
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) { (void)tid; (void)t; (void)v; }
void mtl_reg_image(int tid, const tex_t *t, const view_t *v) { (void)tid; (void)t; (void)v; }
void mtl_reg_samp(int sid, const samp_par *sp) { (void)sid; (void)sp; }
void mtl_set_tex_pool_n(int n) { (void)n; }
void mtl_prog_release(prog_t *p) { (void)p; }
void mtl_prog_forget(const prog_t *p) { (void)p; }
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
int mtl_pso_prebuild_tool(void) { return 0; }   // DM_PSO_PREBUILD=tool has nothing to build here: the game runs
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    (void)t; (void)level; (void)layer0;
    if (r && dst) for (int z = 0; z < (r->depth ? r->depth : 1); z++) for (int y = 0; y < r->height; y++) memset(dst + (size_t)z * img + (size_t)y * row, 0, row);
}

// ---------------- statistics and diagnostics of the Metal backend: nothing to report ----------------
void mtl_vbuf_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e, uint64_t *f) { *a = *b = *c = *d = *e = *f = 0; }
void mtl_res_counts(uint32_t tick, const char *why) { (void)tick; (void)why; }
void mtl_res_snapshot(mtl_res_count *out) { memset(out, 0, sizeof *out); }
void mtl_alloc_stats(uint64_t *w, uint64_t *g, int *n) { *w = *g = 0; *n = 0; }
void mtl_args_stats(uint64_t *e, uint64_t *r) { *e = *r = 0; }
void mtl_barrier_stats(uint64_t *s, uint64_t *f) { *s = *f = 0; }
uint64_t mtl_blit_splits(void) { return 0; }
void mtl_bw(uint64_t *l, uint64_t *s, uint64_t *f) { *l = *s = *f = 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
void mtl_create_ns(uint64_t *l, uint64_t *p) { *l = *p = 0; }
void mtl_create_stats(uint64_t *l, uint64_t *p) { *l = *p = 0; }
void mtl_cshoist_stats(uint64_t *m, uint64_t *d, uint64_t *s) { *m = *d = 0; memset(s, 0, 8 * sizeof *s); }
void mtl_geohoist_stats(uint64_t *m, uint64_t *r, uint64_t *s) { *m = *r = 0; memset(s, 0, 6 * sizeof *s); }
void mtl_geomesh_stats(uint64_t *n, uint64_t *w) { *n = *w = 0; }
void mtl_fb_stats(uint64_t *s, uint64_t *r, uint64_t *t, uint64_t *w) { *s = *r = *t = *w = 0; }
void mtl_hybrid_stats(uint64_t *d, uint64_t *m, uint64_t *n) { *d = *m = *n = 0; }
void mtl_sm_stats(uint64_t *p, uint64_t *c, uint64_t *w) { *p = *c = *w = 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
double mtl_gpu_ms(uint64_t f) { (void)f; return 0; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
void mtl_lc_stats(uint64_t *t, uint64_t *l) { *t = *l = 0; }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) { (void)frame; (void)rss_mb; }
uint64_t mtl_pkg_miss(void) { return 0; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
void mtl_pkg_stats(uint64_t *u, uint64_t *r) { *u = *r = 0; }
void mtl_pso_line(char *o, size_t n) { if (n) o[0] = 0; }
void mtl_res_stats(char *out, size_t n) { if (n) out[0] = 0; }
void mtl_stats(uint64_t *p, uint64_t *d) { *p = *d = 0; }
void mtl_ub_present(double iv) { (void)iv; }
void mtl_ubd_stats(uint64_t *d, uint64_t *c) { *d = *c = 0; }
void mtl_ubp_stats(uint64_t *m, uint64_t *r, uint64_t *f) { *m = *r = *f = 0; }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_upload_ns_take(void) { return 0; }
void mtl_use_stats(uint64_t *c, uint64_t *s) { *c = *s = 0; }
void dm_gfx_passdump_next(long n) { (void)n; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_texlist(int target) { (void)target; }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }
// DM_NVN_CAPTURE reads the Metal backend's tables (nvn_capture.inc): not available
void nvn_cap_submit(int n, const uint64_t *handles) { (void)n; (void)handles; }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; }
