// Windows port, stage 3: what the Vulkan backend (gfx_vk_*.c, from the Android port) lacks of the current host interface. The statistics and Metal-only hooks are
// gfx_null_win.c's (nothing to count or do here); the rest joins the Android port's names to the current host's.
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include "nvn_capture.h"
#include "overlay_layout.h"
#include <string.h>
#include <stdlib.h>

// SyncWait on an acquire sync (hle_nvn.c, fenced 2): the same fence wait; the ledger nvn_mtl.m keeps for it does not exist here
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
// The Android port's hle_nvn.c counted discards and handed SetShaders to the bundle-miss capture (gfx_vk_shader.c shader_set); the current host has neither hook, so the counts stay 0
// and a shader missing from the bundle is logged without being captured
uint64_t g_nvn_discards[2];
void (*g_nvn_on_shader)(const uint8_t *, const void *, uint64_t, uint32_t);
// The Android port's nvn_dirty.c let the texture sweep give up a packaged texture's write protection and remapped dma-buf pools in one piece (the kernel's mapping limit).
// The current nvn_dirty.c has no release: the pages stay protected until the game writes them (its fault handler unprotects and stamps them), which is correct, only slower
// for a texture the game rewrites after the sweep let it go. Nothing is remapped (Windows pools are the game's pages, imported)
void nvn_dirty_release(const void *addr, size_t len) { (void)addr; (void)len; }
uint64_t nvn_dirty_released(void) { return 0; }
int nvn_dirty_runs(void) { return 0; }
int nvn_dirty_defrag(int (*remap)(void *mem, size_t len)) { (void)remap; return 0; }
// The on-screen gamepad's theme (the Android app's assets): none, so the overlay, if shown, draws without one
const ov_theme *dm_overlay_theme(int layout) { (void)layout; return NULL; }
void dm_overlay_theme_done(const ov_theme *t) { (void)t; }

// ---- from gfx_null_win.c ----
void dm_gfx_passdump_next(long n) { (void)n; }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; }
void dm_gfx_texdump_next(const char *path) { (void)path; }
void dm_gfx_texlist(int target) { (void)target; }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; }
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
void mtl_alloc_stats(uint64_t *w, uint64_t *g, int *n) { *w = *g = 0; *n = 0; }
void mtl_args_stats(uint64_t *e, uint64_t *r) { *e = *r = 0; }
void mtl_barrier_stats(uint64_t *s, uint64_t *f) { *s = *f = 0; }
uint64_t mtl_blit_splits(void) { return 0; }
void mtl_bw(uint64_t *l, uint64_t *s, uint64_t *f) { *l = *s = *f = 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
unsigned mtl_cpu_tex_count(void) { return 0; }
// mtl_create_ns / mtl_create_stats: gfx_vk_exec.c (r80, pipelines made on the replay thread)
void mtl_cshoist_stats(uint64_t *m, uint64_t *d, uint64_t *s) { *m = *d = 0; memset(s, 0, 8 * sizeof *s); }
uint64_t mtl_cut_count(void) { return 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
void mtl_fb_stats(uint64_t *s, uint64_t *r, uint64_t *t, uint64_t *w) { *s = *r = *t = *w = 0; }
void mtl_geohoist_stats(uint64_t *m, uint64_t *r, uint64_t *s) { *m = *r = 0; memset(s, 0, 6 * sizeof *s); }
void mtl_geomesh_stats(uint64_t *n, uint64_t *w) { *n = *w = 0; }
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void mtl_hybrid_stats(uint64_t *d, uint64_t *m, uint64_t *n) { *d = *m = *n = 0; }
void mtl_lc_stats(uint64_t *t, uint64_t *l) { *t = *l = 0; }
// r99: mtl_link_rate / mtl_link_stalled / mtl_link_wait_ticks: gfx_vk_present.c (display link)
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) { (void)frame; (void)rss_mb; }
uint64_t mtl_pkg_edge_skips(void) { return 0; }
uint64_t mtl_pkg_miss(void) { return 0; }
void mtl_pkg_stats(uint64_t *u, uint64_t *r) { *u = *r = 0; }
void mtl_prepare(void) {}
// mtl_prog_forget: gfx_vk_exec.c (a finalized program leaves every queue's bindings)
// mtl_prog_prefetch: gfx_vk_shader.c (NXN1X lane B: each shader a program sets goes to Eden there, while its control section is valid)
void mtl_pso_line(char *o, size_t n) { if (n) o[0] = 0; }
int mtl_pso_prebuild_tool(void) { return 0; }   // DM_PSO_PREBUILD=tool has nothing to build here: the game runs
// mtl_queue_activate / mtl_queue_forget: gfx_vk_exec.c (r72, per-queue replay state)
// r70: RegisterImage writes the same table as RegisterTexture (nvn_mtl.m mtl_reg_image). It was empty here, so every image handle missed and storage writes went to the
// dummy view: Bayonetta 3's FilterShaderResolveWithHalf half-size scene copy stayed black (Rodin's liquor, PS_DISTORTION). DM_VK_REG_IMAGE=0 (at start): empty as before
void mtl_reg_image(int tid, const tex_t *t, const view_t *v) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_REG_IMAGE"); on = !(e && *e == '0'); }
    if (!on) return;
    if (!t) { mtl_reg_tex(tid, NULL, v); return; }
    tex_t image = *t; image.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
    mtl_reg_tex(tid, &image, v);
}
void mtl_res_stats(char *out, size_t n) { if (n) out[0] = 0; }
void mtl_sm_stats(uint64_t *p, uint64_t *c, uint64_t *w) { *p = *c = *w = 0; }
uint64_t mtl_snap_acquire_wait(void) { return 0; }
int mtl_snap_on(void) { return 0; }
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
void mtl_vbuf_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e, uint64_t *f) { *a = *b = *c = *d = *e = *f = 0; }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; }
void nvn_cap_submit(int n, const uint64_t *handles) { (void)n; (void)handles; }
