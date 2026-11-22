// Write-protect dirty-page tracking for guest memory: pool pages of packaged textures (CPU writes, GPU reads) are set read-only after upload, and as soon as the game writes it faults -> mark dirty, unprotect;
// before use, check the write generation of the covered pages and re-upload if newer than at upload time. NVN semantics: a texture descriptor points to pool memory, and the GPU reads the current content every time
// (measured: the game writes streamed assets into texture storage that is still being sampled, without re-registering).
// Signal handler constraint: the page-fault path uses only atomic operations and mprotect, no allocation, no locks.
#include "nvn_dirty.h"
#if defined(DM_NXN1X)
#include "../../nxn1x/nvnh_api.h"
#endif
#include "../../host/dm.h"
#include <stdatomic.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <mach/mach_time.h>
#include <sched.h>

// Order of page state and mprotect: the side changing protection first sets the state to BUSY and sets the final state only after mprotect has landed; other threads wait when they see BUSY.
//   set read-only: NONE -> BUSY -> mprotect(read-only) -> PROT; unprotect: PROT -> BUSY -> mprotect(writable) -> NONE.
// Invariant: a page whose state is NONE is always writable, one whose state is PROT is always read-only. Setting read-only used to be "record PROT, then mprotect"; in between, another thread's unprotect-before-write
// (the movie decoder thread writing a frame into texture storage, a disk read into a guest buffer) could set the state back to NONE, then the mprotect landed => the page read-only with state NONE: a game thread writing it got only
// "re-execute" from the fault handler and spun forever; a host system call reading into it saw NONE in prep_write, skipped, and failed with EFAULT (reproduced by t_dirty_race).
// BUSY lasts at most one mprotect (plus a few atomic writes when setting read-only). The wait is capped at BUSY_WAIT_MS and reports red beyond it: in the fault handler it then returns "not our page" so the crash handler reports truthfully
enum { PG_NONE = 0, PG_PROT = 1, PG_BUSY = 2 };
enum { BUSY_WAIT_MS = 2000 };
// Tracked ranges = the memory pools (registered by MemoryPoolInitialize). After a pool is destroyed its live=0 slot is reused; the signal handler scans without locks.
// The table grows in blocks of 1,024 slots, up to 262,144; a block is never moved or freed, so the signal handler reads it without locks. A pool past the end goes untracked: its textures count
// as dirty and are re-uploaded at every use. The table was a fixed 256 slots (Zelda has about 300 pools at the title), then a fixed 4,096, and Breath of the Wild reaches 4,006 pools in its first
// shrine. The scans only cover slots in use, so the capacity itself costs nothing
enum { BLK_SHIFT = 10, BLK = 1 << BLK_SHIFT, NBLK = 256, MAX_RANGE = BLK * NBLK };
// Ranges are rounded to host pages (16 KB) but guest memory pools are only 4 KB aligned: adjacent pools share their first / last pages, so one address can fall in several registered ranges.
// So: "is this texture tracked" looks for a range that holds its whole span; a fault / host pre-write unprotects and stamps every range covering that page; pool destruction identifies its range by the pool's exact start mem.
// (Previously the first range containing the address was always taken: two SMBW ASTC 6x5 textures whose pool's first page is shared with a small preceding pool were judged untracked and re-uploaded on every use, about 6 MB per frame)
// stage2a2: g1 / g2 = the highest generation of any page in each group of 64 / 4,096 pages (raised by stamp_page after the page's own generation), so that
// nvn_dirty_each_since skips whole clean groups: the copied pools' sync looked at every 4 KB page of every pool (HWAoI: about 600,000 a submission, 1,400 dirty)
typedef struct { uintptr_t base, end, mem; _Atomic uint8_t *state; _Atomic uint64_t *gen, *g1, *g2, *fgen; _Atomic(_Atomic uint8_t *) sparse; size_t cap; _Atomic int live; } range_t;   // sparse: per page, protected for a placement sparse tile (nvn_dirty_protect_sparse)
enum { G1_SHIFT = 6, G2_SHIFT = 12 };
static _Atomic(range_t *) g_blk[NBLK];
static inline range_t *rg(int i) { return &atomic_load_explicit(&g_blk[i >> BLK_SHIFT], memory_order_acquire)[i & (BLK - 1)]; }
static _Atomic int g_nr;
static uintptr_t g_pg, g_shift;
static _Atomic uint64_t g_now = 1;
static _Atomic uint64_t g_faults, g_preps;
static _Atomic uint64_t g_adds_total;
static _Atomic int g_live, g_live_peak;   // number of registered pools and peak (for the memory readout line)
void nvn_dirty_stat(int *live, int *peak, int *cap) { *live = atomic_load(&g_live); *peak = atomic_load(&g_live_peak); *cap = MAX_RANGE; }
static _Atomic int g_overflow_logged;
static pthread_mutex_t g_slot_mu = PTHREAD_MUTEX_INITIALIZER;   // Pool creation, retirement and protection serialize descriptor lifetime; the page-fault path does not touch this mutex
static _Atomic uint64_t g_mp_calls, g_mp_pages, g_mp_ticks;   // This module's mprotect: call count, pages covered, time spent (mach timebase, safe in the page-fault path too)
static void mp(uintptr_t a, size_t len, int prot) {
    uint64_t t = mach_absolute_time(); mprotect((void *)a, len, prot);
    atomic_fetch_add(&g_mp_ticks, mach_absolute_time() - t); atomic_fetch_add(&g_mp_calls, 1); atomic_fetch_add(&g_mp_pages, len >> g_shift);
}

static mach_timebase_info_data_t g_tb;   // obtained when the first pool is registered (the fault path only multiplies and divides)
static _Atomic int g_busy_red;
static _Atomic uintptr_t g_busy_red_page;   // nceresume R5: the page busy_wait gave up on, printed later by nvn_dirty_protect (host code, never the fault path)
// Wait for this page to leave BUSY. Returns 1 = it has left; 0 = waited past the limit (reported red once). Usable in a signal handler: only atomic reads, mach_absolute_time and sched_yield
static int busy_wait(range_t *r, uintptr_t p) {
    uint64_t t0 = 0;
    for (unsigned n = 1; atomic_load(&r->state[p]) == PG_BUSY; n++) {
        if (n & 0x3ff) continue;
        uint64_t now = mach_absolute_time(); if (!t0) t0 = now;
        if ((now - t0) * g_tb.numer / g_tb.denom > (uint64_t)BUSY_WAIT_MS * 1000000ull) {
            if (!atomic_exchange(&g_busy_red, 1)) atomic_store(&g_busy_red_page, r->base + (p << g_shift));   // nceresume R5: no log here (this runs in the fault handler); nvn_dirty_protect prints it
            return 0; }
        sched_yield();
    }
    return 1;
}
static int live_at(int i) { return atomic_load_explicit(&rg(i)->live, memory_order_acquire); }
// The registered range that holds all of [a, a+len) (used for a texture's tracking / protection / dirty check)
// s56b2: the live ranges' slots sorted by base (under g_slot_mu: add/remove keep it; nvn_dirty_gpu_write searches it), so that lookup is a binary search
static int *g_sidx, g_sidx_n, g_sidx_cap;
static void sidx_del_locked(int slot) { for (int k = 0; k < g_sidx_n; k++) if (g_sidx[k] == slot) { memmove(&g_sidx[k], &g_sidx[k + 1], (size_t)(g_sidx_n - k - 1) * sizeof *g_sidx); g_sidx_n--; return; } }
static void sidx_add_locked(int slot, uintptr_t base);
static _Atomic int g_span_last;   // the last range found: looks come in runs over one pool, and a scan of Mario Kart World's 5,000 pools per look took a quarter of its submit thread's time
static range_t *find_span(uintptr_t a, size_t len) {
    int n = atomic_load(&g_nr), k = atomic_load_explicit(&g_span_last, memory_order_relaxed);
    if (k < n) { range_t *r = rg(k); if (live_at(k) && a >= r->base && a + len <= r->end) return r; }
    for (int i = 0; i < n; i++) { range_t *r = rg(i); if (live_at(i) && a >= r->base && a + len <= r->end) { atomic_store_explicit(&g_span_last, i, memory_order_relaxed); return r; } }
    return NULL;
}
static inline void raise_max(_Atomic uint64_t *x, uint64_t v) { uint64_t c = atomic_load(x); while (c < v && !atomic_compare_exchange_weak(x, &c, v)) {} }
static void stamp_page(range_t *r, uintptr_t p) {
    const uint64_t g = atomic_fetch_add(&g_now, 1) + 1;
    atomic_store(&r->gen[p], g);
    raise_max(&r->g1[p >> G1_SHIFT], g); raise_max(&r->g2[p >> G2_SHIFT], g);   // stage2a2: after the page (a reader that sees the group raised finds the page too)
    // Publish completion after the generation: a checker may have cached "clean" using the incremented clock before the generation store. Both fault and host-write paths
    // must invalidate that cached clock, as the page stays writable until the next upload protects it again.
    atomic_fetch_add(&g_now, 1);
}
static _Atomic uint64_t g_sparse_writes;   // writes to pages protected for a placement sparse tile (nvn_dirty_sparse_writes)
static void sparse_note(range_t *r, uintptr_t p) { _Atomic uint8_t *sp = atomic_load(&r->sparse); if (sp && atomic_exchange(&sp[p], 0)) atomic_fetch_add(&g_sparse_writes, 1); }
// The snapshot hook (nvn_dirty_set_snap_hook; the Windows Vulkan backend's copied pools keep the content a page had when the CPU first wrote it after a sync)
static void (*_Atomic g_snap)(uintptr_t, size_t, uint64_t);
void nvn_dirty_set_snap_hook(void (*fn)(uintptr_t page, size_t len, uint64_t now)) { atomic_store_explicit(&g_snap, fn, memory_order_release); }
static inline void snap_page(range_t *r, uintptr_t p) { void (*f)(uintptr_t, size_t, uint64_t) = atomic_load_explicit(&g_snap, memory_order_acquire); if (f) f(r->base + (p << g_shift), g_pg, atomic_load(&g_now)); }
static void unprotect_page(range_t *r, uintptr_t p) {   // The caller has already changed the state from PROT to BUSY
    snap_page(r, p);
    mp(r->base + (p << g_shift), g_pg, PROT_READ | PROT_WRITE);
    stamp_page(r, p); sparse_note(r, p);
    atomic_store(&r->state[p], PG_NONE);
}
// Unprotect one page of a range: if protected, unprotect and stamp it; if another thread is changing its protection, wait for it to finish and look again -- if the other side was setting read-only, the page is PROT after the wait and still has to be unprotected
// (an unprotect-before-write never reaches the fault handler, so it must be done here). Returns 1 = this page is managed by this module; 0 = it is not, or the wait on BUSY timed out
static int release_page(range_t *r, uintptr_t p, _Atomic uint64_t *count) {
    for (;;) {
        uint8_t st = PG_PROT;
        if (atomic_compare_exchange_strong(&r->state[p], &st, PG_BUSY)) { unprotect_page(r, p); if (count) atomic_fetch_add(count, 1); return 1; }
        if (st != PG_BUSY) return atomic_load(&r->gen[p]) != 0;   // already NONE: just unprotected by another thread (the fault happened before the unprotect) -> re-execute; pages never protected are not ours
        if (!busy_wait(r, p)) return 0;
    }
}
static _Atomic uint64_t g_edges;   // s28: pages beside a write reported at a page that was not protected, released
uint64_t nvn_dirty_edges(void) { return atomic_load(&g_edges); }
static int release_at(uintptr_t a, int *was_prot) {
    int ours = 0, n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++)   // Every range sharing the page must be released: releasing only one leaves the other range's state at "protected", and later writes no longer fault and are missed
        { range_t *r = rg(i); if (live_at(i) && a >= r->base && a < r->end) { uintptr_t p = (a - r->base) >> g_shift; if (was_prot && atomic_load(&r->state[p]) != PG_NONE) *was_prot = 1;
            ours |= release_page(r, p, &g_faults); } }
    return ours;
}
// stage2b3: the page holding a took a write fault: its fault generation is the generation its unprotect stamped (atomics only: the fault path)
static void mark_fault(uintptr_t a) {
    int n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) { range_t *r = rg(i); if (!live_at(i) || a < r->base || a >= r->end || !r->fgen) continue;
        uintptr_t p = (a - r->base) >> g_shift; atomic_store(&r->fgen[p], atomic_load(&r->gen[p])); }
}
// combined1120: sequential fault-ahead (NXN1X_PC_FAULTAHEAD, default 1; 0 off). HWAoI moving: 20-35k write faults/s, each one fault plus one VirtualProtect, and
// the next sync's protect again. When page P faults and P-1 already faulted since the last sync (g_sync_mark, set by the copied pools' sync), the write is a
// stream: P+1..P+15 of the same range (to its end; protected pages only) are released in one call, stamped as written: they are compared and uploaded at the
// next sync as any dirty page (a GPU write on them not merged yet still takes the exact runs: gfx_vk_core.c pc_wfence). Other ranges over the same pages (aliases)
// keep their protection. Counters: faults that followed their page's neighbour, pages released ahead, time in this handler
static _Atomic uint64_t g_sync_mark, g_seq_faults, g_fa_pages, g_fault_ticks, g_fault_n;
static int g_fa_on = -1;
void nvn_dirty_set_sync_mark(uint64_t gen) { atomic_store(&g_sync_mark, gen); }
void nvn_dirty_fa_stats(uint64_t *seq, uint64_t *ahead, uint64_t *faults, uint64_t *ns) {
    *seq = atomic_load(&g_seq_faults); *ahead = atomic_load(&g_fa_pages); *faults = atomic_load(&g_fault_n);
    *ns = g_tb.denom ? atomic_load(&g_fault_ticks) * g_tb.numer / g_tb.denom : 0;
}
static void release_pages(range_t *r, uintptr_t p, uintptr_t q, _Atomic uint64_t *count);
static void fault_ahead(uintptr_t a) {
    const uint64_t mark = atomic_load(&g_sync_mark); int n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) { range_t *r = rg(i); if (!live_at(i) || a < r->base || a >= r->end || !r->fgen) continue;
        const uintptr_t p = (a - r->base) >> g_shift; if (!p || atomic_load(&r->fgen[p - 1]) <= mark) continue;
        atomic_fetch_add(&g_seq_faults, 1);
        if (g_fa_on <= 0) continue;
        const uintptr_t last = ((r->end - r->base) >> g_shift) - 1, q = p + 15 < last ? p + 15 : last;
        if (q > p) release_pages(r, p + 1, q, &g_fa_pages); }
}
static int on_fault(void *addr, int is_write, void *uc) {
    (void)uc;
    if (!is_write) return 0;
    const uint64_t t0 = mach_absolute_time();
    uintptr_t a = (uintptr_t)addr; int prot = 0, ours = release_at(a, &prot);
    if (ours) { mark_fault(a); fault_ahead(a); }   // stage2b3: this page took a write fault (the copied pools' volatile rule counts these); combined1120: fault-ahead
    // s28: "ours" for a page that was not protected means "run the write again" (another thread released it a moment ago). A store that reaches into the next
    // page (a pair or vector store across the boundary) can be reported at its first byte: run again, it faults at the same writable page again and again, the
    // way a guest thread of Xenoblade 3 did in run 32 (11-12) once a copied pool was protected. So the pages on either side are released as well (released
    // pages count as written: copied at the next sync, nothing missed)
    if (ours && !prot) { uintptr_t pg = a & ~(g_pg - 1); int e = release_at(pg + g_pg, NULL); if (pg >= g_pg) e |= release_at(pg - 1, NULL); if (e) atomic_fetch_add(&g_edges, 1); }
#if defined(DM_NXN1X)
    if (ours) {
        uintptr_t pg = a & ~(g_pg - 1);
        unsigned origin = nvnc_origin() == NVNC_GPU_WRITEBACK ? NVNC_GPU_WRITEBACK : NVNC_CPU_FAULT;
        nvnc_cpu(pg, g_pg, origin); // before re-execution, never a completed store
        if (!prot) { nvnc_cpu(pg + g_pg, g_pg, origin); if (pg >= g_pg) nvnc_cpu(pg - g_pg, g_pg, origin); }
    }
#endif
    if (ours) { atomic_fetch_add(&g_fault_ticks, mach_absolute_time() - t0); atomic_fetch_add(&g_fault_n, 1); }
    return ours;
}
// Unprotect pages p..q of a range with one mprotect per run of protected pages (a host write over a whole texture, e.g. WriteTexels, used to make one system call per 16 KB page:
// Link's Awakening, about 275 a frame). Pages another thread is changing are handled one by one after the run so far is finished, so no BUSY page is held while waiting
// Stamp-unprotected mode (nvn_dirty_stamp_unprotected): a host write also stamps pages that are not protected, for a caller that keeps textures unprotected (Windows: VirtualProtect cost about 1 ms a page,
// so packaged textures are not protected there; a host write still marks them dirty, a guest CPU write does not)
static _Atomic int g_stamp_none;
static _Atomic uint64_t g_stamped;
void nvn_dirty_stamp_unprotected(int on) { atomic_store(&g_stamp_none, on); }
uint64_t nvn_dirty_stamped(void) { return atomic_load(&g_stamped); }
static void release_pages(range_t *r, uintptr_t p, uintptr_t q, _Atomic uint64_t *count) {
    uintptr_t run = 0; int open = 0;
    for (uintptr_t i = p; i <= q + 1; i++) {
        uint8_t st = PG_PROT; int took = i <= q && atomic_compare_exchange_strong(&r->state[i], &st, PG_BUSY);
        if (took) { if (!open) { run = i; open = 1; } continue; }
        if (open) {
            for (uintptr_t k = run; k < i; k++) snap_page(r, k);
            mp(r->base + (run << g_shift), (i - run) << g_shift, PROT_READ | PROT_WRITE);
            for (uintptr_t k = run; k < i; k++) { stamp_page(r, k); sparse_note(r, k); atomic_store(&r->state[k], PG_NONE); }
            if (count) atomic_fetch_add(count, i - run);
            open = 0;
        }
        if (i <= q && st == PG_BUSY) release_page(r, i, count);   // another thread is changing it: wait, then unprotect it if it ended up protected
        if (i <= q && atomic_load_explicit(&g_stamp_none, memory_order_relaxed) && atomic_load(&r->state[i]) == PG_NONE) { stamp_page(r, i); atomic_fetch_add(&g_stamped, 1); }
    }
}
// dirtyidx (11-21, OoT replay2/occkey replaydiag: the replay-time report batches cost 4.4 ms a frame, 6.6 us per page run, nearly all of it here): every host
// pre-write walked every slot in use (g_nr: 5,242 pools in OoT). It now looks in s56b2's index of the live ranges sorted by base (g_sidx, under g_slot_mu)
// with a prefix maximum of their ends (g_spmax): a binary search for the first range starting at or past the end of the write, then back while some range
// before can still reach the write. Every range covering a byte of the write is still released, as before (adjacent pools share their edge pages). The fault
// path (release_at, a signal handler) keeps its scan: it takes no lock
static uintptr_t *g_spmax; static int g_spmax_cap;
static void sidx_pmax_rebuild(void) {
    if (g_spmax_cap < 0) return;   // (sticky: the index missed a range once)
    if (g_spmax_cap < g_sidx_cap) { uintptr_t *x = realloc(g_spmax, (size_t)g_sidx_cap * sizeof *x); if (!x) { g_spmax_cap = -1; return; } g_spmax = x; g_spmax_cap = g_sidx_cap; }
    uintptr_t m = 0; for (int k = 0; k < g_sidx_n; k++) { const uintptr_t e = rg(g_sidx[k])->end; if (e > m) m = e; g_spmax[k] = m; }
}
static void prepare_write_range_locked(uintptr_t a, uintptr_t e) {   // g_slot_mu held
    if (g_spmax_cap < 0 || g_spmax_cap < g_sidx_n) {   // (no index memory: the scan as before)
        int n = atomic_load(&g_nr);
        for (int i = 0; i < n; i++) { range_t *r = rg(i); if (!live_at(i) || e <= r->base || a >= r->end) continue;
            uintptr_t lo = a > r->base ? a : r->base, hi = e < r->end ? e : r->end; release_pages(r, (lo - r->base) >> g_shift, (hi - 1 - r->base) >> g_shift, &g_preps); }
        return;
    }
    int lo = 0, hi = g_sidx_n; while (lo < hi) { int m = (lo + hi) / 2; if (rg(g_sidx[m])->base < e) lo = m + 1; else hi = m; }
    for (int k = lo - 1; k >= 0 && g_spmax[k] > a; k--) {
        range_t *r = rg(g_sidx[k]);
        if (!atomic_load_explicit(&r->live, memory_order_acquire) || a >= r->end) continue;
        uintptr_t l = a > r->base ? a : r->base, h = e < r->end ? e : r->end;
        release_pages(r, (l - r->base) >> g_shift, (h - 1 - r->base) >> g_shift, &g_preps);
    }
}
static void prepare_write_range(void *addr, size_t len) {
    uintptr_t a = (uintptr_t)addr, e = a + len;
    if (!len) return;
    pthread_mutex_lock(&g_slot_mu); prepare_write_range_locked(a, e); pthread_mutex_unlock(&g_slot_mu);
}

static void prep_write(void *addr, size_t len) {
#if defined(DM_NXN1X)
    nvnc_cpu((uint64_t)(uintptr_t)addr, len, nvnc_origin());
#endif
    prepare_write_range(addr, len);
}

void nvn_dirty_add_range(const void *mem, size_t size) {
    static int inited;
    if (!inited) { inited = 1; g_pg = (uintptr_t)getpagesize(); g_shift = __builtin_ctzl(g_pg); mach_timebase_info(&g_tb);
        { const char *e = getenv("NXN1X_PC_FAULTAHEAD"); g_fa_on = !(e && *e == '0'); LOG("[dm] combined1120: sequential fault-ahead %s (NXN1X_PC_FAULTAHEAD)", g_fa_on ? "on" : "off"); } dm_fault_hook_register(on_fault); dm_guest_write_prep_register(prep_write); }
    atomic_fetch_add(&g_adds_total, 1);
    pthread_mutex_lock(&g_slot_mu);
    int n = atomic_load(&g_nr);
    int slot = -1;
    for (int i = 0; i < n; i++) if (!live_at(i)) { slot = i; break; }
    if (slot < 0) {
        if (n >= MAX_RANGE) {
            int expected = 0;
            if (atomic_compare_exchange_strong(&g_overflow_logged, &expected, 1)) LOG("[RED] dirty-page tracking ranges are full (%d), pool %p not tracked", MAX_RANGE, mem);
            pthread_mutex_unlock(&g_slot_mu); return;
        }
        slot = n;
        if (!atomic_load_explicit(&g_blk[n >> BLK_SHIFT], memory_order_acquire)) {   // published before g_nr counts the slot (the scans read g_nr first)
            range_t *b = calloc(BLK, sizeof *b); if (!b) die("dirty-page tracking range table allocation failed (block %d)", n >> BLK_SHIFT);
            atomic_store_explicit(&g_blk[n >> BLK_SHIFT], b, memory_order_release);
        }
    }
    { int lv = atomic_fetch_add(&g_live, 1) + 1, pk = atomic_load(&g_live_peak); while (lv > pk && !atomic_compare_exchange_weak(&g_live_peak, &pk, lv)) {} }
    range_t *r = rg(slot);
    uintptr_t base = (uintptr_t)mem & ~(g_pg - 1), end = ((uintptr_t)mem + size + g_pg - 1) & ~(g_pg - 1);
    size_t np = (end - base) >> g_shift;
    _Atomic uint8_t *state = r->state;
    _Atomic uint64_t *gen = r->gen, *g1 = r->g1, *g2 = r->g2, *fgen = r->fgen;
    const size_t n1 = (np >> G1_SHIFT) + 1, n2 = (np >> G2_SHIFT) + 1;   // stage2a2
    if (r->cap < np) {
        state = calloc(np, 1); gen = calloc(np, sizeof *gen); g1 = calloc(n1, sizeof *g1); g2 = calloc(n2, sizeof *g2); fgen = calloc(np, sizeof *fgen);
        if (!state || !gen || !g1 || !g2 || !fgen) die("dirty-page table allocation failed (%zu pages)", np);
        // The old array is not freed: the signal handler may still have a very brief stale reader; leak = 9 bytes per page.
        atomic_store(&r->sparse, NULL);   // sized for the old cap, not freed for the same reason
        r->cap = np;
    } else {
        memset((void *)state, 0, np); memset((void *)gen, 0, np * sizeof *gen); memset((void *)g1, 0, n1 * sizeof *g1); memset((void *)g2, 0, n2 * sizeof *g2); memset((void *)fgen, 0, np * sizeof *fgen);
        _Atomic uint8_t *sp = atomic_load(&r->sparse); if (sp) memset((void *)sp, 0, np);
    }
    r->base = base; r->end = end; r->mem = (uintptr_t)mem; r->state = state; r->gen = gen; r->g1 = g1; r->g2 = g2; r->fgen = fgen;
    sidx_del_locked(slot); sidx_add_locked(slot, base); sidx_pmax_rebuild();   // s56b2 (dirtyidx: the prefix maximum)
    if (slot == n) atomic_store(&g_nr, n + 1);
    atomic_store_explicit(&r->live, 1, memory_order_release);
    pthread_mutex_unlock(&g_slot_mu);
}
// Pool destruction: first restore still-protected pages to writable (the pool memory may be reused for something else once returned to the game), then mark invalid
void nvn_dirty_remove_range(const void *mem) {
    pthread_mutex_lock(&g_slot_mu);
    range_t *r = NULL; int n = atomic_load(&g_nr);
    int ri = -1;
    for (int i = 0; i < n && !r; i++) if (live_at(i) && rg(i)->mem == (uintptr_t)mem) { r = rg(i); ri = i; }   // Match by the pool's exact start: an adjacent pool sharing a page may have the same rounded base
    if (r) prepare_write_range_locked(r->base, r->end);   // dirtyidx: while r is still in the index (this range and every one sharing its pages)
    if (r) { sidx_del_locked(ri); sidx_pmax_rebuild(); }   // s56b2
    if (r) { atomic_store_explicit(&r->live, 0, memory_order_release); atomic_fetch_sub(&g_live, 1); }
    pthread_mutex_unlock(&g_slot_mu);
}
int nvn_dirty_tracked(const void *addr, size_t len) {
    return find_span((uintptr_t)addr, len) != NULL;
}
// The tracked range holding [addr, addr+len), as a handle a caller keeps (packaged textures: the scan over every range ran twice per texture lookup). NULL = untracked
const void *nvn_dirty_span(const void *addr, size_t len) { return len ? find_span((uintptr_t)addr, len) : NULL; }
// s56b: the host is about to write [addr, addr+len) through another view of the same pages (the backing alias): no page may change protection. 1 = a page is
// protected or busy (someone watches it: the caller writes the normal way instead); 0 = none is, and the pages are stamped as a host write stamps unprotected
// pages (nvn_dirty_stamp_unprotected); -1 = not tracked (nothing to stamp)
static void sidx_add_locked(int slot, uintptr_t base) {
    if (g_sidx_n == g_sidx_cap) { int c = g_sidx_cap ? 2 * g_sidx_cap : 1024; int *x = realloc(g_sidx, (size_t)c * sizeof *x); if (!x) { g_spmax_cap = -1; return; } g_sidx = x; g_sidx_cap = c; }   /* dirtyidx: an incomplete index: the full scan */
    int lo = 0, hi = g_sidx_n; while (lo < hi) { int m = (lo + hi) / 2; if (rg(g_sidx[m])->base <= base) lo = m + 1; else hi = m; }
    memmove(&g_sidx[lo + 1], &g_sidx[lo], (size_t)(g_sidx_n - lo) * sizeof *g_sidx); g_sidx[lo] = slot; g_sidx_n++;
}
static _Atomic uint64_t g_s56b2_prot;   // s56b2: protected pages stamped without unprotecting (nvn_dirty_gpu_write)
uint64_t nvn_dirty_gpu_write_prot(void) { return atomic_exchange(&g_s56b2_prot, 0); }
int nvn_dirty_gpu_write(const void *addr, size_t len) {   // s56b2: a binary search; a PROT page stays protected and is stamped
    if (!len) return -1;
    uintptr_t a = (uintptr_t)addr; range_t *r = NULL;
    pthread_mutex_lock(&g_slot_mu);
    int lo = 0, hi = g_sidx_n; while (lo < hi) { int m = (lo + hi) / 2; if (rg(g_sidx[m])->base <= a) lo = m + 1; else hi = m; }
    for (int k = lo - 1; k >= 0 && k >= lo - 8 && !r; k--) { range_t *x = rg(g_sidx[k]); if (atomic_load_explicit(&x->live, memory_order_acquire) && a >= x->base && a + len <= x->end) r = x; }
    if (!r) { pthread_mutex_unlock(&g_slot_mu); return -1; }
    uintptr_t p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    for (uintptr_t i = p; i <= q; i++) if (atomic_load(&r->state[i]) == PG_BUSY) { pthread_mutex_unlock(&g_slot_mu); return 1; }
    for (uintptr_t i = p; i <= q; i++) {
        uint8_t st = atomic_load(&r->state[i]);
        if (st == PG_PROT) { snap_page(r, i); stamp_page(r, i); atomic_fetch_add(&g_s56b2_prot, 1); }   // stays read-only: the guest's next write faults as today
        else if (atomic_load_explicit(&g_stamp_none, memory_order_relaxed)) { stamp_page(r, i); atomic_fetch_add(&g_stamped, 1); }
    }
    // s56b2: the edge pages are shared with adjacent pools' ranges (the false-sharing pages of the black tiles): stamp every other live range covering them, as the fault path does
    uintptr_t edge[2] = {r->base + (p << g_shift), r->base + (q << g_shift)};
    for (int e = 0; e < (p == q ? 1 : 2); e++) {
        uintptr_t pa = edge[e]; int l2 = 0, h2 = g_sidx_n; while (l2 < h2) { int m = (l2 + h2) / 2; if (rg(g_sidx[m])->base <= pa) l2 = m + 1; else h2 = m; }
        for (int k = l2 - 1; k >= 0 && k >= l2 - 8; k--) {
            range_t *x = rg(g_sidx[k]);
            if (x == r || !atomic_load_explicit(&x->live, memory_order_acquire) || pa < x->base || pa >= x->end) continue;
            uintptr_t j = (pa - x->base) >> g_shift; uint8_t st = atomic_load(&x->state[j]);
            if (st == PG_PROT) { snap_page(x, j); stamp_page(x, j); atomic_fetch_add(&g_s56b2_prot, 1); }
            else if (st == PG_NONE && atomic_load_explicit(&g_stamp_none, memory_order_relaxed)) { stamp_page(x, j); atomic_fetch_add(&g_stamped, 1); }
        }
    }
    pthread_mutex_unlock(&g_slot_mu);
#if defined(DM_NXN1X)
    nvnc_cpu((uint64_t)(uintptr_t)addr, len, NVNC_GPU_WRITEBACK);
#endif
    return 0;
}
// Set read-only: mark the pages BUSY first and mark them PROT only after mprotect has landed (for the order and the invariant see the state notes at the top of the file)
void nvn_dirty_protect(const void *addr, size_t len) {
    if (atomic_load_explicit(&g_busy_red_page, memory_order_relaxed)) {   // nceresume R5
        uintptr_t pg = atomic_exchange(&g_busy_red_page, 0);
        if (pg) LOG("[RED] dirty-page tracking: page %p stuck 'busy' for over %d ms (the thread changing its protection has not finished), no longer waiting", (void *)pg, BUSY_WAIT_MS);
    }
    // Keep this descriptor live through the last mprotect. Retirement restores
    // RW under this same mutex; it must not finish between lookup and PG_BUSY.
    pthread_mutex_lock(&g_slot_mu);
    range_t *r = len ? find_span((uintptr_t)addr, len) : NULL;
    if (!r) { pthread_mutex_unlock(&g_slot_mu); return; }
    uintptr_t a = (uintptr_t)addr, p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    // Consecutive pages are merged into one mprotect (per-page calls reach thousands per frame on the first upload of large textures; character select page startup frame 5: 4888 calls).
    // Pages already PROT / BUSY break the consecutive run: do not protect again, and do not touch pages another thread is changing.
    // A BUSY page is waited out (the run so far is protected first, so none of ours is BUSY meanwhile) and looked at again: an unprotect in progress leaves the page writable
    // with a generation the caller's stamp may cover, so skipped, its later writes never faulted and the texture stayed stale (MacNativeGames a166602)
    uintptr_t run = 0; int open = 0;
    for (uintptr_t i = p; i <= q + 1; i++) {
        uint8_t st = PG_NONE;
        int took = i <= q && atomic_compare_exchange_strong(&r->state[i], &st, PG_BUSY);
        if (took && !open) { run = i; open = 1; }
        if (!took && open) { mp(r->base + (run << g_shift), (i - run) << g_shift, PROT_READ); for (uintptr_t k = run; k < i; k++) atomic_store(&r->state[k], PG_PROT); open = 0; }
        if (!took && i <= q && st == PG_BUSY && busy_wait(r, i)) i--;   // this page again
    }
    pthread_mutex_unlock(&g_slot_mu);
}
// As nvn_dirty_protect, for the pages behind a placement sparse tile: a write to one of them moves nvn_dirty_sparse_writes, which nothing else moves, so its textures need no look at their tiles
// until one is written (the page table's walk per look is what this replaces; nvn_mtl.m sparse_look)
void nvn_dirty_protect_sparse(const void *addr, size_t len) {
    nvn_dirty_protect(addr, len);
    pthread_mutex_lock(&g_slot_mu);
    range_t *r = len ? find_span((uintptr_t)addr, len) : NULL;
    if (r) {
        _Atomic uint8_t *sp = atomic_load(&r->sparse);
        if (!sp) { sp = calloc(r->cap, 1); atomic_store(&r->sparse, sp); }
        if (sp) for (uintptr_t i = ((uintptr_t)addr - r->base) >> g_shift, q = ((uintptr_t)addr + len - 1 - r->base) >> g_shift; i <= q; i++) atomic_store(&sp[i], 1);
    }
    pthread_mutex_unlock(&g_slot_mu);
}
uint64_t nvn_dirty_sparse_writes(void) { return atomic_load(&g_sparse_writes); }
uint64_t nvn_dirty_now(void) { return atomic_load(&g_now); }
// As nvn_dirty_since, with the caller's last range kept in *span (rechecked, scanned again only if it no longer holds the span; a slot reused by another pool over the same memory is as
// good): a packaged texture keeps its handle (the scan over every range ran twice per texture lookup), and a sparse texture's page-table walk (hle_nvn.c vpt_walk) asks once per run of
// physical pages, which mostly lie in one pool, where find_span scans every registered range (about 4,300 in a Mario Kart World race)
int nvn_dirty_since_cached(const void **span, const void *addr, size_t len, uint64_t stamp) {
    if (!len) return 0;
    range_t *r = (range_t *)*span; uintptr_t a = (uintptr_t)addr;
    if (!r || !atomic_load_explicit(&r->live, memory_order_acquire) || a < r->base || a + len > r->end) { r = find_span(a, len); *span = r; }
    if (!r) return 1;   // When untracked, conservatively treat as dirty: better to upload more than to show stale content
    uintptr_t p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift;
    for (uintptr_t i = p; i <= q; i++) if (atomic_load(&r->gen[i]) > stamp) return 1;   // The generation of the last write before upload is <= the stamp; any later write must be > the stamp
    return 0;
}
int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp) { return nvn_dirty_since_cached(&span, addr, len, stamp); }
int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp) { return nvn_dirty_since_span(NULL, addr, len, stamp); }
// stage2b (volatile pages, gfx_vk_core.c): the copied pools' sync found the CPU's bytes changed on [addr, addr+len), pages it leaves writable (so no fault stamped
// them): they are stamped here in every range covering them, without any protection change, so that the other readers of the generations (packaged textures,
// mirrors) see the write as they see a fault's. A page another thread is changing (busy) is stamped too: its stamp after the change is what counts
void nvn_dirty_note_write(const void *addr, size_t len) {
    if (!len) return;
    uintptr_t a = (uintptr_t)addr, e = a + len; int n = atomic_load(&g_nr);
    for (int i = 0; i < n; i++) { range_t *r = rg(i); if (!live_at(i) || e <= r->base || a >= r->end) continue;
        uintptr_t lo = a > r->base ? a : r->base, hi = e < r->end ? e : r->end;
        for (uintptr_t p = (lo - r->base) >> g_shift, q = (hi - 1 - r->base) >> g_shift; p <= q; p++) stamp_page(r, p); }
}
// stage2b3: 1 = a page of [addr, addr+len) took a write fault after stamp (its fault generation is newer). Untracked: 0
int nvn_dirty_faulted_since(const void *span, const void *addr, size_t len, uint64_t stamp) {
    if (!len) return 0;
    range_t *r = (range_t *)span; uintptr_t a = (uintptr_t)addr;
    if (!r || !atomic_load_explicit(&r->live, memory_order_acquire) || a < r->base || a + len > r->end) r = find_span(a, len);
    if (!r || !r->fgen) return 0;
    for (uintptr_t p = (a - r->base) >> g_shift, q = (a + len - 1 - r->base) >> g_shift; p <= q; p++) if (atomic_load(&r->fgen[p]) > stamp) return 1;
    return 0;
}
// pcverify (diagnostic): the page holding addr in the tracked range: its last generation and protection state (0 none, 1 read-only, 2 busy); its group
// generations; how many live ranges cover it. 0 = untracked
int nvn_dirty_page_info(const void *span, const void *addr, uint64_t *gen, int *state, uint64_t *g1, uint64_t *g2, int *ranges) {
    range_t *r = (range_t *)span; uintptr_t a = (uintptr_t)addr;
    if (!r || !atomic_load_explicit(&r->live, memory_order_acquire) || a < r->base || a >= r->end) r = find_span(a, 1);
    if (!r) return 0;
    uintptr_t p = (a - r->base) >> g_shift;
    *gen = atomic_load(&r->gen[p]); *state = atomic_load(&r->state[p]); *g1 = atomic_load(&r->g1[p >> G1_SHIFT]); *g2 = atomic_load(&r->g2[p >> G2_SHIFT]);
    int n = atomic_load(&g_nr), k = 0; for (int i = 0; i < n; i++) { range_t *x = rg(i); if (live_at(i) && a >= x->base && a < x->end) k++; }
    *ranges = k; return 1;
}
// stage2a2: each run of pages in [addr, addr+len) written after stamp (generation > stamp, as nvn_dirty_since) goes to fn(ctx, run start, run length), clipped
// to [addr, addr+len). Groups of 4,096 and 64 pages whose highest generation is <= stamp are skipped whole; only the pages of a raised group are looked at.
// The same generations as nvn_dirty_since, so the same writes count: every stamp (fault, host pre-write, GPU write, stamped unprotected page) goes through
// stamp_page. Untracked memory counts as written (as nvn_dirty_since). *looked += the page generations read. Returns the dirty pages reported
uint64_t nvn_dirty_each_since(const void *span, const void *addr, size_t len, uint64_t stamp, void (*fn)(void *ctx, uintptr_t a, size_t n), void *ctx, uint64_t *looked) {
    if (!len) return 0;
    range_t *r = (range_t *)span; uintptr_t a = (uintptr_t)addr, e = a + len;
    if (!r || !atomic_load_explicit(&r->live, memory_order_acquire) || a < r->base || e > r->end) r = find_span(a, len);
    if (!r) { fn(ctx, a, len); return (len + g_pg - 1) >> g_shift; }
    const uintptr_t p = (a - r->base) >> g_shift, q = (e - 1 - r->base) >> g_shift;
    uint64_t dirty = 0, seen = 0; uintptr_t run = 0; int open = 0;
    const uintptr_t m1 = ((uintptr_t)1 << G1_SHIFT) - 1, m2 = ((uintptr_t)1 << G2_SHIFT) - 1;
    for (uintptr_t i = p; i <= q + 1;) {
        int d = 0; uintptr_t next = i + 1;
        if (i <= q) {
            if (atomic_load(&r->g2[i >> G2_SHIFT]) <= stamp) next = (i | m2) + 1;
            else if (atomic_load(&r->g1[i >> G1_SHIFT]) <= stamp) next = (i | m1) + 1;
            else { seen++; d = atomic_load(&r->gen[i]) > stamp; }
        }
        if (d) { dirty++; if (!open) { run = i; open = 1; } }
        else if (open) {
            uintptr_t ra = r->base + (run << g_shift), re = r->base + (i << g_shift);
            if (ra < a) ra = a;
            if (re > e) re = e;
            if (re > ra) fn(ctx, ra, re - ra);
            open = 0;
        }
        i = next > q + 1 ? q + 1 + (i > q) : next;   // (the step past q closes a run still open)
        if (i > q + 1) break;
    }
    if (looked) *looked += seen;
    return dirty;
}
void nvn_dirty_slot_stats(int *live, int *used_slots, uint64_t *adds_total) {
    int n = atomic_load(&g_nr), active = 0;
    for (int i = 0; i < n; i++) active += live_at(i) != 0;
    *live = active; *used_slots = n; *adds_total = atomic_load(&g_adds_total);
}
void nvn_dirty_stats(uint64_t *faults, uint64_t *preps) { *faults = atomic_load(&g_faults); *preps = atomic_load(&g_preps); }
void nvn_dirty_mp_stats(uint64_t *calls, uint64_t *pages, uint64_t *ns) {
    static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb);
    *calls = atomic_load(&g_mp_calls); *pages = atomic_load(&g_mp_pages); *ns = atomic_load(&g_mp_ticks) * tb.numer / tb.denom;
}
