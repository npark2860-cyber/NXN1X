// NXN1X build of the host NVN layer: the parts of the native host's runtime (dm/host, dm/windows) that the NVN layer (dm/gfx/nvn) and the Vulkan backend
// (dm/windows/gfx_vk_*.c) call, for a process that is the emulator rather than the host. Everything that is the host's job in its own process
// (loading modules, threads, signals and faults, the window, the performance panel, cheats) is NXN1X's here: these are its stand-ins, kept small.
// The symbol list is what the static library leaves undefined without dm/host and dm/windows/{front,perf_overlay,hwwatch}_win.c and posix_win.c
// (llvm-nm over the objects, docs/lane_a/A1_BUILD.md).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>
#include "dm.h"
#include "nvnh_api.h"

// NVN image residency: a private, automatically deleted file, outside the guest's
// pools. Streaming into it avoids an image-sized RAM copy on an integrated GPU.
FILE *nvnh_image_archive_file(uint64_t bytes) {
    wchar_t dir[MAX_PATH + 1], path[MAX_PATH + 1];
    DWORD n = GetTempPathW(MAX_PATH, dir);
    ULARGE_INTEGER available;
    if (!n || n >= MAX_PATH || !GetDiskFreeSpaceExW(dir, &available, NULL, NULL) ||
        bytes > available.QuadPart || available.QuadPart - bytes < (64ull << 20)) return NULL;
    if (!GetTempFileNameW(dir, L"sna", 0, path)) return NULL;
    HANDLE h = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                          FILE_FLAG_DELETE_ON_CLOSE | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (h == INVALID_HANDLE_VALUE) { DeleteFileW(path); return NULL; }
    int fd = _open_osfhandle((intptr_t)h, _O_BINARY | _O_RDWR);
    if (fd < 0) { CloseHandle(h); return NULL; }
    FILE *f = _fdopen(fd, "w+b");
    if (!f) _close(fd);
    return f;
}

// ---- s31: per-title defaults of the native profiles (host dm/games/<title>/game.tsv) where they differ from the build's (Bayonetta 3) ----
static _Atomic unsigned long long g_title;
static void c1_catch_install(void);   // c1 (end of this file)
void nvnh_set_title(unsigned long long id) { atomic_store(&g_title, id); c1_catch_install(); }
int nvnh_default_pass_times(void) {
    const unsigned long long title = atomic_load(&g_title);
    return title == 0x04003F40232DE000ull || title == 0x0400486024DEC000ull;
}
// dkpar (11-22): Donkey Kong Bananza records with a replay thread of its own and 4 parallel recording workers unless DM_VK_ASYNC / DM_VK_PARALLEL say otherwise
// (runs 11-22 08:33: stable after uppool, ~85% of draws on workers, 20 -> 26-30 fps)
int nvnh_default_async(void) { return atomic_load(&g_title) == 0x040074A01BF12000ull ? 1 : 0; }
int nvnh_default_parallel(void) { return atomic_load(&g_title) == 0x040074A01BF12000ull ? 4 : 0; }
int nvnh_oot_upscale_standin(void) { return atomic_load(&g_title) == 0x0400486024DEC000ull; }   // ootbilerp3: gfx_vk_exec.c oot_bilerp_rt, on for Ocarina of Time
int nvnh_xcx_legacy_sectors(void) {
    if (atomic_load(&g_title) != 0x0400453019AA8001ull) return 0;
    const char *v = getenv("NXN1X_XCX_LEGACY_SECTORS");
    return !(v && !strcmp(v, "0"));
}
static int dm_title_default(int key) {   // -1 = none
    static const struct { unsigned long long id; short v[8]; } T[] = {   // s70d: key 7 = DM_VK_S56_IMG (gfx_vk_core.c s70_max)
      // s62: 7 keys (5 barrier policy, 6 DM_VK_CB_DRAWS: dm_title_default_key), shorts for 128
        // Zelda TotK: render_cond (clears are skipped by conditional rendering: black 3D scene without it, runs 45/46b).
        // tex_flush/dirty_upload stay off: with them the boot hung at 10 s (run 46, NCE DeliverPendingBreak livelock)
        // s56: key 3 (DM_VK_S56) on: what it renders into its terrain tiles is written back into their storage (gfx_vk_core.c vk_s56_mirror)
        // Feed the GPU at pass boundaries while the CPU records the rest of the fenced frame.
        {0x0100F2C0115B6000ull, {-1, -1, 1, 1, -1, -1, 32, -1}},
        // z4: Zelda BotW: key 4 (DM_VK_Z4) on: its LINEAR_RENDER_TARGET readback targets (192x192 / 256x256 R16 and others, run 78) are written back into their storage
        // (characters sank slowly into flat terrain on the NVN path only). Block-linear s56 stays TotK-only (z2 reverted)
        {0x01007EF00011E000ull, {-1, -1, -1, -1, 1, -1, -1, -1}},
        // s62: Xenoblade Chronicles 3: the replay cuts before any pass every 128 draws (DM_VK_CB_DRAWS; gfx_vk_exec.c x_read): 10 parts a frame
        // reach the GPU while the rest is recorded (run 91, Colony 9: 30-33 -> 38-40 fps; run 92 ~10 min without a freeze)
        // s70d: s56 on (key 3) and its IMAGE / IMAGE|COMPRESSIBLE extension on (key 7): the save thumbnail (a 320x180 RGBA8 IMAGE|COMPRESSIBLE target the game
        // reads from pool memory, no ReadTexels / copy calls; run 110) was black without them; run 113: right, at 26-138 MB of write-back per 10 s
        {0x010074F013262000ull, {-1, -1, -1, 1, -1, -1, 128, 1}},
        // NVN2: Mario Kart World: render_cond on, as its host profile (host_nx2 dm/games/0400C3F00006E000/game.tsv render_cond on: agl wraps its clears in
        // render conditions). Hyrule Warriors uses cache-only barrier deferral (key 5).
        {0x04003F40232DE000ull, {-1, -1, -1, -1, -1, 2, -1, -1}},
        {0x0400C3F00006E000ull, {-1, -1, 1, -1, -1, -1, -1, -1}},
        // Owl's conditional clear protects the prepass depth used by its EQUAL passes.
        {0x0400486024DEC000ull, {1, 1, 1, -1, -1, -1, -1, -1}},
        // NVN2: DK Bananza: agl wraps composite clears in SetRenderEnableConditional (nxnative DK README:12,34): drawn anyway, the composite was cleared before
        // the upscaler read it and the 3D went black after a few seconds (NXN1X 11-16: the NDSR input went to exact 0; DM_GAME_RENDER_COND was 0)
        {0x040074A01BF12000ull, {-1, -1, 1, -1, -1, -1, -1, -1}},
    };
    unsigned long long id = atomic_load(&g_title);
    for (unsigned i = 0; i < sizeof T / sizeof T[0]; i++)
        if (T[i].id == id) return T[i].v[key];
    return -1;
}
int dm_title_default_key(int key) { return key >= 0 && key < 8 ? dm_title_default(key) : -1; }   // s70d: key 7   // s62 (gfx_vk_exec.c x_read: keys 5, 6; -1 = none)
// ---- s26: profile switches at run time (dm.h) ----
// s56: key 3 = DM_VK_S56 (GPU -> guest write-back of render targets, gfx_vk_core.c vk_s56_mirror): built 0, only a title's default or the environment turns it on
// z4: key 4 = DM_VK_Z4 (GPU -> guest write-back of LINEAR / LINEAR_RENDER_TARGET render targets, gfx_vk_core.c z4_mirror)
static _Atomic int g_pkey[5] = {-1, -1, -1, -1, -1};
int dm_profile_key(int key) {
    if (key < 0 || key > 4) return 0;
    int v = atomic_load_explicit(&g_pkey[key], memory_order_relaxed);
    if (v >= 0) return v;
    static const char *const name[5] = {"DM_GAME_TEX_FLUSH_UPLOAD", "DM_GAME_TEX_DIRTY_UPLOAD", "DM_GAME_RENDER_COND", "DM_VK_S56", "DM_VK_Z4"};
    static const int built[5] = {DM_GAME_BUILT_TEX_FLUSH_UPLOAD, DM_GAME_BUILT_TEX_DIRTY_UPLOAD, DM_GAME_BUILT_RENDER_COND, 0, 1};
    const char *e = getenv(name[key]);
    const int tdef = dm_title_default(key);   // s31
    v = e && *e ? atoi(e) != 0 : tdef >= 0 ? tdef : built[key];
    if (key == 1 && v && !dm_profile_key(0)) v = 0;   // tex_dirty_upload needs tex_flush_upload
    int was = -1;
    if (atomic_compare_exchange_strong(&g_pkey[key], &was, v))
        LOG("s26: profile switch %s = %d (%s; the build's profile has %d)", name[key], v, e && *e ? "from the environment" : tdef >= 0 ? "the title's default" : "the build's", built[key]);
    return atomic_load(&g_pkey[key]);
}

// NVN2: the profile's app_pool for the running title (dm.h DM_GAME_APP_POOL; hle_nvn.c: a virtual pool larger than it gets an address range of its own).
// The one nvn_host build carries Bayonetta 3's 0xcd500000; these Switch 2 titles' host profiles have the Switch 2's 9 GB (host_nx2 dm/games/<id>/game.tsv app_pool)
unsigned long long dm_profile_app_pool(void) {
    static const struct { unsigned long long id, pool; } T[] = {
        {0x04003F40232DE000ull, {-1, -1, -1, -1, -1, 2, -1, -1}},
        {0x0400486024DEC000ull, 0x240000000ull},   // Ocarina of Time
        {0x0400C3F00006E000ull, 0x240000000ull},   // Mario Kart World
        {0x04003F40232DE000ull, 0x240000000ull},   // Hyrule Warriors: Age of Imprisonment
    };
    const unsigned long long id = atomic_load(&g_title);
    for (unsigned i = 0; i < sizeof T / sizeof T[0]; i++)
        if (T[i].id == id) return T[i].pool;
    return dm_game_built_app_pool();
}

// ---- clocks (dm/host/hle_os.c) ----
uint64_t g_t0_ns;   // dm/host/main.c: host start, for relative log times
uint64_t now_ns(void) { static LARGE_INTEGER f; LARGE_INTEGER c; if (!f.QuadPart) QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (uint64_t)((unsigned __int128)c.QuadPart * 1000000000u / (uint64_t)f.QuadPart); }
static uint64_t (*_Atomic g_guest_clock)(void);
void nvnh_set_guest_clock(uint64_t (*fn)(void)) { atomic_store(&g_guest_clock, fn); }
uint64_t guest_ns(void) { uint64_t (*f)(void) = atomic_load(&g_guest_clock); return f ? f() : now_ns(); }
int dm_detclock_enabled(void) { return 0; }   // DM_DETCLOCK (presents advance the guest clock) is a host test mode: the emulator owns the guest clock
static _Atomic uint64_t g_guest_frame;
void dm_guest_frame_tick(void) { atomic_fetch_add(&g_guest_frame, 1); }
_Atomic uint64_t dm_present_ns;   // dm/host/hle_thread.c (DM_HITCH_DUMP watches it there)

// ---- write tracking hooks (dm/host/crash.c): nvn_dirty.c registers them, NXN1X calls them (NVN_HOST_BRIDGE.md §4.2) ----
static _Atomic(nvnh_fault_fn) g_fault;
static _Atomic(nvnh_prep_fn) g_prep;
void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *uc)) { atomic_store(&g_fault, fn); }
void dm_guest_write_prep_register(void (*fn)(void *addr, size_t len)) { atomic_store(&g_prep, fn); }
void dm_guest_write_prepare(void *addr, size_t len) { nvnh_prep_fn f = atomic_load(&g_prep); if (f) f(addr, len); }
nvnh_fault_fn nvnh_fault_hook(void) { return atomic_load(&g_fault); }
nvnh_prep_fn nvnh_write_prep_hook(void) { return atomic_load(&g_prep); }
void dm_backtrace(uint64_t fp) { (void)fp; }   // the emulator has its own crash report

// Write watch self-test (NXN1X_NVN_WATCH_SELFTEST=1, NVN_HOST_BRIDGE.md §4.2). Once per process, on the first pool with CPU access and at least three
// pages (hle_nvn.c MemoryPoolInitialize, after nvn_dirty_add_range): nvn_dirty write-protects the pool's middle page, and this thread writes it once with an
// atomic OR of 0, so the bytes stay as they are. NXN1X's exception handler must pass the fault to on_fault (arm_nce_windows.cpp, NvnLayer::OnWatchedWrite),
// which unprotects and stamps the page, and the write runs again. Passed = the page's generation moved and the fault count grew. A guest thread's write
// goes through the same handler head; this checks the hook, nvn_dirty's protection on NXN1X's guest memory and the re-run, and times one round trip
#include "../gfx/nvn/nvn_dirty.h"
void nvnh_watch_selftest(void *mem, size_t size, int cpu_access) {
    static _Atomic int done;
    static int want = -1;
    if (want < 0) { const char *e = getenv("NXN1X_NVN_WATCH_SELFTEST"); want = e && atoi(e); }
    const size_t pg = 4096;   // nvnh_posix.c getpagesize, the page nvn_dirty protects
    if (!want || !cpu_access || !mem || size < 3 * pg || atomic_load(&done) || atomic_exchange(&done, 1)) return;
    uint8_t *page = (uint8_t *)(((uintptr_t)mem + size / 2) & ~(uintptr_t)(pg - 1));
    if (!nvn_dirty_tracked(page, pg)) { LOG("[RED] write watch self-test: pool %p +%#llx is not tracked by nvn_dirty", mem, (unsigned long long)size); return; }
    uint64_t f0, p0, f1, p1; nvn_dirty_stats(&f0, &p0);
    const uint64_t stamp = nvn_dirty_now();
    nvn_dirty_protect(page, pg);
    const uint64_t t0 = now_ns();
    atomic_fetch_or((_Atomic uint8_t *)page, 0);   // faults while read-only; on_fault makes it writable and the write runs again
    const uint64_t t1 = now_ns();
    nvn_dirty_stats(&f1, &p1);
    const int stamped = nvn_dirty_since(page, pg, stamp);
    LOG("%swrite watch self-test: %s (page %p of pool %p +%#llx; faults %llu -> %llu, stamped %d, write with fault %.1f us)", stamped && f1 > f0 ? "" : "[RED] ",
        stamped && f1 > f0 ? "passed" : "FAILED", (void *)page, mem, (unsigned long long)size, (unsigned long long)f0, (unsigned long long)f1, stamped, (t1 - t0) / 1000.0);
}

// ---- log (dm.h LOG under DM_NXN1X) ----
static void (*_Atomic g_log)(const char *line);
void nvnh_set_log(void (*fn)(const char *line)) { atomic_store(&g_log, fn); }
void dm_log_line(const char *fmt, ...) {
    char buf[2048]; va_list ap; va_start(ap, fmt); int n = vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap); (void)n;
    void (*f)(const char *) = atomic_load(&g_log);
    if (f) { f(buf); return; }
    flockfile(stderr); fprintf(stderr, "[dm] %s\n", buf); funlockfile(stderr);
}

// ---- modules (dm/host/loader.c): the emulator loads the game; call sites are logged without a module name ----
module_t *module_of(const void *p) { (void)p; return NULL; }

// ---- big host stack (dm/host/hle_bigstack.c): the gate already runs every NVN function on the guest thread's host stack, so the entry goes straight through ----
__asm__(".text\n.p2align 2\n.globl dm_bigstack_enter\ndm_bigstack_enter:\n    br x16\n");

// ---- window and present (dm/windows/front_win.c): output goes to NXN1X's presentation (NVN_HOST_BRIDGE.md §4.5), the host makes no window ----
void *dm_front_window(void) { return NULL; }
void dm_front_size(void *w, uint32_t *width, uint32_t *height) { (void)w; *width = 1280; *height = 720; }
VkResult dm_front_surface(VkInstance inst, PFN_vkGetInstanceProcAddr gipa, void *w, VkSurfaceKHR *out) { (void)inst; (void)gipa; (void)w; *out = VK_NULL_HANDLE; return VK_ERROR_INITIALIZATION_FAILED; }
float dm_front_scale(void *w) { (void)w; return 1.0f; }
int front_screen_max_fps(void) { return 60; }
void dm_vi_layer_crop(int c[4]) { c[0] = c[1] = c[2] = c[3] = 0; }   // dm/gfx/hle_vi.c: the game's layer crop; NXN1X's vi keeps its own

// ---- performance panel (dm/windows/perf_overlay_win.c): NXN1X draws its own overlay ----
int perf_ov_on(void) { return 0; }
void perf_ov_driver(const char *s) { (void)s; }
void perf_ov_gpu_ns(uint64_t ns) { (void)ns; }
void perf_ov_vram(int64_t used, int64_t budget) { (void)used; (void)budget; }
void perf_ov_hidden(void) {}
int perf_ov_frame(int64_t now, uint64_t frames, uint32_t sw, uint32_t sh, float k, const uint8_t **px, int *w, int *h, int *x, int *y) {
    (void)now; (void)frames; (void)sw; (void)sh; (void)k; *px = NULL; *w = *h = *x = *y = 0; return 0; }

// ---- the rest ----
void dm_cheats_present(void) {}   // dm/host/hle_cheats.c: NXN1X applies cheats itself
void dm_hwwatch_set(uint64_t a0, uint64_t a1, int n) { (void)a0; (void)a1; (void)n; }

// s10 (nvnh_vk.h): a refused import's range, region by region. VirtualQuery walks the whole view (about 1.5 ms on NXN1X's), so only for a few refusals
void nvnh_range_regions(const void *mem, uint64_t size, char *out, size_t cap) {
    const uint8_t *a = mem, *e = (const uint8_t *)mem + size; size_t n = 0; int k = 0; out[0] = 0;
    while (a < e && k < 12 && n + 160 < cap) {
        MEMORY_BASIC_INFORMATION mi;
        if (!VirtualQuery(a, &mi, sizeof mi)) { n += (size_t)snprintf(out + n, cap - n, " [+%#llx: VirtualQuery failed %lu]", (unsigned long long)(a - (const uint8_t *)mem), GetLastError()); break; }
        const uint8_t *re = (const uint8_t *)mi.BaseAddress + mi.RegionSize; if (re > e) re = e;
        n += (size_t)snprintf(out + n, cap - n, " [+%#llx +%#llx alloc %p%s state %#lx prot %#lx type %#lx]", (unsigned long long)(a - (const uint8_t *)mem),
            (unsigned long long)(re - a), mi.AllocationBase, mi.AllocationBase == mem ? " (=start)" : "", (unsigned long)mi.State, (unsigned long)mi.Protect, (unsigned long)mi.Type);
        a = re; k++;
    }
    if (a < e && n + 40 < cap) snprintf(out + n, cap - n, " ... (+%#llx more)", (unsigned long long)(e - a));
}
typedef BOOL (WINAPI *query_region_fn)(HANDLE, const void *, WIN32_MEMORY_INFORMATION_CLASS,
                                       void *, SIZE_T, PSIZE_T);
static INIT_ONCE region_query_once = INIT_ONCE_STATIC_INIT;
static query_region_fn query_region;
static BOOL CALLBACK resolve_region_query(PINIT_ONCE once, PVOID param, PVOID *context) {
    (void)once; (void)param; (void)context;
    HMODULE module = GetModuleHandleW(L"kernelbase.dll");
    if (module) query_region = (query_region_fn)(void *)GetProcAddress(module, "QueryVirtualMemoryInformation");
    return TRUE;
}
int nvnh_range_views(const void *mem, uint64_t size, uint64_t *cut, int max) {   // s11 (nvnh_vk.h)
    InitOnceExecuteOnce(&region_query_once, resolve_region_query, NULL, NULL);
    uintptr_t start = (uintptr_t)mem, cursor = start;
    void *previous = NULL;
    int n = 0;
    for (int k = 0; cursor - start < size && k < 4096; k++) {
        // The caller needs view boundaries. Query the allocation extent directly so a
        // large RAM backing view is independent of the number of pages or protection runs.
        WIN32_MEMORY_REGION_INFORMATION region;
        void *allocation;
        uintptr_t end;
        if (query_region && query_region(GetCurrentProcess(), (void *)cursor, MemoryRegionInfo,
                                         &region, sizeof region, NULL)) {
            allocation = region.AllocationBase;
            end = (uintptr_t)allocation + region.RegionSize;
        } else {
            // MemoryRegionInfo requires an allocation. Preserve traversal across free space.
            MEMORY_BASIC_INFORMATION mi;
            if (!VirtualQuery((void *)cursor, &mi, sizeof mi)) break;
            allocation = mi.AllocationBase;
            end = (uintptr_t)mi.BaseAddress + mi.RegionSize;
        }
        if (end <= cursor) break;
        if (previous && allocation != previous) {
            if (n == max) break;
            cut[n++] = cursor - start;
        }
        previous = allocation;
        cursor = end;
    }
    return n;
}   // dm/windows/hwwatch_win.c (DM_VK_HWWATCH debug watchpoints): not in NXN1X
uint64_t g_nvnh_vq;   // NVN2: VirtualQuery calls nvnh_range_readable made (gfx_vk_core.c 30 s line)
uint64_t g_nvnh_hm;   // bak84: answers nvnh_range_readable_hm took from HostMemory's own map
int nxn1x_host_range_readable(const void *p, uint64_t n);   // common/host_memory.cpp: 1 readable, 0 not known, -1 outside its reservations
int nvnh_range_readable(const void *mem, uint64_t size);
// bak84 (11-16, MKW 15:00: readable queries 4.2-7.6 s of every 30 s in the race, nearly all for the first uploads of images made again after evictions, about 3 ms a
// VirtualQuery): HostMemory knows which guest ranges are mapped views and which a Protect took reading from; a range it says is readable needs no system call. Anything else
// (not covered by its views, outside its reservation) is asked as before. Its Map / Unmap / Protect keep that map under the lock they already take
int nvnh_range_readable_hm(const void *mem, uint64_t size) {
    if (nxn1x_host_range_readable(mem, size) == 1) { g_nvnh_hm++; return 1; }
    return nvnh_range_readable(mem, size);
}
uint64_t nxn1x_host_map_gen(void);   // common/host_memory.cpp
uint64_t nvn_vpool_unmap_gen(void);   // gfx/nvn/hle_nvn.c: a virtual pool range made inaccessible (vpool_map unmap, vpool_drop) outside HostMemory
uint64_t nvnh_map_gen(void) { return nxn1x_host_map_gen() + nvn_vpool_unmap_gen(); }
int nvnh_range_readable(const void *mem, uint64_t size) {   // NVN2 s24 guard (nvnh_vk.h)
    const uint8_t *a = mem, *e = (const uint8_t *)mem + size;
    for (int k = 0; a < e; k++) {
        MEMORY_BASIC_INFORMATION mi; g_nvnh_vq++;
        if (k == 4096 || !VirtualQuery(a, &mi, sizeof mi) || mi.State != MEM_COMMIT || (mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) || !mi.Protect) return 0;
        a = (const uint8_t *)mi.BaseAddress + mi.RegionSize;
    }
    return 1;
}
// s23 (nvnh_api.h, nvnh_vk.h): the backing alias NXN1X registers (NvnLayer::SetGuestMemory)
static _Atomic(nvnh_alias_fn) g_alias;
void nvnh_set_import_alias(nvnh_alias_fn fn) { atomic_store(&g_alias, fn); }
void *nvnh_import_alias(const void *mem, uint64_t size) { nvnh_alias_fn f = atomic_load(&g_alias); return f ? f(mem, size) : NULL; }
// poolcopy self-test (gfx_vk_core.c vk_poolcopy_selftest): a pagefile section of n bytes mapped three times, as NXN1X's guest memory is: a backing view and two
// views the test registers as pools over the same pages (aliases); the alias function then answers for the two views with the backing view, others as before.
// nvnh_test_alias_end puts the previous alias function back. 0 = made
static nvnh_alias_fn g_test_prev; static uint8_t *g_test_back, *g_test_v[2]; static size_t g_test_n; static HANDLE g_test_sec;
static void *test_alias(const void *mem, uint64_t size) {
    const uint8_t *m = (const uint8_t *)mem;
    for (int k = 0; k < 2; k++) if (g_test_v[k] && m >= g_test_v[k] && m + size <= g_test_v[k] + g_test_n) return g_test_back + (m - g_test_v[k]);
    return g_test_prev ? g_test_prev(mem, size) : NULL;
}
int nvnh_test_alias_views(size_t n, void **va, void **vb) {
    g_test_sec = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, (DWORD)((uint64_t)n >> 32), (DWORD)n, NULL); if (!g_test_sec) return -1;
    g_test_back = MapViewOfFile(g_test_sec, FILE_MAP_ALL_ACCESS, 0, 0, n); g_test_v[0] = MapViewOfFile(g_test_sec, FILE_MAP_ALL_ACCESS, 0, 0, n); g_test_v[1] = MapViewOfFile(g_test_sec, FILE_MAP_ALL_ACCESS, 0, 0, n);
    if (!g_test_back || !g_test_v[0] || !g_test_v[1]) return -2;
    g_test_n = n; g_test_prev = atomic_load(&g_alias); atomic_store(&g_alias, test_alias); *va = g_test_v[0]; *vb = g_test_v[1]; return 0;
}
// poolcopy self-test case 5: 2n bytes that are two allocations side by side (two "views" to nvnh_range_views, as NXN1X's guest memory mapped from two blocks), so
// one pool over them is made of two chunks and a range across the boundary gets a bridge. NULL when the address space could not be had so
void *nvnh_test_two_views(size_t n) {
    for (int t = 0; t < 8; t++) {
        uint8_t *b = VirtualAlloc(NULL, 2 * n, MEM_RESERVE, PAGE_NOACCESS); if (!b) return NULL;
        VirtualFree(b, 0, MEM_RELEASE);
        uint8_t *x = VirtualAlloc(b, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE), *y = x ? VirtualAlloc(b + n, n, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) : NULL;
        if (x == b && y == b + n) return b;
        if (x) VirtualFree(x, 0, MEM_RELEASE); if (y) VirtualFree(y, 0, MEM_RELEASE);
    }
    return NULL;
}
void nvnh_test_alias_end(void) {   // (the views stay mapped: an import or a copy's range made over them may outlive the test's pools)
    atomic_store(&g_alias, g_test_prev); g_test_v[0] = g_test_v[1] = NULL;
}

// c1 (lane A, 2026-11-13): run 51 (TotK, s34) and the 06:48 crash ended with an access violation and left no WER report or dump (and the registry stays as it is).
// When a game starts, an unhandled-exception filter is set (NCE's vectored handlers still see every fault first; only what nothing handled reaches it). It writes
// <exe dir>\nxn1x_crash_<pid>.txt: exception, address as module+offset, x0-x30/sp/pc, and a walk of the frame-pointer chain on the faulting thread's stack, then
// <exe dir>\nxn1x_crash_<pid>.dmp (dbghelp, threads + data segments + memory the registers and stacks point at; not full memory: the guest's views are GBs), and
// passes the exception on. Placed after the rest of the file so windows.h meets none of it. DM_CRASH_CATCH=0: not set
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static LPTOP_LEVEL_EXCEPTION_FILTER c1_prev;
static char c1_dir[MAX_PATH];
static int c1_mod(uint64_t a, char *s, size_t n) {   // "name+0xoff" for an address in a loaded module
    HMODULE h = NULL; char p[MAX_PATH];
    if (!a || !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)(uintptr_t)a, &h) || !h) return snprintf(s, n, "%#llx", (unsigned long long)a);
    DWORD k = GetModuleFileNameA(h, p, sizeof p); const char *b = p; for (DWORD i = 0; i < k; i++) if (p[i] == '\\' || p[i] == '/') b = p + i + 1;
    return snprintf(s, n, "%s+%#llx", k ? b : "?", (unsigned long long)(a - (uint64_t)(uintptr_t)h));
}
// crashsym (11-22, NPARK's 00:44 crash in the driver under sl.interposer: the build that followed replaced NXN1X.exe and its PDB, so the report's NXN1X.exe+offsets
// could not be named any more): a frame in NXN1X.exe gets its function name at the time of the crash, from the PDB beside the exe (dbghelp SymFromAddr, loaded
// on first use). Best effort: with no PDB or no dbghelp the line is as before
#include <dbghelp.h>
typedef BOOL (WINAPI *c1_si_t)(HANDLE, PCSTR, BOOL); typedef BOOL (WINAPI *c1_sfa_t)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO); typedef DWORD (WINAPI *c1_sso_t)(DWORD);
static int c1_sym(uint64_t a, char *s, size_t n) {   // " (function+0xoff)" for an address in NXN1X.exe, "" otherwise
    static int state; static c1_sfa_t sfa;   // state 0 not tried, 1 ready, -1 none
    HMODULE h = NULL;
    if (!a || !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)(uintptr_t)a, &h) || h != GetModuleHandleA(NULL)) { if (n) s[0] = 0; return 0; }
    if (!state) { state = -1; HMODULE dh = LoadLibraryA("dbghelp.dll");
        c1_si_t si = dh ? (c1_si_t)(void *)GetProcAddress(dh, "SymInitialize") : NULL; c1_sso_t sso = dh ? (c1_sso_t)(void *)GetProcAddress(dh, "SymSetOptions") : NULL;
        sfa = dh ? (c1_sfa_t)(void *)GetProcAddress(dh, "SymFromAddr") : NULL;
        if (si && sfa) { if (sso) sso(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_FAIL_CRITICAL_ERRORS); if (si(GetCurrentProcess(), c1_dir, TRUE)) state = 1; } }
    if (state < 0) { if (n) s[0] = 0; return 0; }
    union { SYMBOL_INFO si; char raw[sizeof(SYMBOL_INFO) + 256]; } u; memset(&u, 0, sizeof u); u.si.SizeOfStruct = sizeof(SYMBOL_INFO); u.si.MaxNameLen = 255;
    DWORD64 off = 0;
    if (!sfa(GetCurrentProcess(), (DWORD64)a, &off, &u.si)) { if (n) s[0] = 0; return 0; }
    return snprintf(s, n, " (%s+%#llx)", u.si.Name, (unsigned long long)off);
}
typedef BOOL (WINAPI *c1_mdw_t)(HANDLE, DWORD, HANDLE, int, void *, void *, void *);
// c1c: the report, for the filter (name "nxn1x_crash", with a dump) and the vectored handler (name "nxn1x_av", dump while dump_n > 0); one at a time
static volatile LONG c1_busy;
static void c1_write(EXCEPTION_POINTERS *ep, const char *kind, const char *name, int dump_n) {
    while (InterlockedExchange(&c1_busy, 1)) Sleep(1);
    static char b[16384]; int n = 0; char m[MAX_PATH + 32], path[MAX_PATH + 64];
    const EXCEPTION_RECORD *er = ep->ExceptionRecord; const CONTEXT *c = ep->ContextRecord;
    SYSTEMTIME st; GetLocalTime(&st);
    c1_mod((uint64_t)(uintptr_t)er->ExceptionAddress, m, sizeof m);
    n += snprintf(b + n, sizeof b - n, "c1: %s %08lx at %s (%p), thread %lu, %02u:%02u:%02u.%03u, title %016llx\r\n", kind, (unsigned long)er->ExceptionCode, m, er->ExceptionAddress,
                  (unsigned long)GetCurrentThreadId(), st.wHour, st.wMinute, st.wSecond, st.wMilliseconds, (unsigned long long)atomic_load(&g_title));
    for (DWORD i = 0; i < er->NumberParameters && i < 4; i++) n += snprintf(b + n, sizeof b - n, "  param[%lu] %#llx%s\r\n", (unsigned long)i, (unsigned long long)er->ExceptionInformation[i],
        er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && i == 0 ? (er->ExceptionInformation[0] == 0 ? " (read)" : er->ExceptionInformation[0] == 1 ? " (write)" : " (execute)") : "");
#if defined(__aarch64__) || defined(_M_ARM64)
    for (int i = 0; i < 31; i++) { c1_mod(c->X[i], m, sizeof m); n += snprintf(b + n, sizeof b - n, "  x%-2d %016llx %s\r\n", i, (unsigned long long)c->X[i], m); }
    n += snprintf(b + n, sizeof b - n, "  sp  %016llx\r\n  pc  %016llx\r\n", (unsigned long long)c->Sp, (unsigned long long)c->Pc);
    c1_mod(c->Pc, m, sizeof m); { char sy[300]; c1_sym(c->Pc, sy, sizeof sy); n += snprintf(b + n, sizeof b - n, "frames (frame pointer chain):\r\n  #0  %s%s\r\n", m, sy); }
    c1_mod(c->Lr & 0x00007fffffffffffull, m, sizeof m); n += snprintf(b + n, sizeof b - n, "  lr  %s\r\n", m);
    ULONG_PTR lo = 0, hi = 0; GetCurrentThreadStackLimits(&lo, &hi);
    uint64_t fp = c->Fp;
    for (int i = 1; i < 48 && fp >= lo && fp + 16 <= hi && !(fp & 7) && n < (int)sizeof b - 512; i++) {
        uint64_t nx = ((const uint64_t *)(uintptr_t)fp)[0], ret = ((const uint64_t *)(uintptr_t)fp)[1] & 0x00007fffffffffffull; HMODULE hm = NULL;   // (c1b: a PAC-signed return: its address bits)
        if (!ret || !GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)(uintptr_t)ret, &hm)) {
            n += snprintf(b + n, sizeof b - n, "  (walk stops: return %#llx is in no module)\r\n", (unsigned long long)ret); break; }
        c1_mod(ret, m, sizeof m); { char sy[300]; c1_sym(ret, sy, sizeof sy); n += snprintf(b + n, sizeof b - n, "  #%-2d %s%s\r\n", i, m, sy); }   // crashsym
        if (nx <= fp) break; fp = nx; }
    if (!(c->Fp >= lo && c->Fp < hi)) n += snprintf(b + n, sizeof b - n, "  (fp %#llx is not on this thread's stack %#llx-%#llx)\r\n", (unsigned long long)c->Fp, (unsigned long long)lo, (unsigned long long)hi);
#endif
    snprintf(path, sizeof path, "%s\\%s_%lu.txt", c1_dir, name, (unsigned long)GetCurrentProcessId());
    HANDLE f = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f != INVALID_HANDLE_VALUE) { DWORD w; WriteFile(f, b, (DWORD)n, &w, NULL); FlushFileBuffers(f); CloseHandle(f); }
    OutputDebugStringA(b); fputs(b, stderr); fflush(stderr);
    HMODULE dh = LoadLibraryA("dbghelp.dll"); c1_mdw_t mdw = dh ? (c1_mdw_t)(void *)GetProcAddress(dh, "MiniDumpWriteDump") : NULL;
    if (mdw && dump_n > 0) {
        snprintf(path, sizeof path, "%s\\%s_%lu_%d.dmp", c1_dir, name, (unsigned long)GetCurrentProcessId(), dump_n);
        f = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (f != INVALID_HANDLE_VALUE) {
            struct __attribute__((packed, aligned(4))) { DWORD tid; EXCEPTION_POINTERS *ep; BOOL client; } mei = {GetCurrentThreadId(), ep, FALSE};   // MINIDUMP_EXCEPTION_INFORMATION (pack 4, c1b)
            // MiniDumpWithDataSegs 0x1 | WithHandleData 0x4 | WithIndirectlyReferencedMemory 0x40 | WithUnloadedModules 0x20 | WithProcessThreadData 0x100 | WithThreadInfo 0x1000
            BOOL ok = mdw(GetCurrentProcess(), GetCurrentProcessId(), f, 0x1 | 0x4 | 0x20 | 0x40 | 0x100 | 0x800 | 0x1000, &mei, NULL, NULL); DWORD le = GetLastError();
            CloseHandle(f); if (!ok) fprintf(stderr, "c1: MiniDumpWriteDump failed (%lu)\n", (unsigned long)le); }
    }
    InterlockedExchange(&c1_busy, 0);
}
// hangstk (DK Bananza 11-22: with DM_VK_PARALLEL=4 the game froze twice at the title -> game load, sv1 "no present for 12 s", the guest main thread on core 0 in an
// NVN call, nothing in the log about where the host was): at sv1's first dump, every host thread's PC, LR and frame-pointer chain (24 frames), NXN1X.exe frames named
// from the PDB. Each thread is suspended only while its registers and frame words are copied (ReadProcessMemory: a bad frame pointer ends the walk; nothing is
// allocated while a thread is held), names and symbols come after all are resumed. Log only; NXN1X_HANGSTK=0 off. Lines "hangstk:"
#include <tlhelp32.h>
typedef HRESULT (WINAPI *hs_gtd_t)(HANDLE, PWSTR *);
void nvnh_hang_dump(const char *why) {
    const char *e = getenv("NXN1X_HANGSTK"); if (e && *e == '0') return;
    static volatile LONG once; if (InterlockedExchange(&once, 1)) return;
    enum { HS_T = 160, HS_F = 24 };
    static struct { DWORD tid; int ok, nf; uint64_t pc, lr, sp, f[HS_F]; } t[HS_T]; int nt = 0;
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); if (snap == INVALID_HANDLE_VALUE) { LOG("hangstk: no thread snapshot (%lu)", (unsigned long)GetLastError()); return; }
    THREADENTRY32 te; te.dwSize = sizeof te;
    for (BOOL ok = Thread32First(snap, &te); ok && nt < HS_T; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
        t[nt].tid = te.th32ThreadID; t[nt].ok = 0; t[nt].nf = 0; nt++; }
    CloseHandle(snap);
#if defined(__aarch64__) || defined(_M_ARM64)
    for (int i = 0; i < nt; i++) {
        HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, t[i].tid); if (!h) continue;
        if (SuspendThread(h) != (DWORD)-1) {
            CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (GetThreadContext(h, &c)) { t[i].ok = 1; t[i].pc = c.Pc; t[i].lr = c.Lr & 0x00007fffffffffffull; t[i].sp = c.Sp;
                uint64_t fp = c.Fp;
                for (int k = 0; k < HS_F && fp && !(fp & 7); k++) { uint64_t w[2]; SIZE_T got = 0;
                    if (!ReadProcessMemory(GetCurrentProcess(), (LPCVOID)(uintptr_t)fp, w, sizeof w, &got) || got != sizeof w) break;
                    t[i].f[t[i].nf++] = w[1] & 0x00007fffffffffffull; if (w[0] <= fp) break; fp = w[0]; } }
            ResumeThread(h); }
        CloseHandle(h); }
#endif
    LOG("hangstk: %s: %d host threads (frame-pointer chains; NXN1X.exe frames named)", why ? why : "?", nt);
    hs_gtd_t gtd = (hs_gtd_t)(void *)GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetThreadDescription");
    static char b[4096];
    for (int i = 0; i < nt; i++) {
        char name[96] = "";
        if (gtd) { HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, t[i].tid); PWSTR w = NULL;
            if (h && SUCCEEDED(gtd(h, &w)) && w) { WideCharToMultiByte(CP_UTF8, 0, w, -1, name, sizeof name, NULL, NULL); LocalFree(w); } if (h) CloseHandle(h); }
        if (!t[i].ok) { LOG("hangstk: tid %lu '%s': no context", (unsigned long)t[i].tid, name); continue; }
        char m[MAX_PATH + 32], sy[300]; int n = 0;
        c1_mod(t[i].pc, m, sizeof m); c1_sym(t[i].pc, sy, sizeof sy); n += snprintf(b + n, sizeof b - n, "hangstk: tid %lu '%s': pc %s%s", (unsigned long)t[i].tid, name, m, sy);
        c1_mod(t[i].lr, m, sizeof m); c1_sym(t[i].lr, sy, sizeof sy); n += snprintf(b + n, sizeof b - n, " | lr %s%s |", m, sy);
        for (int k = 0; k < t[i].nf && n < (int)sizeof b - 400; k++) { c1_mod(t[i].f[k], m, sizeof m); c1_sym(t[i].f[k], sy, sizeof sy); n += snprintf(b + n, sizeof b - n, " %s%s", m, sy); }
        LOG("%s", b); }
}
static LONG WINAPI c1_filter(EXCEPTION_POINTERS *ep) {
    static volatile LONG once;
    if (!InterlockedExchange(&once, 1)) c1_write(ep, "unhandled exception", "nxn1x_crash", 1);
    return c1_prev ? c1_prev(ep) : EXCEPTION_CONTINUE_SEARCH;
}
static uint64_t c1_lo, c1_hi;   // NXN1X.exe's image
static volatile LONG c1_avn;
// c1d (TotK run 54, 11-13): the load crashes (06:48, runs 51, 53, 54) are host reads of guest NVN objects (nvnBufferMap's buf->pool) on pages that were not
// readable at that instant, beside pools being made. If the page is accessible again soon, the access is retried; the log says how long it took and what the
// page was at the fault, which tells whether host memory remaps open such a window. DM_CRASH_RETRY=0: no retry
static int c1d_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_CRASH_RETRY"); v = !e || !*e || atoi(e) != 0; } return v; }
static int c1d_can(const MEMORY_BASIC_INFORMATION *mi, int acc) {   // the access (0 read, 1 write, 8 execute) is allowed on this page now
    if (mi->State != MEM_COMMIT || (mi->Protect & PAGE_GUARD)) return 0;
    DWORD p = mi->Protect & 0xff;
    if (acc == 1) return p == PAGE_READWRITE || p == PAGE_WRITECOPY || p == PAGE_EXECUTE_READWRITE || p == PAGE_EXECUTE_WRITECOPY;
    if (acc == 8) return p == PAGE_EXECUTE || p == PAGE_EXECUTE_READ || p == PAGE_EXECUTE_READWRITE || p == PAGE_EXECUTE_WRITECOPY;
    return p != 0 && p != PAGE_NOACCESS;
}
static volatile LONG c1d_n;
static __thread uint64_t c1d_lpc, c1d_lva; static __thread int c1d_reps;
static int c1d_retry(EXCEPTION_POINTERS *ep, uint64_t pc) {   // 1: retry the access
    const EXCEPTION_RECORD *er = ep->ExceptionRecord;
    if (!c1d_on() || er->NumberParameters < 2) return 0;
    uint64_t va = er->ExceptionInformation[1]; int acc = (int)er->ExceptionInformation[0];
    if (pc == c1d_lpc && va == c1d_lva) { if (++c1d_reps > 8) { c1d_reps = 0; fprintf(stderr, "c1d: %#llx at NXN1X.exe+%#llx faulted 8 times in a row: reported\n", (unsigned long long)va, (unsigned long long)(pc - c1_lo)); return 0; } }
    else { c1d_lpc = pc; c1d_lva = va; c1d_reps = 1; }
    MEMORY_BASIC_INFORMATION m0, m; memset(&m0, 0, sizeof m0);
    VirtualQuery((LPCVOID)(uintptr_t)va, &m0, sizeof m0);
    for (int ms = 0; ms <= 200; ms++) {
        if (VirtualQuery((LPCVOID)(uintptr_t)va, &m, sizeof m) && c1d_can(&m, acc)) {
            LONG r = InterlockedIncrement(&c1d_n);
            if (r <= 32 || !(r & (r - 1))) fprintf(stderr, "c1d: host %s of %#llx at NXN1X.exe+%#llx (thread %lu) failed and is retried after %d ms: at the fault the page was state %#lx protect %#lx type %#lx (region %p +%#llx); %ld retried so far\n",
                acc == 1 ? "write" : acc == 8 ? "execute" : "read", (unsigned long long)va, (unsigned long long)(pc - c1_lo), (unsigned long)GetCurrentThreadId(), ms,
                (unsigned long)m0.State, (unsigned long)m0.Protect, (unsigned long)m0.Type, m0.BaseAddress, (unsigned long long)m0.RegionSize, (long)r);
            return 1;
        }
        if (ms) Sleep(1); else SwitchToThread();
    }
    fprintf(stderr, "c1d: %#llx at NXN1X.exe+%#llx stayed inaccessible for 200 ms: state %#lx protect %#lx type %#lx (region %p +%#llx, allocation %p): reported\n", (unsigned long long)va,
            (unsigned long long)(pc - c1_lo), (unsigned long)m.State, (unsigned long)m.Protect, (unsigned long)m.Type, m.BaseAddress, (unsigned long long)m.RegionSize, m.AllocationBase);
    return 0;
}
static __thread int c1_in;   // a fault inside the report itself is not reported again
static LONG WINAPI c1_veh(EXCEPTION_POINTERS *ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || c1_in) return EXCEPTION_CONTINUE_SEARCH;
    uint64_t pc = (uint64_t)(uintptr_t)ep->ExceptionRecord->ExceptionAddress;
    if (pc < c1_lo || pc >= c1_hi) return EXCEPTION_CONTINUE_SEARCH;   // guest code and other modules: NCE's and theirs
    if (c1d_retry(ep, pc)) return EXCEPTION_CONTINUE_EXECUTION;   // c1d
    LONG k = InterlockedIncrement(&c1_avn);
    if (k <= 4096) {   // (more would mean it sees faults that are handled after it: then it stops writing)
        char kind[96]; snprintf(kind, sizeof kind, "access violation no handler before this one took (number %ld)", (long)k);
        c1_in = 1; c1_write(ep, kind, "nxn1x_av", k <= 2 ? (int)k : 0); c1_in = 0;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
static DWORD WINAPI c1_late(LPVOID arg) {
    (void)arg; Sleep(20000);
    HMODULE me = GetModuleHandleA(NULL); const IMAGE_DOS_HEADER *dh = (const IMAGE_DOS_HEADER *)me;
    const IMAGE_NT_HEADERS *nh = (const IMAGE_NT_HEADERS *)((const char *)me + dh->e_lfanew);
    c1_lo = (uint64_t)(uintptr_t)me; c1_hi = c1_lo + nh->OptionalHeader.SizeOfImage;
    PVOID h = AddVectoredExceptionHandler(0, c1_veh);
    fprintf(stderr, "c1c: vectored handler added last (%p), for access violations in NXN1X.exe %#llx-%#llx: %s\\nxn1x_av_%lu.*\n", h, (unsigned long long)c1_lo,
            (unsigned long long)c1_hi, c1_dir, (unsigned long)GetCurrentProcessId());
    return 0;
}
static void c1_catch_install(void) {
    const char *e = getenv("DM_CRASH_CATCH"); if (e && *e && atoi(e) == 0) return;
    static volatile LONG done; if (InterlockedExchange(&done, 1)) return;
    DWORD k = GetModuleFileNameA(NULL, c1_dir, sizeof c1_dir); if (!k || k >= sizeof c1_dir) { strcpy(c1_dir, "."); }
    else { char *s = strrchr(c1_dir, '\\'); if (s) *s = 0; }
    c1_prev = SetUnhandledExceptionFilter(c1_filter);
    { HANDLE th = CreateThread(NULL, 0, c1_late, NULL, 0, NULL); if (th) CloseHandle(th); }   // c1c
    fprintf(stderr, "c1: crash catcher set (%s\\nxn1x_crash_%lu.*; filter before it: %p)\n", c1_dir, (unsigned long)GetCurrentProcessId(), (void *)c1_prev);
}
#else
static void c1_catch_install(void) {}
#endif

// ABI retained for unchanged shader/presentation guard; inactive in F4.
int nvnh_work_get(void) { return atomic_load(&dm_host_work); }
void nvnh_work_restore(int n) { atomic_store(&dm_host_work,n); }
