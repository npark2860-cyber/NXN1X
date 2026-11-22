// The host NVN layer as NXN1X sees it (dm/nxn1x): with nvnh_vk.h (the Vulkan device handover), the only headers NXN1X's C++ includes from the host tree.
// The host's NVN implementation (dm/gfx/nvn) and its Vulkan backend (dm/windows/gfx_vk_*.c) are built unchanged in substance into the static library nvn_host;
// this file is the seam: what NXN1X calls, and the hooks through which NXN1X supplies what the host's own runtime (dm/host) supplied in the native host.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "../windows/nvn_memory_content.h" // F2 versioned, Vulkan-free observation ABI
#ifdef __cplusplus
extern "C" {
#endif

// ---- entry (NVN_HOST_BRIDGE.md §4.1) ----
// Number of NVN functions in the host's name table (nvn_names, dm/gfx/nvn/names.txt order)
unsigned nvnh_nfunc(void);
int nvnh_xcx_legacy_sectors(void);
const char *nvnh_name(unsigned i);
// &nvn_slot[i]: what the gate puts in x16 before calling (*(void **)(slot + 8)), as the host's own thunks do (nvn_thunks.S); the unimplemented and big-stack entries read x16
void *nvnh_slot(unsigned i);
// 1 when the layer counts calls (DM_NVN_COUNT=1, or DM_NVN_TSV with its call rows): the gate's entries then add 1 to nvn_slot[i].cnt as the host's counting thunks do
int nvnh_counting(void);
// The gate's entries, one per function (gates[i] for nvn_names[i]); once set, nvnBootstrapLoader / nvnDeviceGetProcAddress hand these out instead of the host's thunks
void nvnh_set_gates(void *const *gates);
// nvnBootstrapLoader itself (the host's h_nvnBootstrapLoader): what the gate entry for the bootstrap calls
void *nvnh_bootstrap_loader(const char *name);
// The host's take-overs by symbol (HLE_FORCE in dm/gfx/nvn: nvnBootstrapLoader and the native image functions): mangled name and implementation.
// NXN1X hooks each definition it finds in the game's modules to a gate entry calling fn.
typedef struct { const char *name; void *fn; const char *why; } nvnh_force_t;
size_t nvnh_force_entries(const nvnh_force_t **out);
// Guest calls from host code (the command memory callback): 0 = not allowed (default under NXN1X: a host function cannot call guest code from the host stack,
// a guest SVC inside it would return to the emulator's run loop above that stack). The layer then records into host-lent segments, as for a game without a callback.
void nvnh_set_guest_calls(int on);
int nvnh_guest_calls(void);
// s55: the command memory callback, deferred. NXN1X's H gate calls cb(cmd, event, size, data) as guest code right after the host function returns.
// The layer asks for it when a record does not fit (the record itself goes into a host-lent segment); 1 = it will run, 0 = refused (inside an S-gate call,
// not on a guest thread, or one already owed in this call). NULL (DM_S55=0) = never asked
typedef int (*nvnh_defer_fn)(void *cb, void *cmd, int event, size_t size, void *data);
void nvnh_set_defer_call(nvnh_defer_fn fn);
// s52: a guest-kernel sleep of ns (NXN1X sleeps the calling guest thread as svc SleepThread, releasing its core) when the thread is inside an S-gate call;
// returns 1 if it slept, 0 if not allowed there (the host then waits as before)
void nvnh_set_ksleep(int (*fn)(uint64_t ns));
// s52: the calling guest thread's core (-1 unknown) and in *waiter whether that core's scheduler wants another thread there
void nvnh_set_core_state(int (*fn)(int *waiter));

// ---- device and queue (NVN_HOST_BRIDGE.md §4.3): the handover itself is in nvnh_vk.h (Vulkan types) ----
// 1 once the layer's vk_init ran in this process (on any device): its pools, caches and threads stay for the process, so a later boot cannot use it
int nvnh_vk_active(void);

// ---- runtime hooks the native host's dm/host supplied ----
// Guest clock in nanoseconds (nvnDeviceGetCurrentTimestampInNanoseconds, GPU timestamps); default: the host's monotonic clock (now_ns)
void nvnh_set_guest_clock(uint64_t (*fn)(void));
// Write tracking (nvn_dirty.c registers them when the first memory pool is tracked, NVN_HOST_BRIDGE.md §4.2):
// fault(addr, is_write, ctx) -> 1 when the page is the NVN layer's and was made writable (re-execute the write); prep(addr, len) before an emulator write into guest memory
typedef int (*nvnh_fault_fn)(void *addr, int is_write, void *uc);
typedef void (*nvnh_prep_fn)(void *addr, size_t len);
nvnh_fault_fn nvnh_fault_hook(void);
nvnh_prep_fn nvnh_write_prep_hook(void);
// Write watch self-test (NXN1X_NVN_WATCH_SELFTEST=1): hle_nvn.c calls it for each new pool; it runs once, on the first pool with CPU access and three pages or more
void nvnh_watch_selftest(void *mem, size_t size, int cpu_access);
// Log sink: the host writes "[dm] ..." lines to stderr; NXN1X may route them into its own log. NULL = stderr
void nvnh_set_log(void (*fn)(const char *line));
// s46pc (hle_nvn.c): the replay thread (DM_VK_ASYNC > 0) calls the first once when it starts (thread policy) and the second after each job (1 on the fastest core
// class, 0 on another, -1 unknown; *cpu the logical processor). Set by the core; NULL leaves the thread as it was
extern void (*nvnh_replay_thread_hook)(void);
extern int (*nvnh_cpu_class_hook)(int *cpu);
// s23: NXN1X's alias of the game's memory [mem, mem + size) in its backing memory (one view of all of it, never unmapped while the game runs), or NULL (a page
// unmapped, or the pages not one physically contiguous run). The Vulkan backend imports memory pools through it (gfx_vk_core.c pool_zero_copy): the driver locks
// the range it imports, and an import at the game's own address kept NXN1X from unmapping that view when the game remapped its memory (Xenoblade 3 froze, 11-12)
typedef void *(*nvnh_alias_fn)(const void *mem, uint64_t size);
void nvnh_set_import_alias(nvnh_alias_fn fn);
// F2 normal-thread resolver: sparse virtual address -> physical pool address.
// Returns the original address outside sparse pools, zero for an unmapped page.
uint64_t nvnh_content_sparse_address(uint64_t address, uint64_t length);
int nvnh_content_sparse_busy(void); // atomics only; safe in the fault notification

// ---- output (NVN_HOST_BRIDGE.md §4.5, step 6) ----
// With an output set, mtl_present hands each present to it instead of the host's own window and present thread (gfx_vk_present.c).
// image: the window texture's VkImage in GENERAL layout (0 = nothing drawn yet: give it back in its turn); the frame's work is already
// submitted on the graphics queue. crop is x, y, w, h (has_crop 0 = the whole texture); bottom_up: lower-left origin, stored upside down;
// period: the game's frame period in seconds; fence: the layer's last submission (timeline value). The callback runs on the presenting
// thread and must not wait. NXN1X calls nvnh_present_done(n) for every present, in present order, once it no longer reads the image:
// the game's acquire of that texture waits for it (mtl_present_wait), and at most 4 presents are outstanding
typedef struct {
    uint64_t n, image;
    uint32_t vk_format, width, height;
    int32_t crop[4], has_crop, bottom_up, scaled_2x;
    double period;
    uint64_t fence;
} nvnh_frame;
void nvnh_set_output(void (*fn)(const nvnh_frame *f));
void nvnh_present_done(uint64_t n);

// z1: the SDK major version read from the game's sdk module at load (nvn_gate.cpp; 0 = unknown). The NVN API version follows it (hle_nvn.c nvn_api)
void nvnh_set_sdk_major(int major);
// s69: Eden's use_asynchronous_shaders (per game) for the NVN path: 1 = a draw whose Eden pipeline is still being built is skipped (gfx_vk_eden.inc eden_async)
void nvnh_set_eden_async(int on);
// NVN2: the sdk module's build spec (1 = "spec: Ounce") and the constants of the cases of NVN2's own nvnDeviceGetInteger, read at load for Switch 2
// programs (nvn_gate.cpp ReadNvn2; known[i] = 0 where item i's case is not a constant). hle_nvn.c dm_sdk_ounce / dm_sdk_nvn2_item answer from them
void nvnh_set_nvn2(int ounce, const int *values, const unsigned char *known, unsigned count);
// NVN2 native images (nvnTextureInitializeFromNativeImage, hle_nvn.c): 1 and the picture's CPU memory, size, size in pixels, row pitch and layout (gob_h: log2 of the
// block height in GOBs, -1 = pitch-linear) for a native image handle; 0 when no image has it. Set by the core when the Switch 2 graphics server opens on an NVN boot
// (nvn_output.cpp SetNx2NativeImages); NULL clears it
typedef int (*nvnh_native_image_fn)(uint32_t handle, uint8_t **mem, uint64_t *size, uint32_t *w, uint32_t *h, uint32_t *stride, int *gob_h);
void nvnh_set_native_image(nvnh_native_image_fn fn);

#ifdef __cplusplus
}
#endif
