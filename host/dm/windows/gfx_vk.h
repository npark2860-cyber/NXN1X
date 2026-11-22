// Windows port (from the Android port's): the Vulkan backend behind the replay interface of nvn_mtl.h (what nvn_mtl.m implements on Metal).
// Files: gfx_vk_core.c (device, memory pools, submissions, completion), gfx_vk_tex.c (formats, textures, samplers, registration tables),
// gfx_vk_shader.c (shader bundle, modules, layouts), gfx_vk_exec.c (command replay: passes, pipelines, descriptors, draws, copies), gfx_vk_present.c (window, swapchain).
// Version 1 is correct-first: whole-pipeline barriers between passes, full pipelines with a small dynamic state set, descriptor sets per draw.
#pragma once
#define VK_NO_PROTOTYPES
#include "video_core/nvn/nvn_shader_bridge.h"   // NXN1X (lane B): Eden shaders
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <limits.h>   // (S1d (parallel recording): INT_MIN, the dry pass's "no slot named")
#include "dm.h"
#include "nvn_obj.h"
#include "nvn_mtl.h"
#include "nvn_foundation_diag.h"
#if defined(DM_NXN1X)
#include "nvn_memory_content.h"
#endif

// Foundation F0: NVN2-only observation; never changes resource/sync decisions.
int vk_f0_enabled(void);
void vk_f0_begin(nvf0_scope *scope, uint32_t kind, uint64_t bytes);
void vk_f0_end(nvf0_scope *scope, int32_t result);
VkResult vk_f0_result(VkResult result, const char *site, uint32_t part);
void vk_f0_report(void);
void vk_f0_present_done(uint64_t ticket);

#define VKLOG(...) LOG("[vk] " __VA_ARGS__)
#define VKRED_ONCE(id, ...) do { static int once_##id; if (!once_##id++) LOG("[RED] [vk] " __VA_ARGS__); } while (0)
#define VKCHK(x) do { VkResult r_ = (x); if (r_ < 0) die("[vk] %s failed: %d (%s:%d)", #x, r_, __FILE__, __LINE__); } while (0)   // (the site: several share one text)

// ---- Vulkan entry points (loaded from vulkan-1.dll: the loader, which picks the driver VK_DRIVER_FILES names, Turnip by default) ----
#define VK_FNS(X) \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceFormatProperties) X(vkGetPhysicalDeviceImageFormatProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) \
    X(vkEnumerateDeviceExtensionProperties) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) X(vkGetPhysicalDeviceSurfacePresentModesKHR) \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR)
#define VK_DEV_FNS(X) \
    X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkBindImageMemory) X(vkGetBufferMemoryRequirements) \
    X(vkGetImageMemoryRequirements) X(vkCreateImage) X(vkDestroyImage) X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateSampler) X(vkDestroySampler) X(vkMapMemory) X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkUnmapMemory) X(vkResetCommandPool) X(vkResetCommandBuffer) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) \
    X(vkCreateSemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) X(vkCreateFence) X(vkWaitForFences) X(vkResetFences) X(vkDestroyFence) \
    X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments) X(vkCmdResolveImage) \
    X(vkCreateRenderPass) X(vkCreateRenderPass2) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetBlendConstants) X(vkCmdSetStencilReference) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) \
    X(vkCmdSetDepthBias) X(vkCmdSetDepthBounds) X(vkCmdSetLineWidth) X(vkCmdDraw) X(vkCmdPushConstants) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount) X(vkCmdDispatch) X(vkCmdDispatchIndirect) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateGraphicsPipelines) X(vkDestroyPipeline) X(vkCreateComputePipelines) X(vkCreatePipelineCache) \
    X(vkGetPipelineCacheData) X(vkCreateDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateBufferView) X(vkDestroyBufferView) X(vkGetBufferDeviceAddress) \
    X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults) X(vkCmdBeginQuery) X(vkCmdEndQuery) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) \
    X(vkCmdBindTransformFeedbackBuffersEXT) X(vkCmdBeginTransformFeedbackEXT) X(vkCmdEndTransformFeedbackEXT) \
    X(vkGetRefreshCycleDurationGOOGLE) X(vkGetPastPresentationTimingGOOGLE)
#define VK_DECL(f) extern PFN_##f f;
VK_FNS(VK_DECL) VK_DEV_FNS(VK_DECL)

// ---- device ----
typedef struct {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf;
    VkPhysicalDeviceProperties props; VkPhysicalDeviceMemoryProperties mem;
    VkPipelineCache pcache;
    int has_bc, has_astc, has_gs, has_depth_clip_ctl, has_custom_border, has_robust2, has_xfb, has_bda, has_imageless, has_dtiming, has_dynrender, has_quad, has_tess, has_dbounds, has_demote, has_depth_clip_en;
    float max_aniso;
    int has_host_import; VkDeviceSize host_align; PFN_vkVoidFunction get_host_ptr_props;   // VK_EXT_external_memory_host (gfx_vk_core.c pool_zero_copy)
} vkdev_t;
extern vkdev_t V;
static inline int vk_is_n1x(void) { return V.props.vendorID == 0x10de && V.props.deviceID == 0x2e03; }
uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid);
void vk_init(void);   // once, from mtl_init or the first use
void vk_tex_preserve_poll(uint64_t need);
int vk_tex_preserve_enabled(void);
typedef struct nvnr_submission { uint64_t fence; int slot, accepted_parts, active; } nvnr_submission;
void vk_sub_suspend(nvnr_submission *saved);
void vk_sub_resume(const nvnr_submission *saved);
void vk_tex_archive_prepare(void); // bounded transfer reserve, before memory pressure

// ---- memory pools (gfx_vk_core.c): the guest's pool memory is GPU memory where the driver can import it (VK_EXT_external_memory_host), else a GPU copy ----
typedef struct { uint64_t lo, hi; VkBuffer buf; VkDeviceMemory mem; void *ahb; int zero_copy; uint8_t *shadow; uint64_t serial, base, end; int flags, synced, noncoh; uint64_t stamp, stamp0, sync_epoch; const void *span; VkDeviceAddress gpu; uint64_t used, retry, last_binding_fence; VkDeviceMemory mem2; int bridge; int s48; int s48b; uint64_t dg_sync, dg_rev;
    /* poolcopy (gfx_vk_core.c, NXN1X_POOL_MODE=copy): buf/mem/gpu are the device-local copy; shadow is the CPU-mapped baseline (sbuf in smh at soff),
       rb the CPU-mapped readback landing (rbuf in rmh at roff); smem/rmem are owned memories (NULL when bound into a shared backing range pcr) */
    int dl, pcr, rb_whole, rb_noncoh; VkBuffer sbuf, rbuf; VkDeviceMemory smem, rmem, smh, rmh; VkDeviceSize soff, roff; uint8_t *rb; uint64_t rb_sub, pcv_cur, pc_wfence; struct pcvol *pv; } vkpool_t;   // sync_epoch: submission in which the copied contents were refreshed; serial: unique per pool made (caches of views over a pool's buffer key on it); [base, end): the game's pool, of which [lo, hi) is one chunk (ZC_CHUNK); flags: the NVN pool's (vk_pool_set_flags); synced, stamp, stamp0, span: a copied pool's incremental sync
vkpool_t *vk_pool_of(uint64_t addr, uint64_t len);   // the pool containing [addr, addr+len), NULL if none
int vk_buf_at(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off);   // GPU buffer + offset of a pool address range
// Tracked copies refresh at submission begin; NVN2 untracked copies may refresh on first binding, before GPU access.
void vk_pools_sync(VkCommandBuffer cb);

// ---- submissions (gfx_vk_core.c) ----
typedef struct vksub vksub_t;
VkCommandBuffer vk_cmd(void);          // the current submission's main command buffer (recording); R2 (parallel recording): this thread's recording target's
// R2 (parallel recording): the command buffer a thread records into and its generation (bumped at each begin and whenever the host's own binds make the memos
// of the bound state stale: the bind cache g_bc, s35, s45, the vertex stream memos compare it). g_vk_rt points every thread at the replay thread's target, which
// gfx_vk_core.c keeps equal to the current slot's main command buffer (vk_cmd_split, mtl_sub_begin, mtl_sub_commit); a worker may later point its own at its
// own primary command buffer. g_cmd_gen keeps its old name as a macro
// R3 (parallel recording): ar is the arena the thread's per-draw allocations come from (vk_scratch, vk_alloc_set: scratch blocks and descriptor pools,
// gfx_vk_core.c struct vk_arena); NULL = the current slot's own arena, as before (the replay thread's target keeps it NULL)
struct vk_arena;
typedef struct vk_rec_target { VkCommandBuffer cb; uint64_t gen; struct vk_arena *ar; } vk_rec_target;
extern _Thread_local vk_rec_target *g_vk_rt;   // R2 (parallel recording)
#define g_cmd_gen (g_vk_rt->gen)   // R2 (parallel recording): was a global (gfx_vk_core.c)
int vk_sub_open(void);                 // a submission is being recorded
VkCommandBuffer vk_upload_cmd(void);   // the current submission's upload command buffer (runs before the main one)
// R6/R7/R8 (parallel recording), R8: two locks for state the recording threads will share (design\PARALLEL_REPLAY_DESIGN.md R8; gfx_vk_core.c). Each is recursive per
// thread (a depth counter in TLS over a plain pthread mutex: one uncontended lock and unlock at the outermost take only). Order: vk_par_lock -> the pool lock -> g_img_mu
// -> vk_up_lock. vk_par_lock (outer): vk_tex's upload / first-use tail (gfx_vk_tex.c) and the vertex mirrors (gfx_vk_exec.c vbm_bind, vk_vbm_gpu_wrote). vk_up_lock
// (leaf): the upload command buffer, from vk_upload_cmd to the last command recorded into it (with the per-buffer memos that compare g_up_gen), and vk_tex_prepare's
// first transition; nothing that takes another lock (the pool lock, g_img_mu, vk_par_lock) may run while it is held
void vk_par_lock(void); void vk_par_unlock(void);
void vk_up_lock(void); void vk_up_unlock(void);
// S1a (parallel recording): recording worker threads (design\PARALLEL_REPLAY_DESIGN.md Stage 1; gfx_vk_core.c). DM_VK_PARALLEL=<workers> (0..6, default 0: no thread, pool,
// arena or context is made and nothing below runs). Each worker has its own recording contexts (gfx_vk_exec.c struct rec_ctx, struct rec_eden with its own Eden rings),
// recording target (g_vk_rt), and per submission slot a command pool, primary command buffers and an arena, all rewound in that slot's slot_recycle. Nothing submits
// a worker's command buffer yet. A chunk is one primary command buffer of a part's ordered list; the slot keeps the list (vk_chunk_*), recycled with the slot
typedef struct vk_chunk { VkCommandBuffer cb; int worker; int done; } vk_chunk;   // done: 0 recording, 1 ended, -1 failed (cb not usable)
int vk_par_workers(void);                                  // workers running (0: parallel recording off)
void vk_par_run(int w, void (*fn)(void *), void *arg);     // queue fn(arg) on worker w (its g_rec, g_rec_e, g_vk_rt are its own); blocks while its queue is full
void vk_par_drain(void);                                   // wait until every worker's queue is empty and its last job returned
int vk_sub_slot(void);                                     // the open submission's slot (-1 between submissions)
VkCommandBuffer vk_par_chunk_begin(int slot);              // worker thread: begin its next primary of that slot as its recording target (NULL if the begin failed)
VkResult vk_par_chunk_end(void);                           // worker thread: end it; its target has no command buffer until the next begin
int vk_chunk_add(int worker);                              // replay thread: append a chunk to the open slot's list (cb unknown yet), its index
void vk_chunk_finish(int slot, int i, VkCommandBuffer cb, int ok);   // any thread: chunk i of that slot recorded (ok) or not
int vk_chunk_wait_part(void);                              // replay thread: wait for every chunk added since the part's start; the next part starts after them; chunks that failed
const vk_chunk *vk_chunk_list(int *n);                     // replay thread: the open slot's chunks in order
// (gfx_vk_exec.c) a recording context pair made as the replay thread's are at start-up, and the calling thread's contexts set to it (NULL, NULL: the replay thread's)
void *vk_rec_ctx_new(void); void *vk_rec_eden_new(void); void vk_rec_ctx_bind(void *rec, void *rec_e); const void *vk_rec_ctx_cur(void); void *vk_rec_eden_cur(void);
// S1c (parallel recording): a worker's counters added into the replay thread's and zeroed (replay thread, with that worker idle: after vk_chunk_wait_part joined its chunks)
void vk_rec_ctx_fold(void *rec, void *rec_e);
// S1c: DM_VK_PARALLEL_SELFTEST=2 (gfx_vk_core.c par_selftest): the shared-state paths of recording raced on every worker at once, no draw recorded (gfx_vk_exec.c).
// begin (replay thread) sets up nw workers' cases; run (worker w, all at once) races them; check (replay thread, after vk_par_drain) verifies and undoes them: 1 pass
void vk_rec_stress_begin(int nw); void vk_rec_stress_run(int w); int vk_rec_stress_check(void *const *rec, void *const *rec_e, char *out, size_t n);
// S1b (parallel recording): the deferred uniform updates' ranges (lo, hi pairs, at most max; *cap = the list's capacity), read only, for the dry pass (gfx_vk_exec.c)
int vk_ubd_ranges(uint64_t *lohi, int max, int *cap);
// S1d (parallel recording): the deferred uniform update list as data (design R5). An UPDATE_UB is split into what the dry pass does in stream order (vk_sub_update_pre:
// the range's check and the copy of its bytes into the submission's scratch memory) and the list operation (vk_sub_update_apply: the exact-range supersede, the land of a
// full list, the append; returns the superseded entry's index or -1, *landed set when it landed). vk_sub_update is the two in a row (the default path). Each thread works on
// its current list (TLS; the replay thread's main list unless a segment's is set): vk_ubl_new makes a segment list, vk_ubl_snapshot copies the main list into one,
// vk_ubl_use makes it the thread's (NULL: the main list again), vk_ubl_log_apply applies one logged update to the thread's list by its superseded index (no search; with
// check: verified, 0 when the index or the list's room does not hold, the update then applied by search), vk_ubl_same compares a list with the main one. The generation
// the memos compare (g_ubd_gen) is the thread's list's, drawn from one counter for all lists, so a memo never matches across lists
typedef struct { uint64_t dst, n; VkBuffer sb; VkDeviceSize so; const uint8_t *p; int ok; } vk_ubd_pre;
struct vk_ubl;
int vk_sub_update_pre(uint64_t dst, const void *data, uint64_t n, vk_ubd_pre *u);
int vk_sub_update_apply(const vk_ubd_pre *u, int *landed);
struct vk_ubl *vk_ubl_new(void); void vk_ubl_snapshot(struct vk_ubl *l); void vk_ubl_use(struct vk_ubl *l);
int vk_ubl_log_apply(const vk_ubd_pre *u, int sup, int check); int vk_ubl_same(const struct vk_ubl *l);
void vk_ubl_copy(struct vk_ubl *dst, const struct vk_ubl *src);   // S1e (parallel recording): a segment list copied into a worker's own (a fresh generation)
// S1e (parallel recording): segments recorded on the workers (design\PARALLEL_REPLAY_DESIGN.md S1e; gfx_vk_core.c, gfx_vk_exec.c seg_par_*). A worker records a segment
// with a frozen view of what the replay thread's later records may change under it (vk_par_view, gfx_vk_core.c: the open submission's GPU-write list, the GPU writes in
// flight, the landed uniform updates in flight, the occlusion counting state, the submission's write sequence) and puts its post ops into the view; the replay thread
// puts them back at the segment's place among the submission's post ops when the part is joined (vk_par_merge_posts)
typedef struct vk_par_view vk_par_view;
vk_par_view *vk_par_view_new(void);
int vk_par_handout(int worker, vk_par_view *v);   // replay thread, no pass open: the view taken, the segment's post placeholder, the R chunk ended, the W chunk listed (its index), a new R chunk begun
void vk_par_wk_begin(vk_par_view *v); void vk_par_wk_end(void);   // worker thread: its segment's view (post ops, occlusion, GPU-write lookups go to / come from it)
void vk_par_merge_posts(vk_par_view *v);   // replay thread, the part joined: the view's post ops in place of the segment's placeholder
int vk_par_pick(void);                     // the worker with the fewest jobs queued or running
int vk_par_help_one(void);                 // replay thread: one job still queued on a worker run here instead (as the helper context); 0 none
void vk_par_help_run(void (*fn)(void *), void *arg);   // replay thread: fn(arg) run here as the helper context (its own recording contexts, target, pools and arenas)
int vk_par_on_worker(void);                // this thread records as a worker (or as the helper)
int vk_chunk_wait_all(void);               // replay thread: wait for the open part's chunks (not advancing the part); the failed ones
void vk_chunk_get(int i, vk_chunk *out);   // replay thread: chunk i of the open slot
void vk_seg_part_join(void);               // gfx_vk_exec.c: the part's worker segments joined (helped, failures re-recorded here, post ops merged); called by vk_cmd_split / mtl_sub_commit
void vk_par_pin_self(int worker);          // gfx_vk_present.c (Windows): the calling worker thread on the P-cores (DM_VK_PARALLEL_PIN=0 off)
// S1e: worker lookups (gfx_vk_tex.c). While g_vk_tex_wk is set on a thread, vk_tex makes no image, restores no archive, upgrades none and refreshes no sparse tiles: it returns
// NULL with g_vk_tex_refused set (1 new, 2 archived, 3 storage upgrade, 4 sparse tiles), and the worker's segment is recorded again on the replay thread. vk_tex_par_hold:
// image objects dropped meanwhile are kept until vk_tex_par_release (a worker may hold one: P_att, a view lookup)
extern _Thread_local int g_vk_tex_wk, g_vk_tex_refused;
void vk_tex_par_hold(void); void vk_tex_par_release(void);
extern uint64_t g_ubl_seg_lands;   // lands of a segment's list (a land inside a segment: never, by the dry pass's bound; [RED] in the S1d log)
extern _Thread_local uint64_t *g_ubd_genp;
#define g_ubd_gen (*g_ubd_genp)
// S1d: timestamp reports' query slots named by the dry pass. vk_sub_ts_next: the slot the open submission's next timestamp query takes (-1 none: no query pool), *cap one past
// its last; vk_ts_replay_on: DM_VK_TS_REPLAY (then a timestamp's slot depends on GPU writes at replay: the dry pass names none); vk_sub_counter_q: mtl_sub_counter with the
// dry pass's slot q for a timestamp (taken; vk_ts_slot_off counts slots that were not the next one)
int vk_sub_ts_next(int *cap); int vk_ts_replay_on(void); void vk_sub_counter_q(uint64_t addr, uint32_t type, int q); uint64_t vk_ts_slot_off(void);
int vk_gmem_map(VkBuffer *buf, VkDeviceSize *size);
void vk_gml_note(const uint32_t *log);   // its write log, copied back after the submission (gfx_vk_core.c gml_copy_back)   // raw global memory: the memory pool table (dmsh_abi.h DMSH_GMEM_MAP_*), 0 when the device has no buffer device address
VkBuffer vk_atomic_scratch(VkDeviceSize size);
void *vk_scratch(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off);   // host-visible memory that lives until this submission completes
void vk_tex_mem_stats(uint64_t out[5]);   // NVN2 memory report (gfx_vk_core.c): live images, their bytes, evicted over their memory, evicted under memory pressure, NVN2 sparse layers' bytes
int vk_sparse_tiles_ok(void);   // NVN2 2D sparse tiles (gfx_vk_tex.c s2tile_*): sparse binding, sparseResidencyImage2D enabled and the standard 2D block shape (logged once)
void vk_image_sparse_reqs(VkImage img, uint32_t *n, VkSparseImageMemoryRequirements *r);   // vkGetImageSparseMemoryRequirements (0 when the entry point is missing)
uint64_t vk_tex_evict_mem(uint64_t want, uint64_t idle);
uint64_t vk_tex_evict_oom(uint64_t want); // actual allocation failure: may reclaim the recent working set, preserving contents
void vk_tex_mem_attrib(void);   // NVN2 diagnostic: the non-sparse images' device bytes by class and the 20 largest (nvn2_pressure)   // NVN2 memory pressure (gfx_vk_core.c nvn2_pressure): idle re-makeable images dropped, up to want bytes; 0 when g_img_mu is busy
int vk_sparse_bind_ok(void);   // NVN2 sparse layers: the device binds sparse image memory (sparseBinding, a graphics queue with sparse binding, vkQueueBindSparse)
int vk_bind_sparse_wait(const VkBindSparseInfo *bs);   // vkQueueBindSparse under the queue lock, waited for (2 s); 0 = done
void vk_queue_lock(void);
void vk_queue_unlock(void);
void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m);   // destroyed once the GPU has passed the current submission
void vk_defer_free_image(VkImage i, VkDeviceMemory m);
VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub);   // NULL on failure; *sub: a slot of a shared block, or NULL (memory of its own)
uint64_t vk_image_memory_growth(const VkMemoryRequirements *req, uint32_t type); // estimated NEW allocation bytes; zero when an existing free slot fits
void vk_defer_free_image_sub(VkImage i, void *sub);
uint64_t vk_image_release_completed(VkImage i, VkDeviceMemory m, void *sub, uint64_t bytes); // caller has proved all uses completed
uint64_t vk_image_allocation_size(void *sub, uint64_t bytes, uint32_t *live_slots);
uint64_t vk_image_trim_empty(void);
uint64_t vk_nvn2_memory_need(uint64_t extra); // bytes to reclaim before a large allocation (NVN2 only)
void vk_defer_free_view(VkImageView v);
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l);    // from this submission's descriptor pools
int vk_oq_real(void);   // r52: SAMPLES_PASSED reports are measured (r50's occlusion queries; 0 with DM_VK_OQ=0, whose fixed counter is not a measurement)
// occ-s0 / occ-s1 (design\OCCLUSION_REDESIGN.md steps 0 and 1; gfx_vk_core.c "occ-s0 / occ-s1"). g_occdiag: DM_VK_OCCDIAG=1 (live; log only). g_occ2: DM_VK_OCC2=s1 (device init):
// one occlusion query per draw by stream serial (vk_occ2_begin right after vk_oq_draw, vk_occ2_end on every exit of the draw paths), values written as the old path's.
// vk_occ2_serial: the calling thread's stream serial (gfx_vk_exec.c g_oser: the next draw record's); vk_occ2_handout: a worker segment's draws [ser0, ser0 + n) made ready (replay
// thread, at the hand-out); vk_occd_read: a GPU-side read of [a, a + n) checked against report words not landed (kind 0 uniform shadow, 1 copy source, 2 UBO bind, 3 SSBO bind)
extern int g_occdiag, g_occ2;
extern uint64_t g_vk_pends;   // gfx_vk_exec.c: the replay thread's pass ends
void vk_occ2_begin(VkCommandBuffer cb, uint64_t ser); void vk_occ2_end(void); void vk_occ2_handout(uint64_t ser0, uint64_t n); uint64_t vk_occ2_serial(void); uint64_t vk_occ2_ser_off(void);
void vk_occd_read(uint64_t a, uint64_t n, int kind); void vk_occd_exec_log(double frames);
// occ-s2 / occ-s3 (design\OCCLUSION_REDESIGN.md steps 2 and 3; gfx_vk_core.c "occ-s2 / occ-s3"). g_occ2 = 2 (DM_VK_OCC2=s2): samples-passed reports resolved on the GPU in stream order
// (vk_occ2_pass_end: a pass of the replay thread ended, a resolve point if reports are pending); 3 (s3): and render conditions on them evaluated by the GPU. vk_occ2_rseq: the replay
// thread's next report sequence (a report's record and predicate index); vk_occ2_rseq_chk: the dry pass's name for the report being committed; vk_occ2_job_join / join_end: a part's
// worker segments joined (their begun queries; deferred resolves). vk_occ3_pred: a GPU condition's predicate (q: report rx against report ry or the constant yc), 1 with its byte offset
// in the resolve buffer, 0 = evaluated on the CPU; vk_occ3_wait: the CPU's last resort (wait for report r's submission; 1 landed); vk_occ3_begin / end: around a gated draw or clear
typedef struct { uint64_t rx, ry, yc; int yrep; } vk_occ3_q;
void vk_occ2_pass_end(void); uint64_t vk_occ2_rseq(void); void vk_occ2_rseq_chk(uint64_t want); void vk_occ2_job_join(vk_par_view *v, int ok); void vk_occ2_join_end(void);
int vk_occ3_pred(const vk_occ3_q *q, uint32_t *off); int vk_occ3_wait(uint64_t r); void vk_occ3_begin(VkCommandBuffer cb, uint32_t off, int inv); void vk_occ3_end(VkCommandBuffer cb);
void vk_occ3_exec_log(double f, const uint64_t *core);
// occ-s3fast (design\OCCLUSION_REDESIGN.md 6.10): resolve points only where something reads the reports. vk_o3p_want (gfx_vk_exec.c, the plan made by a pre-walk of each list): at
// stream serial ser with reports [plo, phi) pending, 1 = resolve now (a GPU condition or GPU reader of one of them comes before the next opportunity the plan knows of; 1 always
// without a plan). vk_occ2_opp: a record boundary with no pass open (an opportunity). vk_occ2_unres_in: [a, a + n) holds a report word not resolved yet (replay thread).
// vk_o3f_cond_note: a CPU-evaluated render condition's outcome on a timestamp / flag word (diagnostic: flips by frame-time spikes)
int vk_o3p_want(uint64_t ser, uint64_t plo, uint64_t phi); void vk_occ2_opp(void); int vk_occ2_unres_in(uint64_t a, uint64_t n); void vk_o3f_cond_note(uint64_t addr, int kind, int skip);
// vk_o3r_cond / vk_o3r_flush (DM_VK_OCC_CONDDIAG=1, diagnostic): a CPU-evaluated condition's words read back by the GPU at its place in the stream (flushed where no pass is open)
void vk_o3r_cond(uint64_t addr, int mode, int skip, int cls); void vk_o3r_flush(void);
uint64_t vk_fence_done(void);   // highest submission with GPU work AND CPU copy-back/mirrors/post ops completed

// ---- formats and textures (gfx_vk_tex.c) ----
typedef struct { VkFormat vk; VkComponentMapping swz; uint8_t kind; uint8_t depth, stencil, compressed, is_int, is_sint, bw, bh, bpb; } vkfmt_t;
const vkfmt_t *vk_fmt(int nvn_format);   // NULL when the format has no Vulkan counterpart on this device
typedef struct vkimg vkimg_t;
VkImageView vk_rescaled_view(vkimg_t *im, const tex_t *tex, const view_t *view, int dim, int arrayed, int attachment);
void vk_before_native_readback(const vkimg_t *im);
int vk_rescale_eligible(vkimg_t *im);
#define VK_TS_SCALE_WRITEBACK 0xC0FFEFu
void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws);
void vk_rescale_blit(VkCommandBuffer cb, vkimg_t *im, int enlarge, VkImageAspectFlags aspects, VkRect2D native);
struct vkimg {
    vkimg_t *scaled;
    unsigned scaled_valid, scale_rating;
    int scale_caps;
    uint64_t scale_frame;

    uint64_t residency_id; // image incarnation independent of optional F2 observation
    uint64_t content_id; // F2 stable observation incarnation, not a resource handle
    struct nvn_img_archive *archive; // lossless backing; a bounded clean copy may remain while resident
    uint64_t content_serial;         // increment before every possible write; validates retained file contents
    struct archive_save_job *preserve; // pins record and GPU source until collection
    uint64_t preserve_after;
    int preserve_orphan;
    uint64_t allocation_bytes;       // actual VkMemoryRequirements, not guest layout / sparse virtual size
    uint64_t residency_key;          // bounded eviction history; never used to identify image contents
    uint32_t residency_idle;         // learned reuse interval for normal pressure; critical reclamation still uses fences
    uint8_t released_hint;           // API object finalized; a cache hint, NEVER content/lifetime proof
    uint8_t rdi_first;                // slot12 replaydiag: first use since made, 0 none yet, 1 a GPU write (attachment, clear, copy, storage), 2 a read (sampled view)
    uint16_t rdi_slot;                // slot12 replaydiag: its entry in the new-image table (gfx_vk_tex.c rdi_*) + 1, 0 untracked
    uint32_t rdi_gen;                // slot12 replaydiag: the table window it was counted in
    struct s2tile *s2t;   // NVN2 2D sparse tiles (gfx_vk_tex.c s2tile_*)
    const void *rd_mem; uint64_t rd_len, rd_gen, rd_ev; int rd_ok;
    const void *unc_pool;   // bak86: the pool of an uncompressed texture cpu_first_use took (the CPU's writes); once the GPU wrote that pool, its only copy is this image   // NVN2 s24 guard (gfx_vk_tex.c s24_readable): its storage's last "readable" answer and the generation it was taken at
    VkImageUsageFlags usage; VkImageCreateFlags cflags;   // as created (an imageless framebuffer names them)
    VkImage img; VkDeviceMemory mem; void *sub; VkFormat fmt; VkImageViewType vtype; VkImageAspectFlags aspect;
    uint32_t w, h, d, layers, levels, samples; int target, nvn_format;
    VkImageLayout layout;           // version 1 keeps every image in GENERAL after its first use
    int packaged;                   // filled from guest memory (deswizzled upload), re-uploaded when its pages change
    uint64_t pkg_stamp; const void *pkg_span; uint64_t pkg_seen; uint64_t pkg_sgen;   // pkg_sgen: NVN2, the page-table generation at the upload of a texture in a vmode-2 pool
    uint64_t pkg_vhash, pkg_vframe;   // s24 (gfx_vk_tex.c pkg_verify): sampled words of its storage at the upload, and the present it was last checked at
    uint64_t pkg_tser;   // s27 (gfx_vk_tex.c): the texture object (serial) its storage was last uploaded or checked for
    nvn_texdesc pkg_d;              // r44: the description it was uploaded with (DM_VK_TEXDUMP raw blocks)
    const uint8_t *pkg_mem; size_t pkg_len; uint64_t pkg_used; int pkg_released; uint64_t cc_ser;   // its storage, the present it was last drawn at, protection given up (vk_tex_sweep); cc_ser: r73, the last serial cpu_first_use looked at
    uint64_t used;                  // the present it was last looked up at (vk_tex: evict_overlaps)
    uint64_t made;                  // NVN2 (11-16): the present it was made at (img_evict_idle's minimum age)
    uint64_t gaddr; int refs;       // NVN2 (11-16): its storage's guest address; storage keys beyond the first that share it (sparse_share), each dropped once before the image
    uint64_t fdz_frame;             // depth images: 1 + the present at which a fragment shader that replaces depth last wrote it (gfx_vk_exec.c fdz)
    int gpu_w;                      // the GPU wrote it: an attachment, a storage image, a copy or clear destination (gfx_vk_exec.c); lin_copy keeps that until the CPU writes its storage
    int keep_gw;                    // bak91 (NVN2): kept by evict_overlaps although a newer texture was made over its memory (its contents exist only on the GPU); its bytes are in g_keep_gw_bytes
    uint64_t clr_pend, clr_fence;   // clrorder (diagnostic, 11-21, Lichii #185): a clear folded into the next pass's load op not yet begun (its sequence); the submission of its last clear recorded in the main command buffer
    int ovw;                        // relovl (11-21): a newer image over its memory was written by the GPU while it was kept (keep_gw): its contents are gone on the Switch too, dropped by ovw_sweep
    uint64_t lin_stamp; const void *lin_span; const uint8_t *lin_mem; size_t lin_len;   // lin_copy: the storage watched since the GPU wrote it
    uint64_t lin_gen;               // linear textures: the upload command buffer (g_up_gen) that last copied it from its pool (vk_tex: lin_copy)
    VkImageView full;               // a view of the whole image as created
    VkBufferView tbuf;              // buffer textures: a texel buffer view of the pool
    uint64_t last_use;              // submission fence of the last use
    int external_use;               // presented/read outside tracked NVN submissions: keep resident until its pool goes
    uint64_t pool; int64_t off, size; int flags;   // its storage (pool, offset, layout size) and NVN texture flags: images over the same memory find each other
    struct vkimg *ovl[8]; int novl;   // the other images over some of the same memory (a texture of another description over it: on the Switch it sees these texels)
    uint64_t wseq, rseen;             // when it was last rendered to (gfx_vk_tex.c g_wseq); the newest overlapping write already reported
    uint64_t wpres;                   // NVN2 diagnostic: 1 + the present (g_tex_frame) it was last rendered to as an attachment, 0 never
    nvn_texdesc kd;                   // NVN2 diagnostic: the description it was made for ([alias] stale read: what differs)
};
vkimg_t *vk_tex(const tex_t *t);                                 // the image of a texture object (created, uploaded or refreshed as needed)
vkimg_t *vk_tex_peek(const tex_t *t);                            // NVN2 diagnostic: its image if one exists (by storage key), nothing made or uploaded
void vk_tex_gpu_mark(const tex_t *t);
void vk_tex_content_changed(vkimg_t *im); // conservative write invalidation, including uploads/copies/attachments
VkImageView vk_tbuf_rows_view(const tex_t *t);
VkBufferView vk_tbuf_copy_view(const tex_t *t);   // r75: a texture buffer below the texel-buffer alignment, through an aligned copy
void vk_defer_free_bview(VkBufferView v);   // r75   // r74: a texture buffer as 2D rows of NVN_TBUF_ROW texels (gfx_vk_tex.c)   // r72: the GPU wrote this texture (an attachment, a storage image, a copy or clear destination): mtl_tex_gpu_owned
int vk_window_bottom_up(void);                                   // the game draws in lower-left origin mode: its window image is stored bottom-up (gfx_vk_exec.c)
enum vk_view_role { VK_VIEW_SAMPLED, VK_VIEW_ATTACHMENT, VK_VIEW_STORAGE };
VkImageView vk_tex_view(const tex_t *t, const view_t *v, int dim, int arrayed, int as_attachment, vkimg_t **img_out);   // a view for sampling / storage / rendering
void vk_watch(const vkimg_t *im, const char *what);   // DM_VK_WATCH (gfx_vk_exec.c)
vkimg_t *vk_tex_by_id(int tid, const view_t **view_out, const tex_t **tex_out);   // registered texture ID → image (NULL if none)
VkSampler vk_samp_by_id(int sid);
extern __thread VkFormat vk_view_fmt; extern __thread uint32_t vk_view_layers;   // the format and layer count of the view vk_tex_view last returned on this thread
VkImageView vk_dummy_view(int dim, int arrayed, int ms, int storage, int is_int);
VkSampler vk_dummy_sampler(void);
int vk_samp_cmp(int sid);   // r46: the registered sampler's compare mode (-1 not registered)
VkSampler vk_samp_cmp_variant(int sid, int want);   // r46: the sampler with compare on (want 1) or off (0), other parameters as registered
VkBufferView vk_dummy_tbuf(int storage);
void vk_end_pass_land(void);   // vk_end_pass, then the deferred uniform updates land (before copies, dispatches, the submission's end)
void vk_img_barrier_all(VkCommandBuffer cb);   // version 1: one global memory barrier (all images stay in GENERAL)
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im);   // first use: transition from UNDEFINED to GENERAL (and upload a packaged texture)

// ---- shaders (gfx_vk_shader.c) ----
enum { RK_SUPPORT, RK_CBUF, RK_C1, RK_DRV, RK_SBUF, RK_UBO_PTR, RK_SBUF_UBO, RK_TEX, RK_TBUF, RK_IMG, RK_IMG_TBUF, RK_TEX_C2A, RK_TEX_UBO, RK_TBUF_UBO, RK_IMG_UBO, RK_GMEM_MAP, RK_GMEM_LOG, RK_UNKNOWN = 255 };   // RK_GMEM_*: 15, 16 (shader_vk.py KIND)
enum { RD_UBO, RD_SSBO, RD_SAMPLED, RD_STORAGE, RD_TBUF, RD_STORAGE_TBUF, RD_IMAGE, RD_SAMPLER };
enum { RS_NONE, RS_TEX, RS_SLOT, RS_ID, RS_UBO };
typedef struct __attribute__((packed)) { uint8_t kind, desc, dim, flags; uint16_t binding, count; int16_t reg; uint8_t samp, fmt; int32_t off, samp_arg; } dmsv_bind;
_Static_assert(sizeof(dmsv_bind) == 20, "dmsv_bind");
typedef struct {
    uint64_t key; int stage;          // NVN stage: 0 vertex 1 fragment 2 geometry 3 tess control 4 tess eval 5 compute
    VkShaderModule mod;
    VkDescriptorSetLayout dsl; int set;   // the set this stage's bindings were moved to (vertex 0, fragment 1, geometry 2, compute 0)
    int nbind; const dmsv_bind *b;
    const uint8_t *c1; uint32_t c1_len; VkBuffer c1_buf; VkDeviceSize c1_off;
    uint32_t local[3];
    uint8_t dyn[96]; int ndyn;          // bindings of uniform buffers bound with a dynamic offset (gfx_vk_exec.c stage_set), and their count
    int xfb; uint32_t xfb_stride[4];    // a vertex shader that captures transform feedback (its SPIR-V has the Xfb execution mode), and each buffer's stride
    const struct nvn_xfb_layout *xfb_layout;   // s30b, Eden mode: the layout its control section declares (gfx_vk_shader.c vk_shader_xfb), which also sets the two above
    uint32_t *spv, nspv;                // its SPIR-V as the module was made from it, for the variants below
    struct { uint64_t um, im; uint32_t xo; int fd; VkShaderModule mod; uint32_t xp, xr; } var[8]; int nvar;   // variants whose sampled bindings (bit = index in b) of an integer view read uint (um) / int (im) texels,
                                                                                    // that declare the vec4 outputs xo (bit = location) the next stage reads, and (fd) that write gl_FragDepth = gl_FragCoord.z
                                                                                    // (xp, geometry: those of xo it passes on from the vertex shader, spv_gs_pass, read at xr; vertex: outputs also written at xr, spv_vs_dup)
    int layer_out;                      // r76: a vertex, geometry or tessellation shader that writes gl_Layer (BuiltIn Layer)
    int depth_repl;                     // a fragment shader that writes gl_FragDepth (DepthReplacing / DepthGreater / DepthLess)
    uint32_t in_locs, out_locs;         // the vec4 (or per-vertex vec4) inputs and outputs it declares, by location
    uint64_t dref[2];                   // r46: bindings 0-127 whose image type is a depth image (the translator's shadow samplers, read with Dref)
    int setmemo;                        // gfx_vk_exec.c stage_set: 1 its set may be memoized per stage, 2 not (it reads handles or addresses from uniform buffer memory), 0 not known yet
    struct nvn_eden_shader *eden;       // NXN1X (lane B): the shader for Eden's recompiler (video_core/nvn/nvn_shader_bridge.h); with it the fields above from mod on are unused
    uint32_t vpull_inputs; int vpull_known;   // vtxpull (gfx_vk_eden.inc): the generic inputs this vertex shader loads, known once an Eden pipeline with it was ready
    int eden_sw;                        // R6/R7/R8 (parallel recording), R7 (gfx_vk_eden.inc eden_sw_note): this shader's storage writes in the Eden pipelines looked at with it so far,
                                        // from their binds: 0 not known yet, 1 none, 2 writes storage (an image, a storage texel buffer, a written storage buffer); sticky at 2
    // fstage (gfx_vk_eden.inc, DK Bananza 11-22): a fragment shader's written storage buffers seen bound from imported pool chunks (learned by the replay thread at its
    // draws' bindings: Eden bind cbuf, offset, array index, indirect offset, atomic); fst_state 0 none seen, 1 staged at its direct draws, 2 off for the run (reason logged)
    struct { uint32_t cbuf, off, k, ind; uint8_t atomic; } fst_b[4]; int fst_n, fst_state, fst_unused; uint64_t fst_frame; uint32_t fst_in_frame;
} vkshader_t;
vkshader_t *vk_shader(uint64_t key, int stage);   // NULL when the bundle has no such shader (Eden mode: when SetShaders gave no readable code for the key)
int vk_shader_stage(uint64_t key);   // the shader's NVN stage, -1 unknown
int vk_eden_mode(void);   // NXN1X (lane B): 1 = NVN shaders go through Eden's recompiler and pipeline cache (default), 0 = the shaders.dmsv bundle (DM_VK_EDEN=0, bring-up A/B only)
VkShaderModule vk_shader_module(vkshader_t *s, uint64_t um, uint64_t im, uint32_t xo, int fd, uint32_t xp, uint32_t xr);   // the module, or its variant (integer views; extra outputs, xp of them passed on from the vertex shader; fragment depth written)
void vk_shaders_open(const char *path);
VkBuffer vk_support_buffer(void);
VkBuffer vk_zero_buffer(void);
VkBuffer vk_wsink_buffer(void);   // rptfix: written bindings with nowhere to go (gfx_vk_shader.c)

// ---- replay (gfx_vk_exec.c) ----
void vk_exec_reset(void);   // per submission
void vk_end_pass(void);     // close an open render pass

// ---- presentation (gfx_vk_present.c) ----
void vk_present_init(void);
uint64_t vk_present(const tex_t *t, double period, const int crop[4]);

// F1 observes actual allocation boundaries without patching the Vulkan dispatch.
#if defined(DM_NXN1X)
#include "nvn_memory_observe.h"
#else
#define NVNH_HOST_PROPERTIES(f,d,t,p,r) (f)(d,t,p,r)
#endif

uint64_t vk_tex_reclaim_completed(uint64_t want); // owner-safe cheap cohorts, no preservation I/O
uint64_t vk_tex_recover_pool(uint64_t want); // bounded pending preservation; caller releases pool/submission locks
void vk_tex_trim_released(uint64_t budget); // bounded recreatable cache, before submission locks
