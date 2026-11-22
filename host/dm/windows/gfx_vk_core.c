#include "nvn_replay_lease.h"
#include "nvn_memory_transfer.h"
// Windows port (from the Android port's), Vulkan backend: device, memory pools, submissions, completion (see gfx_vk.h).
//   Memory pools: NVN gives the GPU the game's own memory (GPU address = CPU address). Where the driver offers VK_EXT_external_memory_host, each pool's own pages
//   are imported as a VkBuffer (the Android port replaced them with an AHardwareBuffer's instead): the game's pointer is GPU memory. A pool the driver refuses keeps
//   its pages and gets a GPU copy refreshed before every submission (vk_pools_sync), correct but slow, reported once.
//   Submissions: a ring of slots, each with a command pool (upload + main command buffers), descriptor pools, scratch memory and objects freed once the GPU has
//   passed it. A submission signals a timeline semaphore with its fence; a completion thread waits for it and then applies what the CPU may see of its work
//   (event writes, counter reports), as nvn_mtl.m does in its completion handler.
#define _GNU_SOURCE
#include "gfx_vk.h"
#include "nvn_hmap.h"
#include "nvn_dirty.h"
#include "nvn_timestamp.h"   // r79: counter timestamps are 614.4 MHz ticks (384 per 625 ns), as hle_nvn.c nvnDeviceGetTimestampInNanoseconds reads them
#include <dlfcn.h>
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/clonefile.h>
#include <time.h>
#include <unistd.h>

#define VK_DEF(f) PFN_##f f;
VK_FNS(VK_DEF) VK_DEV_FNS(VK_DEF)
static PFN_vkGetInstanceProcAddr p_vkGetInstanceProcAddr;
// s63 (apply_s63.py has the reasoning): GPU work recorded so far (all threads), counted by wrappers over the device's vkCmd* entry points that read or
// write memory, and over command buffer begin / end; vk_img_barrier_all leaves out a full barrier with none since the last one in the same command buffer
static _Atomic int g_s63_mode;   // s64a: default 0 (run 93: no change in fps or GPU time)
//   // DM_VK_S63 (gfx_vk_exec.c x_read, live): 1 dedup, 2 write-only source access
static uint64_t g_s63_work, g_s63_kept, g_s63_skip;
int vk_set_barrier_mode(int mode) {
    int old = atomic_load_explicit(&g_s63_mode, memory_order_relaxed);
    if (old != mode) {
        __atomic_add_fetch(&g_s63_work, 1, __ATOMIC_RELAXED);
        atomic_store_explicit(&g_s63_mode, mode, memory_order_release);
    }
    return old;
}
#define S63_WORK_0() do { if (atomic_load_explicit(&g_s63_mode, memory_order_relaxed) & 1) \
    __atomic_add_fetch(&g_s63_work, 1, __ATOMIC_RELAXED); } while (0)
// cbar2 (11-21, NPARK 07:41, design\OOT_BARRIER_DESIGN.md 10): back-to-back memory barriers merged. Cause (tsdiag, 07:39): OoT's GPU idles 20.5 ms a frame at ~300
// serialization points (~70 us each, real idle, not the timestamps), and 89.8 gaps a frame hold 2 full barriers (8.2 ms): a dispatch's r89 barrier after it, the
// guest's barrier and the next dispatch's barrier before it, or a pass's end barrier and the next pass's begin one, with no GPU work between; each drains the GPU
// on its own. A memory-only barrier (no buffer or image barriers, no dependency flags) is now held per command buffer and recorded at the next command that does GPU
// work in that buffer (the s63 wrappers below: draws, dispatches, copies, fills, clears, render pass begin / end, queries, timestamps, command buffer end; plus the
// mesh draw, gfx_vk_exec.c), as one barrier with the union of the stages and accesses of the ones held: it orders and makes visible at least what each did.
// A barrier with buffer or image barriers records what is held first, then itself. State commands (binds, dynamic state, push constants, descriptors) are not
// ordered by barriers. DM_VK_CBAR2=0: each barrier recorded where it is made (as before)
void vk_hy3_close_for(VkCommandBuffer cb);   // stage2s (gfx_vk_eden.inc): the atomic staging window closed before GPU work in cb
typedef struct { VkCommandBuffer cb; VkPipelineStageFlags src, dst; VkAccessFlags sa, da; int n; } bpend_t;
enum { BPEND_N = 16 };
static bpend_t g_bpend[BPEND_N]; static _Atomic int g_nbpend; static pthread_mutex_t g_bpend_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_bmerge_held, g_bmerge_rec, g_bmerge_abs;   // barriers held, barriers recorded from held ones, barriers absorbed into a held one
static int g_cbar2 = -1;
static int cbar2_on(void) { if (g_cbar2 < 0) { const char *e = getenv("DM_VK_CBAR2"); g_cbar2 = !(e && *e == '0'); } return g_cbar2; }
static PFN_vkCmdPipelineBarrier s63o_vkCmdPipelineBarrier;
static PFN_vkCmdPipelineBarrier g_bpend_rec;   // where held barriers are recorded (s63o_vkCmdPipelineBarrier; the self-test puts a recorder there)
// cbguard (11-22, NPARK's 00:44 crash and the 11-21 08:21 one in another title: c0000005 in nvogla64.dll under sl.interposer.dll's vkCmdPipelineBarrier
// (its export at +0x30ea8 holds the return +0x30ed0), the driver reading 0xfff00000184 from a 0xfff0000015c it took for an object: a barrier recorded into a
// command buffer that was not recording; both crashes came after cbar2 (11-21 07:41) began holding memory barriers per command buffer and recording them
// later): each command buffer's state from its begin and end (the s63 wrappers); a barrier, held or direct, into one that is not recording is dropped and
// logged (where, the command buffer, the stages) and recorded as before (log only, NPARK 11-22: the cause, the upload buffer ended while a worker still recorded into
// it, is fixed at vk_cmd_split: upswap). Command buffers it never saw begin are not looked at. DM_VK_CBGUARD=0 off
enum { CBG_N = 8192 };
static _Atomic(uintptr_t) g_cbg_key[CBG_N]; static _Atomic int g_cbg_st[CBG_N];
static int cbg_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_CBGUARD"); on = !(e && *e == '0'); } return on; }
static int cbg_slot(VkCommandBuffer cb, int insert) {
    const uintptr_t k = (uintptr_t)cb; if (!k) return -1;
    size_t i = (size_t)(((uint64_t)k * 0x9E3779B97F4A7C15ull) >> 51) & (CBG_N - 1);
    for (int p = 0; p < CBG_N; p++, i = (i + 1) & (CBG_N - 1)) {
        uintptr_t cur = atomic_load_explicit(&g_cbg_key[i], memory_order_acquire);
        if (cur == k) return (int)i;
        if (!cur) { if (!insert) return -1; uintptr_t z = 0; if (atomic_compare_exchange_strong(&g_cbg_key[i], &z, k) || z == k) return (int)i; }
    }
    return -1;
}
static void cbg_set(VkCommandBuffer cb, int recording) { if (!cbg_on()) return; const int i = cbg_slot(cb, 1); if (i >= 0) atomic_store_explicit(&g_cbg_st[i], recording, memory_order_release); }
static int cbg_refuse(VkCommandBuffer cb, const char *where, VkPipelineStageFlags s, VkPipelineStageFlags d, uint32_t nb, uint32_t ni) {   // 1: not recording, dropped
    if (!cbg_on()) return 0;
    const int i = cbg_slot(cb, 0); if (i < 0 || atomic_load_explicit(&g_cbg_st[i], memory_order_acquire)) return 0;
    static _Atomic uint64_t n; const uint64_t c = atomic_fetch_add(&n, 1) + 1;
    if (c <= 16 || !(c & (c - 1))) { VKLOG("cbguard: a %s barrier into command buffer %p, which is not recording (ended); recorded as before, log only (stages %#x -> %#x, %u buffer and %u image barriers; %llu so far; DM_VK_CBGUARD=0 off)",
        where, (void *)cb, (unsigned)s, (unsigned)d, nb, ni, (unsigned long long)c); }
    return 0;   // (NPARK 11-22: log only; the cause is fixed at its source: upswap)
}
static void bpend_flush(VkCommandBuffer cb) {   // record what is held for cb (before GPU work in it)
    if (!atomic_load_explicit(&g_nbpend, memory_order_acquire)) return;
    bpend_t p = {0}; pthread_mutex_lock(&g_bpend_mu);
    for (int i = 0; i < BPEND_N; i++) if (g_bpend[i].n && g_bpend[i].cb == cb) { p = g_bpend[i]; g_bpend[i].n = 0; atomic_fetch_sub(&g_nbpend, 1); break; }
    pthread_mutex_unlock(&g_bpend_mu);
    if (!p.n) return;
    if (cbg_refuse(cb, "held", p.src, p.dst, 0, 0)) return;   // cbguard
    VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, p.sa, p.da};
    PFN_vkCmdPipelineBarrier f = g_bpend_rec ? g_bpend_rec : s63o_vkCmdPipelineBarrier ? s63o_vkCmdPipelineBarrier : vkCmdPipelineBarrier;
    f(cb, p.src, p.dst, 0, 1, &m, 0, NULL, 0, NULL); g_bmerge_rec++;
}
static int bpend_hold(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags sa, VkAccessFlags da) {   // 1 = held (merged)
    if (!cbar2_on() || !cb) return 0;
    pthread_mutex_lock(&g_bpend_mu); int k = -1, fr = -1;
    for (int i = 0; i < BPEND_N && k < 0; i++) { if (g_bpend[i].n && g_bpend[i].cb == cb) k = i; else if (!g_bpend[i].n && fr < 0) fr = i; }
    if (k < 0 && fr < 0) { pthread_mutex_unlock(&g_bpend_mu); return 0; }   // (16 command buffers holding one: recorded where made)
    if (k >= 0) { bpend_t *p = &g_bpend[k]; p->src |= src; p->dst |= dst; p->sa |= sa; p->da |= da; p->n++; g_bmerge_abs++; }
    else { g_bpend[fr] = (bpend_t){cb, src, dst, sa, da, 1}; atomic_fetch_add(&g_nbpend, 1); }
    g_bmerge_held++; pthread_mutex_unlock(&g_bpend_mu); return 1;
}
#define S63_WORK() do { S63_WORK_0(); bpend_flush(cb); } while (0)   // (every wrapper below has its command buffer as cb)
// ootgap (diagnostic, og_* below): the GPU commands recorded into the main command buffer between timed passes, by kind (the wrappers below count them)
enum { OG_COPYBUF, OG_B2I, OG_I2B, OG_COPYIMG, OG_BLIT, OG_RESOLVE, OG_FILL, OG_UPDATE, OG_CLEARIMG, OG_CLEARATT, OG_RP, OG_DRAW, OG_DISPATCH, OG_QRESET, OG_QUERY, OG_TS, OG_XFB, OG_BARRIER, OG_FULLBAR, OG_N };
static void og_note(VkCommandBuffer cb, int kind); static void og_note_ts(VkCommandBuffer cb, VkQueryPool q);
static _Atomic uint64_t g_og_locks;   // ootgap: the queue lock taken by this layer (vk_queue_lock)
static void og_report(uint64_t nf);
static PFN_vkCmdCopyBuffer s63o_vkCmdCopyBuffer; static void VKAPI_PTR s63w_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer a, VkBuffer b, uint32_t n, const VkBufferCopy *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_COPYBUF); s63o_vkCmdCopyBuffer(cb, a, b, n, r); }
static PFN_vkCmdCopyBufferToImage s63o_vkCmdCopyBufferToImage; static void VKAPI_PTR s63w_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer a, VkImage b, VkImageLayout l, uint32_t n, const VkBufferImageCopy *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_B2I); s63o_vkCmdCopyBufferToImage(cb, a, b, l, n, r); }
static PFN_vkCmdCopyImageToBuffer s63o_vkCmdCopyImageToBuffer; static void VKAPI_PTR s63w_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage a, VkImageLayout l, VkBuffer b, uint32_t n, const VkBufferImageCopy *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_I2B); s63o_vkCmdCopyImageToBuffer(cb, a, l, b, n, r); }
static PFN_vkCmdCopyImage s63o_vkCmdCopyImage; static void VKAPI_PTR s63w_vkCmdCopyImage(VkCommandBuffer cb, VkImage a, VkImageLayout la, VkImage b, VkImageLayout lb, uint32_t n, const VkImageCopy *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_COPYIMG); s63o_vkCmdCopyImage(cb, a, la, b, lb, n, r); }
static PFN_vkCmdBlitImage s63o_vkCmdBlitImage; static void VKAPI_PTR s63w_vkCmdBlitImage(VkCommandBuffer cb, VkImage a, VkImageLayout la, VkImage b, VkImageLayout lb, uint32_t n, const VkImageBlit *r, VkFilter f) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_BLIT); s63o_vkCmdBlitImage(cb, a, la, b, lb, n, r, f); }
static PFN_vkCmdResolveImage s63o_vkCmdResolveImage; static void VKAPI_PTR s63w_vkCmdResolveImage(VkCommandBuffer cb, VkImage a, VkImageLayout la, VkImage b, VkImageLayout lb, uint32_t n, const VkImageResolve *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_RESOLVE); s63o_vkCmdResolveImage(cb, a, la, b, lb, n, r); }
static PFN_vkCmdFillBuffer s63o_vkCmdFillBuffer; static void VKAPI_PTR s63w_vkCmdFillBuffer(VkCommandBuffer cb, VkBuffer a, VkDeviceSize o, VkDeviceSize s, uint32_t v) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_FILL); s63o_vkCmdFillBuffer(cb, a, o, s, v); }
static PFN_vkCmdUpdateBuffer s63o_vkCmdUpdateBuffer; static void VKAPI_PTR s63w_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer a, VkDeviceSize o, VkDeviceSize s, const void *p) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_UPDATE); s63o_vkCmdUpdateBuffer(cb, a, o, s, p); }
static PFN_vkCmdClearColorImage s63o_vkCmdClearColorImage; static void VKAPI_PTR s63w_vkCmdClearColorImage(VkCommandBuffer cb, VkImage a, VkImageLayout l, const VkClearColorValue *c, uint32_t n, const VkImageSubresourceRange *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_CLEARIMG); s63o_vkCmdClearColorImage(cb, a, l, c, n, r); }
static PFN_vkCmdClearDepthStencilImage s63o_vkCmdClearDepthStencilImage; static void VKAPI_PTR s63w_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage a, VkImageLayout l, const VkClearDepthStencilValue *c, uint32_t n, const VkImageSubresourceRange *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_CLEARIMG); s63o_vkCmdClearDepthStencilImage(cb, a, l, c, n, r); }
static PFN_vkCmdClearAttachments s63o_vkCmdClearAttachments; static void VKAPI_PTR s63w_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr, const VkClearRect *r) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_CLEARATT); s63o_vkCmdClearAttachments(cb, na, a, nr, r); }
static PFN_vkCmdBeginRenderPass s63o_vkCmdBeginRenderPass; static void VKAPI_PTR s63w_vkCmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo *i, VkSubpassContents c) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_RP); s63o_vkCmdBeginRenderPass(cb, i, c); }
static PFN_vkCmdEndRenderPass s63o_vkCmdEndRenderPass; static void VKAPI_PTR s63w_vkCmdEndRenderPass(VkCommandBuffer cb) { S63_WORK(); s63o_vkCmdEndRenderPass(cb); }
static PFN_vkCmdBeginRendering s63o_vkCmdBeginRendering; static void VKAPI_PTR s63w_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *i) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_RP); s63o_vkCmdBeginRendering(cb, i); }
static PFN_vkCmdEndRendering s63o_vkCmdEndRendering; static void VKAPI_PTR s63w_vkCmdEndRendering(VkCommandBuffer cb) { S63_WORK(); s63o_vkCmdEndRendering(cb); }
static PFN_vkCmdDraw s63o_vkCmdDraw; static void VKAPI_PTR s63w_vkCmdDraw(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DRAW); s63o_vkCmdDraw(cb, a, b, c, d); }
static PFN_vkCmdDrawIndexed s63o_vkCmdDrawIndexed; static void VKAPI_PTR s63w_vkCmdDrawIndexed(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DRAW); s63o_vkCmdDrawIndexed(cb, a, b, c, d, e); }
static PFN_vkCmdDrawIndirect s63o_vkCmdDrawIndirect; static void VKAPI_PTR s63w_vkCmdDrawIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DRAW); s63o_vkCmdDrawIndirect(cb, b, o, n, s); }
static PFN_vkCmdDrawIndexedIndirect s63o_vkCmdDrawIndexedIndirect; static void VKAPI_PTR s63w_vkCmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DRAW); s63o_vkCmdDrawIndexedIndirect(cb, b, o, n, s); }
static PFN_vkCmdDrawIndirectCount s63o_vkCmdDrawIndirectCount; static void VKAPI_PTR s63w_vkCmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cb2, VkDeviceSize co, uint32_t n, uint32_t s) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DRAW); s63o_vkCmdDrawIndirectCount(cb, b, o, cb2, co, n, s); }
static PFN_vkCmdDrawIndexedIndirectCount s63o_vkCmdDrawIndexedIndirectCount; static void VKAPI_PTR s63w_vkCmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkBuffer cb2, VkDeviceSize co, uint32_t n, uint32_t s) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DRAW); s63o_vkCmdDrawIndexedIndirectCount(cb, b, o, cb2, co, n, s); }
static PFN_vkCmdDispatch s63o_vkCmdDispatch; static void VKAPI_PTR s63w_vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DISPATCH); s63o_vkCmdDispatch(cb, x, y, z); }
static PFN_vkCmdDispatchIndirect s63o_vkCmdDispatchIndirect; static void VKAPI_PTR s63w_vkCmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_DISPATCH); s63o_vkCmdDispatchIndirect(cb, b, o); }
static PFN_vkCmdResetQueryPool s63o_vkCmdResetQueryPool; static void VKAPI_PTR s63w_vkCmdResetQueryPool(VkCommandBuffer cb, VkQueryPool q, uint32_t f, uint32_t n) { S63_WORK(); og_note(cb, OG_QRESET); s63o_vkCmdResetQueryPool(cb, q, f, n); }
static PFN_vkCmdWriteTimestamp s63o_vkCmdWriteTimestamp; static void VKAPI_PTR s63w_vkCmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits s, VkQueryPool q, uint32_t i) { S63_WORK(); og_note_ts(cb, q); s63o_vkCmdWriteTimestamp(cb, s, q, i); }
static PFN_vkCmdBeginQuery s63o_vkCmdBeginQuery; static void VKAPI_PTR s63w_vkCmdBeginQuery(VkCommandBuffer cb, VkQueryPool q, uint32_t i, VkQueryControlFlags f) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_QUERY); s63o_vkCmdBeginQuery(cb, q, i, f); }
static PFN_vkCmdEndQuery s63o_vkCmdEndQuery; static void VKAPI_PTR s63w_vkCmdEndQuery(VkCommandBuffer cb, VkQueryPool q, uint32_t i) { S63_WORK(); s63o_vkCmdEndQuery(cb, q, i); }
static PFN_vkCmdBeginTransformFeedbackEXT s63o_vkCmdBeginTransformFeedbackEXT; static void VKAPI_PTR s63w_vkCmdBeginTransformFeedbackEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { vk_hy3_close_for(cb); S63_WORK(); og_note(cb, OG_XFB); s63o_vkCmdBeginTransformFeedbackEXT(cb, f, n, b, o); }
static PFN_vkCmdEndTransformFeedbackEXT s63o_vkCmdEndTransformFeedbackEXT; static void VKAPI_PTR s63w_vkCmdEndTransformFeedbackEXT(VkCommandBuffer cb, uint32_t f, uint32_t n, const VkBuffer *b, const VkDeviceSize *o) { S63_WORK(); s63o_vkCmdEndTransformFeedbackEXT(cb, f, n, b, o); }
static void VKAPI_PTR s63w_vkCmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    S63_WORK_0(); og_note(cb, OG_BARRIER);
    if (!f && !nb && !ni && nm <= 1) {   // cbar2: memory only: held, merged with the next ones until GPU work in cb
        VkAccessFlags sa = 0, da = 0; if (nm) { sa = m[0].srcAccessMask; da = m[0].dstAccessMask; }
        if (bpend_hold(cb, s, d, sa, da)) return; }
    if (cbg_refuse(cb, "direct", s, d, nb, ni)) return;   // cbguard
    if (nb || ni) vk_hy3_close_for(cb);   // stage2s: a buffer or image barrier closes the atomic staging window first
    bpend_flush(cb); s63o_vkCmdPipelineBarrier(cb, s, d, f, nm, m, nb, b, ni, im); }
static void bpend_drop(VkCommandBuffer cb) {   // cbar2: a command buffer begun again: what an abandoned recording held is not carried into the new one
    if (!atomic_load_explicit(&g_nbpend, memory_order_acquire)) return;
    pthread_mutex_lock(&g_bpend_mu); for (int i = 0; i < BPEND_N; i++) if (g_bpend[i].n && g_bpend[i].cb == cb) { g_bpend[i].n = 0; atomic_fetch_sub(&g_nbpend, 1); } pthread_mutex_unlock(&g_bpend_mu);
}
static PFN_vkBeginCommandBuffer s63o_vkBeginCommandBuffer; static VkResult VKAPI_PTR s63w_vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo *i) { S63_WORK_0(); bpend_drop(cb); const VkResult r = s63o_vkBeginCommandBuffer(cb, i); cbg_set(cb, r == VK_SUCCESS); return r; }
static PFN_vkEndCommandBuffer s63o_vkEndCommandBuffer; static VkResult VKAPI_PTR s63w_vkEndCommandBuffer(VkCommandBuffer cb) { vk_hy3_close_for(cb); S63_WORK(); const VkResult r = s63o_vkEndCommandBuffer(cb); cbg_set(cb, 0); return r; }
void vk_bar_flush(VkCommandBuffer cb) { vk_hy3_close_for(cb); bpend_flush(cb); }   // cbar2: before GPU work recorded through no wrapper (gfx_vk_exec.c mesh draws)
uint64_t vk_bmerge(int k) { return k == 0 ? g_bmerge_held : k == 1 ? g_bmerge_rec : g_bmerge_abs; }   // cbar2 counters (gfx_vk_exec.c's line)
// cbar2 self-test (cases 180-185): the hold and its union, with a recorder in place of the device's barrier entry point
static struct { int n; VkCommandBuffer cb[4]; VkPipelineStageFlags s[4], d[4]; VkAccessFlags sa[4], da[4]; uint32_t nb[4]; } g_brec; static PFN_vkCmdPipelineBarrier g_brec_keep;
static void VKAPI_PTR brec(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm, const VkMemoryBarrier *m, uint32_t nb, const VkBufferMemoryBarrier *b, uint32_t ni, const VkImageMemoryBarrier *im) {
    if (cb != (VkCommandBuffer)(uintptr_t)0xA0A0 && cb != (VkCommandBuffer)(uintptr_t)0xB0B0) { g_brec_keep(cb, s, d, f, nm, m, nb, b, ni, im); return; }   // (another thread's real barrier during the test)
    (void)f; (void)b; (void)im; if (g_brec.n >= 4) { g_brec.n++; return; } const int i = g_brec.n++;
    g_brec.cb[i] = cb; g_brec.s[i] = s; g_brec.d[i] = d; g_brec.sa[i] = nm ? m[0].srcAccessMask : 0; g_brec.da[i] = nm ? m[0].dstAccessMask : 0; g_brec.nb[i] = nb + ni; }
static int bpend_case(int k, const char **what) {
    if (!cbar2_on()) { *what = "not run (DM_VK_CBAR2=0)"; return k == 0 ? 1 : -1; }
    VkCommandBuffer A = (VkCommandBuffer)(uintptr_t)0xA0A0, B = (VkCommandBuffer)(uintptr_t)0xB0B0; memset(&g_brec, 0, sizeof g_brec);
    PFN_vkCmdPipelineBarrier keep = s63o_vkCmdPipelineBarrier; g_brec_keep = keep ? keep : vkCmdPipelineBarrier; g_bpend_rec = brec; s63o_vkCmdPipelineBarrier = brec; int ok = 0;
    switch (k) {
    case 0: *what = "three barriers with no work between: one recorded, the union of their stages and accesses";
        bpend_hold(A, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
        bpend_hold(A, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT);
        bpend_hold(A, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT);
        bpend_flush(A);
        ok = g_brec.n == 1 && g_brec.cb[0] == A && g_brec.s[0] == (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT) &&
             g_brec.d[0] == (VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) && g_brec.sa[0] == (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) &&
             g_brec.da[0] == (VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT); break;
    case 1: *what = "work between two barriers: both recorded, the first before the work";
        bpend_hold(A, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 1, 1); bpend_flush(A); const int n1 = g_brec.n;
        bpend_hold(A, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 2, 2); bpend_flush(A);
        ok = n1 == 1 && g_brec.n == 2 && g_brec.sa[0] == 1 && g_brec.sa[1] == 2; break;
    case 2: *what = "work in another command buffer leaves this one's held barrier held";
        bpend_hold(A, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 1, 1); bpend_flush(B); const int n0 = g_brec.n; bpend_flush(A);
        ok = n0 == 0 && g_brec.n == 1 && g_brec.cb[0] == A; break;
    case 3: *what = "a barrier with an image barrier: what is held is recorded first, then it";
        bpend_hold(A, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 1, 1);
        { VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; s63w_vkCmdPipelineBarrier(A, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); }
        ok = g_brec.n == 2 && g_brec.nb[0] == 0 && g_brec.nb[1] == 1; break;
    case 4: *what = "a memory barrier through the wrapper is held, then recorded at the next work";
        { VkMemoryBarrier m = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, 4, 8}; s63w_vkCmdPipelineBarrier(A, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &m, 0, NULL, 0, NULL); }
        { const int n0 = g_brec.n; bpend_flush(A); ok = n0 == 0 && g_brec.n == 1 && g_brec.sa[0] == 4 && g_brec.da[0] == 8; } break;
    case 5: *what = "a command buffer begun again drops what an abandoned recording held";
        bpend_hold(A, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 1, 1); bpend_drop(A); bpend_flush(A); ok = g_brec.n == 0; break;
    default: s63o_vkCmdPipelineBarrier = keep; g_bpend_rec = NULL; return -1;
    }
    bpend_drop(A); bpend_drop(B); s63o_vkCmdPipelineBarrier = keep; g_bpend_rec = NULL; return ok;
}
#define S63_I(f) if (f && f != s63w_##f) { s63o_##f = f; f = s63w_##f; }
static void s63_wrap(void) { S63_I(vkCmdCopyBuffer) S63_I(vkCmdCopyBufferToImage) S63_I(vkCmdCopyImageToBuffer) S63_I(vkCmdCopyImage) S63_I(vkCmdBlitImage) S63_I(vkCmdResolveImage) S63_I(vkCmdFillBuffer) S63_I(vkCmdUpdateBuffer) S63_I(vkCmdClearColorImage) S63_I(vkCmdClearDepthStencilImage) S63_I(vkCmdClearAttachments) S63_I(vkCmdBeginRenderPass) S63_I(vkCmdEndRenderPass) S63_I(vkCmdBeginRendering) S63_I(vkCmdEndRendering) S63_I(vkCmdDraw) S63_I(vkCmdDrawIndexed) S63_I(vkCmdDrawIndirect) S63_I(vkCmdDrawIndexedIndirect) S63_I(vkCmdDrawIndirectCount) S63_I(vkCmdDrawIndexedIndirectCount) S63_I(vkCmdDispatch) S63_I(vkCmdDispatchIndirect) S63_I(vkCmdResetQueryPool) S63_I(vkCmdWriteTimestamp) S63_I(vkCmdBeginQuery) S63_I(vkCmdEndQuery) S63_I(vkCmdBeginTransformFeedbackEXT) S63_I(vkCmdEndTransformFeedbackEXT) S63_I(vkCmdPipelineBarrier) S63_I(vkBeginCommandBuffer) S63_I(vkEndCommandBuffer) }   // after the device's entry points are loaded
vkdev_t V;
PFN_vkGetInstanceProcAddr vk_gipa(void) { return p_vkGetInstanceProcAddr; }
pthread_mutex_t g_vk_queue_mu = PTHREAD_MUTEX_INITIALIZER;   // VkQueue access is externally synchronized: submissions and presents
#if defined(DM_NXN1X)
// NXN1X (NVN_HOST_BRIDGE.md §4.3): the device and queue are the emulator's (its Vulkan backend's), and so is the queue's lock, Scheduler::submit_mutex,
// under which its scheduler and PresentManager submit to the same graphics queue. Set by nvnh_vk_use_device before the layer's first vk_init
#include "../nxn1x/nvnh_api.h"
#include "../nxn1x/nvnh_vk.h"
static nvnh_vk_device g_imp;
static int g_imp_set;
int nvnh_vk_use_device(const nvnh_vk_device *d) {
    if (V.dev) { VKLOG("[RED] NXN1X device handed over after the NVN layer's vk_init ran (on an earlier boot's device or its own): ignored"); return -1; }
    g_imp = *d; g_imp_set = 1; return 0;
}
int nvnh_vk_active(void) { return V.dev != VK_NULL_HANDLE; }
// NVN_HOST_BRIDGE.md §6 step 2: vk_init on NXN1X's device, then one empty submission the layer's own way (a ring slot, the timeline signal, the completion thread)
int nvnh_vk_selftest(void) {
    uint64_t t0 = now_ns();
    mtl_sub_begin(); uint64_t f = mtl_sub_commit(); int ok = mtl_wait(f, 2000000000ull);
    VKLOG("%sself-test: an empty submission (timeline value %llu) %s in %.2f ms", ok ? "" : "[RED] ", (unsigned long long)f, ok ? "completed" : "did not complete", (double)(now_ns() - t0) / 1e6);
    { int vk_poolcopy_selftest(void); const char *e = getenv("NXN1X_POOL_SELFTEST"); if (ok && e && atoi(e) != 0 && vk_poolcopy_selftest()) ok = 0; }   // poolcopy (review step 1)
    return ok ? 0 : -1;
}
#endif
// The queue lock: the host's own mutex, or the emulator's (NXN1X)
void vk_queue_lock(void) { atomic_fetch_add(&g_og_locks, 1);   // ootgap
#if defined(DM_NXN1X)
    if (g_imp_set && g_imp.queue_lock) { g_imp.queue_lock(g_imp.queue_ctx); return; }
#endif
    pthread_mutex_lock(&g_vk_queue_mu);
}
void vk_queue_unlock(void) {
#if defined(DM_NXN1X)
    if (g_imp_set && g_imp.queue_unlock) { g_imp.queue_unlock(g_imp.queue_ctx); return; }
#endif
    pthread_mutex_unlock(&g_vk_queue_mu);
}

uint32_t vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < V.mem.memoryTypeCount; i++)
        if ((bits >> i & 1) && (V.mem.memoryTypes[i].propertyFlags & want) == want && !(V.mem.memoryTypes[i].propertyFlags & avoid)) return i;
    for (uint32_t i = 0; i < V.mem.memoryTypeCount; i++) if ((bits >> i & 1) && (V.mem.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }
static int is_on(const char *const *on, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(on[i], name)) return 1; return 0; }
// a feature struct as vk_init queries it (sType, pNext, then VkBool32s; the largest, descriptor indexing, has 20); featx: its VkBool32s, all off if it is not chained
typedef struct { VkBaseOutStructure h; VkBool32 b[24]; } featx_t;
static VkBool32 *featx(featx_t *x, uint32_t n, VkStructureType s) { static VkBool32 off[24]; for (uint32_t i = 0; i < n; i++) if (x[i].h.sType == s) return x[i].b; return off; }   // (off: only 0 is written to it)

static void completion_start(void);
static void slots_init(void);
static VKAPI_ATTR VkBool32 VKAPI_CALL vk_validation_msg(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *d, void *u) {
    (void)type; (void)u;   // each distinct message ID logged its first 3 times
    enum { NID = 512 }; static int32_t ids[NID]; static uint8_t cnt[NID]; static int nid; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu); int k = 0; for (; k < nid; k++) if (ids[k] == d->messageIdNumber) break;
    if (k == nid && nid < NID) ids[nid++] = d->messageIdNumber;
    int show = k < NID && cnt[k] < 3; if (k < NID && cnt[k] < 255) cnt[k]++; pthread_mutex_unlock(&mu);
    if (show) VKLOG("[RED] validation %s: %s", sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning", d->pMessage);
    return VK_FALSE;
}

#if defined(DM_NXN1X)
static int imp_has(const char *name) { for (uint32_t i = 0; i < g_imp.n_ext; i++) if (!strcmp(g_imp.ext[i], name)) return 1; return 0; }
// vk_init on NXN1X's device: what the host's own vk_init reads from the device it creates (below), read from the one the emulator created. The capability flags
// are what that device was created with (extensions enabled, features turned on), not what the GPU could do: the host may use only what is enabled
static void vk_init_import(void) {
    p_vkGetInstanceProcAddr = g_imp.gipa;
    V.inst = g_imp.inst; V.pd = g_imp.pd; V.dev = g_imp.dev; V.q = g_imp.q; V.qf = g_imp.qf;
#define VK_LOADI_IMP(f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);
    VK_FNS(VK_LOADI_IMP)
    vkGetPhysicalDeviceProperties(V.pd, &V.props); vkGetPhysicalDeviceMemoryProperties(V.pd, &V.mem);
    { uint32_t d = V.props.driverVersion;
      VKLOG("GPU %s (vendor %#x, device %#x), Vulkan %u.%u.%u, driver %u.%u.%u (%#x): NXN1X's device and graphics queue (family %u)", V.props.deviceName, V.props.vendorID, V.props.deviceID,
            VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion), VK_API_VERSION_PATCH(V.props.apiVersion), d >> 22, d >> 12 & 0x3ff, d & 0xfff, d, V.qf); }
    V.has_quad = 0;   // The imported device does not enable shader quad control.
    int v12 = V.props.apiVersion >= VK_API_VERSION_1_2;
    if (!g_imp.timeline) die("[vk] NXN1X's device has no timeline semaphores enabled: every submission of the NVN layer signals one");
    V.has_dynrender = g_imp.dynrender;
    V.has_imageless = g_imp.imageless && (v12 || imp_has("VK_KHR_create_renderpass2"));
    V.has_demote = g_imp.demote;
    if (!V.has_imageless && !V.has_dynrender) die("[vk] NXN1X's device has neither dynamic rendering nor imageless framebuffers enabled: passes cannot be begun");
    V.has_xfb = g_imp.xfb; V.has_custom_border = g_imp.custom_border; V.has_robust2 = g_imp.null_descriptor; V.has_depth_clip_ctl = g_imp.depth_clip_ctl;
    V.has_dtiming = 0;          // presentation is NXN1X's (§4.5)
    V.has_depth_clip_en = 0;    // r99c DM_VK_DEPTH_CLIP needs VK_EXT_depth_clip_enable, which NXN1X does not enable
    V.has_host_import = imp_has("VK_EXT_external_memory_host");
    if (V.has_host_import) { VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hp}; vkGetPhysicalDeviceProperties2(V.pd, &p2); V.host_align = hp.minImportedHostPointerAlignment; }
    V.has_bc = g_imp.core.textureCompressionBC; V.has_astc = g_imp.core.textureCompressionASTC_LDR; V.has_gs = g_imp.core.geometryShader; V.has_tess = g_imp.core.tessellationShader;
    V.has_dbounds = g_imp.core.depthBounds;
    V.max_aniso = g_imp.core.samplerAnisotropy ? V.props.limits.maxSamplerAnisotropy : 1.0f;
    V.has_bda = g_imp.bda;
    if (!g_imp.uint8_index) VKLOG("[RED] NXN1X's device has no VK_EXT_index_type_uint8: 8-bit index draws are invalid (gfx_vk_exec.c)");
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr(V.inst, "vkGetDeviceProcAddr");
#define VK_LOADD_IMP(f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f); if (!f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f "KHR"); if (!f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);
    VK_DEV_FNS(VK_LOADD_IMP)
    s63_wrap();   // s63
    if (V.has_host_import) V.get_host_ptr_props = vkGetDeviceProcAddr(V.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};   // the host's own pipelines (blits, clears); the game's are Eden's (shader lane)
    VKCHK(vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache));
    VKLOG("%s, Vulkan %u.%u, driver %#x (NXN1X's device, %u extensions); BC %s, ASTC %s, geometry shaders %s, nullDescriptor %s, custom border colours %s, depth clip control %s, transform feedback %s, "
          "dynamic rendering %s, imageless framebuffers %s, buffer device address %s, host memory import %s (alignment %#llx)",
          V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion), V.props.driverVersion, g_imp.n_ext,
          V.has_bc ? "yes" : "no", V.has_astc ? "yes" : "no", V.has_gs ? "yes" : "no", V.has_robust2 ? "yes" : "no", V.has_custom_border ? "yes" : "no", V.has_depth_clip_ctl ? "yes" : "no", V.has_xfb ? "yes" : "no",
          V.has_dynrender ? "yes" : "no", V.has_imageless ? "yes" : "no", V.has_bda ? "yes" : "no", V.has_host_import ? "yes" : "no", (unsigned long long)V.host_align);
    slots_init();
    completion_start();
    if (dm_sdk_ounce()) (void)vk_sparse_tiles_ok();   // NVN2 (11-16): the 2D sparse tiles line at the device's making, not only at the first big texture
}
#endif
void vk_init(void) {
    static int done; if (done) return; done = 1;
#if defined(DM_NXN1X)
    if (g_imp_set) { vk_init_import(); return; }
    // the gate goes live only after the handover (Core::NvnLayer::GateOn): a layer call without NXN1X's device is a bug, not a reason for a device of its own
    die("[vk] the NVN layer was called before NXN1X handed over its Vulkan device");
#endif
    void *lib = dlopen("vulkan-1.dll", RTLD_NOW | RTLD_LOCAL);
    if (!lib) die("[vk] vulkan-1.dll: %s", dlerror());
    p_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    vkCreateInstance = (PFN_vkCreateInstance)p_vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    const char *iext[] = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_EXT_debug_utils"};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "dmhost", 1, "nxnative", 1, VK_API_VERSION_1_3};
    // DM_VK_VALIDATE=1: Khronos's validation layer, which the APK carries only when built with it (mkapk.py --extra-so); its messages go to the log
    int validate = getenv("DM_VK_VALIDATE") && atoi(getenv("DM_VK_VALIDATE"));
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, validate ? 1u : 0u, layers, validate ? 3u : 2u, iext};
    VkResult ir = vkCreateInstance(&ici, NULL, &V.inst);
    if (ir && validate) { VKLOG("[RED] validation layer not available (%d): running without it", ir); validate = 0; ici.enabledLayerCount = 0; ici.enabledExtensionCount = 2; ir = vkCreateInstance(&ici, NULL, &V.inst); }
    VKCHK(ir);
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT mk = (PFN_vkCreateDebugUtilsMessengerEXT)p_vkGetInstanceProcAddr(V.inst, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0,
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT, vk_validation_msg, NULL};
        VkDebugUtilsMessengerEXT m; if (mk && !mk(V.inst, &mi, NULL, &m)) VKLOG("validation layer on");
    }
#define VK_LOADI(f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);
    VK_FNS(VK_LOADI)
    uint32_t n = 1; VKCHK(vkEnumeratePhysicalDevices(V.inst, &n, &V.pd)); if (!n) die("[vk] no GPU");
    vkGetPhysicalDeviceProperties(V.pd, &V.props); vkGetPhysicalDeviceMemoryProperties(V.pd, &V.mem);
    // the device names itself before any check below can stop: a Moto g 5G 2024's log (Adreno 619) ended at the timeline check with no GPU or driver named.
    // driver: as Qualcomm encodes it (VK_MAKE_VERSION: 512.530.0), with the raw value for other vendors
    { uint32_t d = V.props.driverVersion;
      VKLOG("GPU %s (vendor %#x, device %#x), Vulkan %u.%u.%u, driver %u.%u.%u (%#x)", V.props.deviceName, V.props.vendorID, V.props.deviceID, VK_API_VERSION_MAJOR(V.props.apiVersion),
            VK_API_VERSION_MINOR(V.props.apiVersion), VK_API_VERSION_PATCH(V.props.apiVersion), d >> 22, d >> 12 & 0x3ff, d & 0xfff, d); }
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &qn, NULL);
    VkQueueFamilyProperties qp[16]; if (qn > 16) qn = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &qn, qp);
    V.qf = 0; while (V.qf < qn && (qp[V.qf].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) != (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) V.qf++;
    if (V.qf == qn) die("[vk] no graphics + compute queue");
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, NULL);
    VkExtensionProperties *ex = calloc(ne, sizeof *ex); vkEnumerateDeviceExtensionProperties(V.pd, NULL, &ne, ex);
    const char *want[] = {"VK_KHR_swapchain", "VK_EXT_external_memory_host", "VK_EXT_custom_border_color",
                          "VK_EXT_robustness2", "VK_EXT_depth_clip_control", "VK_EXT_provoking_vertex", "VK_EXT_index_type_uint8", "VK_EXT_transform_feedback",
                          "VK_GOOGLE_display_timing"};
    const char *on[64]; uint32_t non = 0;
    for (unsigned i = 0; i < sizeof want / sizeof *want; i++) if (has_ext(ex, ne, want[i])) on[non++] = want[i]; else VKLOG("device has no %s", want[i]);
    // r59: fragment shaders that shuffle across lanes ask for whole quads (gfx_vk_shader.c spv_full_quads); DM_VK_FULLQUAD=0: not asked
    { const char *e = getenv("DM_VK_FULLQUAD"); V.has_quad = 0;
      int qx = has_ext(ex, ne, "VK_KHR_shader_quad_control"); VkPhysicalDeviceShaderQuadControlFeaturesKHR q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR};
      if (qx) { VkPhysicalDeviceFeatures2 f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &q}; vkGetPhysicalDeviceFeatures2(V.pd, &f); }
      VkPhysicalDeviceDriverProperties dp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
      { VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dp}; vkGetPhysicalDeviceProperties2(V.pd, &p2); }
      if (!(e && *e == '0') && non < 64 && qx && q.shaderQuadControl) { on[non++] = "VK_KHR_shader_quad_control"; V.has_quad = 1; }
      VKLOG("full quads for shuffling fragment shaders: %s (VK_KHR_shader_quad_control %s, shaderQuadControl %u, driver id %d, DM_VK_FULLQUAD %s, %u device extensions)",
            V.has_quad == 1 ? "yes" : "no", qx ? "listed" : "not listed", (unsigned)q.shaderQuadControl, (int)dp.driverID, e ? e : "unset", ne); }
    // A Vulkan 1.1 or 1.2 device (Legion Y700 2023, Adreno 730: its stock driver reports 1.1) has what the replay uses from 1.2 / 1.3 as extensions: they are enabled
    // when present, so the features chained below and the functions loaded by their KHR names are valid there. A 1.3 device needs none of them
    int v12 = V.props.apiVersion >= VK_API_VERSION_1_2, v13 = V.props.apiVersion >= VK_API_VERSION_1_3; uint32_t npromo = 0;
    static const char *promo12[] = {"VK_KHR_create_renderpass2", "VK_KHR_imageless_framebuffer", "VK_KHR_image_format_list", "VK_KHR_depth_stencil_resolve", "VK_KHR_timeline_semaphore",
        "VK_KHR_shader_float16_int8", "VK_KHR_8bit_storage", "VK_EXT_descriptor_indexing", "VK_EXT_scalar_block_layout", "VK_KHR_uniform_buffer_standard_layout",
        "VK_KHR_shader_float_controls", "VK_KHR_spirv_1_4", "VK_KHR_vulkan_memory_model", "VK_KHR_separate_depth_stencil_layouts", "VK_EXT_host_query_reset",
        "VK_KHR_sampler_mirror_clamp_to_edge", "VK_KHR_draw_indirect_count", "VK_KHR_buffer_device_address", "VK_EXT_shader_viewport_index_layer", "VK_EXT_sampler_filter_minmax",
        "VK_KHR_shader_subgroup_extended_types", "VK_KHR_shader_atomic_int64", "VK_KHR_driver_properties"};
    static const char *promo13[] = {"VK_EXT_image_robustness", "VK_KHR_dynamic_rendering", "VK_KHR_synchronization2", "VK_EXT_shader_demote_to_helper_invocation",
        "VK_EXT_subgroup_size_control", "VK_KHR_copy_commands2", "VK_KHR_maintenance4", "VK_KHR_format_feature_flags2", "VK_EXT_extended_dynamic_state",
        "VK_KHR_zero_initialize_workgroup_memory", "VK_KHR_shader_terminate_invocation", "VK_EXT_inline_uniform_block", "VK_EXT_pipeline_creation_cache_control",
        "VK_EXT_texel_buffer_alignment", "VK_EXT_4444_formats", "VK_KHR_shader_integer_dot_product", "VK_EXT_private_data"};
    for (unsigned i = 0; !v12 && i < sizeof promo12 / sizeof *promo12 && non < 64; i++) if (has_ext(ex, ne, promo12[i])) { on[non++] = promo12[i]; npromo++; }
    for (unsigned i = 0; !v13 && i < sizeof promo13 / sizeof *promo13 && non < 64; i++) if (has_ext(ex, ne, promo13[i])) { on[non++] = promo13[i]; npromo++; }
    // features: what the replay uses when the device has it. VkPhysicalDeviceVulkan11/12/13Features are valid only on a 1.2 / 1.3 device (a 1.1 driver ignores
    // them and nothing would be enabled); below that, the same features come through Vulkan 1.1's own structs and those of the extensions enabled above (xf)
    static const struct { const char *ext; VkStructureType s; } XF[] = {   // ext NULL: Vulkan 1.1 core, VkPhysicalDeviceVulkan11Features's members
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTIVIEW_FEATURES},
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES},
        {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES}, {NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES},
        {"VK_KHR_imageless_framebuffer", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES}, {"VK_KHR_timeline_semaphore", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES},
        {"VK_KHR_shader_float16_int8", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}, {"VK_KHR_8bit_storage", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES},
        {"VK_EXT_descriptor_indexing", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES}, {"VK_EXT_scalar_block_layout", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES},
        {"VK_KHR_uniform_buffer_standard_layout", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFORM_BUFFER_STANDARD_LAYOUT_FEATURES}, {"VK_KHR_vulkan_memory_model", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES},
        {"VK_KHR_separate_depth_stencil_layouts", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SEPARATE_DEPTH_STENCIL_LAYOUTS_FEATURES}, {"VK_EXT_host_query_reset", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES},
        {"VK_KHR_buffer_device_address", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES}, {"VK_KHR_shader_subgroup_extended_types", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES},
        {"VK_KHR_shader_atomic_int64", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES},   // (mirror clamp, indirect count, minmax, viewport layer: the extension is the feature)
        {"VK_KHR_dynamic_rendering", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES}, {"VK_KHR_synchronization2", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES},
        {"VK_EXT_shader_demote_to_helper_invocation", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES}, {"VK_EXT_subgroup_size_control", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES},
        {"VK_KHR_maintenance4", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES}, {"VK_KHR_zero_initialize_workgroup_memory", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ZERO_INITIALIZE_WORKGROUP_MEMORY_FEATURES},
        {"VK_KHR_shader_terminate_invocation", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_TERMINATE_INVOCATION_FEATURES}, {"VK_EXT_inline_uniform_block", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INLINE_UNIFORM_BLOCK_FEATURES},
        {"VK_EXT_pipeline_creation_cache_control", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES}, {"VK_KHR_shader_integer_dot_product", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES},
        {"VK_EXT_private_data", VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIVATE_DATA_FEATURES}};   // (VK_EXT_image_robustness: robustImageAccess stays off, as on 1.3)
    uint32_t nxf = sizeof XF / sizeof *XF; featx_t xf[sizeof XF / sizeof *XF]; memset(xf, 0, sizeof xf); VkBaseOutStructure *xs = NULL;
    for (int i = (int)nxf - 1; i >= 0; i--) if (XF[i].ext ? is_on(on, non, XF[i].ext) : !v12) { xf[i].h.sType = XF[i].s; xf[i].h.pNext = xs; xs = &xf[i].h; }
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, xs};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, v13 ? (void *)&f13 : xs};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 ff = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, v12 ? (void *)&f11 : xs};
    vkGetPhysicalDeviceFeatures2(V.pd, &ff);
    if (!(v12 ? f12.timelineSemaphore : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)[0]))
        die("[vk] the device has no timeline semaphores (Vulkan 1.2 or VK_KHR_timeline_semaphore): every submission signals one");
    // dynamic rendering is 1.3 or its extension with the feature on; a loaded pointer does not tell, since the loader hands out a trampoline for core functions the driver lacks
    V.has_dynrender = v13 ? f13.dynamicRendering : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES)[0];
    // imageless framebuffers need render pass 2 (gfx_vk_exec.c rp_get): core in 1.2, else its extension
    V.has_imageless = v12 ? f12.imagelessFramebuffer : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES)[0] && is_on(on, non, "VK_KHR_create_renderpass2");
    V.has_demote = v13 ? f13.shaderDemoteToHelperInvocation : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES)[0] && is_on(on, non, "VK_EXT_shader_demote_to_helper_invocation");   // r83
    VKLOG("demote to helper invocation: %s (r83: discards before shuffles in fragment shaders, DM_VK_DEMOTE=0 off)", V.has_demote ? "yes" : "no");
    if (!V.has_imageless && !V.has_dynrender) die("[vk] the device has neither dynamic rendering nor imageless framebuffers with render pass 2: passes cannot be begun");
    if (!v13) VKLOG("Vulkan %u.%u device: %u extensions enabled for the features promoted to 1.2 / 1.3; dynamic rendering %s", VK_API_VERSION_MAJOR(V.props.apiVersion),
                    VK_API_VERSION_MINOR(V.props.apiVersion), npromo, V.has_dynrender ? "yes" : "no (render pass objects only)");
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fcb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT fr2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, &fcb};
    VkPhysicalDeviceDepthClipControlFeaturesEXT fdc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, &fr2};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT fu8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, &fdc};
    VkPhysicalDeviceTransformFeedbackFeaturesEXT fxf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, &fu8};
    VkPhysicalDeviceFeatures2 fx = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fxf};
    vkGetPhysicalDeviceFeatures2(V.pd, &fx);
    // transform feedback (NVN BeginTransformFeedback; gfx_vk_exec.c xfb_*): Xenoblade Chronicles 3 captures particle state with it
    V.has_xfb = fxf.transformFeedback && has_ext(ex, ne, "VK_EXT_transform_feedback"); fxf.geometryStreams = 0;
    fr2.robustBufferAccess2 = 0; fr2.robustImageAccess2 = 0;   // nullDescriptor only: unbound slots read 0 without a dummy
    V.has_custom_border = fcb.customBorderColors && has_ext(ex, ne, "VK_EXT_custom_border_color");
    V.has_robust2 = fr2.nullDescriptor && has_ext(ex, ne, "VK_EXT_robustness2");
    V.has_depth_clip_ctl = fdc.depthClipControl && has_ext(ex, ne, "VK_EXT_depth_clip_control");
    V.has_dtiming = has_ext(ex, ne, "VK_GOOGLE_display_timing");
    V.has_host_import = has_ext(ex, ne, "VK_EXT_external_memory_host");
    if (V.has_host_import) { VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &hp}; vkGetPhysicalDeviceProperties2(V.pd, &p2); V.host_align = hp.minImportedHostPointerAlignment; }   // frame pacing on the display (gfx_vk_present.c pace_*)
    V.has_bc = ff.features.textureCompressionBC; V.has_astc = ff.features.textureCompressionASTC_LDR; V.has_gs = ff.features.geometryShader; V.has_tess = ff.features.tessellationShader; V.has_dbounds = ff.features.depthBounds;
    V.max_aniso = V.props.limits.maxSamplerAnisotropy;
    // enable: everything the shaders may use that the device has
    VkPhysicalDeviceFeatures en = ff.features; en.robustBufferAccess = 0; en.shaderFloat64 = 0;
    V.has_bda = v12 ? f12.bufferDeviceAddress : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[0] && is_on(on, non, "VK_KHR_buffer_device_address");   // raw global memory's pool table (vk_gmem_map)
    { extern int g_vk_hqr; g_vk_hqr = v12 ? f12.hostQueryReset : featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES)[0] && is_on(on, non, "VK_EXT_host_query_reset"); }   // occ-s1: host query reset (enabled with the queried chain)
    f12.bufferDeviceAddressCaptureReplay = 0; f12.bufferDeviceAddressMultiDevice = 0;
    featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[1] = featx(xf, nxf, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)[2] = 0;   // (its extension's: the same two)
    f13.robustImageAccess = 0;
    fcb.pNext = NULL; fr2.pNext = V.has_custom_border ? (void *)&fcb : NULL; fdc.pNext = V.has_robust2 ? (void *)&fr2 : (V.has_custom_border ? (void *)&fcb : NULL);
    fu8.pNext = V.has_depth_clip_ctl ? (void *)&fdc : fdc.pNext;
    fxf.pNext = &fu8;
    VkBaseOutStructure *t = ff.pNext; while (t->pNext) t = t->pNext;   // the queried chain (core / promoted features), then the extensions'
    t->pNext = V.has_xfb ? (VkBaseOutStructure *)&fxf : (VkBaseOutStructure *)&fu8;
    // r99c DM_VK_DEPTH_CLIP=1 (at start, A/B): draws with NVN's depth clamp keep depth clipping on (VK_EXT_depth_clip_enable), so the near plane still cuts triangles
    // whose vertices lie behind the eye. Turnip turns the near and far clip off with the clamp (tu_pipeline.cc GRAS_CL_CNTL), and Adreno then draws only part of a
    // mesh around the camera (Bayonetta 3's Cheshire tint: a hard octagon, both drivers; NVIDIA covers the screen). Vulkan's switch is the near and far plane together
    VkPhysicalDeviceDepthClipEnableFeaturesEXT fdce = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
    { const char *e = getenv("DM_VK_DEPTH_CLIP"); V.has_depth_clip_en = 0;
      if (e && atoi(e) && non < 64 && has_ext(ex, ne, "VK_EXT_depth_clip_enable")) { VkPhysicalDeviceFeatures2 q = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fdce}; vkGetPhysicalDeviceFeatures2(V.pd, &q);
          if (fdce.depthClipEnable) { on[non++] = "VK_EXT_depth_clip_enable"; fdce.pNext = ff.pNext; V.has_depth_clip_en = 1; } }
      if (e && atoi(e)) VKLOG("r99c: depth clip kept on with depth clamp: %s", V.has_depth_clip_en ? "yes" : "no (the device has no VK_EXT_depth_clip_enable)"); }
    VkPhysicalDeviceShaderQuadControlFeaturesKHR fq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_QUAD_CONTROL_FEATURES_KHR, V.has_depth_clip_en ? (void *)&fdce : ff.pNext, VK_TRUE};
    VkPhysicalDeviceFeatures2 enf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, V.has_quad ? (void *)&fq : V.has_depth_clip_en ? (void *)&fdce : ff.pNext, en};
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, V.qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enf, 0, 1, &qci, 0, NULL, non, on, NULL};
    VKCHK(vkCreateDevice(V.pd, &dci, NULL, &V.dev));
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)p_vkGetInstanceProcAddr(V.inst, "vkGetDeviceProcAddr");
#define VK_LOADD(f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f); if (!f) f = (PFN_##f)vkGetDeviceProcAddr(V.dev, #f "KHR"); if (!f) f = (PFN_##f)p_vkGetInstanceProcAddr(V.inst, #f);   // a promoted function by its extension's name (Vulkan 1.1 / 1.2 devices)
    VK_DEV_FNS(VK_LOADD)
    s63_wrap();   // s63
    vkGetDeviceQueue(V.dev, V.qf, 0, &V.q);
    if (V.has_host_import) V.get_host_ptr_props = vkGetDeviceProcAddr(V.dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    char path[1024]; const char *root = getenv("DM_ROOT"); void *data = NULL; size_t dn = 0;
    if (root && !getenv("DM_VK_NOPCACHE")) { snprintf(path, sizeof path, "%s/vk_pipeline_cache.bin", root); FILE *f = fopen(path, "rb");   // DM_VK_NOPCACHE=1: start empty
        if (f) { fseek(f, 0, SEEK_END); dn = (size_t)ftell(f); fseek(f, 0, SEEK_SET); data = malloc(dn); if (fread(data, 1, dn, f) != dn) dn = 0; fclose(f); } }
    pci.initialDataSize = dn; pci.pInitialData = data;
    if (vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache)) { pci.initialDataSize = 0; pci.pInitialData = NULL; VKCHK(vkCreatePipelineCache(V.dev, &pci, NULL, &V.pcache)); }
    free(data); free(ex);
    VKLOG("%s, Vulkan %u.%u, driver %#x; BC %s, ASTC %s, geometry shaders %s, nullDescriptor %s, custom border colours %s, depth clip control %s, transform feedback %s, host memory import %s (alignment %#llx); pipeline cache %zu bytes",
          V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion), V.props.driverVersion,
          V.has_bc ? "yes" : "no", V.has_astc ? "yes" : "no", V.has_gs ? "yes" : "no", V.has_robust2 ? "yes" : "no", V.has_custom_border ? "yes" : "no", V.has_depth_clip_ctl ? "yes" : "no", V.has_xfb ? "yes" : "no", V.has_host_import ? "yes" : "no", (unsigned long long)V.host_align, dn);
    slots_init();
    completion_start();
}
void vk_pipeline_cache_save(void) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static size_t last;
    const char *root = getenv("DM_ROOT"); if (!root || !V.pcache) return;
    pthread_mutex_lock(&mu);
    size_t n = 0; if (vkGetPipelineCacheData(V.dev, V.pcache, &n, NULL) || !n || n == last) { pthread_mutex_unlock(&mu); return; }   // unchanged since the last save
    last = n; pthread_mutex_unlock(&mu);
    void *d = malloc(n); if (vkGetPipelineCacheData(V.dev, V.pcache, &n, d)) { free(d); return; }
    char path[1024], tmp[1100]; snprintf(path, sizeof path, "%s/vk_pipeline_cache.bin", root); snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb"); if (f) { fwrite(d, 1, n, f); fclose(f); if (renamex_np(tmp, path, 0)) VKLOG("[RED] pipeline cache: %s not replaced", path); }   // r53: the CRT rename fails over an existing file (the cache stayed 36 bytes since 11-03)
    free(d);
}

// ---------------- memory pools ----------------
#define MAXPOOL 16384   // s32: was 4096 (TotK run 48: 2900 chunks at 97 s, and die() at the limit)
static vkpool_t g_pool[MAXPOOL]; static int g_npool;
static pthread_rwlock_t g_pool_mu = PTHREAD_RWLOCK_INITIALIZER;
// Writers retain registry identity across allocation recovery. Readers (including
// CPU copyback/completion) remain able to progress while the physical lock is free.
static nvnr_gate g_pool_writers = NVNR_GATE_INITIALIZER;
#include <sched.h>
// s33 (TotK run 49, 11-13): the replay thread spent about 2/3 of its samples in ntdll waits under pool_find's pthread_rwlock_rdlock: winpthreads' rwlock
// takes an internal mutex even for readers, so each lookup the per-thread cache misses (thousands a frame) met the other threads' lookups in the kernel.
// Now a reader takes the lock with one atomic on a counter (spinning, then yielding, only while a writer holds or waits for it); writers (pools added, made,
// released: rare) are serialized by a mutex, announce themselves so new readers wait, and wait for the readers inside to leave. A thread taking the lock again
// inside its own hold just counts (as the rwlock let readers). DM_VK_S33_LOCK=0: the rwlock as before
enum { PL_W = 1 << 30 };
static int g_pl_state;   // PL_W: a writer holds or waits; low bits: readers inside
static pthread_mutex_t g_pl_wmu = PTHREAD_MUTEX_INITIALIZER;
static __thread int g_pl_depth, g_pl_mode;
static int pl_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S33_LOCK"); on = !e || !*e || atoi(e) != 0; } return on; }
static void pl_pause(int *n) {
#if defined(__aarch64__)
    if (++*n < 64) { __asm__ volatile("yield"); return; }
#else
    if (++*n < 64) return;
#endif
    sched_yield();
}
// s47: times the memory pool lookups' slow parts (TotK fast travel 11-13, run 64: 12.9 s and 11.2 s freezes in the Eden draws' vertex-buffer lookups).
// Slow lookups (pool_find past its per-thread cache), its full scans, bridge_make, pool_make (its import, its first copy), the locks' waits, pool_ord_make. DM_VK_S47=0: off
static int s47_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S47"); on = e && *e && atoi(e) != 0; } return on; }   // s54: default off (DM_VK_S47=1 on)
static uint64_t s47_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
enum { S47_FIND, S47_SCAN, S47_BRIDGE, S47_MAKE, S47_IMPORT, S47_COPY, S47_WAITW, S47_WAITR, S47_ORD, S47_N };
static uint64_t g_s47_ns[S47_N], g_s47_n[S47_N];
static void s47_add(int k, uint64_t ns) { __atomic_fetch_add(&g_s47_ns[k], ns, __ATOMIC_RELAXED); __atomic_fetch_add(&g_s47_n[k], 1, __ATOMIC_RELAXED); }
static void pool_rdlock(void) {
    if (g_pl_depth++) return;
    g_pl_mode = 1;
    if (!pl_on()) { pthread_rwlock_rdlock(&g_pool_mu); return; }
    uint64_t t47 = 0;   // s47
    for (int n = 0;;) {
        int s = __atomic_load_n(&g_pl_state, __ATOMIC_RELAXED);
        if (s & PL_W) { if (!t47 && s47_on()) t47 = s47_ns(); pl_pause(&n); continue; }
        if (__atomic_compare_exchange_n(&g_pl_state, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) { if (t47) s47_add(S47_WAITR, s47_ns() - t47); return; }
    }
}
static void pool_wrlock(void) {
    if (g_pl_depth++) return;
    g_pl_mode = 2;
    nvnr_lock(&g_pool_writers);
    if (!pl_on()) { pthread_rwlock_wrlock(&g_pool_mu); return; }
    uint64_t t47 = s47_on() ? s47_ns() : 0;   // s47
    pthread_mutex_lock(&g_pl_wmu);
    __atomic_fetch_or(&g_pl_state, PL_W, __ATOMIC_SEQ_CST);
    for (int n = 0; __atomic_load_n(&g_pl_state, __ATOMIC_ACQUIRE) != PL_W;) pl_pause(&n);
    if (t47) s47_add(S47_WAITW, s47_ns() - t47);
}
static void pool_unlock(void) {
    if (--g_pl_depth) return;
    if (!pl_on()) { pthread_rwlock_unlock(&g_pool_mu); if (g_pl_mode == 2) nvnr_unlock(&g_pool_writers); return; }
    if (g_pl_mode == 2) { __atomic_fetch_and(&g_pl_state, ~PL_W, __ATOMIC_RELEASE); pthread_mutex_unlock(&g_pl_wmu); nvnr_unlock(&g_pool_writers); }
    else __atomic_fetch_sub(&g_pl_state, 1, __ATOMIC_RELEASE);
}
static uint32_t g_pool_gen;   // bumped under the write lock when a pool is added or removed: vk_pool_of's per-thread cache holds while it is unchanged
// g_pool's indices by lo, remade under the write lock whenever a pool is added or removed: vk_pool_of searches it. A linear scan of g_pool behind the per-thread
// cache was 18% of the profiler's samples of Super Mario Bros. Wonder's replay in its first level (a draw looks up more pools than the cache holds); with the
// search its world map's replay took 4.2 ms a frame instead of 4.6-4.9
static uint16_t g_pool_ord[MAXPOOL];
static int pool_ord_cmp(const void *a, const void *b) { uint64_t x = g_pool[*(const uint16_t *)a].lo, y = g_pool[*(const uint16_t *)b].lo; return x < y ? -1 : x > y; }
// s10 (NXN1X lane A, 2026-11-12): pools over the same memory, one a GPU copy (its import refused) and one imported. pool_find took whichever the per-thread
// cache or the search by lo met first, so a range in the shared memory was read and written through the copy at some lookups and through the game's own pages
// at others (Bayonetta 3's title in NXN1X: a refused 80 MB pool lies 48 MB over an imported one, and the masked head lost its body in the runs where the copy
// was made first). A range several pools hold is now looked up the same way every time: an imported pool, else one with no buffer yet (made at this use and,
// if it is refused too, the others tried: vk_pool_of), a copy last. g_pool_ovl[i]: g_pool[i] shares memory with another pool (with g_pool_ord)
static uint8_t g_pool_ovl[MAXPOOL];
static uint64_t g_ovl_pick, g_ovl_alt;   // lookups the rule sent to another pool than the search found; copies replaced by an imported pool at a first use
static void pool_ord_make_0(void);
static void pool_ord_make(void) { uint64_t t = s47_on() ? s47_ns() : 0; pool_ord_make_0(); if (t) s47_add(S47_ORD, s47_ns() - t); }   // s47: timed
static void pool_ord_make_0(void) { for (int i = 0; i < g_npool; i++) g_pool_ord[i] = (uint16_t)i; qsort(g_pool_ord, (size_t)g_npool, sizeof *g_pool_ord, pool_ord_cmp);
    memset(g_pool_ovl, 0, sizeof g_pool_ovl);
    for (int a = 0; a < g_npool; a++) for (int b = a + 1; b < g_npool && g_pool[g_pool_ord[b]].lo < g_pool[g_pool_ord[a]].hi; b++) g_pool_ovl[g_pool_ord[a]] = g_pool_ovl[g_pool_ord[b]] = 1; }
// s11: g_sub: per-view chunks pool_make added for a chunk refused because it spans mapped views (s12: NXN1X's pools are made per view anyway; this is the fallback)
// (s12: s11's put-back of a copy into the game's memory is gone: in run 12b it put 3.7 MB of stale copy over the game's pages, and the game then read through
// pointers into unmapped memory 478k times and stopped drawing its characters)
enum { MERGE_N = 16 };
static struct { uint64_t lo, hi; } g_sub[MERGE_N]; static int g_nsub;   // under the write lock
static int g_nbridge;   // s13: bridges made (s32: live ones: mtl_pool_del uncounts a deleted pool's)
static int g_bridging;   // s13: pool_make makes the chunk a copy (a bridge across two mapped views, bridge_make)
static int pool_is_sub(const vkpool_t *p) { for (int k = 0; k < g_nsub; k++) if (g_sub[k].lo == p->lo && g_sub[k].hi == p->hi) return 1; return 0; }
// s10: imported, not made yet, copied; s11: a per-view chunk refused too (never made: p->retry ~0)
static int pool_rank(const vkpool_t *p) { return p->buf && p->zero_copy ? 0 : !p->buf ? (p->retry == ~0ull ? 4 : 1) : 2; }
static const VkBufferUsageFlags POOL_USAGE0 = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
#define POOL_USAGE (POOL_USAGE0 | (V.has_xfb ? VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT : 0) | (V.has_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0))   // any pool range may be a transform feedback buffer, or reached by raw global memory (vk_gmem_map)
static VkMemoryAllocateFlagsInfo g_pool_af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};   // with POOL_USAGE's device address bit
static void pool_gpu_addr(vkpool_t *p) { if (V.has_bda) { VkBufferDeviceAddressInfo i = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, NULL, p->buf}; p->gpu = vkGetBufferDeviceAddress(V.dev, &i); } }
// Zero copy: the pages of [mem, mem+size) imported as a VkBuffer's memory (VK_EXT_external_memory_host), the game's pages staying where they are. 0 on success
static int pool_zero_copy_at(vkpool_t *p, uint8_t *mem, uint64_t size);
// s23: a pool chunk is imported through NXN1X's alias of its pages in the backing memory (nvnh_import_alias), not at the game's address. The driver locks the
// virtual range it imports (it cannot be unmapped while the import lives, and a finalized pool's import is freed only once the GPU is past it), while NXN1X unmaps
// and splits the game's views whenever the game remaps its memory. Xenoblade 3 froze leaving its title (11-12): UnmapViewOfFile2 failed on a view, NXN1X's
// placeholders were out of step after that, the game's next mapping failed and two guest threads looped on the missing memory. The backing is one view NXN1X never
// unmaps while the game runs, over the same physical pages. Without an alias (a page unmapped, or not contiguous), or refused there, at the game's address as before.
// DM_VK_S23_ALIAS=0: always at the game's address
// s48: memory pool chunks bound into big imports of NXN1X's backing memory (TotK run 65, 11-13: a WDDM host import costs ~20 ms a call whatever its size,
// and 1,666 of them, one per chunk at its first GPU use, froze the field builds and the play). An aligned range of the backing (one view of the whole section,
// which s23 imports through) is imported once and kept; a chunk inside it is a buffer bound at its offset. DM_VK_S48_MB=<n> (default 32, a power of two;
// 0 off), DM_VK_S48_SPARSE=0: s29's sparse bridges import their pieces alone as before. Under the pool write lock (pool_make's callers), as the table
enum { S48_MAXR = 512 };
static struct { uint64_t lo, hi; VkDeviceMemory mem; uint32_t type; int bad, nb; } g_s48[S48_MAXR];   // nb: buffers bound into it (s48_mem), less the chunks pool_release let go
static int g_s48_n; static uint64_t g_s48_bytes, g_s48_binds, g_s48_alone;
// NVN2 (Switch 2 boots): a chunk bound into a range owns no memory (s48_bind: p->mem stays NULL), so r78's release of an idle imported chunk freed nothing while its
// range lived: HWAoI 11-16 counted 2036 MB of imported chunks while Turnip held 3092 MB of host imports (TU_DEBUG=bos vkImportHost). nb counts the buffers bound into a
// range (s48_mem, every caller); pool_release takes back a chunk's (s48_unref). Only the memory pressure check (nvn2_pressure) frees ranges no buffer holds
// (s48_free_idle), so steady play keeps today's behaviour: a released chunk made again binds into its range with no new import. Buffers bound otherwise (s29's sparse
// pieces, chunks dropped by other paths) are never taken back, so their ranges stay. Switch 1 never frees a range, as before
static int g_s48_last;   // the range s48_mem used last (s48_bind)
static uint64_t g_rel_owned;   // NVN2 pressure: bytes of released chunks that had memory of their own (pool_release)
static uint64_t g_s48_freed, g_s48_freed_n, g_s48_reimp;   // ranges freed (bytes, count); ranges imported again after a free
static uint64_t g_s48_gone[64]; static unsigned g_s48_gone_i;   // the starts of the last ranges freed (a re-import is one of them)
static void s48_unref(int k) { if (k >= 0 && k < g_s48_n && g_s48[k].nb > 0) g_s48[k].nb--; }
// Under the pool write lock, at a submission start (g_cur set): up to maxn ranges no buffer holds are freed with the current submission (vk_defer_free_buffer; every
// buffer bound into them was taken back by pool_release, which deferred its destroy to this submission or an earlier one, so those go first). Returns the bytes
// NVN2 diagnostic (11-16): each range's holders (pool chunks and sparse bridges bound into it, their bytes) against its count, up to 48 lines (under the pool lock)
static void s48_dump(void) {
    uint64_t live[S48_MAXR] = {0}; uint32_t nchunk[S48_MAXR] = {0}, nbr[S48_MAXR] = {0};
    for (int i = 0; i < g_npool; i++) { const vkpool_t *p = &g_pool[i]; if (!p->buf) continue;
        if (p->s48 > 0 && p->s48 <= S48_MAXR) { if (p->bridge || p->s48b) nbr[p->s48 - 1]++; else { nchunk[p->s48 - 1]++; live[p->s48 - 1] += p->hi - p->lo; } }
        if (p->s48b > 0 && p->s48b <= S48_MAXR) nbr[p->s48b - 1]++; }
    int nl = 0;
    for (int k = 0; k < g_s48_n && nl < 48; k++) { if (!g_s48[k].mem) continue; nl++;
        VKLOG("NVN2 s48 range %#llx +%llu MB: count %d, pool chunks %u (%.1f MB), sparse bridges %u%s", (unsigned long long)g_s48[k].lo, (unsigned long long)((g_s48[k].hi - g_s48[k].lo) >> 20),
              g_s48[k].nb, nchunk[k], live[k] / 1048576.0, nbr[k], g_s48[k].nb != (int)(nchunk[k] + nbr[k]) ? " (the count differs: holders not in the pool table)" : ""); }
}
static uint64_t s48_free_idle(int maxn) {
    uint64_t got = 0;
    for (int k = 0; k < g_s48_n && maxn > 0; k++) {
        if (g_s48[k].nb || g_s48[k].bad || !g_s48[k].mem) continue;
        vk_defer_free_buffer(VK_NULL_HANDLE, g_s48[k].mem);
        const uint64_t n = g_s48[k].hi - g_s48[k].lo; g_s48_bytes -= n; g_s48_freed += n; g_s48_freed_n++; got += n; maxn--;
        g_s48_gone[g_s48_gone_i++ % 64] = g_s48[k].lo;
        g_s48[k].lo = g_s48[k].hi = 0; g_s48[k].mem = VK_NULL_HANDLE; g_s48[k].type = 0; g_s48[k].bad = 0; g_s48[k].nb = 0;   // (lo == hi == 0: a free slot)
    }
    return got;
}
// s67 (apply_s67.py): s48 range imports, their ns, zero-copy pool imports, their ns, pools made of chunks (s12); taken per frame by s67_sub
static uint64_t g_s67_ev[5];
static uint64_t s48_blk(void) {
    static uint64_t b = 1;
    if (b == 1) { const char *e = getenv("DM_VK_S48_MB"); long mb = e && *e ? atol(e) : (vk_is_n1x() ? 128 : 32); b = mb > 0 ? (uint64_t)mb << 20 : 0; if (b & (b - 1)) b = 32ull << 20;
        VKLOG("s48: memory pool chunks %s (DM_VK_S48_MB=0 off)", b ? "bound into imports of the backing memory, one per aligned range" : "imported one by one (off)");
        if (b) VKLOG("s48: range size %llu MB", (unsigned long long)(b >> 20)); }
    return b;
}
static int s48_sparse_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S48_SPARSE"); on = !e || !*e || atoi(e) != 0; } return on; }
// [al, al + size) of the backing in an imported range: its memory and offset there. 0 = got; the range is imported here when none holds it yet
static int s48_mem(uint8_t *al, uint64_t size, uint32_t bits, VkDeviceSize align, VkDeviceMemory *mem, VkDeviceSize *off) {
    const uint64_t B = s48_blk(); if (!B || !size || !V.has_host_import || !V.get_host_ptr_props) return -1;
    const uint64_t a = (uint64_t)(uintptr_t)al, e = a + size;
    int k = 0; while (k < g_s48_n && !(a >= g_s48[k].lo && e <= g_s48[k].hi)) k++;
    if (k == g_s48_n) {
        uint64_t lo = a & ~(B - 1), hi = (e + B - 1) & ~(B - 1), cut[1];
        if (nvnh_range_views((const void *)(uintptr_t)lo, hi - lo, cut, 1)) {
            // Padding can cross a Windows allocation boundary even when the chunk fits.
            // Only cache an import for the actual, aligned range in that case.
            lo = a; hi = e;
            if ((V.host_align && ((lo | size) & (V.host_align - 1))) ||
                nvnh_range_views(al, size, cut, 1)) { g_s48_alone++; return -1; }
        }
        for (int j = 0; j < g_s48_n; j++) if (lo < g_s48[j].hi && g_s48[j].lo < hi) { g_s48_alone++; return -1; }   // it would overlap a range made before
        { int d = 0; while (d < g_s48_n && (g_s48[d].lo || g_s48[d].hi)) d++; k = d; }   // NVN2: a slot s48_unref freed, else a new one
        if (k == S48_MAXR) { g_s48_alone++; return -1; }
        g_s48[k].lo = lo; g_s48[k].hi = hi; g_s48[k].mem = VK_NULL_HANDLE; g_s48[k].type = 0; g_s48[k].bad = 1; g_s48[k].nb = 0; if (k == g_s48_n) g_s48_n++;   // (bad until imported: a refusal is remembered)
        const char *why = NULL; uint64_t t0 = s47_ns(); uint32_t type = UINT32_MAX;
        VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        if (NVNH_HOST_PROPERTIES((PFN_vkGetMemoryHostPointerPropertiesEXT)V.get_host_ptr_props, V.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)lo, &hp) || !hp.memoryTypeBits) why = "no host pointer properties";
        else if ((type = vk_mem_type(hp.memoryTypeBits & bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)) == UINT32_MAX) why = "no memory type";
        else {
            VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, V.has_bda ? &g_pool_af : NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)(uintptr_t)lo};
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, hi - lo, type};
            if (vkAllocateMemory(V.dev, &ai, NULL, &g_s48[k].mem)) { g_s48[k].mem = VK_NULL_HANDLE; why = "the driver refused it"; }
            else { g_s48[k].bad = 0; g_s48[k].type = type; g_s48_bytes += hi - lo;
                for (unsigned q = 0; q < 64 && q < g_s48_gone_i; q++) if (g_s48_gone[q] == lo) {   // NVN2: a range the pressure check freed, imported again
                    g_s48_gone[q] = 0; g_s48_reimp++;
                    static int nr; static uint64_t lt; uint64_t tn = s47_ns(); if (nr < 8 || tn - lt > 10000000000ull) { nr++; lt = tn;
                        VKLOG("NVN2 memory pressure: s48 range %#llx +%#llx imported again after it was freed (%llu so far)", (unsigned long long)lo, (unsigned long long)(hi - lo), (unsigned long long)g_s48_reimp); }
                    break; } } }
        __atomic_fetch_add(&g_s67_ev[0], 1, __ATOMIC_RELAXED); __atomic_fetch_add(&g_s67_ev[1], s47_ns() - t0, __ATOMIC_RELAXED);   // s67
        static int nl, nb; double ms = (s47_ns() - t0) / 1e6;
        if (why ? nb++ < 32 : nl++ < 48) VKLOG("%ss48: backing range %#llx +%#llx %s in %.1f ms (%d ranges, %.0f MB imported; chunk %#llx +%#llx; %llu chunks bound so far, %llu imported alone)",
            why ? "[RED] " : "", (unsigned long long)lo, (unsigned long long)(hi - lo), why ? why : "imported", ms, g_s48_n, g_s48_bytes / 1048576.0, (unsigned long long)a,
            (unsigned long long)size, (unsigned long long)g_s48_binds, (unsigned long long)g_s48_alone);
    }
    if (g_s48[k].bad || !(bits >> g_s48[k].type & 1) || (align && (a - g_s48[k].lo) % align)) { g_s48_alone++; return -1; }
    *mem = g_s48[k].mem; *off = a - g_s48[k].lo; g_s48_binds++; g_s48[k].nb++; g_s48_last = k;
    { uint64_t q = g_s48_binds; if (q >= 256 && !(q & (q - 1))) VKLOG("s48: %llu chunks bound into %d ranges (%.0f MB imported), %llu imported alone", (unsigned long long)q, g_s48_n,
        g_s48_bytes / 1048576.0, (unsigned long long)g_s48_alone); }
    return 0;
}
// pool p's chunk [al, al + size) of the backing as a buffer in an imported range. 0 = made (p->buf, p->zero_copy; p->mem stays VK_NULL_HANDLE: the range's)
static int s48_bind(vkpool_t *p, uint8_t *al, uint64_t size) {
    if (!s48_blk() || !V.has_host_import) return -1;
    VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m; VkDeviceSize o;
    if (vkCreateBuffer(V.dev, &bi, NULL, &b)) return -1;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
    if (s48_mem(al, req.size > size ? req.size : size, req.memoryTypeBits, req.alignment, &m, &o)) { vkDestroyBuffer(V.dev, b, NULL); return -1; }
    const int k = g_s48_last;
    if (vkBindBufferMemory(V.dev, b, m, o)) { g_s48[k].nb--; vkDestroyBuffer(V.dev, b, NULL); return -1; }
    p->buf = b; p->mem = VK_NULL_HANDLE; p->zero_copy = 1; p->s48 = k + 1; pool_gpu_addr(p);
    return 0;
}
static int pool_zero_copy(vkpool_t *p, uint8_t *mem, uint64_t size) {
#if defined(DM_NXN1X)
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S23_ALIAS"); on = !e || !*e || atoi(e) != 0; }
    uint8_t *al = on ? (uint8_t *)nvnh_import_alias(mem, size) : NULL;
    if (al) {
        int r = s48_bind(p, al, size) ? pool_zero_copy_at(p, al, size) : 0; static int n_ok, n_bad;   // s48: into a range first
        if (!r) { if (!n_ok++) VKLOG("s23: memory pool chunks are imported through NXN1X's backing memory, not at the game's addresses (first: %p +%#llx at %p; DM_VK_S23_ALIAS=0 off)",
            (void *)mem, (unsigned long long)size, (void *)al); return 0; }
        if (r == -1) return r;
        if (n_bad++ < 8) VKLOG("[RED] s23: memory pool chunk %p +%#llx: its import through NXN1X's backing at %p failed (%d); imported at the game's address", (void *)mem,
            (unsigned long long)size, (void *)al, r);
    } else if (on) { static int n_none; if (n_none++ < 8) VKLOG("s23: memory pool chunk %p +%#llx is not one range of NXN1X's backing memory (a page unmapped, or not contiguous): imported at the game's address",
        (void *)mem, (unsigned long long)size); }
#endif
    return pool_zero_copy_at(p, mem, size);
}
static int pool_zero_copy_at0(vkpool_t *p, uint8_t *mem, uint64_t size);
static int pool_zero_copy_at(vkpool_t *p, uint8_t *mem, uint64_t size) {   // s67: counted and timed
    uint64_t t = s47_ns(); int r = pool_zero_copy_at0(p, mem, size);
    __atomic_fetch_add(&g_s67_ev[2], 1, __ATOMIC_RELAXED); __atomic_fetch_add(&g_s67_ev[3], s47_ns() - t, __ATOMIC_RELAXED); return r;
}
static int pool_zero_copy_at0(vkpool_t *p, uint8_t *mem, uint64_t size) {
    if (!V.has_host_import || !V.get_host_ptr_props) return -1;
    if (V.host_align && (((uintptr_t)mem | size) & (V.host_align - 1))) return -3;
    VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (NVNH_HOST_PROPERTIES((PFN_vkGetMemoryHostPointerPropertiesEXT)V.get_host_ptr_props, V.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, mem, &hp) || !hp.memoryTypeBits) return -6;
    VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &p->buf)) return -2;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, p->buf, &req);
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, V.has_bda ? &g_pool_af : NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, mem};
    uint32_t type = vk_mem_type(hp.memoryTypeBits & req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, type};
#if defined(DM_NXN1X)
    if (type == UINT32_MAX) nvnh_memory_local_reject(__func__, "no-compatible-import-memory-type", size);
#endif
    if (type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &p->mem) || vkBindBufferMemory(V.dev, p->buf, p->mem, 0)) { vkDestroyBuffer(V.dev, p->buf, NULL); p->buf = VK_NULL_HANDLE; return -2; }
    p->zero_copy = 1; pool_gpu_addr(p);
    return 0;
}
#if defined(DM_NXN1X)
// s29: a bridge (s13: a range across two of a pool's per-view chunks) as one sparse buffer over two pieces imported through NXN1X's backing alias (s23), one on
// each side of the view boundary it crosses (vkpool_t mem and mem2), when NXN1X's device was made with sparseBinding (Qualcomm's driver offers it; Turnip only
// where its backend sets has_sparse) and its graphics queue binds sparse memory. The GPU reads and writes the game's own pages through one buffer: no copy before
// each submission, no write watch, nothing copied back. The two views' pages are not adjacent in the backing and the driver refuses one import across two views
// (0xc000000d), so bridges were copies, and Xenoblade 3 run 32 (11-12) froze on a copied bridge's write watch. Fully bound, no residency: only sparseBinding is
// needed. Each piece must be whole sparse blocks: bridge_make places the bridge so (sparse_fit). Otherwise, or when a step fails, the bridge is made as before
// (s25 alias, s28 copy). DM_VK_S29_SPARSE=0: off
static int g_sparse = -1;   // -1: not looked at yet, 0 off, 1 on
static VkDeviceSize g_sparse_align; static uint32_t g_sparse_types;
static PFN_vkQueueBindSparse p_vkQueueBindSparse;
static int sparse_ok(void) {
    if (g_sparse >= 0) return g_sparse;
    const char *why = NULL, *e = getenv("DM_VK_S29_SPARSE");
    if (e && *e && !atoi(e)) why = "DM_VK_S29_SPARSE=0";
    else if (!g_imp_set || !g_imp.core.sparseBinding) why = "NXN1X's device was made without sparseBinding (its driver does not offer it)";
    else if (!V.has_host_import || !V.get_host_ptr_props) why = "no host memory import";
    if (!why) { VkQueueFamilyProperties qp[16]; uint32_t n = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &n, qp);
        if (V.qf >= n || !(qp[V.qf].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) why = "its graphics queue does not bind sparse memory"; }
    if (!why && !(p_vkQueueBindSparse = (PFN_vkQueueBindSparse)vkGetDeviceProcAddr(V.dev, "vkQueueBindSparse"))) why = "no vkQueueBindSparse";
    if (!why) { PFN_vkGetPhysicalDeviceExternalBufferProperties ebp = (PFN_vkGetPhysicalDeviceExternalBufferProperties)p_vkGetInstanceProcAddr(V.inst, "vkGetPhysicalDeviceExternalBufferProperties");
        VkPhysicalDeviceExternalBufferInfo bi = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO, NULL, VK_BUFFER_CREATE_SPARSE_BINDING_BIT, POOL_USAGE, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
        VkExternalBufferProperties bp = {VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        if (!ebp) why = "no vkGetPhysicalDeviceExternalBufferProperties";
        else { ebp(V.pd, &bi, &bp); if (!(bp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) why = "its driver imports no host memory for sparse buffers"; } }
    if (!why) {   // the sparse block size and the memory types such a buffer takes, from a probe buffer
        VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, VK_BUFFER_CREATE_SPARSE_BINDING_BIT, 4u << 20, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer b = VK_NULL_HANDLE;
        if (vkCreateBuffer(V.dev, &bi, NULL, &b)) why = "a sparse buffer could not be made";
        else { VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req); vkDestroyBuffer(V.dev, b, NULL); g_sparse_align = req.alignment; g_sparse_types = req.memoryTypeBits;
            if (!g_sparse_align || (g_sparse_align & (g_sparse_align - 1)) || g_sparse_align > (1u << 20) || (V.host_align > g_sparse_align)) why = "its sparse block size is not usable";
            else if (vk_mem_type(g_sparse_types, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0) == UINT32_MAX)
                why = "no host-visible coherent memory type for the sparse buffer"; } }
    g_sparse = !why;
    VKLOG("s29: bridges as sparse buffers %s%s (sparse block %llu KB, memory types %#x; DM_VK_S29_SPARSE=0 off)", why ? "off: " : "on", why ? why : "", (unsigned long long)(g_sparse_align >> 10), g_sparse_types);
    return g_sparse;
}
// NVN2 sparse layered images (gfx_vk_tex.c s2sp_*): sparse image binding needs only the device's sparseBinding, a queue that binds sparse memory and vkQueueBindSparse
// (sparse_ok above also asks for the host memory import that s29's buffers need; these images take device memory)
int vk_sparse_bind_ok(void) {
    static int ok = -1; if (ok >= 0) return ok;
    ok = 0;
    if (g_imp_set && g_imp.core.sparseBinding) {
        VkQueueFamilyProperties qp[16]; uint32_t n = 16; vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &n, qp);
        if (V.qf < n && (qp[V.qf].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)) {
            if (!p_vkQueueBindSparse) p_vkQueueBindSparse = (PFN_vkQueueBindSparse)vkGetDeviceProcAddr(V.dev, "vkQueueBindSparse");
            ok = p_vkQueueBindSparse != NULL; } }
    VKLOG("NVN2 sparse: sparse image binding %s", ok ? "available" : "not available (sparseBinding, a sparse queue or vkQueueBindSparse missing)");
    return ok;
}
// NVN2 2D sparse tiles (gfx_vk_tex.c s2tile_*, the emu's UsesSparseTiles): images with residency need NXN1X's device made with sparseResidencyImage2D (the emu enables it on
// Turnip) and the standard 2D block shape (a 64 KiB block of the standard texel shape for each size, which a guest sparse texture's 64 KB tile matches). Logged once
int vk_sparse_tiles_ok(void) {
    static int ok = -1; if (ok >= 0) return ok;
    int res = 0;
#if defined(DM_NXN1X)
    res = g_imp_set && g_imp.core.sparseResidencyImage2D;
#endif
    const VkPhysicalDeviceSparseProperties *sp = &V.props.sparseProperties;
    ok = vk_sparse_bind_ok() && res && sp->residencyStandard2DBlockShape;
    VKLOG("NVN2 sparse tiles: %s (sparseResidencyImage2D enabled %d, residencyStandard2DBlockShape %d, residencyNonResidentStrict %d, residencyAlignedMipSize %d)",
        ok ? "available" : "not available: big 2D images in sparse pools take whole memory", res, sp->residencyStandard2DBlockShape, sp->residencyNonResidentStrict, sp->residencyAlignedMipSize);
    return ok;
}
// bak83: NVN2 sparse tiles shared by physical memory (gfx_vk_tex.c ps_*) bind one memory range into several images (and several places of one), which takes
// sparseResidencyAliased on the device (Eden enables the core features its driver offers; our Turnip offers it: _xfer/tile_probe/feat_probe.txt) and SPARSE_ALIASED images
int vk_sparse_aliased_ok(void) {
    static int ok = -1; if (ok >= 0) return ok;
    int en = 0;
#if defined(DM_NXN1X)
    en = g_imp_set && g_imp.core.sparseResidencyAliased;
#endif
    ok = en;
    VKLOG("NVN2 sparse tiles: aliased residency %s (sparseResidencyAliased enabled %d)", ok ? "available" : "not available: tiles are not shared by physical memory", en);
    return ok;
}
void vk_image_sparse_reqs(VkImage img, uint32_t *n, VkSparseImageMemoryRequirements *r) {
    static PFN_vkGetImageSparseMemoryRequirements f; if (!f) f = (PFN_vkGetImageSparseMemoryRequirements)vkGetDeviceProcAddr(V.dev, "vkGetImageSparseMemoryRequirements");
    if (!f) { *n = 0; return; }
    f(V.dev, img, n, r);
}
// Bound before the submission that uploads and samples the memory is recorded further (the replay thread, as the emu path's BindLayers waits for its fence)
int vk_bind_sparse_wait(const VkBindSparseInfo *bs) {
    if (!vk_sparse_bind_ok()) return -1;
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f = VK_NULL_HANDLE;
    if (vkCreateFence(V.dev, &fi, NULL, &f)) return -2;
    vk_queue_lock(); VkResult r = p_vkQueueBindSparse(V.q, 1, bs, f); vk_queue_unlock();
    if (r) { vkDestroyFence(V.dev, f, NULL); return -3; }
    if (vkWaitForFences(V.dev, 1, &f, VK_TRUE, 2000000000ull) != VK_SUCCESS) return -4;   // a bind still pending: its fence is kept
    vkDestroyFence(V.dev, f, NULL); return 0;
}
// s29: [*lo, *hi), the bridge bridge_make wants for [addr, addr + len) across the end b of chunk p, moved onto whole sparse blocks on each side of b, inside the two
// views that meet at b. Left as it is when that cannot hold the range (a view too short there) or b is not where two views meet
static void sparse_fit(const vkpool_t *p, uint64_t addr, uint64_t len, uint64_t *lo, uint64_t *hi) {
    if (!sparse_ok()) return;
    const uint64_t a = g_sparse_align, b = p->hi; uint64_t c[8]; int n;
    if (b <= *lo || b >= *hi) return;
    if (nvnh_range_views((const void *)(uintptr_t)(b - 0x1000), 0x2000, c, 8) != 1 || c[0] != 0x1000) return;
    uint64_t vlo = *lo, vhi = *hi;
    if ((n = nvnh_range_views((const void *)(uintptr_t)vlo, b - vlo, c, 8)) > 0) { if (n >= 8) return; vlo += c[n - 1]; }
    if ((n = nvnh_range_views((const void *)(uintptr_t)b, vhi - b, c, 8)) > 0) vhi = b + c[0];
    const uint64_t l = b - (b - vlo) / a * a, h = b + (vhi - b) / a * a;
    if (l > addr || h < addr + len || l >= b || h <= b) return;
    *lo = l; *hi = h;
}
// s29: the bridge [lo, hi) as one sparse buffer over its two pieces. 0: made (p->buf, p->mem, p->mem2); negative: not, nothing kept (the bridge is made as before)
static int sparse_bridge(vkpool_t *p, uint64_t lo, uint64_t hi) {
    if (!sparse_ok()) return -8;
    static int nlog; const char *step = NULL; uint64_t c[2]; int rc = 0;
    VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory m[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE}; uint64_t n[2] = {0, 0}; uint8_t *at[2] = {NULL, NULL};
    VkDeviceMemory m48[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE}; VkDeviceSize o48[2] = {0, 0};   // s48: a piece in an imported range (never freed here)
    int r48[2] = {-1, -1};   // NVN2 (11-16): the range each s48 piece binds into, given back (s48_unref) with the bridge, so a range no buffer holds can be freed
    if (nvnh_range_views((const void *)(uintptr_t)lo, hi - lo, c, 2) != 1) { rc = -9; step = "it is not two views meeting once"; }
    else { n[0] = c[0]; n[1] = hi - lo - c[0]; if (n[0] % g_sparse_align || n[1] % g_sparse_align) { rc = -10; step = "its pieces are not whole sparse blocks"; } }
    for (int k = 0; k < 2 && !rc; k++) if (!(at[k] = (uint8_t *)nvnh_import_alias((const void *)(uintptr_t)(lo + (k ? n[0] : 0)), n[k]))) { rc = -11; step = "a piece is not one range of NXN1X's backing memory"; }
    VkMemoryRequirements req = {0};
    if (!rc) { VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, VK_BUFFER_CREATE_SPARSE_BINDING_BIT, hi - lo, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(V.dev, &bi, NULL, &buf)) { buf = VK_NULL_HANDLE; rc = -12; step = "making the sparse buffer"; }
        else { vkGetBufferMemoryRequirements(V.dev, buf, &req); if (!req.alignment || n[0] % req.alignment || n[1] % req.alignment) { rc = -10; step = "its pieces are not whole sparse blocks of this buffer"; } } }
    for (int k = 0; k < 2 && !rc; k++) {
        VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        if (NVNH_HOST_PROPERTIES((PFN_vkGetMemoryHostPointerPropertiesEXT)V.get_host_ptr_props, V.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, at[k], &hp)) { rc = -13; step = "the piece's host pointer properties"; break; }
        uint32_t type = vk_mem_type(hp.memoryTypeBits & req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, V.has_bda ? &g_pool_af : NULL, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, at[k]};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, n[k], type};
        if (type == UINT32_MAX) { rc = -14; step = "no memory type both the piece and the sparse buffer take"; }
        else if (s48_sparse_on() && !s48_mem(at[k], n[k], req.memoryTypeBits, req.alignment, &m48[k], &o48[k])) { m[k] = m48[k]; r48[k] = g_s48_last; }   // s48
        else if (vkAllocateMemory(V.dev, &ai, NULL, &m[k])) { m[k] = VK_NULL_HANDLE; rc = -15; step = "importing a piece"; }
    }
    if (!rc) {
        VkSparseMemoryBind sb[2] = {{0, n[0], m[0], o48[0], 0}, {n[0], n[1], m[1], o48[1], 0}};   // (s48: offsets in ranges)
        VkSparseBufferMemoryBindInfo bb = {buf, 2, sb};
        VkBindSparseInfo bs = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, NULL, 0, NULL, 1, &bb, 0, NULL, 0, NULL, 0, NULL};
        VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f = VK_NULL_HANDLE;
        if (vkCreateFence(V.dev, &fi, NULL, &f)) { rc = -16; step = "making a fence"; }
        else { vk_queue_lock(); VkResult r = p_vkQueueBindSparse(V.q, 1, &bs, f); vk_queue_unlock();
            if (r) { rc = -17; step = "vkQueueBindSparse"; vkDestroyFence(V.dev, f, NULL); }
            else if (vkWaitForFences(V.dev, 1, &f, VK_TRUE, 2000000000ull) != VK_SUCCESS) {   // a bind still pending: its buffer, memory and fence are kept (never used)
                if (nlog++ < 16) VKLOG("[RED] s29: bridge %#llx +%#llx: the sparse bind did not finish in 2 s: made as before", (unsigned long long)lo, (unsigned long long)(hi - lo));
                return -18; }
            else vkDestroyFence(V.dev, f, NULL); }
    }
    if (rc) {
        if (buf) vkDestroyBuffer(V.dev, buf, NULL);
        for (int k = 0; k < 2; k++) if (m[k] && m[k] != m48[k]) vkFreeMemory(V.dev, m[k], NULL);   // (s48: a range's memory stays)
        for (int k = 0; k < 2; k++) if (r48[k] >= 0) s48_unref(r48[k]);   // (its count back: the piece is not kept)
        if (nlog++ < 16) VKLOG("%ss29: bridge %#llx +%#llx: %s (%d): made as before", rc <= -11 ? "[RED] " : "", (unsigned long long)lo, (unsigned long long)(hi - lo), step, rc);
        return rc;
    }
    p->buf = buf; p->mem = m48[0] ? VK_NULL_HANDLE : m[0]; p->mem2 = m48[1] ? VK_NULL_HANDLE : m[1]; p->zero_copy = 1; pool_gpu_addr(p);
    p->s48 = r48[0] + 1; p->s48b = r48[1] + 1;   // NVN2 (11-16): its ranges, given back where the pool's chunks give theirs (pool_release, mtl_pool_del)   // (s48: a range's memory is never freed with the pool)
    if (nlog++ < 16) VKLOG("s29: bridge %#llx +%#llx is one sparse buffer over two pieces imported through NXN1X's backing (%#llx + %#llx, views meeting at %#llx): not copied",
        (unsigned long long)lo, (unsigned long long)(hi - lo), (unsigned long long)n[0], (unsigned long long)n[1], (unsigned long long)(lo + n[0]));
    return 0;
}
// s30c: a self-test of s29 at start. The first memory pool the game makes across a mapped-view boundary gets a sparse bridge over one sparse block on each
// side of its first boundary; the GPU copies 4 KB either side of the seam into host memory and that is compared with the game's own bytes (read only: nothing
// is written into the game's memory). Logs "s30c:" once per process: whether bridges can be sparse here (the s29 line comes with it) and whether the GPU reads
// the right pages on both sides. Bayonetta 3 (11-13): a copied bridge across such a boundary lost the GPU's writes and drew stale buildings in some runs.
// DM_VK_S30C_SELFTEST=0: off
static void s30c_selftest(uint64_t base, uint64_t size, const uint64_t *cut, int nc) {
    static int done;
    if (done || !g_imp_set) return;   // (sparse_ok decides once: never before NXN1X's device is known)
    const char *e = getenv("DM_VK_S30C_SELFTEST");
    if (e && *e && !atoi(e)) { done = 1; return; }
    if (!sparse_ok()) { done = 1; VKLOG("s30c: sparse self-test skipped: bridges cannot be sparse buffers here (the s29 line says why)"); return; }
    const uint64_t a = g_sparse_align; const VkDeviceSize n = 0x2000;
    for (int k = 0; k < nc; k++) {
        uint64_t next = k + 1 < nc ? cut[k + 1] : size, c[2];
        if (cut[k] < a || next - cut[k] < a || a < n / 2) continue;
        uint64_t b = base + cut[k], lo = b - a, hi = b + a;
        if (nvnh_range_views((const void *)(uintptr_t)lo, hi - lo, c, 2) != 1 || c[0] != a) continue;
        done = 1;
        vkpool_t t; memset(&t, 0, sizeof t); t.lo = lo; t.hi = hi; t.base = base; t.end = base + size; t.bridge = 1;
        int rc = sparse_bridge(&t, lo, hi);
        if (rc) { VKLOG("[RED] s30c: sparse self-test at the view boundary %#llx: no sparse bridge (%d, the s29 line says why)", (unsigned long long)b, rc); return; }
        VkBuffer rb = VK_NULL_HANDLE; VkDeviceMemory rm = VK_NULL_HANDLE; void *rp = NULL; VkCommandPool cp = VK_NULL_HANDLE; VkCommandBuffer cb = VK_NULL_HANDLE; VkFence f = VK_NULL_HANDLE;
        const char *step = NULL;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        if (vkCreateBuffer(V.dev, &bi, NULL, &rb)) { rb = VK_NULL_HANDLE; step = "a readback buffer"; }
        if (!step) { VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, rb, &req);
            uint32_t ty = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, ty};
            if (ty == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &rm)) { rm = VK_NULL_HANDLE; step = "readback memory"; }
            else if (vkBindBufferMemory(V.dev, rb, rm, 0) || vkMapMemory(V.dev, rm, 0, VK_WHOLE_SIZE, 0, &rp)) step = "mapping the readback memory"; }
        if (!step) { VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, V.qf};
            if (vkCreateCommandPool(V.dev, &pi, NULL, &cp)) { cp = VK_NULL_HANDLE; step = "a command pool"; } }
        if (!step) { VkCommandBufferAllocateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            if (vkAllocateCommandBuffers(V.dev, &ci, &cb)) step = "a command buffer"; }
        if (!step) { VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; if (vkCreateFence(V.dev, &fi, NULL, &f)) { f = VK_NULL_HANDLE; step = "a fence"; } }
        if (!step) {
            VkCommandBufferBeginInfo bb = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            vkBeginCommandBuffer(cb, &bb);
            VkBufferCopy cr = {a - n / 2, 0, n}; vkCmdCopyBuffer(cb, t.buf, rb, 1, &cr);
            VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            vk_f0_result(vkEndCommandBuffer(cb), "vkEndCommandBuffer(pool-bridge)", 0);
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL};
            vk_queue_lock(); VkResult r = vkQueueSubmit(V.q, 1, &si, f); vk_queue_unlock();
            vk_f0_result(r, "vkQueueSubmit(pool-bridge)", 0);
            if (r) step = "the submission";
            else if (vkWaitForFences(V.dev, 1, &f, VK_TRUE, 2000000000ull) != VK_SUCCESS) {   // still pending: everything is kept (never used again)
                VKLOG("[RED] s30c: sparse self-test at the view boundary %#llx: the GPU copy did not finish in 2 s", (unsigned long long)b); return; }
        }
        if (step) VKLOG("[RED] s30c: sparse self-test at the view boundary %#llx: %s failed", (unsigned long long)b, step);
        else {
            const uint8_t *g = (const uint8_t *)(uintptr_t)(b - n / 2), *v = (const uint8_t *)rp; uint64_t bad = 0, first = 0;
            for (uint64_t i = 0; i < n; i++) if (g[i] != v[i]) { if (!bad++) first = i; }
            uint64_t nz = 0; for (uint64_t i = 0; i < n; i++) nz += g[i] != 0;
            if (!bad) VKLOG("s30c: sparse self-test at the view boundary %#llx (pool %#llx +%#llx, sparse block %llu KB): the GPU reads the game's bytes on both sides (%llu of %llu nonzero): sparse bridges work",
                (unsigned long long)b, (unsigned long long)base, (unsigned long long)size, (unsigned long long)(a >> 10), (unsigned long long)nz, (unsigned long long)n);
            else VKLOG("[RED] s30c: sparse self-test at the view boundary %#llx: %llu of %llu bytes differ, first at seam %+lld (the game may have written meanwhile; if it is a whole side, the bind is wrong)",
                (unsigned long long)b, (unsigned long long)bad, (unsigned long long)n, (long long)first - (long long)(n / 2));
        }
        if (f) vkDestroyFence(V.dev, f, NULL);
        (void)cp;   // s30c: kept, as the host's other command pools (vkDestroyCommandPool is not in gfx_vk.h's entry-point list)
        if (rb) vkDestroyBuffer(V.dev, rb, NULL);
        if (rm) vkFreeMemory(V.dev, rm, NULL);
        vkDestroyBuffer(V.dev, t.buf, NULL); vkFreeMemory(V.dev, t.mem, NULL); vkFreeMemory(V.dev, t.mem2, NULL);
        return;
    }
}
#endif
// s25: a bridge (s13: a range across two of a pool's per-view chunks) is imported through NXN1X's backing alias (s23) when its pages are one range there, instead
// of copied. The import WDDM refused across two of the game's views (s12) is one view in the backing, so the bridge is the same memory as the chunks beside it:
// no copy refreshed before each submission, and nothing copied back over what the GPU wrote through those chunks (a copied bridge's copy-back put its stale copy
// there). Xenoblade 3 run 30 (11-12): the first copied bridge, 37 ms after it was made, an access violation in the completion thread's copy. -7: copied as before
// (no alias: the two views' pages are not adjacent in the backing). DM_VK_S25_BRIDGE=0: always copied
static int bridge_zero_copy(vkpool_t *p, uint8_t *mem, uint64_t size) {
#if defined(DM_NXN1X)
    if (!sparse_bridge(p, (uint64_t)(uintptr_t)mem, (uint64_t)(uintptr_t)mem + size)) return 0;   // s29
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S25_BRIDGE"); on = !e || !*e || atoi(e) != 0; }
    uint8_t *al = on ? (uint8_t *)nvnh_import_alias(mem, size) : NULL;
    int r = al ? (s48_bind(p, al, size) ? pool_zero_copy_at(p, al, size) : 0) : -7;   // s48: into a range first
    static int n; if (on && n++ < 16) VKLOG("%ss25: bridge %p +%#llx %s (%d)", r ? "[RED] " : "", (void *)mem, (unsigned long long)size,
        !al ? "is not one range of NXN1X's backing memory: copied" : r ? "refused through NXN1X's backing: copied" : "imported through NXN1X's backing, not copied", r);
    return r ? -7 : 0;
#else
    (void)p; (void)mem; (void)size; return -7;
#endif
}
// (r78's big pools in chunks by stride, made for a Switch 2 game's 2.4 GB pool on a Mac, are deleted: s13, NPARK 11-12. NXN1X makes a pool one chunk per
// mapped view: mtl_pool_add)
static int pool_lazy(void); static uint64_t g_reg_bytes, g_zc_bytes, g_cp_bytes;   // (r77, below)
// nvn_dirty_defrag: the Android port maps a zero-copy pool's dma-buf again in one piece; an imported pool is the game's own pages, nothing to map. 0 = done
int vk_pool_remap_ro(void *mem, size_t len) {
    int rc = 0; uint64_t done = 0;
    pool_rdlock();
    for (int i = 0; i < g_npool && !rc; i++) {
        vkpool_t *p = &g_pool[i]; if (p->base != (uint64_t)(uintptr_t)mem || p->end - p->base != len) continue;
        if (p->buf && !p->zero_copy && !p->shadow) { rc = -1; break; }   // (no buffer yet, r77: nothing mapped either)
        done += p->hi - p->lo;
    }
    pool_unlock();
    return !rc && done == len ? 0 : -1;
}
// Pools over ZC_CHUNK are made of chunks of at most that size, each a pool entry with a buffer of its own (the Android port's limit for its hardware buffers, kept: DK's
// 9 GB pool is more than one buffer's maxStorageBufferRange anyway). A range across two chunks has no buffer (vk_pool_of)
#define ZC_CHUNK 0x10000000ull
// Pages both the CPU and the GPU wrote while a submission ran: pools_copy_back used to leave the whole page as the CPU wrote it (copying it back put old bytes over the
// game's new ones: r14), and the next sync then put the CPU's page over the GPU's copy, so the GPU's writes there were lost (DK, 11-04: the terrain's triangles, which its
// geometry shader writes to memory, flickered in and out; 400-600 such pages every 120 submissions, 150-350 more from raw global memory). Merged word by word instead: the
// content the page had at the sync is kept when the CPU first writes it (the dirty tracking's fault, before the page is made writable: snap_hook), and a word the GPU
// changed from it goes back unless the CPU changed that word too (then the CPU's stays). A page with no snapshot of this sync is left as before.
// SNAP_N slots by page hash, a later page taking a slot over: a slot's first snapshot of the current sync is kept (the page's first write after it)
enum { SNAP_N = 16384, SNAP_PG = 4096 };
typedef struct { _Atomic uint64_t seq, page, taken; uint8_t d[SNAP_PG]; } snap_t;
static snap_t *g_snaps; static _Atomic uint64_t g_snap_epoch;   // epoch: the dirty clock at the end of the latest sync (vk_pools_sync), which the completion thread's copy-back runs under
static _Atomic uint64_t g_snap_taken; static uint64_t g_mg_pages, g_mg_words, g_mg_both, g_mg_none;
static inline unsigned snap_slot(uint64_t pg) { return (unsigned)(((pg >> 12) * 0x9E3779B97F4A7C15ull) >> (64 - 14)); }
static void snap_hook(uintptr_t a, size_t len, uint64_t now) {   // the fault path: atomics and memcpy only
    snap_t *s = g_snaps; if (len != SNAP_PG || !s) return;
    s += snap_slot(a); uint64_t q = atomic_load(&s->seq), ep = atomic_load(&g_snap_epoch);
    if ((q & 1) || (atomic_load(&s->page) == a && atomic_load(&s->taken) >= ep)) return;   // being written; or this sync's first already kept
    if (!atomic_compare_exchange_strong(&s->seq, &q, q + 1)) return;
    if (atomic_load(&s->page) == a && atomic_load(&s->taken) >= ep) { atomic_store(&s->seq, q + 2); return; }
    atomic_store(&s->page, a); atomic_store(&s->taken, now); memcpy(s->d, (const void *)a, SNAP_PG);
    atomic_store(&s->seq, q + 2); atomic_fetch_add(&g_snap_taken, 1);
}
// [a, e) in one page of pool p, which the CPU wrote after the sync: the GPU's words merged into the guest's page. 0 = no snapshot of this sync (left as it is)
void dm_guest_write_prepare(void *addr, size_t len);
static int snap_merge(const vkpool_t *p, uint64_t a, uint64_t e) {
    uint64_t pg = a & ~(uint64_t)(SNAP_PG - 1); if (!g_snaps) return 0;
    snap_t *s = g_snaps + snap_slot(pg); static uint8_t base[SNAP_PG];   // (the completion thread alone)
    uint64_t q = atomic_load(&s->seq);
    if ((q & 1) || atomic_load(&s->page) != pg || atomic_load(&s->taken) < atomic_load(&g_snap_epoch)) { g_mg_none++; return 0; }
    memcpy(base, s->d, SNAP_PG);
    if (atomic_load(&s->seq) != q) { g_mg_none++; return 0; }
    g_mg_pages++;
    uint64_t lo = a & ~3ull, hi = (e + 3) & ~3ull; if (lo < p->lo) lo = p->lo; if (hi > p->hi) hi = p->hi; if (hi > pg + SNAP_PG) hi = pg + SNAP_PG;
#if defined(DM_NXN1X)
    if (hi > lo) nvnc_cpu(lo, hi - lo, NVNC_GPU_WRITEBACK); // possible partial merge, no CPU-generation promotion
#endif
    for (uint64_t x = lo; x + 4 <= hi; x += 4) { uint32_t b, sv, gv; memcpy(&b, base + (x - pg), 4); memcpy(&sv, p->shadow + (x - p->lo), 4);
        if (sv == b) continue;
        memcpy(&gv, (const void *)(uintptr_t)x, 4);
        if (gv == b) { dm_guest_write_prepare((void *)(uintptr_t)x, 4); memcpy((void *)(uintptr_t)x, &sv, 4); g_mg_words++; } else g_mg_both++; }   // (prepare: the page may be protected again)
    return 1;
}
static void pool_add_one(const void *mem, uint64_t size, uint64_t base, uint64_t end);
// DM_VK_ZEROCOPY_MAX_MB=<n>: pools bigger than n MB are copied, not zero-copy, and kept whole (a copied pool takes no hardware buffer, so no chunks). The Legion Y700
// 2023's gralloc aborts the process (no error returned) allocating a hardware buffer for Unicorn Overlord's 860 MB pool, while its pools up to 250 MB are fine
static int zc_over(uint64_t size) { static uint64_t m = 1; if (m == 1) { const char *e = getenv("DM_VK_ZEROCOPY_MAX_MB"); m = e && atoi(e) > 0 ? (uint64_t)atoi(e) << 20 : 0; } return m && size > m; }
void mtl_pool_add(const void *mem, uint64_t size) {
    vk_init();
    if (!size) return;
#if defined(DM_NXN1X)
    // s12: NXN1X maps each block its kernel's page heap hands out (at most 1 GB physically contiguous) as a view of its own, and the driver refuses an import
    // across two views (0xc000000d). r78's chunks by stride (deleted) crossed them: Bayonetta 3's title had an 80 MB chunk refused and copied, and the masked head lost its body
    // when that copy came first. So a pool is made one chunk per mapped view, each imported at its first GPU use; a range across two views has no buffer (as one
    // across two chunks). A view over ZC_CHUNK is not cut further: Windows has no hardware-buffer limit (a binding's range is bounded by the device, not the chunk)
    if (!(((uintptr_t)mem | size) & 0xfff)) {
        uint64_t cut[16]; int nc = nvnh_range_views(mem, size, cut, 16);
        if (nc || size > ZC_CHUNK) {
            uint64_t a = (uint64_t)(uintptr_t)mem, from = 0; int nx = 0;
            // NVN2 (Switch 2 boots only, dm_sdk_ounce): a view over S2_VIEW_MAX is cut further, inside the view, into S2_CHUNK chunks, each imported at its first GPU use.
            // HWAoI (11-15 run 2): one 2 GB view (0x3285000000) was one chunk; with 1460 MB already imported its import was refused (-2; the driver's limit near 3.4 GB,
            // r77 above), so it was copied, and the budget released and re-copied the 2 GB copy 29 times (0.6-1.8 s with the GPU idle each time). In chunks, only
            // what the GPU reads as buffers is imported; a range across two chunks of one view gets a bridge (s13), which imports within the view.
            // Switch 1 keeps one chunk per view (s12, s13). DM_VK_S2_CHUNK_MB=0 off
            static uint64_t s2c = 1; if (s2c == 1) { const char *e = getenv("DM_VK_S2_CHUNK_MB"); s2c = (uint64_t)(e && *e ? atoll(e) : 64) << 20; }
            const uint64_t S2_VIEW_MAX = 512ull << 20;
            const int s2 = dm_sdk_ounce() && s2c && !(s2c & 0xfff);
            for (int k = 0; k <= nc; k++) { uint64_t to = k < nc ? cut[k] : size;
                if (s2 && to - from > S2_VIEW_MAX) {
                    for (uint64_t o = from; o < to; o += s2c) { uint64_t e = to - o < s2c ? to : o + s2c; pool_add_one((const uint8_t *)mem + o, e - o, a, a + size); nx++; }
                } else if (to > from) pool_add_one((const uint8_t *)mem + from, to - from, a, a + size);
                from = to; }
            if (nx) VKLOG("NVN2 s12: memory pool %p +%#llx: views over %llu MB cut into %llu MB chunks inside the view (%d such chunks), each made at its first GPU use (DM_VK_S2_CHUNK_MB=0 off)",
                mem, (unsigned long long)size, (unsigned long long)(S2_VIEW_MAX >> 20), (unsigned long long)(s2c >> 20), nx);
            VKLOG("s12: memory pool %p +%#llx: %d chunk%s, one per mapped view, each made at its first GPU use (%.1f MB in pools)", mem, (unsigned long long)size, nc + 1,
                nc ? "s" : "", g_reg_bytes / 1048576.0);
            __atomic_fetch_add(&g_s67_ev[4], 1, __ATOMIC_RELAXED);   // s67
            s30c_selftest(a, size, cut, nc);   // s30c: once per process, at the first pool across a view boundary
            return; } }
#endif
    uint64_t a = (uint64_t)(uintptr_t)mem, c = ((a | size) & 0xfff) || zc_over(size) ? size : ZC_CHUNK;
    for (uint64_t o = 0; o < size; o += c) pool_add_one((const uint8_t *)mem + o, size - o < c ? size - o : c, a, a + size);
    if (size > c) VKLOG("memory pool %p +%#llx: %llu chunks of at most %#llx", mem, (unsigned long long)size, (unsigned long long)((size + c - 1) / c), c);
}
// r77: a chunk's GPU buffer is made on its first GPU use (vk_pool_of), not when the game makes the pool. Imported or copied whole up front, every pool pinned or
// duplicated all of the game's pool memory: Hyrule Warriors: Age of Imprisonment (11-08) has 5 GB of pools, much of it texture storage the GPU never reads as a buffer
// (textures are uploaded from the CPU's pages through the scratch buffers). 3.4 GB were imported and the driver refused more (0xc0000017), 1.5 GB more were copied,
// and at about 210 s of a battle the GPU's share of system memory (7.8 GB, half the RAM) ran out. Raw global memory reaches the chunks made so far (vk_gmem_map is
// rebuilt when one is made). DM_VK_POOL_LAZY=0: every chunk made when the pool is, as before
static int pool_lazy(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_POOL_LAZY"); on = e && *e ? atoi(e) != 0 : 1; } return on; }
static uint64_t g_reg_bytes, g_zc_bytes, g_cp_bytes;   // pool memory registered; made zero-copy, made copied (under the write lock)
static uint64_t g_sync_bytes, g_sync_full; static int g_shadow_pools;   // (defined below: a chunk copied at its first use counts as a sync)
static VkResult shadow_range(vkpool_t *p, uint64_t off, uint64_t n, int inval);
extern uint64_t g_nsubs;   // submissions begun (mtl_sub_begin): a chunk's last use (used) and its next try after a failure (retry) count in them
static uint64_t g_pool_ser;   // serials, under the write lock: a chunk made again gets a new one (caches of views over its buffer key on it)
// NVN2 copied pools: exact byte comparison, and waits narrowed to work that may
// access a copy or change its guest source. No extra mirror or staging allocation.
// DM_VK_POOL_SYNC_MODE bits: 1 differential copies, 2 dependency waits (default 3).
static int g_pool_sync_mode = 3;
static pthread_once_t g_pool_sync_once = PTHREAD_ONCE_INIT;
static _Atomic uint64_t g_pool_sync_fence;
static uint64_t g_pool_sync_narrow, g_pool_sync_fullwait;
static int g_pool_compare_valid = 1;   // a refused command buffer may have lost its host-read barrier
static int g_pool_demand = 1, g_pool_demand_active;
static uint64_t g_pool_sync_epoch, g_pool_demand_pending;
static uint64_t g_pool_deferred_bytes, g_pool_demand_bytes, g_pool_guard_bytes;
static void pool_sync_deferred(vkpool_t *p, int guard);
static void pool_sync_pending(uint64_t lo, uint64_t hi);
static void pc_free_buffers(vkpool_t *p, int now);   // poolcopy (below)
extern uint64_t g_ns_poolsync;
static void pool_sync_init(void) {
    const char *e = getenv("DM_VK_POOL_SYNC_MODE");
    if (e && *e && atoi(e) >= 0 && atoi(e) <= 3) g_pool_sync_mode = atoi(e);
    e = getenv("DM_VK_POOL_DEMAND"); if (e && *e) g_pool_demand = atoi(e) != 0;
    VKLOG("NVN2 copied-pool synchronization: mode %d (1 exact differential copies, 2 dependency waits; DM_VK_POOL_SYNC_MODE=0 legacy)", g_pool_sync_mode);
    VKLOG("NVN2 copied-pool demand refresh: %s (DM_VK_POOL_DEMAND=0 eager; raw addressing and early submission refresh pending copies)", g_pool_demand ? "on" : "off");
}
static int pool_sync_mode(void) {
    if (!dm_sdk_ounce()) return 0;
    pthread_once(&g_pool_sync_once, pool_sync_init);
    return g_pool_sync_mode;
}
static void pool_fence_max(_Atomic uint64_t *dst, uint64_t fence) {
    uint64_t old = atomic_load(dst);
    while (old < fence && !atomic_compare_exchange_weak(dst, &old, fence)) {}
}
static void pool_used(vkpool_t *p) {
    if (g_pool_demand_active && p->shadow && vk_sub_open()) pool_sync_deferred(p, 0);
    p->used = g_nsubs;
    if (dm_sdk_ounce() && p->shadow && vk_sub_open()) __atomic_store_n(&p->last_binding_fence, mtl_last_fence() + 1, __ATOMIC_RELEASE);
}
static uint64_t g_rel_bytes, g_rel_n;   // r78: chunk memory released, chunks released
// r78: chunks not used for idle submissions, the longest unused first, until want bytes are released (under the write lock). The buffer goes once the GPU is past the
// current submission (deferred); the entry stays, with no buffer, and its next use makes it again (vk_pool_of): an imported chunk is the game's own pages, a copied one
// is copied again from them (the GPU's writes were copied back after each submission). Returns the bytes released
static uint64_t g_gmem_sub;   // r78: the submission that last bound raw global memory's pool table (vk_gmem_map): its chunks are reached with no lookup
static uint64_t g_gmem_rebuilds;   // NVN2 memory report: raw global memory pool tables built (vk_gmem_map)
static int g_rel_zc_only;     // NVN2: pool_budget on a Switch 2 boot releases imported chunks only (see pool_budget)
static uint64_t pool_release(uint64_t want, uint64_t idle, int now) {   // now: destroyed at once (none of them is in a submission in flight or being recorded)
    uint64_t got = 0;
    while (got < want) {
        vkpool_t *o = NULL;
        for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (p->buf && (!g_rel_zc_only || p->zero_copy) && p->used + idle < g_nsubs && (!o || p->used < o->used)) o = p; }
        if (!o) break;
        uint64_t n = o->hi - o->lo;
        if (now) { vkDestroyBuffer(V.dev, o->buf, NULL); if (o->mem) vkFreeMemory(V.dev, o->mem, NULL); if (o->mem2) vkFreeMemory(V.dev, o->mem2, NULL); }
        else { vk_defer_free_buffer(o->buf, o->mem); if (o->mem2) vk_defer_free_buffer(VK_NULL_HANDLE, o->mem2); }   // (mem2: a sparse bridge's second piece, s29)
        if (o->zero_copy) g_zc_bytes -= n; else g_cp_bytes -= n;
        if (o->s48) { s48_unref(o->s48 - 1); o->s48 = 0; }   // NVN2: its range freed when no chunk holds it
        if (o->s48b) { s48_unref(o->s48b - 1); o->s48b = 0; }   // (a sparse bridge's second piece, 11-16)
        if (o->mem) g_rel_owned += n;   // (memory of its own: freed with it)
        if (o->dl) pc_free_buffers(o, now);   // poolcopy: its shadow and readback (made again from the game's pages at its next use)
        o->buf = VK_NULL_HANDLE; o->mem = VK_NULL_HANDLE; o->mem2 = VK_NULL_HANDLE; o->shadow = NULL; o->zero_copy = 0; o->synced = 0; o->gpu = 0; o->noncoh = 0; o->retry = 0;
        got += n; g_rel_bytes += n; g_rel_n++;
    }
    if (got) __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
    return got;
}
// r78: pool chunks made are kept under DM_VK_POOL_BUDGET_MB (default 2048; 0 = no limit): over it, every 32 submissions, chunks unused for 300 submissions are released.
// HWAoI with r77 (11-08) made 256 MB chunks one after another up to the 3.4 GB the driver would import, then a copied one, and died when the next could be neither
// NVN2 (Switch 2 boots): the host's device memory by kind every 30 s, with the driver's heap usage and budget (VK_EXT_memory_budget, where the device reports it).
// HWAoI 11-15 ran out of device memory after 759 s ("FATAL vkAllocateMemory ... failed: -2", Windows event 0xc0000409) with 2358 MB of pools imported and 1305 MB of textures
static void vk_slot_stats(uint64_t o[8]);   // (below, with the submission slots)
static uint64_t g_press_n;   // NVN2 pressure trims (nvn2_pressure)
static int g_scr_squeeze;    // NVN2 pressure: recycles that trim scratch down to its floor (scratch_trim)
static int nslot_on(void);   // slot12 (the submission slots below)
static uint64_t pipes_built_now(void);   // (below, after g_vkst's declaration)
uint64_t g_views_freed;   // image views given to vk_defer_free_view (gfx_vk_tex.c counts those it makes)
#if defined(DM_NXN1X)
void nvn_eden_pipeline_counts(uint64_t out[2]);   // video_core/nvn/nvn_shader_bridge.cpp
#endif
// The heap with the highest usage against its budget: 1 and its usage and budget, 0 when the device reports no budget (VK_EXT_memory_budget absent)
static int vk_heap_worst(uint64_t *use, uint64_t *bud) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &bp};
    static PFN_vkGetPhysicalDeviceMemoryProperties2 gmp2; if (!gmp2) gmp2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)p_vkGetInstanceProcAddr(V.inst, "vkGetPhysicalDeviceMemoryProperties2");
    if (!gmp2) return 0;
    gmp2(V.pd, &mp); int got = 0; *use = *bud = 0;
    for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++) {
        if (!bp.heapBudget[i]) continue;
        if (!got || (double)bp.heapUsage[i] / (double)bp.heapBudget[i] > (double)*use / (double)*bud) { *use = bp.heapUsage[i]; *bud = bp.heapBudget[i]; got = 1; } }
    return got;
}
typedef struct { int on, hi, lo; } nvn2_pressure_limits;
static nvn2_pressure_limits g_nvn2_limits = {1, 85, 80};
static pthread_once_t g_nvn2_limits_once = PTHREAD_ONCE_INIT;
static void nvn2_limits_init(void) {
    nvn2_pressure_limits *limits = &g_nvn2_limits;
    const char *e = getenv("DM_VK_S2_PRESSURE"); limits->on = !(e && *e == '0');
    e = getenv("DM_VK_S2_PRESSURE_HI"); if (e && atoi(e) > 0 && atoi(e) <= 100) limits->hi = atoi(e);
    e = getenv("DM_VK_S2_PRESSURE_LO"); if (e && atoi(e) > 0 && atoi(e) <= 100) limits->lo = atoi(e);
    if (limits->lo > limits->hi) limits->lo = limits->hi;
}
static const nvn2_pressure_limits *nvn2_limits(void) {
    pthread_once(&g_nvn2_limits_once, nvn2_limits_init);
    return &g_nvn2_limits;
}
uint64_t vk_nvn2_memory_need(uint64_t extra) {
    uint64_t use, budget;
    if (!dm_sdk_ounce()) return 0;
    const nvn2_pressure_limits *limits = nvn2_limits();
    if (!limits->on || !vk_heap_worst(&use, &budget) || !budget) return 0;
    // Includes both allocators because this is the driver's heap usage. Keep room
    // for command recording and the 4 MiB preservation buffer before allocating.
    uint64_t reserve = extra > UINT64_MAX - (8ull << 20) ? UINT64_MAX : extra + (8ull << 20);
    uint64_t projected = use > UINT64_MAX - reserve ? UINT64_MAX : use + reserve;
    return projected >= budget / 100 * (uint64_t)limits->hi ? projected - budget / 100 * (uint64_t)limits->lo : 0;
}
// the driver's heap usage and budget (VK_EXT_memory_budget) as text, "" when the device does not report them
// Returns 1 when a heap's usage is at 90% of its budget or more, 0 when none is, -1 when the device reports no budget (VK_EXT_memory_budget absent)
static int vk_heap_text(char *heaps, int cap) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &bp};
    static PFN_vkGetPhysicalDeviceMemoryProperties2 gmp2; if (!gmp2) gmp2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)p_vkGetInstanceProcAddr(V.inst, "vkGetPhysicalDeviceMemoryProperties2");
    int o = 0, full = -1; heaps[0] = 0;
    if (gmp2) { gmp2(V.pd, &mp); for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount && i < 4 && o < cap - 48; i++) {
        o += snprintf(heaps + o, (size_t)(cap - o), " heap%u %.0f/%.0f MB", i, bp.heapUsage[i] / 1048576.0, bp.heapBudget[i] / 1048576.0);
        if (bp.heapBudget[i]) { if (full < 0) full = 0; if (bp.heapUsage[i] * 10 >= bp.heapBudget[i] * 9) full = 1; } } }
    return full;
}
static void nvn2_mem_report(void) {
    static uint64_t last; uint64_t now = s47_ns(); if (!last) last = now;
    if (now - last < 30000000000ull) return;
    last = now;
    uint64_t t[5]; vk_tex_mem_stats(t);
    static uint32_t gen0; static uint64_t rb0; uint32_t gen = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE); uint64_t rb = g_gmem_rebuilds;   // since the last line
    char heaps[256]; vk_heap_text(heaps, (int)sizeof heaps);
    VKLOG("NVN2 memory: pools zero-copy %.0f MB, copied %.0f MB (%d chunks); images %llu, %.0f MB (evicted over their memory %llu, for memory %llu); NVN2 sparse layers %.0f MB; pool table changes %u, raw global memory tables built %llu (since the last line); driver heaps (usage/budget):%s",
        g_zc_bytes / 1048576.0, g_cp_bytes / 1048576.0, g_npool, (unsigned long long)t[0], t[1] / 1048576.0, (unsigned long long)t[2], (unsigned long long)t[3], t[4] / 1048576.0,
        gen - gen0, (unsigned long long)(rb - rb0), heaps[0] ? heaps : " (not reported)");
    gen0 = gen; rb0 = rb;
    uint64_t ss[8]; vk_slot_stats(ss);
    extern uint64_t g_views_made, g_c1_bufs;
    uint64_t ep[2] = {0, 0};
#if defined(DM_NXN1X)
    nvn_eden_pipeline_counts(ep);
#endif
    extern uint64_t g_c1_arena_bytes, g_s24_skips, g_s24_upskips;
    void vk_tex_cpu_stats(uint64_t out[8]); uint64_t vk_replay_ns(void);
    uint64_t cs[8]; vk_tex_cpu_stats(cs); const uint64_t rp = vk_replay_ns(); static uint64_t cs0[8], rp0;   // NVN2 CPU split, ms since the last line
    void vk_tex_tiles_stats(uint64_t out[6]); uint64_t ts[6]; vk_tex_tiles_stats(ts); static uint64_t ts0[6];
    extern uint64_t g_nvnh_vq, g_nvnh_hm; static uint64_t vq0, hm0; const uint64_t vq = g_nvnh_vq, hm = g_nvnh_hm;   // bak84: hm, answers from the host's map
#if defined(DM_NXN1X)
    void vk_tex_s24_range_stats(uint64_t out[4]); uint64_t rs[4]; vk_tex_s24_range_stats(rs);
#else
    uint64_t rs[4] = {0, 0, 0, 0};
#endif
    static uint64_t rs0[4];   // bak82: readable answers kept by range, asked again, ring overflows; the uploads' guard time (part of the readable queries', not of the s24 checks')
    const double s24ms = (double)(cs[2] - cs0[2]) / 1e6, rdms = (double)(cs[4] - cs0[4]) / 1e6, uprd = (double)(rs[3] - rs0[3]) / 1e6;
    VKLOG("NVN2 CPU (ms in the last 30 s): replay %.0f, packaged uploads %.0f (page-table copies %.0f), s24 checks %.0f (readable queries %.0f, %.0f of it for uploads: %llu queries, answered from the host map %llu, VirtualQuery %llu, %llu answers cached; "
          "readable kept by range %llu, re-queried %llu, ring overflow %llu; hashing and the rest %.0f), sparse layers %.0f, sparse tiles %.0f",
        (rp - rp0) / 1e6, (cs[0] - cs0[0]) / 1e6, (cs[1] - cs0[1]) / 1e6, s24ms, rdms, uprd, (unsigned long long)(cs[5] - cs0[5]), (unsigned long long)(hm - hm0),
        (unsigned long long)(vq - vq0), (unsigned long long)(cs[6] - cs0[6]), (unsigned long long)(rs[0] - rs0[0]), (unsigned long long)(rs[1] - rs0[1]), (unsigned long long)(rs[2] - rs0[2]),
        s24ms - (rdms - uprd), (cs[3] - cs0[3]) / 1e6, (ts[5] - ts0[5]) / 1e6);
    vq0 = vq; hm0 = hm; memcpy(rs0, rs, sizeof rs);
    {   // bak89: copied pools (synced bytes, GPU-written pages reverted by a sync) and where indirect arguments / render conditions live
        static int pl; int np = 0; uint64_t tot = 0;
        pool_rdlock();
        for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (!p->shadow) continue; tot += p->dg_sync;
            if ((p->dg_sync || p->dg_rev) && np++ < 8) VKLOG("NVN2 copied pool %#llx +%#llx (NVN flags %#x, %s): synced %.1f MB, GPU-written pages reverted by a sync %.1f KB (last 30 s, reverts sampled every 8th submission)",
                (unsigned long long)p->lo, (unsigned long long)(p->hi - p->lo), p->flags, p->span ? "tracked: dirty blocks" : (pool_sync_mode() & 1) ? "untracked: exact block comparison" : "untracked: whole at every sync", p->dg_sync / 1048576.0, p->dg_rev / 1024.0);
            p->dg_sync = p->dg_rev = 0; }
        pool_unlock();
        void vk_pool_kind_stats(uint64_t out[3][5]); extern uint64_t g_pd_rev_total; uint64_t pk[3][5]; vk_pool_kind_stats(pk);
        VKLOG("NVN2 pool kinds so far: indirect args (no pool/zero-copy/s48/copied/bridge) %llu/%llu/%llu/%llu/%llu, render conditions %llu/%llu/%llu/%llu/%llu; GPU-written bytes reverted by syncs so far %.1f KB; copied pools synced %.0f MB in 30 s",
              (unsigned long long)pk[0][0], (unsigned long long)pk[0][1], (unsigned long long)pk[0][2], (unsigned long long)pk[0][3], (unsigned long long)pk[0][4],
              (unsigned long long)pk[1][0], (unsigned long long)pk[1][1], (unsigned long long)pk[1][2], (unsigned long long)pk[1][3], (unsigned long long)pk[1][4], g_pd_rev_total / 1024.0, tot / 1048576.0);
        (void)pl; }
    {   // bak86: first-invocation stores the host made; images kept as the only copy of GPU-written pools' CPU content, storages zero-filled again
        extern uint64_t g_fs_pre, g_fs_miss; void vk_tex_unc_stats(uint64_t out[2]); uint64_t us[2]; vk_tex_unc_stats(us); static uint64_t fs0, fm0, us0[2];
        VKLOG("NVN2 (last 30 s): first_store host pre-stores %llu (no buffer %llu); evictions: kept (GPU-only content) %llu, zero-filled again %llu",
              (unsigned long long)(g_fs_pre - fs0), (unsigned long long)(g_fs_miss - fm0), (unsigned long long)(us[0] - us0[0]), (unsigned long long)(us[1] - us0[1]));
        fs0 = g_fs_pre; fm0 = g_fs_miss; memcpy(us0, us, sizeof us); }
    VKLOG("NVN2 sparse tiles: %llu images, %llu tiles bound, %.0f MiB bound (tails included); uploads %.1f MB in the last 30 s (%.0f MB in all)", (unsigned long long)ts[0],
        (unsigned long long)ts[1], ts[2] / 1048576.0, (ts[3] - ts0[3]) / 1048576.0, ts[3] / 1048576.0);
    memcpy(cs0, cs, sizeof cs); rp0 = rp; memcpy(ts0, ts, sizeof ts);
    int s48live = 0; for (int k = 0; k < g_s48_n; k++) if (g_s48[k].mem) s48live++;
    VKLOG("NVN2 memory (host objects): scratch blocks %llu, %.0f MB (one submission's use: high-water %.1f MB); main command buffers %llu (most in a slot %llu), upload %llu (most %llu); "
          "descriptor pools %llu; image views made %llu, freed %llu; c1 buffers %llu (blocks %.0f MB); pipelines: Eden graphics %llu, compute %llu, host-made %llu; "
          "s48 ranges %d, %.0f MB imported (freed %llu, %.0f MB; imported again %llu); pressure trims %llu; s24 storage not readable: checks %llu, uploads %llu skipped",
        (unsigned long long)ss[0], ss[1] / 1048576.0, ss[2] / 1048576.0, (unsigned long long)ss[3], (unsigned long long)ss[4], (unsigned long long)ss[5], (unsigned long long)ss[6],
        (unsigned long long)ss[7], (unsigned long long)g_views_made, (unsigned long long)g_views_freed, (unsigned long long)g_c1_bufs, g_c1_arena_bytes / 1048576.0,
        (unsigned long long)ep[0], (unsigned long long)ep[1], (unsigned long long)pipes_built_now(),
        s48live, g_s48_bytes / 1048576.0, (unsigned long long)g_s48_freed_n, g_s48_freed / 1048576.0, (unsigned long long)g_s48_reimp, (unsigned long long)g_press_n,
        (unsigned long long)g_s24_skips, (unsigned long long)g_s24_upskips);
}
// NVN2 (Switch 2 boots): the driver's heaps are kept under their budget before a refusal happens. HWAoI 11-16 06:44: heap0 at 6982 of its 6972 MB budget, a submission
// refused at its end (vkEndCommandBuffer -1), and 1.5 s later an access violation in the driver (also after the 11-15 refusal); the 33-minute run before it held at
// 93-97% with no refusal. At most every 250 ms: a heap at DM_VK_S2_PRESSURE_HI percent of its budget or more (default 85) starts trimming, which goes on at each check
// until every heap is at DM_VK_S2_PRESSURE_LO or less (default 80). Each check takes, in order, until the excess over LO is covered:
// (a) imported pool chunks unused for 60 submissions, then every s48 range no buffer holds,
// (b) up to 512 MiB of completed idle image allocations with recoverable contents, (c) scratch down to its 32 MiB floor
// in the next recycles of every slot (scratch_trim). c1 blocks hold live constants and uploads use scratch: nothing else is let go. The driver's usage reflects the
// deferred frees a submission or more later; a check that leaves usage over LO is followed by the next. DM_VK_S2_PRESSURE=0 off
int g_nvn2_heap_pct;   // bak91: the worst heap's use in % of its budget at the last pressure check (0 until measured)
static void nvn2_pressure(void) {
    const nvn2_pressure_limits *limits = nvn2_limits();
    const int on = limits->on, hi = limits->hi, lo = limits->lo;
    static uint64_t last; static int trimming; uint64_t now = s47_ns();
    if (!on || now - last < 250000000ull) return;
    last = now;
    uint64_t use, bud; if (!vk_heap_worst(&use, &bud) || !bud) return;
    g_nvn2_heap_pct = (int)(use * 100 / bud);
    { static uint64_t heap_t; if (use * 100 >= bud * 85ull && now - heap_t >= 5000000000ull) { heap_t = now;   // NVN2 (11-16): the climb, every 5 s at 85% or more
        VKLOG("NVN2 heap: %.0f of %.0f MB (%.0f%%; trims from %d%% to %d%%, %s)", use / 1048576.0, bud / 1048576.0, 100.0 * (double)use / (double)bud, hi, lo, trimming ? "trimming" : "not trimming"); } }
    if (use * 100 <= bud * (uint64_t)lo) { trimming = 0; return; }
    if (!trimming && use * 100 < bud * (uint64_t)hi) return;
    trimming = 1;
    const uint64_t want = use - bud * (uint64_t)lo / 100;
    pool_wrlock();
    const uint64_t w0 = g_rel_owned;
    g_rel_zc_only = 1; uint64_t a = pool_release(want, 60, 0); g_rel_zc_only = 0;
    const uint64_t s48n0 = g_s48_freed_n;
    const uint64_t own = g_rel_owned - w0, s48f = s48_free_idle(S48_MAXR);   // NVN2 (11-16): every range no buffer holds (was 2 a check)
    const uint64_t s48n = g_s48_freed_n - s48n0;
    { static uint64_t dump_t; if (!dump_t || now - dump_t >= 30000000000ull) { dump_t = now; s48_dump(); } }
    pool_unlock();
    uint64_t freed = s48f + own, b = 0;   // (a chunk bound into a range frees nothing itself: its range when no buffer holds it; a chunk imported alone, its own memory)
    // Protect the recent working set; the shorter critical window still requires
    // completed GPU work and recoverable contents. The return value is physical memory freed.
    const uint64_t img_idle = use * 100 >= bud * 95ull ? 2 : 8;   // completed images only; GPU contents now have lossless backing
    if (freed < want) { uint64_t w = want - freed; b = vk_tex_evict_mem(w < (512ull << 20) ? w : (512ull << 20), img_idle); }
    if (freed + b < want) g_scr_squeeze = 2 * nslot_on();   // (two recycles of each slot in use; slot12: was 12 for 6 slots)
    g_press_n++;
    // NVN2 diagnostic (11-16): what the images take, at the first trim and every 30 s while trimming goes on (Eden's own allocations are not reachable from here: its VMA
    // stats file, vma_stats.json, is written by Eden, not counted in this line)
    { static uint64_t attr_t; if (!attr_t || now - attr_t >= 30000000000ull) { attr_t = now; vk_tex_mem_attrib(); } }
    static int nl; static uint64_t lastlog;
    if (nl < 16 || now - lastlog > 30000000000ull) { nl++; lastlog = now;
        VKLOG("[RED] NVN2 memory pressure: heap at %.0f of %.0f MB (%.0f%%, trims from %d%% to %d%%): imported chunks %.1f MB released, %.1f MB of memory freed (s48 ranges %.1f MB in %llu ranges, "
            "chunks alone %.1f MB), idle images %.1f MB reclaimed (idle %llu presents or more)%s (trim %llu)",
            use / 1048576.0, bud / 1048576.0, 100.0 * (double)use / (double)bud, hi, lo, a / 1048576.0, (s48f + own) / 1048576.0, s48f / 1048576.0, (unsigned long long)s48n, own / 1048576.0,
            b / 1048576.0, (unsigned long long)img_idle, g_scr_squeeze ? ", scratch to its floor" : "", (unsigned long long)g_press_n); }
}
// NVN2 (11-16, MKW: vkBeginCommandBuffer and the scratch block's vkAllocateMemory refused with -2 between two pressure checks, and the host ended at its VKCHK): a host
// allocation or command buffer begin refused for memory (-1 / -2) on a Switch 2 boot first trims at once, in the critical mode: recoverable images by the 2-present
// age (up to 512 MB), every s48 range no buffer holds, imported chunks unused for 60 submissions, scratch to its floor; then the call is made once more. The pool parts
// run only when this thread holds no pool lock (g_pl_depth 0) or holds it for writing: under a read lock pool_wrlock would return without write access
static uint64_t nvn2_emergency(const char *site, VkResult rc) {
    if (!dm_sdk_ounce()) return 0;
    const uint64_t img = vk_tex_evict_oom(512ull << 20);
    uint64_t a = 0, own = 0, s48f = 0, s48n = 0; int pooled = 0;
    if (g_pl_depth == 0 || g_pl_mode == 2) {
        pool_wrlock();
        const uint64_t w0 = g_rel_owned, n0 = g_s48_freed_n;
        g_rel_zc_only = 1; a = pool_release(1ull << 30, 60, 0); g_rel_zc_only = 0;
        own = g_rel_owned - w0; s48f = s48_free_idle(S48_MAXR); s48n = g_s48_freed_n - n0;
        pool_unlock(); pooled = 1;
    }
    g_scr_squeeze = 2 * nslot_on();   // slot12: two recycles of each slot in use
    VKLOG("[RED] NVN2 memory: %s refused (%d): critical trim at once: image memory %.1f MB reclaimed (2 presents), s48 ranges %.1f MB in %llu ranges, imported chunks %.1f MB released "
          "(%.1f MB of their own memory)%s, scratch to its floor; trying once more", site, (int)rc, img / 1048576.0, s48f / 1048576.0, (unsigned long long)s48n, a / 1048576.0,
          own / 1048576.0, pooled ? "" : " (pool parts skipped: a pool read lock held here)");
    return img + s48f + own;
}
static int nvn2_mem_err(VkResult r) { return (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) && dm_sdk_ounce(); }
// vkBeginCommandBuffer with one retry. A begin that failed leaves the command buffer in no defined recording state, so it is reset first: the slots' pools are made with
// VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT (slots_init), which makes vkResetCommandBuffer valid, and the buffer is not pending (it was being begun)
static VkResult nvn2_begin(VkCommandBuffer cb, const VkCommandBufferBeginInfo *bi, const char *site) {
    VkResult r = vkBeginCommandBuffer(cb, bi);
    if (!nvn2_mem_err(r)) return r;
    nvn2_emergency(site, r);
    vkResetCommandBuffer(cb, 0);
    r = vkBeginCommandBuffer(cb, bi);
    VKLOG("%sNVN2 memory: %s after the critical trim: %d", r ? "[RED] " : "", site, (int)r);
    return r;
}
static VkResult nvn2_alloc(const VkMemoryAllocateInfo *ai, VkDeviceMemory *m, const char *site) {
    VkResult r = vkAllocateMemory(V.dev, ai, NULL, m);
    if (!nvn2_mem_err(r)) return r;
    nvn2_emergency(site, r);
    r = vkAllocateMemory(V.dev, ai, NULL, m);
    VKLOG("%sNVN2 memory: %s after the critical trim: %d", r ? "[RED] " : "", site, (int)r);
    return r;
}
static void pool_budget(void) {
    static uint64_t cap = 1; if (cap == 1) { const char *e = getenv("DM_VK_POOL_BUDGET_MB"); cap = (uint64_t)(e && *e ? atoll(e) : 2048) << 20; }
    if (dm_sdk_ounce()) { nvn2_mem_report(); nvn2_pressure(); }
    // NVN2 (Switch 2 boots only): imported bytes alone count, since copies are not released here (below) and would otherwise keep every idle imported chunk released
    const int zo = dm_sdk_ounce();
    if (!cap || (g_nsubs & 31) || (zo ? g_zc_bytes : g_zc_bytes + g_cp_bytes) <= cap) return;   // (read without the lock: a hint)
    pool_wrlock();
    // NVN2 (Switch 2 boots only): copied chunks are not released here. pools_copy_back writes back the GPU's writes only in the ranges a submission records
    // (s->gw); render targets and other GPU-written texture storage in a copied chunk exist only in the copy, and a released copy is made again from the CPU's
    // pages (HWAoI 11-15 run 2: a 2 GB copy released and re-copied 29 times, each a 0.6-1.8 s stall). Imported chunks are the game's own pages: released as before
    g_rel_zc_only = zo;
    uint64_t t = zo ? g_zc_bytes : g_zc_bytes + g_cp_bytes, got = t > cap ? pool_release(t - cap + (cap >> 3), 300, 0) : 0;
    g_rel_zc_only = 0;
    pool_unlock();
    if (got) { static int n; if (n++ < 32) VKLOG("r78: pool chunks over the budget (%.0f MB made, budget %.0f MB): %.1f MB unused for 300 submissions released (%llu chunks so far, made again at their next use)",
        t / 1048576.0, cap / 1048576.0, got / 1048576.0, (unsigned long long)g_rel_n); }
}
// s28: a copied pool of at most DM_VK_S28_UNTRACKED_MB (default 4; 0 = all tracked, as before) is never write-protected: copied whole at each sync instead.
// Protecting a copied pool's pages to copy only what the CPU wrote cost Xenoblade 3 run 32 (11-12) a storm of write faults from its first copied bridge (2.4k
// a second and rising), then a freeze 25 s later with a guest thread faulting on one store over and over. A bridge is about 2 MB: a copy is well under a ms
static int pool_tracked(const vkpool_t *p) {
#if defined(DM_NXN1X)
    static int64_t lim = -1; if (lim < 0) { const char *e = getenv("DM_VK_S28_UNTRACKED_MB"); int64_t mb = e && *e ? atoll(e) : 4; lim = mb > 0 ? mb << 20 : 0; }
    if (p->hi - p->lo <= (uint64_t)lim) { static int said; if (!said) { said = 1; VKLOG("s28: copied pools of up to %lld MB are not write-protected, %s at each sync (first: %p +%#llx; DM_VK_S28_UNTRACKED_MB=0 off)",
        (long long)(lim >> 20), (pool_sync_mode() & 1) ? "compared exactly by block" : "copied whole", (void *)(uintptr_t)p->lo, (unsigned long long)(p->hi - p->lo)); } return 0; }
#endif
    (void)p; return 1;
}
static void s47_big(const char *what, uint64_t a, uint64_t l, uint64_t ns);
// Publish no buffer until allocation, binding and mapping all succeed. In
// particular, completion readers must not see a provisional buffer during recovery.
static VkResult pool_shadow_allocate(uint64_t size, VkBuffer *buffer,
                                     VkDeviceMemory *memory, void **mapping,
                                     uint32_t *memory_type, VkDeviceSize *allocation_size) {
    VkBuffer b=VK_NULL_HANDLE; VkDeviceMemory m=VK_NULL_HANDLE; void *map=NULL;
    VkBufferCreateInfo bi={VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,NULL,0,size,POOL_USAGE,VK_SHARING_MODE_EXCLUSIVE};
    VkResult r=vkCreateBuffer(V.dev,&bi,NULL,&b);
    if (r!=VK_SUCCESS) return r;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev,b,&req);
    *allocation_size=req.size;
    // Preserve the existing cached/coherent memory-type policy.
    uint32_t type=getenv("DM_VK_SHADOW_UNCACHED")?UINT32_MAX:vk_mem_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_CACHED_BIT,0);
    if (type==UINT32_MAX) type=vk_mem_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,0);
    if (type>=V.mem.memoryTypeCount || !(req.memoryTypeBits & (1u<<type))) {
        vkDestroyBuffer(V.dev,b,NULL); return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    VkMemoryAllocateInfo ai={VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,V.has_bda?&g_pool_af:NULL,req.size,type};
    r=vkAllocateMemory(V.dev,&ai,NULL,&m);
    const int allocated=r==VK_SUCCESS;
    if (r==VK_SUCCESS) r=vkBindBufferMemory(V.dev,b,m,0);
    if (r==VK_SUCCESS) r=vkMapMemory(V.dev,m,0,VK_WHOLE_SIZE,0,&map);
    if (r!=VK_SUCCESS) {
        vkDestroyBuffer(V.dev,b,NULL);
        if (allocated) vkFreeMemory(V.dev,m,NULL);
        return r;
    }
    *buffer=b; *memory=m; *mapping=map; *memory_type=type; return VK_SUCCESS;
}
#if defined(DM_NXN1X)
static void reclaim_completed_slots(void);
#endif
static uint64_t pool_allocation_recover(uint64_t want) {
    // Every caller owns a write hold, possibly nested. Keep that exact depth,
    // registry indices (including a not-yet-published bridge), and replay cursor.
    if (!g_pl_depth || g_pl_mode!=2 || !nvnr_suspend(&g_pool_writers))
        die("[vk] pool allocation recovery requires exclusive registry ownership");
    const int depth=g_pl_depth;
    nvnr_submission saved; vk_sub_suspend(&saved);
    g_pl_depth=1; pool_unlock();
#if defined(DM_NXN1X)
    reclaim_completed_slots();
#endif
    const uint64_t freed=vk_tex_recover_pool(want);
    pool_wrlock(); g_pl_depth=depth;
    vk_sub_resume(&saved);
    if (!nvnr_resume(&g_pool_writers)) die("[vk] pool allocation recovery lost registry ownership");
    return freed;
}
static uint32_t pool_shadow_required(vkpool_t *p, uint64_t size) {
    uint32_t ty=UINT32_MAX; VkDeviceSize allocation_size=size;
    void *mapping=NULL;
    p->buf=VK_NULL_HANDLE; p->mem=VK_NULL_HANDLE; p->shadow=NULL;
    VkResult result=pool_shadow_allocate(size,&p->buf,&p->mem,&mapping,&ty,&allocation_size);
    if (nvn2_mem_err(result)) {
        const uint64_t freed=pool_allocation_recover(allocation_size);
        result=pool_shadow_allocate(size,&p->buf,&p->mem,&mapping,&ty,&allocation_size);
        VKLOG("NVN2 F4E pool allocation retry: bytes=%llu reclaimed=%llu result=%d; same command, registry retained",
            (unsigned long long)allocation_size,(unsigned long long)freed,result);
    }
    if (result!=VK_SUCCESS)
        die("[vk] required pool buffer unavailable: address=%p bytes=%llu result=%d; command not continued",(void *)(uintptr_t)p->lo,(unsigned long long)size,result);
    p->shadow=mapping;
    VkMemoryPropertyFlags tf=V.mem.memoryTypes[ty].propertyFlags;
    p->noncoh=!(tf&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    return ty;
}
// ---- poolcopy (design\GPU_MEMORY_CACHE_DESIGN.md 4.6, NXN1X_POOL_MODE=copy; default import) ----
// Cause (11-20): on N1X the GPU's work on imported game pages (VK_EXT_external_memory_host) is slow on several paths (fixed-function vertex/index fetch 20-250x,
// atomics 25x, repeated reads 36x, storage writes 1.8x; membench2/4/5); driver-allocated memory is not. In copy mode every chunk the GPU uses is a device-local
// copy: p->buf / p->gpu are it, so every binding, address table and copy reaches it with no other change. Beside it, two CPU-mapped buffers:
//   shadow: the baseline, equal to the device-local copy at each submission's start (copied pools' existing sync fills it from the game's pages; each run it
//           copies is uploaded to the device-local copy by a GPU copy in the submission's upload command buffer, pc_upload)
//   rb:     the readback landing: at the submission's end the ranges its GPU work may have written (sub_gpu_wrote's list; a whole chunk where the size is not
//           known: raw global memory, storage buffers named in uniform memory) are copied there, then made visible to the host (pc_readback)
// At completion (pools_copy_back) each byte the GPU changed (rb != shadow) goes into the game's page unless the CPU changed that byte since the sync (game !=
// shadow: the CPU's stays), and the shadow takes rb's byte (it is the copy's content). So the CPU's bytes are never put back to older ones (review P1-1)
// and no snapshot is needed. The previous submission's completion is waited for before each sync (as copied pools did), so the shadow is the copy's content then.
// Chunks inside NXN1X's backing memory are bound into one copy per aligned backing range (pcr): every pool or view over the same pages reaches the same
// device-local memory, so aliases see each other's GPU writes in the same submission (review P1-4). A submission refused at its end: every copy is synced
// again in full from the game's pages (its uploads and writes did not all run). Copy mode is chosen before the first pool and kept for the run
static int g_pc_mode = -1, g_pc_test_fail;   // g_pc_test_fail: the self-test's next submission is taken as refused at its end (case 9)
// 0 import, 1 copy (every chunk a device-local copy), 2 hybrid (hybrid1: chunks imported; a unit that an atomic reaches becomes a device-local copy for the run).
// Nonzero: the copy machinery (stage2c sync, pre/post, merges) runs, for the chunks that are copies (p->dl)
int vk_poolcopy_on(void) {
    if (g_pc_mode < 0) { const char *e = getenv("NXN1X_POOL_MODE");
        g_pc_mode = !e ? 0 : !strcmp(e, "copy") || (!strcmp(e, "auto") && vk_is_n1x()) ? 1 : !strcmp(e, "hybrid") ? 2 : 0;
        VKLOG("poolcopy: memory pools %s (NXN1X_POOL_MODE=%s; import / copy / hybrid / auto)", g_pc_mode == 1 ? "are device-local copies of the game's pages: uploaded before each submission, GPU writes read back and merged at its completion" :
              g_pc_mode == 2 ? "are imported; a backing range an atomic reaches (a written storage buffer the shader runs an atomic on) becomes a device-local copy from the next submission on, for the rest of the run (hybrid1)" :
              "are imported where the driver allows it (as before)", e && *e ? e : "import (default)"); }
    return g_pc_mode;
}
static int pc_hybrid(void) { return vk_poolcopy_on() == 2; }
// stage2c (11-20, design/GPU_MEMORY_CACHE_STAGE2_DESIGN.md 8.2): how copied pools stay in step with the GPU. NXN1X_PC_SYNC (A/B and the self-test only):
//   async  (default): no wait for the previous submission. Each submission owns its staging (the CPU patches it uploads, immutable once recorded) and, for every range
//          its GPU work may write, a "pre" copy of the device-local bytes taken just before that work (in the upload buffer of the part that first writes it) and a
//          "post" copy at its end. At completion, in submission order, a byte with post != pre was changed by this submission's GPU work: it goes into the game's page
//          when the page still holds pre there (else the CPU wrote it since: the CPU's stays), and into the shadow either way. The shadow is the game's memory less
//          the CPU changes not yet captured, so a GPU result merged earlier never looks like a CPU write (review P0-1/P0-2)
//   serial: stage 1 (each sync waits for the previous submission's completion; one readback per chunk, merged against the shadow)
//   naive : stage 1's merge without its wait. Unsafe: kept only so the self-test can show that its cases catch the counterexamples
enum { PC_SERIAL = 0, PC_NAIVE = 1, PC_ASYNC = 2 };
static int g_pc_sync = -1;
static int pc_sync(void) {
    if (g_pc_sync < 0) { const char *e = getenv("NXN1X_PC_SYNC"); g_pc_sync = e && !strcmp(e, "serial") ? PC_SERIAL : e && !strcmp(e, "naive") ? PC_NAIVE : PC_ASYNC;
        VKLOG("%spoolcopy: synchronization %s (NXN1X_PC_SYNC=%s)", g_pc_sync == PC_NAIVE ? "[RED] " : "", g_pc_sync == PC_ASYNC ? "per submission: own staging and pre/post copies of GPU-written ranges, no wait for the previous submission" :
              g_pc_sync == PC_SERIAL ? "serial: each sync waits for the previous submission (stage 1)" : "naive: stage 1's merge with no wait (unsafe, for the self-test)", e && *e ? e : "async (default)"); }
    return g_pc_sync;
}
static int pc_async(void) { return vk_poolcopy_on() && pc_sync() == PC_ASYNC; }
struct vksub; static struct vksub *cur(void); static uint64_t g_fence, g_done;   // stage2c (defined below)
static void pv_free(vkpool_t *p); static void pv_reset(vkpool_t *p);   // stage2b (defined below)
static uint64_t g_pc_drain;   // stage2c: a fence the next sync must wait for (all merges up to it): after a refused submission or before a copy's first full sync
static uint64_t g_pc_pre_bytes, g_pc_pre_n, g_pc_post_bytes, g_pc_as_mg, g_pc_as_cpu, g_pc_as_rec, g_pc_as_drop, g_pc_drains;   // stage2c counters
static _Atomic int g_pc_hold;   // stage2c self-test: the completion thread holds its merges while set (so several submissions are in flight at once)
typedef struct { uint64_t lo, hi; VkDeviceMemory dm, sm, rm; uint8_t *smap, *rmap; uint32_t dt, st, rt; int nb, noncoh_s, noncoh_r; } pcr_t;
enum { PCR_MAX = 512 };
static pcr_t g_pcr[PCR_MAX]; static int g_pcr_n;
static uint64_t g_pc_shared, g_pc_private, g_pc_up_bytes, g_pc_up_n, g_pc_rb_bytes, g_pc_rb_whole, g_pc_mg_bytes, g_pc_mg_cpu, g_pc_resync, g_pc_noalias, g_pc_lost;
static uint64_t g_pc_pages_seen, g_pc_pages_dirty, g_pc_runs;   // stage2a: 4 KB pages looked at by the sync, dirty ones, runs copied
static const VkBufferUsageFlags PC_HOST_USAGE = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
static uint32_t pc_type(uint32_t bits, int host) {
    if (!host) return vk_mem_type(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    uint32_t t = vk_mem_type(bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    return t != UINT32_MAX ? t : vk_mem_type(bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
}
static VkResult pc_alloc(uint64_t size, uint32_t type, int bda, VkDeviceMemory *m, uint8_t **map) {
    *m = VK_NULL_HANDLE; if (type == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, bda && V.has_bda ? &g_pool_af : NULL, size, type};
    VkResult r = vkAllocateMemory(V.dev, &ai, NULL, m); if (r) { *m = VK_NULL_HANDLE; return r; }
    if (map) { void *p = NULL; r = vkMapMemory(V.dev, *m, 0, VK_WHOLE_SIZE, 0, &p); if (r) { vkFreeMemory(V.dev, *m, NULL); *m = VK_NULL_HANDLE; return r; } *map = p; }
    return VK_SUCCESS;
}
static int pc_noncoh(uint32_t t) { return t < V.mem.memoryTypeCount && !(V.mem.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); }
// the shared range holding backing [a, a + n): made here when none overlaps it (aligned to 256 MB); -1 = none (the chunk gets memory of its own)
static int pc_range(uint64_t a, uint64_t n, uint32_t dbits, uint32_t hbits) {
    const uint64_t B = 256ull << 20, e = a + n;
    for (int k = 0; k < g_pcr_n; k++) if (g_pcr[k].dm && a >= g_pcr[k].lo && e <= g_pcr[k].hi) return (dbits >> g_pcr[k].dt & 1) && (hbits >> g_pcr[k].st & 1) && (hbits >> g_pcr[k].rt & 1) ? k : -1;
    const uint64_t lo = a & ~(B - 1), hi = (e + B - 1) & ~(B - 1);
    for (int k = 0; k < g_pcr_n; k++) if (g_pcr[k].dm && lo < g_pcr[k].hi && g_pcr[k].lo < hi) return -1;   // it would overlap a range made before
    if (g_pcr_n == PCR_MAX) return -1;
    pcr_t r; memset(&r, 0, sizeof r); r.lo = lo; r.hi = hi; r.dt = pc_type(dbits, 0); r.st = pc_type(hbits, 1); r.rt = r.st;
    if (pc_alloc(hi - lo, r.dt, 1, &r.dm, NULL)) return -1;
    if (pc_alloc(hi - lo, r.st, 0, &r.sm, &r.smap)) { vkFreeMemory(V.dev, r.dm, NULL); return -1; }
    if (pc_alloc(hi - lo, r.rt, 0, &r.rm, &r.rmap)) { vkFreeMemory(V.dev, r.dm, NULL); vkFreeMemory(V.dev, r.sm, NULL); return -1; }
    r.noncoh_s = pc_noncoh(r.st); r.noncoh_r = pc_noncoh(r.rt);
    g_pcr[g_pcr_n] = r;
    { static int nl; if (nl++ < 16) VKLOG("poolcopy: backing range %#llx +%llu MB made: device-local type %u, shadow and readback type %u (%s)", (unsigned long long)lo, (unsigned long long)((hi - lo) >> 20),
        r.dt, r.st, r.noncoh_s ? "not coherent: flushed and invalidated by range" : "coherent"); }
    return g_pcr_n++;
}
static void pc_free_buffers(vkpool_t *p, int now) {   // the chunk's copy buffers and owned memories (pool_release, mtl_pool_del)
    VkBuffer b[2] = {p->sbuf, p->rbuf}; VkDeviceMemory m[2] = {p->smem, p->rmem};
    for (int i = 0; i < 2; i++) { if (now) { if (b[i]) vkDestroyBuffer(V.dev, b[i], NULL); if (m[i]) vkFreeMemory(V.dev, m[i], NULL); }
        else { if (b[i] || m[i]) vk_defer_free_buffer(b[i], m[i]); } }
    if (p->pcr > 0 && p->pcr <= g_pcr_n && g_pcr[p->pcr - 1].nb > 0) g_pcr[p->pcr - 1].nb--;   // (a range's memory is kept, as s48's)
    pv_free(p);   // stage2b
    p->sbuf = p->rbuf = VK_NULL_HANDLE; p->smem = p->rmem = p->smh = p->rmh = VK_NULL_HANDLE; p->rb = NULL; p->dl = 0; p->pcr = 0; p->rb_sub = 0; p->rb_whole = 0;
}
// the chunk [mem, mem + size) as a device-local copy with its shadow and readback (under the pool write lock); 0 = made
static int pc_make(vkpool_t *p, uint8_t *mem, uint64_t size) {
    VkBuffer d = VK_NULL_HANDLE, s = VK_NULL_HANDLE, r = VK_NULL_HANDLE;
    VkBufferCreateInfo bd = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE}, bh = bd; bh.usage = PC_HOST_USAGE;
    if (vkCreateBuffer(V.dev, &bd, NULL, &d) || vkCreateBuffer(V.dev, &bh, NULL, &s) || vkCreateBuffer(V.dev, &bh, NULL, &r)) goto fail;
    VkMemoryRequirements rd, rh; vkGetBufferMemoryRequirements(V.dev, d, &rd); vkGetBufferMemoryRequirements(V.dev, s, &rh);
    uint8_t *al = NULL;
#if defined(DM_NXN1X)
    al = (uint8_t *)nvnh_import_alias(mem, size);
#endif
    int k = al ? pc_range((uint64_t)(uintptr_t)al, size, rd.memoryTypeBits, rh.memoryTypeBits) : -1;
    if (k >= 0) {
        const VkDeviceSize off = (uint64_t)(uintptr_t)al - g_pcr[k].lo, span = g_pcr[k].hi - g_pcr[k].lo;
        if (off % (rd.alignment ? rd.alignment : 1) || off % (rh.alignment ? rh.alignment : 1) || off + rd.size > span || off + rh.size > span) k = -1;
        else if (vkBindBufferMemory(V.dev, d, g_pcr[k].dm, off) || vkBindBufferMemory(V.dev, s, g_pcr[k].sm, off) || vkBindBufferMemory(V.dev, r, g_pcr[k].rm, off)) goto fail;
        else { p->mem = VK_NULL_HANDLE; p->smem = p->rmem = VK_NULL_HANDLE; p->smh = g_pcr[k].sm; p->rmh = g_pcr[k].rm; p->soff = p->roff = off;
            p->shadow = g_pcr[k].smap + off; p->rb = g_pcr[k].rmap + off; p->noncoh = g_pcr[k].noncoh_s; p->rb_noncoh = g_pcr[k].noncoh_r; p->pcr = k + 1; g_pcr[k].nb++; g_pc_shared += size; }
    }
    if (k < 0) {   // memory of its own: not shared with another pool over the same pages (counted; aliases then rely on the sync as copied pools did)
        if (al) g_pc_noalias++;
        VkDeviceMemory dm, sm, rm; uint8_t *smap = NULL, *rmap = NULL; uint32_t dt = pc_type(rd.memoryTypeBits, 0), ht = pc_type(rh.memoryTypeBits, 1);
        if (pc_alloc(rd.size, dt, 1, &dm, NULL)) goto fail;
        if (pc_alloc(rh.size, ht, 0, &sm, &smap)) { vkFreeMemory(V.dev, dm, NULL); goto fail; }
        if (pc_alloc(rh.size, ht, 0, &rm, &rmap)) { vkFreeMemory(V.dev, dm, NULL); vkFreeMemory(V.dev, sm, NULL); goto fail; }
        if (vkBindBufferMemory(V.dev, d, dm, 0) || vkBindBufferMemory(V.dev, s, sm, 0) || vkBindBufferMemory(V.dev, r, rm, 0)) { vkFreeMemory(V.dev, dm, NULL); vkFreeMemory(V.dev, sm, NULL); vkFreeMemory(V.dev, rm, NULL); goto fail; }
        p->mem = dm; p->smem = p->smh = sm; p->rmem = p->rmh = rm; p->soff = p->roff = 0; p->shadow = smap; p->rb = rmap; p->noncoh = pc_noncoh(ht); p->rb_noncoh = p->noncoh; p->pcr = 0; g_pc_private += size;
    }
    p->buf = d; p->sbuf = s; p->rbuf = r; p->dl = 1; p->zero_copy = 0; p->rb_sub = 0; p->rb_whole = 0;
    { static int nl; if (nl++ < 8) VKLOG("poolcopy: chunk %p +%#llx is a device-local copy (%s; shared %.1f MB, own %.1f MB so far)", (void *)mem, (unsigned long long)size,
        p->pcr ? "in its backing range, shared with every pool over the same pages" : al ? "memory of its own: its backing range overlaps another" : "memory of its own: not in NXN1X's backing",
        g_pc_shared / 1048576.0, g_pc_private / 1048576.0); }
    return 0;
fail:
    if (d) vkDestroyBuffer(V.dev, d, NULL); if (s) vkDestroyBuffer(V.dev, s, NULL); if (r) vkDestroyBuffer(V.dev, r, NULL);
    return -1;
}
VkCommandBuffer vk_upload_cmd(void); extern uint64_t g_up_gen;
static uint64_t g_pc_upgen = ~0ull;
// [off, off + n) of the shadow, just written from the game's pages, into the device-local copy: a copy in the open submission's upload command buffer (it runs
// before the part it uploads for; the closing full barrier, vk_img_barrier_all at the upload buffer's end, makes it visible to every later access)
static int pc_stage_upload(vkpool_t *p, uint64_t off, uint64_t n);   // stage2c (async): through the submission's own staging (below, with the slots)
static void pc_upload(vkpool_t *p, uint64_t off, uint64_t n) {
    if (!p->dl || !n) return;
    if (!vk_sub_open()) { p->synced = 0; g_pc_lost++; return; }   // (no submission to carry it: synced again in full at the next one)
    if (pc_async()) { if (pc_stage_upload(p, off, n)) { g_pc_up_bytes += n; g_pc_up_n++; } else { p->synced = 0; g_pc_lost++; } return; }
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer (leaf; this runs under the pool lock)
    VkCommandBuffer cb = vk_upload_cmd();
    if (g_pc_upgen != g_up_gen) { g_pc_upgen = g_up_gen;   // once per upload buffer: earlier GPU work on the copies, and the host's shadow writes, before these copies
        VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
    VkBufferCopy r = {off, off, n}; vkCmdCopyBuffer(cb, p->sbuf, p->buf, 1, &r);
    g_pc_up_bytes += n; g_pc_up_n++;
    vk_up_unlock();
}
static VkResult rb_range(vkpool_t *p, uint64_t off, uint64_t n) {   // a non-coherent readback range made visible to the CPU (after the GPU's completion)
    if (!p->rb_noncoh || !n) return VK_SUCCESS;
    VkDeviceSize a = V.props.limits.nonCoherentAtomSize ? V.props.limits.nonCoherentAtomSize : 1, lo = (p->roff + off) / a * a, hi = (p->roff + off + n + a - 1) / a * a;
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, p->rmh, lo, hi - lo};
    return vkInvalidateMappedMemoryRanges(V.dev, 1, &r);
}
// [lo, hi) of chunk p (game addresses), read back by its submission: the GPU's bytes into the game's page where the CPU left the byte as synced (completion thread)
void dm_guest_write_prepare(void *addr, size_t len);
static void pc_merge(vkpool_t *p, uint64_t lo, uint64_t hi) {
    if (lo < p->lo) lo = p->lo; if (hi > p->hi) hi = p->hi; if (lo >= hi) return;
    rb_range(p, lo - p->lo, hi - lo); shadow_range(p, lo - p->lo, hi - lo, 1);
    for (uint64_t a = lo; a < hi;) { uint64_t e = (a & ~4095ull) + 4096; if (e > hi) e = hi;
        const uint8_t *R = p->rb + (a - p->lo); uint8_t *B = p->shadow + (a - p->lo); volatile uint8_t *G = (volatile uint8_t *)(uintptr_t)a; const size_t n = (size_t)(e - a);
        g_pc_rb_bytes += n;
        if (memcmp(R, B, n)) { int prepared = 0;
#if defined(DM_NXN1X)
            nvnc_cpu(a, n, NVNC_GPU_WRITEBACK);
#endif
            for (size_t x = 0; x < n; x++) { if (R[x] == B[x]) continue;
                if (G[x] == B[x]) { if (!prepared) { dm_guest_write_prepare((void *)(uintptr_t)a, n); prepared = 1; } G[x] = R[x]; g_pc_mg_bytes++; } else g_pc_mg_cpu++;
                B[x] = R[x]; }
            shadow_range(p, a - p->lo, n, 0); }
        a = e; }
}
static void pc_dropped(void) {   // a submission refused at its end: its uploads and GPU writes did not all run: every copy synced in full again
    if (!vk_poolcopy_on()) return;
    int n = 0; pool_wrlock();
    for (int i = 0; i < g_npool; i++) if (g_pool[i].dl && g_pool[i].synced) { g_pool[i].synced = 0; n++; }
    pool_unlock(); g_pc_resync += (uint64_t)n;
    { static int nl; if (nl++ < 8) VKLOG("[RED] poolcopy: a submission was refused at its end: %d device-local copies synced again in full from the game's pages (its GPU writes are dropped with it)", n); }
}
// ---- hybrid1 (11-20, design\GPU_MEMORY_HYBRID_DESIGN.md 3-4, NXN1X_POOL_MODE=hybrid): import by default, the atomic exception path ----
// Cause: on N1X an atomic on imported game pages is 21-25x slower than on device-local memory (membench5, R7); OoT's dispatch 1bba09ec4bc8ff67 waits 122 ms on a
// float atomic add into an imported pool (ootgap). Copying every chunk costs the CPU write tracking everywhere (HWAoI moving: 52-57 fps against 60 with import + vbmir).
// So a chunk stays imported until a written storage buffer that its shader runs an atomic on (Eden StorageBufferDescriptor::is_atomic) is bound in it; from the next
// submission on, its unit is a device-local copy (pc_make: the copy mode's chunk) for the rest of the run, and the copy machinery (stage2c sync, pre/post, merge, volatile
// pages, fault-ahead) serves it as in copy mode. Plain stores do not convert (import store 1.5-1.8x); neither does raw global memory alone (3.1).
// Unit (hybrid2): the chunk's range of NXN1X's backing (aliases, every VA over the same pages, convert together) and its own guest range (chunks over the same
// VA, and chunks outside the backing). A chunk is a copy when either overlaps a converted one; converting one adds both its ranges, so the rule closes over
// every chunk sharing a page with it, and only those (hy_apply loops until nothing changes). The device-local memory itself is still shared per 256 MB (pc_range).
// When: the request is noted while the submission records (its import buffer is already in its descriptors and stays alive until the GPU passed it:
// vk_defer_free_buffer); hy_apply runs at the next submission's start, before the copied pools' scan. The new copy is not synced yet, so that scan waits for every
// earlier submission (their GPU writes went into the game's pages through the import) and then fills the copy from the game's pages whole (design 4.2-4.3)
// hybrid2 (11-21): the unit is the chunk's exact backing range, not the 256 MB blocks it lies in. OoT 00:25 (ab_hybrid_002501): 24 chunks an atomic reached pulled
// 452 chunks (493 MB) into copies through their blocks, chunks sharing no page with them; only pages shared (aliases) need to convert together
enum { HY_MAXR = 8192, HY_MAXQ = 256 };
typedef struct { uint64_t lo, hi; } hy_rng_t;
static hy_rng_t g_hy_al[HY_MAXR]; static int g_hy_nal;    // converted backing ranges (alias addresses)
static hy_rng_t g_hy_va[HY_MAXR]; static int g_hy_nva;    // converted guest ranges
static struct { uint64_t lo, hi, key; int why; } g_hy_q[HY_MAXQ]; static int g_hy_nq, g_hy_dirty;   // requests for the next submission (why: 0 storage atomic, 1 raw atomic log)
static pthread_mutex_t g_hy_mu = PTHREAD_MUTEX_INITIALIZER;                 // g_hy_q (the completion thread adds raw-log requests)
static uint64_t g_ssbo_ro_n, g_ssbo_ro_bytes, g_ssbo_cut_n, g_ssbo_cut_bytes;   // hybrid2 (vk_ssbo_bound): read-only bindings not taken as written; written ones cut to their reach
static uint64_t g_hy_req[2], g_hy_units, g_hy_chunks, g_hy_bytes, g_hy_refused, g_hy_qfull, g_hy_raw_unloc, g_hy_raw_full, g_hy_waits;
static uint64_t hy_alias(const vkpool_t *p) {
#if defined(DM_NXN1X)
    return (uint64_t)(uintptr_t)nvnh_import_alias((const void *)(uintptr_t)p->lo, p->hi - p->lo);
#else
    (void)p; return 0;
#endif
}
static int hy_is_copy(const vkpool_t *p) {   // the pool lock held (either): p shares a page (backing or guest address) with a converted chunk
    const uint64_t al = hy_alias(p), n = p->hi - p->lo;
    if (al) for (int i = 0; i < g_hy_nal; i++) if (al < g_hy_al[i].hi && g_hy_al[i].lo < al + n) return 1;
    for (int i = 0; i < g_hy_nva; i++) if (p->lo < g_hy_va[i].hi && g_hy_va[i].lo < p->hi) return 1;
    return 0;
}
static int hy_add_range(hy_rng_t *t, int *nt, uint64_t lo, uint64_t hi) {   // 1 = new (not inside one held)
    for (int i = 0; i < *nt; i++) if (t[i].lo <= lo && hi <= t[i].hi) return 0;
    if (*nt == HY_MAXR) die("[vk] hybrid: more than %d converted ranges", HY_MAXR);
    t[*nt].lo = lo; t[*nt].hi = hi; (*nt)++; return 1;
}
static int hy_add_unit(const vkpool_t *p) {   // its backing range and guest range converted; 1 = something new (the write lock held)
    const uint64_t al = hy_alias(p), n = p->hi - p->lo; int added = 0;
    if (al) added |= hy_add_range(g_hy_al, &g_hy_nal, al, al + n);
    added |= hy_add_range(g_hy_va, &g_hy_nva, p->lo, p->hi);
    if (added) g_hy_units++;
    return added;
}
static void hy_made_copy(vkpool_t *p, int zc) {   // pool_make_0 made p a copy in hybrid mode (its unit was converted, or its import was refused): its unit is copied
    if (zc != -8) { g_hy_refused++; static int n; if (n++ < 16) VKLOG("hybrid1: chunk %#llx +%#llx could not be imported (%d): a device-local copy, its unit with it", (unsigned long long)p->lo, (unsigned long long)(p->hi - p->lo), zc); }
    if (hy_add_unit(p)) g_hy_dirty = 1;   // (an import beside it, over a block it adds, converts at the next submission's start)
}
// A request: GPU work the open submission records reaches [lo, hi) with an atomic (key: the shader). why 0: a written storage buffer its shader runs an atomic on
// (gfx_vk_eden.inc eden_ssbo); 1: a page a raw global memory write log of an atomic shader lists (gml_copy_back, completion thread)
void vk_hy_request(uint64_t lo, uint64_t hi, uint64_t key, int why) {
    if (!pc_hybrid() || hi <= lo) return;
    pthread_mutex_lock(&g_hy_mu);
    g_hy_req[why & 1]++;
    int had = 0; for (int i = 0; i < g_hy_nq && !had; i++) had = g_hy_q[i].lo <= lo && hi <= g_hy_q[i].hi;
    if (!had) { if (g_hy_nq < HY_MAXQ) { g_hy_q[g_hy_nq].lo = lo; g_hy_q[g_hy_nq].hi = hi; g_hy_q[g_hy_nq].key = key; g_hy_q[g_hy_nq].why = why; g_hy_nq++; } else g_hy_qfull++; }
    pthread_mutex_unlock(&g_hy_mu);
}
// The rule (gfx_vk_eden.inc eden_ssbo, for every storage buffer bound from a pool chunk): written and atomic, in an imported chunk: its unit converts. A store alone does not
void vk_hy_ssbo(const vkpool_t *p, int written, int atomic, uint64_t key) { if (p && written && atomic && p->zero_copy) vk_hy_request(p->lo, p->hi, key, 0); }
static void hy_convert(vkpool_t *p, uint64_t key, int why) {   // an imported chunk made a device-local copy (the write lock held; its old buffer freed once the GPU passed it)
    const uint64_t n = p->hi - p->lo;
    vk_defer_free_buffer(p->buf, p->mem); if (p->mem2) vk_defer_free_buffer(VK_NULL_HANDLE, p->mem2);
    if (p->s48) { s48_unref(p->s48 - 1); p->s48 = 0; } if (p->s48b) { s48_unref(p->s48b - 1); p->s48b = 0; }
    if (p->zero_copy) g_zc_bytes -= n; else g_cp_bytes -= n;
    p->buf = VK_NULL_HANDLE; p->mem = VK_NULL_HANDLE; p->mem2 = VK_NULL_HANDLE; p->zero_copy = 0; p->gpu = 0;
    p->serial = ++g_pool_ser;   // views over the old buffer (vk_tbuf_view), vertex mirrors (vbm_get), pre/post entries key on it
    if (pc_make(p, (uint8_t *)(uintptr_t)p->lo, n)) die("[vk] hybrid1: device-local copy of memory pool chunk %#llx +%#llx could not be made; command not continued", (unsigned long long)p->lo, (unsigned long long)n);
    g_cp_bytes += n; pool_gpu_addr(p); p->synced = 0; p->pc_wfence = 0;   // synced whole at this submission's start, after every earlier submission
    if (!g_snaps && (g_snaps = calloc(SNAP_N, sizeof *g_snaps))) nvn_dirty_set_snap_hook(snap_hook);
    g_hy_chunks++; g_hy_bytes += n;
    static int nl; if (nl++ < 64) VKLOG("hybrid1: chunk %#llx +%#llx (backing %#llx) is a device-local copy from this submission on: %s, shader %016llx (%llu chunks, %.1f MB converted, %llu units)",
        (unsigned long long)p->lo, (unsigned long long)n, (unsigned long long)hy_alias(p), why == 0 ? "a storage buffer its shader runs an atomic on" : why == 1 ? "a page a raw global memory write log of an atomic shader lists" : "it shares pages with a converted chunk",
        (unsigned long long)key, (unsigned long long)g_hy_chunks, g_hy_bytes / 1048576.0, (unsigned long long)g_hy_units);
}
static void hy_apply(void) {   // at a submission's start (mtl_sub_begin, the replay thread), before the copied pools' scan
    if (!pc_hybrid()) return;
    pthread_mutex_lock(&g_hy_mu); int nq = g_hy_nq; __typeof__(g_hy_q) q; memcpy(q, g_hy_q, sizeof q); g_hy_nq = 0; pthread_mutex_unlock(&g_hy_mu);
    if (!nq && !g_hy_dirty) return;
    pool_wrlock(); int changed = 0; g_hy_dirty = 0;
    for (int r = 0; r < nq; r++) for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i];
        if (p->lo >= q[r].hi || q[r].lo >= p->hi) continue;
        if (hy_add_unit(p)) changed = 1;   // (a chunk with no buffer yet, r77: made a copy at its first use, pool_make_0)
        if (p->buf && p->zero_copy) { hy_convert(p, q[r].key, q[r].why); changed = 1; } }
    for (int more = 1; more;) { more = 0;   // closure: every chunk sharing a block or a range with a converted one
        for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i];
            if (!p->buf || !hy_is_copy(p)) continue;
            if (hy_add_unit(p)) more = 1;
            if (p->zero_copy) { hy_convert(p, 0, 2); more = changed = 1; } } }
    if (changed) __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);   // descriptor memos, the raw global memory pool table
    pool_unlock();
    if (changed) g_hy_waits++;
}
// Eden mode has no raw global memory write log (DMSH_GMEM_LOG is the DM_VK_EDEN=0 path's): an atomic that stayed global (the pool table binding, cbuf 0xFF, is_atomic)
// reaches pools only the GPU knows, so nothing converts for it (rule H11). Counted, with the shader, so the conversions it misses show
void vk_hy_raw_unlocated(uint64_t key) {
    g_hy_raw_unloc++;
    static uint64_t seen[64]; static int ns; int k = 0; for (int i = 0; i < ns && !k; i++) k = seen[i] == key;
    if (!k && ns < 64) { seen[ns++] = key; VKLOG("hybrid1: shader %016llx runs an atomic through raw global memory (no write log in Eden mode): where it lands is not known, nothing converts for it (%llu such bindings so far)",
        (unsigned long long)key, (unsigned long long)g_hy_raw_unloc); }
}
// Diagnostic (hybrid1, every mode; changes nothing): storage buffer bindings by kind, and what the GPU-write bookkeeping does for each kind. The compute path takes
// every storage buffer of a dispatch as written (gfx_vk_eden.inc, gfx_vk_exec.c: "taken as written"), read-only ones included: their share of the pre copies, of
// the GPU-write list (g_gw: CPU reads of those bytes wait for the submission) and of the vertex mirrors made stale is counted apart
enum { HYA_NONE, HYA_CRO, HYA_CW, HYA_ATOM, HYA_GW, HYA_N };   // the binding a vk_gpu_wrote / vk_sub_wrote comes from: compute read-only, compute written, atomic, a draw's written
static int g_hy_attr;   // set around the call (the replay thread)
static uint64_t g_hyb_n[HYA_N], g_hyb_bytes[HYA_N], g_hyb_pre[HYA_N], g_hyb_gw[HYA_N], g_hyb_vbm[HYA_N];
void vk_hy_bind(int kind, uint64_t bytes) { if (kind > 0 && kind < HYA_N) { g_hyb_n[kind]++; g_hyb_bytes[kind] += bytes; } g_hy_attr = kind > 0 && kind < HYA_N ? kind : HYA_NONE; }
void vk_hy_bind_end(void) { g_hy_attr = HYA_NONE; }
static void hy_report(void) {   // every 120 submissions
    static uint64_t n0[HYA_N], b0[HYA_N], p0[HYA_N], w0[HYA_N], v0[HYA_N], r0[2];
    double mb[HYA_N][4] = {{0}}; uint64_t nn[HYA_N] = {0};
    for (int k = 1; k < HYA_N; k++) { nn[k] = g_hyb_n[k] - n0[k]; mb[k][0] = (g_hyb_bytes[k] - b0[k]) / 1048576.0; mb[k][1] = (g_hyb_pre[k] - p0[k]) / 1048576.0;
        mb[k][2] = (g_hyb_gw[k] - w0[k]) / 1048576.0; mb[k][3] = (double)(g_hyb_vbm[k] - v0[k]); }
    VKLOG("hybrid1 binds, last 120 submissions: compute storage buffers read-only %llu (%.1f MB bound; taken as written: pre copies %.1f MB, GPU-write list %.1f MB, vertex mirror blocks made stale %.0f), "
          "compute written %llu (%.1f MB; pre %.1f MB, list %.1f MB, mirrors %.0f), atomic %llu (%.1f MB; pre %.1f MB, list %.1f MB, mirrors %.0f), draws written %llu (%.1f MB; pre %.1f MB, list %.1f MB, mirrors %.0f)",
          (unsigned long long)nn[HYA_CRO], mb[HYA_CRO][0], mb[HYA_CRO][1], mb[HYA_CRO][2], mb[HYA_CRO][3], (unsigned long long)nn[HYA_CW], mb[HYA_CW][0], mb[HYA_CW][1], mb[HYA_CW][2], mb[HYA_CW][3],
          (unsigned long long)nn[HYA_ATOM], mb[HYA_ATOM][0], mb[HYA_ATOM][1], mb[HYA_ATOM][2], mb[HYA_ATOM][3], (unsigned long long)nn[HYA_GW], mb[HYA_GW][0], mb[HYA_GW][1], mb[HYA_GW][2], mb[HYA_GW][3]);
    if (pc_hybrid()) VKLOG("hybrid1 conversions: requests %llu storage atomic, %llu raw atomic log (last 120 submissions); %llu units, %llu chunks, %.1f MB device-local copies so far (%llu imports refused); "
          "submissions that waited for the earlier ones to convert %llu; raw atomics with no write log %llu (log full %llu); request list full %llu",
          (unsigned long long)(g_hy_req[0] - r0[0]), (unsigned long long)(g_hy_req[1] - r0[1]), (unsigned long long)g_hy_units, (unsigned long long)g_hy_chunks, g_hy_bytes / 1048576.0,
          (unsigned long long)g_hy_refused, (unsigned long long)g_hy_waits, (unsigned long long)g_hy_raw_unloc, (unsigned long long)g_hy_raw_full, (unsigned long long)g_hy_qfull);
    { static uint64_t a0, a1, a2, a3;
      VKLOG("hybrid2 storage buffer writes, last 120 submissions: read-only bindings not taken as written %llu (%.1f MB), written bindings cut to their stores' reach %llu (%.1f MB not taken)",
            (unsigned long long)(g_ssbo_ro_n - a0), (g_ssbo_ro_bytes - a1) / 1048576.0, (unsigned long long)(g_ssbo_cut_n - a2), (g_ssbo_cut_bytes - a3) / 1048576.0);
      a0 = g_ssbo_ro_n; a1 = g_ssbo_ro_bytes; a2 = g_ssbo_cut_n; a3 = g_ssbo_cut_bytes; }
    for (int k = 1; k < HYA_N; k++) { n0[k] = g_hyb_n[k]; b0[k] = g_hyb_bytes[k]; p0[k] = g_hyb_pre[k]; w0[k] = g_hyb_gw[k]; v0[k] = g_hyb_vbm[k]; }
    r0[0] = g_hy_req[0]; r0[1] = g_hy_req[1];
}
static void pool_make_0(vkpool_t *p, int late);
static void pool_make(vkpool_t *p, int late) {   // s47: timed
    uint64_t t = s47_on() ? s47_ns() : 0, a = p->lo, l = p->hi - p->lo; pool_make_0(p, late);
    if (t) { uint64_t d = s47_ns() - t; s47_add(S47_MAKE, d); s47_big(!p->buf ? "pool_make (no buffer)" : p->zero_copy ? "pool_make (zero-copy)" : "pool_make (copied)", a, l, d); } }
static void pool_make_0(vkpool_t *p, int late) {   // the chunk's buffer: imported, else a GPU copy; under the write lock
    const void *mem = (const void *)(uintptr_t)p->lo; uint64_t size = p->hi - p->lo; p->serial = ++g_pool_ser; p->retry = 0; p->s48 = 0; p->s48b = 0;   // (s48_bind sets it)
    uint64_t t47i = s47_on() ? s47_ns() : 0;   // s47
    // poolcopy: in copy mode no chunk is imported, bridges included (decided before the bridge path: review P1-4). hybrid1: neither is a chunk of a unit already copied
    const int hy = pc_hybrid();
    int zc = vk_poolcopy_on() == 1 || (hy && hy_is_copy(p)) ? -8 :((uintptr_t)mem & 0xfff) || (size & 0xfff) ? -3 : g_bridging || p->bridge ? bridge_zero_copy(p, (uint8_t *)mem, size) : getenv("DM_VK_NO_ZEROCOPY") ? -4 : zc_over(p->end - p->base) ? -5 : pool_zero_copy(p, (uint8_t *)mem, size);
    if (t47i) s47_add(S47_IMPORT, s47_ns() - t47i);   // s47
    if (zc == -2) {   // r78: the driver refused it. Chunks unused for 16 submissions are in no submission in flight (NSLOT) nor in this one (every use this submission
        // stamps them: the descriptor set caches last one submission), so they are destroyed at once and the import tried again; copied only if it is refused still.
        // Not at once while raw global memory's pool table was bound lately (it reaches chunks with no lookup): deferred, and this chunk copied meanwhile
        const uint64_t idle_subs = (uint64_t)nslot_on() + 4 > 16 ? (uint64_t)nslot_on() + 4 : 16;   // slot12: more than the submissions in flight (NSLOT in use), never under the old 16
        int now = g_gmem_sub + idle_subs < g_nsubs; uint64_t r = pool_release(size > (64ull << 20) ? 2 * size : 128ull << 20, idle_subs, now);
        if (r && now) zc = pool_zero_copy(p, (uint8_t *)mem, size);
        static int n; if (n++ < 16) VKLOG("r78: memory pool chunk %p +%#llx not imported (made: zero-copy %.0f MB, copied %.0f MB): %.1f MB of unused chunks released %s%s", mem, (unsigned long long)size,
            g_zc_bytes / 1048576.0, g_cp_bytes / 1048576.0, r / 1048576.0, now ? "at once, imported again: " : "after this submission (raw global memory in use)", now ? (zc ? "refused still, copied" : "done") : ""); }
    if (zc && pool_is_sub(p)) {   // s11: a per-view chunk refused too is not copied a second time: it stays without a buffer, and the copy serves its ranges
        p->retry = ~0ull; static int nl; if (nl++ < 8) VKLOG("[RED] s11: per-view memory pool chunk %p +%#llx refused too (%d): its ranges stay with the copy", mem, (unsigned long long)size, zc);
        return; }
#if defined(DM_NXN1X)
    if (zc == -2) { static int nd; if (nd++ < 4) { char rg[1400]; nvnh_range_regions(mem, size, rg, sizeof rg);   // s10: why the driver refused it (views, protection)
        VKLOG("s10: refused memory pool chunk %p +%#llx, region by region: %s", mem, (unsigned long long)size, rg); } }
#endif
    if (zc) {
        if (zc == -8 || hy) {   // poolcopy; hybrid1: an import refused is a device-local copy too (its unit with it, so its aliases follow at the next submission)
            if (pc_make(p, (uint8_t *)mem, size)) die("[vk] poolcopy: device-local copy of memory pool chunk %p +%#llx could not be made; command not continued", mem, (unsigned long long)size);
            if (hy) hy_made_copy(p, zc);
        } else {
        static int said; if (!said++) VKLOG("[RED] memory pool %p +%#llx could not be zero-copy (%d): a GPU copy is refreshed before each submission (slow)", mem, (unsigned long long)size, zc);
        uint32_t ty=pool_shadow_required(p,size);
        VkMemoryPropertyFlags tf=V.mem.memoryTypes[ty].propertyFlags;
        { static int said_ty; if (!said_ty++) VKLOG("copied pools: memory type %u (%s%s%s)", ty, tf & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? "cached" : "uncached",
            tf & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? ", coherent" : ", not coherent: flushed and invalidated by range", tf & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? ", device-local" : ""); }
        }
        pool_gpu_addr(p);
        if (!g_snaps && (g_snaps = calloc(SNAP_N, sizeof *g_snaps))) nvn_dirty_set_snap_hook(snap_hook);   // (snap_merge)
        if (late) {   // made during a submission's recording, after its sync (mtl_sub_begin): copied whole now, as the sync's first copy of a pool is
            uint64_t t47c = s47_on() ? s47_ns() : 0;   // s47
            uint8_t *lo = (uint8_t *)(uintptr_t)p->lo; uint64_t st = nvn_dirty_now();
            if (!p->span && pool_tracked(p)) p->span = nvn_dirty_span(lo, size);   // (s28)
            if (p->span) nvn_dirty_protect(lo, size);
            memcpy(p->shadow, lo, size); shadow_range(p, 0, size, 0); pc_upload(p, 0, size); g_sync_bytes += size; g_sync_full++; p->synced = 1; p->stamp0 = p->stamp = st; p->sync_epoch = g_pool_sync_epoch; g_shadow_pools = 1; if (t47c) s47_add(S47_COPY, s47_ns() - t47c); }
    }
    if (zc) g_cp_bytes += size; else g_zc_bytes += size;
    static int nlog; static uint64_t mark = 256ull << 20; int big = g_zc_bytes + g_cp_bytes >= mark; if (big) mark += 256ull << 20;
    if (nlog++ < 32 || big) VKLOG("memory pool %p +%#llx: %s%s (zero-copy %.1f MB, copied %.1f MB, of %.1f MB in pools)", mem, (unsigned long long)size, zc ? "copied" : "zero-copy",
        late ? " at its first GPU use (r77)" : "", g_zc_bytes / 1048576.0, g_cp_bytes / 1048576.0, g_reg_bytes / 1048576.0);
#if defined(DM_NXN1X)
    // s11: refused because it spans NXN1X's mapped views (its kernel's page heap hands out at most 1 GB physically contiguous, each block mapped as a view of
    // its own; s10's dump: two allocation bases, 74 + 6 MB): one more chunk per view, each imported at its first use. A range inside one view is looked up
    // there (pool_rank), only a range across views uses the copy
    if (zc == -2 && p->shadow && !hy && p >= g_pool && p < g_pool + g_npool) {   // (hybrid1: the refused chunk is a copy and its unit is copied: no per-view imports beside it)
        uint64_t cut[8]; int nc = nvnh_range_views(mem, size, cut, 8), added = 0; uint64_t from = 0;
        for (int k = 0; nc && k <= nc && g_npool < MAXPOOL && g_nsub < MERGE_N; k++) { uint64_t to = k < nc ? cut[k] : size;
            int had = 0; for (int j = 0; j < g_nsub; j++) if (g_sub[j].lo == p->lo + from && g_sub[j].hi == p->lo + to) had = 1;
            if (!had && to > from && !((from | to) & 0xfff)) { vkpool_t *s = &g_pool[g_npool++]; memset(s, 0, sizeof *s);
                s->lo = p->lo + from; s->hi = p->lo + to; s->base = p->base; s->end = p->end; s->flags = p->flags; s->serial = ++g_pool_ser;
                g_sub[g_nsub].lo = s->lo; g_sub[g_nsub].hi = s->hi; g_nsub++; added++; }
            from = to; }
        if (added) { pool_ord_make(); VKLOG("s11: memory pool chunk %p +%#llx spans %d mapped views: %d chunks added, one per view, each imported at its first use", mem, (unsigned long long)size, nc + 1, added); } }
#endif
}
static void pool_add_one(const void *mem, uint64_t size, uint64_t base, uint64_t end) {
    pool_wrlock();
    if (g_npool == MAXPOOL) { pool_unlock(); die("[vk] too many memory pools"); }
    vkpool_t *p = &g_pool[g_npool]; memset(p, 0, sizeof *p);
    p->lo = (uint64_t)(uintptr_t)mem; p->hi = p->lo + size; p->base = base; p->end = end; p->serial = ++g_pool_ser;
    g_reg_bytes += size;
    if (!pool_lazy()) pool_make(p, 0);
    g_npool++; pool_ord_make(); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
    pool_unlock();
    if (pool_lazy()) { static int nlog; if (nlog++ < 24) VKLOG("memory pool %p +%#llx: registered, its GPU buffer made at its first GPU use (r77; %.1f MB in pools)", mem, (unsigned long long)size, g_reg_bytes / 1048576.0); }
}
void mtl_pool_del(const void *mem) {
    pool_wrlock();
    for (int i = 0; i < g_npool; i++) if (g_pool[i].base == (uint64_t)(uintptr_t)mem)   // s11: the pool's per-view chunk records
        for (int k = 0; k < g_nsub; k++) if (g_sub[k].lo == g_pool[i].lo && g_sub[k].hi == g_pool[i].hi) g_sub[k--] = g_sub[--g_nsub];
    pool_unlock();
    for (;;) {   // every chunk of the pool
        pool_wrlock();
        int i = 0; while (i < g_npool && g_pool[i].base != (uint64_t)(uintptr_t)mem) i++;
        if (i == g_npool) { pool_unlock(); return; }
        vkpool_t p = g_pool[i]; g_pool[i] = g_pool[--g_npool]; if (p.bridge && g_nbridge > 0) g_nbridge--;   /* s32 */ if (p.buf) { if (p.zero_copy) g_zc_bytes -= p.hi - p.lo; else g_cp_bytes -= p.hi - p.lo; } pool_ord_make(); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
        if (p.s48) s48_unref(p.s48 - 1);   // NVN2 (11-16): a finalized pool's chunks gave nothing back, so their ranges stayed held (MKW: 1536 MB of ranges, 267 MB
        if (p.s48b) s48_unref(p.s48b - 1); //  of chunks). Only nvn2_pressure frees a range (s48_free_idle), so Switch 1 keeps every range as before
        pool_unlock();
        if (p.buf) vk_defer_free_buffer(p.buf, p.mem);   // (none: never used by the GPU, r77) an imported pool's pages stay the game's
        if (p.dl) { pool_wrlock(); pc_free_buffers(&p, 0); pool_unlock(); }   // poolcopy: its shadow and readback, after the GPU passed it (a range's memory stays)
        if (p.mem2) vk_defer_free_buffer(VK_NULL_HANDLE, p.mem2);   // s29: a sparse bridge's second piece, after its buffer
    }
}
static void s47_big(const char *what, uint64_t a, uint64_t l, uint64_t ns) {   // s47: one pool operation over 10 ms
    if (ns < 10000000ull) return;
    static int n; if (n++ < 64) VKLOG("[RED] s47: %s %#llx +%#llx took %.1f ms (%d pools, %d bridges)", what, (unsigned long long)a, (unsigned long long)l, ns / 1e6, g_npool, g_nbridge); }
static void s47_report(void) {   // s47: every 2 s, when the slow parts took over 20 ms in all
    static uint64_t last; uint64_t now = s47_ns(); if (!last) last = now;
    if (now - last < 2000000000ull) return;
    double s = (now - last) / 1e9; last = now;
    uint64_t ns[S47_N], n[S47_N];
    for (int k = 0; k < S47_N; k++) { ns[k] = __atomic_exchange_n(&g_s47_ns[k], 0, __ATOMIC_RELAXED); n[k] = __atomic_exchange_n(&g_s47_n[k], 0, __ATOMIC_RELAXED); }
    if (ns[S47_FIND] + ns[S47_BRIDGE] + ns[S47_MAKE] + ns[S47_WAITW] + ns[S47_ORD] < 20000000ull) return;
#define S47_A(k) (unsigned long long)n[k], ns[k] / 1e6
    VKLOG("s47: pool lookups, last %.1f s: slow lookups %llu %.1f ms (full scans %llu %.1f ms), bridge_make %llu %.1f ms, pool_make %llu %.1f ms (import %llu %.1f ms, first copy %llu %.1f ms), "
        "lock waits: write %llu %.1f ms, read %llu %.1f ms; index remade %llu %.1f ms; %d pools, %d bridges (DM_VK_S47=0 off)", s, S47_A(S47_FIND), S47_A(S47_SCAN), S47_A(S47_BRIDGE),
        S47_A(S47_MAKE), S47_A(S47_IMPORT), S47_A(S47_COPY), S47_A(S47_WAITW), S47_A(S47_WAITR), S47_A(S47_ORD), g_npool, g_nbridge);
#undef S47_A
}
static vkpool_t *pool_find(uint64_t addr, uint64_t len) {   // (a chunk with no buffer yet too, r77)
    // the last pools this thread found, while none was added or removed: the replay looks up every buffer binding of every draw, and the read lock alone was ~6% of it
    static __thread struct { uint32_t gen; vkpool_t *p[4]; } c;   // gen: g_pool_gen + 1 (0: empty)
    uint32_t g = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1;
    if (c.gen == g) for (int i = 0; i < 4; i++) { vkpool_t *p = c.p[i]; if (p && addr >= p->lo && addr + len <= p->hi && (!g_pool_ovl[p - g_pool] || !pool_rank(p))) return p; }   // (s10: shared memory searched)
    vkpool_t *r = NULL; uint64_t t47 = s47_on() ? s47_ns() : 0, t47s = 0;   // s47
    pool_rdlock();
    int lo = 0, hi = g_npool;   // the last pool by lo that starts at or below addr
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pool[g_pool_ord[m]].lo <= addr) lo = m + 1; else hi = m; }
    if (lo > 0) { vkpool_t *p = &g_pool[g_pool_ord[lo - 1]]; if (addr + len <= p->hi) r = p; }
    if (!r && t47) t47s = s47_ns();   // s47
    if (!r)   // Check overlapping registrations before the caller handles a cross-chunk range.
        for (int i = 0; i < g_npool; i++) if (addr >= g_pool[i].lo && addr + len <= g_pool[i].hi) { r = &g_pool[i]; break; }
    if (r && g_pool_ovl[r - g_pool] && pool_rank(r)) {   // s10: memory other pools share: the best-ranked pool holding the range (a tie keeps the one found)
        vkpool_t *f = r;
        for (int i = 0; i < g_npool; i++) { vkpool_t *q = &g_pool[i]; if (q != r && addr >= q->lo && addr + len <= q->hi && pool_rank(q) < pool_rank(r)) r = q; }
        if (r != f) __atomic_fetch_add(&g_ovl_pick, 1, __ATOMIC_RELAXED); }
    pool_unlock();
    if (t47) { uint64_t e = s47_ns(); s47_add(S47_FIND, e - t47); if (t47s) s47_add(S47_SCAN, e - t47s); s47_big("lookup", addr, len, e - t47); s47_report(); }   // s47
    if (r) { if (c.gen != g) { memset(&c, 0, sizeof c); c.gen = g; } memmove(&c.p[1], &c.p[0], 3 * sizeof *c.p); c.p[0] = r; }
    return r;
}
#if defined(DM_NXN1X)
// s13: a range across two of a pool's per-view chunks (s12; run 13b: a 10 KB range at 9.3 s had no buffer) gets a copy of the memory around it ("bridge", 1 MB
// either side), made at this lookup and refreshed before each submission as every copy is. Ranges inside one view still go to the imported chunks (pool_rank:
// imported before copied), so the bridge serves only ranges across the boundary. It sees the CPU's writes (its pages are watched, as every copy's) but not the
// GPU's through the imported chunks beside it. Under the write lock it takes; NULL: none (no pool there, no GPU memory, or BRIDGE_MAX bridges made)
enum { BRIDGE_PAD = 1u << 20, BRIDGE_MAX = 1024 };   // s32: was 64
// s32 (TotK run 48, 11-13): once BRIDGE_MAX (then 64, and bridges of deleted pools kept counting) was reached at 97 s, every draw reading a range across two
// views ran bridge_make again: a scan of all chunks, the write lock (the replay thread then sat in ntdll waits, contended with the other threads' lookups),
// two more scans, then NULL; the replay took 23 ms a frame in the field. Now this thread remembers lookups that found nothing: a range in no pool while no pool
// was added or removed (g_pool_gen; *neg 1), and a range no bridge could be made for, for 60 submissions (*neg 2). DM_VK_S32_NEG=0: off
enum { NEG_N = 64 };
static __thread struct { uint64_t addr, len, until; uint32_t gen; uint8_t kind; } g_neg[NEG_N];
static __thread uint64_t g_neg_hits;
static int neg_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S32_NEG"); on = !e || !*e || atoi(e) != 0; } return on; }
static unsigned neg_slot(uint64_t a, uint64_t l) { return (unsigned)(((a ^ (l << 40)) * 0x9E3779B97F4A7C15ull) >> 58); }   // NEG_N = 64 slots
static int neg_hit(uint64_t a, uint64_t l) {
    if (!neg_on()) return 0;
    unsigned s = neg_slot(a, l);
    if (!g_neg[s].kind || g_neg[s].addr != a || g_neg[s].len != l) return 0;
    int hit = g_neg[s].kind == 1 ? g_neg[s].gen == __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) : (uint64_t)g_nsubs < g_neg[s].until;
    if (hit) { uint64_t k = ++g_neg_hits; if (k >= 4096 && !(k & (k - 1))) VKLOG("s32: %llu pool lookups on this thread answered by its cache of misses (this one %#llx +%#llx: %s)",
        (unsigned long long)k, (unsigned long long)a, (unsigned long long)l, g_neg[s].kind == 1 ? "in no pool" : "no bridge can be made"); }
    return hit;
}
static void neg_put(uint64_t a, uint64_t l, int kind, uint32_t gen) {
    if (!neg_on()) return;
    unsigned s = neg_slot(a, l); g_neg[s].addr = a; g_neg[s].len = l; g_neg[s].kind = (uint8_t)kind; g_neg[s].gen = gen; g_neg[s].until = (uint64_t)g_nsubs + 60;
}
static vkpool_t *bridge_make_0(uint64_t addr, uint64_t len, int *neg);
static vkpool_t *bridge_make(uint64_t addr, uint64_t len, int *neg) {   // s47: timed
    uint64_t t = s47_on() ? s47_ns() : 0; vkpool_t *r = bridge_make_0(addr, len, neg);
    if (t) { uint64_t d = s47_ns() - t; s47_add(S47_BRIDGE, d); s47_big(r ? "bridge_make (a bridge)" : *neg == 1 ? "bridge_make (in no pool)" : "bridge_make (none made)", addr, len, d); }
    return r; }
static vkpool_t *bridge_make_0(uint64_t addr, uint64_t len, int *neg) {
    vkpool_t *r = NULL; const vkpool_t *p = NULL; int cross = 0;
    pool_rdlock();   // most misses are in no pool: no write lock for them
    for (int i = 0; i < g_npool && !cross; i++) if (addr >= g_pool[i].lo && addr < g_pool[i].hi && addr + len > g_pool[i].hi && addr + len <= g_pool[i].end) cross = 1;
    pool_unlock();
    if (!cross) { *neg = 1; return NULL; }
    pool_wrlock();
    for (int i = 0; i < g_npool && !r; i++) { vkpool_t *q = &g_pool[i]; if (q->buf && addr >= q->lo && addr + len <= q->hi) r = q; }   // one made meanwhile
    for (int i = 0; i < g_npool && !r && !p; i++) if (addr >= g_pool[i].lo && addr < g_pool[i].hi && addr + len > g_pool[i].hi && addr + len <= g_pool[i].end) p = &g_pool[i];
    if (!r && p && g_npool < MAXPOOL && g_nbridge < BRIDGE_MAX) {
        uint64_t lo = addr > p->base + BRIDGE_PAD ? (addr - BRIDGE_PAD) & ~0xfffull : p->base, hi = (addr + len + BRIDGE_PAD + 0xfff) & ~0xfffull; if (hi > p->end) hi = p->end;
        sparse_fit(p, addr, len, &lo, &hi);   // s29: on whole sparse blocks either side of the view boundary, when bridges can be sparse buffers
        vkpool_t *b = &g_pool[g_npool]; memset(b, 0, sizeof *b);
        b->lo = lo; b->hi = hi; b->base = p->base; b->end = p->end; b->flags = p->flags; b->bridge = 1;   // (s29: made again as a bridge after a release)
        g_bridging = 1; pool_make(b, 1); g_bridging = 0;
        g_npool++; g_nbridge++; pool_ord_make(); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE);
        static int nl; if (nl++ < 16) VKLOG("s13: range %#llx +%#llx crosses two mapped views of memory pool %p: a bridge %#llx +%#llx made for it, %s (%d bridges)",   // s30c: says which
            (unsigned long long)addr, (unsigned long long)len, (void *)(uintptr_t)b->base, (unsigned long long)lo, (unsigned long long)(hi - lo),
            b->mem2 ? "a sparse buffer (not copied)" : !b->buf ? "not made yet" : b->zero_copy ? "zero-copy" : "copied", g_nbridge);
        if (b->buf) r = b; }
    else if (!r && p) {
        die("[vk] required pool range has no bridge capacity: address=%#llx bytes=%llu bridges=%d chunks=%d; command not continued",
            (unsigned long long)addr,(unsigned long long)len,g_nbridge,g_npool);
    }
    if (r) r->used = g_nsubs;
    pool_unlock();
    return r;
}
#endif
// r77: the pool chunk holding [addr, addr+len), its GPU buffer made now if it has none yet (pool_make). The chunk may have gone between the search and the write
// lock (mtl_pool_del moves entries): searched again. Each lookup stamps the chunk's last use (pool_release)
vkpool_t *vk_pool_of(uint64_t addr, uint64_t len) {
    if ((len ? len : 1) > UINT64_MAX - addr) return NULL;
#if defined(DM_NXN1X)
    uint32_t g0 = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE);   // s32
    if (neg_hit(addr, len ? len : 1)) return NULL;
#endif
    for (int tries = 0; tries < 8; tries++) {
        vkpool_t *r = pool_find(addr, len ? len : 1);
        if (r && r->buf) { pool_used(r); return r; }
#if defined(DM_NXN1X)
        if (!r) { int neg = 0; vkpool_t *b = bridge_make(addr, len ? len : 1, &neg); if (!b && neg) neg_put(addr, len ? len : 1, neg, g0); if (b) pool_used(b); return b; }   // s13 (s32: misses remembered): across two mapped views, or in no pool (then NULL as before)
#else
        if (!r) return NULL;
#endif
        int ok = 0;
        pool_wrlock();
        { uint64_t ser = r->serial;
            if (r < g_pool + g_npool && r->serial == ser && addr >= r->lo && addr + len <= r->hi) { ok = 1; if (!r->buf && r->retry <= g_nsubs) { pool_make(r, 1); __atomic_fetch_add(&g_pool_gen, 1, __ATOMIC_RELEASE); }
                // s10/s11: over memory other pools share, the best-ranked one holding the range that has a buffer (the pool made at this use may rank below
                // another: a per-view chunk refused too)
                if (g_pool_ovl[r - g_pool]) for (int i = 0; i < g_npool; i++) { vkpool_t *q = &g_pool[i];
                    if (q != r && q->buf && addr >= q->lo && addr + len <= q->hi && (!r->buf || pool_rank(q) < pool_rank(r))) { r = q; g_ovl_alt++; } }
                // A failed optional per-view import is valid only if a real
                // overlapping buffer supplies this exact required range.
                if (!r->buf) die("[vk] required pool range has no backing buffer: address=%#llx bytes=%llu",(unsigned long long)addr,(unsigned long long)len); } }
        if (r) pool_used(r);
        pool_unlock();
        if (ok) return r;
    }
    return NULL;
}
uint32_t vk_pool_gen(void) { return __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE); }   // pools added or removed (gfx_vk_exec.c's descriptor set memo: vk_pool_of's results hold while it is unchanged)
int vk_buf_at(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    vkpool_t *p = vk_pool_of(addr, len ? len : 1);
    if (!p) return 0;
    *buf = p->buf; *off = addr - p->lo; return 1;
}
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    // Shader code is read through the contiguous CPU mapping. Only GPU bindings
    // need one VkBuffer covering the full range.
    const uint64_t bytes = len ? len : 1;
    if (bytes > UINT64_MAX - addr) return 0;
    pool_rdlock();
    int lo = 0, hi = g_npool;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (g_pool[g_pool_ord[mid]].lo <= addr) lo = mid + 1; else hi = mid;
    }
    while (lo > 0) {
        const vkpool_t *p = &g_pool[g_pool_ord[--lo]];
        if (addr >= p->hi || addr + bytes > p->end) continue;
        const uint64_t base = p->lo;
        int crosses = addr + bytes > p->hi;
        pool_unlock();
        if (buf) *buf = (void *)(uintptr_t)base;
        if (off) *off = addr - base;
        if (crosses) {
            static unsigned reported;
            if (__atomic_fetch_add(&reported, 1, __ATOMIC_RELAXED) < 4)
                VKLOG("CPU pool span %#llx +%#llx resolved across chunks", (unsigned long long)addr, (unsigned long long)len);
        }
        return 1;
    }
    pool_unlock();
    return 0;
}
void mtl_alias_drop_pool(const void *pool) { void vk_tex_drop_pool(const void *); vk_tex_drop_pool(pool); }
// r48 (rebase onto the newer upstream): the backend calls hle_nvn.c / hle_hid.c added since. This backend keeps its own GPU-write tracking (vk_gpu_wrote) and has no
// sparse tile heap, residency set or frame-step GPU clock ramp, so these keep upstream's contract without acting on it.
void mtl_pool_gpu_written(const void *pool) { pool_t *p = (pool_t *)pool; if (p) atomic_store_explicit(&p->gpu_written, 1, memory_order_relaxed); }   // the flag, as nvn_mtl.m sets it
void mtl_sp_phys_gone(const void *mem, uint64_t size) { (void)mem; (void)size; }
int mtl_probe_ramp_on(void) { const char *e = getenv("DM_FSTEP_PROBE_RAMP"); return e ? atoi(e) != 0 : 1; }   // nvn_mtl.m's default: on
void dm_gfx_restop(void) { LOG("DM_CTL restop: this backend has no residency set"); }
// The NVN flags of the pool at mem (hle_nvn.c, after mtl_pool_add): a copied pool the CPU maps uncached (CPU_UNCACHED, 0x2) is one the game writes and the GPU reads;
// the game reads GPU results from cached memory, so such a pool is not copied back after each submission
void vk_pool_set_flags(const void *mem, int flags) {
    pool_rdlock();
    for (int i = 0; i < g_npool; i++) if (g_pool[i].base == (uint64_t)(uintptr_t)mem) g_pool[i].flags = flags;
    pool_unlock();
}
// Raw global memory (binding kind DMSH_KIND_GMEM_MAP; nvn_mtl.m gmem_map_gpu, the same table): a shader that loads or stores at an address it read from memory, which the
// translator could tie to no storage buffer (Ryujinx GlobalToStorage.TryGenerateRawStorageOp), finds the pool holding the address here and goes through a physical storage
// buffer pointer to the pool's buffer. Bound empty before (all reads 0, stores dropped): DK's gsys vertex shader resets its terrain's indirect draw records this way every frame,
// and the cave's walls and floor were dark (11-04). Layout (dmsh_abi.h DMSH_GMEM_MAP_*): sorted by start, entry 0 empty, then the pools as disjoint ranges {start, end, the
// buffer's device address less its guest address}, unused entries starting at ~0. Rebuilt when the pools change (the replay thread only); the replaced table is freed
// when the submission recording now completes
static VkBuffer g_gmem_buf; static VkDeviceMemory g_gmem_mem; static uint32_t g_gmem_gen;
static void pc_pre_all(void);   // stage2c (below)
int vk_gmem_map(VkBuffer *buf, VkDeviceSize *size) {
#if defined(DM_NXN1X)
    nvnc_unknown(NVNC_UNKNOWN_RAW); // arbitrary addressing has no proven destination inventory
#endif
    if (g_gmem_sub != g_nsubs) pc_pre_all();   // stage2c: raw global memory bound in this submission: any chunk may be written (once per submission)
    enum { N = DMSH_GMEM_MAP_ENTRIES, E = DMSH_GMEM_MAP_ENTRY };
    *size = (VkDeviceSize)N * E;
    if (!V.has_bda) return 0;
    pool_sync_pending(0, UINT64_MAX);   // even a cached map can address any copied pool
    // Raw addressing can reach any copied buffer, including on a cached table
    // hit. Its addresses cannot be narrowed to the ordinary binding lookups.
    if (dm_sdk_ounce() && vk_sub_open()) pool_fence_max(&g_pool_sync_fence, mtl_last_fence() + 1);
    g_gmem_sub = g_nsubs;   // (r78 pool_release)
    uint32_t gen = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1;   // + 1: 0 is never built
    if (g_gmem_buf && g_gmem_gen == gen) { *buf = g_gmem_buf; return 1; }
    VkBuffer b; VkDeviceMemory m; void *map;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, *size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_SHARING_MODE_EXCLUSIVE};   // (device address: NVN2 descriptor buffer pipelines bind by address)
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &g_pool_af, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0)); VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &map));
    uint8_t *t = map; memset(t, 0, (size_t)*size);
    for (int k = 1; k < N; k++) memset(t + (size_t)k * E, 0xff, 8);
    int n = 1, over = 0; uint64_t end = 0;
    pool_rdlock();
    gen = __atomic_load_n(&g_pool_gen, __ATOMIC_ACQUIRE) + 1;
    for (int k = 0; k < g_npool; k++) {
        const vkpool_t *p = &g_pool[g_pool_ord[k]];
        uint64_t lo = p->lo > end ? p->lo : end, hi = p->hi;
        if (!p->gpu || lo >= hi) continue;
        if (n == N) { over++; continue; }
        uint64_t e[3] = {lo, hi, (uint64_t)p->gpu - p->lo}; memcpy(t + (size_t)n++ * E, e, sizeof e); end = hi;
    }
    pool_unlock();
    if (over) VKRED_ONCE(94, "raw global memory: %d more memory pools than the %d entries of the pool table; addresses in them are not reached", over, N - 1);
    if (g_gmem_buf) vk_defer_free_buffer(g_gmem_buf, g_gmem_mem);
    g_gmem_buf = b; g_gmem_mem = m; g_gmem_gen = gen; g_gmem_rebuilds++;
    { static int said; if (said++ < 4) VKLOG("raw global memory: pool table of %d ranges (pool generation %u), rebuilt whenever a pool comes or goes", n - 1, gen - 1); }
    *buf = b; return 1;
}
static uint64_t g_sync_bytes, g_sync_full, g_back_bytes, g_back_cmp, g_back_skip;
static uint64_t g_sync_cmp, g_sync_same, g_sync_ranges, g_sync_inval_failed;
static uint64_t g_gml_logs, g_gml_full, g_gml_pages, g_gml_back, g_gml_skip; uint64_t g_sbufu_w;   // raw global memory's write logs read (ran out of slots), pages in copied pools, copied back, left (the CPU wrote them after the sync); draws whose shader writes a storage buffer named in uniform memory (gfx_vk_exec.c: not copied back)   // copied pools: bytes copied by the incremental sync, full copies; bytes copied back, bytes compared for it
static uint64_t g_pool_unknown_write_start;   // preserve waits for writes whose range is unavailable
// a non-coherent copy: the CPU's writes made visible to the GPU (flush) or the GPU's to the CPU (invalidate), the range widened to whole atoms
static VkResult shadow_range(vkpool_t *p, uint64_t off, uint64_t n, int inval) {
    if (!p->noncoh || !n) return VK_SUCCESS;
    VkDeviceSize a = V.props.limits.nonCoherentAtomSize ? V.props.limits.nonCoherentAtomSize : 1, lo = off / a * a, hi = off + n, size = p->hi - p->lo;
    if (p->dl) {   // poolcopy: the shadow is p->smh at p->soff (shared backing ranges hold other chunks' shadows too)
        lo = (p->soff + off) / a * a; hi = (p->soff + off + n + a - 1) / a * a;
        VkMappedMemoryRange rr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, p->smh, lo, hi - lo};
        return inval ? vkInvalidateMappedMemoryRanges(V.dev, 1, &rr) : vkFlushMappedMemoryRanges(V.dev, 1, &rr); }
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, p->mem, lo, hi >= size ? VK_WHOLE_SIZE : (hi + a - 1) / a * a - lo};
    return inval ? vkInvalidateMappedMemoryRanges(V.dev, 1, &r) : vkFlushMappedMemoryRanges(V.dev, 1, &r);
}
// Copied pools: the guest's bytes into the GPU copy. The first sync copies the whole pool; later ones copy only the 64 KB blocks with pages the CPU wrote since the last
// sync (nvn_dirty's write-protect tracking: a block is protected again before it is copied, so a write during the copy is caught by the next sync). An 860 MB
// pool copied whole at every submission (Unicorn Overlord on a Legion Y700 2023) ran at a few frames a second
// bak89 (11-16, NPARK's replay video: whole structures drawn every other frame): copied pools are synced from guest memory before each submission; a range GPU work
// wrote in the copy that was not copied back would be put back to the guest's older bytes. Every 8th submission, before a sync copies over the copy, the 4 KB pages
// of it that GPU work wrote recently (the vk_gpu_wrote ring: dispatch storage buffers, copies) and that differ between the copy and guest memory are counted as
// reverted (dg_rev); bytes synced per pool (dg_sync). Logging only, NVN2 boots; DM_VK_POOL_DIAG=0 off
int vk_gpu_written_recent(uint64_t a, uint64_t n);
uint64_t g_pd_rev_total;
static int pool_diag_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_DIAG"); v = e && *e ? atoi(e) != 0 : dm_sdk_ounce() != 0; } return v; }
static void pool_diag_rev(vkpool_t *p, uint64_t off, uint64_t n) {   // before memcpy(p->shadow + off, guest + off, n)
    if (!pool_diag_on() || (g_nsubs & 7)) return;
    for (uint64_t o = off; o < off + n; o += 4096) { const uint64_t m = off + n - o < 4096 ? off + n - o : 4096;
        if (vk_gpu_written_recent(p->lo + o, m) && memcmp(p->shadow + o, (const void *)(uintptr_t)(p->lo + o), m)) { p->dg_rev += m; g_pd_rev_total += m; } }
}
// stage2c2 (async capture): only the bytes where the game's page differs from the shadow, as exact runs (design 8.2). A dirty 4 KB page may also hold bytes
// an earlier submission's GPU work has changed in the copy and that are not merged yet (the game's page and the shadow both still hold the old value there):
// uploading the whole page put the copy's bytes back to the old ones (self-test T1, 18:37: x came out 1, not 2). Runs are never joined across bytes that are not
// proven equal. One staging allocation and one copy command per 4 KB page with all of its runs
static int pc_stage_regions(vkpool_t *p, const VkBufferCopy *rg, uint32_t nr, uint64_t total);
static uint64_t g_pc_exact_runs, g_pc_exact_bytes, g_pc_exact_calls, g_pc_whole_calls;
static void pc_sync_exact(vkpool_t *p, uint64_t off, uint64_t n) {
    if (shadow_range(p, off, n, 1) != VK_SUCCESS) { g_sync_inval_failed++; p->synced = 0; g_pc_drain = g_fence + 1; return; }   // (synced in full after the merges)
    const uint8_t *G = (const uint8_t *)(uintptr_t)p->lo; uint8_t *B = p->shadow; int touched = 0;
    enum { MAXR = 256 }; VkBufferCopy rg[MAXR];
    for (uint64_t at = off, end = off + n; at < end;) {
        const uint64_t len = end - at < 4096 ? end - at : 4096;
        g_sync_cmp += len;
        if (!memcmp(B + at, G + at, (size_t)len)) { g_sync_same += len; at += len; continue; }
        uint32_t nr = 0; uint64_t total = 0;
        for (uint64_t x = at, e = at + len; x < e;) {
            // stage2c3: 8 bytes at a time to the next word holding a difference, then bytes only at the run's edges
            while (x + 8 <= e) { uint64_t g, b; memcpy(&g, G + x, 8); memcpy(&b, B + x, 8); if (g != b) break; x += 8; }
            while (x < e && G[x] == B[x]) x++;
            if (x >= e) break;
            uint64_t y = x + 1;
            while (y < e && G[y] != B[y] && (y & 7)) y++;
            while (y + 8 <= e) { uint64_t g, b; memcpy(&g, G + y, 8); memcpy(&b, B + y, 8); uint64_t dif = g ^ b; if (!dif) break;
                if ((dif & 0x00000000000000ffull) == 0 || (dif & 0x000000000000ff00ull) == 0 || (dif & 0x0000000000ff0000ull) == 0 || (dif & 0x00000000ff000000ull) == 0 ||
                    (dif & 0x000000ff00000000ull) == 0 || (dif & 0x0000ff0000000000ull) == 0 || (dif & 0x00ff000000000000ull) == 0 || (dif & 0xff00000000000000ull) == 0) break;   // a byte equal inside: the run ends in this word
                y += 8; }
            while (y < e && G[y] != B[y]) y++;
            if (nr == MAXR) { pc_stage_regions(p, rg, nr, total); nr = 0; total = 0; }   // (a page with more than 256 separate runs: in several commands)
            memcpy(B + x, G + x, (size_t)(y - x));
            rg[nr++] = (VkBufferCopy){0, x, y - x}; total += y - x; g_pc_exact_runs++; g_pc_exact_bytes += y - x; x = y;
        }
        if (nr && !pc_stage_regions(p, rg, nr, total)) { p->synced = 0; g_pc_drain = g_fence + 1; g_pc_lost++; }
        touched = 1; g_sync_ranges++; g_sync_bytes += total; p->dg_sync += total;
        at += len;
    }
    if (touched) shadow_range(p, off, n, 0);
}
static void pool_sync_copy(vkpool_t *p, uint64_t off, uint64_t n, int compare) {
    // stage2c3: exact runs only while a submission that has GPU work on this chunk is not merged yet (pc_wfence > g_done): only then can the copy hold GPU bytes the
    // game's page and the shadow do not have yet. With every such merge done, the copy equals the shadow and the whole 4 KB upload of stage 2a is exact as well
    if (p->dl && p->synced && pc_async() && vk_sub_open()) {
        if (p->pc_wfence > __atomic_load_n(&g_done, __ATOMIC_ACQUIRE)) { g_pc_exact_calls++; pc_sync_exact(p, off, n); return; }
        g_pc_whole_calls++; }
    const uint8_t *guest = (const uint8_t *)(uintptr_t)p->lo;
    // Dependencies have completed, including GPU -> CPU copy-back. Invalidate
    // non-coherent memory before comparing; never trust a stale CPU cache line.
    if (compare && shadow_range(p, off, n, 1) != VK_SUCCESS) { compare = 0; g_sync_inval_failed++; }
    if (!compare) {
        if (p->synced) pool_diag_rev(p, off, n);
        memcpy(p->shadow + off, guest + off, n); shadow_range(p, off, n, 0);
        if (p->dl) pc_upload(p, off, n);   // poolcopy: into the device-local copy
        g_sync_bytes += n; p->dg_sync += n;
        if (!off && n == p->hi - p->lo) g_sync_full++;
        return;
    }
    uint64_t end = off + n, run = end, first = end, last = off, copied = 0;
    for (uint64_t at = off; at < end;) {
        uint64_t len = end - at < 4096 ? end - at : 4096;
        g_sync_cmp += len;
        int same = !memcmp(p->shadow + at, guest + at, (size_t)len);
        if (same) g_sync_same += len;
        if (!same && run == end) run = at;
        uint64_t next = at + len;
        if (run != end && (same || next == end)) {
            uint64_t stop = same ? at : next, bytes = stop - run;
            pool_diag_rev(p, run, bytes); memcpy(p->shadow + run, guest + run, (size_t)bytes);
            if (p->dl) pc_upload(p, run, bytes);   // poolcopy (the flush below precedes the submission that runs the copy)
            if (first == end) first = run;
            last = stop; copied += bytes; g_sync_ranges++; run = end;
        }
        at = next;
    }
    // One enclosing flush, rather than a driver call for every changed page.
    // The whole range was invalidated above, so unchanged neighbours are valid.
    if (copied) shadow_range(p, first, last - first, 0);
    g_sync_bytes += copied; p->dg_sync += copied;
}
static void wait_fence(uint64_t f);
// stage2a2: one run of dirty pages of a copied pool (nvn_dirty_each_since): protected again and copied, as the stage2a page loop did per run
uint64_t nvn_dirty_each_since(const void *span, const void *addr, size_t len, uint64_t stamp, void (*fn)(void *ctx, uintptr_t a, size_t n), void *ctx, uint64_t *looked);
typedef struct { vkpool_t *p; int differential; } pc_run_ctx;
// ---- stage2b (11-20, design 8.3 item 2): volatile pages. A 4 KB page of a copied chunk that the CPU changed at 3 syncs in a row, and that no GPU work has ever
// written, is left writable (no fault per CPU write, no protect per sync: HWAoI moving, 19:02: 20k faults/s and 41.5k mprotect calls/s, 298 ms/s). Each sync
// compares it with the shadow 8 bytes at a time and uploads only the changed runs; the shadow takes the bytes from the snapshot that went into staging (one read
// of the game's page), so a CPU write racing the capture differs at the next sync instead of being lost. Changed pages are stamped (nvn_dirty_note_write) for the
// other readers of the generations. A volatile page is protected again after 8 syncs with no change, or at the next sync after any GPU work writes it (its pre
// marks the page). Writable pages do not fault, so the fault path, the host pre-writes (Memory::Write*) and the vfix notices neither stamp nor protect them.
// NXN1X_PC_VOLATILE=0: off (A/B)
struct pcvol { uint32_t np, seq, n, cap; uint8_t *st, *cnt, *idle, *gpu; uint32_t *last, *list, *gseq; };   // gseq: the sync number of the last GPU write noted (stage2b2 decay)   // st: 0 tracked, 1 volatile, 2 volatile until the next sync (a GPU write came)
static int g_pc_vol = -1;
static uint64_t g_run_hist[4], g_fpage_gpu, g_fpage_cpu;   // combined1120: dirty runs at sync by length (1, 2-3, 4-15, 16+ pages); faulted pages on GPU-written vs CPU-only pages
static uint64_t g_vol_pages, g_vol_entries, g_vol_exits, g_vol_evicts, g_vol_cmp_bytes, g_vol_cmp_ns, g_vol_up_bytes, g_vol_up_runs, g_vol_decays, g_pc_pre_all_n;
static int pc_vol_on(void) {
    if (g_pc_vol < 0) { const char *e = getenv("NXN1X_PC_VOLATILE"); g_pc_vol = !(e && *e == '0');
        VKLOG("stage2b: volatile pages %s (NXN1X_PC_VOLATILE=%s): CPU-only pages the CPU changes at 3 syncs in a row stay writable and are compared at every sync", g_pc_vol ? "on" : "off", e && *e ? e : "1 (default)"); }
    return g_pc_vol && pc_async();
}
static struct pcvol *pv_get(vkpool_t *p) {
    if (p->pv) return p->pv;
    struct pcvol *v = calloc(1, sizeof *v); if (!v) return NULL;
    v->np = (uint32_t)((p->hi - p->lo + 4095) >> 12);
    v->st = calloc(v->np, 1); v->cnt = calloc(v->np, 1); v->idle = calloc(v->np, 1); v->gpu = calloc(v->np, 1); v->last = calloc((size_t)v->np * 3, sizeof *v->last); v->gseq = calloc(v->np, sizeof *v->gseq);
    if (!v->st || !v->cnt || !v->idle || !v->gpu || !v->last || !v->gseq) { free(v->st); free(v->cnt); free(v->idle); free(v->gpu); free(v->last); free(v->gseq); free(v); return NULL; }
    v->seq = 1; p->pv = v; return v;
}
static void pv_free(vkpool_t *p) {
    struct pcvol *v = p->pv; if (!v) return;
    g_vol_pages -= v->n; free(v->st); free(v->cnt); free(v->idle); free(v->gpu); free(v->last); free(v->list); free(v->gseq); free(v); p->pv = NULL;
}
static void pv_reset(vkpool_t *p) {   // a full sync (the whole chunk protected again): nothing is volatile
    struct pcvol *v = p->pv; if (!v) return;
    g_vol_pages -= v->n; memset(v->st, 0, v->np); memset(v->cnt, 0, v->np); memset(v->idle, 0, v->np); memset(v->last, 0, (size_t)v->np * 3 * sizeof *v->last); v->n = 0;
}
static void pv_gpu_mark(vkpool_t *p, uint64_t off, uint64_t n) {   // pc_pre_range: GPU work writes [off, off + n): never volatile; a volatile page goes back at the next sync
    struct pcvol *v = pv_get(p); if (!v || !n) return;
    for (uint64_t i = off >> 12, e = (off + n - 1) >> 12; i <= e && i < v->np; i++) { v->gpu[i] = 1; v->gseq[i] = v->seq; if (v->st[i] == 1) v->st[i] = 2; }
}
void nvn_dirty_note_write(const void *addr, size_t len);
int nvn_dirty_faulted_since(const void *span, const void *addr, size_t len, uint64_t stamp);   // stage2b3
// One page [off, off + len) of chunk p: snapshot of the game's bytes, compared with the shadow 8 bytes at a time, changed runs into the shadow (from the snapshot)
// and uploaded (exact runs: safe whatever the GPU's state). 1 = something changed
static int pv_capture(vkpool_t *p, uint64_t off, uint64_t len) {
    uint8_t S[4096]; const uint64_t t0 = now_ns();
    memcpy(S, (const void *)(uintptr_t)(p->lo + off), (size_t)len);
    shadow_range(p, off, len, 1);
    uint8_t *B = p->shadow + off; VkBufferCopy rg[256]; uint32_t nr = 0; uint64_t total = 0;
    for (uint64_t x = 0; x < len;) {
        while (x + 8 <= len) { uint64_t s, b; memcpy(&s, S + x, 8); memcpy(&b, B + x, 8); if (s != b) break; x += 8; }
        while (x < len && S[x] == B[x]) x++;
        if (x >= len) break;
        uint64_t y = x + 1; while (y < len && S[y] != B[y]) y++;
        memcpy(B + x, S + x, (size_t)(y - x));
        if (nr == 256) { pc_stage_regions(p, rg, nr, total); nr = 0; total = 0; }
        rg[nr++] = (VkBufferCopy){0, off + x, y - x}; total += y - x; g_vol_up_runs++; x = y;
    }
    g_vol_cmp_bytes += len;
    if (nr) { shadow_range(p, off, len, 0); if (!pc_stage_regions(p, rg, nr, total)) { p->synced = 0; g_pc_drain = g_fence + 1; } g_vol_up_bytes += total;
        nvn_dirty_note_write((const void *)(uintptr_t)(p->lo + off), (size_t)len); }
    g_vol_cmp_ns += now_ns() - t0;
    return nr != 0;
}
// After the chunk's dirty runs: every volatile page compared (and those leaving: protected first, then compared, so a write before the protect is in the snapshot
// and one after it faults)
static void pv_scan(vkpool_t *p) {
    struct pcvol *v = p->pv; if (!v || !v->n) return;
    const uint64_t size = p->hi - p->lo; uint32_t keep = 0;
    for (uint32_t k = 0; k < v->n; k++) { const uint32_t i = v->list[k]; const uint64_t off = (uint64_t)i << 12, len = size - off < 4096 ? size - off : 4096;
        int leave = v->st[i] == 2;
        if (!leave && v->idle[i] + 1 >= 64) leave = 1;   // stage2b3: no change at the 63 syncs before: this is the 64th
        if (off >= size) { v->st[i] = 0; continue; }   // stage2b3: (a page past the chunk's end: never captured)
        if (leave) { nvn_dirty_protect((const void *)(uintptr_t)(p->lo + off), (size_t)len); pv_capture(p, off, len);
            if (v->st[i] == 2) g_vol_evicts++; else g_vol_exits++;
            v->st[i] = 0; v->cnt[i] = 0; v->idle[i] = 0; continue; }
        if (pv_capture(p, off, len)) v->idle[i] = 0; else v->idle[i]++;
        v->list[keep++] = i; }
    g_vol_pages -= v->n - keep; v->n = keep;
}
static void pc_sync_run(void *ctx, uintptr_t a, size_t n) {
    pc_run_ctx *c = ctx; vkpool_t *p = c->p; const uint64_t off = a - p->lo;
    { const uint64_t np = (n + 4095) >> 12; g_run_hist[np <= 1 ? 0 : np <= 3 ? 1 : np <= 15 ? 2 : 3]++; }   // combined1120
    struct pcvol *v = pc_vol_on() ? pv_get(p) : NULL;
    if (!v) { nvn_dirty_protect((const void *)a, n); pool_sync_copy(p, off, n, c->differential); g_pc_runs++; return; }
    // per page: volatile ones are left to pv_scan; a page changed at 3 syncs in a row (and never GPU-written) becomes volatile: copied, not protected
    uint64_t x = off, end = off + n, rs = off; int ra = -1;   // the run of pages with the same action so far (one protect and one copy per run, as before)
#define PV_FLUSH(upto) do { if (ra >= 1 && (upto) > rs) { if (ra == 1) nvn_dirty_protect((const void *)(uintptr_t)(p->lo + rs), (size_t)((upto) - rs)); \
        pool_sync_copy(p, rs, (upto) - rs, c->differential); g_pc_runs++; } } while (0)
    while (x < end) {
        const uint32_t i = (uint32_t)(x >> 12); const uint64_t pe = ((uint64_t)i + 1) << 12, e = pe < end ? pe : end;
        int act;   // 0 skip (volatile), 1 protect + copy, 2 copy (becomes volatile)
        if (i >= v->np || v->st[i]) act = i >= v->np ? 1 : 0;
        else {   // stage2b3: entry by a window: the page took a write fault at 3 of the last 32 syncs (last[3i..3i+2]: the sync numbers of its last 3 faults). Faults,
            // not stamps: the merges and host pre-writes stamp pages too, and Bayonetta 3 turned such pages volatile while its faulting pages stayed (19:44). And syncs,
            // not consecutive ones: HWAoI submits 6 times a frame, so a page written every frame changed at every 6th sync and never 3 in a row
            if (nvn_dirty_faulted_since(p->span, (const void *)(uintptr_t)(p->lo + x), (size_t)(e - x), p->stamp)) {
                if (v->gpu[i]) g_fpage_gpu++; else g_fpage_cpu++;   // combined1120 (hybrid step 0: faults on pages the GPU writes vs never)
                uint32_t *ev = v->last + 3 * (size_t)i; if (ev[0] != v->seq) { ev[2] = ev[1]; ev[1] = ev[0]; ev[0] = v->seq; } }
            v->cnt[i] = v->last[3 * (size_t)i + 2] && v->seq - v->last[3 * (size_t)i + 2] < 32 ? 3 : 0;
            // stage2b2: the GPU-written mark decays once every GPU write on the chunk is merged and none has come to this page for 8 syncs (raw global memory
            // marks every chunk whole: kept for good, a game binding it once had no volatile page for the rest of its run)
            if (v->gpu[i] && p->pc_wfence <= __atomic_load_n(&g_done, __ATOMIC_ACQUIRE) && v->seq - v->gseq[i] >= 8) { v->gpu[i] = 0; g_vol_decays++; }
            act = v->cnt[i] >= 3 && !v->gpu[i] ? 2 : 1;
            if (act == 2) { v->st[i] = 1; v->idle[i] = 0;
                if (v->n == v->cap) { uint32_t cp = v->cap ? 2 * v->cap : 256; uint32_t *l = realloc(v->list, cp * sizeof *l); if (!l) { v->st[i] = 0; act = 1; } else { v->list = l; v->cap = cp; } }
                if (act == 2) { v->list[v->n++] = i; g_vol_pages++; g_vol_entries++; } } }
        if (act != ra) { PV_FLUSH(x); rs = x; ra = act; }
        x = e;
    }
    PV_FLUSH(end);
#undef PV_FLUSH
}
// pcverify (diagnostic, DM_VK_PC_VERIFY=<MB per sync>, default 0 = off): after a copied pool's incremental sync, up to that many MB of it (a cursor going round
// the pool) are compared page by page against the shadow, which holds what the device-local copy has. A page whose bytes differ while its last generation is
// <= stamp0 was changed without a stamp (no fault, no host pre-write, no GPU-write stamp): the sync will never upload it, and the GPU reads stale bytes there.
// Logged (first 64, then every 1,024th): the page, its generation against the stamps, the tracker's state (0 writable, 1 read-only, 2 busy) against what Windows
// says the protection is, how many tracked ranges cover it, and the first differing byte. Not fixed up: it only shows the path
int nvn_dirty_page_info(const void *span, const void *addr, uint64_t *gen, int *state, uint64_t *g1, uint64_t *g2, int *ranges);
static uint64_t g_pcv_pages, g_pcv_bad; unsigned long dm_win_page_protect(const void *p, unsigned long *type);
static void pc_verify(vkpool_t *p) {
    static int64_t mb = -1; if (mb < 0) { const char *e = getenv("DM_VK_PC_VERIFY"); mb = e ? atoll(e) : 0; if (mb < 0) mb = 0;
        if (mb) VKLOG("pcverify: on, %lld MB of the copied pools compared against their shadow after each sync (DM_VK_PC_VERIFY)", (long long)mb); }
    if (!mb || !p->shadow) return;
    const uint64_t size = p->hi - p->lo, budget = (uint64_t)mb << 20;
    if (shadow_range(p, 0, size, 1) != VK_SUCCESS) return;
    uint64_t off = p->pcv_cur < size ? p->pcv_cur & ~4095ull : 0, done = 0;
    for (; done < budget && done < size; done += 4096, off = off + 4096 >= size ? 0 : off + 4096) {
        const uint64_t n = size - off < 4096 ? size - off : 4096; const uint8_t *G = (const uint8_t *)(uintptr_t)(p->lo + off), *S = p->shadow + off;
        g_pcv_pages++;
        if (!memcmp(G, S, (size_t)n)) continue;
        uint64_t gen = 0, g1 = 0, g2 = 0; int state = -1, ranges = 0;
        const int tracked = nvn_dirty_page_info(p->span, G, &gen, &state, &g1, &g2, &ranges);
        if (tracked && gen > p->stamp0) continue;   // written since: the next sync takes it
        const uint64_t k = ++g_pcv_bad;
        if (k <= 64 || k % 1024 == 0) {
            size_t x = 0; while (x < n && G[x] == S[x]) x++;
            unsigned long wtype = 0; const unsigned long wprot = dm_win_page_protect(G, &wtype);
            VKLOG("[RED] pcverify #%llu: page %p (pool %#llx..%#llx +%#llx) differs from its device-local copy without a stamp: generation %llu (stamp0 %llu, stamp %llu, now %llu; groups %llu / %llu), "
                  "tracker state %d, Windows protection %#lx (type %#lx), %d tracked ranges over it, first byte +%#zx: game %#04x, copy %#04x; pages compared so far %llu",
                (unsigned long long)k, (const void *)G, (unsigned long long)p->lo, (unsigned long long)p->hi, (unsigned long long)off, (unsigned long long)gen, (unsigned long long)p->stamp0,
                (unsigned long long)p->stamp, (unsigned long long)nvn_dirty_now(), (unsigned long long)g1, (unsigned long long)g2, tracked ? state : -1, wprot,
                wtype, ranges, x, x < n ? G[x] : 0, x < n ? S[x] : 0, (unsigned long long)g_pcv_pages);
        }
    }
    p->pcv_cur = off;
    static uint64_t calls; if (++calls % 2048 == 0) VKLOG("pcverify: %llu pages compared, %llu differed without a stamp", (unsigned long long)g_pcv_pages, (unsigned long long)g_pcv_bad);
}
static void pool_sync_one(vkpool_t *p, int differential) {
    // Source writebacks have completed before the registry lock is taken.
    // Ordinary reads need only GPU completion for this particular allocation.
    uint64_t use = __atomic_load_n(&p->last_binding_fence, __ATOMIC_ACQUIRE);
    // stage2c: a copied chunk's sync writes only its shadow and this submission's staging, which no earlier submission reads: no wait for its last use
    if (use && use <= mtl_last_fence() && use > vk_fence_done() && !(p->dl && pc_sync() != PC_SERIAL)) wait_fence(use);
    enum { BLK = 64 << 10 };
    uint8_t *lo = (uint8_t *)(uintptr_t)p->lo; uint64_t size = p->hi - p->lo, st = nvn_dirty_now();
    if (!p->span && pool_tracked(p) && (p->span = nvn_dirty_span(lo, size))) p->synced = 0;
    if (!p->synced || !p->span) {
        if (p->span) nvn_dirty_protect(lo, size);
        pv_reset(p);   // stage2b: protected whole again
        pool_sync_copy(p, 0, size, differential && p->synced);
        p->synced = 1; p->stamp0 = p->stamp = st;
    } else {
        // Keep two-generation dirty tracking: faults can publish a generation
        // after the preceding sync has inspected the block.
        if (p->dl) {   // stage2a (poolcopy): 4 KB pages, the dirty ones only, in runs (one protect and one copy per run of dirty pages). stage2a2: the runs come from
            // nvn_dirty_each_since, which skips clean groups of 64 / 4,096 pages by their highest generation instead of reading every page's (same generations, same writes)
            pc_run_ctx c = { p, differential }; if (p->pv) p->pv->seq++;   // stage2b: this sync's number (3 in a row)
            g_pc_pages_dirty += nvn_dirty_each_since(p->span, lo, size, p->stamp0, pc_sync_run, &c, &g_pc_pages_seen);
            pv_scan(p);   // stage2b: the volatile pages
        } else
        for (uint64_t off = 0; off < size; off += BLK) {
            uint64_t n = size - off < BLK ? size - off : BLK;
            if (!nvn_dirty_since_span(p->span, lo + off, n, p->stamp0)) continue;
            nvn_dirty_protect(lo + off, n); pool_sync_copy(p, off, n, differential);
        }
        p->stamp0 = p->stamp; p->stamp = st;
        if (p->dl) pc_verify(p);   // pcverify (DM_VK_PC_VERIFY=<MB per sync>, default off)
    }
    p->sync_epoch = g_pool_sync_epoch;
}
static void pool_refresh_deferred(vkpool_t *p, int guard) {   // pool write lock held
    if (!p->shadow || p->sync_epoch == g_pool_sync_epoch || (p->span && p->synced)) return;
    int deferred = p->synced;   // a newly made, uninitialized pool is a first copy, not a skipped scan
    uint64_t bytes = p->hi - p->lo;
    pool_sync_one(p, (pool_sync_mode() & 1) && g_pool_compare_valid);
    if (deferred) {
        if (g_pool_demand_pending) g_pool_demand_pending--;
        if (guard) g_pool_guard_bytes += bytes; else g_pool_demand_bytes += bytes;
    }
}
static void pool_sync_deferred(vkpool_t *p, int guard) {
    if (!g_pool_demand_active || !p->shadow || p->sync_epoch == g_pool_sync_epoch) return;
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_POOL_SYNC, p->hi - p->lo);
    uint64_t started = now_ns();
    pool_wrlock(); pool_refresh_deferred(p, guard); pool_unlock();
    g_ns_poolsync += now_ns() - started;
    vk_f0_end(&diag, 0);
}
static void pool_sync_pending(uint64_t lo, uint64_t hi) {
    if (!g_pool_demand_active || !g_pool_demand_pending) return;
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_POOL_SYNC, 0);
    uint64_t started = now_ns();
    pool_wrlock();
    for (int i = 0; i < g_npool; i++) {
        vkpool_t *p = &g_pool[i];
        if (p->lo < hi && lo < p->hi) pool_refresh_deferred(p, 1);
    }
    pool_unlock(); g_ns_poolsync += now_ns() - started;
    vk_f0_end(&diag, 0);
}
static void pool_demand_report(void) {
    if (!dm_sdk_ounce() || !g_nsubs || g_nsubs % 120) return;
    static uint64_t d0, u0, g0;
    uint64_t deferred = g_pool_deferred_bytes - d0, used = g_pool_demand_bytes - u0, guarded = g_pool_guard_bytes - g0;
    VKLOG("NVN2 pool demand: deferred %.1f MiB, refreshed at use %.1f MiB, refreshed by safety guards %.1f MiB, unused comparison avoided %.1f MiB (last 120 submissions)",
        deferred / 1048576.0, used / 1048576.0, guarded / 1048576.0,
        (deferred > used + guarded ? deferred - used - guarded : 0) / 1048576.0);
    d0 = g_pool_deferred_bytes; u0 = g_pool_demand_bytes; g0 = g_pool_guard_bytes;
}
void nvn_dirty_set_sync_mark(uint64_t gen); void nvn_dirty_fa_stats(uint64_t *seq, uint64_t *ahead, uint64_t *faults, uint64_t *ns);   // combined1120
void vk_pools_sync(VkCommandBuffer cb) {
    (void)cb;
    nvn_dirty_set_sync_mark(nvn_dirty_now());   // combined1120: faults from here on are this sync window's (fault-ahead)
    int differential = (pool_sync_mode() & 1) && g_pool_compare_valid;
    pool_wrlock();   // excludes the completion thread while stamps/copies change
    for (int i = 0; i < g_npool; i++) {
        vkpool_t *p = &g_pool[i]; if (!p->shadow) continue;
        // Do not re-enable small-bridge page protection (s28). Only defer an
        // initialized untracked copy; every first binding refreshes the WHOLE
        // pool, since several callers intentionally request only an address.
        if (g_pool_demand_active && p->synced && !p->span && !pool_tracked(p)) {
            g_pool_deferred_bytes += p->hi - p->lo; g_pool_demand_pending++; continue;
        }
        pool_sync_one(p, differential);
    }
    pool_unlock();
    atomic_store(&g_snap_epoch, nvn_dirty_now());   // snapshots from here on are of this sync's content (snap_hook / snap_merge)
    { static uint64_t ns, said = ~0ull; uint64_t k = __atomic_load_n(&g_ovl_pick, __ATOMIC_RELAXED);   // s10, every 600 syncs when it moved
      if (++ns % 600 == 0 && k != said) { said = k; VKLOG("s10: lookups in memory that a copied pool shares with another, sent to the imported or not yet made one: %llu (copies replaced at a first use: %llu)", (unsigned long long)k, (unsigned long long)g_ovl_alt); } }
    static uint64_t n, last, lastb, lastc, lasts; if (++n % 120 == 0) { VKLOG("copied pools, last 120 submissions: %.1f MB synced, %.1f MB copied back of %.1f MB compared (%.1f MB left: the CPU wrote them after the sync), %llu full copies so far", (g_sync_bytes - last) / 1048576.0, (g_back_bytes - lastb) / 1048576.0, (g_back_cmp - lastc) / 1048576.0, (g_back_skip - lasts) / 1048576.0, (unsigned long long)g_sync_full); last = g_sync_bytes; lastb = g_back_bytes; lastc = g_back_cmp; lasts = g_back_skip;
        if (dm_sdk_ounce()) { static uint64_t cmp0, same0, range0, narrow0, wait0;
        VKLOG("NVN2 pool sync cost: compared %.1f MiB, unchanged %.1f MiB skipped, changed runs %llu; dependency waits narrowed %llu, latest required %llu; invalidation failures %llu",
            (g_sync_cmp - cmp0) / 1048576.0, (g_sync_same - same0) / 1048576.0, (unsigned long long)(g_sync_ranges - range0),
            (unsigned long long)(g_pool_sync_narrow - narrow0), (unsigned long long)(g_pool_sync_fullwait - wait0), (unsigned long long)g_sync_inval_failed);
        cmp0 = g_sync_cmp; same0 = g_sync_same; range0 = g_sync_ranges; narrow0 = g_pool_sync_narrow;
        wait0 = g_pool_sync_fullwait; }
        static uint64_t l0, l1, l2, l3, l4, l5; VKLOG("raw global memory stores, last 120 submissions: %llu write logs (%llu ran out of slots), %llu pages in copied pools, %llu differed and were copied back, %llu left (the CPU wrote them after the sync); draws writing a storage buffer named in uniform memory (not copied back) %llu",
            (unsigned long long)(g_gml_logs - l0), (unsigned long long)(g_gml_full - l1), (unsigned long long)(g_gml_pages - l2), (unsigned long long)(g_gml_back - l3), (unsigned long long)(g_gml_skip - l4), (unsigned long long)(g_sbufu_w - l5));
        l0 = g_gml_logs; l1 = g_gml_full; l2 = g_gml_pages; l3 = g_gml_back; l4 = g_gml_skip; l5 = g_sbufu_w;
        static uint64_t m0, m1, m2, m3, m4; uint64_t tk = atomic_load(&g_snap_taken);
        VKLOG("pages both the CPU and the GPU wrote, last 120 submissions: %llu merged word by word (%llu words from the GPU, %llu words both wrote: the CPU's kept), %llu without a snapshot of the sync (left as the CPU wrote them); %llu snapshots taken",
            (unsigned long long)(g_mg_pages - m0), (unsigned long long)(g_mg_words - m1), (unsigned long long)(g_mg_both - m2), (unsigned long long)(g_mg_none - m3), (unsigned long long)(tk - m4));
        m0 = g_mg_pages; m1 = g_mg_words; m2 = g_mg_both; m3 = g_mg_none; m4 = tk;
        if (vk_poolcopy_on()) { static uint64_t q0, q1, q2, q3, q4, q5, q6;   // poolcopy
            VKLOG("poolcopy, last 120 submissions: %.1f MB uploaded in %llu copies, %.1f MB read back and compared (%llu chunks read back whole: writes of unknown size), %llu bytes merged into the game's pages, %llu bytes the CPU had changed too (kept); copies shared %.1f MB, own %.1f MB (%llu could not share their backing range); resyncs after refused submissions %llu, uploads with no submission %llu",
                (g_pc_up_bytes - q0) / 1048576.0, (unsigned long long)(g_pc_up_n - q1), (g_pc_rb_bytes - q2) / 1048576.0, (unsigned long long)(g_pc_rb_whole - q3), (unsigned long long)(g_pc_mg_bytes - q4),
                (unsigned long long)(g_pc_mg_cpu - q5), g_pc_shared / 1048576.0, g_pc_private / 1048576.0, (unsigned long long)g_pc_noalias, (unsigned long long)(g_pc_resync - q6), (unsigned long long)g_pc_lost);
            q0 = g_pc_up_bytes; q1 = g_pc_up_n; q2 = g_pc_rb_bytes; q3 = g_pc_rb_whole; q4 = g_pc_mg_bytes; q5 = g_pc_mg_cpu; q6 = g_pc_resync;
            if (pc_async()) { static uint64_t a0, a1, a2, a3, a4, a5, a6, a7;   // stage2c
                VKLOG("stage2c, last 120 submissions: pre copies %llu (%.1f MB), post %.1f MB; GPU bytes merged into the game's pages %llu, bytes the CPU wrote after the capture (kept) %llu; "
                      "refused submissions recovered %llu, entries dropped with a refused part %llu; waits for earlier merges (full syncs) %llu",
                    (unsigned long long)(g_pc_pre_n - a0), (g_pc_pre_bytes - a1) / 1048576.0, (g_pc_post_bytes - a2) / 1048576.0, (unsigned long long)(g_pc_as_mg - a3), (unsigned long long)(g_pc_as_cpu - a4),
                    (unsigned long long)(g_pc_as_rec - a5), (unsigned long long)(g_pc_as_drop - a6), (unsigned long long)(g_pc_drains - a7));
                a0 = g_pc_pre_n; a1 = g_pc_pre_bytes; a2 = g_pc_post_bytes; a3 = g_pc_as_mg; a4 = g_pc_as_cpu; a5 = g_pc_as_rec; a6 = g_pc_as_drop; a7 = g_pc_drains;
                static uint64_t x0, x1, x2, x3;   // stage2c3
                VKLOG("stage2c3, last 120 submissions: dirty runs captured as exact byte runs (a GPU write on the chunk not merged yet) %llu (%llu runs, %.2f MB), as whole 4 KB pages %llu",
                    (unsigned long long)(g_pc_exact_calls - x0), (unsigned long long)(g_pc_exact_runs - x1), (g_pc_exact_bytes - x2) / 1048576.0, (unsigned long long)(g_pc_whole_calls - x3));
                x0 = g_pc_exact_calls; x1 = g_pc_exact_runs; x2 = g_pc_exact_bytes; x3 = g_pc_whole_calls;
                { void nvn_dirty_mp_stats(uint64_t *, uint64_t *, uint64_t *); void nvn_dirty_stats(uint64_t *, uint64_t *);   // stage2b
                  static uint64_t t0, c0, ns0, f0, v0, v1, v2, v3, v4, v5, v6; uint64_t calls, pages, ns, faults, preps, t = now_ns();
                  nvn_dirty_mp_stats(&calls, &pages, &ns); nvn_dirty_stats(&faults, &preps); const double sec = t0 ? (double)(t - t0) / 1e9 : 0;
                  if (sec > 0) VKLOG("stage2b, last 120 submissions (%.1f s): volatile pages %llu now (%.1f MB); entered %llu, left idle %llu, left for a GPU write %llu; compared %.1f MB in %.2f ms a submission, "
                        "%llu changed runs, %.2f MB uploaded; mprotect %.0f calls/s, %.1f ms/s; write faults %.0f/s; GPU-written marks decayed %llu; raw global memory bound (every chunk marked GPU-written) %llu times so far",
                      sec, (unsigned long long)g_vol_pages, g_vol_pages * 4096.0 / 1048576.0, (unsigned long long)(g_vol_entries - v0), (unsigned long long)(g_vol_exits - v1), (unsigned long long)(g_vol_evicts - v2),
                      (g_vol_cmp_bytes - v3) / 1048576.0, (g_vol_cmp_ns - v4) / 1e6 / 120.0, (unsigned long long)(g_vol_up_runs - v5), (g_vol_up_bytes - v6) / 1048576.0,
                      (calls - c0) / sec, (ns - ns0) / 1e6 / sec, (faults - f0) / sec, (unsigned long long)g_vol_decays, (unsigned long long)g_pc_pre_all_n);
                  { static uint64_t s0, a0, n0, fn0, h0[4], fg0, fc0; uint64_t sq, ah, fn, fns; nvn_dirty_fa_stats(&sq, &ah, &fn, &fns);   // combined1120
                    const uint64_t df = fn - n0; const uint64_t dc = calls - c0;
                    VKLOG("combined1120, last 120 submissions: faults %llu, of which the page before had faulted in the same sync %llu (%.0f%%); pages released ahead %llu; %.1f us a fault, %.1f us a VirtualProtect; "
                          "dirty runs at sync 1 page %llu, 2-3 %llu, 4-15 %llu, 16+ %llu; faulted pages GPU-written %llu, CPU-only %llu (%.0f%% CPU-only)",
                        (unsigned long long)df, (unsigned long long)(sq - s0), df ? 100.0 * (double)(sq - s0) / (double)df : 0.0, (unsigned long long)(ah - a0), df ? (double)(fns - fn0) / 1e3 / (double)df : 0.0,
                        dc ? (double)(ns - ns0) / 1e3 / (double)dc : 0.0, (unsigned long long)(g_run_hist[0] - h0[0]), (unsigned long long)(g_run_hist[1] - h0[1]), (unsigned long long)(g_run_hist[2] - h0[2]),
                        (unsigned long long)(g_run_hist[3] - h0[3]), (unsigned long long)(g_fpage_gpu - fg0), (unsigned long long)(g_fpage_cpu - fc0),
                        (g_fpage_gpu - fg0 + g_fpage_cpu - fc0) ? 100.0 * (double)(g_fpage_cpu - fc0) / (double)(g_fpage_gpu - fg0 + g_fpage_cpu - fc0) : 0.0);
                    s0 = sq; a0 = ah; n0 = fn; fn0 = fns; for (int k = 0; k < 4; k++) h0[k] = g_run_hist[k]; fg0 = g_fpage_gpu; fc0 = g_fpage_cpu; }
                  t0 = t; c0 = calls; ns0 = ns; f0 = faults; v0 = g_vol_entries; v1 = g_vol_exits; v2 = g_vol_evicts; v3 = g_vol_cmp_bytes; v4 = g_vol_cmp_ns; v5 = g_vol_up_runs; v6 = g_vol_up_bytes; } }
            static uint64_t r0, r1, r2, lastc_pc; VKLOG("poolcopy sync (stage2a2, 4 KB pages; clean groups skipped), last 120 submissions: %llu page generations read, %llu dirty (%.1f MB), %llu runs protected and copied (%.1f MB compared)",
                (unsigned long long)(g_pc_pages_seen - r0), (unsigned long long)(g_pc_pages_dirty - r1), (g_pc_pages_dirty - r1) * 4096.0 / 1048576.0, (unsigned long long)(g_pc_runs - r2), (g_sync_cmp - lastc_pc) / 1048576.0);
            r0 = g_pc_pages_seen; r1 = g_pc_pages_dirty; r2 = g_pc_runs; lastc_pc = g_sync_cmp; } }
}
// Copied pools after a submission: what the GPU wrote, back into the guest's pages. A pool never synced yet is left alone (its copy holds nothing of the game's).
// A CPU-uncached pool (vk_pool_set_flags) takes back only the ranges the submission's GPU writes covered (sub_gpu_wrote):
// copying 860 MB back after every submission held Unicorn Overlord to a few frames a second
struct vksub; static void pools_copy_back(const struct vksub *s);

// ---------------- submission slots ----------------
// slot12 (OoT 11-21, replaydiag: 6 submissions a frame, so with 6 slots submission k of a frame waited for submission k of the frame before to complete: 4.5-6.6 ms
// a frame of the replay thread waiting while the GPU was busy only 19-22 ms of a 27-31 ms frame). The arrays are sized for 12; DM_VK_NSLOT=<1..12> uses fewer
// (6: as before). Only the slots in use are taken, so their per-slot rings (hy3 staging, upload buffers) grow only when used
#define NSLOT 12
// rptfix: 6 in use by default again (11-21 10:2x: with 12 the GPU results reached the game a frame later and OoT flashed white; at 6 no flash, no slower: the slot wait
// was the completion thread, which rptfix lightens). DM_VK_NSLOT=12 to test
static int g_nslot = -1;
static int nslot_on(void) { if (g_nslot < 0) { const char *e = getenv("DM_VK_NSLOT"); int n = e && *e ? atoi(e) : 6; g_nslot = n < 1 ? 1 : n > NSLOT ? NSLOT : n;
    VKLOG("slot12: %d submission slots in use (DM_VK_NSLOT=<1..%d>; 6 as before)", g_nslot, NSLOT); } return g_nslot; }
int vk_nslot_in_use(void) { return nslot_on(); }
#define MAIN_CBS 32   // (allocated as used: DM_VK_PASS_SPLIT cuts inside passes as well)
// GPU time: a timestamp at the start and end of each submission's main command buffer (one queue: their sum is the time the GPU was busy)
static VkQueryPool g_tsq; uint64_t g_gpu_ns;
// Timestamp counters (ReportCounter TIMESTAMP): each writes a GPU timestamp where it is in the command stream (CQ per slot), and at completion it gets the guest
// clock now less how long before the submission's end the GPU passed it. They used to all get the completion time of their submission: a game timing its GPU
// work between two counters read 0 when both were in one submission and the gap between completions (our CPU replay included) when not. Xenoblade Chronicles 3
// sizes its frame by that time (dynamic resolution): its main pass went between 960x544 and 776x440 every few frames while the camera turned; with GPU times
// the size holds steady
enum { CQ = 64 };
static VkQueryPool g_cq;
// r50: NVN SAMPLES_PASSED counters as Vulkan occlusion queries (was a fixed 1 << 20: DK's skin SSS read 64x visibility, skin glowed red). After a reset every
// pass with draws gets one query, begun at its first draw and ended at the pass end, a report or a reset; the completion thread sums them per reset (epoch)
enum { OQ = 4096 };   // r50b: 512 overflowed once in DK (a submission with more counted passes)
static VkQueryPool g_oq; static VkQueryControlFlags g_oq_flags;
static int g_oq_counting; static uint32_t g_oq_epoch, g_oq_lost_ep = ~0u;   // replay thread
static _Thread_local int g_oq_active = -1;   // S1e (parallel recording): the query open in this thread's pass (a worker's own; begun and ended within one pass, so within one submission)
// occ-s0 / occ-s1 (design\OCCLUSION_REDESIGN.md steps 0 and 1; the code after sub_counter). g_occdiag: DM_VK_OCCDIAG=1 (live, gfx_vk_exec.c x_read), log only. g_occ2: DM_VK_OCC2=s1
// (device init): one occlusion query per draw by stream serial in a ring; the old path's values are written unchanged and its sums come from those queries (see post_op)
int g_occdiag, g_occ2;
enum { OQ_S1SPAN = 0x7fffffff };   // occ-s1: g_oq_active while the old path's span is open on the ring (no g_oq query)
static void occ_sub_begin(void); static void occ2_span_end(void); static void occd_ovf(void);
static _Thread_local uint64_t g_occ2_slo, g_occ2_shi;   // occ-s1: the old path's open span as draw serials [slo, shi) (per recording thread, as g_oq_active)
static _Thread_local int g_occ2_act = -1; static _Thread_local VkCommandBuffer g_occ2_cb;   // occ-s1: the ring slot of the draw being recorded (-1 none), the buffer it was begun in
// occ-s2 / occ-s3 (design\OCCLUSION_REDESIGN.md steps 2 and 3; the section "occ-s2 / occ-s3" after the occ-s1 ring): g_occ2 = 2 (DM_VK_OCC2=s2: the reports resolved on the GPU in
// stream order and written into the guest's words by the GPU only), 3 (s3: and render conditions on samples-passed reports evaluated by the GPU, VK_EXT_conditional_rendering)
static void occ2_report(uint64_t addr); static void occ2_reset(void); static void occ2_bound(void); static int occ2_counting(void); static void occ2_view(vk_par_view *v);
static void occ2_resolve(int why); static void occ2_dropped(void); enum { O2R_PASS, O2R_PART, O2R_SUB, O2R_FORCED, O2R_JOIN, O2R_READ, O2R_N };   // occ-s2: resolve points (a pass end after reports, a part's end, the submission's end, forced by a condition, a part's join; occ-s3fast: a pass end or record boundary the plan names, a copy reading a pending word)
static void occ2_opp(int why); static int occ2_unres_in(uint64_t a, uint64_t n); static void par_out_moved(int at, int d);   // occ-s3fast: a resolve opportunity (no pass open): taken when the plan or a bound says so; a range holding a report word not resolved yet
enum { RQ_HALF = 0x40000000u };   // occ-s2: a queued replay-time report that writes its timestamp half only (bytes 8..15; the counter word is written by the GPU)
enum { POST_HALF = 9 };           // occ-s2: kind 2 mode 9 = such a report left to the completion thread (key: its timestamp), bytes 8..15 only
#define SCRATCH_BLOCK (8u << 20)
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceSize size, used; } scratch_blk;
// R3 (parallel recording): what recording allocates per draw/dispatch and a slot rewinds when the GPU has passed it (slot_recycle): scratch blocks (vk_scratch,
// uniform shadows, driver segments) and descriptor pools (vk_alloc_set). Were fields of struct vksub; each slot has one (ar), the replay thread's. A thread's
// allocations go to its recording target's arena (g_vk_rt->ar, gfx_vk.h), NULL meaning the current slot's: so far every target's is NULL
typedef struct vk_arena {
    scratch_blk *blk; int nblk, capblk;
    uint64_t scr_peak; uint32_t scr_age;   // NVN2 scratch_trim: the most this arena used in its recent recycles, recycles since it was set
    VkDescriptorPool *dp; int ndp, capdp, dp_cur;
    uint32_t dp_used[9], dp_sets; int dp_counted;   // what pool dp_cur has given out (vk_alloc_set)
} vk_arena;
typedef struct { int kind; void *a; void *b; } dfree_t;   // 0 buffer+memory, 1 image+memory, 2 view
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; int q; uint32_t wseq; uint64_t key, kf, key2;
                 uint64_t s1lo, s1hi; uint32_t pf; uint8_t s1, od; } post_op;   // occ-s1: a samples-passed report's window [s1lo, s1hi) of draw serials (s1 set), compared at completion; occ-s0: od = 1, pf the present at replay (landing age)
// occ-s1: kind 4 mode 2 = a pass span of the old path on the per-draw ring: draw serials [addr, key) (DM_VK_OCC2=s1: only one occlusion query may be active in a command buffer,
// so the old span queries are not recorded; the old path's epoch sums come from the per-draw queries of its spans)   // occkey: kind 2 mode 7 = a keyed occlusion count for the table only (key; kf its present), mode 8 = write key as the counter word   // wseq: GPU writes recorded before it (combined1120)   // q: a timestamp counter's query in g_cq, or -1
// S1e (parallel recording): kind 5 = a worker segment's place among the post ops (v: its view's token), replaced by the segment's own post ops when its part is joined
// (vk_par_merge_posts); the completion thread passes over one (never left: every part is joined before it is submitted)
enum { POST_SEG = 5 };
// S1e (parallel recording): a worker segment's frozen view (gfx_vk.h vk_par_view): taken by the replay thread at the hand-out (vk_par_handout), read by the worker that records
// the segment (tl_wv) where it would have read the replay thread's live lists: the open submission's GPU-write list and write sequence (vk_sub_gw_overlaps, vk_sub_wver: the
// replay thread's serial records after the segment append to them), the GPU writes in flight (vk_gpu_pending: the ring is compacted by later writes), the landed uniform updates
// in flight (vk_ubd_read: g_infl is appended and pruned at later lands), the occlusion counting state (a later reset moves the epoch). Inside a segment none of them can change
// in stream order (no GPU writer, no land, no samples-passed reset), so the values at the hand-out are those the serial replay would have read. The worker's post ops (its
// occlusion queries' kind 4) go into the view and are put back in the segment's place
typedef struct { uint64_t lo, hi, fence, seq; } par_gw_t;
typedef struct { uint64_t lo, hi, fence; const uint8_t *data; } par_infl_t;
struct vk_par_view {
    struct { uint64_t lo, hi; } gw[64]; int ngw, ngml; uint32_t wseq; uint64_t nsubs;   // vksub_t's gw / ngw / ngml / wseq, g_nsubs
    par_gw_t *pg; int npg, cappg;       // g_gw's entries not known complete at the hand-out
    par_infl_t *inf; int ninf, capinf;  // g_infl's entries not known complete at the hand-out
    int oq_counting; uint32_t oq_epoch;
    post_op *post; int npost, cappost;  // the worker's post ops (emptied at vk_par_wk_begin)
    uint32_t token; int post_at;        // the placeholder's token, its index at the hand-out (later insertions only move it on)
    uint64_t *o2b; uint64_t o2lo; uint32_t o2n, o2cap;   // occ-s2: the segment's draw serials [o2lo, o2lo + o2n): a bit per query the worker began (merged into the ring's at the join, if its chunk is submitted)
};
static void par_post(struct vk_par_view *v, post_op o) {
    o.wseq = v->wseq;
    if (v->npost == v->cappost) { int c = v->cappost ? 2 * v->cappost : 64; void *p = realloc(v->post, (size_t)c * sizeof *v->post); if (!p) die("[vk] S1e: no memory for a segment's post ops"); v->post = p; v->cappost = c; }
    v->post[v->npost++] = o;
}
struct vksub {
    uint64_t content_sub; // F2 independent observation identity, copied into completion records
    VkCommandPool pool, up_pool; VkCommandBuffer up, main; int up_used;   // uppool: up_pool holds the upload buffers (ups), touched only under vk_up_lock
    nvf0_submission f0;   // observation; never used for replay or lifetime decisions
    VkCommandBuffer mains[MAIN_CBS]; int nmain_alloc, nmain_used, cb_err;
    VkCommandBuffer ups[MAIN_CBS]; int nup_alloc, nup_used; int nmain_sent;   // upload command buffers in order (one per part sent early), main buffers already submitted   // the submission's main command buffers in order (vk_cmd_split); main = the one being recorded
    uint64_t fence;   // the fence this slot last committed with (0 = never)
    int queries_initialized; // unused queries stay reset across reuse of this slot
    uint32_t pass_queries_written;
    int ncq;          // timestamp counter queries used in this submission (g_cq)
    int noq;          // occlusion queries used in this submission (g_oq)
    VkBuffer hy3_b; VkDeviceMemory hy3_m; VkDeviceSize hy3_cap, hy3_used;   // stage1 (hybrid3): device-local staging of atomic storage buffers, rewound when the slot is recycled
    VkBuffer atomic_scratch;
    VkDeviceMemory atomic_memory;
    int atomic_used;
    vk_arena ar;   // R3 (parallel recording): the slot's scratch blocks and descriptor pools (were blk, nblk, capblk, scr_peak, scr_age, dp, ndp, capdp, dp_cur, dp_used, dp_sets, dp_counted here)
    int no_ts;
    dfree_t *df; int ndf, capdf;
    post_op *post; int npost, cappost;
    struct { uint64_t lo, hi; } gw[64]; int ngw;
    int pc_ok, pc_unk; uint64_t pc_nsub;   // poolcopy: its readback ran (the submission was not refused), a write of unknown size was in it, its number (g_nsubs)
    // stage2c (async): this submission's own staging blocks (CPU patches), pre/post blocks, and its write ranges (pce: chunk, range, its pre and post copies, the part
    // whose upload buffer took the pre). Owned until its merge has run (slot_recycle waits for g_done), never shared with another submission (review: rb_sub)
    struct pcblks *pc_st, *pc_rb; struct pce *pce; int npce, cappce, pc_all; VkCommandBuffer pc_rec;
    struct { uint64_t lo, hi; uint32_t seq; } *wr; int nwr, capwr, wr_over; uint32_t wseq;   // combined1120: this submission's GPU writes in command order (counter/event reports land before later ones)
    vk_chunk *chunk; int nchunk, capchunk, chunk0;   // S1a (parallel recording): the submission's chunks in order (vk_chunk_add), the first of the open part; emptied in slot_recycle
    VkCommandBuffer *rcb; int nrcb_alloc, nrcb_used, caprcb;   // S1e (parallel recording): the replay thread's R chunks begun after a hand-out (from pool, allocated as used, taken from the first each submission)
    uint64_t *gmlk; int capgmlk;   // hybrid1: per write log, the shader's key with bit 0 = it runs an atomic (gml_copy_back: the pages it lists convert)
    const uint32_t **gml; int ngml, capgml;   // raw global memory: the write logs of this submission's draws and dispatches (vk_gml_note), read by pools_copy_back   // GPU writes into pool memory in this submission (sub_gpu_wrote), for pools_copy_back: nearby ones merged, and once full each joins the nearest
};
enum { GW_SUB = 64 };
static vksub_t g_slot[NSLOT]; static int g_cur = -1;   // R2 (parallel recording): g_cmd_gen (main command buffers begun, gfx_vk_exec.c bind cache) is now the recording target's gen
// R2 (parallel recording): the replay thread's recording target (gfx_vk.h): cb is the current slot's main command buffer (NULL while g_cur is -1, so vk_cmd still
// dies outside a submission), gen counts its begins as g_cmd_gen did. Every thread's g_vk_rt points at it until a worker sets its own
static vk_rec_target g_vk_rt_main;
_Thread_local vk_rec_target *g_vk_rt = &g_vk_rt_main;
// S1a (parallel recording): the recording workers (gfx_vk.h; design\PARALLEL_REPLAY_DESIGN.md Stage 1), made in slots_init when DM_VK_PARALLEL=<n> > 0 (none by default).
// Per submission slot a worker has a command pool of its own (a pool is externally synchronized: only that worker records from it, only slot_recycle resets it, once the
// slot's fence has passed and the worker is not recording into that slot), its primaries (allocated as used, reused after the reset) and an arena (scratch blocks,
// descriptor pools) rewound with the slot's own. Jobs run in order from a small queue (mutex + condition variable)
enum { PAR_MAX = 6, PAR_Q = 64 };
typedef struct { VkCommandPool pool; VkCommandBuffer *cb; int ncb, capcb, used; vk_arena ar; } vk_par_slot;
typedef struct { void (*fn)(void *); void *arg; } vk_par_job;
typedef struct vk_worker {
    int id; pthread_t th; void *rec, *rec_e; vk_rec_target rt;   // its recording contexts (gfx_vk_exec.c) and target (g_vk_rt in its thread)
    vk_par_slot slot[NSLOT];
    pthread_mutex_t mu; pthread_cond_t cv; vk_par_job q[PAR_Q]; unsigned qh, qt; int busy;   // qh: the next job to run, qt: the next free; busy: a job is running
} vk_worker;
static vk_worker *g_par; static int g_npar, g_par_selftest;
static _Thread_local vk_worker *tl_worker;   // the worker this thread is (NULL: the replay thread or any other)
// S1e (parallel recording): g_par[g_npar] is the helper context: no thread; the replay thread records as it when it runs a job itself (a queued one at a part's join,
// or a failed segment again). g_npar_ctx counts it (slot_recycle, vk_slot_stats, the fold): g_npar + 1 with workers, 0 without
static int g_npar_ctx, g_par_pin = 1;
struct vk_par_view;
static _Thread_local struct vk_par_view *tl_wv;   // S1e: the segment a worker records (its frozen view and post ops); NULL on the replay thread
static pthread_mutex_t g_chunk_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_chunk_cv = PTHREAD_COND_INITIALIZER;   // the slots' chunk lists (grown by the replay thread, marked by workers)
static pthread_mutex_t g_ar_mu = PTHREAD_MUTEX_INITIALIZER;   // an arena's block list growing (vk_scratch) against a worker's vk_buf_size reading the slot's own arena
static void par_start(void);   // (below, after vk_up_lock)
static void pool_sync_host_barrier(const vksub_t *s, VkCommandBuffer cb) {
    if (!(pool_sync_mode() & 1)) return;
    int unknown = s->ngml || g_sbufu_w != g_pool_unknown_write_start || g_gmem_sub == g_nsubs;
    if (!unknown && !s->ngw) return;
    VkBufferMemoryBarrier barriers[64]; uint32_t count = 0;
    pool_rdlock();
    for (int i = 0; i < g_npool; i++) {
        const vkpool_t *p = &g_pool[i]; if (!p->shadow) continue;
        int written = unknown;
        for (int k = 0; !written && k < s->ngw; k++) written = s->gw[k].lo < p->hi && p->lo < s->gw[k].hi;
        if (!written) continue;
        barriers[count++] = (VkBufferMemoryBarrier){VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL,
            VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            p->buf, 0, VK_WHOLE_SIZE};
        if (count == 64) {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, count, barriers, 0, NULL);
            count = 0;
        }
    }
    if (count) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, count, barriers, 0, NULL);
    pool_unlock();
}
// poolcopy: at the submission's end (its last main command buffer), what its GPU work may have written into the device-local copies, copied into their readback
// buffers and made visible to the host (pools_copy_back merges it at completion). The ranges: the submission's GPU write list (sub_gpu_wrote). A whole chunk
// where where a write lands is not known before it runs (review P1-2): raw global memory bound in this submission (any chunk: its pool table reaches them all
// with no lookup) or a storage buffer named in uniform memory (the chunks used in this submission). Recorded for both Switch 1 and 2 (review R9 step 3)
// ---- stage2c (async): per-submission staging, pre/post copies and merge (see pc_sync) ----
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; uint64_t size, used; int noncoh; } pcblk_t;
struct pcblks { pcblk_t *v; int n, cap; };
struct pce { int pi, part, drop, noncoh; uint64_t serial, off, n, boff; VkBuffer b; VkDeviceMemory m; uint8_t *pre; };   // post = pre + PCE_AL(n)
#define PCE_AL(n) (((n) + 63) & ~63ull)
static void pc_range_flush(VkDeviceMemory m, uint64_t off, uint64_t n, int invalidate) {
    VkDeviceSize a = V.props.limits.nonCoherentAtomSize ? V.props.limits.nonCoherentAtomSize : 1, lo = off / a * a, hi = (off + n + a - 1) / a * a;
    VkMappedMemoryRange r = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, m, lo, hi - lo};
    if (invalidate) vkInvalidateMappedMemoryRanges(V.dev, 1, &r); else vkFlushMappedMemoryRanges(V.dev, 1, &r);
}
// n bytes of one of the slot's block lists (rb: readback, else staging); 0 = no memory
static int pcb_get(struct pcblks **Lp, uint64_t n, int rb, pcblk_t **out, uint64_t *off) {
    if (!*Lp && !(*Lp = calloc(1, sizeof **Lp))) return 0;
    struct pcblks *L = *Lp; n = PCE_AL(n);
    for (int i = 0; i < L->n; i++) { pcblk_t *k = &L->v[i]; if (k->size - k->used >= n) { *out = k; *off = k->used; k->used += n; return 1; } }
    const uint64_t sz = n > (64ull << 20) ? n : (64ull << 20);
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz, rb ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; if (vkCreateBuffer(V.dev, &bi, NULL, &b)) return 0;
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
    const uint32_t t = pc_type(req.memoryTypeBits, 1); VkDeviceMemory m; uint8_t *p = NULL;
    if (pc_alloc(req.size, t, 0, &m, &p)) { vkDestroyBuffer(V.dev, b, NULL); return 0; }
    if (vkBindBufferMemory(V.dev, b, m, 0)) { vkDestroyBuffer(V.dev, b, NULL); vkFreeMemory(V.dev, m, NULL); return 0; }
    if (L->n == L->cap) { int c = L->cap ? 2 * L->cap : 8; pcblk_t *v = realloc(L->v, (size_t)c * sizeof *v); if (!v) { vkDestroyBuffer(V.dev, b, NULL); vkFreeMemory(V.dev, m, NULL); return 0; } L->v = v; L->cap = c; }
    L->v[L->n] = (pcblk_t){b, m, p, sz, n, pc_noncoh(t)}; *out = &L->v[L->n++]; *off = 0;
    { static int nl; if (nl++ < 8) VKLOG("stage2c: a submission slot made a %s block of %.0f MB (memory type %u%s)", rb ? "pre/post" : "staging", sz / 1048576.0, t, pc_noncoh(t) ? ", not coherent" : ""); }
    return 1;
}
static void pcb_reset(struct pcblks *L) { if (L) for (int i = 0; i < L->n; i++) L->v[i].used = 0; }
// One barrier per upload buffer (earlier submissions' work on the copies, the host's staging writes, before its copies), and transfer barriers between this
// buffer's uploads (staging -> copy) and pre copies (copy -> pre) over the same bytes, in whichever order they come
static uint64_t g_pc_cbgen = ~0ull; static int g_pc_up_since, g_pc_pre_since;
static VkCommandBuffer pc_up_cmd(void) {
    VkCommandBuffer cb = vk_upload_cmd();
    if (g_pc_cbgen != g_up_gen) { g_pc_cbgen = g_up_gen; g_pc_up_since = g_pc_pre_since = 0;
        VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
    return cb;
}
static void pc_tbar(VkCommandBuffer cb) {
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}
// [off, off + n) of chunk p, just captured into its shadow (= the game's bytes there): into this submission's staging, then a copy staging -> device-local copy
static int pc_stage_upload(vkpool_t *p, uint64_t off, uint64_t n) {
    vksub_t *s = cur(); pcblk_t *k; uint64_t so;
    if (!pcb_get(&s->pc_st, n, 0, &k, &so)) return 0;
    memcpy(k->p + so, p->shadow + off, (size_t)n);
    if (k->noncoh) pc_range_flush(k->m, so, n, 0);
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and its memos (g_pc_cbgen, g_pc_up_since, g_pc_pre_since)
    VkCommandBuffer cb = pc_up_cmd();
    if (g_pc_pre_since) { pc_tbar(cb); g_pc_pre_since = 0; }
    VkBufferCopy r = {so, off, n}; vkCmdCopyBuffer(cb, k->b, p->buf, 1, &r); g_pc_up_since = 1;
    vk_up_unlock();
    return 1;
}
// stage2c2: several runs of chunk p (dstOffset = offset in the chunk; their bytes already in its shadow) into one staging allocation and one copy command
static int pc_stage_regions(vkpool_t *p, const VkBufferCopy *rg, uint32_t nr, uint64_t total) {
    vksub_t *s = cur(); pcblk_t *k; uint64_t so;
    if (!nr || !pcb_get(&s->pc_st, total, 0, &k, &so)) return 0;
    VkBufferCopy r[256]; uint64_t o = so;
    for (uint32_t i = 0; i < nr; i++) { memcpy(k->p + o, p->shadow + rg[i].dstOffset, (size_t)rg[i].size); r[i] = (VkBufferCopy){o, rg[i].dstOffset, rg[i].size}; o += rg[i].size; }
    if (k->noncoh) pc_range_flush(k->m, so, total, 0);
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and its memos
    VkCommandBuffer cb = pc_up_cmd();
    if (g_pc_pre_since) { pc_tbar(cb); g_pc_pre_since = 0; }
    vkCmdCopyBuffer(cb, k->b, p->buf, nr, r); g_pc_up_since = 1;
    vk_up_unlock();
    g_pc_up_bytes += total; g_pc_up_n++;
    return 1;
}
// A pre copy of [off, off + n) of chunk g_pool[pi] for the open submission, in the upload buffer of the part being recorded (it runs after that part's uploads and
// before its GPU work). Skipped when an entry of this submission already holds the range (its pre was taken before any of this submission's work on it)
static void pc_pre_range(vksub_t *s, int pi, uint64_t off, uint64_t n) {
    vkpool_t *p = &g_pool[pi]; const uint64_t size = p->hi - p->lo; int mine = 0;
    if (p->pc_wfence < g_fence + 1) p->pc_wfence = g_fence + 1;
    if (pc_vol_on()) pv_gpu_mark(p, off, n);   // stage2b: never volatile (a volatile page goes back at the next sync)   // stage2c3: this chunk has GPU work in the open submission (its merge is fence g_fence + 1)
    for (int i = 0; i < s->npce; i++) { const struct pce *e = &s->pce[i]; if (e->pi != pi || e->serial != p->serial) continue; mine++;
        if (off >= e->off && off + n <= e->off + e->n) return; }
    if (mine >= 256 && n < size) { off = 0; n = size; }   // many ranges in one chunk: the whole chunk once (its pre now; earlier entries keep theirs)
    pcblk_t *k; uint64_t bo;
    if (!pcb_get(&s->pc_rb, 2 * PCE_AL(n), 1, &k, &bo)) { p->synced = 0; g_pc_drain = g_fence + 1; g_pc_lost++; return; }   // no memory: this chunk synced again in full after this submission
    if (s->npce == s->cappce) { int c = s->cappce ? 2 * s->cappce : 256; struct pce *v = realloc(s->pce, (size_t)c * sizeof *v); if (!v) { p->synced = 0; g_pc_drain = g_fence + 1; return; } s->pce = v; s->cappce = c; }
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and its memos (under the pool lock: leaf)
    VkCommandBuffer cb = pc_up_cmd();
    if (g_pc_up_since) { pc_tbar(cb); g_pc_up_since = 0; }
    VkBufferCopy r = {off, bo, n}; vkCmdCopyBuffer(cb, p->buf, k->b, 1, &r); g_pc_pre_since = 1;
    vk_up_unlock();
    s->pce[s->npce++] = (struct pce){pi, s->nmain_used, 0, k->noncoh, p->serial, off, n, bo, k->b, k->m, k->p + bo};
    g_pc_pre_bytes += n; g_pc_pre_n++;
}
// The open submission's GPU work may write [a, a + n) (sub_gpu_wrote): pre copies of the copied chunks it overlaps. g_pool_ord is sorted by lo; chunks may overlap
// (pools over the same pages), so a few before the first are looked at too
static void pc_pre_note(uint64_t a, uint64_t n) {
    if (!n || g_cur < 0 || !pc_async()) return;
    vksub_t *s = &g_slot[g_cur]; const uint64_t e = a + n < a ? ~0ull : a + n;
    pool_rdlock();
    int lo = 0, hi = g_npool; while (lo < hi) { int m = (lo + hi) / 2; if (g_pool[g_pool_ord[m]].lo <= a) lo = m + 1; else hi = m; }
    int k0 = lo - 16 > 0 ? lo - 16 : 0;
    for (int k = k0; k < g_npool; k++) { const int pi = g_pool_ord[k]; vkpool_t *p = &g_pool[pi];
        if (p->lo >= e) break;
        if (!p->dl || !p->synced || p->hi <= a) continue;
        const uint64_t l = a > p->lo ? a : p->lo, h = e < p->hi ? e : p->hi;
        pc_pre_range(s, pi, l - p->lo, h - l); }
    pool_unlock();
}
// Writes anywhere (raw global memory: the pool table reaches every chunk): a pre copy of every copied chunk, whole
static void pc_pre_all(void) {
    if (g_cur < 0 || !pc_async()) return;
    vksub_t *s = &g_slot[g_cur]; g_pc_pre_all_n++;   // stage2b2: counted (every chunk marked GPU-written)
    pool_rdlock();
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (p->dl && p->synced) pc_pre_range(s, i, 0, p->hi - p->lo); }
    pool_unlock();
}
// The post copies of entries [0, npce) whose part is <= upto, into cb (the submission's last main buffer, or the recovery buffer of a refused one)
static int pc_post_record(vksub_t *s, VkCommandBuffer cb, int upto) {
    int any = 0; pool_rdlock();
    for (int i = 0; i < s->npce; i++) { struct pce *e = &s->pce[i];
        if (e->drop) continue;
        if (e->part > upto) { e->drop = 1; g_pc_as_drop++; continue; }   // its pre never ran (a part not sent)
        const vkpool_t *p = &g_pool[e->pi];
        if (e->pi >= g_npool || p->serial != e->serial || !p->dl || !p->buf) { e->drop = 1; continue; }
        if (!any) { any = 1; VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
        VkBufferCopy r = {e->off, e->boff + PCE_AL(e->n), e->n}; vkCmdCopyBuffer(cb, p->buf, e->b, 1, &r); g_pc_post_bytes += e->n; }
    pool_unlock();
    if (any) { VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
    return any;
}
// Completion thread, submission order, under the pool read lock (the sync's capture holds the write lock): a byte with post != pre was changed by this submission's
// GPU work. The game's byte takes it when it still holds pre (else the CPU wrote it after the capture: the CPU's stays); the shadow takes it either way
static void pc_merge_async(const vksub_t *s) {
    for (int i = 0; i < s->npce; i++) { const struct pce *e = &s->pce[i];
        if (e->drop || e->pi >= g_npool) continue;
        vkpool_t *p = &g_pool[e->pi];
        if (p->serial != e->serial || !p->dl || !p->shadow) continue;   // (a refused submission's recovery merges into chunks it marked for a full sync)
        const uint64_t al = PCE_AL(e->n);
        if (e->noncoh) pc_range_flush(e->m, e->boff, 2 * al, 1);
        const uint8_t *P = e->pre, *R = e->pre + al; uint8_t *B = p->shadow + e->off; volatile uint8_t *G = (volatile uint8_t *)(uintptr_t)(p->lo + e->off);
        shadow_range(p, e->off, e->n, 1); int touched = 0;
        for (uint64_t x = 0; x < e->n;) {
            const uint64_t ga = p->lo + e->off + x, pe = (ga & ~4095ull) + 4096; uint64_t len = pe - ga; if (len > e->n - x) len = e->n - x;
            g_pc_rb_bytes += len;
            if (memcmp(P + x, R + x, (size_t)len)) { int prepared = 0; touched = 1;
#if defined(DM_NXN1X)
                nvnc_cpu(ga, len, NVNC_GPU_WRITEBACK);
#endif
                for (uint64_t j = x; j < x + len; j++) { if (P[j] == R[j]) continue;
                    if (G[j] == P[j]) { if (!prepared) { dm_guest_write_prepare((void *)(uintptr_t)ga, (size_t)len); prepared = 1; } G[j] = R[j]; g_pc_as_mg++; } else g_pc_as_cpu++;
                    B[j] = R[j]; } }
            x += len; }
        if (touched) shadow_range(p, e->off, e->n, 0);
    }
}
static void pc_readback(vksub_t *s, VkCommandBuffer cb) {
    s->pc_ok = 0; s->pc_unk = 0; s->pc_nsub = g_nsubs;
    if (!vk_poolcopy_on()) return;
    if (pc_async()) { pc_post_record(s, cb, MAIN_CBS + 1); s->pc_ok = 1; return; }   // stage2c: every part is submitted with this one
    const int raw = s->ngml || g_gmem_sub == g_nsubs, unk = g_sbufu_w != g_pool_unknown_write_start;
    s->pc_unk = (unk ? 1 : 0) | (raw ? 2 : 0);
    int opened = 0; VkBufferCopy rg[GW_SUB + 1];
    pool_rdlock();
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i];
        if (!p->dl || !p->synced) continue;
        p->rb_sub = 0; p->rb_whole = 0;
        const uint64_t size = p->hi - p->lo; const int whole = raw || (unk && p->used == g_nsubs);
        uint32_t nr = 0;
        if (whole) rg[nr++] = (VkBufferCopy){0, 0, size};
        else for (int k = 0; k < s->ngw; k++) { uint64_t lo = s->gw[k].lo > p->lo ? s->gw[k].lo : p->lo, hi = s->gw[k].hi < p->hi ? s->gw[k].hi : p->hi;
            if (lo < hi) rg[nr++] = (VkBufferCopy){lo - p->lo, lo - p->lo, hi - lo}; }
        if (!nr) continue;
        if (!opened) { opened = 1;   // every earlier write (shaders, copies, fills, transform feedback) before the copies read the device-local memory
            VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
        vkCmdCopyBuffer(cb, p->buf, p->rbuf, nr, rg);
        p->rb_sub = g_nsubs; p->rb_whole = whole; if (whole) g_pc_rb_whole++;
    }
    pool_unlock();
    if (opened) { VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};   // the readback to the host
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, NULL, 0, NULL); }
    s->pc_ok = 1;
}
// Windows: every pool takes back only the ranges its submission's GPU writes covered (sub_gpu_wrote: copies, fills, texture → buffer copies, transform feedback,
// storage buffers of dispatches, storage buffers and storage texel buffers a draw's shader writes), and of those only the 4 KB pages that differ from the guest's:
// the guest's pages are write-protected by the dirty tracking, so a page written back costs a fault and is copied again at the next sync. Copying every
// whole pool back after each submission held DK (about 1.2 GB of copied pools: Turnip on Windows has no VK_EXT_external_memory_host) to 7-8 s a frame with
// 1 ms of GPU work. DM_VK_COPYBACK_FULL=1: the whole pool as before (CPU-uncached pools still by range)
// Raw global memory's stores (vk_gmem_map) land in the copied pools' GPU copies at addresses only the GPU knew: each shader that makes them also puts every guest page it
// wrote in a write log (dmsh_abi.h DMSH_GMEM_LOG_*; Ryujinx GlobalToStorage.TryGenerateRawStorageOp, nvn_mtl.m gml_read reads the same logs), and those pages come back
// like the other GPU writes. Not copied back, the guest kept its old bytes there, and the next sync of a block the CPU wrote put them over the GPU's (DK, 11-04: holes in
// the cave's walls). A log that ran out of slots is counted (its other pages are not known). Under g_pool_mu's read lock; the completion thread alone
static int u64_cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
// s25: a page the copy-back writes into the game's memory is made writable and stamped first (dm_guest_write_prepare, as every host write), instead of
// being written while nvn_dirty may hold it read-only. Since s23 its protection really applies (the game's view is no longer an imported, locked range), and
// the completion thread's write faulted: run 30 (Xenoblade 3, 11-12) crashed in this memcpy, the fault not taken by NXN1X's handler
void dm_guest_write_prepare(void *addr, size_t len);
static void gml_copy_back(const struct vksub *s) {
    enum { SLOTS = 1 << DMSH_GMEM_LOG_LOG2, PG = 4096 };
    static uint64_t *pg; static size_t cap; size_t n = 0;
    for (int i = 0; i < s->ngml; i++) { const uint32_t *w = s->gml[i];   // hybrid1: an atomic shader's log converts the units of the imported chunks it lists, unless it ran out of slots (rule H11)
        if (s->gmlk && (s->gmlk[i] & 1)) { if (w[0]) g_hy_raw_full++;
            else for (int k = 1; k <= SLOTS; k++) if (w[k]) { const uint64_t a = (uint64_t)w[k] << 12; vk_hy_request(a, a + PG, s->gmlk[i] & ~1ull, 1); } } }
    for (int i = 0; i < s->ngml; i++) { const uint32_t *w = s->gml[i]; g_gml_logs++; g_gml_full += w[0] != 0;
        for (int k = 1; k <= SLOTS; k++) if (w[k]) { if (n == cap) { cap = cap ? 2 * cap : 4096; pg = realloc(pg, cap * sizeof *pg); } pg[n++] = w[k]; } }
    if (!n) return;
    qsort(pg, n, sizeof *pg, u64_cmp);
    for (size_t i = 0; i < n; i++) { if (i && pg[i] == pg[i - 1]) continue;
        uint64_t a = pg[i] << 12; const vkpool_t *p = NULL;
        int lo = 0, hi = g_npool;   // the pool by lo that starts at or below a (vk_pool_of's search, under the lock held here)
        while (lo < hi) { int m = (lo + hi) / 2; if (g_pool[g_pool_ord[m]].lo <= a) lo = m + 1; else hi = m; }
        if (lo > 0 && a + PG <= g_pool[g_pool_ord[lo - 1]].hi) p = &g_pool[g_pool_ord[lo - 1]];
        if (!p || !p->shadow || !p->synced || p->dl) continue;   // (poolcopy chunks: pools_copy_back)
        g_gml_pages++;
        shadow_range((vkpool_t *)p, a - p->lo, PG, 1);
        if (p->span && nvn_dirty_since_span(p->span, (const void *)(uintptr_t)a, PG, p->stamp)) { if (!snap_merge(p, a, a + PG)) g_gml_skip++; continue; }   // as pools_copy_back's pages
        const uint8_t *src = p->shadow + (a - p->lo);
        if (memcmp((const void *)(uintptr_t)a, src, PG)) { dm_guest_write_prepare((void *)(uintptr_t)a, PG); memcpy((void *)(uintptr_t)a, src, PG); g_gml_back++; g_back_bytes += PG; } }
}
static void pools_copy_back(const struct vksub *s) {
    static int full = -1; if (full < 0) full = getenv("DM_VK_COPYBACK_FULL") != NULL;
    enum { PG = 4096 };
    pool_rdlock();
    // poolcopy: a raw global memory write log that ran out of slots: its other pages are not known, so chunks read back whole are merged whole
    int gml_sat = 0; for (int i = 0; i < s->ngml; i++) if (s->gml[i][0]) gml_sat = 1;
    const int async = pc_async(); if (async && s->pc_ok) pc_merge_async(s);   // stage2c: this submission's own pre/post copies
    for (int i = 0; i < g_npool; i++) { vkpool_t *p = &g_pool[i]; if (!p->shadow || !p->synced) continue;
        if (p->dl && async) continue;
        if (p->dl) {   // poolcopy: the readback of this submission merged byte by byte (pc_merge)
            if (!s->pc_ok || p->rb_sub != s->pc_nsub) continue;
            if (p->rb_whole && ((s->pc_unk & 1) || gml_sat || !s->ngml)) { pc_merge(p, p->lo, p->hi); continue; }   // (raw memory with complete write logs: its listed pages below)
            for (int k = 0; k < s->ngw; k++) pc_merge(p, s->gw[k].lo, s->gw[k].hi);
            for (int g = 0; g < s->ngml && p->rb_whole; g++) { const uint32_t *w = s->gml[g];
                for (int k = 1; k <= (1 << DMSH_GMEM_LOG_LOG2); k++) if (w[k]) { uint64_t a = (uint64_t)w[k] << 12; if (a >= p->lo && a < p->hi) pc_merge(p, a, a + 4096); } }
            continue; }
        if (full && !(p->flags & 0x2)) { shadow_range(p, 0, p->hi - p->lo, 1); dm_guest_write_prepare((void *)(uintptr_t)p->lo, p->hi - p->lo); memcpy((void *)(uintptr_t)p->lo, p->shadow, p->hi - p->lo); g_back_bytes += p->hi - p->lo; continue; }
        for (int k = 0; k < s->ngw; k++) { uint64_t lo = s->gw[k].lo > p->lo ? s->gw[k].lo : p->lo, hi = s->gw[k].hi < p->hi ? s->gw[k].hi : p->hi;
            if (lo >= hi) continue;
            shadow_range(p, lo - p->lo, hi - lo, 1); g_back_cmp += hi - lo;
            for (uint64_t a = lo; a < hi; ) { uint64_t e = (a & ~(uint64_t)(PG - 1)) + PG; if (e > hi) e = hi;
                const uint8_t *src = p->shadow + (a - p->lo);
                // A page the CPU wrote after this submission's sync holds newer bytes than the copy: left as it is. This runs on the completion thread while the game goes on;
                // copying the page back put old bytes over the game's new ones (DK, 11-04: command records written after the sync came back as old records, floats and zeros,
                // and the replay CALLed garbage). The next sync (vk_pools_sync waits for this copy-back) is the latest one, so p->stamp is this submission's
                if (p->span && nvn_dirty_since_span(p->span, (const void *)(uintptr_t)a, e - a, p->stamp)) { if (!snap_merge(p, a, e)) g_back_skip += e - a; a = e; continue; }
                if (memcmp((const void *)(uintptr_t)a, src, e - a)) { dm_guest_write_prepare((void *)(uintptr_t)a, e - a); memcpy((void *)(uintptr_t)a, src, e - a); g_back_bytes += e - a; }
                a = e; } } }
    if (!full) gml_copy_back(s);
    pool_unlock();
}
static VkCommandBuffer g_slot_main_cb(void) { return g_vk_rt->cb; }   // R2 (parallel recording): the thread's recording target (was g_slot[g_cur].main, the same buffer on the replay thread)
// DM_VK_PASS_TIMES=1: a timestamp before and after every render pass (the replay's per-frame line then lists the passes that cost the most GPU time)
enum { PASSQ = 512 };   // passes (and, 11-04 r22, dispatches) timed per submission; 128 ran out in DK gameplay
static VkQueryPool g_pq;
static int g_pass_times_requested;
int vk_pass_times_active(void) { return g_pq != VK_NULL_HANDLE; }
#define VK_TS_DISPATCH 0xC0FFEEu   // pass_meta.fmt of a timed dispatch (gfx_vk_exec.c dispatch): w, h = its group counts, draws = the low 32 bits of the shader key
#define VK_TS_OCC2 0x0CC2u         // occ-s3fast: pass_meta.fmt of a timed occlusion resolve point (occ2_resolve: its query copies, dispatch, guest copies and barriers)
static uint64_t g_o2_gpu_ns, g_o2_gpu_n;   // occ-s3fast: the completion thread (atomic adds): GPU time of the timed resolve points (DM_VK_PASS_TIMES=1), how many
typedef struct { uint32_t w, h, fmt, draws, fidx; } pass_meta;   // fidx: the pass's place in its frame (submissions are cut every DM_VK_CB_DRAWS draws)
enum { FPASSQ = 1024 }; static int g_fpass;   // passes of the frame so far
static uint64_t g_r138_nf;   // completed frames with NVN submissions
static void occd_present(void);   // occ-s0 (with the occlusion code)
static void o3f_present(void);    // occ-s3fast (diagnostic, log only): the frame time, spikes, the per-frame ring
void vk_pass_frame(void) { g_fpass = 0; if (g_occdiag) occd_present(); if (g_occ2 >= 2) { occ2_bound(); o3f_present(); } }   // a present: the next pass is the frame's first (occ-s0: its frame time; occ-s2: the landing bound)
static pass_meta g_pmeta[NSLOT][PASSQ]; static int g_npass[NSLOT];
static double g_pacc[FPASSQ]; static uint32_t g_pcnt[FPASSQ]; static pass_meta g_plast[FPASSQ]; static int g_pmax;
// r138 (diagnostic, DM_VK_PASS_TIMES): the timed passes grouped by what they are, not by their place in the frame (the frame's pass list changes, so one place mixed
// different passes): size, format and draw count 0 / 1 / 2-99 / 100-999 / 1000+; dispatches by shader key and group counts. Every report prints all groups with GPU ms
// and passes per frame, and the GPU time of the submissions outside the timed passes
typedef struct { uint32_t w, h, fmt, b, key; double ms; uint64_t n, dr; } r138_grp;
enum { R138N = 1024 };
static r138_grp g_r138[R138N]; static int g_r138_full; static uint64_t g_r138_gpu0, g_r138_nf0;
static void r138_add(const pass_meta *m, double ms) {
    int dsp = m->fmt == VK_TS_DISPATCH; uint32_t d = dsp ? 0 : m->draws & 0xffff, b = dsp ? 0 : d == 0 ? 0 : d == 1 ? 1 : d < 100 ? 2 : d < 1000 ? 3 : 4, key = dsp ? m->draws : 0;
    uint32_t hs = ((m->w * 73856093u) ^ (m->h * 19349663u) ^ (m->fmt * 83492791u) ^ (b * 2654435761u) ^ key) & (R138N - 1);
    for (int i = 0; i < R138N; i++, hs = (hs + 1) & (R138N - 1)) { r138_grp *g = &g_r138[hs];
        if (g->n && g->w == m->w && g->h == m->h && g->fmt == m->fmt && g->b == b && g->key == key) { g->ms += ms; g->n++; g->dr += d; return; }
        if (!g->n) { g->w = m->w; g->h = m->h; g->fmt = m->fmt; g->b = b; g->key = key; g->ms = ms; g->n = 1; g->dr = d; return; } }
    g_r138_full++;
}
static void r138_report(void) {
    uint64_t nf = g_r138_nf - g_r138_nf0; if (!nf) return;
    static int ix[R138N]; int n = 0; double tot = 0;
    for (int i = 0; i < R138N; i++) if (g_r138[i].n) { int j = n++; while (j > 0 && g_r138[ix[j - 1]].ms < g_r138[i].ms) { ix[j] = ix[j - 1]; j--; } ix[j] = i; tot += g_r138[i].ms; }
    double busy = (double)(g_gpu_ns - g_r138_gpu0) / 1e6 / (double)nf;
    VKLOG("r138: pass groups (last %llu completed frames, per frame): %d groups, %.2f ms in timed passes, GPU busy %.2f ms, outside timed passes %.2f ms%s", (unsigned long long)nf, n, tot / (double)nf, busy,
          busy - tot / (double)nf, g_r138_full ? " (group table full: some passes not counted)" : "");
    static const char *bn[5] = {"0", "1", "2-99", "100-999", "1000+"};
    for (int k = 0; k < n; k++) { const r138_grp *g = &g_r138[ix[k]];
        if (g->fmt == VK_TS_SCALE_WRITEBACK) VKLOG("r138:   %6.3f ms %6.2f a frame: 2X writeback %ux%u at %s", g->ms / (double)nf, (double)g->n / (double)nf, g->w, g->h, g->b ? "guest barrier" : "other pass end");
        else if (g->fmt == VK_TS_DISPATCH) VKLOG("r138:   %6.2f ms %6.2f a frame: dispatch ..%08x, %ux%u groups", g->ms / (double)nf, (double)g->n / (double)nf, g->key, g->w, g->h);
        else if (g->fmt == VK_TS_OCC2) VKLOG("r138:   %6.3f ms %6.2f a frame: occlusion resolve points (occ-s3fast)", g->ms / (double)nf, (double)g->n / (double)nf);
        else VKLOG("r138:   %6.2f ms %6.2f a frame: %ux%u fmt %u, draws %s (%.0f on average)", g->ms / (double)nf, (double)g->n / (double)nf, g->w, g->h, g->fmt, bn[g->b], (double)g->dr / (double)g->n); }
    memset(g_r138, 0, sizeof g_r138); g_r138_full = 0; g_r138_gpu0 = g_gpu_ns; g_r138_nf0 = g_r138_nf;
}   // by the pass's place in its frame (the completion thread adds)
static int pass_times_on(void) { return vk_pass_times_active(); }
// ootgap (diagnostic, NPARK 11-20, OoT: its submissions' GPU span about 180 ms a frame, the timed passes and dispatches in it about 11 ms): on while DM_VK_PASS_TIMES
// is (DM_VK_OOTGAP=0 off, read at each submission's start). Per gap between timed passes (before the first, between two, after the last): its GPU time from the pass
// timestamps (one's end, BOTTOM_OF_PIPE, to the next one's start, TOP_OF_PIPE), the GPU commands recorded into the main command buffer in it (by kind: the s63
// wrappers and vk_img_barrier_all) and the record ops replayed in it; the part boundaries in it (a timestamp at the end of a part, one at the start of the next:
// between them the GPU runs the next part's upload command buffer, waits for its submission, or runs other work on the queue); the upload command buffers (a
// timestamp at their begin and end). Lines "ootgap:" at every pass report (120 frames). Logging only: no command is reordered, no synchronization added
enum { OGQ = 4 * MAIN_CBS };   // timestamps per slot: part boundaries (end, begin), then upload command buffers (begin, end)
typedef struct { uint16_t n[OG_N]; uint64_t ops; } og_gap;   // ops: the record ops replayed in the gap (bit per op)
typedef struct { uint16_t gap, q; uint8_t pre; } og_mark;   // q: the pair's first query in the slot; pre: an upload command buffer submitted with the first part (it runs before the span)
typedef struct { int nb, nu, bopen, uopen, skipped; og_mark b[MAIN_CBS], u[MAIN_CBS]; } og_marks;
static VkQueryPool g_ogq; static int g_og_on, g_og_inpass;   // g_og_on: this submission is measured; g_og_inpass: a timed pass is open (both the replay thread's)
static og_gap g_og_gap[NSLOT][PASSQ + 1]; static og_marks g_og[NSLOT];
void vk_pass_ts(int end, uint32_t w, uint32_t h, uint32_t fmt, uint32_t draws) {
    if (!pass_times_on() || g_cur < 0) return; int *n = &g_npass[g_cur]; if (*n >= PASSQ) return;
    const uint32_t query = (uint32_t)(*n * 2 + end);
    vkCmdWriteTimestamp(g_slot_main_cb(), (end || fmt == VK_TS_SCALE_WRITEBACK) ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_pq, (uint32_t)g_cur * PASSQ * 2 + query);
    g_og_inpass = !end;   // ootgap
    g_slot[g_cur].pass_queries_written = query + 1;
    if (end) { g_pmeta[g_cur][*n] = (pass_meta){w, h, fmt, draws, (uint32_t)(g_fpass < FPASSQ - 1 ? g_fpass : FPASSQ - 1)}; (*n)++; g_fpass++; }
}
void vk_pass_report(void) {   // the most expensive passes since the last report, then the counters start again
    if (!pass_times_on()) return;
    og_report(g_r138_nf - g_r138_nf0);   // ootgap (before r138 starts its count again)
    r138_report();   // r138
    static int idx[FPASSQ]; int m = g_pmax; for (int i = 0; i < m; i++) idx[i] = i;
    for (int i = 0; i < m; i++) for (int j = i + 1; j < m; j++) if (g_pcnt[idx[j]] && (!g_pcnt[idx[i]] || g_pacc[idx[j]] / g_pcnt[idx[j]] > g_pacc[idx[i]] / g_pcnt[idx[i]])) { int t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
    double tot = 0, tcs = 0; int ncs = 0; for (int i = 0; i < m; i++) if (g_pcnt[i]) { tot += g_pacc[i] / g_pcnt[i]; if (g_plast[i].fmt == VK_TS_DISPATCH) { tcs += g_pacc[i] / g_pcnt[i]; ncs++; } }
    char b[1600]; int o = snprintf(b, sizeof b, "passes by GPU time (%d passes, %.2f ms in passes per frame; of them %d dispatches, %.2f ms):", m, tot, ncs, tcs);
    for (int i = 0; i < m && i < 12 && o < (int)sizeof b - 80; i++) { int k = idx[i]; if (!g_pcnt[k]) continue;
        if (g_plast[k].fmt == VK_TS_SCALE_WRITEBACK) {
            o += snprintf(b + o, sizeof b - (size_t)o, " #%d %.2f ms (2X writeback %ux%u);",
                          k, g_pacc[k] / g_pcnt[k], g_plast[k].w, g_plast[k].h);
            continue;
        }
        if (g_plast[k].fmt == VK_TS_DISPATCH) { o += snprintf(b + o, sizeof b - (size_t)o, " #%d %.2f ms (dispatch ..%08x, %ux%u groups);", k, g_pacc[k] / g_pcnt[k], g_plast[k].draws, g_plast[k].w, g_plast[k].h); continue; }
        o += snprintf(b + o, sizeof b - (size_t)o, " #%d %.2f ms (%ux%u fmt %u, %u draws over %u px wide);", k, g_pacc[k] / g_pcnt[k], g_plast[k].w, g_plast[k].h, g_plast[k].fmt, g_plast[k].draws & 0xffff, g_plast[k].draws >> 16); }
    { extern uint64_t g_nvn_discards[2]; static uint64_t pd0, pd1;
      uint64_t color = __atomic_load_n(&g_nvn_discards[0], __ATOMIC_RELAXED);
      uint64_t depth = __atomic_load_n(&g_nvn_discards[1], __ATOMIC_RELAXED);
      o += snprintf(b + o, sizeof b - (size_t)o, " discards since: colour %llu, depth %llu", (unsigned long long)(color - pd0), (unsigned long long)(depth - pd1));
      pd0 = color; pd1 = depth; }
    VKLOG("%s", b);
    memset(g_pacc, 0, sizeof g_pacc); memset(g_pcnt, 0, sizeof g_pcnt); g_pmax = 0;
}
// one submission at a time: the game's (hle_nvn.c holds its own lock around them) and the backend's own (texel readback, present)
static nvnr_gate g_sub_lock = NVNR_GATE_INITIALIZER;
static _Thread_local int g_sub_owned;
// ootgap: the recording side (the replay thread, which owns the open submission)
static og_gap *og_cur_gap(void) { int n = g_npass[g_cur]; return &g_og_gap[g_cur][n < PASSQ ? n : PASSQ]; }
static void og_note(VkCommandBuffer cb, int kind) {
    if (!g_sub_owned || !g_og_on || g_cur < 0 || g_og_inpass || cb != g_slot[g_cur].main) return;
    og_gap *g = og_cur_gap(); if (g->n[kind] < 0xffff) g->n[kind]++;
}
static void og_note_ts(VkCommandBuffer cb, VkQueryPool q) { if (q != g_pq && q != g_ogq && q != g_tsq) og_note(cb, OG_TS); }   // (not the diagnostics' own timestamps)
void vk_og_op(unsigned op) { if (g_sub_owned && g_og_on && g_cur >= 0 && !g_og_inpass && op < 64) og_cur_gap()->ops |= 1ull << op; }   // gfx_vk_exec.c exec_ops
static void og_ts(VkCommandBuffer cb, uint32_t q, int bottom) {
    (s63o_vkCmdWriteTimestamp ? s63o_vkCmdWriteTimestamp : vkCmdWriteTimestamp)(cb, bottom ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_ogq, (uint32_t)g_cur * OGQ + q);
}
static void og_reset2(VkCommandBuffer cb, uint32_t q) { (s63o_vkCmdResetQueryPool ? s63o_vkCmdResetQueryPool : vkCmdResetQueryPool)(cb, g_ogq, (uint32_t)g_cur * OGQ + q, 2); }   // outside render passes only
static void og_sub_begin(void) {   // mtl_sub_begin, the slot recycled
    const char *e = getenv("DM_VK_OOTGAP"); g_og_on = g_ogq && pass_times_on() && !(e && *e && atoi(e) == 0); g_og_inpass = 0;
    if (!g_og_on) return;
    memset(&g_og[g_cur], 0, sizeof g_og[g_cur]); memset(g_og_gap[g_cur], 0, sizeof g_og_gap[g_cur]);
}
static void og_part_end(vksub_t *s) {   // vk_cmd_split, right before the part's vkEndCommandBuffer
    og_marks *m = &g_og[g_cur]; if (!g_og_on) return; m->bopen = 0;
    if (g_og_inpass || g_npass[g_cur] >= PASSQ || m->nb >= MAIN_CBS) { m->skipped++; return; }   // (a cut inside a pass: no query reset there)
    uint32_t q = 2 * (uint32_t)m->nb; og_reset2(s->main, q); og_ts(s->main, q, 1); m->bopen = 1;
}
static void og_part_begin(vksub_t *s) {   // vk_cmd_split, the next part begun
    og_marks *m = &g_og[g_cur]; if (!g_og_on || !m->bopen) return; m->bopen = 0;
    uint32_t q = 2 * (uint32_t)m->nb; og_ts(s->main, q + 1, 0); m->b[m->nb++] = (og_mark){(uint16_t)g_npass[g_cur], (uint16_t)q, 0};
}
static void og_up_begin(vksub_t *s) {   // vk_upload_cmd, the upload command buffer begun
    og_marks *m = &g_og[g_cur]; m->uopen = 0; if (!g_og_on || m->nu >= MAIN_CBS) return;
    uint32_t q = 2 * MAIN_CBS + 2 * (uint32_t)m->nu; og_reset2(s->up, q); og_ts(s->up, q, 0); m->uopen = 1;
}
static void og_up_end(vksub_t *s) {   // right before the upload command buffer's vkEndCommandBuffer (it is submitted with the next main part)
    og_marks *m = &g_og[g_cur]; if (!g_og_on || !m->uopen) return; m->uopen = 0;
    uint32_t q = 2 * MAIN_CBS + 2 * (uint32_t)m->nu; og_ts(s->up, q + 1, 1); int n = g_npass[g_cur];
    m->u[m->nu++] = (og_mark){(uint16_t)(n < PASSQ ? n : PASSQ), (uint16_t)q, (uint8_t)(s->nmain_sent == 0)};
}
// ootgap: the completion side (the completion thread only), per submission
static struct { double span, pass, g0, gmid, gtail, between, up_in, up_pre, cls[5], kms[OG_N], fbms[4], trms[4]; uint64_t subs, nb, nskip, nu_in, nu_pre, ncls[5], kn[OG_N], fbn[4], fbs, locks0, nvn0, trn[4]; } g_oga;   // trms/trn (stage1): gaps between passes by transition: dispatch->dispatch, dispatch->pass, pass->dispatch, pass->pass
static double g_ogp_ms[FPASSQ]; static uint32_t g_ogp_n[FPASSQ]; static og_gap g_ogp_last[FPASSQ]; static pass_meta g_ogp_meta[FPASSQ];
static double og_pair(int slot, uint32_t q) {   // ms between the pair's two timestamps; -1: not available
    uint64_t t[2]; if (vkGetQueryPoolResults(V.dev, g_ogq, (uint32_t)slot * OGQ + q, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return -1;
    return t[1] > t[0] ? (double)(t[1] - t[0]) * V.props.limits.timestampPeriod / 1e6 : 0;
}
static void og_done(int slot, const og_gap *gp, const og_marks *m, const uint64_t *pt, int np, const pass_meta *pm, uint64_t T0, uint64_t T1) {
    static double gb[PASSQ + 1]; const double per = V.props.limits.timestampPeriod / 1e6;
    if (np > PASSQ) np = PASSQ;
    for (int g = 0; g <= np; g++) gb[g] = 0;
    g_oga.subs++; g_oga.span += T1 > T0 ? (double)(T1 - T0) * per : 0; g_oga.nskip += (uint64_t)m->skipped;
    for (int i = 0; i < m->nb; i++) { double d = og_pair(slot, m->b[i].q); if (d < 0) { g_oga.nskip++; continue; } g_oga.nb++; if (m->b[i].gap <= np) gb[m->b[i].gap] += d; }
    for (int i = 0; i < m->nu; i++) { double d = og_pair(slot, m->u[i].q); if (d < 0) continue; if (m->u[i].pre) { g_oga.up_pre += d; g_oga.nu_pre++; } else { g_oga.up_in += d; g_oga.nu_in++; } }
    for (int i = 0; i < np; i++) if (pt[2 * i + 1] > pt[2 * i]) g_oga.pass += (double)(pt[2 * i + 1] - pt[2 * i]) * per;
    for (int g = 0; g <= np; g++) {
        const uint64_t s = g ? pt[2 * g - 1] : T0, e = g < np ? pt[2 * g] : T1;
        double raw = e > s ? (double)(e - s) * per : 0, inter = gb[g] < raw ? gb[g] : raw, d = raw - inter;
        g_oga.between += inter;
        if (g == 0) g_oga.g0 += d; else if (g == np) g_oga.gtail += d; else g_oga.gmid += d;
        const uint16_t *n = gp[g].n;
        int tr = n[OG_COPYBUF] | n[OG_B2I] | n[OG_I2B] | n[OG_COPYIMG] | n[OG_BLIT] | n[OG_RESOLVE] | n[OG_FILL] | n[OG_UPDATE] | n[OG_CLEARIMG] | n[OG_CLEARATT];
        int wk = n[OG_RP] | n[OG_DRAW] | n[OG_DISPATCH] | n[OG_XFB], ot = n[OG_BARRIER] | n[OG_QRESET] | n[OG_QUERY] | n[OG_TS];
        int c = tr ? 0 : wk ? 1 : n[OG_FULLBAR] ? 2 : ot ? 3 : 4; g_oga.cls[c] += d; g_oga.ncls[c]++;
        for (int k = 0; k < OG_N; k++) if (n[k]) { g_oga.kms[k] += d; g_oga.kn[k] += n[k]; }
        int b = n[OG_FULLBAR] < 3 ? n[OG_FULLBAR] : 3; g_oga.fbms[b] += d; g_oga.fbn[b]++; g_oga.fbs += n[OG_FULLBAR];
        if (g < np) { int f = (int)pm[g].fidx; g_ogp_ms[f] += d; g_ogp_n[f]++; g_ogp_last[f] = gp[g]; g_ogp_meta[f] = pm[g]; }
        if (g > 0 && g < np) { const int t = (pm[g - 1].fmt == VK_TS_DISPATCH ? 0 : 2) + (pm[g].fmt == VK_TS_DISPATCH ? 0 : 1); g_oga.trms[t] += d; g_oga.trn[t]++; }   // stage1: by transition
    }
}
static const char *const g_og_names[OG_N] = {"copy buffer", "copy buffer to image", "copy image to buffer", "copy image", "blit", "resolve", "fill", "update buffer",
    "clear image", "clear attachments", "render pass", "draw", "dispatch", "query reset", "query", "timestamp", "transform feedback", "pipeline barrier", "full barrier"};
extern const char *const g_op_names[64];   // gfx_vk_exec.c
static void og_report(uint64_t nf) {   // vk_pass_report (the completion thread), every 120 frames
    extern uint64_t g_s61_sent, g_nsubs;
    const uint64_t locks = atomic_load(&g_og_locks), nvn = g_s61_sent + g_nsubs;
    if (!nf || !g_oga.subs) { g_oga.locks0 = locks; g_oga.nvn0 = nvn; return; }
    const double f = (double)nf, gaps = g_oga.g0 + g_oga.gmid + g_oga.gtail;
    VKLOG("ootgap: per frame (last %llu frames, %.1f submissions a frame): span %.2f ms = timed passes %.2f + part boundaries %.2f + gaps %.2f (before the first pass %.2f, between passes %.2f, "
          "after the last %.2f); part boundaries %.1f a frame (%.1f not measured: a cut inside a pass, or no result); upload command buffers in the span %.1f, %.2f ms (they run inside the "
          "part boundaries), before the span %.1f, %.2f ms; queue lock taken by this layer %.1f times a frame, of them NVN parts and submissions %.1f (the rest: present, backend copies, "
          "sparse binds; Eden's own submissions are not counted, they show only as part-boundary time)",
          (unsigned long long)nf, g_oga.subs / f, g_oga.span / f, g_oga.pass / f, g_oga.between / f, gaps / f, g_oga.g0 / f, g_oga.gmid / f, g_oga.gtail / f, g_oga.nb / f, g_oga.nskip / f,
          g_oga.nu_in / f, g_oga.up_in / f, g_oga.nu_pre / f, g_oga.up_pre / f, (double)(locks - g_oga.locks0) / f, (double)(nvn - g_oga.nvn0) / f);
    VKLOG("ootgap: gaps by the main command buffer's GPU commands in them, ms a frame (gaps a frame): with copies/fills/clears %.2f (%.1f), with an untimed render pass/draw/dispatch "
          "%.2f (%.1f), full barriers and nothing heavier %.2f (%.1f), only other barriers/queries/timestamps %.2f (%.1f), no GPU command %.2f (%.1f)",
          g_oga.cls[0] / f, g_oga.ncls[0] / f, g_oga.cls[1] / f, g_oga.ncls[1] / f, g_oga.cls[2] / f, g_oga.ncls[2] / f, g_oga.cls[3] / f, g_oga.ncls[3] / f, g_oga.cls[4] / f, g_oga.ncls[4] / f);
    VKLOG("ootgap: gaps by full barriers in them, ms a frame (gaps a frame): none %.2f (%.1f), 1 %.2f (%.1f), 2 %.2f (%.1f), 3+ %.2f (%.1f); %.1f full barriers a frame in gaps, "
          "%.3f ms of gap time per full barrier",
          g_oga.fbms[0] / f, g_oga.fbn[0] / f, g_oga.fbms[1] / f, g_oga.fbn[1] / f, g_oga.fbms[2] / f, g_oga.fbn[2] / f, g_oga.fbms[3] / f, g_oga.fbn[3] / f, g_oga.fbs / f,
          g_oga.fbs ? (g_oga.fbms[1] + g_oga.fbms[2] + g_oga.fbms[3]) / (double)g_oga.fbs : 0.0);
    VKLOG("ootgap: gaps between passes by transition, ms a frame (gaps a frame): dispatch -> dispatch %.2f (%.1f), dispatch -> pass %.2f (%.1f), pass -> dispatch %.2f (%.1f), pass -> pass %.2f (%.1f)",
          g_oga.trms[0] / f, g_oga.trn[0] / f, g_oga.trms[1] / f, g_oga.trn[1] / f, g_oga.trms[2] / f, g_oga.trn[2] / f, g_oga.trms[3] / f, g_oga.trn[3] / f);   // stage1
    { char b[1400]; int o = 0;
      for (int k = 0; k < OG_N && o < (int)sizeof b - 80; k++) if (g_oga.kn[k]) o += snprintf(b + o, sizeof b - (size_t)o, " %s %.1f (gaps holding it %.2f ms);", g_og_names[k], g_oga.kn[k] / f, g_oga.kms[k] / f);
      VKLOG("ootgap: GPU commands in gaps a frame (a gap's time counts for each kind it holds):%s", o ? b : " none"); }
    { static int ix[FPASSQ]; int n = 0;
      for (int i = 0; i < FPASSQ; i++) if (g_ogp_n[i]) { int j = n++; while (j > 0 && g_ogp_ms[ix[j - 1]] < g_ogp_ms[i]) { ix[j] = ix[j - 1]; j--; } ix[j] = i; }
      for (int r = 0; r < n && r < 8; r++) { const int i = ix[r]; const og_gap *g = &g_ogp_last[i]; const pass_meta *pm = &g_ogp_meta[i]; char c[600]; int o = 0;
          for (int k = 0; k < OG_N && o < (int)sizeof c - 40; k++) if (g->n[k]) o += snprintf(c + o, sizeof c - (size_t)o, "%s%s %u", o ? ", " : "", g_og_names[k], g->n[k]);
          char ops[500]; int p = 0;
          for (int k = 0; k < 64 && p < (int)sizeof ops - 24; k++) if (g->ops >> k & 1) p += snprintf(ops + p, sizeof ops - (size_t)p, "%s%s", p ? " " : "", g_op_names[k] ? g_op_names[k] : "?");
          VKLOG("ootgap: gap before pass #%d: %.3f ms a frame (%.2f times a frame); the pass: %s %ux%u fmt %u, key/draws %#x; last seen with GPU commands: %s; record ops: %s",
                i, g_ogp_ms[i] / f, g_ogp_n[i] / f, pm->fmt == VK_TS_DISPATCH ? "dispatch" : pm->fmt == VK_TS_SCALE_WRITEBACK ? "2X writeback" : "pass", pm->w, pm->h, pm->fmt, pm->draws,
                o ? c : "none", p ? ops : "none"); } }
    memset(&g_oga, 0, sizeof g_oga); g_oga.locks0 = locks; g_oga.nvn0 = nvn;
    memset(g_ogp_ms, 0, sizeof g_ogp_ms); memset(g_ogp_n, 0, sizeof g_ogp_n);
}
static uint64_t g_fence;   // fence of the last commit
static VkSemaphore g_timeline;
static pthread_mutex_t g_done_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_done_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_done;
static uint64_t g_cur_frame;
// F0 milestones are observed lower bounds, not synchronization primitives.
static _Atomic int g_f0_on = -1;
static _Atomic uint64_t g_f0_ids, g_f0_errors, g_f0_scope_drops;
static _Atomic uint64_t g_f0_submitted, g_f0_gpu_observed, g_f0_present_done;
static __thread nvf0_submission *g_f0_sub;
static __thread nvf0_scope *g_f0_scope;
typedef struct { _Atomic uint64_t calls, ns, self, max, slow, failed; } f0_cost;
static f0_cost g_f0_cost[NVF0_KIND_COUNT];
static const char *const g_f0_names[NVF0_KIND_COUNT] = {
    "slot-wait", "pool-wait", "pool-sync", "trim", "save-or-release", "restore",
    "image-allocation", "archive-transfer", "gpu-wait", "copyback", "mirror", "completion"
};
static void f0_max(_Atomic uint64_t *dst, uint64_t value) {
    uint64_t old = atomic_load_explicit(dst, memory_order_relaxed);
    while (old < value && !atomic_compare_exchange_weak_explicit(dst, &old, value, memory_order_relaxed, memory_order_relaxed)) {}
}
int vk_f0_enabled(void) {
    if (!dm_sdk_ounce()) return 0;
    int on = atomic_load_explicit(&g_f0_on, memory_order_relaxed);
    if (on < 0) {
        const char *e = getenv("DM_VK_FOUNDATION_F0"); int wanted = !e || !*e || atoi(e) != 0;
        if (atomic_compare_exchange_strong(&g_f0_on, &on, wanted)) {
            on = wanted;
            if (on) VKLOG("NVN2 F0 diagnostics: on, schema 1; policy unchanged; DM_VK_FOUNDATION_F0=0 disables; cumulative timings, self excludes instrumented children only");
        }
    }
    return on;
}
static nvf0_submission f0_submission(void) {
    nvf0_submission s = {0};
    if (vk_f0_enabled()) { s.id = atomic_fetch_add(&g_f0_ids, 1) + 1; s.frame = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED); }
    return s;
}
void vk_f0_begin(nvf0_scope *s, uint32_t kind, uint64_t bytes) {
    memset(s, 0, sizeof *s);
    if (!vk_f0_enabled()) return;
    uint64_t id = atomic_fetch_add(&g_f0_ids, 1) + 1;
    if (!nvf0_enter(s, &g_f0_scope, id, kind, now_ns(), g_f0_sub ? g_f0_sub->id : 0,
                   g_f0_sub && g_f0_sub->id ? g_f0_sub->frame : __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED), bytes)) atomic_fetch_add(&g_f0_scope_drops, 1);
}
void vk_f0_end(nvf0_scope *s, int32_t result) {
    if (!s->id) return;
    if (!nvf0_leave(s, &g_f0_scope, now_ns())) { atomic_fetch_add(&g_f0_scope_drops, 1); return; }
    f0_cost *c = &g_f0_cost[s->kind];
    atomic_fetch_add(&c->calls, 1); atomic_fetch_add(&c->ns, s->elapsed); atomic_fetch_add(&c->self, s->self_ns); f0_max(&c->max, s->elapsed);
    uint64_t slow = 0, failed = 0;
    if (s->elapsed >= 50000000ull) slow = atomic_fetch_add(&c->slow, 1) + 1;
    if (result != 0) failed = atomic_fetch_add(&c->failed, 1) + 1;
    uint64_t n = failed ? failed : slow;
    if (n && (n <= 24 || !(n & (n - 1))))
        VKLOG("NVN2 F0 span: op=%llu parent=%llu sub=%llu frame=%llu kind=%s bytes=%llu status=%d inclusive_ms=%.3f self_ms=%.3f children_ms=%.3f slow=%llu failed=%llu",
            (unsigned long long)s->id, (unsigned long long)s->parent, (unsigned long long)s->submission, (unsigned long long)s->frame,
            g_f0_names[s->kind], (unsigned long long)s->bytes, result, s->elapsed / 1e6, s->self_ns / 1e6, s->child_ns / 1e6,
            (unsigned long long)slow, (unsigned long long)failed);
}
VkResult vk_f0_result(VkResult result, const char *site, uint32_t part) {
#if defined(DM_NXN1X)
    nvnc_error(part, result);
#endif
    if (result == VK_ERROR_DEVICE_LOST) { void vk_ck_report(void); vk_ck_report(); }   // gpuck: the queue's last checkpoints (gfx_vk_exec.c)
    if (result >= 0 || !vk_f0_enabled()) return result;
    uint64_t event = atomic_fetch_add(&g_f0_errors, 1) + 1;
    nvf0_error(g_f0_sub, result, site, part, event);
    if (event <= 64 || !(event & (event - 1)))
        VKLOG("[RED] NVN2 F0 failure: event=%llu sub=%llu frame=%llu call=%s part=%u result=%d first_result=%d first_part=%u accepted_parts=%u accepted_last=%u submitted_fence=%llu gpu_observed=%llu cpu_completed=%llu present_done_ticket=%llu",
            (unsigned long long)event, (unsigned long long)(g_f0_sub ? g_f0_sub->id : 0),
            (unsigned long long)(g_f0_sub ? g_f0_sub->frame : __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED)), site, part, result,
            g_f0_sub ? g_f0_sub->first_result : result, g_f0_sub ? g_f0_sub->first_part : part,
            g_f0_sub ? g_f0_sub->accepted_parts : 0, g_f0_sub ? g_f0_sub->accepted_last_part : 0,
            (unsigned long long)atomic_load(&g_f0_submitted), (unsigned long long)atomic_load(&g_f0_gpu_observed),
            (unsigned long long)vk_fence_done(), (unsigned long long)atomic_load(&g_f0_present_done));
    return result;   // exact driver result; callers keep their existing decisions
}
void vk_f0_present_done(uint64_t ticket) { if (vk_f0_enabled()) f0_max(&g_f0_present_done, ticket); }
void vk_f0_report(void) {
#if defined(DM_NXN1X)
    if (nvnc_active()) {
        nvnc_stats c; nvnc_get_stats(&c);
#define NC(x) (unsigned long long)c.x
        VKLOG("NVN2 F2 events: queued=%llu consumed=%llu backlog=%llu lost=%llu unknown_epoch=%llu metadata=%llu drain_ms=%.3f max_ms=%.3f bounded=%llu",
            NC(queued),NC(consumed),NC(backlog),NC(dropped),NC(unknown_epoch),NC(metadata_bytes),c.drain_ns/1e6,c.drain_max_ns/1e6,NC(bounded_yields));
        VKLOG("NVN2 F2 ranges: raw=%llu coalesced=%llu pending=%llu passes=%llu range_lost=%llu lifecycle_lost=%llu; tails continued, not discarded",
            NC(range_raw), NC(range_coalesced), NC(range_pending), NC(range_passes), NC(range_dropped), NC(queue_dropped));
        VKLOG("NVN2 F2 sources: cpu=%llu faults=%llu flush=%llu gpu_writeback=%llu cpu_bytes=%llu writeback_bytes=%llu device_bytes=%llu unresolved_bytes=%llu aliases=%llu",
            NC(cpu_events),NC(fault_events),NC(flush_events),NC(writebacks),NC(cpu_bytes),NC(writeback_bytes),NC(device_bytes),NC(unresolved_bytes),NC(alias_hits));
        VKLOG("NVN2 F2 records: pools=%llu images=%llu regions=%llu observed_cpu=%llu observed_gpu=%llu observed_mixed=%llu; authority=Unknown NCE_actual_store_exclusion=0",
            NC(pools),NC(images),NC(regions),NC(observed_cpu_regions),NC(observed_gpu_regions),NC(observed_mixed_regions));
        VKLOG("NVN2 F2 execution: intents=%llu accepted=%llu rejected=%llu gpu_done=%llu cpu_done=%llu first_error=%d sub=%llu part=%llu",
            NC(intents),NC(accepted_intents),NC(rejected_intents),NC(gpu_completed),NC(cpu_completed),c.first_error,NC(first_error_sub),NC(first_error_part));
        VKLOG("NVN2 F2 uploads: cpu_source_intents=%llu; transport does not increment cpu_gen or prove guest-source validity", NC(upload_intents));
        VKLOG("NVN2 F2 protection: checks=%llu pool_written=%llu bytes=%llu no_image_intent=%llu proven_false=0; repeated observations, not unique reclaimable bytes",
            NC(pool_protected_checks),NC(pool_protected_true),NC(pool_protected_bytes),NC(no_image_write_checks));
        VKLOG("NVN2 F2 unknown: reasons=%#llx unsupported=%llu; mapping/order/coverage uncertainty never authorizes reclamation",
            NC(unknown_reasons),NC(unsupported_events));
#undef NC
    }
#endif
    if (!vk_f0_enabled()) return;
    VKLOG("NVN2 F0 progress: submitted_fence=%llu gpu_observed=%llu cpu_completed=%llu present_done_ticket=%llu errors=%llu scope_drops=%llu; present tickets are a separate domain",
        (unsigned long long)atomic_load(&g_f0_submitted), (unsigned long long)atomic_load(&g_f0_gpu_observed),
        (unsigned long long)vk_fence_done(), (unsigned long long)atomic_load(&g_f0_present_done),
        (unsigned long long)atomic_load(&g_f0_errors), (unsigned long long)atomic_load(&g_f0_scope_drops));
    for (int i = 0; i < NVF0_KIND_COUNT; i++) {
        f0_cost *c = &g_f0_cost[i]; uint64_t calls = atomic_load(&c->calls); if (!calls) continue;
        VKLOG("NVN2 F0 cost: kind=%s calls=%llu inclusive_ms=%.3f self_ms=%.3f max_ms=%.3f slow=%llu failed=%llu (cumulative, inclusive overlaps parents)",
            g_f0_names[i], (unsigned long long)calls, atomic_load(&c->ns) / 1e6, atomic_load(&c->self) / 1e6,
            atomic_load(&c->max) / 1e6, (unsigned long long)atomic_load(&c->slow), (unsigned long long)atomic_load(&c->failed));
    }
}
// ---------------- s56: GPU -> guest write-back of render targets ----------------
// (lane A, 2026-11-13; the Mac's rt_mirror: nvn_mtl.m rt_mirror_add / mirror_readback / mirror_rb_land, nvn-objects.md.) Tears of the Kingdom bakes its terrain tiles
// (260x260 RG8, 9 levels, MINIMAL_LAYOUT, a CPU-cached pool) by rendering into them; here the result lived only in the VkImage and the pool memory stayed 0, and the
// game later moves the tiles with memcpy and re-creates the textures at the new address, which took their contents from that memory: giant black tiles.
// At the end of a pass (gfx_vk_exec.c s56_pass_end) each colour attachment that qualifies (vk_s56_mirror: 2D or 2D array, one sample, an uncompressed colour format
// nvn_tex_mirror_b2t writes, not COMPRESSIBLE / LINEAR / LINEAR_RENDER_TARGET / IMAGE / DISPLAY / VIDEO_DECODE / SPARSE, a pool with CPU access, not on the packaged
// path, at most 8 MB) has the level and layers it drew copied into a host-visible staging buffer (kept for reuse, s56_buf_get) in the same submission. The submission's
// fence seals the pending mirrors (s56_seal, mtl_sub_commit). The completion thread lands them (s56_land: nvn_tex_mirror_b2t swizzles into the storage, unprotecting and
// stamping it) right after the GPU passed that fence and before the fence is published (g_done) and before the submission's event and counter writes: whatever tells
// the guest that the work finished (nvnSyncWait, nvnQueueFinish, an acquire, an event or report value) comes after the landing, and nothing waits for a fence here.
// Within one submission a newer mirror whose range covers an older one replaces it (the last drawing of a range wins); across submissions they land in order.
// The dirty tracker's view of CPU writes between the pass and its completion is counted (on the hardware the GPU's write lands after such a write too);
// DM_VK_S56_CPU_DROP=1 drops such a mirror instead. A range no longer tracked at landing (its pool finalized) is not written.
// Switch: profile key 3 (dm_profile_key, nvnh_rt.c): DM_VK_S56=0/1, else the title's default (Tears of the Kingdom on, every other title off). Off, none of this runs
typedef struct { VkBuffer b; VkDeviceMemory m; uint8_t *p; VkDeviceSize size; int coh; } s56_buf;
typedef struct { tex_t t; unsigned level, layer0, n; uint32_t w, h; size_t row, img; uint64_t lo, hi, stamp, ticket; s56_buf buf; int lin; size_t stride; } s56_mir;   // z4: lin, stride   // ticket 0: its submission is still being recorded
enum { S56_FLAGS, S56_TARGET, S56_MSAA, S56_FORMAT, S56_POOL, S56_PACKAGED, S56_RANGE, S56_SIZE, S56_ALLOC, S56_NREASON };
static const char *const g_s56_why[S56_NREASON] = {"texture flags", "target", "multisampled", "format", "pool", "packaged", "level/layer/shape", "over 8 MB", "staging buffer"};
static pthread_mutex_t g_s56_mu = PTHREAD_MUTEX_INITIALIZER;   // the pending list and the kept staging buffers; never held while waiting for a fence
static s56_mir *g_s56_q; static int g_s56_n, g_s56_cap;
static s56_buf g_s56_keep[16]; static int g_s56_nkeep; static VkDeviceSize g_s56_keep_bytes;
static _Atomic uint64_t g_s56_alias, g_s56_normal, g_s56_land_max;   // s56b
static _Atomic uint64_t g_s56_mirrors, g_s56_bytes, g_s56_landed, g_s56_land_ns, g_s56_super, g_s56_failed, g_s56_gone, g_s56_cpu, g_s56_cpu_dropped, g_s56_dropsub, g_s56_skip[S56_NREASON];
static int s56_on(void) { return dm_profile_key(3); }
// s70 (apply_s70.py): DM_VK_S56_IMG=1 lets s56 mirror IMAGE / COMPRESSIBLE targets up to DM_VK_S56_IMG_MAX (default 512) in width and height (XC3 save thumbnails)
static _Atomic uint64_t g_s70_img, g_s70_bytes;
static uint64_t g_s70_key[32]; static _Atomic uint64_t g_s70_kn[32], g_s70_kb[32]; static _Atomic int g_s70_nk;   // s70b: mirrors and bytes per (w, h, format)
static int s70_max(void) { static int v = -1; if (v < 0) { extern int dm_title_default_key(int key); const char *e = getenv("DM_VK_S56_IMG"), *m = getenv("DM_VK_S56_IMG_MAX");
    const int td = dm_title_default_key(7), on = e && *e ? atoi(e) != 0 : td > 0;   // s70d: the title's default (nvnh_rt.c key 7) when the variable is unset or empty
    v = on ? (m && *m && atoi(m) > 0 ? atoi(m) : 512) : 0;
    VKLOG("s70: s56 for IMAGE targets, and IMAGE|COMPRESSIBLE ones not drawn steadily (s70c/s70d), %s (DM_VK_S56_IMG %s, max %d)", v ? "on" : "off",
          e && *e ? "from the environment" : td >= 0 ? "the title's default" : "unset, off", v); } return v; }
// s70c: per storage address, the last frame (present) a target was drawn and its run of consecutive frames; first-case logging per address
typedef struct { const void *mem; uint64_t last; uint32_t hist; uint8_t logged; } s70c_ent;   // s70d: hist bit i = drawn in frame last - i
static s70c_ent g_s70c[2048]; static _Atomic uint64_t g_s70c_steady;
static int s70d_hits(int n) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S56_IMG_STEADY_HITS"); v = e && *e && atoi(e) > 0 ? atoi(e) : 3; } return v < n ? v : n; }
static int s70c_steady_n(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S56_IMG_STEADY"); v = e && *e && atoi(e) > 0 ? atoi(e) : 4;
    VKLOG("s70c: IMAGE|COMPRESSIBLE targets drawn in at least %d of the previous %d frames are not mirrored (DM_VK_S56_IMG_STEADY_HITS, DM_VK_S56_IMG_STEADY; s70d)", s70d_hits(v), v); } return v; }
static s70c_ent *s70c_get(const void *mem) {
    unsigned i = (unsigned)((((uint64_t)(uintptr_t)mem >> 8) * 0x9E3779B97F4A7C15ull) >> 53) & 2047;
    for (unsigned j = 0; j < 2048; j++, i = (i + 1) & 2047) { if (g_s70c[i].mem == mem) return &g_s70c[i]; if (!g_s70c[i].mem) { g_s70c[i].mem = mem; return &g_s70c[i]; } }
    return NULL;
}
// s70e: per kind (w, h, format, flags) the same frame history; the frame a storage was first drawn (or first after an idle stretch) is mirrored
typedef struct { uint64_t key, last; uint32_t hist; _Atomic uint64_t skipped; } s70e_kind;
static s70e_kind g_s70e[128]; static int g_s70e_n;
static int s70e_first_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S56_IMG_FIRST"); v = !(e && *e && atoi(e) == 0); } return v; }
static uint64_t s70e_idle(void) { static int64_t v = -1; if (v < 0) { const char *e = getenv("DM_VK_S56_IMG_IDLE"); v = e && *e && atoi(e) > 0 ? atoi(e) : 120;
    VKLOG("s70e: steadiness per kind and per storage; a storage's first draw or first after %lld idle frames is mirrored (DM_VK_S56_IMG_FIRST=%d, DM_VK_S56_IMG_IDLE)", (long long)v, s70e_first_on()); } return (uint64_t)v; }
static s70e_kind *s70e_get(uint64_t key) {
    for (int i = 0; i < g_s70e_n; i++) if (g_s70e[i].key == key) return &g_s70e[i];
    if (g_s70e_n < 128) { s70e_kind *k = &g_s70e[g_s70e_n++]; k->key = key; return k; }
    return NULL;
}
static void s70e_tick(uint64_t *last, uint32_t *hist, uint64_t f) {
    if (!*hist) { *hist = 1; *last = f; } else if (*last != f) { uint64_t d = f - *last; *hist = d >= 32 ? 1u : *hist << d | 1u; *last = f; }
}
typedef struct { const void *mem; uint64_t grace; } s70e_grace;
static s70e_grace g_s70e_g[2048];
static int s70c_fresh(const tex_t *t) {   // s70e: 1 = mirror this IMAGE|COMPRESSIBLE target
    s70c_ent *e = s70c_get(t->mem); if (!e) return 0;
    const uint64_t f = g_r138_nf, idle = s70e_idle();
    const uint32_t w = t->d.width ? t->d.width : 1, h = t->d.height ? t->d.height : 1;
    const uint64_t key = (uint64_t)w << 44 | (uint64_t)(h & 0xffff) << 28 | (uint64_t)(t->d.flags & 0xfff) << 16 | (uint64_t)(t->d.format & 0xffff);
    s70e_kind *k = s70e_get(key);
    // the grace frame: a storage's first draw, or its first after >= idle frames without one
    s70e_grace *g = &g_s70e_g[e - g_s70c];
    const int is_new = !e->hist, was_idle = e->hist && f - e->last >= idle;
    if (s70e_first_on() && (is_new || was_idle)) g->grace = f + 1;   // (+1: 0 = none)
    s70e_tick(&e->last, &e->hist, f);
    if (k) s70e_tick(&k->last, &k->hist, f);
    const int n = s70c_steady_n() < 31 ? s70c_steady_n() : 31, hits = s70d_hits(n);
    const int own = __builtin_popcount((e->hist >> 1) & ((1u << n) - 1u)) >= hits, kind = k && __builtin_popcount((k->hist >> 1) & ((1u << n) - 1u)) >= hits;
    const int grace = g->grace == f + 1;
    const int ok = grace || !(own || kind);
    if (!ok) { atomic_fetch_add(&g_s70c_steady, 1); if (k) atomic_fetch_add(&k->skipped, 1); }
    if (ok && w == 320 && h == 180 && t->d.format == 0x25) {   // s70e: every mirror decision for a 320x180 RGBA8 storage
        static int nl; if (nl < 400) { nl++; VKLOG("s70e: 320x180 RGBA8 storage %p flags %#x mirrored at frame %llu: %s", t->mem, t->d.flags, (unsigned long long)f,
                                                   grace ? (is_new ? "first draw" : was_idle ? "first after idle" : "grace frame") : "not steady"); } }
    return ok;
}
// z4: LINEAR / LINEAR_RENDER_TARGET targets (pitch layout) written back too; DM_VK_Z4 (default on for every title, 0 off). Non-linear targets follow s56_on()
int vk_z4_on(void) { return dm_profile_key(4); }   // nvnh_rt.c: key 4 = DM_VK_Z4 (env, else the title's default, else the build's)
static int s56_any(void) { return s56_on() || vk_z4_on(); }
static _Atomic uint64_t g_z4_mirrors, g_z4_bytes, g_z4_landed, g_z4_skip[S56_NREASON];
static void z4_skip(int why, const tex_t *t) {
    atomic_fetch_add(&g_z4_skip[why], 1);
    static _Atomic int seen[S56_NREASON];
    if (!atomic_exchange(&seen[why], 1)) VKLOG("z4: linear write-back skipped (%s), first case: texture %ux%u format %#x target %d levels %u samples %u flags %#x stride %u, pool flags %#x, size %llu",
        g_s56_why[why], t->d.width, t->d.height, t->d.format, t->d.target, t->d.levels, t->d.samples, t->d.flags, t->d.stride, t->pool ? t->pool->flags : 0, (unsigned long long)t->L.size);
}
static int s56_cpu_drop(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S56_CPU_DROP"); v = e && atoi(e) != 0; } return v; }
int vk_tex_s56_packaged(const tex_t *t);   // gfx_vk_tex.c
static int s56_buf_get(VkDeviceSize n, s56_buf *out) {   // a kept buffer of the size class, else a new one (HOST_CACHED where the device has it: the CPU reads it)
    VkDeviceSize c = 4096; while (c < n) c <<= 1;
    pthread_mutex_lock(&g_s56_mu);
    for (int i = 0; i < g_s56_nkeep; i++) if (g_s56_keep[i].size == c) { *out = g_s56_keep[i]; g_s56_keep[i] = g_s56_keep[--g_s56_nkeep]; g_s56_keep_bytes -= c; pthread_mutex_unlock(&g_s56_mu); return 1; }
    pthread_mutex_unlock(&g_s56_mu);
    s56_buf b; memset(&b, 0, sizeof b); b.size = c;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, c, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &b.b) != VK_SUCCESS) return 0;
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, b.b, &rq);
    uint32_t ty = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (ty == UINT32_MAX) ty = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (ty == UINT32_MAX) ty = vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, ty};
    void *p = NULL;
    if (ty == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &b.m) != VK_SUCCESS) { vkDestroyBuffer(V.dev, b.b, NULL); return 0; }
    if (vkBindBufferMemory(V.dev, b.b, b.m, 0) != VK_SUCCESS || vkMapMemory(V.dev, b.m, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) { vkDestroyBuffer(V.dev, b.b, NULL); vkFreeMemory(V.dev, b.m, NULL); return 0; }
    b.p = p; b.coh = (V.mem.memoryTypes[ty].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    *out = b; return 1;
}
// Under g_s56_mu. gpu_done: its submission completed (landing); else it is in the open submission's command buffers (superseded, or the submission was refused): kept
// (any later copy into it is recorded after that one and ordered by the barriers around every copy) or freed once the GPU has passed the open submission
static void s56_buf_put_locked(const s56_buf *b, int gpu_done) {
    if (g_s56_nkeep < 16 && g_s56_keep_bytes + b->size <= (16u << 20)) { g_s56_keep[g_s56_nkeep++] = *b; g_s56_keep_bytes += b->size; return; }
    if (gpu_done) { vkDestroyBuffer(V.dev, b->b, NULL); vkFreeMemory(V.dev, b->m, NULL); }
    else vk_defer_free_buffer(b->b, b->m);
}
static void s56_del_locked(int i) { memmove(&g_s56_q[i], &g_s56_q[i + 1], (size_t)(g_s56_n - i - 1) * sizeof *g_s56_q); g_s56_n--; }
// z3: one line for each distinct kind of skipped texture (and depth target), up to 96 kinds
static int z3_new_kind(int why, const tex_t *t) {
    static uint64_t kinds[96]; static int nk; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    uint64_t k = (uint64_t)why * 0x9E3779B97F4A7C15ull ^ ((uint64_t)t->d.width << 1) ^ ((uint64_t)t->d.height << 17) ^ ((uint64_t)t->d.depth << 33) ^ ((uint64_t)t->d.format << 41) ^
                 ((uint64_t)t->d.target << 49) ^ ((uint64_t)t->d.levels << 53) ^ ((uint64_t)t->d.flags << 57) ^ ((uint64_t)(t->pool ? t->pool->flags : 0) * 0xC2B2AE3D27D4EB4Full);
    int fresh = 0;
    pthread_mutex_lock(&mu);
    int i = 0; while (i < nk && kinds[i] != k) i++;
    if (i == nk && nk < 96) { kinds[nk++] = k; fresh = 1; }
    pthread_mutex_unlock(&mu);
    return fresh;
}
void vk_z3_depth(const tex_t *t) {   // gfx_vk_exec.c s56_pass_end: the pass's depth/stencil attachment (s56 never mirrors these)
    if (!t->mem || !t->pool || (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL))) return;
    if (z3_new_kind(-1, t)) VKLOG("z3: depth target in a CPU-accessible pool (not written back): texture %ux%ux%u format %#x target %d levels %u samples %u flags %#x, pool flags %#x, storage %p",
        t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, t->d.samples, t->d.flags, t->pool->flags, (void *)t->mem);
}
static void s56_skip(int why, const tex_t *t) {
    atomic_fetch_add(&g_s56_skip[why], 1);
    if (why != S56_POOL && z3_new_kind(why, t)) VKLOG("z3: s56 skips (%s): texture %ux%ux%u format %#x target %d levels %u samples %u flags %#x, pool flags %#x, storage %p", g_s56_why[why],
        t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, t->d.samples, t->d.flags, t->pool ? t->pool->flags : 0, (void *)t->mem);
    static _Atomic int seen[S56_NREASON];
    if (!atomic_exchange(&seen[why], 1)) VKLOG("s56: write-back skipped (%s), first case: texture %ux%ux%u format %#x target %d levels %u samples %u flags %#x, pool flags %#x%s", g_s56_why[why],
        t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target, t->d.levels, t->d.samples, t->d.flags, t->pool ? t->pool->flags : 0, t->mem ? "" : " (no storage)");
}
// gfx_vk_exec.c s56_pass_end, on the replay thread inside the open submission, after the pass ended: level and layers [layer0, layer0 + n) of the attachment's image
static void z4_mirror(const tex_t *t, vkimg_t *im, unsigned level, unsigned layer0, unsigned n);
void vk_s56_mirror(const tex_t *t, vkimg_t *im, unsigned level, unsigned layer0, unsigned n) {
    if ((t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && vk_z4_on()) { z4_mirror(t, im, level, layer0, n); return; }   // z4
    if (!s56_on()) return;   // z4 on, s56 off for this title: only linear targets
    int bw = 0, bh = 0, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), why = -1;
    const vkfmt_t *f = vk_fmt(t->d.format), *fi = vk_fmt(im->nvn_format);
    const pool_t *pl = t->pool;
    uint32_t w = im->w >> level ? im->w >> level : 1, h = im->h >> level ? im->h >> level : 1;
    size_t row = (size_t)w * (size_t)(bpb > 0 ? bpb : 0), img = row * h, bytes = img * n;
    if (!t->mem || !pl || (pl->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL))) why = S56_POOL;
    else if (t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_IMAGE |
                           NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_VIDEO_DECODE | NVN_TEXTURE_FLAGS_SPARSE) &
             ~(s70_max() && (int)w <= s70_max() && (int)h <= s70_max() ? (uint32_t)NVN_TEXTURE_FLAGS_IMAGE | ((t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) && s70c_fresh(t) ? (uint32_t)NVN_TEXTURE_FLAGS_COMPRESSIBLE : 0u) : 0u)) why = S56_FLAGS;   // s70, s70b: IMAGE; s70c: COMPRESSIBLE when not drawn every frame
    else if (t->d.target != NVN_TEXTURE_TARGET_2D && t->d.target != NVN_TEXTURE_TARGET_2D_ARRAY) why = S56_TARGET;
    else if (t->d.samples > 1 || im->samples > 1) why = S56_MSAA;
    else if (bpb <= 0 || bw != 1 || bh != 1 || (bpb & (bpb - 1)) || !f || !fi || f->depth || f->stencil || f->compressed || f->bpb != bpb || fi->bpb != bpb ||
             im->aspect != VK_IMAGE_ASPECT_COLOR_BIT) why = S56_FORMAT;
    else if (t->has_layout || vk_tex_s56_packaged(t)) why = S56_PACKAGED;
    else if (im->w != (t->d.width ? t->d.width : 1) || im->h != (t->d.height ? t->d.height : 1) || im->d != 1 || !t->d.levels || level >= t->d.levels || level >= im->levels ||
             !n || layer0 + n > im->layers || im->layout != VK_IMAGE_LAYOUT_GENERAL || !t->L.layer_size) why = S56_RANGE;
    else if (bytes > (8u << 20)) why = S56_SIZE;
    if (why >= 0) { s56_skip(why, t); return; }
    if (t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE)) {   // s70: the first mirror of each size / format; s70b: counted per kind
        uint64_t k = (uint64_t)w << 44 | (uint64_t)(h & 0xffff) << 28 | (uint64_t)(t->d.flags & 0xfff) << 16 | (uint64_t)(t->d.format & 0xffff); int ns = atomic_load(&g_s70_nk), ki = -1;   // s70c: flags in the kind
        for (int i = 0; i < ns; i++) if (g_s70_key[i] == k) { ki = i; break; }
        if (ki >= 0) { atomic_fetch_add(&g_s70_kn[ki], 1); atomic_fetch_add(&g_s70_kb[ki], (uint64_t)bytes); }
        if (ki < 0 && ns < 32) { g_s70_key[ns] = k; atomic_store(&g_s70_kn[ns], 1); atomic_store(&g_s70_kb[ns], (uint64_t)bytes); atomic_store(&g_s70_nk, ns + 1); }
        s70c_ent *le = s70c_get(t->mem);   // s70c: the first mirror of each storage address
        if (le && !le->logged) { le->logged = 1; VKLOG("s70: s56 mirrors an IMAGE/COMPRESSIBLE target %ux%u format %#x flags %#x level %u layers %u, pool flags %#x, storage %p", w, h, t->d.format, t->d.flags, level, n, pl->flags, t->mem); }
        atomic_fetch_add(&g_s70_img, 1); atomic_fetch_add(&g_s70_bytes, (uint64_t)bytes); }
    // the guest range it writes (for the alias / superseding and dirty checks): from the level in the first layer to its end in the last one
    uint64_t lvo = nvn_tex_level_offset(&t->d, level), lve = level + 1 < t->d.levels ? nvn_tex_level_offset(&t->d, level + 1) : t->L.layer_size;
    if (lve <= lvo || lve > t->L.layer_size || (uint64_t)(layer0 + n - 1) * t->L.layer_size + lve > t->L.size) { s56_skip(S56_RANGE, t); return; }
    uint64_t lo = (uint64_t)(uintptr_t)t->mem + (uint64_t)layer0 * t->L.layer_size + lvo, hi = (uint64_t)(uintptr_t)t->mem + (uint64_t)(layer0 + n - 1) * t->L.layer_size + lve;
    s56_buf b; if (!s56_buf_get(bytes, &b)) { s56_skip(S56_ALLOC, t); return; }
    VkCommandBuffer cb = vk_cmd();
    vk_before_native_readback(im);
    vk_img_barrier_all(cb);   // after the pass's attachment writes
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, level, layer0, n}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b.b, 1, &c);
    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};   // the copy made visible to the CPU's read at landing
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    vk_img_barrier_all(cb);   // before what comes next writes the image again
    pthread_mutex_lock(&g_s56_mu);
    for (int i = 0; i < g_s56_n; )   // the open submission's mirrors whose range this one covers: replaced (the last drawing of the range wins)
        if (!g_s56_q[i].ticket && g_s56_q[i].lo >= lo && g_s56_q[i].hi <= hi) { s56_buf_put_locked(&g_s56_q[i].buf, 0); s56_del_locked(i); atomic_fetch_add(&g_s56_super, 1); }
        else i++;
    if (g_s56_n == g_s56_cap) { int nc = g_s56_cap ? 2 * g_s56_cap : 64; s56_mir *nq = realloc(g_s56_q, (size_t)nc * sizeof *nq);
        if (!nq) { s56_buf_put_locked(&b, 0); pthread_mutex_unlock(&g_s56_mu); s56_skip(S56_ALLOC, t); return; }
        g_s56_q = nq; g_s56_cap = nc; }
    s56_mir *m = &g_s56_q[g_s56_n++];
    m->lin = 0; m->stride = 0; m->t = *t; m->level = level; m->layer0 = layer0; m->n = n; m->w = w; m->h = h; m->row = row; m->img = img; m->lo = lo; m->hi = hi; m->stamp = nvn_dirty_now(); m->ticket = 0; m->buf = b;
    pthread_mutex_unlock(&g_s56_mu);
    atomic_fetch_add(&g_s56_mirrors, 1); atomic_fetch_add(&g_s56_bytes, bytes);
    { static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < 4) VKLOG("s56: write-back of texture %ux%u format %#x flags %#x level %u layers %u+%u (pool flags %#x): copied out at the pass end, written into its storage when the GPU is done",
        t->d.width, t->d.height, t->d.format, t->d.flags, level, layer0, n, pl->flags); }
}
// z4: a linear target: one level, one layer, rows of the texture's stride (nvn_tex.c: layer_size = stride * h). Copied out tight; s56_land writes each row's
// width*bpb bytes at storage + y*stride (the padding is left alone, as on the hardware)
static void z4_mirror(const tex_t *t, vkimg_t *im, unsigned level, unsigned layer0, unsigned n) {
    int bw = 0, bh = 0, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), why = -1;
    const vkfmt_t *f = vk_fmt(t->d.format), *fi = vk_fmt(im->nvn_format);
    const pool_t *pl = t->pool;
    uint32_t w = im->w, h = im->h;
    size_t row = (size_t)w * (size_t)(bpb > 0 ? bpb : 0), stride = h ? (size_t)(t->L.layer_size / h) : 0;
    if (!t->mem || !pl || (pl->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL))) why = S56_POOL;
    else if (t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_VIDEO_DECODE | NVN_TEXTURE_FLAGS_SPARSE)) why = S56_FLAGS;
    else if (t->d.target != NVN_TEXTURE_TARGET_2D) why = S56_TARGET;
    else if (t->d.samples > 1 || im->samples > 1) why = S56_MSAA;
    else if (bpb <= 0 || bw != 1 || bh != 1 || (bpb & (bpb - 1)) || !f || !fi || f->depth || f->stencil || f->compressed || f->bpb != bpb || fi->bpb != bpb ||
             im->aspect != VK_IMAGE_ASPECT_COLOR_BIT) why = S56_FORMAT;
    else if (t->has_layout) why = S56_PACKAGED;
    else if (level || layer0 || n != 1 || im->w != (t->d.width ? t->d.width : 1) || im->h != (t->d.height ? t->d.height : 1) || im->d != 1 || im->layers != 1 ||
             im->layout != VK_IMAGE_LAYOUT_GENERAL || !h || stride < row || (uint64_t)stride * (h - 1) + row > t->L.size) why = S56_RANGE;
    else if ((uint64_t)stride * h > (8u << 20)) why = S56_SIZE;
    if (why >= 0) { z4_skip(why, t); return; }
    uint64_t lo = (uint64_t)(uintptr_t)t->mem, hi = lo + (uint64_t)stride * (h - 1) + row;
    size_t bytes = row * h;
    s56_buf b; if (!s56_buf_get(bytes, &b)) { z4_skip(S56_ALLOC, t); return; }
    VkCommandBuffer cb = vk_cmd();
    vk_before_native_readback(im);
    vk_img_barrier_all(cb);   // after the pass's attachment writes
    VkBufferImageCopy c = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b.b, 1, &c);
    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
    vk_img_barrier_all(cb);   // before what comes next writes the image again
    pthread_mutex_lock(&g_s56_mu);
    for (int i = 0; i < g_s56_n; )   // the open submission's mirrors whose range this one covers: replaced (the last drawing of the range wins)
        if (!g_s56_q[i].ticket && g_s56_q[i].lo >= lo && g_s56_q[i].hi <= hi) { s56_buf_put_locked(&g_s56_q[i].buf, 0); s56_del_locked(i); atomic_fetch_add(&g_s56_super, 1); }
        else i++;
    if (g_s56_n == g_s56_cap) { int nc = g_s56_cap ? 2 * g_s56_cap : 64; s56_mir *nq = realloc(g_s56_q, (size_t)nc * sizeof *nq);
        if (!nq) { s56_buf_put_locked(&b, 0); pthread_mutex_unlock(&g_s56_mu); z4_skip(S56_ALLOC, t); return; }
        g_s56_q = nq; g_s56_cap = nc; }
    s56_mir *m = &g_s56_q[g_s56_n++];
    m->t = *t; m->level = 0; m->layer0 = 0; m->n = 1; m->w = w; m->h = h; m->row = row; m->img = bytes; m->lo = lo; m->hi = hi; m->stamp = nvn_dirty_now(); m->ticket = 0; m->buf = b;
    m->lin = 1; m->stride = stride;
    pthread_mutex_unlock(&g_s56_mu);
    atomic_fetch_add(&g_s56_mirrors, 1); atomic_fetch_add(&g_s56_bytes, bytes); atomic_fetch_add(&g_z4_mirrors, 1); atomic_fetch_add(&g_z4_bytes, bytes);
    { static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < 8) VKLOG("z4: write-back of linear texture %ux%u format %#x flags %#x stride %zu (pool flags %#x, storage %p): copied out at the pass end, rows written into its storage when the GPU is done",
        t->d.width, t->d.height, t->d.format, t->d.flags, stride, pl->flags, (void *)t->mem); }
}
// Every 10 s, when anything happened (from mtl_sub_commit, one thread at a time under g_sub_lock)
static void s56_report(void) {
    static uint64_t last; uint64_t t = now_ns();
    if (!last) { last = t; return; }
    if (t - last < 10000000000ull) return;
    double s = (t - last) / 1e9; last = t;
    uint64_t mi = atomic_exchange(&g_s56_mirrors, 0), by = atomic_exchange(&g_s56_bytes, 0), la = atomic_exchange(&g_s56_landed, 0), ns = atomic_exchange(&g_s56_land_ns, 0),
             su = atomic_exchange(&g_s56_super, 0), fa = atomic_exchange(&g_s56_failed, 0), go = atomic_exchange(&g_s56_gone, 0), cp = atomic_exchange(&g_s56_cpu, 0),
             cd = atomic_exchange(&g_s56_cpu_dropped, 0), ds = atomic_exchange(&g_s56_dropsub, 0), sk[S56_NREASON], skn = 0;
    for (int i = 0; i < S56_NREASON; i++) { sk[i] = atomic_exchange(&g_s56_skip[i], 0); skn += sk[i]; }
    uint64_t zm = atomic_exchange(&g_z4_mirrors, 0), zb = atomic_exchange(&g_z4_bytes, 0), zl = atomic_exchange(&g_z4_landed, 0), zk[S56_NREASON], zkn = 0;   // z4
    for (int i = 0; i < S56_NREASON; i++) { zk[i] = atomic_exchange(&g_z4_skip[i], 0); zkn += zk[i]; }
    if (zm || zl || zkn) {
        char zb2[300]; int zo = 0; zb2[0] = 0;
        for (int i = 0; i < S56_NREASON && zo < (int)sizeof zb2 - 48; i++) if (zk[i]) zo += snprintf(zb2 + zo, sizeof zb2 - (size_t)zo, "%s%s %llu", zo ? ", " : "", g_s56_why[i], (unsigned long long)zk[i]);
        VKLOG("z4: last %.0f s: linear %llu (%.2f MB), landed %llu; linear skipped by reason {%s}", s, (unsigned long long)zm, zb / 1048576.0, (unsigned long long)zl, zb2);
    }
    { uint64_t im70 = atomic_exchange(&g_s70_img, 0), by70 = atomic_exchange(&g_s70_bytes, 0);
      if (im70) { char kb[1600]; int ko = 0; kb[0] = 0; int nk = atomic_load(&g_s70_nk);   // s70b: per (w, h, format)
          for (int i = 0; i < nk && ko < (int)sizeof kb - 64; i++) { uint64_t c = atomic_exchange(&g_s70_kn[i], 0), b = atomic_exchange(&g_s70_kb[i], 0); if (!c) continue;
              ko += snprintf(kb + ko, sizeof kb - (size_t)ko, "%s%ux%u flags %#x fmt %#x: %llu (%.2f MB)", ko ? ", " : "", (unsigned)(g_s70_key[i] >> 44), (unsigned)(g_s70_key[i] >> 28 & 0xffff),
                             (unsigned)(g_s70_key[i] >> 16 & 0xfff), (unsigned)(g_s70_key[i] & 0xffff), (unsigned long long)c, b / 1048576.0); }
          VKLOG("s70: last %.0f s: img %llu (%.2f MB) of the s56 mirrors were IMAGE targets; by kind: %s", s, (unsigned long long)im70, by70 / 1048576.0, ko ? kb : "-"); }
      uint64_t st70 = atomic_exchange(&g_s70c_steady, 0);
      if (st70) { char sb[1600]; int so = 0; sb[0] = 0;   // s70e: per kind
          for (int i = 0; i < g_s70e_n && so < (int)sizeof sb - 64; i++) { uint64_t c = atomic_exchange(&g_s70e[i].skipped, 0); if (!c) continue; uint64_t q = g_s70e[i].key;
              so += snprintf(sb + so, sizeof sb - (size_t)so, "%s%ux%u flags %#x fmt %#x: %llu", so ? ", " : "", (unsigned)(q >> 44), (unsigned)(q >> 28 & 0xffff), (unsigned)(q >> 16 & 0xfff),
                             (unsigned)(q & 0xffff), (unsigned long long)c); }
          VKLOG("s70c: last %.0f s: skipped steady %llu (IMAGE|COMPRESSIBLE mirrors refused: drawn steadily); by kind: %s", s, (unsigned long long)st70, so ? sb : "-"); } }
    if (!mi && !la && !su && !fa && !go && !cp && !ds && !skn) return;
    char b[400]; int o = 0; b[0] = 0;
    for (int i = 0; i < S56_NREASON && o < (int)sizeof b - 48; i++) if (sk[i]) o += snprintf(b + o, sizeof b - (size_t)o, "%s%s %llu", o ? ", " : "", g_s56_why[i], (unsigned long long)sk[i]);
    pthread_mutex_lock(&g_s56_mu); int pend = g_s56_n; pthread_mutex_unlock(&g_s56_mu);
    VKLOG("s56: write-back last %.0f s: mirrors %llu, %.2f MB, landed %llu (on the completion thread before the fence is published, so before SyncWait/Finish/acquire/events see it), waited 0 ms, landing took %.2f ms; "
          "superseded in their submission %llu, refused by b2t %llu, pool gone %llu, CPU wrote the range meanwhile %llu (dropped %llu, DM_VK_S56_CPU_DROP), submission refused %llu, pending %d; skipped by reason {%s}",
          s, (unsigned long long)mi, by / 1048576.0, (unsigned long long)la, ns / 1e6, (unsigned long long)su, (unsigned long long)fa, (unsigned long long)go, (unsigned long long)cp, (unsigned long long)cd,
          (unsigned long long)ds, pend, b);
    VKLOG("s56b: last %.0f s: layers written through the backing alias %llu, the normal way (a page busy, or no alias: unprotect) %llu; longest landing %.2f ms; protected pages stamped without unprotect %llu",
          s, (unsigned long long)atomic_exchange(&g_s56_alias, 0), (unsigned long long)atomic_exchange(&g_s56_normal, 0), atomic_exchange(&g_s56_land_max, 0) / 1e6, (unsigned long long)nvn_dirty_gpu_write_prot());
}
// mtl_sub_commit, with the submission's fence (refused: the driver refused its command buffer, nothing was copied)
static void s56_seal(uint64_t f, int refused) {
    pthread_mutex_lock(&g_s56_mu);
    for (int i = 0; i < g_s56_n; ) {
        if (g_s56_q[i].ticket) { i++; continue; }
        if (!refused) { pool_fence_max(&g_pool_sync_fence, f); g_s56_q[i++].ticket = f; continue; }
        s56_buf_put_locked(&g_s56_q[i].buf, 0); s56_del_locked(i); atomic_fetch_add(&g_s56_dropsub, 1);
    }
    pthread_mutex_unlock(&g_s56_mu);
    s56_report();
}
// The completion thread, once the GPU passed fence f and before its post-ops and g_done: every sealed mirror up to f is swizzled into its texture's storage, in order
static void s56_land(uint64_t f) {
    pthread_mutex_lock(&g_s56_mu);
    if (!g_s56_n) { pthread_mutex_unlock(&g_s56_mu); return; }
    uint64_t t0 = now_ns(); int cpu_drop = s56_cpu_drop();
    for (int i = 0; i < g_s56_n; ) {
        s56_mir *m = &g_s56_q[i];
        if (!m->ticket || m->ticket > f) { i++; continue; }
        if (!m->buf.coh) { VkMappedMemoryRange mr = {VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, m->buf.m, 0, VK_WHOLE_SIZE}; vkInvalidateMappedMemoryRanges(V.dev, 1, &mr); }
        if (!nvn_dirty_tracked((const void *)(uintptr_t)m->lo, (size_t)(m->hi - m->lo))) atomic_fetch_add(&g_s56_gone, 1);   // its pool was finalized: the memory is not the texture's any more
        else {
            uint64_t a = (m->lo + 4095) & ~4095ull, e = m->hi & ~4095ull;   // pages wholly inside the range (an edge page is shared with the neighbours)
            int cpu = e > a && nvn_dirty_since((const void *)(uintptr_t)a, (size_t)(e - a), m->stamp);
            if (cpu) atomic_fetch_add(&g_s56_cpu, 1);
            if (cpu && cpu_drop) atomic_fetch_add(&g_s56_cpu_dropped, 1);
            else {
                NVNcopyRegion r = {0, 0, 0, (int)m->w, (int)m->h, (int)m->n};
                int nal = 0, nno = 0; uint64_t l0 = now_ns();
                int nvn_tex_mirror_b2t_alias(const tex_t *, unsigned, unsigned, const NVNcopyRegion *, const uint8_t *, size_t, size_t, int *, int *);
                int nvn_tex_mirror_lin_alias(const tex_t *, const uint8_t *, size_t, uint32_t, size_t, int *, int *);
                int rc56 = m->lin ? nvn_tex_mirror_lin_alias(&m->t, m->buf.p, m->row, m->h, m->stride, &nal, &nno)   // z4: rows at the texture's stride
                                  : nvn_tex_mirror_b2t_alias(&m->t, m->level, m->layer0, &r, m->buf.p, m->row, m->img, &nal, &nno);   // s56b: through the backing alias
                if (!rc56 && m->lin) atomic_fetch_add(&g_z4_landed, 1);
                { uint64_t d56 = now_ns() - l0, mx = atomic_load(&g_s56_land_max); atomic_fetch_add(&g_s56_alias, (uint64_t)nal); atomic_fetch_add(&g_s56_normal, (uint64_t)nno);
                  while (d56 > mx && !atomic_compare_exchange_weak(&g_s56_land_max, &mx, d56)) {} }
                if (rc56) {
                    atomic_fetch_add(&g_s56_failed, 1);
                    static int nf; if (nf++ < 4) VKLOG("[RED] s56: write-back of texture %ux%u format %#x level %u refused by nvn_tex_mirror_b2t: not written", m->t.d.width, m->t.d.height, m->t.d.format, m->level);
                } else {
                    atomic_fetch_add(&g_s56_landed, 1);
                    uint64_t now = nvn_dirty_now();   // the later mirrors over this range: this write is not a CPU write to them
                    for (int j = i + 1; j < g_s56_n; j++) if (g_s56_q[j].lo < m->hi && m->lo < g_s56_q[j].hi) g_s56_q[j].stamp = now;
                }
            }
        }
        s56_buf_put_locked(&m->buf, 1); s56_del_locked(i);
    }
    atomic_fetch_add(&g_s56_land_ns, now_ns() - t0);
    pthread_mutex_unlock(&g_s56_mu);
}
static void occ2_init(void);   // occ-s1 (below, with the occlusion code)
static void slots_init(void) {
    VkSemaphoreTypeCreateInfo ti = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &ti};
    VKCHK(vkCreateSemaphore(V.dev, &si, NULL, &g_timeline));
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * NSLOT};
    if (V.props.limits.timestampComputeAndGraphics && vkCreateQueryPool(V.dev, &qi, NULL, &g_tsq)) g_tsq = VK_NULL_HANDLE;
    VkQueryPoolCreateInfo cqi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NSLOT * CQ};
    if (!g_tsq || vkCreateQueryPool(V.dev, &cqi, NULL, &g_cq)) g_cq = VK_NULL_HANDLE;
    { int64_t vk_occ_fixed(void); const char *e = getenv("DM_VK_OQ"); int vk_occ_keyed_on(void); if ((!e || atoi(e)) && (vk_occ_fixed() < 0 || vk_occ_keyed_on())) { /* rptfix: real queries only with DM_VK_OCC_FIXED=-1 (occkey: or the keyed counts) */ VkQueryPoolCreateInfo oi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, NSLOT * OQ}; if (vkCreateQueryPool(V.dev, &oi, NULL, &g_oq)) g_oq = VK_NULL_HANDLE; }
      VkPhysicalDeviceFeatures2 f = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; vkGetPhysicalDeviceFeatures2(V.pd, &f); g_oq_flags = f.features.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0;
      VKLOG("occlusion queries: %s%s", g_oq ? "on" : "off (rptfix DM_VK_OCC_FIXED, DM_VK_OQ=0 or no pool: a fixed count)", g_oq && g_oq_flags ? " (precise)" : "");
      occ2_init(); }   // occ-s1: DM_VK_OCC2=s1, the per-draw ring (after g_oq and its precise flag)
    const char *timers = getenv("DM_VK_PASS_TIMES");
    int default_on = 0;
#ifdef DM_NXN1X
    extern int nvnh_default_pass_times(void);
    default_on = nvnh_default_pass_times();
#endif
    g_pass_times_requested = timers && *timers ? atoi(timers) != 0 : default_on;
    if (g_pass_times_requested) {
        VkQueryPoolCreateInfo pq = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0,
                                    VK_QUERY_TYPE_TIMESTAMP, NSLOT * PASSQ * 2};
        VkResult result = g_tsq ? vkCreateQueryPool(V.dev, &pq, NULL, &g_pq) : VK_ERROR_FEATURE_NOT_PRESENT;
        if (result != VK_SUCCESS) g_pq = VK_NULL_HANDLE;
        VKLOG("DM_VK_PASS_TIMES=1 (%s): timestamp pool %s, VkResult %d, capacity %u pass pairs/submission",
              timers && *timers ? "environment" : "HWAoI title default",
              g_pq ? "active" : "unavailable", (int)result, (unsigned)PASSQ);
        if (g_pq) { VkQueryPoolCreateInfo oq = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, NSLOT * OGQ};   // ootgap
            if (vkCreateQueryPool(V.dev, &oq, NULL, &g_ogq)) g_ogq = VK_NULL_HANDLE;
            VKLOG("ootgap: %s (gaps between timed passes, part boundaries, upload command buffers; DM_VK_OOTGAP=0 off)", g_ogq ? "on" : "off: no query pool"); }
    }
    for (int i = 0; i < NSLOT; i++) {
        // not TRANSIENT, and reset by command buffer (see slot_recycle)
        VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
        VKCHK(vkCreateCommandPool(V.dev, &ci, NULL, &g_slot[i].pool));
        // uppool (DK Bananza 11-22, parallel recording: crashes inside vkQueueSubmit of a part (nvogla64, called from vk_cmd_split) and a device loss in a worker's pass): the
        // upload buffers came from the slot's pool too, and workers record into the open one (under vk_up_lock) while the replay thread records its main and R chunks from
        // that pool with no lock: Vulkan wants a command pool externally synchronized across all its buffers (VkCommandPool host sync), NVIDIA's command memory is per pool.
        // The upload buffers now have a pool of their own, used only under vk_up_lock. DM_VK_UPPOOL=0: one pool as before
        { const char *ue = getenv("DM_VK_UPPOOL"); const int sep = !(ue && *ue == '0'); g_slot[i].up_pool = g_slot[i].pool;
          if (sep && vkCreateCommandPool(V.dev, &ci, NULL, &g_slot[i].up_pool)) { g_slot[i].up_pool = g_slot[i].pool; VKLOG("[RED] uppool: no pool of its own for slot %d: upload buffers share the slot's pool", i); }
          if (!i) VKLOG("uppool: upload command buffers %s (DM_VK_UPPOOL=%s)", g_slot[i].up_pool != g_slot[i].pool ? "in a command pool of their own per slot" : "in the slot's pool (as before)", ue && *ue ? ue : "unset"); }
        VkCommandBuffer cb[2]; VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_slot[i].pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, cb));
        { VkCommandBufferAllocateInfo au = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_slot[i].up_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; if (g_slot[i].up_pool != g_slot[i].pool) VKCHK(vkAllocateCommandBuffers(V.dev, &au, &cb[0])); }
        g_slot[i].up = g_slot[i].ups[0] = cb[0]; g_slot[i].main = g_slot[i].mains[0] = cb[1]; g_slot[i].nmain_alloc = 1; g_slot[i].nup_alloc = 1;
    }
    par_start();   // S1a (parallel recording): the workers, if DM_VK_PARALLEL asks for any
}
int vk_oq_real(void) { return g_oq != VK_NULL_HANDLE || g_occ2 >= 2; }   // (occ-s2: the per-draw ring's counts)
uint64_t vk_fence_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }   // written under g_done_mu (for the waiters) with a release store
static void wait_fence(uint64_t f) {
    if (!f) return;
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &g_timeline, &f};
    vkWaitSemaphores(V.dev, &wi, UINT64_MAX);
}
// The slot's command buffers are reset on the completion thread once its submission is done (completion_main), not when the replay begins them again: an
// implicit reset at vkBeginCommandBuffer took 7% of Xenoblade Chronicles 3's replay thread at a submission's start and more at every part (its battles
// replay ~3,000 draws a frame on a thread that is the frame's limit). The slot is then taken only after the completion thread has passed it (mtl_wait, g_done)
// rather than as soon as the GPU has. DM_VK_X_NOPRERESET=1 (start-up): reset as they are begun
static int g_prereset = -1;
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
static uint64_t g_ss_slot_ns;   // s15: waited for a free slot
static void ss_begin(void), ss_begun(void), ss_end(uint64_t c0, uint64_t qs);   // s15 (below, before mtl_sub_commit)
int rd_on(void); void vk_rd_start(void), vk_rd_mark(int k); extern uint64_t g_rd_ns[64], g_rd_n[64]; extern int g_rdiag;   // replaydiag (gfx_vk_exec.c)
static uint64_t g_scr_hw;   // the most scratch one submission used (slot_recycle; NVN2 memory report)
// NVN2 (Switch 2 boots): a slot keeps scratch blocks for the most it used in its last 300 recycles, at least 32 MB; the blocks past that are freed at once (slot_recycle
// runs once the GPU is past the slot's last submission, so none is in use). HWAoI 11-16 (587 s, out of device memory): 352 MB in 43 blocks, kept after one submission
// that used 195 MB at 137 s, while the driver's heap stayed at 95-97% of its budget. Blocks are taken first-fit in order, so the first ones stay
static uint64_t g_scr_freed, g_scr_freed_n;
static void scratch_trim(vk_arena *s, uint64_t used) {   // R3 (parallel recording): an arena (was the slot)
    if (used >= s->scr_peak) { s->scr_peak = used; s->scr_age = 0; } else if (++s->scr_age > 300) { s->scr_peak = used; s->scr_age = 0; }
    if (g_scr_squeeze > 0) { g_scr_squeeze--; s->scr_peak = 0; s->scr_age = 0; }   // NVN2 pressure (nvn2_pressure): down to the floor
    const uint64_t keep = s->scr_peak > (32ull << 20) ? s->scr_peak : (32ull << 20);
    uint64_t have = 0; int k = 0;
    while (k < s->nblk && have < keep) have += s->blk[k++].size;
    if (k >= s->nblk) return;
    uint64_t freed = 0; int n = s->nblk - k;
    for (int i = k; i < s->nblk; i++) { vkDestroyBuffer(V.dev, s->blk[i].b, NULL); vkFreeMemory(V.dev, s->blk[i].m, NULL); freed += s->blk[i].size; }
    s->nblk = k; g_scr_freed += freed; g_scr_freed_n += (uint64_t)n;
    static int nl; if (nl++ < 8) VKLOG("NVN2 scratch: a slot freed %d blocks (%.1f MB) past its recent need (%.1f MB kept; %.0f MB freed so far)", n, freed / 1048576.0, have / 1048576.0, g_scr_freed / 1048576.0);
}
// NVN2 memory report: scratch blocks and their bytes, the high-water use, main / upload command buffers allocated in all and in the fullest slot, descriptor pools
static void vk_slot_stats(uint64_t o[8]) {
    memset(o, 0, 8 * sizeof *o); o[2] = g_scr_hw;
    for (int k = 0; k < NSLOT; k++) { const vksub_t *s = &g_slot[k];
        for (int i = 0; i < s->ar.nblk; i++) { o[0]++; o[1] += s->ar.blk[i].size; }   // R3 (parallel recording): the slot's arena
        o[3] += (uint64_t)s->nmain_alloc; if ((uint64_t)s->nmain_alloc > o[4]) o[4] = (uint64_t)s->nmain_alloc;
        o[5] += (uint64_t)s->nup_alloc; if ((uint64_t)s->nup_alloc > o[6]) o[6] = (uint64_t)s->nup_alloc;
        o[7] += (uint64_t)s->ar.ndp;
        for (int w = 0; w < g_npar_ctx; w++) { const vk_par_slot *ps = &g_par[w].slot[k];   // S1a (parallel recording): the workers' arenas and primaries of this slot (S1e: and the helper's)
            for (int i = 0; i < ps->ar.nblk; i++) { o[0]++; o[1] += ps->ar.blk[i].size; }
            o[3] += (uint64_t)ps->ncb; o[7] += (uint64_t)ps->ar.ndp; } }
}
// Caller owns submission state. Retirement requires CPU publication as well as
// GPU completion; never consume a pending free as budget credit.
static unsigned slot_release_deferred(vksub_t *s, unsigned limit, uint64_t deadline) {
    unsigned count=0;
    while (count<(unsigned)s->ndf && count<limit && (!deadline || now_ns()<deadline)) {
        dfree_t *d=&s->df[count];
        if (d->kind == 0) { vkDestroyBuffer(V.dev, (VkBuffer)d->a, NULL); if (d->b) vkFreeMemory(V.dev, (VkDeviceMemory)d->b, NULL); }
        else if (d->kind == 1) { vkDestroyImage(V.dev, (VkImage)d->a, NULL); if (d->b) vkFreeMemory(V.dev, (VkDeviceMemory)d->b, NULL); }
        else if (d->kind == 4) { void img_sub_free_(void *); vkDestroyImage(V.dev, (VkImage)d->a, NULL); img_sub_free_(d->b); }
        else if (d->kind == 2) vkDestroyImageView(V.dev, (VkImageView)d->a, NULL);
        else if (d->kind == 3) vkDestroyBufferView(V.dev, (VkBufferView)d->a, NULL);
        ++count;
    }
    s->ndf-=(int)count;
    if (count && s->ndf) memmove(s->df,s->df+count,(size_t)s->ndf*sizeof *s->df);
    return count;
}
#if defined(DM_NXN1X)
static void reclaim_completed_slots(void) {
    if (nvnr_trylock(&g_sub_lock)) return;
    const uint64_t done=vk_fence_done(), deadline=now_ns()+3000000ull;
    unsigned released=0, pending=0;
    for (unsigned i=0;i<NSLOT;i++) {
        vksub_t *s=&g_slot[i];
        if ((int)i==g_cur || !s->fence || s->fence>done) { pending+=(unsigned)s->ndf; continue; }
        released+=slot_release_deferred(s,256-released,deadline);
        pending+=(unsigned)s->ndf;
    }
    nvnr_unlock(&g_sub_lock);
    if (released) {
        static _Atomic uint64_t batches, entries;
        const uint64_t total=atomic_fetch_add(&entries,released)+released;
        const uint64_t batch=atomic_fetch_add(&batches,1)+1;
        if (batch<=16 || !(batch&(batch-1)))
            VKLOG("NVN2 F4B completed retirement: entries=%u remaining=%u cpu_done=%llu total_entries=%llu; no wait or speculative byte credit",
                released,pending,(unsigned long long)done,(unsigned long long)total);
    }
}
#endif
// pwait (NPARK 11-22: with the window sent to the back, the presents stopped for 9 s before the 00:44 crash, and nothing in the log said where they waited): the
// waits of the present path, each in a slot while it waits (nvn_pwait_enter / leave: the present manager's acquire, queue present, pacing and frame fence, the
// render thread's free frame and present queue, the replay's submission slot); a thread looks at them every 100 ms and logs a wait still going at 0.5, 1, 2, 4, 8,
// 16 s and every 16 s after, with the other waits going at that moment (the chain), and a wait over 0.5 s again when it ends. DM_VK_PWAIT=0 off
extern uint64_t g_tex_frame;   // (gfx_vk_tex.c: the present count)
enum { PW_N = 12 };
static struct { _Atomic(const char *) what; _Atomic uint64_t since; _Atomic uint32_t tid; _Atomic int step; } g_pw[PW_N];
static _Atomic int g_pw_on = -1;
static int pwait_on(void) { int v = atomic_load(&g_pw_on); if (v < 0) { const char *e = getenv("DM_VK_PWAIT"); v = !(e && *e == '0'); atomic_store(&g_pw_on, v); } return v; }
static void *pwait_main(void *a) {
    (void)a; static const double marks[] = {0.5, 1, 2, 4, 8, 16};
    for (;;) {
        struct timespec ts = {0, 100000000}; nanosleep(&ts, NULL);
        const uint64_t now = now_ns();
        for (int i = 0; i < PW_N; i++) {
            const char *w = atomic_load(&g_pw[i].what); const uint64_t t0 = atomic_load(&g_pw[i].since); if (!w || !t0 || now < t0) continue;
            const double s = (now - t0) / 1e9; const int st = atomic_load(&g_pw[i].step);
            const double due = st < 6 ? marks[st] : 16.0 * (st - 4);
            if (s < due) continue;
            atomic_store(&g_pw[i].step, st + 1);
            char others[400]; int on = 0; others[0] = 0;
            for (int k = 0; k < PW_N; k++) { if (k == i) continue; const char *o = atomic_load(&g_pw[k].what); const uint64_t ok = atomic_load(&g_pw[k].since);
                if (o && ok && now >= ok && on < (int)sizeof others - 80) on += snprintf(others + on, sizeof others - (size_t)on, " %s %.0f ms;", o, (now - ok) / 1e6); }
            VKLOG("[RED] pwait: %s has waited %.1f s (thread %#x, present %llu); also waiting:%s", w, s, (unsigned)atomic_load(&g_pw[i].tid), (unsigned long long)g_tex_frame, on ? others : " nothing");
        }
    }
    return NULL;
}
void nvn_pwait_enter(int slot, const char *what) {
    if (slot < 0 || slot >= PW_N || !pwait_on()) return;
    static _Atomic int started; if (!atomic_exchange(&started, 1)) { pthread_t t; if (!pthread_create(&t, NULL, pwait_main, NULL)) pthread_detach(t);
        VKLOG("pwait: present-path waits watched (logged when one goes past 0.5 s; DM_VK_PWAIT=0 off)"); }
    atomic_store(&g_pw[slot].step, 0); atomic_store(&g_pw[slot].tid, (uint32_t)(uintptr_t)pthread_self()); atomic_store(&g_pw[slot].since, now_ns()); atomic_store(&g_pw[slot].what, what);
}
void nvn_pwait_leave(int slot) {
    if (slot < 0 || slot >= PW_N || !pwait_on()) return;
    const char *w = atomic_load(&g_pw[slot].what); const uint64_t t0 = atomic_load(&g_pw[slot].since);
    atomic_store(&g_pw[slot].what, NULL); atomic_store(&g_pw[slot].since, 0);
    if (w && t0) { const uint64_t d = now_ns() - t0; if (d >= 500000000ull) VKLOG("[RED] pwait: %s ended after %.2f s (present %llu)", w, d / 1e9, (unsigned long long)g_tex_frame); }
}
static void slot_recycle(vksub_t *s) {   // the GPU has passed this slot's last submission: free what it held, rewind its allocators
    if (g_prereset < 0) { const char *e = getenv("DM_VK_X_NOPRERESET"); g_prereset = !(e && atoi(e)); }
    uint64_t w0 = now_ns();
    if (rd_on() && s->fence && (g_prereset || pc_async())) {   // replaydiag: the GPU fence (slot 40), then the completion thread's g_done after it (slot 41)
        wait_fence(s->fence); const uint64_t w1 = now_ns(); g_rd_ns[40] += w1 - w0; g_rd_n[40]++;
        mtl_wait(s->fence, ~0ull); g_rd_ns[41] += now_ns() - w1; g_rd_n[41]++;
    } else
    if (g_prereset || pc_async()) mtl_wait(s->fence, ~0ull); else wait_fence(s->fence);   // stage2c: its pre/post copies are read by its merge (review T10: not reused before)
    pcb_reset(s->pc_st); pcb_reset(s->pc_rb); s->npce = 0;
    { static uint64_t n, ns, mx; uint64_t d = now_ns() - w0; ns += d; g_ss_slot_ns += d; if (d > mx) mx = d;   // r53 measure: the replay waiting for a free submission slot
      if (!(++n % 2400)) { VKLOG("submission slots: waited %.2f ms a submission for a free slot (max %.1f ms, last 2400)", ns / 2.4e9, mx / 1e6); ns = mx = 0; } }
    if (s->atomic_scratch && !s->atomic_used) {
        vkDestroyBuffer(V.dev, s->atomic_scratch, NULL);
        vkFreeMemory(V.dev, s->atomic_memory, NULL);
        s->atomic_scratch = VK_NULL_HANDLE;
        s->atomic_memory = VK_NULL_HANDLE;
    }
    s->atomic_used = 0;
    s->hy3_used = 0;   // stage1: the GPU has passed this slot's last submission
    slot_release_deferred(s,UINT_MAX,0);
    { // R3 (parallel recording): the slot's arena rewound; S1a: and every worker's arena of this slot, its command pool reset (its primaries reused from the first) and the
      // chunk list emptied. The fence has passed, and a worker records into the open slot only, which this one is not yet: no worker is using them
      const int slot = (int)(s - g_slot); uint64_t su = 0;
      for (int w = -1; w < g_npar_ctx; w++) { vk_arena *const a = w < 0 ? &s->ar : &g_par[w].slot[slot].ar;   // (S1e: the helper context's too)
        { uint64_t u = 0; for (int i = 0; i < a->nblk; i++) u += a->blk[i].used; su += u;
          if (dm_sdk_ounce()) scratch_trim(a, u); }
        for (int i = 0; i < a->nblk; i++) a->blk[i].used = 0;
        for (int i = 0; i < a->ndp; i++) vkResetDescriptorPool(V.dev, a->dp[i], 0);
        a->dp_cur = 0; a->dp_counted = -1;
        if (w >= 0) { vk_par_slot *const ps = &g_par[w].slot[slot]; if (ps->used) { vkResetCommandPool(V.dev, ps->pool, 0); ps->used = 0; } } }
      if (su > g_scr_hw) g_scr_hw = su;   // (NVN2 memory report: the submission's scratch, its workers' included)
      if (g_npar) { pthread_mutex_lock(&g_chunk_mu); s->nchunk = 0; s->chunk0 = 0; pthread_mutex_unlock(&g_chunk_mu); s->nrcb_used = 0; } }
    // The pool is not reset: each command buffer is reset as it is begun again (the pool allows it). A reset of the transient pool made Adreno's driver free the
    // command memory and map it again as the next submission grew (ioctl, mmap and munmap): in Xenoblade Chronicles 3's battles the submission's start took 21%
    // of the replay thread, now 9%, and frames went from about 42 to 38 ms
}
static vksub_t *cur(void) { if (g_cur < 0) die("[vk] no open submission"); return &g_slot[g_cur]; }
// R3 (parallel recording): the arena this thread allocates from: its recording target's, or (NULL, every target so far) the current slot's; dies outside a
// submission as cur() did
static inline vk_arena *rec_arena(void) { vk_arena *const a = g_vk_rt->ar; return a ? a : &cur()->ar; }
int vk_sub_open(void) { return g_cur >= 0; }
void vk_sub_suspend(nvnr_submission *saved) {
    *saved=(nvnr_submission){0};
    if (!g_sub_owned || !vk_sub_open()) return;
    if (!nvnr_suspend(&g_sub_lock)) die("[vk] nested replay suspension");
    *saved=(nvnr_submission){g_fence,g_cur,g_slot[g_cur].nmain_sent,1};
    nvnr_unlock(&g_sub_lock);
}
void vk_sub_resume(const nvnr_submission *saved) {
    if (!saved->active) return;
    if (nvnr_lock(&g_sub_lock)) die("[vk] replay resume lock failed");
    if (!g_sub_owned || g_cur!=saved->slot || g_fence!=saved->fence ||
        g_slot[g_cur].nmain_sent!=saved->accepted_parts)
        die("[vk] suspended replay transaction changed");
    if (!nvnr_resume(&g_sub_lock)) die("[vk] replay resume lost ownership");
}
// A long submission is recorded into several command buffers, submitted together in order: Adreno's driver refuses a command buffer past some size at
// vkEndCommandBuffer (VK_ERROR_OUT_OF_HOST_MEMORY), which NEO's handheld frames reached at about the 712th draw. The caller has ended the render pass.
uint64_t g_s61_sent, g_s61_sub_ns;   // s61 (gfx_vk_exec.c s61_log): parts submitted early, time in their queue lock and vkQueueSubmit
extern int g_ss_on; static uint64_t g_ss_part_n, g_ss_part_lock, g_ss_part_sub, g_ss_main_lock;   // s19 (with the s15 probe): waits for the queue lock
// S1e (parallel recording): a part's command buffers in queue order when workers record: its upload buffer, its chunks (R chunks the replay thread ended at each hand-out,
// W chunks the workers or the replay thread recorded; a failed one left out), then the replay thread's open R chunk (with_main) and extra. Replay thread, the part joined
static VkCommandBuffer *g_pcb; static int g_cappcb;
static uint32_t par_cbs(vksub_t *s, VkCommandBuffer up, int with_main, VkCommandBuffer extra) {
    pthread_mutex_lock(&g_chunk_mu);
    const int need = (s->nchunk - s->chunk0) + 3;
    if (need > g_cappcb) { int c = g_cappcb ? g_cappcb : 64; while (c < need) c *= 2; void *p = realloc(g_pcb, (size_t)c * sizeof *g_pcb); if (!p) die("[vk] S1e: no memory for a part's command buffers"); g_pcb = p; g_cappcb = c; }
    uint32_t n = 0; if (up) g_pcb[n++] = up;
    for (int i = s->chunk0; i < s->nchunk; i++) if (s->chunk[i].done == 1 && s->chunk[i].cb) g_pcb[n++] = s->chunk[i].cb;
    s->chunk0 = s->nchunk;
    pthread_mutex_unlock(&g_chunk_mu);
    if (with_main) g_pcb[n++] = s->main;
    if (extra) g_pcb[n++] = extra;
    return n;
}
static void par_chunk_r(vksub_t *s, VkCommandBuffer cb) {   // S1e: an ended R chunk of the replay thread, listed in queue order
    pthread_mutex_lock(&g_chunk_mu);
    if (s->nchunk == s->capchunk) { s->capchunk = s->capchunk ? 2 * s->capchunk : 16; s->chunk = realloc(s->chunk, (size_t)s->capchunk * sizeof *s->chunk); if (!s->chunk) die("[vk] parallel recording: no memory"); }
    s->chunk[s->nchunk++] = (vk_chunk){cb, -1, 1};
    pthread_mutex_unlock(&g_chunk_mu);
}
void vk_cmd_split(void) {
    vksub_t *s = cur(); if (s->nmain_used == MAIN_CBS || s->nup_used == MAIN_CBS) return;
    if (g_npar) vk_seg_part_join();   // S1e (parallel recording): the part's worker segments recorded (they record into its upload buffer) and their post ops merged before it ends
    if (g_occ2 >= 2) occ2_opp(O2R_PART);   // occ-s2: a resolve point (the caller ended the pass); occ-s3fast: taken only when the plan or a bound needs it (a part end is no guest read: the fence is the submission's)
    og_part_end(s);   // ootgap
    if (vk_f0_result(vkEndCommandBuffer(s->main), "vkEndCommandBuffer(part-main)", s->nmain_used) < 0) { s->cb_err = 1; static int n; if (n++ < 3) VKLOG("[RED] a part of a long submission was refused at its end (part %d)", s->nmain_used); }
    // what is recorded so far goes to the GPU now, as a submission of its own (with its uploads first): Adreno refuses a single submission past about
    // 700 draws (NEO's handheld frames: VK_ERROR_OUT_OF_HOST_MEMORY at vkEndCommandBuffer whatever the draws were, and in one command buffer or several)
    const char *se = getenv("DM_VK_SUBMIT_SPLIT"); int send = !(se && !atoi(se));   // on unless DM_VK_SUBMIT_SPLIT=0
    if (send && !s->cb_err) {
        { void vk_rpt_flush(void); vk_rpt_flush(); }   // replay2: the reports recorded so far are in memory before this part reaches the GPU
        // Freeze still-unused copied sources before any part can write guest
        // memory through a different physical alias. Later parts must not read
        // those sources on the CPU concurrently with this submission's GPU work.
        pool_sync_pending(0, UINT64_MAX);
        VkCommandBuffer cbs2[2]; const VkCommandBuffer *cbs = cbs2; uint32_t n = 0;
        // upswap (NPARK 11-22, the 00:44 crash in vkCmdPipelineBarrier: a barrier into a command buffer that was not recording): the upload buffer was ended here
        // and only swapped for a fresh one after the submit below, with no upload lock: a recording worker (vk_upload_cmd under vk_up_lock, e.g. vk_tex_prepare's first
        // transition) found s->up still in use between the end and the swap and recorded into the ended buffer. Now its end and the swap are one step under
        // vk_up_lock (only the upload buffer's commands, the allocation and the swap: no other lock inside); the ended one is submitted from `up_end`
        VkCommandBuffer up_end = VK_NULL_HANDLE;
        if (s->up_used) { void vk_vbm_up_close(VkCommandBuffer); vk_up_lock(); vk_vbm_up_close(s->up); vk_img_barrier_all(s->up); og_up_end(s); if (vk_f0_result(vkEndCommandBuffer(s->up), "vkEndCommandBuffer(part-upload)", s->nmain_used) < 0) s->cb_err = 1; up_end = s->up;
            if (s->nup_used == s->nup_alloc) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->up_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};   // uppool
                VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->ups[s->nup_alloc])); s->nup_alloc++; }
            s->up = s->ups[s->nup_used++]; s->up_used = 0; vk_up_unlock(); cbs2[n++] = up_end; }
        if (g_npar) { n = par_cbs(s, up_end, 1, VK_NULL_HANDLE); cbs = g_pcb; }   // S1e (parallel recording): upload, chunks in order, the open R chunk
        else cbs2[n++] = s->main;
        if (!s->cb_err) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, n, cbs, 0, NULL};
            uint64_t s61t = now_ns();   // s61
            uint64_t q0 = g_ss_on > 0 ? now_ns() : 0; vk_queue_lock(); uint64_t q1 = g_ss_on > 0 ? now_ns() : 0;
            VkResult r = vkQueueSubmit(V.q, 1, &si, VK_NULL_HANDLE); vk_queue_unlock();
            vk_f0_result(r, "vkQueueSubmit(part)", s->nmain_used);
#if defined(DM_NXN1X)
            nvnc_accept(s->content_sub, s->nmain_used, s->nmain_used, r);
#endif
            g_s61_sent++; g_s61_sub_ns += now_ns() - s61t;   // s61
            if (g_ss_on > 0) { g_ss_part_n++; g_ss_part_lock += q1 - q0; g_ss_part_sub += now_ns() - q1; }   // s19: a part's wait for the queue lock (NXN1X: Scheduler::submit_mutex) and its submit
            if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] vkQueueSubmit (part): %d (device lost?)", r); }
            nvf0_accepted(&s->f0, s->nmain_used, 1);
            { static int n; if (n++ < 3) VKLOG("a long submission: part %d sent early", s->nmain_used); }
            s->nmain_sent = s->nmain_used;
            // (upswap: the next part's fresh upload buffer was taken above, at the end of this one)
        }
    } else if (g_npar) par_chunk_r(s, s->main);   // S1e (parallel recording): not sent now: the ended part's R chunk listed after its chunks, for the commit (in queue order)
    if (s->nmain_used == s->nmain_alloc) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->mains[s->nmain_alloc])); s->nmain_alloc++; }
    s->main = s->mains[s->nmain_used++]; g_vk_rt_main.cb = s->main;   // R2 (parallel recording): the replay thread's target follows the slot's main buffer
#if defined(DM_NXN1X)
    nvnc_part((unsigned)s->nmain_used);
#endif
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(nvn2_begin(s->main, &bi, "vkBeginCommandBuffer (main)")); g_vk_rt_main.gen++;   // R2 (parallel recording): was g_cmd_gen++
    og_part_begin(s);   // ootgap
}   // outside a submission nothing may be recorded (present, a pool finalized between frames)
VkCommandBuffer vk_cmd(void) { const VkCommandBuffer cb = g_vk_rt->cb; if (!cb) die("[vk] no open submission"); return cb; }   // R2 (parallel recording): the thread's recording target (was cur()->main)
int vk_cmd_parts_left(void) { vksub_t *s = cur(); return MAIN_CBS - (s->nmain_used > s->nup_used ? s->nmain_used : s->nup_used); }   // parts vk_cmd_split can still begin (gfx_vk_exec.c pass splits)
uint64_t g_up_gen;   // upload command buffers begun (gfx_vk_tex.c lin_copy: once per upload buffer)
// R6/R7/R8 (parallel recording), R8: the two recording locks (gfx_vk.h has their scope and order). Only the replay thread records so far: always uncontended
static pthread_mutex_t g_vk_par_mu = PTHREAD_MUTEX_INITIALIZER, g_vk_up_mu = PTHREAD_MUTEX_INITIALIZER;
static __thread unsigned tl_par_depth, tl_up_depth;
void vk_par_lock(void) { if (!tl_par_depth++) pthread_mutex_lock(&g_vk_par_mu); }
void vk_par_unlock(void) { if (!--tl_par_depth) pthread_mutex_unlock(&g_vk_par_mu); }
void vk_up_lock(void) { if (!tl_up_depth++) pthread_mutex_lock(&g_vk_up_mu); }
void vk_up_unlock(void) { if (!--tl_up_depth) pthread_mutex_unlock(&g_vk_up_mu); }
// ---- S1a (parallel recording): the workers (struct vk_worker above; gfx_vk.h) ----
int vk_par_workers(void) { return g_npar; }
int vk_sub_slot(void) { return g_cur; }
static void *par_main(void *a) {
    vk_worker *const w = a; char nm[32]; snprintf(nm, sizeof nm, "dm-vk-rec-%d", w->id); pthread_setname_np(pthread_self(), nm);
    tl_worker = w; g_vk_rt = &w->rt; vk_rec_ctx_bind(w->rec, w->rec_e);   // this thread records with its own contexts and target
    if (g_par_pin) vk_par_pin_self(w->id);   // S1e (parallel recording): on the P-cores (gfx_vk_present.c; DM_VK_PARALLEL_PIN=0 off)
    pthread_mutex_lock(&w->mu);
    for (;;) {
        while (w->qh == w->qt) pthread_cond_wait(&w->cv, &w->mu);
        const vk_par_job j = w->q[w->qh % PAR_Q]; w->busy = 1; pthread_mutex_unlock(&w->mu);
        j.fn(j.arg);
        pthread_mutex_lock(&w->mu); w->busy = 0; w->qh++; pthread_cond_broadcast(&w->cv);   // (a full queue's or vk_par_drain's waiter)
    }
    return NULL;
}
void vk_par_run(int wi, void (*fn)(void *), void *arg) {
    if (wi < 0 || wi >= g_npar) die("[vk] parallel recording: no worker %d (%d running)", wi, g_npar);
    vk_worker *const w = &g_par[wi];
    pthread_mutex_lock(&w->mu);
    while (w->qt - w->qh == PAR_Q) pthread_cond_wait(&w->cv, &w->mu);
    w->q[w->qt % PAR_Q] = (vk_par_job){fn, arg}; w->qt++; pthread_cond_broadcast(&w->cv);
    pthread_mutex_unlock(&w->mu);
}
void vk_par_drain(void) {
    for (int i = 0; i < g_npar; i++) { vk_worker *const w = &g_par[i];
        pthread_mutex_lock(&w->mu); while (w->qh != w->qt || w->busy) pthread_cond_wait(&w->cv, &w->mu); pthread_mutex_unlock(&w->mu); }
}
VkCommandBuffer vk_par_chunk_begin(int slot) {
    vk_worker *const w = tl_worker; if (!w) die("[vk] parallel recording: a chunk begun outside a worker");
    if (slot < 0 || slot >= NSLOT) die("[vk] parallel recording: chunk for slot %d", slot);
    vk_par_slot *const ps = &w->slot[slot];
    if (ps->used == ps->ncb) {
        if (ps->ncb == ps->capcb) { ps->capcb = ps->capcb ? 2 * ps->capcb : 4; ps->cb = realloc(ps->cb, (size_t)ps->capcb * sizeof *ps->cb); if (!ps->cb) die("[vk] parallel recording: no memory"); }
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, ps->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (vkAllocateCommandBuffers(V.dev, &ai, &ps->cb[ps->ncb])) return VK_NULL_HANDLE;
        ps->ncb++;
    }
    const VkCommandBuffer cb = ps->cb[ps->used++];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (nvn2_begin(cb, &bi, "vkBeginCommandBuffer (worker chunk)")) return VK_NULL_HANDLE;
    w->rt.cb = cb; w->rt.ar = &ps->ar; w->rt.gen++;   // the worker's memos (g_cmd_gen) see a new command buffer
    return cb;
}
VkResult vk_par_chunk_end(void) {
    vk_worker *const w = tl_worker; if (!w || !w->rt.cb) die("[vk] parallel recording: no chunk to end");
    const VkResult r = vkEndCommandBuffer(w->rt.cb);
    w->rt.cb = VK_NULL_HANDLE; w->rt.ar = NULL;   // (nothing may be recorded or allocated on a worker between chunks)
    return r;
}
int vk_chunk_add(int worker) {
    vksub_t *const s = cur();
    pthread_mutex_lock(&g_chunk_mu);
    if (s->nchunk == s->capchunk) { s->capchunk = s->capchunk ? 2 * s->capchunk : 16; s->chunk = realloc(s->chunk, (size_t)s->capchunk * sizeof *s->chunk); if (!s->chunk) die("[vk] parallel recording: no memory"); }
    const int i = s->nchunk++; s->chunk[i] = (vk_chunk){VK_NULL_HANDLE, worker, 0};
    pthread_mutex_unlock(&g_chunk_mu);
    return i;
}
void vk_chunk_finish(int slot, int i, VkCommandBuffer cb, int ok) {
    pthread_mutex_lock(&g_chunk_mu);
    vksub_t *const s = &g_slot[slot]; if (i < 0 || i >= s->nchunk) die("[vk] parallel recording: chunk %d of slot %d not listed", i, slot);
    s->chunk[i].cb = cb; s->chunk[i].done = ok ? 1 : -1; pthread_cond_broadcast(&g_chunk_cv);
    pthread_mutex_unlock(&g_chunk_mu);
}
int vk_chunk_wait_part(void) {
    vksub_t *const s = cur(); int bad = 0; unsigned joined = 0;
    pthread_mutex_lock(&g_chunk_mu);
    for (int i = s->chunk0; i < s->nchunk; i++) { while (!s->chunk[i].done) pthread_cond_wait(&g_chunk_cv, &g_chunk_mu); bad += s->chunk[i].done < 0;
        if (s->chunk[i].worker >= 0 && s->chunk[i].worker < PAR_MAX) joined |= 1u << s->chunk[i].worker; }
    s->chunk0 = s->nchunk;
    pthread_mutex_unlock(&g_chunk_mu);
    // S1c (parallel recording): the joined workers' counters into the replay thread's (the logs'). A chunk job's vk_chunk_finish is its last count, and the worker records
    // nothing else of this slot until the next part's chunks are added by this thread, so its contexts are not counted into meanwhile
    for (int w = 0; w < g_npar; w++) if (joined >> w & 1) vk_rec_ctx_fold(g_par[w].rec, g_par[w].rec_e);
    return bad;
}
const vk_chunk *vk_chunk_list(int *n) { vksub_t *const s = cur(); *n = s->nchunk; return s->chunk; }   // (the replay thread: the only one that grows the list)
static void par_start(void) {
    const char *e = getenv("DM_VK_PARALLEL"); int nvnh_default_parallel(void); int n = e && *e ? atoi(e) : nvnh_default_parallel(); if (n <= 0) return;   // dkpar: the title's default when unset   // 0 (default): nothing made
    if (n > PAR_MAX) n = PAR_MAX;
    const char *st = getenv("DM_VK_PARALLEL_SELFTEST"); g_par_selftest = st && (atoi(st) == 1 || atoi(st) == 2) ? atoi(st) : 0;   // S1c (parallel recording): 2 adds the stress test
    { const char *pe = getenv("DM_VK_PARALLEL_PIN"); g_par_pin = !(pe && *pe == '0'); }   // S1e (parallel recording)
    g_par = calloc((size_t)n + 1, sizeof *g_par); if (!g_par) die("[vk] parallel recording: no memory for %d workers", n);   // S1e: g_par[n] is the helper context (no thread)
    for (int i = 0; i <= n; i++) { vk_worker *const w = &g_par[i];
        w->id = i; w->rec = vk_rec_ctx_new(); w->rec_e = vk_rec_eden_new();
        pthread_mutex_init(&w->mu, NULL); pthread_cond_init(&w->cv, NULL);
        for (int k = 0; k < NSLOT; k++) {   // (as the slots' pools: buffers resettable one by one, which nvn2_begin's retry needs; reset as a pool in slot_recycle)
            VkCommandPoolCreateInfo ci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
            VKCHK(vkCreateCommandPool(V.dev, &ci, NULL, &w->slot[k].pool)); }
    }
    g_npar = n; g_npar_ctx = n + 1;   // (before the threads: vk_par_run checks it; S1e: the helper context counted for the recycles, stats and folds)
    for (int i = 0; i < n; i++) if (pthread_create(&g_par[i].th, NULL, par_main, &g_par[i])) die("[vk] parallel recording: worker thread %d", i);
    for (int i = 0; i < n; i++) pthread_detach(g_par[i].th);
    VKLOG("S1a: parallel recording: %d worker threads (dm-vk-rec-0..%d) with their own contexts, Eden rings, and per slot a command pool and arena, and a helper context for the "
          "replay thread (S1e: segments recorded on them, %s; DM_VK_PARALLEL=%d%s)", n, n - 1, g_par_pin ? "workers on the P-cores (DM_VK_PARALLEL_PIN=0 off)" : "workers not pinned (DM_VK_PARALLEL_PIN=0)",
          n, g_par_selftest == 2 ? ", self-test and stress self-test at the first submission" : g_par_selftest ? ", self-test at the first submission" : "");
}
// DM_VK_PARALLEL_SELFTEST=1: at the first submission, each worker records an empty chunk of the open slot (and one scratch allocation from its arena), checking that its
// recording target, arena and contexts are its own; one log line. The chunks are never submitted (the slot's next recycle resets them)
typedef struct { int slot, chunk, ok; } par_st;
VkDeviceSize vk_buf_size(VkBuffer b);   // (below)
static void par_selftest_job(void *a) {
    par_st *const t = a; vk_worker *const w = tl_worker;
    const VkCommandBuffer cb = vk_par_chunk_begin(t->slot);
    int ok = cb && vk_cmd() == cb && rec_arena() == &w->slot[t->slot].ar && vk_rec_ctx_cur() == w->rec;
    if (cb) { VkBuffer b; VkDeviceSize o; ok &= vk_scratch(256, 256, &b, &o) && b && vk_buf_size(b) > 0; ok &= vk_par_chunk_end() == VK_SUCCESS; }
    t->ok = ok; vk_chunk_finish(t->slot, t->chunk, cb, ok);
}
// S1c (parallel recording): DM_VK_PARALLEL_SELFTEST=2, after the empty chunks: every worker at once through the shared-state paths of recording (gfx_vk_exec.c
// vk_rec_stress_*: the Eden pipeline get-or-insert and once-publication, the shader objects' draw-time marks, the counters of its own contexts, its scratch), no
// draw recorded; one line `S1c: parallel recording stress self-test: ...` ([RED] on a failure)
static void par_stress_job(void *a) { vk_rec_stress_run((int)(intptr_t)a); }
static void par_stress(void) {
    const uint64_t t0 = now_ns(); void *rec[PAR_MAX], *rec_e[PAR_MAX]; char line[400];
    vk_rec_stress_begin(g_npar);
    for (int i = 0; i < g_npar; i++) { rec[i] = g_par[i].rec; rec_e[i] = g_par[i].rec_e; vk_par_run(i, par_stress_job, (void *)(intptr_t)i); }
    vk_par_drain();
    const int ok = vk_rec_stress_check(rec, rec_e, line, sizeof line);
    VKLOG("%sS1c: parallel recording stress self-test: %s (%.2f ms)", ok ? "" : "[RED] ", line, (now_ns() - t0) / 1e6);
}
static void par_selftest(void) {
    const int stress = g_par_selftest == 2;
    g_par_selftest = -1; par_st t[PAR_MAX]; const uint64_t t0 = now_ns();   // (-1: done; S1c: was 2, now the stress test's mode)
    for (int i = 0; i < g_npar; i++) { t[i] = (par_st){g_cur, vk_chunk_add(i), 0}; vk_par_run(i, par_selftest_job, &t[i]); }
    const int bad = vk_chunk_wait_part(); int n = 0; const vk_chunk *c = vk_chunk_list(&n);
    int okc = 0; for (int i = 0; i < n; i++) okc += c[i].done == 1 && c[i].cb != VK_NULL_HANDLE;
    vk_par_drain();
    VKLOG("%sS1a: parallel recording self-test: %d of %d workers recorded and ended an empty chunk with their own target, arena and contexts (%d failed, %.2f ms)",
          bad || okc != g_npar ? "[RED] " : "", okc, g_npar, bad, (now_ns() - t0) / 1e6);
    pthread_mutex_lock(&g_chunk_mu); cur()->nchunk = cur()->chunk0 = 0; pthread_mutex_unlock(&g_chunk_mu);   // (not part of the submission)
    if (stress) par_stress();   // S1c (parallel recording)
}
VkCommandBuffer vk_upload_cmd(void) {
    vksub_t *s = cur();
    if (!s->up_used) { VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(nvn2_begin(s->up, &bi, "vkBeginCommandBuffer (upload)")); s->up_used = 1; g_up_gen++; og_up_begin(s); }
    return s->up;
}
// The replay queue orders copy-back before the next staged draw. Each submission
// slot owns its buffer until its fence completes; inactive slots release it on recycle.
// stage1 (hybrid3, design\ATOMIC_STAGING_DESIGN.md 2): n bytes of the open submission's device-local staging memory, aligned to al: its buffer, *off. One buffer per
// slot, made larger when a submission needs more (the smaller one freed once the GPU has passed this submission: commands recorded so far still use it)
VkBuffer vk_hy3_mem(VkDeviceSize n, VkDeviceSize al, VkDeviceSize *off) {
    vksub_t *s = cur(); if (!n) return VK_NULL_HANDLE; if (!al) al = 256;
    VkDeviceSize o = (s->hy3_used + al - 1) / al * al;
    if (!s->hy3_b || o + n > s->hy3_cap) {
        VkDeviceSize cap = s->hy3_cap ? 2 * s->hy3_cap : 64ull << 20; while (cap < n) cap *= 2;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, cap, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | (V.has_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0), VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer b; if (vkCreateBuffer(V.dev, &bi, NULL, &b)) return VK_NULL_HANDLE;
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        uint32_t t = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        VkMemoryAllocateFlagsInfo af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, V.has_bda ? &af : NULL, req.size, t}; VkDeviceMemory m;
        if (t == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &m)) { vkDestroyBuffer(V.dev, b, NULL); return VK_NULL_HANDLE; }
        if (vkBindBufferMemory(V.dev, b, m, 0)) { vkDestroyBuffer(V.dev, b, NULL); vkFreeMemory(V.dev, m, NULL); return VK_NULL_HANDLE; }
        if (s->hy3_b) vk_defer_free_buffer(s->hy3_b, s->hy3_m);
        s->hy3_b = b; s->hy3_m = m; s->hy3_cap = cap; o = 0;
        static int nl; if (nl++ < 8) VKLOG("stage1: atomic staging memory of a submission slot: %llu MB device-local", (unsigned long long)(cap >> 20)); }
    s->hy3_used = o + n; *off = o; return s->hy3_b;
}
VkBuffer vk_atomic_scratch(VkDeviceSize n) {
    vksub_t *s = cur();
    if (!n || n > 256 * 1024) return VK_NULL_HANDLE;
    s->atomic_used = 1;
    if (s->atomic_scratch) return s->atomic_scratch;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = 256 * 1024;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (V.has_bda) info.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer buf;
    if (vkCreateBuffer(V.dev, &info, NULL, &buf) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(V.dev, buf, &req);
    uint32_t type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    VkMemoryAllocateFlagsInfo flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, V.has_bda ? &flags : NULL, req.size, type};
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (type == UINT32_MAX || vkAllocateMemory(V.dev, &alloc, NULL, &mem) != VK_SUCCESS ||
        vkBindBufferMemory(V.dev, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(V.dev, buf, NULL);
        if (mem) vkFreeMemory(V.dev, mem, NULL);
        return VK_NULL_HANDLE;
    }
    s->atomic_scratch = buf;
    s->atomic_memory = mem;
    return buf;
}
void *vk_scratch(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off) {
    vk_arena *s = rec_arena(); if (!align) align = 16;   // R3 (parallel recording): the thread's arena (was the slot, cur())
    for (int i = 0; i < s->nblk; i++) {
        scratch_blk *b = &s->blk[i]; VkDeviceSize o = (b->used + align - 1) & ~(align - 1);
        if (o + n <= b->size) { b->used = o + n; *buf = b->b; *off = o; return b->p + o; }
    }
    // S1a (parallel recording): a new block is made in a local and listed under g_ar_mu (a worker's vk_buf_size may read the slot's own list: only whole blocks, never the array moving)
    scratch_blk nb_, *const b = &nb_; memset(b, 0, sizeof *b);
    static VkDeviceSize blk = 0; if (!blk) { const char *e = getenv("DM_VK_SCRATCH_MB"); blk = e && atoi(e) > 0 ? (VkDeviceSize)atoi(e) << 20 : SCRATCH_BLOCK; }
    b->size = n > blk ? (n + 0xffff) & ~0xffffull : blk;
#if defined(DM_NXN1X)
    // NXN1X: with buffer device address on (Eden's descriptor-buffer mode, lane B) Eden's descriptors may name a scratch range by address, so the block has one
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, b->size, POOL_USAGE, VK_SHARING_MODE_EXCLUSIVE};
#else
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, b->size, POOL_USAGE & ~VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_SHARING_MODE_EXCLUSIVE};   // (no device address: its memory is not allocated for one, and the pool table never names it)
#endif
    VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b->b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b->b, &req);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
#if defined(DM_NXN1X)
    if (V.has_bda) ai.pNext = &g_pool_af;
#endif
    VKCHK(nvn2_alloc(&ai, &b->m, "a scratch block's vkAllocateMemory")); VKCHK(vkBindBufferMemory(V.dev, b->b, b->m, 0));
    void *m; VKCHK(vkMapMemory(V.dev, b->m, 0, VK_WHOLE_SIZE, 0, &m)); b->p = m;
    b->used = n; *buf = b->b; *off = 0;
    pthread_mutex_lock(&g_ar_mu);   // S1a (parallel recording)
    if (s->nblk == s->capblk) { s->capblk = s->capblk ? 2 * s->capblk : 8; s->blk = realloc(s->blk, s->capblk * sizeof *s->blk); }
    s->blk[s->nblk++] = *b;
    pthread_mutex_unlock(&g_ar_mu);
    return b->p;
}
VkDeviceSize vk_buf_size(VkBuffer b) {   // r133 (gfx_vk_exec.c DM_VK_X_POFS): a scratch block of the open submission's or a memory pool chunk's buffer size, 0 when not known
    if (g_cur >= 0) { const vk_arena *s = rec_arena(); for (int i = 0; i < s->nblk; i++) if (s->blk[i].b == b) return s->blk[i].size; }   // R3 (parallel recording): the thread's arena
    if (g_cur >= 0 && tl_worker) {   // S1a (parallel recording): a worker may bind a block of the slot's own arena (the replay thread's: e.g. a copy the dry pass made), read under g_ar_mu
        const vk_arena *s = &g_slot[g_cur].ar; VkDeviceSize sz = 0;
        pthread_mutex_lock(&g_ar_mu); for (int i = 0; i < s->nblk && !sz; i++) if (s->blk[i].b == b) sz = s->blk[i].size; pthread_mutex_unlock(&g_ar_mu);
        if (sz) return sz; }
    for (int i = 0; i < g_npool; i++) if (g_pool[i].buf == b) return g_pool[i].hi - g_pool[i].lo;
    return 0;
}
static void df_add(int kind, void *a, void *b) {
    if (g_cur < 0) {   // outside a submission (a pool finalized between frames): after everything submitted so far
        wait_fence(g_fence);
        if (kind == 0) { vkDestroyBuffer(V.dev, (VkBuffer)a, NULL); if (b) vkFreeMemory(V.dev, (VkDeviceMemory)b, NULL); }
        else if (kind == 1) { vkDestroyImage(V.dev, (VkImage)a, NULL); if (b) vkFreeMemory(V.dev, (VkDeviceMemory)b, NULL); }
        else if (kind == 4) { void img_sub_free_(void *); vkDestroyImage(V.dev, (VkImage)a, NULL); img_sub_free_(b); }
        else if (kind == 2) vkDestroyImageView(V.dev, (VkImageView)a, NULL);
        else vkDestroyBufferView(V.dev, (VkBufferView)a, NULL);
        return;
    }
    vksub_t *s = cur(); if (s->ndf == s->capdf) { s->capdf = s->capdf ? 2 * s->capdf : 64; s->df = realloc(s->df, s->capdf * sizeof *s->df); }
    s->df[s->ndf++] = (dfree_t){kind, a, b};
#if defined(DM_NXN1X)
    // A shared-image suballocation does not retire the whole backing block.
    if ((kind == 0 || kind == 1) && b) nvnh_memory_retire((VkDeviceMemory)b);
#endif
}
// Image memory. Small images share blocks: each allocation the driver makes is a mapping of the process (Adreno maps them all), and a game with thousands of textures
// ran towards the kernel's per-process mapping limit (Xenoblade Chronicles 3: about 9,400 driver mappings in its first area, 2,216 of them 172 KB). Images up to
// IMG_SLOT_MAX take a slot of a block (64 slots, 2 MB to IMG_BLOCK bytes), in size classes 2^(k/2) x 4 KB apart (a slot is at most 41% larger than the image); larger ones, or
// ones that need more than 4 KB alignment, get memory of their own
enum { IMG_CLASSES = 18, IMG_BLOCK = 32 << 20 };
#define IMG_SLOT_MAX ((VkDeviceSize)2 << 20)
typedef struct imgblk { VkDeviceMemory m; uint32_t type, cls, nslot, nfree; uint64_t bits[128]; struct imgblk *next; } imgblk;   // bits: 1 = slot free
typedef struct { imgblk *b; uint32_t slot; } imgsub;
static imgblk *g_imgblk[IMG_CLASSES]; static pthread_mutex_t g_imgblk_mu = PTHREAD_MUTEX_INITIALIZER;
// Fill the most occupied compatible block first. List order (newest first)
// otherwise scatters returning slots and lets a few live images pin many blocks.
static imgblk *img_block_reuse(int cls, uint32_t type) {
    imgblk *best=NULL;
    for (imgblk *b=g_imgblk[cls];b;b=b->next)
        if (b->type==type && b->nfree && (!best ||
            (uint64_t)b->nfree*best->nslot<(uint64_t)best->nfree*b->nslot)) best=b;
    return best;
}
static VkDeviceSize img_class_size(int c) { return (c & 1 ? 6144ull : 4096ull) << (c >> 1); }   // 4K, 6K, 8K, 12K, 16K, 24K, ... 2M
uint64_t vk_image_memory_growth(const VkMemoryRequirements *req, uint32_t type) {
    int c = 0; while (c < IMG_CLASSES && img_class_size(c) < req->size) c++;
    if (c == IMG_CLASSES || req->size > IMG_SLOT_MAX || req->alignment > 4096) return req->size;
    pthread_mutex_lock(&g_imgblk_mu);
    imgblk *b = img_block_reuse(c,type);
    int reuse = b != NULL; pthread_mutex_unlock(&g_imgblk_mu);
    if (reuse) return 0;
    VkDeviceSize bytes = img_class_size(c) * 64;
    if (bytes < (2 << 20)) bytes = 2 << 20;
    if (bytes > IMG_BLOCK) bytes = IMG_BLOCK;
#if defined(DM_NXN1X)
    bytes=nvnh_memory_block_size(type,bytes,2ull<<20,img_class_size(c));
#endif
    return bytes / img_class_size(c) * img_class_size(c);
}
VkDeviceMemory vk_image_memory(const VkMemoryRequirements *req, uint32_t type, VkDeviceSize *off, void **sub) {
    *off = 0; *sub = NULL;
    int c = 0; while (c < IMG_CLASSES && img_class_size(c) < req->size) c++;
    if (c == IMG_CLASSES || req->size > IMG_SLOT_MAX || req->alignment > 4096) {   // memory of its own
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req->size, type}; VkDeviceMemory m;
        return vkAllocateMemory(V.dev, &ai, NULL, &m) ? VK_NULL_HANDLE : m;
    }
    imgsub *s = malloc(sizeof *s);
    if (!s) return VK_NULL_HANDLE;
    pthread_mutex_lock(&g_imgblk_mu);
    imgblk *b = img_block_reuse(c,type);
    if (!b) {
        b = calloc(1, sizeof *b);
        if (!b) { pthread_mutex_unlock(&g_imgblk_mu); free(s); return VK_NULL_HANDLE; }
        b->type = type; b->cls = (uint32_t)c; VkDeviceSize bs = img_class_size(c) * 64; if (bs < (2 << 20)) bs = 2 << 20; if (bs > IMG_BLOCK) bs = IMG_BLOCK;
#if defined(DM_NXN1X)
        bs=nvnh_memory_block_size(type,bs,2ull<<20,img_class_size(c));
#endif
        b->nslot = (uint32_t)(bs / img_class_size(c));
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, (VkDeviceSize)b->nslot * img_class_size(c), type};
        if (vkAllocateMemory(V.dev, &ai, NULL, &b->m)) { pthread_mutex_unlock(&g_imgblk_mu); free(b); free(s); return VK_NULL_HANDLE; }
        for (uint32_t i = 0; i < b->nslot; i++) b->bits[i >> 6] |= 1ull << (i & 63);
        b->nfree = b->nslot; b->next = g_imgblk[c]; g_imgblk[c] = b;
    }
    uint32_t i = 0; while (!b->bits[i >> 6]) i += 64; i += (uint32_t)__builtin_ctzll(b->bits[i >> 6]);
    b->bits[i >> 6] &= ~(1ull << (i & 63)); b->nfree--;
    pthread_mutex_unlock(&g_imgblk_mu);
    s->b = b; s->slot = i; *sub = s; *off = (VkDeviceSize)i * img_class_size(c);
    return b->m;
}
static void img_sub_free(imgsub *s) {   // called only after the owning submission's CPU completion
    pthread_mutex_lock(&g_imgblk_mu);
    imgblk *b=s->b; b->bits[s->slot>>6]|=1ull<<(s->slot&63); b->nfree++;
    // One warm empty block per type/class is enough for reuse. An empty block
    // has no live slot or pending retirement; never relocate any live image.
    if (b->nfree==b->nslot) {
        int spare=0;
        for (imgblk *p=g_imgblk[b->cls];p;p=p->next)
            if (p!=b && p->type==b->type && p->nfree==p->nslot) { spare=1; break; }
        if (spare) {
            imgblk **at=&g_imgblk[b->cls];while (*at && *at!=b) at=&(*at)->next;
            if (*at) { *at=b->next;vkFreeMemory(V.dev,b->m,NULL);free(b); }
        }
    }
    pthread_mutex_unlock(&g_imgblk_mu); free(s);
}
uint64_t vk_image_allocation_size(void *allocation, uint64_t bytes, uint32_t *live_slots) {
    if (!allocation) { *live_slots = 1; return bytes; }
    imgsub *s = allocation;
    pthread_mutex_lock(&g_imgblk_mu);
    *live_slots = s->b->nslot - s->b->nfree;
    bytes = img_class_size((int)s->b->cls) * s->b->nslot;
    pthread_mutex_unlock(&g_imgblk_mu);
    return bytes;
}
uint64_t vk_image_trim_empty(void) {
    uint64_t bytes = 0;
    pthread_mutex_lock(&g_imgblk_mu);
    for (int c = 0; c < IMG_CLASSES; c++) {
        imgblk **at = &g_imgblk[c];
        while (*at) {
            imgblk *b = *at;
            if (b->nfree != b->nslot) { at = &b->next; continue; }
            *at = b->next; bytes += img_class_size(c) * b->nslot;
            vkFreeMemory(V.dev, b->m, NULL); free(b);
        }
    }
    pthread_mutex_unlock(&g_imgblk_mu);
    return bytes;
}
uint64_t vk_image_release_completed(VkImage image, VkDeviceMemory memory, void *allocation, uint64_t bytes) {
    vkDestroyImage(V.dev, image, NULL);
    if (!allocation) { if (memory) vkFreeMemory(V.dev, memory, NULL); return memory ? bytes : 0; }
    imgsub *s = allocation; imgblk *b = s->b;
    pthread_mutex_lock(&g_imgblk_mu);
    b->bits[s->slot >> 6] |= 1ull << (s->slot & 63); b->nfree++;
    uint64_t freed = 0;
    if (b->nfree == b->nslot) {
        imgblk **at = &g_imgblk[b->cls]; while (*at && *at != b) at = &(*at)->next;
        if (*at) { *at = b->next; freed = img_class_size((int)b->cls) * b->nslot; vkFreeMemory(V.dev, b->m, NULL); free(b); }
    }
    pthread_mutex_unlock(&g_imgblk_mu); free(s);
    return freed;
}
void vk_defer_free_buffer(VkBuffer b, VkDeviceMemory m) { df_add(0, (void *)b, (void *)m); }
void vk_defer_free_image_sub(VkImage i, void *sub) { df_add(4, (void *)i, sub); }
void img_sub_free_(void *s) { img_sub_free((imgsub *)s); }
void vk_defer_free_image(VkImage i, VkDeviceMemory m) { df_add(1, (void *)i, (void *)m); }
void vk_defer_free_view(VkImageView v) { g_views_freed++; df_add(2, (void *)v, NULL); }
void vk_defer_free_bview(VkBufferView v) { df_add(3, (void *)v, NULL); }   // r75 (df_add and the submission's free list already handled kind 3)
// Descriptor pools: the allocator counts what each set takes and moves to a new pool before any count runs out. Adreno's driver does not refuse an
// allocation past a pool's sizes: it succeeds and the submission then fails as a whole at vkEndCommandBuffer (VK_ERROR_OUT_OF_HOST_MEMORY; NEO's
// handheld frames with 32-slot texture arrays crossed the old 8192 combined-sampler limit at about the 712th draw)
enum { DP_SETS = 4096 };
static const uint32_t DP_CAP[9] = {[VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC] = 16384, [VK_DESCRIPTOR_TYPE_SAMPLER] = 1024, [VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] = 65536, [VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE] = 1024,
    [VK_DESCRIPTOR_TYPE_STORAGE_IMAGE] = 2048, [VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER] = 4096, [VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER] = 2048,
    [VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] = 32768, [VK_DESCRIPTOR_TYPE_STORAGE_BUFFER] = 4096};
static nvn_hmap g_dsl_need = {.klen = sizeof(VkDescriptorSetLayout)}; static pthread_mutex_t g_dsl_mu = PTHREAD_MUTEX_INITIALIZER;
void vk_dsl_register(VkDescriptorSetLayout l, const VkDescriptorSetLayoutBinding *b, uint32_t n) {   // what one set of this layout takes, by type
    uint32_t *need = calloc(9, sizeof *need);
    for (uint32_t i = 0; i < n; i++) if ((uint32_t)b[i].descriptorType < 9) need[b[i].descriptorType] += b[i].descriptorCount;
    pthread_mutex_lock(&g_dsl_mu); nvn_hm_put(&g_dsl_need, &l, need); pthread_mutex_unlock(&g_dsl_mu);
}
VkDescriptorSet vk_alloc_set(VkDescriptorSetLayout l) {
    vk_arena *s = rec_arena();   // R3 (parallel recording): the thread's arena (was the slot, cur())
    pthread_mutex_lock(&g_dsl_mu); const uint32_t *need = nvn_hm_get(&g_dsl_need, &l); pthread_mutex_unlock(&g_dsl_mu);
    for (;;) {
        if (s->dp_cur == s->ndp) {
            VkDescriptorPoolSize sz[9]; uint32_t nz = 0;
            for (int t = 0; t < 9; t++) if (DP_CAP[t]) sz[nz++] = (VkDescriptorPoolSize){(VkDescriptorType)t, DP_CAP[t]};
            VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, DP_SETS, nz, sz};
            if (s->ndp == s->capdp) { s->capdp = s->capdp ? 2 * s->capdp : 4; s->dp = realloc(s->dp, s->capdp * sizeof *s->dp); }
            VKCHK(vkCreateDescriptorPool(V.dev, &pi, NULL, &s->dp[s->ndp++]));
        }
        if (s->dp_cur != s->dp_counted) { memset(s->dp_used, 0, sizeof s->dp_used); s->dp_sets = 0; s->dp_counted = s->dp_cur; }
        int fits = s->dp_sets < DP_SETS;
        if (need) for (int t = 0; t < 9 && fits; t++) if (need[t] && s->dp_used[t] + need[t] > DP_CAP[t]) fits = 0;
        if (!fits) { s->dp_cur++; continue; }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, s->dp[s->dp_cur], 1, &l};
        VkDescriptorSet set; VkResult r = vkAllocateDescriptorSets(V.dev, &ai, &set);
        if (r == VK_SUCCESS) { s->dp_sets++; if (need) for (int t = 0; t < 9; t++) s->dp_used[t] += need[t]; return set; }
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) die("[vk] vkAllocateDescriptorSets: %d", r);
        s->dp_cur++;
    }
}

// ---------------- submissions: begin, CPU-visible effects, commit ----------------
static void post(post_op o) { if (tl_wv) { par_post(tl_wv, o); return; }   // S1e (parallel recording): a worker's into its segment's view
    vksub_t *s = cur(); o.wseq = s->wseq; if (s->npost == s->cappost) { s->cappost = s->cappost ? 2 * s->cappost : 64; s->post = realloc(s->post, s->cappost * sizeof *s->post); } s->post[s->npost++] = o; }
void mtl_init(void) { vk_init(); }
static int g_shadow_pools;
uint64_t g_ns_subwait, g_ns_poolsync, g_nsubs;   // r30: time a submission's start waited for the last one (copied pools), time syncing the copies, submissions
#if defined(DM_NXN1X)
// replay2 (11-21; rptfix replaydiag: the content observer drain 4.6 ms a frame of the replay thread): nvn_memory_content.c is an observation ledger only ("F2
// observation only. None of these records authorizes an upload, free or writeback"): nothing decides from it (nvnc_probe is called by its tests alone, nvnc_get_stats
// by log lines), its producers are lock-free bounded queues that need no consumer to stay correct (a full queue only marks the ledger Unknown), and nvnc_drain is
// a single consumer behind its own trylock. So it has no ordering duty for the next recording, and it runs on a thread of its own, woken at each submission's
// start (requests while it drains fold into one more pass). It takes no submission or image lock; the resolver's owner lock it may wait for, as before, from
// a thread that holds nothing else. DM_VK_DRAIN_ASYNC=0: drained on the replay thread at each submission's start as before
static int drain_async_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_DRAIN_ASYNC"); v = !(e && *e == '0'); VKLOG("replay2: content observer drain %s (DM_VK_DRAIN_ASYNC=%d)", v ? "on its own thread" : "on the replay thread", v); } return v; }
static pthread_mutex_t g_drq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_drq_cv = PTHREAD_COND_INITIALIZER; static int g_drq, g_drq_started;
static void *drain_main(void *a) {
    (void)a; pthread_setname_np(pthread_self(), "dm-vk-observer");
    extern int g_rdiag; extern uint64_t g_rd_ns[64], g_rd_n[64];
    for (;;) {
        pthread_mutex_lock(&g_drq_mu); while (!g_drq) pthread_cond_wait(&g_drq_cv, &g_drq_mu); g_drq = 0; pthread_mutex_unlock(&g_drq_mu);
        const int rd = g_rdiag > 0; nvnc_stats cs0 = {0}; if (rd) nvnc_get_stats(&cs0); const uint64_t t0 = rd ? now_ns() : 0;
        nvnc_drain(8192, 1024);
        if (rd) { g_rd_ns[49] += now_ns() - t0; g_rd_n[49]++; nvnc_stats cs1 = {0}; nvnc_get_stats(&cs1); g_rd_n[18] += cs1.consumed >= cs0.consumed ? cs1.consumed - cs0.consumed : 0; }   // replaydiag: slot 49 the thread's drain time, 18 events
    }
    return NULL;
}
static void drain_request(void) {
    pthread_mutex_lock(&g_drq_mu);
    if (!g_drq_started) { pthread_t t; if (pthread_create(&t, NULL, drain_main, NULL)) { pthread_mutex_unlock(&g_drq_mu); nvnc_drain(8192, 1024); return; } pthread_detach(t); g_drq_started = 1; }
    g_drq = 1; pthread_cond_signal(&g_drq_cv); pthread_mutex_unlock(&g_drq_mu);
}
#endif
void mtl_sub_begin(void) {
    ss_begin();   // s15
    vk_init();
    const int rd = rd_on(); if (rd) vk_rd_start();   // replaydiag (DM_VK_REPLAYDIAG=1): slots 0-8 below
#if defined(DM_NXN1X)
    if (dm_sdk_ounce() && nvnh_memory_policy(NVNH_POLICY_ENABLE,0)) reclaim_completed_slots();
    if (rd) vk_rd_mark(0);   // replaydiag: slot reclaim
    if (drain_async_on()) { drain_request(); if (rd) vk_rd_mark(17); }   // replay2: the drain on its own thread (slot 17: the request only)
    else { nvnc_stats cs0 = {0}; if (rd) nvnc_get_stats(&cs0);
      nvnc_drain(8192, 1024); // bounded observer, outside the submission/owner locks
      if (rd) { vk_rd_mark(17); nvnc_stats cs1 = {0}; nvnc_get_stats(&cs1); g_rd_n[18] += cs1.consumed >= cs0.consumed ? cs1.consumed - cs0.consumed : 0; } }   // replaydiag: the content observer drain (slot 18: events)
#else
    if (rd) vk_rd_mark(0);   // replaydiag: (no NXN1X: slot reclaim, content observer drain)
#endif
    if (dm_sdk_ounce()) {
        vk_tex_archive_prepare(); vk_tex_preserve_poll(0);
        extern uint64_t g_tex_frame;
        static uint64_t budget_frame=UINT64_MAX, cache_budget;
        if (budget_frame!=g_tex_frame) {
            uint64_t use=0; cache_budget=0;
            vk_heap_worst(&use,&cache_budget); budget_frame=g_tex_frame;
        }
        vk_tex_trim_released(cache_budget);
    }
#if defined(DM_NXN1X)
    if (dm_sdk_ounce() && nvnh_memory_policy(NVNH_POLICY_ENABLE,1)) {
        const uint64_t need=nvnh_memory_policy(NVNH_POLICY_TAKE_PRESSURE,0);
        if (need) {
            vk_tex_preserve_poll(need);
            const uint64_t freed=vk_tex_reclaim_completed(need);
            nvnh_memory_policy(NVNH_POLICY_RECLAIMED,freed);
        }
    }
#endif
    if (rd) vk_rd_mark(1);   // replaydiag: texture archive / preserve / trim, memory pressure
    nvnr_lock(&g_sub_lock);
    g_sub_owned=1;
    // the next slot of the ring (g_cur is -1 between submissions, so it cannot say where the ring is: every submission took slot 0 and waited in slot_recycle
    // for the one before to finish on the GPU, which kept the replay of a frame from overlapping the GPU work of the last)
    static int next; if (next >= nslot_on()) next = 0; g_cur = next; next = (next + 1) % nslot_on(); vksub_t *s = &g_slot[g_cur];   // slot12: the slots in use
    g_vk_rt_main.cb = s->main;   // R2 (parallel recording): the replay thread's target follows g_cur (as cur()->main did)
    nvf0_submission diag = f0_submission(); g_f0_sub = &diag;
    nvf0_scope slot_wait; vk_f0_begin(&slot_wait, NVF0_SLOT_WAIT, 0);
    nvn_pwait_enter(6, "replay: a submission slot (slot_recycle, the GPU)");   // pwait
    slot_recycle(s);
    nvn_pwait_leave(6);   // pwait
    vk_f0_end(&slot_wait, 0);
    s->f0 = diag; g_f0_sub = &s->f0;
    if (rd) vk_rd_mark(2);   // replaydiag: submission lock, slot_recycle
    g_npass[g_cur] = 0; og_sub_begin();   // ootgap (before anything is recorded: the pool syncs below record too)
#if defined(DM_NXN1X)
    s->content_sub = nvnc_sub_begin();
#endif
    s->npost = 0; s->up_used = 0; s->ngw = 0; s->ngml = 0; s->nwr = 0; s->wr_over = 0; s->wseq = 0;   // (combined1120)
    g_pool_unknown_write_start = g_sbufu_w;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    s->main = s->mains[0]; s->nmain_used = 1; s->cb_err = 0; s->up = s->ups[0]; s->nup_used = 1; s->nmain_sent = 0; g_vk_rt_main.cb = s->main; s->nrcb_used = 0;   // R2 (parallel recording); S1e: no R chunk taken yet
    VKCHK(nvn2_begin(s->main, &bi, "vkBeginCommandBuffer (main)")); g_vk_rt_main.gen++;   // R2 (parallel recording): was g_cmd_gen++
    if (g_par_selftest > 0) par_selftest();   // S1a (parallel recording): DM_VK_PARALLEL_SELFTEST=1 with workers, once (S1c: =2 adds the stress test)
    g_pool_sync_epoch = g_nsubs + 1; g_pool_demand_pending = 0;
    // Mirrors finish in completion_main before g_done is published. Keep the
    // latest-completion wait below when mirrors are enabled, but they do not
    // require scanning copied pools that this submission never uses.
    g_pool_demand_active = (pool_sync_mode() & 1) && g_pool_demand && g_pool_compare_valid &&
        !getenv("DM_VK_COPYBACK_FULL") && !vk_poolcopy_on();   // poolcopy: every copy synced at the submission's start (the review's reference path)
    static int demand_previous = -1;
    if (demand_previous != g_pool_demand_active && dm_sdk_ounce()) {
        VKLOG("NVN2 pool demand effective: %s; mode %d, requested %d, comparison valid %d, full copyback %d, mirror completion wait %d",
            g_pool_demand_active ? "active" : "eager", pool_sync_mode(), g_pool_demand,
            g_pool_compare_valid, getenv("DM_VK_COPYBACK_FULL") != NULL, s56_any());
        demand_previous = g_pool_demand_active;
    }
    // copied pools: what the game wrote before submitting goes into the GPU copy before any part of the submission can reach the GPU (DM_VK_CB_DRAWS sends
    // parts early; synced at the commit, those parts read the copy as the last submission left it), and only once no earlier submission still reads it
    if (rd) vk_rd_mark(3);   // replaydiag: command buffer begun
    hy_apply();   // hybrid1: chunks an atomic reached in earlier submissions become device-local copies now (not synced yet: the scan below waits for every earlier submission)
    if (rd) vk_rd_mark(4);   // replaydiag: hy_apply
    pool_rdlock(); int shadow = 0, first_sync = 0, legacy = 0; for (int i = 0; i < g_npool; i++) if (g_pool[i].shadow) { shadow = 1; first_sync |= !g_pool[i].synced; legacy |= !g_pool[i].dl; } pool_unlock();   // legacy: a copied pool that is not a device-local copy (its GPU reads its shadow: the wait stays)
    g_shadow_pools = shadow;
    if (rd) { vk_rd_mark(5); g_rd_n[9] += (uint64_t)g_npool; }   // replaydiag: the pool scan (slot 9 counts the pools scanned)
    uint64_t dependency = g_fence;
    if ((pool_sync_mode() & 2) && !first_sync && !getenv("DM_VK_COPYBACK_FULL") && !vk_poolcopy_on()) {   // poolcopy: the latest completion (the shadow is then the copy's content)
        dependency = atomic_load(&g_pool_sync_fence);
        if (dependency > g_fence) dependency = g_fence;
    }
    if (vk_poolcopy_on() && pc_sync() != PC_SERIAL) {   // stage2c: no wait for the previous submission, except (async) where a copy is synced in full: a new copy, after a
        // refused submission, after a copy ran out of pre/post memory. A full sync overwrites the whole device-local copy from the game's pages, so every merge of
        // earlier submissions (their GPU results, into the game's pages) must have run first
        dependency = legacy ? g_fence : pc_async() && (first_sync || g_pc_drain) ? (g_pc_drain > g_fence || first_sync ? g_fence : g_pc_drain) : 0;
        if (dependency) g_pc_drains++;
        g_pc_drain = 0;
    }
    if (shadow) {
      if (dependency < g_fence) g_pool_sync_narrow++; else g_pool_sync_fullwait++;
      struct timespec t0, t1, t2; clock_gettime(CLOCK_MONOTONIC, &t0);
      nvf0_scope pool_wait, pool_sync; vk_f0_begin(&pool_wait, NVF0_POOL_WAIT, 0);
      if (dependency && !mtl_wait(dependency, ~0ull)) die("[vk] copied-pool synchronization dependency did not complete");
      vk_f0_end(&pool_wait, 0);
      clock_gettime(CLOCK_MONOTONIC, &t1); vk_f0_begin(&pool_sync, NVF0_POOL_SYNC, 0);
      vk_pools_sync(NULL); vk_f0_end(&pool_sync, 0); clock_gettime(CLOCK_MONOTONIC, &t2);   // completion includes copy-back, not just the GPU fence
      g_ns_subwait += (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec)); g_ns_poolsync += (uint64_t)((t2.tv_sec - t1.tv_sec) * 1000000000ll + (t2.tv_nsec - t1.tv_nsec)); }   // r30: measured (NPARK: GPU 50 %)
    if (rd) vk_rd_mark(6);   // replaydiag: copied-pool wait and sync
    g_nsubs++; pool_budget();   // r78
    if (g_nsubs % 120 == 0) hy_report();   // hybrid1 (diagnostic lines, every mode)
    if (rd) vk_rd_mark(7);   // replaydiag: pool_budget, hy_report
    if (g_tsq) { vkCmdResetQueryPool(s->main, g_tsq, 2 * (uint32_t)g_cur, 2); vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_tsq, 2 * (uint32_t)g_cur); }
    // On first use (and after a dropped submission) initialize the entire range.
    // Afterwards only the prefix written by the last use needs another reset.
    uint32_t cq_reset = s->queries_initialized ? (uint32_t)s->ncq : CQ;
    uint32_t oq_reset = s->queries_initialized ? (uint32_t)(s->noq < OQ ? s->noq : OQ) : OQ;   // (S1e (parallel recording): noq is taken by atomic adds, which may pass OQ by the threads racing at it)
    if (g_cq && cq_reset) vkCmdResetQueryPool(s->main, g_cq, (uint32_t)g_cur * CQ, cq_reset);
    s->ncq = 0;
    s->noq = 0; if (g_oq) { if (g_oq_active >= 0) { VKRED_ONCE(99, "occlusion query still active at a submission start"); g_oq_active = -1; } if (oq_reset) vkCmdResetQueryPool(s->main, g_oq, (uint32_t)g_cur * OQ, oq_reset); }
    const uint32_t pass_reset = s->queries_initialized ? s->pass_queries_written : PASSQ * 2;
    if (pass_times_on() && pass_reset)
        vkCmdResetQueryPool(s->main, g_pq, (uint32_t)g_cur * PASSQ * 2, pass_reset);
    s->pass_queries_written = 0;
    g_npass[g_cur] = 0;
    if (g_occdiag || g_occ2) occ_sub_begin();   // occ-s0 / occ-s1: the last submission's overflow context, per-submission counts, the 600-present lines
    vk_exec_reset();
    if (rd) vk_rd_mark(8);   // replaydiag: timestamps, query resets, the rest
    ss_begun();   // s15
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, -1}); }
static void oq_end(void) {   // inside the pass (or outside, for none) the query was begun in
    if (g_occ2 >= 2) { vk_occ2_end(); return; }   // occ-s2: no span, no post: the open draw's ring query ended
    if (g_occ2) { occ2_span_end(); return; }   // occ-s1: the open draw's ring query ended, the old span posted as a serial range
    if (g_oq_active < 0) return;
    const uint32_t ep = tl_wv ? tl_wv->oq_epoch : g_oq_epoch;   // S1e (parallel recording): a worker's segment's epoch (frozen at its hand-out)
    vkCmdEndQuery(vk_cmd(), g_oq, (uint32_t)g_oq_active); post((post_op){4, NULL, 0, ep, 0, 0, g_oq_active}); g_oq_active = -1;
}
void vk_oq_draw(VkCommandBuffer cb) {   // gfx_vk_exec.c draw and draw_indexed, with the pass open
    // S1e (parallel recording): a worker counts as its segment's hand-out said (counting, epoch), and takes its query by an atomic add: the slots are the submission's in
    // the order they were taken, which no longer follows the stream; every slot taken is begun and ended (its pass's end), and each post names its own
    if (g_occ2) return;   // occ-s1: no span query (vk_occ2_begin: the draw's own, and the span as serials)
    const int counting = tl_wv ? tl_wv->oq_counting : g_oq_counting; const uint32_t ep = tl_wv ? tl_wv->oq_epoch : g_oq_epoch;
    if (!g_oq || !counting || g_oq_active >= 0) return;
    vksub_t *s = cur();
    const int q = __atomic_load_n(&s->noq, __ATOMIC_RELAXED) >= OQ ? OQ : __atomic_fetch_add(&s->noq, 1, __ATOMIC_RELAXED);
    if (q >= OQ) { if (tl_wv || g_oq_lost_ep != ep) { if (!tl_wv) g_oq_lost_ep = ep; post((post_op){4, NULL, 1, ep, 0, 0, -1}); } VKRED_ONCE(100, "occlusion queries: more than %d passes counted in one submission", OQ);
        if (g_occdiag && !tl_wv) occd_ovf(); return; }   // occ-s0: the submission's context (logged at the next submission's start)
    g_oq_active = g_cur * OQ + q; vkCmdBeginQuery(cb, g_oq, (uint32_t)g_oq_active, g_oq_flags);
}
void vk_oq_pass_end(void) { oq_end(); }
static void occ2_reset_at(void); static uint64_t g_od_sub_rst;
void vk_oq_reset(uint32_t type) { if (g_occ2 >= 2) { if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED) { occ2_reset(); g_od_sub_rst++; } return; }   // occ-s2: a new window (no epoch, no g_oq_counting)
    if (!g_oq || type != NVN_COUNTER_TYPE_SAMPLES_PASSED) return; oq_end(); g_oq_epoch++; g_oq_counting = 1;
    if (g_occ2) occ2_reset_at();   // occ-s1: the window starts at this serial
    g_od_sub_rst++; }   // occ-s0: per submission (overflow context)
// rptfix (OoT 11-21; HeyListen engine/gpu/vk/vk_replay.cpp NOP_REPORT_COUNTER): counter reports written into the game's memory as the replay reaches them, in stream
// order, instead of by the completion thread after the submission's fence. OoT reads report words that are new every frame; written after the fence, a read before
// that found the memory's old value: its sun visibility followed the occlusion counts (the light trembled; DM_VK_OQ=0 "much better", not gone, its constant still late),
// and the completion thread's ~1,500-2,000 occlusion writes a frame held the submission slots (4-5 ms of the replay's wait).
//   occlusion (SAMPLES_PASSED): a constant, as HeyListen's 4096 ("everything visible": their per-draw estimate made OoT's sunlight leave the ground for a frame);
//     DM_VK_OCC_FIXED=<n> that value, -1 the real queries written after the fence as before (r50)
//   other counter types (not measured, written as 0): {0, now} at once
//   timestamps: on the GPU path as before (their value is the GPU's time); DM_VK_TS_REPLAY=1 writes them at once too, on a clock running at 0.1 x real time from
//     each present (HeyListen RX_VK_TS_SCALE: replay-time stamps measure the replay, which moved Mario Kart World's dynamic resolution every frame)
// A report over bytes a GPU write still in flight covers (vk_gpu_pending) is left to the completion thread, after the fence, as before
static uint64_t gpu_ticks(void);
uint64_t guest_ns(void);
int64_t vk_occ_fixed(void) {
    static int64_t v = INT64_MIN;
    if (v == INT64_MIN) { const char *e = getenv("DM_VK_OCC_FIXED"); v = e && *e ? atoll(e) : 4096; if (v < 0) v = -1;
        VKLOG("rptfix: occlusion reports %s (DM_VK_OCC_FIXED=<n>; -1 real queries written after the fence)", v >= 0 ? "a constant written at replay" : "real queries, written after the fence"); }
    return v;
}
static int ts_replay_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_TS_REPLAY"); v = e && *e == '1'; if (v) VKLOG("rptfix: timestamps written at replay on a 0.1 x clock from each present (DM_VK_TS_REPLAY=1)"); } return v; }
uint64_t g_rpt_now, g_rpt_late;   // reports written at replay; left to the completion thread (a GPU write over them in flight)
// replay2 (11-21; rptfix replaydiag: REPORT_COUNTER 12.1 ms a frame on the replay thread, ~2,500 reports at 4.8 us each): each report scanned the GPU write list
// (vk_gpu_pending, up to 1,024 entries), and its dm_guest_write_prepare walked every dirty-tracking range (nvn_dirty.c prepare_write_range: all of g_nr, per call)
// and logged the bytes twice (nvnc_cpu here and in prep_write). Now the reports are queued in stream order with what was recorded before each (the GPU write list's
// g_gw_seq, the submission's post count and wseq) and written once per submission (rpt_flush): before any vkQueueSubmit of it (a part sent early, the commit) and
// before a CPU read of guest memory that stands for a GPU read at this point of the stream (vk_ubd_read), so nothing that reads them can see them later than before.
// Render conditions never read these words at replay (rc_value: occlusion and other counters unknown, a timestamp only once its fence is done). The flush: one scan
// of the GPU write list for the batch's whole range (per report only when that range is hit), sorted by address (stable: the later report at the same address wins),
// one prepare and one ledger record per run of adjacent pages. A report a GPU write recorded before it (or in flight) covers is left to the completion thread as
// before, put back at its own place among the submission's post ops. DM_VK_RPT_BATCH=0: written one by one at replay as rptfix
typedef struct { uint64_t addr, v0, ticks, gwseq; uint32_t type, wseq, npost, ord; } rpt_q;
static rpt_q *g_rq; static uint32_t g_nrq, g_caprq; uint64_t g_rq_flushes, g_rq_runs, g_rq_lo, g_rq_hi;   // lo/hi: the queued reports' bytes (sub_gpu_wrote)
static int rpt_batch_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_RPT_BATCH"); v = !(e && *e == '0'); VKLOG("replay2: replay-time reports %s (DM_VK_RPT_BATCH=%d)", v ? "queued and written once per submission" : "written one by one (rptfix)", v); } return v; }
void vk_rpt_flush(void);
static int report_now(uint64_t addr, uint64_t v0, uint64_t ticks, uint32_t type, int batch) {
    int vk_gpu_pending(uint64_t a, uint64_t n);
    if (addr && batch && rpt_batch_on() && g_cur >= 0) {   // replay2
        vksub_t *s = cur(); extern uint64_t g_gw_seq;
        if (g_nrq == g_caprq) { uint32_t c = g_caprq ? 2 * g_caprq : 1024; void *p = realloc(g_rq, (size_t)c * sizeof *g_rq); if (!p) goto one; g_rq = p; g_caprq = c; }
        if (!g_nrq || addr < g_rq_lo) g_rq_lo = addr; if (!g_nrq || addr + 16 > g_rq_hi) g_rq_hi = addr + 16;
        g_rq[g_nrq] = (rpt_q){addr, v0, ticks, g_gw_seq, type, s->wseq, (uint32_t)s->npost, g_nrq}; g_nrq++;
        return 1;
    }
one:
    if (!addr || vk_gpu_pending(addr, 16)) { g_rpt_late++; return 0; }
    if (type & RQ_HALF) {   // occ-s2: the timestamp half only
#if defined(DM_NXN1X)
        nvnc_cpu(addr + 8, 8, NVNC_GPU_WRITEBACK);
#endif
        dm_guest_write_prepare((void *)(uintptr_t)(addr + 8), 8); memcpy((void *)(uintptr_t)(addr + 8), &ticks, 8); g_rpt_now++; return 1; }
    const uint64_t rec[2] = {v0, ticks};
#if defined(DM_NXN1X)
    nvnc_cpu(addr, sizeof rec, NVNC_GPU_WRITEBACK);
#endif
    dm_guest_write_prepare((void *)(uintptr_t)addr, sizeof rec);
    memcpy((void *)(uintptr_t)addr, rec, sizeof rec);
    g_rpt_now++; return 1;
}
static uint64_t ts_replay_ticks(void) {   // HeyListen's: base = this present's first report; then 0.1 x the real time since
    extern uint64_t g_tex_frame; static uint64_t frame = UINT64_MAX, base;
    const uint64_t ns = guest_ns(); if (frame != g_tex_frame) { frame = g_tex_frame; base = ns; }
    return nvn_timestamp_from_ns(base + (ns - base) / 10);
}
// occkey (11-21, NPARK: Kokiri's haze over the whole picture at one spot with the constant 4096, gone with 0, right but trembling with the real counts written after the
// fence): HeyListen vk_draw.cpp TICK-033 RX_VK_OCC_REAL=5. OoT's report addresses change every frame, but the k-th report of a frame tests the same object as the k-th of
// the frame before (99% agree). A report's key: the program bound at it, the draws since its reset, and its rank among this present's reports with those two
// (gfx_vk_exec.c g_occ_key). The real count of a key (the occlusion queries, summed by the completion thread) goes into a table, and each report is written at replay,
// in stream order, with its key's count from at most 3 presents before (else the fixed value): always about one frame late, never racing the game's read.
// DM_VK_OCC_KEYED=0: the fixed value (rptfix); DM_VK_OCC_FIXED=-1 still writes the real counts after the fence
int vk_occ_keyed_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_OCC_KEYED"); v = !(e && *e == '0') && vk_occ_fixed() >= 0;
    VKLOG("occkey: occlusion reports %s (DM_VK_OCC_KEYED=%s)", v ? "take the real count of the same report one frame before, written at replay" : "as DM_VK_OCC_FIXED says", e && *e ? e : "1 (default)"); } return v; }
enum { OCCK_N = 8192 };
typedef struct { uint64_t key, frame, val; } occk_e;
static occk_e g_occk[OCCK_N]; static pthread_mutex_t g_occk_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_occk_hit, g_occk_miss, g_occk_put, g_occk_flip, g_occk_zero, g_occk_logf, g_occk_ahit;
uint64_t g_occ_key, g_occ_key2;   // gfx_vk_exec.c (key2: occaddr): the key of the report being replayed (0: none)
static uint64_t g_occk_lastf;   // occ-s0: the present the last hit's count was measured at (its report's replay)
static int occk_get(uint64_t key, uint64_t frame, uint64_t *v, int count) {
    int ok = 0; pthread_mutex_lock(&g_occk_mu);
    for (unsigned k = 0, i = (unsigned)(key * 0x9E3779B97F4A7C15ull >> 51); k < 16; k++, i = (i + 1) & (OCCK_N - 1)) {
        const occk_e *e = &g_occk[i]; if (e->key != key) { if (!e->key) break; continue; }
        if (e->frame + 3 >= frame) { *v = e->val; ok = 1; g_occk_lastf = e->frame; } break; }
    if (count || ok) { if (ok) { g_occk_hit++; g_occk_zero += !*v; } else g_occk_miss++; }
    pthread_mutex_unlock(&g_occk_mu);
    extern uint64_t g_tex_frame;
    if (g_tex_frame >= g_occk_logf + 600) { g_occk_logf = g_tex_frame;
        VKLOG("occkey: last 600 presents: reports with a count of their key %llu (of them 0 samples %llu; by their own address, occaddr, %llu), without one (the fixed value) %llu; counts stored %llu, zero/non-zero flips of a key %llu",
              (unsigned long long)g_occk_hit, (unsigned long long)g_occk_zero, (unsigned long long)g_occk_ahit, (unsigned long long)g_occk_miss, (unsigned long long)g_occk_put, (unsigned long long)g_occk_flip);
        g_occk_hit = g_occk_zero = g_occk_miss = g_occk_put = g_occk_flip = g_occk_ahit = 0; }
    return ok;
}
static void occk_put(uint64_t key, uint64_t frame, uint64_t v) {   // the completion thread
    pthread_mutex_lock(&g_occk_mu);
    occk_e *best = NULL;
    for (unsigned k = 0, i = (unsigned)(key * 0x9E3779B97F4A7C15ull >> 51); k < 16; k++, i = (i + 1) & (OCCK_N - 1)) {
        occk_e *e = &g_occk[i]; if (e->key == key || !e->key) { best = e; break; } if (!best || e->frame < best->frame) best = e; }
    if (best->key == key) { if (frame < best->frame) { pthread_mutex_unlock(&g_occk_mu); return; } g_occk_flip += (best->val != 0) != (v != 0); }
    best->key = key; best->frame = frame; best->val = v; g_occk_put++;
    pthread_mutex_unlock(&g_occk_mu);
}
// occland (11-21, NPARK: Pokemon Z-A's dialogue box flickers with occkey, steady with DM_VK_OCC_KEYED=0): a game that reads a report after waiting for its
// submission (as on the console, where the count is this frame's) got the frame-old keyed value: a visibility that feeds what is drawn next then swings.
// Now the keyed value is only what a read before the GPU has finished sees; the completion thread writes the submission's own real count over it, as the GPU
// would have. A read before the fence (OoT's) sees the same object's count of one frame before, a read after it this frame's: two real counts of one object,
// not a stale word of another (OCC_FIXED=-1's tremble). DM_VK_OCC_LAND=0: occkey's keyed value only
static int occ_land_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_OCC_LAND"); v = e && *e == '1';   // (11-21 11:57: off by default: OoT trembled again with it, NPARK; Pokemon's flicker was the rank key, occaddr)
    VKLOG("occland: occlusion reports %s (DM_VK_OCC_LAND=%s)", v ? "keyed at replay, then the submission's own count written after its fence" : "keyed at replay only", e && *e ? e : "0 (default)"); } return v; }
// S1d (parallel recording): the dry pass names each timestamp report's query slot in stream order (vk_sub_ts_next at its list's start, then one slot per timestamp), so a
// worker can record the vkCmdWriteTimestamp of its segment's; the replay takes the named slot (qd; INT_MIN: allocated here as before) and advances ncq past it
static uint64_t g_cq_off;   // named slots that were not the one allocated here (vk_ts_slot_off; 0 unless the replay left the dry pass's path)
int vk_sub_ts_next(int *cap) { vksub_t *s = cur(); *cap = g_cur * CQ + CQ; return g_cq ? g_cur * CQ + (s->ncq < CQ ? s->ncq : CQ) : -1; }
int vk_ts_replay_on(void) { return ts_replay_on(); }
uint64_t vk_ts_slot_off(void) { return g_cq_off; }
static void sub_counter(uint64_t addr, uint32_t type, int qd);
static void occ_sp_report(post_op *o, uint64_t addr, int age);   // occ-s0 / occ-s1 (below)
void mtl_sub_counter(uint64_t addr, uint32_t type) { sub_counter(addr, type, INT_MIN); }
void vk_sub_counter_q(uint64_t addr, uint32_t type, int q) { sub_counter(addr, type, q); }
static void sub_counter(uint64_t addr, uint32_t type, int qd) {
    vksub_t *s = cur(); int q = -1;
    if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED && g_occ2 >= 2) { occ2_report(addr); return; }   // occ-s2: resolved on the GPU in stream order (no occkey, occaddr, occland, fixed value, post)
    if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED && g_oq && vk_occ_keyed_on()) {   // occkey
        // occaddr (11-21, NPARK: Pokemon Z-A's small window over a character flickers as the character moves, with occkey on): its labels' reports share program
        // and draws, so the rank key paired a label with another one whenever a label came or went with the target's position. A report whose address, program
        // and draws were reported within 3 presents (Pokemon: stable report addresses) takes that one's count first (key2); OoT's addresses never repeat within 4
        // presents (HeyListen TICK-033), so it keeps the rank key. Both are stored. DM_VK_OCC_ADDR=0: the rank key only
        static int addr_on = -1; if (addr_on < 0) { const char *e = getenv("DM_VK_OCC_ADDR"); addr_on = !(e && *e == '0'); VKLOG("occaddr: a report's own address first %s (DM_VK_OCC_ADDR)", addr_on ? "on" : "off"); }
        extern uint64_t g_tex_frame, g_occ_key2; const uint64_t key = g_occ_key, key2 = addr_on ? g_occ_key2 : 0; uint64_t v;
        int age = 4;   // occ-s0: the substituted value's age in presents (0..3), 4 = the fixed value
        if (key2 && occk_get(key2, g_tex_frame, &v, 0)) { g_occk_ahit++; age = 0; }
        else if (!key || !occk_get(key, g_tex_frame, &v, 1)) v = (uint64_t)vk_occ_fixed();
        else age = 0;
        if (!age) { const uint64_t d = g_tex_frame - g_occk_lastf; age = d > 3 ? 3 : (int)d; }
        const int ok = report_now(addr, v, gpu_ticks(), type, 1);
        post_op o = {2, NULL, ok ? 7u : 0u, g_oq_epoch, addr, type, -1, 0, key, g_tex_frame, key2};
        oq_end(); if (g_occ2 || g_occdiag) occ_sp_report(&o, addr, age); post(o);   // the real count for the table (mode 0: not written at replay, written as before)
        return;
    }
    if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED && g_occdiag && vk_occ_fixed() >= 0) occ_sp_report(NULL, addr, 4);   // occ-s0: the fixed value (no queries on this path: no post)
    if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED && vk_occ_fixed() >= 0 && report_now(addr, (uint64_t)vk_occ_fixed(), gpu_ticks(), type, 1)) return;   // rptfix (replay2: queued)
    if (type != NVN_COUNTER_TYPE_SAMPLES_PASSED && type != 0 && type != 0x10 && report_now(addr, 0, gpu_ticks(), type, 1)) return;   // rptfix: not measured, 0 (replay2: queued)
    if ((type == 0 || type == 0x10) && ts_replay_on()) { const uint64_t t = ts_replay_ticks(); if (report_now(addr, DM_GAME_RENDER_COND ? 0 : t, t, type, 0)) return; }   // rptfix test (not queued: its fallback is a GPU timestamp)
    if (g_oq && type == NVN_COUNTER_TYPE_SAMPLES_PASSED) { oq_end(); post_op o = {2, NULL, 0, g_oq_epoch, addr, type, -1}; if (g_occ2 || g_occdiag) occ_sp_report(&o, addr, 5); post(o); return; }   // (occ-s0: 5 = the real count after the fence)
    if (qd != INT_MIN && (type == 0 || type == 0x10)) {   // S1d: the dry pass's slot (-1: none, as the allocation below gives when the pool is missing or full)
        const int qa = g_cq && s->ncq < CQ ? g_cur * CQ + s->ncq : -1; if (qa != qd) g_cq_off++;
        if (g_cq && qd >= g_cur * CQ && qd < g_cur * CQ + CQ) { q = qd; if (s->ncq < qd - g_cur * CQ + 1) s->ncq = qd - g_cur * CQ + 1; vkCmdWriteTimestamp(vk_cmd(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_cq, (uint32_t)q); }
    } else if (g_cq && (type == 0 || type == 0x10) && s->ncq < CQ) { q = g_cur * CQ + s->ncq++; vkCmdWriteTimestamp(vk_cmd(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_cq, (uint32_t)q); }
    post((post_op){2, NULL, 0, 0, addr, type, q});
}
static VkMemoryBarrier2 full_barrier(void) {
    return (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                              VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
}
// r89: the full barrier names the device's own kinds of access instead of MEMORY_READ / MEMORY_WRITE: the same scope for work on the queue (ALL_COMMANDS holds no host
// stage, and host writes before a submission are visible to it anyway). Inferred, Turnip's source not at hand: Turnip reads MEMORY_WRITE with ALL_COMMANDS as host writes as
// well and invalidates its L2 (UCHE) at every such barrier, and Hyrule Warriors: Age of Imprisonment records 450-513 passes and up to 372 dispatches a frame, each with one.
// DM_VK_BAR (bits, default 3): 1 these masks, 2 dispatches after a dispatch ordered by compute-only barriers (gfx_vk_exec.c dispatch); 0 the old full masks everywhere
_Thread_local uint64_t g_bar_seq;   // full barriers recorded so far (dispatch: no other one since the previous dispatch's); S1e (parallel recording): per thread (its users compare it with their own command buffer's)
// r107 (diagnostic, after Lucas dkb-pc-recomp 9b12091): full barriers, in-order buffer copies and fills per second, and how many copies/fills came right after another
// one's closing barrier with no barrier, draw or dispatch between (their opening barrier repeats that one). Logged about every 10 s.
static struct { uint64_t bars, copies, fills, bytes, adj, seq, draws, disp, t0; VkCommandBuffer cb; } g_r107;
extern struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst;
static uint64_t pipes_built_now(void) { return g_vkst.pipes_built; }   // (NVN2 memory report)
static void r107_open(VkCommandBuffer cb, uint64_t n, int fill) {
    if (fill) g_r107.fills++; else g_r107.copies++; g_r107.bytes += n;
    if (cb == g_r107.cb && g_bar_seq == g_r107.seq && g_vkst.draws == g_r107.draws && g_vkst.dispatches == g_r107.disp) g_r107.adj++;
}
static void r107_close(VkCommandBuffer cb) { g_r107.cb = cb; g_r107.seq = g_bar_seq; g_r107.draws = g_vkst.draws; g_r107.disp = g_vkst.dispatches; }
static void r107_tick(void) {
    if (g_r107.bars & 1023) return;
    uint64_t t = now_ns(); if (!g_r107.t0) { g_r107.t0 = t; return; }
    if (t - g_r107.t0 < 10000000000ull) return;
    static uint64_t b0, c0, f0, y0, a0; double s = (t - g_r107.t0) / 1e9;
    VKLOG("r107: per second (last %.1f s): full barriers %.0f, buffer copies %.0f (%.0f KB) and fills %.0f, of which %.0f right after another one's closing barrier",
          s, (g_r107.bars - b0) / s, (g_r107.copies - c0) / s, (g_r107.bytes - y0) / s / 1024, (g_r107.fills - f0) / s, (g_r107.adj - a0) / s);
    { static uint64_t k0, s0; VKLOG("s63: full barriers kept %llu, left out %llu (last %.1f s; DM_VK_S63=%d: 1 dedup, 2 write-only source)", (unsigned long long)(g_s63_kept - k0),
                                   (unsigned long long)(g_s63_skip - s0), s, g_s63_mode); k0 = g_s63_kept; s0 = g_s63_skip; }   // s63
    b0 = g_r107.bars; c0 = g_r107.copies; f0 = g_r107.fills; y0 = g_r107.bytes; a0 = g_r107.adj; g_r107.t0 = t;
}
int vk_bar_mode(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_BAR"); v = e ? atoi(e) & 3 : 3; VKLOG("r89: barriers %s, dispatches after a dispatch %s (DM_VK_BAR=%d; 0 = the old full masks)", v & 1 ? "with the device's access kinds" : "with MEMORY_READ / MEMORY_WRITE", v & 2 ? "ordered compute-only" : "after a full barrier", v); }
    return v;
}
VkAccessFlags vk_dev_access(void) {
    VkAccessFlags m = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT |
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    if (V.has_xfb) m |= VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
    if (g_occ2 >= 3) m |= VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;   // occ-s3: the predicates a gated pass reads (the resolve's compute writes them); ALL_COMMANDS holds the CONDITIONAL_RENDERING stage
    return m;
}
void vk_img_barrier_all(VkCommandBuffer cb) {
    static _Thread_local VkCommandBuffer lcb; static _Thread_local uint64_t lwork;   // s63: this thread's last kept full barrier
    const int mode = atomic_load_explicit(&g_s63_mode, memory_order_acquire);
    const uint64_t w = (mode & 1) ? __atomic_load_n(&g_s63_work, __ATOMIC_RELAXED) : 0;
    if ((mode & 1) && cb == lcb && w == lwork) { __atomic_fetch_add(&g_s63_skip, 1, __ATOMIC_RELAXED); return; }   // s63: no work since the last one in this command buffer (S1e (parallel recording): counted by every recording thread)
    VkAccessFlags m = (vk_bar_mode() & 1) ? vk_dev_access() : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkAccessFlags src = m;
    if (mode & 2) {   // s63: only writes are made available (reads need no availability operation; the execution dependency orders them)
        const VkAccessFlags wr = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                 VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT | VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT;
        src = (vk_bar_mode() & 1) ? (m & wr) : VK_ACCESS_MEMORY_WRITE_BIT;
    }
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, src, m};
    if (!bpend_hold(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, src, m))   // cbar2: held and merged with the barriers next to it
        (s63o_vkCmdPipelineBarrier ? s63o_vkCmdPipelineBarrier : vkCmdPipelineBarrier)(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL, 0, NULL);
    if (mode & 1) { lcb = cb; lwork = w; } else lcb = VK_NULL_HANDLE;
    __atomic_fetch_add(&g_s63_kept, 1, __ATOMIC_RELAXED);   // (the barrier itself is not counted as work)
    og_note(cb, OG_FULLBAR);   // ootgap
    g_bar_seq++; if (!tl_worker) { g_r107.bars++; r107_tick(); }   // (S1e (parallel recording): r107 counts the replay thread's barriers)
}
void vk_gpu_wrote(uint64_t a, uint64_t n);
void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!n) return;
    if (!vk_buf_at(src, n, &sb, &so) || !vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(1, "buffer copy outside the memory pools (%#llx → %#llx, %llu bytes)", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n); return; }
    if (g_occ2 >= 2 && !tl_wv && occ2_unres_in(src, n)) { vk_end_pass_land(); occ2_resolve(O2R_READ); }   // occ-s3fast: a GPU copy reading a report word not resolved yet: the resolve first, in stream order (as the console's write before it)
    if (g_occdiag || g_occ2 >= 2) vk_occd_read(src, n, 1);   // occ-s0: a GPU copy reading a samples-passed report word whose result has not landed (occ-s2: counted always)
    vk_end_pass_land();
    VkCommandBuffer cb = vk_cmd(); r107_open(cb, n, 0); vk_img_barrier_all(cb);
    VkBufferCopy r = {so, dof, n}; vkCmdCopyBuffer(cb, sb, db, 1, &r);
    vk_img_barrier_all(cb); vk_gpu_wrote(dst, n); r107_close(cb);
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    VkBuffer db; VkDeviceSize dof;
    if (!n) return;
    if (!vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(2, "buffer fill outside the memory pools"); return; }
    vk_end_pass_land();
    VkCommandBuffer cb = vk_cmd(); r107_open(cb, n, 1); vk_img_barrier_all(cb);
    vkCmdFillBuffer(cb, db, dof, n & ~3ull, value);
    vk_img_barrier_all(cb); vk_gpu_wrote(dst, n); r107_close(cb);
}
// an inline uniform update (UpdateUniformBuffer): the bytes are taken at record time, copied in GPU order from scratch memory
// Inline uniform updates (UpdateUniformBuffer) are deferred instead of ending the render pass (as nvn_mtl.m's ubd_*): a copy cannot run inside a pass, and
// ending it stores every attachment and loads it again in the next pass, which on a tiler is a round trip of the whole tile memory per update (Unity updates
// before nearly every draw: NEO had about 145 passes for 148 draws a frame). NVN applies the update in command order, so a draw after it whose uniform
// binding overlaps it reads a shadow: memory with the deferred updates applied in order (vk_ubd_shadow). The real copies are made in order when a pass
// ends anyway, and at the end of the submission (vk_ubd_land), so later passes, copies, dispatches and the CPU see them as before.
typedef struct { uint64_t lo, hi; VkBuffer sb; VkDeviceSize so; const uint8_t *p; } ubd_t;
enum { UBD_MAX = 512 };
uint64_t g_ubd_watch; int g_ubd_wn;   // DM_VK_UBD_WATCH (gfx_vk_exec.c x_read)
// the watched word's memory, logged when it differs from the last time it was looked at (a CPU write shows up between two of these)
void vk_watch_poll(const char *where) {
    static uint64_t last_a; static uint32_t last; extern uint64_t g_tex_frame; uint32_t m; float o, v;
    if (!g_ubd_watch || g_ubd_wn <= 0) return;
    memcpy(&m, (const void *)(uintptr_t)g_ubd_watch, 4); if (last_a == g_ubd_watch && m == last) return;
    memcpy(&o, &last, 4); memcpy(&v, &m, 4); g_ubd_wn--;
    VKLOG("ubd watch: memory at %#llx is %g (%s %g), seen at %s, present %llu, fence %llu", (unsigned long long)g_ubd_watch, v, last_a == g_ubd_watch ? "was" : "first look, last",
          o, where, (unsigned long long)g_tex_frame, (unsigned long long)g_fence);
    last_a = g_ubd_watch; last = m;
}
// S1d (parallel recording): the list as one object per thread (design R5). The replay thread's main list is what every serial path uses; a segment's list is a copy of it at
// the segment's start that the segment's own updates are applied to from the dry pass's log (gfx_vk.h vk_ubl_*). g_ubd / g_nubd name the thread's current list, g_ubd_gen
// (gfx_vk.h, a macro over g_ubd_genp) its generation; every bump takes the next value of one counter, so two lists never share a generation (the memos compare equality)
struct vk_ubl { uint64_t gen; int n; ubd_t e[UBD_MAX]; };
static struct vk_ubl g_ubl_main = {.gen = 1};
static _Thread_local struct vk_ubl *g_ubl = &g_ubl_main;
_Thread_local uint64_t *g_ubd_genp = &g_ubl_main.gen;
static uint64_t g_ubd_gseq = 1;   // the last generation given to any list
uint64_t g_ubl_seg_lands;
#define g_ubd (g_ubl->e)
#define g_nubd (g_ubl->n)
static inline void ubd_bump(void) { g_ubl->gen = __atomic_add_fetch(&g_ubd_gseq, 1, __ATOMIC_RELAXED); }   // (was g_ubd_gen++)
uint64_t g_ubd_landed, g_ubd_shadows, g_ubd_bytes, g_ns_submit, g_ubd_superseded;
void vk_end_pass_land(void);
// S1d: the dry pass's part of an UPDATE_UB, in stream order: the range checked (as before) and the bytes copied into this submission's scratch memory (kept until it completes)
int vk_sub_update_pre(uint64_t dst, const void *data, uint64_t n, vk_ubd_pre *u) {
    VkBuffer db; VkDeviceSize dof;
    memset(u, 0, sizeof *u); u->dst = dst; u->n = n;
    if (!n) return 0;
    if (!vk_buf_at(dst, n, &db, &dof)) { VKRED_ONCE(3, "uniform update outside the memory pools"); return 0; }
    uint8_t *p = vk_scratch(n, 4, &u->sb, &u->so); memcpy(p, data, n); u->p = p; u->ok = 1;
    return 1;
}
// S1d: the list operation of an UPDATE_UB (the replay's, in stream order, on the thread's list): the superseded entry's index or -1
int vk_sub_update_apply(const vk_ubd_pre *u, int *landed) {
    const uint64_t dst = u->dst, n = u->n; int sup = -1;
    if (landed) *landed = 0;
    if (!u->ok) return -1;
    // an update of exactly the range of a deferred one supersedes it: the older one goes (Unity rewrites the same few constant buffers before every draw,
    // so the list keeps about one entry per buffer region, which keeps the landing to a few copies and the shadows cheap to build)
    for (int i = g_nubd - 1; i >= 0; i--) if (g_ubd[i].lo == dst && g_ubd[i].hi == dst + n) { memmove(&g_ubd[i], &g_ubd[i + 1], sizeof *g_ubd * (size_t)(g_nubd - i - 1)); g_nubd--; g_ubd_superseded++; sup = i; break; }
    if (g_nubd == UBD_MAX) { vk_end_pass_land(); if (landed) *landed = 1; }   // too many distinct ranges: they land now
    g_ubd[g_nubd++] = (ubd_t){dst, dst + n, u->sb, u->so, u->p}; ubd_bump();
    vk_watch_poll("update");
    if (g_ubd_watch && g_ubd_wn > 0 && dst <= g_ubd_watch && g_ubd_watch + 4 <= dst + n) { float v, m; memcpy(&v, u->p + (g_ubd_watch - dst), 4); memcpy(&m, (const void *)(uintptr_t)g_ubd_watch, 4); g_ubd_wn--;
        VKLOG("ubd watch: inline update %#llx +%llu sets %#llx to %g (memory %g)", (unsigned long long)dst, (unsigned long long)n, (unsigned long long)g_ubd_watch, v, m); }
    return sup;
}
void vk_sub_update(uint64_t dst, const void *data, uint64_t n) { vk_ubd_pre u; if (vk_sub_update_pre(dst, data, n, &u)) (void)vk_sub_update_apply(&u, NULL); }   // (as before: the two in a row)
// S1d: segment lists (gfx_vk.h). Replay thread only in S1d (SERIALSEG); a worker would own its segment's list the same way
struct vk_ubl *vk_ubl_new(void) { struct vk_ubl *l = calloc(1, sizeof *l); if (!l) die("[vk] no memory for a segment's uniform update list"); return l; }
void vk_ubl_snapshot(struct vk_ubl *l) { l->gen = g_ubl_main.gen; l->n = g_ubl_main.n; memcpy(l->e, g_ubl_main.e, sizeof *l->e * (size_t)g_ubl_main.n); }
void vk_ubl_use(struct vk_ubl *l) { g_ubl = l ? l : &g_ubl_main; g_ubd_genp = &g_ubl->gen; }
void vk_ubl_copy(struct vk_ubl *dst, const struct vk_ubl *src) {   // S1e (parallel recording): a fresh generation (a worker's memos never match another list's)
    dst->n = src->n; memcpy(dst->e, src->e, sizeof *dst->e * (size_t)src->n); dst->gen = __atomic_add_fetch(&g_ubd_gseq, 1, __ATOMIC_RELAXED); }
int vk_ubl_log_apply(const vk_ubd_pre *u, int sup, int check) {   // no land, no counters, no watch: those are the main list's (vk_sub_update_apply)
    if (!u->ok) return 1;
    const uint64_t lo = u->dst, hi = u->dst + u->n; int ok = 1;
    if (check) {   // the logged index must be what the search finds
        int at = -1; for (int i = g_nubd - 1; i >= 0; i--) if (g_ubd[i].lo == lo && g_ubd[i].hi == hi) { at = i; break; }
        if (at != sup) { ok = 0; sup = at; } }
    if (sup >= 0 && sup < g_nubd && g_ubd[sup].lo == lo && g_ubd[sup].hi == hi) { memmove(&g_ubd[sup], &g_ubd[sup + 1], sizeof *g_ubd * (size_t)(g_nubd - sup - 1)); g_nubd--; }
    else if (sup >= 0) ok = 0;
    if (g_nubd == UBD_MAX) return 0;   // it would land: never inside a segment (the dry pass's bound); the update is not applied
    g_ubd[g_nubd++] = (ubd_t){lo, hi, u->sb, u->so, u->p}; ubd_bump();
    return ok;
}
int vk_ubl_same(const struct vk_ubl *l) {
    if (l->n != g_ubl_main.n) return 0;
    for (int i = 0; i < l->n; i++) if (l->e[i].lo != g_ubl_main.e[i].lo || l->e[i].hi != g_ubl_main.e[i].hi || l->e[i].p != g_ubl_main.e[i].p) return 0;
    return 1;
}
// the draw bisection (gfx_vk_exec.c draw_allowed): the outcome of this submission
extern int g_bis_on, g_bis_lo, g_bis_hi, g_bis_trace, g_bis_var; void vk_bisect_reset_count(void);
static void vk_bisect_result(int ok) {
    static int done, draws_seen; extern uint32_t g_sub_ops[64]; int nd = (int)(g_sub_ops[25] + g_sub_ops[26]);
    if (done) {   // the found draw: one change per submission, then the draw is no longer traced
        if (g_bis_trace >= 0) {
            static const char *what[] = {"as recorded", "depth bias values 0", "no depth bias", "dummy textures", "vertex offset 0", "zero vertex buffer", "in a new pass", "skipped alone (the rest drawn)", "without their fragment shader", "with zero uniform buffers", "with their quad draws skipped", "with all but their quad draws skipped"};
            VKLOG("bisection: draws from %d on %s: %s", g_bis_trace, what[g_bis_var], ok ? "ACCEPTED" : "refused");
            if (++g_bis_var > 11) { g_bis_trace = -1; g_bis_var = 0; }
        }
        vk_bisect_reset_count(); return; }
    if (!g_bis_on) { if (!ok) { g_bis_on = 1; g_bis_lo = 0; g_bis_hi = nd + 1; draws_seen = nd; VKLOG("bisecting the refused submission's %d draws", nd); } vk_bisect_reset_count(); return; }
    int mid = (g_bis_lo + g_bis_hi) / 2;
    if (ok) g_bis_lo = mid; else g_bis_hi = mid;
    if (g_bis_hi - g_bis_lo <= 1) { done = 1; g_bis_on = 0; g_bis_trace = g_bis_lo; VKLOG("bisection: draw %d of %d is the first the driver refuses (traced in the next submission)", g_bis_lo, draws_seen); }
    vk_bisect_reset_count();
}
void vk_ubd_land_drop(void) { g_nubd = 0; ubd_bump(); }
// Landed updates still in flight: their copies are queued but the GPU may not have run them, so memory may not hold them yet; a shadow built on the
// CPU (vk_ubd_read) applies them over memory until their submission completes. Without this a shadow mixed the newest update with the frame before's
// memory (shadows detached from moving characters, far pedestrians flickering).
typedef struct { uint64_t lo, hi, fence; const uint8_t *data; } ubd_infl_t;   // data: the update's bytes in its submission's scratch memory, which is not reused before that submission completes
static ubd_infl_t *g_infl; static int g_ninfl, g_capinfl;
static void infl_prune(void) {
    uint64_t done = vk_fence_done(); int o = 0;
    for (int i = 0; i < g_ninfl; i++) if (g_infl[i].fence > done) g_infl[o++] = g_infl[i];
    g_ninfl = o;
}
static void sub_gpu_wrote(uint64_t a, uint64_t n);
void vk_ubd_land(VkCommandBuffer cb) {   // outside a render pass
    if (!g_nubd) return;
    if (g_ubl != &g_ubl_main) g_ubl_seg_lands++;   // S1d (parallel recording): a segment's list landing (never: the dry pass ends a segment before anything that lands)
    infl_prune();
    for (int i = 0; i < g_nubd; i++) {
        if (g_ninfl == g_capinfl) { g_capinfl = g_capinfl ? 2 * g_capinfl : 256; g_infl = realloc(g_infl, sizeof *g_infl * (size_t)g_capinfl); }
        // (the bytes used to be copied into the heap here: a malloc, memcpy and free of up to 64 KB per update, for bytes the scratch memory keeps anyway)
        g_infl[g_ninfl++] = (ubd_infl_t){g_ubd[i].lo, g_ubd[i].hi, g_fence + 1, g_ubd[i].p};   // the fence of the submission being recorded
    }
    vk_img_barrier_all(cb);
    for (int i = 0; i < g_nubd; i++) { VkBuffer db; VkDeviceSize dof; if (vk_buf_at(g_ubd[i].lo, g_ubd[i].hi - g_ubd[i].lo, &db, &dof)) {
        for (int j = i - 1; j >= 0; --j) {
            if (g_ubd[j].hi <= g_ubd[i].lo || g_ubd[i].hi <= g_ubd[j].lo) continue;
            VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, NULL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, db, dof, g_ubd[i].hi - g_ubd[i].lo};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, NULL, 1, &barrier, 0, NULL);
            break;
        }
        VkBufferCopy r = {g_ubd[i].so, dof, g_ubd[i].hi - g_ubd[i].lo}; vkCmdCopyBuffer(cb, g_ubd[i].sb, db, 1, &r); sub_gpu_wrote(g_ubd[i].lo, g_ubd[i].hi - g_ubd[i].lo); } }
    vk_img_barrier_all(cb);
    g_ubd_landed += (uint64_t)g_nubd; g_nubd = 0; ubd_bump();
}
int vk_ubd_overlaps(uint64_t a, uint64_t n) { for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo < a + n && a < g_ubd[i].hi) return 1; return 0; }
// S1b (parallel recording): the deferred updates' ranges (lo, hi pairs, at most max), read only (the dry pass's "may land" bound, gfx_vk_exec.c seg_ub_*); *cap = UBD_MAX
int vk_ubd_ranges(uint64_t *lohi, int max, int *cap) {
    int n = g_nubd < max ? g_nubd : max; for (int i = 0; i < n; i++) { lohi[2 * i] = g_ubd[i].lo; lohi[2 * i + 1] = g_ubd[i].hi; }
    *cap = UBD_MAX; return n;
}
// GPU writes to pool memory (buffer copies and fills, texture → buffer copies, storage buffers of dispatches) whose submission may not have completed:
// until it has, a CPU read of that memory (vk_ubd_read: a shadow, a handle in a uniform buffer) may see the bytes from before them or after them
typedef struct { uint64_t lo, hi, fence, seq; } gpuw_t; enum { GW_MAX = 1024 };   // seq (replay2): g_gw_seq when listed, so a batched report knows which writes were recorded before it
static gpuw_t g_gw[GW_MAX]; static int g_ngw; uint64_t g_ubd_racy, g_gw_seq;
void vk_vbm_gpu_wrote(uint64_t a, uint64_t n);   // vbmir (gfx_vk_exec.c): the device-local vertex/index mirrors over a GPU-written range are stale
static void sub_gpu_wrote(uint64_t a, uint64_t n) {   // into the open submission's list (pools_copy_back)
    { extern uint64_t g_rq_lo, g_rq_hi; void vk_rpt_flush(void); if (g_rq_hi > g_rq_lo && a < g_rq_hi && g_rq_lo < a + n) vk_rpt_flush(); }   // replay2: a GPU write over queued reports: they are in memory before it is recorded (stage2c's pre copy, the merge, report_later_mask see them as before)
#if defined(DM_NXN1X)
    nvnc_range_intent(a, n); // includes deferred inline updates at their actual copy command
#endif
    { extern uint64_t g_vbm_stale; const uint64_t v0 = g_vbm_stale; vk_vbm_gpu_wrote(a, n); g_hyb_vbm[g_hy_attr] += g_vbm_stale - v0; }   // vbmir (hybrid1: counted by binding kind)
    if (g_cur < 0 || !n) return;
    { const uint64_t p0 = g_pc_pre_bytes; pc_pre_note(a, n); g_hyb_pre[g_hy_attr] += g_pc_pre_bytes - p0; }   // stage2c: the bytes as they are before this submission's work on them (the exact range, before it joins a nearby one)
    { vksub_t *w = &g_slot[g_cur];   // combined1120: in command order, for the reports recorded before it
      if (w->nwr == w->capwr && !w->wr_over) { int cp = w->capwr ? 2 * w->capwr : 256; void *v = cp <= 65536 ? realloc(w->wr, (size_t)cp * sizeof *w->wr) : NULL; if (v) { w->wr = v; w->capwr = cp; } else w->wr_over = 1; }
      w->wseq++; if (!w->wr_over) { w->wr[w->nwr].lo = a; w->wr[w->nwr].hi = a + n; w->wr[w->nwr].seq = w->wseq; w->nwr++; } }
    vksub_t *s = &g_slot[g_cur]; uint64_t lo = a, hi = a + n, bd = ~0ull; int best = -1;
    for (int k = 0; k < s->ngw; k++) { uint64_t d = lo > s->gw[k].hi ? lo - s->gw[k].hi : s->gw[k].lo > hi ? s->gw[k].lo - hi : 0; if (d < bd) { bd = d; best = k; } }
    if (best >= 0 && (bd <= 65536 || s->ngw == GW_SUB)) {
        if (lo >= s->gw[best].lo && hi <= s->gw[best].hi) return;
        if (lo < s->gw[best].lo) s->gw[best].lo = lo; if (hi > s->gw[best].hi) s->gw[best].hi = hi;
        lo = s->gw[best].lo; hi = s->gw[best].hi;
    } else { s->gw[s->ngw].lo = lo; s->gw[s->ngw].hi = hi; s->ngw++; }
    // Copy-back visits all copied overlaps of the merged range, including gaps
    // coalesced above. Keep their baseline current even if a different imported
    // buffer was selected for the write. This only records work, before submit.
    pool_sync_pending(lo, hi);
}
void vk_sub_wrote(uint64_t a, uint64_t n) {
#if defined(DM_NXN1X)
    nvnc_range_intent(a, n);
#endif
    { void vk_tbuf_wrote(uint64_t, uint64_t); vk_tbuf_wrote(a, n); }   // r150 (gfx_vk_tex.c)
    if (g_ubd_watch && g_ubd_wn > 0 && a <= g_ubd_watch && g_ubd_watch < a + n) { g_ubd_wn--; VKLOG("ubd watch: shader write %#llx +%llu covers %#llx", (unsigned long long)a, (unsigned long long)n, (unsigned long long)g_ubd_watch); }
    sub_gpu_wrote(a, n); }
void vk_gml_note2(const uint32_t *log, int atomic, uint64_t key) {   // a raw global memory write log, cleared, for this submission's copy-back (gml_copy_back): the scratch memory holding it lives until then
    if (g_cur < 0) return; vksub_t *s = &g_slot[g_cur];
    vk_vbm_gpu_wrote(0, ~0ull);   // vbmir: writes anywhere: every mirror is copied again before its next use
    pc_pre_all();   // stage2c: raw global memory: any chunk
    if (s->ngml == s->capgml) { s->capgml = s->capgml ? 2 * s->capgml : 256; s->gml = realloc(s->gml, (size_t)s->capgml * sizeof *s->gml); }
    if (s->ngml >= s->capgmlk) { s->capgmlk = s->capgml; s->gmlk = realloc(s->gmlk, (size_t)s->capgmlk * sizeof *s->gmlk); }
    if (!s->gml || !s->gmlk) die("[vk] raw global memory write log list allocation failed");
    s->gmlk[s->ngml] = (key & ~1ull) | (atomic ? 1 : 0);
    s->gml[s->ngml++] = log;
}
// the DM_VK_EDEN=0 path (gfx_vk_exec.c RK_GMEM_LOG): its shader bundle says nothing of atomics, so its logs never convert anything (hybrid1 rule H11)
void vk_gml_note(const uint32_t *log) { vk_gml_note2(log, 0, 0); }   // a draw's shader writes (gfx_vk_exec.c stage_set): copied back only, not a pending write for CPU reads
void vk_gpu_wrote(uint64_t a, uint64_t n) {
#if defined(DM_NXN1X)
    nvnc_range_intent(a, n); // before legacy ring deduplication
#endif
    { void vk_tbuf_wrote(uint64_t, uint64_t); vk_tbuf_wrote(a, n); }   // r150 (gfx_vk_tex.c): before the early return below
    vk_watch_poll("gpu write");
    if (g_ubd_watch && g_ubd_wn > 0 && a <= g_ubd_watch && g_ubd_watch < a + n) { float m; memcpy(&m, (const void *)(uintptr_t)g_ubd_watch, 4); g_ubd_wn--;
        VKLOG("ubd watch: GPU write %#llx +%llu covers %#llx (memory now %g)", (unsigned long long)a, (unsigned long long)n, (unsigned long long)g_ubd_watch, m); }
    uint64_t f = g_fence + 1;
    for (int i = g_ngw - 1; i >= 0 && g_gw[i].fence == f; i--) if (g_gw[i].lo <= a && a + n <= g_gw[i].hi) return;
    if (g_ngw == GW_MAX) { uint64_t done = vk_fence_done(); int o = 0; for (int i = 0; i < g_ngw; i++) if (g_gw[i].fence > done) g_gw[o++] = g_gw[i]; g_ngw = o; }
    if (g_ngw == GW_MAX) { memmove(g_gw, g_gw + 1, sizeof *g_gw * (GW_MAX - 1)); g_ngw--; }
    g_gw[g_ngw++] = (gpuw_t){a, a + n, f, ++g_gw_seq}; g_hyb_gw[g_hy_attr] += n;   // (hybrid1 diagnostic)
    sub_gpu_wrote(a, n);
}
// hybrid2 (11-21, NPARK 00:30): a storage buffer binding's GPU-write bookkeeping (gfx_vk_eden.inc eden_ssbo). Only a binding its shader writes (Eden is_written)
// is a GPU write, and a dispatch's only as far as its stores reach (has_reach: Eden write_extent resolved for this dispatch, nvn_eden_ssbo_reach). The CPU -> GPU
// side is not here: a copy's sync uploads every CPU change whatever binds it. Kept as before: storage buffers whose pointer is read from memory (indirect), and the
// raw pool table (its own path). NXN1X_SSBO_WRITES=all: every dispatch's storage buffer taken as written, whole (as before; A/B)
static int ssbo_all(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("NXN1X_SSBO_WRITES"); v = e && !strcmp(e, "all");
        VKLOG("hybrid2: storage buffer bindings %s (NXN1X_SSBO_WRITES=%s)", v ? "of a dispatch all taken as written, whole (as before)" :
              "taken as written only when the shader writes them, a dispatch's only as far as its stores reach", e && *e ? e : "written (default)"); }
    return v;
}
void vk_ssbo_bound(int st, uint64_t a, uint64_t n, int written, int indirect, int has_reach, uint64_t reach) {
    if (ssbo_all() || indirect) { if (st == 5) vk_gpu_wrote(a, n); else if (written) vk_sub_wrote(a, n); return; }
    if (!written) { g_ssbo_ro_n++; g_ssbo_ro_bytes += n; return; }
    if (has_reach && reach < n) { g_ssbo_cut_n++; g_ssbo_cut_bytes += n - reach; n = reach; }
    if (!n) return;
    if (st == 5) vk_gpu_wrote(a, n); else vk_sub_wrote(a, n);
}
// vbmir (gfx_vk_exec.c): whether the open submission has a GPU write over [a, a+n) recorded so far (its merged list, so it may answer yes for a gap between
// two writes), or a raw global memory write log (where its writes go is known only at copy-back): such a range is bound in the pool, not from its mirror
int vk_sub_gw_overlaps(uint64_t a, uint64_t n) {
    if (tl_wv) { const struct vk_par_view *v = tl_wv; if (v->ngml) return 1;   // S1e (parallel recording): a worker's segment: the list as at its hand-out
        for (int k = 0; k < v->ngw; k++) if (v->gw[k].lo < a + n && a < v->gw[k].hi) return 1;
        return 0; }
    if (g_cur < 0) return 0; const vksub_t *s = &g_slot[g_cur];
    if (s->ngml) return 1;
    for (int k = 0; k < s->ngw; k++) if (s->gw[k].lo < a + n && a < s->gw[k].hi) return 1;
    return 0;
}
uint64_t vk_fence_next(void) { return g_fence + 1; }   // vbmir: the fence the open submission will commit with
uint64_t vk_sub_wver(void) { if (tl_wv) return ((uint64_t)tl_wv->wseq << 24) ^ (uint64_t)tl_wv->ngml ^ (tl_wv->nsubs << 44);   // S1e (parallel recording): a worker's segment: as at its hand-out
    if (g_cur < 0) return ~0ull; const vksub_t *s = &g_slot[g_cur]; return ((uint64_t)s->wseq << 24) ^ (uint64_t)s->ngml ^ ((uint64_t)g_nsubs << 44); }   // replay2: the open submission's GPU writes so far (gfx_vk_eden.inc vertex lookups)
int vk_gpu_written_recent(uint64_t a, uint64_t n) {   // bak89: any entry of the ring of GPU writes, finished or not
    for (int i = g_ngw - 1; i >= 0; i--) if (g_gw[i].lo < a + n && a < g_gw[i].hi) return 1;
    return 0;
}
// bak89: where an address GPU-driven drawing reads lives (indirect arguments, render conditions): the pool kind, once per address (40 lines), and counts per kind
static uint64_t g_pk_n[3][5];   // [indirect args, render condition, other][no pool, zero-copy, s48 range, copied, sparse bridge]
void vk_pool_kind_note(uint64_t a, int what) {
    static uint64_t seen[256]; static int ns;
    if (what < 0 || what > 2) what = 2;
    vkpool_t *p = vk_pool_of(a, 4); const int k = !p ? 0 : p->shadow ? 3 : p->bridge ? 4 : (p->s48 || p->s48b) ? 2 : 1; g_pk_n[what][k]++;
    for (int i = 0; i < ns; i++) if (seen[i] == a) return;
    if (ns < 256) seen[ns++] = a;
    static int nl; if (nl++ >= 40) return;
    static const char *const kn[5] = {"in no pool", "zero-copy", "an s48 range", "a COPIED pool", "a sparse bridge"}, *const wn[3] = {"indirect draw arguments", "render condition", "other"};
    VKLOG("NVN2 pool kind: %s at %#llx: %s%s (pool %#llx +%#llx, NVN flags %#x%s)", wn[what], (unsigned long long)a, kn[k], p && p->shadow && !p->span ? ", untracked: copied whole at every sync" : "",
          (unsigned long long)(p ? p->lo : 0), (unsigned long long)(p ? p->hi - p->lo : 0), p ? p->flags : 0, p && p->shadow ? "" : "");
}
void vk_pool_kind_stats(uint64_t out[3][5]) { memcpy(out, g_pk_n, sizeof g_pk_n); }
// bak91c (11-16, MKW trees / cacti flicker in NVN2 mode, not in emu mode, with the same Eden shaders): the storage buffers the scenery-list compute shaders write
// (e04783bda5a52269, 2ce9071c2f22d74f, 0823b08c2027a38d, fb3fdaea2e0f1779, 5fea9b7ddb30e8a8) are resolved with the binding's length (vk_pool_of(a, n), gfx_vk_eden.inc
// eden_ssbo), indirect arguments with 20 bytes (vk_buf_at): over overlapping pools the two could be different buffers over the same pages. Logging only: each written
// range (first 16 per shader) with its pool, a ring of the last 64, and at each indirect draw / dispatch whose arguments one of them covers, whether the arguments'
// pool is the same buffer (a RED line when not, first 16). DM_VK_POOL_ID_DIAG=0 off. NVN2 boots only
static int pid_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_POOL_ID_DIAG"); v = dm_sdk_ounce() && !(e && *e == '0'); } return v; }
static struct { uint64_t lo, hi, plo, key; VkBuffer buf; int kind; } g_pid[64]; static int g_pid_n, g_pid_w;
static uint64_t g_pid_same, g_pid_diff;
static int pid_slot(uint64_t key) { static const uint64_t ks[5] = {0xe04783bda5a52269ull, 0x2ce9071c2f22d74full, 0x0823b08c2027a38dull, 0xfb3fdaea2e0f1779ull, 0x5fea9b7ddb30e8a8ull};
    for (int i = 0; i < 5; i++) if (ks[i] == key) return i; return -1; }
static int pid_kind(const vkpool_t *p) { return !p ? 0 : p->shadow ? 3 : p->bridge ? 4 : (p->s48 || p->s48b) ? 2 : 1; }
static const char *const g_pid_kn[5] = {"no pool", "zero-copy", "s48 range", "COPIED pool", "sparse bridge"};
void vk_pool_id_ssbo(uint64_t key, uint64_t a, uint64_t n) {   // gfx_vk_eden.inc eden_ssbo: a written storage buffer of a compute dispatch
    const int s = pid_slot(key); if (s < 0 || !a || !pid_on()) return;
    vkpool_t *p = vk_pool_of(a, n ? n : 1); if (!p) p = vk_pool_of(a, 1);
    const int i = g_pid_w; g_pid_w = (g_pid_w + 1) % 64; if (g_pid_n < 64) g_pid_n++;
    g_pid[i].lo = a; g_pid[i].hi = a + (n ? n : 1); g_pid[i].buf = p ? p->buf : VK_NULL_HANDLE; g_pid[i].plo = p ? p->lo : 0; g_pid[i].key = key; g_pid[i].kind = pid_kind(p);
    static int nl[5]; if (nl[s]++ < 16) VKLOG("bak91c pool id: dispatch %016llx writes %#llx +%#llx: %s (pool %#llx..%#llx, buffer %p)", (unsigned long long)key, (unsigned long long)a, (unsigned long long)n,
        g_pid_kn[pid_kind(p)], (unsigned long long)(p ? p->lo : 0), (unsigned long long)(p ? p->hi : 0), p ? (void *)p->buf : NULL);
}
void vk_pool_id_args(uint64_t a, int len, int dispatch) {   // gfx_vk_exec.c: an indirect draw's or dispatch's arguments (len: the bytes it reads)
    if (!a || !pid_on() || !g_pid_n) return;
    int hit = -1; for (int j = 0; j < g_pid_n; j++) { int i = (g_pid_w - 1 - j + 128) % 64; if (a >= g_pid[i].lo && a < g_pid[i].hi) { hit = i; break; } }
    if (hit < 0) return;
    VkBuffer b; VkDeviceSize o; vkpool_t *p = vk_pool_of(a, (uint64_t)len); b = p ? p->buf : VK_NULL_HANDLE; (void)o;
    if (b == g_pid[hit].buf) g_pid_same++;
    else { g_pid_diff++; static int nr; if (nr++ < 16) VKLOG("[RED] bak91c pool id: indirect %s arguments at %#llx read through %s (pool %#llx..%#llx, buffer %p), but dispatch %016llx wrote them through %s (pool from %#llx, buffer %p): two buffers over the same pages (%llu so far)",
        dispatch ? "dispatch" : "draw", (unsigned long long)a, g_pid_kn[pid_kind(p)], (unsigned long long)(p ? p->lo : 0), (unsigned long long)(p ? p->hi : 0), p ? (void *)p->buf : NULL,
        (unsigned long long)g_pid[hit].key, g_pid_kn[g_pid[hit].kind], (unsigned long long)g_pid[hit].plo, (void *)g_pid[hit].buf, (unsigned long long)g_pid_diff); }
    static uint64_t t; const uint64_t ns = s47_ns(); if (ns - t >= 10000000000ull) { t = ns;
        VKLOG("bak91c pool id: indirect arguments inside a scenery shader's written range: same buffer %llu, a different buffer %llu", (unsigned long long)g_pid_same, (unsigned long long)g_pid_diff); }
}
// gwsuffix (11-21, OoT after dirtyidx: images and samplers 1.64 us a draw, 0.94 of it outside tex_desc_for, where a bindless handle read from a uniform buffer goes
// through vk_ubd_read and this scan of up to 1,024 entries, finished or not): g_gw is appended in fence order (vk_gpu_wrote: g_fence + 1, which only grows; its
// compaction and its drop of the oldest keep the order), so the entries not known complete are a suffix: a binary search for its start, then that suffix alone
static int gw_pending_from(uint64_t done) { int lo = 0, hi = g_ngw; while (lo < hi) { int m = (lo + hi) / 2; if (g_gw[m].fence > done) hi = m; else lo = m + 1; } return lo; }
int vk_gpu_pending(uint64_t a, uint64_t n) {
    if (tl_wv) { const uint64_t done = vk_fence_done();   // S1e (parallel recording): a worker: the writes in flight at its hand-out (the ring moves under the replay thread's later writes)
        for (int i = tl_wv->npg - 1; i >= 0; i--) if (tl_wv->pg[i].fence > done && tl_wv->pg[i].lo < a + n && a < tl_wv->pg[i].hi) return 1;
        return 0; }
    if (!g_ngw) return 0;
    const uint64_t done = vk_fence_done(); const int from = gw_pending_from(done);
    for (int i = g_ngw - 1; i >= from; i--) if (g_gw[i].lo < a + n && a < g_gw[i].hi) return 1;
    return 0;
}
// replay2 (see report_now): the queued replay-time reports of the open submission, written into the game's memory now
static int rq_cmp(const void *x, const void *y) { const rpt_q *a = x, *b = y; return a->addr < b->addr ? -1 : a->addr > b->addr ? 1 : a->ord < b->ord ? -1 : a->ord > b->ord; }
static void vk_rpt_flush_(vksub_t *s, const uint32_t n);
void vk_rpt_flush(void) {
    if (!g_nrq) return;
    g_rq_lo = g_rq_hi = 0;
    if (g_cur < 0) { g_nrq = 0; return; }
    vksub_t *s = &g_slot[g_cur]; const uint32_t n = g_nrq; g_nrq = 0; g_rq_flushes++;
    extern int g_rdiag; extern uint64_t g_rd_ns[64], g_rd_n[64]; const uint64_t rt0 = g_rdiag > 0 ? now_ns() : 0;   // replaydiag slot 48
    vk_rpt_flush_(s, n);
    if (rt0) { g_rd_ns[48] += now_ns() - rt0; g_rd_n[48]++; }
}
static void vk_rpt_flush_(vksub_t *s, const uint32_t n) {
    // a GPU write recorded before a report (gwseq) and not known complete: that report waits for the fence, as before
    const uint64_t done = vk_fence_done(); uint64_t lo = ~0ull, hi = 0, seqmax = 0; int sorted = 1;
    for (uint32_t i = 0; i < n; i++) { if (g_rq[i].addr < lo) lo = g_rq[i].addr; if (g_rq[i].addr + 16 > hi) hi = g_rq[i].addr + 16; if (g_rq[i].gwseq > seqmax) seqmax = g_rq[i].gwseq;
        if (i && g_rq[i].addr < g_rq[i - 1].addr) sorted = 0; }
    const int gfrom = gw_pending_from(done); int hit = 0; for (int i = g_ngw - 1; i >= gfrom && !hit; i--) if (g_gw[i].seq <= seqmax && g_gw[i].lo < hi && lo < g_gw[i].hi) hit = 1;
    uint32_t nlate = 0;
    if (hit) for (uint32_t k = 0; k < n; k++) { rpt_q *q = &g_rq[k];
        for (int i = g_ngw - 1; i >= gfrom; i--) if (g_gw[i].seq <= q->gwseq && g_gw[i].lo < q->addr + 16 && q->addr < g_gw[i].hi) { q->type |= 0x80000000u; nlate++; break; } }
    if (nlate) {   // back among the post ops at their own places (last first: an earlier one at the same place ends up before it)
        if (s->npost + (int)nlate > s->cappost) { int c = s->cappost ? s->cappost : 64; while (c < s->npost + (int)nlate) c *= 2; s->post = realloc(s->post, (size_t)c * sizeof *s->post); if (!s->post) die("[vk] post op list"); s->cappost = c; }
        for (uint32_t k = n; k-- > 0;) { const rpt_q *q = &g_rq[k]; if (!(q->type & 0x80000000u)) continue;
            const int at = (int)q->npost < s->npost ? (int)q->npost : s->npost;
            memmove(s->post + at + 1, s->post + at, (size_t)(s->npost - at) * sizeof *s->post); par_out_moved(at, 1);   // (occ-s3fast, S1e fix: the placeholders from at on moved on)
            s->post[at] = (q->type & RQ_HALF) ? (post_op){2, NULL, POST_HALF, 0, q->addr, q->type & 0x3fffffffu, -1, q->wseq, q->ticks, 0}   // occ-s2: its timestamp half only
                                              : (post_op){2, NULL, 8, 0, q->addr, q->type & 0x7fffffffu, -1, q->wseq, q->v0, 0}; s->npost++; }   // occkey: mode 8, the queued value itself (not the completion's occlusion sum)
        g_rpt_late += nlate;
    }
    if (!sorted) qsort(g_rq, n, sizeof *g_rq, rq_cmp);
    // runs of reports over adjacent pages: one prepare (dirty tracking: the pages made writable and marked) and one ledger record each
    const uint64_t PGM = 4095;
    for (uint32_t i = 0; i < n;) {
        if (g_rq[i].type & 0x80000000u) { i++; continue; }
        uint64_t rlo = g_rq[i].addr, rhi = rlo + 16; uint32_t j = i + 1;
        while (j < n && ((g_rq[j].type & 0x80000000u) || (g_rq[j].addr & ~PGM) <= ((rhi - 1) & ~PGM) + PGM + 1)) { if (!(g_rq[j].type & 0x80000000u) && g_rq[j].addr + 16 > rhi) rhi = g_rq[j].addr + 16; j++; }
#if defined(DM_NXN1X)
        nvnc_cpu(rlo, rhi - rlo, NVNC_GPU_WRITEBACK);
#endif
        dm_guest_write_prepare((void *)(uintptr_t)rlo, (size_t)(rhi - rlo));
        for (uint32_t k = i; k < j; k++) if (!(g_rq[k].type & 0x80000000u)) { const uint64_t rec[2] = {g_rq[k].v0, g_rq[k].ticks};
            if (g_rq[k].type & RQ_HALF) memcpy((void *)(uintptr_t)(g_rq[k].addr + 8), &rec[1], 8);   // occ-s2: the timestamp half; the counter word is the GPU's
            else memcpy((void *)(uintptr_t)g_rq[k].addr, rec, sizeof rec); g_rpt_now++; }
        g_rq_runs++; i = j;
    }
}
// what a GPU read of [a, a + n) sees at this point of the command stream, into dst (memory, then the deferred updates in order)
void vk_ubd_read(uint64_t a, uint64_t n, uint8_t *dst) {
    // S1e (parallel recording): a worker never flushes (the queue is the replay thread's; the hand-out flushed the reports recorded before its segment) and reads the landed
    // updates in flight as at its hand-out (g_infl is appended and pruned at the replay thread's later lands); its deferred updates are its own list's (g_ubd)
    if (tl_wv) {
        if (vk_gpu_pending(a, n)) { static _Atomic int k; __atomic_fetch_add(&g_ubd_racy, 1, __ATOMIC_RELAXED); if (atomic_fetch_add(&k, 1) < 20) VKLOG("[RED] a CPU read of %#llx +%llu (worker segment) overlaps a GPU write not known to be complete", (unsigned long long)a, (unsigned long long)n); }
        memcpy(dst, (const void *)(uintptr_t)a, n);
        if (tl_wv->ninf) { const uint64_t done = vk_fence_done();
            for (int i = 0; i < tl_wv->ninf; i++) { const par_infl_t *f = &tl_wv->inf[i]; if (f->fence <= done) continue; uint64_t lo = f->lo > a ? f->lo : a, hi = f->hi < a + n ? f->hi : a + n;
                if (lo < hi) memcpy(dst + (lo - a), f->data + (lo - f->lo), hi - lo); } }
        for (int i = 0; i < g_nubd; i++) { uint64_t lo = g_ubd[i].lo > a ? g_ubd[i].lo : a, hi = g_ubd[i].hi < a + n ? g_ubd[i].hi : a + n;
            if (lo < hi) memcpy(dst + (lo - a), g_ubd[i].p + (lo - g_ubd[i].lo), hi - lo); }
        return;
    }
    if (g_nrq) vk_rpt_flush();   // replay2: a GPU read at this point of the stream sees the reports recorded before it
    if (g_occdiag || g_occ2 >= 2) vk_occd_read(a, n, 0);   // occ-s0: a uniform shadow over a samples-passed report word whose GPU result has not landed (occ-s2: counted always)
    if (vk_gpu_pending(a, n)) { static int k; g_ubd_racy++; if (k < 20) { k++; int cov = 0; for (int i = 0; i < g_nubd; i++) if (g_ubd[i].lo <= a && a + n <= g_ubd[i].hi) cov = 1;
        VKLOG("[RED] a CPU read of %#llx +%llu (%s) overlaps a GPU write not known to be complete", (unsigned long long)a, (unsigned long long)n, cov ? "covered by a deferred update" : "partly from memory"); } }
    memcpy(dst, (const void *)(uintptr_t)a, n);
    if (g_ninfl) { uint64_t done = vk_fence_done();
        for (int i = 0; i < g_ninfl; i++) { if (g_infl[i].fence <= done) continue; uint64_t lo = g_infl[i].lo > a ? g_infl[i].lo : a, hi = g_infl[i].hi < a + n ? g_infl[i].hi : a + n;
            if (lo < hi) memcpy(dst + (lo - a), g_infl[i].data + (lo - g_infl[i].lo), hi - lo); } }
    for (int i = 0; i < g_nubd; i++) { uint64_t lo = g_ubd[i].lo > a ? g_ubd[i].lo : a, hi = g_ubd[i].hi < a + n ? g_ubd[i].hi : a + n;
        if (lo < hi) memcpy(dst + (lo - a), g_ubd[i].p + (lo - g_ubd[i].lo), hi - lo); }
}
// a uniform binding [a, a + n) that deferred updates overlap: its shadow (0 when none overlaps). Shadows are kept per (address, size) until the next update.
int vk_ubd_shadow(uint64_t a, uint64_t n, VkBuffer *buf, VkDeviceSize *off) {
    if (!g_nubd || !vk_ubd_overlaps(a, n)) return 0;   // (a binding no deferred update touches reads memory on the GPU, after the in-flight copies: no shadow needed)
    enum { NC = 64 }; static _Thread_local struct { uint64_t a, n, gen; VkBuffer b; VkDeviceSize o; } c[NC];   // R5 (parallel recording): per thread (inventory item 12; only the replay thread builds shadows today)
    unsigned k = (unsigned)((a ^ n * 0x9E3779B97F4A7C15ull) >> 6) % NC;
    extern int g_x_noshc; if (c[k].a == a && c[k].n == n && c[k].gen == g_ubd_gen && !g_x_noshc) { *buf = c[k].b; *off = c[k].o; return 1; }
    uint8_t *d = vk_scratch(n, V.props.limits.minUniformBufferOffsetAlignment, buf, off); vk_ubd_read(a, n, d); __atomic_fetch_add(&g_ubd_bytes, n, __ATOMIC_RELAXED);   // (S1e (parallel recording): workers count here too)
    c[k].a = a; c[k].n = n; c[k].gen = g_ubd_gen; c[k].b = *buf; c[k].o = *off; __atomic_fetch_add(&g_ubd_shadows, 1, __ATOMIC_RELAXED);
    return 1;
}
// ---- S1e (parallel recording): the hand-out, the workers' view, the join's helpers (gfx_vk.h; gfx_vk_exec.c seg_par_*) ----
int vk_par_on_worker(void) { return tl_worker != NULL; }
vk_par_view *vk_par_view_new(void) { vk_par_view *v = calloc(1, sizeof *v); if (!v) die("[vk] S1e: no memory for a segment's view"); return v; }
static uint32_t g_par_token;
// occ-s3fast (S1e fix, the "[RED] S1e: a worker segment's place among the post ops is gone ... its 0 post ops" lines with DM_VK_PARALLEL=2 under s2 / s3): a merge replaces a
// placeholder by the segment's n post ops, so every later placeholder moves by n - 1. Under s0 / s1 a segment always had post ops (its passes' kind-4 occlusion spans: n >= 1,
// later placeholders moved on, the search from post_at forward found them); under s2 / s3 there are none (no span posts on that path), n = 0 moved the later placeholders one
// back, before the post_at their search started from. The views handed out and not merged yet are listed, and every move of the post list (a merge, a late report put back at
// its place) moves their post_at with it, so the search starts at the placeholder itself
static vk_par_view **g_par_out; static int g_par_nout, g_par_capout;
static void par_out_add(vk_par_view *v) {
    if (g_par_nout == g_par_capout) { const int c = g_par_capout ? 2 * g_par_capout : 16; void *p = realloc(g_par_out, sizeof *g_par_out * (size_t)c); if (!p) die("[vk] S1e: no memory for the handed-out views"); g_par_out = p; g_par_capout = c; }
    g_par_out[g_par_nout++] = v;
}
static void par_out_moved(int at, int d) {   // the post ops from index at on moved by d (a merge: at is past the merged placeholder; an insertion: at is the inserted index, d 1)
    for (int i = 0; i < g_par_nout; i++) if (g_par_out[i]->post_at >= at) g_par_out[i]->post_at += d;
}
static void par_out_drop(const vk_par_view *v) { for (int i = 0; i < g_par_nout; i++) if (g_par_out[i] == v) { g_par_out[i] = g_par_out[--g_par_nout]; return; } }
static void par_view_take(vksub_t *s, vk_par_view *v) {   // the replay thread's lists as at this point of the stream
    v->ngw = s->ngw; memcpy(v->gw, s->gw, sizeof *s->gw * (size_t)s->ngw); v->ngml = s->ngml; v->wseq = s->wseq; v->nsubs = g_nsubs;
    const uint64_t done = vk_fence_done(); const int from = gw_pending_from(done);
    v->npg = 0; if (g_ngw - from > v->cappg) { int c = v->cappg ? v->cappg : 64; while (c < g_ngw - from) c *= 2; void *p = realloc(v->pg, (size_t)c * sizeof *v->pg); if (!p) die("[vk] S1e: no memory for a segment's view"); v->pg = p; v->cappg = c; }
    for (int i = from; i < g_ngw; i++) v->pg[v->npg++] = (par_gw_t){g_gw[i].lo, g_gw[i].hi, g_gw[i].fence, g_gw[i].seq};
    v->ninf = 0;
    for (int i = 0; i < g_ninfl; i++) { if (g_infl[i].fence <= done) continue;
        if (v->ninf == v->capinf) { int c = v->capinf ? 2 * v->capinf : 64; void *p = realloc(v->inf, (size_t)c * sizeof *v->inf); if (!p) die("[vk] S1e: no memory for a segment's view"); v->inf = p; v->capinf = c; }
        v->inf[v->ninf++] = (par_infl_t){g_infl[i].lo, g_infl[i].hi, g_infl[i].fence, g_infl[i].data}; }
    v->oq_counting = g_occ2 >= 2 ? occ2_counting() : g_oq_counting; v->oq_epoch = g_oq_epoch; v->npost = 0;   // (occ-s2: the window state, no epoch)
    if (g_occ2 >= 2) occ2_view(v);   // occ-s2: the bits of the queries the worker begins
}
// The segment's place among the post ops (a placeholder), the replay thread's R chunk ended and listed, the worker's W chunk listed after it, a new R chunk begun for what
// follows the segment. The caller ended its pass (no pass may span two command buffers) and flushed the queued reports (the worker's uniform reads see them)
int vk_par_handout(int worker, vk_par_view *v) {
    vksub_t *const s = cur();
    par_view_take(s, v);
    v->token = ++g_par_token; v->post_at = s->npost; post((post_op){POST_SEG, NULL, 0, v->token, 0, 0, -1});
    par_out_add(v);   // occ-s3fast (S1e fix): its placeholder's index kept exact while the part's earlier segments are merged
    if (vk_f0_result(vkEndCommandBuffer(s->main), "vkEndCommandBuffer(R chunk)", s->nmain_used) < 0) { s->cb_err = 1; static int n; if (n++ < 3) VKLOG("[RED] S1e: a replay-thread chunk was refused at its end (part %d): the submission is dropped", s->nmain_used); }
    par_chunk_r(s, s->main);
    const int wi = vk_chunk_add(worker);
    if (s->nrcb_used == s->nrcb_alloc) {
        if (s->nrcb_alloc == s->caprcb) { int c = s->caprcb ? 2 * s->caprcb : 16; void *p = realloc(s->rcb, (size_t)c * sizeof *s->rcb); if (!p) die("[vk] S1e: no memory for the replay thread's chunks"); s->rcb = p; s->caprcb = c; }
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        VKCHK(vkAllocateCommandBuffers(V.dev, &ai, &s->rcb[s->nrcb_alloc])); s->nrcb_alloc++; }
    s->main = s->rcb[s->nrcb_used++]; g_vk_rt_main.cb = s->main;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(nvn2_begin(s->main, &bi, "vkBeginCommandBuffer (R chunk)")); g_vk_rt_main.gen++;
    return wi;
}
static _Thread_local int tl_oq_saved = -1; static _Thread_local uint64_t tl_o2_saved[2];   // (occ-s1: the old span's serials too)
void vk_par_wk_begin(vk_par_view *v) { v->npost = 0; if (v->o2b && v->o2n) memset(v->o2b, 0, sizeof *v->o2b * (size_t)((v->o2n + 63) / 64)); tl_wv = v; tl_oq_saved = g_oq_active; g_oq_active = -1; tl_o2_saved[0] = g_occ2_slo; tl_o2_saved[1] = g_occ2_shi; }   // (the helper on the replay thread: its open query, if any, kept aside; occ-s2: a run again begins its queries again)
void vk_par_wk_end(void) { tl_wv = NULL; g_oq_active = tl_oq_saved; tl_oq_saved = -1; g_occ2_slo = tl_o2_saved[0]; g_occ2_shi = tl_o2_saved[1]; }
void vk_par_merge_posts(vk_par_view *v) {   // replay thread: the segment's post ops where its placeholder is; the queued reports' places (npost) after it move with it
    vksub_t *const s = cur(); int at = -1;
    for (int i = v->post_at < 0 ? 0 : v->post_at; i < s->npost; i++) if (s->post[i].kind == POST_SEG && s->post[i].v == v->token) { at = i; break; }
    par_out_drop(v);   // occ-s3fast (S1e fix): no longer handed out
    if (at < 0) { static int n; if (n++ < 8) VKLOG("[RED] S1e: a worker segment's place among the post ops is gone (token %u): its %d post ops are dropped", v->token, v->npost); return; }
    const int n = v->npost, d = n - 1;
    if (s->npost + d > s->cappost) { int c = s->cappost ? s->cappost : 64; while (c < s->npost + d) c *= 2; void *p = realloc(s->post, (size_t)c * sizeof *s->post); if (!p) die("[vk] post op list"); s->post = p; s->cappost = c; }
    memmove(s->post + at + n, s->post + at + 1, (size_t)(s->npost - at - 1) * sizeof *s->post);
    if (n) memcpy(s->post + at, v->post, (size_t)n * sizeof *s->post);
    s->npost += d;
    if (d) par_out_moved(at + 1, d);   // occ-s3fast (S1e fix): the later placeholders moved by d (n = 0 under s2 / s3: one back)
    for (uint32_t k = 0; k < g_nrq; k++) if ((int)g_rq[k].npost > at) g_rq[k].npost = (uint32_t)((int)g_rq[k].npost + d);
    v->npost = 0; v->post_at = -1;
}
int vk_par_pick(void) {
    int best = 0; unsigned bl = ~0u;
    for (int i = 0; i < g_npar; i++) { vk_worker *const w = &g_par[i]; pthread_mutex_lock(&w->mu); const unsigned l = w->qt - w->qh; pthread_mutex_unlock(&w->mu);   // (the running job is still at qh)
        if (l < bl) { bl = l; best = i; } }
    return best;
}
void vk_par_help_run(void (*fn)(void *), void *arg) {
    vk_worker *const h = &g_par[g_npar]; vk_worker *const sw = tl_worker; vk_rec_target *const srt = g_vk_rt;
    void *const src = (void *)vk_rec_ctx_cur(), *const sre = vk_rec_eden_cur();
    tl_worker = h; g_vk_rt = &h->rt; vk_rec_ctx_bind(h->rec, h->rec_e);
    fn(arg);
    tl_worker = sw; g_vk_rt = srt; vk_rec_ctx_bind(src, sre);
}
int vk_par_help_one(void) {   // the last job still queued (not running) on the worker with the most queued
    int bw = -1; unsigned bq = 0;
    for (int i = 0; i < g_npar; i++) { vk_worker *const w = &g_par[i]; pthread_mutex_lock(&w->mu); const unsigned q = w->qt - w->qh - (w->busy ? 1u : 0u); pthread_mutex_unlock(&w->mu); if (q > bq) { bq = q; bw = i; } }
    if (bw < 0) return 0;
    vk_worker *const w = &g_par[bw]; vk_par_job j = {0}; int got = 0;
    pthread_mutex_lock(&w->mu);
    if (w->qt - w->qh > (unsigned)(w->busy ? 1 : 0)) { w->qt--; j = w->q[w->qt % PAR_Q]; got = 1; pthread_cond_broadcast(&w->cv); }
    pthread_mutex_unlock(&w->mu);
    if (!got) return 0;
    vk_par_help_run(j.fn, j.arg);
    return 1;
}
int vk_chunk_wait_all(void) {
    vksub_t *const s = cur(); int bad = 0; unsigned joined = 0;
    pthread_mutex_lock(&g_chunk_mu);
    for (int i = s->chunk0; i < s->nchunk; i++) { if (s->chunk[i].worker < 0) continue; while (!s->chunk[i].done) pthread_cond_wait(&g_chunk_cv, &g_chunk_mu); bad += s->chunk[i].done < 0;
        if (s->chunk[i].worker < PAR_MAX) joined |= 1u << s->chunk[i].worker; }
    pthread_mutex_unlock(&g_chunk_mu);
    for (int w = 0; w < g_npar_ctx; w++) if ((joined >> w & 1) || w == g_npar) vk_rec_ctx_fold(g_par[w].rec, g_par[w].rec_e);   // (the helper's always: it may have run any of them)
    return bad;
}
void vk_chunk_get(int i, vk_chunk *out) { vksub_t *const s = cur(); pthread_mutex_lock(&g_chunk_mu); if (i < 0 || i >= s->nchunk) die("[vk] S1e: chunk %d of %d", i, s->nchunk); *out = s->chunk[i]; pthread_mutex_unlock(&g_chunk_mu); }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *h, int n) { (void)h; (void)n; }

// completion: a queue of committed submissions, each with its CPU-visible effects
typedef struct cpend { uint64_t fence, frame, content_sub; nvf0_submission f0; post_op *ops; int n; int slot, rslot; int npass; pass_meta pmeta[PASSQ]; og_gap *og; og_marks ogm; struct cpend *next; } cpend;   // og: ootgap (NULL: not measured)
static cpend *g_ch, *g_ct; static pthread_mutex_t g_cmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_ccv = PTHREAD_COND_INITIALIZER;
uint64_t guest_ns(void);
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; } return nvn_timestamp_from_ns(g); }   // r79: was g * 12 / 625 (19.2 MHz): GPU spans read 32x short through nvnDeviceGetTimestampInNanoseconds
// GPU time per NVN frame (the sum of its submissions' timestamp spans), for the timing line's GPU column: the frame a submission belongs to is the one being replayed
static uint64_t g_frame_gpu[64], g_frame_gpu_of[64];
void mtl_set_frame(uint64_t f) { __atomic_store_n(&g_cur_frame, f, __ATOMIC_RELAXED); }
double mtl_gpu_ms(uint64_t f) { return g_frame_gpu_of[f % 64] == f + 1 ? g_frame_gpu[f % 64] / 1e6 : -1; }
// s67 (apply_s67.py): per NVN frame, what its submissions held (s67_sub, vk_submit on the replay thread) and, once a later frame's submission completes,
// one line for a frame whose GPU time (as mtl_gpu_ms) is >= DM_VK_S67_GPU_MS (default 60, 0 off; s67_done on the completion thread)
typedef struct { uint64_t of, passes, draws, exec_ns, ev[5]; uint32_t subs, parts; } s67_rec;
static s67_rec g_s67[64];
static void s67_sub(uint64_t frame) {
    static uint64_t pp, pd, pe, ps, pv[5];
    s67_rec *r = &g_s67[frame % 64]; if (r->of != frame + 1) { memset(r, 0, sizeof *r); r->of = frame + 1; }
    r->subs++; r->parts += 1 + (uint32_t)(g_s61_sent - ps); r->passes += g_vkst.passes - pp; r->draws += g_vkst.draws - pd; r->exec_ns += g_vkst.ns_exec - pe;
    for (int i = 0; i < 5; i++) { uint64_t v = __atomic_load_n(&g_s67_ev[i], __ATOMIC_RELAXED); r->ev[i] += v - pv[i]; pv[i] = v; }
    ps = g_s61_sent; pp = g_vkst.passes; pd = g_vkst.draws; pe = g_vkst.ns_exec;
}
static double s67_thr(void) { static double t = -1; if (t < 0) { const char *e = getenv("DM_VK_S67_GPU_MS"); t = e && *e ? atof(e) : 60; if (t < 0) t = 0; } return t; }
typedef struct { double ms; pass_meta m; } s67_top_t;
static s67_top_t g_s67_top[5]; static int g_s67_ntop; static uint64_t g_s67_pf = ~0ull, g_s67_t0, g_s67_t1;
static void s67_pass(double ms, const pass_meta *m) {   // the completion thread, DM_VK_PASS_TIMES=1 only: the frame's 5 most expensive passes
    int i; if (g_s67_ntop < 5) i = g_s67_ntop++; else { if (g_s67_top[4].ms >= ms) return; i = 4; }
    while (i > 0 && g_s67_top[i - 1].ms < ms) { g_s67_top[i] = g_s67_top[i - 1]; i--; }
    g_s67_top[i].ms = ms; g_s67_top[i].m = *m;
}
static void s67_span(uint64_t a, uint64_t b) { if (!g_s67_t0 || a < g_s67_t0) g_s67_t0 = a; if (b > g_s67_t1) g_s67_t1 = b; }
static int pass_times_on(void);
static void s67_done(uint64_t pf) {
    double thr = s67_thr(), g; if (thr <= 0 || pf == ~0ull || (g = mtl_gpu_ms(pf)) < thr) return;
    static int nlog; if (nlog >= 4000) return; nlog++;
    s67_rec r = g_s67[pf % 64]; if (r.of != pf + 1) memset(&r, 0, sizeof r);
    char top[480]; int o = 0;
    if (!pass_times_on()) o = snprintf(top, sizeof top, "%s", g_pass_times_requested ?
                                      "n/a (timestamp pool unavailable)" : "n/a (DM_VK_PASS_TIMES=0)");
    else for (int i = 0; i < g_s67_ntop && o < (int)sizeof top - 80; i++) { const pass_meta *m = &g_s67_top[i].m;
        if (m->fmt == VK_TS_SCALE_WRITEBACK)
            o += snprintf(top + o, sizeof top - (size_t)o, "%s%.1f ms 2X writeback %ux%u (#%u)",
                          i ? ", " : "", g_s67_top[i].ms, m->w, m->h, m->fidx);
        else if (m->fmt == VK_TS_DISPATCH) o += snprintf(top + o, sizeof top - (size_t)o, "%s%.1f ms dispatch ..%08x %ux%u (#%u)", i ? ", " : "", g_s67_top[i].ms, m->draws, m->w, m->h, m->fidx);
        else o += snprintf(top + o, sizeof top - (size_t)o, "%s%.1f ms %ux%u fmt %u, %u draws (#%u)", i ? ", " : "", g_s67_top[i].ms, m->w, m->h, m->fmt, m->draws & 0xffffu, m->fidx); }
    if (!o) snprintf(top, sizeof top, "-");
    VKLOG("s67: frame %llu GPU %.1f ms (first start to last end %.1f ms): passes %llu, draws %llu, submissions %u in %u parts, replay %.1f ms; "
          "s48 range imports %llu (%.1f ms), zero-copy pool imports %llu (%.1f ms), pools made of chunks %llu; top passes %s",
          (unsigned long long)pf, g, g_s67_t1 > g_s67_t0 ? (double)(g_s67_t1 - g_s67_t0) * V.props.limits.timestampPeriod / 1e6 : 0.0, (unsigned long long)r.passes,
          (unsigned long long)r.draws, r.subs, r.parts, r.exec_ns / 1e6, (unsigned long long)r.ev[0], r.ev[1] / 1e6, (unsigned long long)r.ev[2], r.ev[3] / 1e6,
          (unsigned long long)r.ev[4], top);
}
static void s67_frame(uint64_t f) {
    if (f == g_s67_pf) return;
    s67_done(g_s67_pf);
    // Only the completion thread owns the pass accumulators. Report before
    // adding timestamps from the first submission of the next frame.
    if (g_s67_pf != ~0ull && pass_times_on() && ++g_r138_nf % 120 == 0)
        vk_pass_report();
    g_s67_pf = f;
    g_s67_ntop = 0;
    g_s67_t0 = g_s67_t1 = 0;
}
double vk_gpu_ms_recent(void) {   // the latest finished frame's GPU time among the last 8 (-1: none known)
    uint64_t f = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED);
    for (uint64_t k = 2; k < 8 && k <= f; k++) { double g = mtl_gpu_ms(f - k); if (g >= 0) return g; }
    return -1;
}
// ---------------- occ-s0 / occ-s1: occlusion instruments and the per-draw query ring (design\OCCLUSION_REDESIGN.md steps 0 and 1) ----------------
// Stream serial (gfx_vk_exec.c g_oser, vk_occ2_serial): every draw-type record (DRAW, DRAW_IDX, MULTI_DRAW, DRAW_TEX) advances the recording thread's serial by one, recorded
// or not (a false render condition, an indirect-count follower, a draw prepare_draw drops). The dry pass names the serials when it runs (a segment carries its first draw's,
// seg_desc.oser0; a worker counts from it; the replay thread's count is checked against the dry pass's); without it the replay thread counts the same records. No atomics.
// occ-s1 (DM_VK_OCC2=s1, read at device init, off by default): one VkQueryPool of 64K occlusion queries, slot = serial mod 64K, reset in chunks of 4K from the host before the
// chunk's first use in a lap (vkResetQueryPool; without hostQueryReset vkCmdResetQueryPool in the part's upload command buffer), after the submission that last recorded into or
// read the chunk's previous lap has completed (a wait: logged, expected never). Each draw recorded while counting (after the first samples-passed reset, as g_oq_counting) begins
// its own query right before its draw and ends it on every exit of the draw paths (vk_occ2_end), at the pass end, a report or a reset. Only one occlusion query may be active in
// a command buffer (VUID-vkCmdBeginQuery-queryPool-01922), so on this path the old per-span queries are not recorded: each old span becomes a serial range (kind 4 mode 2) and
// the completion thread sums the old epoch from the same per-draw queries. The values written into guest memory are the old path's (occkey / fixed / occland as configured);
// the new part is the report window [serial at the last reset, serial at the report) whose sum is compared with the old span sum of the same report (equal by construction
// unless the serial naming or a span's bookkeeping is off). The 4096 per-submission bound is not used on this path (the old pool is never begun)
uint64_t vk_occ2_serial(void);   // gfx_vk_exec.c: the calling thread's stream serial (the next draw's)
extern uint64_t g_vk_pends;      // gfx_vk_exec.c: the replay thread's pass ends (occ-s0 distances, the overflow context)
int g_vk_hqr;                    // vk_init: hostQueryReset (Vulkan 1.2 / VK_EXT_host_query_reset) enabled
enum { OCC2_RING = 1 << 16, OCC2_CHUNK = 1 << 12, OCC2_NCK = OCC2_RING / OCC2_CHUNK };
static VkQueryPool g_occ2_q; static PFN_vkResetQueryPool g_occ2_hreset;
static struct { uint64_t lap, fence; } g_occ2_ck[OCC2_NCK];   // lap: 1 + the ring lap the chunk was last reset for (0: never); fence: the last submission recording into it or reading it
static uint64_t g_occ2_win_lo;   // replay thread: the serial at the last samples-passed reset
static uint64_t g_o2_bits[OCC2_RING / 64];   // occ-s2: a bit per ring slot whose query was begun in its chunk's current lap (replay thread; a worker's come in at its join)
static uint64_t g_o2_wklo, g_o2_wkhi, g_o2_holo, g_o2_hon;   // occ-s2: the draw serials of the open part's handed-out segments (until its join); the last hand-out's
static struct { uint64_t q, sub_q, sub_max, resets, waits, wait_ns, red, longw, lost, cb_off; } g_o2;   // replay thread, since the last line
static uint64_t g_o2_wq, g_o2_wunready, g_o2_wcb_off;   // workers (atomic adds): queries begun, draws whose chunk the hand-out had not reset (never), end on another buffer
static uint64_t g_o2c_cmp, g_o2c_mis, g_o2c_maxd, g_o2c_unrec, g_o2c_zero_unrec, g_o2c_lostold, g_o2c_slots;   // the completion thread, since start (the line takes differences)
static int g_occ2_redn;
static void occ2_red(const char *what, uint64_t a, uint64_t b) { g_o2.red++; if (g_occ2_redn++ < 16) VKLOG("[RED] occ-s1: %s (%llu, %llu)", what, (unsigned long long)a, (unsigned long long)b); }
static int occ2_s2_init(int want);   // occ-s2 / occ-s3 (below): the resolve's buffer and compute pipeline, the conditional rendering entry points; the level taken
static void occ2_init(void) {   // slots_init, after g_oq
    // occ-s2 / occ-s3: DM_VK_OCC2 = 0 (unset) the old path exactly, s1 step 1, s2 step 2 (GPU resolve, stream-order landing), s3 steps 2 and 3 (render conditions on the GPU)
    const char *e = getenv("DM_VK_OCC2"); const int want = !e || !*e ? 0 : (e[0] == 's' || e[0] == 'S') && e[1] >= '1' && e[1] <= '3' && !e[2] ? e[1] - '0' : -1;
    if (want < 0 && strcmp(e, "0")) VKLOG("[RED] occ-s2: DM_VK_OCC2=%s is not known in this build (0, s1, s2, s3): the old path", e);
    if (want <= 0) return;
    if (want == 1 && !g_oq) { VKLOG("occ-s1: DM_VK_OCC2=s1 not taken: the old path has no occlusion queries to compare with (DM_VK_OQ=0, or DM_VK_OCC_KEYED=0 with DM_VK_OCC_FIXED >= 0, or no pool)"); return; }
    VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, OCC2_RING};
    if (vkCreateQueryPool(V.dev, &qi, NULL, &g_occ2_q) != VK_SUCCESS) { g_occ2_q = VK_NULL_HANDLE; VKLOG("[RED] occ-s1: the 64K occlusion query ring could not be made: the old path"); return; }
    if (g_vk_hqr) { g_occ2_hreset = (PFN_vkResetQueryPool)vkGetDeviceProcAddr(V.dev, "vkResetQueryPool"); if (!g_occ2_hreset) g_occ2_hreset = (PFN_vkResetQueryPool)vkGetDeviceProcAddr(V.dev, "vkResetQueryPoolEXT"); }
    if (want >= 2) { g_occ2 = occ2_s2_init(want); if (g_occ2 < 2) { g_occ2 = 0; VKLOG("[RED] occ-s2: DM_VK_OCC2=s%d not taken (above): the old path", want); } return; }   // (its own line)
    g_occ2 = 1;
    VKLOG("occ-s1: DM_VK_OCC2=s1: one occlusion query per draw by stream serial (ring %d, chunks of %d reset %s), %s; the old path's values written unchanged, its span sums taken from "
          "the same queries; per samples-passed report the window sum compared with the old span sum (log every 600 presents)", OCC2_RING, OCC2_CHUNK,
          g_occ2_hreset ? "from the host: vkResetQueryPool, hostQueryReset" : "by vkCmdResetQueryPool in the part's upload command buffer: no hostQueryReset", g_oq_flags ? "precise" : "not precise (no occlusionQueryPrecise)");
}
// replay thread: every chunk holding serials [lo, hi) reset for the lap those serials are in (made so now if not yet). 0 = some of them are gone or cannot be had
static int occ2_prep(uint64_t lo, uint64_t hi) {
    if (hi <= lo) return 1;
    const uint64_t cur = g_fence + 1; int ok = 1;
    for (uint64_t c = lo / OCC2_CHUNK; c <= (hi - 1) / OCC2_CHUNK; c++) {
        const unsigned i = (unsigned)(c % OCC2_NCK); const uint64_t lap = c / OCC2_NCK + 1;
        if (g_occ2_ck[i].lap == lap) continue;
        if (g_occ2_ck[i].lap > lap) { ok = 0; continue; }   // reset for a later lap already: those serials' queries are gone (a window longer than the ring)
        const uint64_t f = g_occ2_ck[i].fence;
        if (f >= cur) { occ2_red("a chunk's previous lap is still in the open submission (more than 60K draws in one submission): its draws not counted", c, f); ok = 0; continue; }
        if (f > vk_fence_done()) { const uint64_t t0 = now_ns(); g_o2.waits++;
            if (!mtl_wait(f, 2000000000ull)) { occ2_red("the wait for a chunk's previous lap did not end in 2 s: its draws not counted", c, f); ok = 0; continue; }
            g_o2.wait_ns += now_ns() - t0; }
        if (g_occ2_hreset) g_occ2_hreset(V.dev, g_occ2_q, i * OCC2_CHUNK, OCC2_CHUNK);
        else { vk_up_lock(); vkCmdResetQueryPool(vk_upload_cmd(), g_occ2_q, i * OCC2_CHUNK, OCC2_CHUNK); vk_up_unlock(); }   // (occ-s2: the upload buffer's lock, R8: workers record into it)
        memset(&g_o2_bits[i * (OCC2_CHUNK / 64)], 0, OCC2_CHUNK / 8);   // occ-s2: none of its queries begun in this lap yet
        __atomic_store_n(&g_occ2_ck[i].lap, lap, __ATOMIC_RELEASE); g_occ2_ck[i].fence = cur; g_o2.resets++;
    }
    return ok;
}
static void occ2_mark(uint64_t lo, uint64_t hi) {   // replay thread: the open submission records into or reads serials [lo, hi) (their chunks' next reset waits for it)
    if (hi <= lo) return; const uint64_t cur = g_fence + 1;
    for (uint64_t c = lo / OCC2_CHUNK; c <= (hi - 1) / OCC2_CHUNK; c++) g_occ2_ck[c % OCC2_NCK].fence = cur;
}
// gfx_vk_exec.c seg_par_handout: a worker segment's draws [ser0, ser0 + n): their chunks reset here, in stream order (the worker only checks)
void vk_occ2_handout(uint64_t ser0, uint64_t n) { g_o2_holo = ser0; g_o2_hon = n; if (!g_occ2 || !n) return; (void)occ2_prep(ser0, ser0 + n); occ2_mark(ser0, ser0 + n);
    if (g_occ2 >= 2) { if (g_o2_wkhi <= g_o2_wklo) { g_o2_wklo = ser0; g_o2_wkhi = ser0 + n; } else { if (ser0 < g_o2_wklo) g_o2_wklo = ser0; if (ser0 + n > g_o2_wkhi) g_o2_wkhi = ser0 + n; } } }   // occ-s2: no resolve over them before the join
void vk_occ2_end(void) {   // the draw's query ended (gfx_vk_exec.c draw / draw_idx exits; the pass end, a report and a reset through oq_end)
    if (g_occ2_act < 0) return;
    if (g_occ2_cb != g_vk_rt->cb) { if (tl_wv) __atomic_fetch_add(&g_o2_wcb_off, 1, __ATOMIC_RELAXED); else g_o2.cb_off++; }   // (never: a pass end inside a draw ends it first)
    vkCmdEndQuery(g_occ2_cb, g_occ2_q, (uint32_t)g_occ2_act); g_occ2_act = -1;
}
// gfx_vk_exec.c draw paths, right after vk_oq_draw with the pass open: draw serial ser's own query (any recording thread)
void vk_occ2_begin(VkCommandBuffer cb, uint64_t ser) {
    if (!g_occ2) return;
    const int counting = tl_wv ? tl_wv->oq_counting : g_occ2 >= 2 ? occ2_counting() : g_oq_counting;   // (occ-s2: the window state; no g_oq_counting)
    if (!counting) return;
    if (g_occ2 == 1) { if (g_oq_active < 0) { g_oq_active = OQ_S1SPAN; g_occ2_slo = ser; }   // the old path's span, as serials
        g_occ2_shi = ser + 1; }
    if (g_occ2_act >= 0) vk_occ2_end();
    const unsigned i = (unsigned)(ser / OCC2_CHUNK % OCC2_NCK); const uint64_t lap = ser / OCC2_RING + 1; const uint32_t slot = (uint32_t)(ser % OCC2_RING);
    if (tl_wv) { if (__atomic_load_n(&g_occ2_ck[i].lap, __ATOMIC_ACQUIRE) != lap) { __atomic_fetch_add(&g_o2_wunready, 1, __ATOMIC_RELAXED); return; } __atomic_fetch_add(&g_o2_wq, 1, __ATOMIC_RELAXED);
        if (tl_wv->o2b && ser >= tl_wv->o2lo && ser - tl_wv->o2lo < tl_wv->o2n) { const uint64_t k = ser - tl_wv->o2lo; tl_wv->o2b[k >> 6] |= 1ull << (k & 63); } }   // occ-s2: its own bits (merged at the join)
    else { if (g_occ2_ck[i].lap != lap && !occ2_prep(ser, ser + 1)) return; g_occ2_ck[i].fence = g_fence + 1; g_o2.q++; g_o2.sub_q++; g_o2_bits[slot >> 6] |= 1ull << (slot & 63); }
    g_occ2_act = (int)slot; g_occ2_cb = cb;
    vkCmdBeginQuery(cb, g_occ2_q, (uint32_t)g_occ2_act, g_oq_flags);
}
static void occ2_span_end(void) {   // oq_end with DM_VK_OCC2=s1: the draw's query ended, the old span posted as its serials (summed per epoch by the completion thread)
    vk_occ2_end();
    if (g_oq_active != OQ_S1SPAN) return;
    const uint32_t ep = tl_wv ? tl_wv->oq_epoch : g_oq_epoch;
    if (!tl_wv) { (void)occ2_prep(g_occ2_slo, g_occ2_shi); occ2_mark(g_occ2_slo, g_occ2_shi); }   // (a chunk inside it no draw was recorded in this lap: reset, so it reads 0; a worker's: its hand-out)
    post((post_op){4, NULL, 2, ep, g_occ2_slo, 0, -1, 0, g_occ2_shi}); g_oq_active = -1;
}
static void occ2_reset_at(void) { g_occ2_win_lo = vk_occ2_serial(); }   // vk_oq_reset (replay thread)
// the completion thread: the available results of ring slots [lo, hi) summed (*unrec: slots not available = no query recorded since their reset)
static uint64_t *g_o2buf;
static uint64_t occ2_sum(uint64_t lo, uint64_t hi, uint64_t *unrec) {
    if (!g_o2buf && !(g_o2buf = malloc(2 * sizeof *g_o2buf * OCC2_RING))) { if (unrec) *unrec += hi - lo; return 0; }
    uint64_t sum = 0, un = 0;
    while (lo < hi) { const uint32_t f = (uint32_t)(lo % OCC2_RING); const uint32_t n = (uint32_t)(hi - lo < OCC2_RING - f ? hi - lo : OCC2_RING - f);
        const VkResult r = vkGetQueryPoolResults(V.dev, g_occ2_q, f, n, (size_t)n * 16, g_o2buf, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (r != VK_SUCCESS && r != VK_NOT_READY) un += n;
        else for (uint32_t k = 0; k < n; k++) { if (g_o2buf[2 * k + 1]) sum += g_o2buf[2 * k]; else un++; }
        lo += n; }
    if (unrec) *unrec += un;
    return sum;
}
// ---- occ-s0 (DM_VK_OCCDIAG=1): report landing age, guest reads before landing, the overflow context (the condition census: gfx_vk_exec.c vk_occd_exec_log)
static uint64_t g_od_ft[16];   // per present (& 15): (present & 0xffffff) << 40 | its frame time in ns (the present path)
static uint64_t g_od_frames, g_od_slow;   // presents and frames over 33 ms (the present path; the line takes differences)
static uint64_t g_od_lag[5];   // the completion thread: presents from a report's replay to its GPU result available (fence done): 0, 1, 2, 3, 4+
static void occd_present(void) {   // vk_pass_frame, at each present (before g_tex_frame moves on)
    extern uint64_t g_tex_frame; static uint64_t t0; const uint64_t t = now_ns(), pf = g_tex_frame;
    if (t0) { uint64_t ns = t - t0; if (ns >= 1ull << 40) ns = (1ull << 40) - 1;
        __atomic_store_n(&g_od_ft[pf & 15], (pf & 0xffffff) << 40 | ns, __ATOMIC_RELEASE);
        __atomic_fetch_add(&g_od_frames, 1, __ATOMIC_RELAXED); if (ns > 33000000ull) __atomic_fetch_add(&g_od_slow, 1, __ATOMIC_RELAXED); }
    t0 = t;
}
typedef struct { uint64_t pf; uint64_t n[6]; int used; } occd_pa;   // a present's samples-passed reports by the age of the value written (0..3 presents, 4 fixed, 5 the real count after the fence)
static occd_pa g_od_pa[16];
static uint64_t g_od_age[3][6];   // the line's window: [0] frames <= 33 ms, [1] frames > 33 ms, [2] frame time not known
static void occd_fold(occd_pa *a) {
    if (!a->used) return;
    const uint64_t w = __atomic_load_n(&g_od_ft[a->pf & 15], __ATOMIC_ACQUIRE);
    const int k = (w >> 40) == (a->pf & 0xffffff) ? (w & ((1ull << 40) - 1)) > 33000000ull : 2;
    for (int i = 0; i < 6; i++) g_od_age[k][i] += a->n[i];
    a->used = 0;
}
static void occd_age(int age) {
    extern uint64_t g_tex_frame; occd_pa *a = &g_od_pa[g_tex_frame & 15];
    if (!a->used || a->pf != g_tex_frame) { occd_fold(a); memset(a, 0, sizeof *a); a->pf = g_tex_frame; a->used = 1; }
    a->n[age < 0 ? 4 : age > 5 ? 5 : age]++;
}
// samples-passed report words whose GPU result has not landed (their submission not done): open addressing by address and by 4 KB page, rebuilt with the pending ones when half full
typedef struct { uint64_t a, f, pf, ser, nsub; uint32_t res; } occd_w;   // (occ-s2: res = its GPU write is recorded, a resolve point passed: GPU readers after it in the stream see it)
typedef struct { uint64_t pg, f; } occd_pg;
enum { ODW_N = 1 << 15 };
static occd_w *g_odw; static occd_pg *g_odp; static uint32_t g_odw_n, g_odp_n; static uint64_t g_odw_full;
static inline uint32_t odh(uint64_t k) { return (uint32_t)((k * 0x9E3779B97F4A7C15ull) >> 49); }   // 15 bits
static occd_w *odw_find(uint64_t a) { for (uint32_t i = odh(a >> 4), k = 0; k < ODW_N; k++, i = (i + 1) & (ODW_N - 1)) { if (g_odw[i].a == a) return &g_odw[i]; if (!g_odw[i].a) return NULL; } return NULL; }
static occd_pg *odp_find(uint64_t pg) { for (uint32_t i = odh(pg), k = 0; k < ODW_N; k++, i = (i + 1) & (ODW_N - 1)) { if (g_odp[i].pg == pg + 1) return &g_odp[i]; if (!g_odp[i].pg) return NULL; } return NULL; }
static void odw_put(const occd_w *w) {
    uint32_t i = odh(w->a >> 4); while (g_odw[i].a && g_odw[i].a != w->a) i = (i + 1) & (ODW_N - 1);
    if (!g_odw[i].a) g_odw_n++; g_odw[i] = *w;
    const uint64_t pg = w->a >> 12; uint32_t j = odh(pg); while (g_odp[j].pg && g_odp[j].pg != pg + 1) j = (j + 1) & (ODW_N - 1);
    if (!g_odp[j].pg) { g_odp_n++; g_odp[j].pg = pg + 1; g_odp[j].f = 0; } if (w->f > g_odp[j].f) g_odp[j].f = w->f;
}
static void occd_rebuild(void) {
    const uint64_t done = vk_fence_done(); occd_w *old = malloc(sizeof *old * ODW_N); if (!old) return;
    memcpy(old, g_odw, sizeof *old * ODW_N); memset(g_odw, 0, sizeof *g_odw * ODW_N); memset(g_odp, 0, sizeof *g_odp * ODW_N); g_odw_n = g_odp_n = 0;
    for (uint32_t i = 0; i < ODW_N; i++) if (old[i].a && old[i].f > done) odw_put(&old[i]);
    free(old);
}
static void occd_word_add(uint64_t a, uint64_t ser) {
    extern uint64_t g_tex_frame;
    if (!a) return;
    if (!g_odw) { g_odw = calloc(ODW_N, sizeof *g_odw); g_odp = calloc(ODW_N, sizeof *g_odp); if (!g_odw || !g_odp) { free(g_odw); free(g_odp); g_odw = NULL; g_odp = NULL; return; } }
    if ((g_odw_n + 1) * 2 > ODW_N || (g_odp_n + 1) * 2 > ODW_N) occd_rebuild();
    if ((g_odw_n + 1) * 2 > ODW_N || (g_odp_n + 1) * 2 > ODW_N) { g_odw_full++; return; }
    const occd_w w = {a, g_fence + 1, g_tex_frame, ser, g_nsubs}; odw_put(&w);
}
static struct { uint64_t reads, hit, words, sub[3], pres[3], dmin, dsum; } g_odr[4];   // per reader kind: 0 uniform shadows, 1 copies, 2 UBO binds, 3 SSBO binds
static const char *const g_odr_names[4] = {"uniform shadows (vk_ubd_read)", "buffer copies (source)", "uniform buffer binds", "storage buffer binds"};
void vk_occd_read(uint64_t a, uint64_t n, int kind) {   // replay thread: a GPU-side read of [a, a + n) at this point of the stream
    extern uint64_t g_tex_frame;
    if (!g_odw || !n || kind < 0 || kind > 3) return;
    g_odr[kind].reads++;
    const uint64_t done = vk_fence_done(); const occd_w *best = NULL; uint64_t nw = 0;
    uint64_t pg = a >> 12, pg1 = (a + n - 1) >> 12; if (pg1 - pg > 4096) pg1 = pg + 4096;
    for (; pg <= pg1; pg++) { const occd_pg *p = odp_find(pg); if (!p || p->f <= done) continue;
        const uint64_t lo = (pg << 12 > a ? pg << 12 : a) & ~15ull, hi = (pg + 1) << 12 < a + n ? (pg + 1) << 12 : a + n;
        for (uint64_t w = lo; w < hi; w += 16) { const occd_w *e = odw_find(w); if (!e || e->f <= done) continue;
            if (g_occ2 >= 2 && kind && e->res) continue;   // occ-s2: a GPU reader recorded after the word's resolve point reads it after the GPU wrote it (only a CPU shadow, kind 0, is early)
            nw++; if (!best || e->ser > best->ser) best = e; } }
    if (!best) return;
    const uint64_t ds = g_nsubs - best->nsub, dp = g_tex_frame - best->pf, dd = vk_occ2_serial() - best->ser;
    g_odr[kind].hit++; g_odr[kind].words += nw; g_odr[kind].sub[ds > 2 ? 2 : ds]++; g_odr[kind].pres[dp > 2 ? 2 : dp]++;
    if (g_odr[kind].hit == 1 || dd < g_odr[kind].dmin) g_odr[kind].dmin = dd; g_odr[kind].dsum += dd;
}
// the overflow context: per submission (reset at its start)
static uint64_t g_od_sub_rep, g_od_sub_ser0, g_od_sub_pe0, g_od_ovf_sub, g_od_ovf_n, g_od_ovf_at[4];
static void occd_ovf(void) {   // vk_oq_draw, the first overflow of this submission
    if (g_od_ovf_sub == g_nsubs) return;
    g_od_ovf_sub = g_nsubs; g_od_ovf_at[0] = vk_occ2_serial() - g_od_sub_ser0; g_od_ovf_at[1] = g_vk_pends - g_od_sub_pe0; g_od_ovf_at[2] = g_od_sub_rep; g_od_ovf_at[3] = g_od_sub_rst;
}
// the replay thread, at a samples-passed report (sub_counter; o: its post op, NULL for the fixed path without queries); age: occ-s0's class of the value written
static void occ_sp_report(post_op *o, uint64_t addr, int age) {
    extern uint64_t g_tex_frame; const uint64_t ser = vk_occ2_serial();
    g_od_sub_rep++;
    if (g_occdiag) { occd_age(age); if (o) { o->od = 1; o->pf = (uint32_t)g_tex_frame; } occd_word_add(addr, ser); }
    if (g_occ2 && o && g_oq_counting) { const uint64_t lo = g_occ2_win_lo, hi = ser;   // (before the first reset nothing counts: nothing to compare)
        if (hi - lo > OCC2_RING - OCC2_CHUNK) { g_o2.longw++; occ2_red("a samples-passed window longer than the ring: not compared", lo, hi); }
        else if (!occ2_prep(lo, hi)) g_o2.lost++;
        else { occ2_mark(lo, hi); o->s1 = 1; o->s1lo = lo; o->s1hi = hi; } }
}
// the completion thread, at a samples-passed report's post op (old: the old path's epoch sum, lost: some of its spans lost)
static void occ_sp_done(const post_op *o, uint64_t old, int lost) {
    extern uint64_t g_tex_frame;
    if (o->od) { const uint32_t d = (uint32_t)__atomic_load_n(&g_tex_frame, __ATOMIC_RELAXED) - o->pf; __atomic_fetch_add(&g_od_lag[d > 3 ? 4 : d], 1, __ATOMIC_RELAXED); }
    if (!o->s1 || !g_occ2) return;
    if (lost) { __atomic_fetch_add(&g_o2c_lostold, 1, __ATOMIC_RELAXED); return; }
    uint64_t un = 0; const uint64_t v = occ2_sum(o->s1lo, o->s1hi, &un);
    __atomic_fetch_add(&g_o2c_cmp, 1, __ATOMIC_RELAXED); __atomic_fetch_add(&g_o2c_slots, o->s1hi - o->s1lo, __ATOMIC_RELAXED); __atomic_fetch_add(&g_o2c_unrec, un, __ATOMIC_RELAXED);
    if (!v && un) __atomic_fetch_add(&g_o2c_zero_unrec, 1, __ATOMIC_RELAXED);
    if (v != old) { const uint64_t d = v > old ? v - old : old - v; __atomic_fetch_add(&g_o2c_mis, 1, __ATOMIC_RELAXED);
        uint64_t m = __atomic_load_n(&g_o2c_maxd, __ATOMIC_RELAXED); while (d > m && !__atomic_compare_exchange_n(&g_o2c_maxd, &m, d, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
        static int nl; if (nl < 8) { nl++; VKLOG("[RED] occ-s1: report %#llx: window [%llu, %llu) sums %llu, the old span sum %llu", (unsigned long long)o->addr, (unsigned long long)o->s1lo, (unsigned long long)o->s1hi, (unsigned long long)v, (unsigned long long)old); } }
}
// ---------------- occ-s2 / occ-s3: the GPU resolve in stream order, render conditions on the GPU (design\OCCLUSION_REDESIGN.md steps 2 and 3) ----------------
// DM_VK_OCC2=s2 / s3 (device init; occ2_init): the per-draw query ring of step 1 (vk_occ2_begin / vk_occ2_end, every draw while a window is open), and no old-path value anywhere: no
// occkey / occ_key_make, occaddr, occland, post modes 7 / 8, DM_VK_OCC_FIXED for samples, 1 << 20 or "1 if lost", kind-4 posts, epochs or g_oq_counting on this path.
// - A samples-passed report (occ2_report, replay thread) is {its address, its timestamp (gpu_ticks at replay), its report sequence (stream order: the dry pass names the same
//   numbers, vk_occ2_rseq), its window [cut, serial now) of draw serials and the window's running base}. Only its timestamp half (bytes 8..15) is written at replay (rptfix's
//   queue, RQ_HALF); the counter word is written by the GPU only, as on the console.
// - A resolve point (occ2_resolve): the first pass end after reports (gfx_vk_exec.c vk_end_pass -> vk_occ2_pass_end), each part's end (vk_cmd_split), the submission's end
//   (mtl_sub_commit), a render condition whose report is still pending (forced, occ-s3), a part's join (resolves deferred while worker segments were out). All on the replay
//   thread, in stream order, outside a render pass. It records: an empty begin/end for each slot of the span no draw began (draws not recorded: a false condition, a draw
//   prepare_draw dropped), so every slot it copies is available; vkCmdCopyQueryPoolResults (64-bit, with availability, waited) of the span [min lo, max hi) into RES, a barrier,
//   ONE dispatch (occ2_resolve.comp) that sums each window into REC[record] = {count, timestamp} and PRED[record] = count != 0 (and the conditions' own predicates), a barrier
//   (CONDITIONAL_RENDERING_READ at the CONDITIONAL_RENDERING stage with s3), then vkCmdCopyBuffer of each report's 16 bytes from REC into the guest's word (vk_buf_at: the
//   imported pages, or a copied pool's device-local copy, whose write-back the GPU-write tracking does: vk_gpu_wrote per run of adjacent words, as a buffer copy's). Atomics:
//   none (NPARK: never on imported memory; none needed here: each entry writes its own words of a device-local buffer). The latest report at one address in a resolve owns
//   its word (R6). A window open for more than a quarter of the ring gets a running base (kind 1 entries into BASE), so a window longer than the ring still sums every draw.
// - Worker segments: none holds a samples-passed report or reset (WR_SPREP / WR_SPRESET, unchanged until step 4); their draws' queries are begun in their chunks, which come
//   before the replay thread's later chunks in queue order. A resolve whose span meets a segment handed out in the open part waits for the part's join (deferred), where the
//   workers' begun-query bits come in (a failed chunk's are dropped: its slots get empty queries), so a copy never waits on a query no command before it ends.
// - Landing bound (occ2_bound, at each present): the previous present's samples-passed reports landed (their submissions done) before the guest goes on: the console's GPU
//   is at most about one frame behind; ours may be several submissions behind. Waits counted (DM_VK_OCC_RDWATCH measures the guest's reads against it).
#include "occ2_resolve_comp.spv.h"
enum { O2_NREC = 1 << 16, O2_NBASE = 4096, O2_NCP = 4096, O2_NEAR = 1024, O2_NQ = 64, O2_NSTG = 1 << 15 };
// occ-s3fast: STG = per resolve point, the guest's 16-byte words in address order (one staging slot per word written), so a run of adjacent guest words is one copy region
enum { O2W_RES = 0, O2W_REC = OCC2_RING * 4, O2W_BASE = O2W_REC + O2_NREC * 4, O2W_PRED = O2W_BASE + O2_NBASE * 2, O2W_STG = O2W_PRED + O2_NREC + O2_NCP, O2W_N = O2W_STG + O2_NSTG * 4 };   // 32-bit words of D
_Static_assert(O2W_REC == 262144 && O2W_BASE == 524288 && O2W_PRED == 532480 && O2W_STG == 602112, "occ-s2 / occ-s3fast: the layout dm/windows/shaders/occ2_resolve.comp names");
static struct { VkBuffer buf; VkDeviceMemory mem; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; int zeroed; } g_o2r;
static PFN_vkCmdCopyQueryPoolResults g_o2_cqr; static PFN_vkCmdBeginConditionalRenderingEXT g_o3_cbeg; static PFN_vkCmdEndConditionalRenderingEXT g_o3_cend;
typedef struct { uint64_t addr, ticks, rseq, lo, hi; uint32_t base; } o2_rep;   // a pending report: its window [lo, hi) of draw serials (lo: the window's cut), its base slot (~0: none)
typedef struct { uint32_t t; uint64_t a, b; uint32_t c; } o2_opd;              // a condition operand: t 0 the constant a, 1 the record a, 2 the window [a, b) with base c (a report of this resolve)
typedef struct { uint32_t pw; o2_opd x, y; } o2_cond;                           // a condition's own predicate word (index in PRED) = x != y
static o2_rep *g_o2p; static uint32_t g_o2p_n, g_o2p_cap;   // replay thread: the reports since the last resolve
static o2_cond g_o2q[O2_NQ]; static uint32_t g_o2q_n;       // occ-s3: the conditions' own predicates since the last resolve
static struct { uint64_t rseq1, fence; uint32_t pi; uint8_t st; } g_o2s[O2_NREC];   // per record (report sequence mod O2_NREC): 1 + its sequence; st 1 pending (pi: its index in g_o2p), 2 resolved in submission fence
static struct { uint64_t lo, cut; uint32_t base; int counting; } g_o2w = {0, 0, ~0u, 0};   // the open window: its reset's serial, the serial its base sums up to, the base slot (~0: none)
static uint64_t g_o2_rseq; static uint32_t g_o2_nb, g_o3_ncp; static int g_o2_defer, g_o2_forcing;
static struct { uint64_t rep, nores, longw, res[O2R_N], defer, disp, slots, empty, words, copies, runs, dup, outside, adv, over;
                uint64_t kept, qruns, regions, ns, bars, must; } g_o2c;   // since the last line (occ-s3fast: opportunities passed over, query copy runs, guest copy regions, resolve CPU ns, barriers recorded, resolves a bound forced)
static struct { uint64_t pf, f; } g_o2_pff[16];   // per present (& 15): the last submission holding its samples-passed reports
static struct { uint64_t waits, ns, unreach, skip; } g_o2b;   // the landing bound (occ-s3fast: skip = would have waited, DM_VK_OCC_BOUND=0)
static struct { uint64_t forced, defer, outring, waits, wait_ns, blocks; } g_o3;   // occ-s3, since the last line
// occ-s3fast (diagnostic, log only, s2 / s3; NPARK 11-21: under s3 the light still trembles slightly, only when the frame time changes abruptly): per present its frame time
// and whether it is a spike (over 33 ms, or over 1.5x the previous frame's); a ring of the last O3F_RING frames (frame ms, the CPU-evaluated render conditions on timestamp /
// flag words: evaluated, drawn, skipped, outcome flips; landing-bound waits and ms), dumped 3 presents after a spike (at most one dump a second, 64 in all); per condition
// address its flips split by spike / calm frames (the 600-present line); the RDWATCH reads split the same way
enum { O3F_RING = 32, O3F_DUMP = 12, O3F_NA = 256 };   // (32: the flash detector names a frame ~10 presents after it)
typedef struct { uint64_t pf; uint32_t ms10, ev, drawn, skip, flips, bw, bskip, res, gpuc, nfa; uint64_t bns, fa[4]; uint8_t spike; } o3f_fr;   // res: resolve points, gpuc: GPU conditions, fa: flipped addresses
static struct {
    o3f_fr r[O3F_RING]; uint64_t tprev, dtprev, dump_at, last_dump, ndump, bw, bns, bskip, res, gpuc, flash;   // res / gpuc: the open frame's resolve points, GPU conditions; flash: 1 + a frame the flash detector named   // bw / bns: the open frame's landing-bound waits (occ2_bound); bskip: would-wait with DM_VK_OCC_BOUND=0
    struct { uint64_t pf; uint8_t spike; } spk[64];
    struct { uint64_t a; int last; uint8_t kind; uint32_t ecur, fcur; uint64_t espk, ecalm, fspk, fcalm; } at[O3F_NA]; uint32_t nat, full;
    uint64_t frames, spikes, ev[2], fl[2], bw2[2], bns2[2], sw[2], slow;   // the line's window, [0] calm frames, [1] spike frames; sw: frames with a landing wait; slow: frames over 33 ms
} g_o3f;
static void occ2_rw_sample(uint64_t addr);   // DM_VK_OCC_RDWATCH (below)
static int g_o2_redn;
static void o2_red(const char *what, uint64_t a, uint64_t b) { if (g_o2_redn++ < 16) VKLOG("[RED] occ-s2: %s (%#llx, %llu)", what, (unsigned long long)a, (unsigned long long)b); }
static int occ2_s2_init(int want) {   // occ2_init, the ring made: the level taken (2, 3), 0 = cannot
    int lvl = want;
    g_o2_cqr = (PFN_vkCmdCopyQueryPoolResults)vkGetDeviceProcAddr(V.dev, "vkCmdCopyQueryPoolResults");
    if (!g_o2_cqr) { VKLOG("[RED] occ-s2: the device has no vkCmdCopyQueryPoolResults"); return 0; }
    if (lvl >= 3) {   // VK_EXT_conditional_rendering enabled on the device (NXN1X's: Eden enables it when the driver has it; no feature struct is needed for its commands)
        int ext = 0;
#if defined(DM_NXN1X)
        if (g_imp_set) ext = imp_has("VK_EXT_conditional_rendering");
#endif
        if (ext) { g_o3_cbeg = (PFN_vkCmdBeginConditionalRenderingEXT)vkGetDeviceProcAddr(V.dev, "vkCmdBeginConditionalRenderingEXT"); g_o3_cend = (PFN_vkCmdEndConditionalRenderingEXT)vkGetDeviceProcAddr(V.dev, "vkCmdEndConditionalRenderingEXT"); }
        if (!g_o3_cbeg || !g_o3_cend) { g_o3_cbeg = NULL; g_o3_cend = NULL; lvl = 2;
            VKLOG("occ-s3: VK_EXT_conditional_rendering is %s on the device: DM_VK_OCC2=s3 runs as s2 (render conditions on the CPU, as before)", ext ? "enabled but its entry points are missing" : "not enabled"); }
    }
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)O2W_N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             (lvl >= 3 ? VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT : 0), VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &g_o2r.buf)) { g_o2r.buf = VK_NULL_HANDLE; VKLOG("[RED] occ-s2: the resolve buffer could not be made"); return 0; }
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_o2r.buf, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};   // device-local
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_o2r.mem) || vkBindBufferMemory(V.dev, g_o2r.buf, g_o2r.mem, 0)) { VKLOG("[RED] occ-s2: no device-local memory for the resolve buffer"); return 0; }
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, b};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_o2r.dsl)) { VKLOG("[RED] occ-s2: descriptor set layout failed"); return 0; }
    vk_dsl_register(g_o2r.dsl, b, 2);
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_o2r.dsl, 0, NULL};   // no push constants (they would touch the draws': as ixq)
    if (vkCreatePipelineLayout(V.dev, &pi, NULL, &g_o2r.pl)) { VKLOG("[RED] occ-s2: pipeline layout failed"); return 0; }
    VkShaderModule m; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof occ2_resolve_comp_spv, occ2_resolve_comp_spv};
    if (vkCreateShaderModule(V.dev, &mi, NULL, &m)) { VKLOG("[RED] occ-s2: shader module failed"); return 0; }
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", NULL}, g_o2r.pl, VK_NULL_HANDLE, -1};
    const VkResult r = vkCreateComputePipelines(V.dev, V.pcache, 1, &ci, NULL, &g_o2r.pipe); vkDestroyShaderModule(V.dev, m, NULL);
    if (r) { g_o2r.pipe = VK_NULL_HANDLE; VKLOG("[RED] occ-s2: compute pipeline failed (%d)", r); return 0; }
    VKLOG("occ-s%d: DM_VK_OCC2=s%d: samples-passed reports resolved on the GPU in stream order (per-draw queries on the 64K ring, chunks reset %s, %s; one copy of the query results, one compute "
          "dispatch and the guest words copied from a device-local ring per resolve point; no atomics), the counter word written by the GPU only, the old substitutes off on this path%s; "
          "landing bound: the previous present's reports land before each present (log every 600 presents)", lvl, want,
          g_occ2_hreset ? "from the host" : "by vkCmdResetQueryPool in the part's upload command buffer", g_oq_flags ? "precise" : "not precise (no occlusionQueryPrecise)",
          lvl >= 3 ? "; render conditions on such reports evaluated by the GPU (VK_EXT_conditional_rendering)" : "");
    return lvl;
}
static int occ2_counting(void) { return g_o2w.counting; }
uint64_t vk_occ2_rseq(void) { return g_o2_rseq; }   // the replay thread's next samples-passed report's sequence (the dry pass names the same: gfx_vk_exec.c g_sd.orseq)
static uint64_t g_o2_rseq_off;
void vk_occ2_rseq_chk(uint64_t want) {   // gfx_vk_exec.c seg_report: the dry pass named this report want (its predicate index follows it)
    if (g_occ2 < 2 || g_o2_rseq == want) return;
    if (!g_o2_rseq_off++) VKLOG("[RED] occ-s2: the replay thread's report sequence %llu is not the dry pass's %llu: the dry pass's taken", (unsigned long long)g_o2_rseq, (unsigned long long)want);
    g_o2_rseq = want;
}
static void occ2_reset(void) {   // vk_oq_reset, SAMPLES_PASSED (replay thread): a new window from this serial
    vk_occ2_end(); const uint64_t s = vk_occ2_serial();
    g_o2w.lo = g_o2w.cut = s; g_o2w.base = ~0u; g_o2w.counting = 1;
}
static void occ2_report(uint64_t addr) {   // sub_counter, SAMPLES_PASSED (replay thread, in stream order)
    extern uint64_t g_tex_frame;
    vk_occ2_end();
    const uint64_t ser = vk_occ2_serial(), rs = g_o2_rseq++, ticks = gpu_ticks();
    (void)report_now(addr, 0, ticks, NVN_COUNTER_TYPE_SAMPLES_PASSED | RQ_HALF, 1);   // the timestamp half at replay (queued), the counter word by the GPU
    o2_rep r = {addr, ticks, rs, ser, ser, ~0u};
    if (g_o2w.counting) { r.lo = g_o2w.cut; r.base = g_o2w.base; } else g_o2c.nores++;   // (before any reset: no draw counted, 0)
    if (g_o2p_n == g_o2p_cap) { const uint32_t c = g_o2p_cap ? 2 * g_o2p_cap : 1024; void *p = realloc(g_o2p, (size_t)c * sizeof *g_o2p); if (!p) die("[vk] occ-s2: no memory for the pending reports"); g_o2p = p; g_o2p_cap = c; }
    if (g_o2p_n >= O2_NREC / 2) { g_o2c.over++; o2_red("more than 32K reports since the last resolve point (one pass): their records would collide", addr, g_o2p_n); }
    const unsigned i = (unsigned)(rs % O2_NREC); g_o2s[i].rseq1 = rs + 1; g_o2s[i].st = 1; g_o2s[i].pi = g_o2p_n; g_o2s[i].fence = g_fence + 1;
    g_o2p[g_o2p_n++] = r; g_o2c.rep++; g_od_sub_rep++;
    occd_word_add(addr, ser);   // GPU-side readers of the word before it lands (uniform shadows, copies, binds: vk_occd_read)
    { const unsigned k = (unsigned)(g_tex_frame & 15); if (g_o2_pff[k].pf != g_tex_frame) { g_o2_pff[k].pf = g_tex_frame; g_o2_pff[k].f = 0; } g_o2_pff[k].f = g_fence + 1; }   // the landing bound
    occ2_rw_sample(addr);
}
static void occ2_view(vk_par_view *v) {   // par_view_take (the hand-out): the bits of the queries this segment's worker begins, its serials as vk_occ2_handout named them
    const uint64_t n = g_o2_hon; const uint32_t w = (uint32_t)((n + 63) / 64);
    v->o2lo = g_o2_holo; v->o2n = 0;
    if (!n) return;
    if (w > v->o2cap) { void *p = realloc(v->o2b, sizeof *v->o2b * (size_t)w); if (!p) { o2_red("no memory for a worker segment's query bits: its slots get empty queries", g_o2_holo, n); return; } v->o2b = p; v->o2cap = w; }
    memset(v->o2b, 0, sizeof *v->o2b * (size_t)w); v->o2n = (uint32_t)n;
}
void vk_occ2_job_join(vk_par_view *v, int ok) {   // gfx_vk_exec.c vk_seg_part_join, per job: its begun queries count once its chunk is submitted (ok); a failed one's slots stay unbegun
    if (g_occ2 < 2 || !v || !v->o2b || !v->o2n) return;
    if (ok) for (uint64_t k = 0; k < v->o2n; k++) if (v->o2b[k >> 6] >> (k & 63) & 1) { const uint32_t s = (uint32_t)((v->o2lo + k) % OCC2_RING); g_o2_bits[s >> 6] |= 1ull << (s & 63); }
    v->o2n = 0;
}
void vk_occ2_join_end(void) {   // gfx_vk_exec.c vk_seg_part_join, the part joined (no pass open): the resolves deferred over its segments
    if (g_occ2 < 2) return;
    g_o2_wklo = g_o2_wkhi = 0;
    if (g_o2_defer) { g_o2_defer = 0; occ2_resolve(O2R_JOIN); }
}
static int o2_cmp_addr(const void *x, const void *y) { const o2_rep *a = *(o2_rep *const *)x, *b = *(o2_rep *const *)y; return a->addr < b->addr ? -1 : a->addr > b->addr ? 1 : a->rseq < b->rseq ? -1 : a->rseq > b->rseq; }
typedef struct { VkBuffer b; VkBufferCopy r; } o2_cp;
static int o2_cmp_buf(const void *x, const void *y) { const o2_cp *a = x, *b = y; const uintptr_t p = (uintptr_t)a->b, q = (uintptr_t)b->b; return p < q ? -1 : p > q ? 1 : a->r.dstOffset < b->r.dstOffset ? -1 : a->r.dstOffset > b->r.dstOffset; }
static void o2_put_opd(uint32_t *w, const o2_opd *o) { w[0] = o->t; if (o->t == 0) { w[1] = (uint32_t)o->a; w[2] = (uint32_t)(o->a >> 32); } else if (o->t == 1) w[1] = (uint32_t)o->a;
    else { w[1] = (uint32_t)(o->a % OCC2_RING); w[2] = (uint32_t)(o->b - o->a); w[3] = o->c; } }
// replay thread, outside a render pass, a submission open: the pending reports (and conditions' predicates, and a running base) resolved on the GPU here, in stream order.
// occ-s3fast: called only where something reads them (occ2_opp: before the pass holding the next GPU condition or GPU reader of a pending report, as the plan of
// gfx_vk_exec.c o3p_* names it; a copy reading one; the submission's end; a bound), so one resolve point batches many passes' reports. The mechanism is Eden's
// (source\src\video_core\renderer_vulkan\vk_query_cache.cpp SamplesStreamer::PresyncWrites / QueryCacheRuntime::SyncValues / Barriers, query_cache.h NotifyWFI): one
// vkCmdCopyQueryPoolResults per contiguous run of slots (ApplyBanksWideOp's min / max per bank), one compute dispatch for the whole batch, then the guest's words copied
// with one vkCmdCopyBuffer per destination buffer (SyncValues' copies_setup), here from a staging area in guest address order so a run of adjacent words is one region; a
// full barrier before the guest copies and one after them (Eden's Barriers(true) / Barriers(false))
typedef struct { uint64_t lo, hi; } o2_run;
static int o2_cmp_run(const void *x, const void *y) { const o2_run *a = x, *b = y; return a->lo < b->lo ? -1 : a->lo > b->lo; }
enum { O2_GAP = 256 };   // occ-s3fast: runs of slots closer than this are one query copy (the slots between hold begun queries of draws no window counts, or get empty ones)
static void occ2_resolve(int why) {
    if (g_occ2 < 2 || g_cur < 0 || !g_o2r.pipe) return;
    const uint64_t now = vk_occ2_serial();
    const int adv = g_o2w.counting && now - g_o2w.cut > OCC2_RING / 4;   // the open window's base moves on (a window longer than the ring still sums every draw)
    if (!g_o2p_n && !g_o2q_n && !adv) return;
    const uint64_t t0 = now_ns();
    uint64_t lo = UINT64_MAX, hi = 0;
#define O2_SPAN(l, h) do { if ((h) > (l)) { if ((l) < lo) lo = (l); if ((h) > hi) hi = (h); } } while (0)
    for (uint32_t i = 0; i < g_o2p_n; i++) O2_SPAN(g_o2p[i].lo, g_o2p[i].hi);
    for (uint32_t i = 0; i < g_o2q_n; i++) { if (g_o2q[i].x.t == 2) O2_SPAN(g_o2q[i].x.a, g_o2q[i].x.b); if (g_o2q[i].y.t == 2) O2_SPAN(g_o2q[i].y.a, g_o2q[i].y.b); }
    if (adv) O2_SPAN(g_o2w.cut, now);
#undef O2_SPAN
    if (hi > lo && now - lo > OCC2_RING - OCC2_CHUNK) {   // (a window that grew past the ring between two resolve points: its oldest draws' queries are gone or going; measured from now,
        // so no slot of the span is in a chunk reset for a later lap: every slot copied is begun in its lap or gets an empty query, and the waited copy cannot wait forever)
        const uint64_t nlo = now - (OCC2_RING - OCC2_CHUNK); o2_red("a samples-passed window longer than the ring between two resolve points: its oldest draws not counted", nlo, now - lo);
        for (uint32_t i = 0; i < g_o2p_n; i++) if (g_o2p[i].hi > g_o2p[i].lo && g_o2p[i].lo < nlo) { g_o2p[i].lo = nlo < g_o2p[i].hi ? nlo : g_o2p[i].hi; g_o2c.longw++; }
        for (uint32_t i = 0; i < g_o2q_n; i++) { o2_opd *o[2] = {&g_o2q[i].x, &g_o2q[i].y}; for (int k = 0; k < 2; k++) if (o[k]->t == 2 && o[k]->a < nlo) o[k]->a = nlo < o[k]->b ? nlo : o[k]->b; }
        lo = nlo; }
    // occ-s3fast: the slots the windows cover as sorted, merged runs (one waited copy each) instead of one span [min lo, max hi): a batch of a frame's reports spans the
    // frame's draws, and a gap over a worker segment of the open part is never merged in (its slots are begun at the join: a copy over them would have to wait for it)
    static o2_run *rn; static uint32_t rn_cap; uint32_t nr = 0;
    { const uint32_t rmax = g_o2p_n + 2 * g_o2q_n + 1;
      if (rmax > rn_cap) { void *p = realloc(rn, sizeof *rn * rmax); if (!p) die("[vk] occ-s3fast: no memory for a resolve's runs"); rn = p; rn_cap = rmax; } }
#define O2_RUN(l, h) do { if ((h) > (l)) { rn[nr].lo = (l); rn[nr].hi = (h); nr++; } } while (0)
    for (uint32_t i = 0; i < g_o2p_n; i++) O2_RUN(g_o2p[i].lo, g_o2p[i].hi);
    for (uint32_t i = 0; i < g_o2q_n; i++) { if (g_o2q[i].x.t == 2) O2_RUN(g_o2q[i].x.a, g_o2q[i].x.b); if (g_o2q[i].y.t == 2) O2_RUN(g_o2q[i].y.a, g_o2q[i].y.b); }
    if (adv) O2_RUN(g_o2w.cut < lo ? lo : g_o2w.cut, now);
#undef O2_RUN
    if (nr > 1) qsort(rn, nr, sizeof *rn, o2_cmp_run);
    const int wk = g_o2_wkhi > g_o2_wklo;
    { uint32_t m = 0;
      for (uint32_t i = 0; i < nr; i++) {
          if (m && rn[i].lo <= rn[m - 1].hi + O2_GAP && !(wk && rn[i].lo > rn[m - 1].hi && rn[m - 1].hi < g_o2_wkhi && g_o2_wklo < rn[i].lo)) { if (rn[i].hi > rn[m - 1].hi) rn[m - 1].hi = rn[i].hi; continue; }
          rn[m++] = rn[i]; }
      nr = m; }
    for (uint32_t i = 0; wk && i < nr; i++) if (rn[i].lo < g_o2_wkhi && g_o2_wklo < rn[i].hi) { g_o2_defer = 1; g_o2c.defer++; return; }   // worker segments out over a run: at the part's join
    VkCommandBuffer cb = vk_cmd();
    vk_occ2_end();   // (none is open at a pass end; a draw's query never spans one)
    { int unsafe = 0;   // every slot of a run begun in its current lap: an empty query for each one no draw began (so the waited copy below never waits on one no command ends)
      for (uint32_t r = 0; r < nr; r++) {
          (void)occ2_prep(rn[r].lo, rn[r].hi); occ2_mark(rn[r].lo, rn[r].hi);
          for (uint64_t s = rn[r].lo; s < rn[r].hi; s++) { const uint32_t k = (uint32_t)(s % OCC2_RING); if (g_o2_bits[k >> 6] >> (k & 63) & 1) continue;
              if (g_occ2_ck[k / OCC2_CHUNK].lap != s / OCC2_RING + 1) { unsafe = 1; continue; }   // (a chunk occ2_prep could not reset ([RED] there): a slot of it may never become available)
              vkCmdBeginQuery(cb, g_occ2_q, k, 0); vkCmdEndQuery(cb, g_occ2_q, k); g_o2_bits[k >> 6] |= 1ull << (k & 63); g_o2c.empty++; } }
      if (unsafe) {   // no waited copy over a slot that may never become available: this resolve's windows counted empty (their bases only)
          o2_red("a resolve's span holds slots of a chunk not reset for their lap: its windows counted empty", lo, hi - lo);
          for (uint32_t i = 0; i < g_o2p_n; i++) g_o2p[i].lo = g_o2p[i].hi;
          for (uint32_t i = 0; i < g_o2q_n; i++) { if (g_o2q[i].x.t == 2) g_o2q[i].x.a = g_o2q[i].x.b; if (g_o2q[i].y.t == 2) g_o2q[i].y.a = g_o2q[i].y.b; }
          lo = hi = now; nr = 0; } }
    // the guest's words: the latest report at each address (R6), in address order; occ-s3fast: each one a staging slot in that order (the dispatch writes its 16 bytes there
    // too), so adjacent words are adjacent in the source as well and a run of them is one copy region
    static o2_rep **srt; static uint32_t srt_cap; static o2_cp *cps; static VkBufferCopy *rgs; static struct { uint64_t lo, hi; } *runs; static uint32_t *stg;
    if (g_o2p_n > srt_cap) { void *a = realloc(srt, sizeof *srt * g_o2p_n), *b2 = a ? realloc(cps, sizeof *cps * g_o2p_n) : NULL; if (a) srt = a; if (b2) cps = b2;
        void *c2 = b2 ? realloc(rgs, sizeof *rgs * g_o2p_n) : NULL; if (c2) rgs = c2; void *d2 = c2 ? realloc(runs, sizeof *runs * g_o2p_n) : NULL; if (d2) runs = d2;
        void *e2 = d2 ? realloc(stg, sizeof *stg * g_o2p_n) : NULL; if (e2) stg = e2;
        if (!a || !b2 || !c2 || !d2 || !e2) die("[vk] occ-s2: no memory for a resolve's copies"); srt_cap = g_o2p_n; }
    uint32_t ncp = 0, nrun = 0, nstg = 0;
    for (uint32_t i = 0; i < g_o2p_n; i++) { srt[i] = &g_o2p[i]; stg[i] = ~0u; }
    if (g_o2p_n > 1) qsort(srt, g_o2p_n, sizeof *srt, o2_cmp_addr);
    for (uint32_t i = 0; i < g_o2p_n; i++) {
        const o2_rep *r = srt[i];
        if (i + 1 < g_o2p_n && srt[i + 1]->addr == r->addr) { g_o2c.dup++; continue; }   // a later report at the same address owns the word
        VkBuffer db; VkDeviceSize dof;
        if (!r->addr || !vk_buf_at(r->addr, 16, &db, &dof)) { g_o2c.outside++; o2_red("a samples-passed report outside the memory pools: not written", r->addr, r->rseq); continue; }
        VkDeviceSize src;
        if (nstg < O2_NSTG) { stg[r - g_o2p] = nstg; src = (VkDeviceSize)(O2W_STG + 4 * nstg) * 4; nstg++; }
        else src = (VkDeviceSize)(O2W_REC + 4 * (r->rseq % O2_NREC)) * 4;   // (more than 32K words at one point: the rest from REC, a region each; RED "over 32K" counts it)
        cps[ncp++] = (o2_cp){db, {src, dof, 16}};
        if (nrun && runs[nrun - 1].hi == r->addr) runs[nrun - 1].hi = r->addr + 16; else { runs[nrun].lo = r->addr; runs[nrun].hi = r->addr + 16; nrun++; }
        if (g_odw) { occd_w *w = odw_find(r->addr); if (w) w->res = 1; }   // (the GPU-side readers' check: recorded after this point, they read the GPU's word)
    }
    // the entries (16 words each) in this submission's host-visible scratch memory
    const uint32_t ne = g_o2p_n + g_o2q_n + (adv ? 1u : 0u);
    const VkDeviceSize al = V.props.limits.minStorageBufferOffsetAlignment ? V.props.limits.minStorageBufferOffsetAlignment : 16, pbytes = (VkDeviceSize)(4 + 16 * ne) * 4;
    VkBuffer pb; VkDeviceSize po; uint32_t *P = vk_scratch(pbytes, al, &pb, &po); memset(P, 0, (size_t)pbytes); P[0] = ne;
    uint32_t k = 0;
    for (uint32_t i = 0; i < g_o2p_n; i++, k++) { const o2_rep *r = &g_o2p[i]; uint32_t *e = P + 4 + 16 * k;
        e[0] = 0; e[1] = (uint32_t)(r->lo % OCC2_RING); e[2] = (uint32_t)(r->hi - r->lo); e[3] = r->base; e[4] = (uint32_t)(r->rseq % O2_NREC); e[5] = (uint32_t)r->ticks; e[6] = (uint32_t)(r->ticks >> 32);
        e[7] = stg[i]; }   // (occ-s3fast: its staging slot, ~0 none)
    for (uint32_t i = 0; i < g_o2q_n; i++, k++) { uint32_t *e = P + 4 + 16 * k; e[0] = 2; e[1] = g_o2q[i].pw; o2_put_opd(e + 2, &g_o2q[i].x); o2_put_opd(e + 8, &g_o2q[i].y); }
    if (adv) { uint32_t *e = P + 4 + 16 * k; const uint32_t bo = g_o2_nb++ % O2_NBASE; const uint64_t c0 = g_o2w.cut < lo ? lo : g_o2w.cut;   // (lo: the span, clamped above)
        if (c0 != g_o2w.cut) g_o2c.longw++;
        e[0] = 1; e[1] = (uint32_t)(c0 % OCC2_RING); e[2] = (uint32_t)(now - c0); e[3] = g_o2w.base; e[4] = bo; g_o2w.cut = now; g_o2w.base = bo; g_o2c.adv++; }
    vk_bar_flush(cb);   // (the held barriers before this point's first command)
    vk_pass_ts(0, 0, 0, VK_TS_OCC2, 0);   // occ-s3fast: its GPU time with DM_VK_PASS_TIMES=1 (r138's group, the occ line's cost)
    if (!g_o2r.zeroed) { vkCmdFillBuffer(cb, g_o2r.buf, 0, VK_WHOLE_SIZE, 0); g_o2r.zeroed = 1;   // (once: nothing reads a word before it is written, but the first lap starts known)
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); g_o2c.bars++; }
    // occ-s3fast: no full barrier before the query copies (the old A): an earlier resolve's reads of RES / REC / STG are ordered by its closing barrier D, the queries' ends by
    // the copy's WAIT bit (Eden records none there either), the guest words' earlier writers by C below
    if (nr) { S63_WORK_0();   // (vkCmdCopyQueryPoolResults has no s63 wrapper: it counts as work)
        const VkQueryResultFlags fl = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_WAIT_BIT;
        for (uint32_t r = 0; r < nr; r++) {
            const uint32_t f0 = (uint32_t)(rn[r].lo % OCC2_RING); const uint64_t n = rn[r].hi - rn[r].lo; const uint32_t n0 = (uint32_t)(n < (uint64_t)(OCC2_RING - f0) ? n : (uint64_t)(OCC2_RING - f0));
            g_o2_cqr(cb, g_occ2_q, f0, n0, g_o2r.buf, (VkDeviceSize)(O2W_RES + 4 * f0) * 4, 16, fl); g_o2c.qruns++;
            if (n > n0) { g_o2_cqr(cb, g_occ2_q, 0, (uint32_t)(n - n0), g_o2r.buf, (VkDeviceSize)O2W_RES * 4, 16, fl); g_o2c.qruns++; }
            g_o2c.slots += n; } }
    { VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};   // B: the results before the sums
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL); g_o2c.bars++; }
    VkDescriptorSet set = vk_alloc_set(g_o2r.dsl);
    VkDescriptorBufferInfo bi[2] = {{pb, po, pbytes}, {g_o2r.buf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[0], NULL},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[1], NULL}};
    vkUpdateDescriptorSets(V.dev, 2, wr, 0, NULL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_o2r.pipe);   // (dispatch binds its own pipeline and set every time)
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_o2r.pl, 0, 1, &set, 0, NULL);
    vkCmdDispatch(cb, (ne + 63) / 64, 1, 1); g_o2c.disp++;
    vk_img_barrier_all(cb); g_o2c.bars++;   // C (occ-s3fast: full, Eden's Barriers(true)): REC / STG before the guest copies, PRED before the gated passes (CONDITIONAL_RENDERING_READ with s3, vk_dev_access), the words' earlier writers before the copies
    uint32_t nreg = 0;
    if (ncp > 1) qsort(cps, ncp, sizeof *cps, o2_cmp_buf);
    for (uint32_t i = 0; i < ncp;) { uint32_t j = i, m = 0;   // one vkCmdCopyBuffer per destination buffer; adjacent words (adjacent in STG too) one region
        while (j < ncp && cps[j].b == cps[i].b) {
            if (m && rgs[m - 1].dstOffset + rgs[m - 1].size == cps[j].r.dstOffset && rgs[m - 1].srcOffset + rgs[m - 1].size == cps[j].r.srcOffset) rgs[m - 1].size += cps[j].r.size;
            else rgs[m++] = cps[j].r;
            j++; }
        vkCmdCopyBuffer(cb, g_o2r.buf, cps[i].b, m, rgs); g_o2c.copies++; nreg += m; i = j; }
    if (ncp) { vk_img_barrier_all(cb); g_o2c.bars++; }   // D: the guest's words before every later reader (as a buffer copy's closing barrier; Eden's Barriers(false))
    vk_pass_ts(1, 0, 0, VK_TS_OCC2, 0);
    for (uint32_t i = 0; i < nrun; i++) vk_gpu_wrote(runs[i].lo, runs[i].hi - runs[i].lo);   // the GPU-write tracking: in flight for CPU readers, written back for copied pools
    g_o2c.words += ncp; g_o2c.runs += nrun; g_o2c.regions += nreg; g_o2c.res[why < O2R_N ? why : 0]++;
    for (uint32_t i = 0; i < g_o2p_n; i++) { const unsigned s = (unsigned)(g_o2p[i].rseq % O2_NREC); if (g_o2s[s].rseq1 == g_o2p[i].rseq + 1) { g_o2s[s].st = 2; g_o2s[s].fence = g_fence + 1; } }
    g_o2p_n = 0; g_o2q_n = 0;
    g_o2c.ns += now_ns() - t0; g_o3f.res++;   // (occ-s3fast diagnostic: the frame's resolve points)
}
// occ-s3fast: the pending state's bounds, whatever the plan says: a condition's own predicate waiting (kind 2), a running base due, 8K reports pending, the oldest pending
// window reaching half the ring (so a batch never comes near the ring's lap, where its oldest slots' chunk would be reset for a later lap)
static int occ2_must(void) {
    const uint64_t now = vk_occ2_serial();
    if (g_o2q_n) return 1;
    if (g_o2w.counting && now - g_o2w.cut > OCC2_RING / 4) return 1;
    if (g_o2p_n >= O2_NREC / 8) return 1;
    if (g_o2p_n && now - g_o2p[0].lo > OCC2_RING / 2) return 1;   // (g_o2p[0] the oldest: windows start at the cut, which only moves on)
    return 0;
}
// occ-s3fast: a resolve opportunity (a pass of the replay thread ended, a record boundary with no pass open, a part's end): taken when the plan (gfx_vk_exec.c vk_o3p_want:
// a GPU condition or GPU reader of a pending report comes before the next opportunity it knows of) or a bound says so; else the reports stay pending (the submission's end
// resolves the rest). No plan for the list: every opportunity is taken (the s2 / s3 rule)
static void occ2_opp(int why) {
    if (g_occ2 < 2 || tl_wv || (!g_o2p_n && !g_o2q_n)) return;
    if (g_o2_forcing) { occ2_resolve(O2R_FORCED); return; }
    const int must = occ2_must();
    if (must || vk_o3p_want(vk_occ2_serial(), g_o2p_n ? g_o2p[0].rseq : g_o2_rseq, g_o2_rseq)) { if (must) g_o2c.must++; occ2_resolve(why); }
    else g_o2c.kept++;
}
void vk_occ2_pass_end(void) { occ2_opp(O2R_PASS); }   // gfx_vk_exec.c vk_end_pass: a pass of the replay thread ended
void vk_occ2_opp(void) { occ2_opp(O2R_PASS); }        // gfx_vk_exec.c: a record boundary with no pass open that the plan counts as an opportunity (SET_RT, copies, dispatches)
static int occ2_unres_in(uint64_t a, uint64_t n) {   // [a, a + n) holds a samples-passed report word not resolved yet (replay thread)
    if (g_occ2 < 2 || !g_o2p_n || !n) return 0;
    if (!g_odw || n / 16 > g_o2p_n) { for (uint32_t i = 0; i < g_o2p_n; i++) if (g_o2p[i].addr < a + n && a < g_o2p[i].addr + 16) return 1; return 0; }   // (a long range: the pending list itself)
    const uint64_t done = vk_fence_done(); uint64_t pg = a >> 12, pg1 = (a + n - 1) >> 12;
    for (; pg <= pg1; pg++) { const occd_pg *p = odp_find(pg); if (!p || p->f <= done) continue;
        const uint64_t lo = (pg << 12 > a ? pg << 12 : a) & ~15ull, hi = (pg + 1) << 12 < a + n ? (pg + 1) << 12 : a + n;
        for (uint64_t w = lo; w < hi; w += 16) { const occd_w *e = odw_find(w); if (e && e->f > done && !e->res) return 1; } }
    return 0;
}
int vk_occ2_unres_in(uint64_t a, uint64_t n) { return occ2_unres_in(a, n); }   // gfx_vk_exec.c the plan: a reader of a report pending at the list's start
static void occ2_dropped(void) {   // mtl_sub_commit, the submission refused (its unsent parts never run: their query begins, chunk resets and resolves): those chunks are reset again
    const uint64_t cur = g_fence + 1; uint32_t n = 0;   // (parts sent early did run: their counts are lost with the re-reset, as the refused part's)
    for (unsigned i = 0; i < OCC2_NCK; i++) if (g_occ2_ck[i].fence == cur) { g_occ2_ck[i].lap = 0; memset(&g_o2_bits[i * (OCC2_CHUNK / 64)], 0, OCC2_CHUNK / 8); n++; }
    for (unsigned i = 0; i < O2_NREC; i++) if (g_o2s[i].st && g_o2s[i].fence == cur) g_o2s[i].st = 0;   // its reports never land: no condition takes their predicates
    g_o2p_n = 0; g_o2q_n = 0;   // (nothing pending: the submission's resolves are gone with it)
    if (n) o2_red("a refused submission: its query ring chunks are reset again before their next use", cur, n);
}
static void occ2_bound(void) {   // vk_pass_frame (a present): the previous present's samples-passed reports landed before the guest goes on
    extern uint64_t g_tex_frame; const uint64_t pf = g_tex_frame;
    if (!pf) return;
    const unsigned k = (unsigned)((pf - 1) & 15); if (g_o2_pff[k].pf != pf - 1 || !g_o2_pff[k].f) return;
    const uint64_t f = g_o2_pff[k].f; if (f <= vk_fence_done()) return;
    if (f > mtl_last_fence()) { g_o2b.unreach++; return; }   // (still the open submission: no wait without its commit)
    // occ-s3fast: DM_VK_OCC_BOUND=0 (live, DM_CTL env; default 1 = as before) leaves the wait out, for the A/B NPARK asked for (11-21: s2 / s3 doubled the frames over 33 ms,
    // the bound is a suspect; step 0 measured no GPU-side reads before landing, the guest's CPU reads are what DM_VK_OCC_RDWATCH=1 measures). Counted "would wait" when off
    { extern volatile long dm_env_gen; static long gen = -1; static int on = 1;
      if (gen != dm_env_gen) { gen = dm_env_gen; const char *e = getenv("DM_VK_OCC_BOUND"); const int n = !(e && *e == '0');
          if (n != on) VKLOG("occ-s3fast: landing bound %s (DM_VK_OCC_BOUND, live)", n ? "on: the previous present's reports land before each present" : "off (A/B): not waited for, counted as would-wait"); on = n; }
      if (!on) { g_o2b.skip++; g_o3f.bskip++; return; } }
    const uint64_t t0 = now_ns(); (void)mtl_wait(f, 2000000000ull); g_o2b.waits++; g_o2b.ns += now_ns() - t0;
    g_o3f.bw++; g_o3f.bns += now_ns() - t0;   // occ-s3fast (diagnostic): this frame's landing-bound waits (o3f_present folds them)
}
// ---- occ-s3: render conditions on the GPU (gfx_vk_exec.c rcg_*: the classification, the gated draws and clears)
static void o3_force(void) {   // the condition's report (or its own predicate) resolved before its draws: the open pass ends there (a GPU-side split, no CPU wait)
    const uint64_t r0 = g_o2c.res[O2R_FORCED];
    g_o2_forcing = 1; vk_end_pass(); if (g_o2p_n || g_o2q_n) occ2_resolve(O2R_FORCED); g_o2_forcing = 0;
    if (g_o2c.res[O2R_FORCED] != r0) g_o3.forced++;
}
static int o3_in(uint64_t r) { return r < g_o2_rseq && g_o2_rseq - r <= O2_NREC - O2_NEAR && g_o2s[r % O2_NREC].rseq1 == r + 1 && g_o2s[r % O2_NREC].st; }   // its record and predicate are still the ring's (st 0: a refused submission's)
static void o3_opd(o2_opd *o, uint64_t r) { const unsigned i = (unsigned)(r % O2_NREC);
    if (g_o2s[i].st == 1) { const o2_rep *p = &g_o2p[g_o2s[i].pi]; *o = (o2_opd){2, p->lo, p->hi, p->base}; } else *o = (o2_opd){1, i, 0, 0}; }
// replay thread: a condition the GPU evaluates (q: x the report rx; y the report ry when yrep, else the constant yc): 1 with *off the byte offset of its predicate word in the
// resolve buffer (x != y), 0 = not on the GPU (its report left the ring, or a resolve deferred to a part's join): the caller evaluates it on the CPU
int vk_occ3_pred(const vk_occ3_q *q, uint32_t *off) {
    if (g_occ2 < 3 || !g_o3_cbeg) return 0;
    if (!o3_in(q->rx) || (q->yrep && !o3_in(q->ry))) { g_o3.outring++; return 0; }
    const unsigned ix = (unsigned)(q->rx % O2_NREC);
    if (!q->yrep && !q->yc) {   // OoT's: y is 0 in memory: the report's own predicate (count != 0)
        if (g_o2s[ix].st == 1) { o3_force(); if (g_o2s[ix].st == 1) { g_o3.defer++; return 0; } }
        *off = (uint32_t)(O2W_PRED + ix) * 4; g_o3f.gpuc++; return 1; }
    if (g_o2q_n == O2_NQ) o3_force();
    if (g_o2q_n == O2_NQ) { g_o3.defer++; return 0; }
    o2_cond c; c.pw = O2_NREC + g_o3_ncp++ % O2_NCP; o3_opd(&c.x, q->rx);
    if (q->yrep) o3_opd(&c.y, q->ry); else c.y = (o2_opd){0, q->yc, 0, 0};
    g_o2q[g_o2q_n++] = c;
    o3_force();
    if (g_o2q_n) { g_o3.defer++; return 0; }   // (deferred: it lands at the join, harmlessly; this condition goes to the CPU)
    *off = (uint32_t)(O2W_PRED + c.pw) * 4; g_o3f.gpuc++; return 1;
}
int vk_occ3_wait(uint64_t r) {   // the CPU's last resort for a condition on report r the GPU does not take: wait for the submission it was resolved in (1: its word has landed)
    if (g_occ2 < 2) return 0;
    const unsigned i = (unsigned)(r % O2_NREC); if (g_o2s[i].rseq1 != r + 1 || g_o2s[i].st != 2) return 0;
    const uint64_t f = g_o2s[i].fence; if (f <= vk_fence_done()) return 1;
    if (f > mtl_last_fence()) return 0;   // the open submission: no wait without its submit
    const uint64_t t0 = now_ns(); const int ok = mtl_wait(f, 2000000000ull); g_o3.waits++; g_o3.wait_ns += now_ns() - t0; return ok;
}
void vk_occ3_begin(VkCommandBuffer cb, uint32_t off, int inv) {   // inside the pass, right before a gated draw or clear
    VkConditionalRenderingBeginInfoEXT bi = {VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, NULL, g_o2r.buf, off, inv ? VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT : 0};
    g_o3_cbeg(cb, &bi); g_o3.blocks++;
}
void vk_occ3_end(VkCommandBuffer cb) { g_o3_cend(cb); }
// ---- occ-s2 diagnostic: guest CPU reads of report words (DM_VK_OCC_RDWATCH=1, live): one sampled report word a second watched for loads with an ARM64 hardware watchpoint on
// every thread (gfx_vk_present.c vk_hwr_*), from its replay until a guest read or 8 presents; the read's age (presents since the report's replay), whether its result had
// landed (its submission done), the host PC (the guest's code runs as the JIT's: not a guest PC). Host reads (our own threads) are counted apart. Page protection (the step-0
// idea) is not used: nvn_dirty and the core own the pages' protection (memory note "core writable bypasses nvn_dirty"), and a fault in JIT code would take the JIT's slow path
int vk_hwr_arm(uint64_t a, uint64_t fence); int vk_hwr_poll(uint64_t *pc, int *landed, uint64_t *frame, uint64_t *host); void vk_hwr_off(void);
static struct { int on, busy, npc; uint64_t pf, last, sampled, armfail, nohit, guest, before, after, ab[3], aa[3], host, pc[4]; uint64_t sb[2][3], sa[2][3]; } g_o2rw;   // (occ-s3fast: sb / sa [calm, spike][age] before / after landing)
static int o3f_spike_in(uint64_t pf0, uint64_t pf1);   // occ-s3fast: a frame-time spike among presents [pf0, pf1]
static void occ2_rw_poll(void) {   // occ_sub_begin: the switch (live), the sampled word's watch
    extern uint64_t g_tex_frame; extern volatile long dm_env_gen; static long gen = -1;
    if (gen != dm_env_gen) { gen = dm_env_gen; const char *e = getenv("DM_VK_OCC_RDWATCH"); const int on = e && atoi(e) != 0;
        if (on != g_o2rw.on) { VKLOG("occ-s2: guest reads of report words %s (DM_VK_OCC_RDWATCH, live; a hardware load watchpoint on one sampled word a second, diagnostic)", on ? "watched" : "not watched");
            if (!on && g_o2rw.busy) { vk_hwr_off(); g_o2rw.busy = 0; } g_o2rw.on = on; } }
    if (!g_o2rw.busy) return;
    uint64_t pc = 0, fr = 0, host = 0; int landed = 0;
    if (vk_hwr_poll(&pc, &landed, &fr, &host)) { const uint64_t d = fr >= g_o2rw.pf ? fr - g_o2rw.pf : 0; g_o2rw.guest++;
        if (landed) { g_o2rw.after++; g_o2rw.aa[d > 2 ? 2 : d]++; } else { g_o2rw.before++; g_o2rw.ab[d > 2 ? 2 : d]++; }
        { const int sp = o3f_spike_in(g_o2rw.pf, fr); if (landed) g_o2rw.sa[sp][d > 2 ? 2 : d]++; else g_o2rw.sb[sp][d > 2 ? 2 : d]++; }   // occ-s3fast: split by a frame-time spike from the report's present to the read's
        if (g_o2rw.npc < 4) g_o2rw.pc[g_o2rw.npc++] = pc; }
    else if (g_tex_frame <= g_o2rw.pf + 8) return;
    else g_o2rw.nohit++;
    g_o2rw.host += host; vk_hwr_off(); g_o2rw.busy = 0;
}
static void occ2_rw_sample(uint64_t addr) {   // occ2_report: this report's word watched (one a second)
    extern uint64_t g_tex_frame;
    if (!g_o2rw.on || g_o2rw.busy || (g_o2rw.last && g_tex_frame < g_o2rw.last + 60)) return;
    if (vk_hwr_arm(addr, g_fence + 1)) { g_o2rw.busy = 1; g_o2rw.pf = g_o2rw.last = g_tex_frame; g_o2rw.sampled++; } else g_o2rw.armfail++;
}
// ---- occ-s3fast diagnostic (log only): frame-time spikes against the CPU-evaluated conditions, the landing bound and the RDWATCH reads (g_o3f above)
static o3f_fr *o3f_cur(void) { extern uint64_t g_tex_frame; o3f_fr *e = &g_o3f.r[g_tex_frame % O3F_RING]; if (e->pf != g_tex_frame + 1) { memset(e, 0, sizeof *e); e->pf = g_tex_frame + 1; } return e; }
static int o3f_spike_in(uint64_t pf0, uint64_t pf1) {
    if (pf1 < pf0 || pf1 - pf0 > 63) return 0;
    for (uint64_t p = pf0; p <= pf1; p++) if (g_o3f.spk[p & 63].pf == p + 1 && g_o3f.spk[p & 63].spike) return 1;
    return 0;
}
void vk_o3f_cond_note(uint64_t addr, int kind, int skip) {   // gfx_vk_exec.c (replay thread): a CPU-evaluated condition on a timestamp (kind 1) or flag (kind 2) word, its outcome
    if (g_occ2 < 2 || tl_wv) return;
    o3f_fr *e = o3f_cur(); e->ev++; if (skip) e->skip++; else e->drawn++;
    uint32_t h = (uint32_t)((addr >> 4) * 0x9E3779B1u) & (O3F_NA - 1);
    for (uint32_t k = 0; k < O3F_NA; k++, h = (h + 1) & (O3F_NA - 1)) {
        if (g_o3f.at[h].a == addr) break;
        if (!g_o3f.at[h].a) { if (g_o3f.nat * 4 >= O3F_NA * 3) { g_o3f.full++; return; } g_o3f.at[h].a = addr; g_o3f.at[h].last = -1; g_o3f.at[h].kind = (uint8_t)kind; g_o3f.nat++; break; } }
    if (g_o3f.at[h].a != addr) { g_o3f.full++; return; }
    g_o3f.at[h].ecur++;
    if (g_o3f.at[h].last >= 0 && g_o3f.at[h].last != (skip != 0)) { g_o3f.at[h].fcur++; e->flips++; if (e->nfa < 4) e->fa[e->nfa++] = addr; }
    g_o3f.at[h].last = skip != 0;
}
static void o3f_dump(uint64_t pf) {   // the ring's frames up to pf, two lines of six
    char b[2][300];
    for (int half = 0; half < 2; half++) { size_t k = 0; b[half][0] = 0;
        for (int i = 0; i < O3F_DUMP / 2; i++) { const uint64_t p = pf - (O3F_DUMP - 1) + (uint64_t)(half * (O3F_DUMP / 2) + i); const o3f_fr *e = &g_o3f.r[p % O3F_RING];
            if (p > pf || e->pf != p + 1) { k += (size_t)snprintf(b[half] + k, sizeof b[half] - k, " %llu:-", (unsigned long long)(p % 1000)); continue; }
            k += (size_t)snprintf(b[half] + k, sizeof b[half] - k, " %llu:%u.%u%s %u/%u/%u f%u b%u/%.1f%s", (unsigned long long)(p % 1000), e->ms10 / 10, e->ms10 % 10, e->spike ? "*" : "",
                                  e->ev, e->drawn, e->skip, e->flips, e->bw, e->bns / 1e6, e->bskip ? "w" : "");
            if (k >= sizeof b[half] - 48) break; } }
    VKLOG("occ-s3fast spike %llu (present%%1000:ms[*] ts/flag conds ev/drawn/skip, f flips, b landing waits/ms[w would-wait]):%s", (unsigned long long)pf, b[0]);
    VKLOG("occ-s3fast spike %llu (cont.):%s", (unsigned long long)pf, b[1]);
}
void vk_o3f_flash(uint64_t frame) { __atomic_store_n(&g_o3f.flash, frame + 1, __ATOMIC_RELEASE); }   // gfx_vk_present.c (the present thread): a one-frame flash in frame `frame`
static void o3f_flash_ctx(uint64_t f) {   // the replay thread: frames f - 1, f, f + 1 from the ring, a line each
    for (uint64_t p = f ? f - 1 : 0; p <= f + 1; p++) { const o3f_fr *e = &g_o3f.r[p % O3F_RING];
        if (e->pf != p + 1) { VKLOG("occ-s3fast flash ctx frame %llu: not in the ring", (unsigned long long)p); continue; }
        VKLOG("occ-s3fast flash ctx frame %llu%s: %u.%u ms%s; CPU ts/flag conditions %u (drawn %u, dropped %u), flips %u (%#llx %#llx %#llx %#llx); GPU conditions %u, resolve points %u, landing waits %u (%.2f ms)",
              (unsigned long long)p, p == f ? " (flash)" : "", e->ms10 / 10, e->ms10 % 10, e->spike ? " spike" : "", e->ev, e->drawn, e->skip, e->flips,
              (unsigned long long)(e->nfa > 0 ? e->fa[0] : 0), (unsigned long long)(e->nfa > 1 ? e->fa[1] : 0), (unsigned long long)(e->nfa > 2 ? e->fa[2] : 0), (unsigned long long)(e->nfa > 3 ? e->fa[3] : 0),
              e->gpuc, e->res, e->bw, e->bns / 1e6); }
}
static void o3f_present(void) {   // vk_pass_frame (a present, before g_tex_frame moves on), s2 / s3
    extern uint64_t g_tex_frame; const uint64_t t = now_ns(), pf = g_tex_frame;
    o3f_fr *e = o3f_cur(); e->bw = (uint32_t)g_o3f.bw; e->bns = g_o3f.bns; e->bskip = (uint32_t)g_o3f.bskip; e->res = (uint32_t)g_o3f.res; e->gpuc = (uint32_t)g_o3f.gpuc;
    g_o3f.bw = 0; g_o3f.bns = 0; g_o3f.bskip = 0; g_o3f.res = 0; g_o3f.gpuc = 0;
    { const uint64_t fx = __atomic_exchange_n(&g_o3f.flash, 0, __ATOMIC_ACQ_REL); if (fx) o3f_flash_ctx(fx - 1); }   // DM_VK_FLASHDIAG: the named frame's context
    int spike = 0;
    if (g_o3f.tprev) { const uint64_t dt = t - g_o3f.tprev; e->ms10 = (uint32_t)(dt / 100000ull);
        spike = dt > 33000000ull || (g_o3f.dtprev && 2 * dt > 3 * g_o3f.dtprev); g_o3f.dtprev = dt; g_o3f.frames++; g_o3f.spikes += (uint64_t)spike; }
    g_o3f.tprev = t; e->spike = (uint8_t)spike; g_o3f.spk[pf & 63].pf = pf + 1; g_o3f.spk[pf & 63].spike = (uint8_t)spike;
    g_o3f.ev[spike] += e->ev; g_o3f.fl[spike] += e->flips; g_o3f.bw2[spike] += e->bw; g_o3f.bns2[spike] += e->bns; g_o3f.sw[spike] += e->bw > 0; g_o3f.slow += e->ms10 > 330;
    for (uint32_t i = 0; i < O3F_NA; i++) if (g_o3f.at[i].a && (g_o3f.at[i].ecur || g_o3f.at[i].fcur)) {
        if (spike) { g_o3f.at[i].espk += g_o3f.at[i].ecur; g_o3f.at[i].fspk += g_o3f.at[i].fcur; } else { g_o3f.at[i].ecalm += g_o3f.at[i].ecur; g_o3f.at[i].fcalm += g_o3f.at[i].fcur; }
        g_o3f.at[i].ecur = g_o3f.at[i].fcur = 0; }
    if (spike && !g_o3f.dump_at && g_o3f.ndump < 64 && (!g_o3f.last_dump || pf >= g_o3f.last_dump + 60)) g_o3f.dump_at = pf + 3;
    if (g_o3f.dump_at && pf >= g_o3f.dump_at) { o3f_dump(pf); g_o3f.last_dump = pf; g_o3f.dump_at = 0; g_o3f.ndump++; }
}
static void o3f_log(double f) {   // occ2_log: the window's split and the most-flipping addresses (several lines, each short)
    VKLOG("occ-s3fast frames: %llu, over 33 ms %llu, spikes (> 33 ms or > 1.5x the previous) %llu; CPU ts/flag conditions a frame calm %.1f / spike %.1f, flips a frame calm %.2f / spike %.2f; "
          "landing waits a frame calm %.2f (%.3f ms) / spike %.2f (%.3f ms), frames with a wait calm %llu / spike %llu; table full %u",
          (unsigned long long)g_o3f.frames, (unsigned long long)g_o3f.slow, (unsigned long long)g_o3f.spikes, g_o3f.frames > g_o3f.spikes ? g_o3f.ev[0] / (double)(g_o3f.frames - g_o3f.spikes) : 0.0,
          g_o3f.spikes ? g_o3f.ev[1] / (double)g_o3f.spikes : 0.0, g_o3f.frames > g_o3f.spikes ? g_o3f.fl[0] / (double)(g_o3f.frames - g_o3f.spikes) : 0.0,
          g_o3f.spikes ? g_o3f.fl[1] / (double)g_o3f.spikes : 0.0, g_o3f.frames > g_o3f.spikes ? g_o3f.bw2[0] / (double)(g_o3f.frames - g_o3f.spikes) : 0.0,
          g_o3f.frames > g_o3f.spikes ? g_o3f.bns2[0] / 1e6 / (double)(g_o3f.frames - g_o3f.spikes) : 0.0, g_o3f.spikes ? g_o3f.bw2[1] / (double)g_o3f.spikes : 0.0,
          g_o3f.spikes ? g_o3f.bns2[1] / 1e6 / (double)g_o3f.spikes : 0.0, (unsigned long long)g_o3f.sw[0], (unsigned long long)g_o3f.sw[1], g_o3f.full);
    // the 8 addresses with the most flips, 4 a line: address kind evaluated calm/spike, flips calm/spike
    int top[8], nt = 0;
    for (int t = 0; t < 8; t++) { int b = -1; for (uint32_t i = 0; i < O3F_NA; i++) { if (!g_o3f.at[i].a || !(g_o3f.at[i].fspk + g_o3f.at[i].fcalm)) continue; int used = 0; for (int j = 0; j < nt; j++) used |= top[j] == (int)i;
            if (!used && (b < 0 || g_o3f.at[i].fspk + g_o3f.at[i].fcalm > g_o3f.at[b].fspk + g_o3f.at[b].fcalm)) b = (int)i; }
        if (b < 0) break; top[nt++] = b; }
    for (int l = 0; l * 4 < nt; l++) { char b[360]; size_t k = 0; b[0] = 0;
        for (int j = l * 4; j < nt && j < l * 4 + 4; j++) { const __typeof__(g_o3f.at[0]) *a = &g_o3f.at[top[j]];
            k += (size_t)snprintf(b + k, sizeof b - k, "%s %#llx %s ev %llu/%llu fl %llu/%llu", j > l * 4 ? ";" : "", (unsigned long long)a->a, a->kind == 1 ? "ts" : "flag",
                                  (unsigned long long)a->ecalm, (unsigned long long)a->espk, (unsigned long long)a->fcalm, (unsigned long long)a->fspk); }
        VKLOG("occ-s3fast flips by address (%d/%d; evaluated calm/spike, flips calm/spike):%s", l + 1, (nt + 3) / 4, b); }
    (void)f;
    g_o3f.frames = g_o3f.spikes = g_o3f.slow = 0; memset(g_o3f.sw, 0, sizeof g_o3f.sw); memset(g_o3f.ev, 0, sizeof g_o3f.ev); memset(g_o3f.fl, 0, sizeof g_o3f.fl); memset(g_o3f.bw2, 0, sizeof g_o3f.bw2); memset(g_o3f.bns2, 0, sizeof g_o3f.bns2);
    for (uint32_t i = 0; i < O3F_NA; i++) { g_o3f.at[i].espk = g_o3f.at[i].ecalm = g_o3f.at[i].fspk = g_o3f.at[i].fcalm = 0; }
    g_o3f.full = 0;
}
void vk_occ3_exec_log(double f, const uint64_t *core);   // gfx_vk_exec.c: the condition census line (core: forced, deferred, out of the ring, cpu waits, wait ns, blocks)
// ---- occ-s3fast diagnostic (DM_VK_OCC_CONDDIAG=1, live through DM_CTL env lines, off by default: it records a barrier and copies): the CPU-evaluated render conditions
// against the words the GPU sees. At each such condition (gfx_vk_exec.c o3g_cond) the two 64-bit words (addr, addr + 16) are copied by the GPU into a host-visible ring at
// the first point after the condition where no pass is open (the condition itself when none is, else its pass's end, before any resolve point there: draws of that pass are
// the only work between, and they write no report words), after a full barrier; once the submission is done the CPU's decision is compared with the decision those words
// give (mode 0 draws while equal, 1 while different). Per address: conditions, mismatches, the words written by the GPU per the GPU-write tracking at the decision
// (vk_gpu_pending: in flight; vk_sub_gw_overlaps: earlier in the open submission) and at the copy
enum { O3R_N = 16384, O3R_NA = 128 };
static struct {
    VkBuffer buf; VkDeviceMemory mem; uint8_t *map; int ok, tried;
    uint64_t head, tail, lost, nocopy;
    struct { uint64_t a, fence; uint8_t mode, skip, cls, dpend, dsub, epend, copied; } e[O3R_N];
    uint32_t q[64]; uint32_t nq;
    struct { uint64_t a, n, mis, dpend, dsub, epend, x, y; uint8_t cls; } at[O3R_NA]; uint32_t nat;
} g_o3r;
static int o3r_make(void) {
    if (g_o3r.ok || g_o3r.tried) return g_o3r.ok; g_o3r.tried = 1;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)O3R_N * 32, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &g_o3r.buf)) { VKLOG("[RED] occ-s3fast: the condition readback buffer could not be made"); return 0; }
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, g_o3r.buf, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    void *m = NULL;
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_o3r.mem) || vkBindBufferMemory(V.dev, g_o3r.buf, g_o3r.mem, 0) || vkMapMemory(V.dev, g_o3r.mem, 0, VK_WHOLE_SIZE, 0, &m)) { VKLOG("[RED] occ-s3fast: no host-visible memory for the condition readback"); return 0; }
    g_o3r.map = m; g_o3r.ok = 1; VKLOG("occ-s3fast: DM_VK_OCC_CONDDIAG=1: CPU-evaluated render conditions' words read back by the GPU at their place in the stream (diagnostic: a barrier and a copy per condition)");
    return 1;
}
void vk_o3r_cond(uint64_t addr, int mode, int skip, int cls) {   // gfx_vk_exec.c (replay thread): a CPU-evaluated condition (mode 0 / 1), its decision (1 = dropped), its class
    if (g_occ2 < 2 || tl_wv || g_cur < 0 || !o3r_make()) return;
    if (g_o3r.head - g_o3r.tail >= O3R_N || g_o3r.nq >= 64) { g_o3r.lost++; return; }
    const uint32_t i = (uint32_t)(g_o3r.head++ % O3R_N);
    g_o3r.e[i].a = addr; g_o3r.e[i].mode = (uint8_t)mode; g_o3r.e[i].skip = (uint8_t)(skip != 0); g_o3r.e[i].cls = (uint8_t)cls; g_o3r.e[i].fence = 0; g_o3r.e[i].copied = 0;
    g_o3r.e[i].dpend = (uint8_t)vk_gpu_pending(addr, 24); g_o3r.e[i].dsub = (uint8_t)vk_sub_gw_overlaps(addr, 24); g_o3r.e[i].epend = 0;
    g_o3r.q[g_o3r.nq++] = i;
}
void vk_o3r_flush(void) {   // gfx_vk_exec.c (replay thread), no pass open: the queued conditions' words copied here
    if (!g_o3r.nq || g_cur < 0) { g_o3r.nq = 0; return; }
    VkCommandBuffer cb = vk_cmd(); int bar = 0;
    for (uint32_t k = 0; k < g_o3r.nq; k++) { const uint32_t i = g_o3r.q[k]; VkBuffer sb; VkDeviceSize so;
        g_o3r.e[i].fence = g_fence + 1; g_o3r.e[i].epend = (uint8_t)vk_gpu_pending(g_o3r.e[i].a, 24);
        if (!vk_buf_at(g_o3r.e[i].a, 24, &sb, &so)) { g_o3r.nocopy++; continue; }
        if (!bar) { vk_img_barrier_all(cb); bar = 1; }
        VkBufferCopy r = {so, (VkDeviceSize)i * 32, 24}; vkCmdCopyBuffer(cb, sb, g_o3r.buf, 1, &r); g_o3r.e[i].copied = 1; }
    g_o3r.nq = 0;
}
static void o3r_poll(void) {   // occ_sub_begin (replay thread): the done entries compared
    const uint64_t done = vk_fence_done();
    while (g_o3r.tail < g_o3r.head) { const uint32_t i = (uint32_t)(g_o3r.tail % O3R_N);
        if (!g_o3r.e[i].fence) break;   // (not copied yet: queued in the open pass)
        if (g_o3r.e[i].fence > done) { if (g_o3r.head - g_o3r.tail > O3R_N / 2 && g_o3r.e[i].fence + 8 < g_fence) { g_o3r.lost++; g_o3r.tail++; continue; } break; }   // (a refused submission's never land)
        g_o3r.tail++;
        if (!g_o3r.e[i].copied) continue;
        uint64_t x, y; memcpy(&x, g_o3r.map + (size_t)i * 32, 8); memcpy(&y, g_o3r.map + (size_t)i * 32 + 16, 8);
        const int gskip = g_o3r.e[i].mode == 0 ? x != y : x == y;
        uint32_t h = (uint32_t)((g_o3r.e[i].a >> 4) * 0x9E3779B1u) & (O3R_NA - 1), k = 0;
        for (; k < O3R_NA; k++, h = (h + 1) & (O3R_NA - 1)) { if (g_o3r.at[h].a == g_o3r.e[i].a) break; if (!g_o3r.at[h].a) { if (g_o3r.nat * 4 >= O3R_NA * 3) { k = O3R_NA; break; } g_o3r.at[h].a = g_o3r.e[i].a; g_o3r.at[h].cls = g_o3r.e[i].cls; g_o3r.nat++; break; } }
        if (k == O3R_NA) continue;
        __typeof__(g_o3r.at[0]) *s = &g_o3r.at[h];
        s->n++; s->mis += (uint64_t)(gskip != g_o3r.e[i].skip); s->dpend += g_o3r.e[i].dpend; s->dsub += g_o3r.e[i].dsub; s->epend += g_o3r.e[i].epend; s->x = x; s->y = y;
    }
}
static void o3r_log(void) {   // occ2_log: the 8 addresses with the most mismatches (else the most conditions), a line each
    static const char *const cn[] = {"gpu", "ts", "flag", "mem", "samples", "?"};
    if (!g_o3r.ok) return;
    VKLOG("occ-s3fast condread: lost %llu, outside the pools %llu, addresses %u (per address: conditions, CPU decision != the words the GPU saw, GPU-written: in flight / this submission at the decision, in flight at the copy)",
          (unsigned long long)g_o3r.lost, (unsigned long long)g_o3r.nocopy, g_o3r.nat);
    int used[8], nu = 0;
    for (int t = 0; t < 8; t++) { int b = -1; for (int i = 0; i < O3R_NA; i++) { if (!g_o3r.at[i].a || !g_o3r.at[i].n) continue; int u = 0; for (int j = 0; j < nu; j++) u |= used[j] == i;
            if (!u && (b < 0 || g_o3r.at[i].mis > g_o3r.at[b].mis || (g_o3r.at[i].mis == g_o3r.at[b].mis && g_o3r.at[i].n > g_o3r.at[b].n))) b = i; }
        if (b < 0) break; used[nu++] = b; const __typeof__(g_o3r.at[0]) *s = &g_o3r.at[b];
        VKLOG("occ-s3fast condread %#llx %s: conditions %llu, mismatches %llu, GPU-written at the decision %llu / %llu, at the copy %llu; last words x %#llx y %#llx", (unsigned long long)s->a,
              cn[s->cls < 5 ? s->cls : 5], (unsigned long long)s->n, (unsigned long long)s->mis, (unsigned long long)s->dpend, (unsigned long long)s->dsub, (unsigned long long)s->epend,
              (unsigned long long)s->x, (unsigned long long)s->y); }
    memset(g_o3r.at, 0, sizeof g_o3r.at); g_o3r.nat = 0; g_o3r.lost = g_o3r.nocopy = 0;
}
void vk_o3p_log(double f);   // gfx_vk_exec.c: the plan's line (occ-s3fast)
// occ_log, every 600 presents. occ-s3fast: several lines, each under ~400 characters (the logger cuts a line at ~590: the old single line lost its RED counts, landing waits and cost)
static void occ2_log(double f, uint64_t ne, uint64_t ng) {
    uint64_t gh = 0, gw = 0; for (int i = 0; i < 4; i++) { gh += g_odr[i].hit; gw += g_odr[i].words; }
    const uint64_t wq = __atomic_load_n(&g_o2_wq, __ATOMIC_RELAXED), wu = __atomic_load_n(&g_o2_wunready, __ATOMIC_RELAXED), wc = __atomic_load_n(&g_o2_wcb_off, __ATOMIC_RELAXED);
    const uint64_t gpn = __atomic_load_n(&g_o2_gpu_ns, __ATOMIC_RELAXED), gpc = __atomic_load_n(&g_o2_gpu_n, __ATOMIC_RELAXED);
    static uint64_t wq0, wu0, wc0, gpn0, gpc0;
    const uint64_t sm = g_o2.sub_max > g_o2.sub_q ? g_o2.sub_max : g_o2.sub_q, red = g_o2.red + g_o2c.longw + g_o2c.outside + g_o2c.over + (wu - wu0) + g_o2.cb_off + (wc - wc0) + gh;
    uint64_t np = 0; for (int i = 0; i < O2R_N; i++) np += g_o2c.res[i];
    const int L = g_occ2;
    VKLOG("occ-s%d (1/5) last %.0f presents: samples-passed reports %llu (%.0f a frame), written by the GPU %llu (a later report at the same address in one batch %llu); "
          "resolve points %llu (%.1f a frame): plan / pass ends %llu, part ends %llu, submission ends %llu, forced by a condition %llu, part joins %llu, copies reading one %llu",
          L, f, (unsigned long long)g_o2c.rep, g_o2c.rep / f, (unsigned long long)g_o2c.words, (unsigned long long)g_o2c.dup, (unsigned long long)np, np / f,
          (unsigned long long)g_o2c.res[O2R_PASS], (unsigned long long)g_o2c.res[O2R_PART], (unsigned long long)g_o2c.res[O2R_SUB], (unsigned long long)g_o2c.res[O2R_FORCED],
          (unsigned long long)g_o2c.res[O2R_JOIN], (unsigned long long)g_o2c.res[O2R_READ]);
    VKLOG("occ-s%d (2/5) opportunities passed over %.1f a frame, taken by a bound %llu, deferred to a part join %llu; dispatches %.1f a frame; query slots copied %.0f a frame in %.1f copies "
          "(empty queries %llu); guest copies %.1f vkCmdCopyBuffer a frame, %.1f regions (runs of adjacent words %.1f)",
          L, g_o2c.kept / f, (unsigned long long)g_o2c.must, (unsigned long long)g_o2c.defer, g_o2c.disp / f, g_o2c.slots / f, g_o2c.qruns / f, (unsigned long long)g_o2c.empty,
          g_o2c.copies / f, g_o2c.regions / f, g_o2c.runs / f);
    VKLOG("occ-s%d (3/5) per-draw queries %.0f a frame (by workers %.0f), most in one submission %llu, chunk resets %llu, ring waits %llu (%.2f ms); reports before any reset %llu, running bases %llu; "
          "landing bound: waits %llu (%.2f ms), not reachable %llu, would-wait (DM_VK_OCC_BOUND=0) %llu",
          L, g_o2.q / f + (wq - wq0) / f, (wq - wq0) / f, (unsigned long long)sm, (unsigned long long)g_o2.resets, (unsigned long long)g_o2.waits, g_o2.wait_ns / 1e6,
          (unsigned long long)g_o2c.nores, (unsigned long long)g_o2c.adv, (unsigned long long)g_o2b.waits, g_o2b.ns / 1e6, (unsigned long long)g_o2b.unreach, (unsigned long long)g_o2b.skip);
    VKLOG("%socc-s%d (4/5) RED (must be 0): windows longer than the ring %llu, outside the pools %llu, over 32K at one point %llu, ring faults %llu, chunk not reset at a hand-out %llu, "
          "query ended in another buffer %llu, report sequences off the dry pass's (since start) %llu, GPU-side reads before the GPU's write %llu (words %llu, table full %llu)",
          red || g_o2_rseq_off ? "[RED] " : "", L, (unsigned long long)g_o2c.longw, (unsigned long long)g_o2c.outside, (unsigned long long)g_o2c.over, (unsigned long long)g_o2.red,
          (unsigned long long)(wu - wu0), (unsigned long long)(g_o2.cb_off + (wc - wc0)), (unsigned long long)g_o2_rseq_off, (unsigned long long)gh, (unsigned long long)gw, (unsigned long long)g_odw_full);
    char gp[96]; if (gpc > gpc0) snprintf(gp, sizeof gp, "%.3f ms a frame (%.1f us a point)", (gpn - gpn0) / 1e6 / f, (double)(gpn - gpn0) / 1e3 / (double)(gpc - gpc0)); else snprintf(gp, sizeof gp, "not measured (DM_VK_PASS_TIMES=1)");
    VKLOG("occ-s%d (5/5) cost a frame: replay %.2f ms, GPU busy %.2f ms (compare DM_VK_OCC2=0); resolve CPU %.3f ms (%.1f us a point), barriers recorded %.1f, forced pass splits %.2f; resolve GPU %s",
          L, ne / 1e6 / f, ng / 1e6 / f, g_o2c.ns / 1e6 / f, np ? g_o2c.ns / 1e3 / (double)np : 0.0, g_o2c.bars / f, g_o3.forced / f, gp);
    vk_o3p_log(f);
    if (g_o2rw.on || g_o2rw.sampled) { char pcs[100]; size_t k = 0; pcs[0] = 0; for (int i = 0; i < g_o2rw.npc; i++) k += (size_t)snprintf(pcs + k, sizeof pcs - k, "%s%#llx", i ? " " : "", (unsigned long long)g_o2rw.pc[i]);
        VKLOG("occ-s%d rdwatch (DM_VK_OCC_RDWATCH, live): words watched %llu (not armed %llu), guest reads %llu: before landing %llu (presents 0/1/2+ %llu/%llu/%llu), after %llu (%llu/%llu/%llu), "
              "none in 8 presents %llu, host reads %llu; PCs %s", L, (unsigned long long)g_o2rw.sampled, (unsigned long long)g_o2rw.armfail, (unsigned long long)g_o2rw.guest,
              (unsigned long long)g_o2rw.before, (unsigned long long)g_o2rw.ab[0], (unsigned long long)g_o2rw.ab[1], (unsigned long long)g_o2rw.ab[2], (unsigned long long)g_o2rw.after,
              (unsigned long long)g_o2rw.aa[0], (unsigned long long)g_o2rw.aa[1], (unsigned long long)g_o2rw.aa[2], (unsigned long long)g_o2rw.nohit, (unsigned long long)g_o2rw.host, k ? pcs : "none");
        VKLOG("occ-s%d rdwatch by frame time (a spike from the report's present to the read's): before landing calm %llu/%llu/%llu, spike %llu/%llu/%llu; after landing calm %llu/%llu/%llu, "
              "spike %llu/%llu/%llu (presents 0/1/2+)", L, (unsigned long long)g_o2rw.sb[0][0], (unsigned long long)g_o2rw.sb[0][1], (unsigned long long)g_o2rw.sb[0][2],
              (unsigned long long)g_o2rw.sb[1][0], (unsigned long long)g_o2rw.sb[1][1], (unsigned long long)g_o2rw.sb[1][2], (unsigned long long)g_o2rw.sa[0][0], (unsigned long long)g_o2rw.sa[0][1],
              (unsigned long long)g_o2rw.sa[0][2], (unsigned long long)g_o2rw.sa[1][0], (unsigned long long)g_o2rw.sa[1][1], (unsigned long long)g_o2rw.sa[1][2]);
        const int on = g_o2rw.on, busy = g_o2rw.busy; const uint64_t pf = g_o2rw.pf, last = g_o2rw.last; memset(&g_o2rw, 0, sizeof g_o2rw); g_o2rw.on = on; g_o2rw.busy = busy; g_o2rw.pf = pf; g_o2rw.last = last; }
    else VKLOG("occ-s%d rdwatch: CPU reads of report words not watched (DM_VK_OCC_RDWATCH=1, live)", L);
    o3f_log(f);
    o3r_log();
    if (g_occ2 >= 3) { const uint64_t c[6] = {g_o3.forced, g_o3.defer, g_o3.outring, g_o3.waits, g_o3.wait_ns, g_o3.blocks}; vk_occ3_exec_log(f, c); }
    wq0 = wq; wu0 = wu; wc0 = wc; gpn0 = gpn; gpc0 = gpc;
    memset(&g_o2c, 0, sizeof g_o2c); memset(&g_o2b, 0, sizeof g_o2b); memset(&g_o3, 0, sizeof g_o3); memset(g_odr, 0, sizeof g_odr); g_odw_full = 0;
    const uint64_t sq = g_o2.sub_q; memset(&g_o2, 0, sizeof g_o2); g_o2.sub_q = sq;
}
void vk_occd_exec_log(double frames);   // gfx_vk_exec.c: the condition census line
static void occ_log(double f) {   // every 600 presents (occ_sub_begin)
    extern uint64_t g_tex_frame; static uint64_t ne0, ng0, fr0, sl0, lag0[5], cmp0, mis0, unr0, zu0, lo0, sl1, wq0, wu0, wc0;
    const uint64_t ne = g_vkst.ns_exec, ng = g_gpu_ns;
    if (g_occdiag) {
        for (int i = 0; i < 16; i++) if (g_od_pa[i].used && g_od_pa[i].pf < g_tex_frame) occd_fold(&g_od_pa[i]);
        const uint64_t fr = __atomic_load_n(&g_od_frames, __ATOMIC_RELAXED), sl = __atomic_load_n(&g_od_slow, __ATOMIC_RELAXED); uint64_t lag[5], nrep = 0;
        for (int i = 0; i < 5; i++) lag[i] = __atomic_load_n(&g_od_lag[i], __ATOMIC_RELAXED);
        for (int k = 0; k < 3; k++) for (int i = 0; i < 6; i++) nrep += g_od_age[k][i];
        char ab[3][160]; for (int k = 0; k < 3; k++) snprintf(ab[k], sizeof ab[k], "%llu/%llu/%llu/%llu/%llu/%llu", (unsigned long long)g_od_age[k][0], (unsigned long long)g_od_age[k][1],
            (unsigned long long)g_od_age[k][2], (unsigned long long)g_od_age[k][3], (unsigned long long)g_od_age[k][4], (unsigned long long)g_od_age[k][5]);
        VKLOG("occ-s0: report landing (1/2), last %.0f presents: samples-passed reports %llu (%.0f a frame); value written, by age in presents 0/1/2/3/fixed/real-after-fence: in frames <= 33 ms %s, "
              "in frames > 33 ms %s, frame time not known %s", f, (unsigned long long)nrep, nrep / f, ab[0], ab[1], ab[2]);   // (occ-s3fast: two lines, each under ~400 characters)
        VKLOG("occ-s0: report landing (2/2): frames %llu, over 33 ms %llu; GPU result available (fence done) 0/1/2/3/4+ presents after the replay: %llu/%llu/%llu/%llu/%llu",
              (unsigned long long)(fr - fr0), (unsigned long long)(sl - sl0),
              (unsigned long long)(lag[0] - lag0[0]), (unsigned long long)(lag[1] - lag0[1]), (unsigned long long)(lag[2] - lag0[2]), (unsigned long long)(lag[3] - lag0[3]), (unsigned long long)(lag[4] - lag0[4]));
        fr0 = fr; sl0 = sl; memcpy(lag0, lag, sizeof lag0); memset(g_od_age, 0, sizeof g_od_age);
        for (int i = 0; i < 4; i++) { const __typeof__(g_odr[0]) *r = &g_odr[i];   // (occ-s3fast: a line per reader kind, each under ~400 characters)
            VKLOG("occ-s0: GPU-side reads of samples-passed report words before their result landed (%d/4), last %.0f presents: %s: reads %llu, over an unlanded word %llu (words %llu; "
                  "submissions after the report 0/1/2+ %llu/%llu/%llu; presents 0/1/2+ %llu/%llu/%llu; draws after it min %llu avg %.0f)", i + 1, f, g_odr_names[i], (unsigned long long)r->reads, (unsigned long long)r->hit, (unsigned long long)r->words,
                                  (unsigned long long)r->sub[0], (unsigned long long)r->sub[1], (unsigned long long)r->sub[2], (unsigned long long)r->pres[0], (unsigned long long)r->pres[1], (unsigned long long)r->pres[2],
                                  (unsigned long long)(r->hit ? r->dmin : 0), r->hit ? (double)r->dsum / (double)r->hit : 0.0); }
        VKLOG("occ-s0: GPU-side reads before landing: pending-word table full %llu (CPU reads: DM_VK_OCC_RDWATCH under s2 / s3)", (unsigned long long)g_odw_full);
        if (g_occ2 < 2) { memset(g_odr, 0, sizeof g_odr); g_odw_full = 0; }   // (occ-s2: its line takes them, below)
        vk_occd_exec_log(f);
    }
    if (g_occ2 >= 2) occ2_log(f, ne - ne0, ng - ng0);   // occ-s2 / occ-s3
    else if (g_occ2) {
        const uint64_t cmp = __atomic_load_n(&g_o2c_cmp, __ATOMIC_RELAXED), mis = __atomic_load_n(&g_o2c_mis, __ATOMIC_RELAXED), unr = __atomic_load_n(&g_o2c_unrec, __ATOMIC_RELAXED),
            zu = __atomic_load_n(&g_o2c_zero_unrec, __ATOMIC_RELAXED), lo = __atomic_load_n(&g_o2c_lostold, __ATOMIC_RELAXED), sl = __atomic_load_n(&g_o2c_slots, __ATOMIC_RELAXED),
            wq = __atomic_load_n(&g_o2_wq, __ATOMIC_RELAXED), wu = __atomic_load_n(&g_o2_wunready, __ATOMIC_RELAXED), wc = __atomic_load_n(&g_o2_wcb_off, __ATOMIC_RELAXED),
            md = __atomic_exchange_n(&g_o2c_maxd, 0, __ATOMIC_RELAXED);
        const uint64_t sm = g_o2.sub_max > g_o2.sub_q ? g_o2.sub_max : g_o2.sub_q;
        const char *rd = g_o2.red + (mis - mis0) + (wu - wu0) + g_o2.cb_off + (wc - wc0) ? "[RED] " : "";   // (occ-s3fast: three lines, each under ~400 characters)
        VKLOG("%socc-s1 (1/3) last %.0f presents: per-draw queries %.0f a frame (by workers %.0f), most in one submission %llu; chunk resets %llu, ring waits %llu (%.2f ms; expected 0); reports compared %llu "
              "(window slots %.0f a report), window sum differs from the old span sum %llu (largest difference %llu)",
              rd, f, g_o2.q / f + (wq - wq0) / f, (wq - wq0) / f, (unsigned long long)sm, (unsigned long long)g_o2.resets,
              (unsigned long long)g_o2.waits, g_o2.wait_ns / 1e6, (unsigned long long)(cmp - cmp0), cmp - cmp0 ? (double)(sl - sl1) / (double)(cmp - cmp0) : 0.0, (unsigned long long)(mis - mis0), (unsigned long long)md);
        VKLOG("%socc-s1 (2/3) not compared: old spans lost %llu, chunk gone %llu, window longer than the ring %llu; unrecorded slots in windows %llu, zero-count reports with unrecorded draws %llu; "
              "RED (must be 0): ring overflows and faults %llu, chunk not reset at a hand-out %llu, query ended in another buffer %llu, stream serials off the dry pass's (since start) %llu",
              rd, (unsigned long long)(lo - lo0), (unsigned long long)g_o2.lost, (unsigned long long)g_o2.longw, (unsigned long long)(unr - unr0), (unsigned long long)(zu - zu0),
              (unsigned long long)g_o2.red, (unsigned long long)(wu - wu0), (unsigned long long)(g_o2.cb_off + (wc - wc0)), (unsigned long long)vk_occ2_ser_off());
        VKLOG("occ-s1 (3/3) replay %.2f ms a frame, GPU busy %.2f ms a frame (compare with DM_VK_OCC2 unset)", (ne - ne0) / 1e6 / f, (ng - ng0) / 1e6 / f);
        cmp0 = cmp; mis0 = mis; unr0 = unr; zu0 = zu; lo0 = lo; sl1 = sl; wq0 = wq; wu0 = wu; wc0 = wc;
        const uint64_t sq = g_o2.sub_q; memset(&g_o2, 0, sizeof g_o2); g_o2.sub_q = sq;
    } else if (g_occdiag) VKLOG("occ-s0: replay %.2f ms a frame, GPU busy %.2f ms a frame", (ne - ne0) / 1e6 / f, (ng - ng0) / 1e6 / f);
    ne0 = ne; ng0 = ng;
}
static void occ_sub_begin(void) {   // mtl_sub_begin (replay thread), g_nsubs already this submission's
    extern uint64_t g_tex_frame;
    if (g_occdiag && g_od_ovf_sub && g_od_ovf_sub + 1 == g_nsubs && g_od_ovf_n < 32) { g_od_ovf_n++;
        VKLOG("[RED] occ-s0: occlusion query overflow context: submission %llu (present %llu): draws %llu, passes (replay thread) %llu, samples-passed reports %llu, resets %llu; "
              "at the first overflow (pass span %d): draws %llu, passes %llu, reports %llu, resets %llu", (unsigned long long)g_od_ovf_sub, (unsigned long long)g_tex_frame,
              (unsigned long long)(vk_occ2_serial() - g_od_sub_ser0), (unsigned long long)(g_vk_pends - g_od_sub_pe0), (unsigned long long)g_od_sub_rep, (unsigned long long)g_od_sub_rst, OQ,
              (unsigned long long)g_od_ovf_at[0], (unsigned long long)g_od_ovf_at[1], (unsigned long long)g_od_ovf_at[2], (unsigned long long)g_od_ovf_at[3]); }
    g_od_sub_ser0 = vk_occ2_serial(); g_od_sub_pe0 = g_vk_pends; g_od_sub_rep = 0; g_od_sub_rst = 0;
    if (g_occ2) { if (g_occ2_act >= 0) { occ2_red("a per-draw query still active at a submission start", (uint64_t)g_occ2_act, g_nsubs); g_occ2_act = -1; }
        if (g_o2.sub_q > g_o2.sub_max) g_o2.sub_max = g_o2.sub_q; g_o2.sub_q = 0; }
    if (g_occ2 >= 2) occ2_rw_poll();   // occ-s2: DM_VK_OCC_RDWATCH (live)
    if (g_occ2 >= 2 && g_o3r.ok) o3r_poll();   // occ-s3fast: DM_VK_OCC_CONDDIAG's readbacks of finished submissions
    static uint64_t last; if (!last) last = g_tex_frame;
    if (g_tex_frame >= last + 600) { occ_log((double)(g_tex_frame - last)); last = g_tex_frame; }
}
// combined1120 (self-test case 8, "counter report then fill: the fill stays", failing in both modes since it was written): a report or event is written by the completion
// thread after the whole submission ran, so a GPU write recorded after it in the same submission was overwritten by it. The submission's GPU writes are kept in
// command order (sub_gpu_wrote: wr, wseq) and every report knows how many came before it (post: wseq): the bytes a later one covers keep the GPU's value (already in
// the game's memory: imported, or merged just before). A list that ran out (wr_over) leaves the report written as before
static uint64_t g_report_later;
static void report_later_mask(const vksub_t *s, uint32_t seq, uint64_t a, uint32_t n, uint8_t *m) {
    memset(m, 0, n); if (!s || s->wr_over) return;
    for (int i = 0; i < s->nwr; i++) { if (s->wr[i].seq <= seq) continue; const uint64_t lo = s->wr[i].lo, hi = s->wr[i].hi; if (hi <= a || lo >= a + n) continue;
        for (uint32_t b = 0; b < n; b++) if (a + b >= lo && a + b < hi) m[b] = 1; }
}
static void *completion_main(void *a) {
    (void)a; pthread_setname_np(pthread_self(), "dm-vk-complete");
    uint32_t oq_ep = ~0u; uint64_t oq_sum = 0; int oq_n = 0, oq_lost = 0, oq_logged = 0;   // r50: the current counter epoch's samples (vk_oq_reset)
    for (;;) {
        pthread_mutex_lock(&g_cmu);
        while (!g_ch) pthread_cond_wait(&g_ccv, &g_cmu);
        cpend *c = g_ch; pthread_mutex_unlock(&g_cmu);
        g_f0_sub = &c->f0;
        nvf0_scope completion, gpu_wait; vk_f0_begin(&completion, NVF0_COMPLETION, 0);
        vk_f0_begin(&gpu_wait, NVF0_GPU_WAIT, 0);
        wait_fence(c->fence);
#if defined(DM_NXN1X)
        void nvn_trace_gpu_completion(uint64_t fence, int reports_done);
        nvn_trace_gpu_completion(c->fence, 0);
        nvnc_gpu_done(c->content_sub);
        unsigned content_origin = nvnc_origin_push(NVNC_GPU_WRITEBACK);
#endif
        if (c->f0.id) f0_max(&g_f0_gpu_observed, c->fence);
        vk_f0_end(&gpu_wait, 0);
        const int crd = g_rdiag > 0; uint64_t ct = crd ? now_ns() : 0;   // replaydiag: the completion thread's work after the GPU fence (slots 43-46, 47 post ops)
#define CRD(k) do { if (crd) { const uint64_t tn = now_ns(); g_rd_ns[k] += tn - ct; g_rd_n[k]++; ct = tn; } } while (0)
        s67_frame(c->frame);   // s67: a later frame: the one before has ended on the GPU
        while (atomic_load(&g_pc_hold) && vk_poolcopy_on() && pc_sync() != PC_SERIAL) usleep(500);   // stage2c self-test: merges held (several submissions in flight)
        if (g_shadow_pools && c->rslot >= 0) { nvf0_scope copyback; vk_f0_begin(&copyback, NVF0_COPYBACK, 0);
            pools_copy_back(&g_slot[c->rslot]); vk_f0_end(&copyback, 0); }
        if (s56_any()) { nvf0_scope mirror; vk_f0_begin(&mirror, NVF0_MIRROR, 0);
            s56_land(c->fence); vk_f0_end(&mirror, 0); }   // s56: render targets written back into their storage, before the event / counter writes and g_done
        CRD(43);   // copy-backs, mirrors
        uint64_t tend = 0, tbeg = 0;   // (tbeg: ootgap)
        if (g_tsq && c->slot >= 0) { uint64_t t[2]; if (vkGetQueryPoolResults(V.dev, g_tsq, 2 * (uint32_t)c->slot, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            tend = t[1]; tbeg = t[0];
            if (t[1] > t[0])
            { uint64_t ns = (uint64_t)((double)(t[1] - t[0]) * V.props.limits.timestampPeriod); g_gpu_ns += ns; int k = (int)(c->frame % 64);
              if (g_frame_gpu_of[k] != c->frame + 1) { g_frame_gpu[k] = 0; g_frame_gpu_of[k] = c->frame + 1; } g_frame_gpu[k] += ns; s67_span(t[0], t[1]); } } }
        if (g_pq && pass_times_on() && c->slot >= 0) {   // this submission's passes
            int np = c->npass; uint64_t t[PASSQ * 2];
            if (np && vkGetQueryPoolResults(V.dev, g_pq, (uint32_t)(c->slot * PASSQ * 2), (uint32_t)np * 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                for (int i = 0; i < np; i++) if (t[2 * i + 1] > t[2 * i]) { int f = (int)c->pmeta[i].fidx; g_pacc[f] += (double)(t[2 * i + 1] - t[2 * i]) * V.props.limits.timestampPeriod / 1e6; g_pcnt[f]++; g_plast[f] = c->pmeta[i]; if (f + 1 > g_pmax) g_pmax = f + 1; r138_add(&c->pmeta[i], (double)(t[2 * i + 1] - t[2 * i]) * V.props.limits.timestampPeriod / 1e6); s67_pass((double)(t[2 * i + 1] - t[2 * i]) * V.props.limits.timestampPeriod / 1e6, &c->pmeta[i]);
                    if (c->pmeta[i].fmt == VK_TS_OCC2) { __atomic_fetch_add(&g_o2_gpu_ns, (uint64_t)((double)(t[2 * i + 1] - t[2 * i]) * V.props.limits.timestampPeriod), __ATOMIC_RELAXED); __atomic_fetch_add(&g_o2_gpu_n, 1, __ATOMIC_RELAXED); } }   // occ-s3fast
        }
        if (c->og && g_ogq && tend && c->slot >= 0) {   // ootgap: this submission's gaps
            uint64_t pt[PASSQ * 2]; const int np = c->npass > PASSQ ? PASSQ : c->npass;
            if (!np || vkGetQueryPoolResults(V.dev, g_pq, (uint32_t)(c->slot * PASSQ * 2), (uint32_t)np * 2, sizeof pt, pt, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
                og_done(c->slot, c->og, &c->ogm, pt, np, c->pmeta, tbeg, tend);
        }
        uint64_t now = c->n ? gpu_ticks() : 0;
        uint64_t occlusion[OQ], counter[CQ];
        int got_occlusion = 0, got_counter = 0;
        if (c->slot >= 0) {
            const vksub_t *slot = &g_slot[c->slot];
            if (slot->noq && g_oq)
                got_occlusion = vkGetQueryPoolResults(V.dev, g_oq, c->slot * OQ, (uint32_t)(slot->noq < OQ ? slot->noq : OQ),   // (S1e: at most OQ)
                    sizeof occlusion, occlusion, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
            if (slot->ncq && tend && g_cq)
                got_counter = vkGetQueryPoolResults(V.dev, g_cq, c->slot * CQ, slot->ncq,
                    sizeof counter, counter, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
        }
        CRD(44);   // replaydiag: s67, timestamps, pass times, ootgap, query results
        for (int i = 0; i < c->n; i++) {
            post_op *o = &c->ops[i];
            if (o->kind == POST_SEG) continue;   // S1e (parallel recording): a segment's placeholder (none is left once its part is joined)
            if (o->kind == 1) {
#if defined(DM_NXN1X)
                nvnc_cpu((uint64_t)(uintptr_t)o->val, sizeof *o->val, NVNC_GPU_WRITEBACK);
#endif
                // poolcopy (self-test 11-20 case 7): the event's page may be write-protected by the dirty tracking (a synced copied pool): made writable and
                // stamped first, as every host write into the game's memory (the completion thread's store faulted: c0000005, NXN1X died)
                { uint8_t m[4]; report_later_mask(c->rslot >= 0 ? &g_slot[c->rslot] : NULL, o->wseq, (uint64_t)(uintptr_t)o->val, 4, m);   // combined1120
                  if (m[0] | m[1] | m[2] | m[3]) { g_report_later++; continue; } }   // a GPU write recorded after it covers it: that one is the latest
                dm_guest_write_prepare((void *)o->val, sizeof *o->val);
                if (o->mode == 1) *o->val += o->v; else *o->val = o->v; continue; }
            if (o->kind == 4) { if (o->v != oq_ep) { oq_ep = o->v; oq_sum = 0; oq_n = 0; oq_lost = 0; }   // ops complete in order: one epoch at a time
                if (o->mode == 2) { oq_sum += occ2_sum(o->addr, o->key, NULL); oq_n++; continue; }   // occ-s1: an old span's draws on the per-draw ring
                uint64_t r = 0; if (o->q < 0 || o->mode) oq_lost = 1; else if (got_occlusion) { oq_sum += occlusion[o->q % OQ]; oq_n++; } else if (vkGetQueryPoolResults(V.dev, g_oq, (uint32_t)o->q, 1, 8, &r, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) { oq_sum += r; oq_n++; } else oq_lost = 1;
                continue; }
            uint64_t oqv = vk_occ_fixed() >= 0 ? (uint64_t)vk_occ_fixed() : 1u << 20;   // rptfix: a fixed report left to this thread (a GPU write was in flight) gets the same constant
            if (g_oq && o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED && o->mode != 8 && o->mode != POST_HALF) {   // (occ-s2: a timestamp half has no count here) if (o->v != oq_ep) { oq_ep = o->v; oq_sum = 0; oq_n = 0; oq_lost = 0; } oqv = oq_lost && !oq_sum ? 1 : oq_sum;
                if (o->s1 || o->od) occ_sp_done(o, oq_sum, oq_lost);   // occ-s1: the window's sum against the old span sum; occ-s0: the landing age
                if (oq_logged < 8) { oq_logged++; VKLOG("occlusion report %#llx: %llu samples (%d queries%s)", (unsigned long long)o->addr, (unsigned long long)oqv, oq_n, oq_lost ? ", some lost" : ""); }
                if (o->key && o->mode != 8) { occk_put(o->key, o->kf, oqv); if (o->key2) occk_put(o->key2, o->kf, oqv); }   // occkey: the key's real count, for the same report in the next frames (occaddr: and by its address)
                if (o->mode == 7 && !occ_land_on()) continue; }   // occkey: written at replay already (occland: and now this submission's own count over it)
            uint64_t ts = now, tq;
            if (o->q >= 0 && tend) {
                if (got_counter) tq = counter[o->q % CQ];
                else if (vkGetQueryPoolResults(V.dev, g_cq, (uint32_t)o->q, 1, 8, &tq, 8,
                                               VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) tq = UINT64_MAX;
            }
            if (o->q >= 0 && tend && tq <= tend) {
                uint64_t back = (uint64_t)((double)(tend - tq) * V.props.limits.timestampPeriod * 384 / 625); if (back < now) ts = now - back; }
            // render_cond: a timestamp report's counter field is 0, as on the GPU (gfx_vk_exec.c render_cond evaluates conditions on it)
            uint64_t rec[2] = {o->type == 0 || o->type == 0x10 ? (DM_GAME_RENDER_COND ? 0 : ts) : o->type == NVN_COUNTER_TYPE_SAMPLES_PASSED ? oqv : 0, ts};
            if (o->mode == 8) rec[0] = o->key;   // occkey / replay2: a replay-time report left to this thread keeps the value it was queued with
            if (o->mode == POST_HALF) rec[1] = o->key;   // occ-s2: the queued timestamp, bytes 8..15 only (the counter word is the GPU's)
#if defined(DM_NXN1X)
            nvnc_cpu(o->addr, sizeof rec, NVNC_GPU_WRITEBACK);
#endif
            { uint8_t m[16]; report_later_mask(c->rslot >= 0 ? &g_slot[c->rslot] : NULL, o->wseq, o->addr, 16, m);   // combined1120: bytes a later GPU write covers keep it
              if (o->mode == POST_HALF) memset(m, 1, 8);   // occ-s2: the counter word is not this thread's to write
              int any = 0, all = 1; for (int b = 0; b < 16; b++) { any |= m[b]; all &= m[b]; }
              if (all) { g_report_later++; continue; }
              dm_guest_write_prepare((void *)(uintptr_t)o->addr, sizeof rec);   // poolcopy: as the event above (a counter report into a protected page)
              if (!any) memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec);
              else { g_report_later++; for (int b = 0; b < 16; b++) if (!m[b]) ((volatile uint8_t *)(uintptr_t)o->addr)[b] = ((const uint8_t *)rec)[b]; } }
        }
        pthread_mutex_lock(&g_cmu); g_ch = c->next; if (!g_ch) g_ct = NULL; pthread_mutex_unlock(&g_cmu);
        if (g_prereset > 0 && c->rslot >= 0) { vksub_t *s = &g_slot[c->rslot];   // its command buffers reset here (slot_recycle); the slot is not taken before g_done passes this fence
            if (s->fence == c->fence) { for (int i = 0; i < s->nmain_alloc; i++) vkResetCommandBuffer(s->mains[i], 0); for (int i = 0; i < s->nup_alloc; i++) vkResetCommandBuffer(s->ups[i], 0);
                for (int i = 0; i < s->nrcb_alloc; i++) vkResetCommandBuffer(s->rcb[i], 0); } }   // (S1e (parallel recording): its R chunks too)
#if defined(DM_NXN1X)
        nvnc_origin_pop(content_origin);
        nvn_trace_gpu_completion(c->fence, 1);
        nvnc_cpu_done(c->content_sub); // copyback, mirror landing and reports finished
#endif
        CRD(46); if (crd) g_rd_n[47] += (uint64_t)c->n;   // replaydiag: after the post ops (command buffer pre-reset, the rest); slot 47 post ops
        pthread_mutex_lock(&g_done_mu); if (c->fence > g_done) __atomic_store_n(&g_done, c->fence, __ATOMIC_RELEASE); pthread_cond_broadcast(&g_done_cv); pthread_mutex_unlock(&g_done_mu);
        vk_f0_end(&completion, 0);
        if (c->f0.first_error_event && (c->f0.first_error_event <= 64 || !(c->f0.first_error_event & (c->f0.first_error_event - 1))))
            VKLOG("NVN2 F0 completion: sub=%llu frame=%llu fence=%llu error_event=%llu first_result=%d first_part=%u accepted_parts=%u accepted_last_part=%u dropped=%d final_buffers=%u gpu_observed=%llu cpu_completed=%llu present_done_ticket=%llu",
                (unsigned long long)c->f0.id, (unsigned long long)c->f0.frame, (unsigned long long)c->fence,
                (unsigned long long)c->f0.first_error_event, c->f0.first_result, c->f0.first_part,
                c->f0.accepted_parts, c->f0.accepted_last_part, c->f0.dropped, c->f0.final_buffers,
                (unsigned long long)atomic_load(&g_f0_gpu_observed), (unsigned long long)vk_fence_done(),
                (unsigned long long)atomic_load(&g_f0_present_done));
        g_f0_sub = NULL;
        free(c->ops); free(c->og); free(c);
    }
    return NULL;
}
static void completion_start(void) { pthread_t t; if (pthread_create(&t, NULL, completion_main, NULL)) die("[vk] completion thread"); pthread_detach(t); }
// s15 (diagnostic, DM_VK_SLOWSUB=<ms>, off by default): a submission over that many ms logs where its time went: its start (and the wait for a free slot), the
// replay of its lists by record op (gfx_vk_exec.c ss_mark), its commit (and vkQueueSubmit with the queue lock, NXN1X's submit_mutex: NXN1X's present holds it
// across vkQueuePresentKHR); this thread's cycles, as ms at the cycles per ns of the normal submissions (a wall time far over it is time the thread did not run),
// the process's page faults, and the replay's own counters (descriptor sets, pipeline keys, new pipelines, texture uploads). First 120 slow ones, then every 40th
extern int g_ss_on; extern uint64_t g_ss_ns[64], g_ss_n[64]; extern const char *const g_op_names[64]; extern uint64_t g_pkg_n, g_pkg_ns;
void vk_ss_counts(uint64_t *cyc, uint64_t *faults);   // gfx_vk_present.c (Windows)
static uint64_t g_ss_thr, g_ss_b0, g_ss_b1, g_ss_v0[8], g_ss_cal_cyc, g_ss_cal_ns, g_ss_nslow, g_ss_nall, g_ss_nnorm, g_ss_norm_ns[64], g_ss_norm_n[64], g_ss_norm_wall;
static void ss_snap(uint64_t v[8]) { vk_ss_counts(&v[0], &v[1]); v[2] = g_vkst.ns_desc; v[3] = g_vkst.ns_pipe; v[4] = g_vkst.pipes_built; v[5] = g_pkg_ns; v[6] = g_pkg_n; v[7] = g_ss_slot_ns; }
static void ss_begin(void) {
    if (g_ss_on < 0) { const char *e = getenv("DM_VK_SLOWSUB"); double ms = e ? atof(e) : 0; g_ss_thr = ms > 0 ? (uint64_t)(ms * 1e6) : 0; g_ss_on = g_ss_thr > 0;
        if (g_ss_on) VKLOG("s15: submissions over %.1f ms are logged with where their time went (DM_VK_SLOWSUB)", ms); }
    if (g_ss_on <= 0) return;
    g_ss_b0 = now_ns(); ss_snap(g_ss_v0); memset(g_ss_ns, 0, sizeof g_ss_ns); memset(g_ss_n, 0, sizeof g_ss_n);
    { extern uint64_t g_ss_dr[8], g_ss_drn[8]; memset(g_ss_dr, 0, sizeof g_ss_dr); memset(g_ss_drn, 0, sizeof g_ss_drn); }   // s18
    g_ss_part_n = g_ss_part_lock = g_ss_part_sub = g_ss_main_lock = 0;   // s19
}
static void ss_begun(void) { if (g_ss_on > 0) g_ss_b1 = now_ns(); }
static void ss_end(uint64_t c0, uint64_t qs) {
    if (g_ss_on <= 0) return;
    uint64_t e = now_ns(), v[8], wall = e - g_ss_b0; ss_snap(v); for (int i = 0; i < 8; i++) v[i] -= g_ss_v0[i];
    g_ss_nall++;
    if (wall < g_ss_thr) {
        if (wall < g_ss_thr / 2) { g_ss_cal_cyc += v[0]; g_ss_cal_ns += wall; }
        for (int i = 0; i < 64; i++) { g_ss_norm_ns[i] += g_ss_ns[i]; g_ss_norm_n[i] += g_ss_n[i]; }
        g_ss_nnorm++; g_ss_norm_wall += wall;
        if (!(g_ss_nall % 1200)) VKLOG("s15: %llu submissions, %llu slow; normal ones %.2f ms each, %.3f cycles per ns", (unsigned long long)g_ss_nall, (unsigned long long)g_ss_nslow,
            g_ss_norm_wall / 1e6 / (double)g_ss_nnorm, g_ss_cal_ns ? (double)g_ss_cal_cyc / (double)g_ss_cal_ns : 0.0);
        return; }
    if (++g_ss_nslow > 120 && g_ss_nslow % 40) return;
    double cpn = g_ss_cal_ns ? (double)g_ss_cal_cyc / (double)g_ss_cal_ns : 0; uint64_t ops = 0; for (int i = 0; i < 64; i++) ops += g_ss_ns[i];
    char top[900]; int o = 0; uint64_t taken = 0;
    for (int k = 0; k < 8; k++) {
        int b = -1; for (int i = 0; i < 64; i++) if (!(taken >> i & 1) && g_ss_ns[i] && (b < 0 || g_ss_ns[i] > g_ss_ns[b])) b = i;
        if (b < 0) break; taken |= 1ull << b; double nn = g_ss_nnorm ? (double)g_ss_nnorm : 1;
        o += snprintf(top + o, sizeof top - (size_t)o, " %s %.2f ms/%llu (normal %.2f/%.0f);", g_op_names[b] ? g_op_names[b] : "?", g_ss_ns[b] / 1e6, (unsigned long long)g_ss_n[b], g_ss_norm_ns[b] / 1e6 / nn, g_ss_norm_n[b] / nn);
        if (o >= (int)sizeof top - 80) break; }
    VKLOG("s15: slow submission %.1f ms (%llu of %llu): start %.2f (slot wait %.2f), lists %.2f (ops %.2f), commit %.2f (vkQueueSubmit with the queue lock %.2f); thread CPU ~%.1f ms (%.1f Mcycles), page faults %llu; descriptor sets %.2f ms, pipeline keys %.2f ms (new pipelines %llu), texture uploads %.2f ms (%llu); ops:%s",
          wall / 1e6, (unsigned long long)g_ss_nslow, (unsigned long long)g_ss_nall, (g_ss_b1 - g_ss_b0) / 1e6, v[7] / 1e6, (c0 - g_ss_b1) / 1e6, ops / 1e6, (e - c0) / 1e6, qs / 1e6,
          cpn > 0 ? v[0] / cpn / 1e6 : -1.0, v[0] / 1e6, (unsigned long long)v[1], v[2] / 1e6, v[3] / 1e6, (unsigned long long)v[4], v[5] / 1e6, (unsigned long long)v[6], top);
    { extern uint64_t g_ss_dr[8], g_ss_drn[8];   // s18: the non-indexed draws by part (gfx_vk_exec.c draw)
        VKLOG("s18: DRAW parts: quad conversion %.2f ms/%llu, prepare_draw %.2f/%llu, command buffer+queries %.2f/%llu, indirect %.2f/%llu, quads %.2f/%llu, plain %.2f/%llu",
              g_ss_dr[0] / 1e6, (unsigned long long)g_ss_drn[0], g_ss_dr[1] / 1e6, (unsigned long long)g_ss_drn[1], g_ss_dr[2] / 1e6, (unsigned long long)g_ss_drn[2],
              g_ss_dr[3] / 1e6, (unsigned long long)g_ss_drn[3], g_ss_dr[4] / 1e6, (unsigned long long)g_ss_drn[4], g_ss_dr[5] / 1e6, (unsigned long long)g_ss_drn[5]); }
    VKLOG("s19: queue lock (NXN1X: Scheduler::submit_mutex): %llu early parts waited %.2f ms for it and took %.2f ms to submit; the final submit waited %.2f ms for it",
          (unsigned long long)g_ss_part_n, g_ss_part_lock / 1e6, g_ss_part_sub / 1e6, g_ss_main_lock / 1e6);
}
uint64_t mtl_sub_commit(void) {
    uint64_t ss_c0 = g_ss_on > 0 ? now_ns() : 0;   // s15
    const int rd = rd_on(); if (rd) vk_rd_start();   // replaydiag: slots 10-16
    vksub_t *s = cur();
    vk_end_pass_land();
    if (g_npar) vk_seg_part_join();   // S1e (parallel recording): the last part's worker segments recorded and their post ops merged (after the replay thread's pass ended: no query of it open) before the rest
    if (g_occ2 >= 2) occ2_resolve(O2R_SUB);   // occ-s2: the submission's last resolve point (its reports land with its fence), before the readback and the end timestamp
    vk_rpt_flush();   // replay2: this submission's replay-time reports, before its completion guards (post ops) and vkQueueSubmit
    if (rd) vk_rd_mark(10);   // replaydiag: the open pass ended (and the reports written: replay2)
    pool_demand_report();
    pool_sync_host_barrier(s, s->main);
    pc_readback(s, s->main);   // poolcopy
    if (g_tsq) vkCmdWriteTimestamp(s->main, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsq, 2 * (uint32_t)g_cur + 1);
    if (rd) vk_rd_mark(11);   // replaydiag: pool demand report, host barrier, readback, end timestamp
    VkResult raw_end = vk_f0_result(vkEndCommandBuffer(s->main), "vkEndCommandBuffer(final-main)", s->nmain_used);
    if (rd) vk_rd_mark(12);   // replaydiag: vkEndCommandBuffer (its s63 wrapper closes a staging window: the merge)
    VkResult er = raw_end; if (s->cb_err && er >= 0) er = VK_ERROR_OUT_OF_HOST_MEMORY;
    const int pc_test = g_pc_test_fail; if (pc_test) { g_pc_test_fail = 0; if (er >= 0) er = VK_ERROR_OUT_OF_HOST_MEMORY; }   // poolcopy self-test (case 9): this submission taken as refused
    if (er < 0) { s->pc_ok = 0; pc_dropped(); }   // poolcopy
    int pc_rec = 0;
    if (er < 0 && pc_async()) {   // stage2c (review P0-6): parts already sent ran (or will): their GPU writes are taken back by a recovery buffer of post copies for the ranges whose
        // pre ran in those parts; the rest (its pre and uploads were in the refused part) are dropped. Every copy is then synced in full once these merges have run (pc_dropped,
        // g_pc_drain): a capture whose upload was refused left its shadow ahead of the device-local copy
        if (s->nmain_sent && s->npce) {
            if (!s->pc_rec) { VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, s->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; if (vkAllocateCommandBuffers(V.dev, &ai, &s->pc_rec)) s->pc_rec = VK_NULL_HANDLE; }
            VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            if (s->pc_rec && !vkBeginCommandBuffer(s->pc_rec, &bi)) { pc_rec = pc_post_record(s, s->pc_rec, s->nmain_sent); if (vkEndCommandBuffer(s->pc_rec)) pc_rec = 0; }
            if (pc_rec) g_pc_as_rec++;
        }
        if (!pc_rec) for (int i = 0; i < s->npce; i++) s->pce[i].drop = 1;
        s->pc_ok = pc_rec; g_pc_drain = g_fence + 1;
    }
    s->f0.dropped = er < 0;   // keep the legacy policy result separate from the actual driver result
    if (s->f0.id && er < 0 && (s->f0.first_error_event <= 64 || !(s->f0.first_error_event & (s->f0.first_error_event - 1))))
        VKLOG("NVN2 F0 drop: sub=%llu frame=%llu error_event=%llu first_site=%s first_result=%d first_part=%u final_end=%d legacy_policy_result=%d accepted_parts=%u accepted_last_part=%u recorded_parts=%d",
            (unsigned long long)s->f0.id, (unsigned long long)s->f0.frame, (unsigned long long)s->f0.first_error_event,
            s->f0.first_site ? s->f0.first_site : "unknown", s->f0.first_result, s->f0.first_part, raw_end, er,
            s->f0.accepted_parts, s->f0.accepted_last_part, s->nmain_used);
    if (er < 0) {   // the driver refused something recorded in it: logged with what the submission held, then it is dropped and the game goes on
        if (g_occ2 >= 2) occ2_dropped();   // occ-s2: its query begins and chunk resets never run: those chunks are reset again before their next use (no waited copy on them)
        if (dm_sdk_ounce() && g_pool_compare_valid && !pc_test) {
            g_pool_compare_valid = 0;   // exact comparisons require an executed host-read barrier
            VKLOG("NVN2 pool sync: differential copies disabled after command-buffer failure; full copies retained until restart");
        }
        extern uint32_t g_sub_ops[64]; char ops[600]; int o = 0;
        for (int i = 0; i < 64 && o < (int)sizeof ops - 16; i++) if (g_sub_ops[i]) o += snprintf(ops + o, sizeof ops - (size_t)o, " %d:%u", i, g_sub_ops[i]);
        static int n; if (n++ < 3) VKLOG("[RED] vkEndCommandBuffer failed (%d): submission dropped; record ops (op:count)%s", er, ops);
        // NVN2 (Switch 2 boots): a memory code (out of host or device memory) with a driver heap at 90% of its budget or more is taken as an allocation failure and not
        // bisected: bisection drops ranges of the game's draws in the following submissions to find a draw the driver refuses. Only this submission is dropped. Adreno's
        // refusal of a too-big submission is a memory code too, so with the heaps under 90% (or no budget reported) it is bisected as before. HWAoI 11-15 (526 s, heap0
        // at 97% of its budget): a part refused at its end, bisection began, and the process stalled and faulted in the driver 12 s later
        int s2mem = 0;
        if (dm_sdk_ounce() && (er == VK_ERROR_OUT_OF_HOST_MEMORY || er == VK_ERROR_OUT_OF_DEVICE_MEMORY)) {
            char heaps[256]; int full = vk_heap_text(heaps, (int)sizeof heaps); s2mem = full > 0;
            static int nm; if (nm++ < 8) VKLOG("[RED] NVN2: the submission was refused with a memory code (%d): %s; driver heaps (usage/budget):%s", er,
                full > 0 ? "a heap is at 90% of its budget or more: dropped, not bisected" : full == 0 ? "the heaps are under 90% of their budget: bisected" : "no budget reported: bisected",
                heaps[0] ? heaps : " (not reported)"); }
        if (!s2mem && !pc_test) vk_bisect_result(0);   // (poolcopy self-test: no bisection of the game's draws)
        if (s->nmain_sent) {   // parts already sent may still run: nothing more is submitted (the submission below only signals its fence)
            // NVN2: the open upload buffer (begun after the last part sent) is ended, so no command buffer of the slot is left recording when it is begun again
            // (vkBeginCommandBuffer on a recording buffer: VUID-vkBeginCommandBuffer-commandBuffer-00049). Its uploads go nowhere, as the rest of the submission
            if (dm_sdk_ounce() && s->up_used) { VkResult ue = vk_f0_result(vkEndCommandBuffer(s->up), "vkEndCommandBuffer(dropped-upload)", s->nmain_used); if (ue < 0) { static int nu; if (nu++ < 4) VKLOG("[RED] NVN2: the dropped submission's upload buffer did not end either (%d)", ue); } }
            s->nmain_sent = s->nmain_used; s->up_used = 0; s->cb_err = 0; s->no_ts = 1; vk_ubd_land_drop();
            if (g_npar) { pthread_mutex_lock(&g_chunk_mu); s->chunk0 = s->nchunk; pthread_mutex_unlock(&g_chunk_mu); }   // S1e (parallel recording): the unsent chunks go with the rest
        } else {
            if (g_npar) { pthread_mutex_lock(&g_chunk_mu); s->chunk0 = s->nchunk; pthread_mutex_unlock(&g_chunk_mu); s->nrcb_used = 0; }   // S1e: the chunks dropped (the pool reset below takes the R chunks)
            vkResetCommandPool(V.dev, s->pool, 0); if (s->up_pool != s->pool) { vk_up_lock(); vkResetCommandPool(V.dev, s->up_pool, 0); vk_up_unlock(); }   /* uppool: the upload buffers' own pool too */ VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            s->main = s->mains[0]; s->nmain_used = 1; s->cb_err = 0; s->no_ts = 1; g_vk_rt_main.cb = s->main;   // no timestamps: the slot's queries still hold an older submission's   // R2 (parallel recording)
            if (s->nup_used > 1) { s->up = s->ups[0]; s->nup_used = 1; }
            VKCHK(nvn2_begin(s->main, &bi, "vkBeginCommandBuffer (main)")); g_vk_rt_main.gen++; s->up_used = 0; vk_ubd_land_drop();   // R2 (parallel recording): was g_cmd_gen++
            VKCHK(vk_f0_result(vkEndCommandBuffer(s->main), "vkEndCommandBuffer(empty-recovery)", s->nmain_used));
        }
    }
    else vk_bisect_result(1);
    { extern uint32_t g_sub_ops[64]; memset(g_sub_ops, 0, sizeof g_sub_ops); }
    if (s->up_used) { void vk_vbm_up_close(VkCommandBuffer); vk_vbm_up_close(s->up); vk_img_barrier_all(s->up); og_up_end(s); VKCHK(vk_f0_result(vkEndCommandBuffer(s->up), "vkEndCommandBuffer(final-upload)", s->nmain_used)); }
    if (rd) vk_rd_mark(13);   // replaydiag: refusal handling, the upload buffer ended
    uint64_t f = ++g_fence;
    if (dm_sdk_ounce()) {
        // Guest-source writes may happen through imported aliases, raw stores,
        // mirrors, or completion operations even when no copied buffer was bound.
        // Do not narrow writes using virtual-address overlap: a distinct guest
        // address may alias the same physical pages. Any buffer write is a guard.
        int post_writes = 0;
        for (int i = 0; i < s->npost; ++i)
            if (s->post[i].kind == 1 || (s->post[i].kind == 2 && (s->post[i].mode != 7 || occ_land_on()))) { post_writes = 1; break; }   // (occkey: mode 7 writes no memory unless occland)
        int needs_completion = s->ngw || post_writes || s->ngml || g_sbufu_w != g_pool_unknown_write_start ||
            g_gmem_sub == g_nsubs || getenv("DM_VK_COPYBACK_FULL");
        if (needs_completion) pool_fence_max(&g_pool_sync_fence, f);
    }
    if (g_ubd_watch && g_ubd_wn > 0) { extern uint64_t g_tex_frame; vk_watch_poll("submit"); g_ubd_wn--; VKLOG("ubd watch: submit fence %llu, present %llu, %d deferred updates open", (unsigned long long)f, (unsigned long long)g_tex_frame, g_nubd); }
    VkCommandBuffer cbs2[2 + MAIN_CBS]; const VkCommandBuffer *cbs = cbs2; uint32_t ncb = 0;
    if (g_npar) { ncb = par_cbs(s, s->up_used ? s->up : VK_NULL_HANDLE, s->nmain_sent < s->nmain_used, pc_rec ? s->pc_rec : VK_NULL_HANDLE); cbs = g_pcb; }   // S1e (parallel recording): the unsent parts' chunks in order, the open R chunk
    else {
    if (s->up_used) cbs2[ncb++] = s->up; for (int i = s->nmain_sent; i < s->nmain_used; i++) cbs2[ncb++] = s->mains[i];
    if (pc_rec) cbs2[ncb++] = s->pc_rec;   // stage2c: a refused submission's recovery (post copies of the parts that ran)
    }
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 0, NULL, 1, &f};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 0, NULL, NULL, ncb, cbs, 1, &g_timeline};
    if (rd) vk_rd_mark(14);   // replaydiag: completion guards, buffers listed
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); vk_queue_lock();
    if (g_ss_on > 0) { struct timespec tl; clock_gettime(CLOCK_MONOTONIC, &tl); g_ss_main_lock = (uint64_t)((tl.tv_sec - t0.tv_sec) * 1000000000ll + (tl.tv_nsec - t0.tv_nsec)); }   // s19
    VkResult r = vkQueueSubmit(V.q, 1, &si, VK_NULL_HANDLE); vk_queue_unlock(); clock_gettime(CLOCK_MONOTONIC, &t1); g_ns_submit += (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec));
    vk_f0_result(r, "vkQueueSubmit(final)", s->nmain_used);
    if (rd) vk_rd_mark(15);   // replaydiag: queue lock and vkQueueSubmit
#if defined(DM_NXN1X)
    if (er >= 0) nvnc_accept(s->content_sub, (unsigned)s->nmain_sent + 1, (unsigned)s->nmain_used, r);
    nvnc_sub_end(); // a signal-only recovery never accepts the missing commands
#endif
    if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] vkQueueSubmit: %d (device lost?)", r); }
    if (er >= 0) nvf0_accepted(&s->f0, s->nmain_used, s->nmain_used - s->nmain_sent);
    s->f0.fence = f; s->f0.final_buffers = ncb;
    if (s->f0.id) f0_max(&g_f0_submitted, f);
    s->fence = f;
    if (s56_any()) s56_seal(f, er < 0);   // s56: this submission's write-backs land when it completes (refused: dropped)
    cpend *c = calloc(1, sizeof *c); c->fence = f; c->frame = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED); c->n = s->npost; c->rslot = g_cur; c->slot = s->no_ts ? -1 : g_cur; s->queries_initialized = !s->no_ts && er >= 0; s->no_ts = 0;
    c->f0 = s->f0;   // independent of ring-slot reuse once CPU completion is published
    c->content_sub = s->content_sub;
    c->npass = g_npass[g_cur]; memcpy(c->pmeta, g_pmeta[g_cur], sizeof(pass_meta) * (size_t)c->npass);
    if (g_og_on && c->slot >= 0 && (c->og = malloc(sizeof(og_gap) * (size_t)(c->npass + 1)))) { memcpy(c->og, g_og_gap[g_cur], sizeof(og_gap) * (size_t)(c->npass + 1)); c->ogm = g_og[g_cur]; }   // ootgap
    g_og_on = 0;
    s67_sub(c->frame);   // s67
    if (s->npost) { c->ops = malloc(s->npost * sizeof *c->ops); memcpy(c->ops, s->post, s->npost * sizeof *c->ops); }
    pthread_mutex_lock(&g_cmu); if (g_ct) g_ct->next = c; else g_ch = c; g_ct = c; pthread_cond_signal(&g_ccv); pthread_mutex_unlock(&g_cmu);
    s->npost = 0; g_cur = -1; g_vk_rt_main.cb = VK_NULL_HANDLE; g_pool_demand_active = 0;   // R2 (parallel recording): no target outside a submission (vk_cmd dies as cur() did)
    g_f0_sub = NULL;
    g_sub_owned=0;
    nvnr_unlock(&g_sub_lock);
    if (rd) vk_rd_mark(16);   // replaydiag: completion record handed to the completion thread, the rest
    ss_end(ss_c0, (uint64_t)((t1.tv_sec - t0.tv_sec) * 1000000000ll + (t1.tv_nsec - t0.tv_nsec)));   // s15
    return f;
}
// s58b: a command buffer of the backend's own (gfx_vk_present.c s58's ring copy), submitted as a host submission: it takes the next fence
// (g_fence), signals the timeline with it, and the completion thread passes it in order, so g_done and every deferred free cover it.
// Outside a submission only (g_sub_lock); fence: an extra VkFence to signal, or VK_NULL_HANDLE
uint64_t vk_submit_extra(VkCommandBuffer cb, VkFence fence) {
    nvnr_lock(&g_sub_lock);
    nvf0_submission diag = f0_submission(), *previous = g_f0_sub; g_f0_sub = &diag;
    uint64_t f = ++g_fence;
    if (dm_sdk_ounce()) pool_fence_max(&g_pool_sync_fence, f);   // external commands have no binding inventory
    VkTimelineSemaphoreSubmitInfo ts = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, NULL, 0, NULL, 1, &f};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, &ts, 0, NULL, NULL, 1, &cb, 1, &g_timeline};
    vk_queue_lock(); VkResult r = vkQueueSubmit(V.q, 1, &si, fence); vk_queue_unlock();
    vk_f0_result(r, "vkQueueSubmit(backend)", 0);
    if (r) { void vk_crumbs_report(void); vk_crumbs_report(); die("[vk] s58b: vkQueueSubmit (backend command buffer): %d (device lost?)", r); }
    diag.fence = f; diag.final_buffers = 1;
    if (diag.id) f0_max(&g_f0_submitted, f);
    cpend *c = calloc(1, sizeof *c); c->fence = f; c->frame = __atomic_load_n(&g_cur_frame, __ATOMIC_RELAXED); c->slot = -1; c->rslot = -1;
    c->f0 = diag;
    pthread_mutex_lock(&g_cmu); if (g_ct) g_ct->next = c; else g_ch = c; g_ct = c; pthread_cond_signal(&g_ccv); pthread_mutex_unlock(&g_cmu);
    g_f0_sub = previous;
    nvnr_unlock(&g_sub_lock);
    return f;
}
uint64_t mtl_last_fence(void) { return g_fence; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    // A poll observes completion (including host writeback) without entering an OS wait.
    if (__atomic_load_n(&g_done, __ATOMIC_ACQUIRE) >= fence) return 1;
    if (!timeout_ns) return 0;
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600ull * 1000000000ull ? 3600ull * 1000000000ull : timeout_ns);
    dl.tv_sec += (time_t)(ns / 1000000000ull); dl.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_done_mu);
    while (g_done < fence) if (pthread_cond_timedwait(&g_done_cv, &g_done_mu, &dl)) break;
    int ok = g_done >= fence; pthread_mutex_unlock(&g_done_mu);
    return ok;
}
void mtl_finish(void) { mtl_wait(g_fence, ~0ull); }
void mtl_drain(void) {
    mtl_finish();
#if defined(DM_NXN1X)
    extern void vk_fg_auto_drain(void);
    vk_fg_auto_drain();
#endif
    vk_pipeline_cache_save();
}

#if defined(DM_NXN1X)
// ---- poolcopy self-test (design\GPU_MEMORY_CACHE_REVIEW.md "검증 순서" 1): NXN1X_NVN_VK_SELFTEST=1 with NXN1X_POOL_SELFTEST=1, at the device handover before any game
// code. It runs the host's own paths (mtl_pool_add, vk_buf_at, mtl_sub_fill / mtl_sub_copy / mtl_sub_event / mtl_sub_counter, mtl_sub_commit, the completion thread) on pools
// of its own and compares the bytes the game would see with the bytes expected. The same cases run in either mode (NXN1X_POOL_MODE=import / copy): the
// expectations are the guest's, so import and copy are both held to them. Each case logs PASS or FAIL with what it found
int nvnh_test_alias_views(size_t n, void **va, void **vb); void nvnh_test_alias_end(void); void *nvnh_test_two_views(size_t n);
void nvn_dirty_add_range(const void *mem, size_t size); void nvn_dirty_remove_range(const void *mem);
static uint8_t *pct_alloc(size_t n) { void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); return p == MAP_FAILED ? NULL : (uint8_t *)p; }
static void pct_pool(uint8_t *m, size_t n) { mtl_pool_add(m, n); nvn_dirty_add_range(m, n); }
static void pct_unpool(uint8_t *m) { nvn_dirty_remove_range(m); mtl_pool_del(m); }
static void pct_cpu(uint8_t *a, uint8_t v, size_t n) { dm_guest_write_prepare(a, n); memset(a, v, n); }   // as a game thread's store (the dirty tracking sees it)
static size_t pct_bad(const uint8_t *a, uint8_t v, size_t n) { for (size_t i = 0; i < n; i++) if (a[i] != v) return i + 1; return 0; }   // 0 = all v, else 1 + the first other
static void pct_rawfill(uint8_t *a, size_t n, uint32_t v) {   // a GPU write the submission's write list does not hold (a shader's store of unknown size, raw memory)
    VkBuffer b; VkDeviceSize o; if (!vk_buf_at((uint64_t)(uintptr_t)a, n, &b, &o)) { VKLOG("[RED] poolcopy self-test: no buffer at %p", (void *)a); return; }
    vk_end_pass_land(); VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb); vkCmdFillBuffer(cb, b, o, n, v); vk_img_barrier_all(cb);
}
static int pct_ok, pct_n;
static void pct_result(int c, const char *what, size_t bad, const uint8_t *at, uint8_t want) {
    pct_n++; if (!bad) pct_ok++;
    if (!bad) VKLOG("poolcopy self-test case %d (%s): PASS", c, what);
    else VKLOG("[RED] poolcopy self-test case %d (%s): FAIL at byte %llu: %#04x, expected %#04x", c, what, (unsigned long long)(bad - 1), at[bad - 1], want);
}
static int pct_vol_state(const void *a) {   // stage2b self-test: the volatile state of the page holding a (0 tracked, 1 volatile, 2 leaving; -1 no pool)
    const uint64_t x = (uint64_t)(uintptr_t)a; int r = -1; pool_rdlock();
    for (int i = 0; i < g_npool; i++) { const vkpool_t *p = &g_pool[i]; if (!p->dl || x < p->lo || x >= p->hi) continue;
        r = p->pv && ((x - p->lo) >> 12) < p->pv->np ? p->pv->st[(x - p->lo) >> 12] : 0; break; }
    pool_unlock(); return r;
}
static void pct_run_mark(void *ctx, uintptr_t a, size_t n) {   // case 12: the 4 KB pages of a run (relative to the test pool Z, its first page passed as ctx's base)
    extern uint8_t *g_pct_z; uint8_t *got = ctx;
    for (uintptr_t x = a & ~(uintptr_t)4095; x < a + n; x += 4096) got[(x - (uintptr_t)g_pct_z) >> 12] = 1;
}
uint8_t *g_pct_z;
static uint64_t pct_go(void) { uint64_t f = mtl_sub_commit(); if (!mtl_wait(f, 10000000000ull)) VKLOG("[RED] poolcopy self-test: submission %llu did not complete in 10 s", (unsigned long long)f); return f; }
static int pct_dl(const void *a) {   // hybrid1 self-test: the chunk holding a: 1 a device-local copy, 0 imported, -1 none made
    const uint64_t x = (uint64_t)(uintptr_t)a; int r = -1; pool_rdlock();
    for (int i = 0; i < g_npool && r < 0; i++) { const vkpool_t *p = &g_pool[i]; if (p->buf && x >= p->lo && x < p->hi) r = p->dl ? 1 : 0; }
    pool_unlock(); return r;
}
static void pct_atomic(const void *a, uint64_t key) { vk_hy_ssbo(vk_pool_of((uint64_t)(uintptr_t)a, 1), 1, 1, key); }   // as eden_ssbo binds a written storage buffer its shader runs an atomic on
// hybrid1 (design\GPU_MEMORY_HYBRID_DESIGN.md 8, H1-H12). Import mode: H1. Hybrid mode: H2-H12 on pools of their own, then A, C, VA and VB are made copies (A and C by a request,
// VA and VB by H4), so the copy cases after this run on device-local copies inside hybrid mode. 1 = memory for the pools not made
static int pct_hybrid_cases(uint8_t *A, uint8_t *C, uint8_t *VA, uint8_t *VB) {
    enum { P = 8u << 20, PG = 4096 };
    if (!pc_hybrid()) {   // H1: import (or copy) mode: the rule never converts, and in import mode no copy is made
        if (vk_poolcopy_on()) return 0;
        mtl_sub_begin(); pct_atomic(A, 0x4831); mtl_sub_fill((uint64_t)(uintptr_t)A, 4, 0x48484848u); pct_go(); mtl_sub_begin(); pct_go();
        pct_result(140, "H1 import mode: an atomic binding converts nothing, no device-local copy is made", pct_dl(A) != 0 || g_pc_shared + g_pc_private ? 1 : pct_bad(A, 0x48, 4), A, pct_dl(A) != 0 ? 0xff : 0x48);
        return 0; }
    uint8_t *HA = pct_alloc(P), *HB = pct_alloc(P), *HR = pct_alloc(P), *HS = pct_alloc(P), *K = C + 40 * PG;
    if (!HA || !HB || !HR || !HS) { VKLOG("[RED] hybrid1 self-test: memory for its pools not made"); return 1; }
    pct_pool(HA, P); pct_pool(HB, P); pct_pool(HR, P); pct_pool(HS, P);
#define HCOPY(src, dst, n) mtl_sub_copy((uint64_t)(uintptr_t)(src), (uint64_t)(uintptr_t)(dst), (n))
#define HFILL(a, n, v) mtl_sub_fill((uint64_t)(uintptr_t)(a), (n), 0x01010101u * (uint32_t)(v))
    pct_cpu(HA, 0, 64); pct_cpu(HB, 0, 64); pct_cpu(HR, 0, 2 * PG); pct_cpu(HS, 0, 64);
    mtl_sub_begin(); HCOPY(HA, K, 4); HCOPY(HB, K, 4); HCOPY(HR, K, 4); HCOPY(HS, K, 4); pct_go();
    pct_result(141, "H0 hybrid mode: chunks start imported", pct_dl(HA) == 0 && pct_dl(HB) == 0 && pct_dl(HR) == 0 && pct_dl(HS) == 0 ? 0 : 1, HA, 0);
    // H5: transfer writes only (fill, copy into it): it stays imported, the GPU's bytes in the game's page at once. A written storage buffer with no atomic: the same
    mtl_sub_begin(); HFILL(HB, 16, 0x55); vk_hy_ssbo(vk_pool_of((uint64_t)(uintptr_t)HB, 1), 1, 0, 0x4835); pct_go(); mtl_sub_begin(); pct_go();
    pct_result(145, "H5 transfer writes and a store-only storage buffer leave the chunk imported", pct_dl(HB) != 0 ? 1 : pct_bad(HB, 0x55, 16), HB, pct_dl(HB) != 0 ? 0xff : 0x55);
    // H2: N writes x on the GPU through the import and binds a storage buffer its shader runs an atomic on; the CPU writes y; N+1: a copy holding x and y
    mtl_sub_begin(); HFILL(HA, 4, 0x21); pct_atomic(HA, 0x4832); pct_go(); pct_cpu(HA + 8, 0x22, 4);
    mtl_sub_begin(); HCOPY(HA, K, 12); pct_go();
    { size_t b1 = pct_bad(K, 0x21, 4), b2 = pct_bad(K + 8, 0x22, 4);
      pct_result(142, "H2 an atomic binding: a device-local copy from the next submission, with the GPU's bytes written through the import and the CPU's", pct_dl(HA) != 1 ? 1 : b1 ? b1 : b2 ? 8 + b2 : 0, K, pct_dl(HA) != 1 ? 0xff : b1 ? 0x21 : 0x22); }
    // H3: N (the atomic binding, a GPU write through the import) still in flight when the CPU writes and N+1 starts: N+1 converts after N and reads both
    { mtl_sub_begin(); for (int i = 0; i < 8; i++) HFILL(C + 64 * PG, 256 * PG, i); HFILL(HS, 4, 0x31); pct_atomic(HS, 0x4833); const uint64_t f = mtl_sub_commit(); pct_cpu(HS + 8, 0x32, 4);
      mtl_sub_begin(); HCOPY(HS, K + 64, 12); pct_go(); (void)f;
      size_t b1 = pct_bad(K + 64, 0x31, 4), b2 = pct_bad(K + 72, 0x32, 4), b3 = pct_bad(HS, 0x31, 4);
      pct_result(143, "H3 converted while the submission that wrote through the import was in flight: both bytes read, the game keeps the GPU's", pct_dl(HS) != 1 ? 1 : b1 ? b1 : b2 ? 8 + b2 : b3, b1 || b2 ? K + 64 : HS, pct_dl(HS) != 1 ? 0xff : b1 || b3 ? 0x31 : 0x32); }
    // H13 (hybrid2): two chunks over different pages of one backing range (well inside one 256 MB block): an atomic on one converts it alone, the other stays imported.
    // VA and VB are taken out for it (whole pools over those pages would join the two), then registered again: X's range is converted, so VA and VB become copies
    { uint8_t *X = VA + 64 * PG, *Y = VB + 512 * PG; pct_unpool(VA); pct_unpool(VB); pct_pool(X, 64 * PG); pct_pool(Y, 64 * PG);
      pct_cpu(X, 0, 16); pct_cpu(Y, 0, 16); mtl_sub_begin(); HCOPY(X, K, 4); HCOPY(Y, K, 4); pct_go();
      mtl_sub_begin(); pct_atomic(X, 0x483d); HFILL(Y, 4, 0x13); pct_go();
      mtl_sub_begin(); HFILL(X, 4, 0x14); HCOPY(X, K + 320, 4); pct_go();
      const int ok = pct_dl(X) == 1 && pct_dl(Y) == 0; size_t b1 = pct_bad(Y, 0x13, 4), b2 = pct_bad(K + 320, 0x14, 4);
      pct_result(154, "H13 an atomic converts only the chunks sharing its pages, not its backing block's others", !ok ? 1 : b1 ? b1 : b2, !ok ? X : b1 ? Y : K + 320, !ok ? 0xff : b1 ? 0x13 : 0x14);
      pct_unpool(X); pct_unpool(Y); pct_pool(VA, P); pct_pool(VB, P); }
    // H4: aliases: an atomic through VA converts VB with it (one backing range); a GPU write through VB is read through VA in the same submission
    mtl_sub_begin(); HCOPY(VA, K, 4); HCOPY(VB, K, 4); pct_go();
    mtl_sub_begin(); pct_atomic(VA, 0x4834); pct_go();
    mtl_sub_begin(); HFILL(VB + 20 * PG, PG, 0x41); HCOPY(VA + 20 * PG, K + 128, PG); pct_go();
    { const int both = pct_dl(VA) == 1 && pct_dl(VB) == 1; size_t b1 = pct_bad(K + 128, 0x41, PG), b2 = pct_bad(VA + 20 * PG, 0x41, PG);
      pct_result(144, "H4 aliases convert together: a GPU write through one read through the other", !both ? 1 : b1 ? b1 : b2, b1 || !both ? K + 128 : VA + 20 * PG, !both ? 0xff : 0x41); }
    // H6: one way: submissions with no atomic leave it a copy, and a CPU write still reaches the GPU
    for (int i = 0; i < 12; i++) { mtl_sub_begin(); pct_go(); }
    pct_cpu(HA + 16, 0x66, 4); mtl_sub_begin(); HCOPY(HA + 16, K + 192, 4); pct_go();
    pct_result(146, "H6 one way: still a copy after 12 submissions with no atomic, a CPU write read", pct_dl(HA) != 1 ? 1 : pct_bad(K + 192, 0x66, 4), K + 192, pct_dl(HA) != 1 ? 0xff : 0x66);
    // H9-H11: raw global memory write logs (the DM_VK_EDEN=0 path's; dmsh_abi.h): page numbers of 32 bits
    if ((uint64_t)(uintptr_t)HR >> 44) VKLOG("hybrid1 self-test: H9-H11 not run (the pool's page numbers do not fit a write log slot)");
    else { static uint32_t lg[(1 << DMSH_GMEM_LOG_LOG2) + 1];
      // H9: a log of plain stores (no atomic): nothing converts, the store in the game's page
      memset(lg, 0, sizeof lg); lg[1] = (uint32_t)((uint64_t)(uintptr_t)HR >> 12);
      mtl_sub_begin(); vk_gml_note2(lg, 0, 0x4839); pct_rawfill(HR, 16, 0x91919191u); pct_go(); mtl_sub_begin(); pct_go();
      pct_result(149, "H9 raw global memory stores (no atomic): the chunk stays imported, the store seen", pct_dl(HR) != 0 ? 1 : pct_bad(HR, 0x91, 16), HR, pct_dl(HR) != 0 ? 0xff : 0x91);
      // H11: an atomic shader's log that ran out of slots: nothing converts (where it wrote is not known), the store seen
      memset(lg, 0, sizeof lg); lg[0] = 1;
      mtl_sub_begin(); vk_gml_note2(lg, 1, 0x483b); pct_rawfill(HR + 32, 16, 0xb1b1b1b1u); pct_go(); mtl_sub_begin(); pct_go();
      pct_result(151, "H11 an atomic log that ran out of slots converts nothing, the store seen", pct_dl(HR) != 0 ? 1 : pct_bad(HR + 32, 0xb1, 16), HR + 32, pct_dl(HR) != 0 ? 0xff : 0xb1);
      // H10: an atomic shader's log listing a page of HR: HR converts, HB (not listed) stays imported, both values seen
      memset(lg, 0, sizeof lg); lg[1] = (uint32_t)((uint64_t)(uintptr_t)(HR + PG) >> 12);
      mtl_sub_begin(); vk_gml_note2(lg, 1, 0x483a); pct_rawfill(HR + PG, 16, 0xa1a1a1a1u); pct_go();
      mtl_sub_begin(); HCOPY(HR + PG, K + 256, 16); pct_go();
      pct_result(150, "H10 an atomic log converts the chunk of the pages it lists, not the others", pct_dl(HR) != 1 || pct_dl(HB) != 0 ? 1 : pct_bad(K + 256, 0xa1, 16), K + 256, pct_dl(HR) != 1 || pct_dl(HB) != 0 ? 0xff : 0xa1); }
    // H12: Eden's raw atomic (the pool table binding, is_atomic): counted with its shader, converts nothing
    { const uint64_t u0 = g_hy_units, r0 = g_hy_raw_unloc; vk_hy_raw_unlocated(0x483c); mtl_sub_begin(); pct_go();
      pct_result(152, "H12 a raw atomic with no write log is counted and converts nothing", g_hy_units != u0 || g_hy_raw_unloc != r0 + 1 ? 1 : 0, HB, 0); }
    // the copy cases below on device-local copies (A and C; VA and VB since H4)
    mtl_sub_begin(); pct_atomic(A, 0x48a0); pct_atomic(C, 0x48a1); pct_go(); mtl_sub_begin(); pct_go();
    pct_result(153, "A and C made device-local copies for the copy cases", pct_dl(A) == 1 && pct_dl(C) == 1 ? 0 : 1, A, 0);
    VKLOG("hybrid1 self-test: %llu units, %llu chunks, %.1f MB converted; storage atomic requests %llu, raw atomic log requests %llu", (unsigned long long)g_hy_units, (unsigned long long)g_hy_chunks,
          g_hy_bytes / 1048576.0, (unsigned long long)g_hy_req[0], (unsigned long long)g_hy_req[1]);
#undef HCOPY
#undef HFILL
    return 0;   // (H pools stay registered: their units are converted for the run)
}
// hybrid2 R1-R6 (every mode; NXN1X_SSBO_WRITES=all: not run): vk_ssbo_bound, as eden_ssbo calls it for each storage buffer binding. In copy and hybrid mode A is a
// device-local copy here (the pre copies are checked there), in import mode imported (only the GPU-write list and the bytes are checked)
static void pct_ssbo_cases(uint8_t *A, uint8_t *C) {
    enum { PG = 4096 };
    if (ssbo_all()) { VKLOG("hybrid2 self-test: R1-R6 not run (NXN1X_SSBO_WRITES=all)"); return; }
    uint8_t *W = A + 40 * PG, *K = C + 48 * PG; const int dl = pct_dl(A) == 1;
    pct_cpu(W, 0, 4 * PG); mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)W, (uint64_t)(uintptr_t)K, 4); pct_go();
    const uint64_t a = (uint64_t)(uintptr_t)W;
    // R1: a dispatch's read-only storage buffer: no pre copy, nothing in the GPU-write list
    mtl_sub_begin(); { const uint64_t p0 = g_pc_pre_bytes; const int w0 = g_ngw; vk_ssbo_bound(5, a, 2 * PG, 0, 0, 0, 0);
      pct_result(160, "R1 a dispatch's read-only storage buffer: no pre copy, no GPU write listed", g_pc_pre_bytes != p0 || g_ngw != w0 ? 1 : 0, W, 0); }
    // R2: written, its stores reaching 256 bytes: a pre copy of those 256 bytes (copies) and one listed write of 256 bytes
    { const uint64_t p0 = g_pc_pre_bytes; vk_ssbo_bound(5, a, 2 * PG, 1, 0, 1, 256);
      const int lw = g_ngw > 0 && g_gw[g_ngw - 1].lo == a && g_gw[g_ngw - 1].hi == a + 256, pre = !dl || g_pc_pre_bytes - p0 == 256;
      pct_result(161, "R2 a written storage buffer: taken as written only as far as its stores reach", !lw || !pre ? 1 : 0, W, 0); }
    // R3: written, reach not known: the whole binding (its pre copy is of the whole binding: no entry holds it yet)
    { const uint64_t p0 = g_pc_pre_bytes; vk_ssbo_bound(5, a, 2 * PG, 1, 0, 0, 0);
      const int lw = g_ngw > 0 && g_gw[g_ngw - 1].lo == a && g_gw[g_ngw - 1].hi == a + 2 * PG, pre = !dl || g_pc_pre_bytes - p0 == 2 * PG;
      pct_result(162, "R3 a written storage buffer with no known reach: the whole binding", !lw || !pre ? 1 : 0, W, 0); }
    // R4: an indirect one (pointer read from memory): as before, whole, even read-only
    { const int w0 = g_ngw; vk_ssbo_bound(5, a + 3 * PG, PG, 0, 1, 0, 0);
      pct_result(163, "R4 an indirect storage buffer is taken as written as before", g_ngw == w0 ? 1 : 0, W, 0); }
    pct_go();
    // R5: a store within the reach comes back to the game (the write registered by the binding alone: the fill itself is not listed)
    mtl_sub_begin(); vk_ssbo_bound(5, a + PG, PG, 1, 0, 1, 64); pct_rawfill(W + PG, 64, 0x2b2b2b2bu); pct_go();
    pct_result(164, "R5 a dispatch's store inside its reach reaches the game's page", pct_bad(W + PG, 0x2b, 64), W + PG, 0x2b);
    // R6: a read-only binding does not stop a CPU write from reaching the GPU (the copy's sync uploads it whatever binds the page)
    pct_cpu(W + 2 * PG, 0x6c, 8); mtl_sub_begin(); vk_ssbo_bound(5, a + 2 * PG, PG, 0, 0, 0, 0); mtl_sub_copy(a + 2 * PG, (uint64_t)(uintptr_t)(K + 64), 8); pct_go();
    pct_result(165, "R6 a CPU write under a read-only binding reaches the GPU", pct_bad(K + 64, 0x6c, 8), K + 64, 0x6c);
}
int vk_poolcopy_selftest(void) {
    enum { P = 8u << 20, PG = 4096 };
    pct_ok = pct_n = 0;
    VKLOG("poolcopy self-test: begins, pools %s", pc_hybrid() ? "in hybrid mode (imported; units an atomic reaches become device-local copies)" : vk_poolcopy_on() ? "in copy mode (device-local copies)" : "in import mode");
    uint8_t *A = pct_alloc(P), *C = pct_alloc(P), *Z = pct_alloc(64u << 20), *M = nvnh_test_two_views(P), *D = pct_alloc(P); void *va = NULL, *vb = NULL;   // M: two views side by side
    if (!A || !C || !Z || !M || !D || nvnh_test_alias_views(P, &va, &vb)) { VKLOG("[RED] poolcopy self-test: memory for its pools not made"); return -1; }
    uint8_t *VA = va, *VB = vb;
    pct_pool(A, P); pct_pool(C, P); pct_pool(Z, 64u << 20); pct_pool(M, 2 * P); pct_pool(D, P); pct_pool(VA, P); pct_pool(VB, P);   // (M: one pool, a chunk per view)
    if (pct_hybrid_cases(A, C, VA, VB)) return -1;   // hybrid1 H1-H12
    pct_ssbo_cases(A, C);   // hybrid2 R1-R6 (every mode)
    { int vk_cbar_case(int, const char **); const char *w = NULL; char nm[128];   // cbar1: which guest barriers between dispatches are left out (gfx_vk_exec.c)
      for (int k = 0; vk_cbar_case(k, &w) >= 0; k++) { snprintf(nm, sizeof nm, "cbar1 guest barrier lookahead: %s", w); pct_result(170 + k, nm, vk_cbar_case(k, &w) ? 0 : 1, A, 0); }
      for (int k = 0; ; k++) { const int r = bpend_case(k, &w); if (r < 0) break; snprintf(nm, sizeof nm, "cbar2 barriers held and merged: %s", w); pct_result(180 + k, nm, r ? 0 : 1, A, 0); }
      // stage1 (atomic staging): the binding rules (every mode), then the GPU path on pool D, imported in import and hybrid mode (copy mode: a copy, not run)
      int vk_hy3_case(int, const char **, uint64_t, uint64_t, int); int vk_hy3_gpu_case(uint64_t, uint64_t, int);
      mtl_sub_begin(); for (int k = 0; ; k++) { const int r = vk_hy3_case(k, &w, 0x100000000ull, 32u << 20, 1); if (r < 0) break; snprintf(nm, sizeof nm, "stage1 binding rule: %s", w); pct_result(190 + k, nm, r ? 0 : 1, A, 0); } pct_go();
      if (vk_poolcopy_on() != 1) { enum { PG = 4096 };
        pct_cpu(D, 0, 32); mtl_sub_begin(); const int g1 = vk_hy3_gpu_case((uint64_t)(uintptr_t)D, 256, 0); pct_go();
        if (g1 < 0) VKLOG("stage1 self-test: GPU cases not run (no imported chunk or no merge pipeline)");
        else { pct_result(198, "stage1 GPU path: the bytes the dispatch changed in staging reach the game's page, the others stay", !g1 ? 1 : pct_bad(D, 0x5a, 16) ? pct_bad(D, 0x5a, 16) : pct_bad(D + 16, 0, 16) ? 16 + pct_bad(D + 16, 0, 16) : 0, D, 0x5a);
          pct_cpu(D + PG, 0, 8); mtl_sub_begin(); const int g2 = vk_hy3_gpu_case((uint64_t)(uintptr_t)(D + PG), 256, 1); pct_go();
          pct_result(199, "stage1 GPU path: GPU byte 0 and a byte 1 written meanwhile in the same word both stay", !g2 ? 1 : D[PG] != 0xc3 ? 1 : D[PG + 1] != 0xd4 ? 2 : 0, D + PG, D[PG] != 0xc3 ? 0xc3 : 0xd4);
          int vk_hy3_win_case(int, const char **, uint8_t *, uint8_t *);   // stage2s: windows across dispatches and their closes
          for (int k = 0; ; k++) { const int r = vk_hy3_win_case(k, &w, D, C + 56 * PG); if (r < 0) break; snprintf(nm, sizeof nm, "stage2s window: %s", w); pct_result(210 + k, nm, r ? 0 : 1, D, 0); } } } }
    // hybrid1 H1-H12 (import mode: H1); in hybrid mode A, C, VA, VB are device-local copies after it, so the cases below run the copy path
    // 0: a CPU write reaches the GPU; a GPU write reaches the CPU
    pct_cpu(A, 0x5a, PG); mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)A, (uint64_t)(uintptr_t)C, PG); pct_go();
    pct_result(0, "CPU write then GPU copy", pct_bad(C, 0x5a, PG), C, 0x5a);
    mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)(A + PG), PG, 0x11111111u); pct_go();
    pct_result(0, "GPU fill seen by the CPU", pct_bad(A + PG, 0x11, PG), A + PG, 0x11);
    // 1: a write of unknown size (a storage buffer named in uniform memory: counted, not listed)
    mtl_sub_begin(); { extern uint64_t g_sbufu_w; g_sbufu_w++; vk_sub_wrote((uint64_t)(uintptr_t)(A + 2 * PG), P - 2 * PG); }   // (as gfx_vk_exec.c binds one: all it may reach, to its chunk's end, taken as written)
    pct_rawfill(A + 2 * PG, PG, 0x21212121u); pct_go();
    pct_result(1, "unknown-size storage write", pct_bad(A + 2 * PG, 0x21, PG), A + 2 * PG, 0x21);
    // 2: raw global memory with a write log that ran out of slots
    { static uint32_t lg[(1 << DMSH_GMEM_LOG_LOG2) + 1]; memset(lg, 0, sizeof lg); lg[0] = 1;
      mtl_sub_begin(); vk_gml_note(lg); pct_rawfill(A + 3 * PG, PG, 0x23232323u); pct_go(); }
    pct_result(2, "raw memory store, log full", pct_bad(A + 3 * PG, 0x23, PG), A + 3 * PG, 0x23);
    // 3: one page: the GPU writes bytes 0-3 while the CPU writes bytes 4-7 before the submission completes (slow GPU work first)
    { mtl_sub_begin(); for (int i = 0; i < 16; i++) mtl_sub_fill((uint64_t)(uintptr_t)Z, 64u << 20, 0x01010101u * (uint32_t)i);
      mtl_sub_fill((uint64_t)(uintptr_t)(A + 4 * PG), 4, 0x31313131u); uint64_t f = mtl_sub_commit();
      pct_cpu(A + 4 * PG + 4, 0x44, 4); mtl_wait(f, 10000000000ull);
      size_t b1 = pct_bad(A + 4 * PG, 0x31, 4), b2 = pct_bad(A + 4 * PG + 4, 0x44, 4);
      pct_result(3, "same page, GPU bytes 0-3 and CPU bytes 4-7", b1 ? b1 : b2 ? 4 + b2 : 0, A + 4 * PG, b1 ? 0x31 : 0x44); }
    // 4: two pools over the same pages (aliases): a GPU write through one, read by the GPU through the other in the same submission
    mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)VA, PG, 0x41414141u); mtl_sub_copy((uint64_t)(uintptr_t)VB, (uint64_t)(uintptr_t)(C + PG), PG); pct_go();
    { size_t b1 = pct_bad(C + PG, 0x41, PG), b2 = pct_bad(VB, 0x41, PG);
      pct_result(4, "alias: GPU write through one pool, GPU read through the other", b1 ? b1 : b2, b1 ? C + PG : VB, 0x41); }
    // 5: one pool over two mapped views: a range across the views' boundary (a bridge): GPU write across it, then GPU reads of it through the bridge and through the first view's chunk
    mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)(M + P - 2048), 4096, 0x51515151u); mtl_sub_copy((uint64_t)(uintptr_t)(M + P - 2048), (uint64_t)(uintptr_t)(C + 2 * PG), 4096); pct_go();
    mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)(M + P - 2048), (uint64_t)(uintptr_t)(C + 4 * PG), 2048); pct_go();
    { size_t b1 = pct_bad(C + 2 * PG, 0x51, 4096), b2 = pct_bad(M + P - 2048, 0x51, 4096), b3 = pct_bad(C + 4 * PG, 0x51, 2048);
      pct_result(5, "across two views of one pool (bridge), then through the first view's chunk", b1 ? b1 : b2 ? b2 : b3, b1 ? C + 2 * PG : b2 ? M + P - 2048 : C + 4 * PG, 0x51); }
    // 6: a pool deleted and registered again at the same address: the GPU reads what the CPU wrote between, not the old copy
    mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)D, PG, 0xaaaaaaaau); pct_go();
    pct_unpool(D); memset(D, 0x33, PG); pct_pool(D, P);
    mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)D, (uint64_t)(uintptr_t)(C + 5 * PG), PG); pct_go();
    pct_result(6, "pool registered again at the same address", pct_bad(C + 5 * PG, 0x33, PG), C + 5 * PG, 0x33);
    // 7: an event the CPU sees set, then the payload the GPU wrote before it (no fence wait)
    { volatile uint32_t *ev = (volatile uint32_t *)(A + 6 * PG); pct_cpu(A + 6 * PG, 0, 4);
      mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)(A + 7 * PG), PG, 0x71717171u); mtl_sub_event(ev, 0, 1); mtl_sub_commit();
      uint64_t t0 = now_ns(); while (*ev != 1 && now_ns() - t0 < 10000000000ull) ;
      pct_result(7, "payload read right after its event is seen", *ev != 1 ? 1 : pct_bad(A + 7 * PG, 0x71, PG), *ev != 1 ? (const uint8_t *)ev : A + 7 * PG, *ev != 1 ? 1 : 0x71); mtl_finish(); }
    // 8: a counter report and a buffer write to the same bytes: the later command wins, in both orders
    { uint8_t *X = A + 8 * PG, *Y = A + 8 * PG + 64;
      mtl_sub_begin(); mtl_sub_fill((uint64_t)(uintptr_t)X, 16, 0x81818181u); mtl_sub_counter((uint64_t)(uintptr_t)X, 0); pct_go();
      pct_result(8, "fill then counter report: the report stays", pct_bad(X + 8, 0x81, 8) ? 0 : 9, X + 8, 0);
      mtl_sub_begin(); mtl_sub_counter((uint64_t)(uintptr_t)Y, 0); mtl_sub_fill((uint64_t)(uintptr_t)Y, 16, 0x82828282u); pct_go();
      pct_result(8, "counter report then fill: the fill stays", pct_bad(Y, 0x82, 16), Y, 0x82); }
    // 9: a submission refused at its end: none of its GPU writes reach the game, and the next submission reads the CPU's bytes
    { uint8_t *F = A + 9 * PG, *H = A + 10 * PG; pct_cpu(F, 0x91, PG);
      mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)F, (uint64_t)(uintptr_t)(C + 6 * PG), PG); mtl_sub_fill((uint64_t)(uintptr_t)H, PG, 0x92929292u); g_pc_test_fail = 1; pct_go();
      mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)F, (uint64_t)(uintptr_t)(C + 7 * PG), PG); pct_go();
      size_t b1 = pct_bad(C + 6 * PG, 0, PG), b2 = pct_bad(H, 0, PG), b3 = pct_bad(C + 7 * PG, 0x91, PG);
      pct_result(9, "refused submission: its writes dropped, the next one reads the CPU's bytes", b1 ? b1 : b2 ? b2 : b3, b1 ? C + 6 * PG : b2 ? H : C + 7 * PG, b1 || b2 ? 0 : 0x91); }
    // 10 (stage2a): a write as Memory::Write32 now makes it: the page is marked first (dm_guest_write_prepare, as NvnLayer::BeforeGuestWrite), then stored
    // through NXN1X's backing view (an alias of the pool's pages the write watch never protects): the GPU must read it
    { uint8_t *bk = (uint8_t *)nvnh_import_alias(VA + 2 * PG, PG);
      if (!bk) pct_result(10, "write through the backing view (Memory::Write32 path)", 1, VA + 2 * PG, 0x0a);
      else { dm_guest_write_prepare(VA + 2 * PG, 4); *(volatile uint32_t *)bk = 0x0a0a0a0au;
        mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)(VA + 2 * PG), (uint64_t)(uintptr_t)(C + 8 * PG), 4); pct_go();
        pct_result(10, "write through the backing view (Memory::Write32 path)", pct_bad(C + 8 * PG, 0x0a, 4), C + 8 * PG, 0x0a); } }
    // 11 (stage2a): exclusive stores through the backing view: one that succeeds is marked after it (as WriteExclusive*), one that fails writes nothing
    { uint8_t *bk = (uint8_t *)nvnh_import_alias(VA + 3 * PG, PG);
      if (!bk) pct_result(11, "exclusive store through the backing view", 1, VA + 3 * PG, 0x0b);
      else { uint32_t expect = 0; int ok = __atomic_compare_exchange_n((uint32_t *)bk, &expect, 0x0b0b0b0bu, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        if (ok) dm_guest_write_prepare(VA + 3 * PG, 4);
        expect = 0x12345678u; int bad = __atomic_compare_exchange_n((uint32_t *)(bk + 4), &expect, 0x99999999u, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);   // fails: the word is 0
        mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)(VA + 3 * PG), (uint64_t)(uintptr_t)(C + 9 * PG), 8); pct_go();
        size_t b1 = ok ? pct_bad(C + 9 * PG, 0x0b, 4) : 1, b2 = bad ? 5 : pct_bad(C + 9 * PG + 4, 0, 4);
        pct_result(11, "exclusive store through the backing view: the succeeded one seen, the failed one wrote nothing", b1 ? b1 : b2 ? 4 + b2 : 0, C + 9 * PG, b1 ? 0x0b : 0); } }
    // 12 (stage2a2): the dirty runs from the group skip (nvn_dirty_each_since) are exactly the pages the per-page check (nvn_dirty_since) finds: CPU writes at group
    // edges (pages 0, 63, 64, 4095, 4096, 4097, ...) and at the pool's last page of a 64 MB pool; then the sync carries one of them to the GPU
    { enum { NP = (64u << 20) / PG };
      static const uint32_t pgs[] = {0, 1, 63, 64, 65, 127, 4095, 4096, 4097, 8191, 8192, 12345, NP - 64, NP - 2, NP - 1};
      static uint8_t want[NP], got[NP];
      nvn_dirty_protect(Z, 64u << 20); const uint64_t s = nvn_dirty_now();
      for (size_t k = 0; k < sizeof pgs / sizeof *pgs; k++) pct_cpu(Z + (size_t)pgs[k] * PG, (uint8_t)(0xc0 + k), 8);
      memset(got, 0, sizeof got); g_pct_z = Z;
      uint64_t looked = 0; nvn_dirty_each_since(NULL, Z, 64u << 20, s, pct_run_mark, got, &looked);
      size_t bad = 0; for (uint32_t i = 0; i < NP && !bad; i++) { want[i] = (uint8_t)nvn_dirty_since(Z + (size_t)i * PG, PG, s); if (want[i] != got[i]) bad = (size_t)i * PG + 1; }
      VKLOG("poolcopy self-test case 12: %llu page generations read for %u pages (%zu written)", (unsigned long long)looked, (unsigned)NP, sizeof pgs / sizeof *pgs);
      pct_result(12, "dirty runs by group skip equal the per-page check", bad, Z, bad ? (uint8_t)nvn_dirty_since(Z + (bad - 1), PG, s) : 0);
      mtl_sub_begin(); mtl_sub_copy((uint64_t)(uintptr_t)(Z + (size_t)4097 * PG), (uint64_t)(uintptr_t)(C + 10 * PG), 8); pct_go();
      pct_result(12, "a page written at a group edge reaches the GPU", pct_bad(C + 10 * PG, 0xc0 + 8, 8), C + 10 * PG, 0xc0 + 8); }
    // stage2c T1-T14 (design 8.4, review 12): several submissions in flight at once. The completion thread holds its merges while g_pc_hold is set (async and
    // naive modes; serial waits for each previous submission, so there the same steps run one after another). Expected values are what one-at-a-time execution gives
    { const int hold_ok = vk_poolcopy_on() && pc_sync() != PC_SERIAL;
#define HOLD(on) do { if (hold_ok) atomic_store(&g_pc_hold, (on)); } while (0)
#define FILL(a, n, v) mtl_sub_fill((uint64_t)(uintptr_t)(a), (n), 0x01010101u * (uint32_t)(v))
#define COPY(src, dst, n) mtl_sub_copy((uint64_t)(uintptr_t)(src), (uint64_t)(uintptr_t)(dst), (n))
#define WAITF(f) do { if (!mtl_wait((f), 10000000000ull)) VKLOG("[RED] stage2c self-test: submission %llu did not complete in 10 s", (unsigned long long)(f)); } while (0)
      uint8_t *T = A + 16 * PG, *K = C + 16 * PG; uint64_t f1, f2;
      // T1 (x, y): N writes x=1 on the GPU, the CPU writes y=7, N+1 writes x=2, the CPU writes y=8, merges late: the game ends with (2, 8)
      pct_cpu(T, 0, 8); mtl_sub_begin(); pct_go();
      HOLD(1); mtl_sub_begin(); FILL(T, 4, 1); f1 = mtl_sub_commit(); pct_cpu(T + 4, 7, 4);
      mtl_sub_begin(); FILL(T, 4, 2); f2 = mtl_sub_commit(); pct_cpu(T + 4, 8, 4); HOLD(0); WAITF(f2); (void)f1;
      { size_t b1 = pct_bad(T, 2, 4), b2 = pct_bad(T + 4, 8, 4); pct_result(101, "T1 late merges of two submissions: x from the GPU (2), y from the CPU (8)", b1 ? b1 : b2 ? 4 + b2 : 0, T, b1 ? 2 : 8); }
      // T2: N+2's sync after those merges must not put the copy's x back: the GPU reads (2, 8)
      mtl_sub_begin(); COPY(T, K, 8); pct_go();
      { size_t b1 = pct_bad(K, 2, 4), b2 = pct_bad(K + 4, 8, 4); pct_result(102, "T2 the next upload does not rewind the copy", b1 ? b1 : b2 ? 4 + b2 : 0, K, b1 ? 2 : 8); }
      // T3: no CPU write in N; the CPU's first write lands while N is in flight; N+1 reads both
      uint8_t *Y = A + 17 * PG, *KY = C + 17 * PG; pct_cpu(Y, 0, 16); mtl_sub_begin(); pct_go();
      HOLD(1); mtl_sub_begin(); FILL(Y, 4, 0x31); f1 = mtl_sub_commit(); pct_cpu(Y + 8, 0x44, 4);
      mtl_sub_begin(); COPY(Y, KY, 16); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
      { size_t b1 = pct_bad(KY, 0x31, 4), b2 = pct_bad(KY + 8, 0x44, 4), b3 = pct_bad(Y, 0x31, 4), b4 = pct_bad(Y + 8, 0x44, 4);
        pct_result(103, "T3 the CPU's first write while N is in flight: N+1 reads the GPU's and the CPU's bytes, the game keeps both", b1 ? b1 : b2 ? 8 + b2 : b3 ? b3 : b4 ? 8 + b4 : 0, b1 || b2 ? KY : Y, b1 || b3 ? 0x31 : 0x44); }
      // T4: A -> B -> A before a sync (no change), then the same value written again while N writes beside it
      uint8_t *Z4 = A + 18 * PG, *K4 = C + 18 * PG; pct_cpu(Z4, 0x10, 4); pct_cpu(Z4 + 16, 0, 4); mtl_sub_begin(); pct_go();
      pct_cpu(Z4, 0x20, 4); pct_cpu(Z4, 0x10, 4);
      HOLD(1); mtl_sub_begin(); COPY(Z4, K4, 4); FILL(Z4 + 16, 4, 0x99); f1 = mtl_sub_commit(); pct_cpu(Z4, 0x10, 4); HOLD(0); WAITF(f1);
      mtl_sub_begin(); COPY(Z4, K4 + 64, 20); pct_go();
      { size_t b1 = pct_bad(K4, 0x10, 4), b2 = pct_bad(Z4 + 16, 0x99, 4), b3 = pct_bad(K4 + 64, 0x10, 4), b4 = pct_bad(K4 + 80, 0x99, 4);
        pct_result(104, "T4 A->B->A and the same value written again", b1 ? b1 : b2 ? b2 : b3 ? b3 : b4, b1 ? K4 : b2 ? Z4 + 16 : b3 ? K4 + 64 : K4 + 80, b1 || b3 ? 0x10 : 0x99); }
      // T5: a CPU write between N's capture and N+1's: N+1 reads it and N's GPU byte beside it
      uint8_t *W = A + 19 * PG, *KW = C + 19 * PG; pct_cpu(W, 0, 8); mtl_sub_begin(); pct_go();
      HOLD(1); mtl_sub_begin(); FILL(W, 4, 1); f1 = mtl_sub_commit(); pct_cpu(W + 4, 5, 4);
      mtl_sub_begin(); COPY(W, KW, 8); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
      { size_t b1 = pct_bad(KW, 1, 4), b2 = pct_bad(KW + 4, 5, 4), b3 = pct_bad(W, 1, 4);
        pct_result(105, "T5 a CPU write between two captures", b1 ? b1 : b2 ? 4 + b2 : b3, b1 || b2 ? KW : W, b1 || b3 ? 1 : 5); }
      // T6: two addresses of the same pages (VA, VB), a first CPU write through each while N is in flight
      uint8_t *V6 = VA + 8 * PG, *U6 = VB + 8 * PG, *K6 = C + 20 * PG; pct_cpu(V6, 0, 8); mtl_sub_begin(); pct_go();
      HOLD(1); mtl_sub_begin(); FILL(V6 + 64, 4, 0x60); f1 = mtl_sub_commit(); pct_cpu(V6, 0x61, 4); pct_cpu(U6 + 4, 0x62, 4);
      mtl_sub_begin(); COPY(V6, K6, 8); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
      { size_t b1 = pct_bad(K6, 0x61, 4), b2 = pct_bad(K6 + 4, 0x62, 4);
        pct_result(106, "T6 first CPU writes through two addresses of the same pages", b1 ? b1 : b2 ? 4 + b2 : 0, K6, b1 ? 0x61 : 0x62); }
      // T7: a host pre-write, a sync, then the store: the next submission reads the stored value
      uint8_t *Q = A + 20 * PG, *KQ = C + 21 * PG; pct_cpu(Q, 0, 4); mtl_sub_begin(); pct_go();
      // (the store as a guest store reaches it: the page, protected again by the sync, faults, and the write watch's fault path releases it first. At self-test time
      // NXN1X's exception handler is not installed yet, so that path is called here as NvnLayer::OnWatchedWrite calls it)
      dm_guest_write_prepare(Q, 4); mtl_sub_begin(); pct_go();
      { nvnh_fault_fn h = nvnh_fault_hook(); if (h) h(Q, 1, NULL); } *(volatile uint32_t *)Q = 0x77777777u;
      mtl_sub_begin(); COPY(Q, KQ, 4); pct_go();
      pct_result(107, "T7 a sync between the pre-write and the store", pct_bad(KQ, 0x77, 4), KQ, 0x77);
      // T8: merges only (the GPU writes the same bytes three times, no CPU write): the game and the next read have the last
      uint8_t *R8 = A + 21 * PG, *K8 = C + 22 * PG; pct_cpu(R8, 0, 4); mtl_sub_begin(); pct_go();
      HOLD(1); for (int i = 0; i < 3; i++) { mtl_sub_begin(); FILL(R8, 4, 0x81 + i); f1 = mtl_sub_commit(); } HOLD(0); WAITF(f1);
      mtl_sub_begin(); COPY(R8, K8, 4); pct_go();
      { size_t b1 = pct_bad(R8, 0x83, 4), b2 = pct_bad(K8, 0x83, 4); pct_result(108, "T8 three GPU writes in a row, merged late", b1 ? b1 : b2, b1 ? R8 : K8, 0x83); }
      // T9: N's GPU write and a later CPU write of the same bytes (N+1's upload of the same copy range): the CPU's is the latest
      uint8_t *S9 = A + 22 * PG, *K9 = C + 23 * PG; pct_cpu(S9, 0, 4); mtl_sub_begin(); pct_go();
      HOLD(1); mtl_sub_begin(); FILL(S9, 4, 0x91); f1 = mtl_sub_commit(); pct_cpu(S9, 0x92, 4);
      mtl_sub_begin(); COPY(S9, K9, 4); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
      { size_t b1 = pct_bad(K9, 0x92, 4), b2 = pct_bad(S9, 0x92, 4); pct_result(109, "T9 the CPU writes after N's GPU write of the same bytes: the CPU's stays", b1 ? b1 : b2, b1 ? K9 : S9, 0x92); }
      // T10: more submissions than slots, none waited for: every slot is reused only after its merge (no hold here: a reuse waits for it)
      // replay2: the GPU fills start 8 * NSLOT bytes in (slot12 made NSLOT 12: 2 * NSLOT CPU words of 4 bytes ran past 64 into the fills' first words: case 110 failed on its own layout)
      uint8_t *T10 = A + 23 * PG; pct_cpu(T10, 0, 16 * NSLOT); mtl_sub_begin(); pct_go();
      for (int i = 0; i < 2 * NSLOT; i++) { pct_cpu(T10 + 4 * i, (uint8_t)(1 + i), 4); mtl_sub_begin(); FILL(T10 + 8 * NSLOT + 4 * i, 4, 0xa0 + i); f1 = mtl_sub_commit(); }
      WAITF(f1);
      { size_t bad = 0; for (int i = 0; i < 2 * NSLOT && !bad; i++) { if (pct_bad(T10 + 4 * i, (uint8_t)(1 + i), 4)) bad = 4 * i + 1; else if (pct_bad(T10 + 8 * NSLOT + 4 * i, (uint8_t)(0xa0 + i), 4)) bad = 8 * NSLOT + 4 * i + 1; }
        pct_result(110, "T10 more submissions in flight than slots", bad, T10, bad ? (bad - 1 < 8 * NSLOT ? (uint8_t)(1 + (bad - 1) / 4) : (uint8_t)(0xa0 + (bad - 1 - 8 * NSLOT) / 4)) : 0); }
      // T11: a long submission whose first part was sent and whose last part is refused: the first part's GPU write reaches the game, the second's does not
      uint8_t *P11 = A + 24 * PG, *K11 = C + 24 * PG; pct_cpu(P11, 0, 12); mtl_sub_begin(); pct_go();
      mtl_sub_begin(); FILL(P11, 4, 0xb1); vk_cmd_split(); FILL(P11 + 4, 4, 0xb2); g_pc_test_fail = 1; pct_go();
      { size_t b1 = pct_bad(P11, 0xb1, 4), b2 = pct_bad(P11 + 4, 0, 4); pct_result(111, "T11 the sent part's GPU write is kept, the refused part's is not", b1 ? b1 : b2 ? 4 + b2 : 0, P11, b1 ? 0xb1 : 0); }
      // T12: a CPU write while the refusal is being recovered: the next submission reads the GPU's, the refused part's (none) and the CPU's bytes
      pct_cpu(P11 + 8, 0xc1, 4); mtl_sub_begin(); COPY(P11, K11, 12); pct_go();
      { size_t b1 = pct_bad(K11, 0xb1, 4), b2 = pct_bad(K11 + 4, 0, 4), b3 = pct_bad(K11 + 8, 0xc1, 4);
        pct_result(112, "T12 a CPU write during the recovery", b1 ? b1 : b2 ? 4 + b2 : b3 ? 8 + b3 : 0, K11, b1 ? 0xb1 : b2 ? 0 : 0xc1); }
      // T13: a Memory::Write32-path write (marked, then stored through the backing view) while N is in flight
      { uint8_t *V13 = VA + 9 * PG, *K13 = C + 25 * PG, *bk = (uint8_t *)nvnh_import_alias(V13, PG);
        if (!bk) pct_result(113, "T13 Write32 path in flight", 1, V13, 0xd2);
        else { pct_cpu(V13, 0, 8); mtl_sub_begin(); pct_go();
          HOLD(1); mtl_sub_begin(); FILL(V13, 4, 0xd1); f1 = mtl_sub_commit(); dm_guest_write_prepare(V13 + 4, 4); *(volatile uint32_t *)(bk + 4) = 0xd2d2d2d2u;
          mtl_sub_begin(); COPY(V13, K13, 8); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
          size_t b1 = pct_bad(K13, 0xd1, 4), b2 = pct_bad(K13 + 4, 0xd2, 4); pct_result(113, "T13 a Write32-path write while N is in flight", b1 ? b1 : b2 ? 4 + b2 : 0, K13, b1 ? 0xd1 : 0xd2); } }
      // T14: a failed exclusive store (writes nothing) and a succeeded one while N is in flight
      { uint8_t *V14 = VA + 10 * PG, *K14 = C + 26 * PG, *bk = (uint8_t *)nvnh_import_alias(V14, PG);
        if (!bk) pct_result(114, "T14 exclusive stores in flight", 1, V14, 0xe2);
        else { pct_cpu(V14, 0, 8); mtl_sub_begin(); pct_go();
          HOLD(1); mtl_sub_begin(); FILL(V14, 4, 0xe1); f1 = mtl_sub_commit();
          uint32_t ex = 0x12345678u; int failed = !__atomic_compare_exchange_n((uint32_t *)bk, &ex, 0x99999999u, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
          ex = 0; int ok = __atomic_compare_exchange_n((uint32_t *)(bk + 4), &ex, 0xe2e2e2e2u, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); if (ok) dm_guest_write_prepare(V14 + 4, 4);
          mtl_sub_begin(); COPY(V14, K14, 8); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
          size_t b1 = !failed ? 1 : pct_bad(K14, 0xe1, 4), b2 = !ok ? 1 : pct_bad(K14 + 4, 0xe2, 4), b3 = pct_bad(V14, 0xe1, 4);
          pct_result(114, "T14 a failed and a succeeded exclusive store while N is in flight", b1 ? b1 : b2 ? 4 + b2 : b3, b1 || b2 ? K14 : V14, b1 || b3 ? 0xe1 : 0xe2); } }
      // T15 (a known limit, reported, not counted): N's GPU work writes 1 into the copy, not merged yet; the CPU then writes 0, the value the shadow and the game's
      // page already hold, before N+1's capture. The byte compare cannot see that write (game == shadow), so N's merge (game still == pre) puts 1 in: the GPU's
      // value wins and the CPU's write of the old value is lost. On hardware the CPU's later write would stand
      { uint8_t *H = A + 25 * PG, *KH = C + 27 * PG; pct_cpu(H, 0, 4); mtl_sub_begin(); pct_go();
        HOLD(1); mtl_sub_begin(); FILL(H, 4, 1); f1 = mtl_sub_commit(); pct_cpu(H, 0, 4);
        mtl_sub_begin(); COPY(H, KH, 4); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
        VKLOG("poolcopy self-test T15 (known limit, not counted): the CPU wrote back the old value 0 while N's GPU write of 1 was in flight: the game holds %#x, N+1 read %#x "
              "(%s)", H[0], KH[0], H[0] == 1 && KH[0] == 1 ? "the GPU's value won: the limit as described" : H[0] == 0 && KH[0] == 0 ? "the CPU's value stood" : "mixed"); }
      // T16 / T17 (stage2c3): the capture uses exact runs only while the chunk has GPU work not merged yet. T16: N merged, then a CPU write: a whole-page upload
      // (no exact run); T17: N in flight, then a CPU write on the same page: exact runs, and both N's GPU bytes and the CPU's reach the next read
      { const int as = pc_async(); uint8_t *H16 = A + 26 * PG, *K16 = C + 28 * PG;
        pct_cpu(H16, 0, 16); mtl_sub_begin(); pct_go();
        mtl_sub_begin(); FILL(H16, 4, 0x16); pct_go();   // merged (pct_go waits for its completion)
        pct_cpu(H16 + 8, 0x61, 4); const uint64_t e0 = g_pc_exact_calls, w0 = g_pc_whole_calls;
        mtl_sub_begin(); COPY(H16, K16, 12); pct_go();
        const int whole = !as || (g_pc_exact_calls == e0 && g_pc_whole_calls > w0);
        size_t b1 = pct_bad(K16, 0x16, 4), b2 = pct_bad(K16 + 8, 0x61, 4);
        pct_result(116, "T16 N merged, then a CPU write: whole-page upload, both bytes read", !whole ? 1 : b1 ? b1 : b2 ? 8 + b2 : 0, K16, !whole ? 0xff : b1 ? 0x16 : 0x61);
        uint8_t *H17 = A + 27 * PG, *K17 = C + 29 * PG;
        pct_cpu(H17, 0, 16); mtl_sub_begin(); pct_go();
        HOLD(1); mtl_sub_begin(); FILL(H17, 4, 0x17); f1 = mtl_sub_commit(); pct_cpu(H17 + 8, 0x71, 4); const uint64_t e1 = g_pc_exact_calls;
        mtl_sub_begin(); COPY(H17, K17, 12); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
        const int exact = !as || !hold_ok || g_pc_exact_calls > e1;
        b1 = pct_bad(K17, 0x17, 4); b2 = pct_bad(K17 + 8, 0x71, 4); size_t b3 = pct_bad(H17, 0x17, 4);
        pct_result(117, "T17 N in flight, then a CPU write on its page: exact runs, the GPU's and the CPU's bytes read", !exact ? 1 : b1 ? b1 : b2 ? 8 + b2 : b3, b3 && !b1 && !b2 ? H17 : K17, !exact ? 0xff : b1 || b3 ? 0x17 : 0x71); }
      // V1-V5 (stage2b, volatile pages; their state is checked where they apply: async with NXN1X_PC_VOLATILE on; the bytes are checked in every mode)
      { const int vol = pc_vol_on();
        // stage2b2: a pool of their own: the cases above mark pool A as GPU-written (case 1's storage binding to the chunk's end, case 2's raw global memory)
        uint8_t *VP = pct_alloc(P); if (!VP) { pct_result(120, "V tests: memory for their pool", 1, A, 0); goto vol_done; } pct_pool(VP, P);
        if (pc_hybrid()) { mtl_sub_begin(); pct_atomic(VP, 0x48b0); pct_go(); }   // hybrid1: VP a device-local copy from the next submission (the V cases need one)
#define VST(a) pct_vol_state(a)
// stage2b3: entry counts write faults: the store goes through the fault path (as a guest store: the hook, then the store), not a host pre-write
#define FAULTW(pg, val) do { nvnh_fault_fn h_ = nvnh_fault_hook(); if (h_) h_((pg), 1, NULL); memset((pg), (val), 4); } while (0)
#define MAKEVOL(pg, base) do { for (int k_ = 0; k_ < 3; k_++) { FAULTW((pg), (uint8_t)((base) + k_)); mtl_sub_begin(); pct_go(); } } while (0)
        // V1: entry after 3 syncs with a change, a write captured while volatile, protected again after 8 syncs with none
        // (stage2b3: V1 failed at 19:46 because the pool's copy did not exist yet: a pool is made at its first GPU use, so its first syncs did not run. A GPU read first)
        uint8_t *E = VP + 0 * PG, *KE = C + 30 * PG; pct_cpu(E, 0, 8); mtl_sub_begin(); COPY(E, KE, 4); pct_go(); mtl_sub_begin(); pct_go();
        MAKEVOL(E, 0x30); const int in1 = !vol || VST(E) == 1;
        pct_cpu(E, 0x40, 4); mtl_sub_begin(); COPY(E, KE, 4); pct_go(); const size_t b1 = pct_bad(KE, 0x40, 4);
        for (int k = 0; k < 66; k++) { mtl_sub_begin(); pct_go(); } const int out1 = !vol || VST(E) == 0;
        pct_cpu(E, 0x41, 4); mtl_sub_begin(); COPY(E, KE, 4); pct_go(); const size_t b2 = pct_bad(KE, 0x41, 4);
        pct_result(121, "V1 volatile after write faults at 3 of 32 syncs, a write read while volatile, protected again after 64 idle syncs", !in1 ? 1 : !out1 ? 2 : b1 ? b1 : b2, !in1 || !out1 ? E : KE, !in1 || !out1 ? 0xff : b1 ? 0x40 : 0x41);
        // V2: a CPU write right after the capture of a volatile page (no fault: it is writable): this submission reads the captured bytes, the next one the new
        uint8_t *E2 = VP + 1 * PG, *K2 = C + 31 * PG; pct_cpu(E2, 0, 8); mtl_sub_begin(); pct_go(); MAKEVOL(E2, 0x50);
        // (the host store below needs the page writable: only when it is volatile; else recorded as a failure, never a store into a protected page)
        const int vol2 = VST(E2) == 1;
        pct_cpu(E2, 0x58, 4); mtl_sub_begin(); if (vol2) memset(E2, 0x59, 4); else pct_cpu(E2, 0x59, 4); COPY(E2, K2, 4); pct_go(); const size_t c1 = vol2 ? pct_bad(K2, 0x58, 4) : 0;
        mtl_sub_begin(); COPY(E2, K2 + 64, 4); pct_go(); const size_t c2 = pct_bad(K2 + 64, 0x59, 4);
        pct_result(122, "V2 a CPU write racing the capture of a volatile page: caught at the next sync", vol && !vol2 ? 1 : c1 ? c1 : c2 ? 64 + c2 : 0, K2, vol && !vol2 ? 0xff : c1 ? 0x58 : 0x59);
        // V3: GPU work writes a volatile page: it is compared and protected again at the next sync, and both the GPU's and the CPU's bytes are read
        uint8_t *E3 = VP + 2 * PG, *K3 = C + 32 * PG; pct_cpu(E3, 0, 128); mtl_sub_begin(); pct_go(); MAKEVOL(E3, 0x60);
        const int in3 = !vol || VST(E3) == 1;
        mtl_sub_begin(); FILL(E3 + 64, 4, 0x6a); pct_go(); pct_cpu(E3, 0x6b, 4);
        mtl_sub_begin(); COPY(E3, K3, 68); pct_go();
        const int out3 = !vol || VST(E3) == 0; const size_t d1 = pct_bad(K3, 0x6b, 4), d2 = pct_bad(K3 + 64, 0x6a, 4), d3 = pct_bad(E3 + 64, 0x6a, 4);
        pct_result(123, "V3 a GPU write on a volatile page: tracked again, the GPU's and the CPU's bytes read", !in3 ? 1 : !out3 ? 2 : d1 ? d1 : d2 ? 64 + d2 : d3 ? 64 + d3 : 0, !in3 || !out3 ? E3 : d3 ? E3 + 64 : K3, !in3 || !out3 ? 0xff : d1 ? 0x6b : 0x6a);
        // V4: protected again after 8 syncs with no change, and a write after that is taken through the fault path (here: the fault hook)
        uint8_t *E4 = VP + 3 * PG, *K4v = C + 33 * PG; pct_cpu(E4, 0, 8); mtl_sub_begin(); pct_go(); MAKEVOL(E4, 0x70);
        for (int k = 0; k < 66; k++) { mtl_sub_begin(); pct_go(); }
        const int out4 = !vol || VST(E4) == 0;
        { nvnh_fault_fn h = nvnh_fault_hook(); if (h) h(E4, 1, NULL); } *(volatile uint32_t *)E4 = 0x7f7f7f7fu;
        mtl_sub_begin(); COPY(E4, K4v, 4); pct_go();
        pct_result(124, "V4 protected again after 64 idle syncs: a later write goes through the fault path and is read", !out4 ? 1 : pct_bad(K4v, 0x7f, 4), !out4 ? E4 : K4v, !out4 ? 0xff : 0x7f);
        // V5: a volatile page written through another address of the same pages (VB): the next read through VA has it
        uint8_t *E5 = VA + 11 * PG, *U5 = VB + 11 * PG, *K5 = C + 34 * PG; pct_cpu(E5, 0, 8); mtl_sub_begin(); pct_go(); MAKEVOL(E5, 0x80);
        const int in5 = !vol || VST(E5) == 1;
        pct_cpu(U5 + 4, 0x8a, 4); mtl_sub_begin(); COPY(E5, K5, 8); pct_go();
        const size_t e1 = pct_bad(K5, 0x82, 4), e2 = pct_bad(K5 + 4, 0x8a, 4);
        pct_result(125, "V5 a volatile page written through an alias address", !in5 ? 1 : e1 ? e1 : e2 ? 4 + e2 : 0, !in5 ? E5 : K5, !in5 ? 0xff : e1 ? 0x82 : 0x8a);
        // V6 (stage2b2): a GPU-written page decays (merged, 8 syncs with no GPU write), becomes volatile, then GPU work writes it again: tracked again, both bytes kept
        uint8_t *E6 = VP + 4 * PG, *K6v = C + 35 * PG; pct_cpu(E6, 0, 128); mtl_sub_begin(); pct_go();
        mtl_sub_begin(); FILL(E6 + 64, 4, 0x90); pct_go();
        for (int k = 0; k < 9; k++) { mtl_sub_begin(); pct_go(); }
        MAKEVOL(E6, 0xa0); const int in6 = !vol || VST(E6) == 1;
        mtl_sub_begin(); FILL(E6 + 64, 4, 0x91); pct_go(); pct_cpu(E6, 0xa5, 4);
        mtl_sub_begin(); COPY(E6, K6v, 68); pct_go();
        const int out6 = !vol || VST(E6) == 0; const size_t g1 = pct_bad(K6v, 0xa5, 4), g2 = pct_bad(K6v + 64, 0x91, 4);
        pct_result(126, "V6 a GPU-written page decays, becomes volatile, is GPU-written again: tracked again, both bytes kept", !in6 ? 1 : !out6 ? 2 : g1 ? g1 : g2 ? 64 + g2 : 0, !in6 || !out6 ? E6 : K6v, !in6 || !out6 ? 0xff : g1 ? 0xa5 : 0x91);
        // V7 (stage2b3): a page written once every 6 syncs (a game submitting 6 times a frame): volatile after its third write, within 32 syncs
        uint8_t *E7 = VP + 5 * PG, *K7 = C + 36 * PG; pct_cpu(E7, 0, 8); mtl_sub_begin(); COPY(E7, K7, 4); pct_go();
        for (int k = 0; k < 3; k++) { FAULTW(E7, (uint8_t)(0xb0 + k)); for (int j = 0; j < 6; j++) { mtl_sub_begin(); pct_go(); } }
        const int in7 = !vol || VST(E7) == 1;
        mtl_sub_begin(); COPY(E7, K7, 4); pct_go();
        pct_result(127, "V7 a page written once every 6 syncs becomes volatile", !in7 ? 1 : pct_bad(K7, 0xb2, 4), !in7 ? E7 : K7, !in7 ? 0xff : 0xb2);
        // FA1-FA3 (combined1120, sequential fault-ahead; with NXN1X_PC_FAULTAHEAD=0 the same bytes are checked, through ordinary faults)
        { uint8_t *Q = VP + 8 * PG, *KQ = C + 37 * PG; for (int k = 0; k < 4; k++) pct_cpu(Q + k * PG, 0, 16);
          mtl_sub_begin(); COPY(Q, KQ, 4); pct_go(); mtl_sub_begin(); pct_go();
          // FA1: a stream of faults runs ahead over a page with a GPU write not merged yet: the GPU's bytes and the CPU's both reach the next read
          HOLD(1); mtl_sub_begin(); FILL(Q + 2 * PG, 4, 0xc1); f1 = mtl_sub_commit();
          FAULTW(Q, 0x11); FAULTW(Q + PG, 0x12); FAULTW(Q + 2 * PG + 8, 0xc2);
          mtl_sub_begin(); COPY(Q + 2 * PG, KQ + 64, 12); f2 = mtl_sub_commit(); HOLD(0); WAITF(f2);
          size_t a1 = pct_bad(KQ + 64, 0xc1, 4), a2 = pct_bad(KQ + 72, 0xc2, 4), a3 = pct_bad(Q + 2 * PG, 0xc1, 4);
          pct_result(131, "FA1 fault-ahead over a page with an unmerged GPU write: both bytes kept", a1 ? a1 : a2 ? 8 + a2 : a3, a1 || a2 ? KQ + 64 : Q + 2 * PG, a1 || a3 ? 0xc1 : 0xc2);
          // FA2: a stream of faults at the pool's last pages: the run ahead stops at its end
          uint8_t *L = VP + P - 2 * PG; pct_cpu(L, 0, 4); pct_cpu(L + PG, 0, 4); mtl_sub_begin(); COPY(L, KQ + 128, 4); pct_go(); mtl_sub_begin(); pct_go();
          FAULTW(L, 0x21); FAULTW(L + PG, 0x22); mtl_sub_begin(); COPY(L + PG, KQ + 128, 4); pct_go();
          pct_result(132, "FA2 fault-ahead at the pool's end", pct_bad(KQ + 128, 0x22, 4), KQ + 128, 0x22);
          // FA3: the run ahead in VA leaves the alias VB's protection alone: a write through VB is still seen
          uint8_t *VA3 = VA + 12 * PG, *VB3 = VB + 12 * PG; pct_cpu(VA3, 0, 4); pct_cpu(VA3 + PG, 0, 4); pct_cpu(VA3 + 2 * PG, 0, 4); mtl_sub_begin(); COPY(VA3, KQ + 192, 4); pct_go(); mtl_sub_begin(); pct_go();
          FAULTW(VA3, 0x31); FAULTW(VA3 + PG, 0x32); pct_cpu(VB3 + 2 * PG, 0x35, 4);
          mtl_sub_begin(); COPY(VA3 + 2 * PG, KQ + 192, 4); pct_go();
          pct_result(133, "FA3 fault-ahead in one address of the pages, a write through the alias", pct_bad(KQ + 192, 0x35, 4), KQ + 192, 0x35); }
        pct_unpool(VP);
      vol_done:;
#undef VST
#undef MAKEVOL
#undef FAULTW
      }
      HOLD(0);
#undef HOLD
#undef FILL
#undef COPY
#undef WAITF
    }
    pct_unpool(A); pct_unpool(C); pct_unpool(Z); pct_unpool(M); pct_unpool(D); pct_unpool(VA); pct_unpool(VB); nvnh_test_alias_end();
    VKLOG("%spoolcopy self-test: %d of %d checks passed (%s mode)", pct_ok == pct_n ? "" : "[RED] ", pct_ok, pct_n, pc_hybrid() ? "hybrid" : vk_poolcopy_on() ? "copy" : "import");
    return pct_ok == pct_n ? 0 : -1;
}
#endif
