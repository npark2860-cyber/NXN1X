#include "nvn_replay_lease.h"
#include "nvn_image_eligibility.h"
// Android port, Vulkan backend: formats, textures, views, samplers and the texture / sampler ID tables (see gfx_vk.h).
//   A texture object's image is found by its storage: textures over the same pool, offset and description share one image (as nvn_mtl.m's storage-sharing
//   table), which is how a render target and the texture a later pass samples it through meet. A packaged texture (packaged layout, CPU-written) is filled
//   from the guest's block-linear bytes (nvn_deswizzle on the CPU into scratch memory, copied in by the upload command buffer) and filled again when the
//   dirty-page tracker saw a write to its pages; any other texture is the GPU's (render targets, images), filled only by GPU work.
//   Version 1 keeps every image in VK_IMAGE_LAYOUT_GENERAL after its first use, ordered by whole-pipeline barriers.
#include "gfx_vk.h"
#include "nvn_image_archive.h"
#include "nvn_memory_transfer.h"
#include <time.h>
#include "nvn_tex.h"
#include "../gfx/tex/nvn_deswizzle.h"
#include "nvn_hmap.h"
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>

int nvn_dirty_since(const void *addr, size_t len, uint64_t stamp);   // gwrange2
int nvn_dirty_since_span(const void *span, const void *addr, size_t len, uint64_t stamp);
const void *nvn_dirty_span(const void *addr, size_t len);
void nvn_dirty_protect(const void *addr, size_t len); void nvn_dirty_release(const void *addr, size_t len); uint64_t nvn_dirty_released(void); int nvn_dirty_runs(void);
uint64_t nvn_dirty_now(void);
void nvn_dirty_stamp_unprotected(int on); uint64_t nvn_dirty_stamped(void);
// Packaged textures are not write-protected unless DM_VK_PKG_PROTECT=1: VirtualProtect took about 1 ms a page (r54: 1,243 calls, 46,386 pages, 45.6 s of loading) and caught no write (0 faults);
// host writes (WriteTexels, mirrors) still stamp the pages, so those re-upload; a guest CPU write into a packaged texture is no longer seen
static int pkg_protect(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_PKG_PROTECT"); v = e && atoi(e); if (!v) nvn_dirty_stamp_unprotected(1);
        VKLOG("packaged textures: write protection %s (DM_VK_PKG_PROTECT)", v ? "on" : "off, host writes still mark them"); }
    return v;
}

// ---------------- formats ----------------
enum { K_NONE, K_FLOAT, K_INT, K_SINT };
typedef struct { VkFormat vk; uint8_t noalpha, rb, kind; } fent;
#define F(v) {v, 0, 0, K_FLOAT}
#define FX(v) {v, 1, 0, K_FLOAT}
#define U(v) {v, 0, 0, K_INT}
#define UX(v) {v, 1, 0, K_INT}
#define I(v) {v, 0, 0, K_SINT}
#define IX(v) {v, 1, 0, K_SINT}
static const fent FT[0x9D] = {
    [0x01] = F(VK_FORMAT_R8_UNORM), F(VK_FORMAT_R8_SNORM), U(VK_FORMAT_R8_UINT), I(VK_FORMAT_R8_SINT),
    [0x05] = F(VK_FORMAT_R16_SFLOAT), F(VK_FORMAT_R16_UNORM), F(VK_FORMAT_R16_SNORM), U(VK_FORMAT_R16_UINT), I(VK_FORMAT_R16_SINT),
    [0x0A] = F(VK_FORMAT_R32_SFLOAT), U(VK_FORMAT_R32_UINT), I(VK_FORMAT_R32_SINT),
    [0x0D] = F(VK_FORMAT_R8G8_UNORM), F(VK_FORMAT_R8G8_SNORM), U(VK_FORMAT_R8G8_UINT), I(VK_FORMAT_R8G8_SINT),
    [0x11] = F(VK_FORMAT_R16G16_SFLOAT), F(VK_FORMAT_R16G16_UNORM), F(VK_FORMAT_R16G16_SNORM), U(VK_FORMAT_R16G16_UINT), I(VK_FORMAT_R16G16_SINT),
    [0x16] = F(VK_FORMAT_R32G32_SFLOAT), U(VK_FORMAT_R32G32_UINT), I(VK_FORMAT_R32G32_SINT),
    // 0x19-0x24 three-component formats: not sampleable on mobile GPUs (vertex input only, see vk_vertex_format)
    [0x25] = F(VK_FORMAT_R8G8B8A8_UNORM), F(VK_FORMAT_R8G8B8A8_SNORM), U(VK_FORMAT_R8G8B8A8_UINT), I(VK_FORMAT_R8G8B8A8_SINT),
    [0x29] = F(VK_FORMAT_R16G16B16A16_SFLOAT), F(VK_FORMAT_R16G16B16A16_UNORM), F(VK_FORMAT_R16G16B16A16_SNORM), U(VK_FORMAT_R16G16B16A16_UINT), I(VK_FORMAT_R16G16B16A16_SINT),
    [0x2E] = F(VK_FORMAT_R32G32B32A32_SFLOAT), U(VK_FORMAT_R32G32B32A32_UINT), I(VK_FORMAT_R32G32B32A32_SINT),
    [0x31] = U(VK_FORMAT_S8_UINT), F(VK_FORMAT_D16_UNORM), F(VK_FORMAT_X8_D24_UNORM_PACK32), F(VK_FORMAT_D32_SFLOAT), F(VK_FORMAT_D24_UNORM_S8_UINT), F(VK_FORMAT_D32_SFLOAT_S8_UINT),
    [0x37] = FX(VK_FORMAT_R8G8B8A8_SRGB), F(VK_FORMAT_R8G8B8A8_SRGB),
    [0x39] = F(VK_FORMAT_A4B4G4R4_UNORM_PACK16), FX(VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR), F(VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR), F(VK_FORMAT_R5G6B5_UNORM_PACK16),
    [0x3D] = F(VK_FORMAT_A2B10G10R10_UNORM_PACK32), U(VK_FORMAT_A2B10G10R10_UINT_PACK32), F(VK_FORMAT_B10G11R11_UFLOAT_PACK32), F(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32),
    [0x41] = F(VK_FORMAT_BC1_RGB_UNORM_BLOCK), F(VK_FORMAT_BC1_RGBA_UNORM_BLOCK), F(VK_FORMAT_BC2_UNORM_BLOCK), F(VK_FORMAT_BC3_UNORM_BLOCK),
             F(VK_FORMAT_BC1_RGB_SRGB_BLOCK), F(VK_FORMAT_BC1_RGBA_SRGB_BLOCK), F(VK_FORMAT_BC2_SRGB_BLOCK), F(VK_FORMAT_BC3_SRGB_BLOCK),
    [0x49] = F(VK_FORMAT_BC4_UNORM_BLOCK), F(VK_FORMAT_BC4_SNORM_BLOCK), F(VK_FORMAT_BC5_UNORM_BLOCK), F(VK_FORMAT_BC5_SNORM_BLOCK),
             F(VK_FORMAT_BC7_UNORM_BLOCK), F(VK_FORMAT_BC7_SRGB_BLOCK), F(VK_FORMAT_BC6H_SFLOAT_BLOCK), F(VK_FORMAT_BC6H_UFLOAT_BLOCK),
    // *_UI2F / *_I2F: integer storage read as float without normalization = Vulkan's scaled formats
    [0x51] = F(VK_FORMAT_R8_USCALED), F(VK_FORMAT_R8_SSCALED), F(VK_FORMAT_R16_USCALED), F(VK_FORMAT_R16_SSCALED), U(VK_FORMAT_R32_UINT), I(VK_FORMAT_R32_SINT),
    [0x57] = F(VK_FORMAT_R8G8_USCALED), F(VK_FORMAT_R8G8_SSCALED), F(VK_FORMAT_R16G16_USCALED), F(VK_FORMAT_R16G16_SSCALED), U(VK_FORMAT_R32G32_UINT), I(VK_FORMAT_R32G32_SINT),
    [0x63] = F(VK_FORMAT_R8G8B8A8_USCALED), F(VK_FORMAT_R8G8B8A8_SSCALED), F(VK_FORMAT_R16G16B16A16_USCALED), F(VK_FORMAT_R16G16B16A16_SSCALED), U(VK_FORMAT_R32G32B32A32_UINT), I(VK_FORMAT_R32G32B32A32_SINT),
    [0x69] = F(VK_FORMAT_A2B10G10R10_SNORM_PACK32), I(VK_FORMAT_A2B10G10R10_SINT_PACK32), F(VK_FORMAT_A2B10G10R10_USCALED_PACK32), F(VK_FORMAT_A2B10G10R10_SSCALED_PACK32),
    [0x6D] = FX(VK_FORMAT_R8G8B8A8_UNORM), FX(VK_FORMAT_R8G8B8A8_SNORM), UX(VK_FORMAT_R8G8B8A8_UINT), IX(VK_FORMAT_R8G8B8A8_SINT),
    [0x71] = FX(VK_FORMAT_R16G16B16A16_SFLOAT), FX(VK_FORMAT_R16G16B16A16_UNORM), FX(VK_FORMAT_R16G16B16A16_SNORM), UX(VK_FORMAT_R16G16B16A16_UINT), IX(VK_FORMAT_R16G16B16A16_SINT),
    [0x76] = FX(VK_FORMAT_R32G32B32A32_SFLOAT), UX(VK_FORMAT_R32G32B32A32_UINT), IX(VK_FORMAT_R32G32B32A32_SINT),
    [0x95] = F(VK_FORMAT_B5G6R5_UNORM_PACK16), FX(VK_FORMAT_A1R5G5B5_UNORM_PACK16), F(VK_FORMAT_A1R5G5B5_UNORM_PACK16), F(VK_FORMAT_R5G5B5A1_UNORM_PACK16),
    [0x99] = FX(VK_FORMAT_B8G8R8A8_UNORM), F(VK_FORMAT_B8G8R8A8_UNORM), FX(VK_FORMAT_B8G8R8A8_SRGB), F(VK_FORMAT_B8G8R8A8_SRGB),
};
static vkfmt_t g_fmt[0x9D]; static uint8_t g_fmt_done[0x9D];
static pthread_mutex_t g_fmt_mu = PTHREAD_MUTEX_INITIALIZER;
const vkfmt_t *vk_fmt(int f) {
    if (f <= 0 || f >= 0x9D) return NULL;
    if (__atomic_load_n(&g_fmt_done[f], __ATOMIC_ACQUIRE)) return g_fmt[f].kind ? &g_fmt[f] : NULL;   // made already: no lock (the replay looks formats up for every texture it binds)
    pthread_mutex_lock(&g_fmt_mu);
    if (!g_fmt_done[f]) {
        vkfmt_t *o = &g_fmt[f]; memset(o, 0, sizeof *o);
        fent e = FT[f];
        if (f >= 0x79 && f <= 0x94) { int k = (f - 0x79) % 14; e = (fent){(VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * k + (f >= 0x87)), 0, 0, K_FLOAT}; }
        if (e.vk) {
            int bw, bh; o->bpb = (uint8_t)nvn_fmt_bpb(f, &bw, &bh); o->bw = (uint8_t)bw; o->bh = (uint8_t)bh;
            o->vk = e.vk;
            if (f == 0x33) {   // DEPTH24: X8_D24 where the device renders to it, else D32
                VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_X8_D24_UNORM_PACK32, &fp);
                if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) o->vk = VK_FORMAT_D32_SFLOAT;
            }
            if (f == 0x31) {   // STENCIL8 alone: a combined format where the device has no S8 attachments
                VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, VK_FORMAT_S8_UINT, &fp);
                if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) o->vk = VK_FORMAT_D24_UNORM_S8_UINT;
            }
            o->depth = f >= 0x32 && f <= 0x36; o->stencil = f == 0x31 || f == 0x35 || f == 0x36 || (f == 0x31);
            o->compressed = o->bw > 1 || o->bh > 1;
            o->is_int = e.kind == K_INT || e.kind == K_SINT; o->is_sint = e.kind == K_SINT; o->kind = 1;
            o->swz = (VkComponentMapping){VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, e.noalpha ? VK_COMPONENT_SWIZZLE_ONE : VK_COMPONENT_SWIZZLE_IDENTITY};
            if ((o->compressed && o->vk >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && o->vk <= VK_FORMAT_BC7_SRGB_BLOCK && !V.has_bc) ||
                (o->vk >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && o->vk <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK && !V.has_astc)) {
                o->kind = 0;   // said once per format: these textures went missing with nothing in the log (Adreno 6xx drivers of 2021 have no BC)
                VKLOG("[RED] texture format %#x (VkFormat %d) needs %s, which the device lacks: textures in it are not drawn", f, o->vk,
                      o->vk <= VK_FORMAT_BC7_SRGB_BLOCK ? "BC (textureCompressionBC)" : "ASTC (textureCompressionASTC_LDR)");
            }
        }
        __atomic_store_n(&g_fmt_done[f], 1, __ATOMIC_RELEASE);   // published once complete
    }
    pthread_mutex_unlock(&g_fmt_mu);
    return g_fmt[f].kind ? &g_fmt[f] : NULL;
}
static VkImageAspectFlags aspect_of(const vkfmt_t *f) {
    if (f->vk == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (f->depth || f->stencil) return VK_IMAGE_ASPECT_DEPTH_BIT | (f->vk == VK_FORMAT_D24_UNORM_S8_UINT || f->vk == VK_FORMAT_D32_SFLOAT_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

// ---------------- images ----------------
static unsigned layers_of(const nvn_texdesc *d) { return d->target == 4 || d->target == 9 ? (d->depth ? d->depth : 1) : d->target == 8 ? 6 : d->target == 3 ? (d->height ? d->height : 1) : 1; }
static VkImageViewType vtype_of(int target) {
    switch (target) {
    case 0: return VK_IMAGE_VIEW_TYPE_1D; case 3: return VK_IMAGE_VIEW_TYPE_1D_ARRAY; case 2: return VK_IMAGE_VIEW_TYPE_3D;
    case 4: case 6: return VK_IMAGE_VIEW_TYPE_2D_ARRAY; case 8: return VK_IMAGE_VIEW_TYPE_CUBE; case 9: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}
// storage key: textures with the same pool, offset and description share one image
typedef struct { uint64_t pool; int64_t off; nvn_texdesc d; } akey;
static nvn_hmap g_by_key = {.klen = sizeof(akey)};     // akey → vkimg_t*
static nvn_hmap g_by_serial = {.klen = sizeof(uint64_t)};  // serial → vkimg_t* (images with no storage key: no pool memory)
static nvnr_gate g_img_mu = NVNR_GATE_INITIALIZER;
static uint64_t g_residency_id; // img_new is serialized by g_img_mu; never reuse after wrap
static void image_version_advance(vkimg_t *im) {
    uint64_t v=__atomic_load_n(&im->content_serial,__ATOMIC_RELAXED);
    while (v!=UINT64_MAX && !__atomic_compare_exchange_n(&im->content_serial,&v,v+1,1,__ATOMIC_RELAXED,__ATOMIC_RELAXED)) {}
}
// slot12 replaydiag (DM_VK_REPLAYDIAG=1, log only): the images vk_tex makes (OoT 11-21: 5,357 -> 10,280 GPU-only images kept in 50 s, ~2-8 new a frame while
// ~35 textures a frame are finalized), by Vulkan format, size and class (0 colour, 1 depth/stencil, 2 storage), whether each one's first use is a GPU write
// (attachment, clear, copy, storage store) or a read (a sampled view), and how many finalized images come back under the same key. With the replay line (vk_rdi_log)
extern int g_rdiag;
typedef struct { uint32_t fmt, w, h, cls; uint64_t n, wf, rf; } rdi_ent;
static rdi_ent g_rdi[128]; static int g_rdin; static uint32_t g_rdi_gen = 1;
static uint64_t g_rdi_new, g_rdi_wf, g_rdi_rf, g_rdi_rel, g_rdi_back;
static void rdi_new(vkimg_t *im, uint32_t nvn_flags) {
    if (g_rdiag <= 0 || !im) return;
    const uint32_t cls = (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) ? 1 : (nvn_flags & NVN_TEXTURE_FLAGS_IMAGE) ? 2 : 0;
    int i; for (i = 0; i < g_rdin; i++) if (g_rdi[i].fmt == (uint32_t)im->fmt && g_rdi[i].w == im->w && g_rdi[i].h == im->h && g_rdi[i].cls == cls) break;
    if (i == g_rdin) { if (g_rdin < 128) g_rdi[g_rdin++] = (rdi_ent){(uint32_t)im->fmt, im->w, im->h, cls, 0, 0, 0}; else i = 127; }   // (a full table: the rest in the last entry)
    g_rdi[i].n++; g_rdi_new++; im->rdi_slot = (uint16_t)(i + 1); im->rdi_gen = g_rdi_gen; im->rdi_first = 0;
}
static inline void rdi_use(vkimg_t *im, int write) {
    if (!im || !im->rdi_slot || im->rdi_first) return;
    im->rdi_first = write ? 1 : 2;
    if (im->rdi_gen != g_rdi_gen) return;   // made in an earlier window
    if (write) { g_rdi[im->rdi_slot - 1].wf++; g_rdi_wf++; } else { g_rdi[im->rdi_slot - 1].rf++; g_rdi_rf++; }
}
void vk_rdi_log(void) {
    if (g_rdiag <= 0) return;
    char line[900]; int o = 0; uint64_t taken[2] = {0, 0};
    for (int k = 0; k < 8; k++) { int b = -1;
        for (int i = 0; i < g_rdin; i++) if (!(taken[i >> 6] >> (i & 63) & 1) && g_rdi[i].n && (b < 0 || g_rdi[i].n > g_rdi[b].n)) b = i;
        if (b < 0) break; taken[b >> 6] |= 1ull << (b & 63);
        static const char *const cn[3] = {"colour", "depth", "storage"};
        o += snprintf(line + o, sizeof line - (size_t)o, " fmt %u %ux%u %s %llu (written first %llu, read first %llu);", g_rdi[b].fmt, g_rdi[b].w, g_rdi[b].h, cn[g_rdi[b].cls],
                      (unsigned long long)g_rdi[b].n, (unsigned long long)g_rdi[b].wf, (unsigned long long)g_rdi[b].rf);
        if (o >= (int)sizeof line - 96) break; }
    VKLOG("replaydiag: new images (last 600 presents) %llu: first use a GPU write %llu, a read %llu, none yet %llu; finalized %llu, taken back by the same key %llu; top:%s",
          (unsigned long long)g_rdi_new, (unsigned long long)g_rdi_wf, (unsigned long long)g_rdi_rf, (unsigned long long)(g_rdi_new - g_rdi_wf - g_rdi_rf),
          (unsigned long long)g_rdi_rel, (unsigned long long)g_rdi_back, o ? line : " none");
    memset(g_rdi, 0, sizeof g_rdi); g_rdin = 0; g_rdi_gen++; g_rdi_new = g_rdi_wf = g_rdi_rf = g_rdi_rel = g_rdi_back = 0;
}
// relovl (11-21, OoT: 47,057 images / 86 GB after 11 minutes, driver heap at 100%, "[RED] NVN2 memory pressure"): evict_overlaps keeps an older GPU-only image that a newer texture
// was made over (keep_gw, bak91), so that a texture made again under the old key finds its contents. OoT makes its small render targets (213x120, 427x240, 640x360 ...) again and
// again over memory other ones had: 43,000 kept images, 82 GB, none ever released (the released-cache trim takes only images it can rebuild). On the Switch those contents last
// only until something writes that memory; once the GPU writes a newer image over it they are gone there too. So a kept image is marked when an image linked over its memory
// (alias_link) is written by the GPU, and ovw_sweep drops the marked ones not looked up for EVICT_KEEP presents. DM_VK_KEPT_OVERWRITE_DROP=0 turns it off
static int g_ovw_pending; static uint64_t g_ovw_dropped, g_ovw_bytes;
static int ovw_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_KEPT_OVERWRITE_DROP"); on = !e || atoi(e) != 0; } return on; }
static void note_image_write(vkimg_t *im) {
    rdi_use(im, 1);   // slot12 replaydiag
    if (im && dm_sdk_ounce()) image_version_advance(im);
    if (im && im->novl && ovw_on()) for (int i = 0; i < im->novl; i++) { vkimg_t *o = im->ovl[i]; if (o->keep_gw && !o->ovw) { o->ovw = 1; g_ovw_pending = 1; } }   // relovl
#if defined(DM_NXN1X)
    if (im) nvnc_image_intent(im->content_id); // conservative intent, not completed contents
#endif
}
// clrorder (diagnostic, 11-21, NPARK, Lichii #185: OoT clears its occlusion depth, blits the half-resolution scene depth over it and its terrain layer reads that
// depth; Eden ran the clear, parked as the next pass's load op, after the blit). Here a whole-attachment clear before any draw is folded into the next pass's load
// op too (clear_color / clear_ds: vk_clrdiag_fold) and runs at that pass's begin (pass_commit: vk_clrdiag_done). Logged: an out-of-pass write (copy, blit, texture
// clear, storage write, vk_tex_content_changed) to an image, or to an image linked over its memory, while such a clear is still parked: the clear would land after the
// write. And an upload (the upload command buffer runs before the submission's main one) of an image cleared earlier in the same submission's main buffer
extern uint64_t g_tex_frame; static uint64_t g_clr_seq, g_clr_bad[3];
void vk_clrdiag_fold(vkimg_t *im) { if (im) im->clr_pend = ++g_clr_seq; }
void vk_clrdiag_done(vkimg_t *im) { if (im) { im->clr_pend = 0; im->clr_fence = mtl_last_fence() + 1; } }
static void clrdiag_log(int kind, const vkimg_t *im, const vkimg_t *o, const char *what) {
    const uint64_t c = ++g_clr_bad[kind]; if (c > 32 && (c & (c - 1))) return;
    static const char *const kn[3] = {"[RED] clrorder: an out-of-pass write to an image whose clear is still parked in the next pass's load op (the clear would run after it)",
        "[RED] clrorder: an out-of-pass write to an image over the memory of an image whose clear is still parked",
        "[RED] clrorder: an upload (upload command buffer, runs first) to an image cleared earlier in this submission's main command buffer"};
    VKLOG("%s: %s %ux%u nvn format %#x at +%#llx%s%s (other %ux%u format %#x at +%#llx) at present %llu (%llu so far)", kn[kind], what, im->w, im->h, im->nvn_format, (unsigned long long)im->off,
          "", "", o ? o->w : 0, o ? o->h : 0, o ? o->nvn_format : 0, o ? (unsigned long long)o->off : 0ull, (unsigned long long)g_tex_frame, (unsigned long long)c);
}
static void clrdiag_write(const vkimg_t *im) {
    if (!im) return;
    if (im->clr_pend) clrdiag_log(0, im, NULL, "write");
    for (int i = 0; i < im->novl; i++) if (im->ovl[i]->clr_pend) clrdiag_log(1, im, im->ovl[i], "write");
}
static void clrdiag_upload(const vkimg_t *im) {
    if (!im) return; const uint64_t cur = mtl_last_fence() + 1;
    if (im->clr_fence == cur) clrdiag_log(2, im, NULL, "upload");
    for (int i = 0; i < im->novl; i++) if (im->ovl[i]->clr_fence == cur) clrdiag_log(2, im, im->ovl[i], "upload over");
}
void vk_tex_content_changed(vkimg_t *im) {
    clrdiag_write(im);   // clrorder (diagnostic)
    if (im) im->scaled_valid = 0;
    note_image_write(im);
}
static void content_cpu_upload(vkimg_t *im) {
    clrdiag_upload(im);   // clrorder (diagnostic)
    if (im) im->scaled_valid = 0;
    if (im && dm_sdk_ounce()) image_version_advance(im);
    if (im && im->novl && ovw_on()) for (int i = 0; i < im->novl; i++) { vkimg_t *o = im->ovl[i]; if (o->keep_gw && !o->ovw) { o->ovw = 1; g_ovw_pending = 1; } }   // relovl: the CPU's data over that memory too
#if defined(DM_NXN1X)
    if (im) nvnc_image_upload(im->content_id);
#endif
}
// The sRGB / UNORM twin of a format (the one reinterpretation games make of uncompressed colour targets), VK_FORMAT_UNDEFINED when it has none
static VkFormat srgb_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB; case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return VK_FORMAT_A8B8G8R8_SRGB_PACK32; case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}
// r72: can an image of format f be a storage image: by its own format, or (via_twin) through its sRGB / UNORM twin, the format its image views then take (Hyrule
// Warriors: Age of Imprisonment registers RGBA8 sRGB textures as UNORM images, 0x38 -> 0x25); the image then needs EXTENDED_USAGE. Cached per format (vk_tex calls it
// on every lookup of an IMAGE texture that has no storage usage, e.g. its compressed material textures)
static int img_storage_ok(const vkfmt_t *f, int *via_twin) {
    static signed char cache[256];   // 0 unknown, 1 no, 2 own format, 3 twin
    *via_twin = 0; if (f->depth || f->stencil) return 0;
    int c = (unsigned)f->vk < 256 ? cache[f->vk] : 0;
    if (!c) {
        VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, f->vk, &fp); c = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT ? 2 : 1;
        VkFormat tw = f->compressed ? VK_FORMAT_UNDEFINED : srgb_twin(f->vk);
        if (c == 1 && tw) { vkGetPhysicalDeviceFormatProperties(V.pd, tw, &fp); if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) c = 3; }
        if ((unsigned)f->vk < 256) cache[f->vk] = (signed char)c;
    }
    *via_twin = c == 3; return c >= 2;
}
// ootu32 (11-20, Ocarina of Time black 3D): its upscale network's first pass (a335c1db6270c355) packs the scene into a storage image of NVN format 0x40 (E5B9G9R9) with
// typed stores (the shader declares R32_UINT), and the next pass (d99fbeeb53d38847) samples it into the 2560x1440 layer. E5B9G9R9 has no storage support here (N1X), so the
// image got no STORAGE usage, the store was dropped and the network read zeros. As Eden's StorageView: a typed store takes a view in the format the shader declares (same
// texel size; the image is MUTABLE_FORMAT + EXTENDED_USAGE), and an image whose own format cannot be stored gets STORAGE usage when such a uint format can
int g_e5_storage_used;   // a declared-format storage view was made over an E5B9G9R9 image (gfx_vk_exec.c oot_bilerp_rt: the network's own path is live)
_Thread_local int g_vk_storage_want;   // R1 (parallel recording): per thread. The declared format (Shader::ImageFormat, 0 typeless) of the storage binding being looked up (gfx_vk_eden.inc eden_image)
static VkFormat typed_fmt(int k) { static const VkFormat m[8] = {VK_FORMAT_UNDEFINED, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R32_UINT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32B32A32_UINT}; return k > 0 && k < 8 ? m[k] : VK_FORMAT_UNDEFINED; }
static unsigned typed_bytes(int k) { static const unsigned b[8] = {0, 1, 1, 2, 2, 4, 8, 16}; return k > 0 && k < 8 ? b[k] : 0; }
static int img_typed_storage_ok(const vkfmt_t *f) {
    static signed char cache[256]; int c = (unsigned)f->vk < 256 ? cache[f->vk] : 0;
    if (!c) {
        VkFormat u = f->depth || f->stencil || f->compressed ? VK_FORMAT_UNDEFINED : f->bpb == 1 ? VK_FORMAT_R8_UINT : f->bpb == 2 ? VK_FORMAT_R16_UINT : f->bpb == 4 ? VK_FORMAT_R32_UINT :
                     f->bpb == 8 ? VK_FORMAT_R32G32_UINT : f->bpb == 16 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_UNDEFINED;
        VkFormatProperties fp = {0}; if (u) vkGetPhysicalDeviceFormatProperties(V.pd, u, &fp); c = fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT ? 2 : 1;
        if ((unsigned)f->vk < 256) cache[f->vk] = (signed char)c;
    }
    return c == 2;
}
int nvn_sparse_any(const void *pool, uint64_t a, uint64_t n); uint64_t nvn_sparse_gen(void); uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n);
int nvn_sparse_since(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp);
// NVN2 (Switch 2 boots): a texture in a vmode-2 (sparse) virtual pool. Its storage address lies in the pool's own range, which on Windows never holds pages (hle_nvn.c
// vpool_map: the mapping is kept in the page table only), so the CPU reads it through the page table (nvn_sparse_copy: unmapped pages read 0), never directly
static int vp2_tex(const tex_t *t) { return dm_sdk_ounce() && t->pool && t->pool->vp && t->pool->vmode == 2 && t->mem; }
extern uint64_t g_tex_frame;   // (below)
static VkDeviceSize s2tile_cap(void) {   // Lichii SparseLayersCap (vk_texture_cache.cpp BindLayers): by the device-local memory
    static VkDeviceSize cap = 0; if (cap) return cap;
    const char *e = getenv("DM_VK_S2_SPARSE_MB"); if (e && atoll(e) > 0) return cap = (VkDeviceSize)atoll(e) << 20;
    VkDeviceSize local = 0; for (uint32_t i = 0; i < V.mem.memoryHeapCount; i++) if (V.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) local += V.mem.memoryHeaps[i].size;
    // The emu's rule takes Device::GetDeviceLocalMemory, which on an integrated GPU is at most 6 GiB (vulkan_device.cpp CollectPhysicalMemoryInfo: min(budget left - 8 GiB,
    // 4 or 6 GiB) against min(local heaps, 4 or 6 GiB)), so an iGPU always gets 384 MiB; the heaps' size alone (15.6 GiB here) gave 768. MKW 11-16: emu 364/364/341 MiB
    if (V.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && local > (6ull << 30)) local = 6ull << 30;
    return cap = local < (7ull << 30) ? (384ull << 20) : (768ull << 20);
}
// 1: the per-submission upload budget, shared by every sparse layered image (reset when a new upload command buffer begins, g_up_gen)
extern uint64_t g_up_gen;
static uint64_t g_s2tile_budget_gen = ~0ull, g_s2tile_budget_bytes;
static uint64_t s2tile_upload_budget(void) { static uint64_t b = 0; if (!b) { const char *e = getenv("DM_VK_S2_SPARSE_UP_MB"); b = (uint64_t)(e && atoll(e) > 0 ? atoll(e) : 128) << 20; } return b; }

// ---------------- NVN2 2D sparse tiles (the emu's UsesSparseTiles, video_core/renderer_vulkan/vk_texture_cache.cpp:250-262, InitSparseTiles :2394, BindTiles :2446) --------
// A big 2D texture in a sparse virtual pool (vmode 2: MKW's 8192x8192 / 16384x16384 single-layer maps, 341-342 MiB each, of which the game maps a few MiB) is an image with
// residency and no memory: each 64 KiB block under a 64 KB guest tile the game maps (the host page table, nvn_sparse_runs) gets memory, sub-allocated from 4 MiB chunks, and
// is uploaded from the page table (nvn_sparse_copy, the tile deswizzled on its own, nvn_deswizzle_tile); a block whose tile the game unmaps loses it (unbound, its memory
// freed after the current submission; a chunk goes with its last block). The mip tail (the levels from the driver's first tail level on) is bound whole when any page of its
// guest range is mapped, and uploaded level by level. Unbound blocks read as zeros (residencyNonResidentStrict, as the emu relies on). The emu's tiles cap per image: 384 MiB
// on an integrated GPU (s2tile_cap). Taken when the driver's block for the format is the guest tile in texels and the guest's tiled levels cover the driver's; else a whole
// image as before, logged once per reason. Switch 2 boots only. DM_VK_S2_TILES=0 off
// 11-16 (bak80, NPARK: "진짜 쓰는 놈만 실제크기로"): sparse-layer arrays (MKW's 8192x8192x68 ASTC / BC6H terrain, 1.4 / 5.8 GB each) as tiles too, per layer: each 64 KiB
// block under a mapped 64 KB guest tile of any layer gets memory, each layer's mip tail its own opaque range (the driver's imageMipTailOffset + layer x imageMipTailStride),
// in place of a whole layer up to the cap (s2sp). Our Turnip binds single tiles of compressed 2D arrays (_xfer\tile_probe\sparse_arr_test.c, 11-16: ASTC 8x8 and BC6H
// 8192x8192x4, 512x512 / 256x256 texel tiles, bound tiles read back exactly, unbound ones zero). The cap counts bound tile bytes. DM_VK_S2_LAYER_TILES=0: allocate whole arrays
struct s2tile_lv { uint32_t tiles_x, tiles_y, trb, trows, tw, th, first; uint64_t base; };   // per tiled level: tile grid, a tile's bytes a row and block rows, texels, first index; base: layer 0's
struct s2tile_ch { VkDeviceMemory m; uint32_t used; uint64_t mask; };
struct s2tile {
    uint32_t ht, levels, ntiles, nslots, npend; struct s2tile_lv lv[16]; VkDeviceSize blk, cap, bound; uint32_t type, bpb, bw, bh;
    uint32_t nl, ntl; uint64_t lsize;   // layers, tiles a layer (ntiles = nl x ntl, layer-major), the guest's layer stride (t->L.layer_size)
    uint32_t *slot; uint64_t *gen, *stamp; uint8_t *need;   // per tile: 0 unbound, else chunk * nslots + slot + 1; the page-table generation and dirty stamp of its upload; to upload
    struct s2tile_ch *ch; uint32_t nch;
    VkDeviceSize tail_off, tail_size, tail_stride; uint64_t tail_lo; VkDeviceMemory *tails; uint64_t *tgen, *tstamp; uint8_t *tneed;   // per layer: the mip tail's memory, upload stamps, to upload
    int warned, more, gw;   // more: a bind call reached its limit (the rest at the next look)
    uint64_t seen_gen, seen_dirty, seen_frame, log_ns;   // log_ns: its last "bound" line (s2tile_bind)
    // bak83: per tile, the shared physical slot it maps (ps_*, 1-based, 0: its own chunk slot or unbound) and whether this image allocated that slot (mine: counted in own)
    uint32_t *ps; uint8_t *mine; uint64_t sig, obj; uint64_t sw[7];   // sw: the sig's fields (bak84 3c)
    uint64_t tsig;   // bak87: what tiles must have alike to share (format, create flags, usage; extent, levels and layers too with DM_VK_S2_ALIAS_EXTENT=0); tails keep sig
    uint8_t *refused; uint32_t ncapped, capref_new; uint64_t seen_psgen, caplog_ns;   // bak84 3a: tiles the cap refused (retried when slots appear), refusals since the last cap line
    uint32_t *tps; uint8_t *tmine;   // bak84 3b: per layer, the shared mip tail it maps (pt_*, 1-based, 0: its own) and whether this image made it   // sig: what images must have alike to share (format, extent, levels, layers, flags, usage); obj: the texture object
    VkDeviceSize own, priv;   // own: bytes this image allocated (its own slots, the shared slots it made, its tails), what the cap counts; priv: its own slots and tails (g_s2t_bytes)
};
// bak83 (11-16, MKW 14:48: six 8192x8192x68 terrain arrays, three pairs of virtual pools the game maps in lockstep, 1946 MiB of tiles at 99% of the heap): on hardware a
// sparse texture's content is the physical memory mapped there, and the game maps one physical page at several places (nvn_mtl.m sp_share, the Metal path, found the same:
// about three places a page in a race, its BC6H and ASTC 8x8 terrain arrays within and between themselves). A fully covered tile whose 16 pages are consecutive physical
// memory (nvn_sparse_tile_phys) takes the slot of that memory and format: every place that maps it binds the same 64 KiB of device memory (aliased residency), so one copy
// where there were two or three, and a GPU write through one place shows at the others. Images share only when created alike (sig; else counted as not shared). Until the
// GPU writes through an image that maps it (drawn), a slot is uploaded by its first place and by CPU writes after; a drawn slot is never uploaded again by a new place. A slot
// goes with the last place. Edge tiles (cut by the level's size), tiles of scattered pages and the mip tails keep memory of their own. Switch 2 boots; DM_VK_S2_PHYS_SHARE=0 off
struct ps_slot { const uint8_t *p; uint64_t sig; int fmt; uint32_t type, v, rc; uint8_t drawn, up, live; uint64_t obj, sw[7]; };   // obj, sw: the image that made it (bak84 3c)
typedef struct { uintptr_t p; int64_t fmt; } ps_key;
static struct ps_slot *g_ps; static uint32_t g_ps_n, g_ps_cap, *g_ps_free, g_ps_nfree;
static nvn_hmap g_ps_map = {.klen = sizeof(ps_key)};
static pthread_mutex_t g_ps_mu = PTHREAD_MUTEX_INITIALIZER;
static struct s2tile *g_ps_hold[32];   // per memory type: the chunks the shared slots come from (s2tile_slot_new / _free on a holder of nslots 64 KiB slots)
static uint64_t g_ps_live, g_ps_shared_binds, g_ps_unique, g_ps_diff, g_ps_private;
static uint64_t g_ps_made, g_ps_caprefused, g_ps_cap_later, g_ps_xext;   // g_ps_xext (bak87): shared binds with an image of another extent, levels or layers
// bak87 (11-16, _xfer/tile_probe/alias_extent.txt on our Turnip: ALL PASS): one 64 KiB block bound into two sparse images of one format holds the same texels
// whatever their extent, levels or layers (3072x3072x2 level 0 against 6144x6144x2 levels 0 and 1, both ways, sampled-only and with storage + colour attachment
// usage): the standard block shape (residencyStandard2DBlockShape, required here) lays a block out alike. So tiles share by format, create flags and usage alone,
// as Metal keys its heap tiles (nvn_mtl.m sp_share: MKW's RGBA8 SNORM 3072^2 and 6144^2 arrays, which the GPU draws into, map the same pages). Mip tails, whose
// layout depends on the image, keep the whole create info. DM_VK_S2_ALIAS_EXTENT=0: tiles of the same create info only, as bak83
static int ps_xext_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S2_ALIAS_EXTENT"); v = !(e && *e == '0');
    VKLOG("NVN2 sparse tiles: tiles shared across extents, levels and layers %s (DM_VK_S2_ALIAS_EXTENT=%s)", v ? "on" : "off", e && *e ? e : "unset"); } return v; }   // bak84 3a: slots made so far (a capped image looks again when it moves); cap refusals; refused tiles bound later through a shared slot
// bak84 3b (11-16, NPARK: terrain still flickers on bak83): the mip tails of images made alike are laid out alike, so a layer's tail whose guest bytes are one run of
// consecutive physical memory takes the tail of that memory, format, create info and size: a GPU write into a tail level through one place shows at the others, as in the
// tiles. A tail of scattered pages, or of an image made otherwise, keeps its own. DM_VK_S2_TAIL_SHARE=0: every tail its own
struct pt_tail { const uint8_t *p; uint64_t sig, size; int fmt; VkDeviceMemory m; uint32_t rc; uint8_t drawn, up, live; };
typedef struct { uintptr_t p; int64_t fmt; uint64_t sig, size; } pt_key;
static struct pt_tail *g_pt; static uint32_t g_pt_n, g_pt_cap, *g_pt_free, g_pt_nfree;
static nvn_hmap g_pt_map = {.klen = sizeof(pt_key)};
static uint64_t g_pt_live, g_pt_live_bytes, g_pt_shared_binds, g_pt_unique, g_pt_private;
int vk_sparse_aliased_ok(void);
static uint32_t s2tile_slot_new(struct s2tile *s); static void s2tile_slot_free(struct s2tile *s, uint32_t v, int now);
static int ps_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_S2_PHYS_SHARE"); const int want = e && *e ? atoi(e) != 0 : dm_sdk_ounce() != 0; v = want && vk_sparse_aliased_ok();
        VKLOG("NVN2 sparse tiles: shared by physical memory %s (DM_VK_S2_PHYS_SHARE=%s, %s boot)", v ? "on" : want ? "off: aliased residency not available" : "off",
              e && *e ? e : "unset", dm_sdk_ounce() ? "a Switch 2" : "a Switch 1"); }
    return v;
}
static uint32_t ps_find(const uint8_t *p, int fmt) {   // caller holds g_ps_mu; 1-based, 0 none
    ps_key k; memset(&k, 0, sizeof k); k.p = (uintptr_t)p; k.fmt = fmt; return (uint32_t)(uintptr_t)nvn_hm_get(&g_ps_map, &k);
}
static uint32_t ps_new(const uint8_t *p, int fmt, uint64_t sig, uint32_t type, VkDeviceSize blk) {   // caller holds g_ps_mu; a new slot with its memory, rc 1; 0 when none
    if (blk != 65536 || type >= 32) return 0;
    struct s2tile *h = g_ps_hold[type];
    if (!h) { if (!(h = calloc(1, sizeof *h))) return 0; h->blk = blk; h->type = type; h->nslots = 64; g_ps_hold[type] = h; }
    const uint32_t v = s2tile_slot_new(h); if (!v) return 0;
    uint32_t id;
    if (g_ps_nfree) id = g_ps_free[--g_ps_nfree];
    else { if (g_ps_n == g_ps_cap) { const uint32_t nc = g_ps_cap ? 2 * g_ps_cap : 4096; struct ps_slot *ng = realloc(g_ps, nc * sizeof *ng); uint32_t *nf = realloc(g_ps_free, nc * sizeof *nf);
               if (ng) g_ps = ng; if (nf) g_ps_free = nf; if (!ng || !nf) { s2tile_slot_free(h, v, 1); return 0; } g_ps_cap = nc; }
           id = ++g_ps_n; }
    g_ps[id - 1] = (struct ps_slot){p, sig, fmt, type, v, 1, 0, 0, 1, 0, {0}}; g_ps_made++;
    ps_key k; memset(&k, 0, sizeof k); k.p = (uintptr_t)p; k.fmt = fmt; nvn_hm_put(&g_ps_map, &k, (void *)(uintptr_t)id);
    g_ps_live++; return id;
}
static void ps_mem(uint32_t id, VkDeviceMemory *m, VkDeviceSize *mo) {   // caller holds g_ps_mu
    const struct ps_slot *e = &g_ps[id - 1]; const struct s2tile *h = g_ps_hold[e->type];
    *m = h->ch[(e->v - 1) / h->nslots].m; *mo = (VkDeviceSize)((e->v - 1) % h->nslots) * h->blk;
}
// One place stops mapping the slot (now: it was never bound, freed at once; else after the current submission). Returns 1 when the slot went (its memory freed)
static int ps_release(uint32_t id, int now) {   // caller holds g_ps_mu
    struct ps_slot *e = &g_ps[id - 1]; if (!e->live || --e->rc) return 0;
    s2tile_slot_free(g_ps_hold[e->type], e->v, now);
    ps_key k; memset(&k, 0, sizeof k); k.p = (uintptr_t)e->p; k.fmt = e->fmt; nvn_hm_del(&g_ps_map, &k);
    e->live = 0; g_ps_free[g_ps_nfree++] = id; g_ps_live--; return 1;
}
static int pt_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_S2_TAIL_SHARE"); v = ps_on() && !(e && *e == '0');
        VKLOG("NVN2 sparse tiles: mip tails shared by physical memory %s (DM_VK_S2_TAIL_SHARE=%s)", v ? "on" : "off", e && *e ? e : "unset"); }
    return v;
}
static uint32_t pt_find(const uint8_t *p, int fmt, uint64_t sig, uint64_t size) {   // caller holds g_ps_mu
    pt_key k; memset(&k, 0, sizeof k); k.p = (uintptr_t)p; k.fmt = fmt; k.sig = sig; k.size = size; return (uint32_t)(uintptr_t)nvn_hm_get(&g_pt_map, &k);
}
static uint32_t pt_new(const uint8_t *p, int fmt, uint64_t sig, uint64_t size, VkDeviceMemory m) {   // caller holds g_ps_mu; takes m; 0 when the table cannot grow
    uint32_t id;
    if (g_pt_nfree) id = g_pt_free[--g_pt_nfree];
    else { if (g_pt_n == g_pt_cap) { const uint32_t nc = g_pt_cap ? 2 * g_pt_cap : 1024; struct pt_tail *ng = realloc(g_pt, nc * sizeof *ng); uint32_t *nf = realloc(g_pt_free, nc * sizeof *nf);
               if (ng) g_pt = ng; if (nf) g_pt_free = nf; if (!ng || !nf) return 0; g_pt_cap = nc; }
           id = ++g_pt_n; }
    g_pt[id - 1] = (struct pt_tail){p, sig, size, fmt, m, 1, 0, 0, 1};
    pt_key k; memset(&k, 0, sizeof k); k.p = (uintptr_t)p; k.fmt = fmt; k.sig = sig; k.size = size; nvn_hm_put(&g_pt_map, &k, (void *)(uintptr_t)id);
    g_pt_live++; g_pt_live_bytes += size; return id;
}
static int pt_release(uint32_t id, int now) {   // caller holds g_ps_mu; 1 when the tail went (its memory freed: at once, or after the current submission)
    struct pt_tail *e = &g_pt[id - 1]; if (!e->live || --e->rc) return 0;
    if (now) vkFreeMemory(V.dev, e->m, NULL); else vk_defer_free_buffer(VK_NULL_HANDLE, e->m);
    pt_key k; memset(&k, 0, sizeof k); k.p = (uintptr_t)e->p; k.fmt = e->fmt; k.sig = e->sig; k.size = e->size; nvn_hm_del(&g_pt_map, &k);
    e->live = 0; g_pt_free[g_pt_nfree++] = id; g_pt_live--; g_pt_live_bytes -= e->size; return 1;
}
static uint64_t g_s2t_images, g_s2t_tiles, g_s2t_bytes, g_s2t_up_bytes, g_s2t_ns;
void vk_tex_tiles_stats(uint64_t out[6]) { out[0] = g_s2t_images; out[1] = g_s2t_tiles; out[2] = g_s2t_bytes; out[3] = g_s2t_up_bytes; out[4] = 0; out[5] = g_s2t_ns; }
int nvn_sparse_runs(const void *pool, uint64_t a, uint64_t n, uint64_t *out, int max);
static int s2tile_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S2_TILES"); on = !(e && *e == '0'); } return on; }
static int s2tile_layers_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_S2_LAYER_TILES"); on = !(e && *e == '0'); } return on; }
static int s2tile_no_compressible(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_VK_S2_TILES_COMPRESSIBLE"); v = e && *e == '0'; } return v; }
static int s2tile_wanted(const tex_t *t, const vkfmt_t *f) {   // UsesSparseTiles' rule, with the host's sparse region (a vmode-2 pool); arrays (bak80) by s2tile_layers_on
    if (!dm_sdk_ounce() || !s2tile_on() || !t->pool || !t->pool->vp || t->pool->vmode != 2 || !t->mem) return 0;
    const int arr = t->d.target == NVN_TEXTURE_TARGET_2D_ARRAY && layers_of(&t->d) > 1;
    if (arr && !s2tile_layers_on()) return 0;
    if ((t->d.target != NVN_TEXTURE_TARGET_2D && !arr) || (!arr && layers_of(&t->d) != 1) || t->d.samples > 1 || f->depth || f->stencil || t->L.size < (32ull << 20)) return 0;
    // 11-16 (MKW: its six 341-342 MiB maps took whole memory with no line saying why): the same rules in the same order, each refusal named once per texture object
    const char *why = NULL; const uint32_t fl = (uint32_t)t->d.flags;
    if (fl & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) why = "a linear texture (LINEAR / LINEAR_RENDER_TARGET flag)";
    // Not a texture the GPU may write (an unbound tile drops writes): no storage-image use (IMAGE), display or video-decode target, render-target compression or Z-cull hints
    // (COMPRESSIBLE, ADAPTIVE_ZCULL), nor one the host has seen written (mtl_tex_gpu_owned). MKW's 342 MB maps are sampled only
    else if (fl & NVN_TEXTURE_FLAGS_IMAGE) why = "IMAGE flag (storage image use)";
    else if (fl & NVN_TEXTURE_FLAGS_DISPLAY) why = "DISPLAY flag";
    else if (fl & NVN_TEXTURE_FLAGS_VIDEO_DECODE) why = "VIDEO_DECODE flag";
    // (11-16, MKW: its six 341-342 MiB maps carry COMPRESSIBLE | SPARSE, flags 0x28, and are sampled only: the emu's UsesSparseTiles has no usage-flag rule and binds them by
    // tile, 2.7 GB -> 22 MiB. COMPRESSIBLE lets the driver compress the storage; it says nothing about GPU writes, which IMAGE, the GPU-owned check below and s2tile_refresh's
    // im->gpu_w watch cover. Kept as a refusal only with DM_VK_S2_TILES_COMPRESSIBLE=0)
    else if ((fl & NVN_TEXTURE_FLAGS_COMPRESSIBLE) && s2tile_no_compressible()) why = "COMPRESSIBLE flag (DM_VK_S2_TILES_COMPRESSIBLE=0)";
    else if (fl & NVN_TEXTURE_FLAGS_ADAPTIVE_ZCULL) why = "ADAPTIVE_ZCULL flag";
    else if (mtl_tex_gpu_owned(t->serial)) why = "the host has seen the GPU write it (mtl_tex_gpu_owned)";
    else if (!f->bpb || (!arr && (f->compressed || (f->bw && f->bw != 1) || (f->bh && f->bh != 1)))) why = "not a format of one texel per block";   // (the emu's rule; arrays: block formats too)
    else if (!vk_sparse_tiles_ok()) why = "sparse tiles not available on this device";
    if (!why) return 1;
    static uint64_t said[64]; static int nsaid; int known = 0;
    for (int i = 0; i < nsaid && !known; i++) known = said[i] == t->serial;
    if (!known && nsaid < 64) { said[nsaid++] = t->serial;
        VKLOG("[RED] NVN2 sparse tiles: %ux%u nvn format %#x (vk %d), %.1f MiB, flags %#x, texture object %llu: refused, %s; a whole image instead", t->d.width, t->d.height, t->d.format,
              f->vk, t->L.size / 1048576.0, fl, (unsigned long long)t->serial, why); }
    return 0;
}
static void s2tile_why(const vkimg_t *im, const char *why) { static int n; if (n++ < 16) VKLOG("[RED] NVN2 sparse tiles: %ux%ux%u format %#x: %s; %s instead", im->w, im->h, im->layers, im->nvn_format, why,
    "a whole image"); }
static void s2tile_arrays_free(struct s2tile *s) { free(s->slot); free(s->gen); free(s->stamp); free(s->need); free(s->tails); free(s->tgen); free(s->tstamp); free(s->tneed); free(s->ch); free(s->ps); free(s->mine); free(s->refused); free(s->tps); free(s->tmine); free(s); }
static struct s2tile *s2tile_init(vkimg_t *im, const tex_t *t, const vkfmt_t *f, const VkMemoryRequirements *req) {
    uint32_t nr = 8; VkSparseImageMemoryRequirements sr[8]; vk_image_sparse_reqs(im->img, &nr, sr);
    const VkSparseImageMemoryRequirements *r = NULL; for (uint32_t i = 0; i < nr; i++) if (sr[i].formatProperties.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) r = &sr[i];
    if (!r) { s2tile_why(im, "no colour sparse requirements"); return NULL; }
    const VkExtent3D g = r->formatProperties.imageGranularity; const uint32_t bpb = f->bpb, bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1, nl = im->layers ? im->layers : 1;
    { static int nl_; if (nl_++ < 8) VKLOG("NVN2 sparse tiles: format vk %d (%u bytes a %ux%u block): driver block %ux%ux%u, %llu bytes, mip tail from level %u (%llu bytes at %#llx, stride %#llx)%s", f->vk, bpb,
        bw, bh, g.width, g.height, g.depth, (unsigned long long)req->alignment, r->imageMipTailFirstLod, (unsigned long long)r->imageMipTailSize, (unsigned long long)r->imageMipTailOffset,
        (unsigned long long)r->imageMipTailStride, (r->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT) ? ", NOT the standard block" : ""); }
    if (!g.width || !g.height || g.depth != 1 || g.width % bw || g.height % bh || (uint64_t)(g.width / bw) * (g.height / bh) * bpb != req->alignment || req->alignment > (4u << 20)) {
        s2tile_why(im, "the driver's block is not one 64 KiB page"); return NULL; }
    if (nl > 1 && (r->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT)) { s2tile_why(im, "one mip tail for all layers (SINGLE_MIPTAIL)"); return NULL; }
    uint32_t gtw, gth, gtd; const int gt = nvn_tex_sparse_tiles(&t->d, &gtw, &gth, &gtd);   // the guest's tiled levels and its 64 KB tile in texels
    if (gt <= 0) { s2tile_why(im, "the texture is not laid out in sparse tiles"); return NULL; }
    if (gtw != g.width || gth != g.height) { s2tile_why(im, "the guest tile and the driver's block differ in shape"); return NULL; }
    const uint32_t levels = im->levels < 16 ? im->levels : 16, ht = r->imageMipTailFirstLod < levels ? r->imageMipTailFirstLod : levels;
    if ((uint32_t)gt < ht) { s2tile_why(im, "the driver tiles levels the guest does not"); return NULL; }
    struct s2tile *s = calloc(1, sizeof *s); if (!s) return NULL;
    s->ht = ht; s->levels = levels; s->blk = req->alignment; s->cap = s2tile_cap(); s->bpb = bpb; s->bw = bw; s->bh = bh; s->nslots = (uint32_t)((4u << 20) / s->blk); if (s->nslots > 64) s->nslots = 64;
    s->nl = nl; s->lsize = nl > 1 ? t->L.layer_size : 0;
    s->type = vk_mem_type(req->memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    for (uint32_t lv = 0; lv < ht; lv++) {
        uint32_t row_bytes, rows, slices, b; uint64_t base; unsigned th, td;
        if (nvn_deswizzle_geom(&t->d, lv, 0, &row_bytes, &rows, &slices, &base, &th, &td, &b) || td || slices != 1 || th > 7) { free(s); s2tile_why(im, "a tiled level's geometry is not one of 2D tiles"); return NULL; }
        if (nl > 1) { uint32_t rb1, r1, sl1, b1; uint64_t base1; unsigned th1, td1;   // layer 1 at layer 0 + the layer stride (the layout the uploads assume)
            if (nvn_deswizzle_geom(&t->d, lv, 1, &rb1, &r1, &sl1, &base1, &th1, &td1, &b1) || base1 != base + s->lsize) { free(s); s2tile_why(im, "its layers are not at one stride"); return NULL; } }
        struct s2tile_lv *L = &s->lv[lv]; L->trb = (128u >> th) * 64; L->trows = 8u << th; L->tw = L->trb / bpb * bw; L->th = L->trows * bh;   // (texels: blocks x block size)
        if (L->tw != g.width || L->th != g.height) { free(s); s2tile_why(im, "a tiled level's tile differs from the driver's block"); return NULL; }
        L->tiles_x = (row_bytes + L->trb - 1) / L->trb; L->tiles_y = (rows + L->trows - 1) / L->trows; L->base = base; L->first = s->ntl; s->ntl += L->tiles_x * L->tiles_y;
    }
    s->ntiles = s->ntl * nl;
    s->tail_off = r->imageMipTailOffset; s->tail_stride = nl > 1 ? r->imageMipTailStride : 0; s->tail_size = ht < levels ? r->imageMipTailSize : 0;
    s->tail_lo = ht < levels ? nvn_tex_level_offset(&t->d, ht) : 0;   // (within a layer)
    const size_t nt = s->ntiles ? s->ntiles : 1;
    s->slot = calloc(nt, sizeof *s->slot); s->gen = calloc(nt, sizeof *s->gen); s->stamp = calloc(nt, sizeof *s->stamp); s->need = calloc(nt, 1);
    s->tails = calloc(nl, sizeof *s->tails); s->tgen = calloc(nl, sizeof *s->tgen); s->tstamp = calloc(nl, sizeof *s->tstamp); s->tneed = calloc(nl, 1); s->seen_gen = ~0ull;
    s->ps = calloc(nt, sizeof *s->ps); s->mine = calloc(nt, 1);   // bak83
    s->refused = calloc(nt, 1); s->tps = calloc(nl, sizeof *s->tps); s->tmine = calloc(nl, 1);   // bak84
    if (!s->slot || !s->gen || !s->stamp || !s->need || !s->tails || !s->tgen || !s->tstamp || !s->tneed || !s->ps || !s->mine || !s->refused || !s->tps || !s->tmine) { s2tile_arrays_free(s); return NULL; }
    { uint64_t h = 0xcbf29ce484222325ull; const uint64_t w[7] = {(uint64_t)im->fmt, im->w, im->h, im->levels, im->layers, (uint64_t)im->cflags, (uint64_t)im->usage};
      for (int i = 0; i < 7; i++) { h = (h ^ w[i]) * 0x100000001b3ull; s->sw[i] = w[i]; } s->sig = h; s->obj = t->serial;
      uint64_t ht = 0xcbf29ce484222325ull; const int idx[3] = {0, 5, 6}; for (int i = 0; i < 3; i++) ht = (ht ^ w[idx[i]]) * 0x100000001b3ull;   // bak87
      s->tsig = ps_xext_on() ? ht : h; }
    g_s2t_images++;
    static int n; if (n++ < 24) VKLOG("NVN2 sparse tiles: image %ux%ux%u, %u levels (%u tiled), format %#x (vk %d): %u tiles of %ux%u (%u a layer), bound as the game maps them (cap %llu MiB an image)",
        im->w, im->h, nl, levels, ht, t->d.format, f->vk, s->ntiles, g.width, g.height, s->ntl, (unsigned long long)(s->cap >> 20));
    return s;
}
static void s2tile_slot_free(struct s2tile *s, uint32_t v, int now) {   // v: a tile's slot (1-based); now: never bound (a failed bind), freed at once
    uint32_t c = (v - 1) / s->nslots, i = (v - 1) % s->nslots; struct s2tile_ch *ch = &s->ch[c];
    ch->mask &= ~(1ull << i); if (--ch->used) return;
    if (now) vkFreeMemory(V.dev, ch->m, NULL); else vk_defer_free_buffer(VK_NULL_HANDLE, ch->m);
    ch->m = VK_NULL_HANDLE; ch->mask = 0;
}
static uint32_t s2tile_slot_new(struct s2tile *s) {   // a free 64 KiB slot (1-based), 0 when no memory
    for (uint32_t c = 0; c < s->nch; c++) { struct s2tile_ch *ch = &s->ch[c]; if (!ch->m || ch->used == s->nslots) continue;
        for (uint32_t i = 0; i < s->nslots; i++) if (!(ch->mask >> i & 1)) { ch->mask |= 1ull << i; ch->used++; return c * s->nslots + i + 1; } }
    uint32_t c = 0; while (c < s->nch && s->ch[c].m) c++;
    if (c == s->nch) { struct s2tile_ch *nc = realloc(s->ch, (size_t)(s->nch + 8) * sizeof *nc); if (!nc) return 0; memset(nc + s->nch, 0, 8 * sizeof *nc); s->ch = nc; s->nch += 8; }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, s->blk * s->nslots, s->type};
    if (s->type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &s->ch[c].m)) { s->ch[c].m = VK_NULL_HANDLE; VKRED_ONCE(179, "NVN2 sparse tiles: out of device memory for a 4 MiB chunk"); return 0; }
    s->ch[c].mask = 1; s->ch[c].used = 1; return c * s->nslots + 1;
}
static void s2tile_free(vkimg_t *im) {   // with the image (img_drop): every chunk and the tails once the GPU is past the current submission
    struct s2tile *s = im->s2t; if (!s) return;
    for (uint32_t c = 0; c < s->nch; c++) if (s->ch[c].m) vk_defer_free_buffer(VK_NULL_HANDLE, s->ch[c].m);
    for (uint32_t l = 0; l < s->nl; l++) if (s->tails[l] && !s->tps[l]) vk_defer_free_buffer(VK_NULL_HANDLE, s->tails[l]);   // (bak84: a shared tail goes with its last place, below)
    for (uint32_t k = 0; k < s->ntiles; k++) if (s->slot[k]) g_s2t_tiles--;
    pthread_mutex_lock(&g_ps_mu);   // bak83: its places in shared slots; a slot goes with its last place
    for (uint32_t k = 0; k < s->ntiles; k++) if (s->ps[k] && ps_release(s->ps[k], 0)) g_s2t_bytes -= s->blk;
    for (uint32_t l = 0; l < s->nl; l++) if (s->tps[l] && pt_release(s->tps[l], 0)) g_s2t_bytes -= s->tail_size;
    pthread_mutex_unlock(&g_ps_mu);
    g_s2t_bytes -= s->priv;   // (bak83: its own slots and tails; shared slots are counted once, by ps)
    s2tile_arrays_free(s); im->s2t = NULL; g_s2t_images--;
}
static void s2tile_tile_xy(const struct s2tile *s, uint32_t k, uint32_t *ly, uint32_t *lv, uint32_t *tx, uint32_t *ty) {
    *ly = k / s->ntl; const uint32_t kl = k % s->ntl;
    uint32_t l = 0; while (l + 1 < s->ht && kl >= s->lv[l + 1].first) l++;
    uint32_t i = kl - s->lv[l].first; *lv = l; *tx = i % s->lv[l].tiles_x; *ty = i / s->lv[l].tiles_x;
}
// BindTiles: want[k] 1 = bind, -1 = unbind, for tiles; tails[l] 1 / -1 / 0 per layer. One vkQueueBindSparse, waited (vk_bind_sparse_wait), as s2sp. Returns the binds made
enum { S2T_SHARED = 0xFFFFFFFFu };   // bak83: slot[k] of a tile bound to a shared physical slot (its id in ps[k])
const uint8_t *nvn_sparse_tile_phys(const void *pool, uint64_t a); const uint8_t *nvn_sparse_range_phys(const void *pool, uint64_t a, uint64_t n);
static int s2tile_bind(vkimg_t *im, const tex_t *t, const signed char *want, const signed char *tails) {
    struct s2tile *s = im->s2t; enum { MAXB = 1024, MAXT = 256 };
    VkSparseImageMemoryBind *ib = malloc(MAXB * sizeof *ib); uint32_t *who = malloc(MAXB * sizeof *who), *was = malloc(MAXB * sizeof *was); uint32_t nb = 0;
    uint32_t *wps = malloc(MAXB * sizeof *wps), *nps = malloc(MAXB * sizeof *nps); uint8_t *nnew = malloc(MAXB);   // bak83: per bind, the shared slot before / after, made by this bind
    VkSparseMemoryBind *ob = malloc(MAXT * sizeof *ob); uint32_t *tl = malloc(MAXT * sizeof *tl); VkDeviceMemory *tm_old = malloc(MAXT * sizeof *tm_old); uint32_t nt = 0;
    if (!ib || !who || !was || !ob || !tl || !tm_old || !wps || !nps || !nnew) { free(ib); free(who); free(was); free(ob); free(tl); free(tm_old); free(wps); free(nps); free(nnew); return 0; }
    const int share = ps_on(); const uint64_t base = (uint64_t)(uintptr_t)t->mem;
    if (share) pthread_mutex_lock(&g_ps_mu);
    for (uint32_t k = 0; k < s->ntiles && nb < MAXB; k++) {
        if (!want[k] || (want[k] > 0) == (s->slot[k] != 0)) continue;
        uint32_t ly, lv, tx, ty; s2tile_tile_xy(s, k, &ly, &lv, &tx, &ty); const struct s2tile_lv *L = &s->lv[lv];
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, x = tx * L->tw, y = ty * L->th;
        if (x >= w || y >= h) continue;
        VkDeviceMemory m = VK_NULL_HANDLE; VkDeviceSize mo = 0; uint32_t v = 0, pid = 0; uint8_t made = 0;
        if (want[k] > 0) {
            const uint8_t *ph = share && x + L->tw <= w && y + L->th <= h ? nvn_sparse_tile_phys(t->pool, base + (uint64_t)ly * s->lsize + L->base + ((uint64_t)ty * L->tiles_x + tx) * 65536ull) : NULL;
            if (ph && (pid = ps_find(ph, (int)im->fmt)) && g_ps[pid - 1].sig != s->tsig) {   // the same memory in an image made otherwise: its own (bak87: tsig)
                const struct ps_slot *o = &g_ps[pid - 1]; g_ps_diff++;
                static uint64_t said[16][2]; static int nsaid; int known = 0;   // bak84 3c: which two texture objects, which fields, once a pair
                for (int i = 0; i < nsaid && !known; i++) known = said[i][0] == o->obj && said[i][1] == s->obj;
                if (!known && nsaid < 16) { said[nsaid][0] = o->obj; said[nsaid][1] = s->obj; nsaid++;
                    static const char *fn[7] = {"vk format", "width", "height", "levels", "layers", "create flags", "usage"}; char df[256]; int p = 0; df[0] = 0;
                    for (int i = 0; i < 7; i++) if (o->sw[i] != s->sw[i] && (!ps_xext_on() || i == 0 || i == 5 || i == 6) && p < (int)sizeof df - 48) p += snprintf(df + p, sizeof df - (size_t)p, " %s %#llx/%#llx", fn[i], (unsigned long long)o->sw[i], (unsigned long long)s->sw[i]);
                    VKLOG("[RED] NVN2 sparse tiles: phys share: texture objects %llu and %llu map the same memory but were made otherwise (%s):%s", (unsigned long long)o->obj, (unsigned long long)s->obj,
                          "theirs/this", p ? df : " (no field differs: the hash collided)"); }
                pid = 0; ph = NULL; }
            if (pid) { g_ps[pid - 1].rc++; ps_mem(pid, &m, &mo); v = S2T_SHARED; if (s->refused[k]) { s->refused[k] = 0; s->ncapped--; g_ps_cap_later++; }
                if (memcmp(&g_ps[pid - 1].sw[1], &s->sw[1], 4 * sizeof s->sw[0])) g_ps_xext++; }   // bak87: another extent, levels or layers   // another place maps this memory: the same device memory (the cap counts it nowhere new; bak84: even at the cap)
            else {
                if (s->own + s->blk > s->cap) { if (!s->warned) { s->warned = 1; VKLOG("[RED] NVN2 sparse tiles: image %ux%ux%u: %llu MiB bound (the cap), more tiles stay without memory (read as zeros)", im->w, im->h, s->nl, (unsigned long long)(s->bound >> 20)); }
                    g_ps_caprefused++; s->capref_new++; if (!s->refused[k]) { s->refused[k] = 1; s->ncapped++; } continue; }   // bak84 3a: looked at again when new slots appear (s2tile_refresh)
                if (ph && (pid = ps_new(ph, (int)im->fmt, s->tsig, s->type, s->blk))) { ps_mem(pid, &m, &mo); v = S2T_SHARED; made = 1; g_ps[pid - 1].obj = s->obj; memcpy(g_ps[pid - 1].sw, s->sw, sizeof s->sw); }
                else { pid = 0; if (!(v = s2tile_slot_new(s))) break; m = s->ch[(v - 1) / s->nslots].m; mo = (VkDeviceSize)((v - 1) % s->nslots) * s->blk; }
            }
        }
        ib[nb] = (VkSparseImageMemoryBind){{VK_IMAGE_ASPECT_COLOR_BIT, lv, ly}, {(int32_t)x, (int32_t)y, 0}, {w - x < L->tw ? w - x : L->tw, h - y < L->th ? h - y : L->th, 1}, m, mo, 0};
        who[nb] = k; was[nb] = s->slot[k]; wps[nb] = s->ps[k]; nps[nb] = pid; nnew[nb] = made; s->slot[k] = v; s->ps[k] = pid; nb++;
    }
    s->more = nb == MAXB;
    // bak84 3b: per tail bind, the shared tail before / after and whether this bind made it
    uint32_t ttid_old[MAXT], ttid[MAXT]; uint8_t tmade[MAXT];
    const int tshare = share && pt_on(); const uint64_t lsz = s->nl > 1 ? s->lsize : t->L.size;
    for (uint32_t l = 0; l < s->nl && tails && s->tail_size && nt < MAXT; l++) {
        const VkDeviceSize off = s->tail_off + (VkDeviceSize)l * s->tail_stride;
        if (tails[l] > 0 && !s->tails[l]) {
            VkDeviceMemory m = VK_NULL_HANDLE; uint32_t tid = 0; uint8_t made = 0;
            const uint8_t *ph = tshare && lsz > s->tail_lo ? nvn_sparse_range_phys(t->pool, base + (uint64_t)l * s->lsize + s->tail_lo, lsz - s->tail_lo) : NULL;
            if (ph && (tid = pt_find(ph, (int)im->fmt, s->sig, s->tail_size))) { g_pt[tid - 1].rc++; m = g_pt[tid - 1].m; }
            else {
                VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, s->tail_size, s->type};
                if (s->type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &m)) { VKRED_ONCE(180, "NVN2 sparse tiles: out of device memory for a mip tail (%llu KiB)", (unsigned long long)(s->tail_size >> 10)); continue; }
                if (ph && (tid = pt_new(ph, (int)im->fmt, s->sig, s->tail_size, m))) made = 1;
            }
            ob[nt] = (VkSparseMemoryBind){off, s->tail_size, m, 0, 0}; tl[nt] = l; tm_old[nt] = VK_NULL_HANDLE; ttid_old[nt] = s->tps[l]; ttid[nt] = tid; tmade[nt] = made; s->tails[l] = m; s->tps[l] = tid; nt++;
        } else if (tails[l] < 0 && s->tails[l]) { ob[nt] = (VkSparseMemoryBind){off, s->tail_size, VK_NULL_HANDLE, 0, 0}; tl[nt] = l; tm_old[nt] = s->tails[l]; ttid_old[nt] = s->tps[l]; ttid[nt] = 0; tmade[nt] = 0; s->tails[l] = VK_NULL_HANDLE; s->tps[l] = 0; nt++; }
    }
    if (share) pthread_mutex_unlock(&g_ps_mu);
    { const uint64_t ns = now_ns();   // bak84 3a: the cap's refusals, at most once every 10 s an image (was: once)
      if (s->capref_new && ns - s->caplog_ns >= 10000000000ull) { s->caplog_ns = ns;
          VKLOG("[RED] NVN2 sparse tiles: image %ux%ux%u format %#x, texture object %llu: the cap refused %u tiles in the last 10 s (%u waiting for a shared slot); %llu MiB its own of the cap's %llu",
                im->w, im->h, s->nl, im->nvn_format, (unsigned long long)s->obj, s->capref_new, s->ncapped, (unsigned long long)(s->own >> 20), (unsigned long long)(s->cap >> 20));
          s->capref_new = 0; } }
    if (!nb && !nt) { free(ib); free(who); free(was); free(ob); free(tl); free(tm_old); free(wps); free(nps); free(nnew); return 0; }
    VkSparseImageMemoryBindInfo ii = {im->img, nb, ib}; VkSparseImageOpaqueMemoryBindInfo oi = {im->img, nt, ob};
    VkBindSparseInfo bs = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, NULL, 0, NULL, 0, NULL, nt ? 1u : 0u, nt ? &oi : NULL, nb ? 1u : 0u, nb ? &ii : NULL, 0, NULL};
    int r = vk_bind_sparse_wait(&bs);
    if (r == -4) { VKRED_ONCE(181, "NVN2 sparse tiles: a vkQueueBindSparse did not finish in 2 s: kept as bound"); r = 0; }
    if (r) {   // never queued: as BindTiles' roll_back, the tiles and tails stay as they were, tried again at the next look
        VKRED_ONCE(182, "NVN2 sparse tiles: vkQueueBindSparse failed (%d)", r);
        if (share) pthread_mutex_lock(&g_ps_mu);
        for (uint32_t j = 0; j < nb; j++) { uint32_t k = who[j];
            if (nps[j]) ps_release(nps[j], 1); else if (s->slot[k]) s2tile_slot_free(s, s->slot[k], 1);   // (bak83: a shared slot this bind took or made goes back)
            s->slot[k] = was[j]; s->ps[k] = wps[j]; }
        for (uint32_t j = 0; j < nt; j++) { const uint32_t l = tl[j];   // (bak84: a shared tail this bind took or made goes back)
            if (tm_old[j]) { s->tails[l] = tm_old[j]; s->tps[l] = ttid_old[j]; }
            else { if (ttid[j]) pt_release(ttid[j], 1); else vkFreeMemory(V.dev, s->tails[l], NULL); s->tails[l] = VK_NULL_HANDLE; s->tps[l] = 0; } }
        if (share) pthread_mutex_unlock(&g_ps_mu);
        free(ib); free(who); free(was); free(ob); free(tl); free(tm_old); free(wps); free(nps); free(nnew); return 0; }
    if (share) pthread_mutex_lock(&g_ps_mu);
    for (uint32_t j = 0; j < nb; j++) { uint32_t k = who[j];
        if (s->slot[k]) { s->bound += s->blk; g_s2t_tiles++; int up = 1;   // bound: uploaded next (bak83: a shared slot only when no place filled it, nor the GPU drew it)
            if (nps[j]) { struct ps_slot *e = &g_ps[nps[j] - 1];
                if (nnew[j]) { s->own += s->blk; s->mine[k] = 1; g_s2t_bytes += s->blk; g_ps_unique++; } else { up = !e->drawn && !e->up; g_ps_shared_binds++; }
                if (s->gw) e->drawn = 1; }
            else { s->own += s->blk; s->priv += s->blk; g_s2t_bytes += s->blk; g_ps_private++; }
            if (up && !s->need[k]) { s->need[k] = 1; s->npend++; } else if (!up) { if (s->need[k]) { s->need[k] = 0; s->npend--; } s->gen[k] = nvn_sparse_gen(); s->stamp[k] = nvn_dirty_now(); } }   // (not uploaded: as of now)
        else { s->bound -= s->blk; g_s2t_tiles--; if (s->need[k]) { s->need[k] = 0; s->npend--; }
            if (wps[j]) { if (s->mine[k]) { s->own -= s->blk; s->mine[k] = 0; } if (ps_release(wps[j], 0)) g_s2t_bytes -= s->blk; }
            else { s2tile_slot_free(s, was[j], 0); s->own -= s->blk; s->priv -= s->blk; g_s2t_bytes -= s->blk; } } }
    for (uint32_t j = 0; j < nt; j++) { const uint32_t l = tl[j];
        if (!tm_old[j]) { s->bound += s->tail_size; int up = 1;
            if (ttid[j]) { struct pt_tail *e = &g_pt[ttid[j] - 1];   // bak84 3b: a shared tail is uploaded only when no place filled it, nor the GPU drew it
                if (tmade[j]) { s->own += s->tail_size; s->tmine[l] = 1; g_s2t_bytes += s->tail_size; g_pt_unique++; } else { up = !e->drawn && !e->up; g_pt_shared_binds++; }
                if (s->gw) e->drawn = 1; }
            else { s->own += s->tail_size; s->priv += s->tail_size; g_s2t_bytes += s->tail_size; g_pt_private++; }
            s->tneed[l] = (uint8_t)up; if (!up) { s->tgen[l] = nvn_sparse_gen(); s->tstamp[l] = nvn_dirty_now(); } }
        else { s->bound -= s->tail_size; s->tneed[l] = 0;
            if (ttid_old[j]) { if (s->tmine[l]) { s->own -= s->tail_size; s->tmine[l] = 0; } if (pt_release(ttid_old[j], 0)) g_s2t_bytes -= s->tail_size; }
            else { vk_defer_free_buffer(VK_NULL_HANDLE, tm_old[j]); s->own -= s->tail_size; s->priv -= s->tail_size; g_s2t_bytes -= s->tail_size; } } }
    if (share) pthread_mutex_unlock(&g_ps_mu);
    free(ib); free(who); free(was); free(ob); free(tl); free(tm_old); free(wps); free(nps); free(nnew);
    { const uint64_t ns = now_ns();   // 11-16: what each tiled image binds, at most once every 10 s an image (MKW's maps and, bak80, its terrain arrays)
      if (ns - s->log_ns >= 10000000000ull) { s->log_ns = ns;
        uint32_t touched = 0, ntb = 0, ntail = 0;
        for (uint32_t l = 0; l < s->nl; l++) { int any = s->tails[l] != VK_NULL_HANDLE; ntail += any; for (uint32_t i = 0; i < s->ntl; i++) if (s->slot[l * s->ntl + i]) { any = 1; ntb++; } touched += any != 0; }
        uint32_t nsh = 0; for (uint32_t k = 0; k < s->ntiles; k++) nsh += s->ps[k] != 0;
        VKLOG("NVN2 sparse tiles: image %ux%ux%u format %#x, texture object %llu, storage %#llx, looked up %llu presents ago: %llu KiB bound of %.1f MiB (tiles %u of %u, %u in shared slots, %llu KiB its own; "
              "layers touched %u of %u, mip tails %u)", im->w, im->h, s->nl, im->nvn_format, (unsigned long long)s->obj, (unsigned long long)im->gaddr, (unsigned long long)(g_tex_frame - im->used),
              (unsigned long long)(s->bound >> 10), ((double)s->ntiles * (double)s->blk + (double)s->tail_size * s->nl) / 1048576.0, ntb, s->ntiles, nsh, (unsigned long long)(s->own >> 10), touched, s->nl, ntail);
        static uint64_t ps_log_ns; if (ps_on() && ns - ps_log_ns >= 10000000000ull) { ps_log_ns = ns;
            VKLOG("NVN2 sparse tiles: phys share: slots %llu (%.0f MiB), shared binds %llu, unique %llu, not shared (create info differs) %llu, own slots (edge tiles, scattered pages) %llu; "
                  "cap refusals %llu, capped tiles bound later via shared slots %llu, shared across different extents %llu; mip tails: shared %llu (%.1f MiB), shared binds %llu, unique %llu, own %llu",
                  (unsigned long long)g_ps_live, g_ps_live * 65536.0 / 1048576.0, (unsigned long long)g_ps_shared_binds, (unsigned long long)g_ps_unique, (unsigned long long)g_ps_diff,
                  (unsigned long long)g_ps_private, (unsigned long long)g_ps_caprefused, (unsigned long long)g_ps_cap_later, (unsigned long long)g_ps_xext, (unsigned long long)g_pt_live, g_pt_live_bytes / 1048576.0,
                  (unsigned long long)g_pt_shared_binds, (unsigned long long)g_pt_unique, (unsigned long long)g_pt_private); } } }
    return (int)nb + (int)nt;
}
static int s2tile_upload_tile(vkimg_t *im, const tex_t *t, uint32_t k) {   // 0 = recorded
    struct s2tile *s = im->s2t; uint32_t ly, lv, tx, ty; s2tile_tile_xy(s, k, &ly, &lv, &tx, &ty); const struct s2tile_lv *L = &s->lv[lv];
    static uint8_t buf[65536];   // (the replay thread)
    uint64_t a = (uint64_t)(uintptr_t)t->mem + (uint64_t)ly * s->lsize + L->base + ((uint64_t)ty * L->tiles_x + tx) * 65536ull;
    nvn_sparse_copy(t->pool, buf, a, 65536);
    VkBuffer sb; VkDeviceSize so; uint8_t *dst = vk_scratch((VkDeviceSize)L->trb * L->trows, 16, &sb, &so); if (!dst) return -2;
    uint32_t wb = 0, rows = 0; if (nvn_deswizzle_tile(&t->d, lv, tx, ty, buf, dst, L->trb, &wb, &rows)) { VKRED_ONCE(183, "NVN2 sparse tiles: tile deswizzle refused format %#x", t->d.format); return -3; }
    if (!wb || !rows) return 0;
    VkCommandBuffer cb = vk_upload_cmd(); if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    const uint32_t lw = im->w >> lv ? im->w >> lv : 1, lh = im->h >> lv ? im->h >> lv : 1, x = tx * L->tw, y = ty * L->th;
    uint32_t ew = wb / s->bpb * s->bw, eh = rows * s->bh; if (ew > lw - x) ew = lw - x; if (eh > lh - y) eh = lh - y;   // (texels; a block format's edge blocks cover the level's last texels)
    VkBufferImageCopy r = {so, L->trb / s->bpb * s->bw, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {(int32_t)x, (int32_t)y, 0}, {ew, eh, 1}};
    vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    if (s->ps[k]) { pthread_mutex_lock(&g_ps_mu); g_ps[s->ps[k] - 1].up = 1; pthread_mutex_unlock(&g_ps_mu); }   // bak83: filled for every place that maps it
    g_s2t_up_bytes += 65536; return 0;
}
static int s2tile_upload_tail(vkimg_t *im, const tex_t *t, const vkfmt_t *f, uint32_t ly) {   // a layer's tail levels from the page table, level by level (as s2sp_upload_layer)
    struct s2tile *s = im->s2t; const uint64_t lsz = s->nl > 1 ? s->lsize : t->L.size, n = lsz > s->tail_lo ? lsz - s->tail_lo : 0; if (!n) return 0;
    const uint64_t at = (uint64_t)ly * s->lsize + s->tail_lo;   // the gathered bytes' place in the storage (nvn_deswizzle_from's src_at)
    uint8_t *buf = malloc(n); if (!buf) return -1;
    nvn_sparse_copy(t->pool, buf, (uint64_t)(uintptr_t)t->mem + at, n);
    VkCommandBuffer cb = vk_upload_cmd(); if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    for (unsigned lv = s->ht; lv < s->levels; lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1; size_t pitch = (size_t)((w + s->bw - 1) / s->bw) * f->bpb, slab = pitch * ((h + s->bh - 1) / s->bh);
        VkBuffer sb; VkDeviceSize so; uint8_t *dst = vk_scratch(slab, 16, &sb, &so); if (!dst) { free(buf); return -2; }
        if (nvn_deswizzle_from(&t->d, lv, ly, buf, at, dst, pitch)) { VKRED_ONCE(184, "NVN2 sparse tiles: tail deswizzle refused format %#x", t->d.format); free(buf); return -3; }
        VkBufferImageCopy r = {so, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }
    if (s->tps[ly]) { pthread_mutex_lock(&g_ps_mu); g_pt[s->tps[ly] - 1].up = 1; pthread_mutex_unlock(&g_ps_mu); }   // bak84: filled for every place that maps it
    g_s2t_up_bytes += n; free(buf); return 0;
}
// At each resolution in a submission (vk_tex), on lookup: a mapping request since the last look moves the tiles to what the page table maps (its runs, so the cost
// follows what is mapped); CPU writes since (at most once every 4 presents) mark the bound tiles written; the uploads share s2sp's per-submission budget
static void s2tile_refresh(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    struct s2tile *s = im->s2t; uint64_t g = nvn_sparse_gen(), now = nvn_dirty_now();
    const int mapped = s->seen_gen != g || (s->ncapped && s->seen_psgen != g_ps_made && g_tex_frame >= s->seen_frame + 4), look = mapped || (s->seen_dirty != now && g_tex_frame >= s->seen_frame + 4);
    int tpend = 0; for (uint32_t l = 0; l < s->nl && !tpend; l++) tpend = s->tneed[l] && s->tails[l];
    if (!look && !s->npend && !tpend) return;
    const uint64_t base = (uint64_t)(uintptr_t)t->mem, lsz = s->nl > 1 ? s->lsize : t->L.size;
    // Written by the GPU after all (an attachment, a copy or a storage write: gpu_w): left as it is, logged once. Its bound tiles keep what the GPU wrote (no re-upload from the
    // page table over it); writes to unbound tiles are dropped
    if (im->gpu_w && !s->gw) { s->gw = 1; static int n; if (n++ < 16) VKLOG("[RED] NVN2 sparse tiles: image %ux%ux%u format %#x is written by the GPU: kept as sparse tiles, not re-uploaded; writes to tiles without memory are dropped", im->w, im->h, s->nl, im->nvn_format);
        pthread_mutex_lock(&g_ps_mu); for (uint32_t k = 0; k < s->ntiles; k++) if (s->ps[k]) g_ps[s->ps[k] - 1].drawn = 1; for (uint32_t l = 0; l < s->nl; l++) if (s->tps[l]) g_pt[s->tps[l] - 1].drawn = 1; pthread_mutex_unlock(&g_ps_mu); }   // bak83: its shared slots hold what the GPU drew
    if (mapped) {
        signed char *want = calloc(s->ntiles ? s->ntiles : 1, 1), *tw = calloc(s->nl, 1); uint8_t *mp = calloc(s->ntiles ? s->ntiles : 1, 1);
        enum { MAXR = 4096 }; uint64_t *runs = malloc(2 * MAXR * sizeof *runs);
        if (!want || !tw || !mp || !runs) { free(want); free(tw); free(mp); free(runs); return; }
        for (uint32_t ly = 0; ly < s->nl; ly++) {
            const uint64_t lb = base + (uint64_t)ly * s->lsize; uint8_t *m = mp + (size_t)ly * s->ntl;
            for (uint32_t lv = 0; lv < s->ht; lv++) {
                const struct s2tile_lv *L = &s->lv[lv]; const uint64_t lbytes = (uint64_t)L->tiles_x * L->tiles_y * 65536ull;
                int nr = nvn_sparse_runs(t->pool, lb + L->base, lbytes, runs, MAXR);
                if (nr < 0) { for (uint32_t i = 0; i < L->tiles_x * L->tiles_y; i++) m[L->first + i] = (uint8_t)nvn_sparse_any(t->pool, lb + L->base + (uint64_t)i * 65536ull, 65536); continue; }   // too many runs: tile by tile
                for (int j = 0; j < nr; j++) for (uint64_t i = runs[2 * j] / 65536ull, e = (runs[2 * j] + runs[2 * j + 1] - 1) / 65536ull; i <= e && i < (uint64_t)L->tiles_x * L->tiles_y; i++) m[L->first + i] = 1;
            }
            if (s->tail_size) { const int any = nvn_sparse_any(t->pool, lb + s->tail_lo, lsz - s->tail_lo); tw[ly] = any && !s->tails[ly] ? 1 : !any && s->tails[ly] ? -1 : 0; }
        }
        for (uint32_t k = 0; k < s->ntiles; k++) { want[k] = mp[k] && !s->slot[k] ? 1 : !mp[k] && s->slot[k] ? -1 : 0; if (!mp[k] && s->refused[k]) { s->refused[k] = 0; s->ncapped--; } }
        s->seen_psgen = g_ps_made;   // bak84 3a: the slots the refused tiles were looked up against
        s2tile_bind(im, t, want, tw);
        free(want); free(tw); free(mp); free(runs);
    }
    if (look) {
        for (uint32_t k = 0; k < s->ntiles && !s->gw; k++) {
            if (!s->slot[k] || s->need[k]) continue;
            uint32_t ly, lv, tx, ty; s2tile_tile_xy(s, k, &ly, &lv, &tx, &ty); const struct s2tile_lv *L = &s->lv[lv];
            if (nvn_sparse_since(t->pool, base + (uint64_t)ly * s->lsize + L->base + ((uint64_t)ty * L->tiles_x + tx) * 65536ull, 65536, s->gen[k], s->stamp[k])) { s->need[k] = 1; s->npend++; } }
        for (uint32_t l = 0; l < s->nl && !s->gw; l++)
            if (s->tails[l] && !s->tneed[l] && nvn_sparse_since(t->pool, base + (uint64_t)l * s->lsize + s->tail_lo, lsz - s->tail_lo, s->tgen[l], s->tstamp[l])) s->tneed[l] = 1;
        s->seen_gen = s->more ? ~0ull : g; s->seen_dirty = now; s->seen_frame = g_tex_frame;   // (more tiles to move: looked at again)
    }
    if (g_up_gen != g_s2tile_budget_gen) { g_s2tile_budget_gen = g_up_gen; g_s2tile_budget_bytes = 0; }
    for (uint32_t l = 0; l < s->nl; l++) {
        if (!s->tneed[l] || !s->tails[l]) continue;
        if (g_s2tile_budget_bytes + (lsz - s->tail_lo) > s2tile_upload_budget() && g_s2tile_budget_bytes) break;
        uint64_t g0 = nvn_sparse_gen(), d0 = nvn_dirty_now();
        if (!s2tile_upload_tail(im, t, f, l)) { s->tneed[l] = 0; s->tgen[l] = g0; s->tstamp[l] = d0; g_s2tile_budget_bytes += lsz - s->tail_lo; }
        if (g_up_gen != g_s2tile_budget_gen) { g_s2tile_budget_gen = g_up_gen; g_s2tile_budget_bytes = lsz - s->tail_lo; }
    }
    for (uint32_t k = 0; k < s->ntiles && s->npend; k++) {
        if (!s->need[k] || !s->slot[k]) continue;
        if (g_s2tile_budget_bytes + 65536 > s2tile_upload_budget() && g_s2tile_budget_bytes) break;   // past this submission's budget (shared with s2sp): the next submissions take the rest
        uint64_t g0 = nvn_sparse_gen(), d0 = nvn_dirty_now();
        if (s2tile_upload_tile(im, t, k)) break;
        s->gen[k] = g0; s->stamp[k] = d0; s->need[k] = 0; s->npend--; g_s2tile_budget_bytes += 65536;
        if (g_up_gen != g_s2tile_budget_gen) { g_s2tile_budget_gen = g_up_gen; g_s2tile_budget_bytes = 65536; }
    }
}
static uint64_t g_img_n, g_img_bytes, g_img_evicted, g_img_mem_evicted;   // live images and their bytes, images evicted so far (vk_tex_sweep's line); g_img_mem_evicted: NVN2, for memory
static void img_drop(vkimg_t *im);
struct nvn_img_archive {
    FILE *file; uint64_t bytes, last_used, content_serial, touched, image_id;
    nvna_shape shape; vkimg_t *owner;
    struct nvn_img_archive *prev, *next;
    int resident;
};
enum { ARCHIVE_CLEAN_CACHE_BYTES = 512 << 20 };
static struct nvn_img_archive *g_archive_clean_first, *g_archive_clean_last;
static struct nvn_img_archive *g_archive_discard;
static uint64_t g_archive_discard_bytes;
static uint64_t g_archive_clean_bytes, g_archive_clean_releases, g_archive_clean_saved_bytes, g_archive_written_bytes;
static int archive_clean(const vkimg_t *im) {
    return im->archive && im->archive->owner==im && im->residency_id && im->archive->image_id==im->residency_id &&
        im->archive->content_serial!=UINT64_MAX && im->archive->content_serial == __atomic_load_n(&im->content_serial, __ATOMIC_RELAXED);
}
static uint64_t g_archive_bytes, g_archive_saved, g_archive_restored, g_archive_failed;
static uint64_t g_archive_save_ns, g_archive_restore_ns, g_trim_ns, g_trim_freed, g_trim_mixed, g_trim_yielded, g_residency_refaults;
static uint64_t g_archive_save_submits, g_archive_restore_submits;
static nvna_workspace g_archive_workspace;
static nvna_workspace g_preserve_workspace;
struct archive_save_job {
    nvnp_job job;
    struct archive_save_job *next;
    vkimg_t *owner;
    struct nvn_img_archive *archive;
    VkImage image, retired;
    VkDeviceMemory retired_memory; void *retired_sub; uint64_t retired_bytes, retired_fence;
    uint64_t elapsed_ns, submissions, job_id;
    const char *stage;
};
static struct archive_save_job *g_preserve_jobs;
static _Atomic int g_preserve_ready;
static _Atomic int g_preserve_closing;
static pthread_mutex_t g_preserve_lifecycle = PTHREAD_MUTEX_INITIALIZER;
static int g_preserve_draining;
static _Atomic int g_archive_attempted;
static uint64_t g_preserve_queued, g_preserve_complete, g_preserve_stale, g_preserve_failed;
static uint64_t g_preserve_written, g_preserve_ns, g_preserve_submits;
static _Thread_local int g_preserve_plan;
// Same-stack replay suspension and two-phase shutdown own the source leases.
// The OFF switch keeps the original F4 synchronous preservation path.
enum { PRESERVE_CONTINUATION_READY = 1 };
static int preserve_wanted(void) {
    if (!PRESERVE_CONTINUATION_READY) return 0;
    const char *stage=getenv("DM_VK_FOUNDATION_F4C"), *f4=getenv("DM_VK_FOUNDATION_F4");
    return dm_sdk_ounce() && !(stage && !strcmp(stage,"0")) && !(f4 && !strcmp(f4,"0"));
}
int vk_tex_preserve_enabled(void) { return preserve_wanted() && g_preserve_ready && !g_preserve_closing; }
static int preserve_cancelled(void *ctx) { return nvnp_changed(ctx); }
static int preserve_save(nvnp_job *base) {
    struct archive_save_job *j=(struct archive_save_job *)base;
    struct nvn_img_archive *a=j->archive;
    extern FILE *nvnh_image_archive_file(uint64_t bytes);
    uint64_t start=now_ns();
    // No owner lock is held here. The job pins the image record and VkImage;
    // a first write invalidates publication, including writes between batches.
    g_preserve_workspace.cancelled=preserve_cancelled;
    g_preserve_workspace.cancel_context=base;
    VkResult r=VK_NOT_READY;
    if (!nvnp_changed(base)) {
        a->file=nvnh_image_archive_file(a->bytes);
        if (a->file) r=nvna_stream_cached(&g_preserve_workspace,V.dev,V.q,V.qf,&V.mem,
            j->image,&a->shape,a->file,0,vk_queue_lock,vk_queue_unlock,&j->submissions);
        else {
            r=VK_ERROR_UNKNOWN; g_preserve_workspace.stage="create preservation file (local, no Vulkan call)";
        }
    }
    j->stage=g_preserve_workspace.stage;
    // A failed wait is not permission to release the source or reset this
    // command pool. Retain both until completion/device loss establishes safety.
    while (g_preserve_workspace.pending) {
        VkResult wait=nvna_workspace_wait(&g_preserve_workspace);
        if (wait!=VK_SUCCESS) r=wait;
        if (g_preserve_workspace.pending) { struct timespec pause={0,100000000}; nanosleep(&pause,NULL); }
    }
    j->elapsed_ns=now_ns()-start;
    if (r==VK_SUCCESS && nvnp_changed(base)) r=VK_NOT_READY;
    // Failed snapshots never become an archive. Disk cleanup also stays here.
    if (r!=VK_SUCCESS && a->file) { fclose(a->file); a->file=NULL; }
    return r;
}
static int preserve_enqueue(vkimg_t *im);
static VkImage g_archive_pending_image;
// At most one failed restore can still be in flight: the shared transfer
// workspace cannot be reused until its fence completes. Keep that allocation.
static struct { VkImage image; VkDeviceMemory memory; void *sub; uint64_t bytes; } g_archive_retired;
static VkResult archive_ready(void) {   // g_img_mu held
    VkResult r = nvna_workspace_wait(&g_archive_workspace);
    if (g_archive_workspace.pending) return r;
    if (g_archive_retired.image) {
        vk_image_release_completed(g_archive_retired.image, g_archive_retired.memory,
                                   g_archive_retired.sub, g_archive_retired.bytes);
        memset(&g_archive_retired, 0, sizeof g_archive_retired);
    }
    g_archive_pending_image = VK_NULL_HANDLE;
    if (r != VK_SUCCESS) return r;
    g_archive_workspace.non_coherent_atom = V.props.limits.nonCoherentAtomSize;
    return nvna_workspace_init(&g_archive_workspace, V.dev, V.qf, &V.mem, NVNA_STAGING_BYTES);
}
void vk_tex_archive_prepare(void) {
    if (atomic_load_explicit(&g_archive_attempted, memory_order_acquire)) return;
    int initializer=0;
    nvnr_lock(&g_img_mu);
    if (!atomic_load_explicit(&g_archive_attempted, memory_order_relaxed)) {
        initializer=1;
        VkResult r = archive_ready();
        VKLOG("NVN2 archive transfer reserve: result %d, %.1f MiB allocation, stage %s; reused across saves/restores",
            r, g_archive_workspace.allocation_bytes / 1048576.0,
            g_archive_workspace.stage ? g_archive_workspace.stage : "none");
        atomic_store_explicit(&g_archive_attempted, 1, memory_order_release);
    }
    nvnr_unlock(&g_img_mu);
    if (initializer && preserve_wanted()) {
        pthread_mutex_lock(&g_preserve_lifecycle);
        if (atomic_load_explicit(&g_preserve_closing,memory_order_acquire)) {
            pthread_mutex_unlock(&g_preserve_lifecycle); return;
        }
        // Called at sub-begin before submission/pool/image locks, while the
        // heap still has room. The persistent staging reservation is only 4 MiB.
        g_preserve_workspace.non_coherent_atom=V.props.limits.nonCoherentAtomSize;
#if defined(DM_NXN1X)
        nvnh_memory_policy(NVNH_POLICY_ENABLE,1);
        g_preserve_workspace.allocate=nvnh_memory_allocate_optional;
#endif
        VkResult r=nvna_workspace_init(&g_preserve_workspace,V.dev,V.qf,&V.mem,NVNA_STAGING_BYTES);
        if (r==VK_SUCCESS) {
            g_preserve_ready=nvnp_start(preserve_save);
            if (!g_preserve_ready) nvna_workspace_destroy(&g_preserve_workspace);
        }
        VKLOG("NVN2 F4C preservation: active=%d result=%d staging=%llu queue_bytes=%u jobs=%u; failed setup retains synchronous fallback",
            g_preserve_ready,r,(unsigned long long)g_preserve_workspace.allocation_bytes,NVNP_BYTES,NVNP_JOBS);
        pthread_mutex_unlock(&g_preserve_lifecycle);
    }
}
void nvnh_vk_archive_shutdown(void) {
    atomic_store_explicit(&g_preserve_closing,1,memory_order_release);
    pthread_mutex_lock(&g_preserve_lifecycle);
    nvnp_stop();
    nvna_workspace_destroy(&g_preserve_workspace);
    pthread_mutex_unlock(&g_preserve_lifecycle);
    // The archive worker has released its queue ownership. Other completion
    // domains retain their pins; never infer presentation completion here or
    // wait the shared device while the renderer can still submit work.
    vk_tex_preserve_poll(0);
    if (g_preserve_jobs)
        VKLOG("NVN2 F4C shutdown: completed worker sources with outstanding consumers retained until device teardown");
    nvnr_lock(&g_img_mu);
    // Do not initialize a reserve on shutdown of an unused host device.
    VkResult r = nvna_workspace_wait(&g_archive_workspace);
    if (!g_archive_workspace.pending) {
        if (g_archive_retired.image) {
            vk_image_release_completed(g_archive_retired.image, g_archive_retired.memory,
                                       g_archive_retired.sub, g_archive_retired.bytes);
            memset(&g_archive_retired, 0, sizeof g_archive_retired);
        }
        nvna_workspace_destroy(&g_archive_workspace);
        g_archive_pending_image = VK_NULL_HANDLE;
    } else {
        VKLOG("[RED] NVN2 archive reserve still pending at shutdown (%d); retained until device teardown", r);
    }
    nvnr_unlock(&g_img_mu);
}
int nvnh_vk_archive_finish_shutdown(int device_idle) {
    if (!atomic_load_explicit(&g_preserve_ready,memory_order_acquire)) return 1;
    if (!device_idle) return 0; // never manufacture presentation/CPU completion
    // The renderer has stopped its scheduler and completed its existing device
    // wait. Native completion still includes CPU copyback and event publication.
    // No owner/submission mutex is held during this final drain.
    if (!mtl_wait(mtl_last_fence(),UINT64_MAX)) return 0;
    g_preserve_draining=1;
    vk_tex_preserve_poll(0);
    g_preserve_draining=0;
    if (g_preserve_jobs) return 0;
    atomic_store_explicit(&g_preserve_ready,0,memory_order_release);
    return 1;
}
static VkResult archive_transfer(VkImage image, const nvna_shape *shape, FILE *file, int upload) {
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_TRANSFER, nvna_size(shape));
    // A restore's allocation trim may itself have saved another image since
    // its first readiness check. Do not confuse that image's failed wait with
    // a transfer of this image, or overwrite the pending-image owner.
    VkResult r = archive_ready();
    if (r != VK_SUCCESS) { vk_f0_end(&diag, r); return r; }
    r = nvna_stream_cached(&g_archive_workspace, V.dev, V.q, V.qf, &V.mem,
        image, shape, file, upload, vk_queue_lock, vk_queue_unlock,
        upload ? &g_archive_restore_submits : &g_archive_save_submits);
    g_archive_pending_image = g_archive_workspace.pending ? image : VK_NULL_HANDLE;
    if (r < 0 && g_archive_workspace.stage) {
        if (!strncmp(g_archive_workspace.stage, "vkEndCommandBuffer", 18))
            vk_f0_result(r, upload ? "vkEndCommandBuffer(archive-restore)" : "vkEndCommandBuffer(archive-save)", 0);
        else if (!strcmp(g_archive_workspace.stage, "vkQueueSubmit"))
            vk_f0_result(r, upload ? "vkQueueSubmit(archive-restore)" : "vkQueueSubmit(archive-save)", 0);
    }
    vk_f0_end(&diag, r);
    return r;
}
static uint64_t g_trim_hot, g_trim_pressure_calls, g_trim_oom_calls;
static uint64_t g_restore_reuse_slots, g_restore_reuse_bytes;
static uint64_t img_archive_release(vkimg_t *im);
static int img_archive_restore(vkimg_t *im);
static void img_archive_close(vkimg_t *im);
static void vk_tex_view_drop_completed(vkimg_t *im);
static uint64_t img_release_completed(vkimg_t *im);
// Under g_img_mu: reclaim completed idle images. NVN2 uses LRU and preserves GPU
// contents in file backing; guest-authoritative packaged images can be rebuilt.
// Returns actual allocations freed, not the guest layout's logical size.
uint64_t vk_fence_done(void);
// gpuowndiag (11-21, log only): textures gpuown re-uploaded from the CPU's storage (hle_nvn.c nvn_ft_poll, 'GPU before CPU'), and what touches their memory afterwards:
// GPU writes (vk_tex_gpu_mark), uploads (pkg_upload_s) and image drops (img_drop_why) of any texture or image over the range; up to 32 ranges, 12 events each
static struct { uint64_t serial, lo, hi; int n; } g_tw[32]; static int g_ntw; static pthread_mutex_t g_tw_mu = PTHREAD_MUTEX_INITIALIZER;
void vk_tex_watch_add(uint64_t serial, uint64_t lo, uint64_t hi) { pthread_mutex_lock(&g_tw_mu); if (g_ntw < 32) g_tw[g_ntw++] = (typeof(g_tw[0])){serial, lo, hi, 0}; pthread_mutex_unlock(&g_tw_mu); }
static void vk_tex_watch_note(const char *what, uint64_t serial, uint64_t lo, uint64_t hi, unsigned w, unsigned h, unsigned fmt) {
    if (!g_ntw) return;
    pthread_mutex_lock(&g_tw_mu);
    for (int i = 0; i < g_ntw; i++) if (lo < g_tw[i].hi && g_tw[i].lo < hi && g_tw[i].n < 12) { g_tw[i].n++;
        VKLOG("gpuowndiag: watched serial %llu (%#llx..%#llx): %s by %s%llu %ux%u format %#x over %#llx..%#llx at present %llu", (unsigned long long)g_tw[i].serial,
              (unsigned long long)g_tw[i].lo, (unsigned long long)g_tw[i].hi, what, serial == g_tw[i].serial ? "itself, serial " : "serial ", (unsigned long long)serial, w, h, fmt,
              (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)g_tex_frame); }
    pthread_mutex_unlock(&g_tw_mu);
}
static void img_drop_why(vkimg_t *im, const char *why);
// bak86 (11-16, NPARK: terrain disappears and reappears since the bak75-77 eviction rules): MKW's terrain tiles (260x260 R16 / RG8 / RGBA8 / R11G11B10F, flags 0x200)
// are written by the CPU with no flush into CPU_CACHED | GPU_CACHED pools (0x24) and taken at first use (cpu_first_use, the nxnative record "Uncompressed storage the
// CPU wrote, in a pool the GPU has not written"). Once the GPU writes such a pool they leave the CPU-written set (packaged_tex): made again after an eviction, nothing
// fills them (cpu_first_use refuses a GPU-written pool, init_fill needs a flush) and they were cleared to zeros: flat or missing ground until the game wrote them again.
// These images need an archive before eviction; guest memory cannot rebuild them.
static uint64_t g_keep_unc, g_zf_again;   // images kept for that, storages zero-filled again after an upload (30 s line)
void vk_tex_unc_stats(uint64_t out[2]) { out[0] = g_keep_unc; out[1] = g_zf_again; }
static int gpu_only_content(const vkimg_t *im) {
    int protected = dm_sdk_ounce() && im->unc_pool && atomic_load_explicit(&((const pool_t *)im->unc_pool)->gpu_written, memory_order_relaxed);
#if defined(DM_NXN1X)
    nvnc_pool_protected(im->content_id, im->size > 0 ? (uint64_t)im->size : 0, im->gpu_w, protected);
#endif
    return protected; // F2 never relaxes the existing archive requirement
}
// g_img_mu pins registry lookups and archive ownership. last_use includes
// recording uses; vk_fence_done is g_done AFTER CPU copyback/mirror publication.
// external_use stays pinned because the compositor uses a separate timeline.
static nvni_decision image_eligibility(const vkimg_t *im, uint64_t done) {
    const int pool_written=gpu_only_content(im); // independent protection coverage
    nvni_state s={0};
    s.image=im->residency_id; s.version=__atomic_load_n(&im->content_serial,__ATOMIC_RELAXED);
    s.last_use=im->last_use; s.cpu_done=done; s.shared_keys=im->refs!=0;
    s.external_reader=im->external_use; s.transfer_pending=im->preserve || (im->img && im->img==g_archive_pending_image);
    s.supported_layout=!im->s2t && im->samples==1 && im->layout==VK_IMAGE_LAYOUT_GENERAL;
    s.legacy_guest=im->packaged && !im->gpu_w && !pool_written;
    if (im->archive) {
        s.archive_complete=1; s.archive_image=im->archive->image_id;
        s.archive_version=im->archive->content_serial;
    }
    return nvni_eligibility(s);
}
static uint64_t g_f3_cheap_bytes, g_f3_preserve_bytes, g_f3_pinned_bytes;
static _Thread_local int g_f3_cheap_only;
static _Thread_local int g_released_only;
static uint64_t g_keep_gw_bytes, g_keep_gw_n, g_keep_gw_reuse;
// Content-free, bounded ghost history. A refault teaches the cache how long this
// storage key actually goes unused between uses, including guest-backed images.
enum { RESIDENCY_HISTORY = 8192 };
static struct { uint64_t key, used; uint32_t idle; } g_residency_history[RESIDENCY_HISTORY];
static uint64_t residency_key(const void *data, size_t size) {
    const uint8_t *p = data; uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < size; i++) { h ^= p[i]; h *= 1099511628211ull; }
    return h ? h : 1;
}
static void residency_reused(vkimg_t *im, uint64_t used) {
    if (g_tex_frame < used) return;
    uint64_t gap = g_tex_frame - used;
    uint32_t keep = gap >= 150 ? 300 : (uint32_t)(gap * 2);
    if (keep < 16) keep = 16;
    if (im->residency_idle < keep) im->residency_idle = keep;
    g_residency_refaults++;
}
typedef struct {
    vkimg_t *im; nvn_hmap *table;
    uint8_t key[sizeof(akey) > sizeof(uint64_t) ? sizeof(akey) : sizeof(uint64_t)];
} reclaim_image;
typedef struct { size_t first, count; uint64_t used, bytes; int hot, archive; } reclaim_block;
static int reclaim_image_cmp(const void *a, const void *b) {
    uint64_t x = (uint64_t)(uintptr_t)((const reclaim_image *)a)->im->mem;
    uint64_t y = (uint64_t)(uintptr_t)((const reclaim_image *)b)->im->mem;
    return x < y ? -1 : x > y;
}
static int reclaim_block_cmp(const void *a, const void *b) {
    const reclaim_block *x = a, *y = b;
    if (x->hot != y->hot) return x->hot < y->hot ? -1 : 1;
    if (x->archive != y->archive) return x->archive < y->archive ? -1 : 1;
    if (x->used != y->used) return x->used < y->used ? -1 : 1;
    return x->bytes > y->bytes ? -1 : x->bytes < y->bytes;
}
static uint64_t g_trim_frame = UINT64_MAX, g_trim_frame_ns, g_trim_frame_bytes, g_trim_frame_images;
static uint64_t g_trim_max_ns, g_trim_slow, g_restore_max_ns, g_restore_slow;
extern int g_nvn2_heap_pct;
static uint64_t img_evict_nvn2(uint64_t want, uint64_t idle, int allocation_failed) {
    const int urgent = allocation_failed;
    if (allocation_failed) g_trim_oom_calls++; else g_trim_pressure_calls++;
    if (g_trim_frame != g_tex_frame) { g_trim_frame = g_tex_frame; g_trim_frame_ns = g_trim_frame_bytes = g_trim_frame_images = 0; }
    // A percentage is not a failed allocation. Even at 95%+, normal pressure
    // must yield between allocation cohorts instead of synchronously archiving
    // hundreds of MiB on the replay thread. Real OOM retains the recovery path.
    if (!urgent && (g_trim_frame_ns >= 3000000ull || g_trim_frame_bytes >= (64ull << 20))) { g_trim_yielded++; return 0; }
    const uint64_t start = now_ns(), done = vk_fence_done();
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_TRIM, want);
    uint64_t got = g_preserve_plan ? 0 : vk_image_trim_empty();
    size_t cap = (size_t)g_by_key.cap + g_by_serial.cap, n = 0, nb = 0;
    reclaim_image *images = NULL; reclaim_block *blocks = NULL;
    if (got >= want || !cap) goto finish;
    images = calloc(cap, sizeof *images); blocks = calloc(cap, sizeof *blocks);
    if (!images || !blocks) goto finish;
    // One table scan, not up to 128 full scans under the texture lock. Group by
    // actual allocation; dropping one cold slot beside a hot slot frees no RAM.
    for (int pass = 0; pass < 2; pass++) {
        nvn_hmap *h = pass ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            vkimg_t *im = h->vals[i];
            if (!im || !im->img || !im->mem || im->refs || im->external_use || im->s2t || im->samples > 1 ||
                im->used > g_tex_frame || im->last_use > done) continue;
            if (g_tex_frame - im->used < idle) continue;
            if (g_released_only && !im->released_hint) continue;
            if (!allocation_failed && g_tex_frame - im->used < im->residency_idle) { g_trim_hot++; continue; }
            const nvni_decision eligibility=image_eligibility(im,done);
            if (eligibility.source==NVNI_KEEP) { g_f3_pinned_bytes+=im->allocation_bytes; continue; }
            if (g_f3_cheap_only && eligibility.source==NVNI_SAVE_REQUIRED) { g_f3_preserve_bytes+=im->allocation_bytes; continue; }
            images[n].im = im; images[n].table = h;
            memcpy(images[n].key, h->keys + (size_t)i * h->klen, h->klen); n++;
        }
    }
    qsort(images, n, sizeof *images, reclaim_image_cmp);
    for (size_t first = 0; first < n;) {
        size_t end = first + 1; uint64_t used = images[first].im->used;
        while (end < n && images[end].im->mem == images[first].im->mem) {
            if (used < images[end].im->used) used = images[end].im->used;
            end++;
        }
        int hot = 0, archive = 0;
        for (size_t j = first; j < end; j++) {
            const vkimg_t *im = images[j].im;
            hot |= g_tex_frame - im->used < im->residency_idle;
            archive |= (!im->packaged || im->gpu_w || gpu_only_content(im)) && !archive_clean(im);
        }
        uint32_t slots; uint64_t bytes = vk_image_allocation_size(images[first].im->sub, images[first].im->allocation_bytes, &slots);
        if (end - first == slots) blocks[nb++] = (reclaim_block){first, end - first, used, bytes, hot, archive};
        else g_trim_mixed++;   // a live, pinned or hot slot prevents reclaiming this allocation
        first = end;
    }
    qsort(blocks, nb, sizeof *blocks, reclaim_block_cmp);
    for (size_t b = 0; b < nb && got < want; b++) {
        const reclaim_block *block = &blocks[b];
        if (!urgent && g_trim_frame_images && (g_trim_frame_ns + now_ns() - start >= 3000000ull || g_trim_frame_bytes >= (64ull << 20))) {
            g_trim_yielded++; goto finish;
        }
        if (g_preserve_plan) {
            for (size_t i=block->first;i<block->first+block->count;i++)
                if (image_eligibility(images[i].im,done).source==NVNI_SAVE_REQUIRED)
                    preserve_enqueue(images[i].im);
            continue; // scheduling does not release or credit any allocation
        }
        // Complete the selected allocation even if it exceeds the soft budget.
        // Yielding between slots can leave zero memory freed and let a restored
        // slot pin the block again before the remaining slots are reclaimed.
        for (size_t i = block->first; i < block->first + block->count; i++) {
            g_trim_frame_images++;
            vkimg_t *im = images[i].im;
            if (image_eligibility(im,done).source==NVNI_LEGACY_GUEST) {
                if (im->residency_key) {
                    size_t at = im->residency_key % RESIDENCY_HISTORY;
                    g_residency_history[at].key = im->residency_key; g_residency_history[at].used = im->used;
                    g_residency_history[at].idle = im->residency_idle;
                }
                got += img_release_completed(im);
                nvn_hm_del(images[i].table, images[i].key);
                img_drop_why(im, "idle allocation, recoverable from guest memory"); g_img_mem_evicted++;
            } else {
                uint64_t written0 = g_archive_written_bytes;
                got += img_archive_release(im);
                g_trim_frame_bytes += g_archive_written_bytes - written0;
                if (im->img) break;   // a failed save pins this block; do not churn its other slots
            }
        }
    }
finish:
    free(blocks); free(images);
    uint64_t elapsed = now_ns() - start; g_trim_ns += elapsed; g_trim_frame_ns += elapsed; g_trim_freed += got;
    if (elapsed > g_trim_max_ns) g_trim_max_ns = elapsed;
    if (elapsed >= 50000000ull) {
        uint64_t n = ++g_trim_slow;
        if (n <= 16 || !(n & (n - 1))) VKLOG("NVN2 residency stall: trim %.2f ms, cause %s, requested %.1f MiB, freed %.1f MiB, heap %d%% (%llu slow trims)",
            elapsed / 1e6, allocation_failed ? "allocation failure" : "pressure", want / 1048576.0,
            got / 1048576.0, g_nvn2_heap_pct, (unsigned long long)n);
    }
    vk_f0_end(&diag, 0);
    return got;
}
static uint64_t img_evict_idle_mode(uint64_t want, uint64_t idle, int allocation_failed) {
    uint64_t got = 0, done = vk_fence_done();
    if (dm_sdk_ounce()) return img_evict_nvn2(want, idle, allocation_failed);
    for (int pass = 0; pass < 2 && got < want; pass++) {   // the pass over g_by_key, then g_by_serial
        nvn_hmap *h = pass ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap && got < want; i++) {
            vkimg_t *im = h->vals[i];
            if (!im || !im->packaged || im->gpu_w || im->s2t || im->used + idle > g_tex_frame || im->last_use > done) continue;
            if (gpu_only_content(im)) { g_keep_unc++; continue; }   // bak86: its only copy
            if (dm_sdk_ounce() && im->made + 300 > g_tex_frame) continue;   // NVN2 (11-16): made within the last 300 presents (as Eden's minimum age): MKW dropped and made
            // again ~25 images a second at 25 / 10 presents of idleness, with s24 checking 19-22 s of each 30 s (was about 5) and its terrain flickering
            uint8_t key[sizeof(akey) > sizeof(uint64_t) ? sizeof(akey) : sizeof(uint64_t)]; memcpy(key, h->keys + (size_t)i * h->klen, h->klen);
            nvn_hm_del(h, key); i = (uint32_t)-1;   // the table was rehashed by the delete: rescan (as vk_tex_drop_pool)
            got += im->size > 0 ? (uint64_t)im->size : 0; g_img_mem_evicted++; img_drop_why(im, "idle, for memory");
        }
    }
    return got;
}
static uint64_t img_evict_idle(uint64_t want, uint64_t idle) { return img_evict_idle_mode(want, idle, 0); }
static uint64_t img_evict_oom(uint64_t want) { return img_evict_idle_mode(want, 2, 1); }
// Request the allocator's actual growth plus modest recording headroom, not a
// fixed 256 MiB for a 60 KiB texture. A second failure keeps the previous broad
// recovery as a fallback; no image is discarded without recoverable contents.
static uint64_t img_reclaim_target(const VkMemoryRequirements *req, uint32_t type) {
    uint64_t growth = vk_image_memory_growth(req, type);
    return growth > UINT64_MAX - (8ull << 20) ? UINT64_MAX : growth + (8ull << 20);
}
static uint64_t g_reclaim_targeted, g_reclaim_escalated, g_reclaim_requested, g_reclaim_freed;
// Keep this C stack and its command cursor intact. The leases exclude registry
// removal and competing fence assignment, while both real mutexes are released.
// Only completed, cheap victims can retire here; no recursive save/restore.
static VkDeviceMemory img_memory_continue(const VkMemoryRequirements *req,uint32_t type,
                                         VkDeviceSize *offset,void **sub) {
    if (!vk_tex_preserve_enabled() || !nvnp_pending_bytes() || !nvnr_suspend(&g_img_mu))
        return VK_NULL_HANDLE;
    nvnr_submission saved;
    vk_sub_suspend(&saved);
    nvnr_unlock(&g_img_mu);
    const uint64_t deadline=now_ns()+500000000ull;
    VkDeviceMemory memory=VK_NULL_HANDLE;
    unsigned waits=0;
    uint64_t released=0;
    for (;;) {
        const uint64_t progress=nvnp_progress();
        vk_tex_preserve_poll(0);
        nvnr_lock(&g_img_mu);
        const int cheap=g_f3_cheap_only; g_f3_cheap_only=1;
        const uint64_t freed=img_evict_nvn2(img_reclaim_target(req,type),2,1);
        g_f3_cheap_only=cheap; released+=freed;
        if (freed || !vk_image_memory_growth(req,type))
            memory=vk_image_memory(req,type,offset,sub);
        nvnr_unlock(&g_img_mu);
        const uint64_t now=now_ns();
        if (memory || !nvnp_pending_bytes() || now>=deadline) break;
        const uint64_t left=deadline-now;
        const int progressed=nvnp_wait_progress(progress,left<100000000ull?left:100000000ull);
        ++waits;
        if (!progressed && !nvnp_working()) break;
    }
    nvnr_lock(&g_img_mu);
    vk_sub_resume(&saved);
    if (!nvnr_resume(&g_img_mu)) die("[vk] image continuation lost ownership");
    VKLOG("NVN2 F4D allocation continuation: bytes=%llu released=%llu waits=%u resumed=%d; same command, no accepted-part replay",
        (unsigned long long)req->size,(unsigned long long)released,waits,memory!=VK_NULL_HANDLE);
    return memory;
}
static VkDeviceMemory img_memory_recover_impl(const VkMemoryRequirements *req, uint32_t type,
                                        VkDeviceSize *offset, void **sub) {
    VkDeviceMemory memory = vk_image_memory(req, type, offset, sub);
    if (memory || !dm_sdk_ounce()) return memory;
    memory=img_memory_continue(req,type,offset,sub);
    if (memory) return memory;
    uint64_t target = img_reclaim_target(req, type), started = now_ns();
    uint64_t got = img_evict_oom(target);
    g_reclaim_targeted++; g_reclaim_requested += target;
    memory = vk_image_memory(req, type, offset, sub);
    if (!memory) {
        g_reclaim_escalated++;
        uint64_t broad = req->size > (256ull << 20) ? req->size : (256ull << 20);
        got += img_evict_oom(broad);
        memory = vk_image_memory(req, type, offset, sub);
    }
    g_reclaim_freed += got;
    if (g_reclaim_targeted <= 16 || !(g_reclaim_targeted & (g_reclaim_targeted - 1)))
        VKLOG("NVN2 image allocation recovery: request %.3f MiB, first target %.1f MiB, reclaimed %.1f MiB, %.2f ms, result %s; targeted %llu, escalated %llu",
            req->size / 1048576.0, target / 1048576.0, got / 1048576.0, (now_ns() - started) / 1e6,
            memory ? "allocated" : "failed", (unsigned long long)g_reclaim_targeted, (unsigned long long)g_reclaim_escalated);
    return memory;
}
static VkDeviceMemory img_memory_recover(const VkMemoryRequirements *req, uint32_t type,
                                        VkDeviceSize *offset, void **sub) {
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_IMAGE_ALLOC, req->size);
    VkDeviceMemory memory = img_memory_recover_impl(req, type, offset, sub);
    vk_f0_end(&diag, memory ? 0 : 1);   // local outcome; not a driver VkResult
    return memory;
}
void vk_tex_mem_stats(uint64_t out[5]) { out[0] = g_img_n; out[1] = g_img_bytes; out[2] = g_img_evicted; out[3] = g_img_mem_evicted; out[4] = 0; }
// NVN2 memory pressure (gfx_vk_core.c nvn2_pressure, at a submission start): img_evict_idle under g_img_mu; skipped (0) when the lock is held, by this thread or another
uint64_t vk_tex_evict_mem(uint64_t want, uint64_t idle) {   // idle: presents since the last look-up
    if (!want || nvnr_trylock(&g_img_mu)) return 0;
    uint64_t got = img_evict_idle(want, idle ? idle : 300);
    nvnr_unlock(&g_img_mu);
    return got;
}
uint64_t vk_tex_evict_oom(uint64_t want) {
    if (!want || nvnr_trylock(&g_img_mu)) return 0;
    uint64_t got = img_evict_oom(want);
    nvnr_unlock(&g_img_mu);
    return got;
}
// F4 invokes this only before taking submission/queue locks. This pass never
// saves an image, waits for a transfer, or drops a sole current copy. Complete
// allocator cohorts are still required, so bytes returned mean actual frees.
uint64_t vk_tex_reclaim_completed(uint64_t want) {
    if (!want || nvnr_trylock(&g_img_mu)) return 0;
    g_f3_cheap_only=1;
    uint64_t freed=img_evict_nvn2(want,8,0);
    g_f3_cheap_only=0; g_f3_cheap_bytes+=freed;
    nvnr_unlock(&g_img_mu); return freed;
}
// Upstream bounds its released-texture cache by bytes (budget/12). Finalization
// here is only a hint: recorded snapshots may still use this storage. All current
// eligibility, CPU completion, external-reader and learned-idle checks remain.
void vk_tex_trim_released(uint64_t budget) {
    static uint64_t checked_frame=UINT64_MAX;
    if (nvnr_trylock(&g_img_mu)) return;
    if (checked_frame==g_tex_frame) { nvnr_unlock(&g_img_mu);return; }
    checked_frame=g_tex_frame;
    // slot12: a pass that freed nothing while over the cap is not run again every frame. OoT (11-21): ~10,000 images walked and img_evict_nvn2 tried each frame,
    // freeing 0 (the released images are GPU-only render targets, "save required"), 0.8-1.3 ms a frame of the replay thread and growing with play time.
    // Whether an image can go depends on its age in frames (idle, residency), so there is no count to keep up to date: after a pass that freed nothing the
    // next one waits 2, 4, ... 64 frames; one that frees resets it. DM_VK_TRIM_SKIP=0: every frame as before
    static int skip_on = -1, backoff; static uint64_t skip_until, skipped;
    if (skip_on < 0) { const char *e = getenv("DM_VK_TRIM_SKIP"); skip_on = !(e && *e == '0'); }
    if (skip_on && g_tex_frame < skip_until) { skipped++; nvnr_unlock(&g_img_mu); return; }
    uint64_t cap=budget?budget/12:512ull<<20;
    if (cap<(256ull<<20)) cap=256ull<<20;
    if (cap>(2048ull<<20)) cap=2048ull<<20;
    uint64_t retained=0;
    for (int pass=0;pass<2;pass++) {
        nvn_hmap *h=pass?&g_by_serial:&g_by_key;
        for (uint32_t i=0;i<h->cap;i++) {
            vkimg_t *im=h->vals[i];
            if (!im || !im->released_hint || !im->img || im->refs || im->s2t) continue;
            retained+=im->allocation_bytes; // image payload; block slack is not falsely credited
        }
    }
    uint64_t freed=0;
    if (retained>cap) {
        const int cheap=g_f3_cheap_only, released=g_released_only;
        g_f3_cheap_only=1;g_released_only=1;
        freed=img_evict_nvn2(retained-cap,8,0);
        g_f3_cheap_only=cheap;g_released_only=released;
    }
    if (skip_on) { if (retained>cap && !freed) { backoff = backoff ? (backoff >= 32 ? 64 : backoff * 2) : 2; skip_until = g_tex_frame + (uint64_t)backoff; } else backoff = 0; }   // slot12
    if (freed || (retained>cap && !(g_tex_frame%300)))
        VKLOG("NVN2 F4F released cache: retained_payload=%llu cap=%llu actual_freed=%llu; unreconstructable or referenced contents kept (slot12: passes skipped %llu, next in %d frames%s)",
            (unsigned long long)retained,(unsigned long long)cap,(unsigned long long)freed, (unsigned long long)skipped, backoff, skip_on ? "" : "; DM_VK_TRIM_SKIP=0");
    nvnr_unlock(&g_img_mu);
}
// Pool recovery cannot block on an image owner that might itself need the pool
// writer. A failed trylock (including a caller already owning it) skips recovery,
// never skips the required allocation/error check. No new preservation is queued.
uint64_t vk_tex_recover_pool(uint64_t want) {
    if (!want || nvnr_trylock(&g_img_mu)) return 0;
    if (!nvnr_suspend(&g_img_mu)) { nvnr_unlock(&g_img_mu); return 0; }
    nvnr_unlock(&g_img_mu);
    uint64_t freed=0;
    const uint64_t deadline=now_ns()+500000000ull;
    for (;;) {
        const uint64_t progress=nvnp_progress();
        vk_tex_preserve_poll(0);
        nvnr_lock(&g_img_mu);
        const int cheap=g_f3_cheap_only; g_f3_cheap_only=1;
        freed+=img_evict_nvn2(want,2,1);
        g_f3_cheap_only=cheap;
        nvnr_unlock(&g_img_mu);
        const uint64_t now=now_ns();
        if (freed || !vk_tex_preserve_enabled() || !nvnp_pending_bytes() || now>=deadline) break;
        const uint64_t left=deadline-now;
        if (!nvnp_wait_progress(progress,left<100000000ull?left:100000000ull) && !nvnp_working()) break;
    }
    nvnr_lock(&g_img_mu);
    if (!nvnr_resume(&g_img_mu)) die("[vk] pool recovery lost image ownership");
    nvnr_unlock(&g_img_mu);
    return freed;
}
// NVN2 diagnostic (11-16, MKW at its heap budget): what the host's non-sparse images take on the device, from their memory requirements, by class (packaged: made again
// from guest memory; written by the GPU: render targets, storage, copy and clear destinations; other), and the 20 largest. Under g_img_mu (skipped when it is busy). Logging only
void vk_tex_mem_attrib(void) {
    if (nvnr_trylock(&g_img_mu)) { VKLOG("NVN2 memory attribution: skipped (the image table is busy)"); return; }
    enum { TOP = 20 }; const vkimg_t *top[TOP]; VkDeviceSize tsz[TOP]; int nt = 0;
    VkDeviceSize cls[3] = {0, 0, 0}; uint64_t ncls[3] = {0, 0, 0}, nsparse = 0;
    for (int pass = 0; pass < 2; pass++) {   // g_by_key, then g_by_serial (an image is in one of them)
        const nvn_hmap *h = pass ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            const vkimg_t *im = h->vals[i]; if (!im || !im->img) continue;
            if (im->s2t) { nsparse++; continue; }
            VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, im->img, &req);
            const int c = im->packaged ? 0 : im->gpu_w ? 1 : 2; cls[c] += req.size; ncls[c]++;
            int at = nt; while (at > 0 && tsz[at - 1] < req.size) at--;
            if (at < TOP) { int last = nt < TOP ? nt : TOP - 1; for (int k = last; k > at; k--) { top[k] = top[k - 1]; tsz[k] = tsz[k - 1]; } top[at] = im; tsz[at] = req.size; if (nt < TOP) nt++; }
        }
    }
    VKLOG("NVN2 memory attribution: non-sparse images on the device %.0f MB: packaged %llu images %.0f MB, written by the GPU %llu images %.0f MB, other %llu images %.0f MB "
          "(%llu sparse images apart: their bound memory is the sparse layers / tiles lines)", (cls[0] + cls[1] + cls[2]) / 1048576.0, (unsigned long long)ncls[0], cls[0] / 1048576.0,
          (unsigned long long)ncls[1], cls[1] / 1048576.0, (unsigned long long)ncls[2], cls[2] / 1048576.0, (unsigned long long)nsparse);
    for (int k = 0; k < nt; k++) { const vkimg_t *im = top[k];
        VKLOG("NVN2 memory attribution: #%d %ux%ux%u, %u levels, nvn format %#x (vk %d), %.1f MB, %s, last used %llu presents ago", k + 1, im->w, im->h, im->layers, im->levels,
              im->nvn_format, im->fmt, tsz[k] / 1048576.0, im->packaged ? "packaged" : im->gpu_w ? "written by the GPU" : "other",
              (unsigned long long)(g_tex_frame > im->used ? g_tex_frame - im->used : 0)); }
    nvnr_unlock(&g_img_mu);
}
uint64_t g_views_made;   // texture views made (vk_tex_view; gfx_vk_core.c NVN2 memory report)
static vkimg_t *img_new(const tex_t *t, const vkfmt_t *f) {
    vkimg_t *im = calloc(1, sizeof *im);
    if (!im) return NULL;
    if (g_residency_id!=UINT64_MAX) im->residency_id=++g_residency_id;
    const nvn_texdesc *d = &t->d;
    im->w = d->width ? d->width : 1; im->h = d->target == 0 || d->target == 3 ? 1 : (d->height ? d->height : 1);
    im->d = d->target == 2 ? (d->depth ? d->depth : 1) : 1; im->layers = layers_of(d);
    im->levels = d->levels ? d->levels : 1; im->samples = d->samples > 1 ? d->samples : 1; im->target = d->target; im->nvn_format = d->format;
    im->fmt = f->vk; im->vtype = vtype_of(d->target); im->aspect = aspect_of(f);
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, f->vk, &fp);
    VkImageUsageFlags u = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    int st_twin = 0; if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && img_storage_ok(f, &st_twin)) u |= VK_IMAGE_USAGE_STORAGE_BIT;   // r72: also through the sRGB twin
    int typed = 0;   // ootu32: storage only through declared-format views
    if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && !(u & VK_IMAGE_USAGE_STORAGE_BIT) && im->samples <= 1 && img_typed_storage_ok(f)) {
        VkImageFormatProperties ip; const VkImageCreateFlags fl = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | (d->target == 2 ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0);
        const VkImageType ty = d->target == 2 ? VK_IMAGE_TYPE_3D : d->target == 0 || d->target == 3 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D;
        typed = vkGetPhysicalDeviceImageFormatProperties(V.pd, f->vk, ty, VK_IMAGE_TILING_OPTIMAL, u | VK_IMAGE_USAGE_STORAGE_BIT, fl, &ip) == VK_SUCCESS;
        if (typed) u |= VK_IMAGE_USAGE_STORAGE_BIT;
        static int n; if (n++ < 8) VKLOG("ootu32: %ux%u nvn format %#x (vk %d) image %s STORAGE usage for declared-format views (#%d)", im->w, im->h, d->format, f->vk, typed ? "given" : "NOT given (driver refused)", n);
    }
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL,
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | (d->target == 8 || d->target == 9 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) |
        (d->target == 2 ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0) | (f->compressed ? VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT : 0),
        d->target == 2 ? VK_IMAGE_TYPE_3D : d->target == 0 || d->target == 3 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D, f->vk,
        {im->w, im->h, im->d}, im->levels, im->layers, (VkSampleCountFlagBits)im->samples, VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (im->samples > 1) ci.flags &= ~(VkImageCreateFlags)VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    // Zelda encodes each BC4 block through an eight-byte RGBA16UI color view.
    if (f->vk == VK_FORMAT_BC4_UNORM_BLOCK) {
        VkImageFormatProperties limits;
        VkImageUsageFlags raw_usage = ci.usage | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (vkGetPhysicalDeviceImageFormatProperties(V.pd, ci.format, ci.imageType,
                ci.tiling, raw_usage, ci.flags, &limits) == VK_SUCCESS &&
            ci.extent.width <= limits.maxExtent.width && ci.extent.height <= limits.maxExtent.height &&
            ci.extent.depth <= limits.maxExtent.depth && ci.mipLevels <= limits.maxMipLevels &&
            ci.arrayLayers <= limits.maxArrayLayers && (ci.samples & limits.sampleCounts))
            ci.usage = raw_usage;
    }
    if (st_twin) ci.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;   // r72: storage only through the twin's views
    // NVN2 sparse tiles (Switch 2 boots only): 2D maps, and (bak80) sparse-layer arrays per tile; use a fully backed image when sparse residency is unavailable
    int s2t = typed ? 0 : s2tile_wanted(t, f);   // ootu32: not sparse (as oote5 tested)
    const VkImageCreateInfo ci0 = ci;    // (bak81: as made before the residency flags, for the fallback)
    if (s2t) {
        ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
        if (ps_on()) ci.flags |= VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;   // bak83: its tiles may be bound where other images (or other places of it) bind them
    }
    im->usage = ci.usage; im->cflags = ci.flags;
    VkResult r = vkCreateImage(V.dev, &ci, NULL, &im->img);
    if (r && s2t) {   // bak81: residency refused for this format / usage / flags: use a fully backed image. Logged once per format
        static int said[512]; if ((unsigned)f->vk < 512 && !said[f->vk]++) VKLOG("[RED] NVN2 sparse tiles: vkCreateImage refused residency (%d) for %ux%ux%u vk %d (usage %#x flags %#x): %s instead",
            r, im->w, im->h, im->layers, f->vk, ci.usage, ci.flags, "a whole image");
        ci = ci0; s2t = 0;
        im->usage = ci.usage; im->cflags = ci.flags;
        r = vkCreateImage(V.dev, &ci, NULL, &im->img);
    }
    if (r) { VKRED_ONCE(10, "vkCreateImage failed (%d): %ux%ux%u format %#x (vk %d) target %d levels %u layers %u samples %u", r, im->w, im->h, im->d, d->format, f->vk, d->target, im->levels, im->layers, im->samples); free(im); return NULL; }
    VkMemoryRequirements req; vkGetImageMemoryRequirements(V.dev, im->img, &req);
    if (s2t && !(im->s2t = s2tile_init(im, t, f, &req))) {   // NVN2: tiles refused (logged): allocate a fully backed image
        vkDestroyImage(V.dev, im->img, NULL); s2t = 0;
        ci = ci0;   // (bak81: the flags and format list as before the residency ones)
        im->usage = ci.usage; im->cflags = ci.flags;
        if (vkCreateImage(V.dev, &ci, NULL, &im->img)) { free(im); return NULL; }
        vkGetImageMemoryRequirements(V.dev, im->img, &req); }
    if (dm_sdk_ounce() && req.size >= (32ull << 20)) {   // NVN2: every host image of 32 MiB or more, at its making (MKW 11-16: a 342 MB image refused, unnamed)
        static uint64_t nbig; nbig++;
        VKLOG("NVN2 big image %llu: %ux%ux%u, %u levels, %u layers, nvn format %#x (vk %d), target %d, %.1f MiB, pool %s%s, %s; storage %p (pool %p +%#llx), texture object %llu", (unsigned long long)nbig, im->w, im->h, im->d, im->levels, im->layers,
            d->format, f->vk, d->target, req.size / 1048576.0, !t->pool ? "none" : !t->pool->vp ? "plain" : t->pool->vmode == 2 ? "virtual (vmode 2)" : "virtual",
            t->pool && t->pool->vp && t->pool->vmode == 2 ? (nvn_sparse_any(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size) ? ", pages mapped" : ", no page mapped") : "",
            im->s2t ? "sparse tiles" : "whole memory", (const void *)t->mem, (const void *)t->pool, (unsigned long long)t->off, (unsigned long long)t->serial); }
    if (im->s2t) { im->layout = VK_IMAGE_LAYOUT_UNDEFINED; return im; }   // NVN2: no memory now: tiles get theirs as the game maps them (s2tile_refresh)
    if ((ci.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) && im->w >= 256) { static int n; if (n++ < 16)   // UBWC check: a compressed target takes room for its metadata
        VKLOG("target %ux%u format %d: %llu bytes", im->w, im->h, f->vk, (unsigned long long)req.size); }
    im->allocation_bytes = req.size;
    uint32_t memory_type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    uint64_t growth = dm_sdk_ounce() ? vk_image_memory_growth(&req, memory_type) : 0;
    if (dm_sdk_ounce() && growth >= (32ull << 20)) {
        uint64_t need = vk_nvn2_memory_need(growth);
        uint64_t target = img_reclaim_target(&req, memory_type);
        if (need) img_evict_idle(need < target ? need : target, 8);
    }
    VkDeviceSize moff;
    im->mem = img_memory_recover(&req, memory_type, &moff, &im->sub);
    if (!im->mem) {
        VKRED_ONCE(11, "image memory allocation failed (%llu bytes)", (unsigned long long)req.size);
        vkDestroyImage(V.dev, im->img, NULL); free(im);
        if (vk_tex_preserve_enabled())
            die("[vk] required image allocation exhausted after continuation and recovery (%llu bytes); command not skipped", (unsigned long long)req.size);
        return NULL; }
    VKCHK(vkBindImageMemory(V.dev, im->img, im->mem, moff));
    im->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    return im;
}
void vk_tex_prepare(VkCommandBuffer cb, vkimg_t *im) {
    if (im->layout == VK_IMAGE_LAYOUT_GENERAL) return;
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the first transition only (leaf); the layout looked at again under it
    if (im->layout == VK_IMAGE_LAYOUT_GENERAL) { vk_up_unlock(); return; }
    if (vk_par_on_worker()) cb = vk_upload_cmd();   // S1e (parallel recording): a worker's first transition goes into the part's upload buffer, which runs before every chunk of the part
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img, {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    im->layout = VK_IMAGE_LAYOUT_GENERAL;
    vk_up_unlock();
}
// Upload a packaged texture from guest memory: each level and layer deswizzled into scratch memory, copied in by the upload command buffer
uint64_t g_pkg_n, g_pkg_bytes, g_pkg_ns;   // packaged texture uploads: count, bytes, CPU time (the replay's per-frame line)
static uint64_t g_pkg_part[4];   // r53 measure: dirty protect, scratch, deswizzle / copy, recording (ns; logged every 128 uploads)
static void pkg_upload_(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
extern uint64_t g_tex_frame;
static uint64_t r153_hash(const void *m, size_t n) {   // r153
    const uint8_t *b = m; uint64_t h = 0xcbf29ce484222325ull ^ n; size_t w = n / 8, st = w > 4096 ? w / 4096 : 1;
    for (size_t i = 0; b && i < w; i += st) { uint64_t v; memcpy(&v, b + 8 * i, 8); h = (h ^ v) * 0x100000001b3ull; }
    return h; }
// s24: NXN1X (Windows) leaves packaged textures unprotected (pkg_protect: VirtualProtect is slow), so a guest CPU write into one after its upload goes unseen and the image
// keeps the old content. Xenoblade Chronicles 3 writes into its textures' storage from the CPU after their first use (on the Android host, where they are protected, its
// protection split the pools into 43,000 mappings), and on NXN1X's NVN path objects flashed over-bright for seconds (11-12); DM_VK_PKG_PROTECT=1 stalled it (run 29b).
// DM_VK_PKG_VERIFY=<n>: a packaged texture drawn is looked at again at most every n presents: 257 words sampled across its storage against those at its upload, and uploaded
// again when they differ. 0 = off. s27: on by default in NXN1X (8), and a texture is also looked at once at its first use by another texture object than the one it was
// uploaded or checked for: images are kept by storage and layout (g_by_key), and Xenoblade 3 streams textures through fixed slots of one size and format (512x512,
// 256x256, 128x128 BC, full mips), so a texture made over a slot found the image of the slot's previous texture, uploaded, and showed its content (objects in another
// texture's colours, black plant shapes) until a periodic check caught it
#if defined(DM_NXN1X)
enum { PKG_VERIFY_DEFAULT = 8 };
#else
enum { PKG_VERIFY_DEFAULT = 0 };
#endif
static int pkg_verify(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_PKG_VERIFY"); v = e && *e ? atoi(e) : PKG_VERIFY_DEFAULT; if (v < 0) v = 0;
        VKLOG("s24: packaged textures: CPU writes after their upload %s (DM_VK_PKG_VERIFY=%d)", v ? "looked for by sampling their storage" : "not looked for", v); }
    return v;
}
static uint64_t s24_hash(const void *m, size_t n) {
    const uint8_t *b = m; uint64_t h = 0x9e3779b97f4a7c15ull ^ n, v;
    if (!b || n < 8) return h;
    size_t w = n / 8, st = w > 256 ? w / 256 : 1;
    for (size_t i = 0; i < w; i += st) { memcpy(&v, b + 8 * i, 8); h = (h ^ v) * 0x100000001b3ull; }
    memcpy(&v, b + n - 8, 8); return (h ^ v) * 0x100000001b3ull;
}
static uint64_t g_s24_checks, g_s24_hits, g_s24_ns, g_s27_checks, g_s27_hits;   // s27: of them, at another texture object's first use
// NVN2: s24_hash of a vmode-2 texture's storage, the same words sampled, each read through the page table
static uint64_t s24_hash_vp(const void *pool, uint64_t a, size_t n) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ n, v;
    if (!a || n < 8) return h;
    size_t w = n / 8, st = w > 256 ? w / 256 : 1;
    for (size_t i = 0; i < w; i += st) { nvn_sparse_copy(pool, &v, a + 8 * i, 8); h = (h ^ v) * 0x100000001b3ull; }
    nvn_sparse_copy(pool, &v, a + n - 8, 8); return (h ^ v) * 0x100000001b3ull;
}
static void pkg_upload(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
uint64_t g_s24_skips, g_s24_upskips;   // NVN2: checks and uploads not made, their storage not readable (gfx_vk_core.c NVN2 host-objects line)
#if defined(DM_NXN1X)
// NVN2 (Switch 2 boots): a check reads the texture's storage in guest memory, which may be gone since the upload. Mario Kart World 11-16 (twice, 07:05 and 07:07, about
// 4 s after the host's first presents): an access violation in s24_hash, inlined in vk_tex (NXN1X.exe +0xe26f3c). The storage is looked at first (nvnh_range_readable);
// storage not committed and readable is not checked (the image keeps what it has), and an upload from it is not made (pkg_upload_), each logged once per address.
// DM_VK_S24_TRACE=1: every texture's first check logs its storage address and size (up to 512 lines), so a fault names its texture (off by default: DK 11-16 spent most of
// the replay in s24, 1.6 ms a check). The guard's answer is kept per image with the guest mapping generation it was taken at (s24_readable): asked again only when guest
// memory was mapped, unmapped or made unreadable since (nvnh_map_gen), not at every check: VirtualQuery walks the dirty tracker's per-page regions one by one
int nvnh_range_readable(const void *mem, uint64_t size); uint64_t nvnh_map_gen(void);
static uint64_t g_s24_rd_ns, g_s24_rd_n, g_s24_rd_cached;   // NVN2: time and count of the readable queries made, answers taken from the cache
// bak82: a "readable" answer is kept across a move of the generation when none of the mapping changes since it was taken touched its storage (the ring of
// common/host_memory.cpp nxn1x_map_ev_*, recorded by HostMemory Map / Unmap / no-read Protect and hle_nvn.c's virtual pool unmaps), the ring still holds them all, and
// the moves are no more than those changes' (two each: a move without its event, from a place that does not record, makes it ask again). A "not readable" answer is
// asked again at every move, as before. DM_VK_S24_RANGE_INV=0: off (every move asks again); 1: on; default on for NVN2 (Switch 2) boots, the only ones s24_readable runs for
uint64_t nxn1x_map_ev_mark(void); int nxn1x_map_ev_clear(uint64_t from, const void *a, uint64_t n, uint64_t *now);
static uint64_t g_s24_rd_kept, g_s24_rd_requery, g_s24_rd_over, g_s24_up_rd_ns;   // kept by range; asked again (a change in range, a change under way, extra moves); ring overflow
static int s24_range_inv(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_S24_RANGE_INV"); v = e && *e ? atoi(e) != 0 : dm_sdk_ounce() != 0;
        VKLOG("NVN2 s24: readable answers %s (DM_VK_S24_RANGE_INV=%s, %s boot)", v ? "kept across mapping changes outside their storage" : "asked again at every mapping change",
              e && *e ? e : "unset", dm_sdk_ounce() ? "a Switch 2" : "a Switch 1"); }
    return v;
}
int nvnh_range_readable_hm(const void *mem, uint64_t size);
static int s24_hm_on(void) {   // bak84: the guard's answer from HostMemory's map first (nvnh_rt.c nvnh_range_readable_hm). DM_VK_S24_HM_READABLE=0 off; on for NVN2 boots
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_S24_HM_READABLE"); v = e && *e ? atoi(e) != 0 : dm_sdk_ounce() != 0;
        VKLOG("NVN2 s24: readable answers %s (DM_VK_S24_HM_READABLE=%s)", v ? "from the host's own map of guest memory first, VirtualQuery for what it does not cover" : "from VirtualQuery", e && *e ? e : "unset"); }
    return v;
}
static int s24_readable(vkimg_t *im, const tex_t *t) {
    const uint64_t g = nvnh_map_gen();
    if (g && im->rd_mem == (const void *)t->mem && im->rd_len == t->L.size) {
        if (im->rd_gen == g) { g_s24_rd_cached++; return im->rd_ok; }
        if (im->rd_ok && s24_range_inv()) {
            uint64_t now = 0; const int c = nxn1x_map_ev_clear(im->rd_ev, t->mem, t->L.size, &now);
            if (c > 0 && g - im->rd_gen <= 2 * (now - im->rd_ev)) { im->rd_gen = g; im->rd_ev = now; g_s24_rd_kept++; g_s24_rd_cached++; return 1; }
            if (c == -1) g_s24_rd_over++; else g_s24_rd_requery++;
        }
    }
    const uint64_t mk = nxn1x_map_ev_mark(), g2 = nvnh_map_gen();   // (the ring position first: a change finished before it is in g2)
    const uint64_t q = now_ns(); const int ok = t->mem && (s24_hm_on() ? nvnh_range_readable_hm(t->mem, t->L.size) : nvnh_range_readable(t->mem, t->L.size)); g_s24_rd_ns += now_ns() - q; g_s24_rd_n++;
    im->rd_gen = g2; im->rd_ev = mk; im->rd_mem = t->mem; im->rd_len = t->L.size; im->rd_ok = ok;
    return ok;
}
static int s24_readable_up(vkimg_t *im, const tex_t *t) {   // bak82: pkg_upload_s's guard, its time counted apart (it is in the readable queries' time, not in the s24 checks')
    const uint64_t q = now_ns(); const int ok = s24_readable(im, t); g_s24_up_rd_ns += now_ns() - q; return ok;
}
void vk_tex_s24_range_stats(uint64_t out[4]) { out[0] = g_s24_rd_kept; out[1] = g_s24_rd_requery; out[2] = g_s24_rd_over; out[3] = g_s24_up_rd_ns; }
static int s24_seen(const void *m, const void **ring, unsigned *ri, unsigned cap) {   // 1 when m was noted already (in the last cap); else noted
    for (unsigned i = 0; i < cap && i < *ri; i++) if (ring[i] == m) return 1;
    ring[(*ri)++ % cap] = m; return 0;
}
#endif
static void s24_check(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {   // (the replay thread, as pkg_upload)
    uint64_t q = now_ns(); im->pkg_vframe = g_tex_frame; g_s24_checks++;
#if defined(DM_NXN1X)
    if (dm_sdk_ounce()) {
        static int trace = -1; if (trace < 0) { const char *e = getenv("DM_VK_S24_TRACE"); trace = e && atoi(e) > 0; }
        static const void *tseen[1024]; static unsigned tri; static int tn;
        if (trace && tn < 512 && !s24_seen(t->mem, tseen, &tri, 1024)) { tn++;
            VKLOG("s24 trace: texture serial %llu (%ux%ux%u, nvn format %#x, %u levels) storage %p +%#llx, checked first at present %llu", (unsigned long long)t->serial, t->d.width,
                t->d.height, t->d.depth, t->d.format, im->levels, (const void *)t->mem, (unsigned long long)t->L.size, (unsigned long long)g_tex_frame); }
        if (vp2_tex(t)) {   // a vmode-2 texture: through the page table
            const int other = t->serial != im->pkg_tser; im->pkg_tser = t->serial; g_s27_checks += other;
            if (s24_hash_vp(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size) != im->pkg_vhash) { g_s24_hits++; g_s27_hits += other; pkg_upload(im, t, f); }
            g_s24_ns += now_ns() - q;
            return; }
        if (!t->mem || !s24_readable(im, t)) {
            static const void *sseen[256]; static unsigned sri; g_s24_skips++;
            if (!s24_seen(t->mem, sseen, &sri, 256)) VKLOG("[RED] NVN2 s24: packaged texture serial %llu (%ux%u, nvn format %#x): its storage %p +%#llx is not readable any more: not checked (%llu skipped so far)",
                (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, (const void *)t->mem, (unsigned long long)t->L.size, (unsigned long long)g_s24_skips);
            g_s24_ns += now_ns() - q;
            return; }
    }
#endif
    const int other = t->serial != im->pkg_tser; im->pkg_tser = t->serial; g_s27_checks += other;
    if (s24_hash(t->mem, t->L.size) != im->pkg_vhash) {
        g_s24_hits++; g_s27_hits += other;
        static int n; if (n++ < 48) VKLOG("s24: packaged texture serial %llu (%ux%ux%u, nvn format %#x, %llu bytes at %p, %u levels) changed since its upload with no write seen: uploaded again (present %llu)%s",
            (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, (unsigned long long)t->L.size, (const void *)t->mem, im->levels, (unsigned long long)g_tex_frame,
            other ? " (s27: another texture object over its storage, at its first use)" : " (the same texture object)");
        pkg_upload(im, t, f);
    }
    g_s24_ns += now_ns() - q;
    if (!(g_s24_checks & 0x3fff)) VKLOG("s24: %llu checks of packaged textures, %llu found changed and uploaded again, %.1f ms in all (present %llu); s27: %llu of the checks and %llu of the changed at another texture object's first use",
        (unsigned long long)g_s24_checks, (unsigned long long)g_s24_hits, g_s24_ns / 1e6, (unsigned long long)g_tex_frame, (unsigned long long)g_s27_checks, (unsigned long long)g_s27_hits);
}
static void pkg_upload(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    content_cpu_upload(im);
    pkg_upload_(im, t, f);
    clock_gettime(CLOCK_MONOTONIC, &b); g_pkg_ns += (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + (b.tv_nsec - a.tv_nsec)); g_pkg_n++; g_pkg_bytes += t->L.size;
    if (!(g_pkg_n % 128)) VKLOG("packaged uploads: %llu, %.1f MB, %.1f ms in all; parts: protect %.1f, scratch %.1f, deswizzle/copy %.1f, record %.1f ms",
        (unsigned long long)g_pkg_n, g_pkg_bytes / 1048576.0, g_pkg_ns / 1e6, g_pkg_part[0] / 1e6, g_pkg_part[1] / 1e6, g_pkg_part[2] / 1e6, g_pkg_part[3] / 1e6);
    if (!(g_pkg_n % 128)) { void nvn_dirty_mp_stats(uint64_t *, uint64_t *, uint64_t *); void nvn_dirty_stats(uint64_t *, uint64_t *); uint64_t c, pg, ns, fa, pr;   // r54 measure: the dirty-page module's mprotect (all callers, all threads) and write faults
        nvn_dirty_mp_stats(&c, &pg, &ns); nvn_dirty_stats(&fa, &pr);
        uint64_t nvn_dirty_edges(void);   // s28
        VKLOG("dirty pages: mprotect %llu calls, %llu pages, %.1f ms in all; write faults %llu, unprotects before host writes %llu, unprotected pages stamped %llu; s28: faults at an unprotected page with its neighbours released %llu",
            (unsigned long long)c, (unsigned long long)pg, ns / 1e6, (unsigned long long)fa, (unsigned long long)pr, (unsigned long long)nvn_dirty_stamped(), (unsigned long long)nvn_dirty_edges()); }
}
static void pkg_upload_vp(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
static void pkg_upload_s(vkimg_t *im, const tex_t *t, const vkfmt_t *f, const uint8_t *vsrc);
static void pkg_upload_(vkimg_t *im, const tex_t *t, const vkfmt_t *f) { pkg_upload_s(im, t, f, NULL); }
// vsrc: NVN2, a vmode-2 texture's storage as copied through the page table (pkg_upload_vp), read instead of t->mem; NULL: t->mem
static void pkg_upload_s(vkimg_t *im, const tex_t *t, const vkfmt_t *f, const uint8_t *vsrc) {
#if defined(DM_NXN1X)
    if (!vsrc && vp2_tex(t)) { pkg_upload_vp(im, t, f); return; }   // NVN2: a vmode-2 texture, through the page table (which calls back here with its copy)
    if (!vsrc && dm_sdk_ounce() && (!t->mem || !s24_readable_up(im, t))) {   // NVN2: storage gone since the texture was made (s24_check): no upload, the image as it is
        static const void *useen[256]; static unsigned uri; g_s24_upskips++;
        if (!s24_seen(t->mem, useen, &uri, 256)) VKLOG("[RED] NVN2 s24: packaged texture serial %llu (%ux%u, nvn format %#x): its storage %p +%#llx is not readable: not uploaded (%llu uploads skipped so far)",
            (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, (const void *)t->mem, (unsigned long long)t->L.size, (unsigned long long)g_s24_upskips);
        return; }
#endif
    { static int n; const char *e = getenv("DM_VK_LOG_UPLOADS"); if (e && atoi(e) && n < (atoi(e) > 1 ? atoi(e) : 60)) { n++;   // which packaged textures are (re-)uploaded
        VKLOG("upload: serial %llu %ux%ux%u nvn format %#x (vk %d) target %d levels %u layers %u flags %#x stride %u size %llu mem %p, tileH*100+pkg_block_h %d%s, present %llu, content %016llx",
              (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, f->vk, t->d.target, im->levels, im->layers, t->d.flags, t->d.stride, (unsigned long long)t->L.size,
              (const void *)t->mem, (int)t->L.tileH * 100 + t->d.pkg_block_h, im->packaged > 1 ? " (again)" : "", (unsigned long long)g_tex_frame, (unsigned long long)r153_hash(vsrc ? vsrc : t->mem, t->L.size)); } }
    if (t->mem) vk_tex_watch_note(im->packaged > 1 ? "upload again from storage" : "first upload from storage", t->serial, (uint64_t)(uintptr_t)t->mem, (uint64_t)(uintptr_t)t->mem + t->L.size, t->d.width, t->d.height, t->d.format);   // gpuowndiag
    if (t->d.width == 260 && t->d.height == 260) {   // texdiag (11-21, OoT terrain tiles, gwrange / gpuown): whether the tiles reach the GPU, and again after CPU rewrites
        static uint64_t n[2]; const int again = im->packaged > 1; uint64_t c = ++n[again];
        if (c <= 24 || !(c & (c - 1))) { const uint8_t *p = vsrc ? vsrc : t->mem; uint64_t nz = 0, m = t->L.size < (1u << 20) ? t->L.size : (1u << 20); for (uint64_t i = 0; i < m; i++) nz += p[i] != 0;
            VKLOG("texdiag upload: terrain-size texture 260x260x%u format %#x flags %#x pool flags %#x %s: %llu of %llu bytes non-zero, present %llu (first uploads %llu, re-uploads %llu)",
                  t->d.depth, t->d.format, t->d.flags, t->pool ? t->pool->flags : 0, again ? "uploaded again (its storage changed)" : "first upload", (unsigned long long)nz, (unsigned long long)m,
                  (unsigned long long)g_tex_frame, (unsigned long long)n[0], (unsigned long long)n[1]); } }
    if (im->packaged < 100) im->packaged++;
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer, to the last copy below (leaf: scratch memory, the dirty protection and the deswizzle inside)
    VkCommandBuffer cb = vk_upload_cmd();
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    // r75: the stamp before the protect (nvn_mtl.m pkg_upload): a write that faults while the pages are being protected gets a generation above it; a stamp taken after
    // the protect covered that generation, so the write and every later one to that page went unseen (MacNativeGames a166602, Tears of the Kingdom's glyph cache)
    const uint8_t *src = vsrc ? vsrc : t->mem;   // NVN2: pkg_upload_vp passes a copy made through the page table
    uint64_t q0 = now_ns(); im->pkg_stamp = nvn_dirty_now();
    if (src == t->mem) { if (pkg_protect()) nvn_dirty_protect(t->mem, t->L.size); im->pkg_span = nvn_dirty_span(t->mem, t->L.size); }
    else { im->pkg_span = NULL; im->pkg_sgen = nvn_sparse_gen(); }   // a vmode-2 texture: its changes are the page table's (nvn_sparse_since, vk_tex)
    im->pkg_seen = im->pkg_stamp; g_pkg_part[0] += now_ns() - q0;
    im->pkg_mem = src == t->mem ? t->mem : NULL; im->pkg_len = t->L.size; im->pkg_released = 0; im->pkg_d = t->d;   // (no pages of its own to release: img_drop, vk_tex_sweep)
    im->pkg_tser = t->serial;   // s27
    if (pkg_verify()) { im->pkg_vhash = s24_hash(src, t->L.size); im->pkg_vframe = g_tex_frame; }   // s24: before it is read for the copy (a write during it is seen next time)
    int linear = (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0;
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1, bpb = f->bpb;
    if (!bpb) { vk_up_unlock(); VKRED_ONCE(12, "packaged texture format %#x has no block size", t->d.format); return; }
    for (unsigned lv = 0; lv < im->levels; lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, z = im->d >> lv ? im->d >> lv : 1;
        uint32_t bx = (w + bw - 1) / bw, by = (h + bh - 1) / bh; size_t pitch = (size_t)bx * bpb, slab = pitch * by * z;
        for (unsigned ly = 0; ly < im->layers; ly++) {
            uint64_t q1 = now_ns(); VkBuffer sb; VkDeviceSize so; uint8_t *dst = vk_scratch(slab, 16, &sb, &so); uint64_t q2 = now_ns(); g_pkg_part[1] += q2 - q1;
            if (linear) {   // linear storage: rows of the texture's stride, one level, one layer
                size_t stride = t->d.stride ? t->d.stride : pitch;
                for (uint32_t y = 0; y < by * z; y++) memcpy(dst + y * pitch, src + (size_t)y * stride, pitch);
            } else if (nvn_deswizzle(&t->d, lv, ly, src, dst, pitch)) { vk_up_unlock(); VKRED_ONCE(13, "deswizzle refused texture format %#x target %d", t->d.format, t->d.target); return; }
            uint64_t q3 = now_ns(); g_pkg_part[2] += q3 - q2;
            VkBufferImageCopy r = {so, 0, 0, {im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, lv, ly, 1}, {0, 0, 0}, {w, h, z}};
            vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r); g_pkg_part[3] += now_ns() - q3;
        }
    }
    vk_up_unlock();   // R6/R7/R8 (parallel recording), R8
}
static int packaged_tex(const tex_t *t);
#if defined(DM_NXN1X)
// NVN2: a vmode-2 texture's upload. Its storage is copied through the page table into a buffer of its own (unmapped pages read 0), which pkg_upload_ then reads as it
// reads t->mem otherwise (passed as a parameter: no shared state between uploads); the address is never read directly. The copy is freed at once (the staging holds the data)
static uint64_t g_vp_uploads, g_vp_bytes, g_vp_ns;
static void pkg_upload_vp(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    uint8_t *buf = malloc(t->L.size ? t->L.size : 1);
    if (!buf) { VKRED_ONCE(178, "NVN2: no memory for a %llu-byte page-table copy of a sparse pool texture: not uploaded", (unsigned long long)t->L.size); return; }
    { uint64_t q = now_ns(); nvn_sparse_copy(t->pool, buf, (uint64_t)(uintptr_t)t->mem, t->L.size); g_vp_ns += now_ns() - q; }
    g_vp_uploads++; g_vp_bytes += t->L.size;
    if (g_vp_uploads <= 4 || !(g_vp_uploads & (g_vp_uploads - 1))) VKLOG("NVN2: sparse pool texture %ux%u nvn format %#x (%llu bytes) uploaded through the page table (%llu so far, %.0f MB)",
        t->d.width, t->d.height, t->d.format, (unsigned long long)t->L.size, (unsigned long long)g_vp_uploads, g_vp_bytes / 1048576.0);
    pkg_upload_s(im, t, f, buf);
    free(buf);
}
#endif
// NVN2 CPU time so far (ns): packaged uploads (pkg_upload, their page-table copies included), the page-table copies (pkg_upload_vp), s24 checks, the sparse layers'
// refresh (gfx_vk_core.c NVN2 30 s line)
void vk_tex_cpu_stats(uint64_t out[8]) {   // (+ out[4] readable queries' ns, out[5] their count, out[6] answers from the cache)
    out[0] = g_pkg_ns; out[2] = g_s24_ns; out[3] = 0;
#if defined(DM_NXN1X)
    out[1] = g_vp_ns; out[4] = g_s24_rd_ns; out[5] = g_s24_rd_n; out[6] = g_s24_rd_cached;
#else
    out[1] = 0; out[4] = out[5] = out[6] = 0;
#endif
    out[7] = 0;
}
static int cpu_first_use(vkimg_t *im, const tex_t *t, const vkfmt_t *f);   // r73
static int init_fill(vkimg_t *im, const tex_t *t, const vkfmt_t *f); static void zero_fill(vkimg_t *im, const vkfmt_t *f);   // r75
static int lin_copy(vkimg_t *im, const tex_t *t, const vkfmt_t *f);
static void vk_tex_view_drop_all(vkimg_t *im);
// Images over the same memory with different descriptions (the storage key differs) are separate Vulkan images: on the Switch a texture sees what another
// one over its memory rendered, here it does not. They are linked both ways at creation (up to 8 each), and a sampled image older than an overlapping
// image's last rendering is logged ([alias] stale read), to find what a game reinterprets
static uint64_t g_wseq;
static void alias_link(vkimg_t *im) {
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        vkimg_t *o = g_by_key.vals[i]; if (!o || o == im || o->pool != im->pool || o->off >= im->off + im->size || im->off >= o->off + o->size) continue;
        if (im->novl < 8 && o->novl < 8) { im->ovl[im->novl++] = o; o->ovl[o->novl++] = im; }
        else if (dm_sdk_ounce()) { static int nf; if (nf++ < 8) VKLOG("[alias] NVN2: %ux%u fmt %#x at %+lld not linked to %ux%u fmt %#x at %+lld over the same memory: %d / %d links already (8 each; its stale reads are not checked)",
            im->w, im->h, im->nvn_format, (long long)im->off, o->w, o->h, o->nvn_format, (long long)o->off, im->novl, o->novl); }
    }
}
static void alias_unlink(vkimg_t *im) {
    for (int i = 0; i < im->novl; i++) { vkimg_t *o = im->ovl[i]; for (int j = 0; j < o->novl; j++) if (o->ovl[j] == im) { o->ovl[j] = o->ovl[--o->novl]; break; } }
    im->novl = 0;
}
static void alias_check(vkimg_t *im) {   // im is about to be sampled
    for (int i = 0; i < im->novl; i++) {
        vkimg_t *o = im->ovl[i]; if (o->wseq <= im->wseq || o->wseq <= im->rseen) continue;
        im->rseen = o->wseq;
        if (dm_sdk_ounce()) {   // NVN2 diagnostic (DK black 3D): each (sampled, overlapping) pair once, up to 512 pairs, with what differs and both images' last rendered presents
            static const vkimg_t *pr[512][2]; static int np; int seen = 0;
            for (int j = 0; j < np && !seen; j++) seen = pr[j][0] == im && pr[j][1] == o;
            if (seen || np >= 512) continue;
            pr[np][0] = im; pr[np][1] = o; np++;
            const nvn_texdesc *a = &im->kd, *b = &o->kd; char df[240]; int p = 0; df[0] = 0;
            #define KD(fl) if (a->fl != b->fl && p < (int)sizeof df - 40) p += snprintf(df + p, sizeof df - (size_t)p, " " #fl " %#x/%#x", (unsigned)a->fl, (unsigned)b->fl);
            KD(target) KD(format) KD(flags) KD(samples) KD(width) KD(height) KD(depth) KD(levels) KD(stride) KD(pkg_block_h)
            #undef KD
            extern uint64_t g_tex_frame;
            VKLOG("[alias] stale read pair %d at present %llu: %ux%u fmt %#x levels %u at %+lld..%+lld (rendered at present %lld) sampled, but %ux%u fmt %#x levels %u at %+lld..%+lld rendered over its memory later (at present %lld); sampled/other differ in%s",
                np, (unsigned long long)g_tex_frame, im->w, im->h, im->nvn_format, im->levels, (long long)im->off, (long long)(im->off + im->size), (long long)im->wpres - 1,
                o->w, o->h, o->nvn_format, o->levels, (long long)o->off, (long long)(o->off + o->size), (long long)o->wpres - 1, p ? df : " nothing");
            continue;
        }
        static int n; if (n++ < 200) VKLOG("[alias] stale read: %ux%u fmt %#x levels %u at %+lld..%+lld sampled, but %ux%u fmt %#x levels %u at %+lld..%+lld rendered over its memory later",
            im->w, im->h, im->nvn_format, im->levels, (long long)im->off, (long long)(im->off + im->size), o->w, o->h, o->nvn_format, o->levels, (long long)o->off, (long long)(o->off + o->size));
    }
}
extern uint64_t g_tex_frame;   // presents so far, for the packaged textures' last use (vk_tex_sweep)
// Images are kept per storage key until their pool goes, so a texture the game streams into memory another texture had before gets an image of its own and the old one
// stayed: Xenoblade Chronicles 3 streams its field textures through pools that live for good, the device's GPU memory grew by about 470 MB a minute while walking (6.3 GB
// after 20 minutes) and Android's low-memory killer ended the game. A new image takes over: the images of the same pool over its memory that were not looked up in the
// last EVICT_KEEP presents go (their memory now holds another texture; one drawn again is made and filled again). Called with g_img_mu held
enum { EVICT_KEEP = 60 };
// Only resident duplicates are on this LRU. The sole copy of an evicted image
// is never pruned. All list operations are under g_img_mu; write markers only
// change the serial, and a stale duplicate is discarded before it can be used.
static void archive_cache_remove(struct nvn_img_archive *a) {
    if (!a || !a->resident) return;
    if (a->prev) a->prev->next = a->next; else g_archive_clean_first = a->next;
    if (a->next) a->next->prev = a->prev; else g_archive_clean_last = a->prev;
    a->prev = a->next = NULL; a->resident = 0; g_archive_clean_bytes -= a->bytes;
}
static void img_archive_close(vkimg_t *im) {
    if (!im->archive) return;
    archive_cache_remove(im->archive);
    if (atomic_load_explicit(&g_preserve_ready,memory_order_acquire)) {
        struct nvn_img_archive *a=im->archive; im->archive=NULL;
        g_archive_bytes-=a->bytes; g_archive_discard_bytes+=a->bytes;
        a->next=g_archive_discard; g_archive_discard=a;
        return; // close files at the next owner-safe point, outside g_img_mu
    }
    fclose(im->archive->file); g_archive_bytes -= im->archive->bytes;
    free(im->archive); im->archive = NULL;
}
static void archive_cache_touch(vkimg_t *im) {
    struct nvn_img_archive *a = im->archive;
    if (!a || !im->img) return;
    if (!archive_clean(im)) { img_archive_close(im); return; }
    if (a->resident && a->touched == g_tex_frame) return;
    archive_cache_remove(a);
    a->owner = im; a->resident = 1; a->touched = g_tex_frame;
    a->prev = g_archive_clean_last;
    if (a->prev) a->prev->next = a; else g_archive_clean_first = a;
    g_archive_clean_last = a; g_archive_clean_bytes += a->bytes;
    while (g_archive_clean_bytes > ARCHIVE_CLEAN_CACHE_BYTES && g_archive_clean_first)
        img_archive_close(g_archive_clean_first->owner);
}
static int archive_make_room(uint64_t bytes) {
    const uint64_t cap = 4ull << 30;
    uint64_t pending=nvnp_pending_bytes()+g_archive_discard_bytes;
    if (pending>cap || bytes>cap-pending) return 0;
    while (g_archive_bytes > cap - pending - bytes && g_archive_clean_first)
        img_archive_close(g_archive_clean_first->owner);
    // Enqueuing a close does not return disk capacity until fclose completes.
    pending=nvnp_pending_bytes()+g_archive_discard_bytes;
    if (pending>cap || bytes>cap-pending) return 0;
    return g_archive_bytes <= cap - pending - bytes;
}
static uint64_t img_release_completed(vkimg_t *im) {
    if (im->preserve) return 0;
    if (im->img && im->img == g_archive_pending_image) {
        nvna_workspace_wait(&g_archive_workspace);
        if (g_archive_workspace.pending) return 0;
        g_archive_pending_image = VK_NULL_HANDLE;
    }
    archive_cache_remove(im->archive);
    // Caller has excluded current and pending submissions. Views must die first;
    // advancing their generation also invalidates descriptor memoization.
    vk_tex_view_drop_completed(im);
    uint64_t freed = vk_image_release_completed(im->img, im->mem, im->sub, im->allocation_bytes);
    im->img = VK_NULL_HANDLE; im->mem = VK_NULL_HANDLE; im->sub = NULL;
    im->allocation_bytes = 0;
    if (im->scaled) {
        freed += img_release_completed(im->scaled);
        img_drop(im->scaled);
        im->scaled = NULL;
        im->scaled_valid = 0;
    }
    return freed;
}
static void archive_failure_at(vkimg_t *im, const char *op, VkResult result, const char *stage) {
    uint64_t n = ++g_archive_failed;
    if (n <= 16 || !(n & (n - 1))) VKLOG("[RED] NVN2 image archive: %s failed (%d), %ux%ux%u format %d; authoritative copy retained (%llu failures); stage %s",
        op, result, im->w, im->h, im->layers, im->fmt, (unsigned long long)n, stage ? stage : op);
}
static void archive_failure(vkimg_t *im, const char *op, VkResult result) { archive_failure_at(im, op, result, op); }
static uint64_t img_archive_release_impl(vkimg_t *im) {
    if (!im->img || im->refs || im->external_use || im->s2t || im->samples > 1 ||
        im->layout != VK_IMAGE_LAYOUT_GENERAL || im->last_use > vk_fence_done() ||
        im->img==g_archive_pending_image) return 0;
    if (archive_clean(im)) {
        // The completed image has not been written since this exact snapshot.
        // Reclaim it without a GPU download, fence wait or another disk write.
        im->archive->last_used = im->used;
        uint64_t bytes = im->archive->bytes, freed = img_release_completed(im);
        if (!im->img) {
            g_archive_clean_releases++; g_archive_clean_saved_bytes += bytes; g_img_mem_evicted++;
            if (im->keep_gw) { g_keep_gw_n--; g_keep_gw_bytes -= im->size > 0 ? (uint64_t)im->size : 0; im->keep_gw = 0; }
        }
        return freed;
    }
    // The live GPU image is authoritative after any possible write. Replacing
    // a stale duplicate never removes the only copy if a new save fails.
    img_archive_close(im);
    const vkfmt_t *f = vk_fmt(im->nvn_format); if (!f) return 0;
    VkResult r = archive_ready();
    if (r != VK_SUCCESS) { archive_failure_at(im, "save preparation", r, g_archive_workspace.stage); return 0; }
    struct nvn_img_archive *a = calloc(1, sizeof *a); if (!a) return 0;
    a->shape = (nvna_shape){im->fmt, im->aspect, im->w, im->h, im->d, im->layers, im->levels, f->bw, f->bh, f->bpb};
    a->bytes = nvna_size(&a->shape);
    a->last_used = im->used; a->owner = im; a->image_id = im->residency_id;
    a->content_serial = __atomic_load_n(&im->content_serial, __ATOMIC_RELAXED);
    // A bounded disk cache, not an image-sized RAM mirror on a unified-memory GPU.
    // Resident duplicates yield first to sole-copy archives, within the same cap.
    extern FILE *nvnh_image_archive_file(uint64_t bytes);
    if (!a->bytes) { free(a); archive_failure(im, "invalid backing size", VK_ERROR_FORMAT_NOT_SUPPORTED); return 0; }
    if (!archive_make_room(a->bytes)) {
        free(a); archive_failure(im, "backing capacity", VK_ERROR_OUT_OF_HOST_MEMORY); return 0;
    }
    if (!(a->file = nvnh_image_archive_file(a->bytes))) {
        free(a); archive_failure(im, "backing file creation", VK_ERROR_OUT_OF_HOST_MEMORY); return 0;
    }
    uint64_t started = now_ns();
    r = archive_transfer(im->img, &a->shape, a->file, 0);
    g_archive_save_ns += now_ns() - started;
    if (r != VK_SUCCESS) { fclose(a->file); free(a); archive_failure_at(im, "save", r, g_archive_workspace.stage); return 0; }
    if (a->content_serial != __atomic_load_n(&im->content_serial, __ATOMIC_RELAXED)) {
        fclose(a->file); free(a); archive_failure(im, "content changed during save", VK_NOT_READY); return 0;
    }
    // Commit only after the entire file was written and flushed successfully.
    im->archive = a; g_archive_bytes += a->bytes; g_archive_written_bytes += a->bytes;
    uint64_t freed = img_release_completed(im);
    if (im->keep_gw) { g_keep_gw_n--; g_keep_gw_bytes -= im->size > 0 ? (uint64_t)im->size : 0; im->keep_gw = 0; }
    g_img_mem_evicted++;
    uint64_t n = ++g_archive_saved;
    if (n <= 16 || !(n & (n - 1))) VKLOG("NVN2 image archive: saved %ux%ux%u format %d, %.1f MiB on disk; %.1f MiB device allocation freed (%llu saved, %.1f MiB backing)",
        im->w, im->h, im->layers, im->fmt, a->bytes / 1048576.0, freed / 1048576.0, (unsigned long long)n, g_archive_bytes / 1048576.0);
    return freed;
}
static uint64_t img_archive_release(vkimg_t *im) {
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_SAVE, im->allocation_bytes);
    uint64_t failed = g_archive_failed;
    uint64_t freed = img_archive_release_impl(im);
    vk_f0_end(&diag, g_archive_failed != failed);   // observed save failure, not a driver VkResult
    return freed;
}
static void archive_restore_cost(vkimg_t *im, uint64_t started) {
    uint64_t elapsed = now_ns() - started; g_archive_restore_ns += elapsed;
    if (elapsed > g_restore_max_ns) g_restore_max_ns = elapsed;
    if (elapsed >= 50000000ull) {
        uint64_t n = ++g_restore_slow;
        if (n <= 16 || !(n & (n - 1))) VKLOG("NVN2 residency stall: restore %.2f ms, %ux%ux%u format %d, heap %d%% (%llu slow restores; includes nested allocation recovery)",
            elapsed / 1e6, im->w, im->h, im->layers, im->fmt, g_nvn2_heap_pct, (unsigned long long)n);
    }
}
static int img_archive_restore(vkimg_t *im) {
    if (im->img) { archive_cache_touch(im); return 1; }
    if (!im->archive) return 0;
    const uint64_t started = now_ns();
    nvf0_scope diag; vk_f0_begin(&diag, NVF0_RESTORE, im->archive->bytes);
    VkResult r = archive_ready();
    if (r != VK_SUCCESS) {
        archive_restore_cost(im, started);
        archive_failure_at(im, "restore preparation", r, g_archive_workspace.stage); vk_f0_end(&diag, r); return 0;
    }
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, im->cflags,
        im->target == 2 ? VK_IMAGE_TYPE_3D : im->target == 0 || im->target == 3 ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D,
        im->fmt, {im->w, im->h, im->d}, im->levels, im->layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        im->usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage image = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; void *sub = NULL;
    VkMemoryRequirements req = {0}; VkDeviceSize offset = 0;
#if defined(DM_NXN1X)
    nvnh_restore_scope memory_scope = {0};
#endif
    const char *stage = "vkCreateImage";
    r = vkCreateImage(V.dev, &ci, NULL, &image);
    if (r != VK_SUCCESS) goto failed;
    vkGetImageMemoryRequirements(V.dev, image, &req);
    uint32_t type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    // Restoring into an already committed free slot adds no device memory. The
    // previous unconditional pressure trim could evict another live image just
    // to reuse that slot, creating a save/restore cycle with no budget benefit.
    uint64_t growth = vk_image_memory_growth(&req, type);
#if defined(DM_NXN1X)
    memory_scope = nvnh_memory_restore_begin(type, growth, req.size);
#endif
    uint64_t need = growth ? vk_nvn2_memory_need(growth) : 0;
    if (!growth) { g_restore_reuse_slots++; g_restore_reuse_bytes += req.size; }
    uint64_t target = img_reclaim_target(&req, type);
    if (need) img_evict_idle(need < target ? need : target, 8);
    stage = "image allocation";
    memory = img_memory_recover(&req, type, &offset, &sub);
    if (!memory) { r = VK_ERROR_OUT_OF_DEVICE_MEMORY; goto failed; }
    stage = "vkBindImageMemory";
    r = vkBindImageMemory(V.dev, image, memory, offset);
    if (r != VK_SUCCESS) goto failed;
    r = archive_transfer(image, &im->archive->shape, im->archive->file, 1);
    stage = g_archive_workspace.stage;
    if (r != VK_SUCCESS) goto failed;
    im->img = image; im->mem = memory; im->sub = sub; im->allocation_bytes = req.size; im->layout = VK_IMAGE_LAYOUT_GENERAL;
    residency_reused(im, im->archive->last_used);
    archive_cache_touch(im);
    archive_restore_cost(im, started);
    uint64_t n = ++g_archive_restored;
    if (n <= 16 || !(n & (n - 1))) VKLOG("NVN2 image archive: restored %ux%ux%u format %d, all levels and layers (%llu restored)",
        im->w, im->h, im->layers, im->fmt, (unsigned long long)n);
    vk_f0_end(&diag, 0);
#if defined(DM_NXN1X)
    nvnh_memory_restore_end(memory_scope, 1);
#endif
    return 1;
failed:
    if (image && image == g_archive_pending_image && g_archive_workspace.pending) {
        g_archive_retired.image = image; g_archive_retired.memory = memory;
        g_archive_retired.sub = sub; g_archive_retired.bytes = req.size;
    } else if (image) vk_image_release_completed(image, memory, sub, req.size);
    archive_restore_cost(im, started);
    archive_failure_at(im, "restore", r, stage);
    vk_f0_end(&diag, r);
#if defined(DM_NXN1X)
    nvnh_memory_restore_end(memory_scope, 0);
#endif
    if (vk_tex_preserve_enabled())
        die("[vk] required archive restore failed (%d, %s); backing retained, command not skipped",r,stage?stage:"unknown");
    return 0;   // retain the file; never substitute an empty image for it
}
// S1e (parallel recording): while worker segments are out (vk_tex_par_hold .. vk_tex_par_release, the replay thread around a part's hand-outs and its join), an image object
// the replay thread drops (evict_overlaps at a later record's new image, the sweeps) is kept, not freed: a worker may hold it (its pass's P_att, a lookup between vk_tex and
// its view). Its Vulkan objects go as before (deferred to the fence); only the struct's memory waits for the join. Default path: one load of g_par_hold per drop
_Thread_local int g_vk_tex_wk, g_vk_tex_refused;
static _Atomic int g_par_hold; static vkimg_t **g_zomb; static int g_nzomb, g_capzomb; static pthread_mutex_t g_zomb_mu = PTHREAD_MUTEX_INITIALIZER;
static void img_obj_free(vkimg_t *im) {
    if (!atomic_load_explicit(&g_par_hold, memory_order_acquire)) { free(im); return; }
    pthread_mutex_lock(&g_zomb_mu);
    if (g_nzomb == g_capzomb) { int c = g_capzomb ? 2 * g_capzomb : 64; void *p = realloc(g_zomb, (size_t)c * sizeof *g_zomb); if (!p) die("[vk] S1e: no memory for dropped images"); g_zomb = p; g_capzomb = c; }
    g_zomb[g_nzomb++] = im;
    pthread_mutex_unlock(&g_zomb_mu);
}
void vk_tex_par_hold(void) { atomic_store_explicit(&g_par_hold, 1, memory_order_release); }
void vk_tex_par_release(void) {   // replay thread, no worker recording
    atomic_store_explicit(&g_par_hold, 0, memory_order_release);
    pthread_mutex_lock(&g_zomb_mu); for (int i = 0; i < g_nzomb; i++) free(g_zomb[i]); g_nzomb = 0; pthread_mutex_unlock(&g_zomb_mu);
}
static void img_drop(vkimg_t *im) {
    if (im->scaled) { img_drop(im->scaled); im->scaled = NULL; }
    if (im->preserve) {
        // Registry removal is allowed; the worker still owns a source lease.
        // It reads only the pinned version/handle, never the possibly dead pool.
        im->preserve_orphan=1; nvnp_cancel(&im->preserve->job); alias_unlink(im);
        // Release guest tracking while this pool incarnation is still valid.
        // Deferring it could unprotect a new pool at the same guest address.
        if (im->lin_mem) { nvn_dirty_release(im->lin_mem,im->lin_len); im->lin_mem=NULL; }
        if (im->packaged && !im->pkg_released && im->pkg_mem) {
            nvn_dirty_release(im->pkg_mem,im->pkg_len); im->pkg_released=1; im->pkg_mem=NULL;
        }
        im->unc_pool=NULL;
#if defined(DM_NXN1X)
        if (im->content_id) { nvnc_image_del(im->content_id); im->content_id=0; }
#endif
        return;
    }
    if (im->keep_gw) { g_keep_gw_n--; g_keep_gw_bytes -= im->size > 0 ? (uint64_t)im->size : 0; im->keep_gw = 0; }
#if defined(DM_NXN1X)
    if (im->content_id) nvnc_image_del(im->content_id);
#endif
    // A pool can disappear after a failed archive wait. Its pending image must
    // outlive the transfer, even though the guest no longer needs its contents.
    if (im->img && im->img == g_archive_pending_image && g_archive_workspace.pending) {
        g_archive_retired.image = im->img; g_archive_retired.memory = im->mem;
        g_archive_retired.sub = im->sub; g_archive_retired.bytes = im->allocation_bytes;
        im->img = VK_NULL_HANDLE; im->mem = VK_NULL_HANDLE; im->sub = NULL;
    }
    img_archive_close(im);
    if (im->s2t) s2tile_free(im); if (im->lin_mem) nvn_dirty_release(im->lin_mem, im->lin_len); if (im->packaged && !im->pkg_released && im->pkg_mem) nvn_dirty_release(im->pkg_mem, im->pkg_len); g_img_n--; g_img_bytes -= im->size > 0 ? (uint64_t)im->size : 0; alias_unlink(im); vk_tex_view_drop_all(im); if (im->img) { if (im->sub) vk_defer_free_image_sub(im->img, im->sub); else vk_defer_free_image(im->img, im->mem); } img_obj_free(im); }
// NVN2 (Switch 2 boots): an image of 32 MiB or more that goes, with why. MKW 11-16: three 8192x8192x68 arrays made again 5 s after the first ones with no line saying
// what became of those (images are kept per storage key: pool, offset and the whole description)
static void img_drop_why(vkimg_t *im, const char *why) {
    if (im->gaddr && im->size > 0) vk_tex_watch_note(why, 0, im->gaddr, im->gaddr + (uint64_t)im->size, im->w, im->h, im->nvn_format);   // gpuowndiag: an image over the range dropped
    if (im->refs > 0) {   // NVN2 (sparse_share): another storage key still uses the image; this key's entry was removed by the caller
        im->refs--; static int n; if (n++ < 64) VKLOG("NVN2 sparse: a key of the shared image %ux%ux%u (storage %#llx) went (%s); %d more keep it", im->w, im->h, im->layers,
            (unsigned long long)im->gaddr, why, im->refs + 1);
        return; }
    if (dm_sdk_ounce() && im->size >= (int64_t)(32ll << 20)) { static int n; if (n++ < 256)
        VKLOG("NVN2 big image dropped (%s): %ux%u, %u levels, %u layers, nvn format %#x, %.1f MiB%s, pool %#llx +%#llx, last looked up %llu presents ago", why, im->w, im->h, im->levels, im->layers,
            im->nvn_format, im->size / 1048576.0, im->s2t ? ", sparse tiles" : "", (unsigned long long)im->pool, (unsigned long long)im->off, (unsigned long long)(g_tex_frame - im->used)); }
    img_drop(im);
}
// A different texture description over a pool is not proof that its old GPU
// contents are dead. Keep the storage key until the pool is finalized. Pressure
// can move the contents to the archive, but cannot discard the only copy.
static int preserve_enqueue(vkimg_t *im) { // g_img_mu; bounded metadata only
    if (!vk_tex_preserve_enabled() || im->preserve || im->preserve_orphan ||
        im->preserve_after>g_tex_frame || archive_clean(im) ||
        image_eligibility(im,vk_fence_done()).source!=NVNI_SAVE_REQUIRED) return 0;
    const vkfmt_t *f=vk_fmt(im->nvn_format); if (!f) return 0;
    nvna_shape shape={im->fmt,im->aspect,im->w,im->h,im->d,im->layers,im->levels,f->bw,f->bh,f->bpb};
    uint64_t bytes=nvna_size(&shape), pending=nvnp_pending_bytes();
    if (!bytes || bytes>NVNP_BYTES || pending>NVNP_BYTES-bytes || !archive_make_room(bytes)) return 0;
    struct archive_save_job *j=calloc(1,sizeof *j);
    struct nvn_img_archive *a=calloc(1,sizeof *a);
    if (!j || !a) { free(j); free(a); return 0; }
    a->shape=shape; a->bytes=bytes; a->last_used=im->used; a->owner=im;
    a->image_id=im->residency_id; a->content_serial=__atomic_load_n(&im->content_serial,__ATOMIC_ACQUIRE);
    j->owner=im; j->archive=a; j->image=im->img;
    j->job_id=g_preserve_queued+1;
    j->job.live_version=&im->content_serial; j->job.image_id=a->image_id;
    j->job.version=a->content_serial; j->job.bytes=bytes;
    if (!nvnp_submit(&j->job)) { free(a); free(j); return 0; }
    im->preserve=j; j->next=g_preserve_jobs; g_preserve_jobs=j; ++g_preserve_queued;
    return 1;
}
void vk_tex_preserve_poll(uint64_t need) {
    if (!atomic_load_explicit(&g_preserve_ready,memory_order_acquire)) return;
    struct archive_save_job *garbage=NULL;
    nvnr_lock(&g_img_mu);
    struct archive_save_job **at=&g_preserve_jobs;
    while (*at) {
        struct archive_save_job *j=*at;
        if (!nvnp_done(&j->job)) { at=&j->next; continue; }
        vkimg_t *im=j->owner;
        // Do not turn collection into a global wait under the image lock.
        // Future NVN uses of an upgraded/orphaned source retain their own pin.
        if (!g_preserve_draining && ((j->retired && j->retired_fence>vk_fence_done()) ||
            (im->preserve_orphan && (im->last_use>vk_fence_done() || im->external_use)))) {
            at=&j->next; continue;
        }
        *at=j->next;
        im->preserve=NULL; nvnp_release(&j->job);
        g_preserve_ns+=j->elapsed_ns; g_preserve_submits+=j->submissions;
        const int valid=nvnp_publishable(&j->job,im->residency_id,im->img==j->image,im->preserve_orphan);
        if (valid) {
            img_archive_close(im); im->archive=j->archive; j->archive=NULL;
            g_archive_bytes+=im->archive->bytes; g_archive_written_bytes+=im->archive->bytes;
            g_preserve_written+=im->archive->bytes; ++g_preserve_complete; ++g_archive_saved;
            archive_cache_touch(im);
        } else {
            if (j->job.result==VK_SUCCESS || j->job.result==VK_NOT_READY) ++g_preserve_stale;
            else {
                ++g_preserve_failed;
                archive_failure_at(im,"background preservation",j->job.result,j->stage);
            }
            // Do not continually download an image which becomes active again.
            im->preserve_after=g_tex_frame+30;
        }
        const uint64_t completed=g_preserve_complete+g_preserve_stale+g_preserve_failed;
        if (completed<=8 || !(completed&(completed-1)))
            VKLOG("NVN2 F4C job: job=%llu image=%llu version=%llu result=%d published=%d worker_ms=%.3f submits=%llu frame=%llu; source retained until validated; stage=%s",
                (unsigned long long)j->job_id,(unsigned long long)j->job.image_id,
                (unsigned long long)j->job.version,j->job.result,valid,j->elapsed_ns/1e6,
                (unsigned long long)j->submissions,(unsigned long long)g_tex_frame,j->stage?j->stage:"cancelled before transfer");
        if (j->retired) {
            // Private download is done, but an NVN upgrade/copy can still use
            // the old image. Keep its normal submission retirement dependency.
            vk_image_release_completed(j->retired,j->retired_memory,j->retired_sub,j->retired_bytes);
        }
        if (im->preserve_orphan) { img_release_completed(im); img_drop(im); }
        j->next=garbage; garbage=j;
    }
    // Plan at most once per 250 ms, not once for every submission/restore.
    static uint64_t plan_at;
    uint64_t now=now_ns();
    if (need && vk_tex_preserve_enabled() && now-plan_at>=250000000ull) {
        plan_at=now; g_preserve_plan=1;
        img_evict_nvn2(need,8,0); g_preserve_plan=0;
    }
    struct nvn_img_archive *discard=g_archive_discard; g_archive_discard=NULL;
    nvnr_unlock(&g_img_mu);
    while (garbage) {
        struct archive_save_job *j=garbage; garbage=j->next;
        if (j->archive) { if (j->archive->file) fclose(j->archive->file); free(j->archive); }
        free(j);
    }
    uint64_t closed_bytes=0;
    while (discard) {
        struct nvn_img_archive *a=discard; discard=a->next;
        fclose(a->file); closed_bytes+=a->bytes; free(a);
    }
    if (closed_bytes) {
        nvnr_lock(&g_img_mu);
        g_archive_discard_bytes-=closed_bytes;
        nvnr_unlock(&g_img_mu);
    }
}
static void evict_overlaps(uint64_t pool, int64_t off, int64_t size) {
    akey ks[64]; int n = 0;
    for (uint32_t i = 0; i < g_by_key.cap && n < 64; i++) {
        vkimg_t *o = g_by_key.vals[i];
        if (!o || o->pool != pool || o->off >= off + size || off >= o->off + o->size || g_tex_frame - o->used < EVICT_KEEP) continue;
        if (dm_sdk_ounce() && (o->archive || o->external_use || o->gpu_w || !o->packaged || gpu_only_content(o))) {
            if (o->img && !o->keep_gw) {
                o->keep_gw = 1; g_keep_gw_n++; g_keep_gw_bytes += o->size > 0 ? (uint64_t)o->size : 0;
            }
            continue;
        }
        memcpy(&ks[n++], g_by_key.keys + (size_t)i * g_by_key.klen, sizeof(akey));
    }
    for (int j = 0; j < n; j++) {
        vkimg_t *o = nvn_hm_get(&g_by_key, &ks[j]); if (!o) continue;
        nvn_hm_del(&g_by_key, &ks[j]); img_drop_why(o, "a newer texture over its memory"); g_img_evicted++;
    }
}
static void ovw_sweep(void) {   // relovl: under g_img_mu, once a present when an image was marked
    if (!g_ovw_pending) return;
    g_ovw_pending = 0; akey ks[256]; int n = 0;
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        vkimg_t *o = g_by_key.vals[i];
        if (!o || !o->ovw || !o->keep_gw) continue;
        if (g_tex_frame - o->used < EVICT_KEEP || o->last_use > vk_fence_done() || o->refs || o->external_use || o->archive) { g_ovw_pending = 1; continue; }   // looked at again later
        if (n == 256) { g_ovw_pending = 1; break; }
        memcpy(&ks[n++], g_by_key.keys + (size_t)i * g_by_key.klen, sizeof(akey));
    }
    for (int j = 0; j < n; j++) {
        vkimg_t *o = nvn_hm_get(&g_by_key, &ks[j]); if (!o) continue;
        uint64_t b = o->size > 0 ? (uint64_t)o->size : 0;
        if (b >= (1u << 20)) {   // relovl diag (11-21, NPARK: does a dropped big image's memory overlap terrain tiles or textures the terrain reads?): every drop of 1 MiB or more, with
            // the live images over the same memory (pool, offset, size, format, packaged = CPU content / gpu_w = the GPU wrote it), up to 12
            VKLOG("relovl diag: dropping %ux%u nvn format %#x (%.1f MiB) pool %#llx +%#llx..+%#llx (host %#llx), last looked up %llu presents ago; images over it now:", o->w, o->h, o->nvn_format, b / 1048576.0,
                  (unsigned long long)o->pool, (unsigned long long)o->off, (unsigned long long)(o->off + o->size), (unsigned long long)o->gaddr, (unsigned long long)(g_tex_frame - o->used));
            int shown = 0;
            for (uint32_t i = 0; i < g_by_key.cap && shown < 12; i++) { const vkimg_t *x = g_by_key.vals[i];
                if (!x || x == o || x->pool != o->pool || x->off >= o->off + o->size || o->off >= x->off + x->size) continue;
                shown++; VKLOG("relovl diag:   %ux%ux%u nvn format %#x flags %#x at +%#llx..+%#llx: %s%s%s, last looked up %llu presents ago", x->w, x->h, x->layers, x->nvn_format, x->flags,
                    (unsigned long long)x->off, (unsigned long long)(x->off + x->size), x->packaged ? "CPU content (packaged)" : "GPU / zero content", x->gpu_w ? ", GPU-written" : "",
                    x->keep_gw ? ", kept" : "", (unsigned long long)(g_tex_frame - x->used)); }
            if (!shown) VKLOG("relovl diag:   none in the image table");
        }
        nvn_hm_del(&g_by_key, &ks[j]); img_drop_why(o, "relovl: kept GPU-only image whose memory a newer image wrote"); g_ovw_dropped++; g_ovw_bytes += b;
    }
    if (n) { static uint64_t said; if (g_ovw_dropped >= said * 2 || !said) { said = g_ovw_dropped;
        VKLOG("relovl: dropped %llu kept GPU-only images (%.1f MiB guest layout) whose memory a newer image wrote; kept now %llu (DM_VK_KEPT_OVERWRITE_DROP=0 off)",
              (unsigned long long)g_ovw_dropped, g_ovw_bytes / 1048576.0, (unsigned long long)g_keep_gw_n); } }
}
static void keep_gw_report(void) {
    static uint64_t t; uint64_t ns = now_ns(); if (ns - t < 10000000000ull) return; t = ns;
    vk_f0_report();
    VKLOG("NVN2 image residency: retained GPU-only images %llu (%.1f MiB guest layout, not device allocation); "
          "archive %.1f MiB on disk, saved %llu, restored %llu, failures %llu (copy retained)",
          (unsigned long long)g_keep_gw_n, g_keep_gw_bytes / 1048576.0, g_archive_bytes / 1048576.0,
          (unsigned long long)g_archive_saved, (unsigned long long)g_archive_restored, (unsigned long long)g_archive_failed);
    VKLOG("NVN2 F3 eligibility: actual_cheap_freed=%llu preserve_scanned=%llu pinned_scanned=%llu; archive=exact-version guest=legacy-only superseded-alias=disabled NCE-publication=disabled",
        (unsigned long long)g_f3_cheap_bytes,(unsigned long long)g_f3_preserve_bytes,(unsigned long long)g_f3_pinned_bytes);
    // Interval costs, not nested sums: trim includes saves, and a restore can
    // itself trigger trimming before allocating. Frame means a presented frame.
    static uint64_t frame0, save0, restore0, save_ns0, restore_ns0, trim_ns0, freed0, mixed0, yielded0, refault0, save_submit0, restore_submit0;
    uint64_t frames = g_tex_frame - frame0;
    double scale = frames ? 1.0 / (1000000.0 * (double)frames) : 0.0;
    VKLOG("NVN2 residency cost: %llu presents; save %llu, restore %llu, refault %llu; "
          "save %.2f, restore %.2f, trim %.2f ms/present (nested); actual freed %.1f MiB; "
          "mixed blocks skipped %llu, budget yields %llu; transfer submits save %llu, restore %llu",
          (unsigned long long)frames, (unsigned long long)(g_archive_saved - save0),
          (unsigned long long)(g_archive_restored - restore0), (unsigned long long)(g_residency_refaults - refault0),
          (g_archive_save_ns - save_ns0) * scale, (g_archive_restore_ns - restore_ns0) * scale, (g_trim_ns - trim_ns0) * scale,
          (g_trim_freed - freed0) / 1048576.0, (unsigned long long)(g_trim_mixed - mixed0), (unsigned long long)(g_trim_yielded - yielded0),
          (unsigned long long)(g_archive_save_submits - save_submit0), (unsigned long long)(g_archive_restore_submits - restore_submit0));
    frame0 = g_tex_frame; save0 = g_archive_saved; restore0 = g_archive_restored;
    save_ns0 = g_archive_save_ns; restore_ns0 = g_archive_restore_ns; trim_ns0 = g_trim_ns;
    freed0 = g_trim_freed; mixed0 = g_trim_mixed; yielded0 = g_trim_yielded; refault0 = g_residency_refaults;
    save_submit0 = g_archive_save_submits; restore_submit0 = g_archive_restore_submits;
    static uint64_t hot0, pressure0, oom0, slots0, bytes0;
    VKLOG("NVN2 residency reuse: pressure trims %llu, allocation-failure trims %llu, hot candidates protected %llu; restores into existing slots %llu (%.1f MiB without new image allocation / pretrim)",
        (unsigned long long)(g_trim_pressure_calls - pressure0), (unsigned long long)(g_trim_oom_calls - oom0),
        (unsigned long long)(g_trim_hot - hot0), (unsigned long long)(g_restore_reuse_slots - slots0),
        (g_restore_reuse_bytes - bytes0) / 1048576.0);
    hot0 = g_trim_hot; pressure0 = g_trim_pressure_calls; oom0 = g_trim_oom_calls;
    slots0 = g_restore_reuse_slots; bytes0 = g_restore_reuse_bytes;
    static uint64_t clean0, clean_bytes0, targeted0, escalated0, requested0, recovered0, slow_trim0, slow_restore0;
    VKLOG("NVN2 residency backing reuse: clean releases %llu, avoided downloads/writes %.1f MiB; resident disk duplicates %.1f MiB (512 MiB cap within 4096 MiB archive cap)",
        (unsigned long long)(g_archive_clean_releases - clean0), (g_archive_clean_saved_bytes - clean_bytes0) / 1048576.0,
        g_archive_clean_bytes / 1048576.0);
    VKLOG("NVN2 residency recovery: targeted %llu, escalated %llu; first targets %.1f MiB, total recovered %.1f MiB; max trim %.2f ms, restore %.2f ms (nested); over 50 ms trim %llu, restore %llu",
        (unsigned long long)(g_reclaim_targeted - targeted0), (unsigned long long)(g_reclaim_escalated - escalated0),
        (g_reclaim_requested - requested0) / 1048576.0, (g_reclaim_freed - recovered0) / 1048576.0,
        g_trim_max_ns / 1e6, g_restore_max_ns / 1e6, (unsigned long long)(g_trim_slow - slow_trim0),
        (unsigned long long)(g_restore_slow - slow_restore0));
    clean0 = g_archive_clean_releases; clean_bytes0 = g_archive_clean_saved_bytes;
    targeted0 = g_reclaim_targeted; escalated0 = g_reclaim_escalated; requested0 = g_reclaim_requested; recovered0 = g_reclaim_freed;
    slow_trim0 = g_trim_slow; slow_restore0 = g_restore_slow; g_trim_max_ns = g_restore_max_ns = 0;
    // These are current bindings, not the cumulative shared-bind hit counters.
    // Keys include physical storage and compatible format/layout. Unbound tiles,
    // private slots and allocator slack are outside this payload accounting.
    uint64_t mapped = 0, unique = 0, places = 0;
    pthread_mutex_lock(&g_ps_mu);
    for (uint32_t i = 0; i < g_ps_n; i++) if (g_ps[i].live) {
        mapped += (uint64_t)g_ps[i].rc * 65536; unique += 65536; places += g_ps[i].rc;
    }
    for (uint32_t i = 0; i < g_pt_n; i++) if (g_pt[i].live) {
        mapped += (uint64_t)g_pt[i].rc * g_pt[i].size; unique += g_pt[i].size; places += g_pt[i].rc;
    }
    pthread_mutex_unlock(&g_ps_mu);
    VKLOG("NVN2 physical sharing: %llu current bindings, mapped payload %.1f MiB, unique compatible shared payload %.1f MiB, duplication avoided %.1f MiB (excludes private/unbound tiles and allocation slack)",
        (unsigned long long)places, mapped / 1048576.0, unique / 1048576.0, (mapped - unique) / 1048576.0);
}
// r72: an image looked up with IMAGE that has no storage usage takes a new image with it: the old one's levels and layers copied in on the upload command buffer (which
// runs before the submission's main one), its views dropped, and the old image freed once the GPU is done with it; the vkimg_t stays, so the key and the aliases do.
// As the Mac's image_prepare / tex_upgrade. Writes the current submission's main command buffer made to the old image before this lookup are not carried over (counted).
// Not for multisampled images or formats that cannot be storage images (compressed: such a texture is sampled as it is). Called with g_img_mu held, in a submission
static uint64_t g_img_upgraded, g_img_upg_late;
static int img_upgrade(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    int tw; if (im->samples > 1 || (!img_storage_ok(f, &tw) && !img_typed_storage_ok(f))) return 0;   // ootu32: also for declared-format views
    tex_t t2 = *t; t2.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
    vkimg_t *n = img_new(&t2, f); if (!n) return 0;
    if (!(n->usage & VK_IMAGE_USAGE_STORAGE_BIT) || n->w != im->w || n->h != im->h || n->d != im->d || n->layers != im->layers || n->levels != im->levels || n->fmt != im->fmt) {
        if (n->sub) vk_defer_free_image_sub(n->img, n->sub); else vk_defer_free_image(n->img, n->mem);
        free(n); return 0; }
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer (leaf; this runs under g_img_mu)
    VkCommandBuffer cb = vk_upload_cmd();
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    vk_tex_prepare(cb, n);
    if (im->layout == VK_IMAGE_LAYOUT_GENERAL) {   // (an image never used has no contents to carry)
        VkImageCopy rg[16]; uint32_t nr = 0;
        for (uint32_t l = 0; l < im->levels && nr < 16; l++) {
            VkExtent3D e = {im->w >> l ? im->w >> l : 1, im->h >> l ? im->h >> l : 1, im->d >> l ? im->d >> l : 1};
            rg[nr++] = (VkImageCopy){{im->aspect, l, 0, im->layers}, {0, 0, 0}, {im->aspect, l, 0, im->layers}, {0, 0, 0}, e};
        }
        vkCmdCopyImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, n->img, VK_IMAGE_LAYOUT_GENERAL, nr, rg);
    }
    VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb2, 0, NULL, 0, NULL);
    vk_up_unlock();
    if (im->used == g_tex_frame && im->gpu_w) g_img_upg_late++;
    vk_tex_view_drop_all(im);
    if (im->preserve && im->preserve->image==im->img) {
        struct archive_save_job *j=im->preserve; nvnp_cancel(&j->job);
        j->retired=im->img; j->retired_memory=im->mem;
        j->retired_sub=im->sub; j->retired_bytes=im->allocation_bytes; j->retired_fence=im->last_use;
    } else if (im->sub) vk_defer_free_image_sub(im->img, im->sub); else vk_defer_free_image(im->img, im->mem);
    vk_tex_content_changed(im);
    if (im->scaled) { img_drop(im->scaled); im->scaled = NULL; }
    im->img = n->img; im->mem = n->mem; im->sub = n->sub; im->allocation_bytes = n->allocation_bytes; im->usage = n->usage; im->cflags = n->cflags; im->layout = n->layout;
    im->flags |= NVN_TEXTURE_FLAGS_IMAGE;
    free(n);
    uint64_t k = ++g_img_upgraded; if (!(k & (k - 1)))
        VKLOG("r72: %llu textures given storage usage for an image lookup, contents copied (%ux%u format %d levels %u layers %u; %llu also used earlier in their submission)",
              (unsigned long long)k, im->w, im->h, im->fmt, im->levels, im->layers, (unsigned long long)g_img_upg_late);
    return 1;
}
// r75: a texture over the same storage as an image that differs only by the other format of an sRGB / UNORM pair is a view of that image (nvn_mtl.m srgb_twins): the
// key holds the format, so it made its own image, evicting the one with the contents (evict_overlaps), and read an empty image. Mario Kart World's Grand Prix movie (an
// RGBA8 target the YUV pass writes, sampled through an RGBA8_SRGB texture) showed a frozen frame on the Mac; Hyrule Warriors: Age of Imprisonment uses 0x25 and 0x38
// over the same storage. Images are made MUTABLE_FORMAT, and vk_tex_view takes the texture's own format. Uncompressed
// single-sampled images only, as srgb_twin. Called with g_img_mu held. DM_VK_SRGB_TWIN=0 turns it off
static uint64_t g_twin_hits;
// NVN2 (11-16, emu parity: Eden's texture cache keys images by GPU address): MKW made each of its three 8192x8192x68 terrain arrays twice, ~2.5 s apart, both kept
// at the per-image cap (2376 MB against the emu's 1069). A sparse-layer or sparse-tiles image (a vmode-2 pool, 32 MiB or more) asked for under another storage key with the
// same guest storage address and the same layout reuses the live image: the new key is added for it and counted (refs). Its layers and tiles follow the page table of the
// texture looked up (s2tile_refresh takes t). Switch 2 boots only; a different address or layout changes nothing. Under g_img_mu
static vkimg_t *sparse_share(const akey *k, const tex_t *t) {
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        vkimg_t *o = g_by_key.vals[i]; if (!o || !(o->s2t) || o->gaddr != (uint64_t)(uintptr_t)t->mem) continue;
        const nvn_texdesc *a = &o->kd, *b = &t->d;
        if (a->target != b->target || a->format != b->format || a->samples != b->samples || a->width != b->width || a->height != b->height || a->depth != b->depth ||
            a->levels != b->levels || a->stride != b->stride || a->pkg_block_h != b->pkg_block_h) continue;
        o->refs++; nvn_hm_put(&g_by_key, k, o);
        static int n; if (n++ < 64) VKLOG("NVN2 sparse: texture object %llu reuses the image %ux%ux%u of the same storage %p (pool %p +%#llx; the image's first key: pool %#llx +%#llx; %d keys now)",
            (unsigned long long)t->serial, o->w, o->h, o->layers, (const void *)t->mem, (const void *)t->pool, (unsigned long long)t->off, (unsigned long long)o->pool,
            (unsigned long long)o->off, o->refs + 1);
        return o;
    }
    return NULL;
}
static vkimg_t *twin_lookup(const akey *k, const vkfmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_SRGB_TWIN"); on = !e || atoi(e) != 0; }
    static signed char nt[0x9D]; static uint8_t tw[0x9D][4];   // per NVN format: 0 not looked at, -1 none, else the NVN formats of its twin
    int nf = k->d.format; if (!on || f->compressed || nf <= 0 || nf >= 0x9D) return NULL;
    if (!nt[nf]) { VkFormat v = srgb_twin(f->vk); int n = 0;
        for (int c = 1; v && c < 0x9D && n < 4; c++) { const vkfmt_t *o = vk_fmt(c); if (o && o->vk == v) tw[nf][n++] = (uint8_t)c; }
        nt[nf] = n ? (signed char)n : -1; }
    for (int j = 0; j < nt[nf]; j++) {
        akey k2; memcpy(&k2, k, sizeof k2); k2.d.format = tw[nf][j];   // (memcpy keeps the zeroed padding the table hashes)
        vkimg_t *im = nvn_hm_get(&g_by_key, &k2);
        if (!im || im->samples > 1 || !(im->cflags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) continue;
        uint64_t c = ++g_twin_hits; if (c <= 4 || !(c & (c - 1)))
            VKLOG("r75: texture format %#x over the storage of a format %#x image (%ux%u): viewed as its sRGB / UNORM twin, contents shared (%llu so far)", nf, tw[nf][j], im->w, im->h, (unsigned long long)c);
        return im;
    }
    return NULL;
}
static nvn_hmap g_up_seen = {.klen = sizeof(akey)};   // bak86: storage keys an image was uploaded for (under g_img_mu)
vkimg_t *vk_tex(const tex_t *t) {
    if (!t || !t->serial || t->magic != MAGIC('T', 'E', 'X', 0)) return NULL;
    const vkfmt_t *f = vk_fmt(t->d.format);
    if (!f) { VKRED_ONCE(14, "texture format %#x has no Vulkan counterpart here (first case %ux%u target %d)", t->d.format, t->d.width, t->d.height, t->d.target); return NULL; }
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER) return NULL;   // buffer textures: texel buffer views of the pool (vk_tex_view)
    akey k; memset(&k, 0, sizeof k); k.pool = (uint64_t)(uintptr_t)t->pool; k.off = t->off; k.d = t->d;
    // flags that do not change the storage's layout are not part of the key: on the Switch two textures over the same memory that differ only in them see the
    // same texels. Xenoblade Chronicles 3 renders its shadow maps and depth-stencil buffer through textures with COMPRESSIBLE and samples them through others with
    // IMAGE added; as separate images those were never written, the shadows were total and the stencil's material tags 0, and its lit scene black. r72: IMAGE left
    // the key for colour formats too (it was kept for the storage usage it gives). Hyrule Warriors: Age of Imprisonment registers its material textures as images at the
    // same ids (r70 made RegisterImage real): their lookups with IMAGE made new, empty images over the copied-in textures, and its play scene's G-buffer was black. A
    // texture made without storage usage gets it when a lookup with IMAGE comes (img_upgrade), its contents kept
    k.d.flags &= NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT;
    int keyed = t->pool != NULL;
    nvnr_lock(&g_img_mu);
    vkimg_t *im = keyed ? nvn_hm_get(&g_by_key, &k) : nvn_hm_get(&g_by_serial, &t->serial);
    if (im && im->keep_gw == 1) { im->keep_gw = 2; uint64_t c = ++g_keep_gw_reuse;   // bak91: a kept GPU-only image looked up again: its contents are still there
        if (c <= 256 || !(c & (c - 1))) VKLOG("bak91: reuse %llu: texture object %llu found the kept GPU-only image %ux%ux%u nvn format %#x (pool %#llx +%#llx, last looked up %llu presents ago)",
            (unsigned long long)c, (unsigned long long)t->serial, im->w, im->h, im->layers, im->nvn_format, (unsigned long long)im->pool, (unsigned long long)im->off, (unsigned long long)(g_tex_frame - im->used)); }
    if (!im && keyed) im = twin_lookup(&k, f);   // r75: the sRGB / UNORM twin over the same storage
    // S1e (parallel recording): a worker makes no image (nor shares a sparse one, nor restores an archived one, nor upgrades one for storage): its first upload, the
    // evictions it makes and the residency history are the replay thread's, in stream order; the segment is recorded again there (gfx_vk_exec.c seg_par_join)
    if (g_vk_tex_wk && (!im || im->archive || (!im->s2t && (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT)))) {
        g_vk_tex_refused = !im ? 1 : im->archive ? 2 : 3; nvnr_unlock(&g_img_mu); return NULL; }
    if (!im && keyed && dm_sdk_ounce() && t->pool && t->pool->vp && t->pool->vmode == 2 && t->mem && t->L.size >= (32ull << 20)) im = sparse_share(&k, t);   // NVN2 (11-16)
    int fresh = 0;
    if (!im) {
        extern uint64_t g_rd_ns[64], g_rd_n[64]; uint64_t now_ns(void); const uint64_t rn0 = g_rdiag > 0 ? now_ns() : 0;   // replay2 replaydiag: slot 61 evict_overlaps, 60 img_new
        if (keyed) evict_overlaps(k.pool, t->off, (int64_t)t->L.size);
        const uint64_t rn1 = rn0 ? now_ns() : 0;
        im = img_new(t, f); if (!im) { nvnr_unlock(&g_img_mu); return NULL; }
        if (rn0) { const uint64_t rn2 = now_ns(); g_rd_ns[61] += rn1 - rn0; g_rd_ns[60] += rn2 - rn1; g_rd_n[60]++; }
    rdi_new(im, t->d.flags);   // slot12 replaydiag
        im->pool = k.pool; im->off = t->off; im->size = (int64_t)t->L.size; im->made = g_tex_frame; im->gaddr = (uint64_t)(uintptr_t)t->mem; im->flags = t->d.flags; im->kd = t->d; g_img_n++; g_img_bytes += t->L.size;
#if defined(DM_NXN1X)
        im->content_id = nvnc_image_add((uint64_t)(uintptr_t)t->pool, im->gaddr, t->L.size, residency_key(&t->d, sizeof t->d));
#endif
        if (dm_sdk_ounce()) {
            im->residency_key = keyed ? residency_key(&k, sizeof k) : residency_key(&t->serial, sizeof t->serial);
            size_t at = im->residency_key % RESIDENCY_HISTORY;
            if (g_residency_history[at].key == im->residency_key) {
                im->residency_idle = g_residency_history[at].idle;
                residency_reused(im, g_residency_history[at].used);
                g_residency_history[at].key = 0;
            }
        }
        if (keyed) { alias_link(im); nvn_hm_put(&g_by_key, &k, im); } else nvn_hm_put(&g_by_serial, &t->serial, im);
        fresh = 1;
        if (keyed && dm_sdk_ounce() && im->size >= (int64_t)(32ll << 20)) {   // NVN2: a big image over memory other images keep (made after evict_overlaps kept them: recent)
            static int nk; for (uint32_t i = 0; i < g_by_key.cap && nk < 128; i++) {
                vkimg_t *o = g_by_key.vals[i]; if (!o || o == im || o->pool != k.pool || o->off >= im->off + im->size || im->off >= o->off + o->size) continue;
                akey ok_; memcpy(&ok_, g_by_key.keys + (size_t)i * g_by_key.klen, sizeof ok_); const nvn_texdesc *a = &ok_.d, *b = &k.d; char df[200]; int p = 0; df[0] = 0;
                #define KD(fl) if (a->fl != b->fl && p < (int)sizeof df - 40) p += snprintf(df + p, sizeof df - (size_t)p, " " #fl " %#x->%#x", (unsigned)a->fl, (unsigned)b->fl);
                KD(target) KD(format) KD(flags) KD(samples) KD(width) KD(height) KD(depth) KD(levels) KD(stride) KD(pkg_block_h)
                #undef KD
                if (ok_.off != k.off && p < (int)sizeof df - 40) p += snprintf(df + p, sizeof df - (size_t)p, " off %+lld", (long long)(k.off - ok_.off));
                nk++; VKLOG("NVN2 big image %ux%ux%u (%.1f MiB) made over the storage of a kept image (%ux%ux%u, %.1f MiB%s, looked up %llu presents ago): key differs in%s",
                    im->w, im->h, im->layers, im->size / 1048576.0, o->w, o->h, o->layers, o->size / 1048576.0, o->s2t ? ", sparse tiles" : "", (unsigned long long)(g_tex_frame - o->used), p ? df : " nothing (the same key)"); }
        }
    }
    // Pin before any allocation: upgrading/restoring may trim other images. Every
    // use through vk_tex (attachments, copies and clears too) needs a fence stamp.
    im->used = g_tex_frame;
    if (im->released_hint && g_rdiag > 0) g_rdi_back++;   // slot12 replaydiag
    im->released_hint=0; // taking the storage back cancels the cache-release hint
    if (vk_sub_open()) im->last_use = mtl_last_fence() + 1;
    else im->external_use = 1;   // the compositor has a separate completion fence
    if (im->archive && !img_archive_restore(im)) { nvnr_unlock(&g_img_mu); return NULL; }
    if (!fresh && !im->s2t && (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT) && vk_sub_open()) img_upgrade(im, t, f);   // r72 (not a sparse layered image)
    nvnr_unlock(&g_img_mu);
    if (!vk_sub_open()) return fresh ? NULL : im;   // looked up outside a submission (present): nothing can be recorded, a new image has no contents yet
    // R6/R7/R8 (parallel recording), R8: the upload / first-use tail under the outer recording lock (the image's upload state: packaged, pkg_*, lin_*, cc_ser, gpu_w;
    // its uploads take the leaf vk_up_lock inside). The g_img_mu takes below (bak86) are NVN2 only, where parallel recording is off
    if (g_vk_tex_wk && im->s2t) { g_vk_tex_refused = 4; return NULL; }   // S1e (parallel recording): NVN2 sparse tiles bind and upload per lookup (replay thread only)
    vk_par_lock();
    if (im->s2t) { uint64_t q = now_ns(); vk_up_lock(); s2tile_refresh(im, t, f); vk_up_unlock();   /* uppool: its uploads record into the upload buffer, which workers record into under this lock */ g_s2t_ns += now_ns() - q; vk_par_unlock(); return im; }   // NVN2 2D sparse tiles: bound and uploaded tile by tile from the page table
    if (t->mem && lin_copy(im, t, f)) {}
    else if (t->mem && (packaged_tex(t) || cpu_first_use(im, t, f))) {   // r73: or written by the CPU before its first use
        im->pkg_used = g_tex_frame;
        if (!im->packaged) { im->packaged = 1; pkg_upload(im, t, f);
            if (keyed && dm_sdk_ounce()) { nvnr_lock(&g_img_mu); if (!nvn_hm_get(&g_up_seen, &k)) nvn_hm_put(&g_up_seen, &k, (void *)1); nvnr_unlock(&g_img_mu); } }   // bak86: this storage held an uploaded image
        else {
            uint64_t now = nvn_dirty_now();
            if (vp2_tex(t) ? now != im->pkg_seen && nvn_sparse_since(t->pool, (uint64_t)(uintptr_t)t->mem, t->L.size, im->pkg_sgen, im->pkg_stamp)   // NVN2: mapped or written since
                           : now != im->pkg_seen && nvn_dirty_since_span(im->pkg_span, t->mem, t->L.size, im->pkg_stamp)) pkg_upload(im, t, f);   // pages written since the upload
            else if (pkg_verify() && (g_tex_frame >= im->pkg_vframe + (uint64_t)pkg_verify() || (t->serial != im->pkg_tser && im->pkg_vframe != g_tex_frame)))
                s24_check(im, t, f);   // s24: a CPU write no fault saw; s27: at once when another texture object is over its storage (at most once a present)
            im->pkg_seen = now;
        }
    } else if (fresh && !init_fill(im, t, f)) {   // r75: the storage's data, else zeros (was: undefined)
        if (keyed && dm_sdk_ounce()) { nvnr_lock(&g_img_mu); const int seen = nvn_hm_get(&g_up_seen, &k) != NULL; nvnr_unlock(&g_img_mu);
            if (seen) { g_zf_again++; static uint64_t n; if (n++ < 32 || !(n & (n - 1))) VKLOG("[RED] NVN2: texture %ux%u nvn format %#x (flags %#x) made again over storage %p +%#llx that held an uploaded image: zero-filled (%llu so far)",
                t->d.width, t->d.height, t->d.format, t->d.flags, (const void *)t->mem, (unsigned long long)t->L.size, (unsigned long long)n); } }
        zero_fill(im, f); }
    vk_par_unlock();   // R6/R7/R8 (parallel recording), R8
    return im;
}
vkimg_t *vk_tex_peek(const tex_t *t) {   // NVN2 diagnostic (gfx_vk_exec.c NDSR slot lines): the image vk_tex would find by storage key, without making, uploading or touching it
    if (!t || !t->serial || t->magic != MAGIC('T', 'E', 'X', 0) || !t->pool) return NULL;
    akey k; memset(&k, 0, sizeof k); k.pool = (uint64_t)(uintptr_t)t->pool; k.off = t->off; k.d = t->d;
    k.d.flags &= NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT;   // as vk_tex
    nvnr_lock(&g_img_mu); vkimg_t *im = nvn_hm_get(&g_by_key, &k); nvnr_unlock(&g_img_mu);
    return im;
}
// Packaged textures give up their write protection (nvn_dirty_release; drawn again, they are uploaded and protected again) when not drawn for PKG_IDLE presents, and the
// least recently drawn ones beyond that while the protected runs, each splitting its pool's mapping, are more than PKG_RUNS (textures stay protected until the game writes them).
// Textures drawn in the last PKG_KEEP presents keep theirs (released, they would be uploaded again at once). Swept every 300 presents, with the pools whose mapping the
// protection has split most mapped again in one piece (nvn_dirty_defrag: Xenoblade Chronicles 3 went from 43,000 mappings after 25 minutes of walking, toward the kernel's
// limit of 65,530, to about 17,500)
enum { PKG_IDLE = 1800, PKG_KEEP = 60, PKG_RUNS = 8000 };   // about 60 s and 2 s at 30 fps; at most about 16,000 mappings for the protection
uint64_t g_tex_frame;
typedef struct { uint64_t used; vkimg_t *im; } pkg_age;
static int pkg_age_cmp(const void *a, const void *b) { uint64_t x = ((const pkg_age *)a)->used, y = ((const pkg_age *)b)->used; return x < y ? -1 : x > y; }
// resoverlay (11-21, NPARK: "the current resolution is unknown"): the largest render target the game drew into in the last presented frame (its size as the game
// made it: the internal resolution) and the host's render scale on the NVN path (2 = the exact-2x path draws double-size copies; any other Eden scale is not applied
// here). Read by the performance overlay (yuzu/render/performance_overlay.cpp) through nvnh_res_info; plain stores, a torn read only shows one stale number
static volatile uint32_t g_res_cur[2], g_res_last[3];
void vk_res_note_scene(uint32_t w, uint32_t h) { if (w && h && w != h && h > g_res_cur[1]) { g_res_cur[0] = w; g_res_cur[1] = h; } }   // resoverlay2: a pass with a depth attachment (not square: shadow maps are left out)
void nvnh_res_info(uint32_t out[3]) { out[0] = g_res_last[0]; out[1] = g_res_last[1]; out[2] = g_res_last[2]; }
// stripediag (NPARK 11-22: a texture that was right turns into stripes during play, for long now): the hotkey "Dump Textures (stripe diagnostic)" (F3,
// main_window.cpp) asks for it; at the next present each image looked up in the last 3 presents is logged: its VkImage, size, levels, NVN and Vulkan format, NVN
// flags, storage (guest address, pool and offset, layout size), how it is filled (packaged = uploaded from guest memory, written by the GPU, kept for its GPU
// contents, gone on the Switch too), the present it was made and last rendered at, the images over the same memory (their size and format) and the texture
// object it was last uploaded for. Each packaged one is then uploaded again from its storage at its next use (the s24 path: its check made to differ), which logs
// "s24 … found changed and uploaded again": stripes that go with that were stale contents (a missed upload or invalidation); stripes that stay are in the
// storage itself or in how it is sampled (a view's format or swizzle, the descriptor), and this dump names the image to follow
static _Atomic int g_tmark;
void nvnh_tex_mark(void) { atomic_store(&g_tmark, 1); }
static void tmark_dump(void) {
    if (!atomic_exchange(&g_tmark, 0)) return;
    nvnr_lock(&g_img_mu);
    unsigned n = 0, pk = 0, gw = 0;
    VKLOG("stripediag: textures looked up in the last 3 presents (present %llu):", (unsigned long long)g_tex_frame);
    for (int pass = 0; pass < 2; pass++) {
        const nvn_hmap *h = pass ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            vkimg_t *im = h->vals[i]; if (!im || !im->img || im->used + 3 < g_tex_frame) continue;
            n++; pk += im->packaged != 0; gw += im->gpu_w != 0;
            if (n > 600) continue;
            char ov[256]; int on = 0; ov[0] = 0;
            for (int k = 0; k < im->novl && k < 8 && on < (int)sizeof ov - 40; k++) { const vkimg_t *o = im->ovl[k]; if (o) on += snprintf(ov + on, sizeof ov - (size_t)on, " %ux%u/%#x", o->w, o->h, o->nvn_format); }
            VKLOG("stripediag: #%u VkImage %p %ux%ux%u, %u levels, %u samples, nvn format %#x (vk %d), flags %#x; storage %#llx (pool %#llx +%#llx, %lld bytes); %s%s%s%s; made at present %llu, rendered at %lld, used %llu; over the same memory:%s; texture object %llu",
                  n, (void *)im->img, im->w, im->h, im->layers > 1 ? im->layers : im->d, im->levels, im->samples, (unsigned)im->nvn_format, (int)im->fmt, (unsigned)im->flags,
                  (unsigned long long)im->gaddr, (unsigned long long)im->pool, (unsigned long long)im->off, (long long)im->size,
                  im->packaged ? "packaged (from guest memory)" : "not packaged", im->gpu_w ? ", written by the GPU" : "", im->keep_gw ? ", kept for its GPU contents" : "", im->ovw ? ", overwritten (gone on the Switch too)" : "",
                  (unsigned long long)im->made, im->wpres ? (long long)im->wpres - 1 : -1LL, (unsigned long long)im->used, on ? ov : " none", (unsigned long long)im->pkg_tser);
            if (im->packaged) { im->pkg_vhash = ~im->pkg_vhash; im->pkg_vframe = 0; }   // its next use uploads it again (s24_check finds it changed)
        }
    }
    VKLOG("stripediag: %u images (%u packaged: each uploaded again at its next use; %u written by the GPU)%s", n, pk, gw, n > 600 ? ", the first 600 listed" : "");
    nvnr_unlock(&g_img_mu);
}
void vk_tex_sweep(void) {
    tmark_dump();   // stripediag
#if defined(DM_NXN1X)
    { int nvn_eden_resolution_x2(void); g_res_last[2] = nvn_eden_resolution_x2() ? 2 : 1; }
#else
    g_res_last[2] = 1;
#endif
    g_res_last[0] = g_res_cur[0]; g_res_last[1] = g_res_cur[1]; g_res_cur[0] = g_res_cur[1] = 0;   // resoverlay: a present: the frame just shown
    if (dm_sdk_ounce()) { nvnr_lock(&g_img_mu); ovw_sweep(); keep_gw_report(); nvnr_unlock(&g_img_mu); }
    if (++g_tex_frame % 300) return;
    uint64_t idle = 0, lru = 0, before = nvn_dirty_released();
    nvnr_lock(&g_img_mu);
    pkg_age *v = NULL; size_t nv = 0, cap = 0;
    for (int m = 0; m < 2; m++) {
        nvn_hmap *h = m ? &g_by_serial : &g_by_key;
        for (uint32_t i = 0; i < h->cap; i++) {
            vkimg_t *im = h->vals[i];
            if (!im || !im->packaged || im->pkg_released || !im->pkg_mem) continue;
            if (g_tex_frame - im->pkg_used >= PKG_IDLE) { nvn_dirty_release(im->pkg_mem, im->pkg_len); im->pkg_released = 1; idle++; continue; }
            if (g_tex_frame - im->pkg_used < PKG_KEEP) continue;
            if (nv == cap) {
                if (cap > SIZE_MAX / 2 / sizeof *v) { free(v); nvnr_unlock(&g_img_mu); return; }
                const size_t next = cap ? cap * 2 : 4096;
                pkg_age *nw = realloc(v, next * sizeof *v);
                // Abort both table scans. Failed metadata growth cannot publish
                // a larger capacity or discard the existing image contents.
                if (!nw) { free(v); nvnr_unlock(&g_img_mu); return; }
                v = nw; cap = next;
            }
            v[nv++] = (pkg_age){im->pkg_used, im};
        }
    }
    { int vk_pool_remap_ro(void *mem, size_t len); int nvn_dirty_defrag(int (*)(void *, size_t)); struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); int d = nvn_dirty_defrag(vk_pool_remap_ro); clock_gettime(CLOCK_MONOTONIC, &t1); static uint64_t nd;
        if (d && (nd += d) <= 8) VKLOG("dirty-page tracking: %d pool(s) mapped again in one piece in %.1f ms (%llu so far)", d, (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, (unsigned long long)nd); }
    int runs = nvn_dirty_runs();
    if (runs > PKG_RUNS && nv) {   // the least recently drawn first, in batches, until the runs are well under the budget
        qsort(v, nv, sizeof *v, pkg_age_cmp);
        for (size_t i = 0; i < nv && runs > PKG_RUNS * 3 / 4; ) {
            for (size_t e = i + 256; i < nv && i < e; i++) { nvn_dirty_release(v[i].im->pkg_mem, v[i].im->pkg_len); v[i].im->pkg_released = 1; lru++; }
            runs = nvn_dirty_runs();
        }
    }
    nvnr_unlock(&g_img_mu);
    free(v);
    static uint64_t total; total += idle + lru;
    if (g_tex_frame % 3000 == 0) VKLOG("images: %llu live, %.0f MB of textures, %llu evicted so far (a newer texture over their memory)", (unsigned long long)g_img_n, g_img_bytes / 1048576.0, (unsigned long long)g_img_evicted);
    if ((idle || lru) && (g_tex_frame % 3000 == 0 || lru)) VKLOG("packaged textures: %llu not drawn for %d presents and %llu least recently drawn released their write protection (%llu pages; %llu so far); %d protected runs",
                    (unsigned long long)idle, PKG_IDLE, (unsigned long long)lru, (unsigned long long)(nvn_dirty_released() - before), (unsigned long long)total, runs);
}
void vk_tex_drop_pool(const void *pool) {   // the pool is finalized: its textures' images go once the GPU is done with them
    nvnr_lock(&g_img_mu);
    for (uint32_t i = 0; i < g_by_key.cap; i++) {
        if (!g_by_key.vals[i]) continue;
        akey k; memcpy(&k, g_by_key.keys + (size_t)i * g_by_key.klen, sizeof k);
        if (k.pool != (uint64_t)(uintptr_t)pool) continue;
        vkimg_t *im = g_by_key.vals[i]; nvn_hm_del(&g_by_key, &k); i = (uint32_t)-1;   // the table was rehashed by the delete: rescan
        img_drop_why(im, "its pool finalized");
    }
    nvnr_unlock(&g_img_mu);
}
// CPU-written textures (WriteTexels / a dirty first use): packaged from now on
static nvn_hmap g_cpu_written = {.klen = sizeof(uint64_t)}; static pthread_mutex_t g_cw_mu = PTHREAD_MUTEX_INITIALIZER;
void mtl_tex_cpu_written(const tex_t *t) { if (!t->serial) return; pthread_mutex_lock(&g_cw_mu); nvn_hm_put(&g_cpu_written, &t->serial, (void *)1); pthread_mutex_unlock(&g_cw_mu); }
// Linear textures (LINEAR) in a pool are their storage on the Switch: the CPU writes the pool memory and the GPU reads it (Metal's tex_linear makes them on the pool buffer).
// Hades II rasterizes its glyphs on the CPU into a 1024x1216 R8 IMAGE|LINEAR atlas (pool CPU_UNCACHED|GPU_CACHED, no flush, no WriteTexels) and decodes its Bink movies
// into linear planes: as GPU-only images of zeros all text, the title art and the movies were blank. A Vulkan image cannot be bound to the pool's memory (a dedicated
// AHardwareBuffer import, bound to its buffer), so each upload command buffer that a submission's draws need it in copies it from the pool buffer on the GPU (lin_copy).
// The packaged path (CPU upload, re-upload on written pages) is the fallback; as the only path the page faults of the CPU's writes cost Hades II about 120 a frame and
// 2 million while loading. Not LINEAR_RENDER_TARGET: the GPU draws into those, and a copy from storage would replace what it drew
static int linear_tex(const tex_t *t) { return t->pool && (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) == NVN_TEXTURE_FLAGS_LINEAR; }
// lin_copy takes LINEAR_RENDER_TARGET textures too: Super Mario Bros. Wonder's CPU writes its sky's four 64x1 RGBA16F gradients straight into such textures (NVN flags 0x40)
// that nothing draws into, and as GPU-only images of zeros the sky drew diagonal bands of saturated colour over every level. What the GPU draws into one (gpu_w) stays, as for a
// cleared or copied linear texture, until the CPU writes its storage again. The packaged path (linear_tex) still leaves them out: a re-upload there would replace a drawing
static int linear_storage(const tex_t *t) { return t->pool && (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)); }
extern uint64_t g_up_gen;
static int lin_copy(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {   // 1 = the image holds its storage as of this upload buffer
    if (!linear_storage(t) || im->levels != 1 || im->d != 1 || im->layers != 1 || f->bw > 1 || f->bh > 1 || !f->bpb || im->aspect != VK_IMAGE_ASPECT_COLOR_BIT) return 0;
    size_t stride = t->d.stride ? t->d.stride : (size_t)im->w * f->bpb; VkBuffer b; VkDeviceSize o;
    if (stride % f->bpb || stride < (size_t)im->w * f->bpb || !vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)stride * (im->h - 1) + (uint64_t)im->w * f->bpb, &b, &o) || o % f->bpb) return 0;
    // the GPU wrote it (a clear, a copy, a storage write): it holds that until the CPU writes its storage again, which its pages, set read-only, tell
    size_t len = stride * (im->h - 1) + (size_t)im->w * f->bpb;
    if (im->gpu_w && !im->lin_mem) { nvn_dirty_protect(t->mem, len); im->lin_stamp = nvn_dirty_now(); im->lin_span = nvn_dirty_span(t->mem, len); im->lin_mem = t->mem; im->lin_len = len; return 1; }
    if (im->gpu_w) { if (!nvn_dirty_since_span(im->lin_span, t->mem, len, im->lin_stamp)) return 1; nvn_dirty_release(im->lin_mem, im->lin_len); im->lin_mem = NULL; im->gpu_w = 0; im->lin_gen = 0; }
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and its g_up_gen memo (leaf: vk_buf_at above, outside it)
    VkCommandBuffer cb = vk_upload_cmd();
    if (im->lin_gen == g_up_gen) { vk_up_unlock(); return 1; }
    im->lin_gen = g_up_gen;
    vk_img_barrier_all(cb);   // after the reads of earlier submissions
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) vk_tex_prepare(cb, im);
    VkBufferImageCopy r = {o, (uint32_t)(stride / f->bpb), 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vk_tex_content_changed(im);
    vkCmdCopyBufferToImage(cb, b, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    vk_up_unlock();
    return 1;
}
// r73: uncompressed textures cpu_first_use took in a pool the GPU had not written (serial -> pool), under g_cw_mu. At the pool's first GPU write they leave the
// CPU-written set (nvn_mtl.m cpu_unc_drop): its storage is no longer the whole truth, and a re-upload would replace what the GPU drew
static nvn_hmap g_cpu_unc = {.klen = sizeof(uint64_t)};
static int storage_gpu_written(const tex_t *t); static int gwr_cpu_after_on(void); static int storage_gpu_written_at(const tex_t *t, uint64_t *pres, uint64_t *lo, uint64_t *hi); static uint64_t g_gwr_taken, g_gwr_refused;   // gwrange (below, beside g_gpu_owned)
static int packaged_tex(const tex_t *t) {
    if (t->has_layout || linear_tex(t)) return 1;
    pthread_mutex_lock(&g_cw_mu); int r = nvn_hm_get(&g_cpu_written, &t->serial) != NULL;
    if (r && g_cpu_unc.n && t->pool && nvn_hm_get(&g_cpu_unc, &t->serial) && storage_gpu_written(t)) {   // gwrange: a GPU write into its own storage
        nvn_hm_del(&g_cpu_unc, &t->serial); nvn_hm_del(&g_cpu_written, &t->serial); r = 0; }
    pthread_mutex_unlock(&g_cw_mu);
    return r;
}
// s56 (gfx_vk_core.c vk_s56_mirror): such a texture is uploaded again from its storage when its pages are stamped written, and a write-back's own stamps would
// replace a later drawing with an earlier one: those are left out of the write-back (so no absorb step is needed for the ones it takes)
int vk_tex_s56_packaged(const tex_t *t) { return packaged_tex(t); }
// r73: storage the CPU wrote before a texture's first use (nvn_mtl.m tex_cpu_content, the same rules; every game). The Windows host had none of it: a sampled texture
// with no packaged layout took content only from GPU copies, a WriteTexels, or a game profile's tex_flush_upload / tex_dirty_upload. Hyrule Warriors: Age of Imprisonment
// writes its material textures (BC, a CPU_UNCACHED | GPU_CACHED pool, no flush) straight into the pool: guest memory held them (194 KB of 256 KB non-zero for one 256x512
// BC texture) while the image stayed zero, and the play scene's G-buffer was black. Block-compressed textures cannot be rendered into, so data in their storage at first use
// is what the CPU wrote; an uncompressed 2D one only in a pool the GPU has not written (dropped at its first GPU write, packaged_tex); an uncompressed 3D one in any pool.
// Looked at once per serial on an image (cc_ser), and not for an image the GPU wrote. DM_NVN_STORAGE_FIRST_USE=0 turns it off, as on the Mac
static int mem_any(const uint8_t *p, uint64_t n) {   // whether any byte is non-zero (stops at the first)
    uint64_t i = 0, v;
    for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) return 1; }
    for (; i < n; i++) if (p[i]) return 1;
    return 0;
}
static int cpu_first_use(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_STORAGE_FIRST_USE"); on = !e || atoi(e) != 0; }
    if (!on || im->cc_ser == t->serial) return 0;
    im->cc_ser = t->serial;
    if (im->gpu_w || !t->mem || !t->pool || t->has_layout || t->d.samples > 1 || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    int bc = f->compressed || f->bw > 1, unc = !bc && t->d.target != NVN_TEXTURE_TARGET_3D;
    if (unc && !f->depth && !f->stencil) { uint64_t gp = 0, glo = 0, ghi = 0;   // gwrange2: refused (the GPU wrote this storage after the CPU's last write): logged with the GPU write
        if (storage_gpu_written_at(t, &gp, &glo, &ghi)) {
            uint64_t c = ++g_gwr_refused; const int tile = t->d.width == 260 && t->d.height == 260;
            if (c <= 16 || !(c & (c - 1)) || (tile && c <= 256)) VKLOG("gwrange2: refused %ux%ux%u format %#x flags %#x pool flags %#x at %p (+%#llx, %llu bytes, %s storage): the GPU wrote %#llx..%#llx at present %llu (now %llu)%s; %llu so far",
                t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags, t->pool->flags, (const void *)t->mem, (unsigned long long)t->off, (unsigned long long)t->L.size,
                mem_any(t->mem, t->L.size) ? "non-zero" : "zero", (unsigned long long)glo, (unsigned long long)ghi, (unsigned long long)gp, (unsigned long long)g_tex_frame,
                gwr_cpu_after_on() ? ", the CPU has not written it since" : " (DM_VK_GPUW_CPU_AFTER=0)", (unsigned long long)c);
            return 0; } }
    if (unc && (f->depth || f->stencil || (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_VIDEO_DECODE)))) return 0;
    if (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;
    if ((t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE))
        || ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !bc)) {
        static int nx; if (bc && nx++ < 4) VKLOG("r73: block-compressed texture %ux%u format %#x flags %#x pool flags %#x left out of the first-use check by its flags", t->d.width, t->d.height, t->d.format, t->d.flags, t->pool->flags);
        return 0; }
    if (mtl_tex_gpu_owned(t->serial) || !mem_any(t->mem, t->L.size)) return 0;
    if (unc) { uint64_t sr = t->serial; pthread_mutex_lock(&g_cw_mu); nvn_hm_put(&g_cpu_unc, &sr, (void *)t->pool); pthread_mutex_unlock(&g_cw_mu); im->unc_pool = t->pool; }   // (bak86)
    mtl_tex_cpu_written(t);
    if (unc && atomic_load_explicit(&t->pool->gpu_written, memory_order_relaxed)) {   // gwrange: taken although its pool has GPU writes (the pool rule left it as zeros)
        uint64_t k = ++g_gwr_taken, nz = 0, m = t->L.size < (1u << 20) ? t->L.size : (1u << 20); for (uint64_t i = 0; i < m; i++) nz += t->mem[i] != 0;
        if (k <= 16 || !(k & (k - 1))) VKLOG("gwrange: texture %ux%ux%u format %#x flags %#x pool flags %#x taken from its storage in a pool the GPU writes (none of its own bytes written by the GPU): %llu of the first %llu bytes non-zero (%llu so far)",
            t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags, t->pool->flags, (unsigned long long)nz, (unsigned long long)m, (unsigned long long)k); }
    static uint64_t n[2]; if (n[unc]++ < 4 || !(n[unc] & (n[unc] - 1)))
        VKLOG("r73: texture written by the CPU before its first use %ux%ux%u format %#x flags %#x%s pool flags %#x: uploaded from its storage, re-uploaded when it changes (%llu so far)",
              t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags, unc ? " (uncompressed, in a pool the GPU has not written)" : "", t->pool->flags, (unsigned long long)n[unc]);
    return 1;
}
// r75: a new image's contents are undefined in Vulkan, while fresh memory on the Switch reads 0, and games rely on it (nvn_mtl.m tex_zero_fill: Super Mario Bros. Wonder's
// fountain wave texture is read, drawn into and copied back every frame and never cleared; Pokemon Brilliant Diamond's movie quad tests GEQUAL against a window depth
// buffer it never clears). So an image with no other contents is cleared to 0 in the upload command buffer (before the submission's main one). Block-compressed and
// multisampled images are left as they are, as on the Mac. DM_VK_ZERO_FILL=0 turns it off
static uint64_t g_zero_n, g_init_n[3];
static void zero_fill(vkimg_t *im, const vkfmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_ZERO_FILL"); on = !e || atoi(e) != 0; }
    vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer (leaf)
    VkCommandBuffer cb = vk_upload_cmd(); vk_tex_prepare(cb, im);
    if (!on || im->samples > 1 || f->compressed || f->bw > 1 || f->bh > 1) { vk_up_unlock(); return; }
    vk_tex_content_changed(im);
    VkImageSubresourceRange rg = {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (im->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue z = {{0}}; vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg); }
    else { VkClearDepthStencilValue z = {0.0f, 0}; vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg); }
    vk_up_unlock();
    if (!g_zero_n++) VKLOG("r75: new textures start as zeros, as fresh memory on the Switch (first %ux%u VkFormat %d; DM_VK_ZERO_FILL=0 off)", im->w, im->h, im->fmt);
}
// r75: a new texture whose storage already holds data takes it as its contents (nvn_mtl.m tex_init_fill), beyond the first-use rule (cpu_first_use): in a CPU_NO_ACCESS
// pool, whose memory the game filled before making the pool (Rhythm Heaven Groove's sprite masks read zeros on the Mac without it; not with COMPRESSIBLE, whose memory may
// mean nothing to the CPU), in a CPU_UNCACHED pool, where the CPU's writes reach memory with no flush (Donkey Kong Country: Tropical Freeze's logo and menu panels), and in a
// pool the CPU flushed over the storage (nvn_flushed: Breath of the Wild's sky tables). Uploaded once, as a packaged texture is (not re-uploaded unless the texture is
// also marked CPU-written). Not for depth or stencil, multisampled, packaged-layout or virtual-pool textures, nor after the GPU wrote the image. DM_VK_INIT_FILL=0 turns it off
int nvn_flushed(const void *p, uint64_t n);   // hle_nvn.c
static int init_fill(vkimg_t *im, const tex_t *t, const vkfmt_t *f) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_INIT_FILL"); on = !e || atoi(e) != 0; }
    if (!on || im->gpu_w || !t->mem || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || t->has_layout || t->d.samples > 1 || f->depth || f->stencil || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    int no_cpu = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) != 0, uncached = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) != 0;
    if (no_cpu && (t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE)) return 0;
    if (!no_cpu && !uncached && !nvn_flushed(t->mem, t->L.size)) return 0;
    if (!mem_any(t->mem, t->L.size)) return 0;
    pkg_upload(im, t, f);
    int why = no_cpu ? 0 : uncached ? 1 : 2; uint64_t c = ++g_init_n[why];
    if (c <= 4 || !(c & (c - 1))) VKLOG("r75: new texture %ux%ux%u format %#x flags %#x filled from its storage, %s (%llu so far; DM_VK_INIT_FILL=0 off)", t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags,
        (const char *[]){"a CPU_NO_ACCESS pool's", "a CPU_UNCACHED pool's", "flushed by the CPU before its first use"}[why], (unsigned long long)c);
    return 1;
}
void mtl_tex_init(const tex_t *t) { (void)t; }
void mtl_tex_release(tex_t *t) {
    if (!dm_sdk_ounce() || !t || !t->serial) return;
    akey k;memset(&k,0,sizeof k);k.pool=(uint64_t)(uintptr_t)t->pool;k.off=t->off;k.d=t->d;
    k.d.flags &= NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT;
    nvnr_lock(&g_img_mu);
    vkimg_t *im=t->pool?nvn_hm_get(&g_by_key,&k):nvn_hm_get(&g_by_serial,&t->serial);
    if (im && !im->released_hint && g_rdiag > 0) g_rdi_rel++;   // slot12 replaydiag
    if (im) im->released_hint=1; // no destruction, archive or wait on finalization
    nvnr_unlock(&g_img_mu);
}
// r72 (Switch 1 thread's audit): mtl_tex_gpu_owned returned 0, so with a game's tex_dirty_upload profile on, a render target whose pages the CPU also wrote was marked
// CPU-written (hle_nvn.c nvn_ft_poll) and its GPU content could be uploaded over from guest memory. The serials of textures the GPU wrote are kept here (as nvn_mtl.m, which
// reads its texture's render-target / shader-write usage). Marked at the attachment, storage, copy and clear sites; a serial is never reused, so entries only accumulate
static nvn_hmap g_gpu_owned = {.klen = sizeof(uint64_t)}; static pthread_mutex_t g_go_mu = PTHREAD_MUTEX_INITIALIZER;
void vk_tex_gpu_mark(const tex_t *t) {
    if (!t || !t->serial) return;
    if (t->mem) vk_tex_watch_note("GPU write (attachment, storage, copy or clear)", t->serial, (uint64_t)(uintptr_t)t->mem, (uint64_t)(uintptr_t)t->mem + t->L.size, t->d.width, t->d.height, t->d.format);   // gpuowndiag
    void mtl_tex_gpu_range(const tex_t *t); mtl_tex_gpu_range(t);   // gwrange
    // gpuown: the value is the dirty-page generation of the latest GPU write + 1 (mtl_tex_gpu_stamp)
    const uint64_t st = nvn_dirty_now() + 1;
    pthread_mutex_lock(&g_go_mu); void *was = nvn_hm_get(&g_gpu_owned, &t->serial); if ((uint64_t)(uintptr_t)was != st) nvn_hm_put(&g_gpu_owned, &t->serial, (void *)(uintptr_t)st); pthread_mutex_unlock(&g_go_mu);
}
int mtl_tex_gpu_owned(uint64_t serial) { pthread_mutex_lock(&g_go_mu); int r = nvn_hm_get(&g_gpu_owned, &serial) != NULL; pthread_mutex_unlock(&g_go_mu); return r; }
// gpuown (11-21, NPARK, nxnative #1558 / #1566 OoT terrain holes): 0 = the GPU never wrote this texture, else the dirty-page generation (nvn_dirty_now) of its latest GPU write + 1.
// A registered texture the CPU rewrote (hle_nvn.c nvn_ft_poll) is the GPU's only when the GPU wrote it after the CPU's last write, not because the GPU wrote it once
uint64_t mtl_tex_gpu_stamp(uint64_t serial) { pthread_mutex_lock(&g_go_mu); uint64_t r = (uint64_t)(uintptr_t)nvn_hm_get(&g_gpu_owned, &serial); pthread_mutex_unlock(&g_go_mu); return r; }
// gwrange (11-21, NPARK: OoT's Kokiri ground between the grass mirrors the sky, "water" that is not there; nxnative #1566, the same blue holes on the Mac): the uncompressed
// first-use rule (cpu_first_use) and the CPU-written set (packaged_tex) asked whether the texture's POOL had any GPU write. OoT keeps its 260x260 terrain height / blend tiles
// in a CPU_CACHED | GPU_CACHED pool (0x24) that also holds render targets, so tiles made after the pool's first render target were never taken from their storage (the log had
// r73 lines only for the block-compressed 260x260 tiles) and tiles taken earlier left the set at that write, their later CPU rewrites dropped: zeros, a flat mirror.
// What makes storage the GPU's is a GPU write into THAT storage: the byte ranges of textures recorded as attachments, clears and copy destinations (hle_nvn.c, beside
// mtl_pool_gpu_written) and of storage images (vk_tex_gpu_mark), merged in a sorted list. A texture overlapping none of them is CPU content in any pool.
// DM_VK_GPUW_RANGE=0 turns it off (the pool rule)
// gwrange2 (11-21, NPARK: OoT's Deku Tree 2F floor came out in blocks of holes, the same look as the Kokiri "water"): a GPU-written range was kept for good and merged
// with every range it touched, so a terrain tile the game later put where a render target had been was refused for good and stayed zero. Each GPU-written texture's
// range is now kept as it is (not merged), with the dirty-page generation (st) and present (pres) of its latest GPU write. A texture's storage is the GPU's when such
// a range overlaps it and, with DM_VK_GPUW_CPU_AFTER on (default; =0 off: any overlap, as gwrange), the CPU has not written its pages since that write
// (nvn_dirty_since): the later writer owns it, as gpuown's rule. Pages the dirty tracker does not watch keep the GPU write. Ranges sorted by lo; g_gwr_maxlen bounds the scan
typedef struct { uint64_t lo, hi, st, pres; } gwr_t;
static gwr_t *g_gwr; static size_t g_ngwr, g_cgwr; static uint64_t g_gwr_maxlen; static pthread_mutex_t g_gwr_mu = PTHREAD_MUTEX_INITIALIZER;
int vk_gwr_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_GPUW_RANGE"); on = !e || atoi(e) != 0; } return on; }
static int gwr_cpu_after_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_GPUW_CPU_AFTER"); on = !e || atoi(e) != 0;
    VKLOG("gwrange2: a GPU-written range is the CPU's again once the CPU writes its pages after the GPU %s (DM_VK_GPUW_CPU_AFTER=%s)", on ? "(on)" : "is not considered (off)", e ? e : "unset"); } return on; }
static size_t gwr_lower(uint64_t lo) { size_t a = 0, b = g_ngwr; while (a < b) { size_t m = (a + b) / 2; if (g_gwr[m].lo < lo) a = m + 1; else b = m; } return a; }   // first range with .lo >= lo
void vk_gwr_add(const void *mem, uint64_t n) {
    if (!mem || !n || !vk_gwr_on()) return;
    const uint64_t a = (uint64_t)(uintptr_t)mem, b = a + n, st = nvn_dirty_now(), pr = g_tex_frame;
    pthread_mutex_lock(&g_gwr_mu);
    size_t i = gwr_lower(a);
    for (size_t j = i; j < g_ngwr && g_gwr[j].lo == a; j++) if (g_gwr[j].hi == b) { g_gwr[j].st = st; g_gwr[j].pres = pr; pthread_mutex_unlock(&g_gwr_mu); return; }   // the same target again
    if (g_ngwr == g_cgwr) { void *g = realloc(g_gwr, (g_cgwr = g_cgwr ? g_cgwr * 2 : 256) * sizeof *g_gwr); if (!g) { pthread_mutex_unlock(&g_gwr_mu); VKLOG("[RED] gwrange: out of memory"); abort(); } g_gwr = g; }
    memmove(g_gwr + i + 1, g_gwr + i, (g_ngwr - i) * sizeof *g_gwr); g_ngwr++;
    g_gwr[i] = (gwr_t){a, b, st, pr}; if (n > g_gwr_maxlen) g_gwr_maxlen = n;
    pthread_mutex_unlock(&g_gwr_mu);
}
// the latest GPU write over [mem, mem+n): its generation + 1 (0 = none), its present and the range it came from
static uint64_t gwr_latest(const void *mem, uint64_t n, uint64_t *pres, uint64_t *lo, uint64_t *hi) {
    const uint64_t a = (uint64_t)(uintptr_t)mem, e = a + n; uint64_t r = 0;
    pthread_mutex_lock(&g_gwr_mu);
    for (size_t j = gwr_lower(a > g_gwr_maxlen ? a - g_gwr_maxlen : 0); j < g_ngwr && g_gwr[j].lo < e; j++)
        if (g_gwr[j].hi > a && g_gwr[j].st + 1 > r) { r = g_gwr[j].st + 1; if (pres) *pres = g_gwr[j].pres; if (lo) *lo = g_gwr[j].lo; if (hi) *hi = g_gwr[j].hi; }
    pthread_mutex_unlock(&g_gwr_mu);
    return r;
}
void mtl_tex_gpu_range(const tex_t *t) { if (t && t->mem) vk_gwr_add(t->mem, t->L.size); }
// the GPU wrote this texture's storage (after the CPU's last write to it, gwrange2), or its pool (DM_VK_GPUW_RANGE=0)
static int storage_gpu_written_at(const tex_t *t, uint64_t *pres, uint64_t *lo, uint64_t *hi) {
    if (!vk_gwr_on()) return atomic_load_explicit(&t->pool->gpu_written, memory_order_relaxed);
    const uint64_t s = gwr_latest(t->mem, t->L.size, pres, lo, hi);
    if (!s) return 0;
    if (!gwr_cpu_after_on() || !nvn_dirty_span(t->mem, t->L.size)) return 1;   // (pages not tracked: no CPU write can be seen, the GPU write stands)
    return !nvn_dirty_since(t->mem, t->L.size, s - 1);
}
static int storage_gpu_written(const tex_t *t) { return storage_gpu_written_at(t, NULL, NULL, NULL); }

// ---------------- views ----------------
typedef struct vview { VkImageView v; VkImageViewType type; VkFormat fmt; uint32_t lv0, nlv, ly0, nly; VkComponentMapping swz; VkImageAspectFlags aspect; int attach; struct vview *next; } vview;   // attach: an attachment view (its usage differs from a sampling view's)
static nvn_hmap g_views = {.klen = sizeof(vkimg_t *)};   // vkimg_t* → vview list
static pthread_mutex_t g_view_mu = PTHREAD_MUTEX_INITIALIZER;
extern uint64_t g_tex_gen;
static void vk_tex_view_drop(vkimg_t *im, int completed) {
    pthread_mutex_lock(&g_view_mu);
    vview *v = nvn_hm_get(&g_views, &im); g_tex_gen++;
    while (v) {
        vview *n = v->next;
        if (completed) { extern uint64_t g_views_freed; vkDestroyImageView(V.dev, v->v, NULL); g_views_freed++; }
        else vk_defer_free_view(v->v);
        free(v); v = n;
    }
    nvn_hm_del(&g_views, &im);
    pthread_mutex_unlock(&g_view_mu);
}
static void vk_tex_view_drop_all(vkimg_t *im) { vk_tex_view_drop(im, 0); }
static void vk_tex_view_drop_completed(vkimg_t *im) { vk_tex_view_drop(im, 1); }
static VkComponentSwizzle swz_comp(int nvn) {   // NVN swizzle enum: 0 zero 1 one 2 R 3 G 4 B 5 A
    switch (nvn) { case 0: return VK_COMPONENT_SWIZZLE_ZERO; case 1: return VK_COMPONENT_SWIZZLE_ONE; case 2: return VK_COMPONENT_SWIZZLE_R; case 3: return VK_COMPONENT_SWIZZLE_G; case 4: return VK_COMPONENT_SWIZZLE_B; default: return VK_COMPONENT_SWIZZLE_A; }
}
static VkComponentSwizzle apply(VkComponentSwizzle outer, const VkComponentMapping *inner, VkComponentSwizzle self) {   // outer picks a channel of what inner produces
    if (outer == VK_COMPONENT_SWIZZLE_IDENTITY) outer = self;
    switch (outer) {
    case VK_COMPONENT_SWIZZLE_R: return inner->r == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R : inner->r;
    case VK_COMPONENT_SWIZZLE_G: return inner->g == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_G : inner->g;
    case VK_COMPONENT_SWIZZLE_B: return inner->b == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_B : inner->b;
    case VK_COMPONENT_SWIZZLE_A: return inner->a == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_A : inner->a;
    default: return outer;
    }
}
static VkImageView image_view(vkimg_t *, const tex_t *, const view_t *, int, int, int);
int vk_rescale_eligible(vkimg_t *im) {
    // Overlapping guest allocations with different descriptions have independent Vulkan
    // images. Reusing a guest memory range does not change this image's blit support.
    if (!im || im->vtype != VK_IMAGE_VIEW_TYPE_2D || im->levels != 1 || im->layers != 1 ||
        im->samples != 1 || im->s2t ||
        (im->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET |
                      NVN_TEXTURE_FLAGS_SPARSE)) ||
        (im->aspect == VK_IMAGE_ASPECT_COLOR_BIT && im->h <= 288) ||
        im->w > V.props.limits.maxFramebufferWidth / 2 || im->h > V.props.limits.maxFramebufferHeight / 2)
        return 0;
    if (im->scale_caps) return im->scale_caps > 0;
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(V.pd, im->fmt, &props);
    const VkFormatFeatureFlags blit = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    im->scale_caps = (props.optimalTilingFeatures & blit) != blit ? -1 :
        (im->aspect == VK_IMAGE_ASPECT_COLOR_BIT && !vk_fmt(im->nvn_format)->is_int &&
         (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) ? 2 : 1;
    return im->scale_caps > 0;
}
VkImageView vk_rescaled_view(vkimg_t *im, const tex_t *t, const view_t *vw, int dim, int arrayed, int attachment) {
    if (!im->scaled) {
        if (!attachment) return VK_NULL_HANDLE;
        tex_t desc = *t;
        desc.d = im->kd;
        desc.d.flags |= im->flags & NVN_TEXTURE_FLAGS_IMAGE;
        desc.d.width = im->w * 2;
        desc.d.height = im->h * 2;
        desc.pool = NULL;
        nvnr_lock(&g_img_mu);
        im->scaled = img_new(&desc, vk_fmt(im->nvn_format));
        if (im->scaled) ++g_img_n;
        nvnr_unlock(&g_img_mu);
        if (!im->scaled) return VK_NULL_HANDLE;
    }
    return image_view(im->scaled, t, vw, dim, arrayed, attachment);
}
void vk_rescale_blit(VkCommandBuffer cb, vkimg_t *im, int enlarge, VkImageAspectFlags aspects, VkRect2D native) {
    vkimg_t *src = enlarge ? im : im->scaled;
    vkimg_t *dst = enlarge ? im->scaled : im;
    vk_tex_prepare(cb, dst);
    int x = native.offset.x, y = native.offset.y;
    int right = x + native.extent.width, bottom = y + native.extent.height;
    int a = enlarge ? 1 : 2, b = enlarge ? 2 : 1;
    VkImageBlit r = {.srcOffsets = {{x*a,y*a,0}, {right*a,bottom*a,1}},
                     .dstOffsets = {{x*b,y*b,0}, {right*b,bottom*b,1}}};
    for (unsigned bit = 1; bit <= VK_IMAGE_ASPECT_STENCIL_BIT; bit <<= 1) {
        if (!(aspects & bit)) continue;
        r.srcSubresource = r.dstSubresource = (VkImageSubresourceLayers){bit,0,0,1};
        // Keep one rendered sample per guest texel when writing back intermediate targets.
        VkFilter filter = enlarge && im->scale_caps == 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        vkCmdBlitImage(cb, src->img, VK_IMAGE_LAYOUT_GENERAL, dst->img, VK_IMAGE_LAYOUT_GENERAL,
                       1, &r, filter);
    }
}
__thread VkFormat vk_view_fmt; __thread uint32_t vk_view_layers;
#if defined(DM_NXN1X)
// Match the guest addresses of the last few decoder surfaces. A view match shows
// a binding; a present match identifies a direct present of that surface.
static struct video_surface {
    uint64_t address, length, pts, sequence;
    uint32_t id, reported;
} video_surfaces[8];
static pthread_mutex_t video_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int video_active;
static _Atomic uint64_t video_low = UINT64_MAX, video_high;
static uint64_t video_sequence;

void nvnh_video_written(uint64_t address, uint64_t length, uint64_t pts, uint32_t id) {
    pthread_mutex_lock(&video_mutex);
    uint64_t seq = ++video_sequence;
    video_surfaces[seq % 8] = (struct video_surface){address, length, pts, seq, id, 0};
    if (seq <= 4 || seq % 120 == 0) VKLOG("NVN movie write seq=%llu image=%u pts=%llu address=%#llx bytes=%llu",
        (unsigned long long)seq, id, (unsigned long long)pts, (unsigned long long)address, (unsigned long long)length);
    if (address < atomic_load(&video_low)) atomic_store(&video_low, address);
    if (length <= UINT64_MAX - address && address + length > atomic_load(&video_high))
        atomic_store(&video_high, address + length);
    atomic_store_explicit(&video_active, 1, memory_order_release);
    pthread_mutex_unlock(&video_mutex);
}

void vk_video_use(const tex_t *tex, int direct_present) {
    if (!atomic_load_explicit(&video_active, memory_order_relaxed) || !tex || !tex->mem) return;
    uint64_t addr = (uintptr_t)tex->mem;
    if (addr < atomic_load_explicit(&video_low, memory_order_relaxed) ||
        addr >= atomic_load_explicit(&video_high, memory_order_relaxed)) return;
    pthread_mutex_lock(&video_mutex);
    for (unsigned i = 0; i < 8 && video_sequence > i; ++i) {
        struct video_surface *v = &video_surfaces[(video_sequence - i) % 8];
        if (addr < v->address || addr - v->address >= v->length) continue;
        unsigned flag = 1u << direct_present;
        if (!(v->reported & flag) && (v->sequence <= 4 || v->sequence % 120 == 0))
            VKLOG("NVN movie %s seq=%llu image=%u pts=%llu texture=%llu offset=%#llx",
                  direct_present ? "present-source" : "sampled-view", (unsigned long long)v->sequence,
                  v->id, (unsigned long long)v->pts, (unsigned long long)tex->serial, (unsigned long long)(addr-v->address));
        v->reported |= flag;
        break;
    }
    pthread_mutex_unlock(&video_mutex);
}
#endif
VkImageView vk_tex_view(const tex_t *t, const view_t *vw, int dim, int arrayed, int role, vkimg_t **img_out) {
    const int as_attachment = role == VK_VIEW_ATTACHMENT;
    tex_t storage_desc;
    if (role == VK_VIEW_STORAGE && !(t->d.flags & NVN_TEXTURE_FLAGS_IMAGE)) {
        storage_desc = *t;
        storage_desc.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
        t = &storage_desc;
    }
    vkimg_t *im = vk_tex(t); if (img_out) *img_out = im;
    if (!im) return VK_NULL_HANDLE;
    if (as_attachment) {
                         note_image_write(im); im->wseq = ++g_wseq; im->wpres = g_tex_frame + 1; im->gpu_w = 1; vk_tex_gpu_mark(t); } else if (im->novl) alias_check(im);   // r72 mark
    if (!as_attachment && role == VK_VIEW_SAMPLED) rdi_use(im, 0);   // slot12 replaydiag: a sampled view (a storage one is a write: vk_tex_content_changed)
    vk_watch(im, as_attachment ? "attachment" : "sampled or image");
    if (!as_attachment) { void vk_texdump_note(vkimg_t *im); vk_texdump_note(im); }
    VkImageView view = image_view(im, t, vw, dim, arrayed, role);
#if defined(DM_NXN1X)
    if (view && role == VK_VIEW_SAMPLED) vk_video_use(t, 0);
#endif
    return view;
}
static VkImageView image_view(vkimg_t *im, const tex_t *t, const view_t *vw, int dim, int arrayed, int role) {
    const int as_attachment = role == VK_VIEW_ATTACHMENT;
    const vkfmt_t *f = vk_fmt(vw && vw->set_format ? vw->format : t->d.format); if (!f) f = vk_fmt(t->d.format);
    uint32_t lv0 = vw && vw->set_levels ? vw->base_level : 0, nlv = vw && vw->set_levels ? vw->nlevels : im->levels - lv0;
    if (lv0 >= im->levels) lv0 = im->levels - 1; if (!nlv || lv0 + nlv > im->levels) nlv = im->levels - lv0;
    // a 3D image drawn into: the view's layers are depth slices of its level (the image is 2D-array compatible). Xenoblade Chronicles 3 renders its 16x16x16 colour grading
    // LUT one slice per pass through 2D views at layers 0-15; clamped to the image's one layer, every slice went to slice 0, and the graded title sky came out flat blue
    uint32_t nlay = as_attachment && im->vtype == VK_IMAGE_VIEW_TYPE_3D ? (im->d >> lv0 ? im->d >> lv0 : 1) : im->layers;
    uint32_t ly0 = vw && vw->set_layers ? vw->min_layer : 0, nly = vw && vw->set_layers ? vw->nlayers : nlay - ly0;
    if (ly0 >= nlay) ly0 = nlay - 1; if (!nly || ly0 + nly > nlay) nly = nlay - ly0;
    int target = vw && vw->set_target ? vw->target : im->target;
    VkImageViewType type = vtype_of(target);
    if ((type == VK_IMAGE_VIEW_TYPE_CUBE || type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) && !as_attachment && vw && vw->set_layers && vw->nlayers % 6) {   // pkm5: layers counted in cubes (the texture header's way)
        uint32_t c0 = vw->min_layer, cn = vw->nlayers ? vw->nlayers : 1;
        if ((c0 + cn) * 6 <= im->layers) { ly0 = c0 * 6; nly = cn * 6; static int n; if (n++ < 4) VKLOG("pkm5: cube view of %u + %u layers on a %u-layer image: counted in cubes (faces %u + %u)", c0, cn, im->layers, ly0, nly); }
        else { static int n; if (n++ < 4) VKLOG("[RED] pkm5: cube view of %u + %u layers past the %u layers of its image: left as it was", c0, cn, im->layers); } }
    // what the shader declares wins over the object's target (a 2D render target sampled as an array, a cube face read as 2D)
    if (dim != 0xff && !as_attachment) {
        VkImageViewType want = dim == 0 ? (arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : dim == 1 ? (arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D) :
                               dim == 2 ? VK_IMAGE_VIEW_TYPE_3D : dim == 3 ? (arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : type;
        if (want == VK_IMAGE_VIEW_TYPE_2D && type == VK_IMAGE_VIEW_TYPE_2D_ARRAY) nly = 1;
        if ((want == VK_IMAGE_VIEW_TYPE_CUBE || want == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) && (im->layers < 6 || nly % 6)) want = type;   // not a cube: keep it as it is
        if (want == VK_IMAGE_VIEW_TYPE_CUBE) nly = 6;
        if (want == VK_IMAGE_VIEW_TYPE_3D && im->vtype != VK_IMAGE_VIEW_TYPE_3D) return VK_NULL_HANDLE;
        if ((want == VK_IMAGE_VIEW_TYPE_1D || want == VK_IMAGE_VIEW_TYPE_1D_ARRAY) != (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_1D_ARRAY)) {
            // tex1d (A/B, NXN1X_TEX1D_DUMMY=1): a 1D slot holding a 2D texture (or the reverse) kept the image's own view type, which leaves a view type that does
            // not match the shader's image declaration (undefined in Vulkan). Hyrule Warriors' grass VS 1df3a0c17a3e9e80 samples its 1D slot 0x38, which holds the
            // game's 8x8 2D default: with this switch the slot gets the zero dummy of the declared type instead (as a 3D slot over a 2D image already does above)
            static int tex1d = -1; if (tex1d < 0) { const char *e = getenv("NXN1X_TEX1D_DUMMY"); tex1d = e && e[0] == '1'; if (tex1d) VKLOG("tex1d: a view type the shader does not declare (1D vs 2D) gets the zero dummy of the declared type (NXN1X_TEX1D_DUMMY=1)"); }
            if (tex1d) return VK_NULL_HANDLE;
            want = type;
        }
        type = want;
    }
    if (as_attachment) { type = im->layers > 1 && nly > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : (im->vtype == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D); nlv = 1; }
    if (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_3D) nly = 1;
    VkImageAspectFlags aspect = im->aspect;
    if (aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) && !as_attachment)
        aspect = (!f->depth || t->stencil || (vw && vw->set_dsmode && vw->dsmode == 1)) ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;   // (STENCIL8 alone may be stored combined)
    VkComponentMapping swz = f->swz;
    if (role == VK_VIEW_SAMPLED) {
        if (t->set_swz) { VkComponentMapping o = swz; swz.r = apply(swz_comp(t->swz & 7), &o, VK_COMPONENT_SWIZZLE_R); swz.g = apply(swz_comp(t->swz >> 3 & 7), &o, VK_COMPONENT_SWIZZLE_G);
                          swz.b = apply(swz_comp(t->swz >> 6 & 7), &o, VK_COMPONENT_SWIZZLE_B); swz.a = apply(swz_comp(t->swz >> 9 & 7), &o, VK_COMPONENT_SWIZZLE_A); }
        if (vw && vw->set_swizzle) { VkComponentMapping o = swz; swz.r = apply(swz_comp(vw->swz[0]), &o, VK_COMPONENT_SWIZZLE_R); swz.g = apply(swz_comp(vw->swz[1]), &o, VK_COMPONENT_SWIZZLE_G);
                                     swz.b = apply(swz_comp(vw->swz[2]), &o, VK_COMPONENT_SWIZZLE_B); swz.a = apply(swz_comp(vw->swz[3]), &o, VK_COMPONENT_SWIZZLE_A); }
    } else swz = (VkComponentMapping){0};
    VkFormat fmt = f->vk;
    int raw_blocks = as_attachment && im->fmt == VK_FORMAT_BC4_UNORM_BLOCK &&
                     fmt == VK_FORMAT_R16G16B16A16_UINT && nlv == 1 && nly == 1;
    if ((im->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT) ||
        (!raw_blocks && f->compressed != vk_fmt(im->nvn_format)->compressed)) fmt = im->fmt;
    // Eden's render-pass keys use the packed spelling of these little-endian
    // RGBA formats. Vulkan compatibility compares VkFormat, not byte layout.
    if (as_attachment && vk_eden_mode()) switch (fmt) {
    case VK_FORMAT_R8G8B8A8_UNORM: fmt = VK_FORMAT_A8B8G8R8_UNORM_PACK32; break;
    case VK_FORMAT_R8G8B8A8_SNORM: fmt = VK_FORMAT_A8B8G8R8_SNORM_PACK32; break;
    case VK_FORMAT_R8G8B8A8_UINT: fmt = VK_FORMAT_A8B8G8R8_UINT_PACK32; break;
    case VK_FORMAT_R8G8B8A8_SINT: fmt = VK_FORMAT_A8B8G8R8_SINT_PACK32; break;
    case VK_FORMAT_R8G8B8A8_SRGB: fmt = VK_FORMAT_A8B8G8R8_SRGB_PACK32; break;
    default: break;
    }
    if (role == VK_VIEW_STORAGE) {
        int twin;
        if (!(im->usage & VK_IMAGE_USAGE_STORAGE_BIT) || fmt != f->vk) return VK_NULL_HANDLE;
        if (!img_storage_ok(f, &twin)) {   // ootu32: the shader's declared format, same texel size
            const VkFormat tf = typed_fmt(g_vk_storage_want);
            if (!tf || typed_bytes(g_vk_storage_want) != f->bpb || !img_typed_storage_ok(f) || !(im->cflags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) {
                static int n; if (n++ < 8) VKLOG("[RED] ootu32: no storage view of %ux%u nvn format %#x (vk %d): the shader declares format %d", im->w, im->h, im->nvn_format, f->vk, g_vk_storage_want);
                return VK_NULL_HANDLE; }
            fmt = tf; twin = 0; if (f->vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) g_e5_storage_used = 1;
            static int n; if (n++ < 8) VKLOG("ootu32: storage view of %ux%u nvn format %#x (vk %d) as declared vk %d (#%d)", im->w, im->h, im->nvn_format, f->vk, tf, n);
        }
        if (twin) fmt = srgb_twin(fmt);
        { const vkfmt_t *of = vk_fmt(im->nvn_format); if (of && of->vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) g_e5_storage_used = 1; }   // ootu32: also through a view format (OoT: the network's own path is live)
    }
    pthread_mutex_lock(&g_view_mu);
    vview *head = nvn_hm_get(&g_views, &im);
    for (vview *v = head; v; v = v->next)
        if (v->type == type && v->fmt == fmt && v->lv0 == lv0 && v->nlv == nlv && v->ly0 == ly0 && v->nly == nly && v->aspect == aspect && v->attach == role && !memcmp(&v->swz, &swz, sizeof swz)) { pthread_mutex_unlock(&g_view_mu); vk_view_fmt = fmt; vk_view_layers = nly; return v->v; }
    VkImageViewUsageCreateInfo uci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, 0};
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(V.pd, fmt, &fp);
    uci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (as_attachment) uci.usage = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) | (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0);
    else if (role == VK_VIEW_STORAGE) uci.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    uci.usage &= im->usage;
    if (!uci.usage) { pthread_mutex_unlock(&g_view_mu); return VK_NULL_HANDLE; }
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, &uci, 0, im->img, type, fmt, swz, {aspect, lv0, nlv, ly0, nly}};
    VkImageView view; VkResult r = vkCreateImageView(V.dev, &ci, NULL, &view);
    if (r) { pthread_mutex_unlock(&g_view_mu); VKRED_ONCE(15, "vkCreateImageView failed (%d): type %d format %d levels %u+%u layers %u+%u of image %ux%ux%u", r, type, fmt, lv0, nlv, ly0, nly, im->w, im->h, im->layers); return VK_NULL_HANDLE; }
    g_views_made++;   // (under g_view_mu; NVN2 memory report)
    vview *nv = calloc(1, sizeof *nv); *nv = (vview){view, type, fmt, lv0, nlv, ly0, nly, swz, aspect, role, head};
    nvn_hm_put(&g_views, &im, nv);
    pthread_mutex_unlock(&g_view_mu);
    vk_view_fmt = fmt; vk_view_layers = nly;
    return view;
}

// ---------------- texture / sampler ID tables ----------------
// A registration as the replay of each submission sees it. NVN's texture pool is memory the GPU reads as it runs, and a game rewrites an ID once the work
// that used it is done; with the replay on its own thread (DM_NVN_ASYNC) a submission is replayed after the game has gone on, and Xenoblade Chronicles 3
// rewrites about 90 IDs a second with other textures: a frame replayed late drew with the next frame's textures (whole-screen garbage while the camera
// turned). So each registration keeps the number of replay jobs queued when it was made, and a job sees the newest one made before it was queued
// (nvn_jobs_queued / nvn_replaying_job, gfx/nvn/hle_nvn.c); without a replay thread that is always the newest. TREG_OLD older ones per ID, kept for IDs rewritten
enum { TREG_OLD = 3 };
typedef struct { tex_t t; view_t v; uint8_t ok, has_v; uint64_t seq; } treg_ver;
typedef struct { treg_ver cur; treg_ver *old; } treg_t;   // old: TREG_OLD earlier versions, newest first (allocated at the first rewrite)
static treg_t *g_treg; static int g_ntreg;
uint64_t g_tex_gen = 1;   // changes whenever a texture ID is registered again or views go (gfx_vk_exec.c's per-submission view cache)
uint64_t g_samp_gen = 1;  // changes whenever a sampler ID is registered with other parameters (gfx_vk_exec.c samp_cached and the descriptor set memo)
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static VkSampler *g_samp; static int g_nsreg;
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;
__attribute__((weak)) uint64_t nvn_jobs_queued(void) { return 0; }
__attribute__((weak)) uint64_t nvn_replaying_job(void) { return 0; }
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    if (tid < 0) return;
    uint64_t seq = nvn_jobs_queued();
    pthread_mutex_lock(&g_rmu);
    if (tid >= g_ntreg) { int n = tid + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    g_tex_gen++; treg_t *r = &g_treg[tid];
    if (r->cur.ok && seq) {   // keep the one replaced for the jobs queued before this registration
        if (!r->old) r->old = calloc(TREG_OLD, sizeof *r->old);
        if (r->old) { memmove(r->old + 1, r->old, sizeof *r->old * (TREG_OLD - 1)); r->old[0] = r->cur; }
    }
    r->cur.ok = t != NULL; if (t) r->cur.t = *t; r->cur.has_v = v != NULL; if (v) r->cur.v = *v; r->cur.seq = seq;
    pthread_mutex_unlock(&g_rmu);
}
void mtl_set_tex_pool_n(int n) { (void)n; }
static __thread treg_ver t_treg;   // the version a lookup found, copied (the table may be rewritten or grow meanwhile)
vkimg_t *vk_tex_by_id(int tid, const view_t **view_out, const tex_t **tex_out) {
    uint64_t job = nvn_replaying_job();
    pthread_mutex_lock(&g_rmu);
    const treg_ver *r = tid > 0 && tid < g_ntreg ? &g_treg[tid].cur : NULL;
    if (r && job && r->seq >= job && g_treg[tid].old)   // registered after this job was queued: the newest earlier one
        for (int k = 0; k < TREG_OLD; k++) { const treg_ver *o = &g_treg[tid].old[k]; if (!o->ok) break; r = o; if (o->seq < job) break; }
    int ok = r && r->ok; if (ok) t_treg = *r;
    pthread_mutex_unlock(&g_rmu);
    if (!ok) return NULL;
    if (view_out) *view_out = t_treg.has_v ? &t_treg.v : NULL; if (tex_out) *tex_out = &t_treg.t;
    return (vkimg_t *)1;   // presence only: the caller makes the view with vk_tex_view
}
// vsres2 (log only, gfx_vk_eden.inc vsres_diag): one texture ID's registrations, the current one and the TREG_OLD earlier ones (serial, size, target, nvn format,
// storage, the job count when made), the replay job looking it up and which of them vk_tex_by_id hands it: does a draw get the game's current registration?
void vk_treg_describe(int tid, char *out, size_t cap) {
    const uint64_t job = nvn_replaying_job();
    pthread_mutex_lock(&g_rmu);
    if (tid <= 0 || tid >= g_ntreg) { pthread_mutex_unlock(&g_rmu); snprintf(out, cap, "id %d: never registered", tid); return; }
    const treg_t *r = &g_treg[tid];
    int pick = -1;   // as vk_tex_by_id: -1 the current one, k the old one at k
    if (job && r->cur.seq >= job && r->old)
        for (int k = 0; k < TREG_OLD; k++) { const treg_ver *o = &r->old[k]; if (!o->ok) break; pick = k; if (o->seq < job) break; }
    int n = snprintf(out, cap, "id %d job %llu: current %s serial %llu %ux%ux%u target %d fmt %#x mem %p seq %llu", tid, (unsigned long long)job, r->cur.ok ? "" : "(none)",
                     (unsigned long long)r->cur.t.serial, r->cur.t.d.width, r->cur.t.d.height, r->cur.t.d.depth, r->cur.t.d.target, r->cur.t.d.format, (void *)r->cur.t.mem,
                     (unsigned long long)r->cur.seq);
    for (int k = 0; r->old && k < TREG_OLD && n > 0 && (size_t)n < cap; k++) {
        const treg_ver *o = &r->old[k]; if (!o->ok) break;
        n += snprintf(out + n, cap - (size_t)n, "; old%d serial %llu %ux%ux%u target %d seq %llu", k, (unsigned long long)o->t.serial, o->t.d.width, o->t.d.height, o->t.d.depth,
                      o->t.d.target, (unsigned long long)o->seq);
    }
    if (n > 0 && (size_t)n < cap) snprintf(out + n, cap - (size_t)n, "; returned %s", pick < 0 ? "current" : pick == 0 ? "old0" : pick == 1 ? "old1" : "old2");
    pthread_mutex_unlock(&g_rmu);
}
static VkSamplerAddressMode wrap(int w) {
    switch (w) { case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case 2: case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
                 case 4: case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
static VkCompareOp cmp_op(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }   // NVN NEVER..ALWAYS = 1..8, Vulkan 0..7 in the same order
static VkSampler make_sampler(const samp_par *p) {
    VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    VkSamplerReductionModeCreateInfo rd = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL, p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : p->reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, p->reduction ? &rd : NULL};
    ci.magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.minFilter = p->minf == 1 || p->minf == 3 || p->minf == 5 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci.mipmapMode = p->minf == 4 || p->minf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    ci.addressModeU = wrap(p->ws); ci.addressModeV = wrap(p->wt); ci.addressModeW = wrap(p->wr);
    ci.mipLodBias = p->lod_bias;
    ci.anisotropyEnable = p->aniso > 1.0f; ci.maxAnisotropy = p->aniso > V.max_aniso ? V.max_aniso : p->aniso;
    ci.compareEnable = p->cmp_mode == 1; ci.compareOp = cmp_op(p->cmp_func);
    ci.minLod = p->lod_min; ci.maxLod = p->minf <= 1 ? 0.25f : p->lod_max;   // no mipmap filter: level 0 only (GL's rule)
    int border = ci.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (border) {
        const float *b = p->border;
        if (b[0] == 0 && b[1] == 0 && b[2] == 0 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1) ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if ((b[0] || b[1] || b[2] || b[3]) && V.has_custom_border) {
            ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT; memcpy(cb.customBorderColor.float32, b, 16); cb.format = VK_FORMAT_UNDEFINED;
            cb.pNext = (void *)ci.pNext; ci.pNext = &cb;
        }
    }
    VkSampler s; if (vkCreateSampler(V.dev, &ci, NULL, &s)) return VK_NULL_HANDLE;
    return s;
}
void mtl_reg_samp(int sid, const samp_par *sp) {
    if (sid < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (sid >= g_nsreg) { int n = sid + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); g_samp = realloc(g_samp, n * sizeof *g_samp);
        memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); memset(g_samp + g_nsreg, 0, (n - g_nsreg) * sizeof *g_samp); g_nsreg = n; }
    if (!g_sreg_ok[sid] || memcmp(&g_sreg[sid], sp, sizeof *sp)) { g_sreg[sid] = *sp; g_sreg_ok[sid] = 1; g_samp[sid] = VK_NULL_HANDLE; __atomic_fetch_add(&g_samp_gen, 1, __ATOMIC_RELEASE); }   // samplers are cached by parameters below
    pthread_mutex_unlock(&g_rmu);
}
static nvn_hmap g_samp_cache = {.klen = sizeof(samp_par)};   // samp_par → VkSampler (VkSamplers are never destroyed: few distinct parameter sets)
void vk_samp_describe(int sid, char *out, size_t n) {   // the registered parameters (traces)
    pthread_mutex_lock(&g_rmu);
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) { const samp_par *p = &g_sreg[sid]; snprintf(out, n, "sampler %d: wrap %d %d %d min %d mag %d lod %g..%g bias %g aniso %g cmp %d/%d red %d border %g %g %g %g", sid, p->ws, p->wt, p->wr,
        p->minf, p->magf, p->lod_min, p->lod_max, p->lod_bias, p->aniso, p->cmp_mode, p->cmp_func, p->reduction, p->border[0], p->border[1], p->border[2], p->border[3]); }
    else snprintf(out, n, "sampler %d: not registered (dummy)", sid);
    pthread_mutex_unlock(&g_rmu);
}
VkSampler vk_samp_by_id(int sid) {
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE;
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) {
        s = g_samp[sid];
        if (!s) { s = nvn_hm_get(&g_samp_cache, &g_sreg[sid]); if (!s) { s = make_sampler(&g_sreg[sid]); if (s) nvn_hm_put(&g_samp_cache, &g_sreg[sid], s); } g_samp[sid] = s; }
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
int vk_samp_cmp(int sid) {
    pthread_mutex_lock(&g_rmu); int c = sid > 0 && sid < g_nsreg && g_sreg_ok[sid] ? g_sreg[sid].cmp_mode : -1; pthread_mutex_unlock(&g_rmu); return c;
}
VkSampler vk_samp_cmp_variant(int sid, int want) {   // r46: Vulkan leaves a sample undefined when the instruction's Dref and the sampler's compare differ
    pthread_mutex_lock(&g_rmu);
    VkSampler s = VK_NULL_HANDLE;
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid]) {
        samp_par p = g_sreg[sid]; p.cmp_mode = want ? 1 : 0; if (want && (p.cmp_func < 1 || p.cmp_func > 8)) p.cmp_func = 4;   // LEQUAL when none was set
        s = nvn_hm_get(&g_samp_cache, &p); if (!s) { s = make_sampler(&p); if (s) nvn_hm_put(&g_samp_cache, &p, s); }
    }
    pthread_mutex_unlock(&g_rmu);
    return s;
}
VkSampler vk_dummy_sampler(void) {
    static VkSampler s; if (s) return s;
    samp_par p = {0}; p.magf = 1; p.minf = 1; p.lod_max = 1000; p.aniso = 1; s = make_sampler(&p); return s;
}
// ---------------- dummies: what an unbound slot reads (zeros) ----------------
VkImageView vk_dummy_view(int dim, int arrayed, int ms, int storage, int is_int) {
    static VkImageView cache[4][2][2][2][2]; static vkimg_t *imgs[4][2][2][2]; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    int d = dim == 0 ? 0 : dim == 2 ? 2 : dim == 3 ? 3 : 1; arrayed = !!arrayed; ms = !!ms && d == 1; storage = !!storage; is_int = !!is_int;
    pthread_mutex_lock(&mu);
    VkImageView v = cache[d][arrayed][ms][storage][is_int];
    if (!v) {
        vkimg_t **pi = &imgs[d][ms][storage][is_int];
        if (!*pi) {
            tex_t t; memset(&t, 0, sizeof t); t.magic = MAGIC('T', 'E', 'X', 0); t.serial = 1;
            t.d.format = is_int ? 0x27 : 0x25; t.d.width = t.d.height = 1; t.d.depth = d == 3 ? 12 : 6; t.d.levels = 1; t.d.samples = ms ? 4 : 0;
            t.d.target = d == 0 ? 3 : d == 2 ? 2 : d == 3 ? 9 : ms ? 6 : 4; t.d.flags = storage ? NVN_TEXTURE_FLAGS_IMAGE : 0;
            if (d == 2) t.d.depth = 1;
            *pi = img_new(&t, vk_fmt(t.d.format));
            vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer (leaf; under this function's mu)
            VkCommandBuffer cb = vk_upload_cmd(); vk_tex_prepare(cb, *pi);
            VkClearColorValue z = {{0}}; VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
            // dummy1 (A/B, NXN1X_DUMMY_ONE=1): the dummies read 1.0 (integer 1) instead of 0. Hyrule Warriors' grass VS samples two 3D slots that hold the game's 8x8 2D
            // default texture, which gets the 3D dummy: does reading 0 there instead of the default's texels make its vertices degenerate (and expensive)?
            { static int one = -1; if (one < 0) { const char *e = getenv("NXN1X_DUMMY_ONE"); one = e && e[0] == '1'; if (one) VKLOG("dummy1: texture dummies read 1.0 (NXN1X_DUMMY_ONE=1)"); }
              if (one) { if (is_int) { z.uint32[0] = z.uint32[1] = z.uint32[2] = z.uint32[3] = 1; } else { z.float32[0] = z.float32[1] = z.float32[2] = z.float32[3] = 1.0f; } } }
            if (!ms) vkCmdClearColorImage(cb, (*pi)->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg);
            vk_up_unlock();
        }
        vkimg_t *im = *pi;
        VkImageViewType type = d == 0 ? (arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : d == 2 ? VK_IMAGE_VIEW_TYPE_3D : d == 3 ? (arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : (arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D);
        uint32_t nly = type == VK_IMAGE_VIEW_TYPE_CUBE ? 6 : type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY ? 12 : arrayed ? im->layers : 1;
        VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, type, im->fmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nly}};
        VKCHK(vkCreateImageView(V.dev, &ci, NULL, &v)); cache[d][arrayed][ms][storage][is_int] = v;
    }
    pthread_mutex_unlock(&mu);
    return v;
}
VkBufferView vk_dummy_tbuf(int storage) {
    static VkBufferView v[2]; static VkBuffer b; static VkDeviceMemory m; storage = !!storage;
    if (v[storage]) return v[storage];
    if (!b) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 65536, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bi, NULL, &b)); VkMemoryRequirements req; vkGetBufferMemoryRequirements(V.dev, b, &req);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &m)); VKCHK(vkBindBufferMemory(V.dev, b, m, 0));
        void *p; VKCHK(vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)); memset(p, 0, 65536);
    }
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, VK_FORMAT_R32_UINT, 0, 65536};
    VKCHK(vkCreateBufferView(V.dev, &ci, NULL, &v[storage]));
    return v[storage];
}
// a buffer texture's texel buffer view over its pool (cached by texture contents: the registered copy)
VkBufferView vk_tbuf_view(const tex_t *t) {
    static nvn_hmap cache = {.klen = 24}; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    // keyed by the pool too: a pool finalized and another made at the same address has a new VkBuffer, and the old one's memory is freed once the GPU is past it. Keyed by address
    // alone, the view over the old buffer was handed out again and shaders read and wrote freed memory, which the kernel gives to newer buffers (Xenoblade Chronicles 3: the
    // allocator kept inside one of its memory pools had its tree bins overwritten during the opening movie, and malloc crashed at the next scene)
    vkpool_t *pl = t->mem ? vk_pool_of((uint64_t)(uintptr_t)t->mem, 1) : NULL;
    struct { uint64_t mem, pool; uint32_t fmt, n; } k = {(uint64_t)(uintptr_t)t->mem, pl ? pl->serial : 0, (uint32_t)t->d.format, t->d.width};
    pthread_mutex_lock(&mu);
    VkBufferView v = nvn_hm_get(&cache, &k);
    if (!v) {
        const vkfmt_t *f = vk_fmt(t->d.format); VkBuffer b; VkDeviceSize off;
        if (f && t->mem && vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)t->d.width * f->bpb, &b, &off) && off % V.props.limits.minTexelBufferOffsetAlignment == 0) {
            VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, f->vk, off, (VkDeviceSize)t->d.width * f->bpb};
            if (vkCreateBufferView(V.dev, &ci, NULL, &v)) v = VK_NULL_HANDLE; else nvn_hm_put(&cache, &k, v);
        } else if (!(f && t->mem && vk_buf_at((uint64_t)(uintptr_t)t->mem, (uint64_t)t->d.width * f->bpb, &b, &off) && f->bpb && off % f->bpb == 0))   // r75: else vk_tbuf_copy_view's
            VKRED_ONCE(16, "buffer texture not in a pool or misaligned (format %#x, %u elements)", t->d.format, t->d.width);
    }
    pthread_mutex_unlock(&mu);
    return v;
}
// CPU readback of texels (ReadTexels): the image copied to scratch memory by a submission of its own, waited for
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); int mtl_wait(uint64_t, uint64_t);
    mtl_sub_begin();
    vkimg_t *im = vk_tex(t); const vkfmt_t *f = vk_fmt(t->d.format);
    size_t depth = r->depth ? r->depth : 1;
    if (!im || !f) { mtl_sub_commit(); for (size_t z = 0; z < depth; z++) for (int y = 0; y < r->height; y++) memset(dst + z * img + (size_t)y * row, 0, row); return; }
    unsigned bw = f->bw ? f->bw : 1, bh = f->bh ? f->bh : 1;
    size_t pitch = (size_t)((r->width + bw - 1) / bw) * f->bpb, rows = (r->height + bh - 1) / bh;
    VkBuffer sb; VkDeviceSize so; uint8_t *p = vk_scratch(pitch * rows * depth, 16, &sb, &so);
    VkCommandBuffer cb = vk_cmd(); vk_tex_prepare(cb, im); vk_img_barrier_all(cb);
    int is3d = im->d > 1;
    VkBufferImageCopy c = {so, 0, 0, {im->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, level, is3d ? 0 : layer0 + (uint32_t)r->zoffset, is3d ? 1 : (uint32_t)depth},
                           {r->xoffset, r->yoffset, is3d ? r->zoffset : 0}, {(uint32_t)r->width, (uint32_t)r->height, is3d ? (uint32_t)depth : 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, sb, 1, &c);
    mtl_wait(mtl_sub_commit(), ~0ull);
    for (size_t z = 0; z < depth; z++) for (size_t y = 0; y < rows; y++) memcpy(dst + z * img + y * row, p + (z * rows + y) * pitch, pitch < row ? pitch : row);
}

// r74: texture buffers read through a 2D sampled binding (nvn_mtl.m tbuf_as_2d, nvn_tex.c nvn_tbuf_rows). Maxwell reads 1D textures and texture buffers with the same instruction and the
// descriptor decides; with no type answer the translator reads a plain 1D fetch as 2D at (x & 0x3FFF, x >> 14) (InstEmitAmpereMem.cs), so the host must give a texture buffer there as
// rows of 16384 texels. The Windows host had none of it: vk_tex makes no image for a buffer texture, the slot got the dummy view and read zeros. KTGL's light and decal lists come this
// way (Age of Calamity, Pokemon Pokopia on the Mac); Hyrule Warriors: Age of Imprisonment's lighting passes came out 0 and its play scene was black. Vulkan cannot view a buffer as an
// image, so the rows are a device image the buffer is copied into on the GPU once per submission that reads it, in its upload command buffer (as lin_copy): what the pool held when the
// submission began, as the Mac's tbuf_rows_copy. Integer formats get the integer-texel variants through view_num, which hand the raw words over. DM_VK_TBUF2D=0 turns it off
enum { TB2_N = 512, TB2_WAYS = 8 };
static struct { uint64_t addr, n, used, gen; VkFormat fmt; VkImage img; VkDeviceMemory mem; VkImageView view; int ready, dirty; uint32_t w, bpb; } g_tb2[TB2_N];   // r150: dirty, w, bpb
static pthread_mutex_t g_tb2_mu = PTHREAD_MUTEX_INITIALIZER; static uint64_t g_tb2_tick, g_tb2_made, g_tb2_copies, g_tb2_bytes;
VkImageView vk_tbuf_rows_view(const tex_t *t) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_TBUF2D"); on = !e || atoi(e) != 0; }
    if (!on || !t || !t->mem || t->d.target != NVN_TEXTURE_TARGET_BUFFER || !vk_sub_open()) return VK_NULL_HANDLE;
    const vkfmt_t *f = vk_fmt(t->d.format);
    if (!f || f->bw > 1 || f->bh > 1 || !f->bpb || f->depth || f->stencil || f->compressed) { VKRED_ONCE(160, "r74: texture buffer format %#x has no 2D-row form: the slot reads 0", t->d.format); return VK_NULL_HANDLE; }
    uint64_t n = t->d.width; uint32_t w, h, row;
    if (!nvn_tbuf_rows(n, f->bpb, 1, &w, &h, &row) || h > V.props.limits.maxImageDimension2D || w > V.props.limits.maxImageDimension2D) {
        VKRED_ONCE(161, "r74: texture buffer of %llu elements (format %#x) does not fit 2D rows of %u: the slot reads 0", (unsigned long long)n, t->d.format, NVN_TBUF_ROW); return VK_NULL_HANDLE; }
    uint64_t addr = (uint64_t)(uintptr_t)t->mem; VkBuffer b; VkDeviceSize o;
    if (!vk_buf_at(addr, n * f->bpb, &b, &o) || o % f->bpb) { VKRED_ONCE(162, "r74: texture buffer at %#llx (%llu elements) has no pool buffer: the slot reads 0", (unsigned long long)addr, (unsigned long long)n); return VK_NULL_HANDLE; }
    pthread_mutex_lock(&g_tb2_mu);
    unsigned base = (unsigned)(((addr >> 4) ^ n * 0x9E3779B97F4A7C15ull ^ (uint64_t)f->vk * 31) >> 7) % (TB2_N / TB2_WAYS) * TB2_WAYS, k = base, old = base;
    for (unsigned i = base; i < base + TB2_WAYS; i++) {
        if (g_tb2[i].img && g_tb2[i].addr == addr && g_tb2[i].n == n && g_tb2[i].fmt == f->vk) { k = i; goto found; }
        if (!g_tb2[i].img) { old = i; break; }
        if (g_tb2[i].used < g_tb2[old].used) old = i;
    }
    k = old;
    if (g_tb2[k].img) { vk_defer_free_view(g_tb2[k].view); vk_defer_free_image(g_tb2[k].img, g_tb2[k].mem); memset(&g_tb2[k], 0, sizeof g_tb2[k]); }
    {   VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, f->vk, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VkImage img; if (vkCreateImage(V.dev, &ci, NULL, &img)) { pthread_mutex_unlock(&g_tb2_mu); VKRED_ONCE(163, "r74: cannot make %ux%u rows (VkFormat %d) for a texture buffer: the slot reads 0", w, h, f->vk); return VK_NULL_HANDLE; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, img, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)}; VkDeviceMemory m;
        if (vkAllocateMemory(V.dev, &ai, NULL, &m)) { vkDestroyImage(V.dev, img, NULL); pthread_mutex_unlock(&g_tb2_mu); VKRED_ONCE(164, "r74: no memory for %ux%u texture buffer rows: the slot reads 0", w, h); return VK_NULL_HANDLE; }
        vkBindImageMemory(V.dev, img, m, 0);
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, img, VK_IMAGE_VIEW_TYPE_2D, f->vk, f->swz, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}; VkImageView v;
        if (vkCreateImageView(V.dev, &vi, NULL, &v)) { vk_defer_free_image(img, m); pthread_mutex_unlock(&g_tb2_mu); VKRED_ONCE(165, "r74: cannot view %ux%u texture buffer rows: the slot reads 0", w, h); return VK_NULL_HANDLE; }
        g_tb2[k].addr = addr; g_tb2[k].n = n; g_tb2[k].fmt = f->vk; g_tb2[k].img = img; g_tb2[k].mem = m; g_tb2[k].view = v; g_tb2[k].ready = 0; g_tb2[k].gen = 0; g_tb2[k].w = w; g_tb2[k].bpb = f->bpb;   /* r150 */
        if (!g_tb2_made++) VKLOG("r74: texture buffer of %llu elements (nvn format %#x) read through 2D rows %ux%u (first case)", (unsigned long long)n, t->d.format, w, h);
    }
found:
    g_tb2[k].used = ++g_tb2_tick;
    {   vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and the entry's g_up_gen memo (leaf; under g_tb2_mu)
        VkCommandBuffer cb = vk_upload_cmd();
        if (g_tb2[k].gen != g_up_gen) {   // the pool's content as of this submission
            g_tb2[k].gen = g_up_gen;
            vk_img_barrier_all(cb);   // after the reads of earlier submissions
            if (!g_tb2[k].ready) {
                VkImageMemoryBarrier ib = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                           VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_tb2[k].img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &ib); g_tb2[k].ready = 1; }
            uint32_t full = (uint32_t)(n / w), rem = (uint32_t)(n - (uint64_t)full * w); VkBufferImageCopy r[2]; uint32_t nr = 0;
            if (full) r[nr++] = (VkBufferImageCopy){o, w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, full, 1}};
            if (rem) r[nr++] = (VkBufferImageCopy){o + (VkDeviceSize)full * w * f->bpb, w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, (int32_t)full, 0}, {rem, 1, 1}};
            vkCmdCopyBufferToImage(cb, b, g_tb2[k].img, VK_IMAGE_LAYOUT_GENERAL, nr, r);
            vk_img_barrier_all(cb);
            uint64_t c = ++g_tb2_copies; g_tb2_bytes += n * f->bpb;
            if (!(c & (c - 1)) && c >= 64) VKLOG("r74: texture buffer rows: %llu made, %llu copies, %.1f MB copied in all", (unsigned long long)g_tb2_made, (unsigned long long)c, g_tb2_bytes / 1048576.0);
        }
        vk_up_unlock();
    }
    VkImageView v = g_tb2[k].view;
    pthread_mutex_unlock(&g_tb2_mu);
    return v;
}

// r75: a texture buffer at a pool offset below the device's texel-buffer alignment (minTexelBufferOffsetAlignment, 64 bytes on Adreno) had no view and read zeros
// (vk_tbuf_view, [RED] 16). Koei Tecmo's engine keeps two copies of its light list 32 bytes apart (Fire Emblem: Three Houses alternated between lit and unlit frames on
// an iPhone, whose alignment is larger than a Mac's; nvn_mtl.m tbuf_rows_copy / tbuf_shadow). Such a sampled texture buffer is read through a device-local copy at offset
// 0, copied on the GPU from the pool once per upload command buffer that reads it (as r74's rows): what the pool held when that part of the submission began, which
// includes the GPU's writes of the parts before it. Storage texel buffers keep the dummy (writes cannot be redirected). DM_VK_TBUF_COPY=0 turns it off
enum { TBC_N = 256, TBC_WAYS = 8 };
static struct { uint64_t addr, n, used, gen; VkFormat fmt; VkBuffer buf; VkDeviceMemory mem; VkBufferView view; int dirty; uint32_t bpb; } g_tbc[TBC_N];   // r150: dirty, bpb
static pthread_mutex_t g_tbc_mu = PTHREAD_MUTEX_INITIALIZER; static uint64_t g_tbc_tick, g_tbc_made, g_tbc_copies, g_tbc_bytes;
VkBufferView vk_tbuf_copy_view(const tex_t *t) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_TBUF_COPY"); on = !e || atoi(e) != 0; }
    if (!on || !t || !t->mem || t->d.target != NVN_TEXTURE_TARGET_BUFFER || !vk_sub_open()) return VK_NULL_HANDLE;
    const vkfmt_t *f = vk_fmt(t->d.format); if (!f || !f->bpb) return VK_NULL_HANDLE;
    uint64_t n = t->d.width, sz = n * f->bpb, addr = (uint64_t)(uintptr_t)t->mem; VkBuffer b; VkDeviceSize o;
    if (!n || n > V.props.limits.maxTexelBufferElements || !vk_buf_at(addr, sz, &b, &o) || o % f->bpb || !(o % V.props.limits.minTexelBufferOffsetAlignment)) return VK_NULL_HANDLE;   // (aligned: vk_tbuf_view's)
    pthread_mutex_lock(&g_tbc_mu);
    unsigned base = (unsigned)(((addr >> 4) ^ n * 0x9E3779B97F4A7C15ull ^ (uint64_t)f->vk * 31) >> 7) % (TBC_N / TBC_WAYS) * TBC_WAYS, k = base, old = base;
    for (unsigned i = base; i < base + TBC_WAYS; i++) {
        if (g_tbc[i].buf && g_tbc[i].addr == addr && g_tbc[i].n == n && g_tbc[i].fmt == f->vk) { k = i; goto found; }
        if (!g_tbc[i].buf) { old = i; break; }
        if (g_tbc[i].used < g_tbc[old].used) old = i;
    }
    k = old;
    if (g_tbc[k].buf) { vk_defer_free_bview(g_tbc[k].view); vk_defer_free_buffer(g_tbc[k].buf, g_tbc[k].mem); memset(&g_tbc[k], 0, sizeof g_tbc[k]); }   // (the view goes first: the list is freed in order)
    {   VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VkBuffer nb; if (vkCreateBuffer(V.dev, &bi, NULL, &nb)) { pthread_mutex_unlock(&g_tbc_mu); VKRED_ONCE(166, "r75: cannot make an aligned copy of %llu bytes for a texture buffer: the slot reads 0", (unsigned long long)sz); return VK_NULL_HANDLE; }
        VkMemoryRequirements rq; vkGetBufferMemoryRequirements(V.dev, nb, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)}; VkDeviceMemory m;
        if (vkAllocateMemory(V.dev, &ai, NULL, &m)) { vkDestroyBuffer(V.dev, nb, NULL); pthread_mutex_unlock(&g_tbc_mu); VKRED_ONCE(167, "r75: no memory for an aligned copy of %llu bytes of a texture buffer: the slot reads 0", (unsigned long long)sz); return VK_NULL_HANDLE; }
        vkBindBufferMemory(V.dev, nb, m, 0);
        VkBufferViewCreateInfo vi = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, nb, f->vk, 0, sz}; VkBufferView v;
        if (vkCreateBufferView(V.dev, &vi, NULL, &v)) { vk_defer_free_buffer(nb, m); pthread_mutex_unlock(&g_tbc_mu); VKRED_ONCE(168, "r75: cannot view the aligned copy of a texture buffer (VkFormat %d): the slot reads 0", f->vk); return VK_NULL_HANDLE; }
        g_tbc[k].addr = addr; g_tbc[k].n = n; g_tbc[k].fmt = f->vk; g_tbc[k].buf = nb; g_tbc[k].mem = m; g_tbc[k].view = v; g_tbc[k].gen = 0; g_tbc[k].bpb = f->bpb;   /* r150 */
        if (!g_tbc_made++) VKLOG("r75: texture buffer of %llu elements (nvn format %#x) at pool offset %#llx, below the device's %llu-byte texel-buffer alignment: read through an aligned copy (first case)",
                                 (unsigned long long)n, t->d.format, (unsigned long long)o, (unsigned long long)V.props.limits.minTexelBufferOffsetAlignment);
    }
found:
    g_tbc[k].used = ++g_tbc_tick;
    {   vk_up_lock();   // R6/R7/R8 (parallel recording), R8: the upload command buffer and the entry's g_up_gen memo (leaf; under g_tbc_mu)
        VkCommandBuffer cb = vk_upload_cmd();
        if (g_tbc[k].gen != g_up_gen) {   // the pool's content as of this part of the submission
            g_tbc[k].gen = g_up_gen;
            vk_img_barrier_all(cb);   // after the reads of earlier parts and the writes before them
            VkBufferCopy c = {o, 0, sz}; vkCmdCopyBuffer(cb, b, g_tbc[k].buf, 1, &c);
            vk_img_barrier_all(cb);
            uint64_t cc = ++g_tbc_copies; g_tbc_bytes += sz;
            if (!(cc & (cc - 1)) && cc >= 64) VKLOG("r75: aligned texture buffer copies: %llu made, %llu copies, %.1f MB copied in all", (unsigned long long)g_tbc_made, (unsigned long long)cc, g_tbc_bytes / 1048576.0);
        }
        vk_up_unlock();
    }
    VkBufferView v = g_tbc[k].view;
    pthread_mutex_unlock(&g_tbc_mu);
    return v;
}

// r150: the copies above (r74 rows, r75 aligned) hold the pool as it was when that part of the submission began. A list the GPU writes in the same part (Hyrule Warriors:
// Age of Imprisonment's clustered light list, 1,012,000 words at a fixed address, written by compute 04ba2c60 right before the lighting pass reads it) was then read one
// frame late: lights stuck to the previous frame's clusters, so the screen broke into black blocks for a frame or two while the camera moved, and the hand lamp's light
// was lost. With DM_VK_TBUF_FRESH=1 (live) a GPU write into such a buffer (vk_gpu_wrote / vk_sub_wrote) marks its copy, and the next pass copies it again on the GPU
// before it begins (pass_commit, outside any render pass), after the write and before the reads
extern volatile long dm_env_gen;
static int tbuf_fresh_on(void) { static long seen = -1; static int on; long g = dm_env_gen; if (g != seen) { seen = g; const char *e = getenv("DM_VK_TBUF_FRESH"); on = e && atoi(e) != 0; } return on; }
static volatile int g_tb_dirty; static uint64_t g_tb_fresh_n, g_tb_fresh_bytes;
void vk_tbuf_wrote(uint64_t a, uint64_t n) {
    if (!n || !tbuf_fresh_on()) return;
    pthread_mutex_lock(&g_tb2_mu);
    for (int i = 0; i < TB2_N; i++) if (g_tb2[i].img && g_tb2[i].ready && a < g_tb2[i].addr + g_tb2[i].n * g_tb2[i].bpb && g_tb2[i].addr < a + n) { g_tb2[i].dirty = 1; g_tb_dirty = 1; }
    pthread_mutex_unlock(&g_tb2_mu);
    pthread_mutex_lock(&g_tbc_mu);
    for (int i = 0; i < TBC_N; i++) if (g_tbc[i].buf && g_tbc[i].gen && a < g_tbc[i].addr + g_tbc[i].n * g_tbc[i].bpb && g_tbc[i].addr < a + n) { g_tbc[i].dirty = 1; g_tb_dirty = 1; }
    pthread_mutex_unlock(&g_tbc_mu);
}
void vk_tbuf_fresh(VkCommandBuffer cb) {
    if (!g_tb_dirty) return;
    g_tb_dirty = 0; int any = 0;
    pthread_mutex_lock(&g_tb2_mu);
    for (int i = 0; i < TB2_N; i++) if (g_tb2[i].dirty) {
        g_tb2[i].dirty = 0; VkBuffer b; VkDeviceSize o; uint64_t n = g_tb2[i].n; uint32_t w = g_tb2[i].w, bpb = g_tb2[i].bpb;
        if (!g_tb2[i].img || !w || !vk_buf_at(g_tb2[i].addr, n * bpb, &b, &o)) continue;
        if (!any++) vk_img_barrier_all(cb);   // after the writes
        uint32_t full = (uint32_t)(n / w), rem = (uint32_t)(n - (uint64_t)full * w); VkBufferImageCopy r[2]; uint32_t nr = 0;
        if (full) r[nr++] = (VkBufferImageCopy){o, w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, full, 1}};
        if (rem) r[nr++] = (VkBufferImageCopy){o + (VkDeviceSize)full * w * bpb, w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, (int32_t)full, 0}, {rem, 1, 1}};
        vkCmdCopyBufferToImage(cb, b, g_tb2[i].img, VK_IMAGE_LAYOUT_GENERAL, nr, r); g_tb_fresh_n++; g_tb_fresh_bytes += n * bpb; }
    pthread_mutex_unlock(&g_tb2_mu);
    pthread_mutex_lock(&g_tbc_mu);
    for (int i = 0; i < TBC_N; i++) if (g_tbc[i].dirty) {
        g_tbc[i].dirty = 0; VkBuffer b; VkDeviceSize o, sz = g_tbc[i].n * g_tbc[i].bpb;
        if (!g_tbc[i].buf || !vk_buf_at(g_tbc[i].addr, sz, &b, &o)) continue;
        if (!any++) vk_img_barrier_all(cb);
        VkBufferCopy c = {o, 0, sz}; vkCmdCopyBuffer(cb, b, g_tbc[i].buf, 1, &c); g_tb_fresh_n++; g_tb_fresh_bytes += sz; }
    pthread_mutex_unlock(&g_tbc_mu);
    if (any) { vk_img_barrier_all(cb);   // before the reads
        uint64_t c = g_tb_fresh_n; if (c <= 2 || !(c & (c - 1))) VKLOG("r150: texture buffer copies made again after a GPU write in the same part: %llu (%.1f MB in all)", (unsigned long long)c, g_tb_fresh_bytes / 1048576.0); }
}
