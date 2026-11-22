// Windows port (from the Android port's), Vulkan backend: presentation (see gfx_vk.h).
// With a window (front_win.c's, dm_front_window): the game's swap chain texture is scaled into a Vulkan swapchain image (aspect kept, black borders) and
// presented with FIFO (vsync) pacing. Headless (DM_HEADLESS=1): frames are counted, and DM_VK_SHOT=n1,n2,... writes those presented frames as PPM files to
// DM_VK_SHOT_DIR (default DM_ROOT), which is how a run without a screen is checked.
#include "gfx_vk.h"
#include <string.h>
#if defined(DM_NXN1X)
#include "nvnh_api.h"
#include "nvn_fg_input.h"
#endif
#include <math.h>
#include <time.h>
#include "overlay_layout.h"
#include "font5x7.h"
#include "overlay_vert.spv.h"
#include "overlay_frag.spv.h"

// the window (front_win.c): its HWND, its client area, a Vulkan surface for it (VK_KHR_win32_surface, made there with the Win32 headers)
void *dm_front_window(void);
void dm_front_size(void *w, uint32_t *width, uint32_t *height);
VkResult dm_front_surface(VkInstance inst, PFN_vkGetInstanceProcAddr gipa, void *w, VkSurfaceKHR *out);
PFN_vkGetInstanceProcAddr vk_gipa(void);   // gfx_vk_core.c
static uint64_t g_presents;
// Vulkan present IDs include overlay redraws; only the game's tickets have a time for the NVN frame statistics.
enum { PRES_TIMES = 64 };
static struct { uint64_t ticket, ns; } g_pres_t[PRES_TIMES];
static struct { uint32_t id; uint64_t ticket; } g_pres_ids[PRES_TIMES];
static pthread_mutex_t g_pres_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_pres_recorded;   // display times recorded so far (present_time_record): none after 240 presents with display timing on, the hand-over time stands in
static void present_time_begin(uint64_t ticket, uint64_t ns) {   // ns: the time to report now (no display times on this device), or 0 for the display's
    pthread_mutex_lock(&g_pres_mu);
    g_pres_t[ticket % PRES_TIMES] = (typeof(g_pres_t[0])){ticket, ns};
    pthread_mutex_unlock(&g_pres_mu);
}
static void present_time_map(uint32_t id, uint64_t ticket) {   // the present thread, under g_present_mu
    g_pres_ids[id % PRES_TIMES] = (typeof(g_pres_ids[0])){id, ticket};
}
static void present_time_record(uint32_t id, uint64_t ns) {   // the present thread, under g_present_mu
    typeof(g_pres_ids[0]) *m = &g_pres_ids[id % PRES_TIMES];
    if (m->id != id || !m->ticket || !ns) return;
    pthread_mutex_lock(&g_pres_mu);
    typeof(g_pres_t[0]) *p = &g_pres_t[m->ticket % PRES_TIMES];
    if (p->ticket == m->ticket) p->ns = ns;
    pthread_mutex_unlock(&g_pres_mu); g_pres_recorded++;
}
static void present_time_reset(void) {   // the old swapchain's pending results cannot describe the new one
    memset(g_pres_ids, 0, sizeof g_pres_ids);
    pthread_mutex_lock(&g_pres_mu); memset(g_pres_t, 0, sizeof g_pres_t); pthread_mutex_unlock(&g_pres_mu);
}
static double present_time_get(uint64_t ticket) {
    pthread_mutex_lock(&g_pres_mu);
    typeof(g_pres_t[0]) *p = &g_pres_t[ticket % PRES_TIMES];
    double t = ticket && p->ticket == ticket ? p->ns / 1e9 : 0;
    pthread_mutex_unlock(&g_pres_mu);
    return t;
}
static VkSurfaceKHR g_surf; static VkSwapchainKHR g_swap; static VkImage g_simg[8]; static uint32_t g_nsimg; static VkExtent2D g_sext; static VkFormat g_sfmt;
// The swapchain's format follows the window texture's sRGB-ness: a blit from an sRGB image into a UNORM one decodes to linear and stores that unencoded, which darkened the whole
// picture (Donkey Kong Country Returns HD presents RGBA8_SRGB, NVN format 0x38; issue #4). The Switch scans the window texture's bytes out as they are, and an SRGB -> SRGB blit
// decodes and encodes them again. UNORM games keep the UNORM swapchain. The overlay drawn after the blit is encoded too in an sRGB swapchain, so it shows a little lighter there
static VkFormat g_want = VK_FORMAT_R8G8B8A8_UNORM;
static VkSemaphore g_acq[8], g_done_sem[8]; static VkFence g_pfence[8]; static VkCommandPool g_ppool; static VkCommandBuffer g_pcb[8]; static uint32_t g_pi;
static void *g_win;
static VkImageView g_sview[8];
// Where the game's picture lies in the window (fractions: x, y, width, height; aspect kept, black borders around it), for the front end's touch points, which hle_touch.c
// takes relative to the picture: normalized by the whole window, a tap in Unpacking's 3040x1904 window (97 px borders) was off by up to 5% of the height
static _Atomic uint32_t g_pic[4] = {0, 0, 1u << 20, 1u << 20};
static void vk_pic_set(float x, float y, float w, float h) { atomic_store(&g_pic[0], (uint32_t)(x * (1 << 20))); atomic_store(&g_pic[1], (uint32_t)(y * (1 << 20))); atomic_store(&g_pic[2], (uint32_t)(w * (1 << 20))); atomic_store(&g_pic[3], (uint32_t)(h * (1 << 20))); }
void dm_picture_rect(float r[4]) { for (int i = 0; i < 4; i++) r[i] = (float)atomic_load(&g_pic[i]) / (1 << 20); }
__attribute__((weak)) int dm_overlay_visible(void) { return 0; }
__attribute__((weak)) float dm_overlay_scale(void) { return 1; }
__attribute__((weak)) void dm_overlay_knob(int stick, float *dx, float *dy) { (void)stick; *dx = *dy = 0; }
__attribute__((weak)) int64_t dm_overlay_pressed(int ctl, uint64_t bit) { (void)ctl; (void)bit; return 0; }
__attribute__((weak)) int dm_menu_state(void) { return -1; }
__attribute__((weak)) int dm_overlay_fps(void) { return 0; }   // front_android.c: the pause menu's Show FPS
__attribute__((weak)) int dm_overlay_custom(int layout, int ctl, float *x, float *y, float *s) { (void)layout; (void)ctl; (void)x; (void)y; *s = 1; return 0; }
__attribute__((weak)) int dm_overlay_custom_ver(void) { return 0; }
__attribute__((weak)) float dm_overlay_opacity(void) { return 0.6f; }
__attribute__((weak)) int dm_overlay_layout(void) { return 0; }
__attribute__((weak)) int dm_text_entry(char *title, int tcap, char *text, int xcap, int *sel, int *shift) { (void)title; (void)tcap; (void)text; (void)xcap; (void)sel; (void)shift; return 0; }
static void pace_reset(void); static void ov_fb_drop(void);
// r86: the performance overlay (perf_overlay_win.c, NXN1X's panel; F3 on / off), drawn over the picture after the touch overlay; front_win.c: the window's DPI scale
extern uint64_t g_gpu_ns; void perf_ov_gpu_ns(uint64_t ns);   // r102
int perf_ov_on(void); void perf_ov_driver(const char *s); void perf_ov_vram(int64_t used, int64_t budget); void perf_ov_hidden(void);
int perf_ov_frame(int64_t now_ns, uint64_t frames, uint32_t sw, uint32_t sh, float k, const uint8_t **px, int *w, int *h, int *x, int *y);
float dm_front_scale(void *w);
static void swap_destroy(void) {
    pace_reset();
    present_time_reset();
    if (g_swap) { vkDeviceWaitIdle(V.dev); ov_fb_drop(); for (uint32_t i = 0; i < g_nsimg; i++) if (g_sview[i]) { vkDestroyImageView(V.dev, g_sview[i], NULL); g_sview[i] = VK_NULL_HANDLE; }
                  vkDestroySwapchainKHR(V.dev, g_swap, NULL); g_swap = VK_NULL_HANDLE; }
    if (g_surf) { vkDestroySurfaceKHR(V.inst, g_surf, NULL); g_surf = VK_NULL_HANDLE; }
}
static int swap_make(void) {
    void *w = dm_front_window();
    if (w != g_win) { swap_destroy(); g_win = w; }
    if (!w) return 0;
    if (g_swap) return 1;
    if (dm_front_surface(V.inst, vk_gipa(), w, &g_surf)) return 0;
    { VkBool32 ok = 0; vkGetPhysicalDeviceSurfaceSupportKHR(V.pd, V.qf, g_surf, &ok); if (!ok) { VKRED_ONCE(96, "the queue family cannot present to the window"); vkDestroySurfaceKHR(V.inst, g_surf, NULL); g_surf = VK_NULL_HANDLE; return 0; } }
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(V.pd, g_surf, &caps);
    uint32_t nf = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(V.pd, g_surf, &nf, NULL); VkSurfaceFormatKHR sf[32]; if (nf > 32) nf = 32; vkGetPhysicalDeviceSurfaceFormatsKHR(V.pd, g_surf, &nf, sf);
    g_sfmt = sf[0].format; for (uint32_t i = 0; i < nf; i++) if (sf[i].format == VK_FORMAT_R8G8B8A8_UNORM) g_sfmt = sf[i].format;
    for (uint32_t i = 0; i < nf; i++) if (sf[i].format == g_want) g_sfmt = g_want;   // the window texture's sRGB-ness (g_want), so the blit copies its bytes as they are
    // r65: Turnip's WDDM surface offers B8G8R8A8_SRGB but not R8G8B8A8_SRGB (B8G8R8A8_UNORM, R8G8B8A8_UNORM, B8G8R8A8_SRGB), so an sRGB window texture (Hyrule Warriors: Age of
    // Imprisonment, NVN 0x38) went into the UNORM swapchain and its blit stored decoded linear values: the whole picture darker. The blit swaps the channels; only the sRGB-ness matters
    if (g_sfmt != g_want && g_want == VK_FORMAT_R8G8B8A8_SRGB) for (uint32_t i = 0; i < nf; i++) if (sf[i].format == VK_FORMAT_B8G8R8A8_SRGB) g_sfmt = VK_FORMAT_B8G8R8A8_SRGB;
    if ((g_sfmt == VK_FORMAT_R8G8B8A8_SRGB || g_sfmt == VK_FORMAT_B8G8R8A8_SRGB) != (g_want == VK_FORMAT_R8G8B8A8_SRGB))
        VKRED_ONCE(95, "the surface offers no format %d for the window texture: its picture is %s", g_want, g_want == VK_FORMAT_R8G8B8A8_SRGB ? "darker" : "lighter");
    g_sext = caps.currentExtent;   // Win32: the surface's extent is the client area's
    if (g_sext.width == 0xFFFFFFFFu || !g_sext.width || !g_sext.height) dm_front_size(w, &g_sext.width, &g_sext.height);
    if (!g_sext.width || !g_sext.height) { vkDestroySurfaceKHR(V.inst, g_surf, NULL); g_surf = VK_NULL_HANDLE; return 0; }   // minimized: nothing to present to
    uint32_t n = caps.minImageCount < 3 ? 3 : caps.minImageCount; if (caps.maxImageCount && n > caps.maxImageCount) n = caps.maxImageCount;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, NULL, 0, g_surf, n, g_sfmt, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, g_sext, 1,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) ? VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_PRESENT_MODE_FIFO_KHR, 1, VK_NULL_HANDLE};
    VkResult r = vkCreateSwapchainKHR(V.dev, &ci, NULL, &g_swap);
    if (r) { VKLOG("[RED] swapchain creation failed: %d", r); swap_destroy(); return 0; }
    g_nsimg = 8; vkGetSwapchainImagesKHR(V.dev, g_swap, &g_nsimg, g_simg);
    for (uint32_t i = 0; i < g_nsimg; i++) { VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_simg[i], VK_IMAGE_VIEW_TYPE_2D, g_sfmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(V.dev, &vi, NULL, &g_sview[i])); }
    if (!g_ppool) {
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf}; VKCHK(vkCreateCommandPool(V.dev, &pci, NULL, &g_ppool));
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_ppool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 8}; VKCHK(vkAllocateCommandBuffers(V.dev, &ai, g_pcb));
        for (int i = 0; i < 8; i++) {
            VkSemaphoreCreateInfo s = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VKCHK(vkCreateSemaphore(V.dev, &s, NULL, &g_acq[i])); VKCHK(vkCreateSemaphore(V.dev, &s, NULL, &g_done_sem[i]));
            VkFenceCreateInfo f = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT}; VKCHK(vkCreateFence(V.dev, &f, NULL, &g_pfence[i]));
        }
    }
    VKLOG("swapchain %ux%u format %d, %u images", g_sext.width, g_sext.height, g_sfmt, g_nsimg);
    return 1;
}
static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, from, to,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}
// ---- the on-screen gamepad: rasterized on the CPU once per window size (half resolution), blended over each frame ----
// The texture is the static layer (w x h), a strip of sprites under it (sh rows) and the pause menu's panel under that (pw x ph). The sprites: each stick's
// knob, drawn every frame where the finger holds it; the menu button, drawn on its own at OV_MENU_ALPHA whether the gamepad is shown or not; a white disc, the
// glow over a pressed round button (a wider one glows in its own shape); a black block, the dimming behind the menu. sp: each sprite in the strip (centre x, y,
// half width and height, texels; 0 = none). The panel is rasterized again whenever the menu's state (dm_menu_state) changes. Under the panel, the frame rate
// counter's characters (OV_FPS_CHARS, gh rows: one cell each), from which the counter is put together on every present, so a changing number needs no upload.
// With a theme in the APK (overlay_theme.c), the gamepad is its sprites instead: an atlas per layout in a texture of its own (g_tt), uploaded once, every
// control drawn from it on each present; the procedural layer is then only the menu's panel, the dimming and the counter.
enum { SP_DISC = OV_N, SP_BLACK, SP_WHITE, SP_ACCENT, SP_N };   // (black, white, accent: solid blocks, the dimming and the layout editor's frames)
#define OV_FPS_CHARS "0123456789 FPS-"
static struct { VkImage img; VkDeviceMemory mem; VkImageView view; VkSampler samp; VkDescriptorSetLayout dsl; VkPipelineLayout layout; VkDescriptorPool pool; VkDescriptorSet set;
                VkPipeline pipe; VkFormat pfmt; uint32_t w, h, sh, pw, ph, gh, gs; VkBuffer up; VkDeviceMemory upm; uint8_t *px; int uploaded, panel, lay, cver; float sp[SP_N][4];
                VkRenderPass rp; VkFramebuffer fb[8]; VkImageView fbview[8]; VkExtent2D fbext[8]; } g_ov;
static struct { const ov_theme *t; VkImage img; VkDeviceMemory mem; VkImageView view; VkDescriptorSet set; VkBuffer up; VkDeviceMemory upm; uint32_t upk; int ready; } g_tt[2];   // per layout
// Without dynamic rendering (a Vulkan 1.1 / 1.2 device, V.has_dynrender) the overlay draws in a render pass object over the swapchain image: one pass for the swapchain's
// format (load and store, the image already in COLOR_ATTACHMENT_OPTIMAL), and a framebuffer per swapchain image, made again when its view or the extent changes
static VkRenderPass ov_rp(void) {
    static VkFormat fmt; if (g_ov.rp && fmt == g_sfmt) return g_ov.rp;
    if (g_ov.rp) { vkDestroyRenderPass(V.dev, g_ov.rp, NULL); g_ov.rp = VK_NULL_HANDLE; }   // the swapchain's format changed (g_want); swap_destroy dropped its framebuffers
    fmt = g_sfmt;
    VkAttachmentDescription a = {0, g_sfmt, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                 VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference r = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp = {0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, NULL, 1, &r, NULL, NULL, 0, NULL};
    VkRenderPassCreateInfo ci = {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, NULL, 0, 1, &a, 1, &sp, 0, NULL};
    if (vkCreateRenderPass(V.dev, &ci, NULL, &g_ov.rp)) { VKRED_ONCE(93, "overlay render pass failed"); g_ov.rp = VK_NULL_HANDLE; }
    return g_ov.rp;
}
// every overlay framebuffer goes with the views or the pass it was made for: a view handle made again later may have the same value (ov_fb would take it for its own)
static void ov_fb_drop(void) { for (int i = 0; i < 8; i++) { if (g_ov.fb[i]) vkDestroyFramebuffer(V.dev, g_ov.fb[i], NULL); g_ov.fb[i] = VK_NULL_HANDLE; g_ov.fbview[i] = VK_NULL_HANDLE; g_ov.fbext[i] = (VkExtent2D){0, 0}; } }
static VkFramebuffer ov_fb(uint32_t idx) {
    if (idx >= 8 || !ov_rp()) return VK_NULL_HANDLE;
    if (g_ov.fb[idx] && g_ov.fbview[idx] == g_sview[idx] && g_ov.fbext[idx].width == g_sext.width && g_ov.fbext[idx].height == g_sext.height) return g_ov.fb[idx];
    if (g_ov.fb[idx]) vkDestroyFramebuffer(V.dev, g_ov.fb[idx], NULL);   // (swap_destroy drops them all with the views; here only an extent change)
    VkFramebufferCreateInfo ci = {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, NULL, 0, g_ov.rp, 1, &g_sview[idx], g_sext.width, g_sext.height, 1};
    if (vkCreateFramebuffer(V.dev, &ci, NULL, &g_ov.fb[idx])) { VKRED_ONCE(94, "overlay framebuffer failed"); g_ov.fb[idx] = VK_NULL_HANDLE; return VK_NULL_HANDLE; }
    g_ov.fbview[idx] = g_sview[idx]; g_ov.fbext[idx] = g_sext;
    return g_ov.fb[idx];
}
static _Atomic uint64_t g_fps_frames;   // the game's frames shown (mtl_present), which the counter counts
static int ov_round(const ov_ctl *c) { return c->w == c->h; }
static uint32_t ov_layout_sprites(uint32_t w, uint32_t h) {   // places the sprites, returns the strip's height
    float x = 4, top = (float)h, mr = 0, disc = 0, b[4];
    for (int i = 0; i < OV_N; i++) { ov_box(&OV_CTLS[i], (float)w, (float)h, b); if (OV_CTLS[i].kind == OV_BUTTON && ov_round(&OV_CTLS[i]) && b[2] > disc) disc = b[2]; }
    for (int i = 0; i < SP_N; i++) {
        const ov_ctl *c = i < OV_N ? &OV_CTLS[i] : NULL; float hw = 0, hh = 0;
        if (!c) hw = hh = i == SP_DISC ? disc : 6;
        else { ov_box(c, (float)w, (float)h, b);
            if (c->kind == OV_STICK_L || c->kind == OV_STICK_R) hw = hh = b[2] * OV_KNOB;
            else if (c->kind == OV_MENU || (c->kind == OV_BUTTON && !ov_round(c))) { hw = b[2]; hh = b[3]; } }   // a wider button's glow is its own shape
        g_ov.sp[i][2] = hw; g_ov.sp[i][3] = hh; if (!hw) continue; g_ov.sp[i][0] = x + hw + 2; x += 2 * hw + 8; if (hh > mr) mr = hh;
    }
    for (int i = 0; i < SP_N; i++) g_ov.sp[i][1] = top + mr + 4;
    return (uint32_t)(2 * mr + 8);
}
static void ov_plot(uint8_t *px, uint32_t w, uint32_t h, int x, int y, uint8_t a, uint8_t v) {
    if (x < 0 || y < 0 || (uint32_t)x >= w || (uint32_t)y >= h) return;
    uint8_t *p = px + 4 * ((size_t)y * w + (size_t)x); if (p[3] >= a) return;
    p[0] = p[1] = p[2] = v; p[3] = a;
}
static void ov_circle(uint8_t *px, uint32_t w, uint32_t h, float cx, float cy, float r, float ring) {
    for (int y = (int)(cy - r - 2); y <= (int)(cy + r + 2); y++) for (int x = (int)(cx - r - 2); x <= (int)(cx + r + 2); x++) {
        float d = sqrtf((x - cx) * (x - cx) + (y - cy) * (y - cy));
        if (d <= r && d >= r - ring) ov_plot(px, w, h, x, y, 150, 255);   // the ring
        else if (d < r - ring) ov_plot(px, w, h, x, y, 50, 30);           // a dark, see-through inside
    }
}
static void ov_box_ring(uint8_t *px, uint32_t w, uint32_t h, const ov_ctl *c, const float b[4], float ring) {   // ov_circle for any control's shape (b: ov_box)
    for (int y = (int)(b[1] - b[3] - 2); y <= (int)(b[1] + b[3] + 2); y++) for (int x = (int)(b[0] - b[2] - 2); x <= (int)(b[0] + b[2] + 2); x++) {
        float d = ov_sdf(c, b, x - b[0], y - b[1]);
        if (d <= 0 && d >= -ring) ov_plot(px, w, h, x, y, 150, 255);
        else if (d < -ring) ov_plot(px, w, h, x, y, 50, 30);
    }
}
static void ov_text(uint8_t *px, uint32_t w, uint32_t h, float cx, float cy, const char *s, int scale) {
    int n = (int)strlen(s), x0 = (int)(cx - n * 6 * scale / 2.0f), y0 = (int)(cy - 7 * scale / 2.0f);
    for (int i = 0; i < n; i++) {
        unsigned c = (unsigned char)s[i]; const unsigned char *g = FONT5X7[(c >= 32 && c < 127 ? c : 63) - 32];
        for (int col = 0; col < 5; col++) for (int row = 0; row < 7; row++) if (g[col] >> row & 1)
            for (int dy = 0; dy < scale; dy++) for (int dx = 0; dx < scale; dx++) ov_plot(px, w, h, x0 + i * 6 * scale + col * scale + dx, y0 + row * scale + dy, 220, 255);
    }
}
// colour drawing for the menu button and the panel: straight alpha, "over" blending, shapes antialiased by their distance
typedef struct { uint8_t *px; uint32_t stride, x0, y0, w, h; } ov_canvas;   // a w x h window of rows `stride` texels wide, at (x0, y0)
static void cv_blend(const ov_canvas *cv, int x, int y, const uint8_t c[4], float cov) {
    if (x < 0 || y < 0 || (uint32_t)x >= cv->w || (uint32_t)y >= cv->h || cov <= 0) return;
    uint8_t *p = cv->px + 4 * ((size_t)(cv->y0 + y) * cv->stride + cv->x0 + x);
    float a = c[3] / 255.0f * (cov > 1 ? 1 : cov), da = p[3] / 255.0f, oa = a + da * (1 - a);
    if (oa <= 0) return;
    for (int k = 0; k < 3; k++) p[k] = (uint8_t)((c[k] * a + p[k] * da * (1 - a)) / oa + 0.5f);
    p[3] = (uint8_t)(oa * 255 + 0.5f);
}
static void cv_rrect(const ov_canvas *cv, float x0, float y0, float x1, float y1, float rad, const uint8_t c[4]) {   // a filled rounded rectangle
    float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hx = (x1 - x0) / 2 - rad, hy = (y1 - y0) / 2 - rad;
    for (int y = (int)y0 - 1; y <= (int)y1 + 1; y++) for (int x = (int)x0 - 1; x <= (int)x1 + 1; x++) {
        float qx = fabsf(x + 0.5f - cx) - hx, qy = fabsf(y + 0.5f - cy) - hy, ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
        float d = sqrtf(ox * ox + oy * oy) + fminf(fmaxf(qx, qy), 0) - rad;
        cv_blend(cv, x, y, c, 0.5f - d);
    }
}
static void cv_disc(const ov_canvas *cv, float cx, float cy, float r, const uint8_t c[4]) { cv_rrect(cv, cx - r, cy - r, cx + r, cy + r, r, c); }
static void cv_box(const ov_canvas *cv, const ov_ctl *c, const float b[4], float inset, const uint8_t col[4]) {   // a control's shape filled (b: ov_box), inset texels in
    for (int y = (int)(b[1] - b[3]) - 1; y <= (int)(b[1] + b[3]) + 1; y++) for (int x = (int)(b[0] - b[2]) - 1; x <= (int)(b[0] + b[2]) + 1; x++)
        cv_blend(cv, x, y, col, 0.5f - (ov_sdf(c, b, x + 0.5f - b[0], y + 0.5f - b[1]) + inset));
}
static void cv_ring(const ov_canvas *cv, float cx, float cy, float r, float t, const uint8_t c[4]) {
    for (int y = (int)(cy - r) - 1; y <= (int)(cy + r) + 1; y++) for (int x = (int)(cx - r) - 1; x <= (int)(cx + r) + 1; x++) {
        float d = sqrtf((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
        cv_blend(cv, x, y, c, fminf(0.5f - (d - r), 0.5f + (d - (r - t))));
    }
}
static float cv_text_w(const char *s, int scale) { return (float)strlen(s) * 6 * scale - scale; }
static void cv_text(const ov_canvas *cv, float x, float cy, const char *s, int scale, const uint8_t c[4]) {   // x: the left edge
    int x0 = (int)(x + 0.5f), y0 = (int)(cy - 7 * scale / 2.0f + 0.5f);
    for (int i = 0; s[i]; i++) {
        unsigned ch = (unsigned char)s[i]; const unsigned char *g = FONT5X7[(ch >= 32 && ch < 127 ? ch : 63) - 32];
        for (int col = 0; col < 5; col++) for (int row = 0; row < 7; row++) if (g[col] >> row & 1)
            for (int dy = 0; dy < scale; dy++) for (int dx = 0; dx < scale; dx++) cv_blend(cv, x0 + i * 6 * scale + col * scale + dx, y0 + row * scale + dy, c, 1);
    }
}
static const uint8_t C_WHITE[4] = {255, 255, 255, 255}, C_ACCENT[4] = {255, 60, 142, 255};
static void ov_menu_icon(const ov_canvas *cv, const ov_ctl *c, const float b[4]) {   // the menu button: a dark pill with a ring and three bars
    static const uint8_t fill[4] = {20, 20, 26, 170}, ring[4] = {255, 255, 255, 210};
    float r = b[3], t = fmaxf(2, r * 0.07f);
    for (int y = (int)(b[1] - b[3]) - 1; y <= (int)(b[1] + b[3]) + 1; y++) for (int x = (int)(b[0] - b[2]) - 1; x <= (int)(b[0] + b[2]) + 1; x++) {
        float d = ov_sdf(c, b, x + 0.5f - b[0], y + 0.5f - b[1]);
        cv_blend(cv, x, y, fill, 0.5f - d); cv_blend(cv, x, y, ring, fminf(0.5f - d, 0.5f + d + t));
    }
    float bw = r * 0.50f, bt = fmaxf(2, r * 0.11f), gap = r * 0.30f, cx = b[0], cy = b[1];
    for (int k = -1; k <= 1; k++) cv_rrect(cv, cx - bw, cy + k * gap - bt / 2, cx + bw, cy + k * gap + bt / 2, bt / 2, C_WHITE);
}
// the text entry (front_android.c front_text_input), in the pause menu's panel: the game's title for it, the field with the text and a caret, the keyboard
// (overlay_layout.h ov_kb_*: the same rectangles the touches are tested against) with the highlighted key, and the controller's keys
static void ov_text_panel(const ov_canvas *cv) {
    char title[128], text[256]; int sel, shift; if (!dm_text_entry(title, sizeof title, text, sizeof text, &sel, &shift)) return;
    float W = (float)g_ov.pw, H = (float)g_ov.ph, p[4] = {0, 0, W, H}, pad = W * 0.03f;
    static const uint8_t bg[4] = {14, 14, 20, 236}, field[4] = {255, 255, 255, 30}, key[4] = {255, 255, 255, 26}, dim[4] = {200, 200, 210, 255};
    cv_rrect(cv, 0, 0, W, H, H * 0.03f, bg);
    int ts = (int)fmaxf(1, roundf(H * 0.045f / 7)), xs = (int)fmaxf(1, roundf(H * 0.06f / 7)), ks = (int)fmaxf(1, roundf(H * 0.05f / 7)), hs = (int)fmaxf(1, roundf(H * 0.036f / 7));
    float room = W - 4 * pad;
    while (title[0] && cv_text_w(title, ts) > room) title[strlen(title) - 1] = 0;   // a long title is cut
    cv_text(cv, (W - cv_text_w(title, ts)) / 2, H * 0.07f, title, ts, C_WHITE);
    cv_rrect(cv, pad, H * 0.14f, W - pad, H * 0.27f, H * 0.02f, field);
    char shown[260]; snprintf(shown, sizeof shown, "%s_", text);   // the caret
    const char *s = shown; while (*s && cv_text_w(s, xs) > room) s++;   // the end of a long text
    cv_text(cv, 2 * pad, H * 0.205f, s, xs, C_WHITE);
    static const char *wide[4] = {"Shift", "Space", "Del", "OK"};
    for (int i = 0; i < OVK_N; i++) {
        float r[4]; ov_kb_key(i, p, r);
        int on = i == sel || (i == OVK_SHIFT && shift);
        cv_rrect(cv, r[0], r[1], r[0] + r[2], r[1] + r[3], r[3] * 0.18f, i == sel ? C_ACCENT : on ? field : key);
        char one[2] = {(char)ov_kb_char(i, shift), 0}; const char *l = i < 40 ? one : wide[i - 40];
        int ls = ks; while (ls > 1 && cv_text_w(l, ls) > r[2] * 0.85f) ls--;   // a wide key's word fits inside it
        cv_text(cv, r[0] + (r[2] - cv_text_w(l, ls)) / 2, r[1] + r[3] / 2, l, ls, i == OVK_OK && i != sel ? C_ACCENT : C_WHITE);
    }
    const char *k = "A key  B delete  Y shift  + OK  - cancel";   // one line, centred, smaller until it fits
    while (hs > 1 && cv_text_w(k, hs) > W - 2 * pad) hs--;
    cv_text(cv, (W - cv_text_w(k, hs)) / 2, H * 0.95f, k, hs, dim);
}
static const uint8_t C_BG[4] = {14, 14, 20, 236}, C_LINE[4] = {255, 255, 255, 40}, C_HI[4] = {255, 255, 255, 34}, C_DIM[4] = {200, 200, 210, 255}, C_OFF[4] = {80, 80, 92, 255},
                     C_KNOB_OFF[4] = {215, 215, 222, 255};
static void cv_button(const ov_canvas *cv, const float b[4], const char *l, int fill, int held) {   // a rounded button with its word, as large as fits
    static const uint8_t face[4] = {255, 255, 255, 30}, rim[4] = {255, 255, 255, 90};
    float rad = b[3] * 0.28f; cv_rrect(cv, b[0], b[1], b[0] + b[2], b[1] + b[3], rad, held ? C_WHITE : rim);
    cv_rrect(cv, b[0] + 2, b[1] + 2, b[0] + b[2] - 2, b[1] + b[3] - 2, rad - 2, held ? C_ACCENT : fill ? C_ACCENT : face);
    int s = (int)fmaxf(1, roundf(b[3] * 0.40f / 7)); while (s > 1 && cv_text_w(l, s) > b[2] * 0.82f) s--;
    cv_text(cv, b[0] + (b[2] - cv_text_w(l, s)) / 2, b[1] + b[3] / 2, l, s, C_WHITE);
}
// The pause menu (overlay_layout.h: the rows, the tabs, the slider's track), in the raster's panel rectangle. Its letters are as large as a row allows (0.30 of
// its height, about 2.7 mm on a 6.7" phone and 3.6 mm on the 8.8" tablet) and smaller only when the widest row would not fit beside its control
static const char *const OVM_LABELS[OVM_N] = {"Show gamepad", "Opacity", "Edit layout", "Rumble", "Show FPS", "Mode", "Resume", ""};
static const char *const OVM_MODES[2] = {"Handheld", "Docked"};
static void ov_menu_panel(const ov_canvas *cv, int st) {
    float p[4] = {0, 0, (float)cv->w, (float)cv->h}, r[4]; ov_menu_rowrect(p, 0, r);
    float rh = r[3], rw = r[2], gap = rh * 0.35f, sh = rh * 0.40f, sw = sh * 1.85f;
    int tab = st >> 8 & 1, sel = (st >> 4 & 15) - 1, op = st >> 9 & 127;
    int is = (int)fmaxf(1, roundf(rh * 0.30f / 7));
    for (; is > 1; is--) {   // the widest row: its label, a gap and its control
        float need = 0;
        for (int i = 0; i < OVM_RESUME; i++) {
            float lw = cv_text_w(OVM_LABELS[i], is), cw = i == OVM_MODE ? 2 * (cv_text_w(OVM_MODES[0], is) + sh) : i == OVM_EDIT ? cv_text_w("Edit", is) + 2 * sh : sw + gap + cv_text_w("OFF", is);
            need = fmaxf(need, i == OVM_OPACITY ? fmaxf(lw / 0.38f, cv_text_w("80%", is) / 0.14f) : lw + gap + cw + rh * 0.3f);
        }
        if (need <= rw && cv_text_w(OVM_TAB_NAMES[0], is) < rw / 2 - sh) break;
    }
    cv_rrect(cv, 0, 0, p[2], p[3], rh * 0.18f, C_BG);
    for (int row = 0; row < OVM_ROWS; row++) {
        int it = ov_menu_item(tab, row); ov_menu_rowrect(p, row, r); float cy = r[1] + rh / 2, x1 = r[0] + rw - rh * 0.15f;
        if (it < 0) continue;
        if (row == sel) cv_rrect(cv, r[0], r[1] + rh * 0.08f, r[0] + rw, r[1] + rh * 0.92f, rh * 0.14f, C_HI);
        if (it == OVM_TAB) {   // the tabs: a segmented bar, the open one filled
            float th = rh * 0.62f, tw = rw / OVM_TABS;
            cv_rrect(cv, r[0], cy - th / 2, r[0] + rw, cy + th / 2, th / 2, C_OFF);
            cv_rrect(cv, r[0] + tab * tw, cy - th / 2, r[0] + (tab + 1) * tw, cy + th / 2, th / 2, C_ACCENT);
            for (int k = 0; k < OVM_TABS; k++) cv_text(cv, r[0] + k * tw + (tw - cv_text_w(OVM_TAB_NAMES[k], is)) / 2, cy, OVM_TAB_NAMES[k], is, k == tab ? C_WHITE : C_DIM);
            cv_rrect(cv, r[0], r[1] + rh - 1, r[0] + rw, r[1] + rh, 0.5f, C_LINE);
            continue;
        }
        if (it == OVM_RESUME) { cv_rrect(cv, r[0], r[1], r[0] + rw, r[1] + 1, 0.5f, C_LINE); cv_text(cv, r[0] + (rw - cv_text_w("Resume", is)) / 2, cy, "Resume", is, C_ACCENT); continue; }
        cv_text(cv, r[0] + rh * 0.15f, cy, OVM_LABELS[it], is, row == sel ? C_WHITE : C_DIM);
        if (it == OVM_OPACITY) {   // the slider: its track, filled to the value, the knob and the percent
            float t[3]; ov_menu_track(p, row, t); float f = (float)(op - OV_OPACITY_MIN) / (OV_OPACITY_MAX - OV_OPACITY_MIN), th = fmaxf(2, rh * 0.08f), kr = rh * 0.20f;
            f = f < 0 ? 0 : f > 1 ? 1 : f;
            cv_rrect(cv, t[0], t[1] - th / 2, t[0] + t[2], t[1] + th / 2, th / 2, C_OFF);
            cv_rrect(cv, t[0], t[1] - th / 2, t[0] + f * t[2], t[1] + th / 2, th / 2, C_ACCENT);
            cv_disc(cv, t[0] + f * t[2], t[1], kr, C_WHITE);
            char v[8]; snprintf(v, sizeof v, "%d%%", op); cv_text(cv, x1 - cv_text_w(v, is), cy, v, is, C_WHITE);
        } else if (it == OVM_EDIT) {   // a button
            float bw = cv_text_w("Edit", is) + 2 * sh, b[4] = {x1 - bw, cy - rh * 0.30f, bw, rh * 0.60f}; cv_button(cv, b, "Edit", 0, 0);
        } else if (it == OVM_MODE) {   // two segments, the current one filled: handheld (performance) | docked (resolution)
            int docked = st >> 2 & 1; float hh = rh * 0.46f, segw = fmaxf(cv_text_w(OVM_MODES[0], is), cv_text_w(OVM_MODES[1], is)) + sh, x0 = x1 - 2 * segw;
            cv_rrect(cv, x0, cy - hh / 2, x1, cy + hh / 2, hh / 2, C_OFF);
            cv_rrect(cv, x0 + docked * segw, cy - hh / 2, x0 + (docked + 1) * segw, cy + hh / 2, hh / 2, C_ACCENT);
            for (int k = 0; k < 2; k++) cv_text(cv, x0 + k * segw + (segw - cv_text_w(OVM_MODES[k], is)) / 2, cy, OVM_MODES[k], is, k == docked ? C_WHITE : C_DIM);
        } else {   // a switch
            int on = st >> (it == OVM_OVERLAY ? 0 : it == OVM_RUMBLE ? 1 : 3) & 1; float sx0 = x1 - sw;
            cv_rrect(cv, sx0, cy - sh / 2, x1, cy + sh / 2, sh / 2, on ? C_ACCENT : C_OFF);
            cv_disc(cv, on ? x1 - sh / 2 : sx0 + sh / 2, cy, sh / 2 - fmaxf(2, sh * 0.1f), on ? C_WHITE : C_KNOB_OFF);
            const char *word = on ? "ON" : "OFF"; cv_text(cv, sx0 - gap - cv_text_w(word, is), cy, word, is, on ? C_WHITE : C_DIM);
        }
    }
}
// The layout editor's toolbar and prompt (overlay_layout.h ov_ed_*), at half the window's scale in the raster's panel rectangle: the toolbar at its top, the
// prompt under it (gfx_vk_present.c ov_render draws each where overlay_layout.h puts it)
static void ov_ed_rects(float tb[4], float pr[4]) {   // their places in the raster
    float W = (float)g_sext.width, H = (float)g_sext.height, a[4], b[4]; ov_ed_rect(W, H, a); ov_ed_prompt_rect(W, H, b);
    tb[0] = 0; tb[1] = 0; tb[2] = fminf(a[2] / 2, (float)g_ov.pw); tb[3] = a[3] / 2;
    pr[0] = 0; pr[1] = tb[3] + 2; pr[2] = fminf(b[2] / 2, (float)g_ov.pw); pr[3] = fminf(b[3] / 2, (float)g_ov.ph - pr[1]);
}
static const char *ov_ctl_name(const ov_ctl *c) {
    return c->kind == OV_STICK_L ? "Left stick" : c->kind == OV_STICK_R ? "Right stick" : c->kind == OV_DPAD ? "D-pad" : c->kind == OV_MENU ? "Menu button"
         : !strcmp(c->label, "-") ? "Minus" : !strcmp(c->label, "+") ? "Plus" : c->label;
}
static void ov_edit_panel(const ov_canvas *cv, int st) {
    float tb[4], pr[4]; ov_ed_rects(tb, pr);
    int held = (st >> 4 & 15) - 1, sel = (st >> 8 & 255) - 1;
    static const char *const words[OVE_ALL] = {"-", "+", "Reset", "Cancel", "Save", "Discard", "Keep editing"};
    cv_rrect(cv, tb[0], tb[1], tb[0] + tb[2], tb[1] + tb[3], tb[3] * 0.14f, C_BG);
    char hint[96];
    if (sel >= 0 && sel < OV_N) snprintf(hint, sizeof hint, "%s: drag, pinch, - +", ov_ctl_name(&OV_CTLS[sel]));   // (short: 2x-scale letters on a phone)
    else snprintf(hint, sizeof hint, "Drag to move, pinch to resize");
    int hs = (int)fmaxf(1, roundf(tb[3] * 0.17f / 7)); while (hs > 1 && cv_text_w(hint, hs) > tb[2] * 0.94f) hs--;
    cv_text(cv, tb[0] + (tb[2] - cv_text_w(hint, hs)) / 2, tb[1] + tb[3] * 0.21f, hint, hs, sel >= 0 ? C_WHITE : C_DIM);
    for (int i = 0; i < OVE_N; i++) { float b[4]; ov_ed_button(i, tb, b); cv_button(cv, b, words[i], i == OVE_SAVE, i == held); }
    if (!(st & 1)) return;
    cv_rrect(cv, pr[0], pr[1], pr[0] + pr[2], pr[1] + pr[3], pr[3] * 0.10f, C_BG);
    const char *q = "Discard your changes?"; int qs = (int)fmaxf(1, roundf(pr[3] * 0.13f / 7)); while (qs > 1 && cv_text_w(q, qs) > pr[2] * 0.9f) qs--;
    cv_text(cv, pr[0] + (pr[2] - cv_text_w(q, qs)) / 2, pr[1] + pr[3] * 0.27f, q, qs, C_WHITE);
    for (int i = OVE_DISCARD; i <= OVE_KEEP; i++) { float b[4]; ov_ed_button(i, pr, b); cv_button(cv, b, words[i], i == OVE_DISCARD, i == held); }
}
static void ov_panel(int st) {   // the pause menu, the text entry or the layout editor, for menu state st (dm_menu_state), into the panel's rows of the upload buffer
    ov_canvas cv = {g_ov.px, g_ov.w, 0, g_ov.h + g_ov.sh, g_ov.pw, g_ov.ph};
    for (uint32_t y = 0; y < g_ov.ph; y++) memset(g_ov.px + 4 * ((size_t)(cv.y0 + y) * g_ov.w), 0, (size_t)g_ov.pw * 4);
    if (st >> 16 & 1) ov_text_panel(&cv);   // a text entry is open (dm_menu_state)
    else if (st & OVE_ST) ov_edit_panel(&cv, st);
    else ov_menu_panel(&cv, st);
}
static void ov_raster(uint8_t *px, uint32_t w, uint32_t h, uint32_t sh) {
    memset(px, 0, (size_t)w * (h + sh) * 4);
    int ts = h > 600 ? 3 : 2;
    for (int i = 0; i < OV_N && g_tt[g_ov.lay].ready != 1; i++) {   // (with a theme the controls are its sprites: the static layer stays empty)
        const ov_ctl *c = &OV_CTLS[i]; float b[4]; ov_box(c, (float)w, (float)h, b);
        if (c->kind == OV_MENU) continue;   // a sprite only
        if (c->kind == OV_DPAD) {
            for (int k = 0; k < 4; k++) { float ox, oy, dr; ov_dpad_dir(k, &ox, &oy, &dr); ov_circle(px, w, h, b[0] + ox * b[2], b[1] + oy * b[2], dr * b[2], 2.5f); }
            continue;
        }
        if (ov_round(c)) ov_circle(px, w, h, b[0], b[1], b[2], c->kind == OV_BUTTON ? 2.5f : 3.0f);
        else ov_box_ring(px, w, h, c, b, 2.5f);
        int ls = ts; while (ls > 1 && 7 * ls > fminf(b[2], b[3])) ls--;   // smaller on a small button (-, +)
        if (c->label[0]) ov_text(px, w, h, b[0], b[1], c->label, ls);
    }
    ov_canvas strip = {px, w, 0, 0, w, h + sh};
    for (int i = 0; i < SP_N; i++) {   // the sprites (the knobs are drawn only as sprites, where the finger has them)
        float x = g_ov.sp[i][0], y = g_ov.sp[i][1], r = g_ov.sp[i][2], b[4] = {x, y, r, g_ov.sp[i][3]}; if (!r) continue;
        if (i == SP_DISC) cv_disc(&strip, x, y, r, C_WHITE);
        else if (i == SP_BLACK || i == SP_WHITE || i == SP_ACCENT) { static const uint8_t k[4] = {0, 0, 0, 255}; cv_rrect(&strip, x - r, y - r, x + r, y + r, 0, i == SP_BLACK ? k : i == SP_WHITE ? C_WHITE : C_ACCENT); }
        else if (OV_CTLS[i].kind == OV_MENU) ov_menu_icon(&strip, &OV_CTLS[i], b);
        else if (OV_CTLS[i].kind == OV_BUTTON) cv_box(&strip, &OV_CTLS[i], b, fminf(b[2], b[3]) * 0.07f, C_WHITE);   // inset as the disc's 0.93
        else ov_circle(px, w, h + sh, x, y, r, 2.5f);
    }
}
typedef struct { float pos[4], uv[4], alpha; } ov_quad;
static void ov_draw(VkCommandBuffer cb, const ov_quad *q) { vkCmdPushConstants(cb, g_ov.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof *q, q); vkCmdDraw(cb, 6, 1, 0, 0); }
static void ov_rect(VkCommandBuffer cb, float x0, float y0, float x1, float y1, float u0, float v0, float u1, float v1, float alpha) {   // screen pixels, texels
    float W = (float)g_sext.width, H = (float)g_sext.height, tw = (float)g_ov.w, th = (float)(g_ov.h + g_ov.sh + g_ov.ph + g_ov.gh);
    ov_quad q = {{x0 / W * 2 - 1, y0 / H * 2 - 1, x1 / W * 2 - 1, y1 / H * 2 - 1}, {u0 / tw, v0 / th, u1 / tw, v1 / th}, alpha}; ov_draw(cb, &q);
}
static void ov_sprite(VkCommandBuffer cb, int i, float sx, float sy, float r, float alpha) {   // sprite i centred on screen pixel (sx, sy), r screen pixels in half width
    float k = r / g_ov.sp[i][2], hx = g_ov.sp[i][2] + 3, hy = g_ov.sp[i][3] + 3;
    ov_rect(cb, sx - hx * k, sy - hy * k, sx + hx * k, sy + hy * k, g_ov.sp[i][0] - hx, g_ov.sp[i][1] - hy, g_ov.sp[i][0] + hx, g_ov.sp[i][1] + hy, alpha);
}
// The theme's sprites (overlay_layout.h ov_tsprite): the design's CSS pixels scaled by the window's height over the design's (and dm_overlay_scale), each at its
// offset from its control's centre. Decorations first, then each control in the table's order: a held button, d-pad direction or stick in its held sprite,
// a stick's knob where the finger has it (as far as the design lets it travel); the menu button last (ov_theme_menu), over everything
static void ov_tsp(VkCommandBuffer cb, const ov_theme *t, const ov_tsprite *s, float cx, float cy, float k, float alpha) {
    float W = (float)g_sext.width, H = (float)g_sext.height, x0 = cx + s->ox * k, y0 = cy + s->oy * k, x1 = x0 + s->w / t->scale * k, y1 = y0 + s->h / t->scale * k;
    ov_quad q = {{x0 / W * 2 - 1, y0 / H * 2 - 1, x1 / W * 2 - 1, y1 / H * 2 - 1}, {s->u / t->w, s->v / t->h, (s->u + s->w) / t->w, (s->v + s->h) / t->h}, alpha}; ov_draw(cb, &q);
}
static float ov_glow(int64_t t, int64_t now) {   // a held button's glow: bright at the press, fading to a steady one
    if (!t) return 0;
    float f = (float)(now - t) / OV_FLASH_NS; return f >= 1 ? 0.22f : 0.22f + 0.48f * (1 - f) * (1 - f);
}
// The frame rate counter: the game's frames shown per second, measured over half a second, in the window's top left corner on a dark block
static float ov_fps_cell(void) { return 6.0f * g_ov.gs + 2; }   // a character cell's width in texels (a glyph of 5 columns, a gap, a clear texel each side)
static void ov_fps_glyphs(void) {   // the characters into the rows under the panel
    ov_canvas cv = {g_ov.px, g_ov.w, 0, g_ov.h + g_ov.sh + g_ov.ph, g_ov.w, g_ov.gh};
    for (int i = 0; OV_FPS_CHARS[i]; i++) { char one[2] = {OV_FPS_CHARS[i], 0}; cv_text(&cv, i * ov_fps_cell() + 1 + g_ov.gs * 0.5f, g_ov.gh / 2.0f, one, (int)g_ov.gs, C_WHITE); }
}
static void ov_fps(VkCommandBuffer cb, int64_t now) {
    static uint64_t n0; static int64_t t0; static int fps = -1;
    uint64_t n = atomic_load(&g_fps_frames);
    if (!t0 || now - t0 > 2000000000ll) { t0 = now; n0 = n; fps = -1; }   // the first frame, or after a pause in the presents (the menu, the background): measure anew
    else if (now - t0 >= 500000000ll) { fps = (int)((double)(n - n0) * 1e9 / (double)(now - t0) + 0.5); t0 = now; n0 = n; }
    char s[16]; if (fps < 0) snprintf(s, sizeof s, "-- FPS"); else snprintf(s, sizeof s, "%d FPS", fps > 999 ? 999 : fps);
    float k = (float)g_sext.width / g_ov.w, cw = ov_fps_cell() * k, ch = (float)g_ov.gh * k, m = (float)g_sext.height * 0.02f, x = m + ch * 0.25f, pad = ch * 0.2f;
    float bx = g_ov.sp[SP_BLACK][0], by = g_ov.sp[SP_BLACK][1], gy = (float)(g_ov.h + g_ov.sh + g_ov.ph);
    ov_rect(cb, m, m, m + ch * 0.5f + strlen(s) * cw, m + ch + 2 * pad, bx - 2, by - 2, bx + 2, by + 2, 0.45f);
    for (int i = 0; s[i]; i++, x += cw) {
        const char *c = strchr(OV_FPS_CHARS, s[i]); if (!c || s[i] == ' ') continue;
        float u = (float)(c - OV_FPS_CHARS) * ov_fps_cell(); ov_rect(cb, x, m + pad, x + cw, m + pad + ch, u, gy, u + ov_fps_cell(), gy + g_ov.gh, 1);
    }
}
static void ov_theme_draw(VkCommandBuffer cb, const ov_theme *t, float alpha) {   // (alpha: the player's opacity; each control at its own size factor)
    float W = (float)g_sext.width, H = (float)g_sext.height, k = H / t->dh * dm_overlay_scale(), b[4]; const ov_ctl *L = OV_LAYOUTS[t->layout];
    for (int j = 0; j < t->n; j++) if (t->sp[j].state == OVT_DECOR) { const ov_ctl *c = &L[t->sp[j].ctl]; ov_box(c, W, H, b); ov_tsp(cb, t, &t->sp[j], b[0], b[1], k * ov_cscale(c), alpha); }
    for (int i = 0; i < OV_N; i++) {
        const ov_ctl *c = &L[i]; if (c->kind == OV_MENU) continue;
        ov_box(c, W, H, b); float kc = k * ov_cscale(c); int stick = c->kind == OV_STICK_L || c->kind == OV_STICK_R, held = c->kind != OV_DPAD && dm_overlay_pressed(i, 0) != 0;
        float kx = 0, ky = 0; if (stick) { dm_overlay_knob(c->kind == OV_STICK_R, &kx, &ky); kx *= t->travel * kc; ky *= t->travel * kc; }
        for (int j = 0; j < t->n; j++) {
            const ov_tsprite *s = &t->sp[j]; if (s->ctl != i || s->state == OVT_DECOR) continue;
            if (s->state == OVT_KNOB) { ov_tsp(cb, t, s, b[0] + kx, b[1] + ky, kc, alpha); continue; }
            int on = c->kind == OV_DPAD ? s->dir >= 0 && dm_overlay_pressed(i, OV_DPAD_BITS[s->dir]) != 0 : held;
            if (s->state == (on ? OVT_HELD : OVT_IDLE)) ov_tsp(cb, t, s, b[0], b[1], kc, alpha);
        }
    }
}
static void ov_theme_menu(VkCommandBuffer cb, const ov_theme *t, int i, float alpha) {
    const ov_ctl *c = &OV_LAYOUTS[t->layout][i]; float W = (float)g_sext.width, H = (float)g_sext.height, b[4]; ov_box(c, W, H, b);
    int held = dm_overlay_pressed(i, 0) != 0;
    for (int j = 0; j < t->n; j++) if (t->sp[j].ctl == i && t->sp[j].state == (held ? OVT_HELD : OVT_IDLE)) ov_tsp(cb, t, &t->sp[j], b[0], b[1], H / t->dh * dm_overlay_scale() * ov_cscale(c), alpha);
}
static void ov_block(VkCommandBuffer cb, int sp, float x0, float y0, float x1, float y1, float alpha) {   // a solid rectangle in a block sprite's colour
    float bx = g_ov.sp[sp][0], by = g_ov.sp[sp][1]; ov_rect(cb, x0, y0, x1, y1, bx - 2, by - 2, bx + 2, by + 2, alpha);
}
static void ov_frame(VkCommandBuffer cb, int sp, const float b[4], float t, float alpha) {   // a rectangle's outline around a control's box (b: ov_box), t pixels thick
    float x0 = b[0] - b[2] - t, x1 = b[0] + b[2] + t, y0 = b[1] - b[3] - t, y1 = b[1] + b[3] + t;
    ov_block(cb, sp, x0, y0, x1, y0 + t, alpha); ov_block(cb, sp, x0, y1 - t, x1, y1, alpha); ov_block(cb, sp, x0, y0 + t, x0 + t, y1 - t, alpha); ov_block(cb, sp, x1 - t, y0 + t, x1, y1 - t, alpha);
}
static void ov_render(VkCommandBuffer cb, int mode, int menu, int fps) {
    float W = (float)g_sext.width, H = (float)g_sext.height, op = dm_overlay_opacity();
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); int64_t now = (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
    int edit = menu >= 0 && !(menu >> 16 & 1) && (menu & OVE_ST);   // the layout editor: the gamepad over the held frame, every control (shown or not)
    if (edit) mode = 1;
    float bx = g_ov.sp[SP_BLACK][0], by = g_ov.sp[SP_BLACK][1];
    if (menu >= 0 && !edit) {   // paused: the frame dimmed, the panel, and the menu button (which closes it)
        ov_rect(cb, 0, 0, W, H, bx - 2, by - 2, bx + 2, by + 2, 0.55f);
        float r[4]; ov_menu_rect(W, H, r);
        ov_rect(cb, r[0], r[1], r[0] + r[2], r[1] + r[3], 0, (float)(g_ov.h + g_ov.sh), (float)g_ov.pw, (float)(g_ov.h + g_ov.sh + g_ov.ph), 1);
    } else if (edit) ov_rect(cb, 0, 0, W, H, bx - 2, by - 2, bx + 2, by + 2, 0.30f);
    const ov_theme *th = g_tt[g_ov.lay].ready == 1 ? g_tt[g_ov.lay].t : NULL; int tbound = 0;   // (the theme's texture bound)
    if ((menu < 0 || edit) && mode == 1 && th) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_tt[g_ov.lay].set, 0, NULL); tbound = 1;
        ov_theme_draw(cb, th, op);
    } else if ((menu < 0 || edit) && mode == 1) {   // without a theme: the outlines (their opacity: 60% looks as they always did), the knobs and the glows
        float fo = fminf(1, op / 0.6f);
        ov_rect(cb, 0, 0, W, H, 0, 0, (float)g_ov.w, (float)g_ov.h, fo);
        for (int i = 0; i < OV_N; i++) { const ov_ctl *c = &OV_CTLS[i]; float b[4]; ov_box(c, W, H, b);
            if (c->kind == OV_STICK_L || c->kind == OV_STICK_R) {
                float dx, dy; dm_overlay_knob(c->kind == OV_STICK_R, &dx, &dy); float travel = b[2] * (1 - OV_KNOB);   // the knob's edge reaches the ring
                ov_sprite(cb, i, b[0] + dx * travel, b[1] + dy * travel, b[2] * OV_KNOB, fo);
            } else if (c->kind == OV_BUTTON) { float g = ov_glow(dm_overlay_pressed(i, 0), now); if (g <= 0) continue;
                if (!ov_round(c)) ov_sprite(cb, i, b[0], b[1], b[2], g); else ov_sprite(cb, SP_DISC, b[0], b[1], b[2] * 0.93f, g); }
            else if (c->kind == OV_DPAD) for (int k = 0; k < 4; k++) {
                float g = ov_glow(dm_overlay_pressed(i, OV_DPAD_BITS[k]), now), ox, oy, dr; if (g <= 0) continue;
                ov_dpad_dir(k, &ox, &oy, &dr); ov_sprite(cb, SP_DISC, b[0] + ox * b[2], b[1] + oy * b[2], dr * b[2] * 0.93f, g);
            }
        }
    }
    for (int i = 0; i < OV_N; i++) if (OV_CTLS[i].kind == OV_MENU && mode) {   // the menu button, on top of the controls, where the player put it
        float ma = mode == 1 && (menu < 0 || edit) ? fmaxf(op, 0.5f) : OV_MENU_ALPHA;   // (findable at any opacity)
        if (th) { if (!tbound) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_tt[g_ov.lay].set, 0, NULL); tbound = 1; }
                  ov_theme_menu(cb, th, i, ma); continue; }
        const ov_ctl *c = &OV_CTLS[i]; float b[4], g = ov_glow(dm_overlay_pressed(i, 0), now); ov_box(c, W, H, b);
        ov_sprite(cb, i, b[0], b[1], b[2], ma);
        if (g > 0) ov_sprite(cb, i, b[0], b[1], b[2], g);
    }
    if (tbound) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_ov.set, 0, NULL);   // the rest is the procedural texture's
    if (edit) {   // every control's box faintly, the chosen one in the accent colour; the toolbar; the prompt over a darker frame
        int sel = (menu >> 8 & 255) - 1; float t = fmaxf(2, H * 0.003f);
        for (int i = 0; i < OV_N; i++) { float b[4]; ov_box(&OV_CTLS[i], W, H, b); if (i == sel) ov_frame(cb, SP_ACCENT, b, 2 * t, 1); else ov_frame(cb, SP_WHITE, b, t, 0.35f); }
        float tb[4], pr[4], a[4], p[4], y0 = (float)(g_ov.h + g_ov.sh); ov_ed_rects(tb, pr); ov_ed_rect(W, H, a); ov_ed_prompt_rect(W, H, p);
        ov_rect(cb, a[0], a[1], a[0] + a[2], a[1] + a[3], tb[0], y0 + tb[1], tb[0] + tb[2], y0 + tb[1] + tb[3], 1);
        if (menu & 1) {
            ov_rect(cb, 0, 0, W, H, bx - 2, by - 2, bx + 2, by + 2, 0.45f);
            ov_rect(cb, p[0], p[1], p[0] + p[2], p[1] + p[3], pr[0], y0 + pr[1], pr[0] + pr[2], y0 + pr[1] + pr[3], 1);
        }
    }
    if (fps && menu < 0) ov_fps(cb, now);
}
// the theme's atlas for a layout into its texture, once (overlay_theme.c reads it at the first present that shows the gamepad in that layout): copied in this
// present's command buffer; its staging buffer goes at a later present, once the copy has run (its slot's fence signalled, or the slot came round again)
static void ov_theme_upload(VkCommandBuffer cb, uint32_t k, int lay) {
    typeof(g_tt[0]) *T = &g_tt[lay];
    if (T->up && (k == T->upk || vkWaitForFences(V.dev, 1, &g_pfence[T->upk], 1, 0) == VK_SUCCESS)) { vkDestroyBuffer(V.dev, T->up, NULL); vkFreeMemory(V.dev, T->upm, NULL); T->up = VK_NULL_HANDLE; }
    if (T->ready) return;
    const ov_theme *t = dm_overlay_theme(lay); T->ready = -1;
    if (!t || !t->px) return;
    VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {(uint32_t)t->w, (uint32_t)t->h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VKCHK(vkCreateImage(V.dev, &ic, NULL, &T->img)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, T->img, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &T->mem)); VKCHK(vkBindImageMemory(V.dev, T->img, T->mem, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, T->img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VKCHK(vkCreateImageView(V.dev, &vi, NULL, &T->view));
    VkDeviceSize n = (VkDeviceSize)t->w * t->h * 4;
    VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(V.dev, &bc, NULL, &T->up)); VkMemoryRequirements bq; vkGetBufferMemoryRequirements(V.dev, T->up, &bq);
    VkMemoryAllocateInfo bi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, bq.size, vk_mem_type(bq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    VKCHK(vkAllocateMemory(V.dev, &bi, NULL, &T->upm)); VKCHK(vkBindBufferMemory(V.dev, T->up, T->upm, 0));
    void *px; VKCHK(vkMapMemory(V.dev, T->upm, 0, VK_WHOLE_SIZE, 0, &px)); memcpy(px, t->px, (size_t)n);   // (freeing the memory unmaps it)
    dm_overlay_theme_done(t);
    VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_ov.pool, 1, &g_ov.dsl};
    if (vkAllocateDescriptorSets(V.dev, &da, &T->set)) { VKRED_ONCE(95, "overlay theme descriptor set failed: the gamepad is drawn without its theme"); return; }
    VkDescriptorImageInfo ii = {g_ov.samp, T->view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, T->set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
    vkUpdateDescriptorSets(V.dev, 1, &wr, 0, NULL);
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, T->img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    VkBufferImageCopy c = {0, (uint32_t)t->w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t)t->w, (uint32_t)t->h, 1}};
    vkCmdCopyBufferToImage(cb, T->up, T->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
    T->t = t; T->upk = k; T->ready = 1;
}
static int overlay_ready(VkCommandBuffer cb, uint32_t k, int menu) {   // k: this present's slot (its fence is not signalled yet)
    uint32_t w = g_sext.width / 2, h = g_sext.height / 2; int lay = dm_overlay_layout();
    if (!g_ov.dsl) {
        VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b}; VKCHK(vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_ov.dsl));
        VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ov_quad)};
        VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_ov.dsl, 1, &pr}; VKCHK(vkCreatePipelineLayout(V.dev, &pl, NULL, &g_ov.layout));
        VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4}; VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 4, 1, &ps};   // the procedural texture, a theme atlas per layout, the performance panel (r86)
        VKCHK(vkCreateDescriptorPool(V.dev, &pi, NULL, &g_ov.pool));
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_ov.pool, 1, &g_ov.dsl}; VKCHK(vkAllocateDescriptorSets(V.dev, &ai, &g_ov.set));
        VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
        VKCHK(vkCreateSampler(V.dev, &si, NULL, &g_ov.samp));
    }
    if (g_ov.pfmt != g_sfmt) {   // the pipeline renders into the swapchain's format
        VkShaderModule vm, fm;
        VkShaderModuleCreateInfo vc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof overlay_vert_spv, overlay_vert_spv}; VKCHK(vkCreateShaderModule(V.dev, &vc, NULL, &vm));
        VkShaderModuleCreateInfo fc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof overlay_frag_spv, overlay_frag_spv}; VKCHK(vkCreateShaderModule(V.dev, &fc, NULL, &fm));
        VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", NULL},
                                                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", NULL}};
        VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
        VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
        VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1};
        VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState ba = {1, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_ADD, 0xf};
        VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &ba};
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
        VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &g_sfmt};
        if (!V.has_dynrender && g_ov.rp) { ov_fb_drop(); vkDestroyRenderPass(V.dev, g_ov.rp, NULL); g_ov.rp = VK_NULL_HANDLE; }   // the pass follows the swapchain's format, as the pipeline does
        VkRenderPass orp = V.has_dynrender ? VK_NULL_HANDLE : ov_rp();
        if (!V.has_dynrender && !orp) return 0;
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, orp ? NULL : &rci, 0, 2, ss, &vi, &ia, NULL, &vs, &rs, &ms, NULL, &cbs, &dy, g_ov.layout, orp};
        VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, &g_ov.pipe);
        vkDestroyShaderModule(V.dev, vm, NULL); vkDestroyShaderModule(V.dev, fm, NULL);
        if (r) { VKRED_ONCE(92, "overlay pipeline failed: %d", r); return 0; }
        g_ov.pfmt = g_sfmt;
    }
    ov_theme_upload(cb, k, lay);
    if (g_ov.w != w || g_ov.h != h || g_ov.lay != lay || (g_tt[lay].ready != 1 && g_ov.cver != dm_overlay_custom_ver())) {   // (re)rasterize for this window size and layout
        g_ov.cver = dm_overlay_custom_ver();   // (and, without a theme, the player's own layout: the outlines are in the static layer)
        vkDeviceWaitIdle(V.dev);
        if (g_ov.view) { vkDestroyImageView(V.dev, g_ov.view, NULL); vkDestroyImage(V.dev, g_ov.img, NULL); vkFreeMemory(V.dev, g_ov.mem, NULL); vkDestroyBuffer(V.dev, g_ov.up, NULL); vkFreeMemory(V.dev, g_ov.upm, NULL); }
        g_ov.lay = lay; uint32_t sh = ov_layout_sprites(w, h); float mr[4]; ov_menu_rect((float)g_sext.width, (float)g_sext.height, mr);
        g_ov.pw = (uint32_t)(mr[2] / 2) + 1; g_ov.ph = (uint32_t)(mr[3] / 2) + 1; if (g_ov.pw > w) g_ov.pw = w;
        g_ov.gs = (uint32_t)fmaxf(2, roundf(h * 0.035f / 7)); g_ov.gh = 7 * g_ov.gs + 4;   // the counter's characters: 3.5% of the window's height
        uint32_t th = h + sh + g_ov.ph + g_ov.gh;
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {w, th, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        VKCHK(vkCreateImage(V.dev, &ic, NULL, &g_ov.img)); VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_ov.img, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &g_ov.mem)); VKCHK(vkBindImageMemory(V.dev, g_ov.img, g_ov.mem, 0));
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_ov.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(V.dev, &vi, NULL, &g_ov.view));
        VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)w * th * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
        VKCHK(vkCreateBuffer(V.dev, &bc, NULL, &g_ov.up)); VkMemoryRequirements bq; vkGetBufferMemoryRequirements(V.dev, g_ov.up, &bq);
        VkMemoryAllocateInfo bi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, bq.size, vk_mem_type(bq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &bi, NULL, &g_ov.upm)); VKCHK(vkBindBufferMemory(V.dev, g_ov.up, g_ov.upm, 0));
        void *px; VKCHK(vkMapMemory(V.dev, g_ov.upm, 0, VK_WHOLE_SIZE, 0, &px)); g_ov.px = px;
        g_ov.w = w; g_ov.h = h; g_ov.sh = sh; ov_raster(g_ov.px, w, h, sh); memset(g_ov.px + (size_t)w * (h + sh) * 4, 0, (size_t)w * (g_ov.ph + g_ov.gh) * 4); ov_fps_glyphs();
        VkDescriptorImageInfo ii = {g_ov.samp, g_ov.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_ov.set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
        vkUpdateDescriptorSets(V.dev, 1, &wr, 0, NULL);
        g_ov.uploaded = 0; g_ov.panel = -1;
    }
    uint32_t y0 = 0, rows = g_ov.h + g_ov.sh + g_ov.ph + g_ov.gh;   // the rows to upload
    if (g_ov.uploaded && menu >= 0 && menu != g_ov.panel) {   // only the panel, after the last copy out of the buffer has run
        VkFence f[8]; uint32_t n = 0; for (uint32_t i = 0; i < 8; i++) if (i != k) f[n++] = g_pfence[i];
        vkWaitForFences(V.dev, n, f, 1, 1000000000ull);
        y0 = g_ov.h + g_ov.sh; rows = g_ov.ph; g_ov.uploaded = 0;
    }
    if (menu >= 0 && menu != g_ov.panel) { ov_panel(menu); g_ov.panel = menu; }
    if (!g_ov.uploaded) {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, y0 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_ov.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkBufferImageCopy c = {(VkDeviceSize)y0 * g_ov.w * 4, g_ov.w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, (int32_t)y0, 0}, {g_ov.w, rows, 1}};
        vkCmdCopyBufferToImage(cb, g_ov.up, g_ov.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        g_ov.uploaded = 1;
    }
    return 1;
}
// r86: the performance panel's texture: perf_overlay_win.c paints it on the CPU (only when it changed: each 500 ms sample, the spinner's 80 ms steps, a click or the wheel),
// it goes up through this slot's own staging buffer (the slot's fence was waited for) into an image that only grows, drawn 1:1 at the place perf_ov_frame gives
static struct { VkImage img; VkDeviceMemory mem; VkImageView view; VkDescriptorSet set; uint32_t cw, ch; VkBuffer up[8]; VkDeviceMemory upm[8]; uint8_t *upp[8]; VkDeviceSize upn[8];
                int x, y, w, h, has, drv; int64_t vram_t; uint8_t lin[256]; } g_pf;
static void perf_vram(void) {   // VK_EXT_memory_budget: the device-local heaps' usage and budget (NXN1X's VRAM row); "--" without it
    static int ok = -1; static PFN_vkGetPhysicalDeviceMemoryProperties2 f;
    if (ok < 0) {
        ok = 0; uint32_t n = 0; vkEnumerateDeviceExtensionProperties(V.pd, NULL, &n, NULL); VkExtensionProperties *e = calloc(n ? n : 1, sizeof *e);
        if (e) { vkEnumerateDeviceExtensionProperties(V.pd, NULL, &n, e); for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) ok = 1; free(e); }
        f = (PFN_vkGetPhysicalDeviceMemoryProperties2)vk_gipa()(V.inst, "vkGetPhysicalDeviceMemoryProperties2"); if (!f) ok = 0;
        VKLOG("r86: performance overlay VRAM: %s", ok ? "VK_EXT_memory_budget" : "no VK_EXT_memory_budget (shown as --)");
    }
    if (!ok) { perf_ov_vram(-1, -1); return; }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT}; VkPhysicalDeviceMemoryProperties2 p = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &b};
    f(V.pd, &p); uint64_t used = 0, budget = 0;
    for (uint32_t i = 0; i < p.memoryProperties.memoryHeapCount && i < VK_MAX_MEMORY_HEAPS; i++) if (p.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) { used += b.heapUsage[i]; budget += b.heapBudget[i]; }
    perf_ov_vram(budget ? (int64_t)used : -1, budget ? (int64_t)budget : -1);
}
static void perf_driver(void) {   // NXN1X's Driver row: "<backend> · <driver>"
    VkPhysicalDeviceDriverProperties dp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES}; VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &dp};
    vkGetPhysicalDeviceProperties2(V.pd, &p2);
    const char *eff = dp.driverName[0] ? dp.driverName : V.props.deviceName;
    char info[VK_MAX_DRIVER_INFO_SIZE + 1]; snprintf(info, sizeof info, "%s", dp.driverInfo); char *cut = strstr(info, " ("); if (cut) *cut = 0;   // "Mesa 26.0.0-devel (git-...)": the version
    if (!info[0]) { uint32_t d = V.props.driverVersion; snprintf(info, sizeof info, "%u.%u.%u", d >> 22, d >> 12 & 0x3ff, d & 0xfff); }
    char s[160]; snprintf(s, sizeof s, "%s \xc2\xb7 %s", eff, info);
    perf_ov_driver(s);
}
static int perf_prepare(VkCommandBuffer cb, uint32_t k) {   // outside the render pass: the panel's picture, uploaded when it changed; 0: nothing to draw
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); int64_t now = (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
    if (!g_pf.drv) { g_pf.drv = 1; perf_driver(); for (int i = 0; i < 256; i++) { float c = i / 255.0f; g_pf.lin[i] = (uint8_t)(255 * (c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f)) + 0.5f); } }
    if (!g_pf.vram_t || now - g_pf.vram_t >= 500000000ll) { g_pf.vram_t = now; perf_vram(); }
    perf_ov_gpu_ns(g_gpu_ns);   // r102: the GPU row (perf_overlay_win.c sampler)
    const uint8_t *px; int w, h, x, y;
    int changed = perf_ov_frame(now, atomic_load(&g_fps_frames), g_sext.width, g_sext.height, dm_front_scale(g_win), &px, &w, &h, &x, &y);
    if (!px || w <= 0 || h <= 0 || w > 4096 || h > 4096) return 0;
    if ((uint32_t)w > g_pf.cw || (uint32_t)h > g_pf.ch) {   // a larger image (rare: the first panel, a larger DPI scale)
        uint32_t cw = g_pf.cw > (uint32_t)w ? g_pf.cw : (uint32_t)w + 64, ch = g_pf.ch > (uint32_t)h ? g_pf.ch : (uint32_t)h + 64;
        if (g_pf.img) { vkDeviceWaitIdle(V.dev); vkDestroyImageView(V.dev, g_pf.view, NULL); vkDestroyImage(V.dev, g_pf.img, NULL); vkFreeMemory(V.dev, g_pf.mem, NULL); g_pf.img = VK_NULL_HANDLE; }
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {cw, ch, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(V.dev, &ic, NULL, &g_pf.img)) { VKRED_ONCE(89, "r86: performance overlay image failed"); g_pf.img = VK_NULL_HANDLE; return 0; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_pf.img, &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(V.dev, &ai, NULL, &g_pf.mem)); VKCHK(vkBindImageMemory(V.dev, g_pf.img, g_pf.mem, 0));
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_pf.img, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(V.dev, &vi, NULL, &g_pf.view));
        if (!g_pf.set) { VkDescriptorSetAllocateInfo da = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_ov.pool, 1, &g_ov.dsl};
                         if (vkAllocateDescriptorSets(V.dev, &da, &g_pf.set)) { VKRED_ONCE(88, "r86: performance overlay descriptor set failed"); g_pf.set = VK_NULL_HANDLE; return 0; } }
        VkDescriptorImageInfo ii = {g_ov.samp, g_pf.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_pf.set, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii};
        vkUpdateDescriptorSets(V.dev, 1, &wr, 0, NULL);
        g_pf.cw = cw; g_pf.ch = ch; g_pf.has = 0;
    }
    if (changed || !g_pf.has) {
        VkDeviceSize n = (VkDeviceSize)w * h * 4;
        if (g_pf.upn[k] < n) {   // this slot's staging buffer (its last copy ran: the slot's fence was waited for)
            if (g_pf.up[k]) { vkDestroyBuffer(V.dev, g_pf.up[k], NULL); vkFreeMemory(V.dev, g_pf.upm[k], NULL); g_pf.up[k] = VK_NULL_HANDLE; g_pf.upn[k] = 0; }
            VkDeviceSize sz = (VkDeviceSize)g_pf.cw * g_pf.ch * 4;
            VkBufferCreateInfo bc = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
            VKCHK(vkCreateBuffer(V.dev, &bc, NULL, &g_pf.up[k])); VkMemoryRequirements bq; vkGetBufferMemoryRequirements(V.dev, g_pf.up[k], &bq);
            VkMemoryAllocateInfo bi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, bq.size, vk_mem_type(bq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
            VKCHK(vkAllocateMemory(V.dev, &bi, NULL, &g_pf.upm[k])); VKCHK(vkBindBufferMemory(V.dev, g_pf.up[k], g_pf.upm[k], 0));
            void *m; VKCHK(vkMapMemory(V.dev, g_pf.upm[k], 0, VK_WHOLE_SIZE, 0, &m)); g_pf.upp[k] = m; g_pf.upn[k] = sz;
        }
        // An sRGB swapchain encodes what is drawn: the panel's colours go in linear so they come out as NXN1X's (Qt paints them as they are)
        int srgb = g_sfmt == VK_FORMAT_R8G8B8A8_SRGB || g_sfmt == VK_FORMAT_B8G8R8A8_SRGB;
        if (srgb) { uint8_t *d = g_pf.upp[k]; for (size_t i = 0; i < (size_t)w * h * 4; i += 4) { d[i] = g_pf.lin[px[i]]; d[i + 1] = g_pf.lin[px[i + 1]]; d[i + 2] = g_pf.lin[px[i + 2]]; d[i + 3] = px[i + 3]; } }
        else memcpy(g_pf.upp[k], px, (size_t)w * h * 4);
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, g_pf.has ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_pf.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        VkBufferImageCopy c = {0, (uint32_t)w, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t)w, (uint32_t)h, 1}};
        vkCmdCopyBufferToImage(cb, g_pf.up[k], g_pf.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &b);
        g_pf.has = 1;
    }
    g_pf.x = x; g_pf.y = y; g_pf.w = w; g_pf.h = h;
    return 1;
}
static void perf_draw(VkCommandBuffer cb) {   // in the overlay's pass, its pipeline bound: one quad, texel for pixel
    float W = (float)g_sext.width, H = (float)g_sext.height, x0 = (float)g_pf.x, y0 = (float)g_pf.y, x1 = x0 + g_pf.w, y1 = y0 + g_pf.h;
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_pf.set, 0, NULL);
    ov_quad q = {{x0 / W * 2 - 1, y0 / H * 2 - 1, x1 / W * 2 - 1, y1 / H * 2 - 1}, {0, 0, (float)g_pf.w / g_pf.cw, (float)g_pf.h / g_pf.ch}, 1}; ov_draw(cb, &q);
}
// ---- frame pacing on the display (VK_GOOGLE_display_timing) ----
// Without a time of its own a frame goes on screen at the first vsync after it is latched. Mario vs. Donkey Kong at 60 fps on the Legion Y700's 120 Hz panel was ready about
// 4.6 ms before that latch (SurfaceFlinger's latency: 12.9 ms from ready to shown, GPU 9.5-10.5 ms a frame), and about one frame in ten missed it by a millisecond or so: shown
// for three vsyncs, the next one for one (11 of 125 frames each way, a visible stutter; Hades II's title, ready far earlier, showed every frame for two). Each frame now gets a
// desired present time: the last one's plus the game's period, on the display's vsync grid (anchored at a past frame's actual present time), half a vsync before that vsync so
// it cannot go out early. A frame that would not be ready in time moves the cadence to the first vsync it can make (a "resync": one longer frame, and from then on a vsync of
// slack). The pause menu's redraws carry no time. DM_VK_PACE=0 (live) presents without times
static uint64_t g_refresh_ns, g_pace_target, g_pace_anchor, g_pace_n, g_pace_resync, g_pace_slip; static uint32_t g_pace_id; static int g_pace_on = 1;
static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static void pace_reset(void) { g_pace_target = g_pace_anchor = g_refresh_ns = 0; }
static uint64_t pace_snap(uint64_t t) {   // the first vsync at or after t on the grid through the anchor
    uint64_t r = g_refresh_ns, a = g_pace_anchor;
    if (t <= a) return a - (a - t) / r * r;
    return a + (t - a + r - 1) / r * r;
}
static uint64_t pace_desired(double period) {   // the frame's desired present time (0 = none)
    if (!V.has_dtiming || !g_swap) return 0;
    if (g_pace_n % 30 == 0) { const char *e = getenv("DM_VK_PACE"); g_pace_on = !(e && !atoi(e)); }
    if (!g_refresh_ns || g_pace_n % 120 == 0) {
        VkRefreshCycleDurationGOOGLE rc; uint64_t was = g_refresh_ns;
        if (vkGetRefreshCycleDurationGOOGLE(V.dev, g_swap, &rc) == VK_SUCCESS && rc.refreshDuration > 1000000) g_refresh_ns = rc.refreshDuration;
        if (g_refresh_ns && g_refresh_ns != was) { VKLOG("[perf] display timing: refresh %.3f ms", g_refresh_ns / 1e6); g_pace_target = 0; }
    }
    VkPastPresentationTimingGOOGLE pt[16]; uint32_t np = 16;
    if (vkGetPastPresentationTimingGOOGLE(V.dev, g_swap, &np, pt) >= 0)
        for (uint32_t i = 0; i < np; i++) {
            present_time_record(pt[i].presentID, pt[i].actualPresentTime);
            if (pt[i].actualPresentTime > g_pace_anchor) g_pace_anchor = pt[i].actualPresentTime;
            if (pt[i].desiredPresentTime && pt[i].actualPresentTime > pt[i].desiredPresentTime + g_refresh_ns) g_pace_slip++;   // went out a vsync or more after its own
        }
    g_pace_n++;
    if (g_pace_n % 600 == 0) VKLOG("[perf] display timing: %llu frames, %llu resyncs, %llu shown a vsync or more after their time (cumulative)%s",
                                   (unsigned long long)g_pace_n, (unsigned long long)g_pace_resync, (unsigned long long)g_pace_slip, g_pace_on ? "" : "; off (DM_VK_PACE=0)");
    if (!g_pace_on || !g_refresh_ns || !g_pace_anchor || period <= 0) { g_pace_target = 0; return 0; }
    // The earliest vsync this frame can make: its GPU work (submitted just before), the blit, then the latch a vsync before it is shown. When the cadence's next vsync is
    // earlier, the cadence restarts half a vsync further out than that, so the next frames have room for the jitter: restarting at the earliest itself made 31 long frames
    // in 4,200 (5 shown late), and going on until the target was under a vsync away made 54 in 3,600 go out late
    double vk_gpu_ms_recent(void); double g = vk_gpu_ms_recent(); uint64_t gpu = (uint64_t)((g > 4 ? g : 4) * 1e6) + 3000000;
    uint64_t now = mono_ns(), t = g_pace_target ? pace_snap(g_pace_target + (uint64_t)(period * 1e9) - g_refresh_ns / 2) : 0;
    if (t < now + gpu + g_refresh_ns) { if (g_pace_target) g_pace_resync++; t = pace_snap(now + gpu + g_refresh_ns * 3 / 2); }
    g_pace_target = t;
    return t - g_refresh_ns / 2;
}
// r53 measure: where the present thread waits, per 600 presents (fence of the slot's last present, acquire, the queue lock the replay's submits also take, the present call)
static uint64_t g_pt[4], g_ptmax[4], g_ptn;
static void pt_add(int i, uint64_t a, uint64_t b) { uint64_t d = b - a; g_pt[i] += d; if (d > g_ptmax[i]) g_ptmax[i] = d; }
static void pt_log(void) {
    if (++g_ptn % 600) return;
    VKLOG("present thread per present (last 600): fence wait %.2f ms (max %.1f), acquire %.2f (max %.1f), queue lock wait %.2f (max %.1f), vkQueuePresentKHR %.2f (max %.1f)",
          g_pt[0] / 6e8, g_ptmax[0] / 1e6, g_pt[1] / 6e8, g_ptmax[1] / 1e6, g_pt[2] / 6e8, g_ptmax[2] / 1e6, g_pt[3] / 6e8, g_ptmax[3] / 1e6);
    memset(g_pt, 0, sizeof g_pt); memset(g_ptmax, 0, sizeof g_ptmax);
}
// r116 (NPARK 11-09, "창 출력 단계 (모든 게임)"): AMD FidelityFX Super Resolution 1 in place of the bilinear blit of the game's picture into the window (DM_VK_FSR1=1; off by default).
// r121 (11-09, whole frames black with FSR 1 in Bayo3 and Age of Calamity): r116 ran both passes as compute shaders into two storage images shared by every present;
// r120's DM_FSR_DIAG showed the passes harmless (3: computed, not shown: no flicker) and their result black on some frames (1: magenta clear, still black).
// Now as Eden does it (video_core/renderer_vulkan/present/fsr.cpp): two fragment passes (shaders/fsr1.frag, MIT) into colour attachments, a pair of images for each
// swapchain image, with layout transitions around each pass. EASU scales the picture's crop to the window's picture size, RCAS sharpens it (DM_FSR_SHARPNESS 0-100,
// 100 = sharpest); the result is copied into the swapchain image where the blit drew. Only upscales (a window smaller than the picture keeps the blit), and any failure keeps it.
// Both read live (ctl.txt "env" lines). Its own state rather than GSR1's (gfx_vk_exec.c gsr_ready / gsr_tmp): those run on the replay thread; this one on the present thread
// with 8 presents in flight, so its sets and source views are per slot
#include "fsr1_easu_frag.spv.h"
#include "fsr1_rcas_frag.spv.h"
#include "fsr1_vert.spv.h"
extern volatile long dm_env_gen;   // r53 compat/posix_win.c
typedef struct { uint32_t con[16]; int32_t mode[4]; } fsr_pc;
static struct { int state; VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline easu, rcas; VkSampler samp; VkDescriptorPool pool; VkDescriptorSet set[8][2]; VkImageView sv[8];
                VkImage img[8][2]; VkDeviceMemory mem[8][2]; VkImageView view[8][2]; uint32_t w, h, n; int on; float sharp; long gen; uint64_t runs; int diag; } g_fsr = {.gen = -1};
static int fsr_on(void) {
    if (g_fsr.gen != dm_env_gen) { g_fsr.gen = dm_env_gen;
        const char *e = getenv("DM_VK_FSR1"), *s = getenv("DM_FSR_SHARPNESS"); int on = e && atoi(e) != 0; float pct = s && *s ? (float)atof(s) : 90.0f; if (pct < 0) pct = 0; if (pct > 100) pct = 100; float sh = (100.0f - pct) / 50.0f;   // r116b: percent, 100 = 0 stops
        if (on != g_fsr.on || (on && sh != g_fsr.sharp)) VKLOG("r116: FSR 1 window upscale %s (DM_VK_FSR1), sharpness %.0f%% = %.2f stops (DM_FSR_SHARPNESS 0-100, 100 = sharpest)", on ? "on" : "off", 100.0f - sh * 50.0f, sh);
        g_fsr.on = on; g_fsr.sharp = sh;
        const char *dg = getenv("DM_FSR_DIAG"); int d = dg ? atoi(dg) : 0;   // r120 (11-09, diagnosis of the black frames with FSR 1): 1 = clear magenta, 2 = no clear, 3 = FSR runs but the old blit shows the picture
        if (d != g_fsr.diag) VKLOG("r120: FSR 1 diagnosis %d (DM_FSR_DIAG: 0 off, 1 magenta clear, 2 no clear, 3 FSR runs, blit shown)", d);
        g_fsr.diag = d; }
    return g_fsr.on && g_fsr.state >= 0;
}
static int fsr_ready(void) {
    if (g_fsr.state) return g_fsr.state > 0;
    g_fsr.state = -1;
    if (!V.has_dynrender) { VKRED_ONCE(116, "r121: FSR 1 needs dynamic rendering: blit instead"); return 0; }
    VkDescriptorSetLayoutBinding b = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(fsr_pc)};
    if (vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_fsr.dsl)) { VKRED_ONCE(116, "r116: FSR 1 set layout failed: blit instead"); return 0; }
    VkPipelineLayoutCreateInfo pi = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &g_fsr.dsl, 1, &pr};
    if (vkCreatePipelineLayout(V.dev, &pi, NULL, &g_fsr.pl)) { VKRED_ONCE(116, "r116: FSR 1 pipeline layout failed: blit instead"); return 0; }
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    if (vkCreateSampler(V.dev, &si, NULL, &g_fsr.samp)) { VKRED_ONCE(116, "r116: FSR 1 sampler failed: blit instead"); return 0; }
    VkShaderModule vm; VkShaderModuleCreateInfo vc = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sizeof fsr1_vert_spv, fsr1_vert_spv};   // (the overlay's pipeline, overlay_ready, is the model)
    if (vkCreateShaderModule(V.dev, &vc, NULL, &vm)) { VKRED_ONCE(116, "r116: FSR 1 shader module failed: blit instead"); return 0; }
    for (int i = 0; i < 2; i++) {
        VkShaderModule fm; VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, i ? sizeof fsr1_rcas_frag_spv : sizeof fsr1_easu_frag_spv, i ? fsr1_rcas_frag_spv : fsr1_easu_frag_spv};
        if (vkCreateShaderModule(V.dev, &mi, NULL, &fm)) { vkDestroyShaderModule(V.dev, vm, NULL); VKRED_ONCE(116, "r116: FSR 1 shader module failed: blit instead"); return 0; }
        VkPipelineShaderStageCreateInfo ss[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", NULL},
                                                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", NULL}};
        VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0};
        VkPipelineViewportStateCreateInfo vs = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
        VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1};
        VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
        VkPipelineColorBlendAttachmentState ba = {0, 0, 0, 0, 0, 0, 0, 0xf};
        VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &ba};
        VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR}; VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 2, dyn};
        VkFormat f16 = VK_FORMAT_R16G16B16A16_SFLOAT; VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &f16};
        VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci, 0, 2, ss, &vi, &ia, NULL, &vs, &rs, &ms, NULL, &cbs, &dy, g_fsr.pl, VK_NULL_HANDLE};
        VkResult r = vkCreateGraphicsPipelines(V.dev, V.pcache, 1, &ci, NULL, i ? &g_fsr.rcas : &g_fsr.easu); vkDestroyShaderModule(V.dev, fm, NULL);
        if (r) { vkDestroyShaderModule(V.dev, vm, NULL); VKRED_ONCE(116, "r121: FSR 1 graphics pipeline failed (%d): blit instead", r); return 0; } }
    vkDestroyShaderModule(V.dev, vm, NULL);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 16, 1, &ps};
    if (vkCreateDescriptorPool(V.dev, &dp, NULL, &g_fsr.pool)) { VKRED_ONCE(116, "r116: FSR 1 descriptor pool failed: blit instead"); return 0; }
    VkDescriptorSetLayout l16[16]; for (int i = 0; i < 16; i++) l16[i] = g_fsr.dsl;
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, g_fsr.pool, 16, l16};
    if (vkAllocateDescriptorSets(V.dev, &ai, &g_fsr.set[0][0])) { VKRED_ONCE(116, "r116: FSR 1 descriptor sets failed: blit instead"); return 0; }
    g_fsr.state = 1; VKLOG("r121: FSR 1 ready (EASU + RCAS as fragment passes, as Eden)"); return 1;
}
static void fsr_free(void) {
    for (uint32_t s = 0; s < 8; s++) for (int i = 0; i < 2; i++) {
        if (g_fsr.view[s][i]) vkDestroyImageView(V.dev, g_fsr.view[s][i], NULL); if (g_fsr.img[s][i]) vkDestroyImage(V.dev, g_fsr.img[s][i], NULL); if (g_fsr.mem[s][i]) vkFreeMemory(V.dev, g_fsr.mem[s][i], NULL);
        g_fsr.view[s][i] = VK_NULL_HANDLE; g_fsr.img[s][i] = VK_NULL_HANDLE; g_fsr.mem[s][i] = VK_NULL_HANDLE; }
    g_fsr.w = g_fsr.h = g_fsr.n = 0;
}
static int fsr_imgs(uint32_t w, uint32_t h) {   // per swapchain image: [0] EASU -> RCAS, [1] RCAS -> the swapchain image; RGBA16F at the window's picture size (made again when that or the image count changes)
    if (g_fsr.n == g_nsimg && g_fsr.w == w && g_fsr.h == h) return 1;
    if (g_fsr.n) { vkDeviceWaitIdle(V.dev); fsr_free(); }
    for (uint32_t s = 0; s < g_nsimg && s < 8; s++) for (int i = 0; i < 2; i++) {
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (i ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : VK_IMAGE_USAGE_SAMPLED_BIT), VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(V.dev, &ic, NULL, &g_fsr.img[s][i])) { g_fsr.img[s][i] = VK_NULL_HANDLE; goto fail; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_fsr.img[s][i], &rq);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        if (vkAllocateMemory(V.dev, &ai, NULL, &g_fsr.mem[s][i])) { g_fsr.mem[s][i] = VK_NULL_HANDLE; goto fail; }
        if (vkBindImageMemory(V.dev, g_fsr.img[s][i], g_fsr.mem[s][i], 0)) goto fail;
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, g_fsr.img[s][i], VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R16G16B16A16_SFLOAT, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(V.dev, &vi, NULL, &g_fsr.view[s][i])) { g_fsr.view[s][i] = VK_NULL_HANDLE; goto fail; } }
    g_fsr.w = w; g_fsr.h = h; g_fsr.n = g_nsimg; VKLOG("r121: FSR 1 images %ux%u RGBA16F, a pair for each of the %u swapchain images", w, h, g_nsimg); return 1;
fail:
    fsr_free(); VKRED_ONCE(117, "r116: FSR 1 images %ux%u failed: blit instead", w, h); return 0;
}
static int srgb_vk(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32; }
static uint32_t fsr_f(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void fsr_pass(VkCommandBuffer cb, VkPipeline p, VkDescriptorSet set, VkImage img, VkImageView v, const fsr_pc *pc, uint32_t w, uint32_t h, VkImageLayout after) {   // one full-target draw into img
    barrier(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);   // (its whole area is drawn: the old contents are not needed)
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, 1, 0, 1, &ca, NULL, NULL};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_fsr.pl, 0, 1, &set, 0, NULL);
    vkCmdPushConstants(cb, g_fsr.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof *pc, pc);
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1}; VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    barrier(cb, img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, after);
}
// Records both passes into the present's command buffer (slot k, whose fence has signalled: its sets and source view are free; swapchain image idx, whose images only
// presents of that swapchain image use); returns the image holding the picture, dw x dh, in TRANSFER_SRC_OPTIMAL, or VK_NULL_HANDLE
static VkImage fsr_run(VkCommandBuffer cb, uint32_t k, uint32_t idx, vkimg_t *im, int cx, int cy, int cw, int ch, int dw, int dh, int flip) {
    if (!fsr_ready()) return VK_NULL_HANDLE;
    if (!(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || im->samples > 1 || im->d > 1) { VKRED_ONCE(118, "r116: FSR 1: the window texture cannot be sampled (usage %#x, format %d): blit instead", im->usage, im->fmt); return VK_NULL_HANDLE; }
    if (!fsr_imgs((uint32_t)dw, (uint32_t)dh) || idx >= g_fsr.n) return VK_NULL_HANDLE;
    if (g_fsr.sv[k]) { vkDestroyImageView(V.dev, g_fsr.sv[k], NULL); g_fsr.sv[k] = VK_NULL_HANDLE; }   // slot k's previous present has finished (its fence)
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, VK_IMAGE_VIEW_TYPE_2D, im->fmt, {0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(V.dev, &vi, NULL, &g_fsr.sv[k])) { g_fsr.sv[k] = VK_NULL_HANDLE; VKRED_ONCE(119, "r116: FSR 1: no view of the window texture (format %d): blit instead", im->fmt); return VK_NULL_HANDLE; }
    VkDescriptorImageInfo ii[2] = {{g_fsr.samp, g_fsr.sv[k], VK_IMAGE_LAYOUT_GENERAL}, {g_fsr.samp, g_fsr.view[idx][0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkWriteDescriptorSet wr[2];
    for (int i = 0; i < 2; i++) wr[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, g_fsr.set[k][i], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ii[i], NULL, NULL};
    vkUpdateDescriptorSets(V.dev, 2, wr, 0, NULL);
    // EASU constants (ffx_fsr1.h FsrEasuCon, with the crop's origin added to con0's offset: its pixel positions are in the texture's texels)
    float W = (float)im->w, H = (float)im->h, rx = (float)cw / (float)dw, ry = (float)ch / (float)dh;
    fsr_pc pc = {{fsr_f(rx), fsr_f(ry), fsr_f(0.5f * rx - 0.5f + (float)cx), fsr_f(0.5f * ry - 0.5f + (float)cy),
                  fsr_f(1.0f / W), fsr_f(1.0f / H), fsr_f(1.0f / W), fsr_f(-1.0f / H), fsr_f(-1.0f / W), fsr_f(2.0f / H), fsr_f(1.0f / W), fsr_f(2.0f / H), 0, fsr_f(4.0f / H), 0, 0},
                 {srgb_vk(im->fmt), flip, dw, dh}};
    fsr_pass(cb, g_fsr.easu, g_fsr.set[k][0], g_fsr.img[idx][0], g_fsr.view[idx][0], &pc, (uint32_t)dw, (uint32_t)dh, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // RCAS constants (FsrRcasCon: stops to a linear factor); mode.x: decode for an sRGB swapchain, whose copy encodes again
    fsr_pc rc = {{fsr_f(exp2f(-g_fsr.sharp))}, {srgb_vk(g_sfmt), 0, dw, dh}};
    fsr_pass(cb, g_fsr.rcas, g_fsr.set[k][1], g_fsr.img[idx][1], g_fsr.view[idx][1], &rc, (uint32_t)dw, (uint32_t)dh, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    uint64_t n = ++g_fsr.runs; if (n == 1) VKLOG("r116: FSR 1 first present: %dx%d crop of %ux%u format %d -> %dx%d in a %ux%u window (swapchain format %d)", cw, ch, im->w, im->h, im->fmt, dw, dh, g_sext.width, g_sext.height, g_sfmt);
    return g_fsr.img[idx][1];
}
// occ-s3fast diagnostic, DM_VK_FLASHDIAG=1 (live through DM_CTL env lines; off: one compare per present): a one-frame flash detector (NPARK 11-21: the terrain flash lasts one
// frame, only while moving). Each present's picture is blitted down (linear) to 128x72 and copied into a host-visible slot of its present command buffer; when that slot
// comes round again (its fence waited) the 16x9 cells' mean luma (8x8 samples each) goes into a history. Frame t flashes in a cell when |L(t) - L(t-1)| and |L(t) - L(t+1)|
// are both over FL_T with the same sign while |L(t+1) - L(t-1)| is under FL_T / 2. Logged (at most one every 30 presents, 200 in all): the present, the frame id
// (g_tex_frame at its present), the cells (count, brighter / darker, bounding box, largest jump), the three frame times; and the frame id handed to gfx_vk_core.c
// (vk_o3f_flash), whose replay thread logs that frame's context (CPU condition flips and addresses, GPU conditions, resolve points, landing waits). Samples every 15th
// pixel of 1080p: a flash smaller than that may be missed
enum { FL_W = 128, FL_H = 72, FL_CW = 16, FL_CH = 9, FL_T = 24, FL_SLOT = FL_W * FL_H * 4 };
static struct { int on, made, tried; long gen; VkImage img; VkDeviceMemory imem; VkBuffer buf; VkDeviceMemory bmem; uint8_t *map;
                uint64_t pf[8], fr[8], t[8], n, last_log, nlog; struct { uint64_t pres, fr, t; uint8_t L[FL_CW * FL_CH]; } h[3]; int nh; } g_fl;
void vk_o3f_flash(uint64_t frame);   // gfx_vk_core.c (the context lines, on the replay thread)
static int fl_on(void) {
    if (g_fl.gen != dm_env_gen) { g_fl.gen = dm_env_gen; const char *e = getenv("DM_VK_FLASHDIAG"); const int on = e && atoi(e) != 0;
        if (on != g_fl.on) { VKLOG("occ-s3fast: one-frame flash detector %s (DM_VK_FLASHDIAG, live; diagnostic: a 128x72 readback per present)", on ? "on" : "off"); g_fl.nh = 0; memset(g_fl.pf, 0, sizeof g_fl.pf); }
        g_fl.on = on; }
    return g_fl.on;
}
static int fl_make(void) {
    if (g_fl.made || g_fl.tried) return g_fl.made; g_fl.tried = 1;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {FL_W, FL_H, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(V.dev, &ii, NULL, &g_fl.img)) return 0;
    VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_fl.img, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    if (vkAllocateMemory(V.dev, &ai, NULL, &g_fl.imem) || vkBindImageMemory(V.dev, g_fl.img, g_fl.imem, 0)) return 0;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)FL_SLOT * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &g_fl.buf)) return 0;
    vkGetBufferMemoryRequirements(V.dev, g_fl.buf, &rq);
    VkMemoryAllocateInfo bai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    void *m = NULL;
    if (vkAllocateMemory(V.dev, &bai, NULL, &g_fl.bmem) || vkBindBufferMemory(V.dev, g_fl.buf, g_fl.bmem, 0) || vkMapMemory(V.dev, g_fl.bmem, 0, VK_WHOLE_SIZE, 0, &m)) return 0;
    g_fl.map = m; g_fl.made = 1; return 1;
}
static void fl_record(VkCommandBuffer cb, uint32_t k, VkImage src, int cx, int cy, int cw, int ch) {   // present_window, after the picture's blit (src in GENERAL)
    extern uint64_t g_tex_frame;
    if (!fl_make()) { VKRED_ONCE(97, "occ-s3fast: the flash detector's image or buffer could not be made"); return; }
    barrier(cb, g_fl.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageBlit b = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{cx, cy, 0}, {cx + cw, cy + ch, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {FL_W, FL_H, 1}}};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, g_fl.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    barrier(cb, g_fl.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy c = {(VkDeviceSize)FL_SLOT * k, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {FL_W, FL_H, 1}};
    vkCmdCopyImageToBuffer(cb, g_fl.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_fl.buf, 1, &c);
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    g_fl.pf[k] = ++g_fl.n; g_fl.fr[k] = g_tex_frame; g_fl.t[k] = now_ns();
}
static void fl_take(uint32_t k) {   // present_window, slot k's fence waited: its picture into the history, the middle of the last three checked
    if (!g_fl.pf[k] || !g_fl.map) return;
    if (g_fl.nh == 3) { g_fl.h[0] = g_fl.h[1]; g_fl.h[1] = g_fl.h[2]; g_fl.nh = 2; }
    __typeof__(g_fl.h[0]) *e = &g_fl.h[g_fl.nh++]; e->pres = g_fl.pf[k]; e->fr = g_fl.fr[k]; e->t = g_fl.t[k]; g_fl.pf[k] = 0;
    const uint8_t *p = g_fl.map + (size_t)FL_SLOT * k;
    for (int cyy = 0; cyy < FL_CH; cyy++) for (int cxx = 0; cxx < FL_CW; cxx++) { uint32_t s = 0;
        for (int y = 0; y < FL_H / FL_CH; y++) for (int x = 0; x < FL_W / FL_CW; x++) { const uint8_t *q = p + 4 * ((size_t)(cyy * (FL_H / FL_CH) + y) * FL_W + (size_t)(cxx * (FL_W / FL_CW) + x));
            s += (uint32_t)(54 * q[0] + 183 * q[1] + 19 * q[2]) >> 8; }   // (Rec. 709 weights; B/R order does not matter much for a jump)
        e->L[cyy * FL_CW + cxx] = (uint8_t)(s / ((FL_H / FL_CH) * (FL_W / FL_CW))); }
    if (g_fl.nh < 3 || g_fl.h[1].pres != g_fl.h[0].pres + 1 || g_fl.h[2].pres != g_fl.h[1].pres + 1) return;
    int n = 0, up = 0, dn = 0, x0 = FL_CW, x1 = -1, y0 = FL_CH, y1 = -1, dmax = 0;
    for (int i = 0; i < FL_CW * FL_CH; i++) { const int a = g_fl.h[0].L[i], b = g_fl.h[1].L[i], c = g_fl.h[2].L[i], d1 = b - a, d2 = b - c, d3 = c - a;
        if (abs(d1) > FL_T && abs(d2) > FL_T && (d1 > 0) == (d2 > 0) && abs(d3) < FL_T / 2) { n++; if (d1 > 0) up++; else dn++;
            const int cx = i % FL_CW, cy = i / FL_CW; if (cx < x0) x0 = cx; if (cx > x1) x1 = cx; if (cy < y0) y0 = cy; if (cy > y1) y1 = cy; if (abs(d1) > dmax) dmax = abs(d1); } }
    if (!n || g_fl.nlog >= 200 || (g_fl.last_log && g_fl.h[1].pres < g_fl.last_log + 30)) return;
    g_fl.nlog++; g_fl.last_log = g_fl.h[1].pres;
    VKLOG("occ-s3fast flash: present %llu (frame %llu): %d of 144 cells for one frame (brighter %d, darker %d), cells x %d-%d y %d-%d of 16x9, largest jump %d; present intervals %.1f / %.1f ms",
          (unsigned long long)g_fl.h[1].pres, (unsigned long long)g_fl.h[1].fr, n, up, dn, x0, x1, y0, y1, dmax, (g_fl.h[1].t - g_fl.h[0].t) / 1e6, (g_fl.h[2].t - g_fl.h[1].t) / 1e6);
    vk_o3f_flash(g_fl.h[1].fr);
}
static void present_window(vkimg_t *im, const int crop[4], double period, uint64_t ticket) {   // period: the game's frame period (s); < 0 for a redraw (ticket 0)
    uint32_t k = g_pi++ % 8;
    uint64_t t0 = now_ns(); vkWaitForFences(V.dev, 1, &g_pfence[k], 1, UINT64_MAX); pt_add(0, t0, now_ns());
    if (fl_on()) fl_take(k);   // occ-s3fast (DM_VK_FLASHDIAG): this slot's last picture into the flash history
    VkFormat want = im->fmt == VK_FORMAT_R8G8B8A8_SRGB || im->fmt == VK_FORMAT_B8G8R8A8_SRGB || im->fmt == VK_FORMAT_A8B8G8R8_SRGB_PACK32 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    if (want != g_want) { g_want = want; swap_destroy(); if (!swap_make()) return; }   // once, at the first present of the other kind
    { uint32_t cw = 0, ch = 0; dm_front_size(g_win, &cw, &ch);   // r29b (11-04): Win32 + Turnip report no OUT_OF_DATE on a resize (F11): the first extent stayed in a corner, black around it
      if (g_swap && cw && ch && (cw != g_sext.width || ch != g_sext.height)) { VKLOG("window %ux%u: swapchain made again", cw, ch); swap_destroy(); if (!swap_make()) return; } }
    uint32_t idx; uint64_t t1 = now_ns(); VkResult r = vkAcquireNextImageKHR(V.dev, g_swap, UINT64_MAX, g_acq[k], VK_NULL_HANDLE, &idx); pt_add(1, t1, now_ns());
    if (r == VK_ERROR_OUT_OF_DATE_KHR) { swap_destroy(); if (!swap_make()) return; r = vkAcquireNextImageKHR(V.dev, g_swap, UINT64_MAX, g_acq[k], VK_NULL_HANDLE, &idx); }
    if (r < 0) { VKRED_ONCE(90, "vkAcquireNextImageKHR: %d", r); return; }
    vkResetFences(V.dev, 1, &g_pfence[k]);
    VkCommandBuffer cb = g_pcb[k];
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; vkBeginCommandBuffer(cb, &bi);
    vk_img_barrier_all(cb);   // after everything submitted before on this queue
    barrier(cb, g_simg[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    int fdiag = fsr_on() ? g_fsr.diag : 0; if (fdiag == 1) black = (VkClearColorValue){{1, 0, 1, 1}};   // r120
    if (fdiag != 2) vkCmdClearColorImage(cb, g_simg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rg);
    barrier(cb, g_simg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);   // r118 (11-09, Bayo3 with FSR 1: whole frames black): the clear and the picture's
    // copy below both write this image, and nothing ordered them (write after write); with r116's compute passes in between, the clear's black could land after the picture
    int cx = crop ? crop[0] : 0, cy = crop ? crop[1] : 0, cw = crop && crop[2] ? crop[2] : (int)im->w, ch = crop && crop[3] ? crop[3] : (int)im->h;
    double s = fmin((double)g_sext.width / cw, (double)g_sext.height / ch); int dw = (int)(cw * s), dh = (int)(ch * s), dx = ((int)g_sext.width - dw) / 2, dy = ((int)g_sext.height - dh) / 2;
    vk_pic_set((float)dx / g_sext.width, (float)dy / g_sext.height, (float)dw / g_sext.width, (float)dh / g_sext.height);
    VkImageBlit b = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{cx, cy, 0}, {cx + cw, cy + ch, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{dx, dy, 0}, {dx + dw, dy + dh, 1}}};
    if (vk_window_bottom_up()) { b.srcOffsets[0].y = cy + ch; b.srcOffsets[1].y = cy; }   // lower-left origin: the window image is stored bottom-up (gfx_vk_exec.c)
    VkImage fimg = fsr_on() && dw >= cw && dh >= ch && dw > 0 && dh > 0 ? fsr_run(cb, k, idx, im, cx, cy, cw, ch, dw, dh, vk_window_bottom_up()) : VK_NULL_HANDLE;   // r116 / r121
    if (fimg && fdiag != 3) {   // r116: FSR 1 instead of the bilinear blit
        VkImageBlit fb = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {dw, dh, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{dx, dy, 0}, {dx + dw, dy + dh, 1}}};
        vkCmdBlitImage(cb, fimg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_simg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &fb, VK_FILTER_NEAREST);   // same size: a format conversion
    } else
        vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, g_simg[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    if (g_fl.on) fl_record(cb, k, im->img, cx, cy, cw, ch);   // occ-s3fast (DM_VK_FLASHDIAG): the game's picture (before the overlay) reduced for the flash detector
    VkImageLayout lay = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    int ovm = dm_overlay_visible(), menu = dm_menu_state(), fps = dm_overlay_fps(), perf = menu < 0 && perf_ov_on(), perf_drawn = 0;   // r86: the performance panel (not over the menu)
    if ((ovm || menu >= 0 || fps || perf) && overlay_ready(cb, k, menu) && (V.has_dynrender || ov_fb(idx))) {   // without dynamic rendering, only with the image's framebuffer
        if (perf) perf = perf_prepare(cb, k);   // (its upload before the pass)
        barrier(cb, g_simg[idx], lay, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL); lay = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, g_sview[idx], lay, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, g_sext}, 1, 0, 1, &ca, NULL, NULL};
        VkFramebuffer ofb = V.has_dynrender ? VK_NULL_HANDLE : ov_fb(idx);
        if (V.has_dynrender) vkCmdBeginRendering(cb, &ri);
        else { VkRenderPassBeginInfo rbi = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, NULL, g_ov.rp, ofb, {{0, 0}, g_sext}, 0, NULL};
               vkCmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE); }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_ov.layout, 0, 1, &g_ov.set, 0, NULL);
        VkViewport vp = {0, 0, (float)g_sext.width, (float)g_sext.height, 0, 1}; VkRect2D sc = {{0, 0}, g_sext};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &sc);
        ov_render(cb, ovm, menu, fps);
        if (perf) { perf_draw(cb); perf_drawn = 1; }
        if (V.has_dynrender) vkCmdEndRendering(cb); else vkCmdEndRenderPass(cb);
    }
    if (!perf_drawn) perf_ov_hidden();   // (the mouse goes to the game where the panel is not drawn)
    barrier(cb, g_simg[idx], lay, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    vkEndCommandBuffer(cb);
    VkPipelineStageFlags ws = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 1, &g_acq[k], &ws, 1, &cb, 1, &g_done_sem[k]};
    void vk_queue_lock(void), vk_queue_unlock(void);   // gfx_vk_core.c: the host's queue mutex, or NXN1X's Scheduler::submit_mutex
    uint64_t t2 = now_ns(); vk_queue_lock(); pt_add(2, t2, now_ns());
    if (vkQueueSubmit(V.q, 1, &si, g_pfence[k])) { vk_queue_unlock(); VKRED_ONCE(91, "present submit failed"); return; }
    VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, NULL, 1, &g_done_sem[k], 1, &g_swap, &idx, NULL};
    VkPresentTimeGOOGLE ptm = {++g_pace_id, period >= 0 ? pace_desired(period) : 0};
    VkPresentTimesInfoGOOGLE pti = {VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, NULL, 1, &ptm};
    if (V.has_dtiming) pi.pNext = &pti;
    uint64_t t3 = now_ns(); r = vkQueuePresentKHR(V.q, &pi); pt_add(3, t3, now_ns());
    if (V.has_dtiming && r >= 0) present_time_map(ptm.presentID, ticket);
    vk_queue_unlock();
    pt_log();
    if (r == VK_ERROR_OUT_OF_DATE_KHR) swap_destroy();
}
// headless frame shots
static int shot_wanted(uint64_t n) {
    static uint64_t until; const char *k = getenv("DM_VK_SHOTS");   // a DM_CTL "env DM_VK_SHOTS=N" line: each of the next N presents (a transition caught frame by frame)
    if (k && atoi(k) > 0) { until = n + (uint64_t)atoi(k); setenv("DM_VK_SHOTS", "0", 1); VKLOG("frame shots: presents %llu to %llu", (unsigned long long)n, (unsigned long long)until - 1); }
    if (n < until) return 1;
    static const char *e; static int init; if (!init) { init = 1; e = getenv("DM_VK_SHOT"); }
    if (!e) return 0;
    for (const char *p = e; *p;) { char *end; unsigned long long v = strtoull(p, &end, 10); if (end == p) break; if (v == n) return 1; p = *end == ',' ? end + 1 : end; }
    return 0;
}
static void shot(vkimg_t *im, uint64_t n) {
    void mtl_sub_begin(void); uint64_t mtl_sub_commit(void); int mtl_wait(uint64_t, uint64_t);
    uint32_t w = im->w, h = im->h; VkBuffer sb; VkDeviceSize so;
    mtl_sub_begin();
    uint8_t *p = vk_scratch((VkDeviceSize)w * h * 4, 16, &sb, &so);
    VkCommandBuffer cb = vk_cmd(); vk_img_barrier_all(cb);
    VkBufferImageCopy c = {so, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, sb, 1, &c);
    mtl_wait(mtl_sub_commit(), ~0ull);
    const char *dir = getenv("DM_VK_SHOT_DIR"); if (!dir) dir = getenv("DM_ROOT"); if (!dir) dir = ".";
    char path[1024]; snprintf(path, sizeof path, "%s/frame_%06llu.ppm", dir, (unsigned long long)n);
    FILE *f = fopen(path, "wb");
    if (!f) { VKLOG("[RED] cannot write %s", path); return; }
    fprintf(f, "P6\n%u %u\n255\n", w, h);
    int bgra = im->fmt == VK_FORMAT_B8G8R8A8_UNORM || im->fmt == VK_FORMAT_B8G8R8A8_SRGB;
    uint8_t *row = malloc((size_t)w * 3);
    for (uint32_t y = 0; y < h; y++) { const uint8_t *s = p + (size_t)y * w * 4; for (uint32_t x = 0; x < w; x++) { row[3 * x] = s[4 * x + (bgra ? 2 : 0)]; row[3 * x + 1] = s[4 * x + 1]; row[3 * x + 2] = s[4 * x + (bgra ? 0 : 2)]; } fwrite(row, 1, (size_t)w * 3, f); }
    free(row); fclose(f);
    VKLOG("frame %llu written to %s (%ux%u)", (unsigned long long)n, path, w, h);
}
static pthread_mutex_t g_present_mu = PTHREAD_MUTEX_INITIALIZER;   // the swapchain against the window going away (vk_window_lost, on the main thread)
void vk_window_lost(void) { pthread_mutex_lock(&g_present_mu); swap_destroy(); g_win = NULL; pthread_mutex_unlock(&g_present_mu); }
// The present thread: mtl_present only queues the frame and returns (as on Apple), so the game thread never waits in vkAcquireNextImageKHR or on the
// swapchain's fences (a FIFO swapchain blocks there until a vsync); the window texture is not drawn again before its blit, because NVN's acquire waits for
// its ticket (mtl_present_wait), and the blit's submission comes after the frame's own on the one queue. DM_VK_SYNC_PRESENT=1 presents on the game thread.
#define PQ_N 4
static struct { vkimg_t *im; int crop[4], has_crop; uint64_t n; double period; } g_pq[PQ_N];
static uint64_t g_pq_head, g_pq_tail, g_pq_done;   // queued [tail, head), presents up to ticket g_pq_done are blitted
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER;
// While the game is held (the pause menu) no frame comes: dm_overlay_redraw presents the last one again with what the overlay now shows.
static _Atomic int g_redraw;
void dm_overlay_redraw(void) { atomic_store(&g_redraw, 1); pthread_mutex_lock(&g_pq_mu); pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); }
static void *present_thread(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk present");
    typeof(g_pq[0]) last = {0};
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_tail == g_pq_head && !atomic_load(&g_redraw)) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        atomic_store(&g_redraw, 0);
        if (g_pq_tail == g_pq_head) {   // a redraw, no new frame
            pthread_mutex_unlock(&g_pq_mu);
            if (last.im) { pthread_mutex_lock(&g_present_mu); if (swap_make()) present_window(last.im, last.has_crop ? last.crop : NULL, -1, 0); pthread_mutex_unlock(&g_present_mu); }
            continue;
        }
        typeof(g_pq[0]) e = g_pq[g_pq_tail % PQ_N];
        pthread_mutex_unlock(&g_pq_mu);
        if (e.im) { last = e; pthread_mutex_lock(&g_present_mu); if (swap_make()) present_window(e.im, e.has_crop ? e.crop : NULL, e.period, e.n); pthread_mutex_unlock(&g_present_mu); }
        pthread_mutex_lock(&g_pq_mu); g_pq_tail++; g_pq_done = e.n; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
    }
    return NULL;
}
static void present_thread_start(void) { pthread_t t; pthread_create(&t, NULL, present_thread, NULL); pthread_detach(t); }
// s58 (XC3 perf): the window texture's copy into a ring slot of the host's own, so the game's acquire does not wait for NXN1X's compositor.
// On the Switch nvnWindowAcquireTexture returns at once and the GPU waits for the texture; here the acquire waited (mtl_present_wait) until the
// compositor had taken the present at a 60 Hz vsync, composited it behind newer frames and copied it: with 2 window textures the game thread
// stood still for the GPU and the vsync every frame (Xenoblade 3 / BotW fields at 30-40 fps). Now present_out copies the texture into slot
// (g_pq_head % PQ_N) on the same queue, right after the frame's own submissions (queue order; the barrier after the copy keeps the game's later
// writes behind it) and hands that slot's image to the compositor; the window texture is free at once (g_s58_done). A slot is used again only
// after NXN1X gave its present back (at most PQ_N outstanding, given back in order) and its own fence signalled. DM_VK_S58=0: as before
static uint64_t g_s58_done;   // under g_pq_mu: the window textures of presents up to this ticket are free again (their copy is queued)
static int g_s58_dead;        // a texture could not be copied: s58 off from then on (the compositor reads the window texture, as before)
static int s58_on(void) {
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_VK_S58"); v = e ? atoi(e) : 0; if (v < 0 || v > 2) v = 0;   // s58c: default off (loading failures, runs 81-84); 1 full, 2 copy only
        VKLOG("s58: %s", v == 1 ? "the window texture is copied into a ring slot at present and goes back to the game at once (DM_VK_S58=1)" : v == 2 ? "ring copy only, the window texture goes back with NXN1X's compositor as before (DM_VK_S58=2)" : "off (DM_VK_S58 unset or 0): the acquire waits for NXN1X's compositor"); }
    return g_s58_dead ? 0 : v;
}
static int s58_free(uint64_t tk) { return g_s58_done >= tk && g_pq_head - g_pq_tail < PQ_N; }   // under g_pq_mu: copied, and the next present has a ring slot
static struct { VkImage img; VkDeviceMemory mem; uint32_t w, h; VkFormat fmt; int fresh; VkCommandBuffer cb; VkFence fence; } g_s58[PQ_N];
static VkCommandPool g_s58_pool;
static uint64_t g_s58_copies, g_s58_fwait_ns;
static __attribute__((unused)) void s58_off(const char *why, int a, int b) { g_s58_dead = 1; VKLOG("[RED] s58: %s (%d, %d): off, the compositor reads the window texture as before", why, a, b); }
static __attribute__((unused)) VkImage s58_copy(vkimg_t *im, int k) {   // the presenting thread (game thread, or the s46 worker); slot k is free
    void vk_queue_lock(void), vk_queue_unlock(void);
    if (!(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || im->samples > 1 || !im->w || !im->h) { s58_off("the window texture cannot be copied", (int)im->usage, (int)im->samples); return VK_NULL_HANDLE; }
    if (!g_s58_pool) {
        VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, V.qf};
        if (vkCreateCommandPool(V.dev, &pci, NULL, &g_s58_pool)) { g_s58_pool = VK_NULL_HANDLE; s58_off("no command pool", 0, 0); return VK_NULL_HANDLE; }
        VkCommandBuffer cbs[PQ_N]; VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, g_s58_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, PQ_N};
        if (vkAllocateCommandBuffers(V.dev, &ai, cbs)) { s58_off("no command buffers", 0, 0); return VK_NULL_HANDLE; }
        for (int i = 0; i < PQ_N; i++) { g_s58[i].cb = cbs[i]; VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, VK_FENCE_CREATE_SIGNALED_BIT};
            if (vkCreateFence(V.dev, &fi, NULL, &g_s58[i].fence)) { s58_off("no fence", i, 0); return VK_NULL_HANDLE; } }
    }
    uint64_t w0 = now_ns(); vkWaitForFences(V.dev, 1, &g_s58[k].fence, 1, UINT64_MAX); g_s58_fwait_ns += now_ns() - w0;   // (signalled long ago: NXN1X's copy of it came later and finished)
    if (!g_s58[k].img || g_s58[k].w != im->w || g_s58[k].h != im->h || g_s58[k].fmt != im->fmt) {
        if (g_s58[k].img) { vkDestroyImage(V.dev, g_s58[k].img, NULL); vkFreeMemory(V.dev, g_s58[k].mem, NULL); g_s58[k].img = VK_NULL_HANDLE; g_s58[k].mem = VK_NULL_HANDLE; }
        VkImageCreateInfo ic = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, im->fmt, {im->w, im->h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
        if (vkCreateImage(V.dev, &ic, NULL, &g_s58[k].img)) { g_s58[k].img = VK_NULL_HANDLE; s58_off("no ring image", (int)im->fmt, k); return VK_NULL_HANDLE; }
        VkMemoryRequirements rq; vkGetImageMemoryRequirements(V.dev, g_s58[k].img, &rq);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, rq.size, vk_mem_type(rq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        if (vkAllocateMemory(V.dev, &mai, NULL, &g_s58[k].mem) || vkBindImageMemory(V.dev, g_s58[k].img, g_s58[k].mem, 0)) {
            if (g_s58[k].mem) vkFreeMemory(V.dev, g_s58[k].mem, NULL); vkDestroyImage(V.dev, g_s58[k].img, NULL); g_s58[k].img = VK_NULL_HANDLE; g_s58[k].mem = VK_NULL_HANDLE;
            s58_off("no memory for a ring image", (int)(rq.size >> 20), k); return VK_NULL_HANDLE; }
        g_s58[k].w = im->w; g_s58[k].h = im->h; g_s58[k].fmt = im->fmt; g_s58[k].fresh = 1;
        VKLOG("s58: ring slot %d %ux%u format %d", k, im->w, im->h, im->fmt);
    }
    VkCommandBuffer cb = g_s58[k].cb;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (vkBeginCommandBuffer(cb, &bi)) { s58_off("begin failed", k, 0); return VK_NULL_HANDLE; }
    VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier pre[2] = {
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img, rg},   // the frame's writes (earlier submissions on this queue)
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, g_s58[k].fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_s58[k].img, rg}};   // after NXN1X's earlier read of the slot
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 2, pre);
    VkImageCopy c = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {im->w, im->h, 1}};
    vkCmdCopyImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, g_s58[k].img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    VkImageMemoryBarrier post[2] = {
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, 0, 0, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img, rg},   // write after read: the game's next writes wait for the copy
        {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, g_s58[k].img, rg}};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 2, post);
    if (vk_f0_result(vkEndCommandBuffer(cb), "vkEndCommandBuffer(present-copy)", 0)) { s58_off("end failed", k, 0); return VK_NULL_HANDLE; }
    vkResetFences(V.dev, 1, &g_s58[k].fence);
    { uint64_t vk_submit_extra(VkCommandBuffer cb, VkFence fence); vk_submit_extra(cb, g_s58[k].fence); }   // s58b: on the host timeline, so frees during loading wait for the copy
    g_s58[k].fresh = 0;
    if (++g_s58_copies == 1 || g_s58_copies % 3600 == 0) VKLOG("s58: %llu presents copied into the ring (%d slots), waits for a slot's fence %.2f ms in all", (unsigned long long)g_s58_copies, PQ_N, g_s58_fwait_ns / 1e6);
    return g_s58[k].img;
}
// s58b: one present at a time through s58_copy (its command pool and slots need external synchronization); two at once is logged
static __attribute__((unused)) VkImage s58_copy_one(vkimg_t *im, int k) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static int said;
    if (pthread_mutex_trylock(&mu)) { if (!said++) VKLOG("[RED] s58b: two threads presented at once (serialized)"); pthread_mutex_lock(&mu); }
    VkImage r = s58_copy(im, k); pthread_mutex_unlock(&mu); return r;
}
static __attribute__((unused)) void s58_give_back(uint64_t n) { pthread_mutex_lock(&g_pq_mu); if (n > g_s58_done) g_s58_done = n; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); }
#if defined(DM_NXN1X)
// Step 6 (NVN_HOST_BRIDGE.md §4.5): under NXN1X the presents go to its compositor (nvnh_set_output; core/nvn/nvn_output.cpp, video_core
// nvn_display.cpp) instead of this file's window, swapchain and present thread. The ticket bookkeeping stays: NXN1X gives each present
// back in order (nvnh_present_done) once it has copied the image, and the acquire's mtl_present_wait waits for that as it waited for the blit
static void (*_Atomic g_out)(const nvnh_frame *f);
void nvnh_set_output(void (*fn)(const nvnh_frame *f)) { atomic_store(&g_out, fn); }
int nvnh_window_image_returned(uint64_t ticket) {
    if (!ticket || !vk_is_n1x() || !atomic_load(&g_out) || s58_on()) return 0;
    pthread_mutex_lock(&g_pq_mu);
    int returned = g_pq_done >= ticket;
    pthread_mutex_unlock(&g_pq_mu);
    return returned;
}
void nvnh_present_done(uint64_t n) {
    pthread_mutex_lock(&g_pq_mu);
    if (g_pq_tail < g_pq_head) g_pq_tail++;
    if (n > g_pq_done) g_pq_done = n;
    pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
    vk_f0_present_done(n);
}
static int present_out(vkimg_t *im, const int crop[4], uint64_t n, double period) {
    void (*out)(const nvnh_frame *) = atomic_load(&g_out);
    if (!out) return 0;
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_head - g_pq_tail >= PQ_N) {   // never more than PQ_N presents outstanding, as with the present thread
        struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_sec += 2;
        if (pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &dl) != 0) {
            static int said; if (!said++) VKLOG("[RED] present %llu: NXN1X's compositor gave no present back for 2 s (%llu outstanding), going on", (unsigned long long)n, (unsigned long long)(g_pq_head - g_pq_tail));
            break;
        }
    }
    int s58_slot = (int)(g_pq_head % PQ_N);   // s58: this present's ring slot (the present that used it before was given back: fewer than PQ_N outstanding)
    g_pq_head++;
    pthread_mutex_unlock(&g_pq_mu);
    uint64_t mtl_last_fence(void);
    nvnh_frame f; memset(&f, 0, sizeof f);
    f.n = n; f.period = period; f.fence = mtl_last_fence();
    VkImage s58_img = VK_NULL_HANDLE;   // s58: the ring slot's copy, shown instead of the window texture
    if (im) { 
        f.scaled_2x = im->scaled_valid && im->scaled;
        vkimg_t *display = f.scaled_2x ? im->scaled : im;
        s58_img = s58_on() ? s58_copy_one(display, s58_slot) : VK_NULL_HANDLE; f.image = (uint64_t)(uintptr_t)(s58_img ? s58_img : display->img); f.vk_format = (uint32_t)im->fmt; f.width = im->w; f.height = im->h; f.bottom_up = vk_window_bottom_up(); }
    if (crop) { memcpy(f.crop, crop, sizeof f.crop); f.has_crop = 1; }
    // Sample exactly one presented image. The output bridge currently carries final color only.
    static uint64_t audit_frame;
    if (n == 1) {
        const char *requested = getenv("NXN1X_DLSS_AUDIT_FRAME");
        audit_frame = requested ? strtoull(requested, NULL, 10) : 0;
    }
    if (audit_frame && n == audit_frame && im) {
        const vkimg_t *display = f.scaled_2x ? im->scaled : im;
        VKLOG("DLSS inputs: ticket=%llu final=%p format=%d extent=%ux%u fence=%llu FG=off (unsupported inputs: no frame-matched depth, motion, camera or UI in nvnh_frame)",
              (unsigned long long)n, (void *)(uintptr_t)f.image, display->fmt, display->w, display->h,
              (unsigned long long)f.fence);
    }
    nvnh_guard_check();
    out(&f);
    if (s58_on() == 1 && (!im || s58_img)) s58_give_back(n);   // s58c: mode 2 keeps the compositor's give-back   // s58: the window texture is free for the game (its copy is queued)
    return 1;
}
#endif
extern _Atomic uint64_t dm_present_ns;   // r92: hle_thread.c DM_HITCH_DUMP watches it; only nvn_mtl.m (Mac) wrote it, so the stall dump never fired on Windows
static uint64_t present_enqueue(vkimg_t *im, const int crop[4], uint64_t n, double period) {   // im NULL: nothing to show, the ticket still completes in order
    atomic_store(&dm_present_ns, now_ns());
    static int sync = -1; static pthread_once_t once = PTHREAD_ONCE_INIT;
    if (sync < 0) sync = getenv("DM_VK_SYNC_PRESENT") != NULL;
#if defined(DM_NXN1X)
    if (present_out(im, crop, n, period)) return n;   // NXN1X's compositor shows it
#endif
    if (sync || !dm_front_window()) {   // headless: nothing to show; or presenting on this thread
        if (sync && im) { pthread_mutex_lock(&g_present_mu); if (swap_make()) present_window(im, crop, period, n); pthread_mutex_unlock(&g_present_mu); }
        pthread_mutex_lock(&g_pq_mu); if (g_pq_tail == g_pq_head) g_pq_done = n; pthread_mutex_unlock(&g_pq_mu);   // done at once (after any queued ones)
        return n;
    }
    pthread_once(&once, present_thread_start);
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_head - g_pq_tail >= PQ_N) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // never more than PQ_N frames behind
    typeof(g_pq[0]) *e = &g_pq[g_pq_head % PQ_N]; e->im = im; e->n = n; e->period = period; e->has_crop = crop != NULL; if (crop) memcpy(e->crop, crop, sizeof e->crop);
    g_pq_head++; pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    return n;
}
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    vk_init();
    uint64_t n = ++g_presents;
#if defined(DM_NXN1X)
    extern void vk_video_use(const tex_t *, int);
    vk_video_use(t, 1);
    nvn_fg_seal(n);
#endif
    // without display times (no VK_GOOGLE_display_timing, no window, or none came back) the hand-over time stands in, as before: the governor's late check needs readings
    { int disp = V.has_dtiming && dm_front_window() && (n <= 240 || g_pres_recorded); struct timespec ts; if (!disp) clock_gettime(CLOCK_MONOTONIC, &ts);
      present_time_begin(n, disp ? 0 : (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec); }
    vkimg_t *im = vk_tex(t);
    static uint64_t no_img, undrawn, shown;
    if (n % 600 == 0) {
        extern struct { uint64_t draws, skip_prog, skip_rt, skip_pipe, passes, folded, dispatches, pipes_built, rt_kept, ns_exec, ns_desc, ns_pipe, max_pass_draws; } g_vkst; extern uint64_t g_ubd_landed, g_ubd_superseded, g_ubd_shadows, g_ubd_bytes, g_ns_submit, g_gpu_ns, g_pkg_n, g_pkg_bytes, g_pkg_ns, g_set_hits, g_set_new, g_ubd_racy, g_ns_set_build, g_ns_set_hash, g_ns_set_new, g_set_memo, g_set_memo_bad;
        static uint64_t psb, psh, psn, pe, pd, pp, ps, pb, pdr, pg, pkn, pkb, pkt, psm;
        VKLOG("replay up to present %llu: draws %llu, skipped (no program %llu, no target %llu, no pipeline %llu), passes %llu, clears folded %llu, dispatches %llu, pipelines %llu; presents shown %llu, no image %llu, never drawn %llu; uniform updates landed %llu (superseded %llu, shadows %llu, CPU reads racing a GPU write %llu), same targets kept %llu, most draws in one pass %llu",
              (unsigned long long)n, (unsigned long long)g_vkst.draws, (unsigned long long)g_vkst.skip_prog, (unsigned long long)g_vkst.skip_rt, (unsigned long long)g_vkst.skip_pipe,
              (unsigned long long)g_vkst.passes, (unsigned long long)g_vkst.folded, (unsigned long long)g_vkst.dispatches, (unsigned long long)g_vkst.pipes_built,
              (unsigned long long)shown, (unsigned long long)no_img, (unsigned long long)undrawn, (unsigned long long)g_ubd_landed, (unsigned long long)g_ubd_superseded, (unsigned long long)g_ubd_shadows, (unsigned long long)g_ubd_racy, (unsigned long long)g_vkst.rt_kept, (unsigned long long)g_vkst.max_pass_draws);
        double nd = g_vkst.draws > pdr ? (double)(g_vkst.draws - pdr) / 600 : 0;
        VKLOG("replay per frame (last 600): %.0f draws, replay %.2f ms (descriptor sets %.2f, pipeline keys %.2f), queue submit %.2f ms, uniform shadows %.0f KB; GPU busy %.2f ms; texture uploads %.1f (%.2f MB, %.2f ms); descriptor sets reused %llu, new %llu (collecting %.2f ms, hashing %.2f, new ones %.2f), kept per stage %.0f a frame (memo differences %llu)",
              nd, (g_vkst.ns_exec - pe) / 6e8, (g_vkst.ns_desc - pd) / 6e8, (g_vkst.ns_pipe - pp) / 6e8, (g_ns_submit - ps) / 6e8, (g_ubd_bytes - pb) / 600.0 / 1024, (g_gpu_ns - pg) / 6e8, (g_pkg_n - pkn) / 600.0, (g_pkg_bytes - pkb) / 600.0 / 1048576, (g_pkg_ns - pkt) / 6e8, (unsigned long long)g_set_hits, (unsigned long long)g_set_new, (g_ns_set_build - psb) / 6e8, (g_ns_set_hash - psh) / 6e8, (g_ns_set_new - psn) / 6e8,
              (g_set_memo - psm) / 600.0, (unsigned long long)g_set_memo_bad);
        { extern void vk_r80_log(void); vk_r80_log(); }   // r80 (diagnostic): viewports
        { extern void vk_rd_log(double); vk_rd_log(600); }   // replaydiag (DM_VK_REPLAYDIAG=1)
        { extern uint64_t g_ns_subwait, g_ns_poolsync, g_nsubs; static uint64_t w0, ys0, n0;   // r30: how much of a frame the replay spends waiting for the GPU at each submission's start
          VKLOG("submissions per frame (last 600): %.1f; their start waited for the last submission %.2f ms (copied pools), syncing the copies %.2f ms", (g_nsubs - n0) / 600.0, (g_ns_subwait - w0) / 6e8, (g_ns_poolsync - ys0) / 6e8);
          w0 = g_ns_subwait; ys0 = g_ns_poolsync; n0 = g_nsubs; }
        pe = g_vkst.ns_exec; pd = g_vkst.ns_desc; pp = g_vkst.ns_pipe; ps = g_ns_submit; pb = g_ubd_bytes; pdr = g_vkst.draws; pg = g_gpu_ns; pkn = g_pkg_n; pkb = g_pkg_bytes; pkt = g_pkg_ns; psb = g_ns_set_build; psh = g_ns_set_hash; psn = g_ns_set_new; psm = g_set_memo;
    }
    if (!im) { no_img++; return present_enqueue(NULL, NULL, n, period); }
    if (im->layout != VK_IMAGE_LAYOUT_GENERAL) { undrawn++; return present_enqueue(NULL, NULL, n, period); }   // never drawn into: nothing to show yet
    shown++; atomic_fetch_add(&g_fps_frames, 1);
    if (shot_wanted(n)) shot(im->scaled_valid && im->scaled ? im->scaled : im, n);
    { void vk_pd_present(vkimg_t *); vk_pd_present(im); }   // NVN2 diagnostic (gfx_vk_exec.c): the image handed to the screen, at a pass dump's sampled presents
    { void vk_pass_dump_poll(uint64_t done); uint64_t vk_fence_done(void); vk_pass_dump_poll(vk_fence_done()); void vk_pass_frame(void); vk_pass_frame(); void vk_tex_sweep(void); vk_tex_sweep(); }
    if (n % 1800 == 0) {   // every ~30 s: the pipeline cache to disk on a thread of its own (the window can be closed without any exit path)
        void vk_pipeline_cache_save(void); pthread_t th; pthread_attr_t a; pthread_attr_init(&a); pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
        pthread_create(&th, &a, (void *(*)(void *))(void *)vk_pipeline_cache_save, NULL); pthread_attr_destroy(&a);
    }
    return present_enqueue(im, crop, n, period);
}
int mtl_present_done(uint64_t tk) {   // s53: mtl_present_wait's condition, without waiting
    pthread_mutex_lock(&g_pq_mu); int d = g_pq_done >= tk || s58_free(tk); pthread_mutex_unlock(&g_pq_mu); return d;   // s58
}
void mtl_present_wait(uint64_t tk) {   // NVN acquire: the window texture's last present has been blitted (the game may draw into it again)
    if (!tk) return;
    pthread_mutex_lock(&g_pq_mu);
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl); dl.tv_sec += 1;
    while (g_pq_done < tk && !s58_free(tk) && pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &dl) == 0) {}   // s58: or its copy is queued
    pthread_mutex_unlock(&g_pq_mu);
}
double mtl_presented_time(uint64_t n) { return present_time_get(n); }
double vk_refresh_ms(void) { return V.has_dtiming && g_pres_recorded && g_refresh_ns ? g_refresh_ns / 1e6 : 0; }   // perf_android.c's late check: the display's refresh, when display times are what it gets
uint32_t mtl_present_qos(void) { return 0; }
#include <windows.h>
#include <errno.h>
// S1e (parallel recording): a recording worker thread (gfx_vk_core.c par_main) on the host's fastest cores: the CPU sets of the highest EfficiencyClass (the N1X's P-cores), parked
// ones included, as Eden's guest-core threads (source/src/common/thread.cpp PerformanceCpuSets, pcore / pcoreset2), and the power-throttling hint off (HighQoS). A worker on an
// E-core makes its segment the part's slowest (design 0.4). Nothing is set on a host with one core class. DM_VK_PARALLEL_PIN=0 (read at device init) skips the call
void vk_par_pin_self(int worker) {
    static ULONG ids[256]; static int nid = -1; static BYTE cls; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static char lps[400]; int o = 0;
    pthread_mutex_lock(&mu);
    if (nid < 0) { nid = 0; ULONG need = 0; GetSystemCpuSetInformation(NULL, 0, &need, GetCurrentProcess(), 0);
        uint8_t *buf = need ? malloc(need) : NULL;
        if (buf && GetSystemCpuSetInformation((PSYSTEM_CPU_SET_INFORMATION)buf, need, &need, GetCurrentProcess(), 0)) {
            BYTE lo = 0xff, hi = 0;
            for (ULONG off = 0; off < need;) { const SYSTEM_CPU_SET_INFORMATION *s = (const SYSTEM_CPU_SET_INFORMATION *)(buf + off); if (s->Type == CpuSetInformation) { if (s->CpuSet.EfficiencyClass < lo) lo = s->CpuSet.EfficiencyClass; if (s->CpuSet.EfficiencyClass > hi) hi = s->CpuSet.EfficiencyClass; } off += s->Size; }
            if (hi != lo) for (ULONG off = 0; off < need;) { const SYSTEM_CPU_SET_INFORMATION *s = (const SYSTEM_CPU_SET_INFORMATION *)(buf + off);
                if (s->Type == CpuSetInformation && s->CpuSet.EfficiencyClass == hi && nid < 256) { ids[nid++] = s->CpuSet.Id; if (o < (int)sizeof lps - 8) o += snprintf(lps + o, sizeof lps - (size_t)o, " %u", (unsigned)s->CpuSet.LogicalProcessorIndex); }
                off += s->Size; }
            cls = hi; }
        free(buf); }
    const int n = nid; const BYTE c = cls;
    pthread_mutex_unlock(&mu);
    THREAD_POWER_THROTTLING_STATE st = {THREAD_POWER_THROTTLING_CURRENT_VERSION, THREAD_POWER_THROTTLING_EXECUTION_SPEED, 0};
    (void)SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &st, sizeof st);
    if (n <= 0) { if (!worker) VKLOG("S1e: recording workers not pinned: one core class (no P-core CPU sets)"); return; }
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), ids, (ULONG)n)) { VKLOG("[RED] S1e: recording worker %d: SetThreadSelectedCpuSets failed (%lu): not pinned", worker, (unsigned long)GetLastError()); return; }
    if (!worker) VKLOG("S1e: recording workers on the efficiency class %u CPU sets (%d, logical processors%s; DM_VK_PARALLEL_PIN=0 off)", (unsigned)c, n, lps[0] ? lps : " ?");
}
// r99 display link on Windows (nvn_mtl.m's mtl_link_* contract, ticks only: presents stay on the present thread's FIFO swapchain). Before, these were stubs, so the
// game was paced by the host's own 60 Hz sleep clock (hle_nvn.c nvnWindowAcquireTexture) while FIFO showed frames on the screen's vblanks: the two clocks drifted and
// Bayonetta 3 at 60 fps doubled 3-5 frames of every 300 (p99 25-35 ms, 11-08). A thread waits for each vblank of the window's screen (D3DKMTWaitForVerticalBlankEvent,
// gdi32, looked up at run time) and counts it as a tick; the acquire then waits for the present interval's ticks as on the Mac. Only on a 59-61 Hz screen (other rates
// would need the Mac's virtual ticks), and DM_VK_LINK=0 keeps the old clock. A tick missing for 100 ms (screen off, adapter lost) makes the link stalled / inactive,
// so the acquire falls back to its own clock
typedef struct { HDC hDc; UINT hAdapter; LUID AdapterLuid; UINT VidPnSourceId; } dmkt_open_hdc;
typedef struct { UINT hAdapter; UINT hDevice; UINT VidPnSourceId; } dmkt_wait_vblank;
typedef struct { UINT hAdapter; } dmkt_close;
typedef LONG (APIENTRY *pfn_kt_open)(dmkt_open_hdc *);
typedef LONG (APIENTRY *pfn_kt_wait)(const dmkt_wait_vblank *);
typedef LONG (APIENTRY *pfn_kt_close)(const dmkt_close *);
static pthread_mutex_t g_vbl_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_vbl_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_vbl_n;                 // ticks so far (under g_vbl_mu)
static _Atomic uint64_t g_vbl_ns;        // when the last tick came (now_ns)
static _Atomic int g_vbl_on;             // the thread runs and ticks
static int g_vbl_hz;
// r106 (OoT 11-09, r105 log): the vblank wait still came back one to four vblanks late about once a second at time-critical priority, and ticks counted late do not
// help: the vsync event they signal is one flag, so two quick signals are one frame for the game. The ticks now come from a timer thread at the vblanks the waits
// predict (the last on-time vblank plus whole periods); the waits only keep that phase and period. DM_VK_LINK_PLL=0 ticks from the waits as r105
static _Atomic uint64_t g_vbl_obs, g_vbl_per;   // the last vblank seen on time (now_ns), the measured period
static int g_vbl_pll = -1;
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
static void *vbl_ticker(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    HANDLE tm = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!tm) tm = CreateWaitableTimerExW(NULL, NULL, 0, TIMER_ALL_ACCESS);
    uint64_t prev = 0; int late = 0;
    for (;;) {
        uint64_t obs = atomic_load(&g_vbl_obs), per = atomic_load(&g_vbl_per), now = now_ns();
        if (!obs || !per) { Sleep(5); continue; }
        uint64_t from = prev ? prev + per / 2 : now; if (from < now) from = now;
        uint64_t k = from > obs ? (from - obs + per - 1) / per : 1, tgt = obs + k * per;
        // r106b (NPARK 11-09 "엄청 버벅", r106 log): when the screen itself stalls (waits back 50-66 ms late, presents blocked 50-60 ms), ticks at the predicted
        // vblanks ran the game ahead of the screen and its frames bunched. The ticker now stands in for at most two vblanks after the last one seen, then waits for the next
        if (tgt > obs + per * 5 / 2) { Sleep(1); continue; }
        if (tgt > now) { LARGE_INTEGER due; due.QuadPart = -(LONGLONG)((tgt - now) / 100); if (!due.QuadPart) due.QuadPart = -1;
            if (tm && SetWaitableTimer(tm, &due, 0, NULL, NULL, FALSE)) WaitForSingleObject(tm, 100); else Sleep((DWORD)((tgt - now) / 1000000)); }
        uint64_t t = now_ns();
        if (t > tgt + 5000000ull && (late++ < 20 || !(late & (late - 1)))) VKLOG("r106: vblank ticker woke %.1f ms late (%d so far)", (t - tgt) / 1e6, late);
        prev = tgt;
        atomic_store(&g_vbl_ns, t); atomic_store(&g_vbl_on, 1);
        pthread_mutex_lock(&g_vbl_mu); g_vbl_n++; pthread_cond_broadcast(&g_vbl_cv); pthread_mutex_unlock(&g_vbl_mu);
    }
    return NULL;
}
static int vbl_screen_hz(void) {
    DEVMODEA dm; memset(&dm, 0, sizeof dm); dm.dmSize = sizeof dm;
    HWND w = dm_front_window(); char dev[CCHDEVICENAME] = "";
    if (w) { HMONITOR m = MonitorFromWindow(w, MONITOR_DEFAULTTOPRIMARY); MONITORINFOEXA mi; memset(&mi, 0, sizeof mi); mi.cbSize = sizeof mi; if (m && GetMonitorInfoA(m, (MONITORINFO *)&mi)) memcpy(dev, mi.szDevice, sizeof dev); }
    return EnumDisplaySettingsA(dev[0] ? dev : NULL, ENUM_CURRENT_SETTINGS, &dm) ? (int)dm.dmDisplayFrequency : 0;
}
static void *vbl_main(void *arg) {
    (void)arg;
    HMODULE g = LoadLibraryA("gdi32.dll");
    pfn_kt_open op = g ? (pfn_kt_open)(void *)GetProcAddress(g, "D3DKMTOpenAdapterFromHdc") : NULL;
    pfn_kt_wait wt = g ? (pfn_kt_wait)(void *)GetProcAddress(g, "D3DKMTWaitForVerticalBlankEvent") : NULL;
    pfn_kt_close cl = g ? (pfn_kt_close)(void *)GetProcAddress(g, "D3DKMTCloseAdapter") : NULL;
    if (!op || !wt) { VKLOG("r99: display link off: gdi32 has no D3DKMTWaitForVerticalBlankEvent"); return NULL; }
    HWND w = dm_front_window(); char dev[CCHDEVICENAME] = "";
    if (w) { HMONITOR m = MonitorFromWindow(w, MONITOR_DEFAULTTOPRIMARY); MONITORINFOEXA mi; memset(&mi, 0, sizeof mi); mi.cbSize = sizeof mi; if (m && GetMonitorInfoA(m, (MONITORINFO *)&mi)) memcpy(dev, mi.szDevice, sizeof dev); }
    HDC dc = dev[0] ? CreateDCA(NULL, dev, NULL, NULL) : GetDC(NULL);
    dmkt_open_hdc oa; memset(&oa, 0, sizeof oa); oa.hDc = dc;
    LONG st = dc ? op(&oa) : -1; if (dc) { if (dev[0]) DeleteDC(dc); else ReleaseDC(NULL, dc); }
    if (st) { VKLOG("r99: display link off: D3DKMTOpenAdapterFromHdc %ld", (long)st); return NULL; }
    VKLOG("r99: display link on: vblanks of %s (%d Hz) pace the game (DM_VK_LINK=0 off)", dev[0] ? dev : "the primary screen", g_vbl_hz);
    dmkt_wait_vblank wv = {oa.hAdapter, 0, oa.VidPnSourceId}; int fails = 0;
    // r105 (OoT 11-09, r104 log): about once a second this wait returned 33 ms after the previous vblank, one vblank unseen, and the game (on the vsync event and the
    // display link's ticks) then made 59 frames for 60 vblanks: the movie's "beat". The thread now runs at time-critical priority, and a wait that comes back more than
    // 1.5 vblanks late counts the vblanks it missed (at most 4), so the ticks stay one per vblank; DM_VK_LINK_CATCHUP=0 counts one per wait as before
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    const char *ce = getenv("DM_VK_LINK_CATCHUP"); int catchup = !(ce && atoi(ce) == 0); uint64_t vprev = 0, vavg = 16666667ull, vmissed = 0; int vlog = 0;
    for (;;) {
        st = wt(&wv);
        if (st) { if (++fails > 50) { VKLOG("[RED] r99: display link stopped: D3DKMTWaitForVerticalBlankEvent %ld", (long)st); break; } Sleep(16); continue; }
        fails = 0;
        if (!g_vbl_pll) { atomic_store(&g_vbl_ns, now_ns()); atomic_store(&g_vbl_on, 1); }   // (r106: the ticker sets them)
        uint64_t vt = now_ns(), add = 1;   // r105: the vblanks since the previous wait came back
        if (vprev) { uint64_t d = vt - vprev;
            if (d < vavg * 3 / 2) { if (d > vavg / 2) vavg = (uint64_t)((int64_t)vavg + ((int64_t)d - (int64_t)vavg) / 32); }
            else if (catchup) { add = (d + vavg / 2) / vavg; if (add > 5) add = 1;
                if (add > 1) { vmissed += add - 1; if (vlog++ < 20 || !(vlog & (vlog - 1))) VKLOG("r105: vblank wait came back %.1f ms after the previous one: %llu missed counted (%llu so far)", d / 1e6, (unsigned long long)(add - 1), (unsigned long long)vmissed); } } }
        vprev = vt;
        if (g_vbl_pll) {   // r106: the wait keeps the phase (only one that comes back within 2 ms of a whole number of periods) and the period; the ticker ticks
            uint64_t o = atomic_load(&g_vbl_obs), per = vavg;
            if (!o) atomic_store(&g_vbl_obs, vt);
            else { uint64_t m = (vt - o + per / 2) / per, err = vt > o + m * per ? vt - o - m * per : o + m * per - vt; if ((m && err < 2000000ull) || vt - o > 4 * per) atomic_store(&g_vbl_obs, vt); }   // (r106b: or after a stall, any)
            atomic_store(&g_vbl_per, per);
            continue; }
        pthread_mutex_lock(&g_vbl_mu); g_vbl_n += add; pthread_cond_broadcast(&g_vbl_cv); pthread_mutex_unlock(&g_vbl_mu);
    }
    atomic_store(&g_vbl_on, 0);
    if (cl) { dmkt_close c = {oa.hAdapter}; cl(&c); }
    return NULL;
}
static void vbl_start(void) {
    const char *e = getenv("DM_VK_LINK"); if (e && atoi(e) == 0) return;
    g_vbl_hz = vbl_screen_hz();
    if (g_vbl_hz < 59 || g_vbl_hz > 61) { VKLOG("r99: display link off: the screen runs at %d Hz (only 60 Hz screens tick the game)", g_vbl_hz); return; }
    { const char *pe = getenv("DM_VK_LINK_PLL"); g_vbl_pll = !(pe && atoi(pe) == 0); }   // r106
    pthread_t th; if (pthread_create(&th, NULL, vbl_main, NULL) == 0) pthread_detach(th);
    if (g_vbl_pll) { pthread_t tk; if (pthread_create(&tk, NULL, vbl_ticker, NULL) == 0) pthread_detach(tk); VKLOG("r106: display link ticks from a timer at the predicted vblanks (DM_VK_LINK_PLL=0 off)"); }
}
static pthread_once_t g_vbl_once = PTHREAD_ONCE_INIT;
int mtl_link_stalled(void) { uint64_t t = atomic_load(&g_vbl_ns); return t && now_ns() - t > 100000000ull; }
int mtl_link_active(void) {
    static int started; if (!started) { if (!dm_front_window()) return 0; started = 1; pthread_once(&g_vbl_once, vbl_start); }   // (acquire thread only)
    return atomic_load(&g_vbl_on) && !mtl_link_stalled();
}
int mtl_link_rate(void) { return 60; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) {   // nvn_mtl.m: go at once when the ticks already passed since the previous frame started (behind), else wait for them
    if (!mtl_link_active()) return 0;
    static uint64_t started_tick; const uint64_t step = ticks < 1 ? 1 : (uint64_t)ticks;
    pthread_mutex_lock(&g_vbl_mu);
    if (g_vbl_n >= started_tick + step) { started_tick = g_vbl_n; pthread_mutex_unlock(&g_vbl_mu); return 1; }
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ns / 1000000000ull; ts.tv_nsec += timeout_ns % 1000000000ull; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    while (g_vbl_n < started_tick + step) if (pthread_cond_timedwait(&g_vbl_cv, &g_vbl_mu, &ts) == ETIMEDOUT) { pthread_mutex_unlock(&g_vbl_mu); return 0; }
    started_tick = g_vbl_n; pthread_mutex_unlock(&g_vbl_mu); return 1;
}
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { if (!mtl_link_active()) return 0; return mtl_link_wait_ticks(vsyncs < 1 ? 1 : vsyncs, timeout_ns); }
int mtl_link_wait_vsync(uint64_t *seen, uint64_t timeout_ns) {   // vi::GetDisplayVsyncEvent (hle_sdk16.c), as nvn_mtl.m at 60 Hz: the next tick after *seen
    if (!mtl_link_active()) return 0;
    pthread_mutex_lock(&g_vbl_mu);
    if (*seen > g_vbl_n || g_vbl_n > *seen + 4) *seen = g_vbl_n;   // first call, or far behind (a stall): from the current tick on, no burst
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ns / 1000000000ull; ts.tv_nsec += timeout_ns % 1000000000ull; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    while (g_vbl_n < *seen + 1) if (pthread_cond_timedwait(&g_vbl_cv, &g_vbl_mu, &ts) == ETIMEDOUT) { pthread_mutex_unlock(&g_vbl_mu); return 0; }
    *seen += 1; pthread_mutex_unlock(&g_vbl_mu); return 1;
}
// s15: this thread's cycles (user and kernel) and the process's page faults, for gfx_vk_core.c's slow-submission line
#include <psapi.h>
void vk_ss_counts(uint64_t *cyc, uint64_t *faults) {
    ULONG64 c = 0; QueryThreadCycleTime(GetCurrentThread(), &c); *cyc = c;
    PROCESS_MEMORY_COUNTERS m = {sizeof m}; *faults = K32GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof m) ? m.PageFaultCount : 0;
}
// occ-s2 (DM_VK_OCC_RDWATCH=1, gfx_vk_core.c occ2_rw_*; diagnostic, off by default): a guest CPU read of a sampled samples-passed report word, seen with an ARM64 hardware
// watchpoint (watchpoint 0: loads of its 8 bytes, EL0) set on every thread of the process but the helper below (suspended, the debug registers set, resumed; a thread cannot set
// its own). The first load from code in no module (the JIT's: guest code) is recorded with the host PC, the present and whether the word's submission had completed, and the
// watch is cleared everywhere; loads from a module (our own threads, the drivers) are counted and only that thread is cleared. The load goes through once its thread's
// watchpoint is cleared (by the helper, and in the context it resumes with). r49's store watch (hwwatch_win.c) is the model; it is not built in NXN1X
#include <tlhelp32.h>
static volatile LONG g_hwr_st, g_hwr_rec, g_hwr_host;   // st: 0 off, 1 arm wanted, 2 armed, 3 clear wanted; rec: a guest read recorded
static volatile uint64_t g_hwr_a, g_hwr_f, g_hwr_pc, g_hwr_fr; static volatile int g_hwr_landed;
static HANDLE g_hwr_th, g_hwr_wake, g_hwr_req, g_hwr_done; static volatile DWORD g_hwr_req_tid; static SRWLOCK g_hwr_lk = SRWLOCK_INIT; static DWORD g_hwr_tid;
static int hwr_set(DWORD tid, uint64_t a) {   // one thread's watchpoint 0: loads of the 8 bytes at a (0: cleared). Nothing between suspend and resume allocates or logs
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE, tid); if (!h) return 0;
    int ok = 0;
    if (SuspendThread(h) != (DWORD)-1) {
        CONTEXT c; memset(&c, 0, sizeof c); c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        if (GetThreadContext(h, &c)) { c.Wcr[0] = a ? 1u | (2u << 1) | (1u << 3) | (0xffu << 5) : 0; c.Wvr[0] = a & ~7ull; c.ContextFlags = CONTEXT_DEBUG_REGISTERS; ok = SetThreadContext(h, &c) != 0; }
        ResumeThread(h);
    }
    CloseHandle(h); return ok;
}
static void hwr_all(uint64_t a) {   // every thread of the process but this one
    HANDLE sn = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0); if (sn == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 te; memset(&te, 0, sizeof te); te.dwSize = sizeof te; const DWORD pid = GetCurrentProcessId(), self = GetCurrentThreadId();
    if (Thread32First(sn, &te)) do { if (te.th32OwnerProcessID == pid && te.th32ThreadID != self) (void)hwr_set(te.th32ThreadID, a); te.dwSize = sizeof te; } while (Thread32Next(sn, &te));
    CloseHandle(sn);
}
static DWORD WINAPI hwr_main(void *p) {
    (void)p; g_hwr_tid = GetCurrentThreadId(); HANDLE hs[2] = {g_hwr_req, g_hwr_wake};
    for (;;) {
        const DWORD r = WaitForMultipleObjects(2, hs, FALSE, 1000);
        if (r == WAIT_OBJECT_0) { (void)hwr_set(g_hwr_req_tid, 0); SetEvent(g_hwr_done); continue; }   // a hit: that thread's watchpoint cleared so its load goes through
        const LONG st = InterlockedCompareExchange(&g_hwr_st, 0, 0);
        if (st == 1) { hwr_all(g_hwr_a); InterlockedCompareExchange(&g_hwr_st, 2, 1); }
        else if (st == 3) { hwr_all(0); InterlockedCompareExchange(&g_hwr_st, 0, 3); }
    }
    return 0;
}
static LONG CALLBACK hwr_veh(EXCEPTION_POINTERS *ep) {
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (InterlockedCompareExchange(&g_hwr_st, 0, 0) == 0 || (code != EXCEPTION_SINGLE_STEP && code != EXCEPTION_BREAKPOINT)) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT *c = ep->ContextRecord; const uint32_t insn = *(const uint32_t *)(uintptr_t)c->Pc;
    if ((insn & 0xffe0001fu) == 0xd4200000u || (insn & 0x0a000000u) != 0x08000000u) return EXCEPTION_CONTINUE_SEARCH;   // a BRK, or not a load / store: not ours
    HMODULE m = NULL; const int host = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)(uintptr_t)c->Pc, &m) && m;
    if (host) InterlockedIncrement(&g_hwr_host);
    else if (InterlockedCompareExchange(&g_hwr_st, 3, 2) == 2 || InterlockedCompareExchange(&g_hwr_st, 3, 1) == 1) {   // the first guest read: recorded, the watch cleared everywhere
        extern uint64_t g_tex_frame; g_hwr_pc = c->Pc; g_hwr_fr = __atomic_load_n(&g_tex_frame, __ATOMIC_RELAXED); g_hwr_landed = vk_fence_done() >= g_hwr_f;
        InterlockedExchange(&g_hwr_rec, 1); SetEvent(g_hwr_wake); }
    AcquireSRWLockExclusive(&g_hwr_lk);
    g_hwr_req_tid = GetCurrentThreadId(); SetEvent(g_hwr_req); WaitForSingleObject(g_hwr_done, 2000);
    ReleaseSRWLockExclusive(&g_hwr_lk);
    c->Wcr[0] = 0; c->Wvr[0] = 0;   // (and the context this thread resumes with, should it carry the debug registers)
    return EXCEPTION_CONTINUE_EXECUTION;
}
int vk_hwr_arm(uint64_t a, uint64_t fence) {   // 1: the watch asked for (the helper arms it on every thread, asynchronously); 0: busy or not available
    if (!g_hwr_th) {
        g_hwr_wake = CreateEventW(NULL, FALSE, FALSE, NULL); g_hwr_req = CreateEventW(NULL, FALSE, FALSE, NULL); g_hwr_done = CreateEventW(NULL, FALSE, FALSE, NULL);
        if (!g_hwr_wake || !g_hwr_req || !g_hwr_done || !AddVectoredExceptionHandler(1, hwr_veh) || !(g_hwr_th = CreateThread(NULL, 0, hwr_main, NULL, 0, NULL))) {
            static int said; if (!said++) VKLOG("[RED] occ-s2: the read watch cannot start (events, exception handler or thread)"); return 0; }
    }
    if (!a || InterlockedCompareExchange(&g_hwr_st, 0, 0) != 0) return 0;
    g_hwr_a = a & ~7ull; g_hwr_f = fence; InterlockedExchange(&g_hwr_rec, 0); InterlockedExchange(&g_hwr_host, 0);
    InterlockedExchange(&g_hwr_st, 1); SetEvent(g_hwr_wake); return 1;
}
int vk_hwr_poll(uint64_t *pc, int *landed, uint64_t *frame, uint64_t *host) {   // 1: a guest read recorded (pc, landed, the present it came in)
    *host = (uint64_t)InterlockedCompareExchange(&g_hwr_host, 0, 0);
    if (!InterlockedCompareExchange(&g_hwr_rec, 0, 0)) return 0;
    *pc = g_hwr_pc; *landed = g_hwr_landed; *frame = g_hwr_fr; return 1;
}
void vk_hwr_off(void) { if (InterlockedCompareExchange(&g_hwr_st, 3, 2) == 2 || InterlockedCompareExchange(&g_hwr_st, 3, 1) == 1) SetEvent(g_hwr_wake); }
int mtl_has_window(void) { return dm_front_window() != NULL; }
int mtl_display_hz(void) { return 60; }
int mtl_fps_target(void) { return DM_GAME_FPS_TARGET > 0 ? DM_GAME_FPS_TARGET : 60; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_prog_release(prog_t *p) { (void)p; }
