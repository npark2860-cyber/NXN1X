// The Vulkan device handover (NVN_HOST_BRIDGE.md §4.3), apart from nvnh_api.h so that NXN1X's core, which has no Vulkan headers, includes no Vulkan types:
// video_core/renderer_vulkan/nvn_bridge_device.cpp includes this after NXN1X's own Vulkan headers; the host build (gfx_vk_core.c) after the host's
#pragma once
#include "nvnh_memory.h"
#include <stdint.h>
#include <vulkan/vulkan_core.h>   // the types only
#ifdef __cplusplus
extern "C" {
#endif

// ---- device and queue (NVN_HOST_BRIDGE.md §4.3) ----
// NXN1X's Vulkan device, as it was created: the layer creates no instance or device of its own once this is set, and may use only what is listed here
typedef struct nvnh_vk_device {
    VkInstance inst; VkPhysicalDevice pd; VkDevice dev; VkQueue q; uint32_t qf;   // the graphics queue (graphics + compute)
    PFN_vkGetInstanceProcAddr gipa;
    const char *const *ext; uint32_t n_ext;    // device extensions enabled at creation (the strings live as long as the device)
    VkPhysicalDeviceFeatures core;             // core features enabled at creation
    uint8_t timeline, dynrender, imageless, demote, bda, xfb, custom_border, null_descriptor, depth_clip_ctl, uint8_index;   // feature bits enabled at creation
    void (*queue_lock)(void *ctx); void (*queue_unlock)(void *ctx); void *queue_ctx;   // the queue's external synchronization (Scheduler::submit_mutex)
} nvnh_vk_device;
// Copies *d; before the game's first nvnDeviceInitialize (RendererVulkan's constructor). -1 when the layer already made its own device
int nvnh_vk_use_device(const nvnh_vk_device *d);
// Check of the handover (NXN1X_NVN_VK_SELFTEST=1, nvn_bridge_device.cpp): the layer's vk_init on the device, one empty submission, its completion
// through the layer's completion thread. 0 when it completed within 2 s. The layer's Vulkan state then exists for the process (nvnh_vk_active)
int nvnh_vk_selftest(void);
// Release the archive transfer reserve while the borrowed device is still alive.
// Game replay has stopped; the texture lock serializes any final transfer.
void nvnh_vk_archive_shutdown(void);
int nvnh_vk_archive_finish_shutdown(int device_idle);
// s10: [mem, mem + size) as VirtualQuery sees it, region by region (offset, size, allocation base, state, protection, type), up to 12, into out
void nvnh_range_regions(const void *mem, uint64_t size, char *out, size_t cap);
// s11: offsets in (0, size) where [mem, mem + size) passes from one allocation (a mapped view) to the next, up to max (at most 4096 regions looked at); their count
int nvnh_range_views(const void *mem, uint64_t size, uint64_t *cut, int max);
// s23 (nvnh_api.h nvnh_set_import_alias): NXN1X's backing alias of [mem, mem + size), NULL without one
void *nvnh_import_alias(const void *mem, uint64_t size);
// NVN2 s24 guard: 1 when every page of [mem, mem + size) is committed and readable (VirtualQuery, region by region, at most 4096 regions; 0 past that)
int nvnh_range_readable(const void *mem, uint64_t size);
// NVN2 s24 guard: the guest mapping generation (common/host_memory.cpp nxn1x_host_map_gen): it moves with every map, unmap and no-read protect of guest memory
uint64_t nvnh_map_gen(void);

#ifdef __cplusplus
}
#endif
