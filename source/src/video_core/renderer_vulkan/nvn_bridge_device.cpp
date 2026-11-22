// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "common/logging.h"
#include "core/nvn/nvn_layer.h"
#include "video_core/renderer_vulkan/nvn_bridge_device.h"
#include "video_core/renderer_vulkan/vk_hang_watch.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/nvn_memory_coordinator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

#include "nvnh_vk.h"

namespace Vulkan::NvnBridge {

namespace {

std::atomic<const Device*> g_device{};
std::atomic<Scheduler*> g_scheduler{};
std::atomic<PipelineCache*> g_pipeline_cache{};
std::atomic<u64> g_program_id{};

// nvnh_vk_device::ext points here: the enabled extension names, kept until the next handover (the host reads them in vk_init)
std::vector<std::string> g_ext_names;
std::vector<const char*> g_ext_ptrs;

// The graphics queue's external synchronization (Vulkan spec: vkQueueSubmit and vkQueuePresentKHR on one queue must not run
// at once): Scheduler::submit_mutex, under which the scheduler's worker, the PresentManager, the swapchain and LSFG use it.
// Each NVN submission also takes a queue position, so stall reports (vk_hang_watch.h) count it among the queue's operations.
void QueueLock(void* ctx) {
    static_cast<std::mutex*>(ctx)->lock();
    (void)HangWatch::NextQueueSeq();
}

void QueueUnlock(void* ctx) {
    static_cast<std::mutex*>(ctx)->unlock();
}

const char* YesNo(bool value) {
    return value ? "yes" : "no";
}

bool HandOver(VkInstance instance, PFN_vkGetInstanceProcAddr gipa, const Device& device, Scheduler& scheduler) {
    nvnh_vk_device d{};
    d.inst = instance;
    d.pd = device.GetPhysical();
    d.dev = *device.GetLogical();
    d.qf = device.GetGraphicsFamily();
    device.GetDispatchLoader().vkGetDeviceQueue(d.dev, d.qf, 0, &d.q); // the queue the scheduler submits to
    d.gipa = gipa;

    const auto& loaded = device.GetLoadedExtensions();
    g_ext_names.assign(loaded.begin(), loaded.end());
    g_ext_ptrs.clear();
    for (const std::string& name : g_ext_names) {
        g_ext_ptrs.push_back(name.c_str());
    }
    d.ext = g_ext_ptrs.data();
    d.n_ext = static_cast<u32>(g_ext_ptrs.size());
    const auto has = [&loaded](const char* name) { return loaded.contains(name); };

    // The feature bits as the device was created with them: the chain passed to vkCreateDevice
    const VkPhysicalDeviceFeatures2& f2 = device.GetEnabledFeatures2();
    d.core = f2.features;
    for (auto* s = static_cast<const VkBaseInStructure*>(f2.pNext); s != nullptr; s = s->pNext) {
        switch (s->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES:
            d.timeline = reinterpret_cast<const VkPhysicalDeviceTimelineSemaphoreFeatures*>(s)->timelineSemaphore != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES:
            d.dynrender = reinterpret_cast<const VkPhysicalDeviceDynamicRenderingFeatures*>(s)->dynamicRendering != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES:
            d.imageless =
                reinterpret_cast<const VkPhysicalDeviceImagelessFramebufferFeatures*>(s)->imagelessFramebuffer != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DEMOTE_TO_HELPER_INVOCATION_FEATURES:
            d.demote = reinterpret_cast<const VkPhysicalDeviceShaderDemoteToHelperInvocationFeatures*>(s)
                           ->shaderDemoteToHelperInvocation != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES:
            d.bda = reinterpret_cast<const VkPhysicalDeviceBufferDeviceAddressFeatures*>(s)->bufferDeviceAddress != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT:
            d.xfb = reinterpret_cast<const VkPhysicalDeviceTransformFeedbackFeaturesEXT*>(s)->transformFeedback != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT:
            d.custom_border =
                reinterpret_cast<const VkPhysicalDeviceCustomBorderColorFeaturesEXT*>(s)->customBorderColors != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT:
            d.null_descriptor = reinterpret_cast<const VkPhysicalDeviceRobustness2FeaturesEXT*>(s)->nullDescriptor != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT:
            d.depth_clip_ctl =
                reinterpret_cast<const VkPhysicalDeviceDepthClipControlFeaturesEXT*>(s)->depthClipControl != VK_FALSE;
            break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT:
            d.uint8_index = reinterpret_cast<const VkPhysicalDeviceIndexTypeUint8FeaturesEXT*>(s)->indexTypeUint8 != VK_FALSE;
            break;
        default:
            break;
        }
    }
    // An extension's feature bits count only with the extension enabled (Device::RemoveExtension can drop one and keep its
    // structure in the chain). Robustness2 and uint8 indices may be the KHR promotions, same structures
    d.xfb = d.xfb && has(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME);
    d.custom_border = d.custom_border && has(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    d.null_descriptor = d.null_descriptor && (has("VK_EXT_robustness2") || has("VK_KHR_robustness2"));
    d.depth_clip_ctl = d.depth_clip_ctl && has(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);
    d.uint8_index = d.uint8_index && (has("VK_EXT_index_type_uint8") || has("VK_KHR_index_type_uint8"));

    d.queue_lock = QueueLock;
    d.queue_unlock = QueueUnlock;
    d.queue_ctx = &scheduler.submit_mutex;

    // The layer records draws and dispatches into the same command buffers (gfx_vk_core.c picked a graphics + compute family)
    const auto families = device.GetPhysical().GetQueueFamilyProperties();
    if (d.qf >= families.size() || (families[d.qf].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
        LOG_ERROR(Render_Vulkan, "NVN host layer: graphics queue family {} has no compute", d.qf);
    }
    // What the layer's vk_init stops on (gfx_vk_core.c vk_init_import), checked here so the boot can still fall back to Emulation
    if (!d.timeline || (!d.dynrender && !d.imageless)) {
        LOG_ERROR(Render_Vulkan, "NVN host layer: NXN1X's device has {} enabled",
                  !d.timeline ? "no timeline semaphores" : "neither dynamic rendering nor imageless framebuffers");
        return false;
    }
    if (nvnh_vk_use_device(&d) != 0) {
        LOG_ERROR(Render_Vulkan, "NVN host layer refused NXN1X's device: its Vulkan already started");
        return false;
    }
    if (auto* memory = device.GetNvnMemoryCoordinator()) {
        if (nvnh_memory_attach(memory->Api()) != 0)
            LOG_WARNING(Render_Vulkan, "NVN-F1 native observer attach rejected; ownership coverage is incomplete");
    }
    LOG_INFO(Render_Vulkan,
             "NVN host layer: NXN1X's device, graphics queue family {}, {} extensions; timeline {}, dynamic "
             "rendering {}, imageless framebuffers {}, demote {}, buffer device address {}, transform feedback {}, "
             "custom border colours {}, null descriptors {}, depth clip control {}, uint8 indices {}, host memory "
             "import {}",
             d.qf, d.n_ext, YesNo(d.timeline), YesNo(d.dynrender), YesNo(d.imageless), YesNo(d.demote),
             YesNo(d.bda), YesNo(d.xfb), YesNo(d.custom_border), YesNo(d.null_descriptor),
             YesNo(d.depth_clip_ctl), YesNo(d.uint8_index), YesNo(has(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)));
    return true;
}

} // Anonymous namespace

void Attach(VkInstance instance, PFN_vkGetInstanceProcAddr gipa, const Device& device, Scheduler& scheduler,
            PipelineCache& pipeline_cache) {
    g_device.store(&device, std::memory_order_release);
    g_scheduler.store(&scheduler, std::memory_order_release);
    g_pipeline_cache.store(&pipeline_cache, std::memory_order_release);
    if (!device.IsNvnHostDevice() || !HandOver(instance, gipa, device, scheduler)) {
        if (Core::NvnLayer::Wanted()) {
            Core::NvnLayer::Refuse(device.IsNvnHostDevice()
                                       ? "the NVN layer could not take NXN1X's Vulkan device"
                                       : "the renderer's device was made without the NVN layer (no window surface)");
        }
        return;
    }
    // NVN_HOST_BRIDGE.md 6 step 2 check, before any game code runs: the layer's vk_init on this device and one empty
    // submission through the layer (ring slot, timeline signal, completion thread)
    if (const char* test = std::getenv("NXN1X_NVN_VK_SELFTEST"); test != nullptr && test[0] == '1') {
        const int result = nvnh_vk_selftest();
        LOG_INFO(Render_Vulkan, "NVN host layer self-test: {}", result == 0 ? "passed" : "FAILED");
    }
    // The device is the layer's now: the game's NVN calls may reach it (the hooked definitions stop running the game's own code)
    Core::NvnLayer::GateOn();
}

void Detach() {
    nvnh_vk_archive_shutdown();
    g_pipeline_cache.store(nullptr, std::memory_order_release);
    g_scheduler.store(nullptr, std::memory_order_release);
    g_device.store(nullptr, std::memory_order_release);
    g_program_id.store(0, std::memory_order_relaxed);
}

void FinishShutdown(VkResult idle_result) {
    if (!nvnh_vk_archive_finish_shutdown(idle_result == VK_SUCCESS)) {
        LOG_ERROR(Render_Vulkan, "NVN archive shutdown retained source references: device idle result {}",
                  static_cast<int>(idle_result));
    }
    nvnh_memory_detach();
}

const Device* GetDevice() {
    return g_device.load(std::memory_order_acquire);
}

Scheduler* GetScheduler() {
    return g_scheduler.load(std::memory_order_acquire);
}

PipelineCache* GetPipelineCache() {
    return g_pipeline_cache.load(std::memory_order_acquire);
}

u64 GetProgramId() {
    return g_program_id.load(std::memory_order_relaxed);
}

void SetProgramId(u64 program_id) {
    g_program_id.store(program_id, std::memory_order_relaxed);
}

} // namespace Vulkan::NvnBridge
