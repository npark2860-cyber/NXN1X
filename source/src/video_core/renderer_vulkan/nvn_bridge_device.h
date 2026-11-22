// SPDX-License-Identifier: GPL-3.0-or-later
// The NVN host layer's view of NXN1X's Vulkan backend (NXN1X docs NVN_HOST_BRIDGE.md 4.3, 4.4): the device and
// graphics queue it runs on, the queue's lock, and the pipeline cache its shaders share with the Maxwell path.
// RendererVulkan publishes them for its lifetime; the host layer's C side receives the device through nvnh_vk_use_device
// (host tree dm/nxn1x/nvnh_vk.h), the shader bridge (video_core/nvn, lane B) reads the getters.
#pragma once

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"

namespace Vulkan {

class Device;
class PipelineCache;
class Scheduler;

namespace NvnBridge {

/// RendererVulkan's constructor, last, once the rasterizer exists. On a boot that takes the NVN path (the device was made
/// for it: Device::IsNvnHostDevice) also hands the device, its graphics queue and Scheduler::submit_mutex to the host layer.
void Attach(VkInstance instance, PFN_vkGetInstanceProcAddr gipa, const Device& device, Scheduler& scheduler,
            PipelineCache& pipeline_cache);

/// RendererVulkan's destructor, first: the getters return nullptr again.
void Detach();
/// After the renderer's existing scheduler shutdown and device wait.
void FinishShutdown(VkResult idle_result);

/// nullptr outside RendererVulkan's lifetime.
const Device* GetDevice();
Scheduler* GetScheduler();
PipelineCache* GetPipelineCache();

/// The program the pipeline cache loaded its disk resources for (RasterizerVulkan::LoadDiskResources); 0 before that.
u64 GetProgramId();
void SetProgramId(u64 program_id);

} // namespace NvnBridge
} // namespace Vulkan
