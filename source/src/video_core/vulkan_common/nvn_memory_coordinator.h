// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <memory>
#include "video_core/vulkan_common/vulkan.h"
#include "nvnh_memory.h"
struct VmaVulkanFunctions;

namespace Vulkan {
// Device-owned observer. No resource ownership, policy callbacks or GPU waits.
class NvnMemoryCoordinator {
public:
    static bool Enabled();
    static std::shared_ptr<NvnMemoryCoordinator> Create(
        VkPhysicalDevice physical, VkDevice device, PFN_vkGetPhysicalDeviceMemoryProperties2 query,
        PFN_vkAllocateMemory allocate, PFN_vkFreeMemory free_memory, bool budget_extension);
    ~NvnMemoryCoordinator();
    const nvnh_memory_api* Api() const;
    void ConfigureVma(VmaVulkanFunctions& functions);
    void LogicalVma(uint32_t heap, uint64_t bytes);
    uint64_t OwnerPressure(uint32_t owner);
    // Used by the VMA dispatch adapter; accounting only, never changes results.
    VkResult Allocate(const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
    void Free(VkDeviceMemory, const VkAllocationCallbacks*);
private:
    struct Impl;
    explicit NvnMemoryCoordinator(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl;
};
} // namespace Vulkan
