// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/fs/file.h"
#include "common/fs/fs_paths.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include <ranges>
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"
#if defined(NXN1X_NVN_HOST)
#include "video_core/vulkan_common/nvn_memory_coordinator.h"
#endif

namespace Vulkan {
    namespace {

// Helpers translating MemoryUsage to flags/usage

        [[maybe_unused]] VkMemoryPropertyFlags MemoryUsagePropertyFlags(MemoryUsage usage) {
            switch (usage) {
                case MemoryUsage::DeviceLocal:
                    return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                case MemoryUsage::Upload:
                    return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                case MemoryUsage::Download:
                    return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                           VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                case MemoryUsage::Stream:
                    return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            }
            ASSERT_MSG(false, "Invalid memory usage={}", usage);
            return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }

        [[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
            if (usage == MemoryUsage::Download) {
                return VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            }
            return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                     : VkMemoryPropertyFlagBits{};
        }

        [[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
            switch (usage) {
                case MemoryUsage::Upload:
                case MemoryUsage::Stream:
                    return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
                case MemoryUsage::Download:
                    return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
                case MemoryUsage::DeviceLocal:
                    return {};
            }
            return {};
        }

        [[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
            switch (usage) {
                case MemoryUsage::DeviceLocal:
                case MemoryUsage::Stream:
                    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                case MemoryUsage::Upload:
                case MemoryUsage::Download:
                    return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
            }
            return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        }


// This avoids calling vkGetBufferMemoryRequirements* directly.
        template<typename T>
        static VkBuffer GetVkHandleFromBuffer(const T &buf) {
            if constexpr (requires { static_cast<VkBuffer>(buf); }) {
                return static_cast<VkBuffer>(buf);
            } else if constexpr (requires {{ buf.GetHandle() } -> std::convertible_to<VkBuffer>; }) {
                return buf.GetHandle();
            } else if constexpr (requires {{ buf.Handle() } -> std::convertible_to<VkBuffer>; }) {
                return buf.Handle();
            } else if constexpr (requires {{ buf.vk_handle() } -> std::convertible_to<VkBuffer>; }) {
                return buf.vk_handle();
            } else {
                static_assert(sizeof(T) == 0, "Cannot extract VkBuffer handle from vk::Buffer");
                return VK_NULL_HANDLE;
            }
        }

    } // namespace

//MemoryCommit is now VMA-backed
    MemoryCommit::MemoryCommit(VmaAllocator alloc, VmaAllocation a,
                               const VmaAllocationInfo &info) noexcept
            : allocator{alloc}, allocation{a}, memory{info.deviceMemory},
              offset{info.offset}, size{info.size}, mapped_ptr{info.pMappedData} {
        // Log GPU memory allocation
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
                reinterpret_cast<uintptr_t>(memory),
                static_cast<u64>(size),
                0  // Memory property flags (not easily available from VMA)
            );
        }
    }

    MemoryCommit::~MemoryCommit() { Release(); }

    MemoryCommit::MemoryCommit(MemoryCommit &&rhs) noexcept
            : allocator{std::exchange(rhs.allocator, nullptr)},
              allocation{std::exchange(rhs.allocation, nullptr)},
              memory{std::exchange(rhs.memory, VK_NULL_HANDLE)},
              offset{std::exchange(rhs.offset, 0)},
              size{std::exchange(rhs.size, 0)},
              mapped_ptr{std::exchange(rhs.mapped_ptr, nullptr)} {}

    MemoryCommit &MemoryCommit::operator=(MemoryCommit &&rhs) noexcept {
        if (this != &rhs) {
            Release();
            allocator = std::exchange(rhs.allocator, nullptr);
            allocation = std::exchange(rhs.allocation, nullptr);
            memory = std::exchange(rhs.memory, VK_NULL_HANDLE);
            offset = std::exchange(rhs.offset, 0);
            size = std::exchange(rhs.size, 0);
            mapped_ptr = std::exchange(rhs.mapped_ptr, nullptr);
        }
        return *this;
    }

    std::span<u8> MemoryCommit::Map()
    {
        if (!allocation) return {};
        if (!mapped_ptr) {
            if (vmaMapMemory(allocator, allocation, &mapped_ptr) != VK_SUCCESS) return {};
        }
        const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                    (std::numeric_limits<size_t>::max)()));
        return std::span<u8>{static_cast<u8 *>(mapped_ptr), n};
    }

    std::span<const u8> MemoryCommit::Map() const
    {
        if (!allocation) return {};
        if (!mapped_ptr) {
            void *p = nullptr;
            if (vmaMapMemory(allocator, allocation, &p) != VK_SUCCESS) return {};
            const_cast<MemoryCommit *>(this)->mapped_ptr = p;
        }
        const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                    (std::numeric_limits<size_t>::max)()));
        return std::span<const u8>{static_cast<const u8 *>(mapped_ptr), n};
    }

    void MemoryCommit::Unmap()
    {
        if (allocation && mapped_ptr) {
            vmaUnmapMemory(allocator, allocation);
            mapped_ptr = nullptr;
        }
    }

    void MemoryCommit::Release() {
        if (allocation && allocator) {
            // Log GPU memory deallocation
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_memory_tracking.GetValue() &&
                memory != VK_NULL_HANDLE) {
                GPU::Logging::GPULogger::GetInstance().LogMemoryDeallocation(
                    reinterpret_cast<uintptr_t>(memory)
                );
            }

            if (mapped_ptr) {
                vmaUnmapMemory(allocator, allocation);
                mapped_ptr = nullptr;
            }
            vmaFreeMemory(allocator, allocation);
        }
        allocation = nullptr;
        allocator = nullptr;
        memory = VK_NULL_HANDLE;
        offset = 0;
        size = 0;
    }

    MemoryAllocator::MemoryAllocator(const Device &device_)
            : device{device_}, allocator{device.GetAllocator()},
              properties{device_.GetPhysical().GetMemoryProperties().memoryProperties},
              buffer_image_granularity{
                      device_.GetPhysical().GetProperties().limits.bufferImageGranularity} {

        // Preserve the previous "RenderDoc small heap" trimming behavior that we had in original vma minus the heap bug
        if (device.HasDebuggingToolAttached())
        {
            using namespace Common::Literals;
            ForEachDeviceLocalHostVisibleHeap(device, [this](size_t heap_idx, VkMemoryHeap &heap) {
                if (heap.size <= 256_MiB) {
                    for (u32 t = 0; t < properties.memoryTypeCount; ++t) {
                        if (properties.memoryTypes[t].heapIndex == heap_idx) {
                            valid_memory_types &= ~(1u << t);
                        }
                    }
                }
            });
        }
    }

    MemoryAllocator::~MemoryAllocator() = default;

    vk::Image MemoryAllocator::CreateImage(const VkImageCreateInfo &ci) const
    {
        const VmaAllocationCreateInfo alloc_ci = {
                .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                .requiredFlags = 0,
                .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                .memoryTypeBits = 0,
                .pool = VK_NULL_HANDLE,
                .pUserData = nullptr,
                .priority = 0.f,
        };

        VkImage handle{};
        VmaAllocation allocation{};
        VmaAllocationInfo alloc_info{};
        VkResult result = vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        if (result != VK_SUCCESS && ReleaseForRetry(0) != 0) {
            result = vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        }
        if (result != VK_SUCCESS) {
            // Say which image did not fit before the exception ends the session, and let an image
            // that is merely over the budget be allocated anyway (Lichii 2a35b191).
            VkDeviceSize bytes = 0;
            VkImage probe{};
            const auto& dld = device.GetDispatchLoader();
            if (dld.vkCreateImage(*device.GetLogical(), &ci, nullptr, &probe) == VK_SUCCESS) {
                bytes = device.GetLogical().GetImageMemoryRequirements(probe).size;
                dld.vkDestroyImage(*device.GetLogical(), probe, nullptr);
            }
            static u32 logged{};
            if (logged++ < 32) {
                LOG_WARNING(Render_Vulkan,
                            "Image {}x{}x{} ({} levels, {} layers, {} samples, format {}, type {}, "
                            "flags {:#x}, usage {:#x}) needs {} bytes and did not fit ({}); "
                            "retrying without the budget limit",
                            ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                            ci.arrayLayers, static_cast<u32>(ci.samples),
                            static_cast<u32>(ci.format), static_cast<u32>(ci.imageType), ci.flags,
                            ci.usage, bytes, static_cast<int>(result));
            }
            VmaAllocationCreateInfo relaxed = alloc_ci;
            relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
            if (result != VK_SUCCESS) {
                LOG_ERROR(Render_Vulkan, "Image of {} bytes failed again: {}", bytes,
                          static_cast<int>(result));
                LogBudget();
            }
        }
        vk::Check(result);
        if (alloc_info.size >= (64ULL << 20)) {
            // Name every large image in the log, so a heap that runs out can be read back.
            static std::atomic<u32> large_logged{};
            if (large_logged.fetch_add(1, std::memory_order_relaxed) < 256) {
                LOG_INFO(Render_Vulkan,
                         "Large image {} MiB: {}x{}x{} ({} levels, {} layers, {} samples, "
                         "format {}, type {}, flags {:#x}, usage {:#x}, memory type {}, "
                         "offset {:#x})",
                         alloc_info.size >> 20, ci.extent.width, ci.extent.height,
                         ci.extent.depth, ci.mipLevels, ci.arrayLayers,
                         static_cast<u32>(ci.samples), static_cast<u32>(ci.format),
                         static_cast<u32>(ci.imageType), ci.flags, ci.usage,
                         alloc_info.memoryType, alloc_info.offset);
            }
        }

        // Log GPU memory allocation for images
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
                reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
                static_cast<u64>(alloc_info.size),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
        }

        return vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                         device.GetDispatchLoader());
    }

    vk::Image MemoryAllocator::TryCreateImage(const VkImageCreateInfo &ci,
                                              VkDeviceSize reserve) const {
        const auto& dld = device.GetDispatchLoader();
        const VkDevice dev = *device.GetLogical();
        VkImage handle{};
        VkResult result = dld.vkCreateImage(dev, &ci, nullptr, &handle);
        if (result != VK_SUCCESS) {
            if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
                vk::Check(result);
            }
            failed_images.fetch_add(1, std::memory_order_relaxed);
            return vk::Image{};
        }
        VkMemoryRequirements requirements{};
        dld.vkGetImageMemoryRequirements(dev, handle, &requirements);
        // The probe only measures the image; vmaCreateImage makes the real one below (the AUTO
        // memory usages need the create info: vmaAllocateMemoryForImage fails them with
        // VK_ERROR_FEATURE_NOT_PRESENT, which stopped every Switch 2 boot in ee74901a/57a48046).
        dld.vkDestroyImage(dev, handle, nullptr);
        handle = VK_NULL_HANDLE;
        // Only refuse once at least half the budget and 2 GiB are in use, as at the world map
        // (6.5 of 7.0 GB); a nearly empty heap never refuses.
        // Large images (the 170/341 MiB textures) keep the whole reserve; smaller ones, render
        // targets more often than not, a quarter of it. With 512 MiB for large and none for
        // small ones, Mario Kart World's display buffers stayed but the heap reached 6774 of 7131
        // MiB and a command submission failed with OUT_OF_DEVICE_MEMORY (919c14a0 run); with
        // 512 MiB for all, its three 3840x2160 display buffers (31 MiB) were left out in turn.
        const VkDeviceSize keep = requirements.size < (VkDeviceSize{64} << 20) ? reserve / 4 : reserve;
        const auto fits = [&] {
            const auto [used, budget] = GetDeviceLocalBudget();
            return budget == 0 || used + requirements.size + keep <= budget ||
                   used < (std::max<VkDeviceSize>)(budget / 2, VkDeviceSize{2} << 30);
        };
        static std::atomic_flag budget_logged{};
        if (!budget_logged.test_and_set()) {
            const auto [used, budget] = GetDeviceLocalBudget();
            LOG_ERROR(Render_Vulkan,
                      "First Switch 2 texture image: device-local {} of {} MiB used (not an error)",
                      used >> 20, budget >> 20);
        }
        bool room = fits();
        if (!room && ReleaseForRetry(requirements.size) != 0) {
            room = fits();
        }
        VmaAllocation allocation{};
        VmaAllocationInfo alloc_info{};
        result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        if (room) {
            const VmaAllocationCreateInfo alloc_ci = {
                    .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
                    .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    .requiredFlags = 0,
                    .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    .memoryTypeBits = 0,
                    .pool = VK_NULL_HANDLE,
                    .pUserData = nullptr,
                    .priority = 0.f,
            };
            result = vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
            if (result != VK_SUCCESS && ReleaseForRetry(requirements.size) != 0) {
                result = vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation,
                                        &alloc_info);
            }
        }
        if (result != VK_SUCCESS) {
            if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY && result != VK_ERROR_OUT_OF_HOST_MEMORY) {
                LOG_ERROR(Render_Vulkan, "Image {}x{}x{} (format {}) failed: {}", ci.extent.width,
                          ci.extent.height, ci.extent.depth, static_cast<u32>(ci.format),
                          static_cast<int>(result));
                vk::Check(result);
            }
            const u32 failed = failed_images.fetch_add(1, std::memory_order_relaxed);
            if (failed < 32 || failed % 1024 == 0) {
                const auto [used, budget] = GetDeviceLocalBudget();
                LOG_ERROR(Render_Vulkan,
                            "Image {}x{}x{} ({} levels, {} layers, format {}) of {} MiB left out: "
                            "device-local {} of {} MiB used, {} MiB kept free (failure {})",
                            ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                            ci.arrayLayers, static_cast<u32>(ci.format), requirements.size >> 20,
                            used >> 20, budget >> 20, keep >> 20, failed + 1);
            }
            return vk::Image{};
        }
        if (alloc_info.size >= (64ULL << 20)) {
            static std::atomic<u32> large_logged{};
            if (large_logged.fetch_add(1, std::memory_order_relaxed) < 256) {
                LOG_INFO(Render_Vulkan,
                         "Large image {} MiB: {}x{}x{} ({} levels, {} layers, format {}, flags "
                         "{:#x}, usage {:#x}, memory type {}, offset {:#x}, may fail)",
                         alloc_info.size >> 20, ci.extent.width, ci.extent.height,
                         ci.extent.depth, ci.mipLevels, ci.arrayLayers,
                         static_cast<u32>(ci.format), ci.flags, ci.usage, alloc_info.memoryType,
                         alloc_info.offset);
            }
        }
        return vk::Image(handle, ci.usage, dev, allocator, allocation, dld);
    }

    vk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci, bool residency) const {
        if (ci.imageType != VK_IMAGE_TYPE_2D) {
            return vk::Image{};
        }
        VkImageCreateInfo sparse_ci = ci;
        sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                           (residency ? VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT : 0);
        VkImage sparse{};
        if (device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr,
                                                     &sparse) != VK_SUCCESS) {
            return vk::Image{};
        }
        return vk::Image(sparse, ci.usage, *device.GetLogical(), allocator, VK_NULL_HANDLE,
                         device.GetDispatchLoader());
    }

    VmaAllocation MemoryAllocator::AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                                        VkDeviceMemory &memory,
                                                        VkDeviceSize &offset) const {
        const VmaAllocationCreateInfo alloc_ci{
            .flags = 0,
            .usage = VMA_MEMORY_USAGE_UNKNOWN,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .preferredFlags = 0,
            .memoryTypeBits = requirements.memoryTypeBits,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 0.f,
        };
        VmaAllocation allocation{};
        VmaAllocationInfo info{};
        VkResult result = vmaAllocateMemory(allocator, &requirements, &alloc_ci, &allocation, &info);
        if (result != VK_SUCCESS && ReleaseForRetry(requirements.size) != 0) {
            result = vmaAllocateMemory(allocator, &requirements, &alloc_ci, &allocation, &info);
        }
        if (result != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        memory = info.deviceMemory;
        offset = info.offset;
        return allocation;
    }

    void MemoryAllocator::FreeSparseMemory(VmaAllocation allocation) const {
        if (allocation != VK_NULL_HANDLE) {
            vmaFreeMemory(allocator, allocation);
        }
    }

    vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
        // MESA will do memcpy() if not marked as host cached, so just force mark it for most buffers
        auto const anv_flags = (usage == MemoryUsage::Stream
            && device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
            ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
        const VmaAllocationCreateInfo alloc_ci = {
            .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
            .usage = MemoryUsageVma(usage),
            .requiredFlags = 0,
            .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
            .memoryTypeBits = usage == MemoryUsage::Stream ? 0u : valid_memory_types,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 0.f,
        };

        VkBuffer handle{};
        VmaAllocationInfo alloc_info{};
        VmaAllocation allocation{};
        VkMemoryPropertyFlags property_flags{};

        VkResult result =
            vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        if (result != VK_SUCCESS && ReleaseForRetry(ci.size) != 0) {
            result = vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
        }
        if (result != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Buffer of {} bytes (usage {:#x}, memory usage {}) failed: {}",
                      ci.size, ci.usage, static_cast<u32>(usage), static_cast<int>(result));
            LogBudget();
        }
        vk::Check(result);
        vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
        if (ci.size >= (64ULL << 20)) {
            static std::atomic<u32> large_logged{};
            if (large_logged.fetch_add(1, std::memory_order_relaxed) < 256) {
                LOG_INFO(Render_Vulkan,
                         "Large buffer {} MiB: usage {:#x}, memory usage {}, memory type {}",
                         ci.size >> 20, ci.usage, static_cast<u32>(usage),
                         alloc_info.memoryType);
            }
        }

        // Log GPU memory allocation for buffers
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
                reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
                static_cast<u64>(alloc_info.size),
                property_flags
            );
        }

        u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
        const std::span<u8> mapped_data = data ? std::span<u8>{data, ci.size} : std::span<u8>{};
        const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

        return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data,
                          is_coherent,
                          device.GetDispatchLoader());
    }

    MemoryCommit MemoryAllocator::Commit(const VkMemoryRequirements &reqs, MemoryUsage usage)
    {
        const auto vma_usage = MemoryUsageVma(usage);
        VmaAllocationCreateInfo ci{};
        ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
        ci.usage = vma_usage;
        ci.memoryTypeBits = reqs.memoryTypeBits & valid_memory_types;
        ci.requiredFlags = 0;
        ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);

        VmaAllocation a{};
        VmaAllocationInfo info{};

        VkResult res = vmaAllocateMemory(allocator, &reqs, &ci, &a, &info);
        if (res != VK_SUCCESS && ReleaseForRetry(reqs.size) != 0) {
            res = vmaAllocateMemory(allocator, &reqs, &ci, &a, &info);
        }

        if (res != VK_SUCCESS) {
            // Relax 1: drop budget constraint
            auto ci2 = ci;
            ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            res = vmaAllocateMemory(allocator, &reqs, &ci2, &a, &info);

            // Relax 2: if we preferred DEVICE_LOCAL, drop that preference
            if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                auto ci3 = ci2;
                ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                res = vmaAllocateMemory(allocator, &reqs, &ci3, &a, &info);
            }
        }

        if (res != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Memory commit of {} bytes (types {:#x}) failed: {}",
                      reqs.size, reqs.memoryTypeBits, static_cast<int>(res));
            LogBudget();
        }
        vk::Check(res);
        return MemoryCommit(allocator, a, info);
    }

    MemoryCommit MemoryAllocator::Commit(const vk::Buffer &buffer, MemoryUsage usage) {
        // Allocate memory appropriate for this buffer automatically
        const auto vma_usage = MemoryUsageVma(usage);

        VmaAllocationCreateInfo ci{};
        ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
        ci.usage = vma_usage;
        ci.requiredFlags = 0;
        ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);
        ci.pool = VK_NULL_HANDLE;
        ci.pUserData = nullptr;
        ci.priority = 0.0f;

        const VkBuffer raw = *buffer;

        VmaAllocation a{};
        VmaAllocationInfo info{};

        // Let VMA infer memory requirements from the buffer
        VkResult res = vmaAllocateMemoryForBuffer(allocator, raw, &ci, &a, &info);
        if (res != VK_SUCCESS && ReleaseForRetry(0) != 0) {
            res = vmaAllocateMemoryForBuffer(allocator, raw, &ci, &a, &info);
        }

        if (res != VK_SUCCESS) {
            auto ci2 = ci;
            ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            res = vmaAllocateMemoryForBuffer(allocator, raw, &ci2, &a, &info);

            if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                auto ci3 = ci2;
                ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                res = vmaAllocateMemoryForBuffer(allocator, raw, &ci3, &a, &info);
            }
        }

        if (res != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "Buffer memory commit failed: {}", static_cast<int>(res));
            LogBudget();
        }
        vk::Check(res);
        vk::Check(vmaBindBufferMemory2(allocator, a, 0, raw, nullptr));
        return MemoryCommit(allocator, a, info);
    }



std::string ProcessMemorySummary();

VkDeviceSize MemoryAllocator::ReleaseForRetry(VkDeviceSize wanted) const {
    static thread_local bool releasing = false;
    if (!out_of_memory_handler || releasing) {
        return 0;
    }
    releasing = true;
    const VkDeviceSize freed = out_of_memory_handler();
    releasing = false;
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_WARNING(Render_Vulkan,
                    "Allocation of {} bytes did not fit; freed {} MiB of idle memory and retrying",
                    wanted, freed >> 20);
    }
    return freed;
}

void MemoryAllocator::LogBudget() const {
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    // Integer MiB (shift) keeps -Wconversion quiet: u64 -> double is a narrowing conversion.
    constexpr auto ToMiB = [](VkDeviceSize bytes) -> u64 { return bytes >> 20; };
    for (u32 heap = 0; heap < properties.memoryHeapCount; ++heap) {
        const VmaBudget& b = budgets[heap];
#if defined(NXN1X_NVN_HOST)
        if (auto* memory = device.GetNvnMemoryCoordinator())
            memory->LogicalVma(heap, b.statistics.allocationBytes);
#endif
        LOG_INFO(Render_Vulkan,
                 "GPU_MEM heap{} usage={}MiB budget={}MiB allocated={}MiB blocks={} "
                 "allocations={} heap_size={}MiB",
                 heap, ToMiB(b.usage), ToMiB(b.budget), ToMiB(b.statistics.allocationBytes),
                 b.statistics.blockCount, b.statistics.allocationCount,
                 ToMiB(properties.memoryHeaps[heap].size));
    }
    LOG_INFO(Render_Vulkan, "GPU_OBJECTS {} {}", vk::LiveObjectSummary(), ProcessMemorySummary());
    // Once a session: every allocation VMA holds (type, size, usage flags), to see what takes
    // the device memory when an allocation fails.
    static std::atomic_flag dumped{};
    if (!dumped.test_and_set()) {
        char* stats{};
        vmaBuildStatsString(allocator, &stats, VK_TRUE);
        if (stats != nullptr) {
            const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "vma_stats.json";
            Common::FS::IOFile file{path, Common::FS::FileAccessMode::Write,
                                    Common::FS::FileType::TextFile};
            const std::string_view text{stats};
            const size_t written = file.IsOpen() ? file.WriteString(text) : 0;
            LOG_INFO(Render_Vulkan, "VMA allocation map ({} bytes) written to {}", written,
                     Common::FS::PathToUTF8String(path));
            vmaFreeStatsString(allocator, stats);
        }
    }
}

std::pair<VkDeviceSize, VkDeviceSize> MemoryAllocator::GetDeviceLocalBudget() const {
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(allocator, budgets.data());
    VkDeviceSize used = 0;
    VkDeviceSize budget = 0;
    for (u32 heap = 0; heap < properties.memoryHeapCount; ++heap) {
        if (properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            used += budgets[heap].usage;
            budget += budgets[heap].budget;
        }
    }
    return {used, budget};
}

} // namespace Vulkan

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace Vulkan {
std::string ProcessMemorySummary() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(),
                                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                sizeof(counters))) {
        // Page faults since the previous line and free system RAM: a shrinking working set with
        // many faults means Windows is trimming us under memory pressure (stutter).
        static DWORD last_faults = 0;
        const DWORD faults = counters.PageFaultCount - last_faults;
        last_faults = counters.PageFaultCount;
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        return fmt::format("process_private={}MiB working_set={}MiB faults={} sys_avail={}MiB "
                           "sys_commit={}/{}MiB",
                           static_cast<u64>(counters.PrivateUsage) >> 20,
                           static_cast<u64>(counters.WorkingSetSize) >> 20, faults,
                           static_cast<u64>(status.ullAvailPhys) >> 20,
                           static_cast<u64>(status.ullTotalPageFile - status.ullAvailPageFile) >> 20,
                           static_cast<u64>(status.ullTotalPageFile) >> 20);
    }
#endif
    return {};
}
} // namespace Vulkan
