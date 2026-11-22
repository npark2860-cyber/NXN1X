// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <utility>
#include <vector>

#include <fmt/ranges.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/cgt_trace.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "video_core/renderer_vulkan/vk_adreno_profiler.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
namespace {

using namespace Common::Literals;

// Maximum potential alignment of a Vulkan buffer
constexpr VkDeviceSize MAX_ALIGNMENT = 256;

// Stream buffer size in bytes
// *NIX drivers are more sensitive to increased buffers for streaming.
// Windows ones however, can intake bigger buffers and generally do not OOM.
// - GTX 960 on Windows will not OOM with 256mib
// - GT 1030 on ^NIX will OOM with 256mib
#if defined(__FreeBSD__)
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 128_MiB;
#else
constexpr VkDeviceSize MAX_STREAM_BUFFER_SIZE = 256_MiB;
#endif

size_t GetStreamBufferSize(const Device& device) {
    if (!device.HasDebuggingToolAttached()) {
        // An integrated GPU's device memory is system RAM shared with the guest: half the stream
        // buffer (larger uploads take pool buffers anyway).
        return device.IsIntegrated() ? MAX_STREAM_BUFFER_SIZE / 2 : MAX_STREAM_BUFFER_SIZE;
    }

    VkDeviceSize size{0};
    bool has_device_local_host_visible_heap{};
    ForEachDeviceLocalHostVisibleHeap(device, [&size, &has_device_local_host_visible_heap](
                                                  size_t index, VkMemoryHeap& heap) {
        has_device_local_host_visible_heap = true;
        size = (std::max)(size, heap.size);
    });
    if (has_device_local_host_visible_heap) {
        // If rebar is not supported, cut the max heap size to 40%. This will allow 2 captures to be
        // loaded at the same time in RenderDoc. If rebar is supported, this shouldn't be an issue
        // as the heap will be much larger.
        if (size <= MAX_STREAM_BUFFER_SIZE) {
            size = size * 40 / 100;
        }
    } else {
        size = MAX_STREAM_BUFFER_SIZE;
    }
    return (std::min)(Common::AlignUp(size, MAX_ALIGNMENT), MAX_STREAM_BUFFER_SIZE);
}
} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, MemoryAllocator& memory_allocator_,
                                     Scheduler& scheduler_)
    : device{device_}, memory_allocator{memory_allocator_}, scheduler{scheduler_},
      stream_buffer_size{GetStreamBufferSize(device)}, region_size{stream_buffer_size /
                                                                   StagingBufferPool::NUM_SYNCS} {
    VkBufferCreateInfo stream_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = stream_buffer_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    stream_buffer = memory_allocator.CreateBuffer(stream_ci, MemoryUsage::Stream);
    if (device.HasDebuggingToolAttached()) {
        stream_buffer.SetObjectNameEXT("Stream Buffer");
    }
    if (device.IsBufferDeviceAddressSupported()) {
        stream_buffer_address = device.GetLogical().GetBufferDeviceAddress(*stream_buffer);
    }
    stream_pointer = stream_buffer.Mapped();
    ASSERT_MSG(!stream_pointer.empty(), "Stream buffer must be host visible!");
    if (Common::CGT::Enabled()) {
        Common::CGT::StreamInfo(Common::CGT::HandleBits(*stream_buffer), stream_buffer_size,
                                stream_buffer.IsHostCoherent(), NUM_SYNCS);
    }
}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, MemoryUsage usage, bool deferred) {
    std::scoped_lock pool_lock{pool_mutex};
    auto& profiler = AdrenoProfiler::Get();
    if (usage == MemoryUsage::Upload) {
        profiler.RecordStagingRequest(static_cast<u64>(size), false, deferred);
    } else if (usage == MemoryUsage::Download) {
        profiler.RecordStagingRequest(static_cast<u64>(size), true, deferred);
    }
    const bool stream = !deferred && usage == MemoryUsage::Upload && size <= region_size;
    StagingBufferRef ref = stream ? GetStreamBuffer(size) : GetStagingBuffer(size, usage, deferred);
    if (Common::CGT::Enabled()) [[unlikely]] {
        // EDEN_CGT: kind 0 = stream buffer region, 1 = staging buffer.
        Common::CGT::StagingRequest(Common::CGT::HandleBits(ref.buffer), ref.offset, size,
                                    ref.buffer == *stream_buffer ? 0 : 1,
                                    static_cast<u64>(usage), deferred, scheduler.CurrentTick(),
                                    scheduler.GetMasterSemaphore().KnownGpuTick());
    }
    return ref;
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    std::scoped_lock pool_lock{pool_mutex};
    auto& entries = GetCache(ref.usage)[ref.log2_level].entries;
    const auto is_this_one = [&ref](const StagingBuffer& entry) {
        return entry.index == ref.index;
    };
    auto it = std::find_if(entries.begin(), entries.end(), is_this_one);
    ASSERT(it != entries.end());
    ASSERT(it->deferred);
    it->tick = scheduler.CurrentTick();
    it->deferred = false;
}

void StagingBufferPool::TickFrame() {
    std::scoped_lock pool_lock{pool_mutex};
    current_delete_level = (current_delete_level + 1) % NUM_LEVELS;
    ++frame_count;

    ReleaseCache(MemoryUsage::DeviceLocal);
    ReleaseCache(MemoryUsage::Upload);
    ReleaseCache(MemoryUsage::Download);

    // The round robin above visits a level once every NUM_LEVELS frames. A 32 MiB or larger
    // buffer kept that long holds memory a unified-memory GPU shares with the whole system
    // (Mario Kart World kept 2 GiB of idle upload buffers while loading), so release those once
    // they have been idle for a few frames.
    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        for (size_t log2 = LARGE_LEVEL; log2 < NUM_LEVELS; ++log2) {
            auto& staging = (*cache)[log2];
            auto& entries = staging.entries;
            if (entries.empty()) {
                continue;
            }
            const u64 idle_frames = log2 >= HUGE_LEVEL ? HUGE_IDLE_FRAMES : LARGE_IDLE_FRAMES;
            const auto is_idle = [this, idle_frames](const StagingBuffer& entry) {
                return !entry.deferred && entry.last_frame + idle_frames <= frame_count &&
                       scheduler.IsFree(entry.tick);
            };
            entries.erase(std::remove_if(entries.begin(), entries.end(), is_idle), entries.end());
            if (staging.delete_index >= entries.size()) {
                staging.delete_index = 0;
            }
            if (staging.iterate_index > entries.size()) {
                staging.iterate_index = 0;
            }
        }
    }
}

VkDeviceSize StagingBufferPool::TrimFree() {
    std::scoped_lock pool_lock{pool_mutex};
    VkDeviceSize freed = 0;
    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        for (auto& staging : *cache) {
            auto& entries = staging.entries;
            const auto is_free = [this, &freed](const StagingBuffer& entry) {
                if (entry.deferred || !scheduler.IsFree(entry.tick)) {
                    return false;
                }
                freed += entry.size;
                return true;
            };
            entries.erase(std::remove_if(entries.begin(), entries.end(), is_free), entries.end());
            staging.delete_index = 0;
            staging.iterate_index = 0;
        }
    }
    return freed;
}

StagingBufferRef StagingBufferPool::GetStreamBuffer(size_t size) {
    const bool wraps = iterator + size >= stream_buffer_size;
    // A wrapping request goes to offset 0, so check the regions it will use there before changing
    // any state. Checking them only after the wrap left region 0 unchecked for the rest of the lap
    // whenever it was still busy, so later requests overwrote data the GPU had yet to read.
    const bool busy = wraps ? AreRegionsActive(0, Region(size) + 1)
                            : AreRegionsActive(Region(free_iterator) + 1,
                                               (std::min)(Region(iterator + size) + 1, NUM_SYNCS));
    if (busy) {
        // Avoid waiting for the previous usages to be free
        return GetStagingBuffer(size, MemoryUsage::Upload);
    }
    const u64 current_tick = scheduler.CurrentTick();
    std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + Region(iterator),
              current_tick);
    used_iterator = iterator;
    free_iterator = (std::max)(free_iterator, iterator + size);

    if (wraps) {
        std::fill(sync_ticks.begin() + Region(used_iterator), sync_ticks.begin() + NUM_SYNCS,
                  current_tick);
        used_iterator = 0;
        iterator = 0;
        free_iterator = size;
    }
    const size_t offset = iterator;
    iterator = Common::AlignUp(iterator + size, MAX_ALIGNMENT);
    return StagingBufferRef{
        .buffer = *stream_buffer,
        .device_address = stream_buffer_address,
        .offset = static_cast<VkDeviceSize>(offset),
        .mapped_span = stream_pointer.subspan(offset, size),
        .usage{},
        .log2_level{},
        .index{},
    };
}

bool StagingBufferPool::AreRegionsActive(size_t region_begin, size_t region_end) const {
    const u64 gpu_tick = scheduler.GetMasterSemaphore().KnownGpuTick();
    return std::any_of(sync_ticks.begin() + region_begin, sync_ticks.begin() + region_end,
                       [gpu_tick](u64 sync_tick) { return gpu_tick < sync_tick; });
};

StagingBufferRef StagingBufferPool::GetStagingBuffer(size_t size, MemoryUsage usage,
                                                     bool deferred) {
    if (const std::optional<StagingBufferRef> ref = TryGetReservedBuffer(size, usage, deferred)) {
        return *ref;
    }
    return CreateStagingBuffer(size, usage, deferred);
}

std::optional<StagingBufferRef> StagingBufferPool::TryGetReservedBuffer(size_t size,
                                                                        MemoryUsage usage,
                                                                        bool deferred) {
    StagingBuffers& cache_level = GetCache(usage)[Common::Log2Ceil(size)];

    const auto is_free = [this, size](const StagingBuffer& entry) {
        return !entry.deferred && entry.size >= size && scheduler.IsFree(entry.tick);
    };
    auto& entries = cache_level.entries;
    const auto hint_it = entries.begin() + cache_level.iterate_index;
    auto it = std::find_if(entries.begin() + cache_level.iterate_index, entries.end(), is_free);
    if (it == entries.end()) {
        it = std::find_if(entries.begin(), hint_it, is_free);
        if (it == hint_it) {
            return std::nullopt;
        }
    }
    cache_level.iterate_index = std::distance(entries.begin(), it) + 1;
    it->tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick();
    it->last_frame = frame_count;
    ASSERT(!it->deferred);
    it->deferred = deferred;
    return it->Ref();
}

StagingBufferRef StagingBufferPool::CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred) {
    auto const log2_size = Common::Log2Ceil<u32>(u32(size));
    // Large buffers are sized in eighths of their level instead of the full power of two: a
    // 341 MiB texture upload took a 512 MiB buffer. Reuse checks the entry size.
    const VkDeviceSize buffer_size =
        log2_size >= LARGE_LEVEL ? Common::AlignUp(u64{size}, u64{1} << (log2_size - 3))
                                 : u64{1} << log2_size;
    VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = buffer_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    if (device.IsExtTransformFeedbackSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;
    }
    if (device.IsBufferDeviceAddressSupported()) {
        buffer_ci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    // An allocation that does not fit frees the idle staging buffers (TrimFree, installed as the
    // allocator's out-of-memory handler by RasterizerVulkan) and is retried there.
    vk::Buffer buffer = memory_allocator.CreateBuffer(buffer_ci, usage);
    if (device.HasDebuggingToolAttached()) {
        ++buffer_index;
        buffer.SetObjectNameEXT(fmt::format("Staging Buffer {}", buffer_index).c_str());
    }
    const std::span<u8> mapped_span = buffer.Mapped();
    const VkDeviceAddress buffer_address =
        device.IsBufferDeviceAddressSupported()
            ? device.GetLogical().GetBufferDeviceAddress(*buffer)
            : VkDeviceAddress{};
    StagingBuffer& entry = GetCache(usage)[log2_size].entries.emplace_back(StagingBuffer{
        .buffer = std::move(buffer),
        .device_address = buffer_address,
        .mapped_span = mapped_span,
        .usage = usage,
        .log2_level = log2_size,
        .index = unique_ids++,
        .tick = deferred ? (std::numeric_limits<u64>::max)() : scheduler.CurrentTick(),
        .size = buffer_size,
        .last_frame = frame_count,
        .deferred = deferred,
    });
    return entry.Ref();
}

StagingBufferPool::StagingBuffersCache& StagingBufferPool::GetCache(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
        return device_local_cache;
    case MemoryUsage::Upload:
        return upload_cache;
    case MemoryUsage::Download:
        return download_cache;
    default:
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return upload_cache;
    }
}

void StagingBufferPool::ReleaseCache(MemoryUsage usage) {
    ReleaseLevel(GetCache(usage), current_delete_level);
}

void StagingBufferPool::ReleaseLevel(StagingBuffersCache& cache, size_t log2) {
    constexpr size_t deletions_per_tick = 16;
    auto& staging = cache[log2];
    auto& entries = staging.entries;
    const size_t old_size = entries.size();

    const auto is_deletable = [this](const StagingBuffer& entry) {
        return scheduler.IsFree(entry.tick);
    };
    const size_t begin_offset = staging.delete_index;
    const size_t end_offset = (std::min)(begin_offset + deletions_per_tick, old_size);
    const auto begin = entries.begin() + begin_offset;
    const auto end = entries.begin() + end_offset;
    entries.erase(std::remove_if(begin, end, is_deletable), end);

    const size_t new_size = entries.size();
    staging.delete_index += deletions_per_tick;
    if (staging.delete_index >= new_size) {
        staging.delete_index = 0;
    }
    if (staging.iterate_index > new_size) {
        staging.iterate_index = 0;
    }
}

} // namespace Vulkan
