// SPDX-License-Identifier: GPL-3.0-or-later
// NVN path, step 6 (NXN1X docs NVN_HOST_BRIDGE.md 4.5): see nvn_display.h.
// Ordering. Read after write: the host submits a frame's work on the graphics queue (under Scheduler::submit_mutex) before its present
// reaches the compositor, so the copy, recorded later and submitted later on the same queue, comes after it in submission order; the
// barrier before the copy (all commands, all memory writes) makes those writes visible to it. The host's timeline value is not
// waited on: the host submits at commit, not from a thread of its own, so nothing of the frame can still be unsubmitted.
// Write after read: the present goes back only when the copy's submission has finished on the GPU (the scheduler's timeline, watched
// by the release thread), and the game draws into the texture only after its acquire saw that.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "common/logging.h"
#include "core/nvn/nvn_layer.h"
#include "video_core/framebuffer_config.h"
#include "video_core/renderer_vulkan/nvn_display.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::NvnDisplay {
namespace {

struct Release {
    u64 tick;   // the scheduler tick whose submission holds the copy
    u64 ticket; // the present
};

struct State {
    const Device* device{};
    Scheduler* scheduler{};
    vk::Image image;
    vk::ImageView view;
    u32 width{};
    u32 height{};
    VkFormat format{VK_FORMAT_UNDEFINED};
    bool fresh{}; // the image was just made: its first copy starts from UNDEFINED
    u64 copied_ticket{};
    u64 last_use_tick{};
    std::vector<Release> copied_now; // copies recorded in the composite going on (GPU thread)
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Release> releases; // copies waiting for the GPU
    std::jthread thread;
    u64 copies{};
    std::atomic<u64> generation{};
};

State g;

// The window texture's bytes go to the screen as they are, whatever its sRGB-ness (NVN_VULKAN_DETAIL C18, as the Maxwell path's
// display): an sRGB texture is copied into the UNORM format of the same layout and sampled as that
VkFormat UnormOf(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_SRGB:
        return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:
        return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default:
        return format;
    }
}

void ReleaseLoop(std::stop_token stop) {
    while (!stop.stop_requested()) {
        Release next{};
        {
            std::unique_lock lock{g.mutex};
            if (!g.cv.wait(lock, stop, [] { return !g.releases.empty(); })) {
                return;
            }
            next = g.releases.front();
        }
        MasterSemaphore& semaphore = g.scheduler->GetMasterSemaphore();
        while (!semaphore.IsFree(next.tick) && !stop.stop_requested()) {
            semaphore.Refresh();
            if (!semaphore.IsFree(next.tick)) {
                std::this_thread::sleep_for(std::chrono::microseconds(250));
            }
        }
        if (stop.stop_requested()) {
            return;
        }
        {
            std::scoped_lock lock{g.mutex};
            g.releases.pop_front();
        }
        Core::NvnLayer::OutputDone(next.ticket);
    }
}

void MakeImage(const Device& device, MemoryAllocator& allocator, u32 width, u32 height, VkFormat format) {
    auto image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    auto view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    });
    // Publish together; destroy the old view before its image on success.
    g.view = std::move(view);
    g.image = std::move(image);
    g.width = width;
    g.height = height;
    g.format = format;
    g.fresh = true;
    g.generation.fetch_add(1, std::memory_order_relaxed);
    LOG_INFO(Render_Vulkan, "NVN output: copy image {}x{}, VkFormat {}", width, height, static_cast<u32>(format));
}

} // namespace

std::optional<FramebufferTextureInfo> Frame(const Device& device, MemoryAllocator& allocator, Scheduler& scheduler,
                                            const Tegra::FramebufferConfig& config) {
    g.device = &device;
    g.scheduler = &scheduler;
    if (!g.thread.joinable()) {
        g.thread = std::jthread([](std::stop_token stop) { ReleaseLoop(stop); });
    }
    if (config.width == 0 || config.height == 0) {
        return std::nullopt;
    }
    if (config.nvn_ticket != g.copied_ticket || !g.image) {
        const VkFormat format = UnormOf(static_cast<VkFormat>(config.nvn_format));
        const u32 scale = config.nvn_scaled_2x ? 2 : 1;
        const u32 width = config.width * scale;
        const u32 height = config.height * scale;
        if (!g.image || g.width != width || g.height != height || g.format != format) {
            if (g.image) {
                scheduler.Wait(g.last_use_tick); // composites still reading the old copy
            }
            MakeImage(device, allocator, width, height, format);
        }
        scheduler.RequestOutsideRenderPassOperationContext();
        const VkImage src = reinterpret_cast<VkImage>(static_cast<uintptr_t>(config.nvn_image));
        scheduler.Record([src, dst = *g.image, width = g.width, height = g.height,
                          fresh = g.fresh](vk::CommandBuffer cmdbuf) {
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const std::array<VkImageMemoryBarrier, 2> before{{
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = src,
                    .subresourceRange = range,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = fresh ? VkAccessFlags{0} : VkAccessFlags{VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT},
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .oldLayout = fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = dst,
                    .subresourceRange = range,
                },
            }};
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);
            const VkImageCopy region{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .srcOffset = {0, 0, 0},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .dstOffset = {0, 0, 0},
                .extent = {width, height, 1},
            };
            cmdbuf.CopyImage(src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_GENERAL, region);
            const VkImageMemoryBarrier after{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst,
                .subresourceRange = range,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, after);
        });
        g.fresh = false;
        g.copied_ticket = config.nvn_ticket;
        g.copied_now.push_back({scheduler.CurrentTick(), config.nvn_ticket});
        ++g.copies;
    }
    g.last_use_tick = scheduler.CurrentTick();
    return FramebufferTextureInfo{
        .image = *g.image,
        .image_view = *g.view,
        .width = config.width,
        .height = config.height,
        .scaled_width = g.width,
        .scaled_height = g.height,
    };
}

void Finish(std::span<const Tegra::FramebufferConfig> framebuffers) {
    std::vector<u64> now;
    {
        std::scoped_lock lock{g.mutex};
        for (const Release& r : g.copied_now) {
            g.releases.push_back(r);
        }
        for (const Tegra::FramebufferConfig& fb : framebuffers) {
            if (fb.nvn_image == 0) {
                continue;
            }
            bool copied = fb.nvn_ticket == g.copied_ticket && g.image;
            for (const Release& r : g.copied_now) {
                copied = copied || r.ticket == fb.nvn_ticket;
            }
            if (!copied) {
                now.push_back(fb.nvn_ticket); // never read: back at once
            }
        }
        g.copied_now.clear();
    }
    g.cv.notify_one();
    for (const u64 ticket : now) {
        Core::NvnLayer::OutputDone(ticket);
    }
}

u64 Generation() {
    return g.generation.load(std::memory_order_relaxed);
}

void Shutdown() {
    if (g.thread.joinable()) {
        g.thread.request_stop();
        g.thread.join();
    }
    std::deque<Release> left;
    {
        std::scoped_lock lock{g.mutex};
        left.swap(g.releases);
    }
    // The renderer drains pending copies before this on a composite failure.
    // Do not allocate a second queue or return tickets before GPU completion.
    if (g.device != nullptr) {
        void(g.device->GetLogical().WaitIdle());
    }
    for (const Release& r : left) {
        Core::NvnLayer::OutputDone(r.ticket);
    }
    for (const Release& r : g.copied_now) {
        Core::NvnLayer::OutputDone(r.ticket);
    }
    g.copied_now.clear();
    g.view = {};
    g.image = {};
    g.copied_ticket = 0;
    g.scheduler = nullptr;
    g.device = nullptr;
    if (g.copies != 0) {
        LOG_INFO(Render_Vulkan, "NVN output: {} presents copied to the screen", g.copies);
    }
}

} // namespace Vulkan::NvnDisplay
