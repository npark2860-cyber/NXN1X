// SPDX-License-Identifier: GPL-2.0-or-later
// Lossless, bounded-memory Vulkan image <-> file transfer. Used by NVN residency
// and by its headless Vulkan round-trip test. The caller owns queue synchronization.
#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>

typedef struct {
    VkFormat format;
    VkImageAspectFlags aspects;
    uint32_t width, height, depth, layers, levels, block_width, block_height, block_bytes;
} nvna_shape;

static uint32_t nvna_dim(uint32_t n, uint32_t level) { return level < 32 && (n >> level) ? n >> level : 1; }
static uint32_t nvna_aspect_bytes(const nvna_shape *s, VkImageAspectFlags aspect) {
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) return 1;
    if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) return s->format == VK_FORMAT_D16_UNORM || s->format == VK_FORMAT_D16_UNORM_S8_UINT ? 2 : 4;
    return s->block_bytes;
}
static uint64_t nvna_size(const nvna_shape *s) {
    if (!s->width || !s->height || !s->depth || !s->layers || !s->levels || s->levels > 32 ||
        !s->block_width || !s->block_height || !s->block_bytes ||
        !s->aspects || (s->aspects & ~(VK_IMAGE_ASPECT_COLOR_BIT | VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) return 0;
    uint64_t total = 0;
    for (unsigned aspect = 1; aspect <= VK_IMAGE_ASPECT_STENCIL_BIT; aspect <<= 1) {
        if (!(s->aspects & aspect)) continue;
        for (uint32_t level = 0; level < s->levels; level++) {
            uint64_t n = ((uint64_t)nvna_dim(s->width, level) + s->block_width - 1) / s->block_width;
            uint64_t factors[] = {((uint64_t)nvna_dim(s->height, level) + s->block_height - 1) / s->block_height,
                                 nvna_dim(s->depth, level), s->layers, nvna_aspect_bytes(s, aspect)};
            for (unsigned i = 0; i < 4; i++) { if (!factors[i] || n > UINT64_MAX / factors[i]) return 0; n *= factors[i]; }
            if (total > UINT64_MAX - n) return 0;
            total += n;
        }
    }
    return total;
}

// Pack independent mip/layer/plane regions into one bounded staging buffer.
// One submission/fence per filled batch, not per tiny mip or array slice.
enum { NVNA_STAGING_BYTES = 4 << 20, NVNA_REGIONS = 128 };
// Caller serializes access. Keep a small transfer reserve across images so
// reclaiming memory does not itself need a new device allocation each time.
typedef struct {
    VkDevice dev; VkBuffer buffer; VkDeviceMemory memory;
    VkCommandPool pool; VkCommandBuffer cmd; VkFence fence;
    uint8_t *mapped; uint64_t capacity, allocation_bytes, non_coherent_atom;
    uint32_t family; int coherent, pending;
    const char *stage;
    int (*cancelled)(void *); void *cancel_context; // optional pinned-source version check
    PFN_vkAllocateMemory allocate; // optional admission adapter, actual requirements
} nvna_workspace;

static VkResult nvna_workspace_wait(nvna_workspace *w) {
    if (!w->pending) return VK_SUCCESS;
    w->stage = "wait for transfer";
    VkResult r = vkWaitForFences(w->dev, 1, &w->fence, VK_TRUE, UINT64_MAX);
    if (r == VK_SUCCESS || r == VK_ERROR_DEVICE_LOST) w->pending = 0;
    return r;
}
static VkResult nvna_workspace_destroy(nvna_workspace *w) {
    VkResult r = nvna_workspace_wait(w);
    if (w->pending) return r; // still in flight: neither reset nor destroy it
    if (w->fence) vkDestroyFence(w->dev, w->fence, NULL);
    if (w->pool) vkDestroyCommandPool(w->dev, w->pool, NULL);
    if (w->mapped) vkUnmapMemory(w->dev, w->memory);
    if (w->buffer) vkDestroyBuffer(w->dev, w->buffer, NULL);
    if (w->memory) vkFreeMemory(w->dev, w->memory, NULL);
    memset(w, 0, sizeof *w);
    return r;
}
static VkResult nvna_workspace_init(nvna_workspace *w, VkDevice dev, uint32_t family,
                                   const VkPhysicalDeviceMemoryProperties *memory, uint64_t capacity) {
    if (w->buffer) {
        if (w->dev != dev || w->family != family || w->capacity < capacity) return VK_ERROR_INITIALIZATION_FAILED;
        return nvna_workspace_wait(w);
    }
    w->dev = dev; w->family = family; w->capacity = capacity;
    VkResult result;
#define NVNA_INIT_CHECK(call) do { w->stage = #call; result = (call); if (result != VK_SUCCESS) goto failed; } while (0)
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, capacity,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    NVNA_INIT_CHECK(vkCreateBuffer(dev, &bi, NULL, &w->buffer));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, w->buffer, &req);
    uint32_t type = UINT32_MAX; int best = -1;
    for (uint32_t i = 0; i < memory->memoryTypeCount; i++) {
        const VkMemoryPropertyFlags flags = memory->memoryTypes[i].propertyFlags;
        if (!(req.memoryTypeBits & (1u << i)) || !(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
        int score = !!(flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) * 2 + !!(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (score > best) { best = score; type = i; w->coherent = !!(flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); }
    }
    if (type == UINT32_MAX) { w->stage = "host visible memory type"; result = VK_ERROR_FEATURE_NOT_PRESENT; goto failed; }
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, type};
    if (w->allocate) { NVNA_INIT_CHECK(w->allocate(dev, &ai, NULL, &w->memory)); }
    else { NVNA_INIT_CHECK(vkAllocateMemory(dev, &ai, NULL, &w->memory)); }
    w->allocation_bytes = req.size;
    NVNA_INIT_CHECK(vkBindBufferMemory(dev, w->buffer, w->memory, 0));
    void *mapped;
    NVNA_INIT_CHECK(vkMapMemory(dev, w->memory, 0, VK_WHOLE_SIZE, 0, &mapped)); w->mapped = mapped;
    VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, family};
    NVNA_INIT_CHECK(vkCreateCommandPool(dev, &pi, NULL, &w->pool));
    VkCommandBufferAllocateInfo ca = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, w->pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    NVNA_INIT_CHECK(vkAllocateCommandBuffers(dev, &ca, &w->cmd));
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    NVNA_INIT_CHECK(vkCreateFence(dev, &fi, NULL, &w->fence));
    w->stage = "ready"; return VK_SUCCESS;
failed: {
    const char *stage = w->stage;
    nvna_workspace_destroy(w); w->stage = stage;
    return result;
}
#undef NVNA_INIT_CHECK
}
typedef struct {
    VkDevice dev; VkQueue queue; VkImage image; VkBuffer buffer;
    VkCommandPool pool; VkCommandBuffer cmd; VkFence fence;
    VkMappedMemoryRange range; uint8_t *mapped;
    const nvna_shape *shape; FILE *file;
    int upload, coherent, first;
    void (*queue_lock)(void); void (*queue_unlock)(void);
    VkBufferImageCopy copies[NVNA_REGIONS];
    size_t bytes[NVNA_REGIONS];
    uint32_t count; uint64_t used, capacity;
    uint64_t *submissions;
    nvna_workspace *workspace;
} nvna_batch;

static VkResult nvna_flush_batch(nvna_batch *b) {
    if (!b->count) return VK_SUCCESS;
    if (b->workspace->cancelled && b->workspace->cancelled(b->workspace->cancel_context)) {
        b->workspace->stage = "preservation version changed"; return VK_NOT_READY;
    }
    VkResult r;
    // A retained 4 MiB reserve must not make every tiny image flush/invalidate
    // 4 MiB. Round the used span to the device's non-coherent atom size.
    if (!b->coherent && b->workspace->non_coherent_atom) {
        uint64_t atom = b->workspace->non_coherent_atom;
        uint64_t bytes = (b->used + atom - 1) / atom * atom;
        b->range.size = bytes < b->workspace->allocation_bytes ? bytes : VK_WHOLE_SIZE;
    }
#define NVNA_BATCH_CHECK(call) do { b->workspace->stage = #call; r = (call); if (r != VK_SUCCESS) return r; } while (0)
    if (b->upload && !b->coherent) NVNA_BATCH_CHECK(vkFlushMappedMemoryRanges(b->dev, 1, &b->range));
    NVNA_BATCH_CHECK(vkResetCommandPool(b->dev, b->pool, 0));
    VkCommandBufferBeginInfo begin = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    NVNA_BATCH_CHECK(vkBeginCommandBuffer(b->cmd, &begin));
    VkImageMemoryBarrier image_bar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, NULL,
        b->upload && b->first ? 0 : VK_ACCESS_MEMORY_WRITE_BIT,
        b->upload ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT,
        b->upload && b->first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, b->image,
        {b->shape->aspects, 0, b->shape->levels, 0, b->shape->layers}};
    vkCmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &image_bar);
    if (b->upload) {
        vkCmdCopyBufferToImage(b->cmd, b->buffer, b->image, VK_IMAGE_LAYOUT_GENERAL, b->count, b->copies);
        image_bar.oldLayout = image_bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        image_bar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &image_bar);
    } else {
        vkCmdCopyImageToBuffer(b->cmd, b->image, VK_IMAGE_LAYOUT_GENERAL, b->buffer, b->count, b->copies);
        VkMemoryBarrier host = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, NULL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
        // Background downloads may interleave with later game writes on this
        // same queue. Order the transfer reads before those subsequent writes.
        if (b->workspace->cancelled) {
            image_bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            image_bar.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(b->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &image_bar);
        }
    }
    NVNA_BATCH_CHECK(vkEndCommandBuffer(b->cmd));
    NVNA_BATCH_CHECK(vkResetFences(b->dev, 1, &b->fence));
    VkSubmitInfo submit = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &b->cmd};
    if (b->queue_lock) b->queue_lock();
    b->workspace->stage = "vkQueueSubmit";
    r = vkQueueSubmit(b->queue, 1, &submit, b->fence);
    if (b->queue_unlock) b->queue_unlock();
    if (r != VK_SUCCESS) return r;
    b->workspace->pending = 1;
    if (b->submissions) ++*b->submissions;
    NVNA_BATCH_CHECK(nvna_workspace_wait(b->workspace));
    if (!b->upload) {
        if (!b->coherent) NVNA_BATCH_CHECK(vkInvalidateMappedMemoryRanges(b->dev, 1, &b->range));
        b->workspace->stage = "write archive file";
        for (uint32_t i = 0; i < b->count; i++)
            if (fwrite(b->mapped + b->copies[i].bufferOffset, 1, b->bytes[i], b->file) != b->bytes[i]) return VK_ERROR_UNKNOWN;
    }
    b->first = 0; b->count = 0; b->used = 0;
    return VK_SUCCESS;
#undef NVNA_BATCH_CHECK
}

// All mips/layers, including separate depth/stencil planes. At most 4 MiB mapped;
// small images allocate only their payload. Multisampled/sparse images are excluded
// by the caller. The caller commits an archive only after this returns success.
static VkResult nvna_stream_cached(nvna_workspace *workspace, VkDevice dev, VkQueue queue, uint32_t family,
                           const VkPhysicalDeviceMemoryProperties *memory,
                           VkImage image, const nvna_shape *shape, FILE *file, int upload,
                           void (*queue_lock)(void), void (*queue_unlock)(void), uint64_t *submissions) {
    VkResult result = VK_ERROR_INITIALIZATION_FAILED;
    uint64_t total = nvna_size(shape);
    if (!image || !file || !total) return result;
    clearerr(file);
    workspace->stage = "seek archive file";
    if (fseek(file, 0, SEEK_SET)) return VK_ERROR_UNKNOWN;
    nvna_batch b = {.dev = dev, .queue = queue, .image = image, .shape = shape, .file = file,
        .upload = upload, .first = 1, .queue_lock = queue_lock, .queue_unlock = queue_unlock,
        .submissions = submissions, .workspace = workspace};
#define NVNA_CHECK(call) do { result = (call); if (result != VK_SUCCESS) goto cleanup; } while (0)
    NVNA_CHECK(nvna_workspace_init(workspace, dev, family, memory, total < NVNA_STAGING_BYTES ? total : NVNA_STAGING_BYTES));
    b.buffer = workspace->buffer; b.pool = workspace->pool; b.cmd = workspace->cmd; b.fence = workspace->fence;
    b.mapped = workspace->mapped; b.capacity = workspace->capacity; b.coherent = workspace->coherent;
    b.range = (VkMappedMemoryRange){VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, NULL, workspace->memory, 0, VK_WHOLE_SIZE};
    // Offset must align to the format's block size, and to four for depth/stencil.
    uint32_t x = shape->block_bytes, y = 4;
    while (y) { uint32_t rem = x % y; x = y; y = rem; }
    const uint64_t alignment = (uint64_t)shape->block_bytes / x * 4;
    for (unsigned aspect = 1; aspect <= VK_IMAGE_ASPECT_STENCIL_BIT; aspect <<= 1) {
        if (!(shape->aspects & aspect)) continue;
        for (uint32_t level = 0; level < shape->levels; level++) {
            uint32_t w = nvna_dim(shape->width, level), h = nvna_dim(shape->height, level), d = nvna_dim(shape->depth, level);
            uint64_t row = ((uint64_t)w + shape->block_width - 1) / shape->block_width * nvna_aspect_bytes(shape, aspect);
            if (!row || row > b.capacity) { workspace->stage = "row exceeds staging capacity"; result = VK_ERROR_FORMAT_NOT_SUPPORTED; goto cleanup; }
            uint32_t rows = (uint32_t)(((uint64_t)h + shape->block_height - 1) / shape->block_height);
            for (uint32_t layer = 0; layer < shape->layers; layer++) for (uint32_t z = 0; z < d; z++) {
                for (uint32_t by = 0; by < rows;) {
                    uint64_t offset = (b.used + alignment - 1) / alignment * alignment;
                    if (b.count == NVNA_REGIONS || offset + row > b.capacity) {
                        NVNA_CHECK(nvna_flush_batch(&b)); offset = 0;
                    }
                    uint64_t fit = (b.capacity - offset) / row;
                    uint32_t count = rows - by < fit ? rows - by : (uint32_t)fit;
                    size_t bytes = (size_t)(row * count);
                    if (upload) {
                        workspace->stage = "read archive file";
                        if (fread(b.mapped + offset, 1, bytes, file) != bytes) { result = VK_ERROR_UNKNOWN; goto cleanup; }
                    }
                    uint32_t py = by * shape->block_height, ph = count * shape->block_height;
                    if (ph > h - py) ph = h - py;
                    b.copies[b.count] = (VkBufferImageCopy){offset, 0, 0, {aspect, level, layer, 1}, {0, (int32_t)py, (int32_t)z}, {w, ph, 1}};
                    b.bytes[b.count++] = bytes; b.used = offset + bytes; by += count;
                }
            }
        }
    }
    NVNA_CHECK(nvna_flush_batch(&b));
    workspace->stage = "flush archive file";
    result = !upload && fflush(file) ? VK_ERROR_UNKNOWN : VK_SUCCESS;
    if (result==VK_SUCCESS && workspace->cancelled && workspace->cancelled(workspace->cancel_context))
        result=VK_NOT_READY; // never publish mixed versions, including the last batch
cleanup:
    return result;
#undef NVNA_CHECK
}
// Transient compatibility entry point for standalone callers. Production owns
// a persistent workspace and separately retains any image whose wait failed.
static VkResult nvna_stream_counted(VkDevice dev, VkQueue queue, uint32_t family,
                           const VkPhysicalDeviceMemoryProperties *memory,
                           VkImage image, const nvna_shape *shape, FILE *file, int upload,
                           void (*queue_lock)(void), void (*queue_unlock)(void), uint64_t *submissions) {
    nvna_workspace workspace = {0};
    VkResult r = nvna_stream_cached(&workspace, dev, queue, family, memory, image, shape, file, upload, queue_lock, queue_unlock, submissions);
    // If a wait failed, destroy waits again and leaves in-flight resources alive
    // on another error rather than freeing memory still owned by the GPU.
    nvna_workspace_destroy(&workspace);
    return r;
}
static VkResult nvna_stream(VkDevice dev, VkQueue queue, uint32_t family,
                           const VkPhysicalDeviceMemoryProperties *memory,
                           VkImage image, const nvna_shape *shape, FILE *file, int upload,
                           void (*queue_lock)(void), void (*queue_unlock)(void)) {
    return nvna_stream_counted(dev, queue, family, memory, image, shape, file, upload, queue_lock, queue_unlock, NULL);
}
