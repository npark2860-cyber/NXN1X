#include "gfx_vk_fg_math.h"

// All discovery state belongs to the serial NVN replay stream. One outstanding
// readback of 16 depth/motion texels; never wait for it on the rendering thread.
static struct {
    uint64_t frame, shader, draw_frame, last_depth, motion_shader;
    unsigned bank, budget, slot, width, height, votes, convention;
    uint64_t sampled_frame, searched[4], camera_address;
    VkFormat format;
    fg_offsets offsets;
    fg_matrix view, projection, previous_clip, current_clip;
    int camera, history, pass_camera;
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t fence;
    fg_matrix sampled_transform, sampled_projection;
    VkFormat depth_format;
} fg_auto;

static struct {
    unsigned draws, depth_draws, buffers, cameras, motion_passes, usable_pairs, tags, samples, useful;
} fg_counts;
static void fg_auto_draw(void) {
    ++fg_counts.draws;
    if (!S.sh[0] || !P_da.imageView || !S.ds.dt || S.pass_w < 320 || S.pass_w <= S.pass_h || !S.ncol)
        return;
    ++fg_counts.depth_draws;
    uint64_t frame = g_tex_frame + 1, shader = S.sh[0]->key;
    if (fg_auto.frame != frame) {
        fg_auto.history = fg_auto.camera && fg_auto.frame + 1 == frame;
        fg_auto.previous_clip = fg_auto.current_clip;
        fg_auto.camera = fg_auto.pass_camera = 0;
        fg_auto.frame = frame;
        fg_auto.budget = 4;
        memset(fg_auto.searched, 0, sizeof fg_auto.searched);
    }
    if (fg_auto.camera) {
        if (shader != fg_auto.shader || !P_att[8].im || P_att[8].im->residency_id != fg_auto.last_depth)
            return;
        fg_auto.pass_camera = 1;
        if (S.ubo[0][fg_auto.bank].addr == fg_auto.camera_address)
            return;
    }
    if (fg_auto.shader && shader != fg_auto.shader) {
        if (frame - fg_auto.draw_frame < 8)
            return;
        fg_auto.shader = 0;
        fg_auto.history = 0;
    }
    if (!fg_auto.budget)
        return;
    if (!fg_auto.camera)
        for (unsigned i = 0; i < 4; ++i)
            if (fg_auto.searched[i] == shader)
                return;
    if (!fg_auto.shader && ((shader ^ frame) & 3))
        return;
    fg_auto.searched[4 - fg_auto.budget] = shader;
    --fg_auto.budget;
    extern int vk_gpu_pending(uint64_t, uint64_t);
    for (unsigned bank = 0; bank < 16; ++bank) {
        if (fg_auto.shader && bank != fg_auto.bank)
            continue;
        const p_bind_buf *u = &S.ubo[0][bank];
        unsigned bytes = u->size < 4096 ? (unsigned)u->size : 4096;
        if (!u->addr || bytes < 176 || u->addr > UINT64_MAX - bytes || !vk_pool_of(u->addr, bytes) ||
            vk_gpu_pending(u->addr, bytes))
            continue;
        ++fg_counts.buffers;
        float data[1024];
        vk_ubd_read(u->addr, bytes, (uint8_t *)data);
        fg_offsets o = fg_auto.offsets;
        if (!fg_auto.shader) {
            if (!fg_find_camera(data, bytes / 4, &o))
                continue;
        } else if (o.view + (o.transposed ? 12 : 16) > bytes / 4 || o.projection + 16 > bytes / 4 || o.combined + 16 > bytes / 4)
            continue;
        fg_matrix view = fg_decode(data, o.view, o.transposed, 1), p = fg_decode(data, o.projection, o.transposed, 0);
        fg_matrix combined = fg_decode(data, o.combined, o.transposed, 0);
        if (!fg_rigid(&view) || !fg_projection(&p) || !fg_matrix_match(fg_product(view, p), combined)) {
            if (frame - fg_auto.draw_frame >= 8)
                fg_auto.shader = 0;
            continue;
        }
        const float aspect = fabsf(p.v[5] / p.v[0]);
        if (fabsf(aspect - (float)S.pass_w / S.pass_h) > .08f * aspect)
            continue;
        if (!P_att[8].im)
            return;
        fg_matrix world = fg_inverse_rigid(view);
        if (!fg_auto.shader) {
            fg_auto.votes = 0;
            VKLOG("FG auto camera candidate VS %016llx UBO %u", (unsigned long long)shader, bank);
        }
        fg_auto.shader = shader;
        fg_auto.bank = bank;
        fg_auto.offsets = o;
        fg_auto.camera_address = u->addr;
        fg_auto.draw_frame = frame;
        fg_auto.last_depth = P_att[8].im->residency_id;
        fg_auto.camera = fg_auto.pass_camera = 1;
        fg_auto.view = world;
        fg_auto.projection = p;
        fg_auto.current_clip = combined;
        ++fg_counts.cameras;
        nvn_fg_camera(world.v, p.v);
        return;
    }
}

static void fg_auto_dispose(void) {
    if (fg_auto.buffer)
        vkDestroyBuffer(V.dev, fg_auto.buffer, NULL);
    if (fg_auto.memory)
        vkFreeMemory(V.dev, fg_auto.memory, NULL);
    fg_auto.buffer = VK_NULL_HANDLE;
    fg_auto.memory = VK_NULL_HANDLE;
    fg_auto.fence = 0;
}
void vk_fg_auto_drain(void) {
    if (fg_auto.fence && fg_auto.fence > vk_fence_done())
        return;
    fg_auto_dispose();
    memset(&fg_auto, 0, sizeof fg_auto);
}
static void fg_auto_poll(void) {
    if (!fg_auto.fence || fg_auto.fence > vk_fence_done())
        return;
    void *raw = NULL;
    if (vkMapMemory(V.dev, fg_auto.memory, 0, VK_WHOLE_SIZE, 0, &raw) == VK_SUCCESS) {
        unsigned hits[12] = {0}, useful = 0;
        for (unsigned i = 0; i < 16; ++i) {
            const uint8_t *d = (uint8_t *)raw + 16 * i;
            float z, mx, my;
            if (fg_auto.depth_format == VK_FORMAT_D16_UNORM) {
                uint16_t n;
                memcpy(&n, d, 2);
                z = n / 65535.f;
            } else if (fg_auto.depth_format == VK_FORMAT_D24_UNORM_S8_UINT) {
                uint32_t n;
                memcpy(&n, d, 4);
                z = (n & 0xffffff) / 16777215.f;
            } else
                memcpy(&z, d, 4);
            if (fg_auto.format == VK_FORMAT_R32G32_SFLOAT) {
                memcpy(&mx, d + 8, 4);
                memcpy(&my, d + 12, 4);
            } else if (fg_auto.format == VK_FORMAT_R16G16_SNORM) {
                int16_t xy[2];
                memcpy(xy, d + 8, 4);
                mx = xy[0] / 32767.f;
                my = xy[1] / 32767.f;
            } else {
                _Float16 xy[2];
                memcpy(xy, d + 8, 4);
                mx = xy[0];
                my = xy[1];
            }
            float u = ((unsigned)((2 * (i % 4) + 1) * fg_auto.width / 8) + .5f) / fg_auto.width;
            float v = ((unsigned)((2 * (i / 4) + 1) * fg_auto.height / 8) + .5f) / fg_auto.height, dx, dy;
            if (!fg_expected_motion(u, v, z, fg_auto.sampled_projection, fg_auto.sampled_transform, &dx, &dy))
                continue;
            float px = dx * fg_auto.width, py = dy * fg_auto.height;
            if (fabsf(px) + fabsf(py) < 2 || !isfinite(mx) || !isfinite(my))
                continue;
            ++useful;
            for (unsigned c = 0; c < 12; ++c) {
                float sx, sy;
                fg_motion_units(c, fg_auto.width, fg_auto.height, &sx, &sy);
                if (fabsf(mx * sx * fg_auto.width - px) + fabsf(my * sy * fg_auto.height - py) <
                    .6f + .08f * (fabsf(px) + fabsf(py)))
                    ++hits[c];
            }
        }
        ++fg_counts.samples;
        fg_counts.useful += useful;
        int best = fg_winning_convention(hits, useful);
        if (best >= 0) {
            if (fg_auto.convention != best)
                fg_auto.votes = 0;
            fg_auto.convention = best;
            if (fg_auto.votes < 3 && ++fg_auto.votes == 3)
                VKLOG("FG auto motion confirmed FS %016llx attachment %u convention %u",
                      (unsigned long long)fg_auto.motion_shader, fg_auto.slot, best);
        } else if (useful >= 8 && hits[fg_auto.convention] * 2 < useful)
            fg_auto.votes = 0;
        vkUnmapMemory(V.dev, fg_auto.memory);
    }
    fg_auto_dispose();
}
static void fg_auto_sample(vkimg_t *depth, vkimg_t *motion) {
    if (fg_auto.buffer || !fg_auto.history || fg_auto.sampled_frame == fg_auto.frame)
        return;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             NULL,
                             0,
                             256,
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VK_SHARING_MODE_EXCLUSIVE};
    if (vkCreateBuffer(V.dev, &bi, NULL, &fg_auto.buffer) != VK_SUCCESS)
        return;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(V.dev, fg_auto.buffer, &req);
    uint32_t type = vk_mem_type(
        req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, type};
    if (type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &fg_auto.memory) != VK_SUCCESS ||
        vkBindBufferMemory(V.dev, fg_auto.buffer, fg_auto.memory, 0) != VK_SUCCESS) {
        fg_auto_dispose();
        return;
    }
    VkCommandBuffer cb = vk_cmd();
    vk_img_barrier_all(cb);
    for (unsigned j = 0; j < 2; ++j) {
        vkimg_t *im = j ? motion : depth;
        VkBufferImageCopy copies[16];
        for (unsigned i = 0; i < 16; ++i)
            copies[i] = (VkBufferImageCopy){
                16 * i + 8 * j,
                0,
                0,
                {j ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
                {(int)((2 * (i % 4) + 1) * im->w / 8), (int)((2 * (i / 4) + 1) * im->h / 8), 0},
                {1, 1, 1}};
        vkCmdCopyImageToBuffer(cb, im->img, im->layout, fg_auto.buffer, 16, copies);
        im->last_use = mtl_last_fence() + 1;
    }
    VkBufferMemoryBarrier visible = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                     NULL,
                                     VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_HOST_READ_BIT,
                                     VK_QUEUE_FAMILY_IGNORED,
                                     VK_QUEUE_FAMILY_IGNORED,
                                     fg_auto.buffer,
                                     0,
                                     VK_WHOLE_SIZE};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                         &visible, 0, NULL);
    vk_img_barrier_all(cb);
    fg_auto.fence = mtl_last_fence() + 1;
    fg_auto.sampled_frame = fg_auto.frame;
    fg_auto.depth_format = depth->fmt;
    fg_auto.sampled_projection = fg_auto.projection;
    fg_auto.sampled_transform = fg_product(fg_auto.view, fg_auto.previous_clip);
}
