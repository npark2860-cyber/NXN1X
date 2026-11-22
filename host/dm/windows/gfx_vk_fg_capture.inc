// NVN replay thread. Capture two adjacent frames without retaining guest images.
// Each copy is queued where the attachment or descriptor is used. Only the staging
// buffers survive beyond submission, until the host's completion fence signals.
static const char *fg_capture_dir;
static unsigned fg_frames_left, fg_capture_run, fg_samples;
static int fg_capture_init, fg_key_down;
static struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t fence;
    size_t size;
    char name[192];
} fg_pending[128];
static unsigned fg_pending_count;
static uint64_t fg_shaders[32];
static unsigned fg_shader_count;
static void fg_capture_uniforms(uint64_t shader, uint64_t fragment) {
    if (!shader || fg_shader_count == 32) return;
    for (unsigned i = 0; i < fg_shader_count; ++i)
        if (fg_shaders[i] == shader) return;
    fg_shaders[fg_shader_count++] = shader;
    extern int vk_gpu_pending(uint64_t, uint64_t);
    for (unsigned slot = 0; slot < 16; ++slot) {
        const p_bind_buf *u = &S.ubo[0][slot];
        unsigned n = u->size < 4096 ? (unsigned)u->size : 4096;
        if (!u->addr || !n || u->addr > UINT64_MAX - n || !vk_pool_of(u->addr,n) ||
            vk_gpu_pending(u->addr,n)) continue;
        char path[1024];
        int len = snprintf(path, sizeof path, "%s/%u-%llu-vs%016llx-fs%016llx-b%u.ubo",
            fg_capture_dir, fg_capture_run, (unsigned long long)g_tex_frame,
            (unsigned long long)shader, (unsigned long long)fragment, slot);
        if (len < 0 || len >= (int)sizeof path) continue;
        FILE *file = fopen(path, "wb");
        if (!file) continue;
        uint8_t data[4096];
        vk_ubd_read(u->addr, n, data);
        fwrite(data, 1, n, file);
        fclose(file);
    }
}
__declspec(dllimport) short __stdcall GetAsyncKeyState(int key);

static void fg_capture_poll(uint64_t completed) {
    static uint64_t capture_at;
    if (!fg_capture_init) {
        fg_capture_dir = getenv("NXN1X_FG_CAPTURE_DIR");
        const char *frame = getenv("NXN1X_CAPTURE_PRESENT");
        if (frame) capture_at = strtoull(frame, NULL, 10);
        fg_capture_init = 1;
    }
    if (!fg_capture_dir) return;
    for (unsigned i = 0; i < fg_pending_count;) {
        if (fg_pending[i].fence > completed) { ++i; continue; }
        void *data;
        if (vkMapMemory(V.dev, fg_pending[i].memory, 0, VK_WHOLE_SIZE, 0, &data) == VK_SUCCESS) {
            char path[1024];
            if (snprintf(path, sizeof path, "%s/%s", fg_capture_dir, fg_pending[i].name) < (int)sizeof path) {
                FILE *f = fopen(path, "wb");
                if (f) { fwrite(data, 1, fg_pending[i].size, f); fclose(f); }
            }
            vkUnmapMemory(V.dev, fg_pending[i].memory);
        }
        vkDestroyBuffer(V.dev, fg_pending[i].buffer, NULL);
        vkFreeMemory(V.dev, fg_pending[i].memory, NULL);
        fg_pending[i] = fg_pending[--fg_pending_count];
    }
    if (fg_frames_left) --fg_frames_left;
    const int pressed = GetAsyncKeyState(0x11) < 0 && GetAsyncKeyState(0x10) < 0 && GetAsyncKeyState(0x7a) < 0;
    const int scheduled = capture_at && g_tex_frame >= capture_at;
    if ((scheduled || (pressed && !fg_key_down)) && !fg_frames_left && !fg_pending_count) {
        if (scheduled) capture_at = 0;
        ++fg_capture_run;
        fg_frames_left = 2;
        VKLOG("FG capture %u: recording two NVN frames in %s", fg_capture_run, fg_capture_dir);
    }
    fg_key_down = pressed;
    fg_samples = 0;
    fg_shader_count = 0;
}

static unsigned fg_texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_D16_UNORM: return 2;
    case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return 4;
    case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    default: return 0;
    }
}

static void fg_capture_image(vkimg_t *im, unsigned level, unsigned layer, const char *role,
                             unsigned slot, uint64_t shader) {
    if (!fg_frames_left || !im || level >= im->levels || layer >= im->layers) return;
    const unsigned w = im->w >> level, h = im->h >> level;
    const unsigned bytes = fg_texel_bytes(im->fmt);
    if (w < 32 || h < 32 || !bytes || im->samples != 1 || im->vtype == VK_IMAGE_VIEW_TYPE_3D ||
        im->layout != VK_IMAGE_LAYOUT_GENERAL || !(im->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return;
    // Reserve space for compute bindings and the final color even in draw-heavy frames.
    int final = !strcmp(role, "present");
    unsigned limit = !strcmp(role, "attachment") ? 40 : final ? 64 : 63;
    if (fg_samples >= limit || fg_pending_count >= 128) return;
    const VkDeviceSize size = 32 * 32 * bytes;
    VkBufferCreateInfo info = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer;
    if (vkCreateBuffer(V.dev, &info, NULL, &buffer) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(V.dev, buffer, &req);
    uint32_t type = vk_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, type};
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (type == UINT32_MAX || vkAllocateMemory(V.dev, &ai, NULL, &mem) != VK_SUCCESS ||
        vkBindBufferMemory(V.dev, buffer, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(V.dev, buffer, NULL);
        if (mem) vkFreeMemory(V.dev, mem, NULL);
        return;
    }
    const VkImageAspectFlags aspect = im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkBufferImageCopy copy = {0, 0, 0, {aspect, level, layer, 1}, {(int)(w-32)/2, (int)(h-32)/2, 0}, {32,32,1}};
    VkCommandBuffer cb = vk_cmd();
    vk_img_barrier_all(cb);
    vkCmdCopyImageToBuffer(cb, im->img, im->layout, buffer, 1, &copy);
    vk_img_barrier_all(cb);
    typeof(fg_pending[0]) *r = &fg_pending[fg_pending_count++];
    r->buffer = buffer; r->memory = mem; r->size = (size_t)size; r->fence = mtl_last_fence() + 1;
    im->last_use = r->fence;
    snprintf(r->name, sizeof r->name, "%u-%llu-%u-%s%u-%016llx-img%llu-%ux%u-m%u-l%u-f%u.raw",
             fg_capture_run, (unsigned long long)g_tex_frame, fg_samples++, role, slot,
             (unsigned long long)shader, (unsigned long long)im->residency_id, w, h, level, layer, (unsigned)im->fmt);
}
