#include <array>
#include <cstdio>
#include <sl.h>
#include <sl_dlss.h>
#include <stdexcept>
#include <string>
#include <vulkan/vulkan.h>
#include <windows.h>

namespace {
void check(int result, const char *call) {
    if (result)
        throw std::runtime_error(std::string(call) + "=" + std::to_string(result));
}
template <class T> T *sdk(HMODULE lib, const char *name) {
    auto fn = reinterpret_cast<T *>(GetProcAddress(lib, name));
    if (!fn)
        throw std::runtime_error(name);
    return fn;
}
struct Image {
    VkImage image{};
    VkDeviceMemory memory{};
    VkImageView view{};
    sl::Resource tag{};
};
struct Fixture {
    VkDevice device;
    VkPhysicalDeviceMemoryProperties properties;
    std::array<Image, 4> images{};
    VkCommandPool pool{};
    VkBuffer buffer{};
    VkDeviceMemory bufferMemory{};
    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) const {
        for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (properties.memoryTypes[i].propertyFlags & flags) == flags)
                return i;
        throw std::runtime_error("no matching memory type");
    }
    VkDeviceMemory allocate(VkMemoryRequirements r, VkMemoryPropertyFlags flags) {
        VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        info.allocationSize = r.size;
        info.memoryTypeIndex = memoryType(r.memoryTypeBits, flags);
        VkDeviceMemory memory{};
        check(vkAllocateMemory(device, &info, nullptr, &memory), "allocate");
        return memory;
    }
    void createImage(Image &img, uint32_t w, uint32_t h, VkFormat format) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {w, h, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (&img == &images.back())
            info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        check(vkCreateImage(device, &info, nullptr, &img.image), "createImage");
        VkMemoryRequirements r{};
        vkGetImageMemoryRequirements(device, img.image, &r);
        img.memory = allocate(r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindImageMemory(device, img.image, img.memory, 0), "bindImage");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = img.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &view, nullptr, &img.view), "createView");
        img.tag = sl::Resource(sl::ResourceType::eTex2d, img.image, img.memory, img.view,
                               VK_IMAGE_LAYOUT_GENERAL);
        img.tag.width = w;
        img.tag.height = h;
        img.tag.nativeFormat = format;
        img.tag.mipLevels = img.tag.arrayLayers = 1;
        img.tag.flags = 0;
        img.tag.usage = info.usage;
    }
    ~Fixture() {
        vkDeviceWaitIdle(device);
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, bufferMemory, nullptr);
        for (auto &i : images) {
            vkDestroyImageView(device, i.view, nullptr);
            vkDestroyImage(device, i.image, nullptr);
            vkFreeMemory(device, i.memory, nullptr);
        }
    }
};
} // namespace

// A stationary orthographic test scene. Zero motion here is known ground truth for this fixture.
bool TestDlssSR(HMODULE lib, VkPhysicalDevice physical, VkDevice device, uint32_t family) try {
    const sl::ViewportHandle vp(1);
    sl::DLSSOptions opts{};
    opts.mode = sl::DLSSMode::eMaxPerformance;
    opts.outputWidth = 1280;
    opts.outputHeight = 720;
    opts.colorBuffersHDR = sl::Boolean::eFalse;
    opts.useAutoExposure = sl::Boolean::eTrue;
    void *fn{};
    check(int(sdk<PFun_slGetFeatureFunction>(lib, "slGetFeatureFunction")(sl::kFeatureDLSS,
                                                                          "slDLSSSetOptions", fn)),
          "lookup options");
    check(int(reinterpret_cast<PFun_slDLSSSetOptions *>(fn)(vp, opts)), "set options");
    Fixture f{device, {}};
    vkGetPhysicalDeviceMemoryProperties(physical, &f.properties);
    f.createImage(f.images[0], 640, 360, VK_FORMAT_R8G8B8A8_UNORM);
    f.createImage(f.images[1], 640, 360, VK_FORMAT_R32_SFLOAT);
    f.createImage(f.images[2], 640, 360, VK_FORMAT_R16G16_SFLOAT);
    f.createImage(f.images[3], 1280, 720, VK_FORMAT_R8G8B8A8_UNORM);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 4;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(device, &bi, nullptr, &f.buffer), "readback");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, f.buffer, &req);
    f.bufferMemory =
        f.allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkBindBufferMemory(device, f.buffer, f.bufferMemory, 0), "bind readback");
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = family;
    check(vkCreateCommandPool(device, &pi, nullptr, &f.pool), "pool");
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = f.pool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd{};
    check(vkAllocateCommandBuffers(device, &ai, &cmd), "command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(cmd, &begin), "begin");
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkClearColorValue colors[] = {
        {{0.8f, 0.2f, 0.05f, 1}}, {{0.5f, 0, 0, 0}}, {{0, 0, 0, 0}}, {{0, 1, 0, 1}}};
    for (size_t i = 0; i < f.images.size(); ++i) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = f.images[i].image;
        barrier.subresourceRange = range;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
        vkCmdClearColorImage(cmd, barrier.image, VK_IMAGE_LAYOUT_GENERAL, &colors[i], 1, &range);
    }
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         1, &memory, 0, nullptr, 0, nullptr);
    sl::FrameToken *frame{};
    uint32_t index = 1;
    check(int(sdk<PFun_slGetNewFrameToken>(lib, "slGetNewFrameToken")(frame, &index)),
          "frame token");
    sl::Constants c{};
    for (auto m : {&c.cameraViewToClip, &c.clipToCameraView, &c.clipToLensClip, &c.clipToPrevClip,
                   &c.prevClipToClip}) {
        m->setRow(0, {1, 0, 0, 0});
        m->setRow(1, {0, 1, 0, 0});
        m->setRow(2, {0, 0, 1, 0});
        m->setRow(3, {0, 0, 0, 1});
    }
    c.jitterOffset = {0, 0};
    c.mvecScale = {1.f / 640, 1.f / 360};
    c.cameraPinholeOffset = {0, 0};
    c.cameraPos = {0, 0, 0};
    c.cameraUp = {0, 1, 0};
    c.cameraRight = {1, 0, 0};
    c.cameraFwd = {0, 0, 1};
    c.cameraNear = 0;
    c.cameraFar = 1;
    c.cameraFOV = 1;
    c.cameraAspectRatio = 1280.f / 720;
    c.orthographicProjection = c.cameraMotionIncluded = c.reset = sl::Boolean::eTrue;
    c.depthInverted = c.motionVectors3D = sl::Boolean::eFalse;
    check(int(sdk<PFun_slSetConstants>(lib, "slSetConstants")(c, *frame, vp)), "constants");
    const sl::BufferType roles[] = {sl::kBufferTypeScalingInputColor, sl::kBufferTypeDepth,
                                    sl::kBufferTypeMotionVectors,
                                    sl::kBufferTypeScalingOutputColor};
    sl::ResourceTag tags[4]{};
    for (size_t i = 0; i < 4; ++i)
        tags[i] =
            sl::ResourceTag(&f.images[i].tag, roles[i], sl::ResourceLifecycle::eValidUntilPresent);
    check(int(sdk<PFun_slSetTagForFrame>(lib, "slSetTagForFrame")(*frame, vp, tags, 4, cmd)),
          "tag");
    const sl::BaseStructure *inputs[] = {&vp};
    check(int(sdk<PFun_slEvaluateFeature>(lib, "slEvaluateFeature")(sl::kFeatureDLSS, *frame,
                                                                    inputs, 1, cmd)),
          "evaluate");
    memory.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &memory, 0, nullptr, 0, nullptr);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset = {640, 360, 0};
    copy.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(cmd, f.images[3].image, VK_IMAGE_LAYOUT_GENERAL, f.buffer, 1, &copy);
    memory.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    memory.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &memory, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(cmd), "end");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    check(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "submit");
    check(vkQueueWaitIdle(queue), "completion");
    void *mapped{};
    check(vkMapMemory(device, f.bufferMemory, 0, 4, 0, &mapped), "map");
    const auto p = static_cast<unsigned char *>(mapped);
    bool matches = p[0] > 174 && p[0] < 234 && p[1] > 21 && p[1] < 81 && p[2] < 43;
    std::printf(
        "SR 640x360 -> 1280x720 evaluate=0 centerRGB=%u,%u,%u expected=204,51,13 match=%d\n", p[0],
        p[1], p[2], matches);
    vkUnmapMemory(device, f.bufferMemory);
    check(int(sdk<PFun_slFreeResources>(lib, "slFreeResources")(sl::kFeatureDLSS, vp)), "free SR");
    return matches;
} catch (const std::exception &error) {
    std::printf("SR fixture failure: %s\n", error.what());
    return false;
}
