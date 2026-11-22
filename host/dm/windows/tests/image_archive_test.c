// SPDX-License-Identifier: GPL-2.0-or-later
// Headless test of the production image archive transfer, on a real Vulkan device.
#include "../nvn_image_archive.h"
#include <stdlib.h>
#include <string.h>

static VkInstance instance;
static VkPhysicalDevice physical;
static VkDevice device;
static VkQueue queue;
static uint32_t family;
static VkPhysicalDeviceMemoryProperties memory;
static int validation_errors, passed, skipped;

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
#define VK_REQUIRE(x) REQUIRE((x) == VK_SUCCESS)
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)type; (void)user;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { validation_errors++; fprintf(stderr, "%s\n", data->pMessage); }
    return VK_FALSE;
}
static VkDebugUtilsMessengerEXT messenger;
static void init(void) {
    uint32_t count = 0; VK_REQUIRE(vkEnumerateInstanceLayerProperties(&count, NULL));
    VkLayerProperties *layers = calloc(count ? count : 1, sizeof *layers);
    VK_REQUIRE(vkEnumerateInstanceLayerProperties(&count, layers));
    int validation = 0;
    for (uint32_t i = 0; i < count; i++) if (!strcmp(layers[i].layerName, "VK_LAYER_KHRONOS_validation")) validation = 1;
    free(layers);
    const char *layer = "VK_LAYER_KHRONOS_validation", *extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "NVN image archive test", 1, NULL, 0, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, validation, &layer, validation, &extension};
    VK_REQUIRE(vkCreateInstance(&ci, NULL, &instance));
    if (validation) {
        VkDebugUtilsMessengerCreateInfoEXT di = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0,
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            debug_message, NULL};
        PFN_vkCreateDebugUtilsMessengerEXT create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        REQUIRE(create); VK_REQUIRE(create(instance, &di, NULL, &messenger));
    }
    VK_REQUIRE(vkEnumeratePhysicalDevices(instance, &count, NULL)); REQUIRE(count);
    VkPhysicalDevice *devices = calloc(count, sizeof *devices);
    VK_REQUIRE(vkEnumeratePhysicalDevices(instance, &count, devices)); physical = devices[0]; free(devices);
    VkPhysicalDeviceProperties properties; vkGetPhysicalDeviceProperties(physical, &properties);
    printf("Device: %s; driver %u; validation: %s\n", properties.deviceName, properties.driverVersion, validation ? "enabled" : "unavailable");
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *queues = calloc(count, sizeof *queues);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
    family = UINT32_MAX;
    for (uint32_t i = 0; i < count; i++) if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { family = i; break; }
    REQUIRE(family != UINT32_MAX); free(queues);
    float priority = 1;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, family, 1, &priority};
    VkPhysicalDeviceFeatures features; vkGetPhysicalDeviceFeatures(physical, &features);
    VkDeviceCreateInfo dc = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qi, 0, NULL, 0, NULL, &features};
    VK_REQUIRE(vkCreateDevice(physical, &dc, NULL, &device)); vkGetDeviceQueue(device, family, 0, &queue);
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
}
typedef struct { VkImage image; VkDeviceMemory allocation; } image_t;
static image_t make_image(const nvna_shape *s, VkImageType type) {
    image_t im = {0};
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, type, s->format,
        {s->width, s->height, s->depth}, s->levels, s->layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VK_REQUIRE(vkCreateImage(device, &ci, NULL, &im.image));
    VkMemoryRequirements req; vkGetImageMemoryRequirements(device, im.image, &req);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) if (req.memoryTypeBits & (1u << i)) { mt = i; break; }
    REQUIRE(mt != UINT32_MAX);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, mt};
    VK_REQUIRE(vkAllocateMemory(device, &ai, NULL, &im.allocation));
    VK_REQUIRE(vkBindImageMemory(device, im.image, im.allocation, 0)); return im;
}
static void drop_image(image_t im) { vkDestroyImage(device, im.image, NULL); vkFreeMemory(device, im.allocation, NULL); }
static void fill(FILE *file, const nvna_shape *s) {
    // Finite [0,1] floats in depth planes; D24's unused high byte is zero.
    for (unsigned aspect = 1; aspect <= VK_IMAGE_ASPECT_STENCIL_BIT; aspect <<= 1) {
        if (!(s->aspects & aspect)) continue;
        nvna_shape plane = *s; plane.aspects = aspect; uint64_t bytes = nvna_size(&plane);
        uint8_t data[4096];
        for (uint64_t at = 0; at < bytes;) {
            size_t n = bytes - at < sizeof data ? (size_t)(bytes - at) : sizeof data;
            for (size_t i = 0; i < n; i++) data[i] = (uint8_t)((at + i) * 73 + ((at + i) >> 9) * 11 + aspect * 29);
            if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT && nvna_aspect_bytes(s, aspect) == 4) {
                for (size_t i = 0; i < n; i += 4) {
                    if (s->format == VK_FORMAT_D24_UNORM_S8_UINT || s->format == VK_FORMAT_X8_D24_UNORM_PACK32) data[i + 3] = 0;
                    else { float value = (float)(((at + i) / 4) % 1024) / 1024.0f; memcpy(data + i, &value, 4); }
                }
            }
            REQUIRE(fwrite(data, 1, n, file) == n); at += n;
        }
    }
    REQUIRE(!fflush(file));
}
static void compare(FILE *a, FILE *b, const nvna_shape *s, int mask_d24) {
    REQUIRE(!fseek(a, 0, SEEK_SET)); REQUIRE(!fseek(b, 0, SEEK_SET));
    nvna_shape depth = *s; depth.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    uint64_t depth_bytes = nvna_size(&depth), total = nvna_size(s);
    uint8_t x[4096], y[4096];
    for (uint64_t at = 0; at < total;) {
        size_t n = total - at < sizeof x ? (size_t)(total - at) : sizeof x;
        REQUIRE(fread(x, 1, n, a) == n); REQUIRE(fread(y, 1, n, b) == n);
        for (size_t i = 0; i < n; i++) {
            if (mask_d24 && at + i < depth_bytes && (at + i) % 4 == 3) continue;
            if (x[i] != y[i]) { fprintf(stderr, "Mismatch at byte %llu: %02x != %02x\n", (unsigned long long)(at + i), x[i], y[i]); exit(1); }
        }
        at += n;
    }
    REQUIRE(fgetc(a) == EOF); REQUIRE(fgetc(b) == EOF);
}
static void run_case(const char *name, nvna_shape s, VkImageType type) {
    VkImageFormatProperties fp;
    VkResult supported = vkGetPhysicalDeviceImageFormatProperties(physical, s.format, type, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 0, &fp);
    if (supported == VK_ERROR_FORMAT_NOT_SUPPORTED) { printf("SKIP %s: format unsupported\n", name); skipped++; return; }
    VK_REQUIRE(supported);
    FILE *input = tmpfile(), *archive = tmpfile(), *restored = tmpfile(); REQUIRE(input && archive && restored);
    fill(input, &s);
    image_t im = make_image(&s, type);
    VK_REQUIRE(nvna_stream(device, queue, family, &memory, im.image, &s, input, 1, NULL, NULL));
    VK_REQUIRE(nvna_stream(device, queue, family, &memory, im.image, &s, archive, 0, NULL, NULL));
    int d24 = s.format == VK_FORMAT_D24_UNORM_S8_UINT || s.format == VK_FORMAT_X8_D24_UNORM_PACK32;
    compare(input, archive, &s, d24);
    drop_image(im); im = make_image(&s, type);   // restoration must not depend on the old allocation
    VK_REQUIRE(nvna_stream(device, queue, family, &memory, im.image, &s, archive, 1, NULL, NULL));
    VK_REQUIRE(nvna_stream(device, queue, family, &memory, im.image, &s, restored, 0, NULL, NULL));
    compare(archive, restored, &s, d24);
    drop_image(im); fclose(input); fclose(archive); fclose(restored);
    REQUIRE(!validation_errors); passed++;
    printf("PASS %s: %llu bytes, all planes/mips/layers after destroy and recreate\n", name, (unsigned long long)nvna_size(&s)); fflush(stdout);
}
int main(void) {
    nvna_shape s = {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 8, 4, 1, 2, 3, 1, 1, 4};
    REQUIRE(nvna_size(&s) == 336); s.width = 0; REQUIRE(!nvna_size(&s));
    s.width = s.height = s.depth = s.layers = UINT32_MAX; REQUIRE(!nvna_size(&s));
    init();
#define COLOR VK_IMAGE_ASPECT_COLOR_BIT
#define DEPTH VK_IMAGE_ASPECT_DEPTH_BIT
#define STENCIL VK_IMAGE_ASPECT_STENCIL_BIT
    run_case("RGBA8 chunk boundaries", (nvna_shape){VK_FORMAT_R8G8B8A8_UNORM,COLOR,2049,1025,1,2,5,1,1,4}, VK_IMAGE_TYPE_2D);
    run_case("R16 terrain", (nvna_shape){VK_FORMAT_R16_UNORM,COLOR,260,260,1,3,4,1,1,2}, VK_IMAGE_TYPE_2D);
    run_case("RG16 integer", (nvna_shape){VK_FORMAT_R16G16_UINT,COLOR,259,131,1,2,3,1,1,4}, VK_IMAGE_TYPE_2D);
    run_case("BC7 edge blocks", (nvna_shape){VK_FORMAT_BC7_UNORM_BLOCK,COLOR,257,129,1,2,5,4,4,16}, VK_IMAGE_TYPE_2D);
    run_case("ASTC 8x8 edge blocks", (nvna_shape){VK_FORMAT_ASTC_8x8_UNORM_BLOCK,COLOR,260,260,1,2,5,8,8,16}, VK_IMAGE_TYPE_2D);
    run_case("D16 shadow array", (nvna_shape){VK_FORMAT_D16_UNORM,DEPTH,3072,3072,1,3,1,1,1,2}, VK_IMAGE_TYPE_2D);
    run_case("D32 depth", (nvna_shape){VK_FORMAT_D32_SFLOAT,DEPTH,257,129,1,2,4,1,1,4}, VK_IMAGE_TYPE_2D);
    run_case("D32S8 planes", (nvna_shape){VK_FORMAT_D32_SFLOAT_S8_UINT,DEPTH|STENCIL,259,131,1,2,3,1,1,8}, VK_IMAGE_TYPE_2D);
    run_case("D24S8 planes", (nvna_shape){VK_FORMAT_D24_UNORM_S8_UINT,DEPTH|STENCIL,259,131,1,2,3,1,1,4}, VK_IMAGE_TYPE_2D);
    run_case("R8 volume", (nvna_shape){VK_FORMAT_R8_UINT,COLOR,129,65,9,1,4,1,1,1}, VK_IMAGE_TYPE_3D);
    run_case("R8 1D array", (nvna_shape){VK_FORMAT_R8_UNORM,COLOR,33,1,1,3,5,1,1,1}, VK_IMAGE_TYPE_1D);
    // Failed upload must report failure even after earlier chunks were submitted.
    s = (nvna_shape){VK_FORMAT_R8G8B8A8_UNORM,COLOR,2048,1024,1,1,1,1,1,4};
    image_t im = make_image(&s, VK_IMAGE_TYPE_2D); FILE *short_file = tmpfile(); REQUIRE(short_file);
    unsigned char zeros[4096] = {0}; for (int i = 0; i < 1024; i++) REQUIRE(fwrite(zeros, 1, sizeof zeros, short_file) == sizeof zeros);
    REQUIRE(!fflush(short_file));
    REQUIRE(nvna_stream(device, queue, family, &memory, im.image, &s, short_file, 1, NULL, NULL) != VK_SUCCESS);
    fclose(short_file); drop_image(im); passed++; puts("PASS truncated backing rejected after first chunk");
    REQUIRE(!validation_errors); VK_REQUIRE(vkDeviceWaitIdle(device)); vkDestroyDevice(device, NULL);
    if (messenger) ((PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"))(instance, messenger, NULL);
    vkDestroyInstance(instance, NULL);
    printf("RESULT: %d passed, %d skipped, %d validation errors\n", passed, skipped, validation_errors); return 0;
}
