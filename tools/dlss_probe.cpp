#include <cstdio>
#include <filesystem>
#include <sl.h>
#include <sl_dlss_g.h>
#include <vector>
#include <vulkan/vulkan.h>
#include <windows.h>

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error Build the probe as native ARM64
#endif

bool TestDlssSR(HMODULE, VkPhysicalDevice, VkDevice, uint32_t);

// Run separately from the emulator so probing an unsupported driver cannot install hooks in it.
int wmain(int argc, wchar_t **argv) {
    const bool fgOnly = argc == 3 && std::wstring_view(argv[2]) == L"--fg-only";
    if (argc != 2 && !fgOnly) {
        std::puts("Usage: dlss-probe <Streamline directory> [--fg-only]");
        return 2;
    }
    setvbuf(stdout, nullptr, _IONBF, 0);
    const auto dir = std::filesystem::absolute(argv[1]);
    const auto dll =
        LoadLibraryExW((dir / L"sl.interposer.dll").c_str(), nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dll) {
        std::printf("streamline_load_error=%lu fg=unavailable\n", GetLastError());
        return 2;
    }
    const auto init = reinterpret_cast<PFun_slInit *>(GetProcAddress(dll, "slInit"));
    const auto shutdown = reinterpret_cast<PFun_slShutdown *>(GetProcAddress(dll, "slShutdown"));
    const auto requirements = reinterpret_cast<PFun_slGetFeatureRequirements *>(
        GetProcAddress(dll, "slGetFeatureRequirements"));
    const auto supported =
        reinterpret_cast<PFun_slIsFeatureSupported *>(GetProcAddress(dll, "slIsFeatureSupported"));
    const auto get =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(dll, "vkGetInstanceProcAddr"));
    if (!init || !shutdown || !requirements || !supported || !get) {
        std::puts("missing SDK export");
        return 2;
    }
    const wchar_t *path = dir.c_str();
    const sl::Feature features[]{sl::kFeatureDLSS_G, sl::kFeatureReflex, fgOnly ? sl::kFeaturePCL : sl::kFeatureDLSS};
    sl::Preferences prefs{};
    prefs.featuresToLoad = features;
    prefs.numFeaturesToLoad = 3;
    prefs.pathsToPlugins = &path;
    prefs.numPathsToPlugins = 1;
    prefs.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                  sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    prefs.renderAPI = sl::RenderAPI::eVulkan;
    prefs.engineVersion = "NXN1X-1";
    prefs.projectId = "ab2efce3-9b51-4891-a73a-53429294495d";
    prefs.logMessageCallback = [](sl::LogType, const char *message) { std::puts(message); };
    auto result = init(prefs, sl::kSDKVersion);
    std::printf("slInit=%d\n", int(result));
    if (result != sl::Result::eOk)
        return 2;
    sl::FeatureRequirements req{};
    auto rr = requirements(sl::kFeatureDLSS_G, req);
    const bool vk =
        (unsigned(req.flags) & unsigned(sl::FeatureRequirementFlags::eVulkanSupported)) != 0;
    std::printf("slGetFeatureRequirements=%d flags=%x vulkan=%d graphicsQueues=%u computeQueues=%u "
                "opticalFlowQueues=%u driver=%s required=%s os=%s required=%s\n",
                int(rr), unsigned(req.flags), vk, req.vkNumGraphicsQueuesRequired,
                req.vkNumComputeQueuesRequired, req.vkNumOpticalFlowQueuesRequired,
                req.driverVersionDetected.toStr().c_str(),
                req.driverVersionRequired.toStr().c_str(), req.osVersionDetected.toStr().c_str(),
                req.osVersionRequired.toStr().c_str());
    sl::FeatureRequirements srReq{};
    const auto srReqResult = fgOnly ? sl::Result::eErrorFeatureMissing : requirements(sl::kFeatureDLSS, srReq);
    const bool srVk =
        srReqResult == sl::Result::eOk &&
        (unsigned(srReq.flags) & unsigned(sl::FeatureRequirementFlags::eVulkanSupported));
    std::printf("SR requirements=%d flags=0x%x Vulkan=%d\n", int(srReqResult),
                unsigned(srReq.flags), srVk);
    const auto create = reinterpret_cast<PFN_vkCreateInstance>(get(nullptr, "vkCreateInstance"));
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_4;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    const char *instanceExtensions[]{"VK_KHR_surface", "VK_KHR_win32_surface"};
    ci.enabledExtensionCount = 2;
    ci.ppEnabledExtensionNames = instanceExtensions;
    VkInstance inst{};
    int status = 2;
    bool stopped = false;
    if (create && create(&ci, nullptr, &inst) == VK_SUCCESS) {
        const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
            get(inst, "vkEnumeratePhysicalDevices"));
        const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            get(inst, "vkGetPhysicalDeviceProperties2"));
        uint32_t count{};
        if (enumerate(inst, &count, nullptr) == VK_SUCCESS && count) {
            std::vector<VkPhysicalDevice> devices(count);
            if (enumerate(inst, &count, devices.data()) == VK_SUCCESS) {
                for (auto dev : devices) {
                    VkPhysicalDeviceDriverProperties driver{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
                    VkPhysicalDeviceProperties2 props{
                        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
                    properties(dev, &props);
                    if (driver.driverID != VK_DRIVER_ID_NVIDIA_PROPRIETARY)
                        continue;
                    sl::AdapterInfo adapter{};
                    adapter.vkPhysicalDevice = dev;
                    auto support = supported(sl::kFeatureDLSS_G, adapter);
                    std::printf("adapter=%s pci=%04x:%04x driver=%s api=%u.%u.%u "
                                "slIsFeatureSupported=%d reflex=%d\n",
                                props.properties.deviceName, props.properties.vendorID,
                                props.properties.deviceID, driver.driverInfo,
                                VK_API_VERSION_MAJOR(props.properties.apiVersion),
                                VK_API_VERSION_MINOR(props.properties.apiVersion),
                                VK_API_VERSION_PATCH(props.properties.apiVersion), int(support),
                                int(supported(sl::kFeatureReflex, adapter)));
                    status = rr == sl::Result::eOk && vk && support == sl::Result::eOk ? 0 : 3;
                    const auto srPre = fgOnly ? srReqResult : supported(sl::kFeatureDLSS, adapter);
                    std::printf("preDeviceSR=%d\n", int(srPre));
                    if (status == 0 || (srVk && srPre == sl::Result::eOk)) {
                        const auto families =
                            reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                                get(inst, "vkGetPhysicalDeviceQueueFamilyProperties"));
                        uint32_t n{};
                        families(dev, &n, nullptr);
                        std::vector<VkQueueFamilyProperties> queues(n);
                        families(dev, &n, queues.data());
                        uint32_t graphics{};
                        while (graphics < n &&
                               !(queues[graphics].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                            ++graphics;
                        status = 4;
                        if (graphics == n)
                            break;
                        float priority = 1.0f;
                        VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
                        queue.queueFamilyIndex = graphics;
                        queue.queueCount = 1;
                        queue.pQueuePriorities = &priority;
                        const char *extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                                                    VK_KHR_PRESENT_ID_EXTENSION_NAME};
                        VkPhysicalDeviceVulkan13Features available13{
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
                        VkPhysicalDevicePresentIdFeaturesKHR presentId{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR};
                        available13.pNext = &presentId;
                        VkPhysicalDeviceFeatures2 available{
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &available13};
                        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                            get(inst, "vkGetPhysicalDeviceFeatures2"))(dev, &available);
                        VkPhysicalDeviceVulkan13Features enabled13{
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
                        enabled13.privateData = available13.privateData;
                        enabled13.pNext = &presentId;
                        std::printf("presentId=%u privateData=%u\n", presentId.presentId, enabled13.privateData);
                        VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                        info.pQueueCreateInfos = &queue;
                        info.queueCreateInfoCount = 1;
                        info.pNext = &enabled13;
                        info.enabledExtensionCount = 2;
                        info.ppEnabledExtensionNames = extensions;
                        VkDevice device{};
                        auto createDevice =
                            reinterpret_cast<PFN_vkCreateDevice>(get(inst, "vkCreateDevice"));
                        auto created = createDevice(dev, &info, nullptr, &device);
                        std::printf("vkCreateDevice=%d\n", int(created));
                        if (created == VK_SUCCESS) {
                            const auto post = supported(sl::kFeatureDLSS_G, adapter);
                            auto featureFunction = reinterpret_cast<PFun_slGetFeatureFunction *>(
                                GetProcAddress(dll, "slGetFeatureFunction"));
                            void *func{};
                            auto lookup = featureFunction ? featureFunction(sl::kFeatureDLSS_G,
                                                                            "slDLSSGGetState", func)
                                                          : sl::Result::eErrorFeatureMissing;
                            std::printf("postDeviceSupport=%d lookupState=%d\n", int(post),
                                        int(lookup));
                            if (lookup == sl::Result::eOk && func && post == sl::Result::eOk) {
                                sl::DLSSGState state{};
                                auto stateResult = reinterpret_cast<PFun_slDLSSGGetState *>(func)(
                                    sl::ViewportHandle(0), state, nullptr);
                                std::printf("slDLSSGGetState=%d status=0x%x maxGenerated=%u\n",
                                            int(stateResult), unsigned(state.status),
                                            state.numFramesToGenerateMax);
                                if (stateResult == sl::Result::eOk &&
                                    state.status == sl::DLSSGStatus::eOk &&
                                    state.numFramesToGenerateMax >= 1 && vk &&
                                    rr == sl::Result::eOk)
                                    status = 0;
                            }
                            const auto sr = fgOnly ? srReqResult : supported(sl::kFeatureDLSS, adapter);
                            std::printf("postDeviceSR=%d\n", int(sr));
                            if (!fgOnly && (!srVk || sr != sl::Result::eOk ||
                                !TestDlssSR(dll, dev, device, graphics)))
                                status = 5;
                            const auto destroy =
                                reinterpret_cast<PFN_vkDestroyDevice>(get(inst, "vkDestroyDevice"));
                            const auto stop = shutdown();
                            stopped = true;
                            std::printf("slShutdown=%d\n", int(stop));
                            if (stop != sl::Result::eOk)
                                status = 4;
                            destroy(device, nullptr);
                        }
                    }
                    break;
                }
            }
        }
        if (!stopped) {
            std::printf("slShutdown=%d\n", int(shutdown()));
            stopped = true;
        }
        reinterpret_cast<PFN_vkDestroyInstance>(get(inst, "vkDestroyInstance"))(inst, nullptr);
    }
    if (!stopped)
        std::printf("slShutdown=%d\n", int(shutdown()));
    std::printf("probe_exit=%d\n", status);
    FreeLibrary(dll);
    return status;
}
