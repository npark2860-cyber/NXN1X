# N1X capability probe, 2026-11-19

NVIDIA RTX Spark N1X, PCI 10de:2e03, NVIDIA 621.24, Vulkan 1.4.357, Windows 11 ARM64 28000.3501. DxDiag reports WDDM 3.2 and hardware scheduling enabled. The following results came from the native probe using the proprietary Vulkan physical device.

| Query | Result |
| --- | --- |
| slInit, vkCreateDevice, slShutdown | All returned 0 |
| FG requirements / adapter support / post-device support | 0 / 0 / 0. Flags 0x1e. One graphics queue, two compute queues and one optical flow queue requested. |
| Reflex adapter support | 0 |
| slDLSSGGetState (FG off, no swapchain) | Return 0, status 0, maximum generated frames 5. No generated frame was requested or displayed. The plugin reports Vulkan FG VSync control unsupported. |
| SR requirements and post-device adapter support | Return 0, flags 0x07, Vulkan supported. |
| SR offscreen output | One stationary orthographic scene, 640×360 to 1280×720. Red input and green output sentinel. After Evaluate and queue completion, center RGB 204,51,13 matched input. Motion zero is ground truth only for this stationary test. |
| Missing runtime directory | Windows loader error 126, probe exit 2. Emulator requires no SDK DLL to render. |
| Synchronization validation of SR | Two WRITE_AFTER_WRITE errors inside nv.ngx.dlss.Evaluate (layout transition followed by clear). Earlier device-create dependency and privateData errors were fixed and did not recur. Production integration blocked. |

The signed interposer and both feature plugins are Streamline 2.14.16.0; NGX DLSS and DLSS-G DLLs are 310.9.16.0. Checked PE machine 0xaa64. Unpacked the official SDK locally in build/sl-sdk. Keep its license.txt, external/ngx-sdk/license.txt and binary-specific notices with the SDK. No NVIDIA DLL was added to the game build.

## Rebuild and run from the workspace root

Download and extract the [official ARM64 SDK](https://github.com/NVIDIA-RTX/Streamline/releases/tag/v2.14.16) into build/sl-sdk. Use the existing native clangarm64 installation.

~~~powershell
C:/msys64/clangarm64/bin/clang++.exe -std=c++20 -O2 -municode -static -fms-extensions '-Ibuild/sl-sdk/include' '-Itools/cpm/vulkanheaders/v1.4.345/include' tools/dlss_probe.cpp tools/dlss_sr_fixture.cpp C:/msys64/clangarm64/lib/libvulkan-1.dll.a -o build/dlss-probe.exe
./build/dlss-probe.exe build/sl-sdk/bin/arm64
~~~

Evidence: [probe](../build/dlss-validation-final/probe.log), [sync validation](../build/dlss-validation-final/validation.log), [missing DLL](../build/dlss-missing.log), [DLL versions](../build/dlss-module-versions.log).
