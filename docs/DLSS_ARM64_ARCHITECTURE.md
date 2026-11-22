# Windows ARM64 DLSS integration boundaries

The current Release can configure Frame Generation in Graphics → Advanced. NVN explicit input tagging is independent of title profiles. Automatic recognition runs on both draw backends and matches camera matrices and motion observations before tagging ordinary raster-pass attachments. The swapchain interposer receives the final composited color. Scope, limitations and build evidence are in the [current validation report](DLSS_ARM64_TEST_REPORT.md#current-arm64-release-with-automatic-fg-inputs). The following inventory describes the initial audit.

## Switch 2 call paths

| Source in this checkout | Ownership and observations |
| --- | --- |
| host/dm/gfx/nvn/hle_nvn.c, nvnDlssNetworkInitialize/Finalize, nvnCommandBufferEvaluateDlss (see OP_DLSS producer) | The command records six texture roles, input and output sizes, base offsets, motion scales, jitter, pre-exposure and history reset. dlss_set_tex currently discards texture views. |
| host/dm/windows/gfx_vk_exec.c, OP_DLSS | The Windows replay currently leaves the output untouched. Opt-in inspection uses vk_tex_peek on the recorded roles without creating textures. A real SR adapter must preserve the guest output subresource, transition it and make it visible to later guest passes. |
| host/dm/windows/gfx_vk_exec.c, ndsr_blit and dispatch | The 24 Bananza agl_ndsr program keys identify game compute. Original guest compute is the default. DM_VK_NDSR=0 explicitly selects GSR1. A recognized direct dispatch can be logged after vkCmdDispatch with NXN1X_DLSS_AUDIT_FRAME set. No second SR pass is applied. |
| source/src/video_core/native_lowering.cpp and shader_recompiler/backend/spirv/emit_spirv_tensor.cpp | Guest HMMA goes through the Ampere translator. The existence of a cooperative-matrix emitter does not establish hardware Tensor Core use. The current shader profile has no assignment enabling the cooperative HMMA flags. |
| host/dm/windows/gfx_vk_present.c, mtl_present and present_out; source/src/core/nvn/nvn_output.cpp, PushNx2 | The bridge publishes a color image, format, crop, extent, scale flag, fence and return ticket. The image may point to the scaled or copied surface. A separate host swapchain exists for standalone use. The integrated path routes the frame to NXN1X's compositor. |
| source/src/video_core/renderer_vulkan/renderer_vulkan.cpp, RendererVulkan and CompositeImpl | Opens Vulkan, creates an instance, selects a device, obtains the output layout and composites. Existing LSFG, SnapGo, screenshot and compact display branches can consume this image. |
| source/src/video_core/vulkan_common/vulkan_device.cpp, Device::Device; renderer_vulkan/vk_swapchain.cpp; renderer_vulkan/vk_present_manager.cpp | Logical device and queues, swapchain creation and acquire, threaded submission/presentation, fences and resizing. Final FG would require one interposer dispatch chain from instance creation through swapchain teardown. Generated frames must not enter the guest ticket or frame counter. |

## SDK selection

Use one Streamline 2.14.16 Vulkan interposer in the standalone probe. Its SR plugin evaluates NVIDIA NGX. An independent second NGX context or a second Present hook has no role in this prototype. The game executable has no SDK DLL dependency. SR and FG are off in the game. The only new environment switch is the diagnostic frame selector. The capability probe explicitly receives an SDK directory and has OTA disabled.

The TotK recompilation experiments illustrate resource snapshots and alternative presentation providers. NXN1X only receives guest command and image metadata, and does not currently retain verified game camera constants or the dense motion image at presentation. A snapshot needs a producer fence and an FG completion fence before slot reuse. A raw Vulkan handle in a trace has no ownership. The authoritative SDK guide requires Streamline shutdown before destroying Vulkan objects. The probe implements this ordering.

References: [Vulkan FG guide](https://github.com/NVIDIA-RTX/Streamline/blob/v2.14.16/docs/ProgrammingGuideDLSS_G.md), [Vulkan sample](https://github.com/nvpro-samples/vk_streamline), [TotK FG experiment](https://github.com/valentderah/totk-pc-recomp/pull/1069), [provider experiment](https://github.com/valentderah/totk-pc-recomp/pull/1223).
