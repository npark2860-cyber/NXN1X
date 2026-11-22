// SPDX-License-Identifier: GPL-3.0-or-later
// NVN path, step 6 (NXN1X docs NVN_HOST_BRIDGE.md 4.5): the renderer's side of the host NVN layer's output. A composite whose
// framebuffer is a present of the layer (Tegra::FramebufferConfig::nvn_image, from HardwareComposer through nvdisp) gets its picture
// here instead of from guest memory: the window texture is copied into an image of the renderer's own, which the usual Layer pass
// (filters, FSR, LSFG, Go capture, PresentManager) then reads. Once the copy has finished on the GPU the present is given back
// (Core::NvnLayer::OutputDone): the game may draw into that window texture again.
#pragma once

#include <optional>
#include <span>

#include "video_core/renderer_vulkan/vk_blit_screen.h"

namespace Tegra {
struct FramebufferConfig;
}

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

namespace NvnDisplay {

/// RasterizerVulkan::AccelerateDisplay for a framebuffer with nvn_image: the frame's picture, as an image of the renderer's own. The
/// window texture is copied once per present; later composites of the same present read the copy (GPU thread).
std::optional<FramebufferTextureInfo> Frame(const Device& device, MemoryAllocator& allocator, Scheduler& scheduler,
                                            const Tegra::FramebufferConfig& config);

/// RendererVulkan::Composite, at its end (every way out): the presents of this composite go back once their copies finished on the
/// GPU; presents it never copied (no window shown, the renderer failed) go back at once.
void Finish(std::span<const Tegra::FramebufferConfig> framebuffers);

/// RendererVulkan's destructor, before the scheduler goes: stops the release thread and gives every present back.
void Shutdown();

// opthold: counts the copy images made (one per output size or format change), so frame generation can restart after one
u64 Generation();

} // namespace NvnDisplay
} // namespace Vulkan
