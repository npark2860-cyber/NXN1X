// SPDX-License-Identifier: GPL-3.0-or-later
// The Eden graphics pipeline key of an NVN draw (NXN1X docs/lane_b/SHADER_EDEN_B.md B4).
// Two steps, so the key is made by Eden's own rules:
//   1. FillNvnRegs: the NVN state -> the Maxwell 3D registers the NVN driver would have written.
//   2. FillNvnGraphicsKey: those registers -> FixedPipelineState, line for line as
//      FixedPipelineState::Refresh (it needs a Maxwell3D engine, so it is mirrored here; the parts
//      that take only registers, BlendingAttachment/DynamicState::Refresh*, are called as they are).
#pragma once

#include "video_core/engines/maxwell_3d.h"
#include "video_core/nvn/nvn_shader_bridge.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"

namespace Vulkan {
class Device;
}

namespace VideoCommon::Nvn {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

// VkFormat -> Maxwell render target / depth format, from Eden's PixelFormat tables and this
// device's SurfaceFormat choice (the inverse of what the Maxwell path does with the register).
void InitFormatTables(const Vulkan::Device& device);
[[nodiscard]] VideoCore::Surface::PixelFormat PixelFormatOfVk(int vk_format);

// regs is reset first (zeroed) and then filled.
void FillNvnRegs(Maxwell& regs, const nvn_eden_draw_state& draw);

// Mirror of FixedPipelineState::Refresh (video_core/renderer_vulkan/fixed_pipeline_state.cpp):
// keep in step with it. All dirty flags are taken as set (the NVN path re-fills only when its
// state objects change, SHADER_EDEN_B.md §4.4-6).
void FillNvnGraphicsKey(Vulkan::FixedPipelineState& state, const Maxwell& regs,
                        Maxwell::PrimitiveTopology topology,
                        const Vulkan::DynamicFeatures& features);

// NVN value -> Maxwell register value, as FillNvnRegs converts them (for the replay's dynamic
// state, nvn_eden_cmd_state, which reads the draw state without a register file)
[[nodiscard]] Maxwell::ComparisonOp NvnCompare(u32 nvn);
[[nodiscard]] Maxwell::StencilOp::Op NvnStencilOp(u32 nvn);
[[nodiscard]] Maxwell::Blend::Equation NvnEquation(u32 nvn);
[[nodiscard]] Maxwell::Blend::Factor NvnFactor(u32 nvn);
// a vertex attribute's VkFormat -> Maxwell (size, type); false when Maxwell has no such format
[[nodiscard]] bool NvnAttribFormat(int vk_format, Maxwell::VertexAttribute::Size& size,
                                   Maxwell::VertexAttribute::Type& type);
// a colour target's VkFormat -> Maxwell render target format (NONE when unknown)
[[nodiscard]] Tegra::RenderTargetFormat NvnRenderTargetFormat(int vk_format);
[[nodiscard]] bool NvnHasDepthFormat(int vk_format);

} // namespace VideoCommon::Nvn
