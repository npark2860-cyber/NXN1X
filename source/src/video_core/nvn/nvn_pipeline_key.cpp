// SPDX-License-Identifier: GPL-3.0-or-later
// See nvn_pipeline_key.h.

#include <algorithm>
#include <bit>
#include <cstring>
#include <mutex>

#include <ankerl/unordered_dense.h>

#include "common/logging.h"
#include "video_core/nvn/nvn_pipeline_key.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/surface.h"
#include "video_core/transform_feedback.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace VideoCommon::Nvn {

using Vulkan::DynamicFeatures;
using Vulkan::FixedPipelineState;
using VideoCore::Surface::PixelFormat;

namespace {

// The render target formats Eden's PixelFormatFromRenderTargetFormat knows (surface.cpp), in its
// order: for two formats with one VkFormat (A8B8G8R8 / X8B8G8R8) the first is taken
// [구현 때 확인: which one the NVN driver writes for an NVN_FORMAT_RGBX8 target].
constexpr Tegra::RenderTargetFormat RT_FORMATS[] = {
#define F(x) Tegra::RenderTargetFormat::x
    F(R32G32B32A32_FLOAT), F(R32G32B32X32_FLOAT), F(R32G32B32A32_SINT), F(R32G32B32X32_SINT),
    F(R32G32B32A32_UINT), F(R32G32B32X32_UINT), F(R16G16B16A16_UNORM), F(R16G16B16A16_SNORM),
    F(R16G16B16A16_SINT), F(R16G16B16A16_UINT), F(R16G16B16A16_FLOAT), F(R32G32_FLOAT),
    F(R32G32_SINT), F(R32G32_UINT), F(R16G16B16X16_FLOAT), F(A8R8G8B8_UNORM), F(X8R8G8B8_UNORM),
    F(A8R8G8B8_SRGB), F(X8R8G8B8_SRGB), F(A2B10G10R10_UNORM), F(A2B10G10R10_UINT),
    F(A2R10G10B10_UNORM), F(A8B8G8R8_UNORM), F(X8B8G8R8_UNORM), F(A8B8G8R8_SRGB),
    F(X8B8G8R8_SRGB), F(A8B8G8R8_SNORM), F(A8B8G8R8_SINT), F(A8B8G8R8_UINT), F(R16G16_UNORM),
    F(R16G16_SNORM), F(R16G16_SINT), F(R16G16_UINT), F(R16G16_FLOAT), F(B10G11R11_FLOAT),
    F(R32_SINT), F(R32_UINT), F(R32_FLOAT), F(R5G6B5_UNORM), F(A1R5G5B5_UNORM), F(X1R5G5B5_UNORM),
    F(R8G8_UNORM), F(R8G8_SNORM), F(R8G8_SINT), F(R8G8_UINT), F(R16_UNORM), F(R16_SNORM),
    F(R16_SINT), F(R16_UINT), F(R16_FLOAT), F(R8_UNORM), F(R8_SNORM), F(R8_SINT), F(R8_UINT),
#undef F
};
constexpr Tegra::DepthFormat DEPTH_FORMATS[] = {
    Tegra::DepthFormat::Z24_UNORM_S8_UINT, Tegra::DepthFormat::S8Z24_UNORM,
    Tegra::DepthFormat::Z32_FLOAT,         Tegra::DepthFormat::Z16_UNORM,
    Tegra::DepthFormat::S8_UINT,           Tegra::DepthFormat::Z32_FLOAT_X24S8_UINT,
    Tegra::DepthFormat::X8Z24_UNORM,
};

std::once_flag g_tables_once;
ankerl::unordered_dense::map<int, Tegra::RenderTargetFormat> g_rt_of_vk;
ankerl::unordered_dense::map<int, Tegra::DepthFormat> g_depth_of_vk;
ankerl::unordered_dense::map<int, PixelFormat> g_pixel_of_vk;

// VkFormat of a vertex attribute -> Maxwell's (size, type), the inverse of MaxwellToVK::VertexFormat
struct AttribFormat {
    VkFormat vk;
    Maxwell::VertexAttribute::Size size;
    Maxwell::VertexAttribute::Type type;
};
constexpr AttribFormat ATTRIB_FORMATS[] = {
#define S(x) Maxwell::VertexAttribute::Size::Size_##x
#define T(x) Maxwell::VertexAttribute::Type::x
    {VK_FORMAT_R32G32B32A32_SFLOAT, S(R32_G32_B32_A32), T(Float)},
    {VK_FORMAT_R32G32B32A32_UINT, S(R32_G32_B32_A32), T(UInt)},
    {VK_FORMAT_R32G32B32A32_SINT, S(R32_G32_B32_A32), T(SInt)},
    {VK_FORMAT_R32G32B32_SFLOAT, S(R32_G32_B32), T(Float)},
    {VK_FORMAT_R32G32B32_UINT, S(R32_G32_B32), T(UInt)},
    {VK_FORMAT_R32G32B32_SINT, S(R32_G32_B32), T(SInt)},
    {VK_FORMAT_R32G32_SFLOAT, S(R32_G32), T(Float)},
    {VK_FORMAT_R32G32_UINT, S(R32_G32), T(UInt)},
    {VK_FORMAT_R32G32_SINT, S(R32_G32), T(SInt)},
    {VK_FORMAT_R32_SFLOAT, S(R32), T(Float)},
    {VK_FORMAT_R32_UINT, S(R32), T(UInt)},
    {VK_FORMAT_R32_SINT, S(R32), T(SInt)},
    {VK_FORMAT_R16G16B16A16_SFLOAT, S(R16_G16_B16_A16), T(Float)},
    {VK_FORMAT_R16G16B16A16_UNORM, S(R16_G16_B16_A16), T(UNorm)},
    {VK_FORMAT_R16G16B16A16_SNORM, S(R16_G16_B16_A16), T(SNorm)},
    {VK_FORMAT_R16G16B16A16_USCALED, S(R16_G16_B16_A16), T(UScaled)},
    {VK_FORMAT_R16G16B16A16_SSCALED, S(R16_G16_B16_A16), T(SScaled)},
    {VK_FORMAT_R16G16B16A16_UINT, S(R16_G16_B16_A16), T(UInt)},
    {VK_FORMAT_R16G16B16A16_SINT, S(R16_G16_B16_A16), T(SInt)},
    {VK_FORMAT_R16G16B16_SFLOAT, S(R16_G16_B16), T(Float)},
    {VK_FORMAT_R16G16B16_UNORM, S(R16_G16_B16), T(UNorm)},
    {VK_FORMAT_R16G16B16_SNORM, S(R16_G16_B16), T(SNorm)},
    {VK_FORMAT_R16G16B16_USCALED, S(R16_G16_B16), T(UScaled)},
    {VK_FORMAT_R16G16B16_SSCALED, S(R16_G16_B16), T(SScaled)},
    {VK_FORMAT_R16G16B16_UINT, S(R16_G16_B16), T(UInt)},
    {VK_FORMAT_R16G16B16_SINT, S(R16_G16_B16), T(SInt)},
    {VK_FORMAT_R16G16_SFLOAT, S(R16_G16), T(Float)},
    {VK_FORMAT_R16G16_UNORM, S(R16_G16), T(UNorm)},
    {VK_FORMAT_R16G16_SNORM, S(R16_G16), T(SNorm)},
    {VK_FORMAT_R16G16_USCALED, S(R16_G16), T(UScaled)},
    {VK_FORMAT_R16G16_SSCALED, S(R16_G16), T(SScaled)},
    {VK_FORMAT_R16G16_UINT, S(R16_G16), T(UInt)},
    {VK_FORMAT_R16G16_SINT, S(R16_G16), T(SInt)},
    {VK_FORMAT_R16_SFLOAT, S(R16), T(Float)},
    {VK_FORMAT_R16_UNORM, S(R16), T(UNorm)},
    {VK_FORMAT_R16_SNORM, S(R16), T(SNorm)},
    {VK_FORMAT_R16_USCALED, S(R16), T(UScaled)},
    {VK_FORMAT_R16_SSCALED, S(R16), T(SScaled)},
    {VK_FORMAT_R16_UINT, S(R16), T(UInt)},
    {VK_FORMAT_R16_SINT, S(R16), T(SInt)},
    {VK_FORMAT_R8G8B8A8_UNORM, S(R8_G8_B8_A8), T(UNorm)},
    {VK_FORMAT_R8G8B8A8_SNORM, S(R8_G8_B8_A8), T(SNorm)},
    {VK_FORMAT_R8G8B8A8_USCALED, S(R8_G8_B8_A8), T(UScaled)},
    {VK_FORMAT_R8G8B8A8_SSCALED, S(R8_G8_B8_A8), T(SScaled)},
    {VK_FORMAT_R8G8B8A8_UINT, S(R8_G8_B8_A8), T(UInt)},
    {VK_FORMAT_R8G8B8A8_SINT, S(R8_G8_B8_A8), T(SInt)},
    {VK_FORMAT_R8G8B8_UNORM, S(R8_G8_B8), T(UNorm)},
    {VK_FORMAT_R8G8B8_SNORM, S(R8_G8_B8), T(SNorm)},
    {VK_FORMAT_R8G8B8_UINT, S(R8_G8_B8), T(UInt)},
    {VK_FORMAT_R8G8B8_SINT, S(R8_G8_B8), T(SInt)},
    {VK_FORMAT_R8G8_UNORM, S(R8_G8), T(UNorm)},
    {VK_FORMAT_R8G8_SNORM, S(R8_G8), T(SNorm)},
    {VK_FORMAT_R8G8_USCALED, S(R8_G8), T(UScaled)},
    {VK_FORMAT_R8G8_SSCALED, S(R8_G8), T(SScaled)},
    {VK_FORMAT_R8G8_UINT, S(R8_G8), T(UInt)},
    {VK_FORMAT_R8G8_SINT, S(R8_G8), T(SInt)},
    {VK_FORMAT_R8_UNORM, S(R8), T(UNorm)},
    {VK_FORMAT_R8_SNORM, S(R8), T(SNorm)},
    {VK_FORMAT_R8_USCALED, S(R8), T(UScaled)},
    {VK_FORMAT_R8_SSCALED, S(R8), T(SScaled)},
    {VK_FORMAT_R8_UINT, S(R8), T(UInt)},
    {VK_FORMAT_R8_SINT, S(R8), T(SInt)},
    {VK_FORMAT_A2B10G10R10_UNORM_PACK32, S(A2_B10_G10_R10), T(UNorm)},
    {VK_FORMAT_A2B10G10R10_SNORM_PACK32, S(A2_B10_G10_R10), T(SNorm)},
    {VK_FORMAT_A2B10G10R10_USCALED_PACK32, S(A2_B10_G10_R10), T(UScaled)},
    {VK_FORMAT_A2B10G10R10_SSCALED_PACK32, S(A2_B10_G10_R10), T(SScaled)},
    {VK_FORMAT_A2B10G10R10_UINT_PACK32, S(A2_B10_G10_R10), T(UInt)},
    {VK_FORMAT_A2B10G10R10_SINT_PACK32, S(A2_B10_G10_R10), T(SInt)},
    {VK_FORMAT_B10G11R11_UFLOAT_PACK32, S(B10_G11_R11), T(Float)},
#undef S
#undef T
};

// NVN value -> Maxwell register value. NVN's compare, stencil, blend and equation enums are the
// Maxwell D3D-style values (nvn.h; Maxwell3D::Regs ComparisonOp, StencilOp::Op, Blend::*), which
// FixedPipelineState::Pack* fold together with the GL-style ones.
Maxwell::ComparisonOp Compare(u32 nvn) {
    return static_cast<Maxwell::ComparisonOp>(nvn >= 1 && nvn <= 8 ? nvn : 8);
}
Maxwell::StencilOp::Op StencilOp(u32 nvn) {
    return static_cast<Maxwell::StencilOp::Op>(nvn >= 1 && nvn <= 8 ? nvn : 1);
}
Maxwell::Blend::Equation Equation(u32 nvn) {
    return static_cast<Maxwell::Blend::Equation>(nvn >= 1 && nvn <= 5 ? nvn : 1);
}
Maxwell::Blend::Factor Factor(u32 nvn) {
    switch (nvn) {   // NVN's constant factors have their own values
    case 0x61: return Maxwell::Blend::Factor::ConstantColor_GL;
    case 0x62: return Maxwell::Blend::Factor::OneMinusConstantColor_GL;
    case 0x63: return Maxwell::Blend::Factor::ConstantAlpha_GL;
    case 0x64: return Maxwell::Blend::Factor::OneMinusConstantAlpha_GL;
    default:
        return static_cast<Maxwell::Blend::Factor>(nvn >= 1 && nvn <= 0x13 ? nvn : 2);
    }
}
Tegra::Texture::MsaaMode Msaa(u32 samples) {
    switch (samples) {
    case 2: return Tegra::Texture::MsaaMode::Msaa2x1;
    case 4: return Tegra::Texture::MsaaMode::Msaa2x2;
    case 8: return Tegra::Texture::MsaaMode::Msaa4x2;
    case 16: return Tegra::Texture::MsaaMode::Msaa4x4;
    default: return Tegra::Texture::MsaaMode::Msaa1x1;
    }
}

// fixed_pipeline_state.cpp's tables (file-local there), copied: keep in step
constexpr size_t POINT = 0;
constexpr size_t LINE = 1;
constexpr size_t POLYGON = 2;
constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
    POINT,   // Points
    LINE,    // Lines
    LINE,    // LineLoop
    LINE,    // LineStrip
    POLYGON, // Triangles
    POLYGON, // TriangleStrip
    POLYGON, // TriangleFan
    POLYGON, // Quads
    POLYGON, // QuadStrip
    POLYGON, // Polygon
    LINE,    // LinesAdjacency
    LINE,    // LineStripAdjacency
    POLYGON, // TrianglesAdjacency
    POLYGON, // TriangleStripAdjacency
    POLYGON, // Patches
};

constexpr std::array TOPOLOGY_CLASS_REPRESENTATIVE_LUT = {
    Maxwell::PrimitiveTopology::Points,              // Points
    Maxwell::PrimitiveTopology::Lines,               // Lines
    Maxwell::PrimitiveTopology::LineLoop,            // LineLoop
    Maxwell::PrimitiveTopology::LineStrip,           // LineStrip
    Maxwell::PrimitiveTopology::Triangles,           // Triangles
    Maxwell::PrimitiveTopology::Triangles,           // TriangleStrip
    Maxwell::PrimitiveTopology::Triangles,           // TriangleFan
    Maxwell::PrimitiveTopology::Triangles,           // Quads
    Maxwell::PrimitiveTopology::Triangles,           // QuadStrip
    Maxwell::PrimitiveTopology::Triangles,           // Polygon
    Maxwell::PrimitiveTopology::LinesAdjacency,      // LinesAdjacency
    Maxwell::PrimitiveTopology::LinesAdjacency,      // LineStripAdjacency
    Maxwell::PrimitiveTopology::TrianglesAdjacency,  // TrianglesAdjacency
    Maxwell::PrimitiveTopology::TrianglesAdjacency,  // TriangleStripAdjacency
    Maxwell::PrimitiveTopology::Patches,             // Patches
};

bool IsDualSource(Maxwell::Blend::Factor f) {
    switch (f) {
    case Maxwell::Blend::Factor::Source1Color_D3D:
    case Maxwell::Blend::Factor::OneMinusSource1Color_D3D:
    case Maxwell::Blend::Factor::Source1Alpha_D3D:
    case Maxwell::Blend::Factor::OneMinusSource1Alpha_D3D:
    case Maxwell::Blend::Factor::Source1Color_GL:
    case Maxwell::Blend::Factor::OneMinusSource1Color_GL:
    case Maxwell::Blend::Factor::Source1Alpha_GL:
    case Maxwell::Blend::Factor::OneMinusSource1Alpha_GL:
        return true;
    default:
        return false;
    }
}
bool Attachment0DualSource(const Maxwell& regs) {
    if (!regs.blend.enable[0]) {
        return false;
    }
    const auto uses = [](const auto& b) {
        return IsDualSource(b.color_source) || IsDualSource(b.color_dest) ||
               IsDualSource(b.alpha_source) || IsDualSource(b.alpha_dest);
    };
    return regs.blend_per_target_enabled ? uses(regs.blend_per_target[0]) : uses(regs.blend);
}

// fixed_pipeline_state.cpp's RefreshXfbState (file-local there), copied unchanged
void RefreshXfbState(VideoCommon::TransformFeedbackState& state, const Maxwell& regs) {
    std::ranges::transform(regs.transform_feedback.controls, state.layouts.begin(),
                           [](const auto& layout) {
                               return VideoCommon::TransformFeedbackState::Layout{
                                   .stream = layout.stream,
                                   .varying_count = layout.varying_count,
                                   .stride = layout.stride,
                               };
                           });
    state.varyings = regs.stream_out_layout;
}

} // namespace

void InitFormatTables(const Vulkan::Device& device) {
    std::call_once(g_tables_once, [&device] {
        using Vulkan::FormatType;
        using Vulkan::MaxwellToVK::SurfaceFormat;
        for (const auto rt : RT_FORMATS) {
            const PixelFormat pf = VideoCore::Surface::PixelFormatFromRenderTargetFormat(rt);
            g_rt_of_vk.try_emplace(SurfaceFormat(device, FormatType::Optimal, true, pf).format, rt);
        }
        // Eden gives the 8-bit RGBA family its PACK32 VkFormats; the host's NVN format table gives the
        // byte-order formats with the same bits (little endian). Both name the same Maxwell format.
        // [render pass compatibility wants the same VkFormat: open check in SHADER_EDEN_B.md §6]
        constexpr std::array<std::pair<int, int>, 5> ALIASES{{
            {VK_FORMAT_A8B8G8R8_UNORM_PACK32, VK_FORMAT_R8G8B8A8_UNORM},
            {VK_FORMAT_A8B8G8R8_SNORM_PACK32, VK_FORMAT_R8G8B8A8_SNORM},
            {VK_FORMAT_A8B8G8R8_UINT_PACK32, VK_FORMAT_R8G8B8A8_UINT},
            {VK_FORMAT_A8B8G8R8_SINT_PACK32, VK_FORMAT_R8G8B8A8_SINT},
            {VK_FORMAT_A8B8G8R8_SRGB_PACK32, VK_FORMAT_R8G8B8A8_SRGB},
        }};
        for (const auto& [eden_vk, host_vk] : ALIASES) {
            if (const auto it = g_rt_of_vk.find(eden_vk); it != g_rt_of_vk.end()) {
                g_rt_of_vk.try_emplace(host_vk, it->second);
            }
        }
        for (const auto df : DEPTH_FORMATS) {
            const PixelFormat pf = VideoCore::Surface::PixelFormatFromDepthFormat(df);
            g_depth_of_vk.try_emplace(SurfaceFormat(device, FormatType::Optimal, true, pf).format, df);
        }
        for (size_t i = 0; i < VideoCore::Surface::MaxPixelFormat; ++i) {
            const auto pf = static_cast<PixelFormat>(i);
            g_pixel_of_vk.try_emplace(SurfaceFormat(device, FormatType::Optimal, true, pf).format, pf);
        }
    });
}

Maxwell::ComparisonOp NvnCompare(u32 nvn) {
    return Compare(nvn);
}
Maxwell::StencilOp::Op NvnStencilOp(u32 nvn) {
    return StencilOp(nvn);
}
Maxwell::Blend::Equation NvnEquation(u32 nvn) {
    return Equation(nvn);
}
Maxwell::Blend::Factor NvnFactor(u32 nvn) {
    return Factor(nvn);
}
bool NvnAttribFormat(int vk_format, Maxwell::VertexAttribute::Size& size,
                     Maxwell::VertexAttribute::Type& type) {
    const auto it = std::ranges::find(ATTRIB_FORMATS, static_cast<VkFormat>(vk_format),
                                      &AttribFormat::vk);
    if (it == std::end(ATTRIB_FORMATS)) {
        return false;
    }
    size = it->size;
    type = it->type;
    return true;
}
Tegra::RenderTargetFormat NvnRenderTargetFormat(int vk_format) {
    const auto it = g_rt_of_vk.find(vk_format);
    return it != g_rt_of_vk.end() ? it->second : Tegra::RenderTargetFormat::NONE;
}

bool NvnHasDepthFormat(int vk_format) {
    return g_depth_of_vk.contains(vk_format);
}

PixelFormat PixelFormatOfVk(int vk_format) {
    const auto it = g_pixel_of_vk.find(vk_format);
    return it != g_pixel_of_vk.end() ? it->second : PixelFormat::A8B8G8R8_UNORM;
}

void FillNvnRegs(Maxwell& regs, const nvn_eden_draw_state& d) {
    std::memset(&regs, 0, sizeof(regs));

    regs.depth_mode = d.depth_mode_minus_one_to_one ? Maxwell::DepthMode::MinusOneToOne
                                                    : Maxwell::DepthMode::ZeroToOne;
    regs.polygon_mode_front = static_cast<Maxwell::PolygonMode>(0x1B00 + (std::min)(d.polygon_mode, 2u));
    regs.patch_vertices = d.patch_size ? d.patch_size : 32;   // 32 on the Maxwell path when never set (B7)
    // s20: the tessellation parameters (domain, spacing, winding), which NVN's driver writes from the evaluation program at
    // nvnProgramSetShaders, are not in the draw state: left at 0 by the memset above they made every evaluation shader an isoline
    // one (Bayonetta 3's fog, TES 152e86c9 / de621a43, drew as a net of lines over the Tokyo street). Until the program's own values
    // are read from its control section, a patch draw gets the native host translator's defaults: triangles, equal spacing, and the
    // vertex order its evaluation stages use (clockwise for the game's GL counter-clockwise), which the host's front face rule
    // (gfx_vk_eden.inc set_dynamic) and the upper-left domain origin pair with (gfx_vk_exec.c r82). Draws without patches keep their keys
    if (d.primitive == 0xE) {   // NVN_DRAW_PRIMITIVE_PATCHES
        regs.tessellation.params.domain_type.Assign(Maxwell::Tessellation::DomainType::Triangles);
        regs.tessellation.params.spacing.Assign(Maxwell::Tessellation::Spacing::Integer);
        regs.tessellation.params.output_primitives.Assign(Maxwell::Tessellation::OutputPrimitives::Triangles_CW);
    }
    regs.anti_alias_samples_mode = Msaa(d.samples);
    regs.transform_feedback_enabled = d.xfb != nullptr ? 1 : 0;   // s30b (was 0, the B4 follow-up): NVN transform feedback
    if (d.xfb != nullptr) {
        // the vertex shader's capture layout (host gfx_vk_shader.c vk_xfb_note, from its control section, where the NVN driver takes it): per buffer
        // stream 0, the 4-byte words per vertex and the stride, and the stream-out layout's attribute word at each position (the rest 0)
        static_assert(sizeof(regs.stream_out_layout[0]) == 128, "one buffer's stream-out layout");
        for (u32 b = 0; b < Maxwell::NumTransformFeedbackBuffers && b < 4; ++b) {
            auto& c = regs.transform_feedback.controls[b];
            c.stream = 0;
            c.varying_count = d.xfb->count[b];
            c.stride = d.xfb->stride[b];
            std::memcpy(&regs.stream_out_layout[b], d.xfb->words[b], 128);
        }
    }

    // blending
    regs.blend_per_target_enabled = 1;   // NVN blend state is per target (nvnBlendStateSetBlendTarget)
    for (u32 i = 0; i < Maxwell::NumRenderTargets; ++i) {
        regs.blend.enable[i] = (d.blend_enable >> i) & 1;
        auto& b = regs.blend_per_target[i];
        b.separate_alpha = 1;
        b.color_op = Equation(d.blend[i].eq);
        b.alpha_op = Equation(d.blend[i].eq_a);
        b.color_source = Factor(d.blend[i].func_src);
        b.color_dest = Factor(d.blend[i].func_dst);
        b.alpha_source = Factor(d.blend[i].func_src_a);
        b.alpha_dest = Factor(d.blend[i].func_dst_a);
        auto& m = regs.color_mask[i];
        m.R.Assign(d.channel_mask[i] & 1);
        m.G.Assign((d.channel_mask[i] >> 1) & 1);
        m.B.Assign((d.channel_mask[i] >> 2) & 1);
        m.A.Assign((d.channel_mask[i] >> 3) & 1);
    }
    regs.color_mask_common = 0;
    regs.logic_op.enable = d.logic_op_enable;
    regs.logic_op.op = static_cast<Maxwell::LogicOp::Op>(0x1500 + (d.logic_op & 0xF));

    // alpha test, early z, depth target, origin
    regs.alpha_test_enabled = d.alpha_func != 0 && d.alpha_func != 8;
    regs.alpha_test_func = Compare(d.alpha_func ? d.alpha_func : 8);
    regs.alpha_test_ref = d.alpha_ref;
    regs.mandated_early_z = d.early_fragment;
    regs.zeta_enable = d.depth_format != 0;
    // without a depth target the register keeps the last one's format (FixedPipelineState reads it)
    const int32_t zeta_vk = d.depth_format != 0 ? d.depth_format : d.depth_format_kept;
    regs.zeta.format = Tegra::DepthFormat::Z32_FLOAT;
    if (zeta_vk != 0) {
        const auto it = g_depth_of_vk.find(zeta_vk);
        regs.zeta.format = it != g_depth_of_vk.end() ? it->second : Tegra::DepthFormat::Z32_FLOAT;
    }
    regs.window_origin.mode.Assign(d.lower_left_origin ? Maxwell::WindowOrigin::Mode::LowerLeft
                                                       : Maxwell::WindowOrigin::Mode::UpperLeft);
    regs.window_origin.flip_y.Assign(d.lower_left_origin ? 1 : 0);   // [구현 때 확인] the NVN driver's flip_y
    regs.provoking_vertex = d.provoking_last ? Maxwell::ProvokingVertex::Last : Maxwell::ProvokingVertex::First;
    regs.conservative_raster_enable = d.conservative_raster;
    regs.line_anti_alias_enable = d.line_smooth;
    regs.anti_alias_alpha_control.alpha_to_coverage.Assign(d.alpha_to_coverage);
    regs.anti_alias_alpha_control.alpha_to_one.Assign(d.alpha_to_one);
    regs.depth_bounds[0] = d.depth_bounds[0];
    regs.depth_bounds[1] = d.depth_bounds[1];
    regs.line_stipple_enable = d.line_stipple;
    regs.line_stipple_params.factor.Assign(d.line_stipple_factor);
    regs.line_stipple_params.pattern.Assign(d.line_stipple_pattern);
    regs.point_size = d.point_size;

    // render targets
    for (u32 i = 0; i < Maxwell::NumRenderTargets; ++i) {
        regs.rt[i].format = Tegra::RenderTargetFormat::NONE;
        if (i < d.num_colors && d.color_format[i] != 0) {
            const auto it = g_rt_of_vk.find(d.color_format[i]);
            if (it != g_rt_of_vk.end()) {
                regs.rt[i].format = it->second;
            } else {
                LOG_ERROR(Render_Vulkan, "NVN colour target VkFormat {} has no Maxwell format",
                          d.color_format[i]);
            }
        }
    }

    // vertex input
    // an attribute the game did not set: constant R32 float in NVN's 16 attributes, as the NVN
    // driver leaves them; the other 16 registers keep the reset value, constant only (the Maxwell
    // path's FixedPipelineState: 0x09700000 for unused 0-15, 0 for 16-31; B7 runs 3 and 5)
    for (u32 i = 0; i < Maxwell::NumVertexAttributes; ++i) {
        auto& f = regs.vertex_attrib_format[i];
        f.constant.Assign(1);
        if (i < 16) {
            f.size.Assign(Maxwell::VertexAttribute::Size::Size_R32);
            f.type.Assign(Maxwell::VertexAttribute::Type::Float);
        }
    }
    for (u32 a = 0; a < d.num_attribs && a < 16; ++a) {
        const auto& in = d.attrib[a];
        if (in.location >= Maxwell::NumVertexAttributes) {
            continue;
        }
        auto& f = regs.vertex_attrib_format[in.location];
        const auto it = std::ranges::find(ATTRIB_FORMATS, static_cast<VkFormat>(in.vk_format),
                                          &AttribFormat::vk);
        if (it == std::end(ATTRIB_FORMATS)) {
            LOG_ERROR(Render_Vulkan, "NVN vertex attribute VkFormat {} has no Maxwell format",
                      in.vk_format);
            continue;
        }
        f.constant.Assign(0);
        f.buffer.Assign(in.stream);
        f.offset.Assign(in.offset);
        f.size.Assign(it->size);
        f.type.Assign(it->type);
    }
    for (u32 s = 0; s < d.num_streams && s < Maxwell::NumVertexArrays; ++s) {
        regs.vertex_streams[s].stride.Assign(d.stream[s].stride);
        regs.vertex_streams[s].enable.Assign(1);
        regs.vertex_streams[s].frequency = d.stream[s].divisor;
        regs.vertex_stream_instances.is_instanced[s] = d.stream[s].divisor != 0;
    }
    for (u32 v = 0; v < Maxwell::NumViewports; ++v) {
        regs.viewport_transform[v].swizzle.raw = v < 16 && d.viewport_swizzle[v] ? d.viewport_swizzle[v] : 0x6420;
    }

    // depth, stencil, raster
    regs.depth_test_enable = d.depth_test;
    regs.depth_write_enabled = d.depth_write;
    regs.depth_test_func = Compare(d.depth_func);
    regs.depth_bounds_enable = d.depth_bounds_test;
    regs.stencil_enable = d.stencil_test;
    regs.stencil_two_side_enable = 1;   // NVN keeps front and back state apart
    regs.stencil_front_op.func = Compare(d.stencil_func[0]);
    regs.stencil_front_op.fail = StencilOp(d.stencil_fail[0]);
    regs.stencil_front_op.zfail = StencilOp(d.stencil_zfail[0]);
    regs.stencil_front_op.zpass = StencilOp(d.stencil_zpass[0]);
    regs.stencil_back_op.func = Compare(d.stencil_func[1]);
    regs.stencil_back_op.fail = StencilOp(d.stencil_fail[1]);
    regs.stencil_back_op.zfail = StencilOp(d.stencil_zfail[1]);
    regs.stencil_back_op.zpass = StencilOp(d.stencil_zpass[1]);
    regs.gl_front_face = d.front_face ? Maxwell::FrontFace::CounterClockWise : Maxwell::FrontFace::ClockWise;
    regs.gl_cull_test_enabled = d.cull_face != 0;
    regs.gl_cull_face = d.cull_face == 1   ? Maxwell::CullFace::Front
                        : d.cull_face == 2 ? Maxwell::CullFace::Back
                                           : Maxwell::CullFace::FrontAndBack;
    regs.polygon_offset_point_enable = (d.polygon_offset >> 0) & 1;
    regs.polygon_offset_line_enable = (d.polygon_offset >> 1) & 1;
    regs.polygon_offset_fill_enable = (d.polygon_offset >> 2) & 1;
    regs.rasterize_enable = d.rasterizer_discard ? 0 : 1;
    regs.primitive_restart.enabled = d.primitive_restart;
    regs.viewport_clip_control.geometry_clip.Assign(
        d.depth_clamp ? Maxwell::ViewportClipControl::GeometryClip::FrustumXY
                      : Maxwell::ViewportClipControl::GeometryClip::FrustumXYZ);
}

void FillNvnGraphicsKey(FixedPipelineState& s, const Maxwell& regs,
                        Maxwell::PrimitiveTopology topology_, const DynamicFeatures& features) {
    s.driver_id = features.driver_id;
    s.driver_version = features.driver_version;

    s.raw1 = 0;
    s.extended_dynamic_state.Assign(features.has_extended_dynamic_state ? 1 : 0);
    s.extended_dynamic_state_2.Assign(features.has_extended_dynamic_state_2 ? 1 : 0);
    s.extended_dynamic_state_2_logic_op.Assign(features.has_extended_dynamic_state_2_logic_op ? 1 : 0);
    s.extended_dynamic_state_3_blend.Assign(features.has_extended_dynamic_state_3_blend ? 1 : 0);
    s.extended_dynamic_state_3_enables.Assign(features.has_extended_dynamic_state_3_enables ? 1 : 0);
    s.color_write_enable_dynamic.Assign(features.has_color_write_enable ? 1 : 0);
    s.dynamic_vertex_input.Assign(features.has_dynamic_vertex_input ? 1 : 0);
    s.xfb_enabled.Assign(regs.transform_feedback_enabled != 0);
    s.ndc_minus_one_to_one.Assign(regs.depth_mode == Maxwell::DepthMode::MinusOneToOne ? 1 : 0);
    s.polygon_mode.Assign(FixedPipelineState::PackPolygonMode(regs.polygon_mode_front));
    s.tessellation_primitive.Assign(static_cast<u32>(regs.tessellation.params.domain_type.Value()));
    s.tessellation_spacing.Assign(static_cast<u32>(regs.tessellation.params.spacing.Value()));
    s.tessellation_clockwise.Assign(regs.tessellation.params.output_primitives.Value() ==
                                    Maxwell::Tessellation::OutputPrimitives::Triangles_CW);
    s.patch_control_points_minus_one.Assign(regs.patch_vertices - 1);
    const bool collapse = features.has_extended_dynamic_state && features.has_extended_dynamic_state_2;
    s.topology.Assign(collapse ? TOPOLOGY_CLASS_REPRESENTATIVE_LUT[static_cast<size_t>(topology_)]
                               : topology_);
    s.msaa_mode.Assign(regs.anti_alias_samples_mode);
    s.attachment0_dual_source_blend.Assign(Attachment0DualSource(regs) ? 1 : 0);

    s.raw2 = 0;
    const auto test_func =
        regs.alpha_test_enabled != 0 ? regs.alpha_test_func : Maxwell::ComparisonOp::Always_GL;
    s.alpha_test_func.Assign(FixedPipelineState::PackComparisonOp(test_func));
    s.early_z.Assign(regs.mandated_early_z != 0 ? 1 : 0);
    s.depth_enabled.Assign(regs.zeta_enable != 0 ? 1 : 0);
    s.depth_format.Assign(static_cast<u32>(regs.zeta.format));
    s.y_negate.Assign(regs.window_origin.mode != Maxwell::WindowOrigin::Mode::UpperLeft ? 1 : 0);

    bool last = false;
    if (features.has_provoking_vertex &&
        (features.has_provoking_vertex_first_mode || features.has_provoking_vertex_last_mode)) {
        last = regs.provoking_vertex == Maxwell::ProvokingVertex::Last;
        if (regs.transform_feedback_enabled != 0 && !features.has_provoking_vertex_tf_preserve) {
            last = false;
        }
        if (last && !features.has_provoking_vertex_last_mode) {
            last = false;
        } else if (!last && !features.has_provoking_vertex_first_mode) {
            last = true;
        }
    }
    s.provoking_vertex_last.Assign(last ? 1 : 0);
    s.conservative_raster_enable.Assign(regs.conservative_raster_enable != 0 ? 1 : 0);
    s.smooth_lines.Assign(regs.line_anti_alias_enable != 0 ? 1 : 0);
    s.alpha_to_coverage_enabled.Assign(regs.anti_alias_alpha_control.alpha_to_coverage != 0 ? 1 : 0);
    s.alpha_to_one_enabled.Assign(regs.anti_alias_alpha_control.alpha_to_one != 0 ? 1 : 0);

    s.depth_bounds_min = static_cast<u32>(regs.depth_bounds[0]);
    s.depth_bounds_max = static_cast<u32>(regs.depth_bounds[1]);
    s.line_stipple_factor = regs.line_stipple_params.factor;
    s.line_stipple_pattern = regs.line_stipple_params.pattern;

    for (size_t i = 0; i < regs.rt.size(); ++i) {
        s.color_formats[i] = static_cast<u8>(regs.rt[i].format);
    }
    s.alpha_test_ref = std::bit_cast<u32>(regs.alpha_test_ref);
    s.NormalizeUnusedState();
    s.point_size = std::bit_cast<u32>(regs.point_size);

    // vertex input (Refresh does this under Dirty::VertexInput)
    if (features.has_dynamic_vertex_input) {
        static constexpr std::array LUT{0u, 1u, 1u, 2u, 3u, 1u, 1u, 1u};
        s.attribute_types = 0;
        for (size_t i = 0; i < Maxwell::NumVertexAttributes; ++i) {
            const auto& a = regs.vertex_attrib_format[i];
            const u32 mask = a.constant != 0 ? 0 : 3;
            const u32 type = LUT[static_cast<size_t>(a.type.Value())];
            s.attribute_types |= static_cast<u64>(type & mask) << (i * 2);
        }
    } else {
        s.enabled_divisors = 0;
        for (size_t i = 0; i < Maxwell::NumVertexArrays; ++i) {
            const bool on = regs.vertex_stream_instances.IsInstancingEnabled(i);
            s.binding_divisors[i] = on ? regs.vertex_streams[i].frequency : 0;
            s.enabled_divisors |= (on ? u64{1} : 0) << i;
        }
        for (size_t i = 0; i < Maxwell::NumVertexAttributes; ++i) {
            const auto& in = regs.vertex_attrib_format[i];
            auto& a = s.attributes[i];
            a.raw = 0;
            a.enabled.Assign(in.constant ? 0 : 1);
            a.buffer.Assign(in.buffer);
            a.offset.Assign(in.offset);
            a.type.Assign(static_cast<u32>(in.type.Value()));
            a.size.Assign(static_cast<u32>(in.size.Value()));
        }
    }
    std::ranges::transform(regs.viewport_transform, s.viewport_swizzles.begin(),
                           [](const auto& v) { return static_cast<u16>(v.swizzle.raw); });
    s.dynamic_state.raw1 = 0;
    s.dynamic_state.raw2 = 0;
    if (!s.extended_dynamic_state) {
        s.dynamic_state.Refresh(regs);
        std::ranges::transform(regs.vertex_streams, s.vertex_strides.begin(),
                               [](const auto& a) { return static_cast<u16>(a.stride.Value()); });
    }
    if (!s.extended_dynamic_state_2_logic_op) {
        s.dynamic_state.Refresh2(regs, topology_, s.extended_dynamic_state_2);
    }
    if (!s.extended_dynamic_state_3_blend) {
        for (size_t i = 0; i < s.attachments.size(); ++i) {
            auto& a = s.attachments[i];
            a.Refresh(regs, i);
            if (s.color_write_enable_dynamic && a.mask_r == 0 && a.mask_g == 0 && a.mask_b == 0 &&
                a.mask_a == 0) {
                a.mask_r.Assign(1);
                a.mask_g.Assign(1);
                a.mask_b.Assign(1);
                a.mask_a.Assign(1);
            }
        }
    }
    s.dynamic_state.Refresh3(regs, features);
    if (s.xfb_enabled) {
        RefreshXfbState(s.xfb_state, regs);
    }
}

} // namespace VideoCommon::Nvn
