// SPDX-FileCopyrightText: Copyright 2026 NXN1X
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// vtxpull (design\VERTEX_PULLING_DESIGN.md, phase B): whether a graphics pipeline's vertex stage reads its generic inputs through the fetch table
// (buffer device addresses into the pools) instead of the fixed-function vertex input, and how each input is laid out. One decision, used by the
// shader's runtime info (vk_pipeline_cache.cpp) and by the pipeline's vertex input state (vk_graphics_pipeline.cpp), so the two always agree.
// Cause: on N1X (NVIDIA, Windows on ARM) the fixed-function fetch from VK_EXT_external_memory_host memory is ~100-250x slower; shader loads from
// the same memory are full speed (membench4: the production decoder bit-exact against the fixed-function fetch, the grass draw as fast as the mirror).

#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

// the layout of a vertex format the decoder reads (the VkFormat the fixed-function path would fetch); false = not pulled (BGRA, 64-bit, ...)
inline bool DescribeVertexPullFormat(VkFormat format, Shader::VertexPullAttrib& a) {
    using K = Shader::VertexPullKind;
    const auto set{[&a](u8 components, u8 bits, K kind) { a.components = components; a.bits = bits; a.kind = kind; a.packed_layout = 0; return true; }};
    switch (format) {
    case VK_FORMAT_R8_UNORM: return set(1, 8, K::Unorm);
    case VK_FORMAT_R8G8_UNORM: return set(2, 8, K::Unorm);
    case VK_FORMAT_R8G8B8_UNORM: return set(3, 8, K::Unorm);
    case VK_FORMAT_R8G8B8A8_UNORM: return set(4, 8, K::Unorm);
    case VK_FORMAT_R8_SNORM: return set(1, 8, K::Snorm);
    case VK_FORMAT_R8G8_SNORM: return set(2, 8, K::Snorm);
    case VK_FORMAT_R8G8B8_SNORM: return set(3, 8, K::Snorm);
    case VK_FORMAT_R8G8B8A8_SNORM: return set(4, 8, K::Snorm);
    case VK_FORMAT_R8_USCALED: return set(1, 8, K::Uscaled);
    case VK_FORMAT_R8G8_USCALED: return set(2, 8, K::Uscaled);
    case VK_FORMAT_R8G8B8_USCALED: return set(3, 8, K::Uscaled);
    case VK_FORMAT_R8G8B8A8_USCALED: return set(4, 8, K::Uscaled);
    case VK_FORMAT_R8_SSCALED: return set(1, 8, K::Sscaled);
    case VK_FORMAT_R8G8_SSCALED: return set(2, 8, K::Sscaled);
    case VK_FORMAT_R8G8B8_SSCALED: return set(3, 8, K::Sscaled);
    case VK_FORMAT_R8G8B8A8_SSCALED: return set(4, 8, K::Sscaled);
    case VK_FORMAT_R8_UINT: return set(1, 8, K::Uint);
    case VK_FORMAT_R8G8_UINT: return set(2, 8, K::Uint);
    case VK_FORMAT_R8G8B8_UINT: return set(3, 8, K::Uint);
    case VK_FORMAT_R8G8B8A8_UINT: return set(4, 8, K::Uint);
    case VK_FORMAT_R8_SINT: return set(1, 8, K::Sint);
    case VK_FORMAT_R8G8_SINT: return set(2, 8, K::Sint);
    case VK_FORMAT_R8G8B8_SINT: return set(3, 8, K::Sint);
    case VK_FORMAT_R8G8B8A8_SINT: return set(4, 8, K::Sint);
    case VK_FORMAT_R16_UNORM: return set(1, 16, K::Unorm);
    case VK_FORMAT_R16G16_UNORM: return set(2, 16, K::Unorm);
    case VK_FORMAT_R16G16B16_UNORM: return set(3, 16, K::Unorm);
    case VK_FORMAT_R16G16B16A16_UNORM: return set(4, 16, K::Unorm);
    case VK_FORMAT_R16_SNORM: return set(1, 16, K::Snorm);
    case VK_FORMAT_R16G16_SNORM: return set(2, 16, K::Snorm);
    case VK_FORMAT_R16G16B16_SNORM: return set(3, 16, K::Snorm);
    case VK_FORMAT_R16G16B16A16_SNORM: return set(4, 16, K::Snorm);
    case VK_FORMAT_R16_USCALED: return set(1, 16, K::Uscaled);
    case VK_FORMAT_R16G16_USCALED: return set(2, 16, K::Uscaled);
    case VK_FORMAT_R16G16B16_USCALED: return set(3, 16, K::Uscaled);
    case VK_FORMAT_R16G16B16A16_USCALED: return set(4, 16, K::Uscaled);
    case VK_FORMAT_R16_SSCALED: return set(1, 16, K::Sscaled);
    case VK_FORMAT_R16G16_SSCALED: return set(2, 16, K::Sscaled);
    case VK_FORMAT_R16G16B16_SSCALED: return set(3, 16, K::Sscaled);
    case VK_FORMAT_R16G16B16A16_SSCALED: return set(4, 16, K::Sscaled);
    case VK_FORMAT_R16_UINT: return set(1, 16, K::Uint);
    case VK_FORMAT_R16G16_UINT: return set(2, 16, K::Uint);
    case VK_FORMAT_R16G16B16_UINT: return set(3, 16, K::Uint);
    case VK_FORMAT_R16G16B16A16_UINT: return set(4, 16, K::Uint);
    case VK_FORMAT_R16_SINT: return set(1, 16, K::Sint);
    case VK_FORMAT_R16G16_SINT: return set(2, 16, K::Sint);
    case VK_FORMAT_R16G16B16_SINT: return set(3, 16, K::Sint);
    case VK_FORMAT_R16G16B16A16_SINT: return set(4, 16, K::Sint);
    case VK_FORMAT_R16_SFLOAT: return set(1, 16, K::Sfloat);
    case VK_FORMAT_R16G16_SFLOAT: return set(2, 16, K::Sfloat);
    case VK_FORMAT_R16G16B16_SFLOAT: return set(3, 16, K::Sfloat);
    case VK_FORMAT_R16G16B16A16_SFLOAT: return set(4, 16, K::Sfloat);
    case VK_FORMAT_R32_UINT: return set(1, 32, K::Uint);
    case VK_FORMAT_R32G32_UINT: return set(2, 32, K::Uint);
    case VK_FORMAT_R32G32B32_UINT: return set(3, 32, K::Uint);
    case VK_FORMAT_R32G32B32A32_UINT: return set(4, 32, K::Uint);
    case VK_FORMAT_R32_SINT: return set(1, 32, K::Sint);
    case VK_FORMAT_R32G32_SINT: return set(2, 32, K::Sint);
    case VK_FORMAT_R32G32B32_SINT: return set(3, 32, K::Sint);
    case VK_FORMAT_R32G32B32A32_SINT: return set(4, 32, K::Sint);
    case VK_FORMAT_R32_SFLOAT: return set(1, 32, K::Sfloat);
    case VK_FORMAT_R32G32_SFLOAT: return set(2, 32, K::Sfloat);
    case VK_FORMAT_R32G32B32_SFLOAT: return set(3, 32, K::Sfloat);
    case VK_FORMAT_R32G32B32A32_SFLOAT: return set(4, 32, K::Sfloat);
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: set(4, 0, K::Unorm); a.packed_layout = 1; return true;
    case VK_FORMAT_A2B10G10R10_SNORM_PACK32: set(4, 0, K::Snorm); a.packed_layout = 1; return true;
    case VK_FORMAT_A2B10G10R10_USCALED_PACK32: set(4, 0, K::Uscaled); a.packed_layout = 1; return true;
    case VK_FORMAT_A2B10G10R10_SSCALED_PACK32: set(4, 0, K::Sscaled); a.packed_layout = 1; return true;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: set(4, 0, K::Uint); a.packed_layout = 1; return true;
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: set(4, 0, K::Sint); a.packed_layout = 1; return true;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: set(3, 0, K::Ufloat); a.packed_layout = 2; return true;
    default: return false;
    }
}

// The decision for a pipeline: the key asks for pulling, the device has buffer device addresses and 64-bit integers, the vertex stage reads no
// attribute by a computed index, and every attribute it loads has a described format. out (may be NULL)
// gets the layouts. false: the pipeline keeps the fixed-function vertex input (nvn_eden_pipeline_info::vertex_pull says which the host must feed)
inline bool FillVertexPulling(const Device& device, const FixedPipelineState& state, const Shader::Info& vs, Shader::RuntimeInfo* out) {
    if (!state.vertex_pulling || state.dynamic_vertex_input || vs.loads_indexed_attributes || !device.IsBufferDeviceAddressSupported() ||
        !device.IsShaderInt64Supported()) {
        return false;
    }
    std::array<Shader::VertexPullAttrib, 32> attribs{};
    std::array<u32, 32> divisors{};
    for (size_t i = 0; i < Maxwell::NumVertexAttributes; ++i) {
        const auto& attribute = state.attributes[i];
        if (!attribute.enabled || !vs.loads.Generic(i)) {
            continue;
        }
        Shader::VertexPullAttrib& a = attribs[i];
        if (!DescribeVertexPullFormat(MaxwellToVK::VertexFormat(device, attribute.Type(), attribute.Size()), a)) {
            return false;
        }
        a.enabled = 1;
        a.stream = static_cast<u8>(attribute.buffer.Value());
        a.offset = static_cast<u16>(attribute.offset.Value());
    }
    // per stream as GraphicsPipeline::MakePipeline builds its bindings: divisor 0 = per vertex, otherwise per instance with that divisor
    for (size_t s = 0; s < Maxwell::NumVertexArrays; ++s) {
        divisors[s] = state.binding_divisors[s];
    }
    if (out) {
        out->vertex_pulling = true;
        out->vertex_pull_attribs = attribs;
        out->vertex_pull_divisor = divisors;
    }
    return true;
}

} // namespace Vulkan
