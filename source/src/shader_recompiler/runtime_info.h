// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <map>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/varying_state.h"

namespace Shader {

enum class AttributeType : u8 {
    Float,
    SignedInt,
    UnsignedInt,
    SignedScaled,
    UnsignedScaled,
    Disabled,
};

enum class InputTopology {
    Points,
    Lines,
    LinesAdjacency,
    Triangles,
    TrianglesAdjacency,
};

struct InputTopologyVertices {
    static u32 vertices(InputTopology input_topology) {
        switch (input_topology) {
        case InputTopology::Lines:
            return 2;
        case InputTopology::LinesAdjacency:
            return 4;
        case InputTopology::Triangles:
            return 3;
        case InputTopology::TrianglesAdjacency:
            return 6;
        case InputTopology::Points:
        default:
            return 1;
        }
    }
};

enum class CompareFunction {
    Never,
    Less,
    Equal,
    LessThanEqual,
    Greater,
    NotEqual,
    GreaterThanEqual,
    Always,
};

enum class TessPrimitive {
    Isolines,
    Triangles,
    Quads,
};

enum class TessSpacing {
    Equal,
    FractionalOdd,
    FractionalEven,
};

struct TransformFeedbackVarying {
    u32 buffer{};
    u32 stream{};
    u32 stride{};
    u32 offset{};
    u32 components{};
};

/// Placement of one generic varying when VS->FS varyings are packed.
/// num_components == 0 means the generic is not part of the packed interface.
struct VaryingPackEntry {
    u8 location{};
    u8 component{};
    u8 first_element{};
    u8 num_components{};
};

/// vtxpull (NXN1X, design\VERTEX_PULLING_DESIGN.md): one generic input of a vertex stage that reads its attribute itself through a buffer device
/// address from the fetch table instead of the fixed-function vertex input (N1X fetches vertices ~100-250x slower from imported pool memory).
/// The layout is the VkFormat the fixed-function path would have fetched (MaxwellToVK::VertexFormat), described component by component.
enum class VertexPullKind : u8 { Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Sfloat, Ufloat };
struct VertexPullAttrib {
    u8 enabled{};      ///< read through the fetch table
    u8 stream{};       ///< fetch table entry (vertex stream)
    u8 components{};   ///< 1-4 (packed formats: 4 for 10-10-10-2, 3 for 11-11-10)
    u8 bits{};         ///< 8, 16 or 32 per component; 0 = packed (packed_layout)
    VertexPullKind kind{};
    u8 packed_layout{};   ///< 1: 10-10-10-2 (x low bits), 2: 11-11-10 unsigned float
    u16 offset{};      ///< byte offset of the attribute in the vertex
};

struct RuntimeInfo {
    std::array<AttributeType, 32> generic_input_types{};

    /// vtxpull: the vertex stage reads its enabled generic inputs through the fetch table whose 64-bit address is the push constant at
    /// VERTEX_PULL_TABLE_OFFSET; per stream in the table: address (u64), stride (u32), size in bytes (u32)
    bool vertex_pulling{};
    std::array<VertexPullAttrib, 32> vertex_pull_attribs{};
    std::array<u32, 32> vertex_pull_divisor{};   ///< per stream: 0 = per vertex, else per instance with this divisor

    /// Packed VertexB -> Fragment generic interface. Some drivers (Qualcomm on Windows) fail to
    /// link pipelines that declare many full vec4 varyings even when few components are used;
    /// both stages then declare only the used component range of each generic, packed with
    /// Location/Component decorations.
    bool varying_pack_enabled{};
    std::array<VaryingPackEntry, 32> varying_pack{};
    VaryingState previous_stage_stores;
    /// Fragment stage after a mesh program: generic inputs it writes per primitive.
    std::bitset<32> per_primitive_generics{};
    /// Last pre-rasterization stage: generic outputs the fragment stage reads screen-linear.
    /// They are written times Position.w and the fragment stage reads them perspective-correct
    /// times FragCoord.w, so clipping a primitive close to the camera interpolates them right.
    std::bitset<32> noperspective_outputs{};
    /// Fragment stage: the previous stage writes noperspective_outputs (not a mesh program).
    bool scaled_noperspective_inputs{};
    std::map<IR::Attribute, IR::Attribute> previous_stage_legacy_stores_mapping;

    bool convert_depth_mode{};
    bool force_early_z{};

    TessPrimitive tess_primitive{};
    TessSpacing tess_spacing{};
    bool tess_clockwise{};

    InputTopology input_topology{};

    std::optional<float> fixed_state_point_size;
    std::optional<CompareFunction> alpha_test_func;
    float alpha_test_reference{};

    /// Static Y negate value
    bool y_negate{};
    /// Use storage buffers instead of global pointers on GLASM
    bool glasm_use_storage_buffers{};

    /// Transform feedback state for each varying
    std::array<TransformFeedbackVarying, 256> xfb_varyings{};
    u32 xfb_count{0};

    /// Output types for each color attachment
    std::array<AttributeType, 8> color_output_types{};

    /// Dual source blending
    bool dual_source_blend{};

    /// Mesh program on a device without VK_EXT_mesh_shader: emit it as a compute shader that
    /// writes its vertices and primitives to the storage buffer at mesh_output_binding (see
    /// EmitMeshEmulationVertex for the layout), for a vertex shader to draw them.
    bool mesh_as_compute{};
    u32 mesh_output_binding{};
    /// Mesh emulation with a task program (mesh_as_compute on both). The task's compute shader
    /// writes its payload (count at word 1) to the payload buffer, task_payload_words words per
    /// task workgroup. Each mesh workgroup finds its task by a binary search of the offsets
    /// buffer (first mesh workgroup of each task) and copies that task's payload. The arguments
    /// buffer holds the draw's output base and task count (MESH_EMULATION_ARGS_*). ~0 when there
    /// is no task program.
    u32 mesh_task_payload_binding{~0U};
    u32 mesh_task_offsets_binding{~0U};
    u32 mesh_task_args_binding{~0U};
    u32 mesh_task_payload_stride{}; ///< payload buffer words per task workgroup
};

} // namespace Shader
