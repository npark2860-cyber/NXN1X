// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>   // nanguard
#include <cstdlib>
#include <chrono>   // pcache1
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <thread>
#include <vector>
#include <bit>
#include <numeric>
#include <optional>
#include <array>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <cstdio>   // s22
#include <mutex>
#include <unordered_set>
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/thread_worker.h"
#include "common/uuid.h"
#include "core/core.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"   // NVN2: Nvn2GmemMapShaders
#include "shader_recompiler/ir_opt/fragment_output_guard.h"   // nanguard
#include "video_core/renderer_vulkan/nvn_bridge_device.h"   // NVN2: the gate line (bridge attached)
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/s67_timing.h"   // s67
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_vertex_pulling.h"   // vtxpull
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#if NXN1X_DIAGNOSTICS
#include "video_core/renderer_vulkan/shader_record.h"
#endif

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {
// s22 (NXN1X_PIPEKEY_DUMP=<file>): each distinct graphics pipeline key a draw asks for is written once, tagged by the path that asked
// (M: the Maxwell path's CurrentGraphicsPipelineSlowPath, N: the NVN bridge's NvnGraphicsPipeline), as the 6 shader hashes and the
// FixedPipelineState bytes the key holds (state.Size()). Booting the same scene once emulated and once on the NVN path, the keys of
// the same shaders show field by field which state the NVN mapping (video_core/nvn/nvn_pipeline_key.cpp) gives differently. Off unless set.
void S22KeyNote(char path, const Vulkan::GraphicsPipelineCacheKey& key) {
    static const char* const file = std::getenv("NXN1X_PIPEKEY_DUMP");
    if (file == nullptr || *file == '\0') {
        return;
    }
    static std::mutex mutex;
    static std::FILE* out;
    static std::unordered_set<u64> seen;
    const u64 hash = static_cast<u64>(key.Hash()) * 2 + (path == 'N' ? 1 : 0);
    std::scoped_lock lock{mutex};
    if (!seen.insert(hash).second) {
        return;
    }
    if (out == nullptr && (out = std::fopen(file, "w")) == nullptr) {
        return;
    }
    std::fprintf(out, "%c", path);
    for (const u64 h : key.unique_hashes) {
        std::fprintf(out, " %016llx", static_cast<unsigned long long>(h));
    }
    std::fputc(' ', out);
    const auto* const bytes = reinterpret_cast<const unsigned char*>(&key.state);
    for (size_t i = 0; i < key.state.Size(); ++i) {
        std::fprintf(out, "%02x", bytes[i]);
    }
    std::fputc('\n', out);
    std::fflush(out);
}
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = 20; // hybrid1: StorageBufferDescriptor::is_atomic (19: NVN2 draw-parameter replacements must be recorded in rebuilt recipes)
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

// Recipes in vulkan.bin hold guest shaders and pipeline keys, not driver output, and keys the
// current device cannot use are skipped at load. Start a new driver folder from the largest
// compatible sibling so its pipelines are built during boot instead of mid-game.
void SeedRecipeCacheFromSiblings(const std::filesystem::path& base_dir,
                                 const std::filesystem::path& target) {
    std::error_code ec;
    if (std::filesystem::exists(target, ec)) {
        return;
    }
    static constexpr std::array<char, 8> recipe_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
    std::filesystem::path best;
    std::uintmax_t best_size = 0;
    for (const auto& entry : std::filesystem::directory_iterator(base_dir.parent_path(), ec)) {
        if (!entry.is_directory(ec) || entry.path() == base_dir) {
            continue;
        }
        const auto candidate = entry.path() / "vulkan.bin";
        const auto size = std::filesystem::file_size(candidate, ec);
        if (ec || size <= best_size) {
            ec.clear();
            continue;
        }
        std::ifstream file{candidate, std::ios::binary};
        std::array<char, 8> magic{};
        u32 version{};
        file.read(magic.data(), magic.size()).read(reinterpret_cast<char*>(&version), sizeof(version));
        if (file && magic == recipe_magic && version == CACHE_VERSION) {
            best = candidate;
            best_size = size;
        }
    }
    if (best.empty()) {
        return;
    }
    if (std::filesystem::copy_file(best, target, ec)) {
        LOG_INFO(Render_Vulkan, "Seeded recipe cache from {} ({} bytes)",
                 Common::FS::PathToUTF8String(best), best_size);
    } else {
        LOG_WARNING(Render_Vulkan, "Failed to seed recipe cache from {}: {}",
                    Common::FS::PathToUTF8String(best), ec.message());
    }
}

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const GraphicsPipelineCacheKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program,
                                    const Vulkan::Device& device) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        // Generics the mesh program writes per primitive (Lichii 02504aeb).
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const bool gl_ndc{key.state.ndc_minus_one_to_one != 0};
    const float point_size{std::bit_cast<float>(key.state.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.state.topology == Maxwell::PrimitiveTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            if (key.state.xfb_enabled) {
                if (device.IsExtTransformFeedbackSupported()) {
                    auto [varyings, count] =
                        VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                    info.xfb_varyings = varyings;
                    info.xfb_count = count;
                } else {
                    LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
                }
            }
            info.convert_depth_mode = gl_ndc;
        }
        if (key.state.dynamic_vertex_input) {
            for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
                info.generic_input_types[index] = AttributeType(key.state, index);
            }
        } else {
            std::ranges::transform(key.state.attributes, info.generic_input_types.begin(),
                                   &CastAttributeType);
        }
        // vtxpull: the same decision vk_graphics_pipeline.cpp makes for the vertex input state (vk_vertex_pulling.h)
        FillVertexPulling(device, key.state, program.info, &info);
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.state.tessellation_clockwise != 0;
        info.tess_primitive = [&key] {
            const u32 raw{key.state.tessellation_primitive.Value()};
            switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
            case Maxwell::Tessellation::DomainType::Isolines:
                return Shader::TessPrimitive::Isolines;
            case Maxwell::Tessellation::DomainType::Triangles:
                return Shader::TessPrimitive::Triangles;
            case Maxwell::Tessellation::DomainType::Quads:
                return Shader::TessPrimitive::Quads;
            }
            ASSERT(false);
            return Shader::TessPrimitive::Triangles;
        }();
        info.tess_spacing = [&] {
            const u32 raw{key.state.tessellation_spacing};
            switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
            case Maxwell::Tessellation::Spacing::Integer:
                return Shader::TessSpacing::Equal;
            case Maxwell::Tessellation::Spacing::FractionalOdd:
                return Shader::TessSpacing::FractionalOdd;
            case Maxwell::Tessellation::Spacing::FractionalEven:
                return Shader::TessSpacing::FractionalEven;
            }
            ASSERT(false);
            return Shader::TessSpacing::Equal;
        }();
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = gl_ndc;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        if (key.state.xfb_enabled != 0) {
            if (device.IsExtTransformFeedbackSupported()) {
                auto [varyings, count] =
                    VideoCommon::MakeTransformFeedbackVaryings(key.state.xfb_state);
                info.xfb_varyings = varyings;
                info.xfb_count = count;
            } else {
                LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
            }
        }
        info.convert_depth_mode = gl_ndc;
        break;
    case Shader::Stage::Fragment: {
        info.alpha_test_func = MaxwellToCompareFunction(
            key.state.UnpackComparisonOp(key.state.alpha_test_func.Value()));
        info.alpha_test_reference = std::bit_cast<float>(key.state.alpha_test_ref);

        info.dual_source_blend = key.state.attachment0_dual_source_blend != 0;

        {   // pkm4: every device, not only MoltenVK (Vulkan: an output's numeric type must match its attachment; Pokemon Sword/Shield G-buffer colour 1 is RGBA16Uint)
            for (size_t i = 0; i < 8; ++i) {
                const auto format = static_cast<Tegra::RenderTargetFormat>(key.state.color_formats[i]);
                if (format == Tegra::RenderTargetFormat::NONE) {   // pkm6: an unused target (no format lookup, which asserts)
                    info.color_output_types[i] = Shader::AttributeType::Float;
                    continue;
                }
                const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
                if (VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
                    if (VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)) {
                        info.color_output_types[i] = Shader::AttributeType::SignedInt;
                    } else {
                        info.color_output_types[i] = Shader::AttributeType::UnsignedInt;
                    }
                } else {
                    info.color_output_types[i] = Shader::AttributeType::Float;
                }
            }
        }
        break;
    }
    default:
        break;
    }
    switch (key.state.topology) {
    case Maxwell::PrimitiveTopology::Points:
        info.input_topology = Shader::InputTopology::Points;
        break;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        info.input_topology = Shader::InputTopology::Lines;
        break;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        info.input_topology = Shader::InputTopology::Triangles;
        break;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        info.input_topology = Shader::InputTopology::LinesAdjacency;
        break;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        info.input_topology = Shader::InputTopology::TrianglesAdjacency;
        break;
    }
    info.force_early_z = key.state.early_z != 0;
    info.y_negate = key.state.y_negate != 0;
    return info;
}

// Qualcomm's Windows driver fails vkCreateGraphicsPipelines with VK_ERROR_UNKNOWN when a
// VertexB -> Fragment interface declares many full vec4 varyings (TotK: 24 generics, only 55 of
// the 96 components used). Declare only each generic's used component range and pack the ranges
// into shared locations with Component decorations. Both stages get the same table, so the
// interface still matches. EDEN_VARYING_PACK=0 disables, =1 forces it for every pipeline,
// EDEN_VARYING_PACK_THRESHOLD=N packs once more than N generic locations are used (default 16).
std::optional<std::array<Shader::VaryingPackEntry, 32>> MakeVaryingPack(
    std::span<const Shader::IR::Program> programs, const GraphicsPipelineCacheKey& key,
    const Device& device) {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_VARYING_PACK");
        return value != nullptr && value[0] != '\0' ? std::atoi(value) : -1;
    }();
    static const u32 threshold = [] {
        const char* const value = std::getenv("EDEN_VARYING_PACK_THRESHOLD");
        return value != nullptr && value[0] != '\0' ? static_cast<u32>(std::atoi(value)) : 16u;
    }();
    if (mode == 0) {
        return std::nullopt;
    }
    if (mode != 1 && device.GetDriverID() != VK_DRIVER_ID_QUALCOMM_PROPRIETARY) {
        return std::nullopt;
    }
    const bool vertex_b_fragment_only = key.unique_hashes[0] == 0 && key.unique_hashes[1] != 0 &&
                                        key.unique_hashes[2] == 0 && key.unique_hashes[3] == 0 &&
                                        key.unique_hashes[4] == 0 && key.unique_hashes[5] != 0;
    if (!vertex_b_fragment_only || key.state.xfb_enabled != 0) {
        return std::nullopt;
    }
    const Shader::Info& vs{programs[1].info};
    const Shader::Info& fs{programs[5].info};
    if (vs.stores_indexed_attributes || fs.loads_indexed_attributes || vs.stores.Legacy() ||
        fs.loads.Legacy()) {
        return std::nullopt;
    }
    struct Item {
        u32 index;
        u32 first;
        u32 count;
        Shader::Interpolation interpolation;
    };
    std::vector<Item> items;
    for (u32 index = 0; index < 32; ++index) {
        if (!vs.stores.Generic(index)) {
            continue;
        }
        u32 first = 4;
        u32 last = 0;
        for (u32 component = 0; component < 4; ++component) {
            if (vs.stores.Generic(index, component) || fs.loads.Generic(index, component)) {
                first = (std::min)(first, component);
                last = (std::max)(last, component);
            }
        }
        const Shader::Interpolation interpolation{fs.loads.Generic(index)
                                                      ? fs.interpolation[index]
                                                      : Shader::Interpolation::Smooth};
        items.push_back(Item{index, first, last - first + 1, interpolation});
    }
    if (mode != 1 && items.size() <= threshold) {
        return std::nullopt;
    }
    std::stable_sort(items.begin(), items.end(),
                     [](const Item& a, const Item& b) { return a.count > b.count; });
    struct Slot {
        Shader::Interpolation interpolation;
        u32 used;
    };
    std::vector<Slot> slots;
    std::array<Shader::VaryingPackEntry, 32> table{};
    for (const Item& item : items) {
        size_t slot_index = slots.size();
        for (size_t candidate = 0; candidate < slots.size(); ++candidate) {
            if (slots[candidate].interpolation == item.interpolation &&
                slots[candidate].used + item.count <= 4) {
                slot_index = candidate;
                break;
            }
        }
        if (slot_index == slots.size()) {
            slots.push_back(Slot{item.interpolation, 0});
        }
        Slot& slot{slots[slot_index]};
        table[item.index] = Shader::VaryingPackEntry{
            .location = static_cast<u8>(slot_index),
            .component = static_cast<u8>(slot.used),
            .first_element = static_cast<u8>(item.first),
            .num_components = static_cast<u8>(item.count),
        };
        slot.used += item.count;
    }
    if (slots.size() > 32) {
        return std::nullopt;
    }
    LOG_INFO(Render_Vulkan, "Packed {} varyings into {} locations (vs {:016x} fs {:016x})",
             items.size(), slots.size(), key.unique_hashes[1], key.unique_hashes[5]);
    return table;
}

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_},
      render_pass_cache{render_pass_cache_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue()},
      workers(device.HasBrokenParallelShaderCompiling() ? 1ULL : GetTotalPipelineWorkers(),
              "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {},
                           Common::ThreadPlacement::Background) {
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    // EDEN_GPU_NO_SUBGROUP=1: report no subgroup support to the recompiler (diagnostic only;
    // votes/shuffles degrade to single-thread semantics).
    const bool no_subgroup = device.GetOutputPolicy().DisableSubgroups();
    const u32 supported_subgroup_stages_raw{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    const u32 supported_subgroup_stages{no_subgroup ? 0u : supported_subgroup_stages_raw};
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        // NXN1X S3z13: Turnip only (Qualcomm launches the helper lanes anyway). EDEN_FULL_QUADS=0: off
        .support_full_quads = driver_id == VK_DRIVER_ID_MESA_TURNIP && [] {
            const char* const value = std::getenv("EDEN_FULL_QUADS");
            return value == nullptr || value[0] != '0';
        }(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        // ootfadd A/B switch (hybrid+ootfadd): NXN1X_OOT_FADD=0 keeps the compare-and-swap loop; read once, when the profile is made
        .support_float32_atomic_add = [this] {
            const char* e = std::getenv("NXN1X_OOT_FADD");
            const bool hw = device.IsBufferFloat32AtomicAddSupported() && !(e && e[0] == '0');
            LOG_INFO(Render_Vulkan, "ootfadd: float32 atomic add = {} (device {}, NXN1X_OOT_FADD={})", hw ? "hw" : "cas",
                     device.IsBufferFloat32AtomicAddSupported() ? "supports it" : "does not support it", e ? e : "unset");
            return hw;
        }(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        // Maxwell gathers round to eight fractional bits. Adreno X1-85 was measured
        // choosing the preceding texel at n + 0.5 - 0.001 without this offset. Turnip
        // drives the same Adreno texture unit, so it needs the offset too.
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == false,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        // Qualcomm's compiler miscompiles OpVectorExtractDynamic on subgroup ballot masks (yuzu
        // enabled this workaround for the proprietary Adreno driver). With Adreno's 64-wide
        // subgroups every emulated 32-wide guest vote/ballot goes through WarpExtract, so a bad
        // extract corrupts whole screen tiles. EDEN_QCOM_SUBGROUP_EXTRACT_FIX=0 disables it.
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            device.GetOutputPolicy().NeedsSubgroupMaskExtractWorkaround(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = std::min<u32>(
            device.GetPhysical().GetProperties().limits.maxPerStageDescriptorStorageBuffers,
            device.IsNvnHostDevice() ? Shader::Info::MAX_SSBOS : 64),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
    };
    // htarr (FE Fortune's Weave 11-22): support_handle_table_arrays was never set, so a texture handle read from a table at an index each invocation
    // computes (FE's per-light shadow / cookie textures: cbuf0+0x510 light table, a light mask loop) always fell to "a pointer: no texture" and the
    // lit scene came out black. The NVN host fills each element from the table (nvn_eden_eval_handle, HandleOp::Index) and the SPIR-V backend marks
    // the index NonUniform, so the NVN host device takes the arrays where sampled image arrays may be indexed non-uniformly. NXN1X_HANDLE_TABLE_ARRAYS=0 off
    {
        const char* e = std::getenv("NXN1X_HANDLE_TABLE_ARRAYS");
        host_info.support_handle_table_arrays = device.IsNvnHostDevice() && device.IsSampledImageArrayNonUniformIndexingSupported() && !(e && *e == '0');
        LOG_INFO(Render_Vulkan, "htarr: handle table arrays {} (NVN host device {}, non-uniform sampled image indexing {}; NXN1X_HANDLE_TABLE_ARRAYS=0 off)",
                 host_info.support_handle_table_arrays ? "on" : "off", device.IsNvnHostDevice(), device.IsSampledImageArrayNonUniformIndexingSupported());
    }
    host_info.ApplyDescriptorLimitPolicy();

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();
}

PipelineCache::~PipelineCache() {
    serialization_thread.WaitForRequests();
    ReportCacheUsage("shutdown");
    if (host_info.nvn2_gmem_map) {
        u64 no_binding{};
        const u64 with_table{Shader::Optimization::Nvn2GmemMapShaders(&no_binding)};
        LOG_INFO(Render_Vulkan, "NVN2: {} shaders translated with the memory pool table this boot (raw global memory), {} without a free binding for it",
                 with_table, no_binding);
    }
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
}

void PipelineCache::TickNvnDriverCache() {
    const u64 builds = nvn_graphics_requests.load(std::memory_order_relaxed) +
                       nvn_compute_requests.load(std::memory_order_relaxed);
    if (builds == checkpoint_nvn_builds || !use_vulkan_pipeline_cache) return;
    const auto now = std::chrono::steady_clock::now();
    if (builds != observed_nvn_builds) {
        if (!observed_nvn_builds) last_checkpoint = now - std::chrono::seconds{30};
        observed_nvn_builds = builds;
        last_nvn_build = now;
    }
    // Save off the presentation thread once the first compile burst settles.
    // Coalesce later bursts so a busy level does not continually rewrite the blob.
    // pcache1 (OoT 11-21): a game that builds a few pipelines every half minute had the whole driver blob (61 MB, vkGetPipelineCacheData and a file
    // rewrite) saved every 30-34 s during play, a recurring burst beside the frames. During play it is now saved once the builds have been quiet for
    // 20 s, or at the latest 5 minutes after the last save (what a crash can lose); the first save after boot and the one at shutdown are as before.
    // NXN1X_PCACHE_PERIODIC=1: every 30 s as before
    static const bool periodic = [] { const char* e = std::getenv("NXN1X_PCACHE_PERIODIC"); return e != nullptr && *e == '1'; }();
    const bool first = checkpoint_nvn_builds == 0;
    const bool due = periodic || first ? !((now - last_nvn_build < std::chrono::seconds{2} && now - last_checkpoint < std::chrono::minutes{1}) ||
                                           now - last_checkpoint < std::chrono::seconds{30})
                                       : (now - last_nvn_build >= std::chrono::seconds{60} && now - last_checkpoint >= std::chrono::seconds{30}) ||   // replaydiag build: 20 -> 60 s (OoT 09:43: 9 saves in 10 min at 20 s)
                                             now - last_checkpoint >= std::chrono::minutes{5};
    if (!due || vulkan_pipeline_cache_filename.empty() || checkpoint_pending.exchange(true)) return;
    last_checkpoint = now;
    serialization_thread.QueueWork([this, builds] {
        if (SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache, CACHE_VERSION))
            checkpoint_nvn_builds.store(builds, std::memory_order_relaxed);
        checkpoint_pending.store(false);
    });
}

void PipelineCache::ReportCacheUsage(const char* phase) const {
    LOG_INFO(Render_Vulkan,
             "NXN1X_CACHE_SUMMARY phase={} disk_gfx_recipes={} disk_compute_recipes={} "
             "disk_shader_environments={} runtime_gfx_build_requests={} "
             "runtime_compute_build_requests={} driver_blob_bytes_supplied={} "
             "driver_cache_hits=unknown driver_recompiles=unknown",
             phase, disk_graphics_recipes.load(std::memory_order_relaxed),
             disk_compute_recipes.load(std::memory_order_relaxed),
             disk_shader_environments.load(std::memory_order_relaxed),
             runtime_graphics_requests.load(std::memory_order_relaxed),
             runtime_compute_requests.load(std::memory_order_relaxed), driver_cache_bytes_supplied);
    LOG_INFO(Render_Vulkan,
             "NXN1X_NVN_CACHE_SUMMARY phase={} native_gfx_build_requests={} "
             "native_compute_build_requests={} native_unserializable_recipes={} "
             "(native requests included in runtime totals; build requests are not driver recompilation counts)",
             phase, nvn_graphics_requests.load(std::memory_order_relaxed),
             nvn_compute_requests.load(std::memory_order_relaxed),
             nvn_unserializable_recipes.load(std::memory_order_relaxed));
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {

    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& pipeline{pair->second};
    if (!is_new) {
        return pipeline.get();
    }
    pipeline = CreateComputePipeline(key, shader);
    return pipeline.get();
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    // NVN2 (Switch 2 programs on Turnip), as Lichii-build-177 (vk_pipeline_cache.cpp:573-575, 703) does on Turnip: geometry shaders are translated without
    // subgroup operations (one lane), which turns on the geometry compaction pass (shader_recompiler geometry_compaction_pass.cpp: a subgroup-ballot slot
    // append becomes per invocation). Turnip runs a geometry shader's body once per output vertex; Donkey Kong Bananza's terrain GS appends its triangles
    // with a ballot and a storage atomic. Set here, per boot before the disk cache and any pipeline: the cache is made with the renderer, before the program
    // is known (core.cpp SetCurrentProgramIsSwitch2). Switch 1 programs keep the device's subgroup stages
    if (device.GetDriverID() == VK_DRIVER_ID_MESA_TURNIP && Settings::CurrentProgramIsSwitch2() &&
        profile.SupportsSubgroupStage(Shader::Stage::Geometry)) {
        profile.supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
        host_info.single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry);
        LOG_INFO(Render_Vulkan, "NVN2: geometry shaders without subgroups on Turnip (single-lane geometry subgroups, geometry compaction on)");
    }
    // NVN2 raw global memory (shader_recompiler host_translate_info nvn2_gmem_map): Switch 2 programs on the NVN host layer's device, which binds the pool table
    // (host gfx_vk_eden.inc eden_ssbo, gfx_vk_core.c vk_gmem_map). Needs physical storage buffer pointers and 64-bit integers. Switch 1 programs, and Switch 2
    // programs in emu mode (Eden's buffer cache, which has no pool table), are unchanged. The recipe cache re-translates every boot: no cache to delete
    // Each condition from what the device was created with (the vkCreateDevice chain the NVN host layer also reads, nvn_bridge_device.cpp HandOver): buffer device
    // address from its feature structure or the Vulkan 1.2 one, not the extension flag; 64-bit integers from the enabled core features and the shader profile.
    // One line per Switch 2 boot with every value, so a gate that stays off says why
    if (Settings::CurrentProgramIsSwitch2()) {
        const VkPhysicalDeviceFeatures2& enabled = device.GetEnabledFeatures2();
        bool bda = false;
        for (auto* s = static_cast<const VkBaseInStructure*>(enabled.pNext); s != nullptr; s = s->pNext) {
            if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
                bda = bda || reinterpret_cast<const VkPhysicalDeviceBufferDeviceAddressFeatures*>(s)->bufferDeviceAddress != VK_FALSE;
            } else if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) {
                bda = bda || reinterpret_cast<const VkPhysicalDeviceVulkan12Features*>(s)->bufferDeviceAddress != VK_FALSE;
            }
        }
        const bool host_device = device.IsNvnHostDevice();
        const bool int64 = enabled.features.shaderInt64 != VK_FALSE && profile.support_int64;
        // Per title: on by default for Donkey Kong Bananza only (NPARK 11-16: its terrain), other Switch 2 titles translate as before.
        // DM_NVN2_GMEM=1 turns it on for any Switch 2 title, DM_NVN2_GMEM=0 off for all
        const char* const gmem_env = std::getenv("DM_NVN2_GMEM");
        const bool env_off = gmem_env && gmem_env[0] == '0';
        const bool env_on = gmem_env && gmem_env[0] == '1';
        const bool title_on = title_id == 0x040074A01BF12000ULL;   // Donkey Kong Bananza
        host_info.nvn2_gmem_map = host_device && bda && int64 && !env_off && (title_on || env_on);
        LOG_INFO(Render_Vulkan, "NVN2: global memory traced to no storage buffer {} (title {:016X}: {}; NVN host device {}, bridge attached {}, buffer device address {}, "
                 "shaderInt64 {} / profile int64 {}, DM_NVN2_GMEM {}; DM_NVN2_GMEM=1 turns it on for any Switch 2 title, 0 off)",
                 host_info.nvn2_gmem_map ? "goes through the host's memory pool table" : "is dropped as before: the pool table is off", title_id,
                 title_on ? "on by default" : "off by default", host_device, NvnBridge::GetDevice() != nullptr, bda,
                 enabled.features.shaderInt64 != VK_FALSE, profile.support_int64, gmem_env ? gmem_env : "unset");
    }
    const auto shader_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto& policy = device.GetOutputPolicy();
    if (!policy.CanPersistCaches()) {
        LOG_WARNING(Render_Vulkan, "Persistent Vulkan caches disabled by output diagnostics");
        if (use_vulkan_pipeline_cache) {
            vulkan_pipeline_cache = LoadVulkanPipelineCache({}, CACHE_VERSION);
        }
        return;
    }
    auto base_dir = shader_dir / fmt::format("{:016x}", title_id);
    if (policy.GetEffectiveMode() == Settings::VulkanBackendMode::Adreno) {
        base_dir /= "adreno";
        base_dir /= fmt::format("{:016x}", policy.GetDigest());
    }
    if (!Common::FS::CreateDirs(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename = base_dir / "vulkan.bin";
    if (policy.GetEffectiveMode() == Settings::VulkanBackendMode::Adreno) {
        SeedRecipeCacheFromSiblings(base_dir, pipeline_cache_filename);
    }
    LOG_INFO(Render_Vulkan, "Vulkan recipe cache: {} (policy {:016x})",
             Common::FS::PathToUTF8String(pipeline_cache_filename), policy.GetDigest());

    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename = base_dir / "vulkan_pipelines.bin";
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
    } state;

    // Normalize older keys before translation so equivalent recipes share one pipeline.
    std::unordered_set<GraphicsPipelineCacheKey> queued_graphics;
    u64 duplicate_graphics_recipes{};
    std::unordered_set<ComputePipelineCacheKey> queued_compute;
    size_t duplicate_compute_recipes{};

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!file) return;
        if (!queued_compute.emplace(key).second) {
            ++duplicate_compute_recipes;
            return;
        }

        workers.QueueWork([this, key, env_ = std::move(env), &state, &callback]() mutable {
            ShaderPools pools;
            auto pipeline{CreateComputePipeline(pools, key, env_, state.statistics.get(), false)};
            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                const auto [entry, inserted] = compute_cache.emplace(key, std::move(pipeline));
                if (inserted) {
                    disk_compute_recipes.fetch_add(1, std::memory_order_relaxed);
                    disk_shader_environments.fetch_add(1, std::memory_order_relaxed);
                }
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        key.state.NormalizeUnusedState();

        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
            return;
        }

        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        if (!queued_graphics.emplace(key).second) {
            ++duplicate_graphics_recipes;
            return;
        }

        workers.QueueWork([this, key, envs_ = std::move(envs), &state, &callback]() mutable {
            ShaderPools pools;
            boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
            for (auto& env : envs_) {
                env_ptrs.push_back(&env);
            }
            auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs),
                                                 state.statistics.get(), false)};

            std::scoped_lock lock{state.mutex};
            if (pipeline) {
                const auto [entry, inserted] = graphics_cache.emplace(key, std::move(pipeline));
                if (inserted) {
                    disk_graphics_recipes.fetch_add(1, std::memory_order_relaxed);
                    disk_shader_environments.fetch_add(envs_.size(), std::memory_order_relaxed);
                }
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {}", state.total);
    LOG_INFO(Render_Vulkan,
             "Vulkan recipe normalization: {} equivalent graphics recipes skipped before build, {} unique graphics recipes queued",
             duplicate_graphics_recipes, queued_graphics.size());
    LOG_INFO(Render_Vulkan, "Compute recipe normalization: {} duplicates skipped, {} unique queued",
             duplicate_compute_recipes, queued_compute.size());

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);
    ReportCacheUsage(stop_loading.stop_requested() ? "load_cancelled" : "loaded");
    if (host_info.nvn2_gmem_map) {
        u64 no_binding{};
        const u64 with_table{Shader::Optimization::Nvn2GmemMapShaders(&no_binding)};
        LOG_INFO(Render_Vulkan, "NVN2: {} shaders of the disk cache translated with the memory pool table (raw global memory), {} without a free binding for it",
                 with_table, no_binding);
    }

    if (use_vulkan_pipeline_cache && state.built) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }

    if (state.statistics) {
        state.statistics->Report();
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    S22KeyNote('M', graphics_key);   // s22
    const auto [pair, is_new]{graphics_cache.try_emplace(graphics_key)};
    auto& pipeline{pair->second};
    if (is_new) {
        pipeline = CreateGraphicsPipeline();
    }
    if (!pipeline) {
        return nullptr;
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    return BuiltPipeline(current_pipeline);
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    // If games are using a small index count, we can assume these are full screen quads.
    // Usually these shaders are only used once for building textures so we can assume they
    // can't be built async
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (draw_state.index_buffer.count <= 6 || draw_state.vertex_buffer.count <= 6) {
        return pipeline;
    }
    return nullptr;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel) try {
    S67::g_front = {}; S67::g_front.t0 = S67::Now();   // s67: front-end times of this pipeline (s67_timing.h)
    auto hash = key.Hash();
    LOG_INFO(Render_Vulkan, "{:#016x}", hash);
    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            auto topology = MaxwellToOutputTopology(key.state.topology);
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program, topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline: no pipeline,
        // so its draws are skipped (Lichii b00481fc).
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            static std::atomic<bool> logged{};
            if (!logged.exchange(true, std::memory_order_relaxed)) {
                LOG_WARNING(Render_Vulkan,
                            "{} has no VK_EXT_mesh_shader: task/mesh pipelines are skipped",
                            device.GetModelName());
            }
            return nullptr;
        }
        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env, host_info);
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    std::array<vk::ShaderModule, Maxwell::MaxShaderStage> modules;
    const auto varying_pack{MakeVaryingPack(programs, key, device)};
    S67::g_front.a = S67::Now() - S67::g_front.t0;   // s67: environments + translation

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        if (device.IsNvnHostDevice() && index == 1 &&
            (program.info.is_task || program.info.is_mesh)) {
            if (program.info.storage_buffers_descriptors.size() >= Shader::Optimization::StorageBufferLimit(host_info)) {
                throw Shader::NotImplementedException("No storage binding for the NVN first task index");
            }
            program.info.storage_buffers_descriptors.push_back({
                .cbuf_index = Shader::MESH_DRAW_CBUF_INDEX, .count = 1,
            });
        }
        infos[stage_index] = &program.info;

        const int64_t s67_b0 = S67::Now();   // s67
        auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage, device)};
        if (varying_pack && (program.stage == Shader::Stage::VertexB ||
                             program.stage == Shader::Stage::Fragment)) {
            runtime_info.varying_pack_enabled = true;
            runtime_info.varying_pack = *varying_pack;
        }
        ConvertLegacyToGeneric(program, runtime_info);
        if (program.stage == Shader::Stage::Fragment) {   // nanguard (NPARK 11-21, OoT's whole-picture white flash; Longshot GuardFragmentOutputs): DM_VK_NAN_GUARD=0 off
            static const bool guard = [] { const char* e = std::getenv("DM_VK_NAN_GUARD"); return !e || std::atoi(e) != 0; }();
            if (guard) {
                u32 float_targets = 0;
                for (size_t i = 0; i < 8; ++i)
                    if (runtime_info.color_output_types[i] == Shader::AttributeType::Float) float_targets |= 1u << i;
                const size_t n = Shader::Optimization::GuardFragmentOutputs(program, float_targets);
                static std::atomic<u64> programs{0}, writes{0};
                const u64 p = programs.fetch_add(1, std::memory_order_relaxed) + 1; writes.fetch_add(n, std::memory_order_relaxed);
                if (p == 1 || (p & (p - 1)) == 0)
                    LOG_INFO(Render_Vulkan, "nanguard: {} fragment programs with their float colour outputs guarded (NaN -> 0, +-65000 at most), {} component writes (DM_VK_NAN_GUARD=0 off)",
                             p, writes.load(std::memory_order_relaxed));
            }
        }
        std::vector<u32> code{EmitSPIRV(profile, runtime_info, program, binding)};
        S67::g_front.b += S67::Now() - s67_b0;   // s67: runtime info + SPIR-V
#if NXN1X_DIAGNOSTICS
        if (ShaderRecord::Enabled()) [[unlikely]] {
            ShaderRecord::NoteSpirv('g', hash, index, key.unique_hashes[index], code);
        }
#endif
        device.SaveShader(code);
        { const int64_t s67_c0 = S67::Now(); modules[stage_index] = BuildShader(device, code); S67::g_front.c += S67::Now() - s67_c0; }   // s67: vkCreateShaderModule

        // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
        // is independent and gated only by gpu_log_shader_dumps.
        const bool should_log = GPU::Logging::IsActive();
        const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
        if (should_log || should_dump) {
            static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry", "fragment"};
            const std::string shader_name = fmt::format("shader_{:016x}_{}", key.unique_hashes[index], stage_names[stage_index]);
            if (should_log) {
                const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                    code.size() * sizeof(u32), key.unique_hashes[index]);
                GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
            }
            if (should_dump) {
                GPU::Logging::DumpSpirvShader(key.unique_hashes[index],
                                              std::span<const u32>(code.data(), code.size()));
            }
        }

        if (device.HasDebuggingToolAttached()) {
            const std::string name{fmt::format("Shader {:016x}", key.unique_hashes[index])};
            modules[stage_index].SetObjectNameEXT(name.c_str());
        }
        previous_stage = &program;
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    S67::g_front.e = S67::Now(); S67::g_front.set = true;   // s67: taken by the GraphicsPipeline constructor
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, std::move(modules), infos);

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        env.Dump(hash, key.unique_hashes[index]);
    }
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}
#if NXN1X_DIAGNOSTICS
catch (const std::exception& exception) {
    // A non-shader exception (allocation, length) here would otherwise terminate the process.
    // Log it, dump the guest shaders of this pipeline and skip it, so the cause can be studied.
    auto hash = key.Hash();
    LOG_CRITICAL(Render_Vulkan, "PIPELINE_EXCEPTION graphics {:#016x}: {}", hash, exception.what());
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;
        LOG_CRITICAL(Render_Vulkan, "PIPELINE_EXCEPTION stage={} shader={:#016x}", index,
                     key.unique_hashes[index]);
        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        env.Dump(hash, key.unique_hashes[index]);
    }
    return nullptr;
}
#endif

void PipelineCache::QueueNvnRecipe(std::span<const char> key, std::span<const GenericEnvironment* const> envs) {
    // Snapshot the caller-owned environments before the replay thread releases them.
    auto record = VideoCommon::MakePipelineRecord(key, envs);
    if (record.empty()) return;
    serialization_thread.QueueWork([filename = pipeline_cache_filename, record = std::move(record)] {
        VideoCommon::AppendPipelineRecord(filename, CACHE_VERSION, record);
    });
}

GraphicsPipeline* PipelineCache::NvnGraphicsPipeline(
    const GraphicsPipelineCacheKey& key, std::span<VideoCommon::GenericEnvironment* const> envs) {
    std::scoped_lock lock{nvn_mutex};
    S22KeyNote('N', key);   // s22
    auto [it, is_new] = graphics_cache.try_emplace(key);
    if (!is_new) {
        return it->second.get();
    }
    runtime_graphics_requests.fetch_add(1, std::memory_order_relaxed);
    nvn_graphics_requests.fetch_add(1, std::memory_order_relaxed);
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> shader_envs;
    for (VideoCommon::GenericEnvironment* const env : envs) {
        shader_envs.push_back(env);
    }
    nvn_pools.ReleaseContents();
    try {
        it->second = CreateGraphicsPipeline(nvn_pools, key, shader_envs, nullptr, true);
    } catch (...) {   // a non-shader exception (the Maxwell path's is caught only in diagnostics builds)
        graphics_cache.erase(it);
        throw;
    }
    if (!it->second) {
        graphics_cache.erase(it);
        return nullptr;
    }
    if (!pipeline_cache_filename.empty()) {
        boost::container::static_vector<const VideoCommon::GenericEnvironment*,
                                        Maxwell::MaxShaderProgram>
            env_ptrs(envs.begin(), envs.end());
        if (!std::ranges::all_of(env_ptrs, &GenericEnvironment::CanBeSerialized)) {
            nvn_unserializable_recipes.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "NVN graphics recipe {:#016x} not saved: instruction reads outside captured code", key.Hash());
        }
        QueueNvnRecipe({reinterpret_cast<const char*>(&key), sizeof(key)}, env_ptrs);
    }
    return it->second.get();
}

ComputePipeline* PipelineCache::NvnComputePipeline(const ComputePipelineCacheKey& key,
                                                   VideoCommon::GenericEnvironment& env) {
    std::scoped_lock lock{nvn_mutex};
    auto [it, is_new] = compute_cache.try_emplace(key);
    if (!is_new) {
        return it->second.get();
    }
    runtime_compute_requests.fetch_add(1, std::memory_order_relaxed);
    nvn_compute_requests.fetch_add(1, std::memory_order_relaxed);
    nvn_pools.ReleaseContents();
    try {
        it->second = CreateComputePipeline(nvn_pools, key, env, nullptr, true);
    } catch (...) {
        compute_cache.erase(it);
        throw;
    }
    if (!it->second) {
        compute_cache.erase(it);
        return nullptr;
    }
    if (!pipeline_cache_filename.empty()) {
        if (!env.CanBeSerialized()) {
            nvn_unserializable_recipes.fetch_add(1, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan, "NVN compute recipe {:#016x} not saved: instruction reads outside captured code", key.Hash());
        }
        QueueNvnRecipe({reinterpret_cast<const char*>(&key), sizeof(key)}, std::array<const GenericEnvironment*, 1>{&env});
    }
    return it->second.get();
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    runtime_graphics_requests.fetch_add(1, std::memory_order_relaxed);
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    main_pools.ReleaseContents();
    auto pipeline{
        CreateGraphicsPipeline(main_pools, graphics_key, environments.Span(), nullptr, true)};
#if NXN1X_DIAGNOSTICS
    // b61 SHADER_RECORD: the translator inputs of each stage (shader_record.h).
    if (ShaderRecord::Enabled()) [[unlikely]] {
        const u64 record_hash = graphics_key.Hash();
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (graphics_key.unique_hashes[index] != 0) {
                ShaderRecord::NoteEnv('g', record_hash, index, graphics_key.unique_hashes[index],
                                      environments.envs[index], pipeline != nullptr);
            }
        }
    }
#endif
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key = graphics_key, envs = std::move(environments.envs)] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs, pipeline_cache_filename, CACHE_VERSION);
    });
    return pipeline;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {
    runtime_compute_requests.fetch_add(1, std::memory_order_relaxed);
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native}; // null on Switch 1
    env.SetCachedSize(shader->size_bytes);

    main_pools.ReleaseContents();
    auto pipeline{CreateComputePipeline(main_pools, key, env, nullptr, true)};
#if NXN1X_DIAGNOSTICS
    // b61 SHADER_RECORD: the translator inputs (shader_record.h).
    if (ShaderRecord::Enabled()) [[unlikely]] {
        ShaderRecord::NoteEnv('c', key.Hash(), 0, key.unique_hash, env, pipeline != nullptr);
    }
#endif
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION);
    });
    return pipeline;
}

// ---- bak86: first-invocation stores moved to the host (nxnative docs/reference/shaders.md "A store only a dispatch's first invocation makes",
// dmsh_build.py first_store; Mario Kart World's scenery lists, NPARK 11-16: cacti and trees flicker) ----
// A compute shader's preamble `first = (-local.x == workgroup.x << log2(local.x))` (one invocation of the dispatch, a flat local size) guards constant stores
// into a storage buffer (the reset of an append counter), while the other invocations add to it hundreds of instructions later. On the Switch the first workgroup
// starts ahead of the rest; Adreno, like Apple's GPUs, starts workgroups in no order, so entries appended before the reset were lost, a different few each
// rebuild. The guarded stores leave the shader and the host makes them before each dispatch (gfx_vk_eden.inc eden_first_stores). Matched on the IR Eden makes
// (_xfer/bak85/ir/nvn2_ir_2ce9071c2f22d74f.txt): `If (LogicalNot(INotEqual(INeg32(LocalInvocationId.x), ShiftLeftLogical32(WorkgroupId.x, #k))))` bodies holding only
// address arithmetic and `WriteStorage32 #b, (cbuf address [+ #c]) - (cbuf address & ~3), value`, value a constant or a Phi taking a constant from a guarded
// body. Anything else under the guard leaves the shader as it is (logged). Only for the shaders the IR was read for, on NVN2 host boots (NXN1X_NVN2_HOST, set at
// the boot's path decision); DM_NVN2_FIRST_STORE=0 off. 0823b08c2027a38d (2D guard, an address through a Phi) and 649e36162cad9e79 (refused by the record) are not taken
namespace {
std::mutex g_fs_mutex;
std::unordered_map<const ComputePipeline*, std::vector<NvnFirstStore>> g_fs_map;

bool FsOn() {
    static const bool on = [] {
        const char* const host = std::getenv("NXN1X_NVN2_HOST");
        const char* const e = std::getenv("DM_NVN2_FIRST_STORE");
        return host && *host == '1' && !(e && *e == '0');
    }();
    return on;
}
bool FsWanted(u64 unique_hash) {
    return unique_hash == 0xcb5d206e5e9364ffULL     // 2ce9071c2f22d74f (the IR read: four guarded stores)
        || unique_hash == 0x528915b19423ac99ULL;    // bak87: 0823b08c2027a38d (local 8x8: the 2D guard, one store through a Phi of the address)
}
Shader::IR::Inst* FsInst(const Shader::IR::Value& v, Shader::IR::Opcode op) {
    const Shader::IR::Value r = v.Resolve();
    if (r.IsImmediate()) {
        return nullptr;
    }
    Shader::IR::Inst* const i = r.InstRecursive();
    return i && i->GetOpcode() == op ? i : nullptr;
}
bool FsImm(const Shader::IR::Value& v, u32& out) {
    const Shader::IR::Value r = v.Resolve();
    if (!r.IsImmediate() || r.Type() != Shader::IR::Type::U32) {
        return false;
    }
    out = r.U32();
    return true;
}
bool FsExtract(const Shader::IR::Value& v, Shader::IR::Opcode src, u32 index) {
    using Shader::IR::Opcode;
    Shader::IR::Inst* const e = FsInst(v, Opcode::CompositeExtractU32x3);
    u32 i{};
    return e && FsInst(e->Arg(0), src) && FsImm(e->Arg(1), i) && i == index;
}
// one axis's test: (-local.c == workgroup.c << log2(local size c)), as IEqual or as LogicalNot(INotEqual); sets the bit of the axis it holds for
bool FsAxis(const Shader::IR::Value& v, const std::array<u32, 3>& wg, u32& axes) {
    using Shader::IR::Opcode;
    Shader::IR::Inst* eq = FsInst(v, Opcode::IEqual);
    if (!eq) {
        Shader::IR::Inst* const n = FsInst(v, Opcode::LogicalNot);
        eq = n ? FsInst(n->Arg(0), Opcode::INotEqual) : nullptr;
    }
    if (!eq) {
        return false;
    }
    for (int s = 0; s < 2; ++s) {
        Shader::IR::Inst* const neg = FsInst(eq->Arg(s), Opcode::INeg32);
        Shader::IR::Inst* const shl = FsInst(eq->Arg(1 - s), Opcode::ShiftLeftLogical32);
        u32 k{};
        if (!neg || !shl || !FsImm(shl->Arg(1), k) || k >= 31) {
            continue;
        }
        for (u32 c = 0; c < 2; ++c) {
            if (FsExtract(neg->Arg(0), Opcode::LocalInvocationId, c) && FsExtract(shl->Arg(0), Opcode::WorkgroupId, c)) {
                if ((1u << k) != wg[c]) {
                    return false;
                }
                axes |= 1u << c;
                return true;
            }
        }
    }
    return false;
}
// workgroup.z == 0
bool FsZeroZ(const Shader::IR::Value& v) {
    using Shader::IR::Opcode;
    Shader::IR::Inst* const eq = FsInst(v, Opcode::IEqual);
    if (!eq) {
        return false;
    }
    for (int s = 0; s < 2; ++s) {
        u32 z{};
        if (FsExtract(eq->Arg(s), Opcode::WorkgroupId, 2) && FsImm(eq->Arg(1 - s), z) && z == 0) {
            return true;
        }
    }
    return false;
}
// the first invocation of a dispatch: a flat local size, the x test (dmsh_build.py FIRST_STORE); or local x by y, the x and y tests and workgroup z zero, joined
// by LogicalAnd (FIRST_STORE_2D, bak87). Every term must be one of those, each axis once
bool FsGuard(const Shader::IR::Value& cond, const std::array<u32, 3>& wg) {
    using Shader::IR::Opcode;
    // bak87: the If node's condition is ConditionRef(expression) (structured_control_flow.cpp: ir.ConditionRef(VisitExpr(...))); bak86 matched the bare
    // expression only and left 2ce9071c2f22d74f as it was ("no first-invocation guard", bak86 log 13.197 s)
    Shader::IR::Inst* const ref = FsInst(cond, Opcode::ConditionRef);
    std::vector<Shader::IR::Value> terms{ref ? ref->Arg(0) : cond}, leaves;
    while (!terms.empty() && leaves.size() < 8) {
        const Shader::IR::Value v = terms.back();
        terms.pop_back();
        if (Shader::IR::Inst* const a = FsInst(v, Opcode::LogicalAnd)) {
            terms.push_back(a->Arg(0));
            terms.push_back(a->Arg(1));
        } else {
            leaves.push_back(v);
        }
    }
    if (!terms.empty()) {
        return false;
    }
    u32 axes = 0;
    bool zero_z = false;
    for (const auto& l : leaves) {
        u32 before = axes;
        if (FsAxis(l, wg, axes) && axes != before) {
            continue;
        }
        if (!zero_z && FsZeroZ(l)) {
            zero_z = true;
            continue;
        }
        return false;
    }
    if (wg[2] != 1) {
        return false;
    }
    if (wg[1] == 1) {
        return axes == 1u && !zero_z;   // the 1D form has the x test alone
    }
    return axes == 3u && zero_z;
}
// the 64-bit address's low word read from a constant buffer: CompositeExtractU32x2(GetCbufU32x2(#ci, #co), #0), or GetCbufU32(#ci, #co) (bak87), or a Phi
// whose inputs are each undefined or one of those, all of one #ci / #co (0823b08c2027a38d reads it in a guarded block: Phi [Undef, GetCbufU32 #0, #64])
bool FsAddrLo(const Shader::IR::Value& v, u32& ci, u32& co, int depth = 0) {
    using Shader::IR::Opcode;
    if (Shader::IR::Inst* const e = FsInst(v, Opcode::CompositeExtractU32x2)) {
        u32 i{};
        if (!FsImm(e->Arg(1), i) || i != 0) {
            return false;
        }
        Shader::IR::Inst* const g = FsInst(e->Arg(0), Opcode::GetCbufU32x2);
        return g && FsImm(g->Arg(0), ci) && FsImm(g->Arg(1), co);
    }
    if (Shader::IR::Inst* const g = FsInst(v, Opcode::GetCbufU32)) {
        return FsImm(g->Arg(0), ci) && FsImm(g->Arg(1), co);
    }
    Shader::IR::Inst* const phi = FsInst(v, Opcode::Phi);
    if (!phi || depth > 0) {
        return false;
    }
    bool any = false;
    for (size_t a = 0; a < phi->NumArgs(); ++a) {
        if (FsInst(phi->Arg(a), Opcode::UndefU32)) {
            continue;
        }
        u32 i{}, o{};
        if (!FsAddrLo(phi->Arg(a), i, o, depth + 1) || (any && (i != ci || o != co))) {
            return false;
        }
        ci = i;
        co = o;
        any = true;
    }
    return any;
}
// Moves the stores out of the program; returns them (empty: the program is unchanged), why says what refused it
std::vector<NvnFirstStore> FirstStorePass(Shader::IR::Program& program, std::string& why) {
    using Shader::IR::AbstractSyntaxNode;
    using Shader::IR::Opcode;
    std::vector<NvnFirstStore> out;
    std::vector<Shader::IR::Inst*> moved;
    if (program.stage != Shader::Stage::Compute || program.workgroup_size[2] != 1) {
        why = "not a compute shader with a flat or 2D local size";
        return {};
    }
    const std::array<u32, 3> wg = program.workgroup_size;
    std::vector<const Shader::IR::Block*> bodies;
    auto& list = program.syntax_list;
    for (size_t i = 0; i < list.size(); ++i) {   // the guarded bodies
        if (list[i].type != AbstractSyntaxNode::Type::If || !FsGuard(list[i].data.if_node.cond, wg)) {
            continue;
        }
        int depth = 0;
        for (size_t j = i + 1; j < list.size(); ++j) {
            const auto t = list[j].type;
            if (t == AbstractSyntaxNode::Type::If) { ++depth; }
            if (t == AbstractSyntaxNode::Type::EndIf) { if (depth-- == 0) { break; } }
            if (t == AbstractSyntaxNode::Type::Loop || t == AbstractSyntaxNode::Type::Repeat || t == AbstractSyntaxNode::Type::Break || t == AbstractSyntaxNode::Type::Return) {
                why = "a loop, break or return under the guard";
                return {};
            }
            if (depth > 0 && t == AbstractSyntaxNode::Type::If) {
                why = "an if nested under the guard";
                return {};
            }
            if (t == AbstractSyntaxNode::Type::Block) { bodies.push_back(list[j].data.block); }
        }
    }
    if (bodies.empty()) {
        why = "no first-invocation guard";
        return {};
    }
    const auto in_bodies = [&](const Shader::IR::Block* b) { return std::find(bodies.begin(), bodies.end(), b) != bodies.end(); };
    const auto& sb = program.info.storage_buffers_descriptors;
    for (const Shader::IR::Block* body : bodies) {
        for (const Shader::IR::Inst& inst_c : body->Instructions()) {
            Shader::IR::Inst& inst = const_cast<Shader::IR::Inst&>(inst_c);
            switch (inst.GetOpcode()) {
            case Opcode::Identity: case Opcode::IAdd32: case Opcode::ISub32: case Opcode::BitwiseAnd32: case Opcode::GetCbufU32: case Opcode::GetCbufU32x2:
            case Opcode::CompositeExtractU32x2: case Opcode::CompositeExtractU32x3: case Opcode::LogicalNot: case Opcode::INotEqual: case Opcode::ConditionRef:
            case Opcode::Phi: case Opcode::Void: case Opcode::UndefU32:
                continue;
            case Opcode::WriteStorage32:
                break;
            default:
                why = fmt::format("{} under the guard", Shader::IR::NameOf(inst.GetOpcode()));
                return {};
            }
            u32 b{};
            if (!FsImm(inst.Arg(0), b) || b >= sb.size()) { why = "a store to a storage buffer not named by an immediate"; return {}; }
            // offset = (address.lo [+ #c]) - (address.lo & ~3), the address from the buffer's own constant buffer words
            Shader::IR::Inst* const sub = FsInst(inst.Arg(1), Opcode::ISub32);
            Shader::IR::Inst* const msk = sub ? FsInst(sub->Arg(1), Opcode::BitwiseAnd32) : nullptr;
            u32 mask{}, mci{}, mco{};
            Shader::IR::Inst* const mg = msk ? FsInst(msk->Arg(0), Opcode::GetCbufU32) : nullptr;
            if (!mg || !FsImm(msk->Arg(1), mask) || mask != 0xfffffffcu || !FsImm(mg->Arg(0), mci) || !FsImm(mg->Arg(1), mco)) { why = "a store offset not of the aligned-address form"; return {}; }
            u32 ci{}, co{}, c = 0;
            if (!FsAddrLo(sub->Arg(0), ci, co)) {
                Shader::IR::Inst* const add = FsInst(sub->Arg(0), Opcode::IAdd32);
                if (!add || !((FsAddrLo(add->Arg(0), ci, co) && FsImm(add->Arg(1), c)) || (FsAddrLo(add->Arg(1), ci, co) && FsImm(add->Arg(0), c)))) { why = "a store address not a constant buffer address plus a constant"; return {}; }
            }
            if (ci != mci || co != mco || sb[b].cbuf_index != ci || sb[b].cbuf_offset != co || sb[b].count != 1 || (c & 3)) { why = "a store whose address is not its storage buffer's"; return {}; }
            u32 value{};
            if (!FsImm(inst.Arg(2), value)) {   // a Phi that takes a constant from a guarded body (the first invocation's value)
                Shader::IR::Inst* const phi = FsInst(inst.Arg(2), Opcode::Phi);
                bool ok = false;
                for (size_t a = 0; phi && a < phi->NumArgs(); ++a) {
                    u32 v{};
                    if (in_bodies(phi->PhiBlock(a)) && FsImm(phi->Arg(a), v)) { if (ok && v != value) { ok = false; break; } value = v; ok = true; }
                }
                if (!ok) { why = "a stored value that is neither a constant nor a guarded constant"; return {}; }
            }
            if (out.size() == 4) { why = "more than 4 stores"; return {}; }
            out.push_back({ci, co, c, value});
            moved.push_back(&inst);
        }
    }
    if (out.empty()) {
        why = "no store under the guard";
        return {};
    }
    for (Shader::IR::Inst* const inst : moved) {
        inst->Invalidate();
    }
    return out;
}
} // anonymous namespace

std::vector<NvnFirstStore> NvnFirstStoresOf(const ComputePipeline* pipeline) {
    std::scoped_lock lock{g_fs_mutex};
    const auto it = g_fs_map.find(pipeline);
    return it == g_fs_map.end() ? std::vector<NvnFirstStore>{} : it->second;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return nullptr;
    }

    LOG_INFO(Render_Vulkan, "{:#016x}", hash);
    // pcache1 (log only): where a compute pipeline's time goes (OoT 11-21: new compute shaders 3.1 s apiece, one after another, 9.3 s without a frame).
    // "s67c:" with the translation, SPIR-V and layout times here when they pass 50 ms, the driver's build in vk_compute_pipeline.cpp; SLOW past 1 s
    using S67Clock = std::chrono::steady_clock;
    const auto s67c_ms = [](S67Clock::time_point a, S67Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto s67c_t0 = S67Clock::now();

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    const auto s67c_t1 = S67Clock::now();   // pcache1: control flow
    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    const auto s67c_t2 = S67Clock::now();   // pcache1: translated
    std::vector<NvnFirstStore> first_stores;   // bak86
    if (FsOn() && FsWanted(key.unique_hash)) {
        std::string why;
        first_stores = FirstStorePass(program, why);
        if (first_stores.empty()) {
            LOG_ERROR(Render_Vulkan, "[RED] first_store: compute shader {:016x}: left as it is ({}): its first-invocation race stays", key.unique_hash, why);   // bak87: RED
        } else {
            LOG_INFO(Render_Vulkan, "first_store: compute shader {:016x}: {} first-invocation stores moved to the host", key.unique_hash, first_stores.size());
        }
    }
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    if (max_shared_memory != 0 && program.shared_memory_size > max_shared_memory) {
        // More shared memory than the device has (MKW asks for 48 KB, Adreno X1 has 32 KB): no
        // workgroup can run. Turnip's compiler ends the process on such a shader (ir3.c "impossible
        // to have enough concurrent waves"), so it never reaches the driver; its dispatches are
        // skipped like those of any missing compute pipeline.
        size_t tensor_ops{};
        for (const auto* block : program.blocks) {
            for (const auto& inst : block->Instructions()) {
                tensor_ops += inst.GetOpcode() == Shader::IR::Opcode::TensorMulAdd ? 1 : 0;
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} needs {} bytes of shared memory, the device has {}: "
                    "skipped (workgroup {}x{}x{}, {} tensor multiply-adds)",
                    key.unique_hash, program.shared_memory_size, max_shared_memory,
                    program.workgroup_size[0], program.workgroup_size[1],
                    program.workgroup_size[2], tensor_ops);
        return nullptr;
    }
    const auto s67c_t3 = S67Clock::now();   // pcache1: before SPIR-V
    std::vector<u32> code{EmitSPIRV(profile, program)};
#if NXN1X_DIAGNOSTICS
    if (ShaderRecord::Enabled()) [[unlikely]] {
        ShaderRecord::NoteSpirv('c', hash, 0, key.unique_hash, code);
    }
#endif
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};
    const auto s67c_t4 = S67Clock::now();   // pcache1: SPIR-V emitted, module made

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    auto made = std::make_unique<ComputePipeline>(device, scheduler, vulkan_pipeline_cache, descriptor_pool,
                                             guest_descriptor_queue, descriptor_buffer_ring,
                                             thread_worker, statistics,
                                             &shader_notify, program.info, std::move(spv_module),
                                             key.unique_hash);
    {   // pcache1: the host side of this compute pipeline (the driver's part is logged where it builds: inline here, or on a worker)
        const auto s67c_t5 = S67Clock::now();
        const double total = s67c_ms(s67c_t0, s67c_t5);
        if (total >= 50.0) {
            size_t insts = 0;
            for (const auto* block : program.blocks) insts += block->Instructions().size();
            LOG_INFO(Render_Vulkan,
                     "s67c: {}compute {:016x} shader {:016x}: flow {:.1f} translate {:.1f} passes {:.1f} spirv+module {:.1f} layout{} {:.1f} ms, total {:.1f} ms "
                     "(IR {} blocks {} instructions, SPIR-V {} KB, workgroup {}x{}x{}, shared {} B)",
                     total >= 1000.0 ? "SLOW " : "", hash, key.unique_hash, s67c_ms(s67c_t0, s67c_t1), s67c_ms(s67c_t1, s67c_t2),
                     s67c_ms(s67c_t2, s67c_t3), s67c_ms(s67c_t3, s67c_t4), build_in_parallel ? "" : "+driver", s67c_ms(s67c_t4, s67c_t5), total,
                     program.blocks.size(), insts, code.size() * sizeof(u32) / 1024, program.workgroup_size[0], program.workgroup_size[1],
                     program.workgroup_size[2], program.shared_memory_size);
        }
    }
    if (!first_stores.empty()) {   // bak86: the host makes them before each dispatch of this pipeline
        std::scoped_lock lock{g_fs_mutex};
        g_fs_map[made.get()] = std::move(first_stores);
    }
    return made;

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    return nullptr;
}

bool PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) {
    if (!pipeline_cache || filename.empty()) return false;
    size_t size{};
    if (pipeline_cache.Read(&size, nullptr) != VK_SUCCESS || !size) return false;
    std::vector<char> data(size);
    const auto result = pipeline_cache.Read(&size, data.data());
    // A concurrent pipeline build can grow the cache between queries. Keep the
    // previous complete snapshot in that case and retry at the next save.
    if (result != VK_SUCCESS || !size) return false;
    auto tmp = filename;
    tmp += "." + Common::UUID::MakeRandom().FormattedString() + ".tmp";
    try {
        std::ofstream file;
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.open(tmp, std::ios::binary);
        file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
            .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version))
            .write(data.data(), size);
        file.close();
        std::filesystem::rename(tmp, filename);
        LOG_INFO(Render_Vulkan, "Vulkan driver cache saved: {} bytes at {}", size,
                 Common::FS::PathToUTF8String(filename));
        return true;
    } catch (const std::system_error& error) {
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);
        LOG_ERROR(Common_Filesystem, "Failed to save Vulkan driver cache: {}", error.what());
        return false;
    }
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        if (filename.empty()) {
            return create_pipeline_cache(0, nullptr);
        }
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        static constexpr size_t header_size = magic_number.size() + sizeof(cache_version);
        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        auto cache = create_pipeline_cache(cache_size, cache_data.data());
        driver_cache_bytes_supplied = cache_size;
        LOG_INFO(Render_Vulkan, "Loaded Vulkan driver pipeline cache: {} ({} bytes supplied)",
                 Common::FS::PathToUTF8String(filename), cache_size);
        return cache;

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
