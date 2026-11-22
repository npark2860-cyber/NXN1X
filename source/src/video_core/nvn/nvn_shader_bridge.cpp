// SPDX-License-Identifier: GPL-3.0-or-later
// See nvn_shader_bridge.h.

#include <algorithm>
#include <array>
#include <atomic>   // S1c (parallel recording): nvn_eden_shader::has_code
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <boost/container/static_vector.hpp>

#include <cstdio>   // NVN2 compute control dump
#include <chrono>   // NVN2 bridge coverage
#include <string>
#include <filesystem>
#include <set>
#include "common/cityhash.h"
#include "common/fs/path_util.h"   // NVN2 compute control dump
#include "common/logging.h"
#include "common/settings.h"   // NVN2
#include "video_core/nvn/nvn_shader_bridge.h"
#include "video_core/nvn/nvn_pipeline_key.h"
#include "video_core/nvn/nvn_shader_env.h"
#include "video_core/renderer_vulkan/nvn_bridge_device.h"   // lane A: Vulkan::NvnBridge::Get*
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "nvnh_memory_guard.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/program.h"            // bak85: nvn_eden_dump_ir
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"

using VideoCommon::Nvn::Control::CODE_SIZE;
using VideoCommon::Nvn::Control::KEY_SIZE;
using VideoCommon::Nvn::NvnEnvironment;
using VideoCommon::Nvn::NvnShaderSource;

// S1c (parallel recording): a shader's code is set once and then immutable. CopyCode writes source.code, nvn_stage, hash and lowered under lower_mutex and
// publishes them with has_code (release); every reader of the code (the environments' constructors copy it, C1(), the stage) first loads has_code (acquire), so a
// shader whose code is being set on one thread is "no code yet" on another, never half-written. A second CopyCode with other bytes is refused: an environment may
// be copying the published code on another thread (pipeline creation, Eden's async builds), and pipelines made from it stay in Eden's cache under its hash
struct nvn_eden_shader {
    NvnShaderSource source;
    std::atomic<bool> has_code{};
    bool detect_stage{};
    std::mutex lower_mutex;   // NVN2: the lowering, made once by the first environment
    std::shared_ptr<const VideoCommon::LoweredNativeCode> lowered;
    size_t analyzed_size{};
    std::optional<u64> hash;   // computed once, from the code copy
};

namespace {
constexpr u32 MAX_CODE = 1u << 24;   // as the host's keys_hash

// Call only from an exception handler. Translation errors retain their existing
// fallback, but a memory stop must reach the outer native boundary so its C
// locks are released and incomplete replay cannot continue as a missing shader.
void RethrowNativeMemoryFailure() {
    nvnh_guard_check();
    if (!nvnh_guard_enabled()) {
        return;
    }
    try {
        throw;
    } catch (const Vulkan::vk::Exception& e) {
        const VkResult result = e.GetResult();
        if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
            nvnh_memory_latch(NVNH_STOP_DRIVER, result, "native shader bridge allocation", 0);
        }
    } catch (const std::bad_alloc&) {
        nvnh_memory_latch(NVNH_STOP_INTERNAL, 0, "native shader bridge host allocation", 0);
    } catch (...) {
    }
    nvnh_guard_check();
}

// host dm/gfx/tools/shader_split.py glslc_stage / sph_stage
int DetectStage(const NvnShaderSource& src) {
    const auto& c = src.code;
    if (c.size() < 4) {
        return -1;
    }
    if (src.nvn2) {
        // host_nx2 shader_split.py:165-167 nvn2_stage: compute, task and mesh by the control section's stage byte (+0x56: 5, 6, 7), else the SPH at
        // the code area's start (no code header; sph_stage, shader_split.py:74-80: shader type in word 0 bits 10..13)
        const u8 st = src.control[VideoCommon::Nvn::Control2::STAGE];
        if (st == 5 || st == 6 || st == 7) {
            return st;
        }
        u32 w0;
        std::memcpy(&w0, c.data(), 4);
        switch ((w0 >> 10) & 0xf) {
        case 0: return (w0 & 0x1f) == 1 ? 0 : -1;
        case 1: return 0;
        case 2: return 3;
        case 3: return 4;
        case 4: return 2;
        case 5: return 1;
        default: return -1;
        }
    }
    u32 magic;
    std::memcpy(&magic, c.data(), 4);
    if (magic == 0x12345679u) {
        return 5;
    }
    if (magic != 0x12345678u) {
        const bool headerless_cs = src.ControlWord(VideoCommon::Nvn::Control::WORKGROUP) &&
                                   src.ControlWord(VideoCommon::Nvn::Control::WORKGROUP + 4) &&
                                   src.ControlWord(VideoCommon::Nvn::Control::WORKGROUP + 8);
        return headerless_cs ? 5 : -1;
    }
    if (c.size() < 0x34) {
        return -1;
    }
    u32 w;
    std::memcpy(&w, c.data() + 0x30, 4);
    switch ((w >> 10) & 0xf) {
    case 0: return (w & 0x1f) == 1 ? 0 : -1;   // Catherine: Full Body's vertex shaders
    case 1: return 0;
    case 2: return 3;
    case 3: return 4;
    case 4: return 2;
    case 5: return 1;
    default: return -1;
    }
}

bool CopyCode(nvn_eden_shader& s, const void* code, u32 code_size) {
    if (!code || code_size == 0 || code_size >= MAX_CODE) {
        return false;
    }
    std::scoped_lock lock{s.lower_mutex};   // S1c (parallel recording): two setters of one shader one after the other; the readers wait for has_code
    if (s.has_code.load(std::memory_order_acquire)) {   // published: immutable (struct nvn_eden_shader)
        if (s.source.code.size() == code_size && std::memcmp(s.source.code.data(), code, code_size) == 0) {
            return true;
        }
        LOG_ERROR(Shader, "NVN shader (stage {}): new code for a shader whose code is in use: refused (a new shader object takes it)", s.source.nvn_stage);
        return false;
    }
    s.source.code.assign(static_cast<const u8*>(code), static_cast<const u8*>(code) + code_size);
    s.hash.reset();
    s.lowered.reset();
    if (s.detect_stage) {
        s.source.nvn_stage = DetectStage(s.source);
    }
    s.has_code.store(true, std::memory_order_release);   // S1c: last, after the code and its stage (was before the stage was detected)
    return true;
}
// NVN2: one lowering per shader, shared by its environments (the Maxwell path keeps it in ShaderInfo::lowered_native)
std::shared_ptr<const VideoCommon::LoweredNativeCode> Lowered(nvn_eden_shader& s) {
    if (!s.source.nvn2) {
        return {};
    }
    std::scoped_lock lock{s.lower_mutex};
    return s.lowered;
}

void KeepLowered(nvn_eden_shader& s, const NvnEnvironment& env) {
    if (!s.source.nvn2) {
        return;
    }
    std::scoped_lock lock{s.lower_mutex};
    if (!s.lowered) {
        s.lowered = env.LoweredNative();
    }
}
// Graphics state changes can build many pipelines around the same immutable
// program. Keep the scan result on the shader, but give each environment its
// own code window and descriptor queries, just as on a cold build.
u64 PrepareProgram(nvn_eden_shader& shader, NvnEnvironment& env) {
    size_t size;
    u64 hash;
    {
        std::scoped_lock lock{shader.lower_mutex};
        if (!shader.hash) {
            shader.hash = env.Analyze().value_or(0);
            if (*shader.hash) {
                if (shader.source.nvn_stage == 6 || shader.source.nvn_stage == 7) {
                    const auto& registers = env.MeshRegisters();
                    const auto& primitive = env.MeshPrimitiveHeader();
                    const u64 configuration = Common::CityHash64WithSeed(
                        reinterpret_cast<const char*>(registers.data()), sizeof(registers), *shader.hash);
                    shader.hash = Common::CityHash64WithSeed(
                        reinterpret_cast<const char*>(&primitive), sizeof(primitive),
                        configuration ^ static_cast<u64>(shader.source.nvn_stage));
                }
                shader.analyzed_size = env.CachedSizeBytes();
            } else {
                LOG_ERROR(Shader, "NVN shader (stage {}): no code hash", shader.source.nvn_stage);
            }
        }
        hash = *shader.hash;
        size = shader.analyzed_size;
    }
    if (hash) env.SetCachedSize(size);
    return hash;
}
} // namespace

extern "C" {

nvn_eden_shader* nvn_eden_shader_new(const void* control, uint32_t control_size, const void* code,
                                     uint32_t code_size, int nvn_stage) {
    // NVN2 control sections only for a Switch 2 program (Control2; the host passes the key's 0x48 bytes as control_size, hle_nvn.c nvn_ctl_lens)
    const bool nvn2 = Settings::CurrentProgramIsSwitch2() && VideoCommon::Nvn::IsNvn2Control(control, control_size);
    if (!control || (!nvn2 && control_size <= CODE_SIZE + 4) || nvn_stage < -1 || nvn_stage > (nvn2 ? 7 : 5)) {
        LOG_ERROR(Shader, "NVN shader: control section {} bytes, stage {}: not used", control_size,
                  nvn_stage);
        return nullptr;
    }
    auto s = std::make_unique<nvn_eden_shader>();
    s->source.nvn2 = nvn2;
    s->source.control_size = nvn2 ? VideoCommon::Nvn::Control2::READ_SIZE : (std::min<u32>)(control_size, KEY_SIZE);
    std::memcpy(s->source.control.data(), control, s->source.control_size);
    s->source.nvn_stage = nvn_stage;
    s->detect_stage = nvn_stage < 0;
    if (code) {
        CopyCode(*s, code, code_size);
    }
    return s.release();
}

int nvn_eden_shader_set_code(nvn_eden_shader* s, const void* code, uint32_t code_size) {
    return s && CopyCode(*s, code, code_size) ? 1 : 0;
}

int nvn_eden_shader_stage(const nvn_eden_shader* s) {
    return s && s->has_code ? s->source.nvn_stage : -1;
}

void nvn_eden_shader_free(nvn_eden_shader* s) {
    delete s;
}

// Ordinary translation errors report a missing shader/pipeline. Memory failures
// in an armed native rendering call instead unwind to its session error boundary.
uint64_t nvn_eden_shader_hash(nvn_eden_shader* s) try {
    if (!s || !s->has_code || s->source.nvn_stage < 0) {
        return 0;
    }
    {
        std::scoped_lock lock{s->lower_mutex};
        if (s->hash) return *s->hash;
    }
    NvnEnvironment env{s->source, nullptr, Lowered(*s)};
    KeepLowered(*s, env);
    return PrepareProgram(*s, env);
} catch (const std::exception& e) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: exception: {}", "shader hash", e.what());
    return 0;
} catch (...) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: unknown exception", "shader hash");
    return 0;
}

int nvn_eden_shader_nvn2_c0(const nvn_eden_shader* s, uint32_t out[8]) {
    namespace C2 = VideoCommon::Nvn::Control2;
    if (!s || !s->has_code || !s->source.nvn2 || s->source.nvn_stage != 5 || s->source.control_size < C2::READ_SIZE) {
        return 0;
    }
    const auto& c = s->source.control;
    out[0] = u32{c[C2::PUSH_WORDS]} * 4;
    out[1] = c[C2::C0_UBOS];
    out[2] = c[C2::C0_SSBOS];
    out[3] = c[C2::C0_TEXTURES];
    out[4] = c[C2::C0_SEP_TEXTURES];
    out[5] = c[C2::C0_SEP_SAMPLERS];
    out[6] = c[C2::C0_IMAGES];
    out[7] = u32{c[C2::C0_SIZE]} | (u32{c[C2::C0_SIZE + 1]} << 8);
    return 1;
}

uint64_t nvn_eden_shader_c1_hash(const nvn_eden_shader* s) {
    if (!s) {
        return 0;
    }
    const auto c1 = s->has_code.load(std::memory_order_acquire) ? s->source.C1() : std::span<const u8>{};   // S1c (parallel recording): no code yet reads as none (struct nvn_eden_shader)
    return Common::CityHash64(reinterpret_cast<const char*>(c1.data()), c1.size());
}

// ---- pipelines ----

namespace {

struct PipelineRecord {
    Vulkan::GraphicsPipeline* graphics{};
    Vulkan::ComputePipeline* compute{};
    std::vector<nvn_eden_bind> binds;
    u32 num_entries{};
};

std::mutex g_pipe_mutex;
uint64_t g_records_graphics, g_records_compute;   // (under g_pipe_mutex) nvn_eden_pipeline_counts
u32 DynOf(const Vulkan::FixedPipelineState& s);   // B4b, below
ankerl::unordered_dense::map<const void*, std::unique_ptr<PipelineRecord>> g_records;

// Eden's stage index (vk_graphics_pipeline NUM_STAGES order) -> NVN stage
constexpr int NVN_STAGE_OF_EDEN[5] = {0, 3, 4, 2, 1};
// NVN stage -> GraphicsPipelineCacheKey::unique_hashes index (Maxwell program order)
constexpr int PROGRAM_OF_NVN[5] = {1, 5, 4, 2, 3};

void Dim(Shader::TextureType type, nvn_eden_bind& b) {
    using T = Shader::TextureType;
    switch (type) {
    case T::Color1D: b.dim = 0; b.arrayed = 0; break;
    case T::ColorArray1D: b.dim = 0; b.arrayed = 1; break;
    case T::ColorArray2D: b.dim = 1; b.arrayed = 1; break;
    case T::Color3D: b.dim = 2; b.arrayed = 0; break;
    case T::ColorCube: b.dim = 3; b.arrayed = 0; break;
    case T::ColorArrayCube: b.dim = 3; b.arrayed = 1; break;
    default: b.dim = 1; b.arrayed = 0; break;   // Color2D, Color2DRect, Buffer
    }
}

// NVN2 diagnostic (bridge coverage): descriptor features Eden's recompiler can give a shader that this bridge or the NVN host does not carry (or, marked, carries
// since a recent change). Counted when a pipeline is first recorded (no per-draw work); one INFO line the first time each is seen, and a summary when counts changed and
// 30 s passed. Switch 2 programs only
struct BridgeFeature { const char* name; bool handled; };
constexpr BridgeFeature BRIDGE_FEATURES[] = {
    {"texture handle read from a storage buffer table (handle_program)", true},
    {"texture descriptor array indexed per invocation (is_descriptor_array)", false},
    {"texture handle as a pointer word (handle_is_pointer_word)", true},
    {"texture buffer handle from a table (handle_program)", true},
    {"storage texel buffer handle from a table (handle_program)", true},
    {"image handle from a table (handle_program)", true},
    {"byte-addressed image (is_byte_addressed)", false},
    {"storage buffer whose address is read from memory (indirect_offset)", true},
    {"draw-time handle programs (Info::handle_programs)", true},
    {"texture array (count > 1)", true},
};
constexpr size_t NUM_BRIDGE_FEATURES = sizeof(BRIDGE_FEATURES) / sizeof(BRIDGE_FEATURES[0]);
std::array<u64, NUM_BRIDGE_FEATURES> g_cov_counts{};
std::array<u64, NUM_BRIDGE_FEATURES> g_cov_logged{};
std::chrono::steady_clock::time_point g_cov_last{};
void BridgeCoverage(const Shader::Info& info, int nvn_stage, const void* pipeline) {   // (under g_pipe_mutex)
    if (!Settings::CurrentProgramIsSwitch2()) {
        return;
    }
    if (info.storage_buffers_descriptors.size() > 64) {
        static size_t highest_count = 64; // Record holds g_pipe_mutex.
        if (info.storage_buffers_descriptors.size() > highest_count) {
            highest_count = info.storage_buffers_descriptors.size();
            LOG_INFO(Render_Vulkan, "NVN2 pipeline recorded with {} storage buffers in stage {}", highest_count, nvn_stage);
        }
    }
    std::array<u32, NUM_BRIDGE_FEATURES> n{};
    for (const auto& d : info.texture_descriptors) {
        n[0] += d.handle_program != ~0U; n[1] += d.is_descriptor_array; n[2] += d.handle_is_pointer_word; n[9] += d.count > 1;
    }
    for (const auto& d : info.texture_buffer_descriptors) { n[3] += d.handle_program != ~0U; }
    for (const auto& d : info.image_buffer_descriptors) { n[4] += d.handle_program != ~0U; }
    for (const auto& d : info.image_descriptors) { n[5] += d.handle_program != ~0U; n[6] += d.is_byte_addressed; }
    for (const auto& d : info.storage_buffers_descriptors) { n[7] += d.indirect_offset != ~0U; }
    n[8] = static_cast<u32>(info.handle_programs.size());
    bool changed = false;
    for (size_t f = 0; f < NUM_BRIDGE_FEATURES; ++f) {
        if (n[f] == 0) {
            continue;
        }
        changed = true;
        if (g_cov_counts[f]++ == 0) {
            LOG_INFO(Render_Vulkan, "NVN2 bridge: pipeline {} stage {} uses {} x{} ({})", pipeline, nvn_stage, BRIDGE_FEATURES[f].name, n[f],
                     BRIDGE_FEATURES[f].handled ? "handled by the host" : "not handled by the host");
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (changed && g_cov_counts != g_cov_logged && now - g_cov_last >= std::chrono::seconds{30}) {
        g_cov_last = now;
        g_cov_logged = g_cov_counts;
        std::string s;
        for (size_t f = 0; f < NUM_BRIDGE_FEATURES; ++f) {
            if (g_cov_counts[f] != 0) {
                s += fmt::format("{}{} {}", s.empty() ? "" : "; ", BRIDGE_FEATURES[f].name, g_cov_counts[f]);
            }
        }
        LOG_INFO(Render_Vulkan, "NVN2 bridge: shader stages using each feature so far: {}", s);
    }
}

// The binds of one stage in DescriptorLayoutBuilder::Add order (pipeline_helper.h)
void AddStage(const Shader::Info& info, int nvn_stage, std::vector<nvn_eden_bind>& out, u32& entry) {
    const auto push = [&](nvn_eden_bind b) {
        b.nvn_stage = static_cast<u8>(nvn_stage);
        b.first_entry = entry;
        entry += b.count;
        out.push_back(b);
    };
    for (const auto& d : info.constant_buffer_descriptors) {
        push({.kind = NVN_EB_UBO, .count = d.count, .cbuf = d.index});
    }
    for (const auto& d : info.storage_buffers_descriptors) {
        push({.kind = NVN_EB_SSBO, .is_written = d.is_written, .count = d.count,
              .cbuf = d.cbuf_index, .offset = d.cbuf_offset, .indirect_offset = d.indirect_offset,
              .is_atomic = d.is_atomic, .ssbo_desc = &d});
    }
    for (const auto& d : info.texture_buffer_descriptors) {
        push({.kind = NVN_EB_TBUF, .count = d.count, .cbuf = d.cbuf_index, .offset = d.cbuf_offset,
              .shift_left = d.shift_left, .size_shift = d.size_shift,
              .has_secondary = d.has_secondary, .secondary_cbuf = d.secondary_cbuf_index,
              .secondary_offset = d.secondary_cbuf_offset,
              .secondary_shift_left = d.secondary_shift_left,
              .handle_info = d.handle_program == ~0U ? nullptr : &info, .handle_program = d.handle_program});
    }
    for (const auto& d : info.image_buffer_descriptors) {
        push({.kind = NVN_EB_IMG_TBUF, .is_written = d.is_written, .is_integer = d.is_integer,
              .count = d.count, .cbuf = d.cbuf_index, .offset = d.cbuf_offset,
              .size_shift = d.size_shift, .handle_info = d.handle_program == ~0U ? nullptr : &info, .handle_program = d.handle_program});
    }
    for (const auto& d : info.texture_descriptors) {
        nvn_eden_bind b{.kind = NVN_EB_TEX, .is_depth = d.is_depth, .is_ms = d.is_multisample,
                        .is_integer = d.is_integer, .count = d.count, .cbuf = d.cbuf_index,
                        .offset = d.cbuf_offset, .shift_left = d.shift_left,
                        .size_shift = d.size_shift, .has_secondary = d.has_secondary,
                        .secondary_cbuf = d.secondary_cbuf_index,
                        .secondary_offset = d.secondary_cbuf_offset,
                        .secondary_shift_left = d.secondary_shift_left,
                        .handle_info = d.handle_program == ~0U ? nullptr : &info, .handle_program = d.handle_program,
                        .handle_is_pointer = d.handle_is_pointer_word};
        Dim(d.type, b);
        push(b);
    }
    for (const auto& d : info.image_descriptors) {
        nvn_eden_bind b{.kind = NVN_EB_IMG, .is_written = d.is_written, .is_integer = d.is_integer,
                        .count = d.count, .cbuf = d.cbuf_index, .offset = d.cbuf_offset,
                        .size_shift = d.size_shift, .handle_info = d.handle_program == ~0U ? nullptr : &info, .handle_program = d.handle_program,
                        .img_format = static_cast<uint8_t>(d.format)};
        Dim(d.type, b);
        push(b);
    }
}

void EnsureTables() {
    static std::once_flag once;
    std::call_once(once, [] {
        const Vulkan::Device* device = Vulkan::NvnBridge::GetDevice();
        if (!device) {
            return;
        }
        VideoCommon::Nvn::InitFormatTables(*device);
        NvnEnvironment::SetFormatLookup(
            [](int vk_format) { return VideoCommon::Nvn::PixelFormatOfVk(vk_format); });
    });
}

nvn_eden_pipeline* Record(const void* pipeline, Vulkan::GraphicsPipeline* g,
                          Vulkan::ComputePipeline* c, const std::array<u32, 3>* wg = nullptr) {
    std::scoped_lock lock{g_pipe_mutex};
    auto& r = g_records[pipeline];
    if (!r) {
        ++(g ? g_records_graphics : g_records_compute);
        r = std::make_unique<PipelineRecord>();
        r->graphics = g;
        r->compute = c;
        if (g) {
            const auto& infos = g->NvnStageInfos();
            for (int i = 0; i < 5; ++i) {
                const int stage = infos[i].is_task ? 3 : infos[i].is_mesh ? 4 : NVN_STAGE_OF_EDEN[i];
                AddStage(infos[i], stage, r->binds, r->num_entries);
                BridgeCoverage(infos[i], stage, pipeline);   // NVN2 diagnostic
            }
        } else {
            AddStage(c->NvnInfo(), 5, r->binds, r->num_entries);
            BridgeCoverage(c->NvnInfo(), 5, pipeline);   // NVN2 diagnostic
            if (wg) for (auto& b : r->binds) { b.wg[0] = (*wg)[0]; b.wg[1] = (*wg)[1]; b.wg[2] = (*wg)[2]; }   // hybrid2 (nvn_eden_ssbo_reach)
        }
    }
    return reinterpret_cast<nvn_eden_pipeline*>(r.get());
}

} // namespace

int nvn_eden_ssbo_reach(const nvn_eden_bind* bind, const uint32_t* grid, uint64_t* bytes) {
    if (!bind || !bind->ssbo_desc || !bind->wg[0] || !bind->wg[1] || !bind->wg[2]) return 0;
    const auto& desc = *static_cast<const Shader::StorageBufferDescriptor*>(bind->ssbo_desc);
    const std::array<u32, 3> block{bind->wg[0], bind->wg[1], bind->wg[2]};
    std::optional<std::array<u32, 3>> g;
    if (grid) g = std::array<u32, 3>{grid[0], grid[1], grid[2]};
    const auto r = Shader::ResolveStorageWriteExtent(desc.write_extent, block, g);
    if (!r) return 0;
    *bytes = *r;
    return 1;
}

uint32_t nvn_eden_eval_handle(const nvn_eden_bind* bind, uint32_t element, const nvn_eden_query* query,
                              uint64_t alignment, uint32_t (*read_word)(uint64_t)) {
    const auto& info = *static_cast<const Shader::Info*>(bind->handle_info);
    if (bind->handle_program >= info.handle_programs.size()) return 0;
    const auto cbuf = [&](u32 bank, u32 offset) {
        return query->cbuf(query->user, bind->nvn_stage, bank, offset);
    };
    const auto storage = [&](u32 binding, u32 offset) -> u32 {
        if (binding >= info.storage_buffers_descriptors.size()) return 0;
        const auto& desc = info.storage_buffers_descriptors[binding];
        u64 base = cbuf(desc.cbuf_index, desc.cbuf_offset) |
                   (u64{cbuf(desc.cbuf_index, desc.cbuf_offset + 4)} << 32);
        if (base && desc.indirect_offset != ~0U) {
            if (base > UINT64_MAX - desc.indirect_offset - 4) return 0;
            const u64 entry = base + desc.indirect_offset;
            base = read_word(entry) | (u64{read_word(entry + 4)} << 32);
        }
        base &= ~(std::max<u64>(1, alignment) - 1);
        return base && base <= UINT64_MAX - offset ? read_word(base + offset) : 0;
    };
    return Shader::EvaluateHandleProgram(info.handle_programs[bind->handle_program], cbuf, storage, element).value_or(0);
}

nvn_eden_pipeline* nvn_eden_graphics_pipeline(nvn_eden_shader* const shaders[5],
                                              const nvn_eden_draw_state* draw,
                                              const nvn_eden_query* query) try {
    Vulkan::PipelineCache* const cache = Vulkan::NvnBridge::GetPipelineCache();
    if (!cache || !draw) {
        return nullptr;
    }
    EnsureTables();
    // Eden builds the pipeline against its RenderPassCache pass for the key's formats: a target
    // whose VkFormat has no Maxwell format would give a pass the replay's is not compatible with
    for (u32 i = 0; i < draw->num_colors && i < 8; ++i) {
        if (draw->color_format[i] != 0 &&
            VideoCommon::Nvn::NvnRenderTargetFormat(draw->color_format[i]) == Tegra::RenderTargetFormat::NONE) {
            LOG_ERROR(Render_Vulkan, "NVN draw: colour target {} VkFormat {} has no Maxwell format",
                      i, draw->color_format[i]);
            return nullptr;
        }
    }
    if (draw->depth_format != 0 && !VideoCommon::Nvn::NvnHasDepthFormat(draw->depth_format)) {
        LOG_ERROR(Render_Vulkan, "NVN draw: depth VkFormat {} has no Maxwell format", draw->depth_format);
        return nullptr;
    }
    // the fragment program's early fragment tests are the NVN driver's (control +0x718), whatever
    // the replay put in the draw state
    nvn_eden_draw_state state = *draw;
    state.early_fragment = shaders[1] && shaders[1]->has_code
                               ? (shaders[1]->source.ControlWord(VideoCommon::Nvn::Control::EARLY_FRAGMENT) != 0)
                               : 0;
    draw = &state;
    Vulkan::GraphicsPipelineCacheKey key;
    std::memset(&key, 0, sizeof(key));   // unique object representation: no stray bytes in the key

    // environments in program order (VertexB, TCS, TES, GS, FS), as CreateGraphicsPipeline takes them
    std::array<std::optional<NvnEnvironment>, 6> envs;
    for (int st = 0; st < 5; ++st) {
        nvn_eden_shader* const s = shaders[st];
        if (!s || !s->has_code) {
            continue;
        }
        auto& env = envs[PROGRAM_OF_NVN[st]].emplace(s->source, query, Lowered(*s));
        KeepLowered(*s, env);
        const u64 hash = PrepareProgram(*s, env);
        if (hash == 0) {
            return nullptr;
        }
        key.unique_hashes[PROGRAM_OF_NVN[st]] = hash;
    }
    static thread_local VideoCommon::Nvn::Maxwell regs;
    VideoCommon::Nvn::FillNvnRegs(regs, *draw);
    VideoCommon::Nvn::FillNvnGraphicsKey(
        key.state, regs, static_cast<VideoCommon::Nvn::Maxwell::PrimitiveTopology>(draw->primitive),
        cache->NvnDynamicFeatures());
    key.state.vertex_pulling.Assign(draw->vertex_pull ? 1 : 0);   // vtxpull: in the key (and its disk-cache entry), vk_vertex_pulling.h

    boost::container::static_vector<VideoCommon::GenericEnvironment*, 6> env_ptrs;
    for (auto& env : envs) {
        if (env) {
            env_ptrs.push_back(&*env);
        }
    }
    Vulkan::GraphicsPipeline* const pipeline = cache->NvnGraphicsPipeline(key, env_ptrs);
    for (auto& env : envs) {
        if (env) {
            env->SetQuery(nullptr);
        }
    }
    return pipeline ? Record(pipeline, pipeline, nullptr) : nullptr;
} catch (const std::exception& e) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: exception: {}", "graphics pipeline", e.what());
    return nullptr;
} catch (...) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: unknown exception", "graphics pipeline");
    return nullptr;
}

void nvn_eden_pipeline_counts(uint64_t out[2]) {
    std::scoped_lock lock{g_pipe_mutex};
    out[0] = g_records_graphics;
    out[1] = g_records_compute;
}

// NVN2 compute shaders with local memory: their hash (the environment's, as vulkan.bin keys it), workgroup, shared size, the size assumed and the control
// section bytes the bridge keeps (Control2::READ_SIZE), once per shader and process, into the log folder's nvn2_cs_control.txt. Offline, the hashes are
// matched against an emu-mode cache's environments, whose local memory size is the QMD's (the game's NVN2 driver's), to find the control section's field
static void DumpNvn2CsControl(u64 hash, const NvnEnvironment& env, const VideoCommon::Nvn::NvnShaderSource& source) {
    static std::mutex mu;
    static std::set<u64> done;
    static bool started = false;
    std::scoped_lock lock{mu};
    if (!done.insert(hash).second) {
        return;
    }
    const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / "nvn2_cs_control.txt";
    std::FILE* f = std::fopen(path.string().c_str(), started ? "a" : "w");
    if (!f) {
        return;
    }
    started = true;
    const auto wg = env.WorkgroupSize();
    std::fprintf(f, "%016llx wg %u %u %u shared %u local %u control %u:", static_cast<unsigned long long>(hash), wg[0], wg[1], wg[2],
                 env.SharedMemorySize(), env.LocalMemorySize(), source.control_size);
    for (u32 i = 0; i < source.control_size && i < source.control.size(); ++i) {
        std::fprintf(f, "%s%02x", i % 4 ? "" : " ", source.control[i]);
    }
    std::fprintf(f, "\n");
    std::fclose(f);
}

nvn_eden_pipeline* nvn_eden_compute_pipeline(nvn_eden_shader* shader, const nvn_eden_query* query) try {
    Vulkan::PipelineCache* const cache = Vulkan::NvnBridge::GetPipelineCache();
    if (!cache || !shader || !shader->has_code) {
        return nullptr;
    }
    EnsureTables();
    NvnEnvironment env{shader->source, query, Lowered(*shader)};
    KeepLowered(*shader, env);
    Vulkan::ComputePipelineCacheKey key;
    std::memset(&key, 0, sizeof(key));
    key.unique_hash = PrepareProgram(*shader, env);
    if (key.unique_hash == 0) {
        return nullptr;
    }
    if (shader->source.nvn2 && env.LocalMemorySize() != 0) {
        DumpNvn2CsControl(key.unique_hash, env, shader->source);
    }
    key.shared_memory_size = env.SharedMemorySize();
    key.workgroup_size = env.WorkgroupSize();
    Vulkan::ComputePipeline* const pipeline = cache->NvnComputePipeline(key, env);
    env.SetQuery(nullptr);
    return pipeline ? Record(pipeline, nullptr, pipeline, &key.workgroup_size) : nullptr;
} catch (const std::exception& e) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: exception: {}", "compute pipeline", e.what());
    return nullptr;
} catch (...) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: unknown exception", "compute pipeline");
    return nullptr;
}

// bak85 (11-16, MKW scenery / cactus flicker; nxnative docs/reference/shaders.md "A store only a dispatch's first invocation makes", dmsh_build.py first_store):
// the IR of one compute shader as Eden translates it for the NVN path (the same environment and translation settings as nvn_eden_compute_pipeline, its own
// pools), written once into the log folder, so that the first-invocation guard and its stores can be matched in IR before a pass moves them to the host
int nvn_eden_dump_ir(nvn_eden_shader* shader, const nvn_eden_query* query, uint64_t host_key) try {
    Vulkan::PipelineCache* const cache = Vulkan::NvnBridge::GetPipelineCache();
    if (!cache || !shader || !shader->has_code) {
        return 0;
    }
    EnsureTables();
    NvnEnvironment env{shader->source, query, Lowered(*shader)};
    KeepLowered(*shader, env);
    const u64 hash = env.Analyze().value_or(0);
    if (hash == 0) {
        return 0;
    }
    env.SetCachedSize(env.CachedSizeBytes());
    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow{32};
    Shader::Maxwell::Flow::CFG cfg{env, flow, env.StartAddress()};
    const Shader::IR::Program program = Shader::Maxwell::TranslateProgram(inst, block, env, cfg, cache->NvnHostInfo());
    env.SetQuery(nullptr);
    char name[64];
    std::snprintf(name, sizeof(name), "nvn2_ir_%016llx.txt", static_cast<unsigned long long>(host_key));
    const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) / name;
    std::FILE* f = std::fopen(path.string().c_str(), "w");
    if (!f) {
        return 0;
    }
    const auto wg = program.workgroup_size;
    std::fprintf(f, "host key %016llx eden hash %016llx workgroup %u %u %u shared %u local %u\n", static_cast<unsigned long long>(host_key),
                 static_cast<unsigned long long>(hash), wg[0], wg[1], wg[2], program.shared_memory_size, program.local_memory_size);
    const auto& sb = program.info.storage_buffers_descriptors;
    for (size_t i = 0; i < sb.size(); ++i) {
        std::fprintf(f, "storage buffer %zu: cbuf %u offset %#x count %u written %d\n", i, sb[i].cbuf_index, sb[i].cbuf_offset, sb[i].count, sb[i].is_written ? 1 : 0);
    }
    const std::string text = Shader::IR::DumpProgram(program);
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    return 1;
} catch (const std::exception& e) {
    RethrowNativeMemoryFailure();
    LOG_CRITICAL(Render_Vulkan, "NVN {}: exception: {}", "IR dump", e.what());
    return 0;
} catch (...) {
    RethrowNativeMemoryFailure();
    return 0;
}

int nvn_eden_pipeline_get(nvn_eden_pipeline* p, nvn_eden_pipeline_info* info, int wait) {
    if (!p || !info) {
        return 0;
    }
    const auto* r = reinterpret_cast<const PipelineRecord*>(p);
    *info = {};
    const bool built = r->graphics ? (r->graphics->IsBuilt() || (wait && r->graphics->NvnWaitForBuild()))
                                   : (r->compute->NvnIsBuilt() || (wait && r->compute->NvnWaitForBuild()));
    if (!built) return 1;
    info->built = 1;
    const auto fill = [&](const auto& pl) {
        info->pipeline = pl.NvnPipeline();
        info->layout = pl.NvnPipelineLayout();
        info->set_layout = pl.NvnSetLayout();
        info->update_template = pl.NvnUpdateTemplate();
        info->push_descriptor = pl.NvnUsesPushDescriptor() ? 1 : 0;
        info->descriptor_buffer = pl.NvnUsesDescriptorBuffer() ? 1 : 0;
    };
    if (r->graphics) {
        fill(*r->graphics);
        const auto& h = r->graphics->NvnKey().unique_hashes;
        std::copy(h.begin(), h.end(), info->unique_hashes);
    } else {
        fill(*r->compute);
    }
    info->num_entries = r->num_entries;
    info->entry_size = sizeof(Vulkan::DescriptorUpdateEntry);
    info->num_binds = static_cast<u32>(r->binds.size());
    info->binds = r->binds.data();
    info->compute = r->compute ? 1 : 0;
    if (r->graphics) {
        const auto& infos = r->graphics->NvnStageInfos();
        info->dyn = DynOf(r->graphics->NvnKey().state);
        if (const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice()) {   // MakePipeline's
            info->num_viewports = (std::min<u32>)(device->GetMaxViewports(), VideoCommon::Nvn::Maxwell::NumViewports);
        }
        for (u32 i = 0; i < 32; ++i) {
            info->vs_inputs |= infos[0].loads.Generic(i) ? 1u << i : 0u;
        }
        for (u32 i = 0; i < 8; ++i) {
            info->frag_outputs |= infos[4].stores_frag_color[i] ? 1u << i : 0u;
        }
        for (u32 i = 0; i < 4; ++i) {   // pkm2: stages that write gl_Layer (host r76 layered passes)
            info->layer_stages |= infos[i].stores[Shader::IR::Attribute::Layer] ? 1u << i : 0u;
        }
        info->vertex_pull = r->graphics->PullsVertices() ? 1 : 0;   // vtxpull
        if (info->descriptor_buffer) {
            info->dbuf_size = r->graphics->NvnDescriptorBufferLayout().size;
        }
    } else if (info->descriptor_buffer) {
        info->dbuf_size = r->compute->NvnDescriptorBufferLayout().size;
    }
    if (r->compute) {   // bak86
        const auto fs = Vulkan::NvnFirstStoresOf(r->compute);
        for (const auto& s : fs) {
            if (info->nfirst == 4) {
                break;
            }
            info->first[info->nfirst][0] = s.cbuf_index;
            info->first[info->nfirst][1] = s.cbuf_offset;
            info->first[info->nfirst][2] = s.byte_offset;
            info->first[info->nfirst][3] = s.value;
            ++info->nfirst;
        }
    }
    return 1;
}

// ---- B4b: recording (see the header) ----

namespace {

const PipelineRecord* RecordOf(nvn_eden_pipeline* p) {
    return reinterpret_cast<const PipelineRecord*>(p);
}

// vk_graphics_pipeline.cpp MakePipeline's dynamic state list, as NVN_ED_* bits
u32 DynOf(const Vulkan::FixedPipelineState& s) {
    u32 d = 0;
    if (s.extended_dynamic_state) {
        d |= NVN_ED_EDS1;
        if (!s.dynamic_vertex_input) {
            d |= NVN_ED_STRIDE;
        }
    }
    if (s.dynamic_vertex_input) {
        d |= NVN_ED_VERTEX_INPUT;
    }
    if (s.extended_dynamic_state_2) {
        d |= NVN_ED_EDS2;
    }
    if (s.extended_dynamic_state_2_logic_op) {
        d |= NVN_ED_EDS2_LOGIC_OP;
    }
    if (s.extended_dynamic_state_3_blend) {
        d |= NVN_ED_EDS3_BLEND;
    } else if (s.color_write_enable_dynamic) {
        d |= NVN_ED_COLOR_WRITE_ENABLE;
    }
    if (s.extended_dynamic_state_3_enables) {
        d |= NVN_ED_EDS3_ENABLES;
    }
    return d;
}

// vk_rasterizer.cpp IsPrimitiveRestartSupported (file-local there)
bool RestartSupported(const Vulkan::Device& device, VkPrimitiveTopology t) {
    const bool strip = t == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
                       t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
                       t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
                       t == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY ||
                       t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    return (t != VK_PRIMITIVE_TOPOLOGY_PATCH_LIST && device.IsTopologyListPrimitiveRestartSupported()) ||
           strip || (t == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST && device.IsPatchListPrimitiveRestartSupported());
}

// fixed_pipeline_state.cpp / vk_rasterizer.cpp UpdateDepthBiasEnable's table
constexpr std::array<u8, 15> POLYGON_OFFSET_CLASS = {0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2};

} // namespace

int nvn_eden_resolution_x2(void) {
    return Settings::values.resolution_info.up_factor == 2.0f;
}

void nvn_eden_cmd_state(void* cmdbuf, nvn_eden_pipeline* p, const nvn_eden_draw_state* d,
                        const nvn_eden_dyn* o) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !device || !cmdbuf) {
        return;
    }
    using namespace Shader::Backend::SPIRV;
    using VideoCommon::Nvn::Maxwell;
    namespace MaxwellToVK = Vulkan::MaxwellToVK;
    const Vulkan::vk::CommandBuffer cmd = Vulkan::vk::CommandBuffer{static_cast<VkCommandBuffer>(cmdbuf), device->GetDispatchLoader()};
    static constexpr std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> NO_RESCALING{};
    auto scaling = NO_RESCALING;
    static_assert(NUM_TEXTURE_SCALING_WORDS == sizeof(o->scaled_textures) / sizeof(u32));
    if (o) std::copy_n(o->scaled_textures, NUM_TEXTURE_SCALING_WORDS, scaling.begin());
    if (r->compute) {
        cmd.PushConstants(r->compute->NvnPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          RESCALING_LAYOUT_WORDS_OFFSET, sizeof(scaling), scaling.data());
        return;
    }
    if (!d || !o) {
        return;
    }
    const Vulkan::GraphicsPipeline& g = *r->graphics;
    const VkPipelineLayout layout = g.NvnPipelineLayout();
    const VkShaderStageFlags stages = g.IsMeshPipeline()
        ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT
        : VK_SHADER_STAGE_ALL_GRAPHICS;
    cmd.PushConstants(layout, stages, RESCALING_LAYOUT_WORDS_OFFSET,
                      sizeof(NO_RESCALING), scaling.data());
    const f32 down_factor = o->scaled_target ? 0.5f : 1.0f;
    cmd.PushConstants(layout, stages, RESCALING_LAYOUT_DOWN_FACTOR_OFFSET,
                      sizeof(down_factor), &down_factor);
    bool render_area = false;
    for (const auto& info : g.NvnStageInfos()) {
        render_area |= info.uses_render_area;
    }
    if (render_area) {
        const std::array<f32, 4> area{o->render_area[0], o->render_area[1], 0.0f, 0.0f};
        cmd.PushConstants(layout, stages, RENDERAREA_LAYOUT_OFFSET,
                          sizeof(area), area.data());
    }

    // RasterizerVulkan::UpdateDynamicStates, every group the pipeline has, nothing cached
    const u32 dyn = DynOf(g.NvnKey().state);
    const auto topology = static_cast<Maxwell::PrimitiveTopology>(d->primitive);
    const VkPrimitiveTopology vk_topology = MaxwellToVK::PrimitiveTopology(*device, topology);
    const auto pick = [](int32_t v, auto from_draw) { return v >= 0 ? static_cast<u32>(v) : static_cast<u32>(from_draw); };
    if (dyn & NVN_ED_EDS1) {
        const Maxwell::CullFace face = d->cull_face == 1   ? Maxwell::CullFace::Front
                                       : d->cull_face == 2 ? Maxwell::CullFace::Back
                                                           : Maxwell::CullFace::FrontAndBack;
        cmd.SetCullModeEXT(static_cast<VkCullModeFlags>(
            pick(o->cull_mode, d->cull_face ? MaxwellToVK::CullFace(face) : VK_CULL_MODE_NONE)));
        const VkFrontFace ff = MaxwellToVK::FrontFace(d->front_face ? Maxwell::FrontFace::CounterClockWise
                                                                    : Maxwell::FrontFace::ClockWise);
        cmd.SetFrontFaceEXT(static_cast<VkFrontFace>(pick(o->front_face, ff)));
        cmd.SetDepthCompareOpEXT(static_cast<VkCompareOp>(pick(
            o->depth_compare, MaxwellToVK::DepthComparisonOp(VideoCommon::Nvn::NvnCompare(d->depth_func)))));
        const auto sop = [&](int f) {
            cmd.SetStencilOpEXT(f == 0 ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT,
                                MaxwellToVK::StencilOp(VideoCommon::Nvn::NvnStencilOp(d->stencil_fail[f])),
                                MaxwellToVK::StencilOp(VideoCommon::Nvn::NvnStencilOp(d->stencil_zpass[f])),
                                MaxwellToVK::StencilOp(VideoCommon::Nvn::NvnStencilOp(d->stencil_zfail[f])),
                                MaxwellToVK::ComparisonOp(VideoCommon::Nvn::NvnCompare(d->stencil_func[f])));
        };
        sop(0);
        sop(1);   // stencil_two_side_enable is always set for NVN (FillNvnRegs)
        cmd.SetDepthBoundsTestEnableEXT(
            pick(o->depth_bounds_test, d->depth_bounds_test && device->IsDepthBoundsSupported()) != 0);
        cmd.SetDepthTestEnableEXT(pick(o->depth_test, d->depth_test) != 0);
        cmd.SetDepthWriteEnableEXT(pick(o->depth_write, d->depth_write) != 0);
        cmd.SetStencilTestEnableEXT(pick(o->stencil_test, d->stencil_test) != 0);
        cmd.SetPrimitiveTopologyEXT(vk_topology);
    }
    if (dyn & NVN_ED_EDS2) {
        bool restart = d->primitive_restart != 0;
        if (device->IsMoltenVK()) {
            restart = true;
        } else if (restart) {
            restart = RestartSupported(*device, vk_topology);
        }
        cmd.SetPrimitiveRestartEnableEXT(pick(o->primitive_restart, restart) != 0);
        cmd.SetRasterizerDiscardEnableEXT(pick(o->rasterizer_discard, d->rasterizer_discard) != 0);
        const u32 cls = POLYGON_OFFSET_CLASS[d->primitive < 15 ? d->primitive : 4];
        cmd.SetDepthBiasEnableEXT(pick(o->depth_bias_enable, (d->polygon_offset >> cls) & 1) != 0);
    }
    if (dyn & NVN_ED_EDS2_LOGIC_OP) {
        cmd.SetLogicOpEXT(static_cast<VkLogicOp>(d->logic_op & 0xF));
    }
    if (dyn & NVN_ED_EDS3_ENABLES) {
        if (device->SupportsDynamicState3LogicOpEnable()) {
            cmd.SetLogicOpEnableEXT(d->logic_op_enable != 0);
        }
        if (device->SupportsDynamicState3DepthClampEnable()) {
            cmd.SetDepthClampEnableEXT(d->depth_clamp != 0);   // FrustumXY (FillNvnRegs) = clamp
        }
        if (device->IsExtLineRasterizationSupported() && device->SupportsDynamicState3LineRasterizationMode()) {
            cmd.SetLineRasterizationModeEXT(d->line_smooth && device->SupportsSmoothLines()
                                                ? VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_EXT
                                                : VK_LINE_RASTERIZATION_MODE_RECTANGULAR_EXT);
        }
        if (device->SupportsDynamicState3LineStippleEnable()) {
            cmd.SetLineStippleEnableEXT(d->line_stipple != 0);
        }
        if (device->SupportsDynamicState3ConservativeRasterizationMode()) {
            cmd.SetConservativeRasterizationModeEXT(d->conservative_raster
                                                        ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                                        : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
        }
        if (device->SupportsDynamicState3AlphaToCoverageEnable()) {
            cmd.SetAlphaToCoverageEnableEXT(g.SupportsAlphaToCoverage() && d->alpha_to_coverage ? VK_TRUE : VK_FALSE);
        }
        if (device->SupportsDynamicState3AlphaToOneEnable()) {
            cmd.SetAlphaToOneEnableEXT(g.SupportsAlphaToOne() && d->alpha_to_one ? VK_TRUE : VK_FALSE);
        }
    }
    const auto& fs = g.NvnStageInfos()[4];
    const auto mask_of = [&](size_t i) {
        u32 m = i < 8 ? d->channel_mask[i] & 0xF : 0;
        if (o->mask_unwritten && i < 8 && !fs.stores_frag_color[i]) {
            m = 0;
        }
        return m;
    };
    if (dyn & NVN_ED_EDS3_BLEND) {
        std::array<VkColorComponentFlags, Maxwell::NumRenderTargets> masks{};
        std::array<VkBool32, Maxwell::NumRenderTargets> enables{};
        std::array<VkColorBlendEquationEXT, Maxwell::NumRenderTargets> eqs{};
        for (size_t i = 0; i < Maxwell::NumRenderTargets; ++i) {
            masks[i] = static_cast<VkColorComponentFlags>(mask_of(i));   // R G B A bits = VK's
            const bool used = i < d->num_colors && d->color_format[i] != 0;
            const auto rt = used ? VideoCommon::Nvn::NvnRenderTargetFormat(d->color_format[i])
                                 : Tegra::RenderTargetFormat::NONE;
            const bool is_int = rt != Tegra::RenderTargetFormat::NONE &&
                                VideoCore::Surface::IsPixelFormatInteger(
                                    VideoCore::Surface::PixelFormatFromRenderTargetFormat(rt));
            enables[i] = !is_int && ((d->blend_enable >> i) & 1) ? VK_TRUE : VK_FALSE;
            const auto& b = d->blend[i];
            using VideoCommon::Nvn::NvnEquation;
            using VideoCommon::Nvn::NvnFactor;
            eqs[i] = VkColorBlendEquationEXT{
                .srcColorBlendFactor = MaxwellToVK::BlendFactor(NvnFactor(b.func_src)),
                .dstColorBlendFactor = MaxwellToVK::BlendFactor(NvnFactor(b.func_dst)),
                .colorBlendOp = MaxwellToVK::BlendEquation(NvnEquation(b.eq)),
                .srcAlphaBlendFactor = MaxwellToVK::BlendFactor(NvnFactor(b.func_src_a)),
                .dstAlphaBlendFactor = MaxwellToVK::BlendFactor(NvnFactor(b.func_dst_a)),
                .alphaBlendOp = MaxwellToVK::BlendEquation(NvnEquation(b.eq_a)),
            };
        }
        cmd.SetColorWriteMaskEXT(0, masks);
        cmd.SetColorBlendEnableEXT(0, enables);
        cmd.SetColorBlendEquationEXT(0, eqs);
    } else if (dyn & NVN_ED_COLOR_WRITE_ENABLE) {
        std::array<VkBool32, Maxwell::NumRenderTargets> enables{};
        for (size_t i = 0; i < Maxwell::NumRenderTargets; ++i) {
            enables[i] = mask_of(i) ? VK_TRUE : VK_FALSE;
        }
        cmd.SetColorWriteEnableEXT(enables);
    }
    if (dyn & NVN_ED_VERTEX_INPUT) {   // UpdateVertexInput
        boost::container::static_vector<VkVertexInputBindingDescription2EXT, 32> bindings;
        boost::container::static_vector<VkVertexInputAttributeDescription2EXT, 32> attributes;
        const u32 max_bindings = static_cast<u32>(
            (std::min<size_t>)(Maxwell::NumVertexArrays, device->GetMaxVertexInputBindings()));
        const u32 max_attributes = static_cast<u32>(
            (std::min<size_t>)(Maxwell::NumVertexAttributes, device->GetMaxVertexInputAttributes()));
        for (u32 a = 0; a < d->num_attribs && a < 16; ++a) {
            const auto& in = d->attrib[a];
            Maxwell::VertexAttribute::Size size;
            Maxwell::VertexAttribute::Type type;
            if (in.location >= max_attributes || in.stream >= max_bindings ||
                !VideoCommon::Nvn::NvnAttribFormat(in.vk_format, size, type)) {
                continue;   // a constant attribute (FillNvnRegs leaves it constant too)
            }
            attributes.push_back({
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .location = in.location,
                .binding = in.stream,
                .format = MaxwellToVK::VertexFormat(*device, type, size),
                .offset = in.offset,
            });
        }
        for (u32 b = 0; b < max_bindings; ++b) {
            const bool have = b < d->num_streams && b < 16;
            const u32 divisor = have ? d->stream[b].divisor : 0;
            bindings.push_back({
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                .pNext = nullptr,
                .binding = b,
                .stride = have ? d->stream[b].stride : 0,
                .inputRate = divisor ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX,
                .divisor = divisor ? divisor : 1,
            });
        }
        cmd.SetVertexInputEXT(bindings, attributes);
    }
}

void nvn_eden_cmd_vertex_buffers(void* cmdbuf, nvn_eden_pipeline* p, const nvn_eden_draw_state* d,
                                 uint32_t first, uint32_t count, void* const* buffers,
                                 const uint64_t* offsets) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !r->graphics || !device || !cmdbuf || !count || count > 32) {
        return;
    }
    const Vulkan::vk::CommandBuffer cmd = Vulkan::vk::CommandBuffer{static_cast<VkCommandBuffer>(cmdbuf), device->GetDispatchLoader()};
    std::array<VkBuffer, 32> bufs;
    std::array<VkDeviceSize, 32> offs;
    for (u32 i = 0; i < count; ++i) {
        bufs[i] = static_cast<VkBuffer>(buffers[i]);
        offs[i] = offsets[i];
    }
    if (DynOf(r->graphics->NvnKey().state) & NVN_ED_STRIDE) {
        std::array<VkDeviceSize, 32> strides;
        for (u32 i = 0; i < count; ++i) {
            const u32 b = first + i;
            strides[i] = d && b < d->num_streams && b < 16 ? d->stream[b].stride : 0;
        }
        cmd.BindVertexBuffers2EXT(first, count, bufs.data(), offs.data(), nullptr, strides.data());
    } else {
        cmd.BindVertexBuffers(first, count, bufs.data(), offs.data());
    }
}

void nvn_eden_cmd_vertex_table(void* cmdbuf, nvn_eden_pipeline* p, uint64_t address) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !r->graphics || !r->graphics->PullsVertices() || !device || !cmdbuf) {
        return;
    }
    const Vulkan::vk::CommandBuffer cmd = Vulkan::vk::CommandBuffer{static_cast<VkCommandBuffer>(cmdbuf), device->GetDispatchLoader()};
    // the stages of the layout's one range (pipeline_helper.h CreatePipelineLayout), as nvn_eden_cmd_state pushes
    cmd.PushConstants(r->graphics->NvnPipelineLayout(), VK_SHADER_STAGE_ALL_GRAPHICS,
                      Shader::Backend::SPIRV::VERTEX_PULL_TABLE_OFFSET, sizeof(address), &address);
}

void nvn_eden_cmd_push_descriptors(void* cmdbuf, nvn_eden_pipeline* p, const void* payload) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !device || !cmdbuf || !payload) {
        return;
    }
    const Vulkan::vk::CommandBuffer cmd = Vulkan::vk::CommandBuffer{static_cast<VkCommandBuffer>(cmdbuf), device->GetDispatchLoader()};
    if (r->graphics) {
        cmd.PushDescriptorSetWithTemplateKHR(r->graphics->NvnUpdateTemplate(),
                                             r->graphics->NvnPipelineLayout(), 0, payload);
    } else {
        cmd.PushDescriptorSetWithTemplateKHR(r->compute->NvnUpdateTemplate(),
                                             r->compute->NvnPipelineLayout(), 0, payload);
    }
}

void nvn_eden_update_set(nvn_eden_pipeline* p, void* set, const void* payload) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !device || !set || !payload) {
        return;
    }
    const VkDescriptorUpdateTemplate t =
        r->graphics ? r->graphics->NvnUpdateTemplate() : r->compute->NvnUpdateTemplate();
    device->GetLogical().UpdateDescriptorSet(static_cast<VkDescriptorSet>(set), t, payload);
}

uint64_t nvn_eden_dbuf_alignment(void) {
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!device || !device->IsExtDescriptorBufferSupported()) {
        return 0;
    }
    return (std::max<VkDeviceSize>)(device->DescriptorBufferProperties().descriptorBufferOffsetAlignment, 1);
}

void nvn_eden_write_dbuf(nvn_eden_pipeline* p, const void* payload, void* host) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !device || !payload || !host) {
        return;
    }
    const auto& layout = r->graphics ? r->graphics->NvnDescriptorBufferLayout()
                                     : r->compute->NvnDescriptorBufferLayout();
    Vulkan::WriteDescriptorBuffer(*device, layout,
                                  static_cast<const Vulkan::DescriptorUpdateEntry*>(payload),
                                  static_cast<u8*>(host));
}

void nvn_eden_cmd_dbuf(void* cmdbuf, nvn_eden_pipeline* p, uint64_t address, uint64_t offset,
                       int bind_buffer) {
    const PipelineRecord* const r = RecordOf(p);
    const Vulkan::Device* const device = Vulkan::NvnBridge::GetDevice();
    if (!r || !device || !cmdbuf) {
        return;
    }
    const Vulkan::vk::CommandBuffer cmd = Vulkan::vk::CommandBuffer{static_cast<VkCommandBuffer>(cmdbuf), device->GetDispatchLoader()};
    if (bind_buffer) {   // as DescriptorBufferRing::BindingInfo
        const VkDescriptorBufferBindingInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
            .pNext = nullptr,
            .address = address,
            .usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                     VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
        };
        cmd.BindDescriptorBuffersEXT(info);
    }
    const bool compute = r->compute != nullptr;
    const u32 kIndex = 0;
    const VkDeviceSize kOffset = offset;
    cmd.SetDescriptorBufferOffsetsEXT(compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
                                      compute ? r->compute->NvnPipelineLayout() : r->graphics->NvnPipelineLayout(),
                                      0, kIndex, kOffset);
}

const void* nvn_eden_shader_c1(const nvn_eden_shader* s, uint32_t* size) {
    if (!s) {
        if (size) {
            *size = 0;
        }
        return nullptr;
    }
    const auto c1 = s->has_code.load(std::memory_order_acquire) ? s->source.C1() : std::span<const u8>{};   // S1c (parallel recording): no code yet reads as none (struct nvn_eden_shader)
    if (size) {
        *size = static_cast<uint32_t>(c1.size());
    }
    return c1.empty() ? nullptr : c1.data();
}

} // extern "C"
