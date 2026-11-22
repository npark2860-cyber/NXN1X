// SPDX-License-Identifier: GPL-3.0-or-later
// See nvn_shader_env.h.

#include <algorithm>
#include <cstdlib>       // NVN2
#include <cstring>
#include <span>          // NVN2
#include <string_view>   // NVN2

#include "common/logging.h"
#include "video_core/nvn/nvn_shader_env.h"
#include "video_core/surface.h"

namespace VideoCommon::Nvn {

namespace {
constexpr u32 SPH_SIZE = sizeof(Shader::ProgramHeader);

// NVNtextureTarget (host dm/gfx/nvn/include/nvn.h:777-787) -> the type Eden's ConvertTextureType
// gives the same texture on the Maxwell path (shader_environment.cpp ConvertTextureType: a 2D TIC
// without normalized coordinates is Color2DRect; multisample TICs are 2D / 2D array).
Shader::TextureType TargetToType(int target) {
    switch (target) {
    case 0x0: return Shader::TextureType::Color1D;
    case 0x1: return Shader::TextureType::Color2D;
    case 0x2: return Shader::TextureType::Color3D;
    case 0x3: return Shader::TextureType::ColorArray1D;
    case 0x4: return Shader::TextureType::ColorArray2D;
    case 0x5: return Shader::TextureType::Color2D;
    case 0x6: return Shader::TextureType::ColorArray2D;
    case 0x7: return Shader::TextureType::Color2DRect;
    case 0x8: return Shader::TextureType::ColorCube;
    case 0x9: return Shader::TextureType::ColorArrayCube;
    case 0xA: return Shader::TextureType::Buffer;
    default:
        LOG_ERROR(Shader, "NVN texture target {} unknown, answered Color2D", target);
        return Shader::TextureType::Color2D;
    }
}

u64 MakeCbufKey(u32 index, u32 offset) {
    return (static_cast<u64>(index) << 32) | offset;
}

std::function<VideoCore::Surface::PixelFormat(int)> g_format_lookup;
} // namespace

u32 NvnShaderSource::ControlWord(u32 offset) const {
    u32 v{};
    if (offset + 4 <= control_size) {
        std::memcpy(&v, control.data() + offset, 4);
    }
    return v;
}

bool IsNvn2Control(const void* control, u32 control_size) {
    if (!control || control_size < Control2::KEY_SIZE) {
        return false;
    }
    u32 magic, version;
    std::memcpy(&magic, control, 4);
    std::memcpy(&version, static_cast<const u8*>(control) + 4, 4);
    return magic == Control2::MAGIC && version == Control2::VERSION;
}

u32 NvnShaderSource::ProgramStart() const {
    return nvn2 ? 0 : ControlWord(Control::PROGRAM_START);   // NVN2: no code area header
}

std::span<const u8> NvnShaderSource::C1() const {
    if (nvn2) {   // dmsh_ampere.py c1_segment: code[+0x40 .. +0x44]
        const u32 begin = ControlWord(Control2::C1_BEGIN);
        const u32 end = ControlWord(Control2::CODE_SIZE);
        if (begin > end || end > code.size()) {
            return {};
        }
        return std::span<const u8>(code).subspan(begin, end - begin);
    }
    const u32 off = ControlWord(Control::C1_OFFSET);
    const u32 size = ControlWord(Control::C1_SIZE);
    if (off >= code.size()) {
        return {};
    }
    return std::span<const u8>(code).subspan(off, (std::min<size_t>)(size, code.size() - off));
}

Shader::Stage StageFromNvn(int nvn_stage) {
    switch (nvn_stage) {
    case 0: return Shader::Stage::VertexB;
    case 1: return Shader::Stage::Fragment;
    case 2: return Shader::Stage::Geometry;
    case 3: return Shader::Stage::TessellationControl;
    case 4: return Shader::Stage::TessellationEval;
    case 6: return Shader::Stage::Task;   // NVN2
    case 7: return Shader::Stage::Mesh;   // NVN2
    default: return Shader::Stage::Compute;
    }
}

void NvnEnvironment::SetFormatLookup(std::function<VideoCore::Surface::PixelFormat(int)> f) {
    g_format_lookup = std::move(f);
}

namespace {
constexpr u32 GRAPHICS_START = 0x10;   // the SPH's offset in a 32-byte bundle

// LDL / STL in the code before its self branch (Maxwell encodings 1110 1111 0100 0--- and
// 1110 1111 0101 0---; the scheduling word of each 32-byte bundle is skipped)
bool UsesLocalMemory(std::span<const u8> code) {
    constexpr u64 SELF_BRANCH_A = 0xE2400FFFFF87000FULL;
    constexpr u64 SELF_BRANCH_B = 0xE2400FFFFF07000FULL;
    for (size_t off = 0; off + 8 <= code.size(); off += 8) {
        if (off % 32 == 0) {
            continue;
        }
        u64 inst;
        std::memcpy(&inst, code.data() + off, 8);
        if (inst == SELF_BRANCH_A || inst == SELF_BRANCH_B) {
            break;
        }
        const u64 op = inst >> 51;
        if (op == 0x1DE8 || op == 0x1DEA) {
            return true;
        }
    }
    return false;
}

// The same scan over lowered NVN2 code: an SPH (0x50 bytes) and then Maxwell code with a scheduling word at every 32-byte boundary
// (native_lowering.cpp MakeLoweredNativeCode). The whole code is scanned: after the instructions come one zero word and the native address
// map, whose small offsets never have an LDL / STL opcode in their top bits
bool UsesLocalMemoryLowered(std::span<const u64> code) {
    for (size_t i = sizeof(Shader::ProgramHeader) / 8; i < code.size(); ++i) {
        if ((i * 8) % 32 == 0) {
            continue;
        }
        const u64 op = code[i] >> 51;
        if (op == 0x1DE8 || op == 0x1DEA) {
            return true;
        }
    }
    return false;
}
} // namespace

// NVN2 (Switch 2): the code area holds Ampere machine code, lowered to Maxwell code the way the Maxwell path lowers a Switch 2 program
// (shader_environment.cpp GraphicsEnvironment: the SPH read from the program's start, InitializeNativeCode(lowered, 128, geometry); ComputeEnvironment:
// InitializeNativeCode(lowered, 0, false) and start_address = sizeof(sph)), so the lowered code and its hash (NativeShaderHash) are the same as there.
void NvnEnvironment::InitNvn2(const NvnShaderSource& source, std::shared_ptr<const LoweredNativeCode> lowered) {
    host_program = source.code;   // the whole code area from program_base 0
    start_address = 0;
    texture_bound = 2;   // replaced by the lowering's when the program names one (InitializeNativeCode)
    is_proprietary_driver = true;
    // ConstantPropagationPass only asks for draw-parameter replacements with
    // this gate enabled. Compute/fragment c0 layouts must remain untouched.
    has_hle_engine_state = nvn_stage == 0;
    viewport_transform_state = 1;
    stage = StageFromNvn(nvn_stage);
    if (stage == Shader::Stage::Compute) {
        workgroup_size = {source.ControlWord(Control2::WORKGROUP), source.ControlWord(Control2::WORKGROUP + 4),
                          source.ControlWord(Control2::WORKGROUP + 8)};
        // the host's rounding (shader_split.py:2964: (sh + 15) // 16 * 16); the field is not yet seen nonzero there
        shared_memory_size = (source.ControlWord(Control2::SHARED_MEMORY) + 15u) & ~15u;
        initial_offset = sizeof(sph);
        InitializeNativeCode(std::move(lowered), 0, false);
        start_address = sizeof(sph);
        // The control section's local memory field is not located (the Maxwell path takes the QMD's): a local load or store in the lowered code
        // (Maxwell encodings, after its SPH) takes 0x1000 bytes a thread, as host_nx2 runs NVN2 compute (Ryujinx's QueryComputeLocalMemorySize
        // default, with which HWAoI and DK ran on Windows). It was 0x4000, as NVN compute below keeps: HWAoI 11-16 had 852 MB of Turnip private
        // memory (pvtmem) from boot and ran out of device memory. NXN1X_NVN2_CS_LOCAL=<bytes> sets the size
        static const u32 cs_local = [] {
            const char* e = std::getenv("NXN1X_NVN2_CS_LOCAL");
            const long v = e ? std::strtol(e, nullptr, 0) : 0;
            return v > 0 ? static_cast<u32>(v) : 0x1000u;
        }();
        local_memory_size = UsesLocalMemoryLowered(code) ? cs_local : 0;
        if (local_memory_size != 0) {
            LOG_WARNING(Shader, "NVN2 compute shader uses local memory: size {:#x} assumed (host_nx2's default; the control section's field is not known; NXN1X_NVN2_CS_LOCAL)", cs_local);
            static bool noted = false;
            if (!noted) {
                noted = true;
                LOG_INFO(Shader, "NVN2 compute local memory is {:#x} bytes a thread (was 0x4000): Turnip's private memory for it shows as pvtmem in TU_DEBUG=bos", cs_local);
            }
        }
        return;
    }
    if (host_program.size() >= sizeof(sph)) {
        std::memcpy(&sph, host_program.data(), sizeof(sph));
    }
    if (stage == Shader::Stage::Mesh || stage == Shader::Stage::Task) {
        const u32 configuration = source.ControlWord(0x58);
        const u32 threads = configuration & 0xffff;
        const u32 lines = ((configuration >> 16) + 127) / 128;
        const u32 limits = source.ControlWord(0x60);
        const u32 vertices_per_primitive = source.control[0x5f];
        const u32 topology = vertices_per_primitive == 3 ? 4 : vertices_per_primitive == 2 ? 1 : 0;
        mesh_registers = {topology | ((limits & 0x1ff) << 4) | (((limits >> 16) & 0x3ff) << 13),
                          threads << 20, threads | (lines << 12) | (lines << 22)};
        const size_t offset = source.ControlWord(0x64);
        if ((source.ControlWord(0x68) & 0x80) && offset <= host_program.size() &&
            host_program.size() - offset >= sizeof(mesh_primitive_header)) {
            std::memcpy(&mesh_primitive_header, host_program.data() + offset, sizeof(mesh_primitive_header));
        }
    }
    initial_offset = sizeof(sph);
    static const bool geometry_enabled = [] {   // as GraphicsEnvironment (EDEN_NX2_EXPERIMENTAL_GEOMETRY)
        const char* value = std::getenv("EDEN_NX2_EXPERIMENTAL_GEOMETRY");
        return !value || std::string_view{value} != "0";
    }();
    const bool geometry = stage == Shader::Stage::Geometry && geometry_enabled;
    InitializeNativeCode(std::move(lowered), Control2::GRAPHICS_HEADER, geometry);
    local_memory_size = static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
    early_fragment = false;   // where NVN2 keeps the early fragment test flag is not located (NVN: control +0x718)
}

NvnEnvironment::NvnEnvironment(const NvnShaderSource& source, const nvn_eden_query* query_,
                               std::shared_ptr<const LoweredNativeCode> lowered)
    : query{query_}, nvn_stage{source.nvn_stage} {
    nvn2_draw_parameters = source.nvn2;
    if (source.nvn2) {
        const auto c1_bytes = source.C1();
        c1.assign(c1_bytes.begin(), c1_bytes.end());
        InitNvn2(source, std::move(lowered));
        return;
    }
    // gpu_memory stays null and program_base 0: GenericEnvironment reads host_program instead
    const u32 start = (std::min<u32>)(source.ProgramStart(), static_cast<u32>(source.code.size()));
    host_program.assign(source.code.begin() + start, source.code.end());
    start_address = 0;
    texture_bound = 2;   // NVN's handle tables are c[2] (host dmsh_abi.h DMSH_C2_*)
    is_proprietary_driver = true;   // as GraphicsEnvironment / FileEnvironment derive it from texture_bound == 2
    has_hle_engine_state = false;   // c0 0x640/0x644/0x648 are written by the replay (S4 §3.5)
    viewport_transform_state = 1;
    const auto c1_bytes = source.C1();
    c1.assign(c1_bytes.begin(), c1_bytes.end());

    stage = StageFromNvn(nvn_stage);
    if (stage == Shader::Stage::Compute) {
        initial_offset = 0;
        workgroup_size = {source.ControlWord(Control::WORKGROUP),
                          source.ControlWord(Control::WORKGROUP + 4),
                          source.ControlWord(Control::WORKGROUP + 8)};
        // The NVN driver allocates shared memory in 256-byte units: the Maxwell path's QMD value
        // for the same shaders is 256 where the control section says 4 or 8 (B7, vulkan.bin of
        // 2026-11-12 run 3).
        shared_memory_size = (source.ControlWord(Control::SHARED_MEMORY) + 255u) & ~255u;
        // The control section's local memory field for compute is not located yet (S4 §6). The
        // Maxwell path's QMD gives 0 for all 25 Bayonetta 3 compute shaders it saw, so 0 when the
        // code has no local load or store (LDL / STL); a shader that has one keeps the largest
        // size Eden's SPIR-V declares without a dedicated buffer (logged: its SPIR-V may differ).
        local_memory_size = UsesLocalMemory(host_program) ? 0x4000 : 0;
        if (local_memory_size != 0) {
            LOG_WARNING(Shader, "NVN compute shader uses local memory: size 0x4000 assumed (the "
                                "control section's field is not known)");
        }
    } else {
        // Maxwell code comes in 32-byte bundles (a scheduling word, then 3 instructions) and the
        // flow analysis finds the scheduling words by address (Flow::Location::Align: offset % 32
        // == 0). A graphics program's code starts 0x50 bytes after its SPH on a bundle boundary,
        // so the SPH sits 0x10 past one, as in the GPU's program region. Read from 0 (as before
        // 2026-11-12 run 2) every scheduling word was decoded as an instruction. The hashed bytes
        // (from start_address) are the same.
        host_program.insert(host_program.begin(), GRAPHICS_START, u8{0});
        start_address = GRAPHICS_START;
        if (host_program.size() >= GRAPHICS_START + SPH_SIZE) {
            std::memcpy(&sph, host_program.data() + GRAPHICS_START, SPH_SIZE);
        }
        initial_offset = SPH_SIZE;
        local_memory_size = static_cast<u32>(sph.LocalMemorySize()) +
                            sph.common3.shader_local_memory_crs_size;
        early_fragment = stage == Shader::Stage::Fragment &&
                         source.control_size > Control::EARLY_FRAGMENT &&
                         source.control[Control::EARLY_FRAGMENT] != 0;
        // SPH shader type 0 in a vertex shader: Catherine: Full Body's GLSLC writes it in every
        // vertex shader (host shader_split.py sph_stage); a plain VertexB here. Two-part vertex
        // shaders (host dmsh_build.py:545-570) are not split off (not in Bayonetta 3).
    }
}

u32 NvnEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    u32 value{};
    if (cbuf_index == 1) {
        if (cbuf_offset + 4 <= c1.size()) {
            std::memcpy(&value, c1.data() + cbuf_offset, 4);
        }
    } else if (query && query->cbuf) {
        value = query->cbuf(query->user, nvn_stage >= 6 ? nvn_stage - 3 : nvn_stage, cbuf_index, cbuf_offset);
    }
    cbuf_values.emplace(MakeCbufKey(cbuf_index, cbuf_offset), value);
    return value;
}

bool NvnEnvironment::QueryTexture(u32 handle, int& target, int& vk_format) const {
    target = 0x1;
    vk_format = 0;
    return query && query->texture && query->texture(query->user, handle, &target, &vk_format);
}

Shader::TextureType NvnEnvironment::ReadTextureType(u32 handle) {
    int target, vk_format;
    const Shader::TextureType result =
        QueryTexture(handle, target, vk_format) ? TargetToType(target) : Shader::TextureType::Color2D;
    texture_types.emplace(handle, result);
    return result;
}

Shader::TexturePixelFormat NvnEnvironment::ReadTexturePixelFormat(u32 handle) {
    int target, vk_format;
    auto format = VideoCore::Surface::PixelFormat::A8B8G8R8_UNORM;
    if (QueryTexture(handle, target, vk_format) && vk_format != 0 && g_format_lookup) {
        format = g_format_lookup(vk_format);
    }
    const auto result = static_cast<Shader::TexturePixelFormat>(format);
    texture_pixel_formats.emplace(handle, result);
    return result;
}

bool NvnEnvironment::IsTexturePixelFormatInteger(u32 handle) {
    return VideoCore::Surface::IsPixelFormatInteger(
        static_cast<VideoCore::Surface::PixelFormat>(ReadTexturePixelFormat(handle)));
}

u32 NvnEnvironment::ReadViewportTransformState() {
    return viewport_transform_state;
}

std::optional<Shader::ReplaceConstant> NvnEnvironment::GetReplaceConstBuffer(u32 bank, u32 offset) {
    // The indirect record is GPU-written: the replay's c0 holds zero for its base
    // instance. Vulkan's draw built-ins carry the actual record. Vertex only;
    // compute's c0 has a different layout. Record replacements for cached recipes.
    if (!nvn2_draw_parameters || nvn_stage != 0 || bank != 0) {
        return std::nullopt;
    }
    Shader::ReplaceConstant value;
    switch (offset) {
    case 0x640: value = Shader::ReplaceConstant::BaseVertex; break;
    case 0x644: value = Shader::ReplaceConstant::BaseInstance; break;
    case 0x648: value = Shader::ReplaceConstant::DrawID; break;
    default: return std::nullopt;
    }
    cbuf_replacements.emplace((static_cast<u64>(bank) << 32) | offset, value);
    return value;
}

// NVN2: a trap's guard constant, from the bound constant buffer as the Maxwell path reads it from the engine's (none without a query or for task / mesh)
std::optional<u32> NvnEnvironment::ReadTrapGuardConstant(u32 bank, u32 offset) {
    if (!query || !query->cbuf) {
        return std::nullopt;
    }
    if (bank == 1) {
        if (offset + 4 > c1.size()) {
            return std::nullopt;
        }
        u32 value;
        std::memcpy(&value, c1.data() + offset, 4);
        return value;
    }
    return query->cbuf(query->user, nvn_stage >= 6 ? nvn_stage - 3 : nvn_stage, bank, offset);
}

} // namespace VideoCommon::Nvn
