// SPDX-License-Identifier: GPL-3.0-or-later
// NVN shader environment for Eden's recompiler (NXN1X docs/lane_b/SHADER_EDEN_B.md §2,
// reference: docs/verify/S4_shaders.md §3.1). A GenericEnvironment without gpu_memory: the program
// is a host copy of the NVN code area from its SPH (graphics) or first instruction (compute), so
// Analyze/Serialize/CanBeSerialized behave exactly as on the Maxwell path and the hash is the same.
#pragma once

#include <array>
#include <functional>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/nvn_shader_bridge.h"
#include "video_core/shader_environment.h"
#include "video_core/surface.h"

namespace VideoCommon::Nvn {

// The NVN control section fields the environment uses (pokopia nvn_shaders.h:54-68, host
// dm/gfx/tools/shader_split.py:47-48, dmsh_build.py:545-624), all little endian u32.
namespace Control {
constexpr u32 KEY_SIZE = 0x880;          // bytes copied (the host key's range, hle_nvn.c nvn_ctl_lens)
constexpr u32 PROGRAM_SIZE = 0x6f8;      // SPH and instructions
constexpr u32 C1_SIZE = 0x6fc;
constexpr u32 C1_OFFSET = 0x700;         // within the code area
constexpr u32 CODE_SIZE = 0x704;         // the code area's length
constexpr u32 PROGRAM_START = 0x708;     // graphics 0x30 (after the area header), compute 0x100 or 0
constexpr u32 EARLY_FRAGMENT = 0x718;    // byte
constexpr u32 WORKGROUP = 0x748;         // compute: x, y, z
constexpr u32 SHARED_MEMORY = 0x754;     // compute
} // namespace Control

// NVN2 (Switch 2) control sections, from the native host's own code: magic and version 0xc9 at +0 / +4 (host dm/gfx/nvn/hle_nvn.c:722 nvn2_ctl,
// host_nx2 dm/gfx/tools/dmsh_ampere.py:12-15), the key's 0x48 bytes (hle_nvn.c:721 NVN2_KEY_CTL), the code area's length at +0x44 (hle_nvn.c:725,
// dmsh_ampere.py:19-20), its c1 constant segment between +0x40 and +0x44 (dmsh_ampere.py:27-29), the stage byte at +0x56 (5 compute, 6 task, 7 mesh;
// host_nx2 dm/gfx/tools/shader_split.py:165-167 nvn2_stage), compute's thread group at +0x58 and shared memory at +0x64 (shader_split.py:2961-2964, the
// host's note: shared memory not yet seen nonzero). The code is Ampere machine code: graphics stages start with a 0x80-byte SPH (version 4), compute has
// no header (shader_split.py:140-141, dmsh_ampere.py:5-7).
namespace Control2 {
constexpr u32 MAGIC = 0x98761234;
constexpr u32 VERSION = 0xc9;
constexpr u32 KEY_SIZE = 0x48;
constexpr u32 C1_BEGIN = 0x40;
constexpr u32 CODE_SIZE = 0x44;
constexpr u32 STAGE = 0x56;              // byte
constexpr u32 WORKGROUP = 0x58;          // compute: x, y, z
constexpr u32 SHARED_MEMORY = 0x64;      // compute
constexpr u32 PUSH_WORDS = 0xa9;         // compute: push constant words, byte (dmsh_build.py:590-593, Ryujinx InstEmitAmpere.cs:260-276 AMapC0)
constexpr u32 C0_UBOS = 0xb1;            // compute c0 layout counts, bytes: uniform buffer, storage buffer, texture, separate texture, separate sampler, image slots
constexpr u32 C0_SSBOS = 0xb2;
constexpr u32 C0_TEXTURES = 0xb3;
constexpr u32 C0_SEP_TEXTURES = 0xb4;
constexpr u32 C0_SEP_SAMPLERS = 0xb5;
constexpr u32 C0_IMAGES = 0xb6;
constexpr u32 C0_SIZE = 0xba;            // compute c0 layout size, u16
constexpr u32 READ_SIZE = 0xbc;          // bytes copied: through C0_SIZE (sections are 0xd0 bytes or more, shader_split.py:105-106)
constexpr u32 GRAPHICS_HEADER = 0x80;    // the SPH in front of a graphics stage's instructions
} // namespace Control2

[[nodiscard]] bool IsNvn2Control(const void* control, u32 control_size);

// The NVN code and control section of one shader, kept from nvnProgramSetShaders on.
struct NvnShaderSource {
    std::array<u8, Control::KEY_SIZE> control{};
    u32 control_size{};
    std::vector<u8> code;   // the whole code area (header, SPH, instructions, c1)
    int nvn_stage{};        // NVNshaderStage (NVN2 adds 6 task, 7 mesh)
    bool nvn2{};            // an NVN2 control section (Control2): Ampere code, lowered as the Maxwell path lowers it

    [[nodiscard]] u32 ControlWord(u32 offset) const;
    [[nodiscard]] u32 ProgramStart() const;
    [[nodiscard]] std::span<const u8> C1() const;
};

[[nodiscard]] Shader::Stage StageFromNvn(int nvn_stage);

class NvnEnvironment final : public GenericEnvironment {
public:
    // query may be null (a pre-built recipe, or no draw yet): constant and texture reads then
    // answer 0 / Color2D / A8B8G8R8_UNORM, as for an unregistered ID (S4 §3.3).
    // NVN2: lowered is the shader's earlier lowering (LoweredNative() of an environment made before), or null to lower it here
    explicit NvnEnvironment(const NvnShaderSource& source, const nvn_eden_query* query,
                            std::shared_ptr<const LoweredNativeCode> lowered = {});
    ~NvnEnvironment() override = default;

    // A translation's queries happen only while it runs; the environment then keeps only the
    // recorded answers (they go to vulkan.bin with it).
    void SetQuery(const nvn_eden_query* q) noexcept {
        query = q;
    }

    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;
    Shader::TextureType ReadTextureType(u32 handle) override;
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 handle) override;
    bool IsTexturePixelFormatInteger(u32 handle) override;
    u32 ReadViewportTransformState() override;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;

    [[nodiscard]] int NvnStage() const noexcept {
        return nvn_stage;
    }
    [[nodiscard]] bool EarlyFragmentTests() const noexcept {
        return early_fragment;
    }

    // VkFormat -> PixelFormat, set once by the bridge from the device (maxwell_to_vk SurfaceFormat).
    static void SetFormatLookup(std::function<VideoCore::Surface::PixelFormat(int vk_format)> f);

protected:
    std::optional<u32> ReadTrapGuardConstant(u32 bank, u32 offset) override;

private:
    bool QueryTexture(u32 handle, int& target, int& vk_format) const;
    void InitNvn2(const NvnShaderSource& source, std::shared_ptr<const LoweredNativeCode> lowered);

    const nvn_eden_query* query{};
    std::vector<u8> c1;
    int nvn_stage{};
    bool nvn2_draw_parameters{};
    bool early_fragment{};
};

} // namespace VideoCommon::Nvn
