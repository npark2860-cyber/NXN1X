// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <openssl/evp.h>
#include <vector>

#include "common/cityhash.h"
#include "common/common_funcs.h"
#include "common/game_presets.h"
#include "common/hex_util.h"
#include "common/logging.h"
#include "common/lz4_compression.h"
#include "common/settings.h"
#include "common/swap.h"
#include "common/zbic.h"
#include "core/core.h"
#include "core/file_sys/patch_manager.h"
#include "core/hle/kernel/code_set.h"
#include "core/hle/kernel/k_page_table.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/loader/nso.h"
#include "core/memory.h"

#ifdef HAS_NCE
#include "core/arm/nce/patcher.h"
#include "core/arm/nce/x18_site_patcher.h"
#include "core/arm/nce/nx2_probe.h"
#if defined(_WIN32)
#include "core/arm/nce/hwa_pool.h"
#endif
#if defined(NXN1X_NVN_HOST)
#include "core/nvn/nvn_gate.h"
#include "core/nvn/sv1.h"
#include "core/nvn/nvn_layer.h"
#endif
#endif

namespace Loader {
namespace {
struct MODHeader {
    u32_le magic;
    u32_le dynamic_offset;
    u32_le bss_start_offset;
    u32_le bss_end_offset;
    u32_le eh_frame_hdr_start_offset;
    u32_le eh_frame_hdr_end_offset;
    u32_le module_offset; // Offset to runtime-generated module object. typically equal to .bss base
};
static_assert(sizeof(MODHeader) == 0x1c, "MODHeader has incorrect size.");

constexpr u32 PageAlignSize(u32 size) {
    return static_cast<u32>((size + Core::Memory::YUZU_PAGEMASK) & ~Core::Memory::YUZU_PAGEMASK);
}
} // Anonymous namespace

bool NSOHeader::IsSegmentCompressed(size_t segment_num) const {
    ASSERT_MSG(segment_num < 3, "Invalid segment {}", segment_num);
    return ((flags >> segment_num) & 1) != 0;
}

AppLoader_NSO::AppLoader_NSO(FileSys::VirtualFile file_) : AppLoader(std::move(file_)) {}

FileType AppLoader_NSO::IdentifyType(const FileSys::VirtualFile& in_file) {
    u32 magic = 0;
    if (in_file->ReadObject(&magic) != sizeof(magic)) {
        return FileType::Error;
    }

    if (Common::MakeMagic('N', 'S', 'O', '0') != magic) {
        return FileType::Error;
    }

    return FileType::NSO;
}

#ifdef HAS_NCE
namespace {
// titleplan: the text of each module when its NCE plan was made (planning pass), by module name
u64& PlannedTextHash(const std::string& name) {
    static std::unordered_map<std::string, u64> hashes;
    return hashes[name];
}
u64 TextHash(const std::vector<u8>& memory, const Kernel::CodeSet::Segment& code) {
    return Common::CityHash64(reinterpret_cast<const char*>(memory.data() + code.offset), code.size);
}
} // namespace
#endif

std::optional<VAddr> AppLoader_NSO::LoadModule(Kernel::KProcess& process, Core::System& system, const FileSys::VfsFile& nso_file, VAddr load_base, bool should_pass_arguments, bool load_into_process, std::optional<FileSys::PatchManager> pm, std::vector<Core::NCE::Patcher>* patches, s32 patch_index, u64 planning_program_id) {
    if (nso_file.GetSize() < sizeof(NSOHeader))
        return std::nullopt;
    NSOHeader nso_header{};
    if (sizeof(NSOHeader) != nso_file.ReadObject(&nso_header))
        return std::nullopt;
    if (nso_header.magic != Common::MakeMagic('N', 'S', 'O', '0'))
        return std::nullopt;
    if (nso_header.segments.empty())
        return std::nullopt;

    // Allocate some space at the beginning if we are patching in PreText mode.
    const size_t module_start = [&]() -> size_t {
#ifdef HAS_NCE
        if (patches && load_into_process) {
            auto* patch = &patches->operator[](patch_index);
            if (patch->GetPatchMode() == Core::NCE::PatchMode::PreText) {
                return patch->GetSectionSize();
            } else if (patch->GetPatchMode() == Core::NCE::PatchMode::Split) {
                return patch->GetPreSectionSize();
            }
        }
#endif
        return 0;
    }();

    auto const last_segment_it = &nso_header.segments[nso_header.segments.size() - 1];
    // Build program image directly in codeset memory :)
    Kernel::CodeSet codeset;
    codeset.memory.resize(module_start + last_segment_it->location + last_segment_it->size);
    std::vector<u8> compressed_data;
    for (size_t i = 0; i < nso_header.segments.size(); ++i) {
        const auto& segment = nso_header.segments[i];
        const bool compressed = nso_header.IsSegmentCompressed(i);
        const bool zbic = compressed && (nso_header.flags & (1U << 7)) != 0;
        const size_t stored = compressed ? nso_header.segments_compressed_size[i] : segment.size;
        const size_t address = module_start + segment.location;
        const size_t size = segment.size;
        if (segment.offset > nso_file.GetSize() || stored > nso_file.GetSize() - segment.offset ||
            address > codeset.memory.size() || size > codeset.memory.size() - address) {
            LOG_ERROR(Loader, "NSO {} segment {} exceeds file or image bounds", nso_file.GetName(), i);
            return std::nullopt;
        }

        auto* destination = codeset.memory.data() + address;
        if (compressed) {
            compressed_data.resize(stored);
        }
        auto* input = compressed ? compressed_data.data() : destination;
        if (stored && nso_file.Read(input, stored, segment.offset) != stored) {
            LOG_ERROR(Loader, "NSO {} segment {} short read (expected {} bytes)", nso_file.GetName(), i, stored);
            return std::nullopt;
        }
        if (compressed) {
            const bool valid = zbic ? nxn1x_decode_zbic(destination, size, input, stored)
                                    : stored <= std::numeric_limits<int>::max() &&
                                      size <= std::numeric_limits<int>::max() &&
                                      Common::Compression::DecompressDataLZ4(destination, size, input, stored) ==
                                          static_cast<int>(size);
            if (!valid) {
                LOG_ERROR(Loader, "NSO {} segment {} {} decode failed ({} bytes, expected {} bytes)",
                          nso_file.GetName(), i, zbic ? "ZBIC" : "LZ4", stored, size);
                return std::nullopt;
            }
        }
        if (zbic && (nso_header.flags & (1U << (3 + i))) != 0) {
            NSOHeader::SHA256Hash digest{};
            unsigned int length{};
            if (EVP_Digest(destination, size, digest.data(), &length, EVP_sha256(), nullptr) != 1 ||
                length != digest.size() || digest != nso_header.segment_hashes[i]) {
                LOG_ERROR(Loader, "NSO {} segment {} ZBIC SHA-256 mismatch", nso_file.GetName(), i);
                return std::nullopt;
            }
        }
        codeset.segments[i].addr = address;
        codeset.segments[i].offset = address;
        codeset.segments[i].size = size;
    }

    if (should_pass_arguments && !Settings::values.program_args.GetValue().empty()) {
        const auto arg_data{Settings::values.program_args.GetValue()};

        codeset.DataSegment().size += NSO_ARGUMENT_DATA_ALLOCATION_SIZE;
        NSOArgumentHeader args_header{NSO_ARGUMENT_DATA_ALLOCATION_SIZE, static_cast<u32_le>(arg_data.size()), {}};
        const auto end_offset = codeset.memory.size();
        codeset.memory.resize(u32(codeset.memory.size()) + NSO_ARGUMENT_DATA_ALLOCATION_SIZE);
        std::memcpy(codeset.memory.data() + end_offset, &args_header, sizeof(NSOArgumentHeader));
        std::memcpy(codeset.memory.data() + end_offset + sizeof(NSOArgumentHeader), arg_data.data(), arg_data.size());
    }

    codeset.DataSegment().size += nso_header.segments[2].bss_size;
    u32 image_size = PageAlignSize(u32(codeset.memory.size()) + nso_header.segments[2].bss_size);
    codeset.memory.resize(image_size);

    for (std::size_t i = 0; i < nso_header.segments.size(); ++i) {
        codeset.segments[i].size = PageAlignSize(codeset.segments[i].size);
    }

    // titleplan (NCE_CLEANUP_REVIEW.md 1, "structural hazard"): the planning pass (load_into_process false, which runs PatchText) has no program ID yet,
    // so a title patch keyed on it went in only when loading, after the NCE patch plan was made from the unpatched text: an svc/mrs/x18 it added was
    // never NCE-patched, and an instruction the plan patched would be overwritten by RelocateAndCopy's branch. The planning pass now gets the program ID
    // from the metadata (deconstructed_rom_directory.cpp) and the title patches below go in on both passes from the same inputs (program ID, build ID,
    // site words, env), so PatchText plans the patched text. Checked: the text's hash at the plan and before RelocateAndCopy (log "titleplan").
    // NXN1X_TITLE_PATCH_PLAN=0: the old order (title patches only when loading)
    static const bool title_patch_plan = [] {
        const char* const v = std::getenv("NXN1X_TITLE_PATCH_PLAN");
        return v == nullptr || v[0] != '0';
    }();
    const u64 program_id = load_into_process ? process.GetProgramId() : (title_patch_plan ? planning_program_id : 0);
#if defined(HAS_NCE) && defined(NXN1X_NVN_HOST)
    // the DK/OoT stubs read their state at 0 (= the game's own instruction) without the NVN host; Wanted() can change between the passes (Refuse while
    // planning), so it gates only the old order
    const bool title_patch_nvn = title_patch_plan || Core::NvnLayer::Wanted();
#endif

    // Apply patches if necessary
    const auto name = nso_file.GetName();
    // titleplan: also on the planning pass (deconstructed_rom_directory.cpp plan_pm), so the NCE plan sees the patched text; the NSO dump only when loading
    if (pm && (load_into_process || title_patch_plan) &&
        (pm->HasNSOPatch(nso_header.build_id, name) || (Settings::values.dump_nso && load_into_process))) {
        std::span<u8> patchable_section(codeset.memory.data() + module_start, codeset.memory.size() - module_start);
        std::vector<u8> pi_header(sizeof(NSOHeader) + patchable_section.size());
        std::memcpy(pi_header.data(), &nso_header, sizeof(NSOHeader));
        std::memcpy(pi_header.data() + sizeof(NSOHeader), patchable_section.data(),
                    patchable_section.size());

        pi_header = pm->PatchNSO(pi_header, name);

        std::copy(pi_header.begin() + sizeof(NSOHeader), pi_header.end(), patchable_section.data());
    }

    // Mario Kart World selects a 1- or 2-vblank frame mode in its main loop. Force its native
    // 30 FPS mode, or let it choose dynamically. In dynamic mode the game keeps a time account
    // (frame cost - interval * 16.67 ms) and advances 1 or 2 steps per frame, so it stays in real
    // time as long as the cost it reads is the real frame time: use the whole frame time and skip
    // the game's own wait for the second vblank, or every 2-step frame reads as 33 ms and the game
    // never leaves 30 FPS (found by SerkiGohan). Presents are then paced with interval 1.
    // Match the instruction sequences instead of relying on a title-version-specific address.
    const bool mkw_dynamic = Settings::values.mkw_dynamic_fps.GetValue();
    bool mkw_dynamic_fps_patch_applied = false;
    if ((mkw_dynamic || Settings::values.mkw_30fps.GetValue()) && name == "main" &&
        program_id == Settings::MarioKartWorldProgramId) {
        constexpr u32 ModeLoadInsn = 0xB9421268; // ldr w8, [x19, #0x210]
        const u32 mode = mkw_dynamic ? 0U : 2U;
        const u32 mode_instruction = 0x52800008U | (mode << 5); // mov w8, #mode
        const auto& text = nso_header.segments[0];
        u8* const code = codeset.memory.data() + module_start + text.location;
        const size_t count = text.size / sizeof(u32);
        const auto word = [code](size_t index) {
            u32 value;
            std::memcpy(&value, code + index * sizeof(u32), sizeof(value));
            return value;
        };
        size_t mode_site = 0;
        size_t mode_matches = 0;
        for (size_t i = 2; i + 8 < count; ++i) {
            if (word(i) != ModeLoadInsn || word(i - 2) != 0x6B01013F || // cmp w9, w1
                (word(i - 1) & 0xFF00001F) != 0x54000001 ||             // b.ne
                (word(i + 1) & 0xFF00001F) != 0x35000008) {             // cbnz w8
                continue;
            }
            for (size_t j = i + 2; j < i + 8; ++j) {
                if (word(j) == 0x52A84E08) { // mov w8, #60.0f
                    mode_site = i;
                    ++mode_matches;
                    break;
                }
            }
        }

        size_t frame_time_site = 0;
        size_t frame_time_matches = 0;
        if (mode_matches == 1 && mkw_dynamic) {
            for (size_t i = mode_site + 1; i < mode_site + 32 && i < count; ++i) {
                if ((word(i) & 0xFFFFFFE0) == 0xF9409660) { // ldr xN, [x19, #0x128]
                    frame_time_site = i;
                    ++frame_time_matches;
                }
            }
        }

        // The vsync function waits until 0.99 * interval / 60 s after the frame start:
        // ldr w8, [x19, #0x88] ; cmp w8, #2 ; b.lo <done> ; ucvtf ; mov w8, #60.0f
        size_t vblank_wait_site = 0;
        size_t vblank_wait_matches = 0;
        if (mkw_dynamic) {
            for (size_t i = 0; i + 4 < count; ++i) {
                if (word(i) == 0xB9408A68 && word(i + 1) == 0x7100091F &&
                    (word(i + 2) & 0xFF00001F) == 0x54000003 && word(i + 4) == 0x52A84E08) {
                    vblank_wait_site = i + 2;
                    ++vblank_wait_matches;
                }
            }
        }

        if (mode_matches == 1 &&
            (!mkw_dynamic || (frame_time_matches == 1 && vblank_wait_matches == 1))) {
            std::memcpy(code + mode_site * sizeof(u32), &mode_instruction,
                        sizeof(mode_instruction));
            if (mkw_dynamic) {
                const u32 whole_frame_time = 0xF9407660 | (word(frame_time_site) & 0x1F);
                std::memcpy(code + frame_time_site * sizeof(u32), &whole_frame_time,
                            sizeof(whole_frame_time));
                // b.lo -> b to the same target.
                const s32 offset = static_cast<s32>(word(vblank_wait_site) << 8) >> 13;
                const u32 branch = 0x14000000U | (static_cast<u32>(offset) & 0x03FFFFFFU);
                std::memcpy(code + vblank_wait_site * sizeof(u32), &branch, sizeof(branch));
                mkw_dynamic_fps_patch_applied = true;
            }
            if (load_into_process) LOG_INFO(Loader, "Mario Kart World: {} FPS mode enabled (main+{:#x})",
                     mkw_dynamic ? "dynamic" : "30",
                     text.location + mode_site * sizeof(u32));
        } else if (load_into_process) {
            LOG_WARNING(Loader,
                        "Mario Kart World FPS patch skipped (mode matches {}, frame-time matches "
                        "{}, vblank-wait matches {})",
                        mode_matches, frame_time_matches, vblank_wait_matches);
        }
    }

    // Hyrule Warriors' frame interval (core/hle/service/nx2_frame_interval.h). Its only copy into
    // the global is
    //   adrp x9, got; ...; ldr x9, [x9, #got]; ...; adrp x8, global; add x8, x8, #global;
    //   ldr w19, [x9]; str w19, [x8]
    // Nothing is patched; the present path writes both words with hyrule_dynamic_fps or
    // hyrule_30fps.
    u64 frame_interval_offset = 0;
    u64 frame_interval_source_offset = 0;
    if (name == "main" && process.GetProgramId() == Settings::HyruleWarriorsProgramId) {
        const auto& text = nso_header.segments[0];
        const u8* const image = codeset.memory.data() + module_start;
        const u64 image_end = codeset.memory.size() - module_start;
        const auto word = [image](u64 offset) {
            u32 value;
            std::memcpy(&value, image + offset, sizeof(value));
            return value;
        };
        const auto fits = [image_end](u64 offset, u64 size) {
            return image_end >= size && offset <= image_end - size;
        };
        const auto adrp_page = [](u64 pc, u32 insn) {
            s64 pages = static_cast<s64>((((insn >> 5) & 0x7FFFF) << 2) | ((insn >> 29) & 3));
            if (pages & (s64{1} << 20)) {
                pages -= s64{1} << 21;
            }
            return (pc & ~u64{0xFFF}) + static_cast<u64>(pages * 0x1000);
        };
        size_t matches = 0;
        for (u64 pc = text.location + 32; pc + 8 <= text.location + text.size; pc += 4) {
            if (word(pc) != 0xB9400133 || word(pc + 4) != 0xB9000113) { // ldr w19, [x9]; str w19, [x8]
                continue;
            }
            const u32 adrp_global = word(pc - 8);
            const u32 add_global = word(pc - 4);
            const u32 adrp_got = word(pc - 32);
            const u32 ldr_got = word(pc - 24);
            if ((adrp_global & 0x9F00001F) != 0x90000008 || (add_global & 0xFFC003FF) != 0x91000108 ||
                (adrp_got & 0x9F00001F) != 0x90000009 || (ldr_got & 0xFFC003FF) != 0xF9400129) {
                continue;
            }
            ++matches;
            frame_interval_offset = adrp_page(pc - 8, adrp_global) + ((add_global >> 10) & 0xFFF);
            const u64 got = adrp_page(pc - 32, adrp_got) + ((ldr_got >> 10) & 0xFFF) * 8;
            frame_interval_source_offset = 0;
            if (fits(got, sizeof(u64))) {
                // Not relocated yet: the entry still holds the module-relative address.
                std::memcpy(&frame_interval_source_offset, image + got, sizeof(u64));
            }
        }
        if (matches == 1 && fits(frame_interval_offset, sizeof(u32)) &&
            frame_interval_source_offset != 0 && fits(frame_interval_source_offset, sizeof(u32))) {
            LOG_INFO(Loader, "Hyrule Warriors frame interval at main+{:#x} (from main+{:#x})",
                     frame_interval_offset, frame_interval_source_offset);
        } else {
            LOG_WARNING(Loader, "Hyrule Warriors frame interval not found ({} matches)", matches);
            frame_interval_offset = frame_interval_source_offset = 0;
        }
    }

    // r63 realtime: Donkey Kong Bananza v458752 on the NVN host path (the old host's dm/games/040074A01BF12000/patches.py, ported there from DKB Recomp's
    // "realtime"). The game's clocks step 60 / fps a frame for the present interval its controller picked, and the host holds no frame for that interval,
    // so a slow frame stepped less than the time it took: slow motion when the game runs slow. Eight sites take the frame time the host measures instead
    // (host dm/gfx/nvn/hle_nvn.c rt_step_tick at each present: state +0xc0 float, the present interval x 60, the mean of the last 5 frames, each 1/240 to
    // 1/15 s; +0xc4 u32 the same as whole frames a second; 0 = every stub runs the game's own instruction). The stubs and the state (0x220 bytes, 16-byte
    // aligned after the stubs: the offsets hle_nvn.c writes reach +0x21c) both go into the zero padding after the text segment, so the module's image and
    // size stay as they are (11-16: a page appended after the data segment instead hung DK before its first SVC). The guest only reads the state (+0xc0,
    // +0xc4); the host writes it through the backing (NvnLayer::SetRealtimeState). Every site's instruction and the padding are checked first: any mismatch
    // patches nothing. DM_REALTIME=0 (rt_step_tick) keeps the state 0
    u64 realtime_state_offset = 0;
    // slot12 (log only): the main module's build ID, and for Ocarina of Time (Switch 2) the words at the two frame-step sites the nxnative host patches for
    // build e0f3ccfaf50065ae4ea01eb5aeb13b1a (#1543: main+0x6e6cc4 fdiv s8,s1,s0 = 0x1e201828; #1704: main+0x6e6c98 ldr s0,[x21,#0x2c8] = 0xbd42caa0). Nothing is patched
    if (name == "main" && load_into_process) {
        std::string bid;
        for (const u8 v : nso_header.build_id) bid += fmt::format("{:02x}", v);
        LOG_INFO(Loader, "main module build ID {} (program {:016X}, text {:#x} bytes)", bid, process.GetProgramId(), nso_header.segments[0].size);
        if (process.GetProgramId() == 0x0400486024DEC000ULL) {
            const u8* const img = codeset.memory.data() + module_start;
            const auto word = [&](u64 o) -> std::string {
                if (o + 4 > nso_header.segments[0].location + nso_header.segments[0].size) return "outside text";
                u32 v; std::memcpy(&v, img + o, sizeof v); return fmt::format("{:#010x}", v); };
            LOG_INFO(Loader, "Ocarina of Time frame-step sites: main+0x6e6cc4 {} (fdiv s8,s1,s0 is 0x1e201828), main+0x6e6c98 {} (ldr s0,[x21,#0x2c8] is 0xbd42caa0); log only",
                     word(0x6e6cc4), word(0x6e6c98));
        }
    }
#if defined(HAS_NCE) && defined(NXN1X_NVN_HOST)
    // On by default for Donkey Kong Bananza (11-16: the state in the text padding boots and stops the slow motion, NPARK confirmed; the appended page of the
    // first version hung DK before its first SVC). DM_REALTIME_PATCH=0 leaves the game's own steps. The sites are this build's: no other title has them
    const char* const realtime_env = std::getenv("DM_REALTIME_PATCH");
    const bool realtime_patch = realtime_env == nullptr || std::atoi(realtime_env) != 0;
    if (name == "main" && process.GetProgramId() == 0x040074A01BF12000ULL && Core::NvnLayer::Wanted() && load_into_process) {
        LOG_INFO(Loader, "realtime game step: {} for title {:016X} (on by default; DM_REALTIME_PATCH {}; DM_REALTIME_PATCH=0 turns it off)",
                 realtime_patch ? "on" : "off", process.GetProgramId(), realtime_env ? realtime_env : "unset");
    }
    if (name == "main" && program_id == 0x040074A01BF12000ULL && title_patch_nvn && realtime_patch) {
        static constexpr std::array<u8, 16> Build{0xE2, 0x5C, 0x90, 0xC1, 0xC1, 0x36, 0x63, 0x88,
                                                  0x85, 0x65, 0x9F, 0xD5, 0x5B, 0xF8, 0xA0, 0x3B};
        constexpr u64 Controller = 0x204a54;   // mov x0,x19 before the controller's rate call
        constexpr std::array<u64, 7> Clocks{0x1c8c9c, 0x2f2928, 0x2f2a80, 0x372060, 0x3722e0, 0x37232c, 0x6b8260};   // fdiv sD, s(60), s(fps)
        constexpr std::array<u32, 7> ClockInsn{0x1E201820, 0x1E201820, 0x1E201820, 0x1E201820, 0x1E201828, 0x1E201860, 0x1E201820};
        constexpr u64 StubBytes = 4 * (8 + 10 * Clocks.size());
        constexpr u64 StateBytes = 0x220;
        const auto& text = nso_header.segments[0];
        u8* const image = codeset.memory.data() + module_start;   // module-relative offsets from here
        const auto rd = [image](u64 o) { u32 v; std::memcpy(&v, image + o, sizeof v); return v; };
        const auto wr = [image](u64 o, u32 v) { std::memcpy(image + o, &v, sizeof v); };
        const auto b = [](u64 from, u64 to) { return 0x14000000U | (static_cast<u32>((static_cast<s64>(to) - static_cast<s64>(from)) >> 2) & 0x03FFFFFFU); };
        const auto cbz = [](u32 rt, u64 from, u64 to) { return 0x34000000U | ((static_cast<u32>((static_cast<s64>(to) - static_cast<s64>(from)) >> 2) & 0x7FFFFU) << 5) | rt; };
        const auto ldrw = [](u32 t, u32 n, u32 off) { return 0xB9400000U | ((off / 4) << 10) | (n << 5) | t; };
        const auto adrp = [](u32 r, u64 pc, u64 target) {
            const u32 imm = static_cast<u32>(static_cast<s64>(target >> 12) - static_cast<s64>(pc >> 12)) & 0x1FFFFFU;
            return 0x90000000U | ((imm & 3) << 29) | ((imm >> 2) << 5) | r;
        };
        const u64 stub = (static_cast<u64>(text.location) + text.size + 3) & ~u64{3};
        const u64 stub_limit = (std::min)(static_cast<u64>(text.location) + PageAlignSize(text.size), static_cast<u64>(nso_header.segments[1].location));
        const u64 state = (stub + StubBytes + 15) & ~u64{15};   // in the padding, after the stubs
        int sites_ok = std::equal(Build.begin(), Build.end(), nso_header.build_id.begin()) && rd(Controller) == 0xAA1303E0U;
        for (size_t i = 0; i < Clocks.size(); ++i) {
            sites_ok += rd(Clocks[i]) == ClockInsn[i];
        }
        bool room = state + StateBytes <= stub_limit && (module_start & 0xFFF) == 0;
        for (u64 o = stub; room && o < state + StateBytes; o += 4) {
            room = rd(o) == 0;
        }
        if (sites_ok == 8 && room) {
            const u32 ps = static_cast<u32>(state & 0xFFF);   // the state's offset in the page ADRP gives (ldr reaches it and the fields after it)
            // The controller's rate: the host's whole frames a second go to its +0x50 and +0x54 before its rate call (x8 is dead there)
            u64 s = stub;
            wr(s, adrp(8, s, state)); wr(s + 4, ldrw(8, 8, ps + 0xc4)); wr(s + 8, cbz(8, s + 8, s + 0x18)); wr(s + 0xc, 0x2A0803E1U);
            wr(s + 0x10, 0xB9005281U); wr(s + 0x14, 0xB9005681U); wr(s + 0x18, 0xAA1303E0U); wr(s + 0x1c, b(s + 0x1c, Controller + 4));
            wr(Controller, b(Controller, s));
            s += 0x20;
            // The clocks: fdiv sD, s60, s(fps) becomes the host's step x 60 (state +0xc0); x9 saved on the stack, the flags left alone (cbz)
            for (const u64 off : Clocks) {
                const u32 orig = rd(off);
                wr(s, 0xF81F0FE9U); wr(s + 4, adrp(9, s + 4, state)); wr(s + 8, ldrw(9, 9, ps + 0xc0)); wr(s + 0xc, cbz(9, s + 0xc, s + 0x1c));
                wr(s + 0x10, 0x1E270000U | (9U << 5) | (orig & 0x1F)); wr(s + 0x14, 0xF84107E9U); wr(s + 0x18, b(s + 0x18, off + 4));
                wr(s + 0x1c, 0xF84107E9U); wr(s + 0x20, orig); wr(s + 0x24, b(s + 0x24, off + 4));
                wr(off, b(off, s));
                s += 0x28;
            }
            realtime_state_offset = state;
#if NXN1X_DIAGNOSTICS   // dkctl is log only (NCE_CLEANUP_REVIEW.md P3): diagnostic builds. titleplan: its mrs cntvct_el0 is now NCE-planned (the guest counter, not raw)
            // dkctl (11-21, NPARK: Donkey Kong Bananza sets present interval 1 -> 3 at its title and stays at 20 fps; log only, DM_DK_CTL_LOG=0 off): the speed
            // controller's instruction at main+0x204c20 is moved into a stub that first snapshots x0-x30, d0-d7 and the raw CNTVCT_EL0 into a record after the
            // state (state + 0x220: +0 sequence, +8 counter, +16 x0..x30, +264 d0..d7); hle_nvn.c rt_step_tick logs it with the host clock and guest_ns. Only a
            // non-PC-relative instruction is moved; the words around it are logged for disassembly either way
            {
                constexpr u64 Ctl = 0x204c20;
                std::string around;
                for (u64 o = Ctl - 0x40; o <= Ctl + 0x40; o += 4) around += fmt::format("{}{:08x}", o == Ctl ? " [" : " ", rd(o)) + (o == Ctl ? "]" : "");
                if (load_into_process) LOG_INFO(Loader, "dkctl: main+{:#x}-{:#x}:{}", Ctl - 0x40, Ctl + 0x40, around);
                const char* const ce = std::getenv("DM_DK_CTL_LOG");
                const u32 orig = rd(Ctl);
                const bool pcrel = (orig & 0x1F000000U) == 0x10000000U || (orig & 0x7C000000U) == 0x14000000U || (orig & 0xFF000010U) == 0x54000000U ||
                                   (orig & 0x7E000000U) == 0x34000000U || (orig & 0x7E000000U) == 0x36000000U || (orig & 0x3B000000U) == 0x18000000U;
                // the record must be guest-writable: the text padding is read-only to the guest (the first dkctl build hung DK at boot: the stub's stores
                // faulted). It goes at the end of the data segment's last page, after .bss (writable, unused); its offset from the state is left at
                // state + 0x220 for the host (hle_nvn.c dkctl_log), and the stub's code stays in the text padding after it
                const auto& ds = nso_header.segments[2];
                const u64 bss_end = static_cast<u64>(ds.location) + ds.size + ds.bss_size;
                const u64 page_end = (bss_end + 0xFFF) & ~u64{0xFFF};
                const u64 diag = (page_end - 0x150) & ~u64{15}, stub2 = state + StateBytes + 0x10, stub2_end = stub2 + 4 * 40;
                bool room2 = stub2_end <= stub_limit && diag >= bss_end && module_start + page_end <= codeset.memory.size();
                for (u64 o = state + StateBytes; room2 && o < stub2_end; o += 4) room2 = rd(o) == 0;
                for (u64 o = diag; room2 && o < diag + 0x150; o += 4) room2 = rd(o) == 0;

                if (!(ce && *ce == '0') && !pcrel && room2) {
                    { const u64 delta = diag - state; std::memcpy(image + state + StateBytes, &delta, sizeof delta); }   // for the host: where the record is
                    const auto stpx = [](u32 t, u32 t2, u32 n, u32 off) { return 0xA9000000U | (((off / 8) & 0x7FU) << 15) | (t2 << 10) | (n << 5) | t; };
                    const auto stpd = [](u32 t, u32 t2, u32 n, u32 off) { return 0x6D000000U | (((off / 8) & 0x7FU) << 15) | (t2 << 10) | (n << 5) | t; };
                    const auto strx = [](u32 t, u32 n, u32 off) { return 0xF9000000U | ((off / 8) << 10) | (n << 5) | t; };
                    const auto ldrx = [](u32 t, u32 n, u32 off) { return 0xF9400000U | ((off / 8) << 10) | (n << 5) | t; };
                    const auto addi = [](u32 d, u32 n, u32 imm) { return 0x91000000U | ((imm & 0xFFFU) << 10) | (n << 5) | d; };
                    u64 s2 = stub2; const auto put = [&](u32 w) { wr(s2, w); s2 += 4; };
                    put(0xF81F0FE9U);                                   // str x9, [sp, #-16]!
                    put(adrp(9, s2, diag)); put(addi(9, 9, static_cast<u32>(diag & 0xFFF)));
                    for (u32 r = 0; r < 8; r += 2) put(stpx(r, r + 1, 9, 16 + 8 * r));
                    put(strx(8, 9, 16 + 8 * 8));
                    for (u32 r = 10; r < 30; r += 2) put(stpx(r, r + 1, 9, 16 + 8 * r));
                    put(strx(30, 9, 16 + 8 * 30));
                    put(0xF94003EAU); put(strx(10, 9, 16 + 8 * 9));    // ldr x10, [sp]: the game's x9
                    for (u32 r = 0; r < 8; r += 2) put(stpd(r, r + 1, 9, 264 + 8 * r));
                    put(0xD53BE04AU); put(strx(10, 9, 8));             // mrs x10, cntvct_el0
                    put(ldrx(10, 9, 0)); put(addi(10, 10, 1)); put(strx(10, 9, 0));   // the sequence, after the record
                    put(ldrx(10, 9, 16 + 8 * 10));                      // the game's x10 back
                    put(0xF84107E9U);                                   // ldr x9, [sp], #16
                    put(orig); put(b(s2, Ctl + 4));
                    wr(Ctl, b(Ctl, stub2));
                    if (load_into_process) LOG_INFO(Loader, "dkctl: main+{:#x} ({:#010x}) snapshots registers into main+{:#x} (stub main+{:#x}, {} words)", Ctl, orig, diag, stub2, (s2 - stub2) / 4);
                } else {
                    if (load_into_process) LOG_INFO(Loader, "dkctl: main+{:#x} ({:#010x}) not hooked ({})", Ctl, orig, ce && *ce == '0' ? "DM_DK_CTL_LOG=0" : pcrel ? "PC-relative" : "no room");
                }
            }
#endif
            if (load_into_process) {
                LOG_INFO(Loader, "realtime game step: 8/8 sites patched (stubs at main+{:#x}), state at main+{:#x}", stub, state);
            }
        } else if (load_into_process) {
            LOG_WARNING(Loader, "realtime game step: not installed ({}/8 sites as expected, stub room {}); Donkey Kong Bananza keeps its own steps", sites_ok, room);
        }
    }
    // ootspeed (11-21, NPARK: Ocarina of Time runs in slow motion below 60 fps): its frame step is fdiv s8, s1, s0 at main+0x6e6cc4 (s1 = 60, or 30 in one mode by the
    // fcsel before it; s0 = the target rate), a constant: a frame that took longer than the target stepped the game by the target's time only. As nxnative #1543 (the Mac
    // host, same build e0f3ccfa): step = max(s1 / s0, s1 x the measured frame time), so it is never shorter than the game's own. The measured time is the host's mean frame
    // time in seconds at state +0xc8 (hle_nvn.c rt_step_tick: the mean of the last 5 presents, each 1/240 to 1/15 s, so the step stays <= 4 at s1 = 60; 0 = the game's own
    // step). s0, x16 and x17 are free there (#1543 uses them the same way). The stub and the state (0x220 bytes) go into the zero padding after the text segment, as Donkey
    // Kong's; the build ID, the site's word and the padding are checked first. DM_REALTIME_PATCH=0 leaves the game's own step
    if (name == "main" && process.GetProgramId() == 0x0400486024DEC000ULL && Core::NvnLayer::Wanted() && load_into_process) {
        LOG_INFO(Loader, "ootspeed: realtime game step {} for Ocarina of Time (DM_REALTIME_PATCH=0 turns it off)", realtime_patch ? "on" : "off");
    }
    if (name == "main" && program_id == 0x0400486024DEC000ULL && title_patch_nvn && realtime_patch) {
        static constexpr std::array<u8, 16> Build{0xE0, 0xF3, 0xCC, 0xFA, 0xF5, 0x00, 0x65, 0xAE,
                                                  0x4E, 0xA0, 0x1E, 0xB5, 0xAE, 0xB1, 0x3B, 0x1A};
        constexpr u64 Site = 0x6e6cc4;
        constexpr u32 SiteInsn = 0x1E201828U;   // fdiv s8, s1, s0
        constexpr u64 StubBytes = 4 * 6;
        constexpr u64 StateBytes = 0x220;
        const auto& text = nso_header.segments[0];
        u8* const image = codeset.memory.data() + module_start;
        const auto rd = [image](u64 o) { u32 v; std::memcpy(&v, image + o, sizeof v); return v; };
        const auto wr = [image](u64 o, u32 v) { std::memcpy(image + o, &v, sizeof v); };
        const auto b = [](u64 from, u64 to) { return 0x14000000U | (static_cast<u32>((static_cast<s64>(to) - static_cast<s64>(from)) >> 2) & 0x03FFFFFFU); };
        const auto adrp = [](u32 r, u64 pc, u64 target) {
            const u32 imm = static_cast<u32>(static_cast<s64>(target >> 12) - static_cast<s64>(pc >> 12)) & 0x1FFFFFU;
            return 0x90000000U | ((imm & 3) << 29) | ((imm >> 2) << 5) | r;
        };
        const u64 stub = (static_cast<u64>(text.location) + text.size + 3) & ~u64{3};
        const u64 stub_limit = (std::min)(static_cast<u64>(text.location) + PageAlignSize(text.size), static_cast<u64>(nso_header.segments[1].location));
        const u64 state = (stub + StubBytes + 15) & ~u64{15};
        const bool site_ok = std::equal(Build.begin(), Build.end(), nso_header.build_id.begin()) && Site + 4 <= text.location + text.size && rd(Site) == SiteInsn;
        bool room = state + StateBytes <= stub_limit && (module_start & 0xFFF) == 0;
        for (u64 o = stub; room && o < state + StateBytes; o += 4) {
            room = rd(o) == 0;
        }
        if (site_ok && room) {
            const u32 ps = static_cast<u32>(state & 0xFFF) + 0xc8;   // ldr s0, [x16, #ps]: the state's +0xc8 in the page ADRP gives (imm12 x 4 reaches 16 KB)
            u64 s = stub;
            wr(s, SiteInsn);                                          // fdiv s8, s1, s0 (the game's step)
            wr(s + 4, adrp(16, s + 4, state));                        // adrp x16, state
            wr(s + 8, 0xBD400000U | ((ps / 4) << 10) | (16U << 5) | 0U);   // ldr s0, [x16, #ps]: the measured frame time, seconds
            wr(s + 0xc, 0x1E210800U);                                 // fmul s0, s0, s1
            wr(s + 0x10, 0x1E204908U);                                // fmax s8, s8, s0
            wr(s + 0x14, b(s + 0x14, Site + 4));
            wr(Site, b(Site, s));
            realtime_state_offset = state;
            if (load_into_process) LOG_INFO(Loader, "ootspeed: main+{:#x} patched (stub main+{:#x}, state main+{:#x})", Site, stub, state);
            // ootdlss (11-21, NPARK: DLSS in place of the upscale network, gfx_vk_exec.c ootdl_*): the scene camera's sub-pixel jitter, as Longshot's nx2/oot_jitter.py
            // (refer\Longshot, GPL-3.0; same build e0f3ccfa = 1.1.1). The game fills each view's uniform buffers in main+0x6bc520, loading the view's projection
            // (view x28, rows at +0x1a8) at three sites; after each the jitter goes on the third lane of rows 0 and 1 (P[0][2], P[1][2]) in the registers, so the GPU's
            // buffers get the jittered matrix and what is made from it while the CPU's views keep theirs. Each stub runs the site's own instruction, then adds the
            // host's jitter (state +0xd0 x, +0xd4 y, clip units; +0xd8 on) only for a perspective projection (w row 0 0 -1 0, P[0][0] > 0) of 16:9 (|P11/P00 - 16/9|
            // x 100 < 1, with +0xdc = 16/9 and +0xe0 = 100 here): Longshot's dlss_jitter_clip. Only NZCV-free instructions (cbz/cbnz/tbnz); x16, x17, q30, q31 kept
            // on the stack. +0xe4 = 'OJIT' tells the host the sites are in. Each site's word is checked first (another build: left out). DM_VK_OOT_DLSS=0: not installed
            {
                const char* const de = std::getenv("DM_VK_OOT_DLSS");
                struct JSite { u64 off; u32 word; const u32* body; size_t n; };
                // llvm-mc (aarch64) of the stub bodies, 11-21: words 4 and 5 are adrp x16, state / add x16, x16, #lo12 (filled below)
                static constexpr u32 S1[] = {0xd100c3ffU, 0xa90047f0U, 0x3d8007ffU, 0x3d800bfeU, 0x90000010U, 0x91000210U, 0xb940da11U, 0x340003f1U, 0xf940ef91U, 0xb50003b1U,
                    0xb941e791U, 0x35000371U, 0xbd41e39fU, 0x1e3e101eU, 0x1e3e3bffU, 0x1e2603f1U, 0x350002d1U, 0xb941ab91U, 0x34000291U, 0x37f80271U, 0xbd41ab9fU, 0xbd41bf9eU,
                    0x1e3f1bdeU, 0xbd40de1fU, 0x1e3f3bdeU, 0x1e20c3deU, 0xbd40e21fU, 0x1e3f0bdeU, 0x1e3903d1U, 0x35000131U, 0xbd40d21fU, 0x5e14067eU, 0x1e3f2bdeU, 0x6e1407d3U,
                    0xbd40d61fU, 0x5e14069eU, 0x1e3f2bdeU, 0x6e1407d4U, 0x3dc00bfeU, 0x3dc007ffU, 0xa94047f0U, 0x9100c3ffU};   // v19 += x, v20 += y
                static constexpr u32 S2[] = {0xd100c3ffU, 0xa90047f0U, 0x3d8007ffU, 0x3d800bfeU, 0x90000010U, 0x91000210U, 0xb940da11U, 0x34000371U, 0xf940ef91U, 0xb5000331U,
                    0xb941e791U, 0x350002f1U, 0xbd41e39fU, 0x1e3e101eU, 0x1e3e3bffU, 0x1e2603f1U, 0x35000251U, 0xb941ab91U, 0x34000211U, 0x37f801f1U, 0xbd41ab9fU, 0xbd41bf9eU,
                    0x1e3f1bdeU, 0xbd40de1fU, 0x1e3f3bdeU, 0x1e20c3deU, 0xbd40e21fU, 0x1e3f0bdeU, 0x1e3903d1U, 0x350000b1U, 0xbd40d21fU, 0x5e14075eU, 0x1e3f2bdeU, 0x6e1407daU,
                    0x3dc00bfeU, 0x3dc007ffU, 0xa94047f0U, 0x9100c3ffU};   // v26 += x
                static constexpr u32 S3[] = {0xd100c3ffU, 0xa90047f0U, 0x3d8007ffU, 0x3d800bfeU, 0x90000010U, 0x91000210U, 0xb940da11U, 0x34000371U, 0xf940ef91U, 0xb5000331U,
                    0xb941e791U, 0x350002f1U, 0xbd41e39fU, 0x1e3e101eU, 0x1e3e3bffU, 0x1e2603f1U, 0x35000251U, 0xb941ab91U, 0x34000211U, 0x37f801f1U, 0xbd41ab9fU, 0xbd41bf9eU,
                    0x1e3f1bdeU, 0xbd40de1fU, 0x1e3f3bdeU, 0x1e20c3deU, 0xbd40e21fU, 0x1e3f0bdeU, 0x1e3903d1U, 0x350000b1U, 0xbd40d61fU, 0x5e1404deU, 0x1e3f2bdeU, 0x6e1407c6U,
                    0x3dc00bfeU, 0x3dc007ffU, 0xa94047f0U, 0x9100c3ffU};   // v6 += y
                const JSite sites[3] = {{0x6bc708, 0xad415915U, S1, sizeof S1 / 4},   // ldp q21, q22, [x8, #0x20] (rows 0 and 1 already in q19, q20)
                                        {0x6bcf8c, 0x3dc0013aU, S2, sizeof S2 / 4},   // ldr q26, [x9] (row 0)
                                        {0x6bcf9c, 0x3dc00126U, S3, sizeof S3 / 4}};  // ldr q6, [x9] (row 1)
                const u64 jstub = state + StateBytes + 0x10;
                u64 need = 0; for (const auto& js : sites) need += (js.n + 2) * 4;
                bool jroom = jstub + need <= stub_limit;
                for (u64 o = jstub; jroom && o < jstub + need; o += 4) jroom = rd(o) == 0;
                int words_ok = 0; for (const auto& js : sites) words_ok += js.off + 4 <= text.location + text.size && rd(js.off) == js.word;
                if (de && *de == '0') {
                    if (load_into_process) LOG_INFO(Loader, "ootdlss: camera jitter sites not installed (DM_VK_OOT_DLSS=0)");
                } else if (words_ok == 3 && jroom) {
                    u64 s = jstub;
                    for (const auto& js : sites) {
                        const u64 at = s;
                        wr(s, js.word); s += 4;   // the site's own instruction first
                        for (size_t i = 0; i < js.n; ++i, s += 4) {
                            u32 w = js.body[i];
                            if (i == 4) w = adrp(16, s, state);
                            else if (i == 5) w = 0x91000000U | (static_cast<u32>(state & 0xFFF) << 10) | (16U << 5) | 16U;   // add x16, x16, #lo12(state)
                            wr(s, w);
                        }
                        wr(s, b(s, js.off + 4)); s += 4;
                        wr(js.off, b(js.off, at));
                    }
                    const float aspect = 16.0f / 9.0f, scale = 100.0f; const u32 tag = 0x54494A4FU;   // 'OJIT'
                    std::memcpy(image + state + 0xdc, &aspect, 4); std::memcpy(image + state + 0xe0, &scale, 4); std::memcpy(image + state + 0xe4, &tag, 4);
                    if (load_into_process) LOG_INFO(Loader, "ootdlss: camera jitter at main+0x6bc708, main+0x6bcf8c, main+0x6bcf9c (stubs main+{:#x}, {} bytes; off until DLSS runs)", jstub, need);
                } else if (load_into_process) {
                    LOG_WARNING(Loader, "ootdlss: camera jitter not installed ({}/3 site words as expected: {:08x} {:08x} {:08x}, stub room {}); DLSS runs without jitter",
                                words_ok, rd(0x6bc708), rd(0x6bcf8c), rd(0x6bcf9c), jroom);
                }
                // resfix (11-21, NPARK: "lines across the picture, like a notebook" with DLSS): the game's dynamic resolution took the scene to 854x480, under
                // DLSS's smallest input (1280x720), and srfit's 1.5x resample of a jittered picture made fixed stripes that moved with the jitter. The scene is
                // now a fixed share of the picture while DLSS runs, as NX2navi's res_fixed (refer\NX2navi-main recomp/patches-v111.txt, MIT; same build 1.1.1):
                // main+0x805bd0 tbz w8, #0 (config +0x94 bit 0: the GPU-time controller) takes the fixed path; main+0x80641c csel x8, x9, x8, hi returns state 0
                // so the caller (main+0x31d3ee0) applies the scale each frame; main+0x806420 mov w9, #1.0 takes the scale from state +0xec (1.0 = the game's).
                // State +0xe8: on (written by gfx_vk_exec.c ootdl_resfix only while DLSS runs, after the controller ran a while), +0xf0 'ORES' = installed.
                // Each word is checked; x16 is kept on the stack, cbz/cbnz/tbz and ldr/str leave NZCV alone (the csel reads it). DM_VK_OOT_RESFIX=0: not installed
                {
                    const char* const re = std::getenv("DM_VK_OOT_RESFIX");
                    constexpr u64 RA = 0x805bd0, RB = 0x80641c, RC = 0x806420, RFar = 0x80640c;
                    constexpr u32 WA = 0x360041e8U, WB = 0x9a888128U, WC = 0x52a7f009U;
                    const u64 rstub = (jstub + need + 15) & ~u64{15};
                    constexpr u64 rbytes = 4 * (9 + 10 + 3);   // A 9 words, B 10, C 3
                    bool rroom = rstub + rbytes <= stub_limit;
                    for (u64 o = rstub; rroom && o < rstub + rbytes; o += 4) rroom = rd(o) == 0;
                    const bool rwords = RC + 4 <= text.location + text.size && rd(RA) == WA && rd(RB) == WB && rd(RC) == WC;
                    const u32 lo = static_cast<u32>(state & 0xFFF);
                    const auto ldr_w = [](u32 rt, u32 rn, u32 off) { return 0xB9400000U | ((off / 4) << 10) | (rn << 5) | rt; };
                    const auto cbnz16 = [](u64 from, u64 to) { return 0x35000000U | ((static_cast<u32>((static_cast<s64>(to) - static_cast<s64>(from)) >> 2) & 0x7FFFFU) << 5) | 16U; };
                    const auto tbz8 = [](u64 from, u64 to) { return 0x36000000U | ((static_cast<u32>((static_cast<s64>(to) - static_cast<s64>(from)) >> 2) & 0x3FFFU) << 5) | 8U; };
                    constexpr u32 Push16 = 0xF81F0FF0U, Pop16 = 0xF84107F0U;   // str x16, [sp, #-16]! / ldr x16, [sp], #16
                    if (re && *re == '0') {
                        if (load_into_process) LOG_INFO(Loader, "resfix: fixed scene share not installed (DM_VK_OOT_RESFIX=0)");
                    } else if (rwords && rroom) {
                        u64 s = rstub;
                        const u64 a = s;   // A: the controller's branch
                        wr(s, Push16); wr(s + 4, adrp(16, s + 4, state)); wr(s + 8, ldr_w(16, 16, lo + 0xe8)); wr(s + 0xc, cbnz16(s + 0xc, s + 0x1c));
                        wr(s + 0x10, Pop16); wr(s + 0x14, tbz8(s + 0x14, s + 0x20)); wr(s + 0x18, b(s + 0x18, RA + 4));
                        wr(s + 0x1c, Pop16); wr(s + 0x20, b(s + 0x20, RFar)); s += 0x24;
                        const u64 bb = s;   // B: the returned state
                        wr(s, Push16); wr(s + 4, adrp(16, s + 4, state)); wr(s + 8, ldr_w(16, 16, lo + 0xe8)); wr(s + 0xc, cbnz16(s + 0xc, s + 0x1c));
                        wr(s + 0x10, Pop16); wr(s + 0x14, WB); wr(s + 0x18, b(s + 0x18, RB + 4));
                        wr(s + 0x1c, Pop16); wr(s + 0x20, 0xAA1F03E8U); wr(s + 0x24, b(s + 0x24, RB + 4)); s += 0x28;   // mov x8, xzr
                        const u64 cc = s;   // C: the fixed path's scale
                        wr(s, adrp(9, s, state)); wr(s + 4, ldr_w(9, 9, lo + 0xec)); wr(s + 8, b(s + 8, RC + 4)); s += 0xc;
                        wr(RA, b(RA, a)); wr(RB, b(RB, bb)); wr(RC, b(RC, cc));
                        const u32 one = 0x3F800000U, off = 0, tag = 0x5345524FU;   // 'ORES'
                        std::memcpy(image + state + 0xe8, &off, 4); std::memcpy(image + state + 0xec, &one, 4); std::memcpy(image + state + 0xf0, &tag, 4);
                        if (load_into_process) LOG_INFO(Loader, "resfix: fixed scene share at main+0x805bd0, main+0x80641c, main+0x806420 (stubs main+{:#x}; off until DLSS runs)", rstub);
                    } else if (load_into_process) {
                        LOG_WARNING(Loader, "resfix: not installed (site words {:08x} {:08x} {:08x}, expected {:08x} {:08x} {:08x}; stub room {}); the game's dynamic resolution stays",
                                    rd(RA), rd(RB), rd(RC), WA, WB, WC, rroom);
                    }
                }
            }
        } else if (load_into_process) {
            LOG_WARNING(Loader, "ootspeed: not installed (build and site {}, stub room {}); Ocarina of Time keeps its own step", site_ok ? "as expected" : "differ", room);
        }
    }
#endif

#ifdef HAS_NCE
    // If we are computing the process code layout and using nce backend, patch.
    const auto& code = codeset.CodeSegment();
    auto* patch = patches ? &patches->operator[](patch_index) : nullptr;
#if defined(_WIN32)
    bool pool_retry = false;
    if (patch && name == "main") {
        namespace Pool = Core::NCE::HwaPool;
        if (load_into_process) {
            Pool::probe_base.store(0);
            Pool::reports.store(0);
        }
        // stubsvc: no program ID test here. The planning pass (deconstructed_rom_directory.cpp: LoadModule with load_into_process false, which runs
        // PatchText) comes before LoadFromMetadata, so the process has no program ID yet: the stub went in only in the loading pass, after the
        // patch plan, and its svc #0xb (SleepThread) stayed a raw svc that never reached the kernel (svcprobe: 0 SleepThread calls with its 1 ms;
        // 957 retries spun in microseconds). Install itself only patches Hyrule Warriors' build (its build ID and the code words it checks)
        {
            pool_retry = Pool::Install(std::span{codeset.memory}.subspan(code.offset, code.size),
                                       nso_header.segments[0].size, nso_header.build_id);
            if (load_into_process) {
                LOG_INFO(Loader, "HWAoI pool-pressure retry installed={}", pool_retry);
            }
        }
    }
#endif
    if (patch && !load_into_process) {
        //Set module ID using build_id from the NSO header
        patch->SetModuleID(nso_header.build_id);
#if defined(NXN1X_NVN_HOST)
        // NVN path (NVN_HOST_BRIDGE.md 4.1): the definitions of this module the patch takes over (empty off the NVN path)
        const auto nvn_hooks = Core::NvnLayer::PlanHooks(codeset.memory, code, name);
#else
        const std::vector<Core::NCE::EntryHook> nvn_hooks;
#endif
        PlannedTextHash(name) = TextHash(codeset.memory, code);   // titleplan: the text PatchText plans
        // Patch SVCs and MRS calls in the guest code
        while (!patch->PatchText(codeset.memory, code, nvn_hooks)) {
            patch = &patches->emplace_back();
            patch->SetModuleID(nso_header.build_id);  // In case the patcher is changed for big modules, the new patcher should also have the build_id
        }
    } else if (patch) {
#if defined(_WIN32)
        // Collect before ordinary NCE relocation rewrites operand-specific system-register sites.
        // The IMP-006 fallback owns only decoder-supported ordinary instructions touching guest x18.
        auto x18_sites = Core::NCE::X18SitePatcher::Collect(codeset.memory, code);
        Core::NCE::Nx2Probe::AddSites(name, load_base, codeset.memory, code, x18_sites);
        if (pool_retry) {
            namespace Pool = Core::NCE::HwaPool;
            for (auto [pc, op] : Pool::FailureSites) {
                x18_sites.push_back({pc / 4, op, false});
            }
            x18_sites.push_back({Pool::Recovered / 4, 0xd503201f, false});
            x18_sites.push_back({Pool::Failed / 4, 0xd503201f, false});
            Pool::probe_base.store(load_base + GetInteger(code.addr));
        }
#endif

        {   // titleplan: the text RelocateAndCopy branches into must be the text PatchText planned
            const u64 planned = PlannedTextHash(name), now = TextHash(codeset.memory, code);
            if (planned != now) {
                LOG_WARNING(Loader, "titleplan: {} text differs from the NCE plan (planned {:016x}, now {:016x}; an NSO patch or a title patch applied only when "
                            "loading): its svc/mrs/x18 changes are not NCE-patched", name, planned, now);
            } else if (name == "main") {
                LOG_INFO(Loader, "titleplan: main text as planned ({:016x}, title patches planned {})", now, title_patch_plan);
            }
        }
        // Relocate code patch and copy to the program image.
        // Save size before RelocateAndCopy (which may resize)
        const size_t size_before_relocate = codeset.memory.size();
        const bool relocated =
            patch->RelocateAndCopy(load_base, code, codeset.memory, &process.GetPostHandlers()
#if defined(_WIN32)
                                   , &process.GetWindowsPatchRecoveryForLoad()
#endif
                                   );
#if defined(NXN1X_NVN_HOST)
        Core::NvnLayer::KeepHookWords(name, std::span<const u8>{codeset.memory}.subspan(code.offset, code.size));
#endif
        if (relocated) {
            // Update patch section.
            auto& patch_segment = codeset.PatchSegment();
            auto& post_patch_segment = codeset.PostPatchSegment();
            const auto patch_mode = patch->GetPatchMode();
            if (patch_mode == Core::NCE::PatchMode::PreText) {
                patch_segment.addr = 0;
                patch_segment.size = static_cast<u32>(patch->GetSectionSize());
            } else if (patch_mode == Core::NCE::PatchMode::Split) {
                // For Split-mode, we are using pre-patch buffer at start, post-patch buffer at end
                patch_segment.addr = 0;
                patch_segment.size = static_cast<u32>(patch->GetPreSectionSize());
                post_patch_segment.addr = size_before_relocate;
                post_patch_segment.size = static_cast<u32>(patch->GetSectionSize());
            } else {
                patch_segment.addr = image_size;
                patch_segment.size = static_cast<u32>(patch->GetSectionSize());
            }
        }

#if defined(_WIN32)
        // Apply after the regular NCE relocations so x18-sensitive sites are guaranteed to reach
        // the Windows fallback trap rather than a generated helper that assumes physical x18.
        Core::NCE::X18SitePatcher::Apply(Common::ProcessAddress{load_base}, code, codeset.memory,
                                         x18_sites, process.GetPostHandlers());
#endif
#if defined(NXN1X_NVN_HOST)
        // An x18 site patch must not replace a taken-over first instruction (the jump to the NVN gate)
        Core::NvnLayer::RestoreHookWords(name, std::span<u8>{codeset.memory}.subspan(code.offset, code.size),
                                         process.GetPostHandlers());
#endif

        // Refresh image_size to take account the patch section if it was added by RelocateAndCopy
        image_size = static_cast<u32>(codeset.memory.size());
    }
#endif

    // If we aren't actually loading (i.e. just computing the process code layout), we are done
    if (!load_into_process) {
#ifdef HAS_NCE
        // Ok, so for Split mode, we need to account for pre-patch and post-patch space
        // which will be added during RelocateAndCopy in the second pass. Where it crashed
        // in Android Studio at PreText. May be a better way. Works for now.
        if (patch && patch->GetPatchMode() == Core::NCE::PatchMode::Split) {
            return load_base + patch->GetPreSectionSize() + image_size + patch->GetSectionSize();
        } else if (patch && patch->GetPatchMode() == Core::NCE::PatchMode::PreText) {
            return load_base + patch->GetSectionSize() + image_size;
        } else if (patch && patch->GetPatchMode() == Core::NCE::PatchMode::PostData) {
            return load_base + image_size + patch->GetSectionSize();
        }
#endif
        return load_base + image_size;
    }

#if defined(HAS_NCE) && defined(NXN1X_NVN_HOST)
    // sv1: where this module sits, for the guest thread dumps' module+offset
    Core::Sv1::NoteModule(name, load_base, image_size);
#endif

    // Apply cheats if they exist and the program has a valid title ID
    if (pm) {
        system.SetApplicationProcessBuildID(nso_header.build_id);
        const auto cheats = pm->CreateCheatList(nso_header.build_id);
        if (!cheats.empty()) {
            system.RegisterCheatList(cheats, nso_header.build_id, load_base, image_size);
        }
    }

#if NXN1X_DIAGNOSTICS
    // CPUW_TRACE (b71): where each module sits, so a guest pc or return address becomes a module
    // offset (segment addresses are relative to base).
    LOG_INFO(Render_Vulkan,
             "NSO_LAYOUT name={} build_id={} base={:#x} image_size={:#x} text={:#x}+{:#x} "
             "rodata={:#x}+{:#x} data={:#x}+{:#x}",
             name, Common::HexToString(nso_header.build_id), load_base, image_size,
             GetInteger(codeset.CodeSegment().addr), codeset.CodeSegment().size,
             GetInteger(codeset.RODataSegment().addr), codeset.RODataSegment().size,
             GetInteger(codeset.DataSegment().addr), codeset.DataSegment().size);
#endif
    // Load codeset for current process
    LOG_INFO(Loader, "Module {} loaded at {:#x} (size {:#x})", nso_file.GetName(), load_base, image_size);
    process.LoadModule(system.Kernel(), std::move(codeset), load_base);
    if (name == "main" && process.GetProgramId() == Settings::MarioKartWorldProgramId) {
        process.SetMkwDynamicFpsPatchApplied(mkw_dynamic_fps_patch_applied);
    }
    if (frame_interval_offset != 0) {
        process.SetFrameIntervalAddresses(load_base + module_start + frame_interval_offset,
                                          load_base + module_start + frame_interval_source_offset);
    }
#if defined(HAS_NCE) && defined(NXN1X_NVN_HOST)
    if (realtime_state_offset != 0 && Core::NvnLayer::Wanted()) {
        Core::NvnLayer::SetRealtimeState(load_base + module_start + realtime_state_offset);   // r63 realtime (above)
    }
#endif
    return load_base + image_size;
}

AppLoader_NSO::LoadResult AppLoader_NSO::Load(Kernel::KProcess& process, Core::System& system) {
    if (is_loaded) {
        return {ResultStatus::ErrorAlreadyLoaded, {}};
    }

    modules.clear();

    // Load module
    const VAddr base_address = GetInteger(process.GetEntryPoint());
    if (!LoadModule(process, system, *file, base_address, true, true)) {
        return {ResultStatus::ErrorLoadingNSO, {}};
    }

    modules.insert_or_assign(base_address, file->GetName());
    LOG_DEBUG(Loader, "loaded module {} @ {:#x}", file->GetName(), base_address);

    is_loaded = true;
    return {ResultStatus::Success, LoadParameters{Kernel::KThread::DefaultThreadPriority,
                                                  Core::Memory::DEFAULT_STACK_SIZE}};
}

ResultStatus AppLoader_NSO::ReadNSOModules(Modules& out_modules) {
    out_modules = this->modules;
    return ResultStatus::Success;
}

} // namespace Loader
