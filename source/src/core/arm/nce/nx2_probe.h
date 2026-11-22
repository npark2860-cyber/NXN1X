// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// Temporary diagnostic (NXN1X_NX2_PROBE=1, off otherwise): Mario Kart World's main frees a slot
// buffer whose block pointer ([obj+0xC0]) is already 0 about 48 s into the world-map loading
// screen, the heap lists break and a thread spins for good. These three sites go through the x18
// fallback trap (BRK #0xF000 + one dynarmic step of the original instruction), and the trap
// records who allocated, freed and signalled each slot buffer, so the first free of the buffer
// freed twice is logged. The sites are checked word by word and the probe stays off on a mismatch.

#include <array>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "core/arm/nce/x18_fallback.h"
#include "core/arm/nce/x18_site_patcher.h"
#include "core/hle/kernel/code_set.h"

namespace Core::NCE::Nx2Probe {

enum class Site : u8 { None, Free, Event, Alloc };

struct SiteDef {
    u32 offset; ///< from the module base
    u32 word;   ///< the original instruction expected there
    Site site;
};

inline constexpr std::array<SiteDef, 3> MkwSites{{
    {0x47ff44, 0xd10143ffU, Site::Free},  // sub sp, sp, #80: free(x0 heap, x1 object), after pacibsp
    {0xa13d4, 0x39402098U, Site::Event},  // ldrb w24, [x4, #8]: slot event handler (w1 event)
    {0xa1640, 0xf9004ea0U, Site::Alloc},  // str x0, [x21, #152]: the slot's buffer is stored
}};

struct State {
    bool enabled{};
    u64 module_base{};
};
inline State g_state{};

[[nodiscard]] inline bool EnvOn() {
    static const bool on = std::getenv("NXN1X_NX2_PROBE") != nullptr;
    return on;
}

inline void AddSites(std::string_view module, u64 load_base, std::span<const u8> image,
                     const Kernel::CodeSet::Segment& code, std::vector<X18FallbackSite>& sites) {
    if (!EnvOn() || module != "main" || g_state.enabled) {
        return;
    }
    const u64 text_start = GetInteger(code.addr);
    if (code.offset + code.size > image.size()) {
        return;
    }
    const auto text = image.subspan(code.offset, code.size);
    for (const auto& def : MkwSites) {
        if (def.offset < text_start || def.offset - text_start + sizeof(u32) > code.size) {
            LOG_WARNING(Core_ARM, "NX2PROBE main+{:#x} is outside .text: probe off", def.offset);
            return;
        }
        u32 word{};
        std::memcpy(&word, text.data() + (def.offset - text_start), sizeof(word));
        if (word != def.word) {
            LOG_WARNING(Core_ARM, "NX2PROBE main+{:#x} holds {:#010x}, not {:#010x}: probe off",
                        def.offset, word, def.word);
            return;
        }
    }
    for (const auto& def : MkwSites) {
        sites.push_back({
            .text_word_index = static_cast<u32>((def.offset - text_start) / sizeof(u32)),
            .instruction = def.word,
            .may_access_memory = X18Fallback::MayAccessGuestMemory(def.word),
        });
    }
    g_state.module_base = load_base;
    g_state.enabled = true;
    LOG_WARNING(Core_ARM, "NX2PROBE on: {} sites in main at {:#x}", MkwSites.size(), load_base);
}

} // namespace Core::NCE::Nx2Probe
