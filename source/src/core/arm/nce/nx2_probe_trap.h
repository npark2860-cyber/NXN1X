// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// The trap side of nx2_probe.h (Windows NCE only; included by arm_nce_windows.cpp after
// <windows.h>). Guest addresses are host addresses under NCE; every read is checked with
// VirtualQuery first so a bad pointer is skipped instead of faulting in the host.

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "core/arm/nce/guest_context.h"
#include "core/arm/nce/nx2_probe.h"

namespace Core::NCE::Nx2Probe {

struct Record {
    u64 t_us{};
    u64 obj{};  // Free: the object; Event: the slot's buffer then; Alloc: the new buffer
    u64 slot{}; // Free: the heap; Event/Alloc: the slot
    u64 aux{};  // Free: [obj+0xC0]; Event: the handler's this; Alloc: the slot's old buffer
    std::array<u32, 4> frames{}; // module-relative return addresses (lr, then the x29 chain)
    u32 tid{};
    Site kind{};
    u8 event{};
    u8 index{};
    u8 active{};
};

struct TrapState {
    std::mutex mutex;
    std::vector<Record> ring; // 1 << 16 records (~3.5 MiB), allocated at the first record: only when NXN1X_NX2_PROBE put trap sites in
    u64 next{};
    std::unordered_set<u64> buffers; // every buffer stored at the Alloc site
    u32 reports{};
    u32 alloc_logs{};
    bool dumped{};
    std::array<u64, 256> event_counts{};
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
inline TrapState g_trap{};

[[nodiscard]] inline bool Readable(u64 address, size_t size) {
    MEMORY_BASIC_INFORMATION info{};
    if (address == 0 ||
        VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0 ||
        info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 ||
        info.Protect == 0) {
        return false;
    }
    const u64 end = reinterpret_cast<u64>(info.BaseAddress) + info.RegionSize;
    return address + size <= end;
}

template <typename T>
[[nodiscard]] inline T ReadOr(u64 address, T fallback) {
    if (!Readable(address, sizeof(T))) {
        return fallback;
    }
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}

[[nodiscard]] inline std::string Describe(const Record& r) {
    const char* const kind = r.kind == Site::Free    ? "free"
                             : r.kind == Site::Event ? "event"
                                                     : "alloc";
    std::string frames;
    for (const u32 f : r.frames) {
        frames += f == 0xffffffffU ? " -" : fmt::format(" {:x}", f);
    }
    return fmt::format("t={}.{:06}s tid={} {} ev={} idx={} active={:#x} obj={:#x} slot={:#x} "
                       "aux={:#x} frames(main+):{}",
                       r.t_us / 1000000, r.t_us % 1000000, r.tid, kind, r.event, r.index,
                       r.active, r.obj, r.slot, r.aux, frames);
}

inline void Report(const Record& now) {
    auto& s = g_trap;
    LOG_WARNING(Core_ARM, "NX2PROBE double free #{}: {} (buffer from the alloc site: {})",
                s.reports, Describe(now), s.buffers.contains(now.obj));
    // Every earlier record of this object, then every event of the slots it was stored in.
    std::vector<const Record*> mine;
    std::unordered_set<u64> slots;
    const u64 count = std::min<u64>(s.next, s.ring.size());
    for (u64 i = 0; i < count; ++i) {
        const Record& r = s.ring[(s.next - count + i) % s.ring.size()];
        if (r.obj == now.obj) {
            mine.push_back(&r);
            if (r.kind != Site::Free) {
                slots.insert(r.slot);
            }
        }
    }
    const size_t first = mine.size() > 24 ? mine.size() - 24 : 0;
    for (size_t i = first; i < mine.size(); ++i) {
        LOG_WARNING(Core_ARM, "NX2PROBE   same object: {}", Describe(*mine[i]));
    }
    std::vector<const Record*> events;
    for (u64 i = 0; i < count; ++i) {
        const Record& r = s.ring[(s.next - count + i) % s.ring.size()];
        if (r.kind != Site::Free && slots.contains(r.slot) && r.obj != now.obj) {
            events.push_back(&r);
        }
    }
    const size_t first_event = events.size() > 40 ? events.size() - 40 : 0;
    for (size_t i = first_event; i < events.size(); ++i) {
        LOG_WARNING(Core_ARM, "NX2PROBE   same slot: {}", Describe(*events[i]));
    }
}

// At the first guest fault NCE emulates (a null read in this bug): the records just before it,
// with their threads, and how often each slot event has been seen.
inline void DumpRecent(u64 thread_id, u64 pc) {
    auto& s = g_trap;
    std::scoped_lock lock{s.mutex};
    if (s.dumped) {
        return;
    }
    s.dumped = true;
    const u64 base = g_state.module_base;
    LOG_WARNING(Core_ARM, "NX2PROBE first emulated fault: tid={} pc={:#x} (main+{:#x}), {} records",
                thread_id, pc, pc - base, s.next);
    std::string counts;
    for (size_t i = 0; i < s.event_counts.size(); ++i) {
        if (s.event_counts[i] != 0) {
            counts += fmt::format(" ev{}={}", i, s.event_counts[i]);
        }
    }
    LOG_WARNING(Core_ARM, "NX2PROBE   events so far:{}", counts);
    const u64 count = std::min<u64>(s.next, 64);
    for (u64 i = 0; i < count; ++i) {
        LOG_WARNING(Core_ARM, "NX2PROBE   recent: {}",
                    Describe(s.ring[(s.next - count + i) % s.ring.size()]));
    }
}

inline void OnTrap(const GuestContext& guest, u64 thread_id) {
    const u64 base = g_state.module_base;
    Site kind = Site::None;
    for (const auto& def : MkwSites) {
        if (guest.pc == base + def.offset) {
            kind = def.site;
        }
    }
    if (kind == Site::None) {
        return;
    }
    constexpr u64 AddressMask = (1ULL << 40) - 1; // drops the PAC bits of a signed lr
    const auto relative = [base](u64 address) {
        address &= AddressMask;
        return address >= base && address - base < 0x10000000 ? static_cast<u32>(address - base)
                                                               : 0xffffffffU;
    };
    const auto& x = guest.cpu_registers;
    Record r{};
    r.kind = kind;
    r.tid = static_cast<u32>(thread_id);
    r.frames.fill(0xffffffffU);
    r.frames[0] = relative(x[30]);
    u64 fp = x[29];
    for (size_t i = 1; i < r.frames.size() && Readable(fp, 16); ++i) {
        r.frames[i] = relative(ReadOr<u64>(fp + 8, 0));
        fp = ReadOr<u64>(fp, 0);
    }
    switch (kind) {
    case Site::Free:
        r.obj = x[1];
        r.slot = x[0];
        r.aux = ReadOr<u64>(x[1] + 0xC0, ~0ULL);
        break;
    case Site::Event: {
        r.event = static_cast<u8>(x[1]);
        r.index = ReadOr<u8>(x[4] + 8, 0xff);
        r.slot = x[0] + 400 + (r.index < 4 ? u64{r.index} * 496 : 0);
        r.obj = ReadOr<u64>(r.slot + 152, ~0ULL);
        r.active = ReadOr<u8>(r.slot + 492, 0xff);
        r.aux = x[0];
        break;
    }
    case Site::Alloc:
        r.obj = x[0];
        r.slot = x[21];
        r.aux = ReadOr<u64>(x[21] + 152, ~0ULL);
        break;
    case Site::None:
        return;
    }

    auto& s = g_trap;
    std::scoped_lock lock{s.mutex};
    r.t_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - s.start)
                                  .count());
    if (kind == Site::Event) {
        ++s.event_counts[r.event];
    }
    if (kind == Site::Alloc) {
        s.buffers.insert(r.obj);
        if (s.alloc_logs < 16) {
            ++s.alloc_logs;
            LOG_WARNING(Core_ARM, "NX2PROBE {}", Describe(r));
        }
    }
    // Frees are recorded only for alloc-site buffers (the heap frees far more); a free that finds
    // the block pointer already 0 is reported whatever the object.
    if (kind != Site::Free || s.buffers.contains(r.obj)) {
        if (s.ring.empty()) s.ring.resize(1U << 16);
        s.ring[s.next % s.ring.size()] = r;
        ++s.next;
    }
    if (kind == Site::Free && r.aux == 0 && s.reports < 4) {
        ++s.reports;
        Report(r);
    }
}

} // namespace Core::NCE::Nx2Probe
