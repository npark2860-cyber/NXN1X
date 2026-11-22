// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <span>
#include <utility>

#include "common/common_types.h"

namespace Core::NCE::HwaPool {

constexpr u32 Allocator = 0x1adce64;
constexpr u32 Stub = 0x1ed71c0; // End of .text in this build, within its mapped RX page.
constexpr u32 Recovered = Stub + 0x78;
constexpr u32 Failed = Stub + 0x80;
constexpr u32 RetryLimit = 128;
constexpr u32 ReleaseAcknowledged = 0x1add1a4;
constexpr u32 ReserveEntry = 0x1add040;
// pooldiag (core/nvn/sv1.cpp PoolDiag): the allocator's vtable in this build (its slot 48 = Allocator, +0x50 = free); a pool object starts
// with it. Pool fields (main+0x1adce64 / 0x1add040 / 0x1add1c0): +0x2c live blocks, +0x28 block size 0x80, +0x818 release word
constexpr u32 AllocatorVtable = 0x27e4ca0;
constexpr std::array<u8, 32> Build{
    0x65, 0x9c, 0xb2, 0x88, 0xb5, 0xc5, 0xb2, 0xb4, 0x98, 0x53,
    0xad, 0xd6, 0x16, 0xc0, 0xd5, 0xbf, 0x4e, 0xa4, 0xe1, 0xba};

// Wrap slot 48 itself. Keep the receiver in x22 and the original size and descriptor
// on the stack. The original function releases its active reference before returning.
// Only the pre-reservation full-count exit is safe to retry; a failed bitmap scan is not.
constexpr u32 RetryCode[] = {
    0xa9bd5bf3U, // stp x19, x22, [sp, #-0x30]!
    0xa9010be1U, // stp x1, x2, [sp, #0x10]
    0xf90013feU, // str x30, [sp, #0x20]
    0xaa0003f6U, // mov x22, x0
    0x52801013U, // mov w19, #0x80
    0xaa1f03ebU, // mov x11, xzr
    0xf102003fU, // cmp x1, #0x80
    0x94000000U | (((Allocator + 4 - (Stub + 0x1c)) >> 2) & 0x03ffffffU), // bl original body
    0xb5000280U, // cbnz x0, 0x70 <success>
    0x540002e1U, // b.ne 0x80 <failed>
    0x9100b2c8U, // add x8, x22, #0x2c
    0xeb08017fU, // cmp x11, x8
    0x54000281U, // b.ne 0x80 <failed>
    0x912062c8U, // add x8, x22, #0x818
    0xeb08005fU, // cmp x2, x8
    0x54000221U, // b.ne 0x80 <failed>
    0x7100047fU, // cmp w3, #0x1
    0x540001e1U, // b.ne 0x80 <failed>
    0x5287d008U, // mov w8, #0x3e80
    0x6b08015fU, // cmp w10, w8
    0x54000181U, // b.ne 0x80 <failed>
    0x34000173U, // cbz w19, 0x80 <failed>
    0x51000673U, // sub w19, w19, #0x1
    0xd2a00200U, // mov x0, #0x100000
    0xd4000161U, // svc #0xb
    0xaa1603e0U, // mov x0, x22
    0xa9410be1U, // ldp x1, x2, [sp, #0x10]
    0x17ffffeaU, // b 0x14 <again>
    0x7102027fU, // cmp w19, #0x80
    0x54000080U, // b.eq 0x84 <done>
    0xd503201fU, // nop
    0x14000002U, // b 0x84 <done>
    0xd503201fU, // nop
    0xf94013feU, // ldr x30, [sp, #0x20]
    0xa8c35bf3U, // ldp x19, x22, [sp], #0x30
    0xd65f03c0U, // ret
};

constexpr std::pair<u32, u32> FailureSites[] = {
    {0x1adcecc, 0xaa1f03e0}, {0x1add054, 0xaa1f03e0}, {0x1add17c, 0x1400000e},
};
inline std::atomic<u64> probe_base{};
inline std::atomic<unsigned> reports{};

inline bool Install(std::span<u8> text, u32 unaligned_text_size,
                    const std::array<u8, 32>& id) {
    if (id != Build || unaligned_text_size != Stub ||
        text.size() < Stub + sizeof(RetryCode)) return false;
    const auto word = [&](u32 off) {
        u32 value;
        std::memcpy(&value, text.data() + off, sizeof(value));
        return value;
    };
    if (word(Allocator) != 0xf102003f || word(Allocator + 4) != 0x540002e8 ||
        word(0x1adcebc) != 0x94000061 || word(0x1add188) != 0x36fff668 ||
        word(ReserveEntry) != 0x2a1f03e9 || word(ReleaseAcknowledged) != 0x17ffffac) return false;
    for (auto [pc, op] : FailureSites) if (word(pc) != op) return false;
    auto padding = text.subspan(Stub, sizeof(RetryCode));
    if (!std::all_of(padding.begin(), padding.end(), [](u8 b) { return b == 0; })) return false;
    const auto branch = [](u32 from, u32 to) {
        return 0x14000000U | (((to - from) >> 2) & 0x03ffffffU);
    };
    const u32 enter = branch(Allocator, Stub);
    // A free publishes bit 31 before decrementing the live count. The full-pool
    // path atomically consumes that bit without having reserved a slot. Reload
    // stride/count and try the normal reservation CAS with the original arguments.
    // If still full and no new release arrived, the original TBZ returns null.
    const u32 recheck = branch(ReleaseAcknowledged, ReserveEntry);
    std::memcpy(padding.data(), RetryCode, sizeof(RetryCode));
    std::memcpy(text.data() + Allocator, &enter, sizeof(enter));
    std::memcpy(text.data() + ReleaseAcknowledged, &recheck, sizeof(recheck));
    return true;
}

} // namespace Core::NCE::HwaPool
