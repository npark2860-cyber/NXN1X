// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#if !defined(_WIN32)
#error arm_nce_windows.cpp is only available on Windows.
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <array>
#include <csignal>
#include <cstdio>
#include <cwchar>
#include <exception>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "common/cgt_trace.h"
#include "common/game_presets.h"
#include "common/host_memory_windows_lease.h"
#include "core/arm/nce/arm_nce.h"
#include "core/arm/nce/arm_nce_asm_definitions.h"
#include "core/arm/nce/current_nce_context.h"
#include "core/arm/nce/patcher.h"
#if NXN1X_DIAGNOSTICS
#include "common/nce_read_guard.h"
#include "common/nce_write_note.h"
#include "common/settings.h"
#include "core/arm/nce/nce_write_step.h"
#endif
#if NXN1X_DIAGNOSTICS
#include <algorithm>
#include <bit>
#include <fmt/format.h>
#endif
#include "core/arm/nce/windows_cross_thread_break.h"
#include "core/arm/nce/windows_exception_context.h"
#include "core/arm/nce/windows_nce_transition.h"
#include "core/arm/nce/windows_nce_stats.h"
#include "core/nvn/nvn_layer.h"
#include "core/arm/nce/windows_nce_r2a_trace.h"
#include "core/arm/nce/windows_patch_code_metadata.h"
#include "core/arm/nce/windows_patch_recovery_metadata.h"
#include "core/arm/nce/windows_x18_fallback_runner.h"
#include "core/arm/nce/windows_x18_fallback_trap.h"
#include "core/arm/nce/nx2_probe_trap.h"
#include "core/arm/nce/hwa_pool.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/svc_types.h"
#include "core/memory.h"

extern "C" char dm_fatal_msg[512];   // stage2c (host dm.h die)
extern "C" void (*dm_fatal_hook)(void);   // stage2c3 (dm.h die)

namespace Core {

namespace NCE {
extern std::atomic<std::uint64_t> g_tebfix3_bind_other, g_tebfix3_x18_other;   // windows_nce_transition.cpp (tebfix5: counted, printed here)
void WindowsNceRefreshGuestStackBounds(GuestContext& guest) noexcept;   // nceresume R8 (defined below, called by windows_x18_fallback_runner.cpp)
}

namespace {

using NativeExecutionParameters = Kernel::KThread::NativeExecutionParameters;

static_assert(offsetof(NativeExecutionParameters, native_context) == TpidrEl0NativeContext);
static_assert(offsetof(NativeExecutionParameters, lock) == TpidrEl0Lock);
static_assert(offsetof(NativeExecutionParameters, magic) == TpidrEl0TlsMagic);

std::once_flag g_windows_veh_once;

// Incremented on every native entry of RunThread on this OS thread. If the value seen after the
// native call returns differs from the one recorded before it, either another RunThread frame ran
// on this OS thread in between or the host continuation restored a foreign callee-saved state.
thread_local u64 t_stats_epoch_tag = 0;

// Last exception the NCE VEH redirected on this OS thread. Logged when RunThread ends with
// PrefetchAbort, because LogBacktrace output is truncated by the logger.
struct WindowsNceLastException {
    u32 code;
    u64 pc;
    u64 sp;
    u64 lr;
    u64 fault_address;
};
thread_local WindowsNceLastException t_last_nce_exception{};

void RecordNceException(const EXCEPTION_RECORD& record, const ARM64_NT_CONTEXT& context) noexcept {
    t_last_nce_exception.code = static_cast<u32>(record.ExceptionCode);
    t_last_nce_exception.pc = context.Pc;
    t_last_nce_exception.sp = context.Sp;
    t_last_nce_exception.lr = context.Lr;
    t_last_nce_exception.fault_address =
        record.NumberParameters >= 2 ? static_cast<u64>(record.ExceptionInformation[1]) : 0;
}

void LogWindowsNceAbort(const char* site, Kernel::KProcess* process, const GuestContext& guest,
                        u64 last_native_exit) {
    const auto& ex = t_last_nce_exception;
    LOG_ERROR(Core_ARM,
              "NCE_ABORT site={} last_exit={:#x} guest pc={:#x} sp={:#x} lr={:#x} x18={:#x} | "
              "last_exception code={:#x} pc={:#x} sp={:#x} lr={:#x} fault={:#x}",
              site, last_native_exit, guest.pc, guest.sp, guest.cpu_registers[30],
              guest.cpu_registers[18], ex.code, ex.pc, ex.sp, ex.lr, ex.fault_address);
    if (process == nullptr) {
        return;
    }
    for (const u64 at : {guest.pc, ex.pc}) {
        if (at < 8) {
            continue;
        }
        std::string words;
        for (u64 a = at - 8; a <= at + 8; a += 4) {
            const auto* const p = process->GetMemory().GetPointerSilent(Common::ProcessAddress{a});
            words += p != nullptr ? fmt::format(" {}{:08x}", a == at ? "*" : "",
                                                *reinterpret_cast<const u32*>(p))
                                  : std::string{" ????????"};
        }
        LOG_ERROR(Core_ARM, "NCE_ABORT code @ {:#x}:{}", at, words);
    }
}
PVOID g_windows_veh_handle{};

struct WindowsTebStackBounds {
    NT_TIB* tib{};
    void* host_stack_base{};
    void* host_stack_limit{};
};

static_assert(offsetof(NT_TIB, StackBase) == 0x08);
static_assert(offsetof(NT_TIB, StackLimit) == 0x10);

[[nodiscard]] bool InstallGuestTebStackBounds(GuestContext& guest,
                                              WindowsTebStackBounds& bounds) noexcept {
    const u64 guest_sp = guest.sp;
    if (guest_sp == 0) {
        return false;
    }

    // The allocation bounds of a guest stack do not change while it stays mapped. Cache the last
    // few per OS thread so SVC-heavy titles do not pay a VirtualQuery walk on every native entry.
    struct CachedStackRange {
        std::uintptr_t base;
        std::uintptr_t end;
    };
    thread_local std::array<CachedStackRange, 8> cached_ranges{};
    thread_local std::size_t cached_next{};
    const auto sp_value = static_cast<std::uintptr_t>(guest_sp);
    std::uintptr_t allocation_base = 0;
    std::uintptr_t allocation_end = 0;
    for (const auto& range : cached_ranges) {
        if (range.end != 0 && sp_value > range.base && sp_value <= range.end) {
            allocation_base = range.base;
            allocation_end = range.end;
            break;
        }
    }

    if (allocation_end == 0) {
    MEMORY_BASIC_INFORMATION stack_mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(guest_sp - 1), &stack_mbi,
                     sizeof(stack_mbi)) == 0 ||
        stack_mbi.AllocationBase == nullptr) {
        return false;
    }

    allocation_base = reinterpret_cast<std::uintptr_t>(stack_mbi.AllocationBase);
    allocation_end = allocation_base;
    for (std::uintptr_t cursor = allocation_base;;) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &region, sizeof(region)) == 0 ||
            region.AllocationBase != stack_mbi.AllocationBase) {
            break;
        }

        const auto region_end = reinterpret_cast<std::uintptr_t>(region.BaseAddress) +
                                static_cast<std::uintptr_t>(region.RegionSize);
        if (region_end <= cursor) {
            break;
        }
        allocation_end = region_end;
        cursor = region_end;
    }
    if (sp_value > allocation_base && sp_value <= allocation_end) {
        cached_ranges[cached_next++ % cached_ranges.size()] = {allocation_base, allocation_end};
    }
    }

    const auto guest_sp_value = static_cast<std::uintptr_t>(guest_sp);
    // A downward-growing stack may begin with SP exactly at StackBase, which is one-past
    // the highest address in the allocation. VirtualQuery above intentionally probes SP - 1,
    // so accept guest_sp == allocation_end while still rejecting values above the allocation.
    if (guest_sp_value <= allocation_base || guest_sp_value > allocation_end) {
        return false;
    }

    auto* const tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    bounds.tib = tib;
    bounds.host_stack_base = tib->StackBase;
    bounds.host_stack_limit = tib->StackLimit;
    guest.windows_guest_stack_base = static_cast<u64>(allocation_end);
    guest.windows_guest_stack_limit = static_cast<u64>(allocation_base);
    guest.windows_host_stack_base =
        static_cast<u64>(reinterpret_cast<std::uintptr_t>(bounds.host_stack_base));
    guest.windows_host_stack_limit =
        static_cast<u64>(reinterpret_cast<std::uintptr_t>(bounds.host_stack_limit));
    // tebfix (11-20): the TEB keeps the host bounds here. The guest bounds are published only where SP changes with them: the
    // entry stubs (windows_nce_entry.asm WindowsNceEnterGuest, patcher x18 direct path) store both right before MOV SP, and
    // the NtContinue entry (windows_nce_transition.cpp WindowsNceRestoreGuestContext) stores both at its last host-side
    // instruction before NtContinue. Publishing the guest StackBase here (as before) left RunThread's host C++ (stack view,
    // stats, entry-handler lookup) on the host stack with StackBase = guest top and StackLimit = host limit: an exception
    // resumed in that window had an SP RtlRestoreContext refuses (fast-fail 0xd at ntdll+0x34a10: HWAoI 11:11, 11:22;
    // Bayonetta 3 11:53)
    return true;
}

void RestoreHostTebStackBounds(WindowsTebStackBounds& bounds) noexcept {
    if (bounds.tib == nullptr) {
        return;
    }
    bounds.tib->StackBase = bounds.host_stack_base;
    bounds.tib->StackLimit = bounds.host_stack_limit;
    bounds.tib = nullptr;
}

[[nodiscard]] const ArmNce::WindowsGuestStackState* FindGuestStackState(const ArmNce& nce,
                                                                     const Kernel::KThread* thread,
                                                                     u64 guest_sp) noexcept {
    if (thread == nullptr || guest_sp == 0) {
        return nullptr;
    }
    const u64 thread_id = thread->GetThreadId();
    const u64 probe = guest_sp - 1;
    for (const auto& state : nce.m_windows_stack_states) {
        if (state.valid && state.thread_id == thread_id && probe >= state.begin &&
            probe < state.end) {
            return &state;
        }
    }
    return nullptr;
}

// Native guest code runs directly on the section-backed (MEM_MAPPED) guest stack by default.
// Replacing the active stack with a private copy on every SVC epoch cost most of the frame time in
// SVC-heavy titles and was found unnecessary (Gunvolt Trilogy, TotK run correctly without it).
// The private stack lease is kept behind EDEN_NCE_LEASE=1 for A/B comparison.
[[nodiscard]] bool WindowsNceNoLeaseExperiment() noexcept {
    static const bool no_lease = [] {
        const char* const value = std::getenv("EDEN_NCE_LEASE");
        const bool lease = value != nullptr && value[0] != '\0' && value[0] != '0';
        if (lease) {
            LOG_WARNING(Core_ARM, "EDEN_NCE_LEASE: private guest stack lease enabled");
        }
        return !lease;
    }();
    return no_lease;
}

// guest_sp_is_stack comes from the guest page table, queried by ArmNce::PrepareRunThread outside
// the execution epoch. This function must not call into the guest kernel: a contended KLightLock
// would reschedule another guest thread onto this core in the middle of RunThread.
[[nodiscard]] bool EnsureWindowsGuestStackLease(
    System& system, Kernel::KProcess* process, u64 guest_sp, bool guest_sp_is_stack,
    std::optional<Common::HostMemory::PrivateMappingLease>& stack_lease) {
    if (process == nullptr || guest_sp == 0) {
        LOG_ERROR(Core_ARM, "Windows NCE stack lease received an invalid process or SP");
        return false;
    }

    if (WindowsNceNoLeaseExperiment()) {
        return true;
    }

    MEMORY_BASIC_INFORMATION stack_mbi{};
    if (VirtualQuery(reinterpret_cast<const void*>(guest_sp - 1), &stack_mbi,
                     sizeof(stack_mbi)) == 0 ||
        stack_mbi.AllocationBase == nullptr) {
        LOG_ERROR(Core_ARM, "Windows NCE guest stack VirtualQuery failed");
        return false;
    }

    if (WindowsNceNoLeaseExperiment()) {
        return true;
    }

    if (stack_lease.has_value()) {
        const bool active_lease_owns_sp = stack_lease->ContainsAddress(guest_sp - 1);
        if (active_lease_owns_sp) {
            if (!guest_sp_is_stack || stack_mbi.Type != MEM_PRIVATE) {
                LOG_ERROR(Core_ARM,
                          "Windows NCE active private stack lease has invalid guest stack state");
                return false;
            }
            return true;
        }

        // Native guest code may switch SP between user-space stacks without leaving the current
        // RunThread epoch. The previous lease still owns a valid dormant stack, but it must be
        // synchronized/restored before evaluating the newly active guest stack. If that stack later
        // becomes active again, it will be leased again from the section-backed mapping.
        NCE::Stats::Add(NCE::Stats::LeaseRestores);
        if (!stack_lease->Restore()) {
            LOG_ERROR(Core_ARM, "Windows NCE failed to roll over previous private stack lease");
            return false;
        }
        stack_lease.reset();
    }

    // Preserve the V7 eligibility policy: already-private stacks and synthetic/non-Stack mapped
    // probe stacks do not need a lease. Only a real KPageTable Stack backed by MEM_MAPPED does.
    if (stack_mbi.Type == MEM_PRIVATE) {
        return true;
    }
    if (!guest_sp_is_stack && stack_mbi.Type == MEM_MAPPED) {
        return true;
    }
    if (!guest_sp_is_stack || stack_mbi.Type != MEM_MAPPED) {
        LOG_ERROR(Core_ARM, "Windows NCE guest stack has unsupported host backing type/state");
        return false;
    }

    const auto stack_base = reinterpret_cast<std::uintptr_t>(stack_mbi.AllocationBase);
    std::uintptr_t stack_end = stack_base;
    for (std::uintptr_t cursor = stack_base;;) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &region, sizeof(region)) == 0 ||
            region.AllocationBase != stack_mbi.AllocationBase) {
            break;
        }
        const auto region_end = reinterpret_cast<std::uintptr_t>(region.BaseAddress) +
                                static_cast<std::uintptr_t>(region.RegionSize);
        if (region_end <= cursor) {
            break;
        }
        stack_end = region_end;
        cursor = region_end;
    }

    if (stack_end <= stack_base || guest_sp <= stack_base || guest_sp > stack_end ||
        (stack_base & Memory::YUZU_PAGEMASK) != 0 ||
        ((stack_end - stack_base) & Memory::YUZU_PAGEMASK) != 0) {
        LOG_ERROR(Core_ARM, "Windows NCE real guest stack host allocation is invalid");
        return false;
    }

    const auto stack_size = static_cast<size_t>(stack_end - stack_base);
    auto* const backing_pointer =
        process->GetMemory().GetPointer(Common::ProcessAddress{stack_base});
    if (backing_pointer == nullptr) {
        LOG_ERROR(Core_ARM, "Windows NCE real guest stack has no backing pointer");
        return false;
    }

    for (size_t offset = 0; offset < stack_size; offset += Memory::YUZU_PAGESIZE) {
        auto* const page_pointer =
            process->GetMemory().GetPointer(Common::ProcessAddress{stack_base + offset});
        if (page_pointer != backing_pointer + offset) {
            LOG_ERROR(Core_ARM, "Windows NCE real guest stack backing is not contiguous");
            return false;
        }
    }

    auto& buffer = system.DeviceMemory().buffer;
    const auto backing_base = reinterpret_cast<std::uintptr_t>(buffer.BackingBasePointer());
    const auto backing_address = reinterpret_cast<std::uintptr_t>(backing_pointer);
    if (backing_address < backing_base) {
        LOG_ERROR(Core_ARM, "Windows NCE real guest stack backing pointer is outside HostMemory");
        return false;
    }
    const auto backing_offset = static_cast<size_t>(backing_address - backing_base);

    const u64 stats_lease_t0 = NCE::Stats::Now();
    auto acquired = buffer.AcquireDirectMappedPrivateLease(
        reinterpret_cast<void*>(stack_base), backing_offset, stack_size,
        Common::MemoryPermission::ReadWrite, guest_sp);
    if (!acquired.has_value()) {
        LOG_ERROR(Core_ARM, "Windows NCE failed to acquire private HostMemory stack lease");
        return false;
    }

    stack_lease.emplace(std::move(*acquired));
    NCE::Stats::Add(NCE::Stats::LeaseAcquires);
    NCE::Stats::Add(NCE::Stats::LeaseBytes, stack_size);
    NCE::Stats::Add(NCE::Stats::LeaseNanos, NCE::Stats::Now() - stats_lease_t0);
    return true;
}

constexpr u32 P2BGuestX18Register = 18;

[[nodiscard]] u64 ReadP2BGuestRegister(const GuestContext& guest, u32 reg) noexcept {
    // These P2B forms only use ordinary X/W registers for the decoded address operands.
    // Rn==31 is SP for the byte store base; the register-offset load is deliberately limited
    // below to Rn!=31 so no ambiguous ZR/SP role is admitted.
    return reg == 31 ? guest.sp : guest.cpu_registers[reg];
}

[[nodiscard]] std::optional<u64> GetP2BObservedByteAccessAddress(
    u32 instruction, const GuestContext& guest) noexcept {
    constexpr u32 RegisterMask = 0x1FU;

    // LDRB Wt, [Xn, X18] using the observed UXTX/LSL #0 register-offset form.
    // Keep this bounded to the exact operand role seen in the P2A hot loop.
    if ((instruction & 0xFFE0FC00U) == 0x38606800U) {
        const u32 rm = (instruction >> 16) & RegisterMask;
        const u32 rn = (instruction >> 5) & RegisterMask;
        if (rm == P2BGuestX18Register && rn != 31) {
            return guest.cpu_registers[rn] + guest.cpu_registers[P2BGuestX18Register];
        }
    }

    // STRB Wt, [X18], #simm9. The byte access uses the pre-writeback X18 address;
    // Dynarmic still owns the architectural X18 writeback itself.
    if ((instruction & 0xFFE00C00U) == 0x38000400U) {
        const u32 rn = (instruction >> 5) & RegisterMask;
        if (rn == P2BGuestX18Register) {
            return ReadP2BGuestRegister(guest, rn);
        }
    }

    return std::nullopt;
}

[[nodiscard]] bool P2BByteAccessAliasesPrivateStackBacking(
    Kernel::KProcess* process, const GuestContext& guest, u64 address) {
    if (process == nullptr) {
        return true;
    }

    const u64 stack_low = guest.windows_guest_stack_limit;
    const u64 stack_high = guest.windows_guest_stack_base;
    if (stack_low == 0 || stack_high <= stack_low) {
        return true;
    }

    // Direct virtual overlap is the common case.
    if (address >= stack_low && address < stack_high) {
        return true;
    }

    // Also catch a second guest VA aliasing the same physical/backing stack storage.
    // EnsureWindowsGuestStackLease already proved this stack backing is contiguous.
    auto& memory = process->GetMemory();
    const auto* const stack_backing =
        memory.GetPointerSilent(Common::ProcessAddress{stack_low});
    const auto* const access_backing =
        memory.GetPointerSilent(Common::ProcessAddress{address});
    if (stack_backing == nullptr || access_backing == nullptr) {
        // Invalid/unmapped accesses stay on the conservative full-sync path so fallback fault
        // behavior is unchanged.
        return true;
    }

    const auto stack_backing_begin =
        reinterpret_cast<std::uintptr_t>(stack_backing);
    const auto stack_size = static_cast<std::uintptr_t>(stack_high - stack_low);
    const auto stack_backing_end = stack_backing_begin + stack_size;
    if (stack_backing_end < stack_backing_begin) {
        return true;
    }

    const auto access_backing_value =
        reinterpret_cast<std::uintptr_t>(access_backing);
    return access_backing_value >= stack_backing_begin &&
           access_backing_value < stack_backing_end;
}

[[nodiscard]] bool P2CNeedsPrivateStackSync(
    u64 transition_result, Kernel::KProcess* process, const GuestContext& guest,
    const NCE::X18FallbackMetadata& metadata) {
    if (transition_result != NCE::WindowsX18FallbackTrap::ReturnMarker) {
        return true;
    }

    const auto site = NCE::WindowsX18FallbackTrap::FindSiteInfo(guest.pc, metadata);
    if (!site.has_value()) {
        return true;
    }

    // P2C generalizes the P2B register-only optimization. The patcher computes this property once
    // from Dynarmic IR and records it in metadata, so the hot fallback path pays only a bit test.
    // An instruction that emits no guest memory read/write IR cannot observe or modify the private
    // stack backing, regardless of which ordinary ALU/branch family uses virtual guest x18.
    if (!site->may_access_memory) {
        return false;
    }

    // Preserve P2B's bounded byte-memory optimization. Memory-touching instructions skip the
    // private-stack copy only when their effective address neither lies in the stack VA nor aliases
    // the same backing storage.
    if (const auto address = GetP2BObservedByteAccessAddress(site->instruction, guest)) {
        return P2BByteAccessAliasesPrivateStackBacking(process, guest, *address);
    }

    // Every other memory-touching x18 fallback keeps the established V16 full-sync behavior.
    return true;
}

struct BreakTransformState {
    ArmNce* nce{};
    // Halt reasons the transform took out of esr_el1. Given back if the CONTEXT is not installed.
    u64 claimed_reason{};
    // Parameter lock taken by the transform while the target was suspended. Ownership passes to
    // the target's host continuation once the redirect is installed; otherwise the sender drops it.
    NativeExecutionParameters* params{};
    bool params_busy{};
    bool transformed{};
    bool host_window{};
    bool patch_window{};
    // R2b-B2 x18 substitution exit. recovered: the transform rebuilt guest state from a recipe and
    // WindowsBreakCommit publishes it before resume. recovery_aborted: SetThreadContext failed and
    // the commit already gave back the reason and the parameter lock.
    bool recovered{};
    bool recovery_aborted{};
    bool publish_x18{};
    u64 recovered_x18{};
    std::size_t recovered_step{};
    std::optional<ARM64_NT_CONTEXT> recovered_state;
#if NXN1X_DIAGNOSTICS
    NCE::WindowsPatchRecoveryKind recovered_kind{}; // R2b-B2x1
    u64 r2a_pc{};
    u8 r2a_phase{};
    u8 r2a_reject{0xFF};
#endif
};

#if NXN1X_DIAGNOSTICS
#define NXN1X_R2B_FAIL(kind)                                                                     \
    NCE::R2a::G()                                                                                  \
        .recovery_fail[static_cast<std::size_t>(NCE::R2a::RecoveryFail::kind)]                     \
        .fetch_add(1, std::memory_order_relaxed)
#else
#define NXN1X_R2B_FAIL(kind) ((void)0)
#endif

// R2b-B2r1. Reads memory the recovery depends on without dereferencing it. A page the helper ran
// from can still lose read access (HostMemory::Protect); a failed read leaves the capture
// unrecovered instead of faulting in the sender. Only called on the recovery path.
template <typename T>
bool ReadRecoveryValue(std::uintptr_t source, T& destination) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    SIZE_T read{};
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(source),
                             &destination, sizeof(T), &read) != 0 &&
           read == sizeof(T);
}

// R2b-B2x1. Same, for a run of bytes whose length the recipe gives (the helpers differ in size).
bool ReadRecoveryBytes(std::uintptr_t source, void* destination, std::size_t size) noexcept {
    SIZE_T read{};
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(source),
                             destination, size, &read) != 0 &&
           read == size;
}

#if NXN1X_DIAGNOSTICS
// S3z5 diagnostic switch. EDEN_NCE_RECOVER_OFF turns patch-window recovery off per recipe kind,
// or per step of a kind. A switched-off capture is resumed unchanged and retried, which is what
// every kind did before it had a recipe (B2x1 behaviour for mrs/cond, pre-B2 for all).
// Tokens, separated by ',': subst, bitfield, mrs, cond, all; "kind:N" switches off step N only.
// Example: EDEN_NCE_RECOVER_OFF=mrs,cond:7
constexpr std::size_t RecoveryKindCount = 4;
using RecoveryOffMasks = std::array<u32, RecoveryKindCount>;

[[nodiscard]] RecoveryOffMasks ParseRecoveryOff(const char* value) noexcept {
    RecoveryOffMasks masks{};
    if (value == nullptr) {
        return masks;
    }
    constexpr std::array<const char*, RecoveryKindCount> names{"subst", "bitfield", "mrs", "cond"};
    const std::string text{value};
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = std::min(text.find(',', begin), text.size());
        const std::string token = text.substr(begin, end - begin);
        const std::size_t colon = token.find(':');
        const std::string name = token.substr(0, colon);
        u32 bits = ~0U;
        if (colon != std::string::npos) {
            const unsigned long step = std::strtoul(token.c_str() + colon + 1, nullptr, 10);
            bits = step < 32 ? (1U << step) : 0U;
        }
        for (std::size_t kind = 0; kind < RecoveryKindCount; ++kind) {
            if (name == "all" || name == names[kind]) {
                masks[kind] |= bits;
            }
        }
        begin = end + 1;
    }
    return masks;
}

// Read once by ArmNce::Initialize on the emulator thread, before any break worker can capture.
[[nodiscard]] const RecoveryOffMasks& RecoveryOff() noexcept {
    static const RecoveryOffMasks masks = ParseRecoveryOff(std::getenv("EDEN_NCE_RECOVER_OFF"));
    return masks;
}
#endif

// R2b-B2. The target stopped inside a generated x18 substitution helper. Rebuild the guest
// architectural state at that instruction boundary from the recipe the patcher recorded, so the
// break is delivered now instead of being retried until the helper branches back to guest code.
// Nothing real is written here: the rebuilt state goes to state.recovered_state and is published
// by WindowsBreakCommit only after SetThreadContext succeeded, while the target is still stopped.
// Any failed check returns false and the capture is retried exactly as before.
bool TryRecoverX18Substitution(BreakTransformState& state, ARM64_NT_CONTEXT& context,
                               Kernel::KThread& thread,
                               const Kernel::KProcess& process) noexcept {
    using Params = Kernel::KThread::NativeExecutionParameters;

    const auto* const recipe = process.GetWindowsPatchRecovery().Find(context.Pc);
    if (recipe == nullptr) {
        NXN1X_R2B_FAIL(NoRecipe);
        return false;
    }
    const auto* const step = recipe->FindStep(context.Pc);
#if NXN1X_DIAGNOSTICS
    if (step != nullptr) {
        const auto kind = static_cast<std::size_t>(recipe->kind);
        const auto index = static_cast<std::size_t>(step - recipe->steps.data());
        if (kind < RecoveryKindCount && index < 32 && (RecoveryOff()[kind] >> index & 1U) != 0) {
            return false;
        }
    }
#endif
    constexpr u32 SpillContext = static_cast<u32>(offsetof(Params, x18_scratch));
    constexpr u32 SpillValue = SpillContext + static_cast<u32>(sizeof(u64));
    constexpr u32 NativeContext = static_cast<u32>(offsetof(Params, native_context));
    constexpr u32 GuestX18 =
        static_cast<u32>(offsetof(GuestContext, cpu_registers) + sizeof(u64) * 18);
    if (step == nullptr ||
        recipe->params_tls_offset != NCE::CurrentNceContext::TebSlotOffset() ||
        recipe->scratch_tls_offset != NCE::CurrentNceContext::ScratchSlotOffset(0) ||
        recipe->spill_context_offset != SpillContext ||
        recipe->spill_value_offset != SpillValue ||
        recipe->native_context_offset != NativeContext || recipe->guest_x18_offset != GuestX18) {
        NXN1X_R2B_FAIL(RecipeShape);
        return false;
    }
    // R2b-B2x2. The MRS handler loads one params field; it must be the instruction's register.
    constexpr u32 TpidrEl0 = static_cast<u32>(offsetof(Params, tpidr_el0));
    constexpr u32 TpidrroEl0 = static_cast<u32>(offsetof(Params, tpidrro_el0));
    const bool is_mrs = recipe->kind == NCE::WindowsPatchRecoveryKind::MrsTpidrV1;
    const bool reads_tpidrro = (recipe->original_instruction & 0x20U) != 0;
    if (is_mrs && recipe->params_field_offset != (reads_tpidrro ? TpidrroEl0 : TpidrEl0)) {
        NXN1X_R2B_FAIL(RecipeShape);
        return false;
    }
    // Valid() bounds word_count by code_words.size(); only the helper's own words are read.
    std::array<u32, NCE::WindowsPatchRecoveryRecipe::StepCount> words{};
    const std::size_t word_count = recipe->word_count;
    if (!ReadRecoveryBytes(recipe->begin, words.data(), word_count * sizeof(u32))) {
        NXN1X_R2B_FAIL(ReadFailed);
        return false;
    }
    if (!std::equal(words.begin(), words.begin() + word_count, recipe->code_words.begin())) {
        NXN1X_R2B_FAIL(CodeMismatch);
        return false;
    }

    // Generated code never changes physical x18, so it must be this thread's TEB. Only then are
    // its TLS slots read; a mismatch means the CONTEXT is not the one the recipe describes.
    const std::uintptr_t teb = state.nce->m_windows_break->Teb();
    NXN1X_R2A(NCE::R2a::NoteRecoveryX18(context.X[18], teb));
    if (teb == 0 || context.X[18] != teb) {
        NXN1X_R2B_FAIL(TebMismatch);
        return false;
    }
    auto* const params = &thread.GetNativeExecutionParameters();
    u64 params_slot;
    if (!ReadRecoveryValue(teb + recipe->params_tls_offset, params_slot)) {
        NXN1X_R2B_FAIL(ReadFailed);
        return false;
    }
    if (params_slot != reinterpret_cast<u64>(params) ||
        params->native_context != &state.nce->m_guest_ctx) {
        NXN1X_R2B_FAIL(ParamsMismatch);
        return false;
    }

    // T lives in the target TEB and is only meaningful while the helper is between its first
    // store and its reload; read it only for those steps. Read before the lock so a failure has
    // nothing to give back.
    NCE::WindowsPatchRecoveryInputs inputs{};
    inputs.captured_x = context.X;
    if (NCE::WindowsPatchStepNeedsTebScratch(*step) &&
        !ReadRecoveryValue(teb + recipe->scratch_tls_offset, inputs.target_teb_scratch)) {
        NXN1X_R2B_FAIL(ReadFailed);
        return false;
    }
    // R2b-B2x1. Between its STP and LDP the bitfield helper keeps the two original scratches at
    // the captured SP; the SP is not the host stack (checked by the caller).
    if (NCE::WindowsPatchStepNeedsGuestStack(*step)) {
        std::array<u64, 2> saved{};
        if (!ReadRecoveryValue(static_cast<std::uintptr_t>(context.Sp), saved)) {
            NXN1X_R2B_FAIL(ReadFailed);
            return false;
        }
        inputs.stack_slot0 = saved[0];
        inputs.stack_slot8 = saved[1];
    }

    // Same rule as a guest PC: take the parameter lock without waiting, or resume unchanged.
    u32 expected = SpinLockUnlocked;
    if (!params->lock.compare_exchange_strong(expected, SpinLockLocked,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
        state.params_busy = true;
        NXN1X_R2B_FAIL(PBusy);
        return false;
    }

    // The spills are fields of this thread's own parameters, pinned by the epoch.
    inputs.spill_context = params->x18_scratch[0];
    inputs.spill_value = params->x18_scratch[1];
    if (is_mrs) {
        // Read under the parameter lock with the thread stopped, like the spills.
        inputs.params_field = reads_tpidrro ? params->tpidrro_el0 : params->tpidr_el0;
    }
    const auto recovered = NCE::RecoverWindowsX18Substitution(*recipe, *step, inputs);

    // Only the two scratches (one register for the B2x2 helpers), and for the helpers that push
    // to the guest stack SP, are rewritten. Every other register, NZCV, FPCR/FPSR and the vector registers stay as captured;
    // physical x18 stays the TEB.
    auto& snapshot = state.recovered_state.emplace(context);
    snapshot.X[recipe->value_reg] = recovered.value_original;
    snapshot.X[recipe->context_reg] = recovered.context_original;
    snapshot.Sp = context.Sp + recovered.sp_restore;
    snapshot.Pc = recovered.guest_pc;
    state.publish_x18 = recovered.publish_x18;
    state.recovered_x18 = recovered.guest_x18;
    state.recovered_step = static_cast<std::size_t>(step - recipe->steps.data());
    NXN1X_R2A(state.recovered_kind = recipe->kind;)
    state.params = params;

    auto& guest = state.nce->m_guest_ctx;
    const auto reason = guest.esr_el1.exchange(0, std::memory_order_acq_rel);
    state.claimed_reason = reason;
    context = snapshot;
    // tebfix3: the CONTEXT is of the break's suspended thread, the one its handle and its TEB were bound together for (BindCurrentThread)
    NCE::WindowsNceTransition::RedirectToHost(context, guest, false, reason, static_cast<u64>(state.nce->m_windows_break->DeliveryTeb()));
    NCE::Stats::Add(NCE::Stats::RedirectSignal);
    state.recovered = true;
    state.transformed = true;
    return true;
}

// Called by SuspendTransformResume between SetThreadContext and ResumeThread. Only the R2b-B2
// path uses it; the guest-PC path keeps its existing order.
void WindowsBreakCommit(bool set_ok, void* opaque) noexcept {
    auto* const state = static_cast<BreakTransformState*>(opaque);
    if (state == nullptr || !state->recovered) {
        return;
    }
    auto& guest = state->nce->m_guest_ctx;
    if (set_ok) {
        // The redirect is installed; publish the rebuilt guest state before the host continuation
        // can read it. SaveGuestState skips x18, which is published separately when the helper
        // had already produced the new value but not stored it.
        NCE::WindowsExceptionContext::SaveGuestState(guest, *state->recovered_state);
        if (state->publish_x18) {
            guest.cpu_registers[18] = state->recovered_x18;
        }
        NXN1X_R2A(NCE::R2a::RecoveredStep(static_cast<std::size_t>(state->recovered_kind),
                                            state->recovered_step)
                        .fetch_add(1, std::memory_order_relaxed);)
        return;
    }
    // The target keeps its original CONTEXT and finishes the helper itself. Give back the reason
    // and the lock while it is still stopped; GuestContext was never written.
    guest.esr_el1.fetch_or(state->claimed_reason, std::memory_order_acq_rel);
    state->params->lock.store(SpinLockUnlocked, std::memory_order_release);
    state->recovery_aborted = true;
    NXN1X_R2B_FAIL(SetFailed);
}

bool WindowsBreakTransform(ARM64_NT_CONTEXT& context, void* opaque) noexcept {
    auto* const state = static_cast<BreakTransformState*>(opaque);
    if (state == nullptr || state->nce == nullptr || state->nce->m_windows_break == nullptr) {
        return false;
    }
    NXN1X_R2A(state->r2a_pc = context.Pc;
                state->r2a_phase = state->nce->m_r2a_phase.load(std::memory_order_relaxed);)

    // The scheduler lock can become available just before RtlRestoreContext performs the final
    // host-stack -> guest-stack transfer. Never snapshot that transition window as guest state.
    // SignalInterrupt retains the NativeExecutionParameters lock and retries after the target has
    // resumed far enough to own a guest stack.
    if (state->nce->m_windows_break->IsHostStackPointer(context.Sp)) {
        state->host_window = true;
        NXN1X_R2A(state->r2a_reject = static_cast<u8>(NCE::R2a::Reject::HostStack);)
        return false;
    }

    state->host_window = false;
    auto* const thread = state->nce->m_running_thread;
    auto* const process = thread != nullptr ? thread->GetOwnerProcess() : nullptr;
    if (process != nullptr && NCE::WindowsPatchCodeMetadata::Contains(
                                  context.Pc, process->GetPostHandlers())) {
        if (TryRecoverX18Substitution(*state, context, *thread, *process)) {
            return true;
        }
        if (state->params_busy) {
            NXN1X_R2A(state->r2a_reject = static_cast<u8>(NCE::R2a::Reject::PBusyAtGuest);)
            return false;
        }
        state->patch_window = true;
        NXN1X_R2A(state->r2a_reject = static_cast<u8>(NCE::R2a::Reject::PatchWindow);)
        return false;
    }

    state->patch_window = false;

    // A Windows host bridge/helper can transiently execute with the physical guest SP already
    // installed. Stack ownership alone is therefore insufficient to prove that the suspended
    // CONTEXT is guest architectural state. Only snapshot a PC that belongs to the guest process
    // address space; otherwise resume unchanged and let SignalInterrupt retry the host window.
    if (process == nullptr ||
        !process->GetMemory().IsValidVirtualAddressRange(context.Pc, sizeof(u32))) {
        state->host_window = true;
        NXN1X_R2A(state->r2a_reject = static_cast<u8>(NCE::R2a::Reject::HostPc);)
        return false;
    }

    // Take the parameter lock only now, with the target stopped at a guest PC, and never wait for
    // it: a stopped target may own it (or be about to need it), and waiting here would deadlock.
    // The redirected host continuation expects the lock to be held, exactly as before.
    auto* const params = &thread->GetNativeExecutionParameters();
    u32 expected = SpinLockUnlocked;
    if (!params->lock.compare_exchange_strong(expected, SpinLockLocked,
                                              std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
        state->params_busy = true;
        NXN1X_R2A(state->r2a_reject = static_cast<u8>(NCE::R2a::Reject::PBusyAtGuest);)
        return false;
    }
    state->params = params;

    auto& guest = state->nce->m_guest_ctx;
    const auto reason = guest.esr_el1.exchange(0, std::memory_order_acq_rel);
    state->claimed_reason = reason;
    NCE::WindowsNceTransition::RedirectToHost(context, guest, true, reason, static_cast<u64>(state->nce->m_windows_break->DeliveryTeb()));   // tebfix3 (as above)
    NCE::Stats::Add(NCE::Stats::RedirectSignal);
    state->transformed = true;
    return true;
}

// S3: Post found the previous job not yet taken by the worker. That ArmNce would stay Busy with
// nobody to deliver it, and its RunThread would wait for the release forever. This cannot happen
// while at most one RunThread per core owns a Busy gate, so stop the process instead of hanging,
// with the core in the exception record. No logging: the caller holds PhysicalCore's guard.
[[noreturn]] void FailFastOnJobOverwrite(std::size_t core) noexcept {
    EXCEPTION_RECORD record{};
    record.ExceptionCode = 0xE04E4353; // 'NCS': NCE break job slot overwritten
    record.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
    record.NumberParameters = 1;
    record.ExceptionInformation[0] = core;
    RaiseFailFastException(&record, nullptr, 0);
    std::abort();
}

// One persistent break worker per emulated core. A worker is started from ArmNce::Initialize on
// the core's own host thread, never under PhysicalCore's guard. It sleeps until SignalInterrupt
// hands it an ArmNce whose delivery gate it already marked Busy.
struct BreakWorker {
    std::atomic<ArmNce*> job{};
    std::atomic<u32> wake{};
    std::atomic<bool> stop{};
    // R2b-1: bumped by this worker when it releases Busy on an epoch RunThread is retiring, so the
    // retiring thread sleeps instead of yielding. It lives as long as the worker, not the ArmNce,
    // so the worker may touch it after the release.
    std::atomic<u64> retire_seq{};
    std::thread thread;

    void Run() {
        for (;;) {
            const u32 seen = wake.load(std::memory_order_acquire);
            if (auto* const nce = job.exchange(nullptr, std::memory_order_acq_rel)) {
                NXN1X_R2A(const std::size_t r2a_core = nce->m_core_index;)
                nce->DeliverPendingBreak();
                // The job is released; only globals may be touched from here.
                NXN1X_R2A(NCE::R2a::NoteWorkerCpu(r2a_core); NCE::R2a::MaybeFlush();)
                continue;
            }
            if (stop.load(std::memory_order_acquire)) {
                return;
            }
            wake.wait(seen, std::memory_order_acquire);
        }
    }
};

struct BreakWorkers {
    std::array<BreakWorker, Core::Hardware::NUM_CPU_CORES> workers;
    std::array<std::once_flag, Core::Hardware::NUM_CPU_CORES> started;

    ~BreakWorkers() {
        for (auto& worker : workers) {
            worker.stop.store(true, std::memory_order_release);
            worker.wake.fetch_add(1, std::memory_order_release);
            worker.wake.notify_one();
            if (worker.thread.joinable()) {
                worker.thread.join();
            }
        }
        NXN1X_R2A(NCE::R2a::MaybeFlush(true);)
    }

    void Start(std::size_t core) {
        auto& worker = workers[core % workers.size()];
        std::call_once(started[core % started.size()],
                       [&worker] { worker.thread = std::thread([&worker] { worker.Run(); }); });
    }

    std::atomic<u64>& RetireSeq(std::size_t core) {
        return workers[core % workers.size()].retire_seq;
    }

    // Lock-free; called under PhysicalCore's guard.
    void Post(std::size_t core, ArmNce* nce) {
        auto& worker = workers[core % workers.size()];
        if (worker.job.exchange(nce, std::memory_order_acq_rel) != nullptr) {
            FailFastOnJobOverwrite(core);
        }
        worker.wake.fetch_add(1, std::memory_order_release);
        worker.wake.notify_one();
    }
};

BreakWorkers g_break_workers;

using BreakApiError = NCE::WindowsCrossThreadBreak::DeliveryError;

// R2b-S12: a failed SuspendThread, GetThreadContext or SetThreadContext leaves the target running
// its original CONTEXT with the request in esr_el1, which RunThread reads on its next exit. The
// worker logs it only after the target runs again (it may hold a lock the logger needs), and only
// the first eight of each API and then each power of two, so a lasting failure cannot flood the
// log.
std::array<std::atomic<u64>, 6> g_break_api_failures{};

void LogBreakApiFailure(const BreakApiError& error, std::size_t core, bool gave_up) noexcept {
    constexpr std::array<const char*, 6> names{"none",        "unbound",     "suspend",
                                               "get_context", "set_context", "resume"};
    const std::size_t api = (std::min)(static_cast<std::size_t>(error.api), names.size() - 1);
    const u64 n = g_break_api_failures[api].fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > 8 && (n & (n - 1)) != 0) {
        return;
    }
    try {
        LOG_ERROR(Render_Vulkan,
                  "NCE_BREAK_API_FAILED api={} error={} prev_suspend={} core={} count={} gave_up={}",
                  names[api], error.last_error, error.previous_suspend_count, core, n, gave_up);
    } catch (...) {
    }
}

// R2b-S12: ResumeThread failed after this worker suspended the target. The target may stay stopped
// with that suspension outstanding, and its core would then hang without a trace. Nothing here can
// resume it, and another delivery would only nest a second SuspendThread. Stop the process instead,
// with the facts in the exception record that Windows Error Reporting keeps. No logging: the
// stopped target may hold the logger's lock.
[[noreturn]] void FailFastOnResumeFailure(const BreakApiError& error, std::size_t core,
                                          bool transformed, bool set_ok) noexcept {
    EXCEPTION_RECORD record{};
    record.ExceptionCode = 0xE04E4352; // 'NCR': NCE resume failed
    record.ExceptionFlags = EXCEPTION_NONCONTINUABLE;
    record.NumberParameters = 5;
    record.ExceptionInformation[0] = error.last_error;
    record.ExceptionInformation[1] = error.previous_suspend_count;
    record.ExceptionInformation[2] = core;
    record.ExceptionInformation[3] = transformed ? 1 : 0;
    record.ExceptionInformation[4] = set_ok ? 1 : 0;
    RaiseFailFastException(&record, nullptr, 0);
    std::abort();
}

#if NXN1X_DIAGNOSTICS
// b60: nothing in this block logs from the vectored exception handler. Eden's logger writes
// synchronously on the calling thread, and its debugger backend calls OutputDebugStringW, which
// raises DBG_PRINTEXCEPTION_WIDE_C (0x4001000A). Raised inside the handler on a guest stack, that
// exception enters the handler again (b59: NCE_STACK_FAULT n=2 counted an exception that is neither
// an access violation nor a breakpoint) and, when Windows cannot dispatch it from that stack, ends
// the process with its code: both b59 runs exited with 0x4001000A right after the n=2 line. The
// handler now only stores numbers and records; LogDeferredNceFaults (PrepareRunThread, host stack,
// outside the execution epoch) prints them.
struct DeferredFault {
    std::atomic<u32> state{}; // 0 empty, 1 written by the handler, 2 printed
    u64 n{};
    u32 code{};
    u32 instruction{};
    u64 info0{};
    u64 fault{};
    u64 pc{};
    u64 sp{};
    u64 lr{};
    u64 room{};
    u64 depth{};
    u64 begin{};
    u64 end{};
    bool pc_is_guest{};
    const char* where{};
};

std::atomic<bool> g_deferred_fault_log{};

void PublishFault(DeferredFault& record) noexcept {
    record.state.store(1, std::memory_order_release);
    g_deferred_fault_log.store(true, std::memory_order_release);
}

// b56 X18_ATOMIC_FAULT (diagnostic builds only): an access violation at a load/store-exclusive or
// LSE atomic inside NCE patch code, i.e. the native atomic of a Windows x18 exclusive/LSE trampoline
// (patcher.cpp). Before b57 such a fault let Windows write its exception record over two parked guest
// registers; the trampolines now keep their save frame across the atomic. The count is kept as a
// record, e.g. to compare with the time of a crash.
// Runs on every access violation, so the cheap tests come first: an instruction abort (fault on the
// page of Pc) is skipped before Pc is read, and the patch-code lookup (a scan of the post handlers)
// only runs for exclusive/LSE instructions.
// b60: the first 64 are recorded and printed later with their details; after that only the count,
// at powers of two.
constexpr std::size_t X18AtomicRecords = 64;
std::array<DeferredFault, X18AtomicRecords> g_x18_atomic_records{};
std::atomic<u64> g_x18_atomic_faults{};

void NoteX18AtomicFault(const EXCEPTION_POINTERS& exception, const ARM64_NT_CONTEXT& context,
                        Kernel::KProcess* process, u64 fault_address) noexcept {
    if (process == nullptr || (fault_address >> 12) == (context.Pc >> 12)) {
        return;
    }
    u32 instruction{};
    std::memcpy(&instruction, reinterpret_cast<const void*>(context.Pc), sizeof(instruction));
    const bool exclusive_class = (instruction & 0x3F000000U) == 0x08000000U;
    const bool lse_class = (instruction & 0x3F200C00U) == 0x38200000U;
    if ((!exclusive_class && !lse_class) ||
        !NCE::WindowsPatchCodeMetadata::Contains(context.Pc, process->GetPostHandlers())) {
        return;
    }
    const u64 n = g_x18_atomic_faults.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > X18AtomicRecords) {
        if (std::has_single_bit(n)) {
            g_deferred_fault_log.store(true, std::memory_order_release);
        }
        return;
    }
    const u64 context_record = reinterpret_cast<u64>(exception.ContextRecord);
    auto& record = g_x18_atomic_records[n - 1];
    record.n = n;
    record.pc = context.Pc;
    record.instruction = instruction;
    record.where = exclusive_class ? "exclusive" : "lse";
    record.sp = context.Sp;
    record.fault = fault_address;
    record.depth = context.Sp > context_record ? context.Sp - context_record : 0;
    PublishFault(record);
}

// b58 NCE_STACK_FAULT (diagnostic builds only): Windows dispatches a user-mode exception on the
// stack of the faulting thread, which under NCE is the guest stack. The kernel stores the CONTEXT
// and EXCEPTION_RECORD below the guest SP; KiUserExceptionDispatcher, the vectored handler chain and
// this handler run further below. AAPCS64 has no red zone, so this only harms the guest when its
// stack has less room below SP than the dispatch uses: the frames then land in whatever is mapped
// below the stack (a guard gap for a kernel-mapped thread stack, live data for a fiber stack carved
// out of a heap buffer). Every exception taken on a guest SP is classified by where SP is:
//   stack     inside the KMemoryState::Stack block PrepareRunThread cached for this thread;
//             room = SP - block begin
//   other     inside the cached block, which is not a Stack block; room as above
//   switched  outside the cached block: the guest moved SP inside the epoch (fibers). The room is
//             not known here; the SP is handed to PrepareRunThread, which queries its guest block
//             outside the epoch (NCE_STACK_FAULT_SP, first 32 distinct blocks).
// depth = guest SP - the SP of this function (kernel frames, dispatcher, handler chain so far).
// A fault with room < depth + 4 KiB is recorded (NCE_STACK_FAULT_TIGHT, first 64); the totals
// (NCE_STACK_FAULT) are printed when the running count reaches the next power of two.
// b60: the first 8 exceptions are also recorded with code, addresses and access type
// (NCE_FIRST_EXCEPTION). All of it is printed by LogDeferredNceFaults.
struct StackFaultTotals {
    std::atomic<u64> n{};
    std::atomic<u64> access_violation{};
    std::atomic<u64> breakpoint{};
    std::atomic<u64> stack{};
    std::atomic<u64> other{};
    std::atomic<u64> switched{};
    std::atomic<u64> tight{};
    std::atomic<u64> over{};
    std::atomic<u64> max_depth{};
    std::atomic<u64> min_room_stack{~u64{0}};
    std::atomic<u64> min_room_other{~u64{0}};
    std::atomic<const char*> last_where{};
    std::atomic<u64> last_depth{};
};
StackFaultTotals g_stack_fault{};
std::atomic<u64> g_stack_fault_pending_sp{};
std::atomic<u64> g_stack_fault_pending_depth{};
constexpr std::size_t FirstFaultRecords = 8;
constexpr std::size_t TightFaultRecords = 64;
std::array<DeferredFault, FirstFaultRecords> g_first_faults{};
std::array<DeferredFault, TightFaultRecords> g_tight_faults{};

void StoreMax(std::atomic<u64>& target, u64 value) noexcept {
    u64 current = target.load(std::memory_order_relaxed);
    while (value > current &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

void StoreMin(std::atomic<u64>& target, u64 value) noexcept {
    u64 current = target.load(std::memory_order_relaxed);
    while (value < current &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

void NoteGuestStackFault(const ArmNce& nce, const Kernel::KThread* thread, Kernel::KProcess* process,
                         const EXCEPTION_RECORD& record, const ARM64_NT_CONTEXT& context) noexcept {
    constexpr u64 TightMargin = 4096;
    volatile u8 marker = 0;
    const u64 here = reinterpret_cast<u64>(&marker);
    const u64 sp = context.Sp;
    const u64 depth = sp > here ? sp - here : 0;
    auto& t = g_stack_fault;
    const u64 n = t.n.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool access_violation = NCE::WindowsExceptionContext::IsAccessViolation(record);
    if (access_violation) {
        t.access_violation.fetch_add(1, std::memory_order_relaxed);
    } else if (record.ExceptionCode == EXCEPTION_BREAKPOINT) {
        t.breakpoint.fetch_add(1, std::memory_order_relaxed);
    }
    StoreMax(t.max_depth, depth);

    const auto* const state = FindGuestStackState(nce, thread, sp);
    const char* where = "switched";
    u64 room = 0;
    u64 begin = 0;
    u64 end = 0;
    if (state == nullptr || !state->query_ok) {
        t.switched.fetch_add(1, std::memory_order_relaxed);
        u64 expected = 0;
        if (g_stack_fault_pending_sp.compare_exchange_strong(expected, sp,
                                                             std::memory_order_relaxed)) {
            g_stack_fault_pending_depth.store(depth, std::memory_order_relaxed);
        }
    } else {
        room = sp - state->begin;
        begin = state->begin;
        end = state->end;
        where = state->is_stack ? "stack" : "other";
        (state->is_stack ? t.stack : t.other).fetch_add(1, std::memory_order_relaxed);
        StoreMin(state->is_stack ? t.min_room_stack : t.min_room_other, room);
        if (room < depth + TightMargin) {
            const u64 tight = t.tight.fetch_add(1, std::memory_order_relaxed) + 1;
            if (room < depth) {
                t.over.fetch_add(1, std::memory_order_relaxed);
            }
            if (tight <= TightFaultRecords) {
                auto& tight_record = g_tight_faults[tight - 1];
                tight_record.n = tight;
                tight_record.code = static_cast<u32>(record.ExceptionCode);
                tight_record.pc = context.Pc;
                tight_record.sp = sp;
                tight_record.where = where;
                tight_record.room = room;
                tight_record.depth = depth;
                tight_record.begin = begin;
                tight_record.end = end;
                PublishFault(tight_record);
            }
        }
    }
    t.last_where.store(where, std::memory_order_relaxed);
    t.last_depth.store(depth, std::memory_order_relaxed);

    if (n <= FirstFaultRecords) {
        auto& first = g_first_faults[n - 1];
        first.n = n;
        first.code = static_cast<u32>(record.ExceptionCode);
        first.info0 =
            record.NumberParameters >= 1 ? static_cast<u64>(record.ExceptionInformation[0]) : 0;
        first.fault =
            record.NumberParameters >= 2 ? static_cast<u64>(record.ExceptionInformation[1]) : 0;
        first.pc = context.Pc;
        first.sp = sp;
        first.lr = context.Lr;
        first.where = where;
        first.room = room;
        first.depth = depth;
        first.begin = begin;
        first.end = end;
        first.pc_is_guest = process != nullptr && process->GetMemory().IsValidVirtualAddressRange(
                                                      context.Pc, sizeof(u32));
        // An instruction abort faults on the page of Pc itself; do not read it then.
        if (first.pc_is_guest && !(access_violation && (first.fault >> 12) == (context.Pc >> 12))) {
            std::memcpy(&first.instruction, reinterpret_cast<const void*>(context.Pc),
                        sizeof(first.instruction));
        }
        PublishFault(first);
    }
    if (std::has_single_bit(n)) {
        g_deferred_fault_log.store(true, std::memory_order_release);
    }
}

// Runs from PrepareRunThread on the host stack, outside the execution epoch: prints what the
// vectored handler recorded (see the b60 note above). One core prints at a time; a core that finds
// the printer busy leaves the flag set for the next call.
void LogDeferredNceFaults() noexcept {
    if (!g_deferred_fault_log.load(std::memory_order_relaxed)) {
        return;
    }
    static std::mutex mutex;
    static u64 next_total_log = 1;
    static u64 next_x18_atomic_log = X18AtomicRecords * 2;
    std::unique_lock lock{mutex, std::try_to_lock};
    if (!lock.owns_lock() || !g_deferred_fault_log.exchange(false, std::memory_order_acquire)) {
        return;
    }
    const auto take = [](DeferredFault& record) {
        u32 expected = 1;
        return record.state.compare_exchange_strong(expected, 2, std::memory_order_acquire);
    };
    const u64 exe_base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
    for (auto& r : g_first_faults) {
        if (take(r)) {
            LOG_INFO(Render_Vulkan,
                     "NCE_FIRST_EXCEPTION n={} code={:#x} info0={:#x} fault={:#x} pc={:#x} "
                     "pc_is_guest={} insn={:#010x} sp={:#x} lr={:#x} where={} room={:#x} "
                     "depth={:#x} block={:#x}-{:#x} exe_base={:#x}",
                     r.n, r.code, r.info0, r.fault, r.pc, r.pc_is_guest, r.instruction, r.sp, r.lr,
                     r.where, r.room, r.depth, r.begin, r.end, exe_base);
        }
    }
    for (auto& r : g_tight_faults) {
        if (take(r)) {
            LOG_WARNING(Render_Vulkan,
                        "NCE_STACK_FAULT_TIGHT n={} code={:#x} pc={:#x} sp={:#x} where={} "
                        "room={:#x} depth={:#x} block={:#x}-{:#x}",
                        r.n, r.code, r.pc, r.sp, r.where, r.room, r.depth, r.begin, r.end);
        }
    }
    for (auto& r : g_x18_atomic_records) {
        if (take(r)) {
            LOG_INFO(Render_Vulkan,
                     "X18_ATOMIC_FAULT n={} pc={:#x} insn={:#010x} kind={} sp={:#x} fault={:#x} "
                     "context_record_below_sp={:#x}",
                     r.n, r.pc, r.instruction, r.where, r.sp, r.fault, r.depth);
        }
    }
    const u64 x18_atomic = g_x18_atomic_faults.load(std::memory_order_relaxed);
    if (x18_atomic >= next_x18_atomic_log) {
        next_x18_atomic_log = std::bit_ceil(x18_atomic + 1);
        LOG_INFO(Render_Vulkan, "X18_ATOMIC_FAULT n={} (details for the first {} only)",
                 x18_atomic, X18AtomicRecords);
    }
    auto& t = g_stack_fault;
    const auto get = [](const std::atomic<u64>& v) { return v.load(std::memory_order_relaxed); };
    const u64 n = get(t.n);
    if (n >= next_total_log) {
        next_total_log = std::bit_ceil(n + 1);
        const auto room_or_none = [&](const std::atomic<u64>& v) {
            const u64 value = get(v);
            return value == ~u64{0} ? std::string{"none"} : fmt::format("{:#x}", value);
        };
        const char* const last_where = t.last_where.load(std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan,
                 "NCE_STACK_FAULT n={} av={} brk={} stack={} other={} switched={} tight={} over={} "
                 "max_depth={:#x} min_room_stack={} min_room_other={} last_where={} "
                 "last_depth={:#x}",
                 n, get(t.access_violation), get(t.breakpoint), get(t.stack), get(t.other),
                 get(t.switched), get(t.tight), get(t.over), get(t.max_depth),
                 room_or_none(t.min_room_stack), room_or_none(t.min_room_other),
                 last_where != nullptr ? last_where : "none", get(t.last_depth));
    }
}

// Runs from PrepareRunThread, outside the execution epoch, where the guest page-table lock may be
// taken: the guest block of an SP that faulted outside its cached stack block.
void LogPendingStackFaultSp(Kernel::KThread* thread) noexcept {
    if (g_stack_fault_pending_sp.load(std::memory_order_relaxed) == 0 || thread == nullptr ||
        thread->GetOwnerProcess() == nullptr) {
        return;
    }
    const u64 depth = g_stack_fault_pending_depth.load(std::memory_order_relaxed);
    const u64 sp = g_stack_fault_pending_sp.exchange(0, std::memory_order_relaxed);
    if (sp == 0) {
        return;
    }
    Kernel::KMemoryInfo info{};
    Kernel::Svc::PageInfo page_info{};
    const auto result = thread->GetOwnerProcess()->GetPageTable().QueryInfo(
        std::addressof(info), std::addressof(page_info), Kernel::KProcessAddress{sp - 1});
    const u64 begin = result.IsSuccess() ? static_cast<u64>(info.GetAddress()) : 0;
    const u64 end = result.IsSuccess() ? static_cast<u64>(info.GetEndAddress()) : 0;
    static std::mutex mutex;
    static std::array<u64, 32> logged{};
    static std::size_t logged_count = 0;
    std::scoped_lock lock{mutex};
    if (std::find(logged.begin(), logged.begin() + logged_count, begin) !=
            logged.begin() + logged_count ||
        logged_count == logged.size()) {
        return;
    }
    logged[logged_count++] = begin;
    LOG_INFO(Render_Vulkan,
             "NCE_STACK_FAULT_SP n={} sp={:#x} query_ok={} state={:#x} is_stack={} "
             "block={:#x}-{:#x} room={:#x} depth={:#x}",
             logged_count, sp, result.IsSuccess(),
             result.IsSuccess() ? static_cast<u32>(info.GetState()) : 0U,
             result.IsSuccess() && info.GetState() == Kernel::KMemoryState::Stack, begin, end,
             begin != 0 && sp > begin ? sp - begin : 0, depth);
}
#endif

// NXN1X_CRASH_RING (default 1, its own flag apart from NXN1X_DIAGNOSTICS: NCE_CLEANUP_REVIEW.md P3): the ring is the only record a fast-fail 0xd leaves;
// -DNXN1X_CRASH_RING=0 builds it out (CrashRingRecord does nothing, no crashring.bin)
#ifndef NXN1X_CRASH_RING
#define NXN1X_CRASH_RING 1
#endif
// crashring (11-20, diagnostic, log only): fast-fail 0xd at ntdll+0x34a10 is RtlRestoreContext refusing a context whose SP is outside the TEB's
// StackLimit..StackBase (one guard page of grace below StackLimit when StackLimit is not DeallocationStack: the check at ntdll+0x34720). Three crashes
// (HWAoI 11:11, 11:22; Bayonetta 3 11:53), dumps without stacks. Every exception reaching this vectored handler is recorded at entry and at return in a
// 256-entry ring in a file-backed mapping (crashring.bin beside the exe), which outlives a fast-fail: the context's SP against the TEB bounds by that
// same rule, the thread, the code and addresses, the nesting depth and the handler's answer. crashring_dump prints it. Stores into the mapping only
struct CrashRingEntry {
    u64 seq, qpc;
    u32 tid, code;
    u64 address, pc, sp, fp, lr, stack_base, stack_limit, dealloc;
    u32 phase, result, depth, flags;   // phase 1 enter, 2 return, 3 note; flags bit0: SP valid for RtlRestoreContext, bit1: SP in the thread's own reservation
    u64 pad[3];
};
static_assert(sizeof(CrashRingEntry) == 128);
struct CrashRingHeader { u64 magic, entries, next, pid, invalid, skipped_debug_prints; u64 pad[10]; };   // invalid: records whose SP RtlRestoreContext would refuse
// (the first 32 are kept after the ring, as a crash may come long after one); skipped_debug_prints: OutputDebugString exceptions (0x40010006 / 0x4001000a),
// not recorded (a run's last 256 records were all of them, at shutdown)
static CrashRingHeader* g_crash_ring = nullptr;
static CrashRingEntry* g_crash_entries = nullptr;
static thread_local u32 t_crash_depth = 0;
// stage2a (diagnostic): abort / terminate / CRT invalid parameter / pure call (the 16:21 crash: c0000409 at ucrtbase+0x798ec, a CRT fast-fail, no log or ring
// entry) write nxn1x_abort_<pid>.txt beside the exe first: the reason and the calling thread's frames as module+offset. Win32 calls and a fixed buffer only
// stage2c (11-20): the report of 17:18 and 18:03 came out empty: two threads called die() at once (FATAL NVN command memory twice), the first created the file and
// stopped in the frame lookup (GetModuleHandleExA takes the loader lock; snprintf the CRT's), the second returned at once and the process ended. Now: the file
// path and the module bases are fixed when the hooks are installed; the reason (and die()'s message, dm_fatal_msg) is written and flushed first; frames are raw
// addresses with the exe's and ntdll's bases beside them (no lookup, no lock); a second caller waits (up to 2 s) for the first to finish
#if NXN1X_DIAGNOSTICS   // the abort report (process-wide SIGABRT/CRT/purecall/terminate handlers): diagnostic builds (NCE_CLEANUP_REVIEW.md P3)
static wchar_t g_abort_path[MAX_PATH];
static uintptr_t g_abort_exe_base, g_abort_ntdll_base, g_abort_ucrt_base;
static std::atomic<int> g_abort_state{0};   // 0 none, 1 writing, 2 written
static void AbortWrite(HANDLE f, const char* s, size_t n) noexcept { DWORD w = 0; WriteFile(f, s, static_cast<DWORD>(n), &w, nullptr); FlushFileBuffers(f); }
static size_t AbortHex(char* o, uint64_t v) noexcept { static const char h[] = "0123456789abcdef"; char t[16]; int k = 0; do { t[k++] = h[v & 15]; v >>= 4; } while (v && k < 16);
    o[0] = '0'; o[1] = 'x'; for (int i = 0; i < k; i++) o[2 + i] = t[k - 1 - i]; return static_cast<size_t>(k + 2); }
static void AbortReport(const char* why) noexcept {
    int expected = 0;
    if (!g_abort_state.compare_exchange_strong(expected, 1)) {   // another thread is writing it: let it finish
        for (int i = 0; i < 2000 && g_abort_state.load() == 1; i++) Sleep(1);
        return;
    }
    if (g_abort_path[0] == 0) { g_abort_state.store(2); return; }
    HANDLE f = CreateFileW(g_abort_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { g_abort_state.store(2); return; }
    char line[1024]; size_t n = 0;
    const auto put = [&](const char* s) { while (*s && n < sizeof line - 1) line[n++] = *s++; };
    put("abort report: "); put(why); put(", thread "); n += AbortHex(line + n, GetCurrentThreadId()); put("\n");
    if (dm_fatal_msg[0]) { put("FATAL: "); char m[512]; std::memcpy(m, dm_fatal_msg, sizeof m); m[sizeof m - 1] = 0; put(m); put("\n"); }
    put("bases: NXN1X.exe "); n += AbortHex(line + n, g_abort_exe_base); put(", ntdll "); n += AbortHex(line + n, g_abort_ntdll_base); put(", ucrtbase "); n += AbortHex(line + n, g_abort_ucrt_base); put("\n");
    AbortWrite(f, line, n);   // the reason first, flushed, before anything that could stop
    void* frames[48]; const USHORT nf = RtlCaptureStackBackTrace(1, 48, frames, nullptr);
    for (USHORT i = 0; i < nf; i++) {
        n = 0; put("  #"); n += AbortHex(line + n, i); put(" "); n += AbortHex(line + n, reinterpret_cast<uintptr_t>(frames[i]));
        const uintptr_t a = reinterpret_cast<uintptr_t>(frames[i]);
        if (g_abort_exe_base && a >= g_abort_exe_base && a < g_abort_exe_base + 0x8000000) { put(" (NXN1X.exe+"); n += AbortHex(line + n, a - g_abort_exe_base); put(")"); }
        put("\n"); AbortWrite(f, line, n);
    }
    CloseHandle(f);
    g_abort_state.store(2);
}
static void AbortSignal(int) { AbortReport("SIGABRT (abort)"); }
static void AbortInvalidParameter(const wchar_t*, const wchar_t*, const wchar_t*, unsigned int, uintptr_t) { AbortReport("CRT invalid parameter"); }
static void AbortPurecall() { AbortReport("pure virtual call"); }
static void AbortTerminate() { AbortReport("std::terminate"); std::abort(); }
static void InstallAbortReport() {
    {   // stage2c: everything AbortReport needs that could take a lock, fixed now
        const DWORD n = GetModuleFileNameW(nullptr, g_abort_path, MAX_PATH);
        wchar_t* slash = n && n < MAX_PATH ? wcsrchr(g_abort_path, L'\\') : nullptr;
        if (slash != nullptr && slash - g_abort_path + 40 < MAX_PATH) swprintf(slash + 1, MAX_PATH - (slash + 1 - g_abort_path), L"nxn1x_abort_%lu.txt", GetCurrentProcessId());
        else g_abort_path[0] = 0;
        g_abort_exe_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
        g_abort_ntdll_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(L"ntdll.dll"));
        g_abort_ucrt_base = reinterpret_cast<uintptr_t>(GetModuleHandleW(L"ucrtbase.dll"));
    }
    dm_fatal_hook = [] { AbortReport("die (FATAL)"); };   // stage2c3: die() reports directly
    std::signal(SIGABRT, AbortSignal);
    _set_invalid_parameter_handler(AbortInvalidParameter);
    _set_purecall_handler(AbortPurecall);
    std::set_terminate(AbortTerminate);
}
#endif
static void CrashRingInit() {
#if NXN1X_DIAGNOSTICS
    InstallAbortReport();   // stage2a
#endif
#if NXN1X_CRASH_RING
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return;
    wchar_t* slash = wcsrchr(path, L'\\');
    if (slash == nullptr || slash - path + 16 >= MAX_PATH) return;
    wcscpy(slash + 1, L"crashring.bin");
    {   // stage2a: the previous run's ring is kept as crashring_prev.bin (a crash's ring was lost when NXN1X was started again right after it)
        wchar_t prev[MAX_PATH]; wcscpy(prev, path); wchar_t* ps = wcsrchr(prev, L'\\'); wcscpy(ps + 1, L"crashring_prev.bin");
        MoveFileExW(path, prev, MOVEFILE_REPLACE_EXISTING);
    }
    const DWORD size = sizeof(CrashRingHeader) + (256 + 32) * sizeof(CrashRingEntry);
    HANDLE file = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    HANDLE map = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, size, nullptr);
    CloseHandle(file);
    if (map == nullptr) return;
    void* view = MapViewOfFile(map, FILE_MAP_ALL_ACCESS, 0, 0, size);
    CloseHandle(map);
    if (view == nullptr) return;
    auto* h = static_cast<CrashRingHeader*>(view);
    std::memset(view, 0, size);
    h->magic = 0x474e495248534152ull;   // "RASHRING"
    h->entries = 256;
    h->pid = GetCurrentProcessId();
    g_crash_entries = reinterpret_cast<CrashRingEntry*>(h + 1);
    g_crash_ring = h;
    LOG_INFO(Core_ARM, "crashring: exceptions reaching the NCE handler are recorded in crashring.bin beside the exe (crashring_dump prints them)");
#endif
}
static void CrashRingRecord(u32 phase, u32 result, const EXCEPTION_POINTERS* e, u64 note0 = 0, u64 note1 = 0, u64 note2 = 0) {   // notes: pad[0..2] (phase 3)
#if !NXN1X_CRASH_RING
    (void)phase; (void)result; (void)e; (void)note0; (void)note1; (void)note2;
    return;
#else
    CrashRingHeader* const h = g_crash_ring;
    if (h == nullptr) return;
    if (e != nullptr && e->ExceptionRecord != nullptr &&
        (e->ExceptionRecord->ExceptionCode == 0x40010006 || e->ExceptionRecord->ExceptionCode == 0x4001000a)) {
        __atomic_fetch_add(&h->skipped_debug_prints, phase == 1 ? 1 : 0, __ATOMIC_RELAXED);
        return;
    }
    const u64 seq = __atomic_fetch_add(&h->next, 1, __ATOMIC_RELAXED);
    CrashRingEntry& r = g_crash_entries[seq & 255];
    LARGE_INTEGER q{};
    QueryPerformanceCounter(&q);
    const auto* const tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    const u64 base = reinterpret_cast<u64>(tib->StackBase), limit = reinterpret_cast<u64>(tib->StackLimit);
    const u64 dealloc = *reinterpret_cast<const u64*>(reinterpret_cast<const u8*>(NtCurrentTeb()) + 0x1478);   // DeallocationStack (ntdll RtlRestoreContext reads it here)
    r.seq = seq + 1;
    r.qpc = static_cast<u64>(q.QuadPart);
    r.tid = GetCurrentThreadId();
    r.phase = phase;
    r.result = result;
    r.depth = t_crash_depth;
    r.stack_base = base;
    r.stack_limit = limit;
    r.dealloc = dealloc;
    u64 sp = 0;
    if (e != nullptr && e->ExceptionRecord != nullptr && e->ContextRecord != nullptr) {
        r.code = e->ExceptionRecord->ExceptionCode;
        r.address = e->ExceptionRecord->NumberParameters >= 2 ? static_cast<u64>(e->ExceptionRecord->ExceptionInformation[1])
                                                              : reinterpret_cast<u64>(e->ExceptionRecord->ExceptionAddress);
        r.pc = e->ContextRecord->Pc;
        sp = e->ContextRecord->Sp;
        r.fp = e->ContextRecord->Fp;
        r.lr = e->ContextRecord->Lr;
    } else {
        r.code = 0;
        r.address = r.pc = r.fp = r.lr = 0;
    }
    r.sp = sp;
    r.pad[0] = note0; r.pad[1] = note1; r.pad[2] = note2;
    const bool valid = (sp >= limit && sp <= base) || (limit != dealloc && sp >= limit - 0x1000 && sp <= base);
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    r.flags = (valid ? 1u : 0u) | (sp > low && sp <= high ? 2u : 0u);
    if (!valid && sp != 0 && phase == 2 && result == static_cast<u32>(EXCEPTION_CONTINUE_EXECUTION)) {   // tebfix2: counted where RtlRestoreContext checks it (a resume)
        const u64 k = __atomic_fetch_add(&h->invalid, 1, __ATOMIC_RELAXED);
        if (k < 32) g_crash_entries[256 + k] = r;
    }
#endif
}

// tebfix5 (11-20): one rule for the TEB stack bounds, by SP alone. Replaces tebfix2 (refit when the PC is not in an image) and tebfix4 (and when
// the PC is in the asm ranges that run on a guest SP), which classified by PC and needed a new exception for each resume path found:
//   tebfix2: a guest fault resumed on another guest stack (a fiber or coroutine stack), 15:42
//   tebfix4: a cross-thread break moved the PC to WindowsNceHostStackBridge while the thread's own fault was being delivered, 16:54
// and tebfix4 itself crashed (0400E88025B1A000 copy, 17:06): it logged before refitting; the logger's OutputDebugStringW raised 0x4001000A on
// the guest stack, which Windows could not dispatch with SP outside the TEB bounds (WER: KERNELBASE+0x11f144, 0x4001000A; crash ring: enter at
// the bridge's first instruction, no return). The bounds Windows checks (RtlRestoreContext on a resume, the frame walk of any exception raised
// while the handler runs) only mean "the stack SP is on", so they are set from SP:
//   - SP accepted as the bounds stand (RtlRestoreContext's rule): nothing (the usual case; no system call)
//   - SP on this thread's own host stack (DeallocationStack .. the top of that reservation): the host bounds
//   - SP in guest memory (the process's mapped address space; guest code, fibers and game-allocated stacks alike): the allocation holding SP
//   - anything else: left as it is, and counted
// Applied twice per exception: at entry, for the stack the handler itself runs on (so an exception raised inside the handler or a callee can be
// dispatched), and before every EXCEPTION_CONTINUE_EXECUTION, for the SP being resumed. The paths that leave through NtContinue (RedirectToHost to
// WindowsNceHostStackBridge) publish their own bounds in the bridge. Nothing here logs: counters only, printed by LogTebFitCounters (PrepareRunThread,
// host stack, outside the execution epoch)
enum TebFit : int { TebFitValid = 0, TebFitHost = 1, TebFitGuest = 2, TebFitNoGuestContext = 3, TebFitNotGuestMemory = 4, TebFitQueryFailed = 5,
                    TebFitOutsideAllocation = 6, TebFitNoSp = 7, TebFitKinds = 8 };
struct TebFitCounts {
    std::atomic<u64> entry[TebFitKinds]{};    // at handler entry (the handler's own SP)
    std::atomic<u64> resume[TebFitKinds]{};   // before EXCEPTION_CONTINUE_EXECUTION (the context's SP)
};
static TebFitCounts g_teb_fit{};
static std::atomic<bool> g_teb_fit_dirty{false};

static bool TebAcceptsSp(u64 sp, u64 base, u64 limit, u64 dealloc) noexcept {   // RtlRestoreContext's check (ntdll+0x34720)
    return (sp >= limit && sp <= base) || (limit != dealloc && sp >= limit - 0x1000 && sp <= base);
}

// The end of the reservation holding p (VirtualQuery walk over its regions); 0 when it cannot be queried
static u64 AllocationOf(u64 p, u64& alloc_base) noexcept {
    MEMORY_BASIC_INFORMATION first{};
    if (VirtualQuery(reinterpret_cast<const void*>(p), &first, sizeof(first)) == 0 || first.AllocationBase == nullptr) return 0;
    alloc_base = reinterpret_cast<u64>(first.AllocationBase);
    u64 aend = alloc_base;
    for (u64 cursor = alloc_base;;) {
        MEMORY_BASIC_INFORMATION region{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &region, sizeof(region)) == 0 || region.AllocationBase != first.AllocationBase) break;
        const u64 end = reinterpret_cast<u64>(region.BaseAddress) + static_cast<u64>(region.RegionSize);
        if (end <= cursor) break;
        aend = end; cursor = end;
    }
    return aend;
}

static int FitTebToSp(u64 sp) noexcept {
    if (sp == 0) return TebFitNoSp;
    auto* const tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    const u64 base = reinterpret_cast<u64>(tib->StackBase), limit = reinterpret_cast<u64>(tib->StackLimit);
    const u64 dealloc = *reinterpret_cast<const u64*>(reinterpret_cast<const u8*>(NtCurrentTeb()) + 0x1478);   // DeallocationStack (never changed by NCE)
    if (TebAcceptsSp(sp, base, limit, dealloc)) return TebFitValid;
    // this thread's own host stack: DeallocationStack is the bottom of its reservation
    if (dealloc != 0 && sp > dealloc) {
        u64 host_low = 0;
        const u64 host_top = AllocationOf(dealloc, host_low);
        if (host_top != 0 && sp <= host_top) {
            const GuestContext* const g = NCE::WindowsExceptionContext::CurrentGuestContext();
            u64 new_limit = sp & ~u64{0xFFF};
            if (g != nullptr && g->windows_host_stack_base == host_top && g->windows_host_stack_limit != 0 && g->windows_host_stack_limit <= sp)
                new_limit = g->windows_host_stack_limit;   // the limit this thread saved at its guest entry
            tib->StackLimit = reinterpret_cast<void*>(new_limit);
            tib->StackBase = reinterpret_cast<void*>(host_top);
            return TebFitHost;
        }
    }
    // guest memory: only a thread running guest code (a guest context installed) takes guest bounds
    const GuestContext* const g = NCE::WindowsExceptionContext::CurrentGuestContext();
    if (g == nullptr || g->parent == nullptr || g->parent->m_running_thread == nullptr) return TebFitNoGuestContext;
    Kernel::KProcess* const process = g->parent->m_running_thread->GetOwnerProcess();
    if (process == nullptr || !process->GetMemory().IsValidVirtualAddressRange(sp - 1, 1)) return TebFitNotGuestMemory;
    u64 abase = 0;
    const u64 aend = AllocationOf(sp - 1, abase);
    if (aend == 0) return TebFitQueryFailed;
    if (!(sp > abase && sp <= aend)) return TebFitOutsideAllocation;
    tib->StackLimit = reinterpret_cast<void*>(abase);
    tib->StackBase = reinterpret_cast<void*>(aend);
    return TebFitGuest;
}

// Exceptions the handler leaves alone, recorded for LogTebFitCounters (the handler itself never logs)
struct HandlerNote {
    std::atomic<u64> n{};
    std::atomic<u64> pc{}, sp{}, fault{}, extra{};
};
static HandlerNote g_note_illegal{}, g_note_host_fault{};
static void NoteForLater(HandlerNote& h, u64 pc, u64 sp, u64 fault, u64 extra) noexcept {
    if (h.n.fetch_add(1, std::memory_order_relaxed) < 4) {
        h.pc.store(pc, std::memory_order_relaxed); h.sp.store(sp, std::memory_order_relaxed);
        h.fault.store(fault, std::memory_order_relaxed); h.extra.store(extra, std::memory_order_relaxed);
    }
    g_teb_fit_dirty.store(true, std::memory_order_release);
}

// PrepareRunThread (host stack, outside the execution epoch): what the handler counted. A line when a count it shows has reached the next power of two
void LogTebFitCounters() noexcept {
    if (!g_teb_fit_dirty.load(std::memory_order_relaxed) && NCE::g_tebfix3_bind_other.load(std::memory_order_relaxed) == 0 &&
        NCE::g_tebfix3_x18_other.load(std::memory_order_relaxed) == 0)
        return;
    static std::mutex mutex;
    std::unique_lock lock{mutex, std::try_to_lock};
    if (!lock.owns_lock()) return;
    g_teb_fit_dirty.store(false, std::memory_order_relaxed);
    const auto ld = [](const std::atomic<u64>& v) { return v.load(std::memory_order_relaxed); };
    static u64 shown_fit = 0, shown_b3 = 0, shown_x3 = 0, shown_ill = 0, shown_hf = 0;
    u64 fitted = 0;
    for (int k = TebFitHost; k < TebFitKinds; ++k) fitted += ld(g_teb_fit.entry[k]) + ld(g_teb_fit.resume[k]);
    const auto due = [](u64 now, u64& shown) { if (now == 0 || std::bit_width(now) == std::bit_width(shown)) return false; shown = now; return true; };
    if (due(fitted, shown_fit)) {
        const auto& e = g_teb_fit.entry; const auto& r = g_teb_fit.resume;
        LOG_WARNING(Core_ARM, "tebfix5: TEB stack bounds set from SP. At handler entry: valid {}, host stack {}, guest memory {}, left (no guest context {}, "
                    "not guest memory {}, query failed {}, outside allocation {}). Before a resume: valid {}, host stack {}, guest memory {}, left (no guest "
                    "context {}, not guest memory {}, query failed {}, outside allocation {}, no SP {})",
                    ld(e[0]), ld(e[1]), ld(e[2]), ld(e[3]), ld(e[4]), ld(e[5]), ld(e[6]),
                    ld(r[0]), ld(r[1]), ld(r[2]), ld(r[3]), ld(r[4]), ld(r[5]), ld(r[6]), ld(r[7]));
    }
    if (due(NCE::g_tebfix3_bind_other.load(std::memory_order_relaxed), shown_b3))
        LOG_WARNING(Core_ARM, "tebfix3: redirects whose cross-thread break binding was another thread's: {}", shown_b3);
    if (due(NCE::g_tebfix3_x18_other.load(std::memory_order_relaxed), shown_x3))
        LOG_WARNING(Core_ARM, "tebfix3: redirects with an x18 other than the thread's TEB in the CONTEXT (the bridge sets x18 from x4): {}", shown_x3);
    if (due(ld(g_note_illegal.n), shown_ill))
        LOG_ERROR(Core_ARM, "NCE guest illegal instruction: {} (last of the first 4: pc={:#x} sp={:#x} insn={:#010x})", shown_ill,
                  ld(g_note_illegal.pc), ld(g_note_illegal.sp), ld(g_note_illegal.extra));
    if (due(ld(g_note_host_fault.n), shown_hf))
        LOG_ERROR(Core_ARM, "NCE host access fault not handled: {} (last of the first 4: pc={:#x} sp={:#x} fault={:#x} exe_base={:#x})", shown_hf,
                  ld(g_note_host_fault.pc), ld(g_note_host_fault.sp), ld(g_note_host_fault.fault),
                  reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr)));
}

// nceresume (11-20, review R1/R2/R3/R6/R9/R10/R15: design/ASTRA_NCE_RESUME_REVIEW_REQUEST.md section 9). Every resume this handler makes goes through
// ResumeChecked: the VEH's EXCEPTION_CONTINUE_EXECUTION (wrapper below) and the direct NtContinue resumes (ContinueChecked: the x18 trap, breakpoint and
// guest access-fault redirects to the host bridge, the cache-op emulation), and S36's mapping handler (host_memory.cpp, through the hook it is given).
//  1. A PC inside a transition's publish window is moved back to the window's start (R2): between the TEB store and the MOV SP the bounds already
//     describe the next stack while SP is still the previous one; fitting them to the current SP and resuming there would leave them wrong after the
//     MOV SP. The windows re-run safely: WindowsNceHostStackBridge from its first instruction (x1-x4, x16 unchanged), WindowsNceEnterGuest from
//     WindowsNceEnterGuestPublish (x14 / x15 recomputed from x17, x30 = guest SP unchanged).
//  2. The bounds are set from the resumed SP (FitTebToSp, the tebfix5 rule).
//  3. An SP the rule cannot place (no guest context, not guest memory, no allocation, outside it, none) is not resumed (R1): the crash ring gets the
//     decision and the counters are written, then the process ends with NCE fast-fail code 0xE0CE0001 (no log, no allocation). Before nceresume such a
//     resume went on with bounds RtlRestoreContext may refuse (0xd) or the next dispatch may not walk.
// NCE fast-fail codes: 0xE0CE0001 a resume whose SP the rule cannot place; 0xE0CE0004 a redirect with no valid host bounds (windows_nce_transition.cpp)
// R15: t_crash_depth is now the active nesting (decremented before a direct NtContinue too); g_veh_entries counts entries, g_veh_direct direct resumes
extern "C" void WindowsNceEnterGuestPublish();
extern "C" void WindowsNceEnterGuestOnGuestSp();
extern "C" void WindowsNceHostStackBridge();
extern "C" void WindowsNceHostStackBridgeEnd();
static std::atomic<u64> g_resume_restart_bridge{0}, g_resume_restart_enter{0}, g_veh_entries{0}, g_veh_direct{0}, g_guest_bounds_refreshed{0};
static u64 SpNow() noexcept { u64 v; asm volatile("mov %0, sp" : "=r"(v)); return v; }
static void NormalizeTransitionPc(ARM64_NT_CONTEXT& ctx) noexcept {
    const u64 pc = ctx.Pc;
    const u64 hb = reinterpret_cast<u64>(&WindowsNceHostStackBridge), he = reinterpret_cast<u64>(&WindowsNceHostStackBridgeEnd);
    if (pc > hb && pc < he) { ctx.Pc = hb; g_resume_restart_bridge.fetch_add(1, std::memory_order_relaxed); g_teb_fit_dirty.store(true, std::memory_order_relaxed); return; }
    const u64 eb = reinterpret_cast<u64>(&WindowsNceEnterGuestPublish), ee = reinterpret_cast<u64>(&WindowsNceEnterGuestOnGuestSp);
    if (pc > eb && pc <= ee) { ctx.Pc = eb; g_resume_restart_enter.fetch_add(1, std::memory_order_relaxed); g_teb_fit_dirty.store(true, std::memory_order_relaxed); }
}
[[noreturn]] static void NceFailFast(u32 code, PEXCEPTION_POINTERS e, u64 a, u64 b) noexcept {
    EXCEPTION_RECORD r{};
    r.ExceptionCode = code; r.NumberParameters = 4;
    r.ExceptionInformation[0] = static_cast<ULONG_PTR>(a); r.ExceptionInformation[1] = static_cast<ULONG_PTR>(b);
    r.ExceptionInformation[2] = e && e->ContextRecord ? static_cast<ULONG_PTR>(e->ContextRecord->Pc) : 0;
    r.ExceptionInformation[3] = e && e->ContextRecord ? static_cast<ULONG_PTR>(e->ContextRecord->Sp) : 0;
    RaiseFailFastException(&r, e ? e->ContextRecord : nullptr, 0);
    for (;;) {}
}
// The checked resume of ctx (e's context, or a copy of it): see above. Returns only when the resume may go ahead
static void ResumeChecked(PEXCEPTION_POINTERS e, ARM64_NT_CONTEXT& ctx) noexcept {
    NormalizeTransitionPc(ctx);
    const auto* const tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
    const u64 base0 = reinterpret_cast<u64>(tib->StackBase), limit0 = reinterpret_cast<u64>(tib->StackLimit);
    const int why = FitTebToSp(ctx.Sp);
    g_teb_fit.resume[why].fetch_add(1, std::memory_order_relaxed);
    if (why == TebFitValid) return;
    g_teb_fit_dirty.store(true, std::memory_order_relaxed);
    if (e != nullptr) CrashRingRecord(3, static_cast<u32>(why), e, static_cast<u64>(why), base0, limit0);
    if (why >= TebFitNoGuestContext) NceFailFast(0xE0CE0001u, e, static_cast<u64>(why), ctx.Sp);   // R1
}
// nceresume R15: PrepareRunThread (host stack): entries, direct resumes, restarted transition windows, refreshed guest bounds; a line at powers of two
void LogNceResumeCounters() noexcept {
    static std::atomic<u64> shown{0};
    const u64 e = g_veh_entries.load(std::memory_order_relaxed);
    u64 s = shown.load(std::memory_order_relaxed);
    if (e == 0 || std::bit_width(e) == std::bit_width(s) || !shown.compare_exchange_strong(s, e)) return;
    LOG_INFO(Core_ARM, "nceresume: handler entries {}, resumed directly (NtContinue) {}, transition windows restarted (host bridge {}, guest entry {}), "
             "guest stack bounds refreshed for the direct x18 return {}, active nesting on this thread {}",
             e, g_veh_direct.load(std::memory_order_relaxed), g_resume_restart_bridge.load(std::memory_order_relaxed),
             g_resume_restart_enter.load(std::memory_order_relaxed), g_guest_bounds_refreshed.load(std::memory_order_relaxed), t_crash_depth);
}
static bool g_inline_fault_enabled = false;   // nceresume R5: EDEN_NCE_INLINE_FAULT, read when the handler is installed
// nceresume R9: another handler (S36, host_memory.cpp) resuming a thread directly: on a thread running guest code the same check as ours
static void NceResumeCheckHook(void* raw) noexcept {
    auto* const e = static_cast<PEXCEPTION_POINTERS>(raw);
    if (e == nullptr || e->ContextRecord == nullptr || NCE::WindowsExceptionContext::CurrentGuestContext() == nullptr) return;
    ResumeChecked(e, *reinterpret_cast<ARM64_NT_CONTEXT*>(e->ContextRecord));
}
// nceresume R14 (NXN1X_NCE_SELFTEST=1, at handler installation, host thread): the parts of the resume check that run without a guest:
//  T1 the transition-window PCs are moved to the window's start, and only those; T2 a host SP with foreign bounds is fitted to the host stack;
//  T3 an SP in memory no guest owns, on a thread with no guest context, is reported as not placeable (the fail-fast decision, not taken here);
//  T4 an SP already inside the bounds leaves them as they are
static void NceResumeSelfTest() noexcept {
    int ok = 0, n = 0;
    const auto check = [&](const char* name, bool pass) { n++; ok += pass; LOG_INFO(Core_ARM, "nceresume self-test {}: {}", name, pass ? "PASS" : "FAIL"); };
    const u64 hb = reinterpret_cast<u64>(&WindowsNceHostStackBridge), he = reinterpret_cast<u64>(&WindowsNceHostStackBridgeEnd);
    const u64 eb = reinterpret_cast<u64>(&WindowsNceEnterGuestPublish), ee = reinterpret_cast<u64>(&WindowsNceEnterGuestOnGuestSp);
    const auto norm = [](u64 pc) { ARM64_NT_CONTEXT c{}; c.Pc = pc; NormalizeTransitionPc(c); return c.Pc; };
    bool t1 = norm(hb) == hb && norm(he) == he && norm(ee + 4) == ee + 4 && norm(eb) == eb;
    for (u64 pc = hb + 4; pc < he; pc += 4) t1 = t1 && norm(pc) == hb;
    for (u64 pc = eb + 4; pc <= ee; pc += 4) t1 = t1 && norm(pc) == eb;
    check("T1 transition windows restart at their start (host bridge, guest entry publish), other PCs unchanged", t1);
    auto* const tib = reinterpret_cast<NT_TIB*>(NtCurrentTeb());
    void* const base0 = tib->StackBase; void* const limit0 = tib->StackLimit;
    u8* const fake = static_cast<u8*>(VirtualAlloc(nullptr, 1u << 20, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (fake != nullptr) {
        tib->StackLimit = fake; tib->StackBase = fake + (1u << 20);   // foreign bounds (no logging or call that could raise until restored)
        const u64 sp = SpNow(); const int w = FitTebToSp(sp);
        const u64 b = reinterpret_cast<u64>(tib->StackBase), l = reinterpret_cast<u64>(tib->StackLimit);
        tib->StackBase = base0; tib->StackLimit = limit0;
        check("T2 host SP with foreign bounds: set to the host stack and accepted", w == TebFitHost && sp >= l && sp <= b);
        const int w3 = FitTebToSp(reinterpret_cast<u64>(fake) + (1u << 19));
        tib->StackBase = base0; tib->StackLimit = limit0;
        check("T3 SP in memory no guest owns, no guest context: not placeable (fail-fast decision)", w3 == TebFitNoGuestContext);
        VirtualFree(fake, 0, MEM_RELEASE);
    } else { check("T2 (no memory for the fake stack)", false); check("T3 (no memory)", false); }
    const int w4 = FitTebToSp(SpNow());
    check("T4 SP inside the bounds: left as they are", w4 == TebFitValid && tib->StackBase == base0 && tib->StackLimit == limit0);
    LOG_INFO(Core_ARM, "{}nceresume self-test: {} of {} passed", ok == n ? "" : "[RED] ", ok, n);
}
// NXN1X_NCE_SELFTEST=1: PatchText on a small synthetic .text. L1 CASP x17,x18 (odd pair, no rewrite exists) is left as data:
// PatchText returns true (false means "out of branch range" and nso.cpp would retry it forever) and patches nothing.
// L2 the bounded form CASP x18,x19 is still patched (the test reaches the LSE path)
static void PatcherSelfTest() noexcept {
    const auto run = [](u32 word, size_t& sites) {
        std::vector<u8> image(64 * sizeof(u32));
        for (size_t i = 0; i < 64; ++i) {
            const u32 nop = 0xD503201FU;
            std::memcpy(image.data() + i * sizeof(u32), &nop, sizeof(u32));
        }
        std::memcpy(image.data() + 16 * sizeof(u32), &word, sizeof(u32));
        Kernel::CodeSet::Segment seg{};
        seg.offset = 0;
        seg.size = static_cast<u32>(image.size());
        NCE::Patcher p;
        const bool ok = p.PatchText(image, seg);
        sites = p.PatchedSiteCount();
        return ok;
    };
    size_t s1 = 0, s2 = 0;
    const bool r1 = run(0x48317C02U, s1);
    const bool r2 = run(0x48327C02U, s2);
    const bool l1 = r1 && s1 == 0, l2 = r2 && s2 == 1;
    LOG_INFO(Core_ARM, "patcher self-test L1 odd-pair CASP x18 left as data: {} (returned {}, sites {})", l1 ? "PASS" : "FAIL", r1, s1);
    LOG_INFO(Core_ARM, "patcher self-test L2 bounded CASP x18 patched: {} (returned {}, sites {})", l2 ? "PASS" : "FAIL", r2, s2);
    LOG_INFO(Core_ARM, "{}patcher self-test: {} of 2 passed", l1 && l2 ? "" : "[RED] ", int(l1) + int(l2));
}
// A direct resume (NtContinue) from inside the handler: checked, the nesting closed, then continued
[[noreturn]] static void ContinueChecked(PEXCEPTION_POINTERS e, ARM64_NT_CONTEXT& ctx) noexcept {
    ResumeChecked(e, ctx);
    g_veh_direct.fetch_add(1, std::memory_order_relaxed);
    if (e != nullptr) CrashRingRecord(2, 0xC0C0C0C0u, e);   // (phase 2 with this marker: resumed directly)
    if (t_crash_depth) --t_crash_depth;
    NCE::WindowsNceTransition::ContinueContext(ctx);
}

static LONG WindowsNceVectoredExceptionHandlerImpl(PEXCEPTION_POINTERS exception) noexcept;
LONG CALLBACK WindowsNceVectoredExceptionHandler(PEXCEPTION_POINTERS exception) noexcept {
    {   // tebfix5: the stack this handler runs on first, before anything that could raise (nceresume R6: the real SP, not a local's address)
        const int w = FitTebToSp(SpNow());
        g_teb_fit.entry[w].fetch_add(1, std::memory_order_relaxed);
        if (w != TebFitValid) g_teb_fit_dirty.store(true, std::memory_order_relaxed);
    }
    ++t_crash_depth;
    g_veh_entries.fetch_add(1, std::memory_order_relaxed);
    CrashRingRecord(1, 0, exception);
    const LONG r = WindowsNceVectoredExceptionHandlerImpl(exception);
    if (r == EXCEPTION_CONTINUE_EXECUTION && exception != nullptr && exception->ContextRecord != nullptr)
        ResumeChecked(exception, *reinterpret_cast<ARM64_NT_CONTEXT*>(exception->ContextRecord));   // nceresume
    CrashRingRecord(2, static_cast<u32>(r), exception);
    --t_crash_depth;
    return r;
}

static LONG WindowsNceVectoredExceptionHandlerImpl(PEXCEPTION_POINTERS exception) noexcept {
    if (exception == nullptr || exception->ExceptionRecord == nullptr ||
        exception->ContextRecord == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // NVN path, step 4 (NVN_HOST_BRIDGE.md 4.2): a write to a page the NVN layer write-protects to watch it, from any thread
    // (guest code, the NVN gate's host code, an emulator thread). The layer marks the page written and makes it writable, and
    // the write runs again at once, as in the native host's own fault handler: no trip through RunThread for these pages.
    // Atomics and VirtualProtect only. With too little stack left (a guest stack) the usual path reaches Memory::InvalidateNCE,
    // which asks the layer the same question
    if (NCE::WindowsExceptionContext::GetAccessType(*exception->ExceptionRecord) ==
        NCE::WindowsExceptionContext::AccessType::Write) {
        const auto* const tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
        const u64 sp = exception->ContextRecord->Sp;
        const u64 stack_limit = reinterpret_cast<u64>(tib->StackLimit);
        constexpr u64 WatchedWriteStackHeadroom = 16 * 1024;
        // s25: on a host thread's own stack the room is measured to the bottom of its reservation (GetCurrentThreadStackLimits), less a guard margin:
        // TIB StackLimit is only as low as the thread has ever reached, so a thread with a shallow stack (the Vulkan completion thread copying a pool
        // back into the game's memory) had less than the headroom above it and its write to a watched page went unhandled (run 30, 11-12). A guest
        // stack (outside those limits) as before
        ULONG_PTR stack_low = 0;
        ULONG_PTR stack_high = 0;
        GetCurrentThreadStackLimits(&stack_low, &stack_high);
        constexpr u64 HostStackGuard = 64 * 1024;
        const bool host_stack = sp > stack_low && sp <= stack_high;
        // nceresume R7: on a guest stack the room is measured to the bottom of the guest stack block PrepareRunThread found for this thread
        // (FindGuestStackState), not to TIB StackLimit: tebfix5 sets StackLimit to the base of the allocation holding SP, which for a fiber carved out
        // of a heap block is far below the fiber's own bottom. An SP outside the known block (a fiber) has no known room: the usual path below
        (void)stack_limit;
        u64 guest_room = 0;
        if (!host_stack) {
            const auto* const g = NCE::WindowsExceptionContext::CurrentGuestContext();
            if (g != nullptr && g->parent != nullptr && g->parent->m_running_thread != nullptr)
                if (const auto* const st = FindGuestStackState(*g->parent, g->parent->m_running_thread, sp); st != nullptr && st->query_ok && sp > st->begin)
                    guest_room = sp - st->begin;
        }
        const u64 room = host_stack ? (sp - stack_low > HostStackGuard ? sp - stack_low - HostStackGuard : 0) : guest_room;
        const u64 fault_address = reinterpret_cast<u64>(NCE::WindowsExceptionContext::GetFaultAddress(*exception->ExceptionRecord));
        const bool room_ok = room >= WatchedWriteStackHeadroom;
        const bool handled = room_ok && Core::NvnLayer::OnWatchedWrite(fault_address);
        // vbvol (diagnostic): the thread kind and outcome of each write fault the watch saw or could not take; a fault that was not the
        // layer's page (handled false, room true) is the usual NCE path below and is counted as passed
        Core::NvnLayer::NoteWatchFault(host_stack, room_ok, handled, fault_address, exception->ContextRecord->Pc, sp);
        if (handled) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }

    auto* const guest = NCE::WindowsExceptionContext::CurrentGuestContext();
    if (guest == nullptr || guest->parent == nullptr || guest->parent->m_running_thread == nullptr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    auto* const nce = guest->parent;
    auto& context = *reinterpret_cast<ARM64_NT_CONTEXT*>(exception->ContextRecord);

    // The Windows transition and arbitrary-PC restore helpers execute on the original host stack.
    // Exceptions there belong to Windows/host code and must remain chainable.
    if (nce->m_windows_break != nullptr && nce->m_windows_break->IsHostStackPointer(context.Sp)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    auto* const thread = nce->m_running_thread;
    auto* const process = thread->GetOwnerProcess();
    auto* const params = &thread->GetNativeExecutionParameters();
    RecordNceException(*exception->ExceptionRecord, context);
#if NXN1X_DIAGNOSTICS
    NoteGuestStackFault(*nce, thread, process, *exception->ExceptionRecord, context);
#endif

    // IMP-006 ordinary guest-x18 sites deliberately trap with BRK #0xF000. Once the tagged
    // process-owned metadata confirms the site, transfer ownership of the native-parameter lock
    // back to the host return path and reuse the proven x18 trap/transition seam.
    if (exception->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT &&
        NCE::WindowsX18FallbackTrap::FindOriginalInstruction(
            context.Pc, process->GetPostHandlers()).has_value()) {
        params->lock.store(SpinLockLocked, std::memory_order_release);
        const bool redirected = NCE::WindowsX18FallbackTrap::TryRedirect(
            exception, *guest, process->GetPostHandlers());
        if (redirected) {
            NCE::Stats::Add(NCE::Stats::RedirectX18Trap);
            context.X[18] = reinterpret_cast<u64>(NtCurrentTeb());
            ContinueChecked(exception, context);   // nceresume R3
        }
        params->lock.store(SpinLockUnlocked, std::memory_order_release);
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Preserve the working 20da host-return recovery without turning a host/bridge transition
    // breakpoint into a guest prefetch fault. PhysicalCore treats PrefetchAbort as a debug-stop,
    // while BreakLoop is the scheduler's normal "leave native execution and reschedule" reason.
    // Use the already-proven guest-PC ownership test to distinguish the two meanings: a breakpoint
    // outside the guest address space is a transient host/bridge seam and must not close the guest
    // execution resume gate. Unknown breakpoints at a real guest PC retain the existing behavior.
    if (exception->ExceptionRecord->ExceptionCode == EXCEPTION_BREAKPOINT) {
        const bool breakpoint_pc_is_guest =
            process != nullptr &&
            process->GetMemory().IsValidVirtualAddressRange(context.Pc, sizeof(u32));
        const auto reason = breakpoint_pc_is_guest ? HaltReason::PrefetchAbort
                                                   : HaltReason::BreakLoop;

        params->lock.store(SpinLockLocked, std::memory_order_release);
        NCE::Stats::Add(NCE::Stats::RedirectVehBreakpoint);
        // Save guest state for a breakpoint at a real guest PC so diagnostics and any debugger see
        // the faulting instruction instead of the PC of the previous host transition.
        NCE::WindowsNceTransition::RedirectToHost(context, *guest, breakpoint_pc_is_guest,
                                                  static_cast<u64>(reason), reinterpret_cast<u64>(NtCurrentTeb()));   // tebfix3: the faulting thread itself
        context.X[18] = reinterpret_cast<u64>(NtCurrentTeb());
        ContinueChecked(exception, context);   // nceresume R3
    }

    // Windows keeps SCTLR_EL1.UCI clear, so EL0 cache maintenance by VA (DC CVAU/CVAC/CIVAC/
    // CVAP/CVADP, IC IVAU) raises STATUS_ILLEGAL_INSTRUCTION. Guest code that patches itself
    // (exlaunch-based mods such as TotK FPS/UltraCam patches, __clear_cache) relies on these.
    // Emulate them: IC IVAU / DC CVAU become a host instruction-cache flush of the line, the
    // data-cache clean variants are no-ops on the coherent host, then resume after the op.
    if (exception->ExceptionRecord->ExceptionCode == EXCEPTION_ILLEGAL_INSTRUCTION &&
        process != nullptr &&
        process->GetMemory().IsValidVirtualAddressRange(context.Pc, sizeof(u32))) {
        u32 instruction{};
        std::memcpy(&instruction, reinterpret_cast<const void*>(context.Pc), sizeof(instruction));
        const u32 op = instruction & ~0x1FU;
        const u32 rt = instruction & 0x1FU;
        constexpr u32 IcIvau = 0xD50B7520U;
        constexpr u32 DcCvau = 0xD50B7B20U;
        constexpr u32 DcCvac = 0xD50B7A20U;
        constexpr u32 DcCivac = 0xD50B7E20U;
        constexpr u32 DcCvap = 0xD50B7C20U;
        constexpr u32 DcCvadp = 0xD50B7D20U;
        const bool flush_icache = op == IcIvau || op == DcCvau;
        const bool clean_only = op == DcCvac || op == DcCivac || op == DcCvap || op == DcCvadp;
        if ((flush_icache || clean_only) && rt != 18) {
            if (flush_icache) {
                const u64 address = rt == 31 ? 0 : context.X[rt];
                constexpr u64 Line = 64;
                (void)FlushInstructionCache(GetCurrentProcess(),
                                            reinterpret_cast<const void*>(address & ~(Line - 1)),
                                            Line);
            }
            NCE::Stats::Add(NCE::Stats::CacheOpEmulated);
            context.Pc += sizeof(u32);
            context.X[18] = reinterpret_cast<u64>(NtCurrentTeb());
            ContinueChecked(exception, context);   // nceresume R3
        }
        NoteForLater(g_note_illegal, context.Pc, context.Sp, 0, instruction);   // tebfix5: printed by LogTebFitCounters (no logging in the handler)
    }

    if (NCE::WindowsExceptionContext::IsAccessViolation(*exception->ExceptionRecord)) {
        const auto fault_address = reinterpret_cast<u64>(
            NCE::WindowsExceptionContext::GetFaultAddress(*exception->ExceptionRecord));
#if NXN1X_DIAGNOSTICS
        NoteX18AtomicFault(*exception, context, process, fault_address);
#endif
        // A read of a page the NCE read guard protects (core/memory.cpp); never a write.
        const bool read_fault = NCE::WindowsExceptionContext::GetAccessType(
                                    *exception->ExceptionRecord) ==
                                NCE::WindowsExceptionContext::AccessType::Read;

        // Host code can run on a guest-owned thread inside a native epoch (generated-code C
        // helpers, the direct x18 Dynarmic helper and its memory callbacks). An access violation
        // there is a host fault, never guest state: saving it as GuestContext and redirecting would
        // later "resume" host code as guest code on a stale host frame. Mirror Linux
        // HandleHostAccessFault instead: invalidate the guest page and retry, or let the fault
        // propagate to the next handler.
        if (process == nullptr ||
            !process->GetMemory().IsValidVirtualAddressRange(context.Pc, sizeof(u32))) {
            NCE::Stats::Add(NCE::Stats::HostAccessFault);
            // Host code reading a guarded page: let it read guest memory as before the guard. The
            // handler cannot wait for the GPU thread, and invalidating would drop the GPU content.
            if (process != nullptr && read_fault &&
                process->GetMemory().HandleNceReadFault(Common::ProcessAddress{fault_address},
                                                        false)) {
#if NXN1X_DIAGNOSTICS
                Common::NceReadGuard::host_reads.fetch_add(1, std::memory_order_relaxed);
#endif
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            if (process != nullptr &&
                process->GetMemory().InvalidateNCE(
                    Common::ProcessAddress{fault_address & ~Memory::YUZU_PAGEMASK},
                    Memory::YUZU_PAGESIZE)) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            NoteForLater(g_note_host_fault, context.Pc, context.Sp, fault_address, 0);   // tebfix5: printed by LogTebFitCounters (no logging in the handler)
            return EXCEPTION_CONTINUE_SEARCH;
        }

        // Linux HandleGuestAccessFault resolves most faults (rasterizer write tracking) inside the
        // signal handler and resumes guest code directly. Do the same here when the guest stack
        // the VEH is running on has comfortable headroom; otherwise take the host-return path.
        {
            // nceresume R7: room to the bottom of the known guest stack block (see the watched-write path), not TIB StackLimit
            const auto* const st = FindGuestStackState(*nce, thread, context.Sp);
            const u64 stack_bottom = st != nullptr && st->query_ok ? st->begin : ~u64{0};
            constexpr u64 InlineFaultStackHeadroom = 64 * 1024;
            // Opt-in (EDEN_NCE_INLINE_FAULT=1): enabling it unconditionally hung TotK after
            // loading, so the host-return path stays the default until that is understood.
            // nceresume R5: read when the handler is installed (g_inline_fault_enabled), not on first use inside it
            if (g_inline_fault_enabled && !read_fault && context.Sp > stack_bottom &&
                context.Sp - stack_bottom >= InlineFaultStackHeadroom &&
                process->GetMemory().InvalidateNCE(
                    Common::ProcessAddress{fault_address & ~Memory::YUZU_PAGEMASK},
                    Memory::YUZU_PAGESIZE)) {
                NCE::Stats::Add(NCE::Stats::FaultInline);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }

        nce->m_windows_pending_nce_fault = true;
        nce->m_windows_pending_nce_fault_address = fault_address;
        nce->m_windows_pending_nce_fault_page = fault_address & ~Memory::YUZU_PAGEMASK;
        nce->m_windows_pending_nce_fault_read = read_fault;
        params->lock.store(SpinLockLocked, std::memory_order_release);
        NCE::Stats::Add(NCE::Stats::RedirectVehAccess);
        NCE::WindowsNceTransition::RedirectToHost(
            context, *guest, true, static_cast<u64>(HaltReason::PrefetchAbort), reinterpret_cast<u64>(NtCurrentTeb()));   // tebfix3: the faulting thread itself

        // The x18 breakpoint path already uses the proven direct NtContinue transition after
        // RedirectToHost. Do the same for AV recovery instead of returning the mutated CONTEXT to
        // the Windows exception dispatcher. V72 natural dumps showed a reproducible second
        // execute-AV at PC=SP=0 with TEB StackBase/StackLimit=0, exactly matching
        // WindowsNceHostStackBridge executing after its volatile x1/x2/x3/x16 scratch values were
        // lost. Direct NtContinue keeps the redirected register contract intact through resume.
        context.X[18] = reinterpret_cast<u64>(NtCurrentTeb());

        // Resume the redirected host-return context directly. RedirectToHost already selected
        // the standard WindowsNceHostStackBridge and populated its host bounds/SP/return-PC contract.
        ContinueChecked(exception, context);   // nceresume R3
    }

    // IMP-008A does not claim complete game fault compatibility. Unknown host/guest exception
    // classes remain chainable instead of being swallowed by the NCE VEH.
    return EXCEPTION_CONTINUE_SEARCH;
}

#if NXN1X_DIAGNOSTICS
// NCE_WRITE_NOTE (b71): the guest state of a faulting store, for the texture cache trace
// (cpu_write_trace.h). RunThread calls this on the CPU thread before InvalidateNCE, never inside
// the exception handler. Guest memory is read only through host pointers and only while reactive
// flushing is off: with it on, the GPU-cached pages are not readable and a read would fault.
void FillWriteNote(Common::NceWriteNote::Note& note, Memory::Memory& memory,
                   const GuestContext& ctx, u64 thread_id) {
    namespace NW = Common::NceWriteNote;
    note.filled = true;
    note.seq = NW::sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    note.thread = thread_id;
    note.fp = ctx.cpu_registers[29];
    note.lr = ctx.cpu_registers[30];
    note.sp = ctx.sp;
    const NW::Operands op = NW::Decode(note.insn);
    note.store = op.kind;
    const auto gpr = [&ctx](u8 reg) { return reg == 31 ? u64{0} : ctx.cpu_registers[reg]; };
    if (op.kind == NW::StoreVector) {
        note.data[0] = ctx.vector_registers[op.rt][0];
        note.data[1] = ctx.vector_registers[op.rt][1];
        if (op.pair) {
            note.data[2] = ctx.vector_registers[op.rt2][0];
            note.data[3] = ctx.vector_registers[op.rt2][1];
        }
    } else if (op.kind == NW::StoreGpr || op.kind == NW::StoreZr) {
        note.data[0] = gpr(op.rt);
        if (op.pair) {
            note.data[2] = gpr(op.rt2);
        }
    }
    for (size_t i = 0; i < NW::REGS; ++i) {
        note.x[i] = ctx.cpu_registers[i];
    }
    if (Settings::values.use_reactive_flushing.GetValue()) {
        return;
    }
    const auto count = [&memory](u64 va, u8& zeros, u8& bytes) -> const u8* {
        const u8* const host = memory.GetPointerSilent(Common::ProcessAddress{va});
        if (host == nullptr) {
            return nullptr;
        }
        const u64 length = (std::min)(NW::READ_BYTES, NW::GUEST_PAGE - (va & (NW::GUEST_PAGE - 1)));
        zeros = static_cast<u8>(std::count(host, host + length, u8{0}));
        bytes = static_cast<u8>(length);
        return host;
    };
    const u8* const fault_host = count(note.fault, note.dst_zeros, note.dst_bytes);
    for (size_t i = 0; i < NW::REGS; ++i) {
        const u8* const host = count(note.x[i], note.zeros[i], note.bytes[i]);
        note.contig[i] = host != nullptr && fault_host != nullptr &&
                                 reinterpret_cast<uintptr_t>(host) -
                                         reinterpret_cast<uintptr_t>(fault_host) ==
                                     note.x[i] - note.fault
                             ? 1
                             : 0;
    }
    // Frame-pointer chain: [fp] = caller fp, [fp + 8] = return address.
    u64 frame = note.fp;
    for (size_t i = 0; i < NW::FRAMES; ++i) {
        if (frame == 0 || (frame & 0xF) != 0 || (frame & (NW::GUEST_PAGE - 1)) > NW::GUEST_PAGE - 16) {
            break;
        }
        const u8* const host = memory.GetPointerSilent(Common::ProcessAddress{frame});
        if (host == nullptr) {
            break;
        }
        u64 next = 0;
        std::memcpy(&next, host, sizeof(next));
        std::memcpy(&note.frames[i], host + 8, sizeof(u64));
        if (next <= frame) {
            break;
        }
        frame = next;
    }
}
#endif

} // namespace

// nceresume R8 (windows_x18_fallback_runner.cpp): the direct x18 helper returns to the guest through a D2 handoff that publishes
// windows_guest_stack_* (patcher.cpp). Those were computed at the guest entry for the entry SP; after a fiber switch they describe another stack.
// Refreshed here for the SP the guest resumes on (host code, host stack: VirtualQuery is fine). Left as they are when SP is inside them or no
// allocation holds it (the VEH's checked resume then decides at the next fault)
void NCE::WindowsNceRefreshGuestStackBounds(GuestContext& guest) noexcept {
    const u64 sp = guest.sp;
    if (sp == 0 || (sp > guest.windows_guest_stack_limit && sp <= guest.windows_guest_stack_base)) return;
    u64 abase = 0;
    const u64 aend = AllocationOf(sp - 1, abase);
    if (aend == 0 || !(sp > abase && sp <= aend)) return;
    guest.windows_guest_stack_limit = abase;
    guest.windows_guest_stack_base = aend;
    g_guest_bounds_refreshed.fetch_add(1, std::memory_order_relaxed);
    g_teb_fit_dirty.store(true, std::memory_order_relaxed);
}

ArmNce::ArmNce(System& system, bool uses_wall_clock, std::size_t core_index)
    : ArmInterface{uses_wall_clock}, m_system{system}, m_core_index{core_index},
      m_windows_break{std::make_unique<NCE::WindowsCrossThreadBreak>()} {
    m_guest_ctx.system = &m_system;
}

ArmNce::~ArmNce() = default;

void ArmNce::Initialize() {
#if NXN1X_DIAGNOSTICS
    static std::once_flag recover_off_once;
    std::call_once(recover_off_once, [] {
        const auto& off = RecoveryOff();
        LOG_WARNING(Render_Vulkan,
                    "NCE_RECOVER_OFF subst={:#x} bitfield={:#x} mrs={:#x} cond={:#x}",
                    off[0], off[1], off[2], off[3]);
    });
#endif
    if (!NCE::WindowsNceTransition::Initialize()) {
        LOG_CRITICAL(Core_ARM, "Failed to resolve Windows NCE NtContinue transition");
    }

    if (m_windows_break != nullptr) {
        if (!m_windows_break->BindCurrentThread()) {
            LOG_CRITICAL(Core_ARM, "Failed to bind Windows NCE cross-thread break target");
        }
    }
    g_break_workers.Start(m_core_index);

    std::call_once(g_windows_veh_once, [] {
        CrashRingInit();   // crashring (diagnostic)
        {   // nceresume R5: options the handler reads, fixed here (no getenv / first-use initialization inside it)
            const char* const value = std::getenv("EDEN_NCE_INLINE_FAULT");
            g_inline_fault_enabled = value != nullptr && value[0] != '\0' && value[0] != '0';
        }
        Common::SetResumeCheckHook(&NceResumeCheckHook);   // nceresume R9: S36's direct resumes take the same check
        if (const char* const t = std::getenv("NXN1X_NCE_SELFTEST"); t != nullptr && t[0] == '1') { NceResumeSelfTest(); PatcherSelfTest(); }   // nceresume R14, patcher LSE data
        g_windows_veh_handle = AddVectoredExceptionHandler(1, &WindowsNceVectoredExceptionHandler);
        if (g_windows_veh_handle == nullptr) {
            LOG_CRITICAL(Core_ARM, "Failed to install Windows NCE vectored exception handler");
        }
    });
}

void ArmNce::LockThreadParameters(void* raw_params) {
    auto* const params = static_cast<NativeExecutionParameters*>(raw_params);
    for (;;) {
        u32 expected = SpinLockUnlocked;
        if (params->lock.compare_exchange_weak(expected, SpinLockLocked,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
            return;
        }
        std::this_thread::yield();
    }
}

void ArmNce::UnlockThreadParameters(void* raw_params) {
    auto* const params = static_cast<NativeExecutionParameters*>(raw_params);
    params->lock.store(SpinLockUnlocked, std::memory_order_release);
}

void ArmNce::LockThread(Kernel::KThread* thread) {
    LockThreadParameters(&thread->GetNativeExecutionParameters());
}

void ArmNce::UnlockThread(Kernel::KThread* thread) {
    auto* const thread_params = &thread->GetNativeExecutionParameters();
    // Only adopt the TLS registers RunThread published for this context. RunThread can return
    // before publishing (a pending BreakLoop, e.g. the IPI that starts a newly created thread);
    // copying back then would replace the TLS address LoadContext just installed with the
    // thread's stale parameter value (0 for a thread that has never run). If the scheduler then
    // re-runs the same thread without a context switch, guest TPIDRRO_EL0 reads return 0 and
    // nn::os thread entry dereferences TLS+0x1f8 -> null, followed by a userspace panic.
    if (thread_params->native_context == &m_guest_ctx) {
        m_guest_ctx.tpidr_el0 = thread_params->tpidr_el0;
        m_guest_ctx.tpidrro_el0 = thread_params->tpidrro_el0;
    }
    thread_params->native_context = nullptr;
    UnlockThreadParameters(thread_params);
}

bool ArmNce::PrepareRunThread(Kernel::KThread* thread) {
    // Runs before PhysicalCore::EnterContext, where blocking on the guest page-table lock is as safe
    // as inside an SVC handler. RunThread only consumes the cached result.
#if NXN1X_DIAGNOSTICS
    LogDeferredNceFaults();
    LogPendingStackFaultSp(thread);
#endif
    LogTebFitCounters();   // tebfix5
    LogNceResumeCounters();   // nceresume
    const u64 guest_sp = m_guest_ctx.sp;
    if (thread == nullptr || guest_sp == 0 ||
        FindGuestStackState(*this, thread, guest_sp) != nullptr) {
        return false;
    }

    auto* const process = thread->GetOwnerProcess();
    if (process == nullptr) {
        return false;
    }

    Kernel::KMemoryInfo info{};
    Kernel::Svc::PageInfo page_info{};
    const auto result = process->GetPageTable().QueryInfo(
        std::addressof(info), std::addressof(page_info), Kernel::KProcessAddress{guest_sp - 1});

    WindowsGuestStackState state{};
    state.thread_id = thread->GetThreadId();
    state.valid = true;
    if (result.IsFailure()) {
        // Cache the failure for this page so RunThread reports it instead of looping.
        state.begin = (guest_sp - 1) & ~static_cast<u64>(Memory::YUZU_PAGEMASK);
        state.end = state.begin + Memory::YUZU_PAGESIZE;
    } else {
        state.begin = static_cast<u64>(info.GetAddress());
        state.end = static_cast<u64>(info.GetEndAddress());
        state.query_ok = true;
        state.is_stack = info.GetState() == Kernel::KMemoryState::Stack;
    }

    // QueryInfo may have rescheduled; the cache slot is chosen only afterwards.
    m_windows_stack_states[m_windows_stack_state_next++ % m_windows_stack_states.size()] = state;
    return true;
}

HaltReason ArmNce::RunThread(Kernel::KThread* thread) {
    HaltReason hr = static_cast<HaltReason>(m_guest_ctx.esr_el1.exchange(0));
    if (True(hr)) {
        NXN1X_R2A(NCE::R2a::NoteHostSideConsumed(
                        m_r2a_host_side_post_ns.exchange(0, std::memory_order_relaxed),
                        NCE::R2a::HostSideNextEntry);)
        return hr;
    }

    auto* const thread_params = &thread->GetNativeExecutionParameters();
    auto* const process = thread->GetOwnerProcess();

    m_running_thread = thread;
    m_guest_ctx.parent = this;
    m_guest_ctx.windows_native_execution_parameters = thread_params;
    thread_params->native_context = &m_guest_ctx;
    thread_params->tpidr_el0 = m_guest_ctx.tpidr_el0;
    thread_params->tpidrro_el0 = m_guest_ctx.tpidrro_el0;

    std::atomic_thread_fence(std::memory_order_release);
    thread_params->is_running = true;

    // Open a delivery epoch. The previous epoch was fully drained when it retired, so no worker
    // can still hold this gate. HostSide: the loop below reads esr_el1 before any guest entry.
    {
        const u64 generation = (m_windows_delivery_gate.load(std::memory_order_relaxed) &
                                ~DeliveryFlagMask) +
                               (DeliveryFlagMask + 1);
        m_windows_delivery_gate.store(generation | DeliveryActive | DeliveryHostSide,
                                      std::memory_order_seq_cst);
    }

    if (m_windows_x18_runner == nullptr) {
        m_windows_x18_runner = std::make_unique<NCE::WindowsX18FallbackRunner>(
            m_system, m_uses_wall_clock, process, m_core_index);
    }
#if NXN1X_DIAGNOSTICS
    // Logs the EDEN_NCE_WRITE_STEP mode once per process, before any fault.
    (void)NCE::WriteStep::GetMode();
#endif

    std::optional<Common::HostMemory::PrivateMappingLease> private_stack_lease;
    const char* abort_site = "native-exit";
    u64 last_native_exit = 0;

    for (;;) {
        NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::EntryPrepare);)
        // R2b-B3: past this read the thread may enter guest code without reading esr_el1 again, so
        // a request from here on has to be delivered by capture. The worker may already have left
        // a request to this read on seeing HostSide; this acquire RMW on the gate follows that
        // request's gate update, so the read below sees it.
        m_windows_delivery_gate.fetch_and(~DeliveryHostSide, std::memory_order_acquire);
        // SignalInterrupt may defer delivery when another host-side NCE path already owns the
        // parameters lock. Honor that pending break before this lock-owning path can re-enter
        // native guest execution.
        const auto pending_reason =
            static_cast<HaltReason>(m_guest_ctx.esr_el1.exchange(0, std::memory_order_acq_rel));
        if (True(pending_reason)) {
            NXN1X_R2A(NCE::R2a::NoteHostSideConsumed(
                            m_r2a_host_side_post_ns.exchange(0, std::memory_order_relaxed),
                            NCE::R2a::HostSideLoopTop);)
            hr = pending_reason;
            break;
        }

        // The guest page-table state of the active stack is resolved by PrepareRunThread. If guest
        // code switched to a stack outside that block during this epoch, leave with BreakLoop so
        // PhysicalCore re-enters through PrepareRunThread instead of querying the kernel here.
        const auto* const guest_stack_state = FindGuestStackState(*this, thread, m_guest_ctx.sp);
        if (guest_stack_state == nullptr) {
            hr = HaltReason::BreakLoop;
            break;
        }
        if (!guest_stack_state->query_ok) {
            LOG_ERROR(Core_ARM, "Windows NCE guest stack page-table query failed");
            abort_site = "stack-query";
            hr = HaltReason::PrefetchAbort;
            break;
        }
        const u64 stats_prep_t0 = NCE::Stats::Now();
        NCE::CurrentNceContext::Install(thread_params, __LINE__);

        if (!EnsureWindowsGuestStackLease(m_system, process, m_guest_ctx.sp,
                                          guest_stack_state->is_stack,
                                          private_stack_lease)) {
            NCE::CurrentNceContext::Clear(__LINE__);
            abort_site = "stack-lease";
            hr = HaltReason::PrefetchAbort;
            break;
        }

        WindowsTebStackBounds teb_stack_bounds{};
        if (!InstallGuestTebStackBounds(m_guest_ctx, teb_stack_bounds)) {
            NCE::CurrentNceContext::Clear(__LINE__);
            abort_site = "teb-bounds";
            hr = HaltReason::PrefetchAbort;
            break;
        }

        if (private_stack_lease.has_value()) {
            m_windows_x18_runner->SetDirectPrivateStackView(
                m_guest_ctx.windows_guest_stack_limit,
                m_guest_ctx.windows_guest_stack_base - m_guest_ctx.windows_guest_stack_limit);
        } else {
            m_windows_x18_runner->ClearDirectPrivateStackView();
        }

        const u64 stats_native_t0 = NCE::Stats::Now();
        NCE::Stats::Add(NCE::Stats::PrepNanos, stats_native_t0 - stats_prep_t0);
        const u64 stats_epoch_tag = ++t_stats_epoch_tag;
        const DWORD stats_entry_os_thread = GetCurrentThreadId();
        const u64 stats_entry_host_sp = m_guest_ctx.host_ctx.host_sp;
        NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::Transition);)
        const auto& entry_handlers = process->GetPostHandlers();
        if (const auto it = entry_handlers.find(m_guest_ctx.pc); it != entry_handlers.end()) {
            hr = static_cast<HaltReason>(NCE::WindowsNceEnterGuest(
                &m_guest_ctx, reinterpret_cast<const void*>(it->second)));
        } else {
            hr = static_cast<HaltReason>(NCE::WindowsNceEnterGuestContext(&m_guest_ctx));
        }
        // R2b-B3: back in host code. Every path from here reaches the esr_el1 read at the top of
        // this loop, or leaves RunThread, whose next call reads esr_el1 first, before guest code
        // runs again.
        m_windows_delivery_gate.fetch_or(DeliveryHostSide, std::memory_order_relaxed);

        NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::HostReturn);)
        RestoreHostTebStackBounds(teb_stack_bounds);
        last_native_exit = static_cast<u64>(hr);
        NCE::CurrentNceContext::Clear(__LINE__);

        if (NCE::Stats::Enabled()) {
            const u64 raw_hr = static_cast<u64>(hr);
            NCE::Stats::Add(NCE::Stats::NativeEntries);
            if (stats_epoch_tag != t_stats_epoch_tag) {
                NCE::Stats::Add(NCE::Stats::HostFrameMismatch);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                    u8 frame_probe{};
                    LOG_ERROR(Core_ARM,
                              "NCE_FRAME_MISMATCH hr={:#x} tag local={} now={} os_thread "
                              "entry={} now={} host_sp entry={:#x} ctx_now={:#x} frame={:#x} "
                              "t0_age_ms={} guest_pc={:#x} thread={:#x} running={:#x}",
                              raw_hr, stats_epoch_tag, t_stats_epoch_tag,
                              stats_entry_os_thread, GetCurrentThreadId(), stats_entry_host_sp,
                              m_guest_ctx.host_ctx.host_sp,
                              reinterpret_cast<std::uintptr_t>(&frame_probe),
                              (NCE::Stats::Now() - stats_native_t0) / 1'000'000,
                              m_guest_ctx.pc, reinterpret_cast<std::uintptr_t>(thread),
                              reinterpret_cast<std::uintptr_t>(m_running_thread));
                }
            }
            const u64 native_delta = NCE::Stats::Now() - stats_native_t0;
            // A single native epoch longer than 1s means the timestamp did not come from this
            // iteration; count it separately instead of polluting the total.
            if (native_delta > 1'000'000'000ULL) {
                NCE::Stats::Add(NCE::Stats::NativeOutliers);
            } else {
                NCE::Stats::Add(NCE::Stats::NativeNanos, native_delta);
            }
            if (raw_hr == NCE::WindowsX18FallbackRunner::DirectStackRefreshMarker) {
                NCE::Stats::Add(NCE::Stats::ExitStackRefresh);
            } else if (m_windows_pending_nce_fault) {
                NCE::Stats::Add(NCE::Stats::ExitFault);
            } else if (raw_hr == NCE::WindowsX18FallbackTrap::ReturnMarker) {
                NCE::Stats::Add(NCE::Stats::ExitX18Fallback);
                if (const auto original = NCE::WindowsX18FallbackTrap::FindOriginalInstruction(
                        m_guest_ctx.pc, process->GetPostHandlers())) {
                    NCE::Stats::RecordTrapSite(m_guest_ctx.pc, *original);
                }
            } else if (True(hr & HaltReason::SupervisorCall)) {
                NCE::Stats::Add(NCE::Stats::ExitSvc);
            } else if (True(hr & HaltReason::BreakLoop)) {
                NCE::Stats::Add(NCE::Stats::ExitBreakLoop);
            } else if (raw_hr == 0) {
                NCE::Stats::Add(NCE::Stats::ExitNoReason);
            } else {
                NCE::Stats::Add(NCE::Stats::ExitOther);
            }
            NCE::Stats::MaybeReport();
        }

        // P3C stack-rollover marker is internal to this RunThread epoch. The direct trampoline
        // deliberately did not execute the x18 instruction. Loop once so the existing
        // EnsureWindowsGuestStackLease/InstallGuestTebStackBounds path can roll the private lease
        // and publish bounds for the current guest SP, then re-enter at the same guest PC.
        if (static_cast<u64>(hr) ==
            NCE::WindowsX18FallbackRunner::DirectStackRefreshMarker) {
            m_windows_x18_runner->ClearDirectPrivateStackView();
            continue;
        }

        if (m_windows_pending_nce_fault) {
            const u64 pending_fault_address = m_windows_pending_nce_fault_address;
            const u64 pending_fault_page = m_windows_pending_nce_fault_page;
            m_windows_pending_nce_fault = false;
            m_windows_pending_nce_fault_address = 0;
            m_windows_pending_nce_fault_page = 0;
            const bool pending_fault_read = m_windows_pending_nce_fault_read;
            m_windows_pending_nce_fault_read = false;
            // NCE read guard (core/memory.cpp): a guest read of a guarded page takes the Dynarmic
            // read path (GPU::OnCPURead) and reads of the page are let through again. A page that
            // keeps faulting (256 read faults within 50 ms) goes to InvalidateNCE below instead,
            // so the thread always progresses.
            if (pending_fault_read && m_guest_ctx.pc != pending_fault_address) {
                constexpr u32 MaxReadGuardRepeats = 256;
                constexpr s64 ReadGuardWindowNs = 50'000'000;
                const s64 now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now().time_since_epoch())
                                       .count();
                if (pending_fault_page != m_windows_read_guard_page ||
                    now_ns - m_windows_read_guard_since_ns > ReadGuardWindowNs) {
                    m_windows_read_guard_page = pending_fault_page;
                    m_windows_read_guard_since_ns = now_ns;
                    m_windows_read_guard_repeats = 0;
                }
                if (++m_windows_read_guard_repeats <= MaxReadGuardRepeats) {
                    if (process->GetMemory().HandleNceReadFault(
                            Common::ProcessAddress{pending_fault_address}, true)) {
#if NXN1X_DIAGNOSTICS
                        const u64 fault_us = static_cast<u64>(
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count() -
                            now_ns / 1000);
                        Common::NceReadGuard::NoteFault(fault_us);
                        // b73: read faults on the watched page (common/nce_read_guard.h).
                        if (pending_fault_page == Common::NceReadGuard::watch_vaddr.load(
                                                      std::memory_order_relaxed) &&
                            Common::NceReadGuard::TakeWatchLine()) {
                            u32 watch_insn{};
                            if (const auto* const code = process->GetMemory().GetPointerSilent(
                                    Common::ProcessAddress{m_guest_ctx.pc})) {
                                std::memcpy(&watch_insn, code, sizeof(watch_insn));
                            }
                            LOG_INFO(Render_Vulkan,
                                     "NCE_READ_GUARD watch_fault page={:#x} off={:#x} pc={:#x} "
                                     "insn={:08x} lr={:#x} us={} repeats={} tid={}",
                                     pending_fault_page, pending_fault_address - pending_fault_page,
                                     m_guest_ctx.pc, watch_insn, m_guest_ctx.cpu_registers[30],
                                     fault_us, m_windows_read_guard_repeats,
                                     thread->GetThreadId());
                        }
#endif
                        continue;
                    }
                } else if (m_windows_read_guard_repeats == MaxReadGuardRepeats + 1) {
#if NXN1X_DIAGNOSTICS
                    {
                        u32 fallback_insn{};
                        if (const auto* const code = process->GetMemory().GetPointerSilent(
                                Common::ProcessAddress{m_guest_ctx.pc})) {
                            std::memcpy(&fallback_insn, code, sizeof(fallback_insn));
                        }
                        LOG_INFO(Render_Vulkan,
                                 "NCE_READ_GUARD fallback_at page={:#x} off={:#x} pc={:#x} "
                                 "insn={:08x} lr={:#x} tid={}",
                                 pending_fault_page, pending_fault_address - pending_fault_page,
                                 m_guest_ctx.pc, fallback_insn, m_guest_ctx.cpu_registers[30],
                                 thread->GetThreadId());
                    }
#endif
                    static std::atomic<u32> fallback_logged{};
                    if (fallback_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                        LOG_WARNING(Render_Vulkan,
                                    "NCE read guard fallback: page {:#x} faulted {} times on reads "
                                    "within 50 ms; invalidating it",
                                    pending_fault_page, MaxReadGuardRepeats);
                    }
#if NXN1X_DIAGNOSTICS
                    Common::NceReadGuard::fallbacks.fetch_add(1, std::memory_order_relaxed);
#endif
                }
            }

#if NXN1X_DIAGNOSTICS
            // EDEN_NCE_WRITE_STEP=1 (nce_write_step.h): run a fault on a GPU-cached page through
            // the fallback runner. Its stores take the Dynarmic memory path (HandleRasterizerWrite
            // with the exact range) instead of invalidating the whole page here.
            if (NCE::WriteStep::GetMode() == NCE::WriteStep::Mode::Step) {
                const u64 step_fault = NCE::WriteStep::NoteFault();
                const auto& page_table = process->GetPageTable().GetImpl();
                const auto is_cached_page = [&page_table](u64 address) {
                    const u64 index = address >> Memory::YUZU_PAGEBITS;
                    return index < page_table.entries.size() &&
                           page_table.entries[index].ptr.Type() ==
                               Common::PageType::RasterizerCachedMemory;
                };
                const u64 faulting_pc = m_guest_ctx.pc;
                u32 instruction{};
                auto kind = NCE::WriteStep::Kind::NotCached;
                if (faulting_pc == pending_fault_address) {
                    kind = NCE::WriteStep::Kind::Fetch;
                } else if (is_cached_page(pending_fault_address)) {
                    if (const auto* const code = process->GetMemory().GetPointerSilent(
                            Common::ProcessAddress{faulting_pc})) {
                        std::memcpy(&instruction, code, sizeof(instruction));
                        kind = NCE::WriteStep::Classify(instruction);
                    } else {
                        kind = NCE::WriteStep::Kind::NoCode;
                    }
                }
                if (kind == NCE::WriteStep::Kind::Step) {
                    const auto step_start = std::chrono::steady_clock::now();
                    NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::Fallback);)
                    const auto step =
                        m_windows_x18_runner->ExecuteDirect(thread, m_guest_ctx, instruction);
                    NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::HostReturn);)
                    const auto step_ns = static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - step_start)
                            .count());
                    NCE::WriteStep::NoteStep(step_fault, step.completed, True(step.halt_reason),
                                             is_cached_page(pending_fault_address), step_ns,
                                             faulting_pc, instruction, pending_fault_address);
                    if (step.completed) {
                        // The instruction retired (GuestContext::pc advanced). Honour any halt it
                        // raised instead of re-running it.
                        if (True(step.halt_reason)) {
                            hr = step.halt_reason;
                            break;
                        }
                        continue;
                    }
                    // Not retired: InvalidateNCE below lets it run natively, as without the switch.
                } else {
                    NCE::WriteStep::NoteFallback(step_fault, kind);
                }
            }
#endif

            if (Common::CGT::Enabled()) [[unlikely]] {
                u32 cgt_insn{};
                if (const auto* const code = process->GetMemory().GetPointerSilent(
                        Common::ProcessAddress{m_guest_ctx.pc})) {
                    std::memcpy(&cgt_insn, code, sizeof(cgt_insn));
                }
                Common::CGT::SetNceFault(pending_fault_address, m_guest_ctx.pc, cgt_insn);
            }
#if NXN1X_DIAGNOSTICS
            // TILE_TRACE (b64): InvalidateNCE reaches TextureCache::WriteMemory synchronously on
            // this thread; tell it which store faulted (it only sees the whole page).
            u32 note_insn{};
            if (const auto* const code = process->GetMemory().GetPointerSilent(
                    Common::ProcessAddress{m_guest_ctx.pc})) {
                std::memcpy(&note_insn, code, sizeof(note_insn));
            }
            Common::NceWriteNote::Note write_note{};
            write_note.fault = pending_fault_address;
            write_note.pc = m_guest_ctx.pc;
            write_note.insn = note_insn;
            write_note.valid = true;
            // b71: the guest state of the store (registers, the bytes they point at, frames).
            FillWriteNote(write_note, process->GetMemory(), m_guest_ctx, thread->GetThreadId());
            Common::NceWriteNote::current = write_note;
#endif
            if (process->GetMemory().InvalidateNCE(Common::ProcessAddress{pending_fault_page},
                                                   Memory::YUZU_PAGESIZE)) {
#if NXN1X_DIAGNOSTICS
                Common::NceWriteNote::current.valid = false;
#endif
                continue;
            }
#if NXN1X_DIAGNOSTICS
            Common::NceWriteNote::current.valid = false;
#endif

            if (m_guest_ctx.pc != pending_fault_address) {
                // Unresolvable data abort (typically a guest NULL-based field access). Match the
                // Dynarmic backend instead of skipping: execute this one instruction through the
                // fallback runner, whose memory callbacks read unmapped memory as zero and drop
                // unmapped writes. Skipping left the destination register stale, which sent
                // titles/mods that rely on the JIT behaviour into endless loops.
                const u64 faulting_pc = m_guest_ctx.pc;
                if (const auto* const code = process->GetMemory().GetPointerSilent(
                        Common::ProcessAddress{faulting_pc})) {
                    u32 instruction{};
                    std::memcpy(&instruction, code, sizeof(instruction));
                    // XC2 3.0.1 startup: the arena either rejects the 4 GiB + 128 KiB
                    // reservation or fails to commit its first 128 KiB. The guest
                    // clears the resulting null pointer in both cases.
                    if (pending_fault_address == 0 && instruction == 0x39000001 &&
                        m_guest_ctx.cpu_registers[0] == 0 && m_guest_ctx.cpu_registers[2] == 0x20000 &&
                        process->GetProgramId() == 0x0400E95004038001ULL) {
                        const auto& regs = m_guest_ctx.cpu_registers;
                        auto& mem = process->GetMemory();
                        const auto read_word = [&mem](u64 addr, u64& value) {
                            if (!addr || (addr & 7)) return false;
                            const auto* ptr = mem.GetPointerSilent(Common::ProcessAddress{addr});
                            if (!ptr) return false;
                            std::memcpy(&value, ptr, sizeof(value));
                            return true;
                        };
                        u64 base{}, cursor{}, limit{}, first{}, committed{}, reservation{}, output{};
                        if (regs[19] && regs[20] && regs[19] <= UINT64_MAX - 88 &&
                            regs[20] <= UINT64_MAX - 24 &&
                            read_word(regs[20], base) && read_word(regs[20]+8, cursor) &&
                            read_word(regs[20]+24, limit) && read_word(regs[19]+40, first) &&
                            read_word(regs[19]+48, committed) && read_word(regs[19]+56, reservation) &&
                            read_word(regs[19]+88, output)) {
                            LOG_ERROR(Core_ARM, "XC2 startup arena tid={} lr={:#x} base={:#x} cursor={:#x} "
                                      "limit={:#x} first={:#x} committed={:#x} reserved={:#x} clear_ptr={:#x}",
                                      thread->GetThreadId(), regs[30], base, cursor, limit,
                                      first, committed, reservation, output);
                        }
                    }
                    // The allocating virtual call preserves its receiver in x22. Resolve its
                    // slot before the fallback mutates the faulting thread's registers.
                    if (pending_fault_address == 0x18 && instruction == 0xf9000c16 &&
                        m_guest_ctx.cpu_registers[0] == 0 &&
                        m_guest_ctx.cpu_registers[30] + 12 == faulting_pc &&
                        process->GetProgramId() == Settings::HyruleWarriorsProgramId) {
                        auto& memory = process->GetMemory();
                        const auto read_aligned = [&memory](u64 addr, auto& value) {
                            if (addr == 0 || addr % sizeof(value) != 0) {
                                return false;
                            }
                            const auto* p = memory.GetPointerSilent(Common::ProcessAddress{addr});
                            if (!p) {
                                return false;
                            }
                            std::memcpy(&value, p, sizeof(value));
                            return true;
                        };
                        const u64 receiver = m_guest_ctx.cpu_registers[22];
                        u32 call{}, slot_load{};
                        u64 vtable{}, target{};
                        if (read_aligned(faulting_pc - 16, call) && call == 0xd63f0100 &&
                            read_aligned(faulting_pc - 20, slot_load) && slot_load == 0xf9401908) {
                            const bool resolved = read_aligned(receiver, vtable) &&
                                vtable != 0 && vtable <= UINT64_MAX - 48 &&
                                read_aligned(vtable + 48, target);
                            LOG_ERROR(Core_ARM, "HWAoI null allocation tid={} call={:#x} "
                                      "receiver={:#x} vtable={:#x} slot48={:#x} resolved={} "
                                      "result=0 owner={:#x} fp={:#x}",
                                      thread->GetThreadId(), faulting_pc - 16, receiver, vtable,
                                      target, resolved, m_guest_ctx.cpu_registers[19],
                                      m_guest_ctx.cpu_registers[29]);
                        }
                    }
                    NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::Fallback);)
                    const auto step =
                        m_windows_x18_runner->ExecuteDirect(thread, m_guest_ctx, instruction);
                    NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::HostReturn);)
                    if (True(step.halt_reason & HaltReason::PrefetchAbort)) {
                        m_guest_ctx.pc = faulting_pc;
                        abort_site = "fault-fallback-exception";
                        hr = step.halt_reason;
                        break;
                    }
                    if (step.completed) {
                        NCE::Stats::Add(NCE::Stats::FaultEmulated);
                        static std::atomic<u32> emulated_logged{};
                        if (emulated_logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                            const auto& pt = process->GetPageTable();
                            LOG_WARNING(Core_ARM,
                                        "NCE emulated faulting guest instruction pc={:#x} "
                                        "insn={:08x} fault={:#x} (JIT semantics) lr={:#x} "
                                        "x0={:#x} x1={:#x} x2={:#x} sp={:#x} code_base={:#x} "
                                        "alias_base={:#x}",
                                        faulting_pc, instruction, pending_fault_address,
                                        m_guest_ctx.cpu_registers[30],
                                        m_guest_ctx.cpu_registers[0],
                                        m_guest_ctx.cpu_registers[1],
                                        m_guest_ctx.cpu_registers[2], m_guest_ctx.sp,
                                        GetInteger(pt.GetCodeRegionStart()),
                                        GetInteger(pt.GetAliasRegionStart()));
                        }
                        if (NCE::Nx2Probe::g_state.enabled) [[unlikely]] {
                            NCE::Nx2Probe::DumpRecent(thread->GetThreadId(), faulting_pc);
                        }
                        // The instruction retired (GuestContext::pc advanced). Honour any halt it
                        // raised instead of re-running or skipping it.
                        if (True(step.halt_reason)) {
                            hr = step.halt_reason;
                            break;
                        }
                        continue;
                    }
                }

                // Last resort: skip the instruction (inherited Linux NCE policy). Count and log it,
                // since a skipped store silently corrupts data.
                NCE::Stats::Add(NCE::Stats::FaultSkipped);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                    LOG_WARNING(Core_ARM, "NCE skipped faulting guest instruction pc={:#x} fault={:#x}",
                                m_guest_ctx.pc, pending_fault_address);
                }
                m_guest_ctx.pc += sizeof(u32);
                continue;
            }

            abort_site = "execute-fault";
            hr = HaltReason::PrefetchAbort;
            break;
        }

        // P2E: the x18 fallback backend is callback-only. While a private native stack lease is
        // active, its memory callbacks read/write that exact MEM_PRIVATE stack view directly and
        // use Core::Memory for every address outside the stack. Therefore ordinary x18 fallback no
        // longer needs a whole-stack SyncToBacking/SyncFromBacking round trip.
        const bool fallback_uses_private_stack_view =
            private_stack_lease.has_value() &&
            static_cast<u64>(hr) == NCE::WindowsX18FallbackTrap::ReturnMarker;

        const bool needs_private_stack_sync =
            private_stack_lease.has_value() && !fallback_uses_private_stack_view &&
            P2CNeedsPrivateStackSync(static_cast<u64>(hr), process, m_guest_ctx,
                                     process->GetPostHandlers());

        if (needs_private_stack_sync && !private_stack_lease->SyncToBacking()) {
            LOG_ERROR(Core_ARM, "V16 failed to synchronize private NCE stack to backing");
            hr = HaltReason::PrefetchAbort;
            break;
        }

        const u64 private_stack_base =
            fallback_uses_private_stack_view ? m_guest_ctx.windows_guest_stack_limit : 0;
        const u64 private_stack_size =
            fallback_uses_private_stack_view
                ? m_guest_ctx.windows_guest_stack_base - m_guest_ctx.windows_guest_stack_limit
                : 0;

        const u64 stats_fallback_t0 = NCE::Stats::Now();
        if (static_cast<u64>(hr) == NCE::WindowsX18FallbackTrap::ReturnMarker) {
            namespace Pool = NCE::HwaPool;
            const u64 base = Pool::probe_base.load(std::memory_order_relaxed);
            if (base != 0) {
                const auto& x = m_guest_ctx.cpu_registers;
                const u64 site = m_guest_ctx.pc - base;
                if (site == Pool::Recovered) {
                    LOG_INFO(Core_ARM, "HWAoI pool retry recovered tid={} attempts={} result={:#x}",
                             thread->GetThreadId(), Pool::RetryLimit - x[19], x[0]);
                } else if (site == Pool::Failed) {
                    auto& memory = process->GetMemory();
                    const Common::ProcessAddress receiver{x[22]};
                    std::array<u32, 500> bitmap{};
                    // A racing allocation/free may change the count while copying the bitmap.
                    // Record both endpoints; this is diagnostic, never used to change pool state.
                    if (memory.IsValidVirtualAddressRange(receiver, 2076)) {
                        const u32 before = memory.Read32(receiver + 44);
                        if (memory.ReadBlock(receiver + 48, bitmap.data(), sizeof(bitmap))) {
                            unsigned occupied = 0;
                            for (u32 bits : bitmap) occupied += std::popcount(bits);
                            const u32 after = memory.Read32(receiver + 44);
                            LOG_ERROR(Core_ARM, "HWAoI slot48 occupancy tid={} before={} bitmap={} "
                                      "after={} release={:#x} retries={}", thread->GetThreadId(),
                                      before, occupied, after, memory.Read32(receiver + 2072),
                                      Pool::RetryLimit - x[19]);
                        }
                    }
                    LOG_ERROR(Core_ARM, "HWAoI pool terminal tid={} reason={} retries={} "
                              "receiver={:#x} x2={:#x} w3={:#x} w10={:#x} x11={:#x}",
                              thread->GetThreadId(), x[19] ? "guard" : "retry-limit",
                              Pool::RetryLimit - x[19], x[22], x[2], static_cast<u32>(x[3]),
                              static_cast<u32>(x[10]), x[11]);
                } else {
                    // At these exits the tested values and the last CMP flags are still live.
                    u32 edge = 0;
                    if (site == 0x1adcecc) {
                        edge = (x[9] & (1U << 30)) ? 0x1adce98 : 0x1adceac;
                    } else if (x[30] == base + 0x1adcec0 && site == 0x1add17c) {
                        edge = 0x1add17c;
                    } else if (x[30] == base + 0x1adcec0 && site == 0x1add054) {
                        const u32 nzcv = m_guest_ctx.nzcv;
                        if (x[8] == x[0] + 8 && x[9] == 0) edge = 0x1add050;
                        else if (x[11] == x[0] + 12 && (nzcv & (1U << 30)))
                            edge = (x[8] & (1U << 31)) ? 0x1add1a4 : 0x1add188;
                        else if (x[11] == x[0] + 12 &&
                                 ((nzcv >> 31) ^ ((nzcv >> 28) & 1))) edge = 0x1add084;
                        else if (!(nzcv & (1U << 29))) edge = 0x1add0b4;
                    }
                    if (x[19] == x[22] + 36 &&
                        (edge || (site == 0x1add054 && x[30] == base + 0x1adcec0)) &&
                        Pool::reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                        LOG_WARNING(Core_ARM, "HWAoI pool failure tid={} exit=main+{:#x} "
                                    "edge=main+{:#x} x8={:#x} x9={:#x} x10={:#x} x11={:#x} "
                                    "nzcv={:#x} receiver={:#x}", thread->GetThreadId(), site,
                                    edge, x[8], x[9], x[10], x[11], m_guest_ctx.nzcv, x[22]);
                    }
                }
            }
        }
        NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::Fallback);)
        if (NCE::Nx2Probe::g_state.enabled &&
            static_cast<u64>(hr) == NCE::WindowsX18FallbackTrap::ReturnMarker) [[unlikely]] {
            NCE::Nx2Probe::OnTrap(m_guest_ctx, thread->GetThreadId()); // NXN1X_NX2_PROBE
        }
        const auto fallback = m_windows_x18_runner->Dispatch(
            static_cast<u64>(hr), thread, m_guest_ctx, process->GetPostHandlers(),
            private_stack_base,
            private_stack_size);
        NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::HostReturn);)
        if (static_cast<u64>(hr) == NCE::WindowsX18FallbackTrap::ReturnMarker) {
            NCE::Stats::Add(NCE::Stats::FallbackNanos, NCE::Stats::Now() - stats_fallback_t0);
        }

        if (fallback.handled && needs_private_stack_sync &&
            !private_stack_lease->SyncFromBacking()) {
            LOG_ERROR(Core_ARM, "V16 failed to synchronize fallback stack writes to private view");
            hr = HaltReason::PrefetchAbort;
            break;
        }

        if (!fallback.handled) {
            break;
        }

        if (!fallback.metadata_found || !fallback.step.completed) {
            abort_site = fallback.metadata_found ? "x18-fallback-step" : "x18-fallback-metadata";
            hr = fallback.step.halt_reason;
            if (!True(hr)) {
                hr = HaltReason::PrefetchAbort;
            }
            break;
        }

        if (True(fallback.step.halt_reason)) {
            hr = fallback.step.halt_reason;
            break;
        }

        // A normal one-instruction x18 fallback updated GuestContext::pc. Re-enter the native NCE
        // path using the same post-handler/arbitrary-PC selection contract as ordinary RunThread.
    }

    if (m_windows_x18_runner != nullptr) {
        m_windows_x18_runner->ClearDirectPrivateStackView();
    }

    // The private replacement belongs to the complete RunThread epoch. Internal NCE fault/retry
    // iterations reuse it; restore the section-backed mapping exactly once when the epoch exits.
    const bool stats_had_lease = private_stack_lease.has_value();
    const u64 stats_restore_t0 = NCE::Stats::Now();
    // Only [SP, top) of the guest stack is live at epoch exit; copy back just that part.
    if (private_stack_lease.has_value() && !private_stack_lease->Restore(m_guest_ctx.sp)) {
        LOG_ERROR(Core_ARM, "Failed to restore Windows NCE private stack lease");
        hr = HaltReason::PrefetchAbort;
    }
    if (stats_had_lease) {
        NCE::Stats::Add(NCE::Stats::LeaseRestores);
        NCE::Stats::Add(NCE::Stats::LeaseNanos, NCE::Stats::Now() - stats_restore_t0);
    }

    if (True(hr & HaltReason::PrefetchAbort)) {
        LogWindowsNceAbort(abort_site, process, m_guest_ctx, last_native_exit);
    }

    // Close the delivery epoch before anything the worker reads is torn down. Retiring stops new
    // deliveries; a worker already inside one sees it after its current Windows call, resumes us
    // and releases Busy. Wait on the host stack with no lock held, so that worker never needs
    // anything from this thread to finish.
    NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::Retire);)
#if NXN1X_DIAGNOSTICS
    // R2b-B3d: a request left on HostSide is still pending; the next RunThread call takes it.
    if (m_r2a_host_side_post_ns.load(std::memory_order_relaxed) != 0) {
        NCE::R2a::NoteHostSideExit(True(hr & HaltReason::SupervisorCall)  ? 0
                                   : True(hr & HaltReason::BreakLoop)     ? 1
                                   : True(hr & HaltReason::PrefetchAbort) ? 2
                                                                          : 3);
    }
#endif
    {
        // R2b-1: sleep until the worker releases Busy. seen is read before Retiring is set, and a
        // worker that releases Busy after that sees Retiring in its CAS and bumps retire_seq
        // afterwards, so the bump cannot be missed. Other cores' wakes only cause a recheck.
        auto& retire_seq = g_break_workers.RetireSeq(m_core_index);
        u64 seen = retire_seq.load(std::memory_order_acquire);
        u64 gate = m_windows_delivery_gate.fetch_or(DeliveryRetiring, std::memory_order_seq_cst);
        NXN1X_R2A(const bool r2a_waited = (gate & DeliveryBusy) != 0;
                    const u64 r2a_retire_t0 = r2a_waited ? NCE::R2a::NowNs() : 0;)
        while ((gate & DeliveryBusy) != 0) {
            retire_seq.wait(seen, std::memory_order_acquire);
            seen = retire_seq.load(std::memory_order_acquire);
            gate = m_windows_delivery_gate.load(std::memory_order_acquire);
        }
        NXN1X_R2A(if (r2a_waited) {
            NCE::R2a::G().retire_busy_wait.Add(NCE::R2a::NowNs() - r2a_retire_t0);
        })
        m_windows_delivery_gate.store(gate & ~DeliveryFlagMask, std::memory_order_release);
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    const u64 final_tpidr_el0 = thread_params->tpidr_el0;

    thread_params->is_running = false;
    thread_params->native_context = nullptr;
    m_running_thread = nullptr;
    NXN1X_R2A(NCE::R2a::StorePhase(m_r2a_phase, NCE::R2a::Phase::Idle);)
    m_guest_ctx.tpidr_el0 = final_tpidr_el0;

    return hr;
}

HaltReason ArmNce::StepThread(Kernel::KThread* thread) {
    (void)thread;
    return HaltReason::StepThread;
}

u32 ArmNce::GetSvcNumber() const {
    return m_guest_ctx.svc;
}

void ArmNce::GetSvcArguments(std::span<uint64_t, 8> args) const {
    for (size_t i = 0; i < args.size(); ++i) {
        args[i] = m_guest_ctx.cpu_registers[i];
    }
}

void ArmNce::SetSvcArguments(std::span<const uint64_t, 8> args) {
    for (size_t i = 0; i < args.size(); ++i) {
        m_guest_ctx.cpu_registers[i] = args[i];
    }
}

void ArmNce::SetTpidrroEl0(u64 value) {
    m_guest_ctx.tpidrro_el0 = value;
}

void ArmNce::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    for (size_t i = 0; i < 29; ++i) {
        ctx.r[i] = m_guest_ctx.cpu_registers[i];
    }
    ctx.fp = m_guest_ctx.cpu_registers[29];
    ctx.lr = m_guest_ctx.cpu_registers[30];
    ctx.sp = m_guest_ctx.sp;
    ctx.pc = m_guest_ctx.pc;
    ctx.pstate = m_guest_ctx.pstate;
    ctx.v = m_guest_ctx.vector_registers;
    ctx.fpcr = m_guest_ctx.fpcr;
    ctx.fpsr = m_guest_ctx.fpsr;
    ctx.tpidr = m_guest_ctx.tpidr_el0;
}

void ArmNce::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    for (size_t i = 0; i < 29; ++i) {
        m_guest_ctx.cpu_registers[i] = ctx.r[i];
    }
    m_guest_ctx.cpu_registers[29] = ctx.fp;
    m_guest_ctx.cpu_registers[30] = ctx.lr;
    m_guest_ctx.sp = ctx.sp;
    m_guest_ctx.pc = ctx.pc;
    m_guest_ctx.pstate = ctx.pstate;
    m_guest_ctx.nzcv = ctx.pstate & 0xF0000000U;
    m_guest_ctx.vector_registers = ctx.v;
    m_guest_ctx.fpcr = ctx.fpcr;
    m_guest_ctx.fpsr = ctx.fpsr;
    m_guest_ctx.tpidr_el0 = ctx.tpidr;
}

void ArmNce::SignalInterrupt(Kernel::KThread* thread) {
    (void)thread;
    // Called under PhysicalCore's guard. Record the request and hand delivery to this core's
    // worker; never call Windows, wait for a lock, or wait for the target here.
    m_guest_ctx.esr_el1.fetch_or(static_cast<u64>(HaltReason::BreakLoop),
                                 std::memory_order_acq_rel);

    if (m_windows_break == nullptr || !m_windows_break->IsBound()) {
        return;
    }

    u64 gate = m_windows_delivery_gate.load(std::memory_order_seq_cst);
    for (;;) {
        // No native epoch, or it is closing: RunThread reads esr_el1 before it next enters guest
        // code, so the BreakLoop recorded above is enough.
        if ((gate & DeliveryActive) == 0 || (gate & DeliveryRetiring) != 0) {
            return;
        }
        const bool busy = (gate & DeliveryBusy) != 0;
        const u64 next = busy ? (gate | DeliveryPending) : (gate | DeliveryBusy);
        if (m_windows_delivery_gate.compare_exchange_weak(gate, next, std::memory_order_seq_cst,
                                                          std::memory_order_seq_cst)) {
#if NXN1X_DIAGNOSTICS
            if (!busy) {
                const u64 r2a_now = NCE::R2a::NowNs();
                NCE::R2a::NoteFirstEvent(r2a_now);
                m_r2a_post_ns.store(r2a_now, std::memory_order_relaxed);
            } else {
                m_r2a_merged.fetch_add(1, std::memory_order_relaxed);
                u64 r2a_none = 0;
                m_r2a_post_ns.compare_exchange_strong(r2a_none, NCE::R2a::NowNs(),
                                                      std::memory_order_relaxed);
            }
#endif
            if (!busy) {
                // Busy was set while this core's guard proves the ArmNce is running, and RunThread
                // cannot retire the epoch until the worker clears it. The pointer stays valid.
                g_break_workers.Post(m_core_index, this);
            }
            return;
        }
    }
}

void ArmNce::DeliverPendingBreak() {
    using Delivery = NCE::WindowsCrossThreadBreak::DeliveryResult;

    for (;;) {
#if NXN1X_DIAGNOSTICS
        NCE::R2a::EpisodeBuilder r2a{};
        r2a.core = static_cast<u32>(m_core_index);
        r2a.post_ns = m_r2a_post_ns.exchange(0, std::memory_order_relaxed);
        auto r2a_end = NCE::R2a::End::Retired;
#endif
        bool settled = false;
        u32 set_failures = 0;
        while (!settled) {
            const u64 gate_now = m_windows_delivery_gate.load(std::memory_order_acquire);
            if ((gate_now & DeliveryRetiring) != 0) {
                // RunThread is leaving and will read esr_el1 on its next entry.
                break;
            }
            if ((gate_now & DeliveryHostSide) != 0) {
                // R2b-B3: RunThread is in host code and reads esr_el1 before it can enter guest
                // code again; its next gate RMW follows this request's. Nothing to capture.
#if NXN1X_DIAGNOSTICS
                r2a_end = NCE::R2a::End::HostSide;
                if (r2a.post_ns != 0) {
                    u64 r2a_none = 0;
                    m_r2a_host_side_post_ns.compare_exchange_strong(r2a_none, r2a.post_ns,
                                                                    std::memory_order_relaxed);
                }
#endif
                break;
            }

            // While the target owns its parameter lock it is in host code that reads esr_el1 before
            // its next guest entry, or is about to drop the lock and enter. Either way a capture
            // now can only find a host window; wait for the lock to drop instead of suspending.
            // m_running_thread is stable while this epoch is Active and Busy.
            if (m_running_thread != nullptr &&
                m_running_thread->GetNativeExecutionParameters().lock.load(
                    std::memory_order_acquire) != SpinLockUnlocked) {
#if NXN1X_DIAGNOSTICS
                ++r2a.p_busy_spins;
                if (!r2a.in_p_busy_streak) {
                    r2a.in_p_busy_streak = true;
                    r2a.Refused(NCE::R2a::Reject::PBusyPrecheck,
                                m_r2a_phase.load(std::memory_order_relaxed), 0);
                }
#endif
                std::this_thread::yield();
                continue;
            }

            BreakTransformState state{.nce = this};
            bool set_ok = false;
            BreakApiError api_error{};
#if NXN1X_DIAGNOSTICS
            r2a.in_p_busy_streak = false;
            ++r2a.attempts;
            const u64 r2a_t0 = NCE::R2a::NowNs();
#endif
            const Delivery delivery =
                m_windows_break->SuspendTransformResume(&WindowsBreakTransform, &state, &set_ok,
                                                        &WindowsBreakCommit, &api_error);
            NXN1X_R2A(NCE::R2a::G().capture_cost.Add(NCE::R2a::NowNs() - r2a_t0);)

            switch (delivery) {
            case Delivery::Delivered:
                // The target returns to its host continuation holding the parameter lock and
                // PhysicalCore::ExitContext releases it.
                NXN1X_R2A(r2a_end = NCE::R2a::End::Delivered;)
                settled = true;
                break;
            case Delivery::SetFailed:
                // The target still runs its original CONTEXT. Give back what the transform took and
                // try again on a fresh capture.
                NXN1X_R2A(r2a.Refused(NCE::R2a::Reject::SetFailed, state.r2a_phase, state.r2a_pc);)
                if (!state.recovery_aborted) {
                    m_guest_ctx.esr_el1.fetch_or(state.claimed_reason, std::memory_order_acq_rel);
                    UnlockThreadParameters(state.params);
                }
                // R2b-S12: a SetThreadContext that keeps failing must not be retried forever. After
                // three in a row the request stays in esr_el1 for RunThread's next read, as for a
                // failed SuspendThread.
                ++set_failures;
                LogBreakApiFailure(api_error, m_core_index, set_failures >= 3);
                if (set_failures >= 3) {
                    NXN1X_R2A(r2a_end = NCE::R2a::End::SetFailedSticky;)
                    settled = true;
                }
                break;
            case Delivery::SuspendFailed:
            case Delivery::GetFailed:
                // Nothing was changed and the request stays in esr_el1.
                LogBreakApiFailure(api_error, m_core_index, true);
                NXN1X_R2A(r2a_end = NCE::R2a::End::SuspendOrGetFailed;)
                settled = true;
                break;
            case Delivery::NotTransformed:
                // The target was resumed unchanged: it was in a host window, a generated patch
                // window, or a guest PC whose parameter lock it owns. No core or parameter lock is
                // held here, so the target is free to finish that window; capture again.
                // TODO(R2): patch windows with a covered exit will notify instead of being retried.
                set_failures = 0;
#if NXN1X_DIAGNOSTICS
                if (state.r2a_reject != 0xFF) {
                    r2a.Refused(static_cast<NCE::R2a::Reject>(state.r2a_reject), state.r2a_phase,
                                state.r2a_pc);
                }
#endif
                break;
            case Delivery::ResumeFailed:
                FailFastOnResumeFailure(api_error, m_core_index, state.transformed, set_ok);
            }
            if (!settled) {
                std::this_thread::yield();
            }
        }

        NXN1X_R2A(r2a.Finish(r2a_end, m_r2a_merged.exchange(0, std::memory_order_relaxed));)

        // Release Busy unless a request arrived meanwhile. Nothing in this ArmNce may be touched
        // after the release succeeds: RunThread may then retire the epoch and the owner may free it.
        // The worker's retire_seq outlives the ArmNce and is the one thing touched afterwards.
        auto& retire_seq = g_break_workers.RetireSeq(m_core_index);
        u64 gate = m_windows_delivery_gate.load(std::memory_order_seq_cst);
        for (;;) {
            if ((gate & DeliveryPending) != 0 && (gate & DeliveryRetiring) == 0) {
                if (m_windows_delivery_gate.compare_exchange_weak(
                        gate, gate & ~DeliveryPending, std::memory_order_seq_cst,
                        std::memory_order_seq_cst)) {
                    break; // Deliver the newer request.
                }
                continue;
            }
            if (m_windows_delivery_gate.compare_exchange_weak(
                    gate, gate & ~(DeliveryBusy | DeliveryPending), std::memory_order_seq_cst,
                    std::memory_order_seq_cst)) {
                if ((gate & DeliveryRetiring) != 0) {
                    // R2b-1: RunThread is waiting for this release.
                    retire_seq.fetch_add(1, std::memory_order_release);
                    retire_seq.notify_all();
                }
                return;
            }
        }
    }
}

void ArmNce::ClearInstructionCache() {
    FlushInstructionCache(GetCurrentProcess(), nullptr, 0);
}

void ArmNce::InvalidateCacheRange(u64 addr, std::size_t size) {
    FlushInstructionCache(GetCurrentProcess(), reinterpret_cast<const void*>(addr), size);
}

} // namespace Core
