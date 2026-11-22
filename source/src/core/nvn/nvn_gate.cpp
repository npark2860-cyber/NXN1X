// SPDX-License-Identifier: GPL-2.0-or-later
// NVN path, the gate (NXN1X docs NVN_HOST_BRIDGE.md 4.1; design v0.3 chapter 4.2 / D-22 for the hook, chapter 5.3 for the H gate).
//
// The game reaches the host NVN layer (src/nvn_host) in two steps:
//  1. Hook. The first pass of the NCE patcher turns the first instruction of nvnBootstrapLoader's definition (and of the host's other
//     take-overs by symbol) into a branch to the patch section, which jumps to this file's gate block (Patcher::PatchText, EntryHook).
//     While the gate is off (before NXN1X's device is handed over, or after Refuse) the block sends the call to the definition's
//     original start in the patch section instead: the game's own code runs and the boot is on the Emulation path.
//  2. Gate. nvnBootstrapLoader returns, for every NVN function, an entry of the gate block (nvnh_set_gates), as the host's own loader
//     returns its thunks. An entry puts &nvn_slot[i] in x16 and enters the H gate: from the guest stack to the guest thread's host
//     (fiber) stack below host_ctx.host_sp, with the host's TEB stack bounds and FPCR, calls nvn_slot[i].fn as the host's thunks do
//     (x16 still the slot: the unimplemented and big-stack entries read it), and goes back. x0-x8 and v0-v7 go through untouched both ways.
//
// Most functions run in the H gate. A cross-thread break that arrives during an H call waits until the thread is back in guest code (the
// break transform refuses a PC outside guest memory or a host SP, and the break worker keeps retrying), so the functions that wait or
// touch the kernel take the S gate (step 3b, design v0.3 5.4): the entry leaves guest code the way an SVC does, PhysicalCore::RunThread
// runs the function on the core's host thread (GateSlowCall) and the thread resumes at the caller's x30. The S set is SlowNames below
// (function table v3 S functions, and nvnSyncWait, an HS function whose H half is not split off yet). NXN1X_NVN_GATE_SLOW=0 runs
// them all as H.
// Switches: NXN1X_NVN_GATE_TRACE=1 logs the first call of each function through the gate (with x0), and every name the game looks up
// through nvnBootstrapLoader. DM_NVN_COUNT=1 (the host's) makes the entries count calls into nvn_slot[i].cnt, as the host's counting thunks do.

#include <array>
#include <atomic>
#include <cstdlib>
#include <chrono>   // s39
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>   // s39
#include <string>
#include <string_view>   // z1
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <windows.h>

#include <oaknut/oaknut.hpp>

#include "common/elf.h"
#include "common/logging.h"
#include "common/settings.h"   // s69
#include "core/arm/arm_interface.h"
#include "core/arm/nce/arm_nce_asm_definitions.h"
#include "core/arm/nce/current_nce_context.h"
#include "core/arm/nce/guest_context.h"
#include "core/arm/nce/windows_patch_code_metadata.h"
#include "core/hle/kernel/k_thread.h"
#include "core/core.h"                        // s52
#include "core/hle/kernel/k_hardware_timer.h" // s52
#include "core/hle/kernel/k_process.h"        // s52
#include "core/hle/kernel/k_scheduler.h"      // s52
#include "core/hle/kernel/kernel.h"           // s52
#include "core/hle/kernel/physical_core.h"    // s52
#include <algorithm>                          // s52
#include <limits>                             // s52
#include "core/nvn/nvn_gate.h"
#include "core/nvn/nvn_layer.h"

#include "nvnh_api.h"

namespace Core::NvnLayer {

namespace S52 {   // s52: defined below, before GateSlowCall
int KSleep(uint64_t ns);
int CoreState(int* waiter);
} // namespace S52
namespace S55 {   // s55: defined below, after S52
int Defer(void* cb, void* cmd, int event, size_t size, void* data);
u32 SetUp();   // at the block build: the TEB offset of this thread's owed call (0 = s55 off)
} // namespace S55

namespace {

using namespace oaknut::util;
using Params = Kernel::KThread::NativeExecutionParameters;

constexpr size_t MaxHooks = 64;
constexpr size_t MaxForce = 32;
constexpr const char* BootstrapName = "nvnBootstrapLoader";

// The S gate's functions: they wait (GPU, display pacing) or will touch the kernel (pools, windows, present). nvnQueueSubmitCommands
// (S/HS) stays H: it is called many times a frame and the host's submit does not wait for the guest
constexpr std::array SlowNames{
    "nvnDeviceInitialize",       "nvnDeviceFinalize",          "nvnQueueInitialize",      "nvnQueueFinalize",
    "nvnQueueFinish",            "nvnQueuePresentTexture",     "nvnQueueAcquireTexture",  "nvnWindowInitialize",
    "nvnWindowFinalize",         "nvnWindowAcquireTexture",    "nvnMemoryPoolInitialize", "nvnTextureReadTexels",
    "nvnTextureReadTexelsStrided", "nvnTextureInvalidateTexels", "nvnSyncWait",
};

// What an entry puts in x16, laid out as the host's nvn_slot_t: the count at +0, the function at +8
struct alignas(16) Record {
    std::atomic<u64> count;
    void* fn;
};
static_assert(sizeof(Record) == 16 && offsetof(Record, fn) == 8);

// A hooked definition: its record (&nvn_slot[i] or a take-over's), and the guest address of its original start (0: none)
struct alignas(16) HookCell {
    const void* record;
    u64 original_start;
};
static_assert(sizeof(HookCell) == 16 && offsetof(HookCell, original_start) == 8);

struct HookInfo {
    std::string name;
    std::string module;
    u32 text_offset;
    size_t cell;
};

// ---- the block (built once per process: the game keeps its addresses, it is never freed or moved) ----
std::once_flag g_build_once;
bool g_built{};
std::string g_build_failure;
u8* g_block{};
size_t g_block_size{};
std::vector<void*> g_entries;   // nvnh_set_gates: the entry for nvn_names[i]
u64 g_hook_entries{};           // hook entry j at g_hook_entries + 16 * j
bool g_trace{};
bool g_counting{};
std::vector<u8> g_slow;          // per function: 1 = the S gate
std::string g_slow_list;

const nvnh_force_t* g_force{};
size_t g_nforce{};
std::array<Record, MaxForce> g_force_records{};
const u8* g_slot0{};            // &nvn_slot[0]
unsigned g_nfunc{};

std::unique_ptr<std::atomic<u8>[]> g_first_seen; // NXN1X_NVN_GATE_TRACE: one flag per function, then per take-over
std::atomic<u32> g_traced_lookups{};

// ---- this boot ----
std::atomic<u8> g_live{0}; // read by the hook entries (LDRB): 0 = the original start
std::array<HookCell, MaxHooks> g_cells{};
std::mutex g_plan_mutex;
std::vector<HookInfo> g_hooks;
std::unordered_set<std::string> g_imported; // NVN names some module imports (undefined in its symbol table)
std::unordered_map<std::string, std::vector<u32>> g_kept_words; // module -> its hooked first instructions after RelocateAndCopy
bool g_bootstrap_hooked{};
bool g_hold{true};   // s16: the H gate holds the thread's parameter lock during the host call (NXN1X_NVN_GATE_HOLD=0 off)
std::atomic<u64> g_s39_steals{0};   // s39: parameter locks the NVN gate (S or H hold) found left taken and took over
bool g_s39_on{true};   // s39: DM_S39=0 at boot: the old unbounded spins
std::atomic<bool> g_bootstrap_called{false};
bool g_range_registered{};
u32 g_s55_slot{};   // s55: TEB offset of the owed guest call (S55::SetUp), read by the H gate's return; 0 = off

void* NoexceptBootstrap(const char* name) noexcept;
void TraceFirst(const void* record, u64 x0, u64 lr) noexcept;

// ---- the code ----
constexpr u32 TebStackBaseOffset = 0x08; // TEB NT_TIB StackBase, StackLimit (x18 + 8, + 0x10)
constexpr u32 FrameH = 64;               // guest SP, guest x30, guest TEB StackBase, StackLimit, guest FPCR (+ x16 in the trace gate)
constexpr u32 FrameTrace = 256;          // + x0-x8 at 48, q0-q7 at 128
constexpr u32 BlrX17 = 0xD63F0000U | (17U << 5);
constexpr u32 MovzX17One = 0xD2800000U | (1U << 5) | 17U;
constexpr u32 StaddX17X16 = 0xF8200000U | (17U << 16) | (16U << 5) | 31U; // LDADD x17, xzr, [x16]

void EmitGate(oaknut::VectorCodeGenerator& cg, bool counting, bool trace, oaknut::Label& trace_fn) {
    const u32 frame = trace ? FrameTrace : FrameH;
    constexpr u32 HostSp = static_cast<u32>(offsetof(GuestContext, host_ctx) + offsetof(HostContext, host_sp));
    constexpr u32 HostFpcr = static_cast<u32>(offsetof(GuestContext, host_ctx) + offsetof(HostContext, host_fpcr));
    constexpr u32 HostStackBase = static_cast<u32>(offsetof(GuestContext, windows_host_stack_base));
    constexpr u32 HostStackLimit = static_cast<u32>(offsetof(GuestContext, windows_host_stack_limit));
    static_assert(HostStackLimit == HostStackBase + 8);

    if (counting) {
        cg.dw(MovzX17One);
        cg.dw(StaddX17X16);
    }
    // x17 = this thread's GuestContext (RunThread mirrors the parameters into the TEB slot, as the patcher's generated code reads them)
    cg.LDR(X17, X18, Core::NCE::CurrentNceContext::TebSlotOffset());
    cg.LDR(X17, X17, offsetof(Params, native_context));
    // The frame, below the host SP RunThread left when it entered the guest: guest SP and x30, guest TEB stack bounds, guest FPCR
    cg.LDR(X9, X17, HostSp);
    cg.SUB(X9, X9, frame);
    cg.MOV(X10, SP);
    cg.STP(X10, X30, X9, 0);
    cg.LDP(X11, X12, X18, TebStackBaseOffset);
    cg.STP(X11, X12, X9, 16);
    cg.MRS(X13, oaknut::SystemReg::FPCR);
    cg.STR(X13, X9, 32);
    cg.LDR(X11, X17, HostStackBase);
    cg.LDR(X12, X17, HostStackLimit);
    cg.LDR(W14, X17, HostFpcr);
    // D2 rule (patcher.cpp): the whole host TEB interval, then SP at once, nothing in between
    cg.STP(X11, X12, X18, TebStackBaseOffset);
    cg.MOV(SP, X9);
    oaknut::Label same_fpcr_in;
    cg.CMP(X13, X14);
    cg.B(oaknut::Cond::EQ, same_fpcr_in);
    cg.MSR(oaknut::SystemReg::FPCR, X14);
    cg.l(same_fpcr_in);

    if (trace) {
        cg.STR(X16, SP, 40);
        for (int i = 0; i < 8; i += 2) {
            cg.STP(oaknut::XReg{i}, oaknut::XReg{i + 1}, SP, 48 + 8 * i);
        }
        cg.STR(X8, SP, 112);
        for (int i = 0; i < 8; i += 2) {
            cg.STP(oaknut::QReg{i}, oaknut::QReg{i + 1}, SP, 128 + 16 * i);
        }
        cg.MOV(X1, X0);
        cg.MOV(X0, X16);
        cg.LDR(X2, SP, 8);   // xcdedlss: the guest's x30 (the frame's second word)
        cg.LDR(X17, trace_fn);
        cg.dw(BlrX17);
        for (int i = 0; i < 8; i += 2) {
            cg.LDP(oaknut::QReg{i}, oaknut::QReg{i + 1}, SP, 128 + 16 * i);
        }
        cg.LDR(X8, SP, 112);
        for (int i = 0; i < 8; i += 2) {
            cg.LDP(oaknut::XReg{i}, oaknut::XReg{i + 1}, SP, 48 + 8 * i);
        }
        cg.LDR(X16, SP, 40);
    }

    // s16: this thread's parameter lock is held while the host function runs, so NXN1X's break worker (arm_nce_windows.cpp
    // DeliverPendingBreak) waits for it to drop instead of suspending the thread. Before, a break request (preemption, another thread
    // ready on this core) during a host call found a host stack, was refused and tried again at once: SuspendThread, GetThreadContext,
    // ResumeThread of this thread in a loop until the call returned. Run 17: about 1 submission in 6, its replay 10-50 times slower in
    // every op, with this thread's cycles up as well. The lock drops once the guest SP is back; the request is then taken at the guest
    // PC that follows (the gate block is patch code, refused and tried again). Nothing the host functions run takes this lock
    if (g_hold) {
        // s39: while this thread runs guest code its lock can only be held briefly by the break worker, so a lock still taken after ~8M
        // tries (tens of ms) was left taken: pid 23344 froze with this thread spinning here and the break worker yielding in
        // DeliverPendingBreak until it dropped. Taken over (counted, "s39:") and released after the host call as usual. x11-x13 are
        // free here (caller-saved, not arguments)
        oaknut::Label hold_retry, hold_busy, hold_done;
        cg.MOV(X11, XZR);
        cg.l(hold_retry);
        cg.CLREX();
        cg.LDR(X9, X18, Core::NCE::CurrentNceContext::TebSlotOffset());
        cg.ADD(X9, X9, offsetof(Params, lock));
        cg.LDAXR(W10, X9);
        cg.CBZ(W10, g_s39_on ? hold_busy : hold_retry);
        cg.STXR(W10, WZR, X9);
        cg.CBNZ(W10, hold_retry);
        cg.B(hold_done);
        cg.l(hold_busy);
        if (g_s39_on) {
            cg.ADD(X11, X11, 1);
            cg.TBZ(X11, 23, hold_retry);
            cg.CLREX();
            cg.MOV(X12, reinterpret_cast<u64>(&g_s39_steals));
            cg.LDR(X13, X12);
            cg.ADD(X13, X13, 1);
            cg.STR(X13, X12);
        } else {
            cg.B(hold_retry);
        }
        cg.l(hold_done);
    }

    // The host function, x16 = its record (nvn_slot[i]: nvn_unimpl_entry and nvn_big_entry read it)
    cg.LDR(X17, X16, 8);
    cg.dw(BlrX17);

    // Back: x0, x1 and v0-v3 hold the results; only x10-x14 are used
    oaknut::Label same_fpcr_out;
    cg.LDR(X13, SP, 32);
    cg.MRS(X14, oaknut::SystemReg::FPCR);
    cg.CMP(X13, X14);
    cg.B(oaknut::Cond::EQ, same_fpcr_out);
    cg.MSR(oaknut::SystemReg::FPCR, X13);
    cg.l(same_fpcr_out);
    cg.LDP(X11, X12, SP, 16);
    cg.LDP(X10, X30, SP, 0);
    cg.STP(X11, X12, X18, TebStackBaseOffset);
    cg.MOV(SP, X10);
    if (g_hold) {   // s16: the lock drops with the guest SP and TEB stack bounds back (x0, x1 and v0-v3 untouched)
        cg.LDR(X11, X18, Core::NCE::CurrentNceContext::TebSlotOffset());
        cg.ADD(X11, X11, offsetof(Params, lock));
        cg.MOVZ(W12, SpinLockUnlocked);
        cg.STLR(W12, X11);
    }
    // s55: the host function left a guest call owed (the game's command memory callback, S55::Defer): run it now, as guest code. Guest SP, TEB bounds,
    // FPCR are back and the lock is dropped; x0, x1, q0 (the results) and x30 are kept on the guest stack around it; x9-x15 are free at a return.
    // The slot is cleared before the call, so a callback that overflows again owes its own at that inner gate's return
    if (g_s55_slot != 0) {
        oaknut::Label s55_none;
        cg.LDR(X9, X18, g_s55_slot);
        cg.CBZ(X9, s55_none);
        cg.LDR(X10, X9, 0);
        cg.CBZ(X10, s55_none);
        cg.SUB(SP, SP, 48);
        cg.STP(X0, X1, SP, 0);
        cg.STR(X30, SP, 16);
        cg.STR(oaknut::QReg{0}, SP, 32);
        cg.LDP(X0, X1, X9, 8);
        cg.LDP(X2, X3, X9, 24);
        cg.MOV(X11, XZR);
        cg.STR(X11, X9, 0);
        cg.dw(0xD63F0000U | (10U << 5));   // BLR x10
        cg.LDR(oaknut::QReg{0}, SP, 32);
        cg.LDR(X30, SP, 16);
        cg.LDP(X0, X1, SP, 0);
        cg.ADD(SP, SP, 48);
        cg.l(s55_none);
    }
    cg.RET();
}

// The S gate. Entered from a function entry, x16 = &nvn_slot[i], at a call boundary (x9-x17 free, the guest stack untouched).
// As patcher.cpp WriteSvcTrampoline (Windows): take the parameters lock, save the guest context, pc = the caller's x30,
// svc = SlowGateSvcBase + i, take esr_el1 with SupervisorCall added as the host return value, and return to RunThread with the host's
// FPCR, TEB stack bounds, SP and callee-saved registers. RunThread re-enters at pc through the generic context restore, which unlocks.
void EmitSlowGate(oaknut::VectorCodeGenerator& cg, oaknut::Label& slot0) {
    constexpr u32 Vecs = static_cast<u32>(offsetof(GuestContext, vector_registers));
    constexpr u32 HostCtx = static_cast<u32>(offsetof(GuestContext, host_ctx));
    constexpr u32 HostStackBase = static_cast<u32>(offsetof(GuestContext, windows_host_stack_base));
    constexpr u32 HostStackLimit = static_cast<u32>(offsetof(GuestContext, windows_host_stack_limit));
    constexpr u32 HostRegs = static_cast<u32>(offsetof(HostContext, host_saved_regs));
    constexpr u32 HostVregs = static_cast<u32>(offsetof(HostContext, host_saved_vregs));
    constexpr u32 NzcvMask = 0xF0000000U;
    static_assert(offsetof(GuestContext, cpu_registers) == 0);
    static_assert(SpinLockLocked == 0);

    // The lock, re-reading the parameters each try as LockContext does
    // s39: a lock still taken after ~8M tries was left taken (see the H hold): taken over and counted (x11-x13 are free at a call boundary)
    oaknut::Label lock_retry, esr_retry, s39_busy, s39_done;
    cg.MOV(X11, XZR);
    cg.l(lock_retry);
    cg.CLREX();
    cg.LDR(X17, X18, Core::NCE::CurrentNceContext::TebSlotOffset());
    cg.ADD(X9, X17, offsetof(Params, lock));
    cg.LDAXR(W10, X9);
    cg.CBZ(W10, g_s39_on ? s39_busy : lock_retry);
    cg.STXR(W10, WZR, X9);
    cg.CBNZ(W10, lock_retry);
    cg.B(s39_done);
    cg.l(s39_busy);
    if (g_s39_on) {
        cg.ADD(X11, X11, 1);
        cg.TBZ(X11, 23, lock_retry);
        cg.CLREX();
        cg.MOV(X12, reinterpret_cast<u64>(&g_s39_steals));
        cg.LDR(X13, X12);
        cg.ADD(X13, X13, 1);
        cg.STR(X13, X12);
    } else {
        cg.B(lock_retry);
    }
    cg.l(s39_done);
    cg.LDR(X17, X17, offsetof(Params, native_context));

    // The guest context: x0-x16 (x17 is the context, dead at a call), x19-x30 (not x18: the TEB), SP, q0-q31, FPSR, FPCR, NZCV
    for (int i = 0; i <= 14; i += 2) {
        cg.STP(oaknut::XReg{i}, oaknut::XReg{i + 1}, X17, 8 * i);
    }
    cg.STR(X16, X17, 8 * 16);
    for (int i = 19; i <= 29; i += 2) {
        cg.STP(oaknut::XReg{i}, oaknut::XReg{i + 1}, X17, 8 * i);
    }
    for (int i = 0; i <= 30; i += 2) {
        cg.STP(oaknut::QReg{i}, oaknut::QReg{i + 1}, X17, Vecs + 16 * i);
    }
    cg.MOV(X9, SP);
    cg.STR(X9, X17, offsetof(GuestContext, sp));
    cg.STR(X30, X17, offsetof(GuestContext, pc));
    cg.MRS(X9, oaknut::SystemReg::FPSR);
    cg.STR(W9, X17, offsetof(GuestContext, fpsr));
    cg.MRS(X9, oaknut::SystemReg::FPCR);
    cg.STR(W9, X17, offsetof(GuestContext, fpcr));
    cg.MRS(X9, oaknut::SystemReg::NZCV);
    cg.STR(W9, X17, offsetof(GuestContext, nzcv));
    cg.LDR(W10, X17, offsetof(GuestContext, pstate));
    cg.AND(W10, W10, ~NzcvMask);
    cg.AND(W9, W9, NzcvMask);
    cg.ORR(W10, W10, W9);
    cg.STR(W10, X17, offsetof(GuestContext, pstate));

    // svc = base + (x16 - &nvn_slot[0]) / 16
    cg.LDR(X9, slot0);
    cg.SUB(X9, X16, X9);
    cg.LSR(X9, X9, 4);
    cg.MOVZ(W10, SlowGateSvcBase);
    cg.ORR(W9, W9, W10);
    cg.STR(W9, X17, offsetof(GuestContext, svc));

    // esr_el1 -> x0, cleared, plus SupervisorCall
    static_assert(std::is_same_v<std::underlying_type_t<Core::HaltReason>, u64>);
    cg.ADD(X9, X17, offsetof(GuestContext, esr_el1));
    cg.l(esr_retry);
    cg.LDAXR(X0, X9);
    cg.STLXR(W10, XZR, X9);
    cg.CBNZ(W10, esr_retry);
    cg.ORR(X0, X0, static_cast<u64>(Core::HaltReason::SupervisorCall));

    // Back to the host (D2 rule: the TEB interval, then SP at once)
    cg.ADD(X1, X17, HostCtx);
    cg.LDR(W5, X1, offsetof(HostContext, host_fpcr));
    cg.MSR(oaknut::SystemReg::FPCR, X5);
    cg.LDR(X3, X17, HostStackBase);
    cg.LDR(X4, X17, HostStackLimit);
    cg.LDR(X2, X1, offsetof(HostContext, host_sp));
    cg.STP(X3, X4, X18, TebStackBaseOffset);
    cg.MOV(SP, X2);
    for (int i = 0; i < 12; i += 2) {
        cg.LDP(oaknut::XReg{19 + i}, oaknut::XReg{20 + i}, X1, HostRegs + 8 * i);
    }
    for (int i = 0; i < 8; i += 2) {
        cg.LDP(oaknut::QReg{8 + i}, oaknut::QReg{9 + i}, X1, HostVregs + 16 * i);
    }
    cg.RET();
}

bool BuildOnce() {
    if (!Core::NCE::CurrentNceContext::HasTebSlot()) {
        g_build_failure = "no TEB TLS slot for the gate (the NCE parameters would need a C call on the guest stack)";
        return false;
    }
    g_nfunc = nvnh_nfunc();
    g_slot0 = static_cast<const u8*>(nvnh_slot(0));
    g_nforce = nvnh_force_entries(&g_force);
    if (g_nfunc == 0 || g_slot0 == nullptr) {
        g_build_failure = "the host layer has no function table";
        return false;
    }
    if (g_nforce > MaxForce) {
        g_build_failure = fmt::format("the host layer has {} take-overs by symbol, the gate holds {}", g_nforce, MaxForce);
        return false;
    }
    for (size_t k = 0; k < g_nforce; ++k) {
        g_force_records[k].fn = std::strcmp(g_force[k].name, BootstrapName) == 0
                                    ? reinterpret_cast<void*>(&NoexceptBootstrap)
                                    : g_force[k].fn;
    }
    const char* trace = std::getenv("NXN1X_NVN_GATE_TRACE");
    g_trace = trace != nullptr && trace[0] == '1';
    const bool counting = nvnh_counting() != 0;
    g_counting = counting;
    if (g_trace) {
        g_first_seen = std::make_unique<std::atomic<u8>[]>(g_nfunc + g_nforce);
    }
    const char* hold_switch = std::getenv("NXN1X_NVN_GATE_HOLD");   // s16
    g_hold = hold_switch == nullptr || hold_switch[0] != '0';
    LOG_INFO(Core, "NVN gate: H calls hold the thread's parameter lock (no break captures during a host call): {}", g_hold ? "on" : "off (NXN1X_NVN_GATE_HOLD=0)");
    {   // s39
        const char* e = std::getenv("DM_S39");
        g_s39_on = !(e != nullptr && *e != 0 && std::atoi(e) == 0);
        static std::once_flag s39_once;
        if (g_s39_on) {
            std::call_once(s39_once, [] {
                std::thread([] {
                    u64 seen = 0;
                    for (;;) {
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                        const u64 n = g_s39_steals.load(std::memory_order_relaxed);
                        if (n != seen) {
                            LOG_WARNING(Core, "s39: the NVN gate took over a parameter lock left taken ({} so far): it had waited ~8M tries", n);
                            seen = n;
                        }
                    }
                }).detach();
            });
        } else {
            LOG_WARNING(Core, "s39: off (DM_S39=0): NVN gate lock spins are unbounded");
        }
    }
    const char* slow_switch = std::getenv("NXN1X_NVN_GATE_SLOW");
    const bool slow_on = slow_switch == nullptr || slow_switch[0] != '0';
    g_slow.assign(g_nfunc, 0);
    size_t slow_count = 0;
    for (unsigned i = 0; i < g_nfunc && slow_on; ++i) {
        const char* name = nvnh_name(i);
        for (const char* slow_name : SlowNames) {
            if (name != nullptr && std::strcmp(name, slow_name) == 0) {
                g_slow[i] = 1;
                ++slow_count;
                g_slow_list += g_slow_list.empty() ? "" : ", ";
                g_slow_list += name;
            }
        }
    }

    g_s55_slot = S55::SetUp();   // s55: before EmitGate
    std::vector<u32> code;
    oaknut::VectorCodeGenerator cg{code};
    oaknut::Label entries, gate, slow_gate, hook_common, live_flag, trace_fn, slot0, slots_bytes;
    // Function entries (16 bytes): x16 = &nvn_slot[i], into the H gate or the S gate
    cg.l(entries);
    for (unsigned i = 0; i < g_nfunc; ++i) {
        oaknut::Label record;
        cg.LDR(X16, record);
        cg.B(g_slow[i] != 0 ? slow_gate : gate);
        cg.l(record);
        cg.dx(reinterpret_cast<u64>(g_slot0 + 16 * i));
    }
    // Hook entries (16 bytes): x17 = &g_cells[j]
    const size_t hook_entries = static_cast<size_t>(cg.offset());
    for (size_t j = 0; j < MaxHooks; ++j) {
        oaknut::Label cell;
        cg.LDR(X17, cell);
        cg.B(hook_common);
        cg.l(cell);
        cg.dx(reinterpret_cast<u64>(&g_cells[j]));
    }
    // Live: the cell's record, through its function entry when it is nvn_slot[i] (H or S), else (a take-over) the H gate.
    // Off: the original start, or the layer anyway when the definition has none. x9, x10 are free at a call
    oaknut::Label off, to_record;
    cg.l(hook_common);
    cg.LDR(X16, live_flag);
    cg.LDRB(W16, X16);
    cg.CBZ(W16, off);
    cg.l(to_record);
    cg.LDR(X16, X17, 0);
    cg.LDR(X9, slot0);
    cg.SUB(X9, X16, X9);
    cg.LDR(X10, slots_bytes);
    cg.CMP(X9, X10);
    cg.B(oaknut::Cond::HS, gate);
    cg.ADR(X10, entries);
    cg.ADD(X10, X10, X9);
    cg.BR(X10);
    cg.l(off);
    cg.LDR(X16, X17, 8);
    cg.CBZ(X16, to_record);
    cg.BR(X16);
    cg.l(gate);
    EmitGate(cg, counting, g_trace, trace_fn);
    cg.l(slow_gate);
    EmitSlowGate(cg, slot0);
    cg.l(live_flag);
    cg.dx(reinterpret_cast<u64>(&g_live));
    cg.l(trace_fn);
    cg.dx(reinterpret_cast<u64>(&TraceFirst));
    cg.l(slot0);
    cg.dx(reinterpret_cast<u64>(g_slot0));
    cg.l(slots_bytes);
    cg.dx(16 * static_cast<u64>(g_nfunc));

    g_block_size = code.size() * sizeof(u32);
    void* block = VirtualAlloc(nullptr, g_block_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (block == nullptr) {
        g_build_failure = fmt::format("VirtualAlloc of {} bytes failed ({})", g_block_size, GetLastError());
        return false;
    }
    std::memcpy(block, code.data(), g_block_size);
    DWORD old_protect{};
    if (!VirtualProtect(block, g_block_size, PAGE_EXECUTE_READ, &old_protect)) {
        g_build_failure = fmt::format("VirtualProtect failed ({})", GetLastError());
        VirtualFree(block, 0, MEM_RELEASE);
        return false;
    }
    FlushInstructionCache(GetCurrentProcess(), block, g_block_size);
    g_block = static_cast<u8*>(block);
    g_hook_entries = reinterpret_cast<u64>(g_block) + hook_entries;
    g_entries.resize(g_nfunc);
    for (unsigned i = 0; i < g_nfunc; ++i) {
        g_entries[i] = g_block + 16 * i;
    }
    nvnh_set_gates(g_entries.data());
    nvnh_set_ksleep(&S52::KSleep);         // s52
    nvnh_set_core_state(&S52::CoreState);  // s52
    nvnh_set_defer_call(g_s55_slot != 0 ? &S55::Defer : nullptr);   // s55
    LOG_INFO(Core, "NVN gate block: {} function entries, {} hook entries, {} bytes at {}; calls {}counted{}", g_nfunc, MaxHooks,
             g_block_size, fmt::ptr(g_block), counting ? "" : "not ", g_trace ? ", first calls traced" : "");
    LOG_INFO(Core, "NVN gate: {} functions run on the core's host thread (S gate){}{}", slow_count, slow_count ? ": " : "",
             slow_on ? g_slow_list : std::string{"; NXN1X_NVN_GATE_SLOW=0 runs every function as H"});
    return true;
}

bool Build() {
    std::call_once(g_build_once, [] { g_built = BuildOnce(); });
    return g_built;
}

std::string_view RecordName(const void* record) {
    const auto* p = static_cast<const u8*>(record);
    if (p >= g_slot0 && p < g_slot0 + 16 * static_cast<size_t>(g_nfunc)) {
        return nvnh_name(static_cast<unsigned>((p - g_slot0) / 16));
    }
    for (size_t k = 0; k < g_nforce; ++k) {
        if (record == &g_force_records[k]) {
            return g_force[k].name;
        }
    }
    return "?";
}

// xcdedlss (11-22, diagnostic, NXN1X_NVN_GATE_TRACE=1): for the DLSS and texture-size calls, the first 8 distinct guest return addresses, each with the guest
// code around it (1024 words before, 128 after) as hex, to disassemble the game's decision to evaluate DLSS or not (Xenoblade Chronicles DE's title)
static void TraceCaller(std::string_view nv, u64 lr) noexcept {
    const std::string ns_(nv); const char* name = ns_.c_str();
    if (lr == 0 || name == nullptr) return;
    if (!std::strstr(name, "Dlss") && !std::strstr(name, "TextureGetWidth") && !std::strstr(name, "TextureGetHeight")) return;
    static std::mutex mu; static u64 seen[64]; static int ns;
    std::scoped_lock lk{mu};
    int same = 0, k = 0;
    for (int i = 0; i < ns; ++i) { if (seen[i] == lr) return; }
    for (int i = 0; i < ns; ++i) { k += 0; }
    (void)same; (void)k;
    if (ns >= 64) return;
    seen[ns++] = lr;
    int per = 0; static std::unordered_map<std::string, int> per_name; per = ++per_name[name];
    if (per > 8) return;
    LOG_INFO(Core, "xcdedlss: {} called from guest {:#x}", name, lr);
    const u32* p = reinterpret_cast<const u32*>(lr & ~u64{3});
    for (int i = -1024; i < 128; i += 8) {
        std::string line;
        for (int j = 0; j < 8; ++j) line += fmt::format(" {:08x}", p[i + j]);
        LOG_INFO(Core, "xcdedlss: {:#x}:{}", lr + static_cast<s64>(i) * 4, line);
    }
}
void TraceFirst(const void* record, u64 x0, u64 lr) noexcept {
    TraceCaller(RecordName(record), lr);
    const auto* p = static_cast<const u8*>(record);
    size_t index = 0;
    if (p >= g_slot0 && p < g_slot0 + 16 * static_cast<size_t>(g_nfunc)) {
        index = static_cast<size_t>((p - g_slot0) / 16);
    } else {
        index = g_nfunc;
        while (index < g_nfunc + g_nforce && record != &g_force_records[index - g_nfunc]) {
            ++index;
        }
        if (index == g_nfunc + g_nforce) {
            return;
        }
    }
    if (g_first_seen[index].exchange(1, std::memory_order_relaxed) == 0) {
        LOG_INFO(Core, "NVN gate: first call of {} (x0={:#x})", RecordName(record), x0);
    }
}

void* NoexceptBootstrap(const char* name) noexcept {
    if (!g_bootstrap_called.exchange(true, std::memory_order_relaxed)) {
        LOG_INFO(Core, "NVN gate: the game called nvnBootstrapLoader(\"{}\"){}", name != nullptr ? name : "(null)",
                 g_live.load(std::memory_order_relaxed) ? "" : " while the gate is off (a definition without an original start): the layer runs anyway");
    }
    void* const address = nvnh_bootstrap_loader(name);
    if (g_trace && g_traced_lookups.fetch_add(1, std::memory_order_relaxed) < 256) {
        LOG_INFO(Core, "NVN gate: nvnBootstrapLoader(\"{}\") = {}", name != nullptr ? name : "(null)", fmt::ptr(address));
    }
    return address;
}

// The dynamic symbol table of a module image (its MOD0 header), as core/arm/symbols.cpp reads it, with the section index and type kept
template <typename F>
void ForEachSymbol(std::span<const u8> image, F&& f) {
    const auto read = [&](void* out, u64 offset, size_t size) {
        if (offset > image.size() || size > image.size() - offset) {
            return false;
        }
        std::memcpy(out, image.data() + offset, size);
        return true;
    };
    struct {
        u32 version, header_offset, version_offset;
    } loc{};
    struct {
        u32 signature, dynamic_offset;
    } hdr{};
    if (!read(&loc, 0, sizeof loc) || !read(&hdr, loc.header_offset, sizeof hdr) || hdr.signature != Common::MakeMagic('M', 'O', 'D', '0')) {
        return;
    }
    u64 strtab{}, symtab{}, syment = sizeof(Common::ELF::Elf64_Sym);
    for (u64 at = u64{loc.header_offset} + hdr.dynamic_offset;; at += sizeof(Common::ELF::Elf64_Dyn)) {
        Common::ELF::Elf64_Dyn dyn{};
        if (!read(&dyn, at, sizeof dyn) || dyn.d_tag == Common::ELF::ElfDtNull) {
            break;
        }
        if (dyn.d_tag == Common::ELF::ElfDtStrtab) {
            strtab = dyn.d_un.d_ptr;
        } else if (dyn.d_tag == Common::ELF::ElfDtSymtab) {
            symtab = dyn.d_un.d_ptr;
        } else if (dyn.d_tag == Common::ELF::ElfDtSyment) {
            syment = dyn.d_un.d_val;
        }
    }
    if (strtab == 0 || symtab == 0 || symtab >= strtab || syment < sizeof(Common::ELF::Elf64_Sym)) {
        return;
    }
    for (u64 at = symtab; at + sizeof(Common::ELF::Elf64_Sym) <= strtab; at += syment) {
        Common::ELF::Elf64_Sym sym{};
        if (!read(&sym, at, sizeof sym) || strtab + sym.st_name >= image.size()) {
            return;
        }
        const char* name = reinterpret_cast<const char*>(image.data() + strtab + sym.st_name);
        const size_t max_length = image.size() - (strtab + sym.st_name);
        f(std::string_view{name, strnlen(name, max_length)}, sym);
    }
}

} // Anonymous namespace

// z1: the SDK major version, read as the native host's loader.c dm_sdk_major reads it: from the version string the SDK's libraries keep in the sdk
// module's data ("NintendoSDK_nnSdk-10_4_0"; SDK 12 spells it NintendoSdk_; Switch 2 SDKs name a variant first, "nnSdk-Generic_20_5_19").
// The NVN API version the game's own driver reports follows it (hle_nvn.c nvn_api: SDK 10 and older 53.313, later 55.15). Games check it and create
// no device on a mismatch: Breath of the Wild 1.6.0 was told 55.15 and read its null device right after nvnDeviceInitialize (run 75)
static void ReadSdkVersion(std::span<const u8> image) {
    if (const char* e = std::getenv("DM_Z1"); e != nullptr && e[0] == '0') {
        LOG_INFO(Core, "z1: off (DM_Z1=0): the SDK version is not read, the NVN API takes the newer SDKs' 55.15");
        return;
    }
    static constexpr std::string_view Key{"_nnSdk-"};
    static constexpr std::string_view Vendor{"nintendosdk"};
    const std::string_view all{reinterpret_cast<const char*>(image.data()), image.size()};
    int major = 0;
    std::string_view shown;
    for (size_t p = all.find(Key); p != std::string_view::npos && major == 0; p = all.find(Key, p + Key.size())) {
        if (p < Vendor.size()) {
            continue;
        }
        bool vendor = true;
        for (size_t k = 0; k < Vendor.size() && vendor; ++k) {
            const char c = all[p - Vendor.size() + k];
            vendor = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c) == Vendor[k];
        }
        if (!vendor) {
            continue;
        }
        const size_t start = p + Key.size();
        size_t v = start;
        while (v < all.size() && ((all[v] >= 'A' && all[v] <= 'Z') || (all[v] >= 'a' && all[v] <= 'z'))) {
            ++v;   // a variant name ("Generic") and its '_'
        }
        if (v > start && v < all.size() && all[v] == '_') {
            ++v;
        }
        while (v < all.size() && all[v] >= '0' && all[v] <= '9' && major < 1000) {
            major = major * 10 + (all[v++] - '0');
        }
        if (major != 0) {
            size_t end = start;
            while (end < all.size() && end - start < 20 && all[end] != '\0') {
                ++end;
            }
            shown = all.substr(start, end - start);
        }
    }
    nvnh_set_sdk_major(major);
    {   // s69: the NVN path skips draws whose Eden pipeline is still being built when the (per-game) asynchronous shader option is on
        const bool async = Settings::values.use_asynchronous_shaders.GetValue();
        nvnh_set_eden_async(async ? 1 : 0);
        LOG_INFO(Core, "s69: use_asynchronous_shaders = {} for the NVN path (DM_VK_EDEN_ASYNC overrides)", async ? 1 : 0);
    }
    if (major != 0) {
        LOG_INFO(Core, "z1: sdk module: NintendoSDK nnSdk-{} (major {}): NVN device API {}", shown, major, major <= 10 ? "53.313" : "55.15");
    } else {
        LOG_WARNING(Core, "z1: no SDK version string in the sdk module: the NVN API takes the newer SDKs' 55.15");
    }
}

// NVN2 (Switch 2 programs only): what the native host's loader reads from the sdk module for its NVN2 device (host dm/host/loader.c:170-221,
// NXN1X builds no loader.c): dm_sdk_ounce, the build spec of the file system access log line ("spec: Ounce"; Switch 1 SDKs say "spec: NX"), and
// dm_sdk_nvn2_item, the constants of the cases of NVN2's own nvnDeviceGetInteger switch, compiled into the game's sdk (mov w0, #231 with cmp wN, #<last
// item> / adrp + add (the byte table) / adr (the cases) / ldrb / br around it; a constant case is mov w0, #imm16; b). Scanned here from the module image
// the same way, and handed to the layer (hle_nvn.c nvnh_set_nvn2). The image starts at the module's base, so page offsets are module offsets.
static void ReadNvn2(std::span<const u8> image, const Kernel::CodeSet::Segment& code) {
    const size_t text_begin = (std::min)(static_cast<size_t>(code.offset), image.size());
    const size_t text_end = (std::min)(static_cast<size_t>(code.offset) + code.size, image.size());
    const size_t ro_begin = text_end;   // loader.c: ro_off .. key_page_off; here everything after the text
    const std::string_view rest{reinterpret_cast<const char*>(image.data()) + ro_begin, image.size() - ro_begin};
    const bool ounce = rest.find("spec: Ounce") != std::string_view::npos;
    const auto word = [&](size_t off) {
        u32 w{};
        std::memcpy(&w, image.data() + off, 4);
        return w;
    };
    const auto imm21 = [](u32 w) {   // adr / adrp immediate (immhi:immlo), sign-extended
        s64 v = static_cast<s64>(((w >> 3) & 0x1FFFFC) | ((w >> 29) & 3));
        return (v & 0x100000) ? v - 0x200000 : v;
    };
    std::array<int, 256> values{};
    std::array<unsigned char, 256> known{};
    unsigned count = 0;
    unsigned constants = 0;
    size_t at = 0;
    bool found = false;
    // loader.c: for p from 8 while p + 9 < n words, the window p - 8 .. p + 8 (and the word after an adrp)
    for (size_t p = text_begin + 8 * 4; !found && p + 10 * 4 <= text_end; p += 4) {
        if (word(p) != 0x52801CE0) {   // mov w0, #231
            continue;
        }
        s64 table = -1;
        s64 cases = -1;
        int lim = -1;
        bool ldrb = false;
        bool br = false;
        for (size_t q = p - 8 * 4; q <= p + 8 * 4; q += 4) {
            const u32 w = word(q);
            if ((w & 0xFFC0001F) == 0x7100001F) {   // cmp wN, #imm
                lim = static_cast<int>((w >> 10) & 0xFFF);
            } else if ((w & 0x9F000000) == 0x90000000 && (word(q + 4) & 0xFFC00000) == 0x91000000) {   // adrp + add
                table = static_cast<s64>(q & ~size_t{0xFFF}) + imm21(w) * 4096 + ((word(q + 4) >> 10) & 0xFFF);
            } else if ((w & 0x9F000000) == 0x10000000) {   // adr
                cases = static_cast<s64>(q) + imm21(w);
            } else if ((w & 0xFFE0FC00) == 0x38606800) {   // ldrb wK, [xT, xI]
                ldrb = true;
            } else if ((w & 0xFFFFFC1F) == 0xD61F0000) {   // br
                br = true;
            }
        }
        if (table < 0 || cases < 0 || lim <= 1 || !ldrb || !br) {
            continue;
        }
        if (static_cast<size_t>(table) < ro_begin || static_cast<size_t>(table) + lim >= image.size()) {
            continue;
        }
        found = true;
        at = p;
        count = (std::min)(static_cast<unsigned>(lim) + 1, 256u);   // loader.c answers items 0 .. last
        for (unsigned item = 0; item < count; ++item) {
            const size_t c = static_cast<size_t>(cases) + size_t{image[static_cast<size_t>(table) + item]} * 4;
            if (c + 8 > image.size()) {
                continue;
            }
            const u32 c0 = word(c);
            const u32 c1 = word(c + 4);
            if ((c0 & 0xFFE0001F) != 0x52800000 || (c1 & 0xFC000000) != 0x14000000) {   // mov w0, #imm16; b
                continue;
            }
            values[item] = static_cast<int>((c0 >> 5) & 0xFFFF);
            known[item] = 1;
            ++constants;
        }
        if (static_cast<unsigned>(lim) + 1 > 256u) {
            LOG_WARNING(Core, "NVN2: nvnDeviceGetInteger switch has {} items; the layer keeps the first 256", lim + 1);
        }
    }
    nvnh_set_nvn2(ounce ? 1 : 0, values.data(), known.data(), count);
    const std::string minor = known[1] ? std::to_string(values[1]) : std::string{"? (the layer's newest known)"};
    if (found) {
        LOG_INFO(Core, "NVN2: api 231.{}, ounce {}, nvnDeviceGetInteger switch at sdk+{:#x}: {} items, {} constants", minor,
                 ounce ? 1 : 0, at, count, constants);
    } else {
        LOG_WARNING(Core, "NVN2: api 231.{}, ounce {}, nvnDeviceGetInteger switch not found in the sdk module", minor, ounce ? 1 : 0);
    }
}

std::vector<Core::NCE::EntryHook> PlanHooks(std::span<const u8> image, const Kernel::CodeSet::Segment& code, std::string_view module) {
    std::vector<Core::NCE::EntryHook> hooks;
    if (!Wanted()) {
        return hooks;
    }
    if (!Build()) {
        Refuse(fmt::format("the NVN gate could not be built: {}", g_build_failure));
        return hooks;
    }
    if (module == "sdk") {
        ReadSdkVersion(image);   // z1
        if (Settings::CurrentProgramIsSwitch2()) {
            ReadNvn2(image, code);   // NVN2
        }
    }
    std::scoped_lock lock{g_plan_mutex};
    struct Want {
        const void* record;
        bool needs_import;
    };
    std::unordered_map<std::string_view, Want> wanted;
    for (size_t k = 0; k < g_nforce; ++k) {
        wanted.emplace(g_force[k].name, Want{&g_force_records[k], false});
    }
    for (unsigned i = 0; i < g_nfunc; ++i) {
        wanted.emplace(nvnh_name(i), Want{g_slot0 + 16 * i, true});
    }
    // Imports first (this module's own definitions of a name it also imports do not count), then definitions
    ForEachSymbol(image, [&](std::string_view name, const Common::ELF::Elf64_Sym& sym) {
        if (sym.st_shndx == Common::ELF::ElfShnUndef && wanted.contains(name)) {
            g_imported.emplace(name);
        }
    });
    ForEachSymbol(image, [&](std::string_view name, const Common::ELF::Elf64_Sym& sym) {
        const u8 bind = Common::ELF::ElfStBind(sym.st_info);
        if (sym.st_shndx == Common::ELF::ElfShnUndef || Common::ELF::ElfStType(sym.st_info) != Common::ELF::ElfTypeFunc ||
            (bind != Common::ELF::ElfBindGlobal && bind != Common::ELF::ElfBindWeak)) {
            return;
        }
        const auto it = wanted.find(name);
        if (it == wanted.end() || (it->second.needs_import && !g_imported.contains(std::string{name}))) {
            return;
        }
        const u64 code_start = static_cast<u64>(code.offset);
        if (sym.st_value < code_start || sym.st_value - code_start >= code.size || (sym.st_value & 3) != 0) {
            LOG_WARNING(Core, "NVN gate: {} in {} is at {:#x}, outside its text: not taken over", name, module, sym.st_value);
            return;
        }
        if (g_hooks.size() == MaxHooks) {
            Refuse(fmt::format("more than {} NVN definitions to take over ({} in {})", MaxHooks, name, module));
            return;
        }
        const size_t cell = g_hooks.size();
        g_cells[cell] = HookCell{it->second.record, 0};
        const u32 text_offset = static_cast<u32>(sym.st_value - code_start);
        g_hooks.push_back(HookInfo{std::string{name}, std::string{module}, text_offset, cell});
        hooks.push_back(Core::NCE::EntryHook{text_offset, g_hook_entries + 16 * cell, &g_cells[cell].original_start});
        g_bootstrap_hooked = g_bootstrap_hooked || name == BootstrapName;
        LOG_INFO(Core, "NVN gate: {} in {} (text {:#x}) taken over{}", name, module, text_offset,
                 it->second.needs_import ? " (imported directly by another module)" : "");
    });
    if (!Wanted()) {
        hooks.clear();
    }
    return hooks;
}

void KeepHookWords(std::string_view module, std::span<const u8> text) {
    std::scoped_lock lock{g_plan_mutex};
    std::vector<u32> words;
    for (const HookInfo& hook : g_hooks) {
        if (hook.module == module && hook.text_offset + sizeof(u32) <= text.size()) {
            u32 word{};
            std::memcpy(&word, text.data() + hook.text_offset, sizeof word);
            words.push_back(word);
        }
    }
    if (!words.empty()) {
        g_kept_words[std::string{module}] = std::move(words);
    }
}

void RestoreHookWords(std::string_view module, std::span<u8> text, Core::NCE::EntryTrampolines& post_handlers) {
    std::scoped_lock lock{g_plan_mutex};
    if (g_built && !g_range_registered && !g_hooks.empty()) {
        Core::NCE::WindowsPatchCodeMetadata::RegisterRange(post_handlers, reinterpret_cast<u64>(g_block), g_block_size);
        g_range_registered = true;
    }
    const auto kept = g_kept_words.find(std::string{module});
    if (kept == g_kept_words.end()) {
        return;
    }
    size_t k = 0;
    for (const HookInfo& hook : g_hooks) {
        if (hook.module != module || hook.text_offset + sizeof(u32) > text.size() || k >= kept->second.size()) {
            continue;
        }
        const u32 want = kept->second[k++];
        u32 word{};
        std::memcpy(&word, text.data() + hook.text_offset, sizeof word);
        if (word != want) {
            LOG_WARNING(Core, "NVN gate: the x18 site patcher rewrote the first instruction of {} in {} ({:#010x}): put back", hook.name,
                        module, word);
            std::memcpy(text.data() + hook.text_offset, &want, sizeof want);
        }
    }
    g_kept_words.erase(kept);
}

void GateBeginBoot() {
    std::scoped_lock lock{g_plan_mutex};
    g_live.store(0, std::memory_order_relaxed);
    g_hooks.clear();
    g_imported.clear();
    g_kept_words.clear();
    g_cells = {};
    g_bootstrap_hooked = false;
    g_bootstrap_called.store(false, std::memory_order_relaxed);
    g_range_registered = false;
}

void GateSetLive(bool live) {
    g_live.store(live ? 1 : 0, std::memory_order_release);
}

bool GateLive() {
    return g_live.load(std::memory_order_acquire) != 0;
}

bool GateBootstrapHooked() {
    std::scoped_lock lock{g_plan_mutex};
    return g_bootstrap_hooked;
}

bool GateBootstrapCalled() {
    return g_bootstrap_called.load(std::memory_order_relaxed);
}

size_t GateHookCount() {
    std::scoped_lock lock{g_plan_mutex};
    return g_hooks.size();
}

size_t GateHooksWithoutStart() {
    std::scoped_lock lock{g_plan_mutex};
    size_t n = 0;
    for (const HookInfo& hook : g_hooks) {
        n += g_cells[hook.cell].original_start == 0 ? 1 : 0;
    }
    return n;
}

namespace S52 {
// s52 (lane A): the guest threads now inside an S-gate call (only there may the host wait as a guest-kernel sleep: the call runs where an SVC handler
// runs, after RunThread returned), and whether they slept; the kernel, known from the first S call
std::atomic<Kernel::KernelCore*> g_kernel{nullptr};
std::mutex g_mu;
std::vector<std::pair<Kernel::KThread*, bool>> g_in;
int KSleep(uint64_t ns) {   // the host's pacing wait as svc SleepThread does it; 0 = not allowed here (not inside an S call), the host waits as before
    Kernel::KernelCore* const k = g_kernel.load(std::memory_order_acquire);
    if (k == nullptr || ns == 0) {
        return 0;
    }
    Kernel::KThread* const t = Kernel::GetCurrentThreadPointer(*k);
    {
        std::scoped_lock lock{g_mu};
        auto it = std::find_if(g_in.begin(), g_in.end(), [t](const auto& e) { return e.first == t; });
        if (it == g_in.end()) {
            return 0;
        }
        it->second = true;
    }
    s64 timeout = k->HardwareTimer().GetTick() + static_cast<s64>(ns) + 2;
    if (timeout <= 0) {
        timeout = (std::numeric_limits<s64>::max)();
    }
    static_cast<void>(t->Sleep(*k, timeout));
    return 1;
}
int CoreState(int* waiter) {   // the calling thread's guest core, and whether its scheduler wants another thread there
    Kernel::KernelCore* const k = g_kernel.load(std::memory_order_acquire);
    *waiter = 0;
    if (k == nullptr) {
        return -1;
    }
    const size_t c = k->CurrentPhysicalCoreIndex();
    const Kernel::KScheduler& s = k->Scheduler(c);
    Kernel::KThread* const hi = s.S52HighestPriorityThread();
    *waiter = hi != nullptr && hi != s.GetSchedulerCurrentThread();
    return static_cast<int>(c);
}
} // namespace S52

namespace S55 {
// s55 (lane A): one owed guest call per OS thread (the H gate's return reads it through a TEB TLS slot and clears it): the game's command memory callback,
// which the host layer cannot call from the host stack. Refused inside an S-gate call (the thread may sleep and resume on another core's host thread),
// off a guest thread, or with one already owed in this call
struct alignas(16) Pending {
    u64 cb, cmd, event, size, data, pad;
};
static_assert(offsetof(Pending, cmd) == 8 && offsetof(Pending, size) == 24 && offsetof(Pending, data) == 32);
thread_local Pending t_pending{};
DWORD g_tls = TLS_OUT_OF_INDEXES;
u32 SetUp() {
    const char* e = std::getenv("DM_S55");
    if (e != nullptr && *e != 0 && std::atoi(e) == 0) {
        LOG_WARNING(Core, "s55: off (DM_S55=0): the game's command memory callback is never called, records past its memory go to host-lent segments");
        return 0;
    }
    if (g_tls == TLS_OUT_OF_INDEXES) {
        // s55tls: the slot main() reserved before Qt and the drivers (CurrentNceContext::ReserveDirectSlots), not a late TlsAlloc
        const DWORD i = Core::NCE::CurrentNceContext::S55SlotIndex();
        if (i >= Core::NCE::CurrentNceContext::TebTlsSlotCount) {
            LOG_WARNING(Core, "s55: off: no TEB TLS slot below {} for the owed guest call (reserved index {})", Core::NCE::CurrentNceContext::TebTlsSlotCount, i);
            return 0;
        }
        g_tls = i;
    }
    LOG_INFO(Core, "s55: the game's command memory callback runs at the H gate's return (TLS slot {}; DM_S55=0 off)", g_tls);
    return Core::NCE::CurrentNceContext::TebTlsSlotsOffset + static_cast<u32>(g_tls) * 8U;
}
int Defer(void* cb, void* cmd, int event, size_t size, void* data) {
    if (g_tls == TLS_OUT_OF_INDEXES || cb == nullptr || t_pending.cb != 0 || Core::NCE::CurrentNceContext::Get() == nullptr) {
        return 0;
    }
    if (Kernel::KernelCore* const k = S52::g_kernel.load(std::memory_order_acquire)) {
        Kernel::KThread* const t = Kernel::GetCurrentThreadPointer(*k);
        std::scoped_lock lock{S52::g_mu};
        if (std::any_of(S52::g_in.begin(), S52::g_in.end(), [t](const auto& en) { return en.first == t; })) {
            return 0;
        }
    }
    t_pending.cmd = reinterpret_cast<u64>(cmd);
    t_pending.event = static_cast<u64>(static_cast<u32>(event));
    t_pending.size = static_cast<u64>(size);
    t_pending.data = reinterpret_cast<u64>(data);
    TlsSetValue(g_tls, &t_pending);
    t_pending.cb = reinterpret_cast<u64>(cb);
    return 1;
}
} // namespace S55

namespace {
// The host function with x0-x7 = args and x16 = its record, as the H gate calls it (nvn_unimpl_entry and nvn_big_entry read x16)
u64 CallWithRecord(const Record* record, const std::array<u64, 8>& args) {
    register u64 x0 asm("x0") = args[0];
    register u64 x1 asm("x1") = args[1];
    register u64 x2 asm("x2") = args[2];
    register u64 x3 asm("x3") = args[3];
    register u64 x4 asm("x4") = args[4];
    register u64 x5 asm("x5") = args[5];
    register u64 x6 asm("x6") = args[6];
    register u64 x7 asm("x7") = args[7];
    register u64 x16 asm("x16") = reinterpret_cast<u64>(record);
    register u64 x17 asm("x17") = reinterpret_cast<u64>(record->fn);
    asm volatile("blr x17"
                 : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3), "+r"(x4), "+r"(x5), "+r"(x6), "+r"(x7), "+r"(x16), "+r"(x17)
                 :
                 : "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x30", "memory", "cc", "v0", "v1", "v2", "v3", "v4",
                   "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                   "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    return x0;
}
} // Anonymous namespace

void GateSlowCall(Core::ArmInterface& cpu, u32 svc, Core::System& system) {
    const u32 index = svc - SlowGateSvcBase;
    if (index >= g_nfunc || g_slow.empty() || g_slow[index] == 0) {
        LOG_CRITICAL(Core, "NVN gate: S call {:#x} is not an S function", svc);
        return;
    }
    auto* const record = reinterpret_cast<Record*>(const_cast<u8*>(g_slot0) + 16 * static_cast<size_t>(index));
    std::array<u64, 8> args{};
    cpu.GetSvcArguments(args);
    if (g_counting) {
        record->count.fetch_add(1, std::memory_order_relaxed);
    }
    if (g_trace) {
        TraceFirst(record, args[0], 0);
    }
    // s52: the thread is inside an S call (a guest-kernel sleep is allowed); after one it may resume on another core, so the result goes to
    // the current core as Svc::Call does it (LoadSvcArguments), not to the interface it came from
    auto& kernel = system.Kernel();
    S52::g_kernel.store(&kernel, std::memory_order_release);
    Kernel::KThread* const s52_thread = Kernel::GetCurrentThreadPointer(kernel);
    {
        std::scoped_lock lock{S52::g_mu};
        S52::g_in.emplace_back(s52_thread, false);
    }
    args[0] = CallWithRecord(record, args);
    bool s52_slept = false;
    {
        std::scoped_lock lock{S52::g_mu};
        auto it = std::find_if(S52::g_in.begin(), S52::g_in.end(), [s52_thread](const auto& e) { return e.first == s52_thread; });
        if (it != S52::g_in.end()) {
            s52_slept = it->second;
            S52::g_in.erase(it);
        }
    }
    if (s52_slept) {
        kernel.CurrentPhysicalCore().LoadSvcArguments(Kernel::GetCurrentProcess(kernel), args);
    } else {
        cpu.SetSvcArguments(args);
    }
}

} // namespace Core::NvnLayer
