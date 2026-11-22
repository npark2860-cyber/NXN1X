// SPDX-License-Identifier: GPL-2.0-or-later
// NVN path, the gate (NXN1X docs NVN_HOST_BRIDGE.md 4.1; design v0.3 chapter 4.2 / D-22 for the hook, chapter 5.3 for the H gate).
// The module loader's side: which definitions in a game module are taken over, and keeping those first instructions as the patcher left them.
// Windows ARM64 NCE builds with the host layer only (NXN1X_NVN_HOST); everything else is in nvn_layer.h.
#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "core/arm/nce/patcher.h"
#include "core/hle/kernel/code_set.h"

namespace Core {
class ArmInterface;
class System;
}

namespace Core::NvnLayer {

// First pass of the module loader (nso.cpp, before PatchText), one module: the definitions it takes over. nvnBootstrapLoader and the host's
// other take-overs by symbol (nvnh_force_entries) wherever they are defined, and an NVN function another module imports directly. Empty when
// this boot is not on the NVN path, or the gate cannot be built (then the boot falls back to Emulation, logged)
std::vector<Core::NCE::EntryHook> PlanHooks(std::span<const u8> image, const Kernel::CodeSet::Segment& code, std::string_view module);

// Second pass (nso.cpp): the module's hooked first instructions right after RelocateAndCopy wrote them (branches into the patch section),
// and again after the Windows x18 site patcher, which could rewrite one of them (put back, logged). The second call also registers the gate
// block as generated code with the process (cross-thread breaks wait until the thread is back in guest code)
void KeepHookWords(std::string_view module, std::span<const u8> text);
void RestoreHookWords(std::string_view module, std::span<u8> text, Core::NCE::EntryTrampolines& post_handlers);

// ---- for nvn_layer.cpp ----
// A new boot (OnAppLoad): the last boot's hooks are forgotten and the gate is off
void GateBeginBoot();
// On: the hook entries go into the layer. Off: they run the definitions' original starts (the game's own code, the Emulation path)
void GateSetLive(bool live);
bool GateLive();
// Whether a module defines nvnBootstrapLoader (first pass), and whether the game has called it through the gate
bool GateBootstrapHooked();
bool GateBootstrapCalled();
// How many definitions this boot took over, and how many of them have no original start (no fallback to the game's own code)
size_t GateHookCount();
size_t GateHooksWithoutStart();
// SlowGateCall (nvn_layer.h): the S function svc - SlowGateSvcBase with the thread's saved arguments, on the core's host thread
void GateSlowCall(Core::ArmInterface& cpu, u32 svc, Core::System& system);   // s52: system

} // namespace Core::NvnLayer
