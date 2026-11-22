// SPDX-License-Identifier: GPL-2.0-or-later
// NVN path (NXN1X docs NVN_HOST_BRIDGE.md): the game's NVN calls go to the native host's NVN layer (src/nvn_host, the host tree's dm/gfx/nvn and
// dm/windows/gfx_vk_*.c) instead of the game's own NVN driver and the Maxwell GPU emulation. Windows ARM64 NCE only.
// This file is the process-level switch: which path this boot takes (setting nce_gpu_path, read once at boot). The gate (the call from guest code
// into the host layer) is nvn_gate.cpp; the Vulkan side (device, queue, pipeline cache) is video_core/renderer_vulkan/nvn_bridge_device.cpp.
#pragma once

#include <functional>   // NVN2 output (SetNx2Compose)
#include <optional>
#include <string_view>

#include "common/common_types.h"

namespace Core {
class ArmInterface;
class System;
}

namespace Core::Memory {
class Memory;
}

namespace Common {
class HostMemory;
}

namespace Core::NvnLayer {

// Whether this build has the host layer at all (src/nvn_host built and linked)
bool Available();

// The application's program metadata was read (loader, after Settings::SetNceEnabled): decides this boot's path once.
// Nvn when nce_gpu_path is Nvn, NCE runs the game, the host layer is linked and has not run in this process before
// (its state lives for the process); otherwise Emulation, with one log line when Nvn was asked for
void OnAppLoad();
void SetTitleId(unsigned long long title_id);   // s31: per-title profile defaults

// This boot's path, as OnAppLoad decided it (false before it, and in builds without the layer).
// Read by the loader (the gate, NVN_HOST_BRIDGE.md 4.1) and by RendererVulkan (the device, 4.3), both after OnAppLoad
bool Wanted();

// u1: the NVN layer has run in this process (its Vulkan device exists): a later boot here cannot take the NVN path (OnAppLoad), so the frontend
// starts NXN1X again instead (main_window.cpp BootGame)
bool RanInProcess();

// A part of the NVN path could not be set up (the gate could not be built, ...): this boot falls back to Emulation, logged
// once with why. The gate goes off: the hooked definitions run the game's own code. Before the game has called nvnBootstrapLoader
// only: nothing undoes NVN entries the game already holds
void Refuse(std::string_view why);

// The first pass of the module loader is over (every module planned, nvn_gate.h PlanHooks): Emulation when no module defines nvnBootstrapLoader
void OnModulesPlanned();

// The renderer handed its device to the layer (Vulkan::NvnBridge::Attach): the gate goes on, the game's NVN calls reach the layer
// from now on. Until then the hooked definitions run the game's own code
void GateOn();

// RendererVulkan::Composite with a frame of the game: a game that draws before calling nvnBootstrapLoader is not an NVN game (Emulation, logged once)
void OnFrame();

// The S gate (nvn_gate.cpp): an NVN function that waits or touches the kernel stops the guest thread as an SVC does, with this number plus the
// function's index. Guest SVC immediates have 16 bits, so the range never meets a real SVC
constexpr u32 SlowGateSvcBase = 0x4E560000;
constexpr bool IsSlowGateSvc(u32 svc) {
    return (svc & 0xFFFF0000U) == SlowGateSvcBase;
}
// PhysicalCore::RunThread, for such a stop: runs the function on the core's host thread with the saved x0-x7 and puts its result in x0.
// The thread then resumes at the caller's return address
void SlowGateCall(Core::ArmInterface& cpu, u32 svc, Core::System& system);   // s52: system

// Step 4 (NVN_HOST_BRIDGE.md 4.2): a write fault at address on a page the host layer write-protected to watch it (dm/gfx/nvn/nvn_dirty.c).
// true = it was such a page: the layer marked it written and made it writable, so the write can simply run again.
// Called from the NCE exception handler on any thread (guest code, host code, the guest stack) and from Memory::InvalidateNCE:
// atomics, VirtualProtect, QueryPerformanceCounter and (rarely) VirtualQuery only, no log, no allocation, no lock. The usual fault costs
// on_fault alone. Only when the same thread faults on the same page again at once (the write reaches into the next page, or something
// else protected the page) does it release the neighbouring pages and check the page with VirtualQuery (about 1.5 ms on a large view);
// a page still read-only then is false and counted as foreign
bool OnWatchedWrite(u64 address);
// Counters of the write watch's unusual cases (OnWatchedWrite), for the log
struct WatchStats {
    u64 foreign{};        // faults on the layer's pages that were still read-only after on_fault (not the layer's)
    u64 foreign_first{};  // the first such address
    u64 repeats{};        // a fault on the same page right after on_fault let the write run again
    u64 repeat_worst{};   // the longest such run on one page
    u64 repeat_addr{};    // its address
    u64 neighbours{};     // neighbouring pages released for those
};
WatchStats GetWatchStats();
// vbvol (11-20, diagnostic): which threads take the write-watch faults in the NCE exception handler, and what came of them. Two fast-fail 0xd
// (INVALID_SET_OF_CONTEXT) crashes came while vbmir's protected pages faulted 250-2,600 times a second: a guest stack (NCE guest code) or a host
// thread's own stack, handled (CONTINUE_EXECUTION) or passed on (too little stack room, or not the layer's page). Atomics only (exception handler)
void NoteWatchFault(bool host_stack, bool room, bool handled, u64 address, u64 pc, u64 sp);
struct WatchFaultKinds {
    u64 guest_handled{}, guest_passed{}, host_handled{}, host_passed{}, no_room{};
    u32 nfirst{};
    struct First { u32 tid; bool host_stack, room, handled; u64 address, pc, sp; } first[8]{};
};
WatchFaultKinds GetWatchFaultKinds();
// NXN1X itself is about to write guest memory at [address, address + size) (Memory::WriteBlock / ZeroBlock / CopyBlock: HLE services,
// file reads into game buffers). These writes go through the backing view, not the game's addresses, so they never fault: the host layer
// is told first (nvn_dirty prep_write, as the native host does before its own read() into guest memory), which unprotects and stamps the
// pages. Without it a copied pool's GPU copy and the uploaded textures keep the old bytes
void BeforeGuestWrite(u64 address, u64 size);
// F2 provenance only for scalar/exclusive host stores; does not change protection.
void ObserveGuestWrite(u64 address, u64 size);
// s23: the game's memory (its process's page table), from the loader; kept only on NVN boots. The layer imports memory pools for the GPU through
// NXN1X's backing alias of the pool's pages (what the page table points to), not at the game's addresses: the driver locks the range it imports,
// and NXN1X must still be able to unmap and split the game's views when the game remaps its memory (Xenoblade 3 froze leaving its title)
void SetGuestMemory(Core::Memory::Memory& memory);
// s28: NXN1X's backing memory (DeviceMemory), from the loader; kept only on NVN boots. A pool range whose pages are runs of the backing that do not follow
// each other (it crosses two of the game's views) is imported through one new view of those runs (HostMemory::MapBackingRuns) instead of copied
void SetBackingMemory(Common::HostMemory& backing);
// r63 realtime (loader/nso.cpp, Donkey Kong Bananza): the guest address of the state page its stubs read; the host's frame step state (hle_nvn.c dm_fps_state)
// points there, so rt_step_tick writes the measured frame time at each present. NVN boots only
void SetRealtimeState(u64 address);

// ---- Step 6, output (NVN_HOST_BRIDGE.md 4.5): the layer's presents go to NXN1X's compositor instead of a window of its own ----
// A present of the game's window texture: its VkImage (as a number) in GENERAL layout with the frame's work already on the graphics queue, its
// VkFormat and size, the window crop (x, y, w, h; all 0 = the whole texture), whether the game draws lower-left origin (stored bottom-up), and how
// many 60 Hz vsyncs the game wants per frame. ticket names the present: OutputDone(ticket) gives the texture back to the game
struct OutputFrame {
    u64 ticket{};
    u64 image{};
    u32 vk_format{};
    u32 width{};
    u32 height{};
    s32 crop[4]{};
    bool scaled_2x{};
    bool flip_v{};
    u32 interval{1};
};
// The gate went on (GateOn): presents come here from now on, in present order
bool OutputActive();
// HardwareComposer::ComposeLocked, once a vsync (vsync = its frame number): the next present whose turn has come, after the one on screen has
// stayed its interval. Presents without an image are given back on the way. The frame is in flight until OutputDone
std::optional<OutputFrame> OutputTake(u64 vsync);
// The compositor is done with a frame it took: its image was copied and the copy finished on the GPU, or the frame was not drawn at all.
// The window texture goes back to the game (the host's acquire waits for it), in present order. Any thread
void OutputDone(u64 ticket);
// NVN2 output (Switch 2 NVN boots): a Switch 2 game makes no nvnflinger layer (its NVN2 driver presented through nx2:vi2 229, which the layer
// replaces), so HardwareComposer never takes the layer's presents. The nx2:vi2 session sets this when it opens on an NVN boot: each present
// with an image is then handed to it at once (nx2_gsv.h Nx2ComposeNvnFrame, as nx2:vi2 Composite hands a present to the GPU), and is in flight
// until OutputDone as before. Presents before it is set are given back at once (not shown)
void SetNx2Compose(std::function<void(const OutputFrame&)> compose);
// NVN2 native images (Switch 2 NVN boots): the nx2:gsv session sets this when it opens; the host's nvnTextureInitializeFromNativeImage asks it for a handle's picture
// (hle_nvn.c dm_native_image). True and the picture's CPU memory, size, width, height, row pitch and layout (gob_h: log2 of the block height in GOBs, -1 pitch-linear)
struct NativeImage { u8* mem; u64 size; u32 width, height, pitch; int gob_h; };
void SetNx2NativeImages(std::function<bool(u32 handle, NativeImage& out)> lookup);

} // namespace Core::NvnLayer
