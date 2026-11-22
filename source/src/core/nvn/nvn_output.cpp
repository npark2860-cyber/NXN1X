// SPDX-License-Identifier: GPL-2.0-or-later
// NVN path, step 6 (NXN1X docs NVN_HOST_BRIDGE.md 4.5): the host layer's presents, for NXN1X's compositor.
// The host's mtl_present (dm/windows/gfx_vk_present.c) hands each present here instead of to a window and swapchain of its own. Every vsync,
// HardwareComposer::ComposeLocked takes the next present whose turn has come and composes it as the game's layer; the renderer copies its
// image (video_core/renderer_vulkan/nvn_display.cpp) and gives it back once that copy has finished on the GPU. The host's acquire of a window
// texture waits for that (mtl_present_wait), as it waited for its own present thread's blit.
// Pacing: a present stays on screen for its interval (60 Hz vsyncs, from the host's frame period); the host also paces at acquire, as before.
// Presents are given back strictly in present order (the host keeps "presents up to ticket n are done").
#include <deque>
#include <functional>   // NVN2 output
#include <mutex>
#include <vector>

#include "common/logging.h"
#include "core/nvn/nvn_layer.h"

#if defined(NXN1X_NVN_HOST)
#include <atomic>
#include <cmath>
#include <cstdlib>

#include "nvnh_api.h"
#include "common/settings.h"   // NVN2 output
#include "core/nvn/sv1.h"
#endif

namespace Core::NvnLayer {

#if defined(NXN1X_NVN_HOST)
namespace {

struct Flight {
    u64 ticket;
    bool done;
};

std::mutex g_mu;
std::deque<OutputFrame> g_pending; // presented, not taken yet, in present order
std::deque<Flight> g_flight;       // taken (or without an image), not given back yet, in present order
std::atomic<bool> g_active{false};
bool g_on_screen{};                // a frame was taken (the composer has something on screen)
u64 g_taken_vsync{};               // the vsync it was taken at
u32 g_on_screen_interval{1};
[[maybe_unused]] u64 g_presents{}, g_empty{};
u64 g_taken{};
bool g_nx2{};                                         // NVN2 output: a Switch 2 NVN boot (OutputAttach)
std::function<void(const OutputFrame&)> g_nx2_compose; // set by the nx2:vi2 session (SetNx2Compose), under g_mu
u64 g_nx2_dropped{};                                  // presents with an image given back before it was set

// Pops the given-back prefix of g_flight into out (under g_mu)
void CollectDone(std::vector<u64>& out) {
    while (!g_flight.empty() && g_flight.front().done) {
        out.push_back(g_flight.front().ticket);
        g_flight.pop_front();
    }
}

void GiveBack(const std::vector<u64>& tickets) {
    for (const u64 ticket : tickets) {
        nvnh_present_done(ticket);
    }
}

// The host's present (the game's thread, inside nvnQueuePresentTexture): queue it. A present without an image when nothing is waiting
// before it is given back at once, as the host did without a window
// The OutputFrame of a host present (the fields as Push queues them for HardwareComposer)
OutputFrame MakeFrame(const nvnh_frame* f) {
    OutputFrame frame{};
    frame.ticket = f->n;
    frame.image = f->image;
    frame.vk_format = f->vk_format;
    frame.width = f->width;
    frame.height = f->height;
    frame.scaled_2x = f->scaled_2x != 0;
    if (f->has_crop) {
        for (int i = 0; i < 4; ++i) {
            frame.crop[i] = f->crop[i];
        }
    }
    frame.flip_v = f->bottom_up != 0;
    const double vsyncs = f->period > 0.0 ? std::round(f->period * 60.0) : 1.0;
    frame.interval = vsyncs < 1.0 ? 1U : vsyncs > 4.0 ? 4U : static_cast<u32>(vsyncs);
    return frame;
}

// NVN2 output: no game layer to take the present, so it goes to the compositor here, at once (the host paces at acquire, as for Switch 1)
void PushNx2(const nvnh_frame* f) {
    std::vector<u64> done;
    std::function<void(const OutputFrame&)> compose;
    const OutputFrame frame = MakeFrame(f);
    bool first = false;
    bool dropped_first = false;
    {
        std::scoped_lock lock{g_mu};
        ++g_presents;
        Core::Sv1::NotePresent();   // sv1: the watchdog's present count
        if (f->image != 0 && g_nx2_compose) {
            compose = g_nx2_compose;
            g_flight.push_back({f->n, false});   // in flight until the renderer's copy is done (OutputDone)
            first = ++g_taken == 1;
        } else {
            if (f->image == 0) {
                ++g_empty;
            } else {
                dropped_first = ++g_nx2_dropped == 1;
            }
            g_flight.push_back({f->n, true});
        }
        CollectDone(done);
    }
    GiveBack(done);
    if (dropped_first) {
        LOG_WARNING(Core, "NVN2 output: present {} came before the nx2:vi2 session; given back without being shown", f->n);
    }
    if (compose) {
        if (first) {
            LOG_INFO(Core, "NVN2 output: first frame composed ({}x{}, VkFormat {}, present {})", frame.width, frame.height,
                     frame.vk_format, frame.ticket);
        }
        compose(frame);
    }
}

void Push(const nvnh_frame* f) {
    if (g_nx2) {
        PushNx2(f);
        return;
    }
    std::vector<u64> done;
    {
        std::scoped_lock lock{g_mu};
        ++g_presents;
        Core::Sv1::NotePresent();   // sv1: the watchdog's present count
        if (f->image == 0 && g_pending.empty()) {
            ++g_empty;
            g_flight.push_back({f->n, true});
            CollectDone(done);
        } else {
            OutputFrame frame{};
            frame.ticket = f->n;
            frame.image = f->image;
            frame.vk_format = f->vk_format;
            frame.width = f->width;
            frame.height = f->height;
            frame.scaled_2x = f->scaled_2x != 0;
            if (f->has_crop) {
                for (int i = 0; i < 4; ++i) {
                    frame.crop[i] = f->crop[i];
                }
            }
            frame.flip_v = f->bottom_up != 0;
            const double vsyncs = f->period > 0.0 ? std::round(f->period * 60.0) : 1.0;
            frame.interval = vsyncs < 1.0 ? 1U : vsyncs > 4.0 ? 4U : static_cast<u32>(vsyncs);
            g_pending.push_back(frame);
        }
    }
    GiveBack(done);
}

} // namespace

void OutputAttach() {
    if (const char* off = std::getenv("NXN1X_NVN_OUTPUT"); off != nullptr && off[0] == '0') {
        LOG_INFO(Core, "NVN output: off (NXN1X_NVN_OUTPUT=0), the layer's presents are not shown");
        return;
    }
    g_nx2 = Settings::CurrentProgramIsSwitch2();   // before the first present can come (set_output below)
    nvnh_set_output(&Push);
    g_active.store(true, std::memory_order_release);
    LOG_INFO(Core, "NVN output: the layer's presents go to NXN1X's compositor (NVN_HOST_BRIDGE.md 4.5)");
    if (g_nx2) {
        LOG_INFO(Core, "NVN2 output: host frames go to the compositor without a game layer (nx2:vi2 session, Nx2ComposeNvnFrame)");
    }
}

void SetNx2Compose(std::function<void(const OutputFrame&)> compose) {
    std::scoped_lock lock{g_mu};
    if (g_nx2) {
        g_nx2_compose = std::move(compose);
    }
}

namespace {
std::mutex g_ni_mu;
std::function<bool(u32, NativeImage&)> g_ni_lookup; // SetNx2NativeImages, under g_ni_mu
int NativeImageOf(uint32_t handle, uint8_t** mem, uint64_t* size, uint32_t* w, uint32_t* h, uint32_t* stride, int* gob_h) {
    std::function<bool(u32, NativeImage&)> f;
    {
        std::scoped_lock lock{g_ni_mu};
        f = g_ni_lookup;
    }
    NativeImage ni{};
    ni.gob_h = -1;
    if (!f || !f(handle, ni)) {
        return 0;
    }
    *mem = ni.mem;
    *size = ni.size;
    *w = ni.width;
    *h = ni.height;
    *stride = ni.pitch;
    *gob_h = ni.gob_h;
    return 1;
}
} // namespace

void SetNx2NativeImages(std::function<bool(u32 handle, NativeImage& out)> lookup) {
    {
        std::scoped_lock lock{g_mu};
        if (!g_nx2) {
            return;
        }
    }
    std::scoped_lock lock{g_ni_mu};
    g_ni_lookup = std::move(lookup);
    nvnh_set_native_image(g_ni_lookup ? &NativeImageOf : nullptr);
}

bool OutputActive() {
    return g_active.load(std::memory_order_acquire);
}

std::optional<OutputFrame> OutputTake(u64 vsync) {
    if (!OutputActive()) {
        return std::nullopt;
    }
    // The write watch's unusual cases (OnWatchedWrite): reported when they moved, every ~10 s
    if (vsync % 600 == 0) {
        static WatchStats reported{};
        {   // vbvol (diagnostic): the write faults the NCE handler saw, by thread kind and outcome, and the first 8 in detail (once)
            static WatchFaultKinds last{};
            static bool firsts_logged = false;
            const WatchFaultKinds k = GetWatchFaultKinds();
            LOG_INFO(Core, "vbvol: write faults in the NCE handler since the last line: guest stack handled {} passed {}, host stack handled {} passed {}, too little stack room {}",
                     k.guest_handled - last.guest_handled, k.guest_passed - last.guest_passed, k.host_handled - last.host_handled,
                     k.host_passed - last.host_passed, k.no_room - last.no_room);
            if (!firsts_logged && k.nfirst == 8) {
                firsts_logged = true;
                for (u32 i = 0; i < k.nfirst; ++i) {
                    const auto& f = k.first[i];
                    LOG_INFO(Core, "vbvol: fault {}: thread {} {} stack, address {:#x}, pc {:#x}, sp {:#x}, {}", i, f.tid, f.host_stack ? "host" : "guest",
                             f.address, f.pc, f.sp, !f.room ? "too little stack room: passed on" : f.handled ? "CONTINUE_EXECUTION" : "passed on");
                }
            }
            last = k;
        }
        const WatchStats w = GetWatchStats();
        if (w.foreign != reported.foreign || w.repeats != reported.repeats) {
            reported = w;
            LOG_WARNING(Core, "NVN write watch: {} faults came back at once on the same page (longest run {} at {:#x}; {} neighbouring "
                              "pages released), {} were not the layer's (still read-only after on_fault; first at {:#x})",
                        w.repeats, w.repeat_worst, w.repeat_addr, w.neighbours, w.foreign, w.foreign_first);
        }
    }
    std::vector<u64> done;
    std::optional<OutputFrame> taken;
    {
        std::scoped_lock lock{g_mu};
        // Presents without an image (nothing drawn yet) go back on the way, in their place in the order
        while (!g_pending.empty() && g_pending.front().image == 0) {
            ++g_empty;
            g_flight.push_back({g_pending.front().ticket, true});
            g_pending.pop_front();
        }
        const bool turn = !g_on_screen || vsync - g_taken_vsync >= g_on_screen_interval;
        if (turn && !g_pending.empty()) {
            taken = g_pending.front();
            g_pending.pop_front();
            g_flight.push_back({taken->ticket, false});
            g_on_screen = true;
            g_taken_vsync = vsync;
            g_on_screen_interval = taken->interval;
            if (++g_taken == 1) {
                LOG_INFO(Core, "NVN output: first frame composed ({}x{}, VkFormat {}, interval {}, present {})", taken->width,
                         taken->height, taken->vk_format, taken->interval, taken->ticket);
            }
        }
        CollectDone(done);
    }
    GiveBack(done);
    return taken;
}

void OutputDone(u64 ticket) {
    std::vector<u64> done;
    {
        std::scoped_lock lock{g_mu};
        for (Flight& f : g_flight) {
            if (f.ticket == ticket) {
                f.done = true;
                break;
            }
        }
        CollectDone(done);
    }
    GiveBack(done);
}

#else

bool OutputActive() {
    return false;
}

std::optional<OutputFrame> OutputTake(u64 vsync) {
    (void)vsync;
    return std::nullopt;
}

void OutputDone(u64 ticket) {
    (void)ticket;
}

void SetNx2Compose(std::function<void(const OutputFrame&)> compose) {
    (void)compose;
}

void SetNx2NativeImages(std::function<bool(u32 handle, NativeImage& out)> lookup) {
    (void)lookup;
}

#endif

} // namespace Core::NvnLayer
