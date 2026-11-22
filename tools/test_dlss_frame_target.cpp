#include "../source/src/video_core/renderer_vulkan/dlss_frame_target.h"
#include <cassert>
#include <cstdio>

int main() {
    using Vulkan::NvnStreamline::OriginalCadence;
    using namespace std::chrono;
    for (unsigned guest_hz : {19200000u, 31250000u}) {
        for (unsigned native : {30u, 60u, 90u, 120u}) {
            OriginalCadence cadence{};
            for (unsigned i = 0; i <= native * 3; ++i) {
                const std::uint64_t ticks = std::uint64_t(i) * guest_hz / native;
                const auto elapsed = nanoseconds(ticks * 1000000000ull / guest_hz);
                cadence.Observe(i+1, OriginalCadence::Clock::time_point{} + elapsed);
            }
            for (unsigned target : {30u, 60u, 90u, 120u})
                assert(cadence.Multiplier(target,5) == std::max(1u,target/native));
            assert(cadence.Multiplier(120,0)==1);
            assert(cadence.Multiplier(120,1)<=2);
            cadence.Observe(cadence.ticket+1,cadence.last+seconds(1));
            assert(cadence.Multiplier(120,5)==1);
        }
    }
    using Vulkan::NvnStreamline::OpticalCadence;
    for (const unsigned native : {20u, 29u, 30u, 43u, 60u, 90u}) {
        OpticalCadence plan;
        unsigned extras = 0;
        for (unsigned i = 0; i <= native; ++i) {
            auto now = 1000000000LL * i / native;
            extras += plan.Advance(i+1, now, 60);
            assert(plan.Advance(i+1, now, 60) == 0);
        }
        assert(extras == (native < 60 ? 60-native : 0));
        std::printf("60 FPS schedule: %u originals + %u generated per second\n", native, extras);
    }
    OpticalCadence plan;
    plan.Advance(1,0,60);
    // Switch from 43 to 29 without a one-second measurement window.
    unsigned total = 0;
    for (unsigned i = 1; i <= 43; ++i) total += plan.Advance(i+1,1000000000LL*i/43,60);
    for (unsigned i = 1; i <= 29; ++i) total += plan.Advance(44+i,1000000000LL+1000000000LL*i/29,60);
    assert(total == 17+31);
    assert(plan.Advance(74,3000000000LL,60) == 0);
    assert(plan.Advance(75,3033333333LL,120) == 0);
    assert(plan.Advance(76,3066666667LL,120) == 3);
    plan.Advance(77,3100000000LL,60);
    assert(plan.Advance(78,3133333333LL,60) == 1);
    plan.RetryOne(1);
    assert(plan.Advance(79,3166666667LL,60) == 2);
    assert(plan.Advance(82,3200000000LL,60) == 0);
    puts("PASS real-time cadence for 19.2 and 31.25 MHz, 30/60/90/120 FPS targets, SDK cap and pause");
}
