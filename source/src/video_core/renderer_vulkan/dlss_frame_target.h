#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace Vulkan::NvnStreamline {
struct OpticalCadence {
    std::uint64_t ticket{};
    std::int64_t previous_ns{}, balance{};
    unsigned target{};

    // One original pays for one output slot. Keep the fractional remainder
    // in integer units (nanoseconds times Hz) between adjacent originals.
    void RetryOne(unsigned missed) {
        // Carry at most one missed slot into the next pair; discard older debt.
        if (missed) balance = std::min<std::int64_t>(balance + 1000000000, 1999999999);
    }

    unsigned Advance(std::uint64_t next, std::int64_t now, unsigned hz) {
        if (next && next == ticket && hz == target) return 0;
        constexpr std::int64_t unit = 1000000000;
        const auto elapsed = now - previous_ns;
        const bool consecutive = next && ticket && next == ticket + 1 && hz && hz == target &&
                                 elapsed > 0 && elapsed <= unit / 4;
        previous_ns = now;
        ticket = next;
        target = hz;
        if (!consecutive) { balance = unit / 2; return 0; }
        balance = std::clamp<std::int64_t>(balance + elapsed * hz - unit, -unit, 4 * unit - 1);
        const unsigned extra = static_cast<unsigned>(std::max<std::int64_t>(0, balance) / unit);
        balance -= extra * unit;
        return extra;
    }
};

struct OriginalCadence {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start{}, last{};
    std::uint64_t ticket{};
    unsigned intervals{};
    double fps{};

    void Observe(std::uint64_t next, Clock::time_point now) {
        if (!next || next == ticket) return;
        if (!ticket || next != ticket + 1 || now - last > std::chrono::milliseconds(250)) {
            start = now;
            intervals = 0;
            fps = 0;
        } else {
            ++intervals;
            const double seconds = std::chrono::duration<double>(now - start).count();
            if (seconds >= 1.0 && intervals >= 4) {
                const double measured = intervals / seconds;
                fps = fps > 0 && std::abs(measured - fps) < fps * 0.1
                          ? (fps + measured) * 0.5 : measured;
                start = now;
                intervals = 0;
            }
        }
        ticket = next;
        last = now;
    }

    unsigned Multiplier(unsigned target, unsigned max_generated) const {
        if (!(fps > 0) || !max_generated) return 1;
        // Allow the normal timing variation around nominal rates such as 29.97/30.
        const double ratio = (target ? target : 60) / fps;
        return 1 + static_cast<unsigned>(std::clamp(std::floor(ratio * 1.01) - 1.0,
                                                    0.0, double(max_generated)));
    }
};
}
