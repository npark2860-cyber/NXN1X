// SPDX-License-Identifier: GPL-3.0-or-later
// fgtoggle (11-21, NPARK: F10 turns frame generation on and off live, as SnapRyu does; the overlay shows its state): what the presenter's frame generation
// did over its last two-second window (nvn_streamline.cpp AfterPresent), read by the performance overlay without a lock
#pragma once
#include <atomic>
#include "common/common_types.h"

namespace VideoCommon::FgStatus {
enum Mode : u32 { Off = 0, Waiting = 1, Optical = 2, Dlssg = 3 };   // Waiting: on, nothing generated in the window
inline std::atomic<u32> mode{Off}, multiplier{1};
inline std::atomic<u32> original_fps10{0}, output_fps10{0};   // frames a second x 10 over the window
}  // namespace VideoCommon::FgStatus
