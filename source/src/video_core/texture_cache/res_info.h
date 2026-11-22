// SPDX-License-Identifier: GPL-3.0-or-later
// resoverlay (11-21, NPARK): the emulated GPU path's resolution for the performance overlay: the largest render target of the last frame as the game made it (the
// Maxwell surface clip) and as drawn (scaled by the resolution setting when the texture cache rescales). Written by TextureCache::UpdateRenderTargets / TickFrame
#pragma once
#include <atomic>
#include "common/common_types.h"
namespace VideoCommon::ResInfo {
inline std::atomic<u32> cur_w{0}, cur_h{0}, cur_rw{0}, cur_rh{0};
inline std::atomic<u32> last_w{0}, last_h{0}, last_rw{0}, last_rh{0};
// resoverlay2: the frame the game hands to the display (its framebuffer size, both paths: present/layer.cpp)
inline std::atomic<u32> out_w{0}, out_h{0};
} // namespace VideoCommon::ResInfo
