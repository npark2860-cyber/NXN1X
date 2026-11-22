// SPDX-FileCopyrightText: Copyright 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/math_util.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/hle/service/nvnflinger/buffer_transform_flags.h"
#include "core/hle/service/nvnflinger/pixel_format.h"
#include "core/hle/service/nvnflinger/ui/fence.h"

namespace Service::Nvnflinger {

// hwc_layer_t::blending values
enum class LayerBlending : u32 {
    // No blending
    None = 0x100,

    // ONE / ONE_MINUS_SRC_ALPHA
    Premultiplied = 0x105,

    // SRC_ALPHA / ONE_MINUS_SRC_ALPHA
    Coverage = 0x405,
};

struct HwcLayer {
    u32 buffer_handle;
    u32 offset;
    android::PixelFormat format;
    u32 width;
    u32 height;
    u32 stride;
    s32 z_index;
    LayerBlending blending;
    android::BufferTransformFlags transform;
    Common::Rectangle<int> crop_rect;
    android::Fence acquire_fence;
    // NVN path (NXN1X NVN_HOST_BRIDGE.md 4.5): a present of the host NVN layer (core/nvn/nvn_output.cpp) instead of an nvmap
    // buffer: its VkImage (as a number, 0 = an nvmap buffer as usual), VkFormat and present ticket
    u64 nvn_image{};
    u32 nvn_format{};
    u64 nvn_ticket{};
    bool nvn_scaled_2x{};
};

} // namespace Service::Nvnflinger
