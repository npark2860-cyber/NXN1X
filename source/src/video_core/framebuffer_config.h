// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"
#include "common/math_util.h"
#include "core/hle/service/nvnflinger/buffer_transform_flags.h"
#include "core/hle/service/nvnflinger/pixel_format.h"
#include "core/hle/service/nvnflinger/ui/fence.h"

namespace Tegra {

enum class BlendMode {
    Opaque,
    Premultiplied,
    Coverage,
};

enum class ColorSpace {
    RGB709, // Presented as sRGB.
    RGB2020_PQ,
    RGB709_Linear,
};

/**
 * Struct describing framebuffer configuration
 */
struct FramebufferConfig {
    DAddr address{};
    u32 offset{};
    u32 width{};
    u32 height{};
    u32 stride{};
    Service::android::PixelFormat pixel_format{};
    Service::android::BufferTransformFlags transform_flags{};
    Common::Rectangle<int> crop_rect{};
    BlendMode blending{};
    // NVN path (NXN1X NVN_HOST_BRIDGE.md 4.5): a present of the host NVN layer instead of guest memory: its VkImage (as a number;
    // 0 = read address as usual), VkFormat and present ticket. RasterizerVulkan::AccelerateDisplay copies it (nvn_display.cpp)
    u64 nvn_image{};
    u32 nvn_format{};
    u64 nvn_ticket{};
    bool nvn_scaled_2x{};
    bool native_tiling{};
    ColorSpace color_space{ColorSpace::RGB709};
    u16 linear_unit_luminance{}; // Nits per linear 1.0; typically 80 for scRGB (observed in MKW's SDK).
};

Common::Rectangle<f32> NormalizeCrop(const FramebufferConfig& framebuffer, u32 texture_width,
                                     u32 texture_height);

} // namespace Tegra
