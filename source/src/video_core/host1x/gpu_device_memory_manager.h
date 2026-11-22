// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/device_memory_manager.h"

namespace VideoCore {
class RasterizerInterface;
}

namespace Tegra {

struct MaxwellDeviceMethods;

struct MaxwellDeviceTraits {
    // 64 GiB: Switch 2 titles map most of their memory twice plus GPU-only regions (16 GiB ran
    // out in Lichii's tests). Switch 1 titles stay below 16 GiB; the tables behind it are lazy.
    static constexpr size_t device_virtual_bits = 36;
    using DeviceInterface = typename VideoCore::RasterizerInterface;
    using DeviceMethods = MaxwellDeviceMethods;
};

using MaxwellDeviceMemoryManager = Core::DeviceMemoryManager<MaxwellDeviceTraits>;

} // namespace Tegra