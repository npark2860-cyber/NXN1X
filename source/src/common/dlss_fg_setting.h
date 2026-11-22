#pragma once
#include "common/settings_setting.h"
namespace Settings {
extern SwitchableSetting<u32, true> dlss_target_fps;
extern SwitchableSetting<bool> dlss_frame_generation;
extern SwitchableSetting<bool> dlss_upscale;   // dlsstoggle (11-21): NVIDIA DLSS Super Resolution in place of a game's own upscale (Ocarina of Time), live
extern SwitchableSetting<u32, true> dlss_fg_source;   // fgsource (11-21, NPARK): 0 Auto (DLSS-G, optical interpolation where it does not generate), 1 DLSS-G only, 2 optical only; live
}
