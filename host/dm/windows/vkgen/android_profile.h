// The Android port's on-screen overlay profile (gfx_vk_present.c): on Windows only the menu button, which is not drawn with a mouse and keyboard
#pragma once
#define DM_AOV_N 1
#define DM_AOV_TABLET {OV_MENU, 0.5f, 0, 0.05f, 0.08f, 0.08f, 0, "", "menu"}
#define DM_AOV_PHONE DM_AOV_TABLET
