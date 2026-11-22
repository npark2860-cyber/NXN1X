// Windows port: the host front end (dm/host/front.h) with a Win32 window for the Vulkan backend (gfx_vk_present.c), keyboard and XInput controllers.
// The guest runs on the calling thread, as without a window (its 8 MiB stack, build.py); the window lives on a thread of its own that runs its message loop.
// DM_HEADLESS=1: no window, no input (stage 2's behaviour; gfx_vk_present.c then only counts frames). The window starts at DM_WIN_W x DM_WIN_H (default 1280 x 720)
// client pixels; F11 or Alt+Enter toggles borderless fullscreen on its monitor; closing it ends the process.
// Keyboard, as the Mac front end's (front_mac.m key_bit): J K U I = A B X Y, F G = press left / right stick, Q E Z C = L R ZL ZR, Enter Tab = + -, arrows = D-pad,
// Shift + arrows = right stick, WASD = left stick, Space = shake. XInput controllers 1-4 are players 1-4, buttons by position (Xbox A, the bottom one, is the Switch's B).
// Vibration goes to the XInput motors as haptics_apple.m drives a DualShock 4: the strong (left) motor follows the low band, the weak (right) one the high band.
// DM_RUMBLE=0 turns it off; DM_RUMBLE_GAIN (0 to 3) and DM_RUMBLE_MIN (0 to 0.9) as on the Mac.
// r43 (11-05): with SDL3 (DM_SDL3_DLL, else libSDL3.dll / SDL3.dll on the search path) controllers come from SDL instead, as in the TotK recomp's runtime/input.c:
// buttons by position as above, rumble through SDL_RumbleGamepad, and a gyro + accelerometer feeds the six-axis sensor (dm_motion_set_hid). DM_NO_SDL=1: XInput only.
#include "dm.h"
#include "front.h"
#include "dm_input.h"
#include "hle_haptics.h"
#include <windows.h>
#include <xinput.h>
#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static HWND g_hwnd;
static HANDLE g_ready;
static int g_headless = -1;

static int headless(void) { if (g_headless < 0) { const char *e = getenv("DM_HEADLESS"); g_headless = e && atoi(e); } return g_headless; }

// ---- keyboard ----
static uint64_t g_btn; static int g_w, g_a, g_s, g_d, g_rl, g_ru, g_rr, g_rd;
static void push(void) { dm_input_set(g_btn, (g_d - g_a) * DM_STICK_MAX, (g_w - g_s) * DM_STICK_MAX, (g_rr - g_rl) * DM_STICK_MAX, (g_ru - g_rd) * DM_STICK_MAX); }
static uint64_t key_mask(WPARAM vk) {
    switch (vk) {
        case 'J': return DM_BTN_A; case 'K': return DM_BTN_B; case 'U': return DM_BTN_X; case 'I': return DM_BTN_Y;
        case 'F': return DM_BTN_LS; case 'G': return DM_BTN_RS;
        case 'Q': return DM_BTN_L; case 'E': return DM_BTN_R; case 'Z': return DM_BTN_ZL; case 'C': return DM_BTN_ZR;
        case VK_RETURN: return DM_BTN_PLUS; case VK_TAB: return DM_BTN_MINUS;
        case VK_LEFT: return DM_BTN_LEFT; case VK_UP: return DM_BTN_UP; case VK_RIGHT: return DM_BTN_RIGHT; case VK_DOWN: return DM_BTN_DOWN;
        case VK_SPACE: return DM_BTN_SHAKE;
    }
    return 0;
}
static void on_key(WPARAM vk, int down) {
    switch (vk) { case 'W': g_w = down; break; case 'A': g_a = down; break; case 'S': g_s = down; break; case 'D': g_d = down; break; }
    // an arrow pressed with Shift held is the right stick until it is released (front_mac.m: the keyboard had no camera otherwise)
    int *rs = vk == VK_LEFT ? &g_rl : vk == VK_UP ? &g_ru : vk == VK_RIGHT ? &g_rr : vk == VK_DOWN ? &g_rd : NULL;
    if (rs && (down ? (GetKeyState(VK_SHIFT) & 0x8000) != 0 : *rs)) { *rs = down; push(); return; }
    uint64_t m = key_mask(vk);
    g_btn = down ? (g_btn | m) : (g_btn & ~m);
    push();
}
static void keys_release(void) { g_btn = 0; g_w = g_a = g_s = g_d = g_rl = g_ru = g_rr = g_rd = 0; push(); }

// ---- fullscreen (borderless, on the window's monitor) ----
static WINDOWPLACEMENT g_place = {sizeof g_place}; static int g_full;
static void fullscreen_toggle(HWND h) {
    LONG st = GetWindowLongW(h, GWL_STYLE);
    if (!g_full) {
        MONITORINFO mi = {sizeof mi};
        if (!GetWindowPlacement(h, &g_place) || !GetMonitorInfoW(MonitorFromWindow(h, MONITOR_DEFAULTTOPRIMARY), &mi)) return;
        SetWindowLongW(h, GWL_STYLE, st & ~WS_OVERLAPPEDWINDOW);
        SetWindowPos(h, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left, mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        g_full = 1;
    } else {
        SetWindowLongW(h, GWL_STYLE, st | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(h, &g_place);
        SetWindowPos(h, NULL, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        g_full = 0;
    }
}

// ---- vibration: the haptics sink only stores each player's two motor speeds (it runs under the haptics lock on a game thread: no hardware calls there);
// the pad thread sends a changed pair with XInputSetState on its next 4 ms pass ----
typedef DWORD (WINAPI *xset_t)(DWORD, XINPUT_VIBRATION *);
static xset_t g_xset;
static volatile LONG g_rumble[4];   // per XInput slot: strong motor speed << 16 | weak motor speed, 0-65535 each
static float g_gain = 1, g_rumble_min;
static float rumble_level(float x) {   // haptics_apple.m controller_level
    x = isfinite(x) ? fmaxf(0, fminf(1, x)) : 0;
    return x <= g_rumble_min ? 0 : (x - g_rumble_min) / (1 - g_rumble_min);
}
static void rumble_sink(dm_haptic_state s) {
    if (s.player < 0 || s.player >= 4) return;
    const dm_vibration *l = &s.motor[0], *r = &s.motor[1];
    LONG strong = lrintf(rumble_level((0.85f * l->amplitude_low + 0.15f * r->amplitude_low) * g_gain) * 65535);
    LONG weak = lrintf(rumble_level((0.15f * l->amplitude_high + 0.85f * r->amplitude_high) * g_gain) * 65535);
    InterlockedExchange(&g_rumble[s.player], (LONG)((ULONG)strong << 16 | (ULONG)weak));
}
static void rumble_send(int p, LONG v) { XINPUT_VIBRATION x = {(WORD)((ULONG)v >> 16), (WORD)((ULONG)v & 0xffff)}; g_xset((DWORD)p, &x); }
// r52: on exit the thread that reads the pads stops the motors itself and signals; the exit path waits for it a short while (a controller keeps spinning after
// the process is gone). A stop sent from the window thread just before TerminateProcess was often lost: SDL's Switch driver holds back a write that comes
// within 30 ms of the last one until a later SDL_UpdateGamepads (SDL_hidapi_switch.c RUMBLE_WRITE_FREQUENCY_MS), and HIDAPI writes go through SDL's rumble thread
static volatile LONG g_pads_quit;
static HANDLE g_pads_done;   // set by the pad thread once the motors are stopped, or at once when there is no pad thread
void front_pads_quit(void) {   // the window's close and the console's control events (hle_lifecycle.c); more than one call is fine
    if (!g_pads_done) return;
    InterlockedExchange(&g_pads_quit, 1);
    if (WaitForSingleObject(g_pads_done, 500) == WAIT_OBJECT_0) return;
    LOG("[RED] [front] the pad thread did not stop the motors in 0.5 s");
    if (g_xset) for (int p = 0; p < 4; p++) rumble_send(p, 0);
}
static void rumble_setup(const char *what) {
    const char *e = getenv("DM_RUMBLE_GAIN"); if (e) { float v = strtof(e, NULL); if (isfinite(v)) g_gain = fmaxf(0, fminf(3, v)); }
    e = getenv("DM_RUMBLE_MIN"); if (e) { float v = strtof(e, NULL); if (isfinite(v)) g_rumble_min = fmaxf(0, fminf(0.9f, v)); }
    e = getenv("DM_RUMBLE"); dm_haptics_init(rumble_sink, !e || atoi(e) != 0);
    LOG("[front] %s vibration %s (gain %.2f, min %.2f)", what, !e || atoi(e) ? "on" : "off (DM_RUMBLE=0)", g_gain, g_rumble_min);
}

// ---- the mouse is the touch screen (as the Mac's game view): left button down / drag, over the game's picture (gfx_vk_present.c dm_picture_rect) ----
__attribute__((weak)) void dm_picture_rect(float r[4]) { r[0] = r[1] = 0; r[2] = r[3] = 1; }   // gfx_vk_present.c's; the whole window in a null-backend build
static int g_touch;
// r86: the performance overlay (perf_overlay_win.c): the left button over it clicks (FPS only) or drags it, the wheel over it sets its opacity, F3 shows / hides it
int perf_ov_mouse_down(int x, int y); int perf_ov_mouse_move(int x, int y); int perf_ov_mouse_up(int x, int y); int perf_ov_wheel(int x, int y, int delta);
void perf_ov_toggle(void); void perf_ov_drag_distance(int d); void perf_ov_mouse_cancel(void);
static int g_perf_press;
float dm_front_scale(void *w) {   // the window's DPI scale (Qt's devicePixelRatio): NXN1X's panel is laid out in logical pixels
    typedef UINT (WINAPI *dpi_t)(HWND); static dpi_t f; static int got;
    if (!got) { got = 1; f = (dpi_t)(void *)GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForWindow"); }
    UINT d = f && w ? f((HWND)w) : 96; return d ? d / 96.0f : 1;
}
static void touch_at(HWND h, LPARAM lp, int down) {
    RECT rc; GetClientRect(h, &rc); float W = (float)(rc.right - rc.left), H = (float)(rc.bottom - rc.top);
    if (!down || W <= 0 || H <= 0) { dm_touch_set(NULL, 0); return; }
    float pr[4]; dm_picture_rect(pr); if (pr[2] <= 0 || pr[3] <= 0) return;
    float x = ((float)(short)LOWORD(lp) / W - pr[0]) / pr[2], y = ((float)(short)HIWORD(lp) / H - pr[1]) / pr[3];
    if (x < 0 || y < 0 || x >= 1 || y >= 1) return;
    dm_touch t = {0, x, y}; dm_touch_set(&t, 1);
}

static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
    case WM_KEYDOWN: case WM_SYSKEYDOWN:
        if (wp == VK_F11 && !(lp & (1 << 30))) { fullscreen_toggle(h); return 0; }
        if (wp == VK_F3) { if (!(lp & (1 << 30))) perf_ov_toggle(); return 0; }   // r86
        if (wp == VK_RETURN && m == WM_SYSKEYDOWN) { if (!(lp & (1 << 30))) fullscreen_toggle(h); return 0; }   // Alt+Enter
        if (wp == VK_F4 && m == WM_SYSKEYDOWN) break;   // Alt+F4: closes (DefWindowProc)
        if (!(lp & (1 << 30))) on_key(wp, 1);   // not auto-repeat
        return 0;
    case WM_KEYUP: case WM_SYSKEYUP: on_key(wp, 0); return 0;
    case WM_SYSCHAR: return 0;   // no beep for Alt+key
    case WM_ACTIVATEAPP: dm_haptics_suspend(DM_HAPTICS_INACTIVE, !wp); break;   // the Mac stops rumble while the app is in the background
    case WM_KILLFOCUS: keys_release(); if (g_touch) { g_touch = 0; dm_touch_set(NULL, 0); } if (g_perf_press) { g_perf_press = 0; perf_ov_mouse_cancel(); } return 0;
    case WM_LBUTTONDOWN: SetCapture(h); perf_ov_drag_distance(GetSystemMetrics(SM_CXDRAG));
        if (perf_ov_mouse_down((short)LOWORD(lp), (short)HIWORD(lp))) { g_perf_press = 1; return 0; }   // r86: the panel's, not a touch
        g_touch = 1; touch_at(h, lp, 1); return 0;
    case WM_MOUSEMOVE: if (g_perf_press) { perf_ov_mouse_move((short)LOWORD(lp), (short)HIWORD(lp)); return 0; } if (g_touch) touch_at(h, lp, 1); return 0;
    case WM_LBUTTONUP: ReleaseCapture(); if (g_perf_press) { g_perf_press = 0; perf_ov_mouse_up((short)LOWORD(lp), (short)HIWORD(lp)); return 0; } g_touch = 0; touch_at(h, lp, 0); return 0;
    case WM_MOUSEWHEEL: { POINT p = {(short)LOWORD(lp), (short)HIWORD(lp)}; ScreenToClient(h, &p); if (perf_ov_wheel(p.x, p.y, (short)HIWORD(wp))) return 0; break; }   // r86 (screen coordinates)
    case WM_ERASEBKGND: return 1;   // Vulkan draws all of it
    case WM_CLOSE: LOG("[front] window closed: exiting"); front_pads_quit(); TerminateProcess(GetCurrentProcess(), 0); return 0;   // the guest and the driver's threads are not unwound
    }
    return DefWindowProcW(h, m, wp, lp);
}

static DWORD WINAPI window_thread(LPVOID arg) {
    (void)arg;
    { typedef BOOL (WINAPI *dpi_t)(HANDLE); dpi_t f = (dpi_t)(void *)GetProcAddress(GetModuleHandleW(L"user32.dll"), "SetProcessDpiAwarenessContext");   // r30 (11-04): real pixels; unaware at
      if (f && !f((HANDLE)(intptr_t)-4)) LOG("[front] per-monitor DPI awareness refused: %lu", (unsigned long)GetLastError()); }   // 125 % a 2560x1440 fullscreen was a 2048x1152 swapchain stretched by Windows
    HINSTANCE hi = GetModuleHandleW(NULL);
    WNDCLASSEXW wc = {sizeof wc}; wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC; wc.lpfnWndProc = wndproc; wc.hInstance = hi; wc.hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW);
    wc.hIcon = LoadIconW(hi, MAKEINTRESOURCEW(1)); wc.lpszClassName = L"dmhost"; wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    RegisterClassExW(&wc);
    const char *ew = getenv("DM_WIN_W"), *eh = getenv("DM_WIN_H"); int cw = ew && atoi(ew) > 0 ? atoi(ew) : 1280, ch = eh && atoi(eh) > 0 ? atoi(eh) : 720;
    RECT r = {0, 0, cw, ch}; AdjustWindowRectEx(&r, WS_OVERLAPPEDWINDOW, FALSE, 0);
    const char *t = getenv("DM_TITLE"); wchar_t title[256]; MultiByteToWideChar(CP_UTF8, 0, t && *t ? t : "dmhost", -1, title, 256); title[255] = 0;
    g_hwnd = CreateWindowExW(0, wc.lpszClassName, title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, NULL, NULL, hi, NULL);
    if (!g_hwnd) LOG("[RED] [front] CreateWindowEx failed: %lu; running without a window", (unsigned long)GetLastError());
    else { ShowWindow(g_hwnd, SW_SHOWNORMAL); UpdateWindow(g_hwnd); SetForegroundWindow(g_hwnd); LOG("[front] window %dx%d", cw, ch); }
    SetEvent(g_ready);
    if (!g_hwnd) return 0;
    MSG msg; while (GetMessageW(&msg, NULL, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
    return 0;
}

// ---- XInput: polled every 4 ms while a controller is connected; an empty slot is asked again once a second (XInputGetState on an empty slot is slow) ----
typedef DWORD (WINAPI *xget_t)(DWORD, XINPUT_STATE *);
static int32_t xaxis(SHORT v) { return v < -DM_STICK_MAX ? -DM_STICK_MAX : v; }
static DWORD WINAPI pad_thread(LPVOID arg) {
    xget_t get = (xget_t)arg; int conn[4] = {0}; DWORD seen[4] = {0}; ULONGLONG probe[4] = {0}; uint32_t players = 0; LONG sent[4] = {0};
    for (;;) {
        if (g_pads_quit) { if (g_xset) for (int p = 0; p < 4; p++) if (conn[p]) rumble_send(p, 0); LOG("[front] exit: vibration stopped"); SetEvent(g_pads_done); return 0; }
        ULONGLONG now = GetTickCount64(); uint32_t mask = 0;
        for (int p = 0; p < 4; p++) {
            if (!conn[p] && now < probe[p]) continue;
            XINPUT_STATE s; DWORD r = get((DWORD)p, &s);
            if (r != ERROR_SUCCESS) {
                if (conn[p]) { dm_pad_set_player(p, 0, 0, 0, 0, 0); if (g_xset) dm_haptics_route(p, 0); LOG("[front] controller %d disconnected", p + 1); }
                conn[p] = 0; probe[p] = now + 1000; sent[p] = 0; continue;
            }
            if (!conn[p]) { conn[p] = 1; seen[p] = ~0u; if (g_xset) dm_haptics_route(p, 1); LOG("[front] XInput controller %d connected (player %d)", p + 1, p + 1); }
            if (g_xset) { LONG v = InterlockedCompareExchange(&g_rumble[p], 0, 0); if (v != sent[p]) { rumble_send(p, v); sent[p] = v; } }   // only a changed pair
            mask |= 1u << p;
            if (s.dwPacketNumber == seen[p]) continue;
            seen[p] = s.dwPacketNumber;
            const XINPUT_GAMEPAD *g = &s.Gamepad; WORD b = g->wButtons; uint64_t m = 0;
            if (b & XINPUT_GAMEPAD_A) m |= DM_BTN_B; if (b & XINPUT_GAMEPAD_B) m |= DM_BTN_A; if (b & XINPUT_GAMEPAD_X) m |= DM_BTN_Y; if (b & XINPUT_GAMEPAD_Y) m |= DM_BTN_X;
            if (b & XINPUT_GAMEPAD_LEFT_SHOULDER) m |= DM_BTN_L; if (b & XINPUT_GAMEPAD_RIGHT_SHOULDER) m |= DM_BTN_R;
            if (g->bLeftTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD) m |= DM_BTN_ZL; if (g->bRightTrigger > XINPUT_GAMEPAD_TRIGGER_THRESHOLD) m |= DM_BTN_ZR;
            if (b & XINPUT_GAMEPAD_START) m |= DM_BTN_PLUS; if (b & XINPUT_GAMEPAD_BACK) m |= DM_BTN_MINUS;
            if (b & XINPUT_GAMEPAD_LEFT_THUMB) m |= DM_BTN_LS; if (b & XINPUT_GAMEPAD_RIGHT_THUMB) m |= DM_BTN_RS;
            if (b & XINPUT_GAMEPAD_DPAD_LEFT) m |= DM_BTN_LEFT; if (b & XINPUT_GAMEPAD_DPAD_UP) m |= DM_BTN_UP; if (b & XINPUT_GAMEPAD_DPAD_RIGHT) m |= DM_BTN_RIGHT; if (b & XINPUT_GAMEPAD_DPAD_DOWN) m |= DM_BTN_DOWN;
            dm_pad_set_player(p, m, xaxis(g->sThumbLX), xaxis(g->sThumbLY), xaxis(g->sThumbRX), xaxis(g->sThumbRY));   // y up in both
        }
        if (mask != players) { players = mask; dm_pad_players(mask | 1); }
        Sleep(4);
    }
    return 0;
}
static void xinput_start(void) {
    if (getenv("DM_NO_XINPUT")) { SetEvent(g_pads_done); return; }
    HMODULE x = LoadLibraryW(L"xinput1_4.dll"); if (!x) x = LoadLibraryW(L"xinput9_1_0.dll");
    xget_t get = x ? (xget_t)(void *)GetProcAddress(x, "XInputGetState") : NULL;
    if (!get) { LOG("[front] no XInput: keyboard only"); SetEvent(g_pads_done); return; }
    g_xset = (xset_t)(void *)GetProcAddress(x, "XInputSetState");
    if (g_xset) rumble_setup("XInput");
    else LOG("[front] XInput has no XInputSetState: no vibration");
    HANDLE t = CreateThread(NULL, 0, pad_thread, (LPVOID)(void *)get, 0, NULL); if (t) CloseHandle(t); else SetEvent(g_pads_done);
}

// ---- SDL3 controllers (r43): the DLL is loaded at run time (no link dependency; without it XInput as before). One thread does every SDL call: it opens up to
// 4 gamepads as players 1-4 (a new one is looked for once a second), reads them every 4 ms, sends changed motor speeds and feeds the gyro / accelerometer ----
#if __has_include(<SDL3/SDL.h>)
#include <SDL3/SDL.h>
#define DM_SDL3 1
#define DM_SDLF(X) X(SDL_SetHint) X(SDL_Init) X(SDL_GetError) X(SDL_GetGamepads) X(SDL_free) X(SDL_OpenGamepad) X(SDL_CloseGamepad) X(SDL_GamepadConnected) \
    X(SDL_GetGamepadID) X(SDL_GetGamepadName) X(SDL_GetGamepadProperties) X(SDL_GetBooleanProperty) X(SDL_GamepadHasSensor) X(SDL_SetGamepadSensorEnabled) \
    X(SDL_GetGamepadSensorDataRate) X(SDL_GetGamepadSensorData) X(SDL_SetEventEnabled) X(SDL_UpdateGamepads) X(SDL_GetGamepadButton) X(SDL_GetGamepadAxis) X(SDL_RumbleGamepad)
#define DM_SDL_DECL(f) static __typeof__(f) *p_##f;
DM_SDLF(DM_SDL_DECL)
static int sdl_load(void) {
    const char *e = getenv("DM_SDL3_DLL"); HMODULE h = e && *e ? LoadLibraryA(e) : NULL;
    if (!h) h = LoadLibraryW(L"libSDL3.dll");
    if (!h) h = LoadLibraryW(L"SDL3.dll");
    if (!h) { LOG("[front] no SDL3 (DM_SDL3_DLL='%s', libSDL3.dll, SDL3.dll): XInput controllers", e ? e : ""); return 0; }
#define DM_SDL_LOAD(f) if (!(p_##f = (__typeof__(f) *)(void *)GetProcAddress(h, #f))) { LOG("[front] SDL3 has no %s: XInput controllers", #f); return 0; }
    DM_SDLF(DM_SDL_LOAD)
    return 1;
}
typedef struct { SDL_Gamepad *pad; int rumble, motion; LONG sent; ULONGLONG sent_at; } sdl_pad;
static sdl_pad g_sdl[4];
static const struct { SDL_GamepadButton b; uint64_t m; } g_sdl_btn[] = {   // by position, as the XInput mapping: the bottom face button is the Switch's B
    {SDL_GAMEPAD_BUTTON_SOUTH, DM_BTN_B}, {SDL_GAMEPAD_BUTTON_EAST, DM_BTN_A}, {SDL_GAMEPAD_BUTTON_WEST, DM_BTN_Y}, {SDL_GAMEPAD_BUTTON_NORTH, DM_BTN_X},
    {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, DM_BTN_L}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, DM_BTN_R}, {SDL_GAMEPAD_BUTTON_START, DM_BTN_PLUS},
    {SDL_GAMEPAD_BUTTON_BACK, DM_BTN_MINUS}, {SDL_GAMEPAD_BUTTON_LEFT_STICK, DM_BTN_LS}, {SDL_GAMEPAD_BUTTON_RIGHT_STICK, DM_BTN_RS},
    {SDL_GAMEPAD_BUTTON_DPAD_LEFT, DM_BTN_LEFT}, {SDL_GAMEPAD_BUTTON_DPAD_UP, DM_BTN_UP}, {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, DM_BTN_RIGHT},
    {SDL_GAMEPAD_BUTTON_DPAD_DOWN, DM_BTN_DOWN},
};
enum { SDLPAD_TRIGGER_ON = XINPUT_GAMEPAD_TRIGGER_THRESHOLD * 128, SDLPAD_RUMBLE_MS = 1000, SDLPAD_RESEND_MS = 500 };
static int32_t saxis(Sint16 v, int flip) { int32_t r = flip ? -(int32_t)v : v; return r > DM_STICK_MAX ? DM_STICK_MAX : r < -DM_STICK_MAX ? -DM_STICK_MAX : r; }   // SDL's y grows downwards
static void sdl_quit(void) {   // r52: a stop, then 80 ms of updates (the Switch driver's held-back write goes out), then the gamepads closed (SDL's close waits for queued writes)
    int any = 0;
    for (int p = 0; p < 4; p++) if (g_sdl[p].pad && g_sdl[p].rumble) { p_SDL_RumbleGamepad(g_sdl[p].pad, 0, 0, 0); any = 1; }
    if (any) { ULONGLONG end = GetTickCount64() + 80; do { Sleep(4); p_SDL_UpdateGamepads(); } while (GetTickCount64() < end); }
    for (int p = 0; p < 4; p++) if (g_sdl[p].pad) { SDL_Gamepad *g = g_sdl[p].pad; g_sdl[p].pad = NULL; p_SDL_CloseGamepad(g); }
    LOG("[front] exit: vibration stopped, SDL gamepads closed");
}
static void sdl_open(int p, SDL_JoystickID id) {
    sdl_pad *s = &g_sdl[p];
    if (!(s->pad = p_SDL_OpenGamepad(id))) { LOG("[front] SDL gamepad %u not opened: %s", (unsigned)id, p_SDL_GetError()); return; }
    s->rumble = p_SDL_GetBooleanProperty(p_SDL_GetGamepadProperties(s->pad), SDL_PROP_GAMEPAD_CAP_RUMBLE_BOOLEAN, false);
    s->motion = p_SDL_GamepadHasSensor(s->pad, SDL_SENSOR_GYRO) && p_SDL_GamepadHasSensor(s->pad, SDL_SENSOR_ACCEL) &&
                p_SDL_SetGamepadSensorEnabled(s->pad, SDL_SENSOR_GYRO, true) && p_SDL_SetGamepadSensorEnabled(s->pad, SDL_SENSOR_ACCEL, true);
    s->sent = 0; s->sent_at = 0;
    if (s->rumble) dm_haptics_route(p, 1);
    LOG("[front] SDL gamepad \"%s\" connected (player %d, %s, %s)", p_SDL_GetGamepadName(s->pad), p + 1, s->rumble ? "rumble" : "no rumble",
        s->motion ? "gyro and accelerometer" : "no motion sensors");
    if (s->motion) LOG("[front] player %d motion sensors on (gyro %.0f Hz, accelerometer %.0f Hz)", p + 1,
                       (double)p_SDL_GetGamepadSensorDataRate(s->pad, SDL_SENSOR_GYRO), (double)p_SDL_GetGamepadSensorDataRate(s->pad, SDL_SENSOR_ACCEL));
}
static void sdl_close(int p) {
    sdl_pad *s = &g_sdl[p];
    if (s->rumble) dm_haptics_route(p, 0);
    if (s->motion) dm_motion_clear_hid(p);
    dm_pad_set_player(p, 0, 0, 0, 0, 0);
    SDL_Gamepad *g = s->pad; s->pad = NULL; p_SDL_CloseGamepad(g);
    LOG("[front] controller %d disconnected", p + 1);
}
// The gyro and accelerometer in the Switch frame, as the TotK recomp's input_motion_axes (runtime/input.c): SDL x right, y out of the face, z toward the player,
// push against gravity in m/s^2; Switch x right, y toward the top edge, z out of the face, gravity in G (flat: z = -1). Gyro rad/s in both.
static void sdl_motion(int p, uint64_t t) {
    float a[3], g[3];
    if (!p_SDL_GetGamepadSensorData(g_sdl[p].pad, SDL_SENSOR_ACCEL, a, 3) || !p_SDL_GetGamepadSensorData(g_sdl[p].pad, SDL_SENSOR_GYRO, g, 3)) return;
    if (!a[0] && !a[1] && !a[2]) return;   // no sample yet (motion_apple.h drops the leading zero samples too)
    float acc[3] = {-a[0] / SDL_STANDARD_GRAVITY, a[2] / SDL_STANDARD_GRAVITY, -a[1] / SDL_STANDARD_GRAVITY}, rot[3] = {g[0], -g[2], g[1]};
    dm_motion_set_hid(p, acc, rot, t);
}
static DWORD WINAPI sdl_thread(LPVOID arg) {
    (void)arg;
    p_SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");   // no SDL window: read the controllers whatever has the focus, as XInput does
    if (!p_SDL_Init(SDL_INIT_GAMEPAD)) { LOG("[front] SDL gamepad init failed (%s): XInput controllers", p_SDL_GetError()); xinput_start(); return 0; }
    p_SDL_SetEventEnabled(SDL_EVENT_GAMEPAD_SENSOR_UPDATE, false);   // nothing drains SDL's event queue; the sensor data still updates (input.c)
    rumble_setup("SDL3");
    ULONGLONG scan = 0; uint32_t players = 0;
    for (;;) {
        if (g_pads_quit) { sdl_quit(); SetEvent(g_pads_done); return 0; }
        ULONGLONG now = GetTickCount64(); uint32_t mask = 0;
        p_SDL_UpdateGamepads();
        for (int p = 0; p < 4; p++) if (g_sdl[p].pad && !p_SDL_GamepadConnected(g_sdl[p].pad)) sdl_close(p);
        if (now >= scan) {
            scan = now + 1000; int n = 0; SDL_JoystickID *ids = p_SDL_GetGamepads(&n);
            for (int i = 0; ids && i < n; i++) {
                int open = 0, slot = -1;
                for (int p = 0; p < 4; p++) { if (g_sdl[p].pad && p_SDL_GetGamepadID(g_sdl[p].pad) == ids[i]) open = 1; if (!g_sdl[p].pad && slot < 0) slot = p; }
                if (!open && slot >= 0) sdl_open(slot, ids[i]);
            }
            p_SDL_free(ids);
        }
        uint64_t t = now_ns();
        for (int p = 0; p < 4; p++) {
            sdl_pad *s = &g_sdl[p]; SDL_Gamepad *g = s->pad; if (!g) continue;
            mask |= 1u << p;
            uint64_t m = 0;
            for (size_t i = 0; i < sizeof g_sdl_btn / sizeof g_sdl_btn[0]; i++) if (p_SDL_GetGamepadButton(g, g_sdl_btn[i].b)) m |= g_sdl_btn[i].m;
            if (p_SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > SDLPAD_TRIGGER_ON) m |= DM_BTN_ZL;
            if (p_SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > SDLPAD_TRIGGER_ON) m |= DM_BTN_ZR;
            dm_pad_set_player(p, m, saxis(p_SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX), 0), saxis(p_SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY), 1),
                              saxis(p_SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX), 0), saxis(p_SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY), 1));
            if (s->motion) sdl_motion(p, t);
            if (s->rumble) {   // SDL rumble runs for a set time: a changed pair at once, a held one again before it runs out
                LONG v = InterlockedCompareExchange(&g_rumble[p], 0, 0);
                if (v != s->sent || (v && now - s->sent_at >= SDLPAD_RESEND_MS)) {
                    p_SDL_RumbleGamepad(g, (Uint16)((ULONG)v >> 16), (Uint16)((ULONG)v & 0xffff), v ? SDLPAD_RUMBLE_MS : 0);
                    s->sent = v; s->sent_at = now;
                }
            }
        }
        if (mask != players) { players = mask; dm_pad_players(mask | 1); }
        Sleep(4);
    }
    return 0;
}
#endif

static void pads_start(void) {
    g_pads_done = CreateEventW(NULL, TRUE, FALSE, NULL);
#ifdef DM_SDL3
    const char *e = getenv("DM_NO_SDL");
    if (!(e && atoi(e)) && sdl_load()) { HANDLE t = CreateThread(NULL, 0, sdl_thread, NULL, 0, NULL); if (t) { CloseHandle(t); return; } }
#endif
    xinput_start();
}

void front_run(void (*guest)(void)) {
    if (!headless()) {
        g_ready = CreateEventW(NULL, TRUE, FALSE, NULL);
        HANDLE t = CreateThread(NULL, 0, window_thread, NULL, 0, NULL);
        if (t) { WaitForSingleObject(g_ready, INFINITE); CloseHandle(t); }
        if (g_hwnd) pads_start();
    }
    guest();
}
// The backend's window (gfx_vk_present.c): NULL headless
void *dm_front_window(void) { return g_hwnd; }
void dm_front_size(void *w, uint32_t *width, uint32_t *height) {
    RECT r; if (w && GetClientRect((HWND)w, &r)) { *width = (uint32_t)(r.right - r.left); *height = (uint32_t)(r.bottom - r.top); } else *width = *height = 0;
}
VkResult dm_front_surface(VkInstance inst, PFN_vkGetInstanceProcAddr gipa, void *w, VkSurfaceKHR *out) {
    typedef struct { VkStructureType sType; const void *pNext; VkFlags flags; HINSTANCE hinstance; HWND hwnd; } win32_ci;   // VkWin32SurfaceCreateInfoKHR
    typedef VkResult (VKAPI_PTR *create_t)(VkInstance, const win32_ci *, const VkAllocationCallbacks *, VkSurfaceKHR *);
    create_t c = (create_t)gipa(inst, "vkCreateWin32SurfaceKHR");
    if (!c) { LOG("[RED] [front] vkCreateWin32SurfaceKHR missing (VK_KHR_win32_surface not enabled?)"); return VK_ERROR_EXTENSION_NOT_PRESENT; }
    win32_ci ci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, NULL, 0, GetModuleHandleW(NULL), (HWND)w};
    VkResult r = c(inst, &ci, NULL, out);
    if (r) LOG("[RED] [front] vkCreateWin32SurfaceKHR failed: %d", r);
    return r;
}
void *front_metal_layer(void) { return NULL; }   // no Metal layer: the Vulkan backend asks dm_front_window
int front_screen_max_fps(void) { return 0; }
int front_screen_variable_refresh(void) { return 0; }
void front_screen_match(int fps) { (void)fps; }
void (*front_progress_hook)(const char *title, long done, long total);
void front_progress(const char *title, long done, long total) { (void)title; (void)done; (void)total; }
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    (void)title; (void)sub; (void)guide; (void)initial; (void)max_chars; (void)out; (void)out_cap;
    return -1;   // no input box yet (front.h: as without a window)
}
