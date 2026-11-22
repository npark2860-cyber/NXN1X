// r86: the performance overlay of NXN1X (src/yuzu/render/performance_overlay.cpp, process_utilization.cpp), the same panel drawn into a texture that
// gfx_vk_present.c lays over the window: big FPS, then CPU / GPU / RAM / VRAM / Mode / Driver, the shader spinner while pipelines are being made.
// Same fonts (Pretendard Std, embedded), sizes, colours, spacing and rules (the width only grows while the mode stays, digits measured as the widest digit).
// Click: FPS only (GPU % beside it) and back. Drag: move it (from the top-right corner). Wheel: background opacity 40..255 in steps of 15. F3: on / off.
// DM_PERF_OVERLAY=0 starts hidden, =2 starts in the FPS-only mode. The choices are kept in <DM_ROOT or the current folder>/perf_overlay.txt.
// Rendering happens on the present thread (perf_ov_frame), input on the window thread (perf_ov_mouse_*), the counters on a thread of their own (1 s, as NXN1X's).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <pdhmsg.h>
#endif
#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "../third_party/stb_truetype/stb_truetype.h"
#include "../third_party/pretendard/pretendard_otf.h"
#include "../third_party/pretendard/pretendard_kern.h"

void mtl_create_stats(uint64_t *lib, uint64_t *pso);   // gfx_vk_exec.c: pipelines made where a draw waits for them (the shader spinner's count)

enum { Padding = 10, Margin = 8, RowHeight = 18, Radius = 8, UnitGap = 6, LabelGap = 8, OpacityStep = 15, MinOpacity = 40, MaxOpacity = 255,
       SpinnerSize = 14, SpinnerSpokes = 12, SpinnerStepMs = 80, SpinnerAlphaStep = 19 };
static const uint8_t Accent[4] = {240, 128, 60, 255}, Dim[4] = {185, 185, 185, 255}, Divider[4] = {255, 255, 255, 60}, White[4] = {255, 255, 255, 255};

// ---- fonts: QFont pixel sizes and weights as NXN1X's (big 34 Bold, FPS-only 40 Bold, unit 12 DemiBold, label 12 Medium, value 12 Normal) ----
#ifndef TEXT_GAMMA
#define TEXT_GAMMA 2.2f
#endif
enum { F_REG, F_MED, F_SEMI, F_BOLD, F_N };
static stbtt_fontinfo g_font[F_N]; static int g_font_ok = -1; static void text_gamma_init(void);
static float g_asc, g_desc, g_cap;   // per em (OS/2 typo metrics, which Pretendard marks USE_TYPO_METRICS: the ones DirectWrite / Qt take)
typedef struct { int f, px; } pfont;
static const pfont BigF = {F_BOLD, 34}, CompactF = {F_BOLD, 40}, UnitF = {F_SEMI, 12}, LabelF = {F_MED, 12}, ValueF = {F_REG, 12};
static int fonts_init(void) {
    if (g_font_ok >= 0) return g_font_ok;
    const unsigned char *d[F_N] = {pretendard_regular, pretendard_medium, pretendard_semibold, pretendard_bold};
    g_font_ok = 1; text_gamma_init();
    for (int i = 0; i < F_N; i++) if (!stbtt_InitFont(&g_font[i], d[i], stbtt_GetFontOffsetForIndex(d[i], 0))) g_font_ok = 0;
    if (!g_font_ok) return 0;
    int a, de, gap; float em = (float)ttUSHORT(g_font[F_REG].data + g_font[F_REG].head + 18);
    if (!stbtt_GetFontVMetricsOS2(&g_font[F_REG], &a, &de, &gap)) stbtt_GetFontVMetrics(&g_font[F_REG], &a, &de, &gap);
    g_asc = a / em; g_desc = -de / em;
    uint32_t os2 = stbtt__find_table(g_font[F_REG].data, g_font[F_REG].fontstart, "OS/2");
    g_cap = os2 && ttUSHORT(g_font[F_REG].data + os2) >= 2 ? ttSHORT(g_font[F_REG].data + os2 + 88) / em : 0.7f;
    return 1;
}
static const pretendard_kern *const g_kern[F_N] = {pretendard_kern_regular, pretendard_kern_medium, pretendard_kern_semibold, pretendard_kern_bold};
static const int g_nkern[F_N] = {sizeof pretendard_kern_regular / sizeof *pretendard_kern_regular, sizeof pretendard_kern_medium / sizeof *pretendard_kern_medium,
                                 sizeof pretendard_kern_semibold / sizeof *pretendard_kern_semibold, sizeof pretendard_kern_bold / sizeof *pretendard_kern_bold};
static int kern(int f, int a, int b) {   // GPOS pair kerning as Qt (HarfBuzz) applies it, font units (stb_truetype reads only part of Pretendard's GPOS: 4 px off on the Driver row)
    int lo = 0, hi = g_nkern[f] - 1; uint32_t key = (uint32_t)a << 16 | (uint32_t)b;
    while (lo <= hi) { int m = (lo + hi) / 2; uint32_t k = (uint32_t)g_kern[f][m].a << 16 | g_kern[f][m].b; if (k == key) return g_kern[f][m].k; if (k < key) lo = m + 1; else hi = m - 1; }
    return 0;
}
static float em_scale(pfont f) { return stbtt_ScaleForMappingEmToPixels(&g_font[f.f], (float)f.px); }
static int next_cp(const char **s) {   // UTF-8 (the Driver row's middle dot)
    const unsigned char *p = (const unsigned char *)*s; int c = *p++;
    if (c >= 0xc0 && c < 0xe0 && (*p & 0xc0) == 0x80) c = (c & 0x1f) << 6 | (*p++ & 0x3f);
    else if (c >= 0xe0 && c < 0xf0 && (p[0] & 0xc0) == 0x80 && (p[1] & 0xc0) == 0x80) { c = (c & 0x0f) << 12 | (p[0] & 0x3f) << 6 | (p[1] & 0x3f); p += 2; }
    *s = (const char *)p; return c;
}
static float text_wf(pfont f, const char *s) {   // logical pixels, kerned (as QFontMetrics::horizontalAdvance)
    float k = em_scale(f), w = 0; int prev = 0;
    while (*s) { int c = next_cp(&s), g = stbtt_FindGlyphIndex(&g_font[f.f], c), adv, lsb; stbtt_GetGlyphHMetrics(&g_font[f.f], g, &adv, &lsb);
                 if (prev) { w += kern(f.f, prev, c) * k; } w += adv * k; prev = c; }
    return w;
}
static int adv(pfont f, const char *s) { return (int)lroundf(text_wf(f, s)); }
static int cap(pfont f) { return (int)lroundf(g_cap * f.px); }
static void digits(pfont f, const char *in, char *out, size_t n) {   // every digit as the font's widest one
    char wd = '0'; int ww = adv(f, "0");   // (whole pixels, as QFontMetrics compares them)
    for (char d = '1'; d <= '9'; d++) { char s[2] = {d, 0}; int w = adv(f, s); if (w > ww) { ww = w; wd = d; } }
    size_t i = 0; for (; in[i] && i + 1 < n; i++) out[i] = in[i] >= '0' && in[i] <= '9' ? wd : in[i]; out[i] = 0;
}
static int adv_digits(pfont f, const char *s) { char t[160]; digits(f, s, t, sizeof t); return adv(f, t); }

// ---- the canvas: physical pixels, straight alpha (as gfx_vk_present.c's cv_blend) ----
typedef struct { uint8_t *px; int w, h; float k; } canvas;   // k: device pixels per logical pixel
static void blend(const canvas *cv, int x, int y, const uint8_t c[4], float cov) {
    if (x < 0 || y < 0 || x >= cv->w || y >= cv->h || cov <= 0) return;
    uint8_t *p = cv->px + 4 * ((size_t)y * cv->w + x);
    float a = c[3] / 255.0f * (cov > 1 ? 1 : cov), da = p[3] / 255.0f, oa = a + da * (1 - a);
    if (oa <= 0) return;
    for (int i = 0; i < 3; i++) p[i] = (uint8_t)((c[i] * a + p[i] * da * (1 - a)) / oa + 0.5f);
    p[3] = (uint8_t)(oa * 255 + 0.5f);
}
static void fill_rrect(const canvas *cv, float x0, float y0, float x1, float y1, float rad, const uint8_t c[4]) {   // device pixels
    float cx = (x0 + x1) / 2, cy = (y0 + y1) / 2, hx = (x1 - x0) / 2 - rad, hy = (y1 - y0) / 2 - rad;
    for (int y = (int)y0 - 1; y <= (int)y1 + 1; y++) for (int x = (int)x0 - 1; x <= (int)x1 + 1; x++) {
        float qx = fabsf(x + 0.5f - cx) - hx, qy = fabsf(y + 0.5f - cy) - hy, ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
        float d = sqrtf(ox * ox + oy * oy) + fminf(fmaxf(qx, qy), 0) - rad;
        float cov = fminf(1, fmaxf(0, 0.5f - d));
        // the panel's edge against the rectangle's own bounds (the box is the widget: nothing outside it)
        cov *= fminf(1, fmaxf(0, fminf(x + 1 - x0, x1 - x))) * fminf(1, fmaxf(0, fminf(y + 1 - y0, y1 - y)));
        blend(cv, x, y, c, cov);
    }
}
static void hline(const canvas *cv, float x0, float x1, float yl, const uint8_t c[4]) {   // QPainter::drawLine with a cosmetic pen (1 device pixel, square caps), antialiased
    float yc = yl * cv->k, a0 = x0 * cv->k - 0.5f, a1 = x1 * cv->k + 0.5f;
    for (int y = (int)floorf(yc - 0.5f); y <= (int)floorf(yc + 0.5f); y++) {
        float vy = fminf(y + 1, yc + 0.5f) - fmaxf(y, yc - 0.5f); if (vy <= 0) continue;
        for (int x = (int)floorf(a0); x <= (int)floorf(a1); x++) { float vx = fminf(x + 1, a1) - fmaxf(x, a0); if (vx > 0) blend(cv, x, y, c, vx * vy); }
    }
}
static void capsule(const canvas *cv, float ax, float ay, float bx, float by, float r, const uint8_t c[4]) {   // a round-capped line, device pixels
    int x0 = (int)floorf(fminf(ax, bx) - r - 1), x1 = (int)ceilf(fmaxf(ax, bx) + r + 1), y0 = (int)floorf(fminf(ay, by) - r - 1), y1 = (int)ceilf(fmaxf(ay, by) + r + 1);
    float dx = bx - ax, dy = by - ay, ll = dx * dx + dy * dy;
    for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
        float px = x + 0.5f - ax, py = y + 0.5f - ay, t = ll > 0 ? fminf(1, fmaxf(0, (px * dx + py * dy) / ll)) : 0;
        float ex = px - t * dx, ey = py - t * dy, d = sqrtf(ex * ex + ey * ey) - r;
        blend(cv, x, y, c, fminf(1, fmaxf(0, 0.5f - d)));
    }
}
static float g_text_gamma[256];   // glyph coverage made heavier, as Qt's text (its gamma-corrected grey antialiasing) looks beside a plain coverage blend
static void text_gamma_init(void) { const char *e = getenv("DM_PERF_OVERLAY_GAMMA"); float g = e && *e ? (float)atof(e) : TEXT_GAMMA; if (g <= 0) g = 1; for (int i = 0; i < 256; i++) g_text_gamma[i] = powf(i / 255.0f, 1 / g); }
static void text(const canvas *cv, pfont f, float xl, float basel, const char *s, const uint8_t c[4]) {   // QPainter::drawText(QPoint): x logical, baseline logical
    float k = stbtt_ScaleForMappingEmToPixels(&g_font[f.f], f.px * cv->k), x = xl * cv->k; int by = (int)lroundf(basel * cv->k), prev = 0;
    while (*s) {
        int cp = next_cp(&s), g = stbtt_FindGlyphIndex(&g_font[f.f], cp), ad, lsb; stbtt_GetGlyphHMetrics(&g_font[f.f], g, &ad, &lsb);
        if (prev) x += kern(f.f, prev, cp) * k;
        float fx = x - floorf(x); int ix0, iy0, ix1, iy1;
        stbtt_GetGlyphBitmapBoxSubpixel(&g_font[f.f], g, k, k, fx, 0, &ix0, &iy0, &ix1, &iy1);
        int gw = ix1 - ix0, gh = iy1 - iy0;
        if (gw > 0 && gh > 0 && gw < 512 && gh < 512) {
            unsigned char *bm = malloc((size_t)gw * gh);
            stbtt_MakeGlyphBitmapSubpixel(&g_font[f.f], bm, gw, gh, gw, k, k, fx, 0, g);
            int ox = (int)floorf(x) + ix0, oy = by + iy0;
            for (int yy = 0; yy < gh; yy++) for (int xx = 0; xx < gw; xx++) if (bm[yy * gw + xx]) blend(cv, ox + xx, oy + yy, c, g_text_gamma[bm[yy * gw + xx]]);
            free(bm);
        }
        x += ad * k; prev = cp;
    }
}
static void text_vc(const canvas *cv, pfont f, int x, int top, const char *s, const uint8_t c[4]) {   // drawText(QRect{x, top, .., RowHeight}, AlignLeft | AlignVCenter)
    float base = top + (RowHeight - (g_asc + g_desc) * f.px) / 2 + g_asc * f.px;
    text(cv, f, (float)x, base, s, c);
}
// r91 (NPARK 11-08): numbers in fixed columns. Each digit sits centred in a cell as wide as the font's widest digit (r95: its ink), other characters at their own width (no kerning
// around digits), so a changing count keeps its place instead of shifting by the digits' different widths
// r95 (NPARK 11-08, as NXN1X S3z14): the cell is the widest digit's ink (its outline's extent, not its advance) plus a 1 px gap (2 px from 22 px fonts), each digit's ink centred in it,
// so the digits sit as close as the font allows and still keep their places
static void digit_ink(pfont f, char d, float *left, float *width) {
    int x0 = 0, y0, x1 = 0, y1; float k = em_scale(f);
    if (!stbtt_GetGlyphBox(&g_font[f.f], stbtt_FindGlyphIndex(&g_font[f.f], d), &x0, &y0, &x1, &y1)) x0 = x1 = 0;
    *left = x0 * k; *width = (x1 - x0) * k;
}
static float digit_cell(pfont f) { float m = 0, l, w; for (char d = '0'; d <= '9'; d++) { digit_ink(f, d, &l, &w); if (w > m) m = w; } return m + (f.px >= 22 ? 2.0f : 1.0f); }
static int one_char(const char **s, char *u) { const char *p = *s; next_cp(s); size_t n = (size_t)(*s - p); if (n > 7) n = 7; memcpy(u, p, n); u[n] = 0; return n == 1 && u[0] >= '0' && u[0] <= '9'; }
static float tab_wf(pfont f, const char *s) { float cw = digit_cell(f), w = 0; char u[8]; while (*s) w += one_char(&s, u) ? cw : text_wf(f, u); return w; }
static int tab_w(pfont f, const char *s) { return (int)ceilf(tab_wf(f, s)); }
static void text_tab(const canvas *cv, pfont f, float xl, float basel, const char *s, const uint8_t c[4]) {
    float cw = digit_cell(f), x = xl; char u[8];
    while (*s) {
        int dg = one_char(&s, u);
        if (dg) { float l, w; digit_ink(f, u[0], &l, &w); text(cv, f, x + (cw - w) / 2 - l, basel, u, c); x += cw; }   // r95: the ink centred, as NXN1X DigitX
        else { text(cv, f, x, basel, u, c); x += text_wf(f, u); }
    }
}
static void text_vc_tab(const canvas *cv, pfont f, int x, int top, const char *s, const uint8_t c[4]) {
    float base = top + (RowHeight - (g_asc + g_desc) * f.px) / 2 + g_asc * f.px;
    text_tab(cv, f, (float)x, base, s, c);
}

// ---- the counters (NXN1X's ProcessUtilization: a 1 s sample of process CPU time over all logical CPUs, the busiest GPU engine of this process, the working set) ----
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static double g_cpu = -1, g_gpu = -1; static int64_t g_ram = -1, g_phys = -1; static int64_t g_sampled_ms;
// r102 (OoT 11-09, a long frame every 1.002 s with submit+present 23-35 ms, in step with this thread and not with frames): the GPU row came from the PDH counter
// "GPU Engine(*)\\Utilization Percentage", whose collection walks every process's engines in the kernel once a second. The row is now the host's own GPU time
// (g_gpu_ns, the command buffers' timestamps, given by the present thread through perf_ov_gpu_ns) over the second. DM_PERF_PDH=1 brings the PDH counter back
static uint64_t g_gpu_busy_ns; static int g_gpu_busy_set;
static int64_t now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000; }
#ifdef _WIN32
typedef PDH_STATUS (WINAPI *pdh_open_t)(LPCWSTR, DWORD_PTR, PDH_HQUERY *);
typedef PDH_STATUS (WINAPI *pdh_add_t)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER *);
typedef PDH_STATUS (WINAPI *pdh_collect_t)(PDH_HQUERY);
typedef PDH_STATUS (WINAPI *pdh_array_t)(PDH_HCOUNTER, DWORD, LPDWORD, LPDWORD, PPDH_FMT_COUNTERVALUE_ITEM_W);
typedef PDH_STATUS (WINAPI *pdh_close_t)(PDH_HQUERY);
static struct { pdh_open_t open; pdh_add_t add; pdh_collect_t collect; pdh_array_t array; pdh_close_t close; PDH_HQUERY q; PDH_HCOUNTER c; wchar_t prefix[32]; void *buf; DWORD cap; } P;
static void gpu_close(void) { if (P.q) P.close(P.q); P.q = NULL; P.c = NULL; }
static int gpu_open(void) {
    gpu_close();
    if (!P.open) { HMODULE m = LoadLibraryW(L"pdh.dll"); if (!m) return 0;
        P.open = (pdh_open_t)(void *)GetProcAddress(m, "PdhOpenQueryW"); P.add = (pdh_add_t)(void *)GetProcAddress(m, "PdhAddEnglishCounterW");
        P.collect = (pdh_collect_t)(void *)GetProcAddress(m, "PdhCollectQueryData"); P.array = (pdh_array_t)(void *)GetProcAddress(m, "PdhGetFormattedCounterArrayW");
        P.close = (pdh_close_t)(void *)GetProcAddress(m, "PdhCloseQuery");
        if (!P.open || !P.add || !P.collect || !P.array || !P.close) { P.open = NULL; return 0; }
        swprintf(P.prefix, 32, L"pid_%lu_", (unsigned long)GetCurrentProcessId()); }
    if (P.open(NULL, 0, &P.q) != ERROR_SUCCESS || P.add(P.q, L"\\GPU Engine(*)\\Utilization Percentage", 0, &P.c) != ERROR_SUCCESS) { gpu_close(); return 0; }
    P.collect(P.q);   // a rate needs two collections
    return 1;
}
static double gpu_sample(void) {
    if (!P.q || P.collect(P.q) != ERROR_SUCCESS) return -1;
    size_t pl = wcslen(P.prefix);
    for (int attempt = 0; attempt < 2; attempt++) {   // the set of engines can change between the size and the data calls
        DWORD bytes = 0, count = 0, fmt = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100;
        PDH_STATUS st = P.array(P.c, fmt, &bytes, &count, NULL);
        if (st != (PDH_STATUS)PDH_MORE_DATA || !bytes || bytes > 16u << 20) return -1;
        if (bytes > P.cap) { free(P.buf); P.buf = malloc(bytes); P.cap = P.buf ? bytes : 0; if (!P.buf) return -1; }
        PDH_FMT_COUNTERVALUE_ITEM_W *it = P.buf; st = P.array(P.c, fmt, &bytes, &count, it);
        if (st == (PDH_STATUS)PDH_MORE_DATA || st == (PDH_STATUS)PDH_INVALID_ARGUMENT) continue;
        if (st != ERROR_SUCCESS) return -1;
        double best = -1;
        for (DWORD i = 0; i < count; i++) {
            if (!it[i].szName || wcsncmp(it[i].szName, P.prefix, pl) || (it[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA && it[i].FmtValue.CStatus != PDH_CSTATUS_NEW_DATA)) continue;
            double v = it[i].FmtValue.doubleValue; if (isfinite(v) && v >= 0) best = fmax(best, fmin(100, v));
        }
        return best;
    }
    return -1;
}
static uint64_t cpu_ticks(int *ok) {
    FILETIME c, e, k, u; *ok = GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u) != 0;
    return ((uint64_t)k.dwHighDateTime << 32 | k.dwLowDateTime) + ((uint64_t)u.dwHighDateTime << 32 | u.dwLowDateTime);
}
static void *sampler(void *arg) {
    (void)arg;
    DWORD ncpu = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); if (!ncpu) ncpu = 1;
    { MEMORYSTATUSEX m = {sizeof m}; if (GlobalMemoryStatusEx(&m)) g_phys = (int64_t)m.ullTotalPhys; }
    const char *pe = getenv("DM_PERF_PDH"); int pdh = pe && atoi(pe);   // r102
    int pok, pgok = 0; uint64_t prev = cpu_ticks(&pok), pg = 0; int64_t pt = now_ms(), retry = pt + 30000; if (pdh) gpu_open();
    // r103 (OoT 11-09): a 2-vblank frame every 1.000-1.002 s stayed after r102, in step with this loop. DM_PERF_SAMPLER=0 stops it (CPU / GPU / RAM rows "--");
    // each call slower than 2 ms is logged (the first 40)
    const char *se = getenv("DM_PERF_SAMPLER"); if (se && *se == '0') { fprintf(stderr, "[dm] r103: performance overlay counters off (DM_PERF_SAMPLER=0)\n"); return NULL; }
    int slow = 0; LARGE_INTEGER qf; QueryPerformanceFrequency(&qf);
    for (;;) {
        Sleep(1000);
        double cpu = -1, gpu = -1; int64_t ram = -1;
        PROCESS_MEMORY_COUNTERS_EX m = {sizeof m};
        LARGE_INTEGER q0, q1, q2; QueryPerformanceCounter(&q0);
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&m, sizeof m)) ram = (int64_t)m.WorkingSetSize;
        QueryPerformanceCounter(&q1);
        int cok; uint64_t cur = cpu_ticks(&cok); int64_t t = now_ms(); double sec = (t - pt) / 1000.0;
        QueryPerformanceCounter(&q2);
        { double a = (double)(q1.QuadPart - q0.QuadPart) * 1e3 / (double)qf.QuadPart, b = (double)(q2.QuadPart - q1.QuadPart) * 1e3 / (double)qf.QuadPart;
          if ((a > 2 || b > 2) && slow++ < 40) fprintf(stderr, "[dm] r103: overlay counters slow at %.2f s: memory info %.2f ms, process times %.2f ms\n", (double)t / 1e3, a, b); }
        if (cok && pok && cur >= prev && sec > 0) cpu = fmin(100, fmax(0, (double)(cur - prev) * 1e-7 * 100.0 / (sec * ncpu)));   // FILETIME: 100 ns
        prev = cur; pok = cok; pt = t;
        if (pdh) { if (!P.q && t >= retry) { gpu_open(); retry = t + 30000; } else gpu = gpu_sample(); }
        else { pthread_mutex_lock(&g_mu); uint64_t g = g_gpu_busy_ns; int set = g_gpu_busy_set; pthread_mutex_unlock(&g_mu);   // r102: the host's GPU time over the second
               if (set && pgok && g >= pg && sec > 0) gpu = fmin(100, (double)(g - pg) * 1e-9 * 100.0 / sec);
               pg = g; pgok = set; }
        pthread_mutex_lock(&g_mu); g_cpu = cpu; g_gpu = gpu; g_ram = ram; g_sampled_ms = now_ms(); pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}
#else
static void *sampler(void *arg) { (void)arg; return NULL; }
#endif

// ---- the panel's state ----
static int g_on = -1, g_fps_only, g_opacity = 165, g_width_floor, g_dirty = 1;
static float g_dx, g_dy;   // the drag offset from the default place, device pixels
static int g_px, g_py, g_pw, g_ph, g_shown;   // where the panel was last drawn (device pixels), for the window's mouse
static int g_press, g_dragging, g_drag_dist = 4; static float g_press_x, g_press_y, g_grab_x, g_grab_y;
static uint32_t g_sw, g_sh; static float g_k = 1;
static char g_driver[96];
static int64_t g_vram_used = -1, g_vram_budget = -1;
static char g_fps_whole[16] = "--";
static int64_t g_spin_last_ms; static int g_spin_step, g_shader_burst; static int64_t g_sh_last = -1, g_sh_base = -1;
static uint8_t *g_img; static int g_iw, g_ih;
static char g_cfg[512];
static void cfg_save(void) {
    if (!g_cfg[0]) return;
    FILE *f = fopen(g_cfg, "w"); if (!f) return;
    fprintf(f, "show=%d\nfps_only=%d\nopacity=%d\n", g_on, g_fps_only, g_opacity); fclose(f);
}
static void init_once(void) {
    if (g_on >= 0) return;
    const char *r = getenv("DM_ROOT"); snprintf(g_cfg, sizeof g_cfg, "%s%sperf_overlay.txt", r && *r ? r : "", r && *r ? "/" : "");
    g_on = 1;
    FILE *f = fopen(g_cfg, "r");
    if (f) { char l[64]; int v; while (fgets(l, sizeof l, f)) { if (sscanf(l, "show=%d", &v) == 1) g_on = !!v; else if (sscanf(l, "fps_only=%d", &v) == 1) g_fps_only = !!v;
                                                                     else if (sscanf(l, "opacity=%d", &v) == 1) g_opacity = v < MinOpacity ? MinOpacity : v > MaxOpacity ? MaxOpacity : v; } fclose(f); }
    const char *e = getenv("DM_PERF_OVERLAY"); if (e && *e) { g_on = *e != '0'; if (*e == '2') g_fps_only = 1; }
    pthread_t t; if (!pthread_create(&t, NULL, sampler, NULL)) pthread_detach(t);
}

// ---- the layout and painting: NXN1X's drawPanel, line for line ----
typedef struct { char label[16], value[96], share[16]; } row_t;
static int panel(const canvas *cv, row_t *rows, int nrows, int *ow, int *oh) {   // cv NULL: sizes only
    pfont big = g_fps_only ? CompactF : BigF;
    const char *unit_text = "FPS";
    int content = (tab_w(big, g_fps_whole) > adv(big, "00") ? tab_w(big, g_fps_whole) : adv(big, "00")) + UnitGap + adv(UnitF, unit_text);
    int label_width = 0, value_width = 0;
    for (int i = 0; i < nrows; i++) {
        int lw = adv(LabelF, rows[i].label); if (lw > label_width) label_width = lw;
        char sh[24]; snprintf(sh, sizeof sh, " %s", rows[i].share);
        int share = rows[i].share[0] ? tab_w(ValueF, sh) : 0, vw = tab_w(ValueF, rows[i].value) + share; if (vw > value_width) value_width = vw;
    }
    int value_x = Padding + label_width + LabelGap;
    if (label_width + LabelGap + value_width > content) content = label_width + LabelGap + value_width;
    char shader_count[16]; snprintf(shader_count, sizeof shader_count, "%d", g_shader_burst);
    const char *shader_label = "Shaders";
    if (g_shader_burst > 0) { int c = SpinnerSize + LabelGap + adv(LabelF, shader_label) + LabelGap + tab_w(ValueF, shader_count); if (c > content) content = c; }
    int panel_width = content + Padding * 2 > g_width_floor ? content + Padding * 2 : g_width_floor;
    int y = Padding + cap(big);
    if (cv) {
        int slot = tab_w(big, g_fps_whole) > adv(big, "00") ? tab_w(big, g_fps_whole) : adv(big, "00");   // right-aligned in its slot: "FPS" stays put
        text_tab(cv, big, (float)(Padding + slot) - tab_wf(big, g_fps_whole), (float)y, g_fps_whole, White);   // r91
        text(cv, UnitF, (float)(Padding + slot + UnitGap), (float)y, unit_text, Accent);
    }
    y += 8;
    if (cv) hline(cv, Padding, (float)(panel_width - Padding), (float)y, Divider);
    y += 3;
    for (int i = 0; i < nrows; i++) {
        if (cv) {
            text_vc(cv, LabelF, Padding, y, rows[i].label, Accent);
            int x = value_x;
            if (g_fps_only) x = panel_width - Padding - tab_w(ValueF, rows[i].value);   // r91: values in fixed digit columns
            text_vc_tab(cv, ValueF, x, y, rows[i].value, White);
            if (rows[i].share[0]) { char v[112]; snprintf(v, sizeof v, "%s ", rows[i].value); x += tab_w(ValueF, v); text_vc_tab(cv, ValueF, x, y, rows[i].share, Dim); }
        }
        y += RowHeight;
    }
    if (g_shader_burst > 0) {
        if (cv) {
            float k = cv->k, outer = SpinnerSize / 2.0f, inner = outer * 0.45f, pw = fmaxf(1.5f, outer * 0.26f);
            float cx = (Padding + SpinnerSize / 2.0f) * k, cy = (y + (RowHeight - SpinnerSize) / 2.0f + SpinnerSize / 2.0f) * k;
            for (int i = 0; i < SpinnerSpokes; i++) {   // spoke i at i * 30 degrees clockwise from 12 o'clock; the head (g_spin_step) opaque, the ones behind it fading
                int age = (g_spin_step - i + SpinnerSpokes) % SpinnerSpokes; uint8_t c[4] = {255, 255, 255, (uint8_t)(255 - age * SpinnerAlphaStep)};
                float a = i * 6.2831853f / SpinnerSpokes, sx = sinf(a), sy = -cosf(a);
                capsule(cv, cx + sx * inner * k, cy + sy * inner * k, cx + sx * outer * k, cy + sy * outer * k, pw * k / 2, c);
            }
            int label_x = Padding + SpinnerSize + LabelGap;
            text_vc(cv, LabelF, label_x, y, shader_label, Accent);
            text_vc_tab(cv, ValueF, label_x + adv(LabelF, shader_label) + LabelGap, y, shader_count, White);   // r91
        }
        y += RowHeight;
    }
    *ow = panel_width; *oh = y + Padding - 2;
    return 1;
}
static void fmt_pct(char *o, size_t n, double v, int dec) { if (isfinite(v) && v >= 0) snprintf(o, n, "%.*f%%", dec, v); else snprintf(o, n, "--"); }
static void fmt_mb(char *o, size_t n, int64_t b) { if (b >= 0) snprintf(o, n, "%lld MB", (long long)(b / 1000000)); else snprintf(o, n, "--"); }
static void fmt_share(char *o, size_t n, int64_t part, int64_t whole) { if (part < 0 || whole <= 0) o[0] = 0; else snprintf(o, n, "(%ld%%)", lround(100.0 * (double)part / (double)whole)); }
static int build_rows(row_t *r) {
    double cpu = -1, gpu = -1; int64_t ram = -1, phys;
    if (g_sampled_ms && now_ms() - g_sampled_ms <= 3500) { cpu = g_cpu; gpu = g_gpu; ram = g_ram; } phys = g_phys;   // (under g_mu: perf_ov_frame)
    memset(r, 0, sizeof(row_t) * 6);
    if (g_fps_only) { strcpy(r[0].label, "GPU"); fmt_pct(r[0].value, sizeof r[0].value, gpu, 0); return 1; }
    strcpy(r[0].label, "CPU"); fmt_pct(r[0].value, sizeof r[0].value, cpu, 1);
    strcpy(r[1].label, "GPU"); fmt_pct(r[1].value, sizeof r[1].value, gpu, 1);
    strcpy(r[2].label, "RAM"); fmt_mb(r[2].value, sizeof r[2].value, ram); fmt_share(r[2].share, sizeof r[2].share, ram, phys);
    strcpy(r[3].label, "VRAM"); fmt_mb(r[3].value, sizeof r[3].value, g_vram_used); fmt_share(r[3].share, sizeof r[3].share, g_vram_used, g_vram_budget);
    strcpy(r[4].label, "Mode"); strcpy(r[4].value, "Native");
    if (!g_driver[0]) return 5;
    strcpy(r[5].label, "Driver"); snprintf(r[5].value, sizeof r[5].value, "%s", g_driver); return 6;
}
static void place(void) {   // top-right of the window, Margin in from it, plus the drag offset, kept inside
    float k = g_k, x = g_sw - g_iw - Margin * k + g_dx, y = Margin * k + g_dy;
    float mx = g_sw > (uint32_t)g_iw ? (float)(g_sw - g_iw) : 0, my = g_sh > (uint32_t)g_ih ? (float)(g_sh - g_ih) : 0;
    g_px = (int)lroundf(fminf(fmaxf(x, 0), mx)); g_py = (int)lroundf(fminf(fmaxf(y, 0), my)); g_pw = g_iw; g_ph = g_ih;
}

// ---- the present thread ----
int perf_ov_on(void) { init_once(); return g_on && fonts_init(); }
void perf_ov_driver(const char *s) { pthread_mutex_lock(&g_mu); if (strcmp(g_driver, s)) { snprintf(g_driver, sizeof g_driver, "%s", s); g_dirty = 1; } pthread_mutex_unlock(&g_mu); }
void perf_ov_gpu_ns(uint64_t ns) { pthread_mutex_lock(&g_mu); g_gpu_busy_ns = ns; g_gpu_busy_set = 1; pthread_mutex_unlock(&g_mu); }   // r102
void perf_ov_vram(int64_t used, int64_t budget) { pthread_mutex_lock(&g_mu); g_vram_used = used; g_vram_budget = budget; pthread_mutex_unlock(&g_mu); }
// Once per present while on. frames: the game's frames shown so far; sw, sh: the window's pixels; k: its DPI scale. Returns 1 when the picture changed
// (*px: g_iw x g_ih RGBA, straight alpha, the background included), and the panel's place in *x, *y either way.
int perf_ov_frame(int64_t now_ns, uint64_t frames, uint32_t sw, uint32_t sh, float k, const uint8_t **px, int *w, int *h, int *x, int *y) {
    static int64_t t0; static uint64_t n0;
    int64_t ms = now_ns / 1000000;
    pthread_mutex_lock(&g_mu);
    if (sw != g_sw || sh != g_sh || fabsf(k - g_k) > 1e-3f) { g_sw = sw; g_sh = sh; g_k = k; g_dirty = 1; }
    if (!t0 || now_ns - t0 > 2000000000ll) { t0 = now_ns; n0 = frames; snprintf(g_fps_whole, sizeof g_fps_whole, "--"); g_dirty = 1; }   // the first frame, or a pause in the presents
    if (now_ns - t0 >= 500000000ll) {   // NXN1X's 500 ms sample: the rate, the shader burst, the rows
        double fps = (double)(frames - n0) * 1e9 / (double)(now_ns - t0); t0 = now_ns; n0 = frames;
        if (isfinite(fps) && fps >= 0 && fps < 10000) snprintf(g_fps_whole, sizeof g_fps_whole, "%.0f", fps); else snprintf(g_fps_whole, sizeof g_fps_whole, "--");
        uint64_t lib, pso; mtl_create_stats(&lib, &pso); int64_t made = (int64_t)pso;
        if (g_sh_last < 0 || made < g_sh_last) { g_sh_last = made; g_sh_base = -1; }
        int building = made != g_sh_last;   // a pipeline made since the last sample
        if (building && g_sh_base < 0) g_sh_base = g_sh_last; else if (!building) g_sh_base = -1;
        g_sh_last = made; g_shader_burst = building ? (int)(made - g_sh_base) : 0;
        g_dirty = 1;
    }
    if (g_shader_burst > 0 && ms - g_spin_last_ms >= SpinnerStepMs) { g_spin_last_ms = ms; g_spin_step = (g_spin_step + 1) % SpinnerSpokes; g_dirty = 1; }
    int changed = 0;
    if (g_dirty) {
        g_dirty = 0; row_t rows[6]; int n = build_rows(rows), lw, lh;
        panel(NULL, rows, n, &lw, &lh); g_width_floor = lw;
        int iw = (int)ceilf(lw * k), ih = (int)ceilf(lh * k);
        if (iw != g_iw || ih != g_ih || !g_img) { free(g_img); g_img = malloc((size_t)iw * ih * 4); g_iw = iw; g_ih = ih; }
        memset(g_img, 0, (size_t)iw * ih * 4);
        canvas cv = {g_img, iw, ih, k};
        uint8_t bg[4] = {0, 0, 0, (uint8_t)g_opacity}; fill_rrect(&cv, 0, 0, lw * k, lh * k, Radius * k, bg);
        panel(&cv, rows, n, &lw, &lh);
        changed = 1;
    }
    place(); g_shown = 1;
    *px = g_img; *w = g_iw; *h = g_ih; *x = g_px; *y = g_py;
    pthread_mutex_unlock(&g_mu);
    return changed;
}
void perf_ov_hidden(void) { pthread_mutex_lock(&g_mu); g_shown = 0; pthread_mutex_unlock(&g_mu); }

// ---- the window thread: F3, the mouse (window client pixels) ----
void perf_ov_toggle(void) { init_once(); pthread_mutex_lock(&g_mu); g_on = !g_on; g_shown = 0; g_dirty = 1; cfg_save(); pthread_mutex_unlock(&g_mu); }
void perf_ov_drag_distance(int d) { if (d > 0) g_drag_dist = d; }
static int hit(int x, int y) { return g_on > 0 && g_shown && x >= g_px && y >= g_py && x < g_px + g_pw && y < g_py + g_ph; }
int perf_ov_mouse_down(int x, int y) {   // 1: the panel took it (no touch)
    pthread_mutex_lock(&g_mu); int h = hit(x, y);
    if (h) { g_press = 1; g_dragging = 0; g_press_x = (float)x; g_press_y = (float)y; g_grab_x = (float)(x - g_px); g_grab_y = (float)(y - g_py); }
    pthread_mutex_unlock(&g_mu); return h;
}
int perf_ov_mouse_move(int x, int y) {
    pthread_mutex_lock(&g_mu); int h = g_press;
    if (h) {
        if (!g_dragging && fabsf(x - g_press_x) + fabsf(y - g_press_y) < g_drag_dist) { pthread_mutex_unlock(&g_mu); return 1; }   // QApplication::startDragDistance (manhattan)
        g_dragging = 1;
        float defx = g_sw - g_iw - Margin * g_k, defy = Margin * g_k;   // the default place
        g_dx = x - g_grab_x - defx; g_dy = y - g_grab_y - defy; place();
    }
    pthread_mutex_unlock(&g_mu); return h;
}
int perf_ov_mouse_up(int x, int y) {
    (void)x; (void)y;
    pthread_mutex_lock(&g_mu); int h = g_press;
    if (h && !g_dragging) { g_fps_only = !g_fps_only; g_width_floor = 0; g_dirty = 1; cfg_save(); }   // a click: FPS only, and back
    g_press = g_dragging = 0;
    pthread_mutex_unlock(&g_mu); return h;
}
int perf_ov_wheel(int x, int y, int delta) {   // delta: WHEEL_DELTA (120) units; touchpads send less, collected into whole notches
    static int acc;
    pthread_mutex_lock(&g_mu); int h = hit(x, y);
    if (h) { acc += delta; int steps = acc / 120;
             if (steps) { acc -= steps * 120; int o = g_opacity + steps * OpacityStep; g_opacity = o < MinOpacity ? MinOpacity : o > MaxOpacity ? MaxOpacity : o; g_dirty = 1; cfg_save(); } }
    pthread_mutex_unlock(&g_mu); return h;
}
void perf_ov_mouse_cancel(void) { pthread_mutex_lock(&g_mu); g_press = g_dragging = 0; pthread_mutex_unlock(&g_mu); }   // focus lost mid-press: neither a click nor a drag
