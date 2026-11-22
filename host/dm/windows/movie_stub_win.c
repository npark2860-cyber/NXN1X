// Windows port, stage 2: the movie decoders of hle_movie_avc.h and hle_movie_av1.h without a decoder behind them (VideoToolbox is Apple's).
// AVC: every access unit decodes to a black picture at its time, so a movie plays for its length and ends as it would; AV1: nothing to show.
#include "hle.h"
#include "hle_movie_avc.h"
#include "hle_movie_av1.h"
#include <stdlib.h>
#include <string.h>

#define MAVC_Q 64
struct mavc { int64_t q[MAVC_Q]; int head, n, rgba; };
mavc *mavc_new(void) { mavc *a = calloc(1, sizeof *a); if (!a) die("movie: AVC decoder allocation failed"); return a; }
void mavc_free(mavc *a) { free(a); }
int mavc_bt709(const mavc *a, int h) { (void)a; return h >= 720; }   // r48: hle_movie_avc.c's rule when the stream does not declare its matrix
void mavc_param_sets(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
void mavc_avcc(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
int mavc_decode(mavc *a, const uint8_t *p, int n, int64_t pts_us) {
    (void)p; if (n <= 0) return 0;
    if (a->n == MAVC_Q) { a->head = (a->head + 1) % MAVC_Q; a->n--; }   // the oldest picture was never taken
    int k = a->head, i = a->n; while (i > 0 && a->q[(k + i - 1) % MAVC_Q] > pts_us) { a->q[(k + i) % MAVC_Q] = a->q[(k + i - 1) % MAVC_Q]; i--; }   // presentation order
    a->q[(k + i) % MAVC_Q] = pts_us; a->n++;
    return 0;
}
int mavc_take(mavc *a, int flush, uint8_t *dst, int w, int h, int64_t *pts_us) {
    if (!a->n || (!flush && a->n < 2)) return 0;   // as a decoder with a reorder depth of one
    *pts_us = a->q[a->head]; a->head = (a->head + 1) % MAVC_Q; a->n--;
    if (dst && w > 0 && h > 0) {
        if (a->rgba) { for (size_t i = 0; i < (size_t)w * h; i++) { dst[4 * i] = dst[4 * i + 1] = dst[4 * i + 2] = 0; dst[4 * i + 3] = 255; } }
        else { memset(dst, 16, (size_t)w * h); memset(dst + (size_t)w * h, 128, (size_t)w * ((h + 1) / 2)); }   // NV12 black (video range)
    }
    return 1;
}
int mavc_queued(const mavc *a) { return a->n; }
void mavc_set_rgba(mavc *a) { a->rgba = 1; }
void mavc_drop(mavc *a) { a->head = a->n = 0; }

struct mav1 { int unused; };
mav1 *mav1_new(void) { mav1 *a = calloc(1, sizeof *a); if (!a) die("movie: AV1 decoder allocation failed"); return a; }
void mav1_free(mav1 *a) { free(a); }
void mav1_reset(mav1 *a) { (void)a; }
int mav1_decode(mav1 *a, const uint8_t *data, size_t n, mav1_picture *out) { (void)a; (void)data; (void)n; (void)out; return 0; }
const char *mav1_backend(const mav1 *a) { (void)a; return "none (Windows stage 2)"; }
