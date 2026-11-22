#pragma once
#include <math.h>
#include <string.h>

// Row-vector matrices. Confirm the redundant view * projection matrix before
// interpreting a rigid transform in a game constant buffer as a view.
typedef struct {
    float v[16];
} fg_matrix;
static int fg_close(float a, float b) {
    return isfinite(a) && isfinite(b) && fabsf(a - b) <= 0.00015f * (1.f + fabsf(b));
}
static int fg_projection(const fg_matrix *p) {
    const float *m = p->v;
    if (!isfinite(m[0]) || !isfinite(m[5]) || fabsf(m[0]) < .1f || fabsf(m[5]) < .1f ||
        !fg_close(fabsf(m[11]), 1) || !isfinite(m[10]) || !isfinite(m[14]) || !m[14])
        return 0;
    for (unsigned i = 0; i < 16; ++i)
        if (i != 0 && i != 5 && i != 10 && i != 11 && i != 14 && !fg_close(m[i], 0))
            return 0;
    return 1;
}
static int fg_rigid(const fg_matrix *m) {
    for (unsigned i = 0; i < 3; ++i) {
        if (!fg_close(m->v[4 * i + 3], 0) || !isfinite(m->v[12 + i]))
            return 0;
        for (unsigned j = 0; j <= i; ++j) {
            float dot = 0;
            for (unsigned k = 0; k < 3; ++k)
                dot += m->v[4 * i + k] * m->v[4 * j + k];
            if (!fg_close(dot, (float)(i == j)))
                return 0;
        }
    }
    return fg_close(m->v[15], 1);
}
static fg_matrix fg_product(fg_matrix a, fg_matrix b) {
    fg_matrix r = {0};
    for (unsigned i = 0; i < 4; ++i)
        for (unsigned j = 0; j < 4; ++j)
            for (unsigned k = 0; k < 4; ++k)
                r.v[4 * i + j] += a.v[4 * i + k] * b.v[4 * k + j];
    return r;
}
static fg_matrix fg_inverse_rigid(fg_matrix view) {
    fg_matrix w = {0};
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            w.v[i * 4 + j] = view.v[j * 4 + i];
            w.v[12 + j] -= view.v[12 + i] * w.v[i * 4 + j];
        }
    w.v[15] = 1;
    return w;
}
typedef struct {
    unsigned view, projection, combined, transposed;
} fg_offsets;
static fg_matrix fg_matrix_at(const float *data, unsigned offset) {
    fg_matrix m;
    memcpy(&m, data + offset, sizeof m);
    return m;
}
static int fg_matrix_match(fg_matrix a, fg_matrix b) {
    for (unsigned i = 0; i < 16; ++i)
        if (!fg_close(a.v[i], b.v[i]))
            return 0;
    return 1;
}
// Some uniform blocks hold three float4 columns of an affine view matrix.
static fg_matrix fg_decode(const float *data, unsigned offset, unsigned transposed, int affine) {
    if (!transposed)
        return fg_matrix_at(data, offset);
    fg_matrix m = {0};
    m.v[15] = 1;
    for (unsigned col = 0; col < (affine ? 3u : 4u); ++col)
        for (unsigned row = 0; row < 4; ++row)
            m.v[row * 4 + col] = data[offset + col * 4 + row];
    return m;
}
static int fg_find_camera(const float *data, unsigned count, fg_offsets *result) {
    int found = 0;
    fg_matrix accepted_view = {0}, accepted_projection = {0};
    for (unsigned t = 0; t < 2; ++t) {
        unsigned vn = t ? 12 : 16;
        for (unsigned p = 0; p + 16 <= count; p += 4) {
            fg_matrix projection = fg_decode(data, p, t, 0);
            if (!fg_projection(&projection)) continue;
            for (unsigned v = 0; v + vn <= count; v += 4) {
                if (v < p + 16 && p < v + vn) continue;
                fg_matrix view = fg_decode(data, v, t, 1);
                if (!fg_rigid(&view)) continue;
                fg_matrix combined = fg_product(view, projection);
                if (fg_matrix_match(combined, projection)) continue;
                for (unsigned c = 0; c + 16 <= count; c += 4) {
                    if ((c < v + vn && v < c + 16) || (c < p + 16 && p < c + 16)) continue;
                    if (!fg_matrix_match(fg_decode(data, c, t, 0), combined)) continue;
                    if (found && (!fg_matrix_match(view, accepted_view) ||
                                  !fg_matrix_match(projection, accepted_projection))) return 0;
                    *result = (fg_offsets){v, p, c, t};
                    accepted_view = view;
                    accepted_projection = projection;
                    found = 1;
                }
            }
        }
    }
    return found;
}
// Reproject a stored device depth at a pixel into the previous camera. UV
// coordinates have their origin at the top left. Perspective only.
static int fg_expected_motion(float u, float v, float z, fg_matrix p, fg_matrix world_to_previous_clip,
                              float *dx, float *dy) {
    float denom = z * p.v[11] - p.v[10];
    if (fabsf(denom) < 1e-7f || !(z > 0 && z < 1))
        return 0;
    float vz = p.v[14] / denom, w = vz * p.v[11];
    float pos[4] = {(2 * u - 1) * w / p.v[0], (1 - 2 * v) * w / p.v[5], vz, 1}, clip[4] = {0};
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned k = 0; k < 4; ++k)
            clip[j] += pos[k] * world_to_previous_clip.v[4 * k + j];
    if (!(w > 0 && clip[3] > 0))
        return 0;
    *dx = .5f * (clip[0] / clip[3] + 1) - u;
    *dy = .5f * (1 - clip[1] / clip[3]) - v;
    return isfinite(*dx) && isfinite(*dy);
}

static void fg_motion_units(unsigned c, unsigned w, unsigned h, float *x, float *y) {
    // Normalized UV, pixels, or NDC deltas, with independently inverted axes.
    *x = (c / 4 == 0 ? 1.f : c / 4 == 1 ? 1.f / w : .5f) * (c & 1 ? -1.f : 1.f);
    *y = (c / 4 == 0 ? 1.f : c / 4 == 1 ? 1.f / h : .5f) * (c & 2 ? -1.f : 1.f);
}
static int fg_winning_convention(const unsigned hits[12], unsigned useful) {
    unsigned best = 0, runner = 0;
    for (unsigned c = 1; c < 12; ++c) {
        if (hits[c] > hits[best]) {
            runner = hits[best];
            best = c;
        } else if (hits[c] > runner)
            runner = hits[c];
    }
    return useful >= 8 && hits[best] * 4 >= useful * 3 && hits[best] >= runner + 3 ? (int)best : -1;
}
