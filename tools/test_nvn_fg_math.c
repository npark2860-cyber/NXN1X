#include "../host/dm/windows/gfx_vk_fg_math.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    fg_matrix view = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 3, 4, 1}};
    fg_matrix p = {{1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1.01f, 1, 0, 0, -.101f, 0}};
    fg_matrix vp = fg_product(view, p), world = fg_inverse_rigid(view);
    float data[64] = {0};
    memcpy(data + 4, &view, 64);
    memcpy(data + 20, &p, 64);
    memcpy(data + 40, &vp, 64);
    fg_offsets o;
    assert(fg_find_camera(data, 64, &o) && o.view == 4 && o.projection == 20 && o.combined == 40);
    data[54] += 1;
    assert(!fg_find_camera(data, 64, &o));
    float packed[44];
    for (unsigned c = 0; c < 4; ++c)
        for (unsigned r = 0; r < 4; ++r) {
            if (c < 3) packed[c*4+r] = view.v[r*4+c];
            packed[12+c*4+r] = p.v[r*4+c];
            packed[28+c*4+r] = vp.v[r*4+c];
        }
    assert(fg_find_camera(packed, 44, &o) && o.transposed && o.projection == 12);
    assert(fg_matrix_match(view, fg_decode(packed, o.view, o.transposed, 1)));
    assert(!fg_find_camera(packed, 43, &o));
    fg_matrix stationary = fg_product(world, vp);
    float dx, dy;
    assert(fg_expected_motion(.3f, .4f, .909f, p, stationary, &dx, &dy));
    assert(fabsf(dx) + fabsf(dy) < 1e-5f);
    view.v[12] += .1f;
    view.v[13] += .2f;
    assert(fg_expected_motion(.3f, .4f, .909f, p, fg_product(world, fg_product(view, p)), &dx, &dy));
    assert(fg_close(dx, .05f) && fg_close(dy, -.2f));
    assert(!fg_expected_motion(.5f, .5f, 1, p, stationary, &dx, &dy));
    unsigned hits[12] = {0};
    assert(fg_winning_convention(hits, 0) < 0);
    for (unsigned c = 0; c < 12; ++c) {
        float sx, sy;
        fg_motion_units(c, 1920, 1080, &sx, &sy);
        assert(isfinite(sx) && isfinite(sy) && sx && sy);
        hits[c] = 13;
        assert(fg_winning_convention(hits, 16) == (int)c);
        hits[(c + 1) % 12] = 13;
        assert(fg_winning_convention(hits, 16) < 0);
        hits[c] = hits[(c + 1) % 12] = 0;
    }
    puts("PASS camera discovery, invalid products, static and moving reprojection, depth and motion "
         "ambiguity");
}
