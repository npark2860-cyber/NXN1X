#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    void *image, *memory, *view;
    uint32_t width, height, format, layout, usage, flags;
    uint32_t x, y;
} nvn_fg_image;

int nvn_sr_evaluate(void *cmd, const nvn_fg_image images[6], uint32_t iw, uint32_t ih,
                    uint32_t ow, uint32_t oh, const float mv[2], const float jitter[2],
                    float exposure, uint32_t flags, int reset, uint64_t network);
int nvn_fg_collecting(void);
int nvn_dlss_sr_wanted(void);   // dlsstoggle (nvn_streamline.cpp): Settings::dlss_upscale
int nvn_sr_fit(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, uint32_t out[2]);   // srfit (nvn_streamline.cpp): the size DLSS is given for a render size
typedef struct {
    uint64_t vertex, fragment;
    uint32_t camera_ubo, camera_to_world, view_to_clip, motion_attachment;
    float mv_scale_x, mv_scale_y;
} nvn_fg_profile;
const nvn_fg_profile *nvn_fg_get_profile(void);
void nvn_fg_camera(const float *world, const float *projection);
void nvn_fg_seal(uint64_t ticket);
void nvn_fg_tag(void *cmd, const nvn_fg_image *depth, const nvn_fg_image *motion,
                uint32_t width, uint32_t height, const float *scale, const float *jitter, int reset,
                int explicit_inputs);
#ifdef __cplusplus
}
#endif
