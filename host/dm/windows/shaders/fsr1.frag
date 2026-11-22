#version 450
// r121: AMD FidelityFX Super Resolution 1 (ffx_fsr1.h, ffx_a.h; MIT, see FSR1-LICENSE.txt) for the window present (gfx_vk_present.c), drawn as two fragment passes
// into colour attachments as Eden does (video_core/renderer_vulkan/present/fsr.cpp), in place of r116's compute passes into storage images (fsr1.comp).
// FSR_PASS 0 = EASU (upscale of the game's picture to the window's picture size), 1 = RCAS (sharpening at that size).
// pc.mode.x (EASU): 1 = the input is an sRGB image (sampling decodes to linear): the samples are encoded again, as FSR wants display-encoded input.
// pc.mode.x (RCAS): 1 = its output is decoded to linear again (the blit into the sRGB swapchain encodes it once). pc.mode.y (EASU): 1 = lower-left origin (rows mirrored)
#extension GL_GOOGLE_include_directive : require
layout(binding = 0) uniform sampler2D src;
layout(location = 0) out vec4 o;
layout(push_constant) uniform PC { uvec4 con0, con1, con2, con3; ivec4 mode; } pc;   // mode: z, w = the output size
#define A_GPU 1
#define A_GLSL 1
#include "ffx_a.h"
vec4 srgb_enc(vec4 c) { c = max(c, vec4(0.0)); return mix(c * 12.92, 1.055 * pow(c, vec4(1.0 / 2.4)) - 0.055, greaterThan(c, vec4(0.0031308))); }
vec3 srgb_dec(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045))); }
#if FSR_PASS == 0
#define FSR_EASU_F 1
AF4 FsrEasuRF(AF2 p) { AF4 v = textureGather(src, p, 0); return pc.mode.x != 0 ? srgb_enc(v) : v; }
AF4 FsrEasuGF(AF2 p) { AF4 v = textureGather(src, p, 1); return pc.mode.x != 0 ? srgb_enc(v) : v; }
AF4 FsrEasuBF(AF2 p) { AF4 v = textureGather(src, p, 2); return pc.mode.x != 0 ? srgb_enc(v) : v; }
#else
#define FSR_RCAS_F 1
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(src, clamp(p, ivec2(0), pc.mode.zw - 1), 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
#endif
#include "ffx_fsr1.h"
void main() {
    uvec2 p = uvec2(gl_FragCoord.xy);
#if FSR_PASS == 0
    uvec2 ip = pc.mode.y != 0 ? uvec2(p.x, uint(pc.mode.w) - 1u - p.y) : p;
    AF3 c; FsrEasuF(c, ip, pc.con0, pc.con1, pc.con2, pc.con3);
    o = vec4(c, 1.0);
#else
    AF3 c; FsrRcasF(c.r, c.g, c.b, p, pc.con0);
    if (pc.mode.x != 0) c = srgb_dec(clamp(c, 0.0, 1.0));
    o = vec4(c, 1.0);
#endif
}
