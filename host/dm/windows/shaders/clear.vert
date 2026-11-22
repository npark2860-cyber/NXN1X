#version 450
// r76: a full-target triangle for clears that keep a write mask (gfx_vk_exec.c clear_draw)
layout(push_constant) uniform P { uvec4 c; float z; } p;
void main() { vec2 v = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2)); gl_Position = vec4(v * 2.0 - 1.0, p.z, 1.0); }
