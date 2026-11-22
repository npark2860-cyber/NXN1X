#version 450
// r121: the full-target triangle of the FSR 1 fragment passes (fsr1.frag)
void main() { vec2 v = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2)); gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0); }
