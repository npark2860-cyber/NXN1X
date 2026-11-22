#version 450
// r76: the clear colour to every location (the pipeline's write masks pick the target), vec4 outputs
layout(push_constant) uniform P { uvec4 c; float z; } p;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(location = 2) out vec4 o2;
layout(location = 3) out vec4 o3;
layout(location = 4) out vec4 o4;
layout(location = 5) out vec4 o5;
layout(location = 6) out vec4 o6;
layout(location = 7) out vec4 o7;
void main() { vec4 v = uintBitsToFloat(p.c); o0 = v; o1 = v; o2 = v; o3 = v; o4 = v; o5 = v; o6 = v; o7 = v; }
