#version 450
// r76: the clear colour to every location (the pipeline's write masks pick the target), uvec4 outputs
layout(push_constant) uniform P { uvec4 c; float z; } p;
layout(location = 0) out uvec4 o0;
layout(location = 1) out uvec4 o1;
layout(location = 2) out uvec4 o2;
layout(location = 3) out uvec4 o3;
layout(location = 4) out uvec4 o4;
layout(location = 5) out uvec4 o5;
layout(location = 6) out uvec4 o6;
layout(location = 7) out uvec4 o7;
void main() { uvec4 v = p.c; o0 = v; o1 = v; o2 = v; o3 = v; o4 = v; o5 = v; o6 = v; o7 = v; }
