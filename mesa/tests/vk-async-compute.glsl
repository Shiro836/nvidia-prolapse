// Shaders of vk-async-compute.c; split by the build line with -DSTAGE_*.
#version 450
#if defined(STAGE_SPIN)
// Each invocation runs `iters` LCG steps from `seed + index` and stores it.
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) buffer Out { uint v[]; } o;
layout(push_constant) uniform P { uint iters; uint seed; } p;
void main() {
   uint i = gl_GlobalInvocationID.x;
   uint x = p.seed + i;
   for (uint k = 0u; k < p.iters; k++)
      x = x * 1664525u + 1013904223u;
   o.v[i] = x;
}
#elif defined(STAGE_CHAIN)
// One ping-pong step: v[dst] = v[src] + 1, plus `iters` LCG steps of busy work stored at dst + 1024.
layout(local_size_x = 1) in;
layout(std430, set = 0, binding = 0) buffer Out { uint v[]; } o;
layout(push_constant) uniform P { uint src; uint dst; uint iters; } p;
void main() {
   uint x = o.v[p.src];
   uint y = x;
   for (uint k = 0u; k < p.iters; k++)
      y = y * 1664525u + 1013904223u;
   o.v[p.dst] = x + 1u;
   o.v[p.dst + 1024u] = y; /* keeps the busy work */
}
#elif defined(STAGE_VERT)
void main() {
   vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
#elif defined(STAGE_FRAG)
// Busy fragment shader: `iters` LCG steps per fragment.
layout(location = 0) out uvec4 color;
layout(push_constant) uniform P { uint iters; } p;
void main() {
   uint x = uint(gl_FragCoord.x) * 4096u + uint(gl_FragCoord.y);
   for (uint k = 0u; k < p.iters; k++)
      x = x * 1664525u + 1013904223u;
   color = uvec4(x, 0u, 0u, 1u);
}
#endif
