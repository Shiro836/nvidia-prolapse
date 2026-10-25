#version 450
/* Shaders of vk-barrier-order.c, one per -D: QUAD (vertex), FILL / SAMPLE / SSBO (fragment),
 * PRODUCE / CONSUME (compute). The producers spin for `spin` iterations before they write, so a
 * consumer that is not ordered behind them reads the previous round's values.
 */
layout(push_constant) uniform PC { uint value; uint spin; } pc;

#ifdef QUAD
void main() {
   vec2 p = vec2((gl_VertexIndex & 1) * 2 - 1, (gl_VertexIndex >> 1) * 2 - 1);
   gl_Position = vec4(p, 0, 1);
}
#endif

#ifdef FILL
layout(location = 0) out uvec4 o;
void main() {
   uint acc = uint(gl_FragCoord.x) ^ (uint(gl_FragCoord.y) << 16);
   for (uint i = 0u; i < pc.spin; i++)
      acc = acc * 1664525u + 1013904223u;
   /* acc | 1 is never 0: the write is pc.value, but only after the loop. */
   o = uvec4(pc.value + ((acc | 1u) == 0u ? 1u : 0u), 0, 0, 0);
}
#endif

#ifdef SAMPLE
layout(set = 0, binding = 0) uniform usampler2D src;
layout(location = 0) out uvec4 o;
void main() {
   o = uvec4(texelFetch(src, ivec2(gl_FragCoord.xy), 0).x, 0, 0, 0);
}
#endif

#ifdef SSBO
layout(std430, set = 0, binding = 0) readonly buffer B { uint v[]; } b;
layout(location = 0) out uvec4 o;
void main() {
   uint i = uint(gl_FragCoord.y) * 256u + uint(gl_FragCoord.x);
   o = uvec4(b.v[i], 0, 0, 0);
}
#endif

#ifdef PRODUCE
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) writeonly buffer B { uint v[]; } b;
void main() {
   uint acc = gl_GlobalInvocationID.x;
   for (uint i = 0u; i < pc.spin; i++)
      acc = acc * 1664525u + 1013904223u;
   b.v[gl_GlobalInvocationID.x] = pc.value + gl_GlobalInvocationID.x + ((acc | 1u) == 0u ? 1u : 0u);
}
#endif

#ifdef CONSUME
layout(local_size_x = 64) in;
layout(std430, set = 0, binding = 0) readonly buffer B { uint v[]; } b;
layout(std430, set = 0, binding = 1) writeonly buffer O { uint v[]; } o;
void main() {
   o.v[gl_GlobalInvocationID.x] = b.v[gl_GlobalInvocationID.x] + 1u;
}
#endif
