#version 450
/* Shader of vk-cs-join-reuse.c: one read-modify-write per invocation on slice `slice` of A, after
 * spinning `spin` iterations, so a dispatch that is not ordered behind the previous one on the same
 * slice loses increments.
 */
layout(local_size_x = 64) in;
layout(push_constant) uniform PC { uint slice; uint spin; uint n; } pc;
layout(std430, set = 0, binding = 0) buffer A { uint v[]; } a;
void main() {
   uint i = pc.slice * pc.n + gl_GlobalInvocationID.x;
   uint x = a.v[i];
   uint acc = x ^ gl_GlobalInvocationID.x;
   for (uint k = 0u; k < pc.spin; k++)
      acc = acc * 1664525u + 1013904223u;
   /* acc | 1 is never 0: the write is x + 1, but only after the loop. */
   a.v[i] = x + 1u + ((acc | 1u) == 0u ? 1u : 0u);
}
