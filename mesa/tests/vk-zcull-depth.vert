#version 450
// Axis-aligned rectangle at a constant depth: rect = (x0, y0, x1, y1) in NDC.
layout(push_constant) uniform PC { vec4 rect; vec4 color; float z; } pc;
void main() {
   vec2 c = vec2((gl_VertexIndex & 1) != 0 ? pc.rect.z : pc.rect.x,
                 (gl_VertexIndex & 2) != 0 ? pc.rect.w : pc.rect.y);
   gl_Position = vec4(c, pc.z, 1.0);
}
