#version 450
layout(push_constant) uniform Parameters { uint phase; } parameters;
layout(location=0) out vec4 color;
void main() {
    uvec2 pixel = uvec2(gl_FragCoord.xy);
    color = vec4(float((pixel.x + parameters.phase) & 1u),
                 float((pixel.y + parameters.phase) & 1u), 0.25, 1.0);
}
