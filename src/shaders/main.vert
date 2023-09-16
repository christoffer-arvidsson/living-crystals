#version 330 core

uniform vec2 resolution;
uniform float radius;

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

out vec3 particle_color;
out vec2 particle_center;
out float particle_radius;

vec2 screen_to_ndc(vec2 pos) {
    return (pos - resolution * 0.5) / (resolution * 0.5);
}

void main() {
    vec2 uv = vec2(
                   float(gl_VertexID & 1),
                   float((gl_VertexID >> 1) & 1));
    gl_Position = vec4(
                       screen_to_ndc(position + uv * radius * 2),
                       0.0,
                       1.0);
    particle_color = color;
    particle_center =  uv;
    particle_radius = radius;
}
