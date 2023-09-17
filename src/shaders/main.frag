#version 330 core
in vec3 particle_color;
in vec2 particle_center;
in float particle_radius;
void main() {
    vec2 temp = particle_center - vec2(0.5);
    float f = dot(temp, temp);
    if (f>0.25) {
        discard;
    } else {
        gl_FragColor = vec4(particle_color, 1.0);
    }
}
