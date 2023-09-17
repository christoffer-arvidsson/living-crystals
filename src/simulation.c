#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "particle.h"
#include "constants.h"
#include "renderer.h"

float3 particle_type_to_color(ParticleCharge type) {
    switch (type) {
    case ACTIVE:
        return make_float3(0.8f, 0.2f, 0.4f);
    case PASSIVE:
        return make_float3(0.3f, 0.3f, 0.3f);
    default:
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

void setup_particles(ParticleContainer* container, size_t n_particles) {
    clear_particles(container);

    for (size_t p=0; p<n_particles; ++p) {
        float pos_x = (float)rand()/(float)(RAND_MAX/SCREEN_WIDTH);
        float pos_y = (float)rand()/(float)(RAND_MAX/SCREEN_HEIGHT);
        float speed = (float)rand()/(float)(RAND_MAX);
        float orient = (float)rand()/((float)(RAND_MAX)/(3.14f * 2.0f));
        ParticleCharge charge = PASSIVE;
        if ((float)rand()/((float)(RAND_MAX)) < (float)ACTIVE_FRACTION) {
            speed += PARTICLE_SPEED;
            charge = ACTIVE;
        }
        else {
            speed = 0.0f;
        }
        const Particle particle = {
            .pos = make_float2(pos_x, pos_y),
            .velocity = make_float2(speed, speed),
            .radius = PARTICLE_RADIUS,
            .charge = charge,
            .orient = orient,
        };
        push_particle(container, &particle);
    }
}

void particles_to_entities(ParticleContainer* container, Renderer* renderer) {
    for (size_t p = 0; p < container->size; ++p) {
        const Particle* part = get_particle(container, p);

        float3 color = particle_type_to_color(part->charge);
        Entity entity = {
            .x = part->pos.x,
            .y = part->pos.y,
            .r = color.x,
            .g = color.y,
            .b = color.z
        };
        renderer_add_entity(renderer, &entity);
    }
}

int main() {
    Renderer* renderer = renderer_alloc();

    ParticleContainer particles = {0};
    setup_particles(&particles, PARTICLES_CAPACITY);
    init_simulation(&particles);

    while(!glfwWindowShouldClose(renderer->window)) {
        tick_simulation(&particles);
        renderer_clear_entities(renderer);
        particles_to_entities(&particles, renderer);
        renderer_render(renderer);
        renderer_poll(renderer);
    }

    renderer_free(renderer);

    return 0;
}
