#ifndef PARTICLE_H
#define PARTICLE_H

#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "constants.h"


typedef struct {
    unsigned int steps;
    float delta_t;
    float trans_coeff; // 0.22,
    float rot_coeff; // 0.16,
    float r_c;
    float initial_speed;
} Params;

typedef enum {
    ACTIVE = 0,
    PASSIVE
} ParticleType;

typedef struct {
    float2 pos;
    float2 velocity;
    float radius;
    ParticleType charge;
    float orient;
} Particle;

typedef struct {
    Particle particles[PARTICLES_CAPACITY];
    size_t particles_count;
} ParticleContainer;

void clear_particles(ParticleContainer* container);
void push_particle(ParticleContainer* container, float2 pos, float2 speed, float orient, ParticleType charge, float radius);
void print_particle(Particle* particle);

unsigned int get_num_particles(ParticleContainer* container);
const Particle* get_particle(const ParticleContainer* container, unsigned int idx);

void init_simulation(ParticleContainer* container);
void tick_simulation(ParticleContainer* container);


#endif /* PARTICLE_H */
