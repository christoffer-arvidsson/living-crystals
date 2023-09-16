#ifndef PARTICLE_H
#define PARTICLE_H

#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "constants.h"

typedef enum {
    ACTIVE = 0,
    PASSIVE
} ParticleCharge;

typedef struct {
    float2 pos;
    float2 velocity;
    float radius;
    ParticleCharge charge;
    float orient;
} Particle;

typedef struct {
    Particle particles[PARTICLES_CAPACITY];
    size_t particles_count;
} ParticleContainer;

void clear_particles(ParticleContainer* container);
void push_particle(ParticleContainer* container, const Particle* partilcle);
void print_particle(const Particle* particle);

size_t get_num_particles(ParticleContainer* container);
const Particle* get_particle(const ParticleContainer* container, size_t idx);

void init_simulation(ParticleContainer* container);
void tick_simulation(ParticleContainer* container);


#endif /* PARTICLE_H */
