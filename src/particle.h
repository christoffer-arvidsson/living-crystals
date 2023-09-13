#ifndef PARTICLE_H
#define PARTICLE_H

#include "stdint.h"
#include "stdlib.h"
#include <cuda_runtime.h>

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
    float2 pos;  // micro meters
    float speed;  // micro meters/s
    float radius;
    ParticleType charge;
    float orient;
} Particle;

void clear_particles(void);
void push_particle(float2 pos, float speed, float orient, ParticleType charge, float radius);
void print_particle(Particle* particle);
unsigned int get_num_particles();
Particle* get_particle(unsigned int idx);

void init_simulation(void);
void tick_simulation(void);


#endif /* PARTICLE_H */
