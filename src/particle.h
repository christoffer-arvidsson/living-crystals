#ifndef PARTICLE_H
#define PARTICLE_H

#include "stdint.h"
#include "stdlib.h"

typedef struct {
    uint32_t steps;
    float delta_t;
    float trans_coeff; // 0.22,
    float rot_coeff; // 0.16,
    float initial_speed;

} Params;

typedef enum {
    ACTIVE = 0,
    PASSIVE
} ParticleType;

typedef struct {
    float p_x;  // micro meters
    float p_y;  // micro meters
    float speed;  // micro meters/s
    float radius;
    ParticleType charge;
    float orient;
} Particle;

void clear_particles(void);
void push_particle(float p_x, float p_y, float speed, float orient);
void print_particle(Particle* particle);
size_t get_num_particles();
Particle* get_particle(size_t idx);

void init_simulation(void);
void tick_simulation(void);


#endif /* PARTICLE_H */
