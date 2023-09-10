#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "particle.h"

#define RNG_SEED 1234

#define LOOP_AROUND true

#define PARTICLES_CAPACITY 64
Particle particles[PARTICLES_CAPACITY];
size_t particles_count = 0;

__global__ void setup_rng(curandState* state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(RNG_SEED, idx, 0, &state[idx]);
}

__global__ void update_state(Particle* particles, size_t n_particles, curandState* curand_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles) {
        Particle particle = particles[idx];

        // precompute these
        const float trans_coeff = 0.22;
        const float rot_coeff = 0.16;
        const float d_t = 0.1;
        const float sq_trans = sqrtf(2 * trans_coeff);
        const float sq_rot = sqrtf(2 * rot_coeff);

        // normally distributed random numbers
        float weight_x = curand_normal(curand_state);
        float weight_y = curand_normal(curand_state);
        float weight_rot = curand_normal(curand_state);

        // dx(t)/dt = v * cos(theta(t)) + sqrt(2 * D_T) * W_x
        float diff_x = particle.speed * cosf(particle.orient) + sq_trans * weight_x;
        // dy(t)/dt = v * sin(theta(t)) + sqrt(2 * D_T) * W_y
        float diff_y = particle.speed * sinf(particle.orient) + sq_trans * weight_y;
        // dtheta(t)/dt = sqrt(2 * D_R) * W_theta
        float diff_orient = sq_rot * weight_rot;

        particles[idx].p_x += diff_x * d_t;
        particles[idx].p_y += diff_y * d_t;
        particles[idx].orient += diff_orient * d_t;

        #ifdef LOOP_AROUND
        particles[idx].p_x = fmod(particles[idx].p_x, 800.0f);
        particles[idx].p_y = fmod(particles[idx].p_y, 600.0f);
        #endif
    }

}

void clear_particles(void) {
    particles_count = 0;
}

void push_particle(float p_x, float p_y, float speed, float orient) {
    assert(particles_count < PARTICLES_CAPACITY);
    particles[particles_count].p_x = p_x;
    particles[particles_count].p_y = p_y;
    particles[particles_count].speed = speed;
    particles[particles_count].orient = 0.0;
    particles[particles_count].charge = ACTIVE;
    particles[particles_count].radius = 5.0;
    particles_count += 1;
}


void print_particle(Particle* particle) {
    printf("x: %f y: %f speed: %f rad: %f orient: %f\n",
           particle->p_x,
           particle->p_y,
           particle->speed,
           particle->radius,
           particle->orient);
}

Particle* d_particles;
curandState *d_state;

void init_simulation(void) {
    // Setup particles
    cudaMalloc((void**)&d_particles, particles_count * sizeof(Particle));
    cudaMemcpy(d_particles, particles, particles_count * sizeof(Particle), cudaMemcpyHostToDevice);

    // Setup rng
    cudaMalloc(&d_state, sizeof(curandState));
    setup_rng<<<1,1>>>(d_state);
}
void tick_simulation(void) {
    update_state<<<1, particles_count>>>(d_particles, particles_count, d_state);
    cudaMemcpy(particles, d_particles, particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

Particle* get_particle(size_t idx) {
    assert (idx < particles_count);
    return &particles[idx];
}

size_t get_num_particles(void) {
    return particles_count;
}

int main_2() {
    // Particles
    clear_particles();
    push_particle(50.0, 50.0, 0.0, 0.0);
    push_particle(-50.0, 50.0, 1.0, 0.0);
    push_particle(-50.0, -50.0, 2.0, 0.0);
    push_particle(50.0, -50.0, 3.0, 0.0);

    init_simulation();

    for (size_t step=0; step < 100; ++step) {
        tick_simulation();
    }

    return 0;
}
