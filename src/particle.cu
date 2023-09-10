#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "particle.h"
#include "constants.h"

Particle particles[PARTICLES_CAPACITY];
size_t particles_count = 0;


inline __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}

inline __host__ __device__ float2 f3_to_f2(float3 vec) {
    return make_float2(vec.x, vec.y);
}

inline __host__ __device__ float3 f2_to_f3(float2 vec) {
    return make_float3(vec.x, vec.y, 0.0f);
}

__device__ float2 cyclic_distance(float2 a, float2 b, float width, float height) {
    float2 dist = a - b;
    float dx = fabsf(dist.x);
    if (dx > width) {
        dist.x = width / 2.0f;
    }
    float dy = fabsf(dist.y);
    if (dy > height) {
        dist.y = height / 2.0f;
    }
    return dist;
}

__device__ float dot_product2(float2 a, float2 b) {
    return (a.x * b.x) + (a.y * b.y);
}

__device__ float dot_product3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross_product(float3 a, float3 b) {
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__device__ float squared_norm(float2 vec) {
    return vec.x * vec.x + vec.y * vec.y;
}

__device__ float compute_torque(Particle* particles, size_t n_particles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    Particle* part_n = &particles[idx];

    const float attract_strength = ATTR_STRENGTH;
    const float r_c = INTR_CUTOFF;
    float2 unit_vel_n = make_float2(cosf(part_n->orient), sinf(part_n->orient));
    float3 unit_vel_n3 = make_float3(unit_vel_n.x, unit_vel_n.y, 0.0f);
    float3 unit_z = make_float3(0.0f, 0.0f, 1.0f);

    float torque = 0.0f;
    for (size_t i=0; i < n_particles; ++i) {
        if (i == idx) {
            continue;
        }
        Particle* part_i = &particles[i];
        float2 dist_ni = cyclic_distance(part_n->pos, part_i->pos, (float)SCREEN_WIDTH, (float)SCREEN_HEIGHT);
        float s_dist_ni = squared_norm(dist_ni);

        if (sqrtf(s_dist_ni) < r_c) {
            // (unit_vel_n dot dist_ni) / dist_ni**2) cross (dist_ni dot unit_z)
            float3 dist_ni_3 = make_float3(dist_ni.x, dist_ni.y, 0.0f);
            float sign;
            if (part_i->charge != part_n->charge) {
                sign = -1.0f;
            }
            else if (part_i->charge == ACTIVE && part_n->charge == ACTIVE) {
                sign = 1.0f;
            }
            else {
                sign = 0.0f;
            }

            torque = torque + sign * dot_product3(cross_product((dot_product2(unit_vel_n, dist_ni) / s_dist_ni) * unit_vel_n3, dist_ni_3), unit_z);
        }
    }

    torque *= attract_strength;

    return torque;
}

__global__ void setup_rng(curandState* state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(RNG_SEED, idx, 0, &state[idx]);
}

__global__ void update_state(Particle* particles, size_t n_particles, curandState* curand_state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles) {
        Particle particle = particles[idx];

        // precompute these
        const float trans_coeff = TRANS_COEFF;
        const float rot_coeff = ROT_COEFF;
        const float d_t = DELTA_T;
        const float sq_trans = sqrtf(2 * trans_coeff);
        const float sq_rot = sqrtf(2 * rot_coeff);

        // normally distributed random numbers
        float weight_x = curand_normal(&(curand_state[idx]));
        float weight_y = curand_normal(&(curand_state[idx]));
        float weight_rot = curand_normal(&(curand_state[idx]));

        // dx(t)/dt = v * cos(theta(t)) + sqrt(2 * D_T) * W_x
        float diff_x = particle.speed * cosf(particle.orient) + sq_trans * weight_x;
        // dy(t)/dt = v * sin(theta(t)) + sqrt(2 * D_T) * W_y
        float diff_y = particle.speed * sinf(particle.orient) + sq_trans * weight_y;

        // dtheta(t)/dt = torque + sqrt(2 * D_R) * W_theta
        float torque = compute_torque(particles, n_particles);
        float diff_orient = torque + sq_rot * weight_rot;

        // Sync to do a synchronized state update
        __syncthreads();
        particles[idx].pos.x += diff_x * d_t;
        particles[idx].pos.y += diff_y * d_t;
        particles[idx].orient += diff_orient * d_t;

        #ifdef LOOP_AROUND
        particle.pos.x = fmod(particle.pos.x, (float)SCREEN_WIDTH);
        if (particle.pos.x < 0) {
            particle.pos.x += (float)SCREEN_WIDTH;
        }
        particle.pos.y = fmod(particle.pos.y, (float)SCREEN_HEIGHT);
        if (particle.pos.y < 0) {
            particle.pos.y += (float)SCREEN_WIDTH;
        }
        #endif
    }

}

void clear_particles(void) {
    particles_count = 0;
}

void push_particle(float2 pos, float speed, float orient, ParticleType charge) {
    assert(particles_count < PARTICLES_CAPACITY);
    particles[particles_count].pos = pos;
    particles[particles_count].speed = speed;
    particles[particles_count].orient = orient;
    particles[particles_count].charge = charge;
    particles[particles_count].radius = 5.0;
    particles_count += 1;
}


void print_particle(Particle* particle) {
    printf("x: %f y: %f speed: %f rad: %f orient: %f\n",
           particle->pos.x,
           particle->pos.y,
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
    setup_rng<<<3,particles_count>>>(d_state);
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
