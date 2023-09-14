#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "particle.h"
#include "constants.h"
#include "cuda_helpers.h"

Particle particles[PARTICLES_CAPACITY];
unsigned int particles_count = 0;
const unsigned int num_tiles_x = (SCREEN_WIDTH / TILE_SIZE) + 2U;
const unsigned int num_tiles_y = (SCREEN_HEIGHT / TILE_SIZE) + 2U;
const unsigned int num_tiles = num_tiles_x * num_tiles_y;
const unsigned int num_blocks = (PARTICLES_CAPACITY / 1024U) + 1U;

typedef struct {
    unsigned int tw = num_tiles_x;
    unsigned int th = num_tiles_y;
    unsigned int counts[num_tiles_y][num_tiles_x];
    unsigned int cells[num_tiles_y][num_tiles_x][MAX_PARTICLES_PER_CELL]; // H,W,P
} Grid;


void clear_particles(void) {
    particles_count = 0;
}

void push_particle(float2 pos, float speed, float orient, ParticleType charge, float radius) {
    assert(particles_count < PARTICLES_CAPACITY);
    particles[particles_count].pos = pos;
    particles[particles_count].speed = speed;
    particles[particles_count].orient = orient;
    particles[particles_count].charge = charge;
    particles[particles_count].radius = radius;
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

__global__ void setup_rng(curandState* state, unsigned int n_particles) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n_particles) {
        curand_init(RNG_SEED, idx, 0, &state[idx]);
    }
}

__global__ void reset_grid_tiles_count(Grid* grid) {
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx < grid->tw * grid->th) {
        unsigned int tile_x = tile_idx % grid->tw;
        unsigned int tile_y = tile_idx / grid->tw;

        grid->counts[tile_y][tile_x] = 0U;
    }
}

__device__ uint2 get_tile_idx(Particle* particle) {
    unsigned int tile_x = floorf(particle->pos.x / TILE_SIZE);
    unsigned int tile_y = floorf(particle->pos.y / TILE_SIZE);

    return make_uint2(tile_x, tile_y);
}

__global__ void update_grid_tiles(Particle* particles, unsigned int n_particles, Grid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_particles) {
        uint2 tile_idx = get_tile_idx(&particles[idx]);
        unsigned int old = atomicAdd(&grid->counts[tile_idx.y][tile_idx.x], 1U);
        grid->cells[tile_idx.y][tile_idx.x][old] = static_cast<unsigned int>(idx);
    }
}

__device__ void resolve_collisions(Particle* particles, unsigned int n_particles, Grid* grid) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    Particle* part_n = &particles[idx];
    uint2 tile_idx = get_tile_idx(part_n);
    unsigned int num_neighbors = grid->counts[tile_idx.y][tile_idx.x];
    for (unsigned int i=0; i < num_neighbors; ++i) {
        unsigned int neighbor = grid->cells[tile_idx.y][tile_idx.x][i];

        if (neighbor == idx) {
            continue;
        }
        Particle* part_i = &particles[neighbor];
        float2 dist_ni = cyclic_distance(part_n->pos, part_i->pos, (float)SCREEN_WIDTH, (float)SCREEN_HEIGHT);
        float squared_distance_ni = squared_norm(dist_ni);
        float distance_ni = sqrtf(squared_distance_ni);
        float diameter = static_cast<float>(PARTICLE_RADIUS * 2U);

        if (distance_ni < diameter) {
            float overlap = diameter - distance_ni;
            float2 unit_vec_i_to_n = dist_ni / distance_ni;
            part_i->pos = part_i->pos - (overlap / 2) * unit_vec_i_to_n;
            part_n->pos = part_n->pos + (overlap / 2) * unit_vec_i_to_n;
        }
    }
}

__device__ float compute_torque(Particle* particles, unsigned int n_particles, Grid* grid) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    Particle* part_n = &particles[idx];

    uint2 tile_idx = get_tile_idx(part_n);
    unsigned int num_neighbors = grid->counts[tile_idx.y][tile_idx.x];

    const float attract_strength = ATTR_STRENGTH;
    const float r_c = INTR_CUTOFF;
    float2 unit_vel_n = make_float2(cosf(part_n->orient), sinf(part_n->orient));
    float3 unit_vel_n3 = make_float3(unit_vel_n.x, unit_vel_n.y, 0.0f);
    float3 unit_z = make_float3(0.0f, 0.0f, 1.0f);

    float torque = 0.0f;
    for (unsigned int i=0; i < num_neighbors; ++i) {
        unsigned int neighbor = grid->cells[tile_idx.y][tile_idx.x][i];

        if (neighbor == idx) {
            continue;
        }
        Particle* part_i = &particles[neighbor];
        float2 dist_ni = cyclic_distance(part_n->pos, part_i->pos, (float)SCREEN_WIDTH, (float)SCREEN_HEIGHT);
        float squared_distance_ni = squared_norm(dist_ni);
        float distance_ni = sqrtf(squared_distance_ni);

        // Torque
        if (distance_ni < r_c) {
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

            torque = torque + sign *
            dot_product3(cross_product((dot_product2(unit_vel_n,
            dist_ni) / squared_distance_ni) * unit_vel_n3, dist_ni_3),
            unit_z);
        }
    }

    torque *= attract_strength;

    return torque;
}

__global__ void update_state(Particle* particles, unsigned int n_particles, curandState* curand_state, Grid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles) {
        Particle* particle = &particles[idx];

        // precompute these
        const float trans_coeff = TRANS_COEFF;
        const float rot_coeff = ROT_COEFF;
        const float d_t = DELTA_T;
        const float sq_trans = sqrtf(2 * trans_coeff);
        const float sq_rot = sqrtf(2 * rot_coeff);

        // normally distributed random numbers

        uint2 tile_idx = get_tile_idx(&particles[idx]);

        float weight_x = curand_normal(&(curand_state[idx]));
        float weight_y = curand_normal(&(curand_state[idx]));
        float weight_rot = curand_normal(&(curand_state[idx]));

        // dx(t)/dt = v * cos(theta(t)) + sqrt(2 * D_T) * W_x
        float diff_x = particle->speed * cosf(particle->orient) + sq_trans * weight_x;
        // dy(t)/dt = v * sin(theta(t)) + sqrt(2 * D_T) * W_y
        float diff_y = particle->speed * sinf(particle->orient) + sq_trans * weight_y;

        // dtheta(t)/dt = torque + sqrt(2 * D_R) * W_theta
        float torque = compute_torque(particles, n_particles, grid);
        float diff_orient = torque + sq_rot * weight_rot;

        // Sync to do a synchronized state update
        __syncthreads();
        particles[idx].pos.x += diff_x * d_t;
        particles[idx].pos.y += diff_y * d_t;
        particles[idx].orient += diff_orient * d_t;
        __syncthreads();
        resolve_collisions(particles, n_particles, grid);

        #ifdef LOOP_AROUND
        particle->pos.x = fmod(particle->pos.x, (float)SCREEN_WIDTH);
        if (particle->pos.x < 0) {
            particle->pos.x += (float)SCREEN_WIDTH;
        }
        particle->pos.y = fmod(particle->pos.y, (float)SCREEN_HEIGHT);
        if (particle->pos.y < 0) {
            particle->pos.y += (float)SCREEN_HEIGHT;
        }
        #endif
    }

}
Particle* d_particles;
curandState *d_state;
Grid* d_grid;

void init_simulation(void) {
    // Setup particles
    cudaMalloc((void**)&d_particles, particles_count * sizeof(Particle));
    cudaMemcpy(d_particles, particles, particles_count * sizeof(Particle), cudaMemcpyHostToDevice);

    // Setup rng
    cudaMalloc(&d_state, particles_count * sizeof(curandState));
    setup_rng<<<num_blocks,1024>>>(d_state, particles_count);

    // Setup grid
    Grid grid = {
        .tw = num_tiles_x,
        .th = num_tiles_y
    };

    cudaMalloc((void**)&d_grid, sizeof(Grid));
    cudaMemcpy(d_grid, &grid, sizeof(grid), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if (num_tiles < 1024) {
        reset_grid_tiles_count<<<1,num_tiles>>>(d_grid);
    } else {
        reset_grid_tiles_count<<<(unsigned int)((num_tiles)/1024)+1,1024>>>(d_grid);
    }
    cudaDeviceSynchronize();
}
void tick_simulation(void) {
    if (num_tiles < 1024) {
        reset_grid_tiles_count<<<1,num_tiles>>>(d_grid);
    } else {
        reset_grid_tiles_count<<<(unsigned int)((num_tiles)/1024)+1,1024>>>(d_grid);
    }
    cudaDeviceSynchronize();

    update_grid_tiles<<<num_blocks,1024>>>(d_particles, particles_count, d_grid);
    cudaDeviceSynchronize();

    update_state<<<num_blocks, 1024>>>(d_particles, particles_count, d_state, d_grid);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, d_particles, particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

Particle* get_particle(unsigned int idx) {
    assert (idx < particles_count);
    return &particles[idx];
}

unsigned int get_num_particles(void) {
    return particles_count;
}
