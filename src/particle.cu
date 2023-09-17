#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "constants.h"
#include "cuda_helpers.h"

extern "C" {
    #include "particle.h"
}

const size_t num_tiles_x = (SCREEN_WIDTH / TILE_SIZE) + 2U;
const size_t num_tiles_y = (SCREEN_HEIGHT / TILE_SIZE) + 2U;
const size_t num_tiles = num_tiles_x * num_tiles_y;
const size_t num_blocks = (PARTICLES_CAPACITY / 1024U) + 1U;

typedef struct {
    unsigned int tw = num_tiles_x;
    unsigned int th = num_tiles_y;
    unsigned int counts[num_tiles_y][num_tiles_x];
    unsigned int cells[num_tiles_y][num_tiles_x][MAX_PARTICLES_PER_CELL]; // H,W,P
} Grid;


void clear_particles(ParticleContainer* container) {
    container->size = 0;
}

void push_particle(ParticleContainer* container, const Particle* particle) {
    assert(container->size < PARTICLES_CAPACITY);
    size_t current = container->size;
    container->particles[current].pos = particle->pos;
    container->particles[current].velocity = particle->velocity;
    container->particles[current].orient = particle->orient;
    container->particles[current].charge = particle->charge;
    container->particles[current].radius = particle->radius;
    container->size += 1;
}


void print_particle(Particle* particle) {
    printf("x: %f y: %f v_x: %f v_y: %f rad: %f orient: %f\n",
           particle->pos.x,
           particle->pos.y,
           particle->velocity.x,
           particle->velocity.y,
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

// Interactions need to check the 9 neighboring tiles including the
// center for nearby particles. These are the offsets.
__device__ const int NEIGHBOR_OFFSET[9][2] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 0},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1},
};

__device__ void resolve_collisions(Particle* particles, unsigned int n_particles, Grid* grid) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    Particle* part_n = &particles[idx];
    uint2 tile_idx = get_tile_idx(part_n);
    float diameter = static_cast<float>(PARTICLE_RADIUS * 2U);

    for (unsigned int t_idx=0; t_idx<9; ++t_idx) {
        int tx = tile_idx.x + NEIGHBOR_OFFSET[t_idx][0];
        int ty = tile_idx.y + NEIGHBOR_OFFSET[t_idx][1];
        if (!(tx >= 0 && tx < grid->tw && ty >= 0 && ty < grid->th)) {
            continue;
        }
        unsigned int num_neighbors = grid->counts[ty][tx];
        for (unsigned int i=0; i < num_neighbors; ++i) {
            unsigned int neighbor = grid->cells[ty][tx][i];

            if (neighbor == idx) {
                continue;
            }
            Particle* part_i = &particles[neighbor];
            float2 dist_ni = cyclic_distance(part_n->pos, part_i->pos, (float)SCREEN_WIDTH, (float)SCREEN_HEIGHT);
            float squared_distance_ni = squared_norm(dist_ni);
            float distance_ni = sqrtf(squared_distance_ni);

            if (distance_ni < diameter) {
                float overlap = diameter - distance_ni;
                float2 unit_vec_i_to_n = dist_ni / distance_ni;
                part_i->pos = part_i->pos - (overlap / 2) * unit_vec_i_to_n;
                part_n->pos = part_n->pos + (overlap / 2) * unit_vec_i_to_n;
            }
        }
    }
}

__device__ float compute_torque(Particle* particles, unsigned int n_particles, Grid* grid) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    Particle* part_n = &particles[idx];
    uint2 tile_idx = get_tile_idx(part_n);

    float2 unit_vel_n = make_float2(cosf(part_n->orient), sinf(part_n->orient));
    float3 unit_vel_n3 = make_float3(unit_vel_n.x, unit_vel_n.y, 0.0f);
    float3 unit_z = make_float3(0.0f, 0.0f, 1.0f);

    float torque = 0.0f;
    for (unsigned int t_idx=0; t_idx<9; ++t_idx) {
        int tx = tile_idx.x + NEIGHBOR_OFFSET[t_idx][0];
        int ty = tile_idx.y + NEIGHBOR_OFFSET[t_idx][1];
        if (tx < 0) tx = grid->tw - 1;
        if (ty < 0) tx = grid->th - 1;
        if (tx > grid->tw) tx = 0;
        if (ty > grid->th) tx = 0;


        unsigned int num_neighbors = grid->counts[ty][tx];

        for (unsigned int i=0; i < num_neighbors; ++i) {
            unsigned int neighbor = grid->cells[ty][tx][i];

            if (neighbor == idx) {
                continue;
            }
            Particle* part_i = &particles[neighbor];
            float2 dist_ni = cyclic_distance(part_n->pos, part_i->pos, (float)SCREEN_WIDTH, (float)SCREEN_HEIGHT);
            float squared_distance_ni = squared_norm(dist_ni);
            float distance_ni = sqrtf(squared_distance_ni);

            // Torque
            if (distance_ni < INTR_CUTOFF) {
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
    }

    torque *= ATTR_STRENGTH;

    return torque;
}

__global__ void update_state(Particle* particles, unsigned int n_particles, curandState* curand_state, Grid* grid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_particles) {
        Particle* particle = &particles[idx];

        // precompute these
        const float sq_trans = sqrtf(2 * TRANS_COEFF * DELTA_T);
        const float sq_rot = sqrtf(2 * ROT_COEFF * DELTA_T);

        // normally distributed random numbers

        uint2 tile_idx = get_tile_idx(&particles[idx]);

        float weight_x = curand_normal(&(curand_state[idx]));
        float weight_y = curand_normal(&(curand_state[idx]));
        float weight_rot = curand_normal(&(curand_state[idx]));

        // dx(t)/dt = v * cos(theta(t)) + sqrt(2 * D_T) * W_x
        float diff_x = particle->velocity.x * cosf(particle->orient) * DELTA_T + sq_trans * weight_x;
        // dy(t)/dt = v * sin(theta(t)) + sqrt(2 * D_T) * W_y
        float diff_y = particle->velocity.y * sinf(particle->orient) * DELTA_T + sq_trans * weight_y;

        // dtheta(t)/dt = torque + sqrt(2 * D_R) * W_theta
        float torque = compute_torque(particles, n_particles, grid);
        float diff_orient = torque * DELTA_T + sq_rot * weight_rot;

        // Sync to do a synchronized state update
        __syncthreads();
        particles[idx].pos.x += diff_x;
        particles[idx].pos.y += diff_y;
        particles[idx].orient += diff_orient;
        __syncthreads();
        resolve_collisions(particles, n_particles, grid);

        // loop around
        particle->pos.x = fmod(particle->pos.x, (float)SCREEN_WIDTH);
        if (particle->pos.x < 0) {
            particle->pos.x += (float)SCREEN_WIDTH;
        }
        particle->pos.y = fmod(particle->pos.y, (float)SCREEN_HEIGHT);
        if (particle->pos.y < 0) {
            particle->pos.y += (float)SCREEN_HEIGHT;
        }
    }

}
Particle* d_particles;
curandState *d_state;
Grid* d_grid;

void init_simulation(ParticleContainer* container) {
    // Setup particles
    size_t size = container->size;
    cudaMalloc((void**)&d_particles, size * sizeof(Particle));
    cudaMemcpy(d_particles, container->particles, size * sizeof(Particle), cudaMemcpyHostToDevice);

    // Setup rng
    cudaMalloc(&d_state, size * sizeof(curandState));
    setup_rng<<<num_blocks,1024>>>(d_state, size);

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

void tick_simulation(ParticleContainer* container) {
    size_t size = container->size;
    if (num_tiles < 1024) {
        reset_grid_tiles_count<<<1,num_tiles>>>(d_grid);
    } else {
        reset_grid_tiles_count<<<(unsigned int)((num_tiles)/1024)+1,1024>>>(d_grid);
    }
    cudaDeviceSynchronize();

    update_grid_tiles<<<num_blocks,1024>>>(d_particles, size, d_grid);
    cudaDeviceSynchronize();

    update_state<<<num_blocks, 1024>>>(d_particles, size, d_state, d_grid);
    cudaDeviceSynchronize();

    cudaMemcpy(container->particles, d_particles, size * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

const Particle* get_particle(const ParticleContainer* container, size_t idx) {
    assert (idx < container->size);
    return &container->particles[idx];
}
