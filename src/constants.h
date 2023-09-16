#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <math.h>

#define RNG_SEED 1234

// From the paper:
// unit mu m
#define PARTICLE_RADIUS 4.0f
// unit J*K^-1
#define BOLTZMANN_CONSTANT 1.380649 * pow(10.0f, -23.0f)
// unit K
#define TEMPERATURE_KELVIN 300.0f
// unit Ns/m^2 (water)
#define FLUID_VISCOSITY 0.001f
// unit mu m^2/s
#define TRANS_COEFF BOLTZMANN_CONSTANT * TEMPERATURE_KELVIN / \
                        (6.0f * M_PI * FLUID_VISCOSITY * PARTICLE_RADIUS)
// unit rad^2/s
#define ROT_COEFF BOLTZMANN_CONSTANT * TEMPERATURE_KELVIN / \
    (8.0f * M_PI * FLUID_VISCOSITY * PARTICLE_RADIUS * PARTICLE_RADIUS * PARTICLE_RADIUS)

// ---PARAMETERS---
// Tunnels
#define PARTICLES_CAPACITY 1024*16
#define MAX_PARTICLES_PER_CELL 128
#define TILE_SIZE 32
#define DELTA_T 0.01f
#define INTR_CUTOFF 30.0f
#define ATTR_STRENGTH 25.0f
#define ACTIVE_FRACTION 0.1f
#define PARTICLE_SPEED 100.0f

// Clusters
/* #define PARTICLES_CAPACITY 1024 */
/* #define MAX_PARTICLES_PER_CELL 32 */
/* #define TILE_SIZE 32 */
/* #define DELTA_T 0.01f */
/* #define INTR_CUTOFF 10.0f */
/* #define ATTR_STRENGTH 150.0f */
/* #define ACTIVE_FRACTION 1.0f */
/* #define PARTICLE_SPEED 40.0f */

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

#endif /* CONSTANTS_H */
