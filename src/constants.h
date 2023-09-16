#ifndef CONSTANTS_H
#define CONSTANTS_H

#define RNG_SEED 1234

// mu m
/* #define PARTICLE_RADIUS 1.0f */

// From the paper:
// unit mu m
// #define PARTICLE_RADIUS 1.0f

// unit J*K^-1
//   #define BOLTZMANN_CONSTANT 1.380649 * pow(10.0, -23.0)

// unit K
/* #define TEMPERATURE_KELVIN 300.0 */

// unit Ns/m^2 (water)
/* #define FLUID_VISCOSITY 0.001 */

// unit mu m^2/s
/* #define TRANS_COEFF 0.22f */

// unit rad^2/s
/* #define ROT_COEFF 0.16f */

// Tunnels
/* #define PARTICLES_CAPACITY 1024*16 */
/* #define MAX_PARTICLES_PER_CELL 128 */
/* #define TILE_SIZE 32 */
/* #define PARTICLE_RADIUS 4.0f */
/* #define TRANS_COEFF 0.22f / PARTICLE_RADIUS */
/* #define ROT_COEFF 0.16f / powf(PARTICLE_RADIUS, 3) */
/* #define DELTA_T 0.01f */
/* #define INTR_CUTOFF 30.0f */
/* #define ATTR_STRENGTH 25.0f */
/* #define ACTIVE_FRACTION 0.1f */
/* #define PARTICLE_SPEED 100.0f */

// Clusters
#define PARTICLES_CAPACITY 1024
#define MAX_PARTICLES_PER_CELL 32
#define TILE_SIZE 32
#define PARTICLE_RADIUS 4.0f
#define TRANS_COEFF 0.22f / PARTICLE_RADIUS
#define ROT_COEFF 0.16f / powf(PARTICLE_RADIUS, 3)
#define DELTA_T 0.01f
#define INTR_CUTOFF 10.0f
#define ATTR_STRENGTH 150.0f
#define ACTIVE_FRACTION 1.0f
#define PARTICLE_SPEED 40.0f

#define LOOP_AROUND

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

#endif /* CONSTANTS_H */
