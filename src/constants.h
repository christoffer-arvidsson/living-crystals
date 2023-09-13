#ifndef CONSTANTS_H
#define CONSTANTS_H

#define RNG_SEED 1234

// Tunnels
#define PARTICLES_CAPACITY 1024*4
#define MAX_PARTICLES_PER_CELL 128
#define TILE_SIZE 25
#define TRANS_COEFF 1.5f
#define ROT_COEFF 1.5f
#define DELTA_T 0.01f
#define INTR_CUTOFF 50.0f
#define ATTR_STRENGTH 50.0f
#define ACTIVE_FRACTION 0.05f
#define PARTICLE_SPEED 200.0f
#define PARTICLE_RADIUS 4.0f

// Clusters
/* #define PARTICLES_CAPACITY 512 */
/* #define MAX_PARTICLES_PER_CELL 128 */
/* #define TILE_SIZE 25 */
/* #define TRANS_COEFF 0.22f */
/* #define ROT_COEFF 0.16f */
/* #define DELTA_T 0.01f */
/* #define INTR_CUTOFF 25.0f */
/* #define ATTR_STRENGTH 150.0f */
/* #define ACTIVE_FRACTION 1.0f */
/* #define PARTICLE_SPEED 40.0f */
/* #define PARTICLE_RADIUS 4.0f */

#define LOOP_AROUND

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#endif /* CONSTANTS_H */
