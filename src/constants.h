#ifndef CONSTANTS_H
#define CONSTANTS_H

#define RNG_SEED 1234

// Tunnels
#define PARTICLES_CAPACITY 1024*16
#define MAX_PARTICLES_PER_CELL 128
#define TILE_SIZE 32
#define TRANS_COEFF 150.0f
#define ROT_COEFF 15.0f
#define DELTA_T 0.01f
#define INTR_CUTOFF 30.0f
#define ATTR_STRENGTH 25.0f
#define ACTIVE_FRACTION 0.1f
#define PARTICLE_SPEED 100.0f
#define PARTICLE_RADIUS 4.0f

// Clusters
/* #define PARTICLES_CAPACITY 1024 */
/* #define MAX_PARTICLES_PER_CELL 32 */
/* #define TRANS_COEFF 22.0f */
/* #define ROT_COEFF 16.0f */
/* #define DELTA_T 0.01f */
/* #define INTR_CUTOFF 10.0f */
/* #define ATTR_STRENGTH 150.0f */
/* #define ACTIVE_FRACTION 1.0f */
/* #define PARTICLE_SPEED 40.0f */
/* #define PARTICLE_RADIUS 4.0f */
/* #define TILE_SIZE 32 */

#define LOOP_AROUND

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080

#endif /* CONSTANTS_H */
