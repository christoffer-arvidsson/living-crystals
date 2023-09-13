# What
Simulation of micro-swimmers with brownian motion.

Originally I implemented this as part of an assignment for a simulation course,
but in python with pygame and numpy. This reimplementation is written in C with
CUDA and OpenGL and serves as an exploration project for particle physics simulation.

![Simulation video](https://github.com/christoffer-arvidsson/living-crystals/blob/main/assets/sim.gif)

# Requires
- GPU atomic operations (compute capability >= 1.1)

# Todos
- Tiling subregions for better performance than bruteforce n-body
    - Also launch more than one thread block to not be capped to 1024 particles
- FPS counter (on screen, but requires a lot of font stuff)
    - Currently logs in terminal
- Separate simulation and rendering threads
- Collision and accurate particle size rendering

# References
- CUDA particle simulation :: https://developer.download.nvidia.com/assets/cuda/files/particles.pdf
- Brownian motion simulation :: Volpe, Gigan & Volpe (2014) Simulation of the active Brownian motion of a microswimmer, American Journal of Physics.
- Metastable clusters and channels :: Nilsson & Volpe (2017) Metastable clusters and channels formed by active particles with aligning interactions, New Journal of Physics.
