# What
Simulation of micro-swimmers with brownian motion.

Originally I implemented this as part of an assignment for a simulation course,
but in python with pygame and numpy. This reimplementation is written in C with
CUDA and OpenGL and serves as an exploration project for particle physics simulation.

*Tunnels*

https://github.com/christoffer-arvidsson/living-crystals/asset/26621143/6491c366-7754-4a0a-9ad8-2ddfb3a5e40c

*Clusters*

https://github.com/christoffer-arvidsson/living-crystals/assets/26621143/8dc85d3b-3ab1-40b0-957f-be5aa629a53d


# Requires
- GPU atomic operations (compute capability >= 1.1)

# Todos
- Wrap simulation in Simulator struct (for device state)
- Separate simulation and rendering threads
- Reuse the vbo with cuda+opengl interop

# References
- CUDA particle simulation :: https://developer.download.nvidia.com/assets/cuda/files/particles.pdf
- Brownian motion simulation :: Volpe, Gigan & Volpe (2014) Simulation of the active Brownian motion of a microswimmer, American Journal of Physics.
- Metastable clusters and channels :: Nilsson & Volpe (2017) Metastable clusters and channels formed by active particles with aligning interactions, New Journal of Physics.
