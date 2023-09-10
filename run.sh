#!/bin/sh

set -xe

nvcc -o bin/main src/simulation.cu src/particle.cu  -lGL -lglfw -lGLEW -lX11 -lpthread -lXrandr -lXi -ldl  &&
    ./bin/main
