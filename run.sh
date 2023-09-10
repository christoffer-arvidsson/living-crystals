#!/bin/sh

set -xe

nvcc -o bin/main src/particle.cu  -lGL -lglfw -lGLEW -lX11 -lpthread -lXrandr -lXi -ldl  &&
    ./bin/main
