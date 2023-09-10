#!/bin/sh

set -xe

nvcc -o tunnels tunnels.cu  -lGL -lglfw -lGLEW -lX11 -lpthread -lXrandr -lXi -ldl  &&
    ./tunnels
