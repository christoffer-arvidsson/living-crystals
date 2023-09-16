#!/bin/sh

set -xe

CFLAGS="-Xcompiler -Wall"
LIBS="-lGL -lglfw -lGLEW -lX11 -lpthread -lXrandr -lXi -ldl"
SRC="src/simulation.c src/particle.cu"

nvcc $CFLAGS -o bin/main $SRC $LIBS  &&
    ./bin/main
