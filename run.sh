#!/bin/sh

nvcc -o basics basics.cu glad/src/glad.c -lGL -lglfw -lX11 -lpthread -lXrandr -lXi -ldl -Iglad/include &&
    ./basics
