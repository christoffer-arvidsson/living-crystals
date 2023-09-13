#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda_runtime.h>

inline __host__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

inline __host__ __device__ float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

inline __host__ __device__ float2 operator*(float a, float2 b) {
    return make_float2(a * b.x, a * b.y);
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
    return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ float2 f3_to_f2(float3 vec) {
    return make_float2(vec.x, vec.y);
}

inline __host__ __device__ float3 f2_to_f3(float2 vec) {
    return make_float3(vec.x, vec.y, 0.0f);
}

inline __device__ float dot_product2(float2 a, float2 b) {
    return (a.x * b.x) + (a.y * b.y);
}

inline __device__ float dot_product3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross_product(float3 a, float3 b) {
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

inline __device__ float squared_norm(float2 vec) {
    return vec.x * vec.x + vec.y * vec.y;
}

__device__ float2 cyclic_distance(float2 a, float2 b, float width, float height) {
    float2 dist = a - b;
    float dx = fabsf(dist.x);
    if (dx > width) {
        dist.x = width / 2.0f;
    }
    float dy = fabsf(dist.y);
    if (dy > height) {
        dist.y = height / 2.0f;
    }
    return dist;
}



#endif /* CUDA_HELPERS_H */
