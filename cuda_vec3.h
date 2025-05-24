#ifndef CUDA_VEC3_H
#define CUDA_VEC3_H

// Definations for intellisense and gcc compilers
#ifndef __CUDACC__
#define __host__
#define __device__
#endif

#include <math.h>

struct vec3 {
    float x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ vec3 operator+(const vec3& v) const {
        return vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ vec3 operator-(const vec3& v) const {
        return vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ vec3 operator*(float t) const {
        return vec3(x * t, y * t, z * t);
    }

    __host__ __device__ float dot(const vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ vec3 normalized() const {
        float len = length();
        return (*this) * (1.0f / len);
    }
};

#endif
