#ifndef CUDA_RAY_H
#define CUDA_RAY_H

#include "cuda_vec3.h"

struct Ray {
    vec3 origin;
    vec3 direction;

    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const vec3& o, const vec3& d) : origin(o), direction(d) {}

    __host__ __device__ vec3 at(float t) const {
        return origin + direction * t;
    }
};

#endif
    