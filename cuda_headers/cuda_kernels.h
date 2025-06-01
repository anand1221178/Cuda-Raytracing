// DECLARATION FILE
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_sphere.h"
#include "cuda_camera.h"

#define MAX_SPHERES 64          

// ── constant-memory scene
extern __constant__ unsigned char const_spheres_buffer[];


// ── kernel prototypes 
__global__ void rayKernel   (unsigned char* image, Camera* cam,
                                    Sphere* spheres, int n, int maxDepth);

__global__ void rayKernel_constant(unsigned char* image, Camera* cam,
                                   int n, int maxDepth);


// traceRay keeps its original signature
__device__ vec3 traceRay(const Ray& r, Sphere* spheres,
                         int n, int depth, int* seed);

                         
__device__ float lcg(int* seed);
#endif
