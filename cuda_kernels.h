// DECLARTION FILE

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

// INCLUDES
#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_sphere.h"
#include "cuda_camera.h"

__global__ void rayKernel(unsigned char* image, Camera* cam, Sphere* spheres, int n);
__device__ vec3 traceRay(const Ray& r, Sphere* spheres, int n);
#endif
