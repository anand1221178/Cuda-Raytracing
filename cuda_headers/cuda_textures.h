#ifndef CUDA_TEXTURES_H
#define CUDA_TEXTURES_H

#include "cuda_vec3.h"

struct Texture {
    int width, height;
    vec3* data; // Pointer to texture data

    __device__ vec3 value(float u, float v) const{
        int i = static_cast<int>(u*width);
        int j = static_cast<int>((1-v) * height - 0.001f);
        i = i < 0 ? 0 : (i >= width ? width -1 : i);
        j = j < 0 ? 0 : (j >= height ? height -1: j);
        return data[j * width +i];
    }
};

#endif