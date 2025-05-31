// cuda_texobj.h
#ifndef CUDA_TEXOBJ_H
#define CUDA_TEXOBJ_H

#include <cuda_runtime.h>

struct CudaTexture {
    cudaArray_t array;
    cudaTextureObject_t texObj;
};

#endif
