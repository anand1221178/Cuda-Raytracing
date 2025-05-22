// ALL GPU SIDE LOGIC WILL COME HERE!
// DEVICE FUNCTIONS/GLOBAL FUNCTIONS

#include "cuda_kernels.h"

__global__ void render_sky(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixel_index = (y * width + x) * 3;

    float u = float(x) / (width - 1);
    float v = float(y) / (height - 1);

    // Simple vertical gradient: lerp between white and blue
    float r = (1.0f - v) * 1.0f + v * 0.5f;
    float g = (1.0f - v) * 1.0f + v * 0.7f;
    float b = (1.0f - v) * 1.0f + v * 1.0f;

    image[pixel_index + 0] = (unsigned char)(255.99f * r);
    image[pixel_index + 1] = (unsigned char)(255.99f * g);
    image[pixel_index + 2] = (unsigned char)(255.99f * b);
}
