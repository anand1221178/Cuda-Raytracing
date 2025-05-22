// CUDA VERSION OF MAIN.C all mem alloc, freeing and IO will go here
// HOST-SIDE

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_kernels.h"

#define WIDTH 640
#define HEIGHT 640

__host__ void save_image(const char* filename, unsigned char* image) {
    FILE* fp = fopen(filename, "w");
    fprintf(fp, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT * 3; i += 3) {
        fprintf(fp, "%d %d %d\n", image[i], image[i+1], image[i+2]);
    }
    fclose(fp);
}

int main() {
    size_t img_size = WIDTH * HEIGHT * 3;
    unsigned char* h_img = (unsigned char*)malloc(img_size);
    unsigned char* d_img;

    cudaMalloc(&d_img, img_size);

    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    render_sky<<<grid, block>>>(d_img, WIDTH, HEIGHT);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    save_image("CudaOut.ppm", h_img);

    cudaFree(d_img);
    free(h_img);
    return 0;
}
