#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_kernels.h"
#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_sphere.h"
#include "cuda_camera.h"

#define WIDTH 640
#define HEIGHT 640
#define SAMPLES_PER_PIXEL 100
#define MAX_DEPTH 50

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

    // ----------SCENE SETUP--------------- //
    vec3 lookFrom = {13.0f, 2.0f, 3.0f};
    vec3 lookAt = {0.0f, 0.0f, 0.0f};
    vec3 up = {0.0f, 1.0f, 0.0f};

    float distToFocus = 10.0f;
    float aperture = 0.1f;
    float aspect_ratio = float(WIDTH) / HEIGHT;

    Camera h_cam(lookFrom, lookAt, up, 20.0f, aspect_ratio, aperture, distToFocus);
    Camera* d_cam;
    cudaMalloc(&d_cam, sizeof(Camera));
    cudaMemcpy(d_cam, &h_cam, sizeof(Camera), cudaMemcpyHostToDevice);

    cudaMalloc(&d_img, img_size);

    Sphere h_spheres[2] = {
    Sphere{vec3{0.0f, 0.0f, -1.0f}, 0.5f},
    Sphere{vec3{0.0f, -100.5f, -1.0f}, 100.0f}  // ground
    };

    Sphere* d_spheres;
    cudaMalloc(&d_spheres, sizeof(h_spheres));
    cudaMemcpy(d_spheres, h_spheres, sizeof(h_spheres), cudaMemcpyHostToDevice);

    // TODO: Launch rayKernel here
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    rayKernel<<<grid, block>>>(d_img, d_cam, d_spheres, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
    save_image("CudaOut.ppm", h_img);

    cudaFree(d_img);
    cudaFree(d_cam);
    free(h_img);
    return 0;
}
