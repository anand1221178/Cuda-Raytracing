#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_kernels.h"
#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_sphere.h"
#include "cuda_camera.h"
#include "cuda_material.h"

#include <vector>
#include <random>

#define WIDTH 1920
#define HEIGHT 1080
#define SAMPLES_PER_PIXEL 1000
#define MAX_DEPTH 50
#define NUM_SPHERES 30

__host__ void save_image(const char* filename, unsigned char* image) {
    FILE* fp = fopen(filename, "w");
    fprintf(fp, "P3\n%d %d\n255\n", WIDTH, HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT * 3; i += 3) {
        fprintf(fp, "%d %d %d\n", image[i], image[i+1], image[i+2]);
    }
    fclose(fp);
}

void build_scene(std::vector<Sphere>& H)
{
    std::mt19937 gen{42};
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // --- ground ---
    printf("HERE\n");
    H.emplace_back(vec3(0,-1000,0), 1000,static_cast<MaterialType>(3), vec3(1), 0, 1.0f); 
    // --- four large metal balls ---
    for (int i=0;i<4;++i) {
        float x = -3.0f + 2.0f*i;
        H.emplace_back(vec3(x,1,0), 1.0f,
        METAL, vec3(0.9f), 0.05f, 1.0f);
    }

    // --- small random balls grid ---
    for (int a=-NUM_SPHERES;a<=NUM_SPHERES;++a)
    {
        for (int b=-NUM_SPHERES;b<=NUM_SPHERES;++b) {
            float choose = uni(gen);
            vec3 center(a+0.9f*uni(gen), 0.2f, b+0.9f*uni(gen));
            if ((center-vec3(4,0.2,0)).length() <= 0.9f) continue;

            if (choose < 0.6f) {                 // diffuse
                vec3 col = vec3(uni(gen),uni(gen),uni(gen));
                col = col*col;                   // bias towards bright
                H.emplace_back(center,0.2f,LAMBERTIAN,col,0,1.0f);
            }
            else if (choose < 0.85f) {           // metal
                vec3 col = 0.5f*vec3(1+uni(gen),1+uni(gen),1+uni(gen));
                float fuzz = 0.01f + 0.08f*uni(gen);
                H.emplace_back(center,0.2f,METAL,col,fuzz,1.0f);
            }
            else {                               // glass
                H.emplace_back(center,0.2f,DIELECTRIC,vec3(1),0,1.5f);
            }
        }
    }
}

int main() {
    size_t img_size = WIDTH * HEIGHT * 3;
    unsigned char* h_img = (unsigned char*)malloc(img_size);
    unsigned char* d_img;

    // ----------SCENE SETUP--------------- //
    vec3 lookFrom = vec3(20.0f, 5.0f, 5.0f);  // farther & slightly higher
    vec3 lookAt   = vec3(0.0f, 0.5f, 0.0f);   // aim just above ground
    vec3 up = {0.0f, 1.0f, 0.0f};

    float distToFocus = (lookFrom - lookAt).length();
    float aperture = 0.03f;
    float aspect_ratio = float(WIDTH) / HEIGHT;

    Camera h_cam(lookFrom, lookAt, up, 20.0f, aspect_ratio, aperture, distToFocus);
    Camera* d_cam;
    cudaMalloc(&d_cam, sizeof(Camera));
    cudaMemcpy(d_cam, &h_cam, sizeof(Camera), cudaMemcpyHostToDevice);

    cudaMalloc(&d_img, img_size);

    std::vector<Sphere> host_spheres;
    build_scene(host_spheres);
    
    
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, host_spheres.size()*sizeof(Sphere));
    cudaMemcpy(d_spheres, host_spheres.data(),
            host_spheres.size()*sizeof(Sphere), cudaMemcpyHostToDevice);


    // TODO: Launch rayKernel here
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    rayKernel<<<grid,block>>>(d_img,d_cam,d_spheres,
        static_cast<int>(host_spheres.size()));
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    save_image("CudaOut.ppm", h_img);
    printf("Image saved as CudaOut.ppm\n");
    printf("Time take to compute image using global memory :%f ms\n", milliseconds);

    cudaFree(d_img);
    cudaFree(d_cam);
    free(h_img);
    return 0;
}
