#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_kernels.h"
#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_sphere.h"
#include "cuda_camera.h"
#include "cuda_material.h"
#include "stb_image.h"
#include "cuda_texobj.h"
#include <vector>
#include <random>

extern __constant__ cudaTextureObject_t dev_textures[5];

// FINAL RUN PARAMS
// #define WIDTH 1920
// #define HEIGHT 1080
// #define SAMPLES_PER_PIXEL 300
// #define MAX_DEPTH 30
// #define NUM_SPHERES 25

#define WIDTH 1280
#define HEIGHT 720
#define SAMPLES_PER_PIXEL 30
#define MAX_DEPTH 10
#define NUM_SPHERES 8

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
    H.emplace_back(vec3(0, -1000, 0), 1000, CHECKER, vec3(1), 0, 1.0f);

    // --- five large spheres for texture testing ---
    vec3 textured_positions[5] = {
        vec3(-6.0f, 1.0f, 6.0f),
        vec3(-3.0f, 1.0f, 6.0f),
        vec3( 0.0f, 1.0f, 6.0f),
        vec3( 3.0f, 1.0f, 6.0f),
        vec3( 6.0f, 1.0f, 6.0f)
    };
    for (int i = 0; i < 5; ++i) {
    H.emplace_back(textured_positions[i], 1.0f, TEXTURED, vec3(1.0f), 0.0f, 1.0f, i); // bind i-th texture
    }


    // --- two fixed known spheres (for control testing) ---
    H.emplace_back(vec3(-4, 1.0f, 0), 1.0f, DIELECTRIC, vec3(1), 0, 1.5f); // glass
    H.emplace_back(vec3( 4, 1.0f, 0), 1.0f, LAMBERTIAN, vec3(0.4f, 0.2f, 0.1f), 0, 1.0f); // diffuse



    // --- small random balls grid ---
    for (int a = -NUM_SPHERES; a <= NUM_SPHERES; ++a)
    {
        for (int b = -NUM_SPHERES; b <= NUM_SPHERES; ++b) {
            vec3 center(a + 0.9f * uni(gen), 0.2f, b + 0.9f * uni(gen));

            // Avoid overlapping the main spheres (control + textured)
            vec3 exclusion_centers[] = {
                vec3(-4,1,0), vec3(4,1,0),
                vec3(-6,1,6), vec3(-3,1,6), vec3(0,1,6), vec3(3,1,6), vec3(6,1,6)
            };

            bool skip = false;
            for (const vec3& p : exclusion_centers)
                if ((center - p).length() < 1.2f) { skip = true; break; }
            if (skip) continue;

            float choose = uni(gen);
            if (choose < 0.6f) {
                vec3 col = vec3(uni(gen), uni(gen), uni(gen));
                col = col * col;
                H.emplace_back(center, 0.2f, LAMBERTIAN, col, 0, 1.0f);
            } else if (choose < 0.85f) {
                vec3 col = 0.5f * vec3(1 + uni(gen), 1 + uni(gen), 1 + uni(gen));
                float fuzz = 0.01f + 0.08f * uni(gen);
                H.emplace_back(center, 0.2f, METAL, col, fuzz, 1.0f);
            } else {
                H.emplace_back(center, 0.2f, DIELECTRIC, vec3(1), 0, 1.5f);
            }
        }
    }
}

int main() {
    size_t img_size = WIDTH * HEIGHT * 3;
    unsigned char* h_img = (unsigned char*)malloc(img_size);
    unsigned char* d_img;
    // -----------TEXTURE MEMORY SETUP ---------//
    CudaTexture h_textures[5];
    unsigned char* h_tex_data[5];
    int tex_width[5], tex_height[5], tex_channels[5];

    const char* tex_files[5] = {
        "./test_textures/beach_probe.jpg",
        "./test_textures/building_probe.jpg",
        "./test_textures/campus_probe.jpg",
        "./test_textures/kitchen_probe.jpg",
        "./test_textures/tex.jpg"
    };

    for (int i = 0; i < 5; ++i) {
        h_tex_data[i] = stbi_load(tex_files[i], &tex_width[i], &tex_height[i], &tex_channels[i], 4);
        if (!h_tex_data[i]) {
            printf("Failed to load texture %s\n", tex_files[i]);
            exit(1);
        }

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        cudaMallocArray(&h_textures[i].array, &desc, tex_width[i], tex_height[i]);

        cudaMemcpy2DToArray(h_textures[i].array, 0, 0, h_tex_data[i],
                            tex_width[i] * 4, tex_width[i] * 4, tex_height[i],
                            cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = h_textures[i].array;

        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&h_textures[i].texObj, &resDesc, &texDesc, nullptr);
    }
    cudaTextureObject_t h_textureObjs[5];
    for (int i = 0; i < 5; ++i) {
        h_textureObjs[i] = h_textures[i].texObj;
    }

    cudaMemcpyToSymbol(dev_textures, h_textureObjs, sizeof(h_textureObjs));

    // ----------SCENE SETUP--------------- //
    vec3 lookFrom = vec3(20.0f, 10.0f, 5.0f);  // farther & slightly higher
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
    cudaDeviceSetLimit(cudaLimitStackSize, 32768); // or 32768 for deep recursion

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
