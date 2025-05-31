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
#include <iostream>
#include <iomanip>

extern __constant__ cudaTextureObject_t dev_textures[5];
__constant__ unsigned char const_spheres_buffer[64 * sizeof(Sphere)];

#define MAX_SPHERES 64
#define NUM_SPHERES 3  // Fixed for constant memory


// Test configuration
struct TestConfig {
    int width, height, samples, depth;
    const char* name;
};

__host__ void save_image(const char* filename, unsigned char* image, int width, int height) {
    FILE* fp = fopen(filename, "w");
    fprintf(fp, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height * 3; i += 3) {
        fprintf(fp, "%d %d %d\n", image[i], image[i+1], image[i+2]);
    }
    fclose(fp);
}

void build_scene(std::vector<Sphere>& H) {
    std::mt19937 gen{42};
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    // Ground
    H.push_back({vec3(0, -1000, 0), 1000.0f, CHECKER, vec3(1), 0.0f, 1.0f, -1});

    // Five textured spheres
    vec3 textured_positions[5] = {
        vec3(-6.0f, 1.0f, 6.0f),
        vec3(-3.0f, 1.0f, 6.0f),
        vec3( 0.0f, 1.0f, 6.0f),
        vec3( 3.0f, 1.0f, 6.0f),
        vec3( 6.0f, 1.0f, 6.0f)
    };
    for (int i = 0; i < 5; ++i) {
        H.push_back({textured_positions[i], 1.0f, TEXTURED, vec3(1.0f), 0.0f, 1.0f, i});
    }

    // Two dielectric spheres
    H.push_back({vec3(-4, 1.0f, 0), 1.0f, DIELECTRIC, vec3(1), 0.0f, 1.5f,-1});
    H.push_back({vec3( 4, 1.0f, 0), 1.0f, DIELECTRIC, vec3(1), 0.0f, 1.5f, -1});

    // Random small spheres
    for (int a = -NUM_SPHERES; a <= NUM_SPHERES; ++a) {
        for (int b = -NUM_SPHERES; b <= NUM_SPHERES; ++b) {
            vec3 center(a + 0.9f * uni(gen), 0.2f, b + 0.9f * uni(gen));

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
                H.push_back({center, 0.2f, LAMBERTIAN, col, 0.0f, 1.0f,-1}); 
            } else if (choose < 0.85f) {
                vec3 col = 0.5f * vec3(1 + uni(gen), 1 + uni(gen), 1 + uni(gen));
                float fuzz = 0.01f + 0.08f * uni(gen);
                H.push_back({center, 0.2f, METAL, col, fuzz, 1.0f,-1});
            } else {
                H.push_back({center, 0.2f, DIELECTRIC, vec3(1), 0.0f, 1.5f,-1});
            }
        }
    }
}

template<typename KernelPtr>
float time_kernel_3D(KernelPtr k,
                     unsigned char* d_img,
                     Camera* d_cam,
                     Sphere* d_spheres,
                     int n, int maxDepth,
                     dim3 grid, dim3 block,
                     int width, int height, int samples_per_pixel)
{
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    
    // Warm up
    k<<<grid,block>>>(d_img, d_cam, d_spheres, n, maxDepth, width, height, samples_per_pixel);
    cudaDeviceSynchronize();
    
    // Actual timing
    cudaEventRecord(t0);
    k<<<grid,block>>>(d_img, d_cam, d_spheres, n, maxDepth, width, height, samples_per_pixel);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, t0, t1);
    
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    
    return ms;
}

// Modified kernel that accepts dynamic parameters
__global__ void rayKernel_benchmark(unsigned char* image, Camera* cam, Sphere* spheres, 
                                   int n, int maxDepth, int width, int height, int samples_per_pixel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int seed = x + y * width;
    vec3 pixel_color = vec3(0, 0, 0);

    for (int s = 0; s < samples_per_pixel; ++s) {
        float u = (x + lcg(&seed)) / float(width - 1);
        float v = (y + lcg(&seed)) / float(height - 1);

        Ray r = cam->get_ray(u, v, &seed, lcg);
        pixel_color += traceRay(r, spheres, n, maxDepth, &seed);
    }

    pixel_color = pixel_color * (1.0f / samples_per_pixel);

    // Gamma correction
    pixel_color.x = sqrtf(fminf(fmaxf(pixel_color.x, 0.0f), 1.0f));
    pixel_color.y = sqrtf(fminf(fmaxf(pixel_color.y, 0.0f), 1.0f));
    pixel_color.z = sqrtf(fminf(fmaxf(pixel_color.z, 0.0f), 1.0f));

    int idx = (y * width + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * pixel_color.x);
    image[idx + 1] = (unsigned char)(255.99f * pixel_color.y);
    image[idx + 2] = (unsigned char)(255.99f * pixel_color.z);
}

int main() {
    // Setup textures
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

    // Build scene
    std::vector<Sphere> host_spheres;
    build_scene(host_spheres);
    
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, host_spheres.size() * sizeof(Sphere));
    cudaMemcpy(d_spheres, host_spheres.data(),
               host_spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(const_spheres_buffer,
                       host_spheres.data(),
                       host_spheres.size() * sizeof(Sphere));

    // Test configurations
    TestConfig configs[] = {
        {640, 360, 10, 5, "360p Low"},
        {640, 360, 50, 10, "360p Medium"},
        {640, 360, 100, 15, "360p High"},
        {1280, 720, 10, 5, "720p Low"},
        {1280, 720, 30, 10, "720p Medium"},
        {1280, 720, 100, 20, "720p High"},
        {1920, 1080, 10, 5, "1080p Low"},
        {1920, 1080, 50, 15, "1080p Medium"},
        {1920, 1080, 100, 20, "1080p High"},
        {1920, 1080, 300, 30, "1080p Ultra"}
    };

    // Print header
    std::cout << "\n===== PERFORMANCE SCALING ANALYSIS =====\n";
    std::cout << std::left << std::setw(20) << "Configuration"
              << std::setw(15) << "Resolution"
              << std::setw(10) << "Samples"
              << std::setw(10) << "Depth"
              << std::setw(12) << "Time(ms)"
              << std::setw(15) << "MRays/sec"
              << std::setw(15) << "Pixels/sec"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    // Run tests
    for (const auto& config : configs) {
        // Setup camera for aspect ratio
        float aspect_ratio = float(config.width) / config.height;
        vec3 lookFrom = vec3(20.0f, 5.0f, 20.0f);
        vec3 lookAt = vec3(0.0f, 0.5f, 0.0f);
        vec3 up = {0.0f, 1.0f, 0.0f};
        float distToFocus = (lookFrom - lookAt).length();
        float aperture = 0.03f;

        Camera h_cam(lookFrom, lookAt, up, 20.0f, aspect_ratio, aperture, distToFocus);
        Camera* d_cam;
        cudaMalloc(&d_cam, sizeof(Camera));
        cudaMemcpy(d_cam, &h_cam, sizeof(Camera), cudaMemcpyHostToDevice);

        // Allocate image
        size_t img_size = config.width * config.height * 3;
        unsigned char* d_img;
        cudaMalloc(&d_img, img_size);

        // Set stack size for recursion
        cudaDeviceSetLimit(cudaLimitStackSize, 32768);

        // Configure grid/block
        dim3 block(16, 16);
        dim3 grid((config.width + block.x - 1) / block.x,
                  (config.height + block.y - 1) / block.y);

        // Time the kernel
        float time_ms = time_kernel_3D(rayKernel_benchmark,
                                       d_img, d_cam, d_spheres,
                                       host_spheres.size(), config.depth,
                                       grid, block,
                                       config.width, config.height, config.samples);

        // Calculate metrics
        long long primary_rays = (long long)config.width * config.height * config.samples;
        float avg_secondary_rays = 1.5f * config.depth / 2.0f;
        long long total_rays = primary_rays * avg_secondary_rays;
        double mrays_per_sec = (total_rays / 1e6) / (time_ms / 1000.0);
        double pixels_per_sec = (config.width * config.height) / (time_ms / 1000.0);

        // Print results
        std::cout << std::left << std::setw(20) << config.name
                  << std::setw(15) << (std::to_string(config.width) + "x" + std::to_string(config.height))
                  << std::setw(10) << config.samples
                  << std::setw(10) << config.depth
                  << std::setw(12) << std::fixed << std::setprecision(2) << time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << mrays_per_sec
                  << std::setw(15) << std::scientific << std::setprecision(2) << pixels_per_sec
                  << std::endl;

        // Cleanup
        cudaFree(d_img);
        cudaFree(d_cam);
    }

    // Scaling analysis graphs
    std::cout << "\n===== SCALING TRENDS =====\n";
    std::cout << "Resolution Scaling (fixed samples=50, depth=10):\n";
    std::cout << "360p → 720p: ~4x pixels, expect ~4x time\n";
    std::cout << "720p → 1080p: ~2.25x pixels, expect ~2.25x time\n";
    
    std::cout << "\nSample Scaling (fixed resolution=1280x720, depth=10):\n";
    std::cout << "10 → 30 samples: 3x samples, expect ~3x time\n";
    std::cout << "30 → 100 samples: ~3.3x samples, expect ~3.3x time\n";

    // Cleanup
    cudaFree(d_spheres);
    for (int i = 0; i < 5; ++i) {
        cudaDestroyTextureObject(h_textures[i].texObj);
        cudaFreeArray(h_textures[i].array);
        stbi_image_free(h_tex_data[i]);
    }

    return 0;
}