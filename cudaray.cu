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
#include <iostream>              // <<< add this right after the other #includes


extern __constant__ cudaTextureObject_t dev_textures[5];

// Define constant memory as a byte buffer to avoid constructor issues
__constant__ unsigned char const_spheres_buffer[64 * sizeof(Sphere)];

// FINAL RUN PARAMS
#define WIDTH 1920
#define HEIGHT 1080
#define SAMPLES_PER_PIXEL 300
#define MAX_DEPTH 30
#define NUM_SPHERES 3  // Must stay at 3 due to constant memory constraints (generates ~55 spheres total)

// #define WIDTH 1280
// #define HEIGHT 720
// #define SAMPLES_PER_PIXEL 30
// #define MAX_DEPTH 10
// #define NUM_SPHERES 3
#define MAX_SPHERES 64 

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
    H.push_back({vec3(0, -1000, 0), 1000.0f,           // radius  -> float
               CHECKER, vec3(1), 0.0f, 1.0f, -1});

    // --- five large spheres for texture testing ---
    vec3 textured_positions[5] = {
        vec3(-6.0f, 1.0f, 6.0f),
        vec3(-3.0f, 1.0f, 6.0f),
        vec3( 0.0f, 1.0f, 6.0f),
        vec3( 3.0f, 1.0f, 6.0f),
        vec3( 6.0f, 1.0f, 6.0f)
    };
    for (int i = 0; i < 5; ++i) {
    H.push_back({textured_positions[i], 1.0f, TEXTURED, vec3(1.0f), 0.0f, 1.0f, i}); // bind i-th texture
    }


    // --- two fixed known spheres (for control testing) ---
    H.push_back({vec3(-4, 1.0f, 0), 1.0f, DIELECTRIC,
               vec3(1), 0.0f, 1.5f,-1});                 // fuzz -> 0.0f

    H.push_back({vec3( 4, 1.0f, 0), 1.0f, DIELECTRIC,
                vec3(1), 0.0f, 1.5f, -1});



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
                H.push_back({center, 0.2f, LAMBERTIAN, col,
                    0.0f, 1.0f,-1}); 
            } else if (choose < 0.85f) {
                vec3 col = 0.5f * vec3(1 + uni(gen), 1 + uni(gen), 1 + uni(gen));
                float fuzz = 0.01f + 0.08f * uni(gen);
                H.push_back({center, 0.2f, METAL, col, fuzz, 1.0f,-1});
            } else {
                H.push_back({center, 0.2f, DIELECTRIC,
                    vec3(1), 0.0f, 1.5f,-1});
            }
        }
    }
}

// TIMING TEMPLATE
template<typename KernelPtr>
float time_kernel_3D(KernelPtr k,
                     unsigned char* d_img,
                     Camera* d_cam,
                     Sphere* d_spheres,
                     int n, int maxDepth,
                     dim3 grid, dim3 block)
{
    cudaEvent_t t0, t1;  cudaEventCreate(&t0);  cudaEventCreate(&t1);
    cudaEventRecord(t0);
    k<<<grid,block>>>(d_img, d_cam, d_spheres, n, maxDepth);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    return ms;
}

// overload for constant-memory kernel (no Sphere* param)
template<typename KernelPtr>
float time_kernel_const(KernelPtr k,
                        unsigned char* d_img,
                        Camera* d_cam,
                        int n, int maxDepth,
                        dim3 grid, dim3 block)
{
    cudaEvent_t t0, t1;  cudaEventCreate(&t0);  cudaEventCreate(&t1);
    cudaEventRecord(t0);
    k<<<grid,block>>>(d_img, d_cam, n, maxDepth);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    return ms;
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
    vec3 lookFrom = vec3(20.0f, 5.0f, 20.0f);  // farther & slightly higher
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
    
    // Debug: print sphere count and size
    std::cout << "Number of spheres: " << host_spheres.size() << std::endl;
    std::cout << "Size of Sphere: " << sizeof(Sphere) << " bytes" << std::endl;
    std::cout << "Total size needed: " << host_spheres.size() * sizeof(Sphere) << " bytes" << std::endl;
    std::cout << "Constant memory buffer size: " << sizeof(const_spheres_buffer) << " bytes" << std::endl;
    
    std::cout << "Total spheres in scene: " << host_spheres.size() << std::endl;
    if (host_spheres.size() > MAX_SPHERES) {
        std::cerr << "ERROR: Scene has " << host_spheres.size() 
                  << " spheres, but constant memory can only hold " << MAX_SPHERES << std::endl;
    }


    
    
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, host_spheres.size()*sizeof(Sphere));
    cudaMemcpy(d_spheres, host_spheres.data(),
            host_spheres.size()*sizeof(Sphere), cudaMemcpyHostToDevice);

    // ------------ copy the same scene to constant memory once ------------
    cudaMemcpyToSymbol(const_spheres_buffer,
        host_spheres.data(),
        host_spheres.size() * sizeof(Sphere));

    // TODO: Launch rayKernel here
    cudaDeviceSetLimit(cudaLimitStackSize, 32768); // or 32768 for deep recursion

    dim3 block2D(16,16);                                    // for original 2-D kernel
    dim3 grid2D((WIDTH+block2D.x-1)/block2D.x,
                (HEIGHT+block2D.y-1)/block2D.y);

    dim3 block1D(256);                                      // for shared/constant
    dim3 grid1D((WIDTH*HEIGHT + block1D.x - 1)/block1D.x);


    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x, (HEIGHT + block.y - 1) / block.y);
    rayKernel<<<grid, block>>>(d_img, d_cam, d_spheres,
        static_cast<int>(host_spheres.size()),
        MAX_DEPTH);      // add arg
    cudaDeviceSynchronize();


    float t_global = time_kernel_3D(rayKernel,           // your original
        d_img, d_cam, d_spheres,
        host_spheres.size(), MAX_DEPTH,
        grid2D, block2D);

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
    save_image("out_global.ppm", h_img);
    std::cout << "Global   " << t_global << "  ms  (out_global.ppm)\n";

    // -------------------------------------------------------------------
    float t_const  = time_kernel_const(rayKernel_constant,
            d_img, d_cam,
            host_spheres.size(), MAX_DEPTH,
            grid1D, block1D);

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);
    save_image("out_const.ppm", h_img);
    std::cout << "Const    " << t_const  << "  ms  (out_const.ppm)\n";

    // summary
    std::cout << "Speed-up vs Global  →  Const: "
    << t_global/t_const  << "×\n";

    // ===== BENCHMARK 1: Rays Per Second (RPS) =====
    std::cout << "\n===== RAYS PER SECOND ANALYSIS =====\n";
    
    // Estimate average secondary rays (bounces) - typically 1.5-2x for diffuse scenes
    float avg_secondary_rays = 1.5f * MAX_DEPTH / 2.0f; // Rough estimate
    long long primary_rays = (long long)WIDTH * HEIGHT * SAMPLES_PER_PIXEL;
    long long total_rays = primary_rays * avg_secondary_rays;
    
    double rays_per_sec_global = total_rays / (t_global / 1000.0);
    double rays_per_sec_const = total_rays / (t_const / 1000.0);
    
    std::cout << "Primary rays: " << primary_rays << std::endl;
    std::cout << "Estimated total rays (with bounces): " << total_rays << std::endl;
    std::cout << "Global kernel: " << rays_per_sec_global / 1e9 << " billion rays/sec" << std::endl;
    std::cout << "Const kernel: " << rays_per_sec_const / 1e9 << " billion rays/sec" << std::endl;

    // ===== BENCHMARK 2: Memory Bandwidth Utilization =====
    std::cout << "\n===== MEMORY BANDWIDTH ANALYSIS =====\n";
    
    // More realistic memory access calculation
    // Each ray typically checks ~10-20 spheres on average (not all)
    size_t sphere_data_size = sizeof(Sphere);
    size_t avg_spheres_checked = host_spheres.size() / 3; // Rough estimate
    size_t reads_per_ray = avg_spheres_checked * sphere_data_size;
    size_t writes_per_pixel = 3 * sizeof(unsigned char);
    
    // Account for texture reads (5 textured spheres)
    size_t texture_reads_per_ray = 4 * sizeof(float) * 5; // float4 per texture sample
    
    // Total memory accessed (primary rays only for more accurate estimate)
    size_t total_sphere_reads = primary_rays * reads_per_ray;
    size_t total_texture_reads = primary_rays * texture_reads_per_ray / 10; // ~10% rays hit textured spheres
    size_t total_writes = WIDTH * HEIGHT * writes_per_pixel;
    size_t total_bytes = total_sphere_reads + total_texture_reads + total_writes;
    
    double bandwidth_global_GBps = (total_bytes / 1e9) / (t_global / 1000.0);
    double bandwidth_const_GBps = (total_bytes / 1e9) / (t_const / 1000.0);
    
    // Get device properties for theoretical bandwidth
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double theoretical_bandwidth_GBps = (prop.memoryClockRate * 1000.0 * 2.0 * prop.memoryBusWidth / 8) / 1e9;
    
    std::cout << "Total memory accessed: " << total_bytes / 1e9 << " GB" << std::endl;
    std::cout << "Global kernel bandwidth: " << bandwidth_global_GBps << " GB/s (" 
              << (bandwidth_global_GBps / theoretical_bandwidth_GBps * 100) << "% of theoretical)" << std::endl;
    std::cout << "Const kernel bandwidth: " << bandwidth_const_GBps << " GB/s (" 
              << (bandwidth_const_GBps / theoretical_bandwidth_GBps * 100) << "% of theoretical)" << std::endl;
    std::cout << "Theoretical max bandwidth: " << theoretical_bandwidth_GBps << " GB/s" << std::endl;

    // ===== BENCHMARK 3: Occupancy Analysis =====
    std::cout << "\n===== OCCUPANCY ANALYSIS =====\n";
    
    int blockSize2D = 16 * 16;
    int blockSize1D = 256;
    int numBlocksGlobal, numBlocksConst;
    
    // Calculate occupancy for both kernels
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksGlobal, 
        rayKernel, blockSize2D, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksConst, 
        rayKernel_constant, blockSize1D, 0);
    
    int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize2D;
    float occupancy_global = (float)numBlocksGlobal / maxBlocksPerSM;
    
    maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize1D;
    float occupancy_const = (float)numBlocksConst / maxBlocksPerSM;
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Global kernel occupancy: " << occupancy_global * 100 << "%" << std::endl;
    std::cout << "Const kernel occupancy: " << occupancy_const * 100 << "%" << std::endl;

    // ===== BENCHMARK 4: Performance Scaling Analysis =====
    std::cout << "\n===== PERFORMANCE SCALING ANALYSIS =====\n";
    std::cout << "Note: For full scaling analysis across multiple configurations,\n";
    std::cout << "compile and run 'make benchmark && ./benchmark_scaling'\n";
    
    // Show scaling formulas
    std::cout << "\nExpected Scaling Relationships:\n";
    std::cout << "- Time ∝ Width × Height (linear with pixel count)\n";
    std::cout << "- Time ∝ Samples per pixel (linear)\n";
    std::cout << "- Time ∝ Max depth × probability of bounce (sub-linear)\n";
    
    // Current configuration performance
    std::cout << "\nCurrent Configuration Performance:\n";
    std::cout << "Resolution: " << WIDTH << "x" << HEIGHT << " (" << (WIDTH*HEIGHT/1e6) << " MP)\n";
    std::cout << "Samples: " << SAMPLES_PER_PIXEL << "\n";
    std::cout << "Max Depth: " << MAX_DEPTH << "\n";
    std::cout << "Time: " << t_global << " ms\n";
    std::cout << "Throughput: " << (WIDTH * HEIGHT) / (t_global / 1000.0) / 1e6 << " MPixels/sec\n";
    
    // Estimate performance for other resolutions
    std::cout << "\nEstimated Performance at Other Resolutions (same samples/depth):\n";
    float base_pixels = WIDTH * HEIGHT;
    float base_time = t_global;
    
    struct Resolution { int w, h; const char* name; };
    Resolution resolutions[] = {
        {640, 360, "360p"},
        {1280, 720, "720p"}, 
        {2560, 1440, "1440p"},
        {3840, 2160, "4K"}
    };
    
    for (const auto& res : resolutions) {
        float pixels = res.w * res.h;
        float estimated_time = base_time * (pixels / base_pixels);
        float estimated_fps = 1000.0f / estimated_time;
        std::cout << res.name << " (" << res.w << "x" << res.h << "): " 
                  << estimated_time << " ms (" << estimated_fps << " FPS)\n";
    }

    cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_img);
    cudaFree(d_cam);
    cudaFree(d_spheres);
    free(h_img);
    
    // Cleanup textures
    for (int i = 0; i < 5; ++i) {
        cudaDestroyTextureObject(h_textures[i].texObj);
        cudaFreeArray(h_textures[i].array);
        stbi_image_free(h_tex_data[i]);
    }
    
    return 0;
}
