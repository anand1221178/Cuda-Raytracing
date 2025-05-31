# GPU-Accelerated Ray Tracing: A CUDA Implementation Study

## Abstract

This report presents the implementation and optimization of a physically-based ray tracer using NVIDIA CUDA. The project explores various GPU memory hierarchies and optimization techniques to achieve real-time rendering performance for complex scenes with multiple material types including Lambertian diffuse, metallic, dielectric, and texture-mapped surfaces. Through careful analysis of memory access patterns and kernel configurations, we demonstrate the performance characteristics of different implementation strategies on an NVIDIA GeForce GTX 1070 GPU.

## 1. Introduction

### 1.1 Problem Statement

Ray tracing is a computationally intensive rendering technique that simulates the physical behavior of light to produce photorealistic images. The core challenge lies in the algorithm's inherent complexity: for each pixel in the output image, multiple rays must be traced through the scene, potentially bouncing off surfaces multiple times. With modern display resolutions and the demand for high-quality rendering, the computational requirements become substantial.

For a 1920×1080 image with 300 samples per pixel and maximum recursion depth of 30, our implementation processes approximately 14 billion rays, making efficient GPU utilization critical for achieving reasonable render times.

### 1.2 Project Objectives

- Implement a feature-complete ray tracer supporting multiple material types
- Leverage CUDA's various memory hierarchies for optimal performance
- Analyze performance characteristics and scaling behavior
- Achieve interactive frame rates for moderate scene complexity

## 2. Implementation

### 2.1 Architecture Overview

The ray tracer is implemented using CUDA C++ with a modular architecture consisting of:

- **Core Data Structures**: `vec3` for 3D vectors, `Ray` for ray representation, `Sphere` for scene geometry
- **Material System**: Supporting Lambertian, Metal, Dielectric, Checker pattern, and Textured materials
- **Memory Management**: Utilizing global, constant, and texture memory for different data types
- **Kernel Design**: Both 2D grid-based and 1D linearized kernel configurations

### 2.2 Scene Representation

The scene consists of 57 spheres total:
- 1 large ground sphere with checker pattern
- 5 textured spheres using environment maps
- 2 perfect mirror spheres
- 49 randomly placed small spheres with mixed materials

```cpp
struct Sphere {
    vec3  center;
    float radius;
    MaterialType mat;
    vec3  albedo;
    float fuzz;
    float ir;        // index of refraction
    int   texture_id;
};
```

### 2.3 Ray Tracing Algorithm

The core ray tracing algorithm implements:

1. **Primary Ray Generation**: Camera generates rays through each pixel with multiple samples for anti-aliasing
2. **Intersection Testing**: Each ray tests against all spheres in the scene
3. **Shading**: Material-specific shading based on intersection results
4. **Recursive Bounces**: Secondary rays for reflections/refractions up to MAX_DEPTH

### 2.4 Material Implementation

Each material type implements physically-based light transport:

- **Lambertian**: Diffuse scattering with cosine-weighted hemisphere sampling
- **Metal**: Perfect and fuzzy reflections using Schlick's approximation
- **Dielectric**: Refraction and reflection based on Fresnel equations
- **Textured**: Environment mapping with spherical UV coordinates

## 3. GPU Optimization Strategies

### 3.1 Memory Hierarchy Utilization

#### Global Memory Kernel
- Stores sphere data in global memory
- Simple implementation but high memory bandwidth requirements
- Each thread potentially accesses all sphere data multiple times

#### Constant Memory Kernel
- Leverages 64KB constant memory cache
- Sphere data (2508 bytes) fits entirely in constant memory
- Optimized for broadcast access patterns where all threads read same data

#### Texture Memory
- Environment maps stored as CUDA texture objects
- Hardware-accelerated bilinear filtering
- Efficient 2D spatial locality for texture lookups

### 3.2 Kernel Configuration

Two kernel launch configurations were tested:

1. **2D Grid (16×16 blocks)**: Natural mapping to image dimensions
2. **1D Grid (256 threads/block)**: Better for constant memory access patterns

### 3.3 Performance Optimizations

- **Ray Offset Bias**: 1e-3f offset to prevent self-intersection artifacts
- **Early Ray Termination**: Rays stop at first hit for primary visibility
- **Normalized Coordinates**: Texture coordinates use normalized [0,1] range
- **Stack Size Management**: Increased to 32KB for deep recursion support

## 4. Performance Analysis

### 4.1 Benchmark Results

| Metric | Global Memory | Constant Memory |
|--------|---------------|-----------------|
| Render Time | 74,702.7 ms | 105,280 ms |
| Rays/second | 187.4 M | 132.9 M |
| Memory Bandwidth | 7.03 GB/s (2.74%) | 4.99 GB/s (1.95%) |
| Speedup | 1.0× | 0.71× |

### 4.2 Performance Characteristics

**Surprising Result**: The constant memory implementation performed 29% slower than global memory, contrary to expectations. Analysis reveals:

1. **Access Pattern Mismatch**: While constant memory excels at broadcast reads, ray tracing exhibits divergent access patterns as different rays hit different spheres
2. **Cache Thrashing**: The 8KB constant cache may experience conflicts with divergent warp execution
3. **Occupancy Limitations**: Both kernels achieve only 12.5% occupancy, limiting latency hiding

### 4.3 Scaling Analysis

Performance scales linearly with pixel count and sample count:

| Resolution | Estimated Time | FPS |
|------------|----------------|-----|
| 360p | 8.3 seconds | 0.120 |
| 720p | 33.2 seconds | 0.030 |
| 1080p | 74.7 seconds | 0.013 |
| 4K | 298.8 seconds | 0.003 |

### 4.4 Memory Bandwidth Analysis

The implementation achieves only 2.74% of theoretical memory bandwidth (256 GB/s), indicating:
- Compute-bound rather than memory-bound workload
- Significant thread divergence in ray traversal
- Poor memory coalescing due to scattered sphere access

## 5. Challenges and Solutions

### 5.1 Technical Challenges

1. **Texture Coordinate Singularities**: Addressed using clamped texture addressing for polar coordinates
2. **Numerical Precision**: Self-intersection artifacts resolved with appropriate epsilon values
3. **Memory Constraints**: Constant memory limited scene to 64 spheres maximum

### 5.2 Performance Bottlenecks

- **Thread Divergence**: Different rays take vastly different paths through recursion
- **Register Pressure**: Complex shading calculations limit occupancy
- **Random Memory Access**: Sphere intersection tests lack spatial coherence

## 6. Conclusions and Future Work

### 6.1 Key Findings

1. Ray tracing exhibits inherently divergent execution patterns that challenge GPU efficiency
2. Traditional GPU memory optimizations may not apply directly to ray tracing workloads
3. Despite low efficiency metrics, GPU acceleration still provides significant speedup over CPU implementation

### 6.2 Future Improvements

1. **Spatial Acceleration Structures**: Implement BVH or k-d trees to reduce intersection tests
2. **Wavefront Path Tracing**: Reorganize computation to improve SIMD efficiency
3. **OptiX Integration**: Leverage NVIDIA's specialized ray tracing cores (RT cores)
4. **Multi-GPU Scaling**: Distribute rendering across multiple GPUs

### 6.3 Learning Outcomes

This project provided hands-on experience with:
- CUDA programming and GPU memory hierarchies
- Performance analysis and optimization techniques
- Physical rendering algorithms and mathematics
- Challenges of mapping irregular algorithms to GPU architectures

## References

1. Shirley, P. (2020). *Ray Tracing in One Weekend*. https://raytracing.github.io/
2. NVIDIA. (2023). *CUDA C++ Programming Guide*. NVIDIA Corporation.
3. Pharr, M., Jakob, W., & Humphreys, G. (2016). *Physically Based Rendering: From Theory to Implementation*. Morgan Kaufmann.
4. Aila, T., & Laine, S. (2009). *Understanding the efficiency of ray traversal on GPUs*. Proceedings of HPG 2009.

---

*Appendix: Sample rendered output available in out_global.ppm and out_const.ppm*