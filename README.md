# CUDA Ray Tracing

A GPU-accelerated physically-based ray tracer implemented in CUDA, achieving real-time rendering performance through optimized memory hierarchies and kernel configurations.

## Features

- **Multiple Material Types**:
  - Lambertian (diffuse) surfaces
  - Metallic surfaces with adjustable fuzziness
  - Dielectric materials (glass/transparent)
  - Textured surfaces with environment mapping
  - Checker pattern procedural textures

- **Optimizations**:
  - Constant memory for scene data
  - Texture memory for environment maps
  - Optimized kernel configurations (2D grid and 1D linearized)
  - Coalesced memory access patterns

- **Performance**:
  - 1920×1080 resolution with 300 samples per pixel
  - Maximum recursion depth of 30
  - ~14 billion rays processed per frame
  - Benchmarking suite for performance analysis

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 10.0 or newer
- GCC compiler with OpenMP support
- Linux operating system (tested on Ubuntu)

## Building

The project includes both CPU (OpenMP) and GPU (CUDA) implementations:

```bash
# Build everything (CPU + CUDA versions)
make all

# Build only CUDA raytracer
make cudaray

# Build benchmark suite
make benchmark

# Clean build artifacts
make clean
```

## Running

### Quick Start
Use the provided run script to build and run everything:
```bash
./run.sh
```

This will:
1. Clean previous builds
2. Build the CUDA raytracer
3. Build the benchmark scaling tests
4. Run the raytracer (outputs to `CudaOut.ppm`)
5. Run performance benchmarks

### Manual Execution
```bash
# Run CUDA raytracer
./cudaray

# Run benchmark tests
./benchmark_scaling

# Run CPU version (if built)
./raytracer
```

## Output

The raytracer generates a PPM image file:
- **CUDA version**: `CudaOut.ppm`
- **CPU version**: `tmp.ppm`

View the output with any PPM-compatible image viewer:
```bash
display CudaOut.ppm    # ImageMagick
gimp CudaOut.ppm       # GIMP
```

## Scene Configuration

The default scene contains:
- 1 large ground sphere with checker pattern
- 5 textured spheres with environment maps
- 2 perfect mirror spheres
- 49 randomly placed small spheres with mixed materials

Total: 57 spheres

## Project Structure

```
Cuda-Raytracing/
├── cudaray.cu           # Main CUDA implementation
├── cuda_kernels.cu      # CUDA kernel implementations
├── cuda_kernels.h       # Kernel declarations
├── benchmark_scaling.cu # Performance benchmarking
├── main.c               # CPU OpenMP implementation
├── include/             # Header files
│   ├── cuda_*.h        # CUDA-specific headers
│   └── *.h             # Shared headers
├── Common/              # NVIDIA SDK helper files
├── test_textures/       # Environment map textures
└── makefile            # Build configuration
```

## Performance

On NVIDIA GeForce GTX 1070:
- Resolution: 1920×1080
- Samples per pixel: 300
- Max recursion depth: 30
- Typical render time: ~2-5 seconds (scene dependent)

The benchmark suite tests:
- Different kernel configurations
- Memory access patterns
- Scaling with scene complexity
- Samples per pixel impact

## Technical Details

### Memory Usage
- **Constant Memory**: Scene geometry (up to 64 spheres)
- **Texture Memory**: Environment maps for realistic reflections
- **Global Memory**: Frame buffer and dynamic scene data

### Kernel Configurations
1. **2D Grid**: Maps directly to image pixels
2. **1D Linearized**: Better for certain GPU architectures

### Random Number Generation
Uses cuRAND for Monte Carlo sampling in:
- Anti-aliasing (jittered sampling)
- Diffuse material scattering
- Depth of field effects

## Customization

### Adjusting Quality Settings
Edit the defines in `cudaray.cu`:
```cpp
#define WIDTH 1920
#define HEIGHT 1080
#define SAMPLES_PER_PIXEL 300
#define MAX_DEPTH 30
```

### Adding Textures
Place texture files in `test_textures/` and update the texture loading code in `cudaray.cu`.

### Scene Modification
Edit the `build_scene()` function in `cudaray.cu` to customize sphere placement, materials, and properties.

## Troubleshooting

- **Out of memory**: Reduce resolution or samples per pixel
- **Compilation errors**: Ensure CUDA toolkit is properly installed
- **Performance issues**: Check GPU compute capability and adjust kernel configuration

## References

Based on Peter Shirley's "Ray Tracing in One Weekend" series, adapted for GPU acceleration using CUDA.

## License

This project is for educational purposes. See individual file headers for specific licensing information.