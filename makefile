# Compiler and flags
CC = gcc
NVCC = nvcc
CFLAGS = -fopenmp -I ./include 
CUDAFLAGS = -I ./include -O3
CUDAFLAGS = -I ./Common -I ./include -O3 -rdc=true -diag-suppress 20054 -diag-suppress 26 -std=c++14 --expt-relaxed-constexpr
LDFLAGS = -lm

# Source files
CPUSRCS = main.c \
       sources/allocator.c \
       sources/camera.c \
       sources/color.c \
       sources/hitRecord.c \
       sources/material.c \
       sources/outfile.c \
       sources/ray.c \
       sources/sphere.c \
       sources/texture.c \
       sources/util.c

CUDASRCS = cudaray.cu cuda_kernels.cu

# Output binaries
CPUTARGET = raytracer
CUDATARGET = cudaray

# Default target builds both
all: $(CPUTARGET) $(CUDATARGET)

# Build CPU OpenMP raytracer
$(CPUTARGET): $(CPUSRCS)
	$(CC) $(CFLAGS) $(CPUSRCS) -o $(CPUTARGET) $(LDFLAGS)

# Build CUDA raytracer
$(CUDATARGET): $(CUDASRCS)
	$(NVCC) $(CUDAFLAGS) stb_image_impl.cpp $(CUDASRCS) -o $(CUDATARGET)

# Clean up
clean:
	rm -f $(CPUTARGET) $(CUDATARGET) raytracer tmp.ppm CudaOut CudaOut.ppm *.o
