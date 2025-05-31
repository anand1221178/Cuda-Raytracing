#!/bin/bash

# CUDA Raytracing Project Runner
# Cleans, builds, and runs the entire project

echo "=== Cleaning previous builds ==="
make clean

echo "=== Building CUDA raytracer ==="
make cudaray

echo "=== Building benchmark scaling ==="
make benchmark

echo "=== Running CUDA raytracer ==="
./cudaray

echo "=== Running benchmark scaling ==="
./benchmark_scaling

echo "=== All tasks completed ==="