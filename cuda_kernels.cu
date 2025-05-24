// ALL GPU SIDE LOGIC WILL COME HERE!
// DEVICE FUNCTIONS/GLOBAL FUNCTIONS

#include "cuda_kernels.h"
#define WIDTH 640
#define HEIGHT 640
#define SAMPLES_PER_PIXEL 100
#define MAX_DEPTH 50

__device__ vec3 traceRay(const Ray& r, Sphere* spheres, int n) {
    float t_min = 0.001f;
    float t_max = 1.0e20f;
    HitRecord rec;
    bool hit_anything = false;
    float closest = t_max;

    for (int i = 0; i < n; ++i) {
        HitRecord temp;
        if (spheres[i].hit(r, t_min, closest, temp)) {
            hit_anything = true;
            closest = temp.t;
            rec = temp;
        }
    }

    if (hit_anything) {
        // Visualize the surface normal as color
        vec3 normal = rec.normal.normalized();
        return 0.5f * vec3(normal.x + 1, normal.y + 1, normal.z + 1);
    }

    // Sky gradient
    vec3 unit_dir = r.direction.normalized();
    float t = 0.5f * (unit_dir.y + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

__global__ void rayKernel(unsigned char* image, Camera* cam, Sphere* spheres, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    float u = float(x) / (WIDTH - 1);
    float v = float(y) / (HEIGHT - 1);

    Ray r = cam->get_ray(u, v);
    vec3 color = traceRay(r, spheres, n);

    int idx = (y * WIDTH + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * color.x);
    image[idx + 1] = (unsigned char)(255.99f * color.y);
    image[idx + 2] = (unsigned char)(255.99f * color.z);
}
