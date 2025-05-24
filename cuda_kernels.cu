// ALL GPU SIDE LOGIC WILL COME HERE!
// DEVICE FUNCTIONS/GLOBAL FUNCTIONS

#include "cuda_kernels.h"
#define WIDTH 640
#define HEIGHT 640
#define SAMPLES_PER_PIXEL 100
#define MAX_DEPTH 50

// GPU side code -> takes Ray r, array of sphere obj and n = num spheres -> return vec3 of color based on whether ray has hit anything or not.
__device__ vec3 traceRay(const Ray& r, Sphere* spheres, int n) {
    // Only worry about intersections beyon t_min
    float t_min = 0.001f;
    // t_mx furthest distance to check if ray has hit anything
    float t_max = 1.0e20f;

    // Store info about closest hit
    HitRecord rec;
    // Was sphere hit
    bool hit_anything = false;
    // Current closest valid t for hit
    float closest = t_max;

    // Loop over all spheres
    for (int i = 0; i < n; ++i) {
        HitRecord temp;
        if (spheres[i].hit(r, t_min, closest, temp)) {
            // If ray hits anything closer than t then mark hit_anything as true
            hit_anything = true;
            // Update closest to new
            closest = temp.t;
            // Store
            rec = temp;
        }
    }

    // Return color based on hit
    if (hit_anything) {
        // Normalise the hit normal (upward from surafce)
        vec3 normal = rec.normal.normalized();
        return 0.5f * vec3(normal.x + 1, normal.y + 1, normal.z + 1);
    }

    // Sky gradient -> no spheres were hit!
    vec3 unit_dir = r.direction.normalized();
    float t = 0.5f * (unit_dir.y + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}


// Main cuda kernal -> each thread computes one pixels color
// Image : GPU memory buffer
// cam : ptr to camera
// spheres -> scene in this case
// n : number of spheres
__global__ void rayKernel(unsigned char* image, Camera* cam, Sphere* spheres, int n) {
    // Basic pizel coords 
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (x >= WIDTH || y >= HEIGHT) return;

    // Mapping to screen space
    // Converts pixel coords from aobve into noramlised UV space
    // Used to geenrate primary rays
    float u = float(x) / (WIDTH - 1);
    float v = float(y) / (HEIGHT - 1);

    // Generate and tracre rays
    

    // Call camera to get ray for this pixel
    Ray r = cam->get_ray(u, v);
    // CALL ABOVE DEVICE side kernal to get pixel color
    vec3 color = traceRay(r, spheres, n);

    // Write color to image from above
    int idx = (y * WIDTH + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * color.x);
    image[idx + 1] = (unsigned char)(255.99f * color.y);
    image[idx + 2] = (unsigned char)(255.99f * color.z);
}
