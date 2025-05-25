// ALL GPU SIDE LOGIC WILL COME HERE!
// DEVICE FUNCTIONS/GLOBAL FUNCTIONS

#include "cuda_kernels.h"
#define WIDTH 640
#define HEIGHT 640
#define SAMPLES_PER_PIXEL 1000
#define MAX_DEPTH 50

// linear congruential generator -> used to control randomness on cuda
__device__ float lcg(int* seed) {
    const int a = 1664525;
    const int c = 1013904223;
    const int m = 0x7fffffff;
    *seed = (a * (*seed) + c) & m;
    return float(*seed) / m;
}


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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int seed = x + y * WIDTH;
    vec3 pixel_color = vec3(0, 0, 0);

    for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
        float u = (x + lcg(&seed)) / float(WIDTH - 1);
        float v = (y + lcg(&seed)) / float(HEIGHT - 1);

        Ray r = cam->get_ray(u, v);
        pixel_color = pixel_color + traceRay(r, spheres, n);  // single-bounce traceRay for now
    }

    pixel_color = pixel_color * (1.0f / SAMPLES_PER_PIXEL);  // average

    int idx = (y * WIDTH + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * pixel_color.x);
    image[idx + 1] = (unsigned char)(255.99f * pixel_color.y);
    image[idx + 2] = (unsigned char)(255.99f * pixel_color.z);
}

__device__ bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, int* seed, const Sphere& sphere) {
    if (sphere.mat == LAMBERTIAN) {
        vec3 scatter_dir = rec.normal + random_unit_vector(seed);
        scattered = Ray(rec.p, scatter_dir);
        attenuation = sphere.albedo;
        return true;
    }

    if (sphere.mat == METAL) {
        vec3 reflected = reflect(r_in.direction.normalized(), rec.normal);
        scattered = Ray(rec.p, reflected + sphere.fuzz * random_unit_vector(seed));
        attenuation = sphere.albedo;
        return dot(scattered.direction, rec.normal) > 0;
    }

    if (sphere.mat == DIELECTRIC) {
        attenuation = vec3(1.0f, 1.0f, 1.0f);  // glass = no attenuation
        float refraction_ratio = rec.front_face ? (1.0f / sphere.ir) : sphere.ir;

        vec3 unit_direction = r_in.direction.normalized();
        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > lcg(seed))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.p, direction);
        return true;
    }

    return false;
}
