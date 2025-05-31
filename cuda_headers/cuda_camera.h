#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include "cuda_vec3.h"
#include "cuda_ray.h"

// ----------- device helper -------------
__device__ inline vec3 random_in_unit_disk(int* seed, float (*rng)(int*)) {
    while (true) {
        float x = 2.0f * rng(seed) - 1.0f;
        float y = 2.0f * rng(seed) - 1.0f;
        if (x*x + y*y < 1.0f) return vec3(x,y,0);
    }
}
// ---------------------------------------

struct Camera {
    vec3 origin, lower_left_corner, horizontal, vertical;
    vec3 u, v, w;
    float lens_radius;

    __host__ __device__
    Camera(vec3 lookfrom, vec3 lookat, vec3 vup,
           float vfov, float aspect, float aperture, float focus_dist)
    {
        float theta = vfov * 3.14159265f / 180.0f;
        float h = tanf(theta * 0.5f);
        float viewport_h = 2.0f * h;
        float viewport_w = aspect * viewport_h;

        w = (lookfrom - lookat).normalized();
        u = (w.cross(vup)).normalized();   // 
        v = w.cross(u);                  


        origin      = lookfrom;
        horizontal  = u * viewport_w * focus_dist;
        vertical    = v * viewport_h * focus_dist;
        lower_left_corner =
            origin - horizontal*0.5f - vertical*0.5f - w*focus_dist;

        lens_radius = aperture * 0.5f;
    }

    // NOTE: extra 'seed' arg so each thread passes its RNG pointer
    __device__
    Ray get_ray(float s, float t, int* seed, float (*rng)(int*)) const {
        vec3 rd     = lens_radius * random_in_unit_disk(seed, rng);
        vec3 offset = u * rd.x + v * rd.y;
        return Ray(origin + offset,
                   lower_left_corner + s*horizontal + t*vertical
                   - origin - offset);
    }
};

#endif
