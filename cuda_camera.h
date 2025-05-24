#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include "cuda_vec3.h"
#include "cuda_ray.h"

struct Camera {
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 w,u,v;
    float lens_radius;

    __host__ __device__
    Camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist) {
    float theta = vfov * 3.14159265f / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    w = (lookfrom - lookat).normalized();
    u = vup.cross(w).normalized();
    v = w.cross(u);

    origin = lookfrom;
    horizontal = u * viewport_width * focus_dist;
    vertical = v * viewport_height * focus_dist;
    lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f - w * focus_dist;

    lens_radius = aperture / 2.0f;
}

    __host__ __device__
    Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left_corner + horizontal * s + vertical * t - origin);
    }
};

#endif
