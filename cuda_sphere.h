#ifndef CUDA_SPHERE_H
#define CUDA_SPHERE_H

#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_hit.h"

struct Sphere {
    vec3 center;
    float radius;

    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
        vec3 oc = r.origin - center;
        float a = r.direction.dot(r.direction);
        float half_b = oc.dot(r.direction);
        float c = oc.dot(oc) - radius * radius;
        float discriminant = half_b * half_b - a * c;

        if (discriminant < 0) return false;
        float sqrt_d = sqrtf(discriminant);

        float root = (-half_b - sqrt_d) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_d) / a;
            if (root < t_min || root > t_max)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) * (1.0f / radius);
        rec.set_face_normal(r, outward_normal);
        return true;
    }
};

#endif
