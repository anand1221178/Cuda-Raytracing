#ifndef CUDA_SPHERE_H
#define CUDA_SPHERE_H

#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_hit.h"

enum MaterialType {
    LAMBERTIAN,
    METAL,
    DIELECTRIC
};

struct Sphere {
    vec3 center;
    float radius;
    MaterialType mat;
    vec3 albedo;  // color
    float fuzz;   // for METAL
    float ir;     // index of refraction for DIELECTRIC


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

        // MODIFY HIT RECORD TO ACCOUNT FOR MATERIAL PROPERTIES
        rec.material = mat;
        rec.albedo = albedo; // Set the color for the material at the hit point
        rec.fuzz = fuzz; // Set the fuzziness factor for the material at the hit point
        rec.ir = ir; // Set the index of refraction for the material at the hit point
        return true;
    }



};

#endif
