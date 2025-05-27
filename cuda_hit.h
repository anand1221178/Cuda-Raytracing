#ifndef CUDA_HIT_H
#define CUDA_HIT_H

#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_sphere.h"
#include "cuda_material.h"

struct HitRecord {
    vec3 p;         // point of intersection
    vec3 normal;
    float t;
    bool front_face;
    vec3 albedo; // color for the material at the hit point
    MaterialType material; // ID of the material at the hit point
    float fuzz; // Fuzziness factor for the material at the hit point
    float ir; //Dielectric constant for the material at the hit point

    __host__ __device__ void set_face_normal(const Ray& r, const vec3& outward_normal) {
        front_face = r.direction.dot(outward_normal) < 0.0f;
        normal = front_face ? outward_normal : outward_normal * -1.0f;
    }
};

#endif
