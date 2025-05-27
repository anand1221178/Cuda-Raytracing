#ifndef CUDA_SPHERE_H
#define CUDA_SPHERE_H

#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_hit.h"
#include "cuda_material.h"

// enum MaterialType {
//     LAMBERTIAN,
//     METAL,
//     DIELECTRIC
// };

struct Sphere {
    // Constructor
    __host__ __device__ Sphere(vec3 c, float r, MaterialType m, vec3 a, float f, float i): center(c), radius(r), mat(m), albedo(a), fuzz(f), ir(i) {}
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
        // rec.albedo = albedo; // Set the color for the material at the hit point
        rec.fuzz = fuzz; // Set the fuzziness factor for the material at the hit point
        rec.ir = ir; // Set the index of refraction for the material at the hit point

        if (mat == static_cast<MaterialType>(3))          // CHECKER
        {
            // Project hit-point into the X-Z plane of the sphere centre.
            vec3 p_local = rec.p - center;

            // ONE square == 4 units wide  ➜  big enough to see on a 1000-unit sphere
            int ix = static_cast<int>(floorf(p_local.x * 0.25f));
            int iz = static_cast<int>(floorf(p_local.z * 0.25f));

            bool dark = ((ix + iz) & 1);

            rec.albedo = dark ? vec3(0.05f)          // almost black
                            : vec3(0.95f);         // very bright
        }
        else
        {
            rec.albedo = albedo;
        }


        // DeBUG CODE
        // if (mat == static_cast<MaterialType>(3)) {
        //     vec3 p_local = rec.p - center;
        //     float pattern = sinf(10.0f * p_local.x) * sinf(10.0f * p_local.z);
        //     rec.albedo = (pattern < 0.0f) ? vec3(0.1f) : vec3(0.9f);
        // } else {
        //     rec.albedo = albedo;
        // }
        return true;
    }



};

#endif
