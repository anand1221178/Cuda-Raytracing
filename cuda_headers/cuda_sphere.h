#ifndef CUDA_SPHERE_H
#define CUDA_SPHERE_H

#include "cuda_vec3.h"
#include "cuda_ray.h"
#include "cuda_hit.h"
#include "cuda_material.h"

struct Sphere {
      vec3  center;
      float radius;
      MaterialType mat;
      vec3  albedo;
      float fuzz;
      float ir;
      int   texture_id;


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

        rec.material = mat;
        rec.fuzz = fuzz;
        rec.ir = ir;
        rec.texture_id = this->texture_id;

        if (mat == CHECKER) {
            vec3 p_local = rec.p - center;
            int ix = static_cast<int>(floorf(p_local.x * 0.25f));
            int iz = static_cast<int>(floorf(p_local.z * 0.25f));
            bool dark = ((ix + iz) & 1);
            rec.albedo = dark ? vec3(0.05f) : vec3(0.95f);
        } else if (mat == TEXTURED) {
            const float EPS = 1.0e-4f; 

            vec3 p_local = (rec.p - center).normalized();

        /* longitude ------------------------------------------------------------ */
        float u = 0.5f + atan2f(p_local.z, p_local.x) / (2.0f * M_PI);
        u = u - floorf(u);
        u = fminf(u, 1.0f - EPS);

        /* latitude ------------------------------------------------------------- */
        float v = 0.5f - asinf(p_local.y) / M_PI;
        v = fminf(fmaxf(v, EPS), 1.0f - EPS);

        rec.u = u;
        rec.v = v;
        } else {
            rec.albedo = albedo;
        }

        return true;
    }
};

#endif
