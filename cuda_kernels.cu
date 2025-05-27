// ALL GPU SIDE LOGIC WILL COME HERE!
// DEVICE FUNCTIONS/GLOBAL FUNCTIONS

#include "cuda_kernels.h"
#define WIDTH 1920
#define HEIGHT 1080
#define SAMPLES_PER_PIXEL 1000
#define MAX_DEPTH 10

// FORWARD DECLARIONS
__device__ bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, int* seed);



// linear congruential generator -> used to control randomness on cuda
__device__ float lcg(int* seed) {
    const int a = 1664525;
    const int c = 1013904223;
    const int m = 0x7fffffff;
    *seed = (a * (*seed) + c) & m;
    return float(*seed) / m;
}

// -------------_RANDOM VECTOR KERNALS------------//
__device__ vec3 random_in_unit_sphere(int * seed){
    while(true)
    {
        vec3 p = vec3(2.0f * lcg(seed) - 1.0f,2.0f * lcg(seed) - 1.0f,2.0f * lcg(seed) - 1.0f);
        if(p.length_squared() < 1.0f){
            return p;
        }
    }
}

// RANDOM UNIT VECTOR -> WILL BE NORMAL
__device__ vec3 random_unit_vector(int* seed) {
    return random_in_unit_sphere(seed).normalized();
}

// 3d considerations
__device__ vec3 random_on_hemisphere(const vec3& normal, int* seed) {
    vec3 dir = random_unit_vector(seed);
    return (dir.dot(normal) > 0.0f) ? dir : -1.0f * dir;
}


// DEVICE REFLECT FUNCTION
__device__ vec3 reflect(const vec3& V, const vec3& n)
{
    return V - 2 * V.dot(n) * n;
}

// DEVICE REFRACT FUNCTION -> using snells law
// n1sin(theta1) = n2sin(theta2)
__device__ vec3 refract(const vec3& uv, const vec3& n, float eta_ratio)
{
    float cos_theta = fminf((-uv).dot(n), 1.0f);
    vec3 r_out_perp = eta_ratio * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

//approximate reflectance of dielectric surfaces
__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}




// GPU side code -> takes Ray r, array of sphere obj and n = num spheres -> return vec3 of color based on whether ray has hit anything or not.
__device__ vec3 traceRay(const Ray& r, Sphere* spheres, int n, int depth, int* seed) {
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
    if (depth <= 0)
    return vec3(0, 0, 0);  // absorption / recursion cap

    if (hit_anything) {
        Ray scattered;
        vec3 attenuation;
        if (scatter(r, rec, attenuation, scattered, seed)) {
            return attenuation * traceRay(scattered, spheres, n, depth - 1, seed);
        }
        return vec3(0, 0, 0);
        // return rec.albedo;
    }



    /* DEBUG
     if (hit_anything) {
         Ray dummy;
         vec3 attenuation = vec3(0.9f, 0.9f, 0.9f);
         return attenuation;
     }*/
    
    

    if (depth <= 0) return vec3(1, 0, 1); // purple for recursion cap

    // Sky gradient -> no spheres were hit!
    // --------- in traceRay(), *after* the hit_anything block ----------
    vec3 unit_dir = r.direction.normalized();
    float t = 0.5f * (unit_dir.y + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f)   // white at horizon
        +           t  * vec3(0.5f, 0.7f, 1.0f); // blue at zenith


}


/*Main cuda kernal -> each thread computes one pixels color
Image : GPU memory buffer
cam : ptr to camera
spheres -> scene in this case
n : number of spheres*/
__global__ void rayKernel(unsigned char* image, Camera* cam, Sphere* spheres, int n) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int seed = x + y * WIDTH;
    vec3 pixel_color = vec3(0, 0, 0);

    for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
        float u = (x + lcg(&seed)) / float(WIDTH - 1);
        float v = (y + lcg(&seed)) / float(HEIGHT - 1);

        Ray r = cam->get_ray(u, v, &seed, lcg);

        pixel_color += traceRay(r, spheres, n, MAX_DEPTH, &seed); //MULTI DEPTH RAY BOUNCE
    }

    pixel_color = pixel_color * (1.0f / SAMPLES_PER_PIXEL);

    // Gamma correction (sRGB, gamma = 2.0)
    pixel_color.x = sqrtf(fminf(fmaxf(pixel_color.x, 0.0f), 1.0f));
    pixel_color.y = sqrtf(fminf(fmaxf(pixel_color.y, 0.0f), 1.0f));
    pixel_color.z = sqrtf(fminf(fmaxf(pixel_color.z, 0.0f), 1.0f));

    int idx = (y * WIDTH + x) * 3;
    image[idx + 0] = (unsigned char)(255.99f * pixel_color.x);
    image[idx + 1] = (unsigned char)(255.99f * pixel_color.y);
    image[idx + 2] = (unsigned char)(255.99f * pixel_color.z);
}

/* TAKES incoming ray and a hitrecord
 Outpuit scattered ray and attenuation
 ABOVE BEHAVIOUR DIFFERS BASED ON rec.material
 Include seed for reproducebility
 r_in -> incoming ray
 rec -> hit info - did we hit anything?
Attenuation -> color tp multiply into the pixel
scattered -> new ray dir*/

__device__ bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, int* seed)
{
// FUCNTION TOO HANDLE DIFFERENT SCATTERING RULES
    switch(rec.material){
        // ------FLOOR-----//
        case CHECKER:
        // FALL THROUGH TO LAMBERT
        //---------LAMBERT--------//
        case LAMBERTIAN:{
            vec3 scatter_dir = rec.normal + random_unit_vector(seed);
            if (scatter_dir.length_squared() < 1e-8f)
                scatter_dir = rec.normal;

            // just before you build the secondary ray
            const vec3 offset = rec.normal * 0.001f;          // 1 mm bias
            scattered = Ray(rec.p + offset, scatter_dir);     // <-- use p + bias
            attenuation = rec.albedo;
            return true;
        }
        break;
        //---------METAL --------//
        case METAL: {
            vec3 unit_dir = r_in.direction.normalized();
            vec3 reflected = reflect(unit_dir, rec.normal);
            vec3 perturbed = reflected + rec.fuzz * random_unit_vector(seed);
            const vec3 offset = rec.normal * 0.001f;          // 1 mm bias
            scattered = Ray(rec.p + offset, perturbed);     // <-- use p + bias
            attenuation = rec.albedo;
            return (scattered.direction.dot(rec.normal) > 0.0f);
        }
        break;
        //-----------DIELETRIC-------//
        case DIELECTRIC: {
            attenuation = vec3(1.0f);  // dielectric is transparent
        
            // Decide if we're entering or exiting
            float refraction_ratio = rec.front_face ? (1.0f / rec.ir) : rec.ir;
        
            // Normalize incoming ray
            vec3 unit_dir = r_in.direction.normalized();
        
            // Compute cosθ and sin²θ
            float cos_theta = fminf((-unit_dir).dot(rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        
            // Check for total internal reflection
            bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        
            // Decide: refract or reflect?
            vec3 direction;
            if (cannot_refract || schlick(cos_theta, refraction_ratio) > lcg(seed)) {
                direction = reflect(unit_dir, rec.normal); // reflect
            } else {
                direction = refract(unit_dir, rec.normal, refraction_ratio); // refract
            }
        
            // just before you build the secondary ray
            const vec3 offset = rec.normal * 0.001f;          // 1 mm bias
            scattered = Ray(rec.p + offset, direction);     // <-- use p + bias

            return true;
        }        

        break;
    }
}


__device__ vec3 checker_color(const vec3& p) {
    float s = sinf(10.0f*p.x) * sinf(10.0f*p.z);
    return s < 0 ? vec3(0.1f) : vec3(0.9f);
}





