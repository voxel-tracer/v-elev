#pragma once

#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
//#define M_PI_4     0.785398163397448309616  // pi/4

#define EPS     0.1f

#include <vector_functions.h>
#include <helper_math.h>

#include <curand_kernel.h>
#define seed_t curandStatePhilox4_32_10_t*
__device__ float RandomFloat01(seed_t seed) {
	return curand_uniform(seed);
}

typedef union {
	float3 v;
	float a[3];
} float3u;

typedef union {
	int3 v;
	int a[3];
} int3u;

#define NO_HIT  0
#define X       1
#define Y       2
#define Z       3

struct cu_hit {
	float hit_t = 0;
	uint hit_face = 0; //TODO use enum instead

	__host__ __device__ cu_hit() {}
	__device__ cu_hit(const cu_hit& h) : hit_t(h.hit_t), hit_face(h.hit_face) {}
	__device__ cu_hit(const float t, const uint face) : hit_t(t), hit_face(face) {}
};

__device__ float3 random_cosine_direction(seed_t seed) {
	float r1 = RandomFloat01(seed);
	float r2 = RandomFloat01(seed);
	float z = sqrtf(1 - r2);
	float phi = 2 * M_PI*r1;
	float x = cosf(phi) * 2 * sqrtf(r2);
	float y = sinf(phi) * 2 * sqrtf(r2);
	return make_float3(x, y, z);
}

__device__ float3 random_to_sphere(seed_t seed) {
	float3 p;
	do {
		p = 2.0*make_float3(RandomFloat01(seed), RandomFloat01(seed), RandomFloat01(seed)) - make_float3(1, 1, 1);
	} while (dot(p, p) >= 1.0);
	return normalize(p);
}

__device__ float3 random_to_sphere(seed_t seed, float radius, float distance_squared) {
	float r1 = RandomFloat01(seed);
	float r2 = RandomFloat01(seed);
	float z = 1 + r2*(sqrt(1 - radius*radius / distance_squared) - 1);
	float phi = 2 * M_PI*r1;
	float x = cos(phi)*sqrt(1 - z*z);
	float y = sin(phi)*sqrt(1 - z*z);
	return make_float3(x, y, z);
}

__device__ bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
	float3 uv = normalize(v);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (discriminant > 0) {
		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
		return true;
	}
	return false;
}
__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0*r0;
	return r0 + (1 - r0)*pow((1 - cosine), 5);
}
__device__ float squared_length(const float3& v) {
	return dot(v, v);
}
__device__ int signum(const float f) {
	return f == 0 ? 0 : (f < 0 ? -1 : +1);
}
__device__ float max(const float3& v) {
	return v.x >= v.y ? (v.x >= v.z ? v.x : v.z) : (v.y >= v.z ? v.y : v.z);
}
__device__ float min(const float3& v) {
	return v.x <= v.y ? (v.x <= v.z ? v.x : v.y) : (v.y <= v.z ? v.y : v.z);
}
__device__ uint max_id(const float3& v) {
	return v.x >= v.y ? (v.x >= v.z ? 0 : 2) : (v.y >= v.z ? 1 : 2);
}
__device__ uint min_id(const float3& v) {
	return v.x <= v.y ? (v.x <= v.z ? 0 : 2) : (v.y <= v.z ? 1 : 2);
}
