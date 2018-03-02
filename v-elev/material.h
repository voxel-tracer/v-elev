#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "utils.h"
#include "ray.h"

struct scatter_record {
	ray scattered;
	float3 attenuation;
};

struct hit_record {
	__device__ hit_record() {}
	__device__ hit_record(const float3& p, const float3& n) : hit_p(p), normal(n) {}

	float3 hit_p;
	float3 normal;
};

__device__ bool scatter_lambertian(const float3& albedo, const hit_record& hrec, seed_t seed, scatter_record& srec);

#endif /* MATERIAL_H_ */
