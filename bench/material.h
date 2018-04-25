#pragma once

#include "pdf.h"
#include "ray.h"

struct scatter_record {
	ray scattered;
	float3 attenuation;
	pdf *pdf_ptr;
};

struct hit_record {
	__device__ hit_record() {}
	__device__ hit_record(const float3& p, const float3& n) : hit_p(p), normal(n) {}

	float3 hit_p;
	float3 normal;
};

class material {
public:
	__device__ virtual bool scatter(const hit_record& hrec, scatter_record& srec) const {
		return false;
	}
	__device__ virtual float scattering_pdf(const hit_record& rec, const ray& scattered) const {
		return false;
	}
	__device__ virtual float3 emitted(const ray& r_in, const hit_record& rec, float u, float v, const float3& p) const { return make_float3(0); }
};

class lambertian : public material {
public:
	__device__ lambertian(const float3& a) : albedo(a) {}
	__device__ float scattering_pdf(const hit_record& rec, const ray& scattered) const {
		float cosine = dot(rec.normal, normalize(scattered.direction));
		if (cosine < 0) return 0;
		return cosine / M_PI;
	}
	__device__ bool scatter(const hit_record& hrec, scatter_record& srec) const {
		srec.attenuation = albedo;
		srec.pdf_ptr = new cosine_pdf(hrec.normal);
		return true; // caller need to make sure pdf.value > 0
	}

	float3 albedo;
};

