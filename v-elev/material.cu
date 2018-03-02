#include "material.h"
#include "pdf.h"

__device__ float lambertian_scattering_pdf(const hit_record& rec, const ray& scattered) {
	float cosine = dot(rec.normal, normalize(scattered.direction));
	if (cosine < 0) return 0;
	return cosine / M_PI;
}

__device__ bool scatter_lambertian(const float3& albedo, const hit_record& hrec, seed_t seed, scatter_record& srec) {
	cosine_pdf p = cosine_pdf(hrec.normal);
	srec.scattered = ray(hrec.hit_p, p.generate(seed));
	float pdf_val = p.value(srec.scattered.direction);
	float scattering_pdf = lambertian_scattering_pdf(hrec, srec.scattered);
	srec.attenuation = albedo*scattering_pdf / pdf_val;
	return pdf_val > 0;
}
