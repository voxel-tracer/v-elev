#include "pdf.h"

__device__ float cosine_pdf::value(const float3& direction) const {
	float cosine = dot(normalize(direction), uvw.w());
	if (cosine > 0)
		return cosine / M_PI;
	return 0;
}

__device__ float3 cosine_pdf::generate(seed_t seed) const {
	return uvw.local(random_cosine_direction(seed));
}
