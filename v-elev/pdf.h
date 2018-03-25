#pragma once

#include <limits.h>
#include "onb.h"
#include "sun.h"

class pdf {
public:
	__device__ virtual float value(const float3& direction) const = 0;
	__device__ virtual float3 generate(seed_t seed) const = 0;
	__device__ ~pdf() {}
};

class cosine_pdf : public pdf {
public:
	__device__ cosine_pdf(const float3& w) { uvw.build_from_w(w); }
	__device__ virtual float value(const float3& direction) const {
		float cosine = dot(normalize(direction), uvw.w());
		if (cosine > 0)
			return cosine / M_PI;
		else
			return 0;
	}
	__device__ virtual float3 generate(seed_t seed) const {
		return uvw.local(random_cosine_direction(seed));
	}
	onb uvw;
};

class mixture_pdf : public pdf {
public:
	__device__ mixture_pdf(pdf *p0, pdf *p1) { p[0] = p0; p[1] = p1; }
	__device__ virtual float value(const float3& direction) const {
		return 0.5 * p[0]->value(direction) + 0.5 *p[1]->value(direction);
	}
	__device__ virtual float3 generate(seed_t seed) const {
		if (cu_drand48(seed) < 0.5)
			return p[0]->generate(seed);
		else
			return p[1]->generate(seed);
	}
	pdf *p[2];
};

class sun_pdf : public pdf {
public:
	__device__ sun_pdf(const sun *p, const float3& origin) : ptr(p), o(origin) {}
	__device__ virtual float value(const float3& direction) const {
		return ptr->pdf_value(o, direction);
	}
	__device__ virtual float3 generate(seed_t seed) const {
		return ptr->random(seed, o);
	}
	float3 o;
	const sun *ptr;
};
