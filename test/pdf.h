#pragma once

#include "onb.h"

class pdf {
public:
	virtual float value(const float3& direction) const = 0;
	virtual float3 generate(seed_t seed) const = 0;
	~pdf() {}
};

class cosine_pdf : public pdf {
public:
	cosine_pdf(const float3& w) { uvw.build_from_w(w); }
	virtual float value(const float3& direction) const {
		float cosine = dot(normalize(direction), uvw.w());
		if (cosine > 0)
			return cosine / M_PI;
		else
			return 0;
	}
	virtual float3 generate(seed_t seed) const {
		return uvw.local(random_cosine_direction(seed));
	}
	onb uvw;
};