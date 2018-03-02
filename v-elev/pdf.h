#pragma once

#include <limits.h>
#include "onb.h"

struct cosine_pdf {
	__device__ cosine_pdf(const float3& w): uvw(w) {}

	__device__ float value(const float3& direction) const;
	__device__ float3 generate(seed_t seed) const;

	const onb uvw;
};
