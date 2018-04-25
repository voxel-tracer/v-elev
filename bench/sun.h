#pragma once

#include "onb.h"

class sun {
public:
	sun(const float3& center, const float radius, const float3& color) : c(center), r(radius), clr(color) {}
	__device__ float pdf_value(const float3 o, const float3 v) const {
		const float dot_v_co = dot(v, c - o) / length(v);
		const float r_sqr = r*r;
		const float co_len_sqr = dot(c - o, c - o);

		if (dot_v_co*dot_v_co > (co_len_sqr - r_sqr)) {
			float solid_angle = 2 * M_PI*(1 - sqrt(1 - r_sqr / co_len_sqr));
			return  1 / solid_angle;
		}
		else
			return 0;
	}
	__device__ float3 random(seed_t seed, const float3& o) const {
		float3 direction = c - o;
		float distance_squared = dot(direction, direction);
		onb uvw;
		uvw.build_from_w(direction);
		return uvw.local(random_to_sphere(seed, r, distance_squared));
	}

	float3 c;
	float r;
	float3 clr;
};
