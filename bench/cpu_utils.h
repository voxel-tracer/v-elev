#pragma once

#include <vector_functions.h>
#include <helper_math.h>

float drand48(void);

float3 random_in_unit_disk() {
	float3 p;
	do {
		p = 2.0*make_float3(drand48(), drand48(), 0) - make_float3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}
