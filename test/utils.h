#pragma once

#include <math.h>

#define uint unsigned int
#define seed_t uint&
#define M_PI       3.14159265358979323846   // pi

typedef struct float3 {
	float x;
	float y;
	float z;
} float3;

inline float3 make_float3(float x, float y, float z) {
	float3 f3;
	f3.x = x;
	f3.y = y;
	f3.z = z;
	return f3;
}

float3 normalize(float3 v) {
	float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
	return make_float3(v.x / len, v.y / len, v.z / len);
}

float3 cross(float3 a, float3 b) {
	return make_float3((a.y * b.z - a.z * b.y), (-(a.x *b.z - a.z * b.x)), (a.x * b.y - a.y * b.x));
}

float dot(float3 a, float3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline float3 operator*(float t, const float3 &v) {
	return make_float3(t*v.x, t*v.y, t*v.z);
}

inline float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


float RandomFloat01(uint& seed) {
	seed = (214013 * seed + 2531011);
	return (float)((seed >> 16) & 0x7FFF) / 32767;
}

float3 random_cosine_direction(seed_t seed) {
	float r1 = RandomFloat01(seed);
	float r2 = RandomFloat01(seed);
	float z = sqrtf(1 - r2);
	float phi = 2 * M_PI*r1;
	float x = cosf(phi) * 2 * sqrtf(r2);
	float y = sinf(phi) * 2 * sqrtf(r2);
	return make_float3(x, y, z);
}
