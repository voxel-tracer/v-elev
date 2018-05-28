#ifndef VOXEL_MODEL_H_
#define VOXEL_MODEL_H_

#include <float.h>
#include "utils.h"

#include "ray.h"

struct voxelModel {
	__host__ voxelModel(const unsigned char *hm, uint w, uint h): heightmap(hm) { 
		// compute max elevation
		uint max_e = 0;
		for (uint y = 0, idx = 0; y < h; y++) {
			for (uint x = 0; x < w; x++, idx++) {
				if (heightmap[idx] > max_e)
					max_e = heightmap[idx];
			}
		}

		size = make_uint3(w, max_e, h);
	}

	__device__ voxelModel(const unsigned char *hm, const uint3& s) : heightmap(hm), size(s) { }

	__device__ bool hitDist(const ray& r, float3& hit_min) const {
		float invD, t0, t1, tmp;
		float tMin = EPS;
		float tMax = FLT_MAX;
		// x axis
		invD = 1 / r.direction.x;
		t0 = (0 - r.origin.x) * invD;
		t1 = (size.x - r.origin.x) * invD;
		if (t1 < t0) {
			tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = t0 > tMin ? t0 : tMin;
		tMax = t1 < tMax ? t1 : tMax;
		hit_min.x = t0;
		if (tMax <= tMin)
			return false;

		// y axis
		invD = 1 / r.direction.y;
		t0 = (0 - r.origin.y) * invD;
		t1 = (size.y - r.origin.y) * invD;
		if (t1 < t0) {
			tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = t0 > tMin ? t0 : tMin;
		tMax = t1 < tMax ? t1 : tMax;
		hit_min.y = t0;
		if (tMax <= tMin)
			return false;

		// z axis
		invD = 1 / r.direction.z;
		t0 = (0 - r.origin.z) * invD;
		t1 = (size.z - r.origin.z) * invD;
		if (t1 < t0) {
			tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tMin = t0 > tMin ? t0 : tMin;
		tMax = t1 < tMax ? t1 : tMax;
		hit_min.z = t0;
		if (tMax <= tMin)
			return false;

		return true;
	}

	__device__ bool hit(const ray& r, cu_hit& hit) const {
		//TODO I only need face and hit_min[face]
		float3u hit_min;
		if (!hitDist(r, hit_min.v))
			return false;

		float3u d; d.v = r.direction; //TODO make ray.direction float3u
		int3u d_sign;
		d_sign.v = make_int3(signum(d.v.x), signum(d.v.y), signum(d.v.z));

		// find which face was hit
		int face = max_id(hit_min.v);
		float minT = fmaxf(0, hit_min.a[face]); // t at intersection
		const float3 h_vec = r.point_at_parameter(minT); // intersected point
		int3u voxel; // intersected voxel at minT
		voxel.v = make_int3(
			h_vec.x + d_sign.v.x*EPS,
			h_vec.y + d_sign.v.y*EPS,
			h_vec.z + d_sign.v.z*EPS
		);

		float3u tDelta; // how much we need to travel in each axis to move from one voxel to another
		tDelta.v = 1 / fabs(d.v);

		float3u tVec; // how much we need to travel in each axis to hit next voxel 
		tVec.v.x = fabsf(voxel.v.x - r.origin.x + ((d_sign.v.x + 1) >> 1))*tDelta.v.x;
		tVec.v.y = fabsf(voxel.v.y - r.origin.y + ((d_sign.v.y + 1) >> 1))*tDelta.v.y;
		tVec.v.z = fabsf(voxel.v.z - r.origin.z + ((d_sign.v.z + 1) >> 1))*tDelta.v.z;

		//TODO just check if voxel is inside model instead of checking agains hit_max
		while (isInside(voxel.v) && !isVoxelFull(voxel.v)) {
			// we need to move enough to hit another voxel
			face = min_id(tVec.v);
			voxel.a[face] += d_sign.a[face];
			tVec.a[face] += tDelta.a[face];
		}

		if (!isInside(voxel.v) || !isVoxelFull(voxel.v))
			return false;

		//COMPUTE HIT RECORD
		hit.hit_t = tVec.a[face] - tDelta.a[face];
		hit.hit_face = face + 1;
		return true;
	}

#ifdef DBG_TRACING
	__device__ bool hit_trace(const ray& r, cu_hit& hit) const {
		printf("- hit_trace: -----------------------------------\n");

		float3u hit_min;
		if (!hitDist(r, hit_min.v)) {
			printf("bbox NO_HIT\n");
			return false;
		}
		
		printf("hit_min(%f, %f, %f)\n", hit_min.v.x, hit_min.v.y, hit_min.v.z);

		float3u d; d.v = r.direction; //TODO make ray.direction float3u
		int3u d_sign;
		d_sign.v = make_int3(signum(d.v.x), signum(d.v.y), signum(d.v.z));

		// find which face was hit
		int face = max_id(hit_min.v);
		const float minT = fmaxf(0, hit_min.a[face]); // t at intersection
		const float3 h_vec = r.point_at_parameter(minT); // intersected point
		int3u voxel; // intersected voxel at minT
		voxel.v = make_int3(
			h_vec.x + d_sign.v.x*EPS,
			h_vec.y + d_sign.v.y*EPS,
			h_vec.z + d_sign.v.z*EPS
		);
		printf("face: %d, voxel(%d, %d, %d)\n", face, voxel.v.x, voxel.v.y, voxel.v.z);

		float3u tDelta; // how much we need to travel in each axis to move from one voxel to another
		tDelta.v = 1 / fabs(d.v);
		printf("t_delta(%f, %f, %f)\n", tDelta.v.x, tDelta.v.y, tDelta.v.z);

		float3u tVec; // how much we need to travel in each axis to hit next voxel 
		tVec.v.x = fabsf(voxel.v.x - r.origin.x + ((d_sign.v.x + 1) >> 1))*tDelta.v.x;
		tVec.v.y = fabsf(voxel.v.y - r.origin.y + ((d_sign.v.y + 1) >> 1))*tDelta.v.y;
		tVec.v.z = fabsf(voxel.v.z - r.origin.z + ((d_sign.v.z + 1) >> 1))*tDelta.v.z;

		printf("t_vec(%f, %f, %f)\n", tVec.v.x, tVec.v.y, tVec.v.z);

		//TODO just check if voxel is inside model instead of checking agains hit_max
		while (isInside(voxel.v) && !isVoxelFull(voxel.v)) {
			// we need to move enough to hit another voxel
			face = min_id(tVec.v);
			voxel.a[face] += d_sign.a[face];
			tVec.a[face] += tDelta.a[face];

			printf("next face %d, voxel(%d, %d, %d), t_vec(%f, %f, %f)\n", face, voxel.v.x, voxel.v.y, voxel.v.z, tVec.v.x, tVec.v.y, tVec.v.z);
		}

		if (!isInside(voxel.v) || !isVoxelFull(voxel.v)) {
			printf("NO_HIT\n");
			return false;
		}

		//COMPUTE HIT RECORD
		hit.hit_t = tVec.a[face] - tDelta.a[face];
		hit.hit_face = face + 1;
		printf("HIT(%d, %f)\n", hit.hit_face, hit.hit_t);
		return true;
	}
#endif // DBG_TRACING

	__device__ bool isInside(const int3& voxel) const {
		if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0) return false;
		if (voxel.x > size.x || voxel.y > size.y || voxel.z > size.z) return false;
		return true;
	}
	__device__ bool isVoxelFull(const int3& voxel) const {
		const int elevation = heightmap[voxel.z*size.x + voxel.x] + 1; // raise the model by 1 to get a floor we can intersect with
		return voxel.y < elevation;
	}

	uint3 size; // bounding box
	const unsigned char *heightmap;
};

#endif /* VOXEL_MODEL_H_ */
