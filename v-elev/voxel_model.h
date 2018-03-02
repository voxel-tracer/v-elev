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

	__device__ bool hitDist(const ray& r, float3& hit_min, float3& hit_max) const {
		// min = (0, 0, 0), max = size
		float tMin = -FLT_MAX;
		float tMax = FLT_MAX;
		float invD, t0, t1, tmp;
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
		hit_max.x = t1;
		if (tMax <= tMin) {
			return false;
		}
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
		hit_max.y = t1;
		if (tMax <= tMin) {
			return false;
		}
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
		hit_max.z = t1;
		if (tMax <= tMin) {
			return false;
		}
		return true;
	}

	__device__ bool hit(const ray& r, float t_min, float t_max, cu_hit& hit, const bool dbg) const {
		float3u hit_min, hit_max;
		hit_min.v = make_float3(t_min);
		hit_max.v = make_float3(t_max);
		if (!hitDist(r, hit_min.v, hit_max.v) || min(hit_max.v) < t_min) {
			if (dbg) printf("NO_HIT\n");
			return false;
		}

		float3u d; d.v = r.direction; //TODO make ray.direction float3u
		int3u d_sign;
		d_sign.v = make_int3(signum(d.v.x), signum(d.v.y), signum(d.v.z));
		if (dbg) printf("direction(%f, %f, %f), d(%f, %f, %f), d_sign(%d, %d, %d)\n", 
			r.direction.x, r.direction.y, r.direction.z, 
			d.v.x, d.v.y, d.v.z, 
			d_sign.v.x, d_sign.v.y, d_sign.v.z);

		// find which face was hit
		int face = max_id(hit_min.v);
		float minT = hit_min.a[face]; // t at intersection
		float3 h_vec = r.point_at_parameter(minT); // intersected point
		if (dbg) printf("minT %f, h_vec (%f, %f, %f)\n", minT, h_vec.x, h_vec.y, h_vec.z);
		int3u voxel;
		voxel.v = make_int3(
			h_vec.x + d_sign.v.x*EPS,
			h_vec.y + d_sign.v.y*EPS,
			h_vec.z + d_sign.v.z*EPS
		); // intersected voxel
		if (dbg) printf("start (%d, %d, %d)\n", voxel.v.x, voxel.v.y, voxel.v.z);

		// compute tVec: how much we need to travel in each axis to hit next voxel
		// next_voxel = voxel + step (+1 if step < 0)
		// this is equivalent to
		// next_voxel = voxel (+1 if step > 0)
		float3u tVec;
		//TODO if I just divide by 0 I will get +/- infinity, won't that be enough ?
		tVec.v = make_float3(
			(d_sign.v.x == 0 ? FLT_MAX : (voxel.v.x + (d_sign.v.x > 0 ? 1:0) - r.origin.x) / d.v.x),
			(d_sign.v.y == 0 ? FLT_MAX : (voxel.v.y + (d_sign.v.y > 0 ? 1:0) - r.origin.y) / d.v.y),
			(d_sign.v.z == 0 ? FLT_MAX : (voxel.v.z + (d_sign.v.z > 0 ? 1:0) - r.origin.z) / d.v.z)
		);
		if (dbg) printf("      t_vec(%f, %f, %f)\n", tVec.v.x, tVec.v.y, tVec.v.z);

		// compute tDelta: how much we need to travel in each axis to move from one voxel to another
		// final Vec3 tDelta = rD.abs().div(gridSize).inv();
		float3u tDelta;
		tDelta.v = 1 / fabs(d.v);
		if (dbg) printf("      t_delta(%f, %f, %f)\n", tDelta.v.x, tDelta.v.y, tDelta.v.z);

		// remember tVec.get(face) is the intersection with the next voxel
		// substract tDelta[face] to get the intersection with current voxel
		while ((tVec.a[face] - tDelta.a[face] + EPS) < t_min || (!isVoxelFull(voxel.v) && (tVec.a[face] - EPS) <= hit_max.a[face])) {
			// we need to move enough to hit another voxel
			face = min_id(tVec.v);
			voxel.a[face] += d_sign.a[face];
			tVec.a[face] += tDelta.a[face];
			if (dbg) printf("      (%d, %d, %d), (%f, %f, %f): %d\n", voxel.v.x, voxel.v.y, voxel.v.z, tVec.v.x, tVec.v.y, tVec.v.z, face);
		}

		if (!isVoxelFull(voxel.v)) {
			if (dbg) printf("NO_HIT\n");
			return false;
		}

		//COMPUTE HIT RECORD
		hit.hit_t = tVec.a[face] - tDelta.a[face];
		hit.hit_face = face + 1;
		if (dbg) printf("HIT\n");
		return true;
	}

	__device__ bool isVoxelFull(const int3& voxel) const {
		// if voxel outside bbox, its considered empty
		if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0) return false;
		if (voxel.x > size.x || voxel.y > size.y || voxel.z > size.z) return false;
		const int elevation = heightmap[voxel.z*size.x + voxel.x] + 1; // raise the model by 1 to get a floor we can intersect with
		return voxel.y < elevation;
	}

	uint3 size; // bounding box
	const unsigned char *heightmap;
};

#endif /* VOXEL_MODEL_H_ */
