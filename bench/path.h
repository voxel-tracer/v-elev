#pragma once

struct paths {
	// ray.origin
	float *oxs;
	float *oys;
	float *ozs;
	// ray.direction
	float *dxs;
	float *dys;
	float *dzs;
	// color
	float *cxs;
	float *cys;
	float *czs;
	//hit
	unsigned char *hit_faces;
	float *hit_ts;
	// done
	bool *done;
};