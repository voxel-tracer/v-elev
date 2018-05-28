#pragma once

#include <ctime>
#include <fstream>
#include <iostream>

#include "camera.h"
#include "voxel_model.h"
#include "sun.h"

//#define DBG_FILE
#define DBG_RENDER

struct pixel {
	uint samples = 0;
	uint done = 0; // needed to differentiate between done vs ongoing samples, when doing progressive rendering

	pixel() {}
	pixel(const pixel &p) : samples(p.samples), done(p.done) {}
};

struct work_unit {
	cudaStream_t stream;
	const uint start_idx;
	const uint end_idx;
	ray* h_rays;
	ray* d_rays;
#ifdef DBG_FILE
	cu_hit* h_hits;
#endif
	cu_hit* d_hits;
	clr_rec* h_clrs;
	clr_rec* d_clrs;
	int * pixel_idx;
	sample* samples;
	pixel* pixels;
	float3* h_colors;

	bool done = false;
	uint gpu_time = 0;
	uint cpu_time = 0;
	uint num_iter = 0;

	work_unit(uint start, uint end) :start_idx(start), end_idx(end) {}
	uint length() const { return end_idx - start_idx; }
	uint total_rays() const { return num_iter*length(); }
};

class renderer {
public:
	renderer(const camera* _cam, const voxelModel* vm, const float3& albedo, const sun& s, uint _nx, uint _ny, uint _ns, uint _max_depth, float _min_attenuation, uint nunits):
		cam(_cam), model(vm), model_albedo(albedo), scene_sun(s), nx(_nx), ny(_ny), ns(_ns), max_depth(_max_depth), min_attenuation(_min_attenuation), num_units(nunits) {
#ifdef DBG_FILE
		output_file = new std::ofstream("render.dat", std::ios::binary);
#endif // DBG_FILE
	}
	~renderer();

	uint numpixels() const { return nx*ny; }
	bool is_not_done() const { return !(wunits[0]->done && wunits[1]->done); }
	uint get_pixelId(int x, int y) const { return (ny - y - 1)*nx + x; }

	uint get_unitIdx(uint pixelId) const {
		const uint unit_numpixels = numpixels() / num_units;
		return pixelId / unit_numpixels; 
	}

	float3 get_pixel_color(int x, int y) const {
		const uint pixelId = get_pixelId(x, y);
		const uint unitIdx = get_unitIdx(pixelId);
		work_unit* wu = wunits[unitIdx];
		const uint local_idx = pixelId - wu->start_idx;
		const uint num_done = wu->pixels[local_idx].done;
		if (num_done == 0) return make_float3(0, 0, 0);
		return wu->h_colors[local_idx] / float(num_done);
	}

	uint totalrays() const { return total_rays; }
	const work_unit *get_wunit(uint idx) { return wunits[idx]; }

	void prepare_kernel();
	void update_camera();

	void generate_rays();
	
	void render_work_unit(uint unit_idx);
	const camera* const cam;
	const voxelModel * const model;
	const float3 model_albedo;
	const sun scene_sun;
	const uint nx;
	const uint ny;
	const uint ns;
	const uint max_depth;
	const float min_attenuation;

	unsigned char* d_heightmap;
	bool init_rnds = true;

	clock_t kernel = 0;
	clock_t generate = 0;
	clock_t compact = 0;

	uint ray_tracing_id = -1;

private:
	void copy_rays_to_gpu(const work_unit* wu);
	void start_kernel(const work_unit* wu);
	void copy_colors_from_gpu(const work_unit* wu);
	void compact_rays(work_unit* wu);
	inline void generate_ray(work_unit* wu, const uint ray_idx, int x, int y);

	uint total_rays = 0;
	work_unit **wunits;
	const uint num_units;
	uint next_pixel = 0;
	int remaining_pixels = 0;
	uint num_runs = 0;
#ifdef DBG_FILE
	std::ofstream *output_file;
#endif // DBG_FILE
};