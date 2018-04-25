
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include "utils.h"
#include "ray.h"
#include "voxel_model.h"
#include "material.h"
#include "human-readable.h"

__global__ void hit_scene(const ray* rays, const uint num_rays, const unsigned char* heightmap, const uint3 model_size, float t_min, float t_max, cu_hit* hits)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

	const ray *r = &(rays[i]);
	const voxelModel model(heightmap, model_size);
	cu_hit hit;
	if (!model.hit(*r, t_min, t_max, hit)) {
		hits[i].hit_face = NO_HIT;
		return;
	}

	hits[i].hit_face = hit.hit_face;
	hits[i].hit_t = hit.hit_t;
}


__global__ void simple_color(const ray* rays, const uint num_rays, const cu_hit* hits, clr_rec* clrs, const uint seed, const float3 albedo, const sun s) {

	const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (ray_idx >= num_rays) return;

	const ray& r = rays[ray_idx];
	const cu_hit hit(hits[ray_idx]);
	clr_rec& crec = clrs[ray_idx];

	if (hit.hit_face == NO_HIT) {
		// no intersection with spheres, return sky color
		if (s.pdf_value(r.origin, r.direction) > 0) {
			crec.color = s.clr;
			crec.done = true;
		}
		else {
			crec.color = make_float3(0);
			crec.done = true;
		}
		return;
	}

	const float3 hit_n = make_float3(
		-1 * (hit.hit_face == X)*signum(r.direction.x),
		-1 * (hit.hit_face == Y)*signum(r.direction.y),
		-1 * (hit.hit_face == Z)*signum(r.direction.z)
	);

	hit_record rec(r.point_at_parameter(hit.hit_t), hit_n);
	curandStatePhilox4_32_10_t localState;
	curand_init(0, seed*blockDim.x + threadIdx.x, 0, &localState);
	const lambertian mat(albedo);

	scatter_record srec;
	mat.scatter(rec, srec);

	sun_pdf plight(&s, rec.hit_p);
	mixture_pdf p(&plight, srec.pdf_ptr);

	srec.scattered = ray(rec.hit_p, p.generate(&localState));
	const float pdf_val = p.value(srec.scattered.direction);
	if (pdf_val > 0) {
		const float scattering_pdf = mat.scattering_pdf(rec, srec.scattered);
		srec.attenuation *= scattering_pdf / pdf_val;

		crec.origin = srec.scattered.origin;
		crec.direction = srec.scattered.direction;
		crec.color = srec.attenuation;
		crec.done = false;
	}
	else {
		crec.color = make_float3(0, 0, 0);
		crec.done = true;
	}
	delete srec.pdf_ptr;
}

void err(cudaError_t err, char *msg)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to %s (error code %s)!\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void city_scene(sun **s, voxelModel** model, float aspect) {
	// load heightmap image
	int image_x, image_y, image_n;
	int image_desired_channels = 1; // grayscale
	unsigned char *data = stbi_load("city512.png", &image_x, &image_y, &image_n, image_desired_channels);
	if (data == NULL) {
		*model = NULL;
		return;
	}

	*s = new sun(make_float3(700, 1400, 1400), 200, make_float3(50));
	*model = new voxelModel(data, image_x, image_y);
}

int main()
{
	// load voxel model
	const uint nx = 500, ny = 500;
	voxelModel *model;
	sun *s;
	float3 albedo = make_float3(.45f);
	city_scene(&s, &model, float(nx) / float(ny));
	if (model == NULL) {
		std::cerr << "couldn't load image" << std::endl;
		return 1;
	}
	printf("voxel size (%d, %d, %d)\n", model->size.x, model->size.y, model->size.z);

	// copy model to gpu
	unsigned char* d_heightmap = NULL;
	err(cudaMalloc((void **)&d_heightmap, model->size.x*model->size.z * sizeof(unsigned char)), "allocate device d_heightmap");
	err(cudaMemcpy(d_heightmap, model->heightmap, model->size.x*model->size.z * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy heightmap from host to device");

	// load rays/hits
	std::ifstream input_file("render.dat", std::ios::binary);
	uint num_iter = 0;
	uint num_rays;
	input_file.read((char*)&num_rays, sizeof(uint));
	std::cout << "num_rays per iteration " << sscale(num_rays) << std::endl;

	ray* rays = new ray[num_rays];
	cu_hit* hits = new cu_hit[num_rays];
	clr_rec* clrs = new clr_rec[num_rays];

	ray* d_rays = NULL;
	err(cudaMalloc((void **)&d_rays, num_rays * sizeof(ray)), "allocate device d_rays");
	cu_hit* d_hits = NULL;
	err(cudaMalloc((void **)&d_hits, num_rays * sizeof(cu_hit)), "allocate device d_hits");
	clr_rec* d_clrs = NULL;
	err(cudaMalloc((void **)&d_clrs, num_rays * sizeof(clr_rec)), "allocate device d_clrs");

	const int threadsPerBlock = 128;
	const int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;

	clock_t hit_duration = 0;
	clock_t color_duration = 0;

	clock_t start = clock();
	while (!input_file.eof()) {
		//std::cout << "reading iteration " << num_iter << std::endl;

		input_file.read((char*)rays, num_rays * sizeof(ray));
		input_file.read((char*)hits, num_rays * sizeof(cu_hit));

		// copy rays to gpu and run kernel
		clock_t begin = clock();
		//err(cudaMemcpyAsync(d_rays, rays, num_rays * sizeof(ray), cudaMemcpyHostToDevice), "copy rays from host to device");
		//hit_scene <<<blocksPerGrid, threadsPerBlock, 0 >>>(d_rays, num_rays, d_heightmap, model->size, 0.1f, FLT_MAX, d_hits);
		//cudaDeviceSynchronize();
		//hit_duration += clock() - begin;
								 
		err(cudaMemcpy(d_hits, hits, num_rays * sizeof(cu_hit), cudaMemcpyHostToDevice), "copy hits from host to device");
		begin = clock();
		simple_color <<<blocksPerGrid, threadsPerBlock, 0 >>>(d_rays, num_rays, d_hits, d_clrs, num_iter, albedo, *s);
		err(cudaMemcpy(clrs, d_clrs, num_rays * sizeof(clr_rec), cudaMemcpyDeviceToHost), "copy results from device to host");
		color_duration += clock() - begin;

		// copy hits to gpu and run kernel
		num_iter++;
	}
	const uint total_time = (clock() - start) / CLOCKS_PER_SEC;
	
	std::cout << "num iterations " << num_iter << " in " << total_time << " seconds" << std::endl;
	{
		//const uint total_exec_time = hit_duration / CLOCKS_PER_SEC;
		//std::cout << "hit_scene took " << total_exec_time << " seconds" << std::endl;
		//std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
	}
	{
		const uint total_exec_time = color_duration / CLOCKS_PER_SEC;
		std::cout << "simple_color took " << total_exec_time << " seconds" << std::endl;
		std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
	}

	delete[] rays;
	delete[] hits;
	delete[] clrs;
	err(cudaFree(d_rays), "free device d_rays");
	err(cudaFree(d_hits), "free device d_hits");
	err(cudaFree(d_clrs), "free device d_clrs");
	input_file.close();

    return 0;
}