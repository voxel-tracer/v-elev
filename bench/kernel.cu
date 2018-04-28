
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

typedef struct ray_soa {
	float *ox;
	float *oy;
	float *oz;
	float *dx;
	float *dy;
	float *dz;
} ray_soa;

typedef struct hit_soa {
	float *hit_t;
	uint *hit_face;
} hit_soa;

__global__ void hit_scene(const ray_soa rays, const hit_soa hits, const uint num_rays, const unsigned char* heightmap, const uint3 model_size, float t_min, float t_max)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

	const ray r(
		make_float3(rays.ox[i], rays.oy[i], rays.oz[i]),
		make_float3(rays.dx[i], rays.dy[i], rays.dz[i])
	);
	const voxelModel model(heightmap, model_size);
	cu_hit hit;
	if (!model.hit(r, t_min, t_max, hit)) {
		hits.hit_face[i] = NO_HIT;
		return;
	}

	hits.hit_face[i] = hit.hit_face;
	hits.hit_t[i] = hit.hit_t;
}

__global__ void simple_color(const ray_soa rays, const hit_soa hits, const uint num_rays, clr_rec* clrs, const uint seed, const float3 albedo, const sun s) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

	//clr_rec& crec = clrs[i];

	const uint hit_face = hits.hit_face[i];
	if (hit_face == NO_HIT) {
		// no intersection with spheres, return sky color
		//if (s.pdf_value(r.origin, r.direction) > 0) {
		//	//crec.color = s.clr;
		//	//crec.done = true;
		//}
		//else {
		//	//crec.color = make_float3(0);
		//	//crec.done = true;
		//}
		return;
	}

	const float3 direction = make_float3(rays.dx[i], rays.dy[i], rays.dz[i]);
	const float3 hit_n = make_float3(
		-1 * (hit_face == X)*signum(direction.x),
		-1 * (hit_face == Y)*signum(direction.y),
		-1 * (hit_face == Z)*signum(direction.z)
	);

	pdf* scatter_pdf = new cosine_pdf(hit_n);

	const float3 origin = make_float3(rays.ox[i], rays.oy[i], rays.oz[i]);
	const float3 hit_p(origin + hits.hit_t[i] * direction);
	sun_pdf plight(&s, hit_p);
	mixture_pdf p(&plight, scatter_pdf);

	curandStatePhilox4_32_10_t lseed;
	curand_init(0, seed*blockDim.x + threadIdx.x, 0, &lseed);
	const float3 scattered(p.generate(&lseed));
	const float pdf_val = p.value(scattered);
	if (pdf_val > 0) {
		const float scattering_pdf = fmaxf(0, dot(hit_n, scattered) / M_PI);

		//crec.origin = hit_p;
		//crec.direction = scattered;
		//crec.color = albedo*scattering_pdf / pdf_val;
		//crec.done = false;
	}
	else {
		//crec.color = make_float3(0, 0, 0);
		//crec.done = true;
	}
	delete scatter_pdf;
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

void print_stats(cu_hit *hits, const uint num_hits) 
{
	uint num_no_hit = 0;

	for (uint i = 0; i < num_hits; i++)
		if (hits[i].hit_face == NO_HIT) num_no_hit++;

	printf("%d/%d (%d%%) are NO_HIT\n", num_no_hit, num_hits, 100*num_no_hit / num_hits);
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
	float* temp_floats = new float[num_rays];
	uint* temp_uints = new uint[num_rays];
	cu_hit* hits = new cu_hit[num_rays];
	clr_rec* clrs = new clr_rec[num_rays];

	// prepare ray_soa
	ray_soa ray_soas;
	err(cudaMalloc((void**)&ray_soas.ox, num_rays * sizeof(float)), "allocate ray_soa.ox");
	err(cudaMalloc((void**)&ray_soas.oy, num_rays * sizeof(float)), "allocate ray_soa.oy");
	err(cudaMalloc((void**)&ray_soas.oz, num_rays * sizeof(float)), "allocate ray_soa.oz");
	err(cudaMalloc((void**)&ray_soas.dx, num_rays * sizeof(float)), "allocate ray_soa.dx");
	err(cudaMalloc((void**)&ray_soas.dy, num_rays * sizeof(float)), "allocate ray_soa.dy");
	err(cudaMalloc((void**)&ray_soas.dz, num_rays * sizeof(float)), "allocate ray_soa.dz");

	hit_soa hit_soas;
	err(cudaMalloc((void**)&hit_soas.hit_t, num_rays * sizeof(float)), "allocate hit_soa.hit_t");
	err(cudaMalloc((void**)&hit_soas.hit_face, num_rays * sizeof(uint)), "allocate hit_soa.hit_face");

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
		//print_stats(hits, num_rays);

		// copy rays to ray_soas
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].origin.x;
		err(cudaMemcpy(ray_soas.ox, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy ray.ox from host to device");
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].origin.y;
		err(cudaMemcpy(ray_soas.oy, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy ray.oy from host to device");
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].origin.z;
		err(cudaMemcpy(ray_soas.oz, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy ray.oz from host to device");
		
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].direction.x;
		err(cudaMemcpy(ray_soas.dx, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy ray.dx from host to device");
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].direction.y;
		err(cudaMemcpy(ray_soas.dy, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy ray.dy from host to device");
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].direction.z;
		err(cudaMemcpy(ray_soas.dz, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy ray.dz from host to device");

		// copy rays to gpu and run kernel
		clock_t begin = clock();
		hit_scene <<<blocksPerGrid, threadsPerBlock, 0 >>>(ray_soas, hit_soas, num_rays, d_heightmap, model->size, 0.1f, FLT_MAX);
		cudaDeviceSynchronize();
		hit_duration += clock() - begin;

		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = hits[i].hit_t;
		err(cudaMemcpy(hit_soas.hit_t, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice), "copy hit_soa.hit_t from host to device");
		for (uint i = 0; i < num_rays; i++)
			temp_uints[i] = hits[i].hit_face;
		err(cudaMemcpy(hit_soas.hit_face, temp_uints, num_rays * sizeof(uint), cudaMemcpyHostToDevice), "copy hit_soa.hit_face from host to device");

		begin = clock();
		simple_color <<<blocksPerGrid, threadsPerBlock, 0 >>>(ray_soas, hit_soas, num_rays, d_clrs, num_iter, albedo, *s);
		err(cudaMemcpy(clrs, d_clrs, num_rays * sizeof(clr_rec), cudaMemcpyDeviceToHost), "copy results from device to host");
		color_duration += clock() - begin;

		// copy hits to gpu and run kernel
		num_iter++;
	}
	const uint total_time = (clock() - start) / CLOCKS_PER_SEC;
	
	std::cout << "num iterations " << num_iter << " in " << total_time << " seconds" << std::endl;
	{
		const uint total_exec_time = hit_duration / CLOCKS_PER_SEC;
		std::cout << "hit_scene took " << total_exec_time << " seconds" << std::endl;
		std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
	}
	{
		const uint total_exec_time = color_duration / CLOCKS_PER_SEC;
		std::cout << "simple_color took " << total_exec_time << " seconds" << std::endl;
		std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
	}

	delete[] rays;
	delete[] temp_floats;
	delete[] hits;
	delete[] clrs;
	err(cudaFree(ray_soas.ox), "free device ray_soa.ox");
	err(cudaFree(ray_soas.oy), "free device ray_soa.oy");
	err(cudaFree(ray_soas.oz), "free device ray_soa.oz");
	err(cudaFree(ray_soas.dx), "free device ray_soa.dx");
	err(cudaFree(ray_soas.dy), "free device ray_soa.dy");
	err(cudaFree(ray_soas.dz), "free device ray_soa.dz");
	err(cudaFree(hit_soas.hit_face), "free device hit_soa.hit_face");
	err(cudaFree(hit_soas.hit_t), "free device hit_soa.hit_t");
	err(cudaFree(d_clrs), "free device d_clrs");
	input_file.close();

    return 0;
}