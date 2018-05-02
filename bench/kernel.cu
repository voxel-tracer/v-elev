
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

typedef union f3u {
	float3 f;
	float a[3];

	__device__ f3u(float3 v) : f(v) {}
} f3u;

typedef struct paths {
	// ray.origin
	float *ox;
	float *oy;
	float *oz;
	// ray.direction
	float *dx;
	float *dy;
	float *dz;
	// hit_face + 2*signbit(direction[hit_face])
	char *hit_case;
	// hit_p
	float *px;
	float *py;
	float *pz;
	// scattered.direction
	float *sx;
	float *sy;
	float *sz;
	// sun pdf
	float *sun_pdf;
	// scattered pdf
	float *scattered_pdf;
} paths;

__global__ void hit_scene(const paths p, const uint num_rays, const unsigned char* heightmap, const uint3 model_size, float t_min, float t_max)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

	const ray r(
		make_float3(p.ox[i], p.oy[i], p.oz[i]),
		make_float3(p.dx[i], p.dy[i], p.dz[i])
	);
	const voxelModel model(heightmap, model_size);
	cu_hit hit;
	if (!model.hit(r, t_min, t_max, hit)) {
		p.hit_case[i] = NO_HIT;
		return;
	}

	const f3u dir(r.direction);
	p.hit_case[i] = 1 + hit.hit_face * 2 * signbit(dir.a[hit.hit_face]);

	const float3 hit_p = r.point_at_parameter(hit.hit_t);
	p.px[i] = hit_p.x;
	p.py[i] = hit_p.y;
	p.pz[i] = hit_p.z;
}

__global__ void simple_color(const paths p, const uint num_rays, const uint seed, const float3 albedo, const sun s) {

	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

	const char hit_case = p.hit_case[i];
	if (hit_case == NO_HIT)
		return;

	const char hit_face = hit_face >> 2;
	const char hit_sign = ((hit_face & 1) << 1) - 1;
	const float3 hit_n = make_float3(
		(hit_face == 0)*hit_sign,
		(hit_face == 1)*hit_sign,
		(hit_face == 2)*hit_sign);

	cosine_pdf scatter_pdf(hit_n);
	//cosine_x scatter_pdf;
	const float3 hit_p = make_float3(p.px[i], p.py[i], p.pz[i]);
	sun_pdf plight(&s, hit_p);
	mixture_pdf mix(&plight, &scatter_pdf);

	curandStatePhilox4_32_10_t lseed;
	curand_init(0, seed*blockDim.x + threadIdx.x, 0, &lseed);
	const float3 scattered(mix.generate(&lseed));

	const float sun_pdf = plight.value(scattered);
	const float scattering_pdf = scatter_pdf.value(scattered);
	p.sun_pdf[i] = sun_pdf;

	if (scattering_pdf > 0) {
		p.sx[i] = scattered.x;
		p.sy[i] = scattered.y;
		p.sz[i] = scattered.z;
	}
	else {
		p.scattered_pdf[i] = 0;
	}
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

	// prepare paths
	paths p;
	cudaMalloc((void**)&p.ox, num_rays * sizeof(float));
	cudaMalloc((void**)&p.oy, num_rays * sizeof(float));
	cudaMalloc((void**)&p.oz, num_rays * sizeof(float));
	cudaMalloc((void**)&p.dx, num_rays * sizeof(float));
	cudaMalloc((void**)&p.dy, num_rays * sizeof(float));
	cudaMalloc((void**)&p.dz, num_rays * sizeof(float));
	cudaMalloc((void**)&p.hit_case, num_rays * sizeof(char));
	cudaMalloc((void**)&p.px, num_rays * sizeof(float));
	cudaMalloc((void**)&p.py, num_rays * sizeof(float));
	cudaMalloc((void**)&p.pz, num_rays * sizeof(float));
	cudaMalloc((void**)&p.sx, num_rays * sizeof(float));
	cudaMalloc((void**)&p.sy, num_rays * sizeof(float));
	cudaMalloc((void**)&p.sz, num_rays * sizeof(float));
	cudaMalloc((void**)&p.sun_pdf, num_rays * sizeof(float));
	cudaMalloc((void**)&p.scattered_pdf, num_rays * sizeof(float));

	const int threadsPerBlock = 256;
	const int blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;

	clock_t hit_duration = 0;
	clock_t color_duration = 0;

	cudaEvent_t startEvent, stopEvent;
	err(cudaEventCreate(&startEvent), "create startEvent");
	err(cudaEventCreate(&stopEvent), "create endEvent");

	float ms;
	clock_t start = clock();
	while (!input_file.eof()) {
		input_file.read((char*)rays, num_rays * sizeof(ray));
		input_file.ignore(num_rays * sizeof(cu_hit));

		// copy rays to ray_soas
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].origin.x;
		cudaMemcpy(p.ox, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].origin.y;
		cudaMemcpy(p.oy, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].origin.z;
		cudaMemcpy(p.oz, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice);

		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].direction.x;
		cudaMemcpy(p.dx, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].direction.y;
		cudaMemcpy(p.dy, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice);
		for (uint i = 0; i < num_rays; i++)
			temp_floats[i] = rays[i].direction.z;
		cudaMemcpy(p.dz, temp_floats, num_rays * sizeof(float), cudaMemcpyHostToDevice);

		err(cudaEventRecord(startEvent, 0), "record startEvent");
		hit_scene <<<blocksPerGrid, threadsPerBlock, 0 >>>(p, num_rays, d_heightmap, model->size, 0.1f, FLT_MAX);
		err(cudaEventRecord(stopEvent, 0), "record endEvent");
		cudaDeviceSynchronize();
		err(cudaEventElapsedTime(&ms, startEvent, stopEvent), "compute elapsed time");
		hit_duration += ms;

		err(cudaEventRecord(startEvent, 0), "record startEvent");
		simple_color <<<blocksPerGrid, threadsPerBlock, 0 >>>(p, num_rays, num_iter, albedo, *s);
		err(cudaEventRecord(stopEvent, 0), "record endEvent");
		cudaDeviceSynchronize();
		err(cudaEventElapsedTime(&ms, startEvent, stopEvent), "compute elapsed time");
		color_duration += ms;

		// copy hits to gpu and run kernel
		num_iter++;
	}
	const uint total_time = (clock() - start) / CLOCKS_PER_SEC;
	
	std::cout << "num iterations " << num_iter << " in " << total_time << " seconds" << std::endl;
	{
		const uint total_exec_time = hit_duration / 1000;
		std::cout << "hit_scene took " << total_exec_time << " seconds" << std::endl;
		std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
	}
	{
		const uint total_exec_time = color_duration / CLOCKS_PER_SEC;
		std::cout << "simple_color took " << total_exec_time << " seconds" << std::endl;
		std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
	}
	
	input_file.close();

	delete[] rays;
	delete[] temp_floats;
	cudaFree(p.ox);
	cudaFree(p.oy);
	cudaFree(p.oz);
	cudaFree(p.dx);
	cudaFree(p.dy);
	cudaFree(p.dz);
	cudaFree(p.hit_case);
	cudaFree(p.px);
	cudaFree(p.py);
	cudaFree(p.pz);
	cudaFree(p.sx);
	cudaFree(p.sy);
	cudaFree(p.sz);
	cudaFree(p.sun_pdf);
	cudaFree(p.scattered_pdf);
	err(cudaEventDestroy(startEvent), "free startEvent");
	err(cudaEventDestroy(stopEvent), "free endEvent");

    return 0;
}