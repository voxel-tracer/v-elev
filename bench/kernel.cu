
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <ctime>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include "sdl.h"
#undef main

#include "utils.h"
#include "ray.h"
#include "camera.h"
#include "voxel_model.h"
#include "material.h"
#include "human-readable.h"


__global__ void hit_scene(const ray* rays, const uint num_rays, const unsigned char* heightmap, const uint3 model_size, cu_hit* hits)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num_rays) return;

	const ray *r = &(rays[i]);
	const voxelModel model(heightmap, model_size);
	cu_hit hit;
	if (!model.hit(*r, hit)) {
		hits[i].hit_face = NO_HIT;
		return;
	}

	hits[i].hit_face = hit.hit_face;
	hits[i].hit_t = hit.hit_t;
}

__global__ void simple_color(ray* rays, const uint num_rays, const cu_hit* hits, clr_rec* clrs, const uint seed, const float3 albedo, const sun s) {

	const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (ray_idx >= num_rays) return;

	clr_rec& crec = clrs[ray_idx];
	if (crec.done) return; // nothing more to do

	ray& r = rays[ray_idx];
	const cu_hit hit(hits[ray_idx]);

	if (hit.hit_face == NO_HIT) {
		// no intersection with spheres, return sky color
		crec.done = true;
		crec.color *= (s.pdf_value(r.origin, r.direction) > 0) ? s.clr : make_float3(0);
		return;
	}

	const float3 hit_n = make_float3(
		-1 * (hit.hit_face == X)*signum(r.direction.x),
		-1 * (hit.hit_face == Y)*signum(r.direction.y),
		-1 * (hit.hit_face == Z)*signum(r.direction.z)
	);

	cosine_pdf scatter_pdf(hit_n);

	const float3 hit_p(r.point_at_parameter(hit.hit_t));
	sun_pdf plight(&s, hit_p);
	mixture_pdf p(&plight, &scatter_pdf);

	curandStatePhilox4_32_10_t lseed;
	curand_init(0, seed*blockDim.x + threadIdx.x, 0, &lseed);
	const float3 scattered(p.generate(&lseed));
	const float pdf_val = p.value(scattered);
	if (pdf_val > 0) {
		const float scattering_pdf = fmaxf(0, dot(hit_n, scattered) / M_PI);

		r.origin = hit_p;
		r.direction = scattered;
		crec.color *= albedo*scattering_pdf / pdf_val;
		crec.done = false;
	}
	else {
		crec.color = make_float3(0);
		crec.done = true;
	}
}

__global__ void debug_color(ray* rays, const uint num_rays, const cu_hit* hits, clr_rec* clrs, const uint seed, const float3 albedo, const sun s) {

	const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (ray_idx >= num_rays) return;

	clr_rec& crec = clrs[ray_idx];
	if (crec.done) return; // nothing more to do

	ray& r = rays[ray_idx];
	const cu_hit hit(hits[ray_idx]);

	if (hit.hit_face == NO_HIT) {
		// no intersection with spheres, return sky color
		crec.done = true;
		crec.color *= (s.pdf_value(r.origin, r.direction) > 0) ? s.clr : make_float3(0);
		return;
	}

	const float3 hit_n = make_float3(
		-1 * (hit.hit_face == X)*signum(r.direction.x),
		-1 * (hit.hit_face == Y)*signum(r.direction.y),
		-1 * (hit.hit_face == Z)*signum(r.direction.z)
	);

	cosine_pdf scatter_pdf(hit_n);

	const float3 hit_p(r.point_at_parameter(hit.hit_t));
	sun_pdf plight(&s, hit_p);
	mixture_pdf p(&plight, &scatter_pdf);

	curandStatePhilox4_32_10_t lseed;
	curand_init(0, seed*blockDim.x + threadIdx.x, 0, &lseed);
	const float3 scattered(p.generate(&lseed));
	const float pdf_val = p.value(scattered);
	if (pdf_val > 0) {
		const float scattering_pdf = fmaxf(0, dot(hit_n, scattered) / M_PI);

		const uint max_dir = max_id(scattered);
		crec.color = (make_float3(
			(max_dir == 0)*signum(scattered.x),
			(max_dir == 1)*signum(scattered.y),
			(max_dir == 2)*signum(scattered.z)
		) + 1) / 2;
		crec.color = (normalize(hit_n) + 1) / 2;
		crec.done = true;
	}
	else {
		crec.color = make_float3(0);
		crec.done = true;
	}
}

void err(cudaError_t err, char *msg) {
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to %s (error code %s)!\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void city_scene(camera **cam, sun **s, voxelModel** model, float aspect) {
	// load heightmap image
	int image_x, image_y, image_n;
	int image_desired_channels = 1; // grayscale
	unsigned char *data = stbi_load("city512.png", &image_x, &image_y, &image_n, image_desired_channels);
	if (data == NULL) {
		*model = NULL;
		return;
	}

	*cam = new camera(make_float3(700), make_float3(image_x / 2, 0, image_y / 2), make_float3(0, 1, 0), 20, aspect, 0, 1.0);
	*s = new sun(make_float3(700, 1400, 1400), 200, make_float3(50));
	*model = new voxelModel(data, image_x, image_y);
}

void print_stats(cu_hit *hits, const uint num_hits) {
	uint num_no_hit = 0;

	for (uint i = 0; i < num_hits; i++)
		if (hits[i].hit_face == NO_HIT) num_no_hit++;

	printf("%d/%d (%d%%) are NO_HIT\n", num_no_hit, num_hits, 100*num_no_hit / num_hits);
}

void generate_ray(ray& ray, const camera *cam, int x, int y, const uint nx, const uint ny) {
	// even though we can compute pixelId from (x,y), we still need the sampleId as its not necessarely the same (as more than a single sample point to the same pixel)
	const float u = float(x + drand48()) / float(nx);
	const float v = float(y + drand48()) / float(ny);
	cam->get_ray(u, v, ray);
}

void generate_rays(ray* rays, const camera *cam, const uint nx, const uint ny, const uint spp) {
	uint ray_idx = 0;
	for (int j = ny - 1; j >= 0; j--)
		for (int i = 0; i < nx; ++i)
			for (uint s = 0; s < spp; s++, ++ray_idx)
				generate_ray(rays[ray_idx], cam, i, j, nx, ny);
}

void display_image(clr_rec *clrs, const uint nx, const uint ny, const uint spp) {
	uint *pixels = new uint[nx*ny];

	uint ray_idx = 0;
	uint pixel_idx = 0;
	for (int y = ny-1; y >= 0; y--) {
		for (int x = 0; x < nx; x++, ++pixel_idx) {
			// compute average of all samples
			float3 color;
			for (uint s = 0; s < spp; s++, ++ray_idx)
				color += clrs[ray_idx].color;
			color = color / spp;
			unsigned char red = min(255, int(255.99*color.x));
			unsigned char green = min(255, int(255.99*color.y));
			unsigned char blue = min(255, int(255.99*color.z));
			pixels[pixel_idx] = (red << 16) + (green << 8) + blue;
		}
	}

	// display the pixels in a window
	SDL_Window *win = NULL;
	SDL_Renderer *renderer = NULL;
	SDL_Texture *img = NULL;

	// Initialize SDL.
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		delete[] pixels;
		return;
	}

	win = SDL_CreateWindow("voxel elevation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, nx, ny, 0);
	renderer = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
	img = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, nx, ny);
	SDL_UpdateTexture(img, NULL, pixels, nx * sizeof(uint));
	SDL_RenderCopy(renderer, img, NULL, NULL);
	SDL_RenderPresent(renderer);

	// wait until the window is closed
	SDL_Event event;
	bool quit = false;
	while (!quit && SDL_WaitEvent(&event)) {
		switch (event.type) {
		case SDL_QUIT:
			quit = true;
			break;
		}
	}

	// free all resources
	SDL_DestroyTexture(img);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(win);
	SDL_Quit();

	delete[] pixels;
}

int main(int argc, char** argv) {
	const uint nx = 500, ny = 500, spp = 32, max_depth = 50;
	const uint num_rays = nx*ny*spp;
	const uint threadsPerBlock = 128;
	const uint blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;

	// load voxel model
	voxelModel *model;
	sun *s;
	camera *c;
	float3 albedo = make_float3(.45f);
	city_scene(&c, &s, &model, float(nx) / float(ny));
	if (model == NULL) {
		std::cerr << "couldn't load image" << std::endl;
		return 1;
	}
	printf("voxel size (%d, %d, %d)\n", model->size.x, model->size.y, model->size.z);

	// copy model to gpu
	unsigned char* d_heightmap = NULL;
	err(cudaMalloc((void **)&d_heightmap, model->size.x*model->size.z * sizeof(unsigned char)), "allocate device d_heightmap");
	err(cudaMemcpy(d_heightmap, model->heightmap, model->size.x*model->size.z * sizeof(unsigned char), cudaMemcpyHostToDevice), "copy heightmap from host to device");

	// allocate all samples
	ray* rays = new ray[num_rays];
	clr_rec* clrs = new clr_rec[num_rays];

	// allocate buffers on device
	ray* d_rays = NULL;
	err(cudaMalloc((void **)&d_rays, num_rays * sizeof(ray)), "allocate device d_rays");
	cu_hit* d_hits = NULL;
	err(cudaMalloc((void **)&d_hits, num_rays * sizeof(cu_hit)), "allocate device d_hits");
	clr_rec* d_clrs = NULL;
	err(cudaMalloc((void **)&d_clrs, num_rays * sizeof(clr_rec)), "allocate device d_clrs");

	// start by generating all primary rays
	generate_rays(rays, c, nx, ny, spp);
	// copy rays to gpu and run kernel
	err(cudaMemcpyAsync(d_rays, rays, num_rays * sizeof(ray), cudaMemcpyHostToDevice), "copy rays to device");
	delete[] rays;

	cudaEvent_t start, hit_done, color_done;
	cudaEventCreate(&start);
	cudaEventCreate(&hit_done);
	cudaEventCreate(&color_done);

	// intersect all primary rays with the scene
	uint num_iter = 1;
	cudaEventRecord(start);
	hit_scene <<<blocksPerGrid, threadsPerBlock, 0 >>>(d_rays, num_rays, d_heightmap, model->size, d_hits);
	cudaEventRecord(hit_done);
	simple_color <<<blocksPerGrid, threadsPerBlock, 0 >>>(d_rays, num_rays, d_hits, d_clrs, num_iter, albedo, *s);
	cudaEventRecord(color_done);
	err(cudaMemcpy(clrs, d_clrs, num_rays * sizeof(clr_rec), cudaMemcpyDeviceToHost), "copy results from device to host");

	float hit_duration_ms = 0;
	float color_duration_ms = 0;
	cudaEventElapsedTime(&hit_duration_ms, start, hit_done);
	cudaEventElapsedTime(&color_duration_ms, hit_done, color_done);

	{
		const float total_exec_time = hit_duration_ms / 1000;
		if (total_exec_time > 0) {
			std::cout << "hit_scene took " << total_exec_time << " seconds" << std::endl;
			std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
		}
	}
	{
		const float total_exec_time = color_duration_ms / 1000;
		if (total_exec_time > 0) {
			std::cout << "simple_color took " << total_exec_time << " seconds" << std::endl;
			std::cout << "  " << sscale(num_iter*num_rays / total_exec_time) << " rays/s" << std::endl;
		}
	}
	err(cudaFree(d_rays), "free device d_rays");
	err(cudaFree(d_hits), "free device d_hits");
	err(cudaFree(d_clrs), "free device d_clrs");

	display_image(clrs, nx, ny, spp);

	delete[] clrs;

    return 0;
}