
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
#include "path.h"
#include "ray.h"
#include "camera.h"
#include "voxel_model.h"
#include "material.h"
#include "human-readable.h"
#include "options.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void hit_scene(paths paths, const uint path_offset, const uint num_paths, const unsigned char* heightmap, const uint3 model_size)
{
	uint path_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (path_idx >= num_paths) return;
	path_idx += path_offset;

	if (paths.done[path_idx]) return;

	const ray r(
		make_float3(paths.oxs[path_idx], paths.oys[path_idx], paths.ozs[path_idx]),
		make_float3(paths.dxs[path_idx], paths.dys[path_idx], paths.dzs[path_idx]));
	const voxelModel model(heightmap, model_size);
	cu_hit hit;
	if (!model.hit(r, hit)) {
		paths.hit_faces[path_idx] = NO_HIT;
		return;
	}

	paths.hit_faces[path_idx] = hit.hit_face;
	paths.hit_ts[path_idx] = hit.hit_t;
}

__global__ void simple_color(paths paths, const uint num_paths, const uint seed, const uint iteration, const float3 albedo, const sun s, const uint spp) {

	const int path_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (path_idx >= num_paths) return;

	if (paths.done[path_idx]) return;

	const ray r(
		make_float3(paths.oxs[path_idx], paths.oys[path_idx], paths.ozs[path_idx]),
		make_float3(paths.dxs[path_idx], paths.dys[path_idx], paths.dzs[path_idx]));
	const cu_hit hit(
		paths.hit_ts[path_idx],
		paths.hit_faces[path_idx]
	);

	if (hit.hit_face == NO_HIT) {
		// no intersection with spheres, return sky color
		paths.done[path_idx] = true;
		const bool is_sun_hit = s.pdf_value(r.origin, r.direction) > 0;
		paths.cxs[path_idx] *= is_sun_hit ? s.clr.x : 0;
		paths.cys[path_idx] *= is_sun_hit ? s.clr.y : 0;
		paths.czs[path_idx] *= is_sun_hit ? s.clr.z : 0;
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
	const uint pixel_idx = path_idx / spp;
	const uint sample_idx = path_idx % spp;
	curand_init(seed,
		pixel_idx, // different subsequence per pixel
		iteration*spp * 5 + sample_idx * 5, // we assume each sample generates 5 random numbers per iteration
		&lseed);
	const float3 scattered(p.generate(&lseed));
	const float pdf_val = p.value(scattered);
	if (pdf_val > 0) {
		const float scattering_pdf = fmaxf(0, dot(hit_n, scattered) / M_PI);

		paths.oxs[path_idx] = hit_p.x;
		paths.oys[path_idx] = hit_p.y;
		paths.ozs[path_idx] = hit_p.z;
		paths.dxs[path_idx] = scattered.x;
		paths.dys[path_idx] = scattered.y;
		paths.dzs[path_idx] = scattered.z;
		paths.cxs[path_idx] *= albedo.x*scattering_pdf / pdf_val;
		paths.cys[path_idx] *= albedo.y*scattering_pdf / pdf_val;
		paths.czs[path_idx] *= albedo.z*scattering_pdf / pdf_val;
		paths.done[path_idx] = false; // is it really needed ?
	}
	else {
		paths.cxs[path_idx] = 0;
		paths.cys[path_idx] = 0;
		paths.czs[path_idx] = 0;
		paths.done[path_idx] = true;
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

	*cam = new camera(make_float3(500), make_float3(image_x / 2, 0, image_y / 2), make_float3(0, 1, 0), 20, aspect, 0, 1.0);
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
				if (clrs[ray_idx].done) color += clrs[ray_idx].color;
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

void init_paths(paths &p, const ray *rays, const uint num_rays) {
	float *temp = new float[num_rays];

	// ray.origin
	gpuErrchk(cudaMalloc((void**)&p.oxs, num_rays * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p.oys, num_rays * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p.ozs, num_rays * sizeof(float)));
	// ray.direction
	gpuErrchk(cudaMalloc((void**)&p.dxs, num_rays * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p.dys, num_rays * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p.dzs, num_rays * sizeof(float)));
	// color
	gpuErrchk(cudaMalloc((void**)&p.cxs, num_rays * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p.cys, num_rays * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&p.czs, num_rays * sizeof(float)));
	// hit.face
	gpuErrchk(cudaMalloc((void**)&p.hit_faces, num_rays * sizeof(unsigned char)));
	// hit.t
	gpuErrchk(cudaMalloc((void**)&p.hit_ts, num_rays * sizeof(float)));
	// done
	gpuErrchk(cudaMalloc((void**)&p.done, num_rays * sizeof(bool)));

	// copy ray.origin
	for (uint i = 0; i < num_rays; i++) temp[i] = rays[i].origin.x;
	gpuErrchk(cudaMemcpyAsync(p.oxs, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	for (uint i = 0; i < num_rays; i++) temp[i] = rays[i].origin.y;
	gpuErrchk(cudaMemcpyAsync(p.oys, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	for (uint i = 0; i < num_rays; i++) temp[i] = rays[i].origin.z;
	gpuErrchk(cudaMemcpyAsync(p.ozs, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	// copy ray.direction
	for (uint i = 0; i < num_rays; i++) temp[i] = rays[i].direction.x;
	gpuErrchk(cudaMemcpyAsync(p.dxs, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	for (uint i = 0; i < num_rays; i++) temp[i] = rays[i].direction.y;
	gpuErrchk(cudaMemcpyAsync(p.dys, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	for (uint i = 0; i < num_rays; i++) temp[i] = rays[i].direction.z;
	gpuErrchk(cudaMemcpyAsync(p.dzs, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	// init colors to 1 (should be done in a kernel)
	for (uint i = 0; i < num_rays; i++) temp[i] = 1;
	gpuErrchk(cudaMemcpyAsync(p.cxs, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(p.cys, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyAsync(p.czs, temp, num_rays * sizeof(float), cudaMemcpyHostToDevice));
	// set done to false
	gpuErrchk(cudaMemset(p.done, 0, num_rays * sizeof(bool)));

	delete[] temp;
}

void copyClrsToHost(const paths &p, clr_rec *clrs, const uint num_rays) {
	float *temp = new float[num_rays];
	bool *done = new bool[num_rays];
	gpuErrchk(cudaMemcpy(temp, p.cxs, num_rays * sizeof(float), cudaMemcpyDeviceToHost));
	for (uint i = 0; i < num_rays; i++) clrs[i].color.x = temp[i];
	gpuErrchk(cudaMemcpy(temp, p.cys, num_rays * sizeof(float), cudaMemcpyDeviceToHost));
	for (uint i = 0; i < num_rays; i++) clrs[i].color.y = temp[i];
	gpuErrchk(cudaMemcpy(temp, p.czs, num_rays * sizeof(float), cudaMemcpyDeviceToHost));
	for (uint i = 0; i < num_rays; i++) clrs[i].color.z = temp[i];
	gpuErrchk(cudaMemcpy(done, p.done, num_rays * sizeof(bool), cudaMemcpyDeviceToHost));
	for (uint i = 0; i < num_rays; i++) clrs[i].done = done[i];

	delete[] temp;
	delete[] done;
}

void releasePaths(const paths &p) {
	// ray.origin
	gpuErrchk(cudaFree(p.oxs));
	gpuErrchk(cudaFree(p.oys));
	gpuErrchk(cudaFree(p.ozs));
	// ray.direction
	gpuErrchk(cudaFree(p.dxs));
	gpuErrchk(cudaFree(p.dys));
	gpuErrchk(cudaFree(p.dzs));
	// color
	gpuErrchk(cudaFree(p.cxs));
	gpuErrchk(cudaFree(p.cys));
	gpuErrchk(cudaFree(p.czs));
	// hit
	gpuErrchk(cudaFree(p.hit_faces));
	gpuErrchk(cudaFree(p.hit_ts));
	// done
	gpuErrchk(cudaFree(p.done));
}

int main(int argc, char** argv) {
	options o;
	if (!parse_args(o, argc, argv)) return;

	const uint nx = o.nx, ny = o.ny, spp = o.spp, max_depth = o.max_depth;
	const uint num_rays = nx*ny*spp;
	const uint threadsPerBlock = 64;

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
	gpuErrchk(cudaMalloc((void **)&d_heightmap, model->size.x*model->size.z * sizeof(unsigned char)));
	gpuErrchk(cudaMemcpy(d_heightmap, model->heightmap, model->size.x*model->size.z * sizeof(unsigned char), cudaMemcpyHostToDevice));

	// allocate all samples
	ray* rays = new ray[num_rays];

	// start by generating all primary rays
	generate_rays(rays, c, nx, ny, spp);

	// prepare paths
	paths p;
	init_paths(p, rays, num_rays);

	delete[] rays;

	cudaEvent_t hit_start, hit_done, color_done, iter_done;
	if (o.kernel_perf) {
		gpuErrchk(cudaEventCreate(&hit_start));
		gpuErrchk(cudaEventCreate(&hit_done));
		gpuErrchk(cudaEventCreate(&color_done));
	}
	else {
		gpuErrchk(cudaEventCreate(&iter_done));
	}
	float hit_duration_ms = 0;
	float color_duration_ms = 0;

	const clock_t start = clock();
	for (uint iteration = 0; iteration < max_depth; iteration++) {
		if (o.kernel_perf) gpuErrchk(cudaEventRecord(hit_start));
		{
			const uint rays_per_strip = num_rays / o.num_strips;
			const uint blocksPerGrid = ceilf(rays_per_strip / threadsPerBlock);
			for (uint j = 0; j < o.num_strips; j++) {
				hit_scene <<<blocksPerGrid, threadsPerBlock, 0 >>> (p, rays_per_strip*j, rays_per_strip*(j+1), d_heightmap, model->size);
			}
		}
		gpuErrchk(cudaPeekAtLastError());
		if (o.kernel_perf) gpuErrchk(cudaEventRecord(hit_done));
		{
			const uint blocksPerGrid = (num_rays + threadsPerBlock - 1) / threadsPerBlock;
			simple_color <<<blocksPerGrid, threadsPerBlock, 0 >>> (p, num_rays, 0, iteration, albedo, *s, spp);
		}
		gpuErrchk(cudaPeekAtLastError());
		if (o.kernel_perf) gpuErrchk(cudaEventRecord(color_done));
		if (!o.kernel_perf)	gpuErrchk(cudaEventRecord(iter_done));

		if (o.kernel_perf) {
			gpuErrchk(cudaEventSynchronize(color_done));
			float duration_ms = 0;
			gpuErrchk(cudaEventElapsedTime(&duration_ms, hit_start, hit_done));
			hit_duration_ms += duration_ms;
			if (o.per_iter_perf && duration_ms > 0) std::cout << " hit_scene " << sscale(num_rays*1000.0 / duration_ms) << " rays/s" << std::endl;
			gpuErrchk(cudaEventElapsedTime(&duration_ms, hit_done, color_done));
			color_duration_ms += duration_ms;
			if (o.per_iter_perf && duration_ms > 0) std::cout << " simple_color " << sscale(num_rays*1000.0 / duration_ms) << " rays/s" << std::endl;
		} else {
			gpuErrchk(cudaEventSynchronize(iter_done));
		}
	}
	const float total_duration_ms = clock() - start;
	std::cout << "total duration " << (total_duration_ms / 1000.0) << " seconds" << std::endl;
	std::cout << "  " << sscale(max_depth*num_rays*1000.0 / total_duration_ms) << " rays/s" << std::endl;

	// copy colors to cpu
	clr_rec *clrs = new clr_rec[num_rays];
	copyClrsToHost(p, clrs, num_rays);

	releasePaths(p);

	if (o.kernel_perf) {
		if (hit_duration_ms > 0) {
			std::cout << "hit_scene took " << (hit_duration_ms / 1000.0) << " seconds" << std::endl;
			std::cout << "  " << sscale(max_depth*num_rays*1000.0 / hit_duration_ms) << " rays/s" << std::endl;
		}
		if (color_duration_ms > 0) {
			std::cout << "color_scene took " << (color_duration_ms / 1000.0) << " seconds" << std::endl;
			std::cout << "  " << sscale(max_depth*num_rays*1000.0 / color_duration_ms) << " rays/s" << std::endl;
		}
	}

	if (o.show_image) display_image(clrs, nx, ny, spp);

	delete[] clrs;

    return 0;
}