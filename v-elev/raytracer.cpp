#include <iostream>
#include <float.h>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <thread>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include "stb_image.h"

#include "renderer.h"
#include "utils.h"
#include "material.h"

#include "sdl.h"

using namespace std;

void test_scene0(camera **cam, voxelModel **model, float aspect) {
	const uint scene_size = 10;
	unsigned char *data = new unsigned char[scene_size*scene_size];
	for (uint i = 0; i < scene_size*scene_size; i++)
		data[i] = 0;
	data[0] = 10;
	data[1] = 1;
	data[scene_size] = 1;
	data[scene_size+1] = 1;

	*model = new voxelModel(data, scene_size, scene_size);
	*cam = new camera(make_float3(10), make_float3(1.5, 0.5, 1.5), make_float3(0, 1, 0), 20, aspect, 0, 1.0);
}

void test_scene1(camera **cam, voxelModel **model, float aspect) {
	const uint scene_size = 10;
	unsigned char *data = new unsigned char[scene_size*scene_size];
	for (uint i = 0; i < scene_size*scene_size; i++)
		data[i] = 0;
	for (uint x = 0; x < 4; x++) {
		for (uint y = 0; y < 4; y++) {
			if (x > 0 && x < 3 && y > 0 && y < 3)
				data[y * scene_size + x] = 2;
			else
				data[y * scene_size + x] = 1;
		}
	}
	data[scene_size * 1 + 1] = 3;

	*model = new voxelModel(data, scene_size, scene_size);
	*cam = new camera(make_float3(10), make_float3(1.5, 0.5, 1.5), make_float3(0, 1, 0), 20, aspect, 0, 1.0);
}

void city_scene(camera **cam, voxelModel** model, float aspect) {
	// load heightmap image
	int image_x, image_y, image_n;
	int image_desired_channels = 1; // grayscale
	unsigned char *data = stbi_load("city512.png", &image_x, &image_y, &image_n, image_desired_channels);
	if (data == NULL) {
		*model = NULL;
		return;
	}

	*model = new voxelModel(data, image_x, image_y);
	*cam = new camera(make_float3(700), make_float3(image_x / 2, 0, image_y / 2), make_float3(0, 1, 0), 20, aspect, 0, 1.0);
}

void call_from_thread(renderer& r, const uint unit_idx) {
	r.render_work_unit(unit_idx);
}

void write_image(uint nx, uint ny, const renderer& r) {
	char *data = new char[nx*ny * 3];
	int idx = 0;
	for (int y = ny - 1; y >= 0; y--) {
		for (int x = 0; x < nx; x++) {
			const float3 col = r.get_pixel_color(x, y);
			data[idx++] = min(255, int(255.99*col.x));
			data[idx++] = min(255, int(255.99*col.y));
			data[idx++] = min(255, int(255.99*col.z));
		}
	}
	stbi_write_png("picture.png", nx, ny, 3, (void*)data, nx * 3);
	delete[] data;
}

void display_image(uint nx, uint ny, const renderer& r) {
	uint *pixels = new uint[nx*ny];

	// copy pixels from RGB to ARGB
	for (int y = ny - 1, idx = 0; y >= 0; y--) {
		for (int x = 0; x < nx; x++, idx++) {
			const float3 col = r.get_pixel_color(x, y);
			unsigned char red = min(255, int(255.99*col.x));
			unsigned char green = min(255, int(255.99*col.y));
			unsigned char blue = min(255, int(255.99*col.z));
			pixels[idx] = (red << 16) + (green << 8) + blue;
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

	win = SDL_CreateWindow("Image Loading", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, nx, ny, 0);
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

	delete[] pixels;
	SDL_Quit();
}

/**
 * Host main routine
 */
int main(int argc, char** argv)
{
	bool writeImage = true;
	bool showImage = true;

	const uint num_threads = 1;
	const int nx = 500;
	const int ny = 500;
	const int ns = 500;
	const uint max_depth = 50;

	camera *cam;
	voxelModel *model;
	city_scene(&cam, &model, float(nx) / float(ny));
	if (model == NULL) {
		cerr << "couldn't load image" << endl;
		return 1;
	}
	printf("voxel size (%d, %d, %d)\n", model->size.x, model->size.y, model->size.z);

	const float albedo = .45;
	renderer r(cam, model, make_float3(albedo), nx, ny, ns, max_depth, 0.001f, num_threads);
	r.prepare_kernel();

	clock_t begin = clock();
	thread t[num_threads];
	// launch a group of threads 
	for (uint i = 0; i < num_threads; i++)
		t[i] = thread(call_from_thread, ref(r), i);
	// join the threads with the main thread
	for (uint i = 0; i < num_threads; i++)
		t[i].join();
	cout << "total execution took " << double(clock() - begin) / CLOCKS_PER_SEC << endl;
	for (uint i = 0; i < num_threads; i++) {
		const work_unit * wu = r.get_wunit(i);
		const double wu_cpu = double(wu->cpu_time);
		const double wu_gpu = double(wu->gpu_time);
		cout << "thread: " << i << " " << wu->num_iter << 
			" iterations, cpu: " << wu_cpu / CLOCKS_PER_SEC << " (" << wu_cpu / wu->num_iter << ")"
			", gpu: " << wu_gpu / CLOCKS_PER_SEC << "(" << wu_gpu / wu->num_iter << ")" << endl;
	}

	if (writeImage)
		write_image(nx, ny, r);
	if (showImage)
		display_image(nx, ny, r);
	
	r.destroy();

	stbi_image_free((void*) (model->heightmap));
	//delete[] model->heightmap;

    return 0;
}