/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_math.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

#define SIZE	5

typedef struct soa {
	float *arr[SIZE];
} soa;

typedef struct ray {
	float arr[SIZE];
} ray;

__global__ void offset_soa(soa a, const int s)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float r = a.arr[0][i];
	r += a.arr[1][i];
	r += a.arr[2][i];
	r += a.arr[3][i];
	r += a.arr[4][i];
	a.arr[0][i] = r + 1;
}

__global__ void offset_ray(ray *r, const int s) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	float c = r[i].arr[0];
	c += r[i].arr[1];
	c += r[i].arr[2];
	c += r[i].arr[4];
	r[i].arr[0] = c + 1;
}

void runTest(int deviceId, const int n)
{
	int blockSize = 256;
	float ms;

	soa s;
	ray *r;
	cudaEvent_t startEvent, stopEvent;

	for (uint i = 0; i < SIZE; i++)
		checkCuda(cudaMalloc(&(s.arr[i]), n * sizeof(float)));
	checkCuda(cudaMalloc(&r, n * sizeof(ray)));

	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	printf("Offset SOA, Bandwidth (GB/s):\n");

	offset_soa << <n / blockSize, blockSize >> > (s, SIZE - 1); // warm up

	for (int i = SIZE-1; i < SIZE; i++) {
		for (uint j = 0; j < SIZE; j++)
			checkCuda(cudaMemset(s.arr[j], 0, n * sizeof(float)));

		checkCuda(cudaEventRecord(startEvent, 0));
		offset_soa << <n / blockSize, blockSize >> >(s, i);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		const uint nMB = n * (i + 2) * sizeof(float) / (1024*1024); // mem ops in MB (x2 as we have 1 read + 1 write per column)
		printf("%d, %f\n", i, nMB / ms);
	}

	printf("\n");

	printf("Offset AOS, Bandwidth (GB/s):\n");

	offset_ray << <n / blockSize, blockSize >> >(r, 0); // warm up

	for (int i = SIZE-1; i < SIZE; i++) {
		checkCuda(cudaMemset(r, 0, n * sizeof(ray)));

		checkCuda(cudaEventRecord(startEvent, 0));
		offset_ray << <n / blockSize, blockSize >> >(r, i);
		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));

		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		const uint nMB = n * (i + 2) * sizeof(float) / (1024 * 1024); // mem ops in MB (x2 as we have 1 read + 1 write per column)
		printf("%d, %f\n", i, nMB / ms);
	}

	printf("\n");

	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));
	for (uint i = 0; i < SIZE; i++)
		cudaFree(s.arr[i]);
	cudaFree(r);
}

int main(int argc, char **argv)
{
	int deviceId = 0;

	cudaDeviceProp prop;

	checkCuda(cudaSetDevice(deviceId));
	checkCuda(cudaGetDeviceProperties(&prop, deviceId));
	printf("Device: %s\n", prop.name);
	printf("Memory bus width: %d\n", prop.memoryBusWidth);

	runTest(deviceId, 10*1024*1024);
}