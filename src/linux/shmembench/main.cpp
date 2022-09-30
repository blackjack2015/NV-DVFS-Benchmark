/**
 * main.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <lcutil.h>
#include "shmem_kernels.h"
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

#define VECTOR_SIZE (1024*1024)

int main(int argc, char* argv[]) {
	printf("CUDA shmembench (shared memory bandwidth microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(double) * 6;

    // indicate device id
    findCudaDevice(argc, (const char **)argv);
	StoreDeviceInfo(stdout);

    int ratio = 1;
    int secs = 2;
    int benchtype = 0;  // 0 for float, 1 for float2, 2 for float4
    // parse input args
    if (checkCmdLineFlag(argc, (const char **)argv, "ratio"))
    {
        ratio = getCmdLineArgumentInt(argc, (const char **)argv, "ratio");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "secs"))
    {
        secs = getCmdLineArgumentInt(argc, (const char **)argv, "secs");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "type"))
    {
        benchtype = getCmdLineArgumentInt(argc, (const char **)argv, "type");
    }

	StoreDeviceInfo(stdout);

	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);

	printf("Buffer sizes: %dMB\n", datasize/(1024*1024));
	
	double *c;
	c = (double*)malloc(datasize);
	// memset(c, 0, sizeof(int)*VECTOR_SIZE);
	memset(c, 0, datasize);

	// benchmark execution
	shmembenchGPU(c, VECTOR_SIZE, ratio, benchtype, secs);

	free(c);

	return 0;
}

