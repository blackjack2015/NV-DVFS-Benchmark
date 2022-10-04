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
#include "cache_kernels.h"
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

#define VECTOR_SIZE (/*4*/32*1024*1024)

// Initialize vector data
void init_vector(double *v, size_t datasize){
	for(int i=0; i<(int)datasize; i++){
		v[i] = i;
	}
}


int main(int argc, char* argv[]) {
	printf("CUDA cachebench (repeated memory cached operations microbenchmark)\n");

	unsigned int datasize = VECTOR_SIZE*sizeof(int4); // reserve space for int4 types

    // indicate device id
    findCudaDevice(argc, (const char **)argv);

    bool readonly = false;
    int step = 1;
    int idxClamp = 0;

    int secs = 2;
    int benchtype = 0;  // 0 for int1, 1 for int2, 2 for int4
    // parse input args
    if (checkCmdLineFlag(argc, (const char **)argv, "readonly"))
    {
        readonly = getCmdLineArgumentInt(argc, (const char **)argv, "readonly") == 1 ? true : false;
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "step"))
    {
        step = getCmdLineArgumentInt(argc, (const char **)argv, "step");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "idxClamp"))
    {
        idxClamp = getCmdLineArgumentInt(argc, (const char **)argv, "idxClamp");
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

	printf("Buffer size: %dMB\n", datasize/(1024*1024));
	
	double *c = (double*)malloc(datasize);
	init_vector(c, VECTOR_SIZE);

	// benchmark execution
	cachebenchGPU(c, VECTOR_SIZE, true, readonly, benchtype, step, idxClamp, secs);

	free(c);

	return 0;
}

