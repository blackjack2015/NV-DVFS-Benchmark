/**
 * main-cuda.cpp: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"
#include "version_info.h"
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

#ifdef _MSC_VER 
//not #if defined(_WIN32) || defined(_WIN64) because we have strncasecmp in mingw
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

#ifdef READONLY
#define VECTOR_SIZE (32*1024*1024)
#else
#define VECTOR_SIZE (8*1024*1024)
#endif

int main(int argc, char* argv[]) {
#ifdef READONLY
	printf("mixbench/read-only (%s)\n", VERSION_INFO);
#else
	printf("mixbench/alternating (%s)\n", VERSION_INFO);
#endif

	unsigned int datasize = VECTOR_SIZE*sizeof(double);

    // indicate device id
    findCudaDevice(argc, (const char **)argv);
	StoreDeviceInfo(stdout);

    int ratio = 1;
    int secs = 2;
    BenchType benchtype = BENCH_FLOAT;
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
        char *typeChoice = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "type", &typeChoice);

        if (!strcasecmp(typeChoice, "float"))
        {
            benchtype = BENCH_FLOAT;
        }
        else if (!strcasecmp(typeChoice, "double"))
        {
            benchtype = BENCH_DOUBLE; 
        }
        else if (!strcasecmp(typeChoice, "int"))
        {
            benchtype = BENCH_INT; 
        }
    }

	size_t freeCUDAMem, totalCUDAMem;
	cudaMemGetInfo(&freeCUDAMem, &totalCUDAMem);
	printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
	printf("Buffer size:          %dMB\n", datasize/(1024*1024));
	
	double *c;
	c = (double*)malloc(datasize);

	mixbenchGPU(c, VECTOR_SIZE, ratio, benchtype, secs);

	free(c);

	return 0;
}
