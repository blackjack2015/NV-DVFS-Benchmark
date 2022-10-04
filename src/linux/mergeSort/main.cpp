/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "mergeSort_common.h"


////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int nIter = 100;
int secs = 180;
bool timeRestrict = false;

int main(int argc, char **argv)
{
    uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
    uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
    StopWatchInterface *hTimer = NULL;

    uint   N = 1024 * 1024;
    const uint DIR = 1;
    const uint numValues = 65536;

    printf("%s Starting...\n\n", argv[0]);

	// array length
	if (checkCmdLineFlag(argc, (const char **)argv, "N"))
	{
		N = getCmdLineArgumentInt(argc, (const char **)argv, "N");
	}

	// Iteration count
	if (checkCmdLineFlag(argc, (const char **)argv, "iters"))
	{
		nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iters");
	}

	// Power Running Time
	if (checkCmdLineFlag(argc, (const char **)argv, "secs"))
	{
		secs = getCmdLineArgumentInt(argc, (const char **)argv, "secs");
		timeRestrict = true;
	}

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_SrcKey = (uint *)malloc(N * sizeof(uint));
    h_SrcVal = (uint *)malloc(N * sizeof(uint));
    h_DstKey = (uint *)malloc(N * sizeof(uint));
    h_DstVal = (uint *)malloc(N * sizeof(uint));

    srand(2009);

    for (uint i = 0; i < N; i++)
    {
        h_SrcKey[i] = rand() % numValues;
    }

    fillValues(h_SrcVal, N);

    printf("Allocating and initializing CUDA arrays...\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_DstKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_DstVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_BufVal, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));

    printf("Initializing GPU merge sort...\n");
    initMergeSort();

    printf("Running GPU merge sort...\n");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    //sdkStartTimer(&hTimer);

	cudaEvent_t start, stop;
	// Record the start event
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	double totalTime = 0.0;

	int k = -1;
	double gpuTime = 0;
	while (k < nIter)
	{
		// Run kernel and record the time
		checkCudaErrors(cudaEventRecord(start, NULL));

		mergeSort(
			d_DstKey,
			d_DstVal,
			d_BufKey,
			d_BufVal,
			d_SrcKey,
			d_SrcVal,
			N,
			DIR
		);

		cudaThreadSynchronize();

		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));
		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		//iter == -1 -- warmup iteration
		if (k == -1)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			//sdkResetTimer(&hTimer);
			//sdkStartTimer(&hTimer);
			if (timeRestrict){
				nIter = secs / (msecTotal * 0.001);
				printf("Adjust Iters to %d for meeting time requirement %d secs.\n", nIter, secs);
			}
		}
		else
		{
			totalTime += msecTotal;
		}

		k++;

	}

    checkCudaErrors(cudaDeviceSynchronize());
    //sdkStopTimer(&hTimer);
    printf("Total   Time: %f ms\n", totalTime);
	printf("Average Time: %f ms\n", totalTime/(double)nIter);

    printf("Reading back GPU merge sort results...\n");
    checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));

    printf("Inspecting the results...\n");
    uint keysFlag = validateSortedKeys(
                        h_DstKey,
                        h_SrcKey,
                        1,
                        N,
                        numValues,
                        DIR
                    );

    uint valuesFlag = validateSortedValues(
                          h_DstKey,
                          h_DstVal,
                          h_SrcKey,
                          1,
                          N
                      );

    printf("Shutting down...\n");
    closeMergeSort();
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_SrcVal));
    checkCudaErrors(cudaFree(d_SrcKey));
    checkCudaErrors(cudaFree(d_BufVal));
    checkCudaErrors(cudaFree(d_BufKey));
    checkCudaErrors(cudaFree(d_DstVal));
    checkCudaErrors(cudaFree(d_DstKey));
    free(h_DstVal);
    free(h_DstKey);
    free(h_SrcVal);
    free(h_SrcKey);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}
