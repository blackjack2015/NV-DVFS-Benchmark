/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * This sample implemenets bitonic sort and odd-even merge sort, algorithms
 * belonging to the class of sorting networks.
 * While generally subefficient on large sequences
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 *
 * Victor Podlozhnyuk, 07/09/2009
 */

// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_timer.h>

#include "sortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int nIter = 100;
int secs = 180;
bool timeRestrict = false;

int main(int argc, char **argv)
{
	uint             N = 1048576;
	const uint           DIR = 0;
	const uint     numValues = 65536;

    cudaError_t error;
    printf("%s Starting...\n\n", argv[0]);

    printf("Starting up CUDA context...\n");

	// array size
	if (checkCmdLineFlag(argc, (const char **)argv, "N"))
	{
		N = getCmdLineArgumentInt(argc, (const char **)argv, "N") * 1048576;
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

	int dev = findCudaDevice(argc, (const char **)argv);

    uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
    uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;
    StopWatchInterface *hTimer = NULL;




    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_InputKey     = (uint *)malloc(N * sizeof(uint));
    h_InputVal     = (uint *)malloc(N * sizeof(uint));
    h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
    h_OutputValGPU = (uint *)malloc(N * sizeof(uint));
    srand(2001);

    for (uint i = 0; i < N; i++)
    {
        h_InputKey[i] = rand() % numValues;
        h_InputVal[i] = i;
    }

    printf("Allocating and initializing CUDA arrays...\n\n");
    error = cudaMalloc((void **)&d_InputKey,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_InputVal,  N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputKey, N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMalloc((void **)&d_OutputVal, N * sizeof(uint));
    checkCudaErrors(error);
    error = cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice);
    checkCudaErrors(error);
    error = cudaMemcpy(d_InputVal, h_InputVal, N * sizeof(uint), cudaMemcpyHostToDevice);
    checkCudaErrors(error);

    int flag = 1;
    printf("Running GPU bitonic sort (%u identical iterations)...\n\n", nIter);

    for (uint arrayLength = N; arrayLength <= N; arrayLength *= 2)
    {
        printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        sdkResetTimer(&hTimer);
        //sdkStartTimer(&hTimer);
        uint threadCount = 0;

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

            threadCount = bitonicSort(
                              d_OutputKey,
                              d_OutputVal,
                              d_InputKey,
                              d_InputVal,
                              N / arrayLength,
                              arrayLength,
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

        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        //sdkStopTimer(&hTimer);
        printf("iterated %d, average time is %f msec.\n", nIter, totalTime / nIter);

        if (arrayLength == N)
        {
            double dTimeSecs = 1.0e-3 * totalTime / nIter;
            printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
                   (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);
        }

        printf("\nValidating the results...\n");
        printf("...reading back GPU results\n");
        error = cudaMemcpy(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);
        error = cudaMemcpy(h_OutputValGPU, d_OutputVal, N * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);

        int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, DIR);
        int valuesFlag = validateValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, N / arrayLength, arrayLength);
        flag = flag && keysFlag && valuesFlag;

        printf("\n");
    }

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    cudaFree(d_OutputVal);
    cudaFree(d_OutputKey);
    cudaFree(d_InputVal);
    cudaFree(d_InputKey);
    free(h_OutputValGPU);
    free(h_OutputKeyGPU);
    free(h_InputVal);
    free(h_InputKey);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
