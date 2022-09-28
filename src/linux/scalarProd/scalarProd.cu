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

/*
 * This sample calculates scalar products of a
 * given set of input vector pairs
 */



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>

///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
    float *h_C,
    float *h_A,
    float *h_B,
    int vectorN,
    int elementN
);



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cuh"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Total number of input vector pairs; arbitrary
int VECTOR_N = 256;
//Number of elements per vector; arbitrary,
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
const int ELEMENT_N = 4096;
//Total number of data elements
int    DATA_N = VECTOR_N * ELEMENT_N;
int   DATA_SZ = DATA_N * sizeof(float);
int RESULT_SZ = VECTOR_N  * sizeof(float);



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int nIter = 100;
int secs = 180;
bool timeRestrict = false;

int main(int argc, char **argv)
{
    float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    float *d_A, *d_B, *d_C;
    double delta, ref, sum_delta, sum_ref, L1norm;
    StopWatchInterface *hTimer = NULL;
    int i;

    printf("%s Starting...\n\n", argv[0]);

	// array size
	if (checkCmdLineFlag(argc, (const char **)argv, "N"))
	{
		VECTOR_N = getCmdLineArgumentInt(argc, (const char **)argv, "N") * 1024;
		DATA_N = VECTOR_N * ELEMENT_N;

		DATA_SZ = DATA_N * sizeof(float);
		RESULT_SZ = VECTOR_N  * sizeof(float);
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

	if (dev == -1)
	{
		return EXIT_FAILURE;
	}

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_A     = (float *)malloc(DATA_SZ);
    h_B     = (float *)malloc(DATA_SZ);
    h_C_CPU = (float *)malloc(RESULT_SZ);
    h_C_GPU = (float *)malloc(RESULT_SZ);

    printf("...allocating GPU memory.\n");
    checkCudaErrors(cudaMalloc((void **)&d_A, DATA_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_B, DATA_SZ));
    checkCudaErrors(cudaMalloc((void **)&d_C, RESULT_SZ));

    printf("...generating input data in CPU mem.\n");
    srand(123);

    //Generating input data on CPU
    for (i = 0; i < DATA_N; i++)
    {
        h_A[i] = RandFloat(0.0f, 1.0f);
        h_B[i] = RandFloat(0.0f, 1.0f);
    }

    printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpy(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice));
    printf("Data init done.\n");


    printf("Executing GPU kernel...\n");
    checkCudaErrors(cudaDeviceSynchronize());
	
	cudaEvent_t start, stop;
	// Record the start event
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	double totalTime = 0.0;

	int k = -1;
	while (k < nIter)
	{
		// Run kernel and record the time
		checkCudaErrors(cudaEventRecord(start, NULL));

		scalarProdGPU<<<128, 256>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N);

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

    getLastCudaError("scalarProdGPU() execution failed\n");
    checkCudaErrors(cudaDeviceSynchronize());
    //sdkStopTimer(&hTimer);
    printf("Total   Kernel Time scalarProdGPU() time: %f msec\n", totalTime);
    printf("Average Kernel Time scalarProdGPU() time: %f msec\n", totalTime / (double)nIter);
    printf("\n");

    printf("Reading back GPU result...\n");
    //Read back GPU results to compare them to CPU results
    checkCudaErrors(cudaMemcpy(h_C_GPU, d_C, RESULT_SZ, cudaMemcpyDeviceToHost));


    printf("Checking GPU results...\n");
    printf("..running CPU scalar product calculation\n");
    scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);

    printf("...comparing the results\n");
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;

    for (i = 0; i < VECTOR_N; i++)
    {
        delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
        ref   = h_C_CPU[i];
        sum_delta += delta;
        sum_ref   += ref;
    }

    L1norm = sum_delta / sum_ref;

    printf("Shutting down...\n");
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_A));
    free(h_C_GPU);
    free(h_C_CPU);
    free(h_B);
    free(h_A);
    sdkDeleteTimer(&hTimer);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    printf("L1 error: %E\n", L1norm);
    printf((L1norm < 1e-6) ? "Test passed\n" : "Test failed!\n");
    exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}
