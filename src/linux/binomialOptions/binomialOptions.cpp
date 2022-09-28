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
 * This sample evaluates fair call price for a
 * given set of European options under binomial model.
 * See supplied whitepaper for more explanations.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include "binomialOptions_common.h"

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCall(
    float &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsCPU(
    float &callResult,
    TOptionData optionData
);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
    float *callValue,
    TOptionData  *optionData,
    int optN,
	int nIter,
	int secs,
	bool timeRestrict
);

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float randData(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}


int nIter = 100;
int secs = 180;
bool timeRestrict = false;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    unsigned int OPT_N_MAX = 512;

    printf("[%s] - Starting...\n", argv[0]);

	// opt n max
	if (checkCmdLineFlag(argc, (const char **)argv, "optN"))
	{
		OPT_N_MAX = getCmdLineArgumentInt(argc, (const char **)argv, "optN");
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

    cudaDeviceProp deviceProp;
    int devID = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (((deviceProp.major << 4) + deviceProp.minor) < 0x20)
    {
        fprintf(stderr, "binomialOptions requires Compute Capability of SM 2.0 or higher to run.\n");
        cudaDeviceReset();
        exit(EXIT_WAIVED);
    }

    int OPT_N = OPT_N_MAX;

	TOptionData* optionData = new TOptionData[OPT_N];
	float* callValueBS = new float[OPT_N];
	float* callValueGPU = new float[OPT_N];
	float* callValueCPU = new float[OPT_N];

    double
    sumDelta, sumRef, gpuTime, errorVal;

    StopWatchInterface *hTimer = NULL;
    int i;

    sdkCreateTimer(&hTimer);

    printf("Generating input data...\n");
    //Generate options set
    srand(123);

    for (i = 0; i < OPT_N; i++)
    {
        optionData[i].S = randData(5.0f, 30.0f);
        optionData[i].X = randData(1.0f, 100.0f);
        optionData[i].T = randData(0.25f, 10.0f);
        optionData[i].R = 0.06f;
        optionData[i].V = 0.10f;
        BlackScholesCall(callValueBS[i], optionData[i]);
    }

    printf("Running GPU binomial tree...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    binomialOptionsGPU(callValueGPU, optionData, OPT_N, nIter, secs, timeRestrict);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("Options count            : %i     \n", OPT_N);
    printf("Time steps               : %i     \n", NUM_STEPS);
    
    printf("Running CPU binomial tree...\n");

    for (i = 0; i < OPT_N; i++)
    {
        binomialOptionsCPU(callValueCPU[i], optionData[i]);
    }

    printf("Comparing the results...\n");
    sumDelta = 0;
    sumRef   = 0;
    printf("GPU binomial vs. Black-Scholes\n");

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueBS[i] - callValueGPU[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", sumDelta / (double)OPT_N);
    }

    printf("CPU binomial vs. Black-Scholes\n");
    sumDelta = 0;
    sumRef   = 0;

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueBS[i]- callValueCPU[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", sumDelta / (double)OPT_N);
    }

    printf("CPU binomial vs. GPU binomial\n");
    sumDelta = 0;
    sumRef   = 0;

    for (i = 0; i < OPT_N; i++)
    {
        sumDelta += fabs(callValueGPU[i] - callValueCPU[i]);
        sumRef += callValueCPU[i];
    }

    if (sumRef > 1E-5)
    {
        printf("L1 norm: %E\n", errorVal = sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", errorVal = sumDelta / (double)OPT_N);
    }

    printf("Shutting down...\n");

    sdkDeleteTimer(&hTimer);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    if (errorVal > 5e-4)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
