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



////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <ctime>
#include "binomialOptions_common.h"



////////////////////////////////////////////////////////////////////////////////
// Internal GPU-side constants and data structures
////////////////////////////////////////////////////////////////////////////////
#define  TIME_STEPS 16
#define CACHE_DELTA (2 * TIME_STEPS)
#define  CACHE_SIZE (256)
#define  CACHE_STEP (CACHE_SIZE - CACHE_DELTA)

#if NUM_STEPS % CACHE_DELTA
#error Bad constants
#endif

//Preprocessed input option data
typedef struct
{
    double S;
    double X;
    double vDt;
    double puByDf;
    double pdByDf;
} __TOptionData;
static __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
static __device__           float d_CallValue[MAX_OPTIONS];
static __device__          double d_CallBuffer[MAX_OPTIONS * (NUM_STEPS + 16)];



////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////
#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
    double d = S * expf(vDt * (2.0f * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}
#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
static __global__ void binomialOptionsKernel()
{
    __shared__ double callA[CACHE_SIZE+1];
    __shared__ double callB[CACHE_SIZE+1];
    //Global memory frame for current option (thread block)
    double *const d_Call = &d_CallBuffer[blockIdx.x * (NUM_STEPS + 16)];

    const int       tid = threadIdx.x;
    const double      S = d_OptionData[blockIdx.x].S;
    const double      X = d_OptionData[blockIdx.x].X;
    const double    vDt = d_OptionData[blockIdx.x].vDt;
    const double puByDf = d_OptionData[blockIdx.x].puByDf;
    const double pdByDf = d_OptionData[blockIdx.x].pdByDf;

    //Compute values at expiry date
    for (int i = tid; i <= NUM_STEPS; i += CACHE_SIZE)
    {
        d_Call[i] = expiryCallValue(S, X, vDt, i);
    }

    //Walk down binomial tree
    //So double-buffer and synchronize to avoid read-after-write hazards.
    for (int i = NUM_STEPS; i > 0; i -= CACHE_DELTA)
        for (int c_base = 0; c_base < i; c_base += CACHE_STEP)
        {
            //Start and end positions within shared memory cache
            int c_start = min(CACHE_SIZE - 1, i - c_base);
            int c_end   = c_start - CACHE_DELTA;

            //Read data(with apron) to shared memory
            __syncthreads();

            if (tid <= c_start)
            {
                callA[tid] = d_Call[c_base + tid];
            }

            //Calculations within shared memory
            for (int k = c_start - 1; k >= c_end;)
            {
                //Compute discounted expected value
                __syncthreads();
                callB[tid] = puByDf * callA[tid + 1] + pdByDf * callA[tid];
                k--;

                //Compute discounted expected value
                __syncthreads();
                callA[tid] = puByDf * callB[tid + 1] + pdByDf * callB[tid];
                k--;
            }

            //Flush shared memory cache
            __syncthreads();

            if (tid <= c_end)
            {
                d_Call[c_base + tid] = callA[tid];
            }
        }

    //Write the value at the top of the tree to destination buffer
    if (threadIdx.x == 0)
    {
        d_CallValue[blockIdx.x] = (float)callA[0];
    }
}



////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU binomialOptions
////////////////////////////////////////////////////////////////////////////////
extern "C" void binomialOptionsGPU(
	float *callValue,
	TOptionData  *optionData,
	int optN,
	int nIter,
	int secs,
	bool timeRestrict
	)
{
	__TOptionData h_OptionData[MAX_OPTIONS];

	for (int i = 0; i < optN; i++)
	{
		const double      T = optionData[i].T;
		const double      R = optionData[i].R;
		const double      V = optionData[i].V;

		const double     dt = T / (double)NUM_STEPS;
		const double    vDt = V * sqrt(dt);
		const double    rDt = R * dt;
		//Per-step interest and discount factors
		const double     If = exp(rDt);
		const double     Df = exp(-rDt);
		//Values and pseudoprobabilities of upward and downward moves
		const double      u = exp(vDt);
		const double      d = exp(-vDt);
		const double     pu = (If - d) / (u - d);
		const double     pd = 1.0 - pu;
		const double puByDf = pu * Df;
		const double pdByDf = pd * Df;

		h_OptionData[i].S = (double)optionData[i].S;
		h_OptionData[i].X = (double)optionData[i].X;
		h_OptionData[i].vDt = (double)vDt;
		h_OptionData[i].puByDf = (double)puByDf;
		h_OptionData[i].pdByDf = (double)pdByDf;
	}

	double gpuTime;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	checkCudaErrors(cudaMemcpyToSymbol(d_OptionData, h_OptionData, optN * sizeof(__TOptionData)));
	checkCudaErrors(cudaDeviceSynchronize());
	//sdkResetTimer(&hTimer);
	//sdkStartTimer(&hTimer);
	
	
	cudaEvent_t start, stop;
	// Record the start event
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	double totalTime = 0.0;

	for (int i = -1; i < nIter; i++){
		// Run kernel and record the time
		checkCudaErrors(cudaEventRecord(start, NULL));
		binomialOptionsKernel << <optN, CACHE_SIZE >> >();
		cudaDeviceSynchronize();
		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));
		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		//iter == -1 -- warmup iteration
		if (i == -1)
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
	}

	getLastCudaError("binomialOptionsKernel() execution failed.\n");

	//sdkStopTimer(&hTimer);
	//gpuTime = sdkGetTimerValue(&hTimer);
	gpuTime = totalTime / nIter;
	printf("iterated %d, average time is %f msec.\n", nIter, gpuTime);

    checkCudaErrors(cudaMemcpyFromSymbol(callValue, d_CallValue, optN *sizeof(float)));
}
