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
* This sample implements 64-bin histogram calculation
* of arbitrary-sized 8-bit data array
*/

// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <ctime>

// project include
#include "histogram_common.h"

int nIter = 100;
int secs = 180;
bool timeRestrict = false;

static const char *sSDKsample = "[histogram]\0";

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    StopWatchInterface *hTimer = NULL;
    int PassFailFlag = 1;
    
	uint byteNum = 1048576;
	uint byteCount = 64 * byteNum;

    uint uiSizeMult = 1;

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

    // set logfile name and start logs
    printf("[%s] - Starting...\n", sSDKsample);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	// bytes
	if (checkCmdLineFlag(argc, (const char **)argv, "bytes"))
	{
		byteNum = getCmdLineArgumentInt(argc, (const char **)argv, "bytes");
		byteCount = 64 * byteNum;
	}

	// Iteration Count
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

    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",
           deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    int version = deviceProp.major * 0x10 + deviceProp.minor;

    if (version < 0x11)
    {
        printf("There is no device supporting a minimum of CUDA compute capability 1.1 for this CUDA Sample\n");

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }

    sdkCreateTimer(&hTimer);

    // Optional Command-line multiplier to increase size of array to histogram
    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        uiSizeMult = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
        uiSizeMult = MAX(1,MIN(uiSizeMult, 10));
        byteCount *= uiSizeMult;
    }

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
    h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

    printf("...generating input data\n");
    srand(2009);

    for (uint i = 0; i < byteCount; i++)
    {
        h_Data[i] = rand() % 256;
    }

    printf("...allocating GPU memory and copying input data\n\n");
    checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));

  //  {
  //      printf("Starting up 64-bin histogram...\n\n");
  //      initHistogram64();

  //      printf("Running 64-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, nIter);

		//for (int iter = -1; iter < nIter; iter++)
  //      {
  //          //iter == -1 -- warmup iteration
  //          if (iter == 0)
  //          {
  //              cudaDeviceSynchronize();
  //              sdkResetTimer(&hTimer);
  //              sdkStartTimer(&hTimer);
  //          }

  //          histogram64(d_Histogram, d_Data, byteCount);
  //      }

  //      cudaDeviceSynchronize();
  //      sdkStopTimer(&hTimer);
		//double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)nIter;
		//printf("histogram64() time (total) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs * (double)nIter, ((double)byteCount * 1.0e-6) / dAvgSecs);
  //      printf("histogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
  //             (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM64_THREADBLOCK_SIZE);

  //      printf("\nValidating GPU results...\n");
  //      printf(" ...reading back GPU results\n");
  //      checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM64_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

  //      printf(" ...histogram64CPU()\n");
  //      histogram64CPU(
  //          h_HistogramCPU,
  //          h_Data,
  //          byteCount
  //      );

  //      printf(" ...comparing the results...\n");

  //      for (uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
  //          if (h_HistogramGPU[i] != h_HistogramCPU[i])
  //          {
  //              PassFailFlag = 0;
  //          }

  //      printf(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n");

  //      printf("Shutting down 64-bin histogram...\n\n\n");
  //      closeHistogram64();
  //  }

	cudaEvent_t start, stop;
	// Record the start event
	clock_t startTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	double totalTime = 0.0;

    {
        printf("Initializing 256-bin histogram...\n");
        initHistogram256();

		printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, nIter);

		for (int iter = -1; iter < nIter; iter++)
        {
			// Run kernel and record the time
			checkCudaErrors(cudaEventRecord(start, NULL));

			histogram256(d_Histogram, d_Data, byteCount);

			cudaDeviceSynchronize();
			checkCudaErrors(cudaEventRecord(stop, NULL));

			// Wait for the stop event to complete
			checkCudaErrors(cudaEventSynchronize(stop));
			float msecTotal = 0.0f;
			checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

			//iter == -1 -- warmup iteration
			if (iter == -1)
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

        cudaDeviceSynchronize();

		merge256(d_Histogram, d_Data, byteCount);
		cudaDeviceSynchronize();

		double averMsecs = totalTime / nIter;
		printf("iterated %d, average time is %f msec.\n", nIter, averMsecs);

        //sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * totalTime / (double)nIter;
		printf("histogram256() time (total) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs * (double)nIter, ((double)byteCount * 1.0e-6) / dAvgSecs);
        printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
               (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);

        printf("\nValidating GPU results...\n");
        printf(" ...reading back GPU results\n");
        checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));

        printf(" ...histogram256CPU()\n");
        histogram256CPU(
            h_HistogramCPU,
            h_Data,
            byteCount
        );

        printf(" ...comparing the results\n");

        for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
            if (h_HistogramGPU[i] != h_HistogramCPU[i])
            {
                PassFailFlag = 0;
            }

        printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");

        printf("Shutting down 256-bin histogram...\n\n\n");
        closeHistogram256();
    }

    printf("Shutting down...\n");
    sdkDeleteTimer(&hTimer);
    checkCudaErrors(cudaFree(d_Histogram));
    checkCudaErrors(cudaFree(d_Data));
    free(h_HistogramGPU);
    free(h_HistogramCPU);
    free(h_Data);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    printf("%s - Test Summary\n", sSDKsample);

    // pass or fail (for both 64 bit and 256 bit histograms)
    if (!PassFailFlag)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
