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
 * This sample implements the same algorithm as the convolutionSeparable
 * CUDA Sample, but without using the shared memory at all.
 * Instead, it uses textures in exactly the same way an OpenGL-based
 * implementation would do.
 * Refer to the "Performance" section of convolutionSeparable whitepaper.
 */




#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <ctime>

#include "convolutionTexture_common.h"

int nIter = 100;
int secs = 180;
bool timeRestrict = false;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    float
    *h_Kernel,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU;

    cudaArray
    *a_Src;

    cudaChannelFormatDesc floatTex = cudaCreateChannelDesc<float>();

    float
    *d_Output;

    float
    gpuTime;

    StopWatchInterface *hTimer = NULL;

    int imageW = 3072;
    int imageH = 3072 / 2;

    printf("[%s] - Starting...\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

	// Image size
	if (checkCmdLineFlag(argc, (const char **)argv, "W"))
	{
		imageW = getCmdLineArgumentInt(argc, (const char **)argv, "W");
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "H"))
	{
		imageH = getCmdLineArgumentInt(argc, (const char **)argv, "H");
	}

	// Iteration count
	if (checkCmdLineFlag(argc, (const char **)argv, "iters"))
	{
		nIter = getCmdLineArgumentInt(argc, (const char **)argv, "iters");
	}

	// Iteration count
	if (checkCmdLineFlag(argc, (const char **)argv, "secs"))
	{
		secs = getCmdLineArgumentInt(argc, (const char **)argv, "secs");
		timeRestrict = true;
	}

    sdkCreateTimer(&hTimer);

    printf("Initializing data...\n");
    h_Kernel    = (float *)malloc(KERNEL_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU = (float *)malloc(imageW * imageH * sizeof(float));
    checkCudaErrors(cudaMallocArray(&a_Src, &floatTex, imageW, imageH));
    checkCudaErrors(cudaMalloc((void **)&d_Output, imageW * imageH * sizeof(float)));

    srand(2009);

    for (unsigned int i = 0; i < KERNEL_LENGTH; i++)
    {
        h_Kernel[i] = (float)(rand() % 16);
    }

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        h_Input[i] = (float)(rand() % 16);
    }

    setConvolutionKernel(h_Kernel);
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice));


    printf("Running GPU rows convolution (%u identical iterations)...\n", nIter);
    checkCudaErrors(cudaDeviceSynchronize());
    //sdkResetTimer(&hTimer);
    //sdkStartTimer(&hTimer);

	cudaEvent_t start, stop;
	// Record the start event
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	double totalTime = 0.0;
	double averMsecs = 0.0;
	int c = 0;

	for (int i = -10; i < nIter; i++)
    {
		// Run kernel and record the time
		checkCudaErrors(cudaEventRecord(start, NULL));

		convolutionRowsGPU(
			d_Output,
			a_Src,
			imageW,
			imageH
			);

		cudaDeviceSynchronize();
		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));
		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		totalTime += msecTotal;
		c++;

		//iter == -1 -- warmup iteration
		if (i == -1)
		{
			checkCudaErrors(cudaDeviceSynchronize());
			//sdkResetTimer(&hTimer);
			//sdkStartTimer(&hTimer);
			double timePerIter = totalTime / c;
			if (timeRestrict){
				nIter = int(double(secs * 1000) / timePerIter);
				printf("Adjust Iters to %d for meeting time requirement %d secs.\n", nIter, secs);
			}
			c = 0;
			totalTime = 0;
		}

    }

    checkCudaErrors(cudaDeviceSynchronize());
    //sdkStopTimer(&hTimer);
	gpuTime = totalTime / nIter;
	printf("iterated %d, average time is %f msec.\n", nIter, gpuTime);


	printf("Total   convolutionRowsGPU() time: %f msecs\n", gpuTime * (float)nIter);
    printf("Average convolutionRowsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    //While CUDA kernels can't write to textures directly, this copy is inevitable
    printf("Copying convolutionRowGPU() output back to the texture...\n");
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);
    checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("cudaMemcpyToArray() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

	//printf("Running GPU columns convolution (%i iterations)\n", nIter);
	printf("Running GPU columns convolution (%i iterations)\n", 1);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

	//for (int i = 0; i < nIter; i++)
    {
        convolutionColumnsGPU(
            d_Output,
            a_Src,
            imageW,
            imageH
        );
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
	//gpuTime = sdkGetTimerValue(&hTimer) / (float)nIter;
	gpuTime = sdkGetTimerValue(&hTimer);
	printf("Total   convolutionColumnsGPU() time: %f msecs\n", gpuTime);
    printf("Average convolutionColumnsGPU() time: %f msecs; //%f Mpix/s\n", gpuTime, imageW * imageH * 1e-6 / (0.001 * gpuTime));

    printf("Reading back GPU results...\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Checking the results...\n");
    printf("...running convolutionRowsCPU()\n");
    convolutionRowsCPU(
        h_Buffer,
        h_Input,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    printf("...running convolutionColumnsCPU()\n");
    convolutionColumnsCPU(
        h_OutputCPU,
        h_Buffer,
        h_Kernel,
        imageW,
        imageH,
        KERNEL_RADIUS
    );

    double delta = 0;
    double sum = 0;

    for (unsigned int i = 0; i < imageW * imageH; i++)
    {
        sum += h_OutputCPU[i] * h_OutputCPU[i];
        delta += (h_OutputGPU[i] - h_OutputCPU[i]) * (h_OutputGPU[i] - h_OutputCPU[i]);
    }

    double L2norm = sqrt(delta / sum);
    printf("Relative L2 norm: %E\n", L2norm);
    printf("Shutting down...\n");

    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFreeArray(a_Src));
    free(h_OutputGPU);
    free(h_Buffer);
    free(h_Input);
    free(h_Kernel);

    sdkDeleteTimer(&hTimer);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

    if (L2norm > 1e-6)
    {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
