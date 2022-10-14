/**
 * mix_kernels_cuda_ro.cu: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <math_constants.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"

#define ELEMENTS_PER_THREAD (8)
#define FUSION_DEGREE (4)

BenchType datatype = BENCH_FLOAT;
int secs = 5;

template <class T, int blockdim, unsigned int granularity, unsigned int fusion_degree>
__global__ void benchmark_func(T seed, T *g_data, int compute_iterations){
	const unsigned int blockSize = blockdim;
	const int stride = blockSize;
	int idx = blockIdx.x*blockSize*granularity + threadIdx.x;
	const int big_stride = gridDim.x*blockSize*granularity;

	T tmps[granularity];
	for(int k=0; k<fusion_degree; k++){
		#pragma unroll
		for(int j=0; j<granularity; j++){
			// Load elements (memory intensive part)
			tmps[j] = g_data[idx+j*stride+k*big_stride];
			// Perform computations (compute intensive part)
			for(int i=0; i<compute_iterations; i++){
				tmps[j] = tmps[j]*tmps[j]+seed;//tmps[(j+granularity/2)%granularity];
			}
		}
		// Multiply add reduction
		T sum = (T)0;
		#pragma unroll
		for(int j=0; j<granularity; j+=2)
			sum += tmps[j]*tmps[j+1];
		// Dummy code
		if( sum==(T)-1 ) // Designed so it never executes
			g_data[idx+k*big_stride] = sum;
	}
}

void initializeEvents(cudaEvent_t *start, cudaEvent_t *stop){
	CUDA_SAFE_CALL( cudaEventCreate(start) );
	CUDA_SAFE_CALL( cudaEventCreate(stop) );
	CUDA_SAFE_CALL( cudaEventRecord(*start, 0) );
}

float finalizeEvents(cudaEvent_t start, cudaEvent_t stop){
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaEventRecord(stop, 0) );
	CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	float kernel_time;
	CUDA_SAFE_CALL( cudaEventElapsedTime(&kernel_time, start, stop) );
	CUDA_SAFE_CALL( cudaEventDestroy(start) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop) );
	return kernel_time;
}

void runbench_warmup(double *cd, long size){
	const long reduced_grid_size = size/(ELEMENTS_PER_THREAD)/128;
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< short, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE ><<< dimReducedGrid, dimBlock >>>((short)1, (short*)cd, 0);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

int out_config = 1;

void runbench(double *cd, long size, int compute_iterations){
	const long compute_grid_size = size/ELEMENTS_PER_THREAD/FUSION_DEGREE;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = (ELEMENTS_PER_THREAD*(long long)compute_grid_size+(2*ELEMENTS_PER_THREAD*compute_iterations)*(long long)compute_grid_size)*FUSION_DEGREE;
	const long long memoryoperations = size;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;

    float kernel_time_mad = 0.0;
    unsigned int size_of_data = sizeof(float);
    switch (datatype)
    {
        default:
        case BENCH_FLOAT:{
	        initializeEvents(&start, &stop);
	        benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE><<< dimGrid, dimBlock >>>(1.0f, (float*)cd, compute_iterations);
	        kernel_time_mad = finalizeEvents(start, stop);
            size_of_data = sizeof(float);
            break;
        }

        case BENCH_DOUBLE:{
	        initializeEvents(&start, &stop);
	        benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE><<< dimGrid, dimBlock >>>(1.0, cd, compute_iterations);
	        kernel_time_mad = finalizeEvents(start, stop);
            size_of_data = sizeof(double);
            break;
        }

        case BENCH_INT:{
	        initializeEvents(&start, &stop);
	        benchmark_func< int, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE><<< dimGrid, dimBlock >>>(1, (int*)cd, compute_iterations);
	        kernel_time_mad = finalizeEvents(start, stop);
            size_of_data = sizeof(int);
            break;
        }
    }


	// printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  %8.3f,%8.2f,%8.2f,%7.2f\n",
	// 	compute_iterations,
	// 	((double)computations)/((double)memoryoperations*sizeof(float)),
	// 	kernel_time_mad_sp,
	// 	((double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
	// 	((double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
	// 	((double)computations)/((double)memoryoperations*sizeof(double)),
	// 	kernel_time_mad_dp,
	// 	((double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
	// 	((double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
	// 	((double)computations)/((double)memoryoperations*sizeof(int)),
	// 	kernel_time_mad_int,
	// 	((double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
	// 	((double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );
	printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f \n",
		compute_iterations,
		((double)computations)/((double)memoryoperations*sizeof(float)),
		kernel_time_mad,
	    ((double)computations)/kernel_time_mad*1000./(double)(1000*1000*1000),
	    ((double)memoryoperations*size_of_data)/kernel_time_mad*1000./(1000.*1000.*1000.));

    // collect power data
    int iters = int((double)secs * 1000 / kernel_time_mad);
    printf("Adjust %d iterations to achieve time duration %d.\n", iters, secs);

    for (int i = 0 ; i < iters; i++)
        switch (datatype)
        {
            default:
            case BENCH_FLOAT:{
	            benchmark_func< float, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE><<< dimGrid, dimBlock >>>(1.0f, (float*)cd, compute_iterations);
                break;
            }

            case BENCH_DOUBLE:{
	            benchmark_func< double, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE><<< dimGrid, dimBlock >>>(1.0, cd, compute_iterations);
                break;
            }

            case BENCH_INT:{
	            benchmark_func< int, BLOCK_SIZE, ELEMENTS_PER_THREAD, FUSION_DEGREE><<< dimGrid, dimBlock >>>(1, (int*)cd, compute_iterations);
                break;
            }
        }

}

extern "C" void mixbenchGPU(double *c, long size, int compute_iterations, BenchType bt, int s){
	const char *benchtype = "compute with global memory (block strided)";

	printf("Trade-off type:       %s\n", benchtype);
	printf("Elements per thread:  %d\n", ELEMENTS_PER_THREAD);
	printf("Thread fusion degree: %d\n", FUSION_DEGREE);
	double *cd;
    datatype = bt;
    secs = s;

	CUDA_SAFE_CALL( cudaMalloc((void**)&cd, size*sizeof(double)) );

	// Copy data to device memory
	CUDA_SAFE_CALL( cudaMemset(cd, 0, size*sizeof(double)) );  // initialize to zeros

	// Synchronize in order to wait for memory operations to finish
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	printf("---------------------------------------------------------- CSV data ----------------------------------------------------------\n");
	// printf("Experiment ID, Single Precision ops,,,,              Double precision ops,,,,              Integer operations,,, \n");
	// printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec, Flops/byte, ex.time,  GFLOPS, GB/sec, Iops/byte, ex.time,   GIOPS, GB/sec\n");
    if (datatype == BENCH_FLOAT)
	    printf("Experiment ID, Single Precision ops,,,,              \n");
    if (datatype == BENCH_DOUBLE)
	    printf("Experiment ID, Double Precision ops,,,,              \n");
    if (datatype == BENCH_INT)
	    printf("Experiment ID, Int Precision ops,,,,              \n");
	printf("Compute iters, Flops/byte, ex.time,  GFLOPS, GB/sec  \n");

	// runbench_warmup(cd, size);

	runbench(cd, size, compute_iterations); // 0~256

	printf("------------------------------------------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
