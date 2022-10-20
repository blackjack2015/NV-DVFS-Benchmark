/**
 * mix_kernels_cuda.cu: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <math_constants.h>
#include "lcutil.h"
#include "mix_kernels_cuda.h"

#define COMP_ITERATIONS (8192)
#define UNROLL_ITERATIONS (32)
#define REGBLOCK_SIZE (8)

#define UNROLLED_MEMORY_ACCESSES (UNROLL_ITERATIONS/2)

BenchType datatype = BENCH_FLOAT;
int secs = 5;

template <class T, int blockdim>
__global__ void benchmark_func(T seed, volatile T *g_data, int memory_ratio){
	const int index_stride = blockdim;
	const int index_base = blockIdx.x*blockdim*UNROLLED_MEMORY_ACCESSES + threadIdx.x;

	const int halfarraysize = gridDim.x*blockdim*UNROLLED_MEMORY_ACCESSES;
	const int offset_slips = 1+UNROLLED_MEMORY_ACCESSES-((memory_ratio+1)/2);
	const int array_index_bound = index_base+offset_slips*index_stride;
	const int initial_index_range = memory_ratio>0 ? UNROLLED_MEMORY_ACCESSES % ((memory_ratio+1)/2) : 1;
	int initial_index_factor = 0;
	volatile T *data = g_data;

	int array_index = index_base;
	T r0 = seed + blockIdx.x * blockdim + threadIdx.x,
	  r1 = r0+(T)(2),
	  r2 = r0+(T)(3),
	  r3 = r0+(T)(5),
	  r4 = r0+(T)(7),
	  r5 = r0+(T)(11),
	  r6 = r0+(T)(13),
	  r7 = r0+(T)(17);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS-memory_ratio; i++){
			r0 = r0 * r0 + r4;
			r1 = r1 * r1 + r5;
			r2 = r2 * r2 + r6;
			r3 = r3 * r3 + r7;
			r4 = r4 * r4 + r0;
			r5 = r5 * r5 + r1;
			r6 = r6 * r6 + r2;
			r7 = r7 * r7 + r3;
		}
		bool do_write = true;
		int reg_idx = 0;
		#pragma unroll
		for(int i=UNROLL_ITERATIONS-memory_ratio; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to one memory operation
			T& r = reg_idx==0 ? r0 : (reg_idx==1 ? r1 : (reg_idx==2 ? r2 : (reg_idx==3 ? r3 : (reg_idx==4 ? r4 : (reg_idx==5 ? r5 : (reg_idx==6 ? r6 : r7))))));
			if( do_write )
				data[ array_index+halfarraysize ] = r;
			else {
				r = data[ array_index ];
				if( ++reg_idx>=REGBLOCK_SIZE )
					reg_idx = 0;
				array_index += index_stride;
			}
			do_write = !do_write;
		}
		if( array_index >= array_index_bound ){
			if( ++initial_index_factor > initial_index_range)
				initial_index_factor = 0;
			array_index = index_base + initial_index_factor*index_stride;
		}
	}
	if( (r0==(T)CUDART_INF) && (r1==(T)CUDART_INF) && (r2==(T)CUDART_INF) && (r3==(T)CUDART_INF) &&
	    (r4==(T)CUDART_INF) && (r5==(T)CUDART_INF) && (r6==(T)CUDART_INF) && (r7==(T)CUDART_INF) ){ // extremely unlikely to happen
		g_data[0] = r0+r1+r2+r3+r4+r5+r6+r7;
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
	const long reduced_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/32;
	const int BLOCK_SIZE = 256;
	const int TOTAL_REDUCED_BLOCKS = reduced_grid_size/BLOCK_SIZE;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	benchmark_func< short, BLOCK_SIZE><<< dimReducedGrid, dimBlock >>>((short)1, (short*)cd, 0);
	CUDA_SAFE_CALL( cudaGetLastError() );
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
}

void runbench(double *cd, long size, int memory_ratio){
	if( memory_ratio>UNROLL_ITERATIONS ){
		fprintf(stderr, "ERROR: memory_ratio exceeds UNROLL_ITERATIONS\n");
		exit(1);
	}
		
	const long compute_grid_size = size/(UNROLLED_MEMORY_ACCESSES)/2;
	const int BLOCK_SIZE = 256;
	const int TOTAL_BLOCKS = compute_grid_size/BLOCK_SIZE;
	const long long computations = 2*(long long)(COMP_ITERATIONS)*REGBLOCK_SIZE*compute_grid_size;
	const long long memoryoperations = (long long)(COMP_ITERATIONS)*compute_grid_size;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
    dim3 dimGrid(TOTAL_BLOCKS, 1, 1);
	cudaEvent_t start, stop;

    float kernel_time_mad;
    unsigned int size_of_data = sizeof(float);
    switch (datatype)
    {
        default:
        case BENCH_FLOAT:{
	        initializeEvents(&start, &stop);
	        benchmark_func< float, BLOCK_SIZE><<< dimGrid, dimBlock >>>(1.0f, (float*)cd, memory_ratio);
	        kernel_time_mad = finalizeEvents(start, stop);
            size_of_data = sizeof(float);
            break;
        }

        case BENCH_DOUBLE:{
	        initializeEvents(&start, &stop);
	        benchmark_func< double, BLOCK_SIZE><<< dimGrid, dimBlock >>>(1.0, cd, memory_ratio);
	        kernel_time_mad = finalizeEvents(start, stop);
            size_of_data = sizeof(double);
            break;
        }

        case BENCH_INT:{
	        initializeEvents(&start, &stop);
	        benchmark_func< int, BLOCK_SIZE><<< dimGrid, dimBlock >>>(1, (int*)cd, memory_ratio);
	        kernel_time_mad = finalizeEvents(start, stop);
            size_of_data = sizeof(int);
            break;
        }
    }

	const double memaccesses_ratio = (double)(memory_ratio)/UNROLL_ITERATIONS;
	const double computations_ratio = 1.0-memaccesses_ratio;

	// printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f,   %8.3f,%8.2f,%8.2f,%7.2f,  %8.3f,%8.2f,%8.2f,%7.2f\n",
	// 	UNROLL_ITERATIONS-memory_ratio,
	// 	(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(float)),
	// 	kernel_time_mad_sp,
	// 	(computations_ratio*(double)computations)/kernel_time_mad_sp*1000./(double)(1000*1000*1000),
	// 	(memaccesses_ratio*(double)memoryoperations*sizeof(float))/kernel_time_mad_sp*1000./(1000.*1000.*1000.),
	// 	(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(double)),
	// 	kernel_time_mad_dp,
	// 	(computations_ratio*(double)computations)/kernel_time_mad_dp*1000./(double)(1000*1000*1000),
	// 	(memaccesses_ratio*(double)memoryoperations*sizeof(double))/kernel_time_mad_dp*1000./(1000.*1000.*1000.),
	// 	(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(int)),
	// 	kernel_time_mad_int,
	// 	(computations_ratio*(double)computations)/kernel_time_mad_int*1000./(double)(1000*1000*1000),
	// 	(memaccesses_ratio*(double)memoryoperations*sizeof(int))/kernel_time_mad_int*1000./(1000.*1000.*1000.) );
	printf("         %4d,   %8.3f,%8.2f,%8.2f,%7.2f \n",
		UNROLL_ITERATIONS-memory_ratio,
		(computations_ratio*(double)computations)/(memaccesses_ratio*(double)memoryoperations*sizeof(float)),
		kernel_time_mad,
	    (computations_ratio*(double)computations)/kernel_time_mad*1000./(double)(1000*1000*1000),
	    (memaccesses_ratio*(double)memoryoperations*size_of_data)/kernel_time_mad*1000./(1000.*1000.*1000.));

    // collect power data
    int iters = int((double)secs * 1000 / kernel_time_mad);
    printf("Adjust %d iterations to achieve time duration %d.\n", iters, secs);

	initializeEvents(&start, &stop);
    for (int i = 0 ; i < iters; i++)
        switch (datatype)
        {
            default:
            case BENCH_FLOAT:{
	            benchmark_func< float, BLOCK_SIZE><<< dimGrid, dimBlock >>>(1.0f, (float*)cd, memory_ratio);
                break;
            }

            case BENCH_DOUBLE:{
	            benchmark_func< double, BLOCK_SIZE><<< dimGrid, dimBlock >>>(1.0, cd, memory_ratio);
                break;
            }

            case BENCH_INT:{
	            benchmark_func< int, BLOCK_SIZE><<< dimGrid, dimBlock >>>(1, (int*)cd, memory_ratio);
                break;
            }
        }
    
    float avg_msec = finalizeEvents(start, stop) / iters;
    printf("benchmark_func() iterated %d, average time is %f msec\n", iters, avg_msec);
}

extern "C" void mixbenchGPU(double *c, long size, int memory_ratio, BenchType bt, int s){
	const char *benchtype = "compute with global memory (block strided)";

	printf("Trade-off type:       %s\n", benchtype);
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

	runbench_warmup(cd, size);

	runbench(cd, size, memory_ratio);

	printf("------------------------------------------------------------------------------------------------------------------------------\n");

	// Copy results back to host memory
	CUDA_SAFE_CALL( cudaMemcpy(c, cd, size*sizeof(double), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree(cd) );

	CUDA_SAFE_CALL( cudaDeviceReset() );
}
