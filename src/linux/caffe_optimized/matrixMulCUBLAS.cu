// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include <math.h>
#include <ctime>


#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

// Coalesced transpose with no bank conflicts

#define TILE_DIM    16
#define BLOCK_ROWS  16
static cudaDeviceProp g_devProp;
__global__ void transGPU(float *dmt, float *dm, int w, int h)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int old_idx = row*w + col;
	int new_idx = col*h + row;
	if(row < h && col < w){ 
		tile[threadIdx.y][threadIdx.x] = dm[old_idx];
		__syncthreads();
		dmt[new_idx] = tile[threadIdx.y][threadIdx.x];
	}   

}

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
	__shared__ float tile[TILE_DIM][TILE_DIM+1];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex)*width;

	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex)*height;

	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
	{
		tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
	}

	__syncthreads();

	for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
	{
		odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
	}
}



typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
	size_t uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
	void
matrixMulCPU(float *C, const float *A, const float *B, size_t hA, size_t wA, size_t wB)
{
	for (size_t i = 0; i < hA; ++i)
		for (size_t j = 0; j < wB; ++j)
		{
			double sum = 0;
			for (size_t k = 0; k < wA; ++k)
			{
				double a = A[i * wA + k];
				//double b = B[k * wB + j];
				double b = B[j * wA + k];
				sum += a * b;
			}
			C[i * wB + j] = (float)sum;
		}
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B^T
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wA         width of matrix A
//! @param hB         height of matrix B, wB = wA
////////////////////////////////////////////////////////////////////////////////
	void
matrixMulNTCPU(float *C, const float *A, const float *B, size_t hA, size_t wA, size_t hB)
{
	for (size_t i = 0; i < hA; ++i)
		for (size_t j = 0; j < hB; ++j)
		{
			double sum = 0;
			for (size_t k = 0; k < wA; ++k)
			{
				double a = A[i * wA + k];
				double b = B[j * wA + k];
				sum += a * b;
			}
			C[i * hB + j] = (float)sum;
		}
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
	printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
	int i,j,k;
	int error_count=0;

	for (j = 0; j < height; j++)
	{
		if (error_count < iListLength)
		{
			printf("\n  Row %d:\n", j);
		}

		for (i = 0; i < width; i++)
		{
			k = j * width + i;
			float fDiff = fabs(data1[k] - data2[k]);

			if (fDiff > fListTol)
			{
				if (error_count < iListLength)
				{
					printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
				}

				error_count++;
			}
		}
	}

	printf(" \n  Total Errors = %d\n", error_count);
}

void getDeviceProp(cudaDeviceProp &deviceProp, int devID=0) {
    cudaGetDeviceProperties(&deviceProp, devID);
    // deviceProp.major: major capability
    // deviceProp.minor: minor capability
    // deviceProp.totalGlobalMem: total global memory in bytes 
    // deviceProp.multiProcessorCount
    // deviceProp.clockRate
    // deviceProp.memoryClockRate
    // deviceProp.memoryBusWidth
    // deviceProp.l2CacheSize
    // deviceProp.sharedMemPerBlock
    // deviceProp.totalConstMem
    // deviceProp.regsPerBlock
}

void generateFeatureVector(size_t M, size_t N, size_t K, cudaDeviceProp &deviceProp)
{
    printf("%d", deviceProp.major);
    printf(",");
    printf("%d", deviceProp.minor);
    printf(",");
    printf("%ld", deviceProp.totalGlobalMem);
    printf(",");
    printf("%d", deviceProp.multiProcessorCount);
    printf(",");
    printf("%d", deviceProp.clockRate);
    printf(",");
    printf("%d", deviceProp.memoryClockRate);
    printf(",");
    printf("%d", deviceProp.memoryBusWidth);
    printf(",");
    printf("%ld", deviceProp.l2CacheSize);
    printf(",");
    printf("%d", deviceProp.sharedMemPerBlock);
    printf(",");
    printf("%d", deviceProp.totalConstMem);
    printf(",");
    printf("%d", deviceProp.regsPerBlock);
    printf(",");
    printf("%ld,%ld,%ld,", M, N, K);
}

void initializeCUDA(int argc, char **argv, int devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
	cudaError_t error;
	int hA = 64, hB = 128, w = 1627;
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		error = cudaSetDevice(devID);
		if (error != cudaSuccess)
		{
			printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
			exit(EXIT_FAILURE);
		}
	}

	if (
			checkCmdLineFlag(argc, (const char **)argv, "hA") 
			&& checkCmdLineFlag(argc, (const char **)argv, "hB") 
			&& checkCmdLineFlag(argc, (const char **)argv, "w")
	   )
	{
		hA = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
		hB = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
		w = getCmdLineArgumentInt(argc, (const char **)argv, "w");
	}


	// get number of SMs on this GPU
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}


	if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
	{
		iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
	}

	iSizeMultiple = min(iSizeMultiple, 10);
	iSizeMultiple = max(iSizeMultiple, 1);

	cudaDeviceProp deviceProp;

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}

	printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", 
			devID, deviceProp.name, deviceProp.major, deviceProp.minor);

	matrix_size.uiWA = w; 
	matrix_size.uiHA = hA;
	matrix_size.uiWB = w; 
	matrix_size.uiHB = hB;
	matrix_size.uiWC = hB;
	matrix_size.uiHC = hA;
	printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
			matrix_size.uiHA, matrix_size.uiWA,
			matrix_size.uiHB, matrix_size.uiWB,
			matrix_size.uiHC, matrix_size.uiWC);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
	cudaDeviceProp deviceProp;

	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	// set seed for rand()
	srand(2006);

	// allocate host memory for matrices A and B
	size_t size_A = matrix_size.uiWA * matrix_size.uiHA;
	size_t mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	size_t size_B = matrix_size.uiWB * matrix_size.uiHB;
	size_t mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// set seed for rand()
	srand(2006);

	// initialize host memory
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	// allocate device memory
	float *d_A, *d_B, *d_BT, *d_C;
	size_t size_C = matrix_size.uiWC * matrix_size.uiHC;
	size_t mem_size_C = sizeof(float) * size_C;

	// allocate host memory for the result
	float *h_C      = (float *) malloc(mem_size_C);
	float *h_CUBLAS = (float *) malloc(mem_size_C);

	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
	checkCudaErrors(cudaMalloc((void **) &d_BT, mem_size_B));
	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

	int size_x, size_y;
	size_x = matrix_size.uiWB;
	size_y = matrix_size.uiHB;
	dim3 gridt((size_x-1)/TILE_DIM + 1, (size_y-1)/TILE_DIM + 1), threadst(TILE_DIM,BLOCK_ROWS);

	// setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid((matrix_size.uiWC - 1) / threads.x + 1, (matrix_size.uiHC - 1) / threads.y + 1);

	// create and start timer
	printf("Computing result using CUBLAS...");

	// execute the kernel
	int nIter = 1;

	// CUBLAS version 2.0
	{
		const float alpha = 1.0f;
		const float beta  = 0.0f;
		cublasHandle_t handle;
		cudaEvent_t start, stop;

		checkCudaErrors(cublasCreate(&handle));

		// Allocate CUDA events that we'll use for timing
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));
		int trans = 1;
		if (checkCmdLineFlag(argc, (const char **)argv, "transpose"))
		{
			trans = getCmdLineArgumentInt(argc, (const char **)argv, "transpose");
		}
		// Record the start event
		checkCudaErrors(cudaEventRecord(start, NULL));
		for (int j = 0; j < nIter; j++)
		{
			if(trans)
			{

				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
							matrix_size.uiHB, matrix_size.uiHA, matrix_size.uiWB,  
							&alpha, 
							d_B, matrix_size.uiWB, 
							d_A, matrix_size.uiWB, 
							&beta, 
							d_C, matrix_size.uiWC));
			}
			else	
			{
				int m = matrix_size.uiHB, n = matrix_size.uiHA, k = matrix_size.uiWA;
				int ldbt = matrix_size.uiHB, lda = matrix_size.uiWA, ldc = matrix_size.uiWC;
				/* for debug only
				   printf("transGPU param: %d %d\n", matrix_size.uiHB, matrix_size.uiWB);
				   printf("cublasSgemm param: N, N, %d, %d, %d, 1, *d_BT, %d, *d_A, %d 0, *d_C, %d\n" ,
				   m, n, k,
				   ldbt, 
				   lda,
				   ldc);
				 */
//				transGPU<<<gridt, threadst>>>(d_BT, d_B, matrix_size.uiWB, matrix_size.uiHB);
//				checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
//							matrix_size.uiHB, matrix_size.uiWB, 
//							&alpha, d_B, matrix_size.uiWB, 
//							&beta, NULL, matrix_size.uiWB,
//							d_BT, matrix_size.uiHB));
//				checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//							m, n, k, 
//							&alpha, 
//							d_BT, ldbt, 
//							d_A, lda, 
//							&beta, 
//							d_C, ldc));
                checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
							m, n, k, 
							&alpha, 
							d_B, ldbt, 
							d_A, lda, 
							&beta, 
							d_C, ldc));


			}
		}

		printf("done.\n");

		// Record the stop event
		checkCudaErrors(cudaEventRecord(stop, NULL));

		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		// Compute and print the performance
		float msecPerMatrixMul = msecTotal / nIter;
		double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiWB;
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
		printf(
				"Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
				gigaFlops,
				msecPerMatrixMul,
				flopsPerMatrixMul);

		// copy result from device to host
		checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost));

		// Destroy the handle
		checkCudaErrors(cublasDestroy(handle));
	}
	int compare = 1;
	bool resCUBLAS = true;
	if (checkCmdLineFlag(argc, (const char **)argv, "compare"))
	{
		compare = getCmdLineArgumentInt(argc, (const char **)argv, "compare");
	}
	if (compare)
	{
		// compute reference solution
		printf("Computing result using host CPU...");
		float *reference = (float *)malloc(mem_size_C);
		printf("\nHA: %ld, WA: %ld, WB: %ld\n", matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiHB);
		matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiHB);
		printf("done.\n");

		// check result (CUBLAS)
		resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);

		if (resCUBLAS != true)
		{
			printDiff(reference, h_CUBLAS, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
		}
		printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
		free(reference);
	}
	// clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

	if (resCUBLAS == true)
	{
		return EXIT_SUCCESS;    // return value = 1
	}
	else
	{
		return EXIT_FAILURE;     // return value = 0
	}
}


/**
  * nWB = nWA
**/
double bmMatrixMatTranspose(size_t nWA, size_t nHA, size_t nHB, bool bUseTranspose=false)
{

    // M = hA, N = hB, K = wA
    // wA = wB
	size_t size_A = nWA * nHA; 
	size_t mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	size_t size_B = nWA * nHB;//matrix_size.uiWB * matrix_size.uiHB;
	size_t mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// set seed for rand()
	srand(2006);
	// initialize host memory
	randomInit(h_A, size_A);
	randomInit(h_B, size_B);

	size_t size_C = nHB * nHA; 
	size_t mem_size_C = sizeof(float) * size_C;
	float *h_C      = (float *) malloc(mem_size_C);

	float *d_A, *d_B, *d_BT, *d_C;
	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));

    if (bUseTranspose) {
        checkCudaErrors(cudaMalloc((void **) &d_BT, mem_size_B));
    }

	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

    const float alpha = 1.0f;
	const float beta  = 0.0f;
	cublasHandle_t handle;
	cudaEvent_t start, stop;
    checkCudaErrors(cublasCreate(&handle));
    // Record the start event
	size_t maxValue = (65536/nHA) * (65536/nHB) * (65536/nWA);
	size_t nIter = 5; // 1 is for warming 
	printf("nIter:%ld\n", nIter);
    clock_t startTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	double totalTime = 0.0;
    //for (size_t j = 0; j < nIter; j++)
	size_t j = 0;
	while (j < nIter)
    {
        //startTime = clock();
		checkCudaErrors(cudaEventRecord(start, NULL));
        if (bUseTranspose) {
            checkCudaErrors(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                        nHB, nWA, 
                        &alpha, d_B, nWA, 
                        &beta, NULL, nWA,
                        d_BT, nHB));
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                        nHB, nHA, nWA, 
                        &alpha, 
                        d_BT, nHB, 
                        d_A, nWA, 
                        &beta, 
                        d_C, nHB));
        } else {
            checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                        nHB, nHA, nWA,  
                        &alpha, 
                        d_B, nWA, 
                        d_A, nWA, 
                        &beta, 
                        d_C, nHB));
        }
		checkCudaErrors(cudaEventRecord(stop, NULL));
		// Wait for the stop event to complete
		checkCudaErrors(cudaEventSynchronize(stop));
		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
		if (j >= 1) {
			totalTime += msecTotal;
		}
		// else {
		// 	nIter = secs / (msecTotal * 0.001);
		// }
		j++;
    }
	checkCudaErrors(cublasDestroy(handle));
    clock_t endTime = clock();

    

    ;

    // Compare
	//checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
	//float *reference = (float *)malloc(mem_size_C);
	//printf("\nHA: %d, WA: %d, HB: %d\n", nHA, nWA, nHB);
	//matrixMulNTCPU(reference, h_A, h_B, nHA, nWA, nHB);
    //bool resCUBLAS = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
    //if (!resCUBLAS) {
    //    printDiff(reference, h_C, nHA, nHB, 100, 1.0e-5f);
    //}
    //printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
    //float msecTotal = (endTime - startTime) * 1000.0f/(double) CLOCKS_PER_SEC;
    float msecPerMatrixMul = totalTime / (nIter-1);
    double flopsPerMatrixMul = 2.0 * (double)nWA * (double)nHA * (double)nHB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf(
            "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
            gigaFlops,
            msecPerMatrixMul,
			flopsPerMatrixMul);
	free(h_A);
	free(h_B);
	free(h_C);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
    if (bUseTranspose) {
	    checkCudaErrors(cudaFree(d_BT));
    }
	checkCudaErrors(cudaFree(d_C));
	cudaDeviceReset();

    return gigaFlops;
}


bool checkIfEnoughMemory(size_t nWA, size_t nHA, unsigned nHB, bool bUseTranspose)
{
    size_t size_A = nWA * nHA; 
	size_t mem_size_A = sizeof(float) * size_A;
	size_t size_B = nWA * nHB;
	size_t mem_size_B = sizeof(float) * size_B;
	size_t size_C = nHB * nHA; 
	size_t mem_size_C = sizeof(float) * size_C;
    size_t nTotalMemNeeded = mem_size_A + mem_size_B + mem_size_C;
    if (bUseTranspose) {
        nTotalMemNeeded += mem_size_B;
    }
    size_t availableMemory, totalMemory;
    cudaMemGetInfo(&availableMemory, &totalMemory);
    //if (mem_size_A >= 4096l*1024*1024 || mem_size_B >= 4096l*1024*1024 || mem_size_C >= 4096l*1024*1024) {
    //    printf("No enough memory!\n");
    //    return false;
    //}
    return availableMemory > nTotalMemNeeded;
}

void validateSVM()
{
    int nNumAttr = 3;

    size_t nNumTrue = 0;
    size_t nNumFalse = 0;
    size_t nMinMatrixW = 128; 
    size_t nMaxMatrixW = 65536;
    size_t nStep = 24;
    //size_t nMaxMatrixW = 128;//nMinMatrixW;
    for (int i = nMinMatrixW; i <= nMaxMatrixW; i *= 2) {
        for (int j = nMinMatrixW; j <= nMaxMatrixW; j *= 2) {
            for (int k = nMinMatrixW; k <= nMaxMatrixW; k *= 2) {
                //size_t nWA = 8192, nHA = 8192, nHB = 8192;
                try {
                    size_t nWA = i, nHA = j, nHB = k;
                    //setSVMNodeFeature(x, nWA, nHA, nHB);
                    double label = 1;//predict(model, x);
                    //printf("M:%d, N: %d, K:%d:[%.1f]\n", nHA, nHB, nWA, label);
                    if (!checkIfEnoughMemory(nWA, nHA, nHB, true)) {
                        continue;
                    }
                    double naiveThroughput = bmMatrixMatTranspose(nWA, nHA, nHB);
					double forceThroughput = 0;// bmMatrixMatTranspose(nWA, nHA, nHB, true);
                    generateFeatureVector(nHA, nHB, nWA, g_devProp);
                    printf("%.3f,%.3f,%.3f\n", naiveThroughput, forceThroughput, naiveThroughput-forceThroughput);
                    //printf("%ld,%ld,%ld,%.3f,%.3f,%.3f\n", nHA, nHB, nWA, naiveThroughput, forceThroughput, naiveThroughput-forceThroughput);
                    if (label * (naiveThroughput - forceThroughput) > 0) {
                        // true 
                        //printf("+\n");
                        nNumTrue++;
                    } else {
                        // false 
                        nNumFalse++;
                        //printf("-\n");
                    }
                    //printf("Current Precision: %.3f\n", nNumTrue*1.0/(nNumTrue+nNumFalse));
                } catch (const std::exception &e) {
                    continue;
                }
            }
        }
    }
    //printf("Total: %d, True: %d, False: %d\n", nNumTrue+nNumFalse, nNumTrue, nNumFalse);
    //printf("Precision: %.3f\n", nNumTrue*1.0/(nNumTrue+nNumFalse));
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    
    //getDeviceProp(g_devProp);
    //validateSVM();
    //return 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "h"))
	{
		printf("This program calculates matrix multiplication A*transpose(B)\nYou can use cublas with transpose param or launch a transpose kernel before mutiplication.\nNote that if the matrix is too large, the CPU will take painfully long time to calculate the reference answer matrix.\nThis program is modified version of CUDA toolkit sample matrixMulCUBLAS\n\n");
		printf("Arguments:\n \t-device=YOUR_GPU_NUM\n{\n \t-hA=HIGH_OF_A\n \t-hB=HIGHT_OF_B\n \t-w=WIDTH\n} These parameters must be set together\n \t-transpose=0|1\tuse transpose kernel or not\n\t-compare=0|1 need compare with CPU or not\n");
		exit(EXIT_SUCCESS);
	}
	printf("[Matrix Multiply CUBLAS] - Starting...\n");
	int devID = 0, sizeMult = 5;
	sMatrixSize matrix_size;

	initializeCUDA(argc, argv, devID, sizeMult, matrix_size);
    getDeviceProp(g_devProp, devID);
	int isTranspose = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "transpose")) {
        isTranspose = getCmdLineArgumentInt(argc, (const char **)argv, "transpose");
    }
	bmMatrixMatTranspose(matrix_size.uiWA, matrix_size.uiHA, matrix_size.uiHB, isTranspose);

	return 0; 
}
