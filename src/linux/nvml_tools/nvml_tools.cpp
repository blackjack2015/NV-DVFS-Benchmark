// System includes
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <fstream>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <nvml.h>

using namespace std;

const char * convertToComputeModeString(nvmlComputeMode_t mode)
{
	switch (mode)
	{
	case NVML_COMPUTEMODE_DEFAULT:
		return "Default";

	case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
		return "Exclusive_Thread";

	case NVML_COMPUTEMODE_PROHIBITED:
		return "Prohibited";

	case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
		return "Exclusive Process";

	default:
		return "Unknown";

	}

}



int main(int argc, char **argv)
{
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;
	int sampleInterval = 200;
	char* outputF = new char[150];

	for (int i = 0; i < 150; i++)
		outputF[i] = '\0';

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		cudaSetDevice(devID);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "output"))
	{
		getCmdLineArgumentString(argc, (const char **)argv, "output", &outputF);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "si"))
	{
		sampleInterval = getCmdLineArgumentInt(argc, (const char **)argv, "si");
	}

	printf(outputF);
	ofstream ofile;
	ofile.open(outputF);

	nvmlReturn_t result;
	//unsigned int device_count, i;

	// First initialize NVML library
	result = nvmlInit();

	if (NVML_SUCCESS != result)
	{
		printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
		printf("Press ENTER to continue...\n");
		getchar();
		return 1;
	}

	//result = nvmlDeviceGetCount(&device_count);

	//if (NVML_SUCCESS != result)
	//{
	//	printf("Failed to query device count: %s\n", nvmlErrorString(result));
	//}

	//printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");
	//printf("Listing devices:\n");

	nvmlDevice_t device;
	char name[64];

	result = nvmlDeviceGetHandleByIndex(devID, &device);
	if (NVML_SUCCESS != result)
	{
		printf("Failed to get handle for device %i: %s\n", devID, nvmlErrorString(result));
	}

	result = nvmlDeviceGetName(device, name, sizeof(name) / sizeof(name[0]));
	if (NVML_SUCCESS != result)
	{
		printf("Failed to get name of device %i: %s\n", devID, nvmlErrorString(result));
	}
	printf("Device %i: is %s\n", devID, name);
	ofile << "Device " << devID << ": is " << name << '\n';

	clock_t start = 0;
	clock_t now;
	double msec = 0.0;

	printf("Time-Stamp\tP-state\tCore-F(MHz)\tMem-F(MHz)\tPower(mW)\t\n");
	ofile << "Time-Stamp\tP-state\tCore-F(MHz)\tMem-F(MHz)\tPower(mW)\t\n";
	while (1)
	{
		now = clock();
		printf("%u-%u-%u\n", start, now, CLOCKS_PER_SEC);
		msec = double(now - start);
		printf("%.1f ms\t", msec);
		ofile << msec << " ms\t";
		start = now;

		//get pState of GPU
		nvmlPstates_t pState;
		result = nvmlDeviceGetPowerState(device, &pState);
		if (NVML_ERROR_NOT_SUPPORTED == result)
			printf("\t pState is not CUDA capable device\n");
		else if (NVML_SUCCESS != result)
		{
			printf("Failed to get PState Information for device %i: %s\n", devID, nvmlErrorString(result));
		}
		printf("%d\t", pState);
		ofile << pState << "\t";

		//get SM clock and Memory clock of GPU
		unsigned int sm_clock, memory_clock;
		unsigned int power;
		result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock);
		result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &memory_clock);
		if (NVML_ERROR_NOT_SUPPORTED == result)
			printf("\t Clock Info is not CUDA capable device\n");
		else if (NVML_SUCCESS != result)
		{
			printf("Failed to get clock Information for device %i: %s\n", devID, nvmlErrorString(result));
		}
		printf("%u\t%u\t", sm_clock, memory_clock);
		ofile << sm_clock << "\t" << memory_clock << "\t";

		result = nvmlDeviceGetPowerUsage(device, &power);
		if (NVML_ERROR_NOT_SUPPORTED == result)
			printf("\t Power Info is not CUDA capable device\n");
		else if (NVML_SUCCESS != result)
		{
			printf("Failed to get power Information for device %i: %s\n", devID, nvmlErrorString(result));
		}
		printf("%u\t\n", power);
		ofile << power << "\t" << endl;

		// Sleep(sampleInterval);
		usleep(sampleInterval*1000.0);
	}

	result = nvmlShutdown();

	if (NVML_SUCCESS != result)
		printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

	printf("All done.\n");
	//printf("Press ENTER to continue...\n");
	//getchar();
	return 0;

}
