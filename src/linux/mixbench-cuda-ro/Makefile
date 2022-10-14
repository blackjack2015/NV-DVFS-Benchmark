include ../common.mk

################################################################################

# Target rules
all: build

build: mixbench-cuda-ro

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

mix_kernels_cuda_ro.o:mix_kernels_cuda_ro.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main-cuda.o:main-cuda.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mixbench-cuda-ro:mix_kernels_cuda_ro.o main-cuda.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./mixbench-cuda-ro

clean:
	rm -f *.o
	rm -rf ../bin/mixbench-cuda-ro

clobber: clean
