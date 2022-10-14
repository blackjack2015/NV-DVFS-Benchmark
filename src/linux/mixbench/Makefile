include ../common.mk

################################################################################

# Target rules
all: build

build: mixbench

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

mix_kernels_cuda.o:mix_kernels_cuda.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main-cuda.o:main-cuda.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mixbench:mix_kernels_cuda.o main-cuda.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./mixbench

clean:
	rm -f *.o
	rm -rf ../bin/mixbench

clobber: clean
