include ../common.mk

################################################################################

# Target rules
all: build

build: shmembench

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

shmem_kernels.o:shmem_kernels.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

shmembench:shmem_kernels.o main.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./shmembench

clean:
	rm -f *.o
	rm -rf ../bin/shmembench

clobber: clean
