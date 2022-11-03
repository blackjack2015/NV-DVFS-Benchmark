include ../common.mk

################################################################################

# Target rules
all: build

build: cachebench-l2-only

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

cache_kernels.o:cache_kernels.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -DL2_ONLY -o $@ -c $<

main.o:main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

cachebench-l2-only:cache_kernels.o main.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./cachebench-l2-only

clean:
	rm -f *.o
	rm -rf ../bin/cachebench-l2-only

clobber: clean
