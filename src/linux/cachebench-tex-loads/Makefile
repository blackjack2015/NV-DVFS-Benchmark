include ../common.mk

################################################################################

# Target rules
all: build

build: cachebench-tex-loads

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

cache_kernels.o:cache_kernels.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -DTEX_LOADS -o $@ -c $<

main.o:main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -DTEX_LOADS -o $@ -c $<

cachebench-tex-loads: main.o cache_kernels.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./cachebench-tex-loads

clean:
	rm -f *.o
	rm -rf ../bin/cachebench-tex-loads

clobber: clean
