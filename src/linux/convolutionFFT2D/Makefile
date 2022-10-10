include ../common.mk

################################################################################
LIBRARIES := -lcufft

# Target rules
all: build

build: convolutionFFT2D

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

convolutionFFT2D.o:convolutionFFT2D.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

convolutionFFT2D_gold.o:convolutionFFT2D_gold.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

convolutionFFT2D: convolutionFFT2D.o convolutionFFT2D_gold.o main.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./convolutionFFT2D

clean:
	rm -f convolutionFFT2D *.o
	rm -rf ../bin/convolutionFFT2D

clobber: clean
