include ../common.mk

################################################################################

# Target rules
all: build

build: BlackScholes

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

BlackScholes.o:BlackScholes.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

BlackScholes_gold.o:BlackScholes_gold.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

BlackScholes: BlackScholes.o BlackScholes_gold.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
	$(EXEC) mkdir -p ../bin/
	$(EXEC) mv $@ ../bin/

run: build
	$(EXEC) ./BlackScholes

clean:
	rm -f *.o
	rm -rf ../bin/BlackScholes

clobber: clean
