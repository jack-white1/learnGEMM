# Makefile for compiling CUDA programs

# Compiler
NVCC = nvcc

# Architecture
ARCH = -arch=sm_89

# Source files
SRCS = basic_mma.cu matmul.cu

# Executable names
EXES = basic_mma matmul

# Suppress specific warnings
DIAG_SUPPRESS = -diag-suppress=550

# Default target
all: $(EXES)

# Compile basic_mma
basic_mma: basic_mma.cu
	$(NVCC) $(ARCH) $(DIAG_SUPPRESS) -o $@ $<

# Compile matmul
matmul: matmul.cu
	$(NVCC) $(ARCH) $(DIAG_SUPPRESS) -o $@ $<

# Clean target
clean:
	rm -f $(EXES)

# Phony targets
.PHONY: all clean
