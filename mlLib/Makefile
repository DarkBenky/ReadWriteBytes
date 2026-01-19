# CNN Denoising Library & Examples Makefile
DEEPCL_DIR = dist

CXX = g++
CC = gcc
CXXFLAGS = -std=c++11 -Wall -O2 \
	-I$(DEEPCL_DIR)/include \
	-I$(DEEPCL_DIR)/include/easycl \
	-I$(DEEPCL_DIR)/include/deepcl

CFLAGS = -O3 -Wall

LDFLAGS = -L$(DEEPCL_DIR)/lib \
	-lDeepCL -lEasyCL -lclBLAS -lclew \
	-Wl,-rpath,$(PWD)/$(DEEPCL_DIR)/lib

OPENCL_LIBS = -lOpenCL -lm

# Main targets
all: example libcnn_denoise.a

# Library: Your custom CNN denoising library
libcnn_denoise.a: cnn_denoise.o
	ar rcs libcnn_denoise.a cnn_denoise.o

cnn_denoise.o: cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -c cnn_denoise.c -o cnn_denoise.o

# Example using DeepCL (default)
example: example.cpp
	$(CXX) $(CXXFLAGS) -o example example.cpp $(LDFLAGS)

# Example using custom C library (commented out in example.cpp by default)
example_custom: example.cpp libcnn_denoise.a
	$(CC) $(CFLAGS) -o example_custom example.cpp -L. -lcnn_denoise $(OPENCL_LIBS)

# Benchmarks
benchmark_deepcl: benchmark_deepcl.cpp
	$(CXX) $(CXXFLAGS) -o benchmark_deepcl benchmark_deepcl.cpp $(LDFLAGS)

benchmark_custom: benchmark_custom.c
	$(CC) $(CFLAGS) -o benchmark_custom benchmark_custom.c $(OPENCL_LIBS)

# Test programs
test_lib: test_lib.c libcnn_denoise.a
	$(CC) $(CFLAGS) -o test_lib test_lib.c -L. -lcnn_denoise $(OPENCL_LIBS)

clean:
	rm -f *.o *.a example example_custom benchmark_deepcl benchmark_custom test_lib

# Run examples
run_example: example
	./example

run_custom: example_custom
	./example_custom

run_benchmarks: benchmark_deepcl benchmark_custom
	@echo "=== DeepCL Benchmark ==="
	@./benchmark_deepcl 2>&1 | tail -20 || true
	@echo ""
	@echo "=== Custom OpenCL Benchmark ==="
	@./benchmark_custom 2>&1 | tail -20 || true

.PHONY: all clean run_example run_custom run_benchmarks
