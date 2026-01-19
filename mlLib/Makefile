# CNN Denoising Library - Optimized Makefile
CC = gcc
CFLAGS = -O3 -Wall
OPENCL_LIBS = -lOpenCL -lm

# Main targets
all: libcnn_denoise.a example_rgb_helpers benchmark_network benchmark_inference

# Library: Optimized CNN denoising library
libcnn_denoise.a: cnn_denoise.o
	ar rcs libcnn_denoise.a cnn_denoise.o

cnn_denoise.o: cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -c cnn_denoise.c -o cnn_denoise.o

# Examples and benchmarks
example_rgb_helpers: example_rgb_helpers.c cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -o example_rgb_helpers example_rgb_helpers.c cnn_denoise.c $(OPENCL_LIBS)

benchmark_custom: benchmark_custom.c
	$(CC) $(CFLAGS) -o benchmark_custom benchmark_custom.c $(OPENCL_LIBS)

benchmark_inference: benchmark_inference.c
	$(CC) $(CFLAGS) -o benchmark_inference benchmark_inference.c $(OPENCL_LIBS)

benchmark_network: benchmark_network.c
	$(CC) $(CFLAGS) -o benchmark_network benchmark_network.c $(OPENCL_LIBS)

test_lib: test_lib.c libcnn_denoise.a
	$(CC) $(CFLAGS) -o test_lib test_lib.c -L. -lcnn_denoise $(OPENCL_LIBS)

# Utility targets
clean:
	rm -f *.o *.a example_rgb_helpers benchmark_* test_lib

rebuild: clean all

# Run examples
run_example: example_rgb_helpers
	@echo "=== RGB Helper Functions Example ==="
	./example_rgb_helpers

run_benchmarks: benchmark_inference benchmark_network benchmark_custom
	@echo "=== Single Layer Inference (optimized) ==="
	@./benchmark_inference 2>&1 | tail -15 || true
	@echo ""
	@echo "=== Multi-Layer Network (real-time) ==="
	@./benchmark_network 2>&1 | tail -15 || true
	@echo ""
	@echo "=== Training Benchmark (kernel fusion) ==="
	@./benchmark_custom 2>&1 | tail -15 || true

help:
	@echo "CNN Denoising Library - Build Targets"
	@echo ""
	@echo "Main targets:"
	@echo "  all                  - Build library and all examples"
	@echo "  libcnn_denoise.a     - Build only the library"
	@echo "  example_rgb_helpers  - RGB helper functions example"
	@echo "  benchmark_inference  - Single layer inference benchmark"
	@echo "  benchmark_network    - Multi-layer network benchmark"
	@echo "  benchmark_custom     - Training benchmark with kernel fusion"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean                - Remove all built files"
	@echo "  rebuild              - Clean and rebuild everything"
	@echo "  run_example          - Run RGB helpers example"
	@echo "  run_benchmarks       - Run all benchmarks"
	@echo "  help                 - Show this message"

.PHONY: all clean rebuild run_example run_benchmarks help
