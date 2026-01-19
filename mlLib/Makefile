# CNN Denoising Library - Optimized Makefile
CC = clang
CL_BUILD_FLAGS = -cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only
CFLAGS = -O3 -march=native -mtune=native -ffast-math -funroll-loops -Wall -DCL_BUILD_OPTIONS="\"$(CL_BUILD_FLAGS)\""
LDFLAGS = -O3 -march=native -mtune=native -ffast-math
OPENCL_LIBS = -lOpenCL -lm

# Main targets
all: libcnn_denoise.a example_rgb_helpers test_training

# Library: Optimized CNN denoising library
libcnn_denoise.a: cnn_denoise.o
	ar rcs libcnn_denoise.a cnn_denoise.o

cnn_denoise.o: cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -c cnn_denoise.c -o cnn_denoise.o

# Examples and tests
example_rgb_helpers: example_rgb_helpers.c cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -o example_rgb_helpers example_rgb_helpers.c cnn_denoise.c $(LDFLAGS) $(OPENCL_LIBS)

test_training: test_training.c cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -o test_training test_training.c cnn_denoise.c $(LDFLAGS) $(OPENCL_LIBS)

benchmark_custom: benchmark_custom.c
	$(CC) $(CFLAGS) -o benchmark_custom benchmark_custom.c $(LDFLAGS) $(OPENCL_LIBS)

benchmark_inference: benchmark_inference.c
	$(CC) $(CFLAGS) -o benchmark_inference benchmark_inference.c $(LDFLAGS) $(OPENCL_LIBS)

benchmark_network: benchmark_network.c
	$(CC) $(CFLAGS) -o benchmark_network benchmark_network.c $(LDFLAGS) $(OPENCL_LIBS)

test_lib: test_lib.c libcnn_denoise.a
	$(CC) $(CFLAGS) -o test_lib test_lib.c -L. -lcnn_denoise $(LDFLAGS) $(OPENCL_LIBS)

# Utility targets
clean:
	rm -f *.o *.a example_rgb_helpers benchmark_* test_lib test_training

rebuild: clean all

# Run examples
run_example: example_rgb_helpers
	@echo "=== RGB Helper Functions Example ==="
	./example_rgb_helpers

run_test: test_training
	@echo "=== Training System Test ==="
	./test_training

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
	@echo "  all                  - Build library and examples"
	@echo "  libcnn_denoise.a     - Build static library"
	@echo "  example_rgb_helpers  - RGB helper example"
	@echo "  test_training        - Training validation test"
	@echo "  test_lib             - Library unit tests"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean                - Remove all build artifacts"
	@echo "  rebuild              - Clean and rebuild"
	@echo "  run_example          - Run RGB example"
	@echo "  run_test             - Run training test"
	@echo "  help                 - Show this message"

.PHONY: all clean rebuild run_example run_test help

.PHONY: all clean rebuild run_example run_benchmarks help
