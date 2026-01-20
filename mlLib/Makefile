# CNN Denoising Library - Optimized Makefile
CC = clang
CL_BUILD_FLAGS = -cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only
CFLAGS = -O3 -march=native -mtune=native -ffast-math -funroll-loops -Wall -DCL_BUILD_OPTIONS="\"$(CL_BUILD_FLAGS)\""
LDFLAGS = -O3 -march=native -mtune=native -ffast-math
OPENCL_LIBS = -lOpenCL -lm

# Main targets
all: example_easy benchmark test_features

# Example: Comprehensive usage example showing all features
example_easy: example_easy.c cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -o example_easy example_easy.c cnn_denoise.c $(LDFLAGS) $(OPENCL_LIBS)

# Benchmark: Performance testing for all components
benchmark: benchmark.c cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -o benchmark benchmark.c cnn_denoise.c $(LDFLAGS) $(OPENCL_LIBS)

# Test: Correctness validation
test_features: test_features.c cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -o test_features test_features.c cnn_denoise.c $(LDFLAGS) $(OPENCL_LIBS)

# Library (optional)
libcnn_denoise.a: cnn_denoise.o
	ar rcs libcnn_denoise.a cnn_denoise.o

cnn_denoise.o: cnn_denoise.c cnn_denoise.h
	$(CC) $(CFLAGS) -c cnn_denoise.c -o cnn_denoise.o

# Utility targets
clean:
	rm -f *.o *.a example_rgb_helpers benchmark_* test_lib test_training test_residual test_features

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
# Utility targets
clean:
	rm -f *.o *.a example_easy benchmark test_features

rebuild: clean all

run_example:
	./example_easy

run_benchmark:
	./benchmark

run_test:
	./test_features

help:
	@echo "CNN Denoising Library - Build Targets"
	@echo ""
	@echo "Main targets:"
	@echo "  all           - Build everything (example, benchmark, test)"
	@echo "  example_easy  - Comprehensive usage example"
	@echo "  benchmark     - Performance measurements"
	@echo "  test_features - Correctness validation"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean         - Remove build artifacts"
	@echo "  rebuild       - Clean and rebuild"
	@echo "  run_example   - Run usage example"
	@echo "  run_benchmark - Run performance tests"
	@echo "  run_test      - Run correctness tests"
	@echo "  help          - Show this message"

.PHONY: all clean rebuild run_example run_benchmark run_test help
