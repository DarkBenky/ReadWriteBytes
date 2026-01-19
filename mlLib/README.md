# CNN Denoising Library - Optimized for GPU

High-performance CNN denoising library using OpenCL with state-of-the-art optimizations.

## Features

- **Real-time performance:** ~4ms inference @ 800x600 (231.9 FPS)
- **GPU optimized:** 4 outputs per thread, float4 vectorization
- **Easy-to-use:** RGB helper functions for training and inference
- **Production-ready:** Comprehensive API with examples

## Quick Start

```bash
# Build everything
make

# Run RGB helper example
make run_example

# Run all benchmarks
make run_benchmarks

# Show all targets
make help
```

## Example Usage

```c
#include "cnn_denoise.h"

// Create network (800x600, RGB + Luminance = 4 channels)
CNNConfig config = {800, 600, 4, 4, 0.001f, 0};
CNNDenoiser *cnn = cnn_create(config);

// Build architecture
cnn_add_layer(cnn, (LayerConfig){4, 24, 1, "encoder1"});
cnn_add_layer(cnn, (LayerConfig){24, 48, 1, "bottleneck"});
cnn_add_layer(cnn, (LayerConfig){48, 24, 1, "decoder1"});
cnn_add_layer(cnn, (LayerConfig){24, 4, 0, "output"});
cnn_finalize(cnn);

// Easy RGB inference (RGB in, RGB out)
unsigned char *noisy_rgb = load_image("noisy.jpg", 800, 600);
unsigned char *clean_rgb = malloc(800 * 600 * 3);
cnn_inference_rgb(cnn, noisy_rgb, clean_rgb, 800, 600);
```

## Documentation

See [LIBRARY_OPTIMIZATIONS.md](LIBRARY_OPTIMIZATIONS.md) for:
- Complete optimization details
- RGB + Luminance format explanation
- Helper function reference
- Training examples
- Performance benchmarks

## Files

- `cnn_denoise.c/h` - Main library (optimized kernels)
- `example_rgb_helpers.c` - RGB helper function examples
- `benchmark_inference.c` - Single layer benchmark
- `benchmark_network.c` - Multi-layer real-time benchmark
- `benchmark_custom.c` - Training benchmark with kernel fusion

## Performance

| Configuration | Inference Time | Throughput |
|--------------|----------------|------------|
| Single layer (4→32) | 0.25 ms | 4069 img/sec |
| Real-time (4→24→48→24→4) | 4.31 ms | 231.9 FPS |
| Training (kernel fusion) | 1.74 ms/iter | 3.06x faster |

**Hardware:** NVIDIA RTX 3090, OpenCL 3.0

## Requirements

- GCC (C compiler)
- OpenCL development libraries
- GPU with OpenCL support

## License

See project root for license information.
