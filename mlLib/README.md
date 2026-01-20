# CNN Denoiser Library

Fast GPU-accelerated image denoising with OpenCL. Features residual learning, Adam optimizer, and multi-loss training.

## Quick Start

```bash
make all           # Build everything
./example_easy     # See usage examples
./benchmark        # Run performance tests
./test_features    # Validate correctness
```

## Files

**Core Library**:
- `cnn_denoise.c/h` - Main implementation (974 lines)

**Usage**:
- `example_easy.c` - Comprehensive example showing:
  - Optimizer selection (SGD vs Adam)
  - Loss functions (MAE, MSE, Laplace)
  - Multi-loss configuration
  - Residual mode
  - Training loop with LR scheduling

**Testing**:
- `benchmark.c` - Performance measurements for all components
- `test_features.c` - Correctness validation

**Documentation**:
- `QUICK_REFERENCE.txt` - API overview
- `DOCUMENTATION.txt` - Detailed guide
- `OPTIMIZATION_SUMMARY.md` - Performance improvements

## Example Usage

```c
// 1. Configure
CNNConfig cfg = cnn_default_config(256, 256, 4);
cfg.optimizer = OPTIMIZER_ADAM;
cfg.learning_rate = 0.001f;
cfg.residual_mode = 1;  // Predict noise

// 2. Multi-loss for edge preservation
cfg.loss_config.num_losses = 2;
cfg.loss_config.types[0] = LOSS_MAE;
cfg.loss_config.weights[0] = 1.0f;
cfg.loss_config.types[1] = LOSS_LAPLACE;
cfg.loss_config.weights[1] = 0.1f;

// 3. Build network
CNNDenoiser* cnn = cnn_create(cfg);
cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "enc1"});
cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "enc2"});
cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "dec1"});
cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "dec2"});
cnn_finalize(cnn);

// 4. Train
for (int epoch = 0; epoch < 100; epoch++) {
    float loss = cnn_train_step(cnn, noisy, target, 1);
}

// 5. Inference
cnn_denoise(cnn, noisy_image, clean_image, 1);
```

## Performance (RTX 3090)

**Inference**:
- 128×128: 0.17ms (98 MP/s)
- 1024×1024: 7.0ms (150 MP/s)

**Training**:
- SGD: 8.3 ms/iteration
- Adam: 11.3 ms/iteration

**Multi-loss overhead**: <0.1ms for 3 losses

## Features

✓ Residual learning (67% faster convergence)  
✓ Adam optimizer with adaptive LR  
✓ Multi-loss training (MAE, MSE, Laplace)  
✓ Float4 vectorization  
✓ Fused GPU kernels  
✓ Non-square image support  

## Build Requirements

- OpenCL 1.2+
- Clang or GCC
- GPU with OpenCL support

## License

See main repository for license information.
