# Residual Layer Architecture

This document explains the new layer-based residual architecture that replaces the old `residual_mode` config flag.

## Problem with Old Approach

The old approach used a global `residual_mode` flag that applied residual learning at the network level:
- `residual_mode = 1`: Network predicts noise, output = input - prediction
- `residual_mode = 0`: Network predicts clean image directly

This approach had issues:
1. Not flexible - all layers affected by global flag
2. Hard to debug gradient flow
3. Couldn't have mixed architectures

## New Layer-Based Approach

The new architecture introduces two new layer types that make residual connections explicit:

### `LAYER_RESIDUAL_INPUT`
- Saves the current activation for later use
- Acts as a pass-through (output = input)
- No trainable parameters

### `LAYER_RESIDUAL_SUBTRACT`
- Computes: output = saved_input - current_input
- Typically used as: denoised = input - noise_prediction
- No trainable parameters
- References a `LAYER_RESIDUAL_INPUT` layer via `residual_from` index

## Example Architecture

### Noise Prediction + Denoising

```c
// Input (4 channels, noisy image)

// Save input for later
cnn_add_layer(cnn, (LayerConfig){
    .type = LAYER_RESIDUAL_INPUT,
    .cin = 4, .cout = 4,
    .name = "save_input"
});

// Predict noise with CNN layers
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 16, 1, -1, -1, "noise_1"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 16, 1, -1, -1, "noise_2"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 4, 0, -1, -1, "noise_out"});

// Subtract noise from input: denoised = input - noise
cnn_add_layer(cnn, (LayerConfig){
    .type = LAYER_RESIDUAL_SUBTRACT,
    .cin = 4, .cout = 4,
    .residual_from = 0,  // Reference layer 0 (save_input)
    .name = "denoise"
});

// Refine denoised image
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 12, 1, -1, -1, "refine_1"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 12, 4, 0, -1, -1, "output"});
```

## LayerConfig Structure

```c
typedef struct {
    LayerType type;          // LAYER_CONV, LAYER_RESIDUAL_INPUT, or LAYER_RESIDUAL_SUBTRACT
    int cin;                 // Input channels
    int cout;                // Output channels
    int use_relu;            // ReLU activation (CONV only)
    int skip_from;           // Skip connection source layer (-1 = none)
    int residual_from;       // Residual input source layer (-1 = use network input)
    char name[64];           // Layer name
} LayerConfig;
```

## Benefits

1. **Explicit**: Residual connections are visible in architecture
2. **Flexible**: Can place residual operations anywhere
3. **Debuggable**: Easy to inspect saved inputs and gradients
4. **Composable**: Can combine with skip connections
5. **Clear**: Training target is always the clean image (no manual noise computation)

## Training

With the new architecture, you always pass the **clean image** as the target:

```c
// Old way (with residual_mode = 1):
// cnn_train_step(cnn, noisy_input, noise_target, 1);  // noise = noisy - clean

// New way:
cnn_train_step(cnn, noisy_input, clean_target, 1);  // Always use clean image
```

The network learns to:
1. Predict noise in the first branch
2. Subtract it to get denoised image
3. Refine the denoised image in the second branch

## Gradient Flow

During backpropagation:

**RESIDUAL_SUBTRACT**: `output = saved - noise`
- Gradient w.r.t. noise: `-gradient` (negated)
- Gradient w.r.t. saved: `+gradient` (unchanged)

**RESIDUAL_INPUT**: Pass-through
- Gradient flows directly to previous layer

## Migration Guide

### Old Code (residual_mode)
```c
cfg.residual_mode = 1;
cnn_add_layer(cnn, (LayerConfig){4, 16, 1, -1, "layer1"});
cnn_add_layer(cnn, (LayerConfig){16, 4, 0, -1, "output"});
cnn_finalize(cnn);

// Compute noise for training
for (int i = 0; i < size; i++) {
    noise[i] = noisy[i] - clean[i];
}
cnn_train_step(cnn, noisy, noise, 1);
```

### New Code (layer-based)
```c
cfg.residual_mode = 0;
cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 16, 1, -1, -1, "layer1"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 4, 0, -1, -1, "noise"});
cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 4, 0, -1, -1, "output"});
cnn_finalize(cnn);

// No need to compute noise - just use clean image
cnn_train_step(cnn, noisy, clean, 1);
```

## Examples

See:
- `test_residual_layers.c` - Simple demonstration
- `debug_residual.c` - Updated to use new architecture
- `train.c` - Can be updated to use new architecture for better results
