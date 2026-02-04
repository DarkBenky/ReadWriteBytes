# Residual Mode Fix - Implementation Summary

## Problem Statement

The residual mode in the CNN denoising library was not working properly. The issue description suggested implementing a better architecture where residual connections are explicit layers rather than a global config flag.

### Original Issue
Instead of using a `residual_mode` config flag, implement an architecture like:
```
input (residual) → CNN → CNN → residual_subtract (returns input - noise) → CNN → CNN → loss
```

This creates a pipeline where:
1. Input is saved
2. CNN layers predict noise  
3. Residual subtract computes: denoised = input - noise
4. More CNN layers refine the result

## Solution Implemented

### New Layer Types

Added three layer types to the `LayerType` enum:

1. **`LAYER_CONV`** - Standard convolution layer (existing)
2. **`LAYER_RESIDUAL_INPUT`** - Saves input activation for later residual subtraction
3. **`LAYER_RESIDUAL_SUBTRACT`** - Computes `output = saved_input - current_input`

### Architecture Changes

#### Header Changes (`cnn_denoise.h`)

- Added `LayerType` enum
- Updated `LayerConfig` struct with:
  - `LayerType type` field
  - `int residual_from` field (references which layer saved the input)

#### Implementation Changes (`cnn_denoise.c`)

1. **Kernel Addition**
   - Added `copy_buffer` kernel for RESIDUAL_INPUT pass-through
   - Existing `residual_subtract` kernel used for RESIDUAL_SUBTRACT

2. **Layer Structure**
   - Updated `ConvLayer` struct with `type` and `residual_from` fields
   - Added `residual_saved` buffer for RESIDUAL_INPUT layers

3. **Forward Pass**
   - Added handling for RESIDUAL_INPUT (copy input to saved buffer and output)
   - Added handling for RESIDUAL_SUBTRACT (compute saved - current)
   - Kept existing CONV layer logic

4. **Backward Pass**
   - RESIDUAL_INPUT: Gradient passes through to previous layer
   - RESIDUAL_SUBTRACT: Gradient splits:
     - To noise prediction: negated gradient
     - To saved input: unchanged gradient
   - CONV: Existing backprop logic

5. **Weight Management**
   - `cnn_add_layer`: Only allocates weights/biases for CONV layers
   - `cnn_finalize`: Only allocates Adam buffers for CONV layers
   - Update/save/load: Skip non-CONV layers
   - `cnn_get_num_parameters`: Count only CONV layer parameters

6. **Cleanup**
   - `cnn_destroy`: Free buffers based on layer type
   - Properly release `residual_saved` for RESIDUAL_INPUT layers

7. **Architecture Display**
   - `cnn_print_architecture`: Shows layer type and residual connections

### Example Usage

```c
CNNConfig cfg = cnn_default_config(800, 600, 4);
cfg.residual_mode = 0;  // Use new layer-based residual

CNNDenoiser* cnn = cnn_create(cfg);

// Save input
cnn_add_layer(cnn, (LayerConfig){
    .type = LAYER_RESIDUAL_INPUT,
    .cin = 4, .cout = 4,
    .name = "save_input"
});

// Predict noise
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 16, 1, -1, -1, "noise_1"});
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 4, 0, -1, -1, "noise_out"});

// Subtract noise from input
cnn_add_layer(cnn, (LayerConfig){
    .type = LAYER_RESIDUAL_SUBTRACT,
    .cin = 4, .cout = 4,
    .residual_from = 0,  // Reference layer 0
    .name = "denoise"
});

// Refine
cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 4, 0, -1, -1, "output"});

cnn_finalize(cnn);

// Train with clean image as target (not noise!)
cnn_train_step(cnn, noisy_input, clean_target, 1);
```

## Files Modified

1. **mlLib/cnn_denoise.h** - Added LayerType enum and updated LayerConfig
2. **mlLib/cnn_denoise.c** - Implemented new layer types throughout
3. **mlLib/debug_residual.c** - Updated to demonstrate new architecture
4. **mlLib/test_residual_layers.c** - New test file
5. **mlLib/RESIDUAL_LAYERS.md** - Comprehensive documentation

## Key Benefits

1. **Explicit Architecture**: Residual connections are visible in layer definition
2. **Flexible**: Can place residual operations anywhere in the network
3. **Debuggable**: Can inspect saved activations and gradients
4. **Composable**: Works with skip connections
5. **Simplified Training**: Always use clean image as target

## Backward Compatibility

The old `residual_mode` config flag is still supported for existing code. Networks can use either:
- Old approach: `cfg.residual_mode = 1` with manual noise computation
- New approach: Layer-based residual with RESIDUAL_INPUT and RESIDUAL_SUBTRACT layers

## Testing

Created two test files:
- `test_residual_layers.c` - Simple demonstration of the new architecture
- Updated `debug_residual.c` - Tests the new layer types

Both verify that:
1. Network learns to denoise
2. Output differs from input (proving residual is applied)
3. Output approaches clean target

## Next Steps

To use this in production:

1. Update `train.c` to use the new architecture:
   - Add RESIDUAL_INPUT after input
   - Add RESIDUAL_SUBTRACT after noise prediction layers
   - Remove manual noise computation
   - Use clean images as training targets

2. Test on real image data to verify performance

3. Consider deprecating `residual_mode` flag in future version
