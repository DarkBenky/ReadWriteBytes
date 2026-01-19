/* Example: Easy training with RGB images using helper functions */

#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== CNN Denoising Library - RGB Helper Example ===\n\n");
    
    /* Step 1: Create network configuration */
    CNNConfig config = {
        .input_width = 800,
        .input_height = 600,
        .input_channels = 4,    /* RGB + Luminance for float4 optimization */
        .output_channels = 4,   /* RGB + Luminance output */
        .learning_rate = 0.00001f,
        .use_profiling = 0
    };
    
    CNNDenoiser *cnn = cnn_create(config);
    if (!cnn) {
        printf("Failed to create CNN\n");
        return 1;
    }
    
    /* Step 2: Build real-time optimized architecture (3->24->48->24->3 + luminance) */
    cnn_add_layer(cnn, (LayerConfig){4, 24, 1, "encoder1"});
    cnn_add_layer(cnn, (LayerConfig){24, 48, 1, "bottleneck"});
    cnn_add_layer(cnn, (LayerConfig){48, 24, 1, "decoder1"});
    cnn_add_layer(cnn, (LayerConfig){24, 4, 0, "output"});  /* No ReLU on output */
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    
    /* Step 3: Prepare synthetic RGB training data (800x600x3) */
    int width = 800, height = 600;
    unsigned char *clean_rgb = malloc(width * height * 3);
    unsigned char *noisy_rgb = malloc(width * height * 3);
    float *clean_rgba = malloc(width * height * 4 * sizeof(float));
    float *noisy_rgba = malloc(width * height * 4 * sizeof(float));
    
    /* Create synthetic clean image (gradient pattern) */
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            clean_rgb[idx + 0] = (x * 255) / width;       /* Red gradient */
            clean_rgb[idx + 1] = (y * 255) / height;      /* Green gradient */
            clean_rgb[idx + 2] = 128;                     /* Constant blue */
        }
    }
    
    printf("Image size: %dx%d RGB\n", width, height);
    printf("Training data size: %d bytes (RGB) -> %d floats (RGBA)\n", 
           width * height * 3, width * height * 4);
    printf("\n");
    
    /* Step 4: Use helper function to prepare training batch */
    printf("Preparing training batch with noise_sigma=0.05...\n");
    float noise_sigma = 0.05f;
    
    int ret = cnn_prepare_training_batch(clean_rgb, noisy_rgb, clean_rgba, noisy_rgba,
                                         width, height, noise_sigma);
    if (ret != 0) {
        printf("Failed to prepare training batch\n");
        return 1;
    }
    
    printf("RGB -> RGBA conversion done\n");
    printf("Gaussian noise added (sigma=%.3f)\n", noise_sigma);
    printf("Values clamped to [0,1]\n");
    printf("\n");
    
    /* Step 5: Training loop */
    printf("Running %d training iterations...\n", 100);
    for (int iter = 0; iter < 100; iter++) {
        float loss = cnn_train_step(cnn, noisy_rgba, clean_rgba, 1);
        
        if (iter % 10 == 0) {
            printf("  Iteration %d: Loss = %.6f\n", iter, loss);
        }
    }
    printf("\n");
    
    /* Step 6: Inference using easy RGB helper */
    printf("Running inference with RGB helper function...\n");
    unsigned char *output_rgb = malloc(width * height * 3);
    
    ret = cnn_inference_rgb(cnn, noisy_rgb, output_rgb, width, height);
    if (ret != 0) {
        printf("Failed to run inference\n");
        return 1;
    }
    
    printf("Inference complete (RGB in, RGB out)\n");
    printf("Internal RGBA conversion handled automatically\n");
    printf("\n");
    
    /* Step 7: Verify output */
    printf("Sample pixel comparison (center of image):\n");
    int center_idx = (height/2 * width + width/2) * 3;
    printf("  Clean:  RGB(%3d, %3d, %3d)\n", 
           clean_rgb[center_idx], clean_rgb[center_idx+1], clean_rgb[center_idx+2]);
    printf("  Noisy:  RGB(%3d, %3d, %3d)\n", 
           noisy_rgb[center_idx], noisy_rgb[center_idx+1], noisy_rgb[center_idx+2]);
    printf("  Output: RGB(%3d, %3d, %3d)\n", 
           output_rgb[center_idx], output_rgb[center_idx+1], output_rgb[center_idx+2]);
    printf("\n");
    
    printf("=== Summary ===\n");
    printf("Network architecture: 4->24->48->24->4 (optimized for real-time)\n");
    printf("Input format: RGB (3 channels) -> automatically converted to RGBA (4 channels)\n");
    printf("4th channel: Luminance (0.299*R + 0.587*G + 0.114*B)\n");
    printf("GPU optimization: float4 vectorization, 4 outputs per thread\n");
    printf("Expected inference time: ~4ms for 800x600 (from benchmark)\n");
    printf("\n");
    
    printf("Helper functions available:\n");
    printf("  - cnn_rgb_to_rgba_luminance()     : RGB uint8 -> RGBA float\n");
    printf("  - cnn_rgba_luminance_to_rgb()     : RGBA float -> RGB uint8\n");
    printf("  - cnn_prepare_training_batch()    : All-in-one: RGB + noise -> RGBA\n");
    printf("  - cnn_inference_rgb()             : Easy inference: RGB in, RGB out\n");
    
    /* Cleanup */
    free(clean_rgb);
    free(noisy_rgb);
    free(clean_rgba);
    free(noisy_rgba);
    free(output_rgb);
    cnn_destroy(cnn);
    
    printf("\nExample complete!\n");
    return 0;
}
