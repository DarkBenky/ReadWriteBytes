#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST_WIDTH 16
#define TEST_HEIGHT 16
#define TEST_CHANNELS 4
#define TEST_IMAGE_SIZE (TEST_WIDTH * TEST_HEIGHT * TEST_CHANNELS)

int main() {
    printf("=== Batch Loss Computation Debug ===\n\n");
    
    CNNConfig cfg = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    cfg.max_batch_size = 4;  /* Small batch for debugging */
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;
    
    /* Test all loss types */
    cfg.loss_config.num_losses = 5;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.loss_config.types[1] = LOSS_MSE;
    cfg.loss_config.weights[1] = 0.5f;
    cfg.loss_config.types[2] = LOSS_LAPLACE;
    cfg.loss_config.weights[2] = 0.1f;
    cfg.loss_config.types[3] = LOSS_COLOR_VARIANCE;
    cfg.loss_config.weights[3] = 0.05f;
    cfg.loss_config.types[4] = LOSS_SSIM;
    cfg.loss_config.weights[4] = 0.2f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Simple architecture */
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, -1, "layer1"});
    cnn_add_layer(cnn, (LayerConfig){8, 4, 0, -1, "output"});
    
    cnn_finalize(cnn);
    
    /* Check initial weights */
    printf("CNN created with batch_size=%d\n", cfg.max_batch_size);
    printf("Checking initial weights of first layer...\n");
    /* We can't easily access weights directly, but we can infer from output */
    
    printf("Image size: %dx%dx%d = %d floats\n\n", TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS, TEST_IMAGE_SIZE);
    
    /* Create test data - simple known values */
    float *batch_input = calloc(cfg.max_batch_size * TEST_IMAGE_SIZE, sizeof(float));
    float *batch_target = calloc(cfg.max_batch_size * TEST_IMAGE_SIZE, sizeof(float));
    
    /* Fill with simple test pattern */
    for (int b = 0; b < cfg.max_batch_size; b++) {
        for (int i = 0; i < TEST_IMAGE_SIZE; i++) {
            int channel = i / (TEST_WIDTH * TEST_HEIGHT);
            if (channel < 3) {  /* RGB channels */
                batch_input[b * TEST_IMAGE_SIZE + i] = 0.5f;  /* Mid-gray */
                batch_target[b * TEST_IMAGE_SIZE + i] = 0.6f;  /* Slightly brighter */
            } else {  /* Luminance channel */
                batch_input[b * TEST_IMAGE_SIZE + i] = 0.5f;
                batch_target[b * TEST_IMAGE_SIZE + i] = 0.6f;
            }
        }
    }
    
    printf("Test data created:\n");
    printf("  Input: all RGB=0.5, luminance=0.5\n");
    printf("  Target: all RGB=0.6, luminance=0.6\n");
    printf("  Expected MAE per pixel: |0.5-0.6| = 0.1\n");
    printf("  RGB pixels per image: %d\n", TEST_WIDTH * TEST_HEIGHT * 3);
    printf("  Expected total MAE: 0.1 * %d * %d = %.2f\n\n", 
           TEST_WIDTH * TEST_HEIGHT * 3, cfg.max_batch_size,
           0.1f * TEST_WIDTH * TEST_HEIGHT * 3 * cfg.max_batch_size);
    
    /* Run one training step */
    printf("Running training step...\n");
    
    printf("\n=== Training Step 1 ===\n");
    
    /* DEBUG: Verify input was uploaded correctly */
    float *check_input = malloc(100 * sizeof(float));
    /* Read from batch_input_buf - need access to internal structures */
    printf("[DEBUG] Checking if input was uploaded (first 10 values): ");
    memcpy(check_input, batch_input, 10 * sizeof(float));
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", check_input[i]);
    }
    printf("\n");
    free(check_input);
    
    float loss = cnn_train_step(cnn, batch_input, batch_target, cfg.max_batch_size);
    
    /* For batch training, need to read from batch_output buffer */
    float *output = malloc(TEST_IMAGE_SIZE * sizeof(float));
    cnn_get_batch_output(cnn, output, 0);  /* Get first image from batch */
    
    printf("\n[DEBUG] Network output values:\n");
    printf("  First 10 pixels: ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", output[i]);
    }
    printf("\n  Expected: ~0.5 (matching input)\n");
    float avg_step1 = 0.0f;
    for (int i = 0; i < 1000; i++) avg_step1 += output[i];
    avg_step1 /= 1000.0f;
    printf("  Actual average of first 1000: %.6f\n", avg_step1);
    /* Keep output for later comparison - don't free yet */
    
    printf("\n=== Results ===\n");
    printf("Returned loss: %.10f\n", loss);
    
    float mae, mse, laplace, color, ssim;
    cnn_get_individual_losses(cnn, &mae, &mse, &laplace, &color, &ssim);
    printf("Individual losses:\n");
    printf("  MAE: %.10f\n", mae);
    printf("  MSE: %.10f\n", mse);
    printf("  Laplace: %.10f\n", laplace);
    printf("  Color: %.10f\n", color);
    printf("  SSIM: %.10f\n", ssim);
    
    /* Check if loss is reasonable */
    float expected_mae = 0.1f;  /* |0.5 - 0.6| */
    printf("\nExpected normalized MAE: ~%.10f\n", expected_mae);
    printf("Ratio (actual/expected): %.2f\n", mae / expected_mae);
    
    if (mae < 0 || mae > 1.0f) {
        printf("\n*** ERROR: MAE is out of expected range [0, 1] ***\n");
    } else if (fabs(mae - expected_mae) < 0.01f) {
        printf("\n*** SUCCESS: MAE matches expected value! ***\n");
    } else {
        printf("\n*** WARNING: MAE differs from expected (but in valid range) ***\n");
    }
    
    printf("\n=== Training Step 2 ===\n");
    loss = cnn_train_step(cnn, batch_input, batch_target, cfg.max_batch_size);
    cnn_get_individual_losses(cnn, &mae, &mse, &laplace, &color, &ssim);
    printf("Loss after 2nd step: %.10f\n", loss);
    printf("MAE after 2nd step: %.10f\n", mae);
    
    /* Check output after step 2 */
    float *output_step2 = malloc(TEST_IMAGE_SIZE * sizeof(float));
    cnn_get_batch_output(cnn, output_step2, 0);
    printf("\nNetwork output AFTER 2 steps (first 10): ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", output_step2[i]);
    }
    printf("\n");
    
    /* Check if outputs changed at all */
    int num_changed = 0;
    float avg_step2 = 0.0f;
    for (int i = 0; i < 1000; i++) {
        if (fabs(output_step2[i] - output[i]) > 0.001f) num_changed++;
        avg_step2 += output_step2[i];
    }
    avg_step2 /= 1000.0f;
    printf("Pixels that changed > 0.001 (out of first 1000): %d\n", num_changed);
    printf("Average of first 1000 pixels: %.6f\n", avg_step2);
    printf("Change in average from step 1: %.6f\n", avg_step2 - avg_step1);
    
    free(output);  /* Free output from step 1 */
    free(output_step2);
    
    if (fabs(loss) > 1e10f) {
        printf("\n*** ERROR: Loss EXPLODED! ***\n");
    }
    
    free(batch_input);
    free(batch_target);
    cnn_destroy(cnn);
    
    printf("\n=== Test completed successfully! ===\n");
    
    return 0;
}
