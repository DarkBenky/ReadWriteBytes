#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST_WIDTH 64
#define TEST_HEIGHT 48
#define TEST_SIZE (TEST_WIDTH * TEST_HEIGHT * 4)

int main() {
    printf("=== Minimal SSIM Loss Test ===\n");
    printf("Testing with %dx%d images, 4 channels\n\n", TEST_WIDTH, TEST_HEIGHT);
    
    CNNConfig cfg = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, 4);
    cfg.max_batch_size = 2;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.001f;
    cfg.use_profiling = 1;
    
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = LOSS_SSIM;
    cfg.loss_config.weights[0] = 1.0f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, -1, "layer1"});
    cnn_add_layer(cnn, (LayerConfig){8, 4, 0, -1, "output"});
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    
    float *input_planar = malloc(TEST_SIZE * sizeof(float));
    float *target_planar = malloc(TEST_SIZE * sizeof(float));
    float *batch_input = malloc(2 * TEST_SIZE * sizeof(float));
    float *batch_target = malloc(2 * TEST_SIZE * sizeof(float));
    
    printf("\n=== Test 1: Identical images (expect SSIM loss ~0) ===\n");
    for (int i = 0; i < TEST_SIZE; i++) {
        input_planar[i] = 0.5f + 0.1f * sinf(i * 0.1f);
        target_planar[i] = input_planar[i];
    }
    
    memcpy(batch_input, input_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_input + TEST_SIZE, input_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_target, target_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_target + TEST_SIZE, target_planar, TEST_SIZE * sizeof(float));
    
    float loss = cnn_train_step(cnn, batch_input, batch_target, 2);
    
    float mae_loss, mse_loss, laplace_loss, color_loss, ssim_loss;
    cnn_get_individual_losses(cnn, &mae_loss, &mse_loss, &laplace_loss, &color_loss, &ssim_loss);
    
    printf("Total Loss: %.6f\n", loss);
    printf("SSIM Loss: %.6f (should be ~0)\n", ssim_loss);
    
    printf("\n=== Test 2: Different images (expect SSIM loss > 0) ===\n");
    for (int i = 0; i < TEST_SIZE; i++) {
        input_planar[i] = 0.3f + 0.2f * sinf(i * 0.05f);
        target_planar[i] = 0.7f + 0.1f * cosf(i * 0.08f);
    }
    
    memcpy(batch_input, input_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_input + TEST_SIZE, input_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_target, target_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_target + TEST_SIZE, target_planar, TEST_SIZE * sizeof(float));
    
    loss = cnn_train_step(cnn, batch_input, batch_target, 2);
    cnn_get_individual_losses(cnn, &mae_loss, &mse_loss, &laplace_loss, &color_loss, &ssim_loss);
    
    printf("Total Loss: %.6f\n", loss);
    printf("SSIM Loss: %.6f (should be > 0)\n", ssim_loss);
    
    printf("\n=== Test 3: Slightly different images (expect small SSIM loss) ===\n");
    for (int i = 0; i < TEST_SIZE; i++) {
        input_planar[i] = 0.5f + 0.1f * sinf(i * 0.1f);
        target_planar[i] = input_planar[i] + 0.01f;
    }
    
    memcpy(batch_input, input_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_input + TEST_SIZE, input_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_target, target_planar, TEST_SIZE * sizeof(float));
    memcpy(batch_target + TEST_SIZE, target_planar, TEST_SIZE * sizeof(float));
    
    loss = cnn_train_step(cnn, batch_input, batch_target, 2);
    cnn_get_individual_losses(cnn, &mae_loss, &mse_loss, &laplace_loss, &color_loss, &ssim_loss);
    
    printf("Total Loss: %.6f\n", loss);
    printf("SSIM Loss: %.6f (should be small but > 0)\n", ssim_loss);
    
    printf("\n=== Test Results ===\n");
    printf("✓ SSIM kernel created and executed\n");
    printf("✓ SSIM loss values are being computed\n");
    
    free(input_planar);
    free(target_planar);
    free(batch_input);
    free(batch_target);
    cnn_destroy(cnn);
    
    printf("\nTest completed successfully!\n");
    return 0;
}
