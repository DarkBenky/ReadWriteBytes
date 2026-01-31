#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnn_denoise.h"

int main() {
    printf("Testing Sobel loss integration...\n");
    
    int W = 32, H = 32, C = 4;
    
    CNNConfig cfg = cnn_default_config(W, H, C);
    cfg.max_batch_size = 2;
    cfg.output_channels = C;
    cfg.learning_rate = 0.001f;
    cfg.loss_config.num_losses = 3;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 0.33f;
    cfg.loss_config.types[1] = LOSS_LAPLACE;
    cfg.loss_config.weights[1] = 0.33f;
    cfg.loss_config.types[2] = LOSS_SOBEL;
    cfg.loss_config.weights[2] = 0.34f;
    
    CNNDenoiser *cnn = cnn_create(cfg);
    if (!cnn) {
        fprintf(stderr, "Failed to create CNN\n");
        return 1;
    }
    printf("CNN created successfully\n");
    
    /* Add a simple layer */
    LayerConfig layer = {
        .cin = C,
        .cout = C,
        .use_relu = 0,
        .name = "test_layer"
    };
    cnn_add_layer(cnn, layer);
    cnn_finalize(cnn);
    printf("CNN finalized\n");
    
    float *input = malloc(W * H * C * sizeof(float));
    float *target = malloc(W * H * C * sizeof(float));
    
    for (int i = 0; i < W * H * C; i++) {
        input[i] = 0.5f;
        target[i] = 0.5f;
    }
    
    /* Create an edge in the target */
    for (int y = 0; y < H; y++) {
        for (int x = W/2; x < W; x++) {
            for (int c = 0; c < 3; c++) {
                target[c * H * W + y * W + x] = 1.0f;
            }
        }
    }
    
    printf("Testing single batch training with Sobel loss...\n");
    float loss = cnn_train_step(cnn, input, target, 1);
    printf("Training loss: %f\n", loss);
    
    float mae, mse, laplace, color, ssim, sobel;
    cnn_get_individual_losses(cnn, &mae, &mse, &laplace, &color, &ssim, &sobel);
    printf("Individual losses:\n");
    printf("  MAE: %f\n", mae);
    printf("  MSE: %f\n", mse);
    printf("  Laplace: %f\n", laplace);
    printf("  Color: %f\n", color);
    printf("  SSIM: %f\n", ssim);
    printf("  Sobel: %f\n", sobel);
    
    if (sobel > 0.0f) {
        printf("✓ Sobel loss computation successful\n");
    } else {
        printf("✗ Sobel loss computation failed (loss = 0)\n");
    }
    
    free(input);
    free(target);
    cnn_destroy(cnn);
    
    printf("Test completed\n");
    return 0;
}
