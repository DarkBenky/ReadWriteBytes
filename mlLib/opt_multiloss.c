#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn_denoise.h"

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("Multi-Loss Gradient Accumulation Benchmark\n");
    printf("==========================================\n\n");
    
    /* Initialize CNN with multi-loss configuration */
    CNNConfig cfg = cnn_default_config(128, 128, 4);
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.001f;
    
    /* Multi-loss configuration: MAE + MSE + Laplace */
    cfg.loss_config.num_losses = 3;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 0.4f;
    cfg.loss_config.types[1] = LOSS_MSE;
    cfg.loss_config.weights[1] = 0.3f;
    cfg.loss_config.types[2] = LOSS_LAPLACE;
    cfg.loss_config.weights[2] = 0.3f;
    
    CNNDenoiser *cnn = cnn_create(cfg);
    
    /* Add layers */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "enc1"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "enc2"});
    cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "dec1"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "dec2"});
    cnn_finalize(cnn);
    
    /* Create dummy input and target data */
    int size = 128 * 128 * 4;
    float *input = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        input[i] = (float)(i % 256) / 255.0f;
        target[i] = (float)((i + 100) % 256) / 255.0f;
    }
    
    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cnn_train_step(cnn, input, target, 1);
    }
    
    /* Benchmark multi-loss training */
    double start = get_time_ms();
    int iterations = 30;
    for (int i = 0; i < iterations; i++) {
        cnn_train_step(cnn, input, target, 1);
    }
    double end = get_time_ms();
    double avg_multi = (end - start) / iterations;
    
    printf("Multi-loss (MAE + MSE + Laplace): %.3f ms\n", avg_multi);
    
    /* Now test single loss for comparison */
    cnn_destroy(cnn);
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cnn = cnn_create(cfg);
    
    /* Add layers again */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "enc1"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "enc2"});
    cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "dec1"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "dec2"});
    cnn_finalize(cnn);
    
    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cnn_train_step(cnn, input, target, 1);
    }
    
    /* Benchmark single loss */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        cnn_train_step(cnn, input, target, 1);
    }
    end = get_time_ms();
    double avg_single = (end - start) / iterations;
    
    printf("Single loss (MAE only):            %.3f ms\n", avg_single);
    printf("\nOverhead from multi-loss:          %.3f ms (%.1f%%)\n",
           avg_multi - avg_single, 
           ((avg_multi - avg_single) / avg_single) * 100.0);
    
    printf("\nCurrent implementation creates/destroys temp buffers\n");
    printf("for each loss type. Optimization: write gradients directly\n");
    printf("to accumulated buffer to eliminate temp buffer overhead.\n");
    
    free(input);
    free(target);
    cnn_destroy(cnn);
    return 0;
}
