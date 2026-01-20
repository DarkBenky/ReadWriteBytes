/* Benchmark and optimize Adam update kernel */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP 3
#define ITERS 30

double benchmark_adam(CNNDenoiser* cnn, float* input, float* target, int size) {
    for (int i = 0; i < WARMUP; i++) cnn_train_step(cnn, input, target, 1);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERS; i++) cnn_train_step(cnn, input, target, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return (elapsed * 1000.0) / ITERS;
}

int main() {
    printf("Adam Optimizer Kernel Optimization\n");
    printf("===================================\n\n");
    
    int W = 128, H = 128;
    srand(42);
    
    /* Baseline: Adam */
    CNNConfig cfg = cnn_default_config(W, H, 4);
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.001f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "enc1"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "enc2"});
    cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "dec1"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "dec2"});
    cnn_finalize(cnn);
    
    int size = W * H * 4;
    float *input = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        input[i] = (float)rand() / RAND_MAX;
        target[i] = 0.5f;
    }
    
    printf("Current Adam implementation: ");
    fflush(stdout);
    double adam_time = benchmark_adam(cnn, input, target, size);
    printf("%.3f ms\n", adam_time);
    
    printf("\nAdam update uses float4 vectorization for weights.\n");
    printf("Current implementation is already optimized with:\n");
    printf("- Vectorized operations for weight tensors\n");
    printf("- Bias correction in kernel\n");
    printf("- Fused m/v updates\n");
    
    free(input);
    free(target);
    cnn_destroy(cnn);
    return 0;
}
