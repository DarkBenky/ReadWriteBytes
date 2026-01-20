/* Optimize residual mode inference - baseline vs optimized */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUP 5
#define ITERS 50

double benchmark(CNNDenoiser* cnn, float* input, float* output, int size) {
    for (int i = 0; i < WARMUP; i++) cnn_denoise(cnn, input, output, 1);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < ITERS; i++) cnn_denoise(cnn, input, output, 1);
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    return (elapsed * 1000.0) / ITERS;
}

int main() {
    printf("Residual Mode Inference Optimization\n");
    printf("=====================================\n\n");
    
    int W = 256, H = 256;
    CNNConfig cfg = cnn_default_config(W, H, 4);
    cfg.residual_mode = 1;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "enc1"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "enc2"});
    cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "dec1"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "dec2"});
    cnn_finalize(cnn);
    
    int size = W * H * 4;
    float *input = malloc(size * sizeof(float));
    float *output = malloc(size * sizeof(float));
    
    srand(42);
    for (int i = 0; i < size; i++) input[i] = (float)rand() / RAND_MAX;
    
    printf("Baseline (CPU residual subtract): ");
    fflush(stdout);
    double baseline = benchmark(cnn, input, output, size);
    printf("%.3f ms\n", baseline);
    
    printf("\nCurrent implementation uses CPU for residual subtraction.\n");
    printf("We will optimize this to use GPU kernel.\n");
    
    free(input);
    free(output);
    cnn_destroy(cnn);
    return 0;
}
