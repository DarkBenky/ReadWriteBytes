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
    printf("Inference Performance Benchmark\n");
    printf("================================\n\n");
    
    /* Test various resolutions */
    int resolutions[][2] = {
        {128, 128},
        {256, 256},
        {512, 512},
        {800, 600},
        {1024, 1024}
    };
    int num_res = 5;
    
    for (int r = 0; r < num_res; r++) {
        int W = resolutions[r][0];
        int H = resolutions[r][1];
        
        /* Create CNN */
        CNNConfig cfg = cnn_default_config(W, H, 4);
        cfg.residual_mode = 1;  /* Residual mode (optimized) */
        CNNDenoiser *cnn = cnn_create(cfg);
        
        /* Standard 4-layer encoder-decoder */
        cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "enc1"});
        cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "enc2"});
        cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "dec1"});
        cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "dec2"});
        cnn_finalize(cnn);
        
        /* Create test input */
        int size = W * H * 4;
        float *input = malloc(size * sizeof(float));
        float *output = malloc(size * sizeof(float));
        for (int i = 0; i < size; i++) {
            input[i] = (float)(i % 256) / 255.0f;
        }
        
        /* Warmup */
        for (int i = 0; i < 5; i++) {
            cnn_denoise(cnn, input, output, 1);
        }
        
        /* Benchmark inference */
        int iterations = 50;
        double start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            cnn_denoise(cnn, input, output, 1);
        }
        double end = get_time_ms();
        double avg = (end - start) / iterations;
        
        float mpix = (W * H) / 1000000.0f;
        printf("%4dx%4d (%4.2f MP): %6.3f ms  (%6.1f MP/s)\n",
               W, H, mpix, avg, mpix * 1000.0 / avg);
        
        free(input);
        free(output);
        cnn_destroy(cnn);
    }
    
    printf("\nInference is already highly optimized with:\n");
    printf("- Float4 vectorization for 4-channel processing\n");
    printf("- GPU-accelerated residual subtraction\n");
    printf("- Optimized conv3x3 kernels with ReLU fusion\n");
    
    return 0;
}
