/* Test program for cnn_denoise library with optimization iterations */

#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main() {
    printf("========================================\n");
    printf("  CNN Denoising Library Test            \n");
    printf("========================================\n");
    
    CNNConfig config = {
        .input_width = 800,
        .input_height = 600,
        .input_channels = 4,   /* Must be multiple of 4 */
        .output_channels = 4,
        .learning_rate = 0.001f,
        .use_profiling = 1
    };
    
    printf("\nConfiguration:\n");
    printf("  Image size: %dx%dx%d\n", config.input_width, config.input_height, config.input_channels);
    printf("  Learning rate: %.4f\n", config.learning_rate);
    printf("  Profiling: %s\n\n", config.use_profiling ? "enabled" : "disabled");
    
    printf("Creating CNN denoiser...\n");
    CNNDenoiser *cnn = cnn_create(config);
    if (!cnn) {
        printf("ERROR: Failed to create CNN denoiser\n");
        return 1;
    }
    
    /* Add layers: encoder-decoder architecture */
    printf("Building network architecture...\n");
    LayerConfig l1 = {.cin = 4, .cout = 32, .use_relu = 1, .name = "conv1"};
    LayerConfig l2 = {.cin = 32, .cout = 64, .use_relu = 1, .name = "conv2"};
    LayerConfig l3 = {.cin = 64, .cout = 32, .use_relu = 1, .name = "conv3"};
    LayerConfig l4 = {.cin = 32, .cout = 4, .use_relu = 0, .name = "conv4"};
    cnn_add_layer(cnn, l1);
    cnn_add_layer(cnn, l2);
    cnn_add_layer(cnn, l3);
    cnn_add_layer(cnn, l4);
    
    printf("Finalizing network...\n");
    if (!cnn_finalize(cnn)) {
        printf("ERROR: Failed to finalize network\n");
        return 1;
    }
    
    /* Prepare test data */
    int img_size = config.input_width * config.input_height * config.input_channels;
    float *input_img = malloc(img_size * sizeof(float));
    float *target_img = malloc(img_size * sizeof(float));
    
    srand(42);
    for (int i = 0; i < img_size; i++) {
        input_img[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        target_img[i] = input_img[i] * 0.9f;  /* Target is slightly darker */
    }
    
    /* Warmup */
    printf("\nWarming up (5 iterations)...\n");
    for (int i = 0; i < 5; i++) {
        float loss = cnn_train_step(cnn, input_img, target_img, 1);
        printf(".");
        fflush(stdout);
    }
    printf(" done!\n\n");
    
    /* Benchmark training */
    printf("Running training benchmark (10 iterations)...\n");
    printf("Iter | Loss      | Time (ms)\n");
    printf("-----|-----------|----------\n");
    
    double total_time = 0;
    for (int i = 0; i < 10; i++) {
        double t0 = get_time_ms();
        float loss = cnn_train_step(cnn, input_img, target_img, 1);
        double t1 = get_time_ms();
        double iter_time = t1 - t0;
        total_time += iter_time;
        
        printf("%4d | %.6f | %.2f\n", i, loss, iter_time);
    }
    
    /* Get detailed timing stats */
    TimingStats stats;
    cnn_get_timing_stats(cnn, &stats);
    
    printf("\n=== Detailed Timing Breakdown ===\n");
    printf("Forward pass:   %.2f ms (%.1f%%)\n", stats.forward_time_ms, 
           100.0 * stats.forward_time_ms / stats.total_time_ms);
    printf("Backward pass:  %.2f ms (%.1f%%)\n", stats.backward_time_ms,
           100.0 * stats.backward_time_ms / stats.total_time_ms);
    printf("Loss calc:      %.2f ms (%.1f%%)\n", stats.loss_time_ms,
           100.0 * stats.loss_time_ms / stats.total_time_ms);
    printf("Weight update:  %.2f ms (%.1f%%)\n", stats.update_time_ms,
           100.0 * stats.update_time_ms / stats.total_time_ms);
    printf("----------------\n");
    printf("Total per iter: %.2f ms\n", stats.total_time_ms);
    printf("Average time:   %.2f ms\n", total_time / 10.0);
    printf("Throughput:     %.2f images/sec\n", 1000.0 / (total_time / 10.0));
    printf("====================================\n");
    
    /* Test inference */
    printf("\nTesting inference (forward only)...\n");
    float *output = malloc(img_size * sizeof(float));
    double inf_start = get_time_ms();
    for (int i = 0; i < 20; i++) {
        cnn_denoise(cnn, input_img, output, 1);
    }
    double inf_end = get_time_ms();
    printf("Inference: %.2f ms/image (%.2f img/sec)\n", 
           (inf_end - inf_start) / 20.0, 20000.0 / (inf_end - inf_start));
    
    /* Cleanup */
    free(output);
    free(input_img);
    free(target_img);
    cnn_destroy(cnn);
    
    printf("\nTest complete!\n");
    return 0;
}
