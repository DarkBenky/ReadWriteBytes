/* test_batch_simple.c - Simple test of batch training functionality */

#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BATCH_SIZE 8

void generate_synthetic_data(float *data, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.25f;
    }
}

void add_noise_to_data(float *clean, float *noisy, int size, float noise_level) {
    for (int i = 0; i < size; i++) {
        float noise = ((float)rand() / RAND_MAX - 0.5f) * noise_level;
        noisy[i] = fminf(fmaxf(clean[i] + noise, 0.0f), 1.0f);
    }
}

int main() {
    printf("=== Batch Training Test ===\n\n");
    
    /* Configure network with batch training */
    CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
    cfg.max_batch_size = BATCH_SIZE;  /* Enable batch mode */
    cfg.learning_rate = 0.001f;
    cfg.optimizer = OPTIMIZER_SGD;
    
    printf("Creating network with max_batch_size=%d...\n", BATCH_SIZE);
    CNNDenoiser *cnn = cnn_create(cfg);
    
    /* Simple 2-layer network */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, -1, "encode"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 0, -1, "decode"});
    
    cnn_finalize(cnn);
    printf("\n");
    
    /* Allocate data buffers */
    int img_size = WIDTH * HEIGHT * 4;
    
    /* Single image buffers */
    float *single_clean = malloc(img_size * sizeof(float));
    float *single_noisy = malloc(img_size * sizeof(float));
    
    /* Batch buffers */
    float *batch_clean = malloc(BATCH_SIZE * img_size * sizeof(float));
    float *batch_noisy = malloc(BATCH_SIZE * img_size * sizeof(float));
    
    /* Generate test data */
    printf("Generating synthetic test data...\n");
    generate_synthetic_data(single_clean, img_size, 42);
    memcpy(single_noisy, single_clean, img_size * sizeof(float));
    add_noise_to_data(single_clean, single_noisy, img_size, 0.1f);
    
    for (int i = 0; i < BATCH_SIZE; i++) {
        generate_synthetic_data(&batch_clean[i * img_size], img_size, 100 + i);
        memcpy(&batch_noisy[i * img_size], &batch_clean[i * img_size], img_size * sizeof(float));
        add_noise_to_data(&batch_clean[i * img_size], &batch_noisy[i * img_size], img_size, 0.1f);
    }
    
    printf("\n--- Test 1: Single Image Training (batch_size=1) ---\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int step = 0; step < 10; step++) {
        float loss = cnn_train_step(cnn, single_noisy, single_clean, 1);
        if (step % 2 == 0) {
            printf("Step %d: Loss = %.6f\n", step, loss);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double single_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time for 10 single images: %.3f seconds\n", single_time);
    
    printf("\n--- Test 2: Batch Training (batch_size=%d) ---\n", BATCH_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int step = 0; step < 10; step++) {
        float loss = cnn_train_step(cnn, batch_noisy, batch_clean, BATCH_SIZE);
        if (step % 2 == 0) {
            printf("Batch %d: Loss = %.6f\n", step, loss);
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double batch_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time for 10 batches (%d images): %.3f seconds\n", BATCH_SIZE * 10, batch_time);
    
    /* Calculate speedup */
    double single_time_per_img = single_time / 10.0;
    double batch_time_per_img = batch_time / (10.0 * BATCH_SIZE);
    double speedup = single_time_per_img / batch_time_per_img;
    
    printf("\n--- Performance Summary ---\n");
    printf("Single-image mode: %.1f ms/image\n", single_time_per_img * 1000);
    printf("Batch mode:        %.1f ms/image\n", batch_time_per_img * 1000);
    printf("Speedup:           %.2fx faster\n", speedup);
    
    if (speedup > 1.5f) {
        printf("\n✓ SUCCESS: Batch training is %.1fx faster!\n", speedup);
    } else {
        printf("\n! Note: Speedup lower than expected (may be due to small test size)\n");
    }
    
    /* Cleanup */
    free(single_clean);
    free(single_noisy);
    free(batch_clean);
    free(batch_noisy);
    cnn_destroy(cnn);
    
    printf("\nBatch training test complete!\n");
    return 0;
}
