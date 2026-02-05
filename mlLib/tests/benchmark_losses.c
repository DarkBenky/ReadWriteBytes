#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TEST_WIDTH 32
#define TEST_HEIGHT 32
#define TEST_CHANNELS 4
#define TEST_IMAGE_SIZE (TEST_WIDTH * TEST_HEIGHT * TEST_CHANNELS)
#define NUM_ITERATIONS 100

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

void benchmark_loss(const char* name, LossType type, int batch_size) {
    CNNConfig cfg = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    cfg.max_batch_size = batch_size;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;
    
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = type;
    cfg.loss_config.weights[0] = 1.0f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 4, 1, -1, "layer1"});
    cnn_add_layer(cnn, (LayerConfig){4, 4, 0, -1, "output"});
    cnn_finalize(cnn);
    
    float *batch_input = calloc(batch_size * TEST_IMAGE_SIZE, sizeof(float));
    float *batch_target = calloc(batch_size * TEST_IMAGE_SIZE, sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < TEST_IMAGE_SIZE; i++) {
            batch_input[b * TEST_IMAGE_SIZE + i] = 0.5f + (rand() % 100) / 1000.0f;
            batch_target[b * TEST_IMAGE_SIZE + i] = 0.6f + (rand() % 100) / 1000.0f;
        }
    }
    
    /* Warmup */
    for (int i = 0; i < 5; i++) {
        cnn_train_step(cnn, batch_input, batch_target, batch_size);
    }
    
    /* Benchmark */
    double start = get_time_ms();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cnn_train_step(cnn, batch_input, batch_target, batch_size);
    }
    double end = get_time_ms();
    
    double avg_time = (end - start) / NUM_ITERATIONS;
    double throughput = (batch_size * 1000.0) / avg_time;
    
    printf("%-20s: %7.2f ms/batch  %8.1f img/s  (batch=%d, %dx%d)\n", 
           name, avg_time, throughput, batch_size, TEST_WIDTH, TEST_HEIGHT);
    
    free(batch_input);
    free(batch_target);
    cnn_destroy(cnn);
}

int main() {
    printf("=== Loss Function Benchmark ===\n");
    printf("Image size: %dx%d, Iterations: %d\n\n", TEST_WIDTH, TEST_HEIGHT, NUM_ITERATIONS);
    
    int batch_size = 4;
    
    benchmark_loss("MAE Loss", LOSS_MAE, batch_size);
    benchmark_loss("MSE Loss", LOSS_MSE, batch_size);
    benchmark_loss("Laplace Loss", LOSS_LAPLACE, batch_size);
    benchmark_loss("Color Variance", LOSS_COLOR_VARIANCE, batch_size);
    benchmark_loss("SSIM Loss", LOSS_SSIM, batch_size);
    
    printf("\n=== Combined Loss (All 5) ===\n");
    
    CNNConfig cfg = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
    cfg.max_batch_size = batch_size;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;
    
    cfg.loss_config.num_losses = 3;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.loss_config.types[1] = LOSS_MSE;
    cfg.loss_config.weights[1] = 0.5f;
    cfg.loss_config.types[2] = LOSS_LAPLACE;
    cfg.loss_config.weights[2] = 0.1f;
    cfg.loss_config.types[1] = LOSS_COLOR_VARIANCE;
    cfg.loss_config.weights[1] = 0.05f;
    cfg.loss_config.types[2] = LOSS_SSIM;
    cfg.loss_config.weights[2] = 0.5f;
    cfg.input_height = 32;
    cfg.input_width = 32;

    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 4, 1, -1, "layer1"});
    cnn_add_layer(cnn, (LayerConfig){4, 4, 0, -1, "output"});
    cnn_finalize(cnn);
    
    float *batch_input = calloc(batch_size * 32 * 32 * TEST_CHANNELS, sizeof(float));
    float *batch_target = calloc(batch_size * 32 * 32 * TEST_CHANNELS, sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < 32 * 32 * TEST_CHANNELS; i++) {
            batch_input[b * 32 * 32 * TEST_CHANNELS + i] = 0.5f;
            batch_target[b * 32 * 32 * TEST_CHANNELS + i] = 0.6f;
        }
    }
    
    for (int i = 0; i < 5; i++) {
        cnn_train_step(cnn, batch_input, batch_target, batch_size);
    }
    
    double start = get_time_ms();
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        cnn_train_step(cnn, batch_input, batch_target, batch_size);
    }
    double end = get_time_ms();
    
    double avg_time = (end - start) / NUM_ITERATIONS;
    double throughput = (batch_size * 1000.0) / avg_time;
    
    printf("All 5 losses       : %7.2f ms/batch  %8.1f img/s\n", avg_time, throughput);
    
    free(batch_input);
    free(batch_target);
    cnn_destroy(cnn);
    
    return 0;
}
