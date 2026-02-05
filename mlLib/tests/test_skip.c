#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(void) {
    printf("=== Testing Skip Connections ===\n\n");
    
    /* Small test configuration */
    CNNConfig cfg = cnn_default_config(64, 64, 4);
    cfg.learning_rate = 0.001f;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.adam_beta1 = 0.9f;
    cfg.adam_beta2 = 0.999f;
    cfg.adam_epsilon = 1e-8f;
    cfg.residual_mode = 0;  /* Direct prediction mode for testing */
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Simple architecture with skip connections:
     * Layer 0: 4->8
     * Layer 1: 8->8 (skip from 0)
     * Layer 2: 8->4
     */
    printf("Creating architecture with skip connection...\n");
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, -1, "layer0"});
    cnn_add_layer(cnn, (LayerConfig){8, 8, 1, 0, "layer1_skip0"});  /* Skip from layer 0 */
    cnn_add_layer(cnn, (LayerConfig){8, 4, 0, -1, "output"});
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    
    /* Create test data */
    int size = 64 * 64 * 4;
    float *input = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    
    srand(42);
    for (int i = 0; i < size; i++) {
        input[i] = (float)rand() / RAND_MAX;
        target[i] = (float)rand() / RAND_MAX;
    }
    
    printf("\nRunning forward pass...\n");
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    float loss = cnn_train_step(cnn, input, target, 1);
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double train_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
                      (t_end.tv_nsec - t_start.tv_nsec) / 1e6;
    
    TimingStats stats;
    cnn_get_timing_stats(cnn, &stats);
    printf("  Forward pass: %.2f ms\n", stats.forward_time_ms);
    printf("  Backward pass: %.2f ms\n", stats.backward_time_ms);
    printf("  Loss: %.6f\n", loss);
    
    /* Check output */
    float *output = malloc(size * sizeof(float));
    cnn_get_output(cnn, output);
    double out_mean = 0.0, out_min = 1e9, out_max = -1e9;
    for (int i = 0; i < size; i++) {
        out_mean += output[i];
        if (output[i] < out_min) out_min = output[i];
        if (output[i] > out_max) out_max = output[i];
    }
    out_mean /= size;
    printf("  Output stats: mean=%.4f, min=%.4f, max=%.4f\n", out_mean, out_min, out_max);
    
    /* Second iteration to see if weights are updating */
    printf("\nRunning second iteration...\n");
    cnn_train_step(cnn, input, target, 1);
    float *output2 = malloc(size * sizeof(float));
    cnn_get_output(cnn, output2);
    
    /* Check if output changed */
    double diff_sum = 0.0;
    for (int i = 0; i < size; i++) {
        diff_sum += fabs(output2[i] - output[i]);
    }
    printf("  Output change after update: %.6f\n", diff_sum / size);
    
    if (stats.backward_time_ms < 1.0) {
        printf("\n❌ FAILURE: Backward pass too fast (%.2f ms) - gradients not computing!\n", stats.backward_time_ms);
        return 1;
    }
    
    if (diff_sum / size < 1e-6) {
        printf("\n❌ FAILURE: Output didn't change after update - weights not updating!\n");
        return 1;
    }
    
    printf("\n✅ SUCCESS: Skip connections working correctly!\n");
    printf("  - Forward pass: %.2f ms\n", stats.forward_time_ms);
    printf("  - Backward pass: %.2f ms\n", stats.backward_time_ms);
    printf("  - Gradients computed and weights updated\n");
    
    free(input);
    free(target);
    free(output);
    free(output2);
    cnn_destroy(cnn);
    
    return 0;
}
