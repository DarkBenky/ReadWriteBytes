#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("=== Debugging Residual Mode Output ===\n\n");
    
    CNNConfig cfg = cnn_default_config(800, 600, 4);
    cfg.learning_rate = 0.001f;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.adam_beta1 = 0.95f;
    cfg.adam_beta2 = 0.999f;
    cfg.residual_mode = 0;  /* Using new layer-based residual architecture */
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* New residual architecture with explicit layers */
    cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save_input"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 16, 1, -1, -1, "encode_1"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 20, 1, -1, -1, "encode_2"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 20, 24, 1, -1, -1, "bottleneck"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 24, 20, 1, 2, -1, "decode_1_skip2"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 20, 16, 1, 1, -1, "decode_2_skip1"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 4, 0, -1, -1, "noise_output"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 4, 0, -1, -1, "output"});
    
    cnn_finalize(cnn);
    
    int size = 800 * 600 * 4;
    float *input = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    float *output = malloc(size * sizeof(float));
    
    /* Create clear test pattern */
    printf("Creating test data:\n");
    float input_val = 0.8f;    /* Noisy input */
    float clean_val = 0.5f;    /* Clean target */
    float noise_val = input_val - clean_val;  /* = 0.3 */
    
    for (int i = 0; i < size; i++) {
        input[i] = input_val;
        target[i] = clean_val;  /* Target is clean image */
    }
    
    printf("  Input (noisy): %.3f\n", input_val);
    printf("  Clean (expected): %.3f\n", clean_val);
    printf("  Noise: %.3f (what network should internally predict)\n", noise_val);
    printf("  Expected output: %.3f\n\n", clean_val);
    
    /* Train for a few steps */
    printf("Training for 30 steps...\n");
    for (int step = 0; step < 30; step++) {
        cnn_train_step(cnn, input, target, 1);
        
        if (step % 10 == 0) {
            cnn_get_output(cnn, output);
            
            printf("Step %2d:\n", step);
            printf("  Output[0] = %.6f\n", output[0]);
            printf("  Output[1000] = %.6f\n", output[1000]);
            printf("  Output[5000] = %.6f\n", output[5000]);
            
            /* Check if output == input */
            int same_count = 0;
            for (int i = 0; i < 100; i++) {
                if (fabs(output[i] - input[i]) < 0.0001) same_count++;
            }
            printf("  Pixels identical to input (first 100): %d/100\n", same_count);
            
            /* Check distance to clean target */
            double dist_to_input = fabs(output[0] - input_val);
            double dist_to_clean = fabs(output[0] - clean_val);
            printf("  |output - input|: %.6f\n", dist_to_input);
            printf("  |output - clean|: %.6f\n", dist_to_clean);
            
            if (same_count > 95) {
                printf("  ERROR: OUTPUT IS IDENTICAL TO INPUT!\n");
            } else if (dist_to_clean < dist_to_input) {
                printf("  SUCCESS: Moving toward clean target\n");
            }
            printf("\n");
        }
    }
    
    cnn_get_output(cnn, output);
    
    printf("\n=== Final Analysis ===\n");
    printf("After 30 training steps:\n");
    printf("  Input:  %.6f\n", input_val);
    printf("  Clean:  %.6f\n", clean_val);
    printf("  Output: %.6f\n", output[0]);
    
    if (fabs(output[0] - input_val) < 0.001) {
        printf("\nERROR: Output equals input!\n");
        printf("   Residual architecture is NOT working.\n");
    } else if (fabs(output[0] - clean_val) < 0.05) {
        printf("\nSUCCESS: Output close to clean target!\n");
        printf("   New residual layer architecture is working correctly.\n");
    } else {
        printf("\nResidual architecture working but network needs more training.\n");
    }
    
    free(input);
    free(target);
    free(output);
    cnn_destroy(cnn);
    
    return 0;
}
