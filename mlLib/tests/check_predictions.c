#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    printf("=== Checking Prediction Differences ===\n\n");
    
    CNNConfig cfg = cnn_default_config(800, 600, 4);
    cfg.learning_rate = 0.001f;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.adam_beta1 = 0.95f;
    cfg.adam_beta2 = 0.999f;
    cfg.adam_epsilon = 1e-8f;
    cfg.residual_mode = 1;  /* Test residual mode */
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Same architecture as train.c */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, -1, "encode_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 20, 1, -1, "encode_2"});
    cnn_add_layer(cnn, (LayerConfig){20, 24, 1, -1, "bottleneck"});
    cnn_add_layer(cnn, (LayerConfig){24, 20, 1, 1, "decode_1_skip1"});
    cnn_add_layer(cnn, (LayerConfig){20, 16, 1, 0, "decode_2_skip0"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 0, -1, "output"});
    
    cnn_finalize(cnn);
    
    int size = 800 * 600 * 4;
    float *input = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    float *output = malloc(size * sizeof(float));
    
    /* Create test pattern with clear differences */
    printf("Setting up test data:\n");
    for (int i = 0; i < size; i++) {
        input[i] = 0.5f;   /* Noisy input */
        /* In residual mode, we pass NOISE as target, not clean image */
    }
    
    /* Test with residual mode */
    printf("\n=== TEST: Residual Mode ===\n");
    
    /* Compute noise target: noise = input - clean */
    /* If clean image is 0.3, and input is 0.5, then noise = 0.5 - 0.3 = 0.2 */
    float clean_value = 0.3f;
    float noise_value = 0.5f - clean_value;  /* = 0.2 */
    
    for (int i = 0; i < size; i++) {
        target[i] = noise_value;  /* Target is the noise to predict */
    }
    
    printf("  Input (noisy): all %.1f\n", input[0]);
    printf("  Clean (expected): all %.1f\n", clean_value);
    printf("  Target (noise): all %.1f\n", noise_value);
    printf("  Expected output = input - predicted_noise\n");
    printf("  If network predicts zero → output = %.1f (BAD: same as input)\n", input[0]);
    printf("  If network learns → output ≈ %.1f (GOOD: clean image)\n", clean_value);
    
    cnn_train_step(cnn, input, target, 1);
    cnn_get_output(cnn, output);
    
    /* Check if output == input (BAD) or output moving toward clean (GOOD) */
    double sum_to_input = 0.0, sum_to_clean = 0.0;
    int same_as_input = 0;
    
    for (int i = 0; i < size; i++) {
        sum_to_input += fabs(output[i] - input[i]);
        sum_to_clean += fabs(output[i] - clean_value);
        if (fabs(output[i] - input[i]) < 1e-7) same_as_input++;
    }
    
    printf("\nAfter 1 iteration:\n");
    printf("  |output - input|: %.6f\n", sum_to_input/size);
    printf("  |output - clean|: %.6f\n", sum_to_clean/size);
    printf("  Pixels same as input: %d / %d (%.1f%%)\n", 
           same_as_input, size, 100.0 * same_as_input / size);
    printf("  Average output value: %.6f (input=%.1f, clean=%.1f)\n",
           output[0], input[0], clean_value);
    
    printf("\n=== Training for 20 steps ===\n");
    for (int step = 0; step < 20; step++) {
        float loss = cnn_train_step(cnn, input, target, 1);
        
        if (step % 5 == 0) {
            cnn_get_output(cnn, output);
            
            sum_to_input = 0.0;
            sum_to_clean = 0.0;
            for (int i = 0; i < size; i++) {
                sum_to_input += fabs(output[i] - input[i]);
                sum_to_clean += fabs(output[i] - clean_value);
            }
            
            printf("  Step %2d: Loss=%.6f, |out-input|=%.6f, |out-clean|=%.6f, output[0]=%.6f\n",
                   step, loss, sum_to_input/size, sum_to_clean/size, output[0]);
        }
    }
    
    cnn_get_output(cnn, output);
    
    /* Final check */
    sum_to_input = 0.0;
    sum_to_clean = 0.0;
    same_as_input = 0;
    
    for (int i = 0; i < size; i++) {
        sum_to_input += fabs(output[i] - input[i]);
        sum_to_clean += fabs(output[i] - clean_value);
        if (fabs(output[i] - input[i]) < 1e-7) same_as_input++;
    }
    
    printf("\nFinal results after 20 iterations:\n");
    printf("  |output - input|: %.6f\n", sum_to_input/size);
    printf("  |output - clean|: %.6f\n", sum_to_clean/size);
    printf("  Identical to input: %d / %d (%.1f%%)\n", 
           same_as_input, size, 100.0 * same_as_input / size);
    
    /* Diagnosis */
    printf("\n=== DIAGNOSIS ===\n");
    if (same_as_input > size * 0.95) {
        printf("❌ CRITICAL BUG: Output is identical to input!\n");
        printf("   In residual mode with noise target = %.2f:\n", noise_value);
        printf("   - Network should predict noise ≈ %.2f\n", noise_value);
        printf("   - Output should be: input - predicted_noise = %.1f - %.1f = %.1f\n", 
               input[0], noise_value, clean_value);
        printf("   - But output = %.6f (same as input!)\n", output[0]);
        printf("\n   Possible causes:\n");
        printf("   1. cnn_get_output() not applying residual mode correctly\n");
        printf("   2. Network always outputs zero (not learning)\n");
        printf("   3. Residual subtraction kernel has a bug\n");
    } else if (sum_to_clean/size < 0.05) {
        printf("✅ EXCELLENT: Output very close to clean target!\n");
        printf("   Residual mode working correctly.\n");
    } else if (sum_to_clean/size < sum_to_input/size) {
        printf("✅ GOOD: Output moving toward clean!\n");
        printf("   |out-clean| < |out-input| means residual mode is working.\n");
    } else {
        printf("⚠️  PROBLEM: Output not moving toward clean.\n");
        printf("   Residual mode may not be working correctly.\n");
    }
    
    free(input);
    free(target);
    free(output);
    cnn_destroy(cnn);
    
    return 0;
}
