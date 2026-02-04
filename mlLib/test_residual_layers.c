#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

/* Example usage of the new residual layer architecture
 * 
 * Pipeline:
 * Input (noisy) → 
 * RESIDUAL_INPUT (save input) → 
 * CNN layers (predict noise) → 
 * RESIDUAL_SUBTRACT (input - noise = denoised) → 
 * CNN layers (refine denoised) → 
 * Output (final)
 */

int main(void) {
    printf("=== Testing New Residual Layer Architecture ===\n\n");
    
    CNNConfig cfg = cnn_default_config(800, 600, 4);
    cfg.learning_rate = 0.001f;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.adam_beta1 = 0.95f;
    cfg.adam_beta2 = 0.999f;
    cfg.residual_mode = 0;  /* Use new layer-based residual instead */
    
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Build network with explicit residual layers
     * 
     * Architecture:
     * 1. Input (4 channels) - noisy image
     * 2. ResidualInput - saves the noisy input for later
     * 3. Conv layers - predict the noise
     * 4. ResidualSubtract - compute (saved_input - noise_prediction) = denoised
     * 5. Conv layers - refine the denoised image
     * 6. Output
     */
    
    printf("Building network architecture:\n");
    printf("  Layer 0: Input (noisy image)\n");
    
    /* Save input for residual subtraction */
    printf("  Layer 1: RESIDUAL_INPUT (save input)\n");
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_RESIDUAL_INPUT,
        .cin = 4,
        .cout = 4,
        .use_relu = 0,
        .skip_from = -1,
        .residual_from = -1,
        .name = "save_input"
    });
    
    /* First branch: predict noise */
    printf("  Layer 2-4: CNN layers to predict noise\n");
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_CONV,
        .cin = 4,
        .cout = 16,
        .use_relu = 1,
        .skip_from = -1,
        .residual_from = -1,
        .name = "noise_pred_1"
    });
    
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_CONV,
        .cin = 16,
        .cout = 16,
        .use_relu = 1,
        .skip_from = -1,
        .residual_from = -1,
        .name = "noise_pred_2"
    });
    
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_CONV,
        .cin = 16,
        .cout = 4,
        .use_relu = 0,
        .skip_from = -1,
        .residual_from = -1,
        .name = "noise_output"
    });
    
    /* Subtract noise from saved input: denoised = input - noise */
    printf("  Layer 5: RESIDUAL_SUBTRACT (input - noise)\n");
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_RESIDUAL_SUBTRACT,
        .cin = 4,
        .cout = 4,
        .use_relu = 0,
        .skip_from = -1,
        .residual_from = 1,  /* Reference layer 1 (save_input) */
        .name = "denoise"
    });
    
    /* Second branch: refine denoised image */
    printf("  Layer 6-8: CNN layers to refine denoised image\n");
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_CONV,
        .cin = 4,
        .cout = 12,
        .use_relu = 1,
        .skip_from = -1,
        .residual_from = -1,
        .name = "refine_1"
    });
    
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_CONV,
        .cin = 12,
        .cout = 8,
        .use_relu = 1,
        .skip_from = -1,
        .residual_from = -1,
        .name = "refine_2"
    });
    
    cnn_add_layer(cnn, (LayerConfig){
        .type = LAYER_CONV,
        .cin = 8,
        .cout = 4,
        .use_relu = 0,
        .skip_from = -1,
        .residual_from = -1,
        .name = "output"
    });
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    
    /* Test with simple data */
    int size = 800 * 600 * 4;
    float *noisy_input = malloc(size * sizeof(float));
    float *clean_target = malloc(size * sizeof(float));
    float *output = malloc(size * sizeof(float));
    
    /* Create test pattern: 
     * Input = 0.8 (noisy)
     * Clean = 0.5 (target)
     * Noise = 0.3 (what should be predicted in layer 4)
     */
    float input_val = 0.8f;
    float clean_val = 0.5f;
    
    for (int i = 0; i < size; i++) {
        noisy_input[i] = input_val;
        clean_target[i] = clean_val;
    }
    
    printf("\nTraining for 10 steps...\n");
    printf("Input (noisy): %.3f\n", input_val);
    printf("Target (clean): %.3f\n", clean_val);
    printf("Expected noise to predict: %.3f\n\n", input_val - clean_val);
    
    /* Train */
    for (int step = 0; step < 10; step++) {
        float loss = cnn_train_step(cnn, noisy_input, clean_target, 1);
        
        if (step % 5 == 0) {
            cnn_get_output(cnn, output);
            printf("Step %2d: Loss=%.6f, Output[0]=%.6f\n", step, loss, output[0]);
        }
    }
    
    cnn_get_output(cnn, output);
    printf("\nFinal output: %.6f (target: %.6f)\n", output[0], clean_val);
    printf("Difference: %.6f\n", fabs(output[0] - clean_val));
    
    /* Success if output is close to clean target */
    if (fabs(output[0] - clean_val) < 0.1f) {
        printf("\n✓ SUCCESS: Network learned the denoising task!\n");
    } else {
        printf("\n⚠ Network needs more training.\n");
    }
    
    free(noisy_input);
    free(clean_target);
    free(output);
    cnn_destroy(cnn);
    
    return 0;
}
