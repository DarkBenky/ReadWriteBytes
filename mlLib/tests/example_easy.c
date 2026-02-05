/* Comprehensive example showing all features:
 * - Optimizer selection (SGD vs Adam)
 * - Loss configuration (single and multi-loss)
 * - Layer architecture
 * - Residual mode
 * - Training loop
 */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

void print_section(const char* title) {
    printf("\n========================================\n");
    printf("  %s\n", title);
    printf("========================================\n\n");
}

int main() {
    print_section("CNN Denoiser - Complete Example");
    
    /* ===========================================
     * PART 1: Basic Configuration
     * =========================================== */
    printf("1. CREATE CONFIGURATION\n");
    printf("   CNNConfig cfg = cnn_default_config(256, 256, 4);\n\n");
    
    CNNConfig cfg = cnn_default_config(256, 256, 4);
    
    /* ===========================================
     * PART 2: Choose Optimizer
     * =========================================== */
    printf("2. SELECT OPTIMIZER\n");
    printf("   Options:\n");
    printf("   - OPTIMIZER_SGD:  Simple, needs small LR (~0.0001)\n");
    printf("   - OPTIMIZER_ADAM: Adaptive, works with LR ~0.001\n\n");
    printf("   cfg.optimizer = OPTIMIZER_ADAM;\n");
    printf("   cfg.learning_rate = 0.001f;\n\n");
    
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.001f;
    
    /* Optional: Configure Adam parameters */
    cfg.adam_beta1 = 0.9f;   /* Momentum decay (default) */
    cfg.adam_beta2 = 0.999f; /* RMSprop decay (default) */
    cfg.adam_epsilon = 1e-8f; /* Numerical stability */
    
    /* ===========================================
     * PART 3: Configure Loss Function
     * =========================================== */
    printf("3. CONFIGURE LOSS FUNCTION\n");
    printf("   Available losses:\n");
    printf("   - LOSS_MAE:     Mean Absolute Error (L1)\n");
    printf("   - LOSS_MSE:     Mean Squared Error (L2)\n");
    printf("   - LOSS_LAPLACE: Edge preservation\n\n");
    printf("   Multi-loss example (MAE + edge preservation):\n");
    printf("   cfg.loss_config.num_losses = 2;\n");
    printf("   cfg.loss_config.types[0] = LOSS_MAE;\n");
    printf("   cfg.loss_config.weights[0] = 1.0f;\n");
    printf("   cfg.loss_config.types[1] = LOSS_LAPLACE;\n");
    printf("   cfg.loss_config.weights[1] = 0.1f;\n\n");
    
    cfg.loss_config.num_losses = 2;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.loss_config.types[1] = LOSS_LAPLACE;
    cfg.loss_config.weights[1] = 0.1f;
    
    /* ===========================================
     * PART 4: Residual Mode (Optional)
     * =========================================== */
    printf("4. ENABLE RESIDUAL MODE\n");
    printf("   Predict noise instead of clean image (faster convergence)\n");
    printf("   cfg.residual_mode = 1;\n\n");
    
    cfg.residual_mode = 1;
    
    /* ===========================================
     * PART 5: Build Network Architecture
     * =========================================== */
    printf("5. BUILD NETWORK LAYERS\n");
    printf("   Encoder-decoder architecture:\n");
    printf("   4 -> 16 -> 32 -> 16 -> 4 channels\n\n");
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Encoder (compress) */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "encoder_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "encoder_2"});
    
    /* Decoder (expand) */
    cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "decoder_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "decoder_2"});
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    
    /* ===========================================
     * PART 6: Prepare Training Data
     * =========================================== */
    printf("\n6. PREPARE TRAINING DATA\n");
    
    int size = 256 * 256 * 4;
    float *noisy_image = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    
    /* Generate synthetic data */
    srand(42);
    for (int i = 0; i < size; i++) {
        float clean = 0.5f;
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        noisy_image[i] = clean + noise;
        
        /* In residual mode: target is the noise
         * In direct mode: target is the clean image */
        target[i] = cfg.residual_mode ? noise : clean;
    }
    
    /* ===========================================
     * PART 7: Training Loop
     * =========================================== */
    printf("\n7. TRAINING LOOP\n");
    printf("   Training for 50 epochs...\n\n");
    
    for (int epoch = 0; epoch < 50; epoch++) {
        float loss = cnn_train_step(cnn, noisy_image, target, 1);
        
        if (epoch % 10 == 0) {
            printf("   Epoch %3d: Loss = %.6f\n", epoch, loss);
        }
        
        /* Learning rate scheduling */
        if (epoch == 30) {
            cnn_set_learning_rate(cnn, 0.0005f);
            printf("   -> Reduced learning rate to 0.0005\n");
        }
    }
    
    /* ===========================================
     * PART 8: Inference (Denoising)
     * =========================================== */
    printf("\n8. INFERENCE (DENOISING)\n");
    
    float *denoised = malloc(size * sizeof(float));
    cnn_denoise(cnn, noisy_image, denoised, 1);
    
    printf("   Denoised image ready!\n");
    
    /* ===========================================
     * PART 9: Tips and Variations
     * =========================================== */
    print_section("Tips and Variations");
    
    printf("Switch to SGD optimizer:\n");
    printf("  cfg.optimizer = OPTIMIZER_SGD;\n");
    printf("  cfg.learning_rate = 0.0001f;  // Smaller LR for SGD\n\n");
    
    printf("Use MSE loss instead:\n");
    printf("  cfg.loss_config.num_losses = 1;\n");
    printf("  cfg.loss_config.types[0] = LOSS_MSE;\n");
    printf("  cfg.loss_config.weights[0] = 1.0f;\n\n");
    
    printf("Combine all three losses:\n");
    printf("  cfg.loss_config.num_losses = 3;\n");
    printf("  cfg.loss_config.types[0] = LOSS_MAE;\n");
    printf("  cfg.loss_config.weights[0] = 1.0f;\n");
    printf("  cfg.loss_config.types[1] = LOSS_MSE;\n");
    printf("  cfg.loss_config.weights[1] = 0.3f;\n");
    printf("  cfg.loss_config.types[2] = LOSS_LAPLACE;\n");
    printf("  cfg.loss_config.weights[2] = 0.1f;\n\n");
    
    printf("Deeper network:\n");
    printf("  cnn_add_layer(cnn, (LayerConfig){4, 16, 1, \"enc1\"});\n");
    printf("  cnn_add_layer(cnn, (LayerConfig){16, 32, 1, \"enc2\"});\n");
    printf("  cnn_add_layer(cnn, (LayerConfig){32, 64, 1, \"bottleneck\"});\n");
    printf("  cnn_add_layer(cnn, (LayerConfig){64, 32, 1, \"dec1\"});\n");
    printf("  cnn_add_layer(cnn, (LayerConfig){32, 16, 1, \"dec2\"});\n");
    printf("  cnn_add_layer(cnn, (LayerConfig){16, 4, 1, \"dec3\"});\n\n");
    
    /* Cleanup */
    free(noisy_image);
    free(target);
    free(denoised);
    cnn_destroy(cnn);
    
    print_section("Example Complete!");
    printf("See benchmark.c for performance measurements\n");
    printf("See test_features.c for correctness validation\n\n");
    
    return 0;
}
