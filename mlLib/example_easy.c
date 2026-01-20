/* Easy-to-use example: Image denoising with all features */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("====================================\n");
    printf("  CNN Image Denoiser - Easy Example\n");
    printf("====================================\n\n");
    
    /* Step 1: Create configuration with default values */
    CNNConfig cfg = cnn_default_config(256, 256, 4);
    
    /* Step 2: Enable advanced features */
    cfg.residual_mode = 1;              /* Predict noise instead of clean image */
    cfg.optimizer = OPTIMIZER_ADAM;     /* Use Adam optimizer */
    cfg.learning_rate = 0.001f;
    
    /* Step 3: Configure multi-loss for edge preservation */
    cfg.loss_config.num_losses = 2;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;       /* Main loss */
    cfg.loss_config.types[1] = LOSS_LAPLACE;
    cfg.loss_config.weights[1] = 0.1f;       /* Edge preservation */
    
    /* Step 4: Create network */
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Step 5: Build encoder-decoder architecture */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "encoder_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "encoder_2"});
    cnn_add_layer(cnn, (LayerConfig){32, 16, 1, "decoder_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 1, "decoder_2"});
    cnn_finalize(cnn);
    
    cnn_print_architecture(cnn);
    
    /* Step 6: Prepare training data */
    int size = 256 * 256 * 4;
    float *noisy_image = malloc(size * sizeof(float));
    float *noise_target = malloc(size * sizeof(float));
    
    /* Generate synthetic noisy image */
    srand(42);
    for (int i = 0; i < size; i++) {
        float clean = 0.5f;  /* Original clean value */
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        noisy_image[i] = clean + noise;
        noise_target[i] = noise;  /* In residual mode, target is the noise */
    }
    
    /* Step 7: Train the network */
    printf("Training progress:\n");
    for (int epoch = 0; epoch < 100; epoch++) {
        float loss = cnn_train_step(cnn, noisy_image, noise_target, 1);
        
        if (epoch % 10 == 0) {
            printf("  Epoch %3d: Loss = %.6f\n", epoch, loss);
        }
        
        /* Optional: Adjust learning rate */
        if (epoch == 50) {
            cnn_set_learning_rate(cnn, 0.0005f);  /* Reduce learning rate */
        }
    }
    
    printf("\nTraining complete!\n\n");
    
    /* Step 8: Show how to switch optimizers */
    printf("To use SGD instead of Adam:\n");
    printf("  cfg.optimizer = OPTIMIZER_SGD;\n");
    printf("  cfg.learning_rate = 0.0001f;  // SGD needs smaller LR\n\n");
    
    /* Step 9: Show how to use different losses */
    printf("Available loss functions:\n");
    printf("  - LOSS_MAE: Mean Absolute Error (L1)\n");
    printf("  - LOSS_MSE: Mean Squared Error (L2)\n");
    printf("  - LOSS_LAPLACE: Laplacian edge loss\n\n");
    
    printf("Example multi-loss configuration:\n");
    printf("  cfg.loss_config.num_losses = 3;\n");
    printf("  cfg.loss_config.types[0] = LOSS_MAE;\n");
    printf("  cfg.loss_config.weights[0] = 1.0f;\n");
    printf("  cfg.loss_config.types[1] = LOSS_LAPLACE;\n");
    printf("  cfg.loss_config.weights[1] = 0.1f;\n");
    printf("  cfg.loss_config.types[2] = LOSS_MSE;\n");
    printf("  cfg.loss_config.weights[2] = 0.05f;\n\n");
    
    /* Cleanup */
    free(noisy_image);
    free(noise_target);
    cnn_destroy(cnn);
    
    printf("====================================\n");
    printf("  Example complete!\n");
    printf("====================================\n");
    
    return 0;
}
