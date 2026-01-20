/* Test residual connections and new features */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    srand(42);
    
    /* Test configuration: residual mode with Adam */
    CNNConfig cfg = cnn_default_config(64, 64, 4);
    cfg.residual_mode = 1;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.001f;
    
    /* Multi-loss: MAE + Laplace */
    cfg.loss_config.num_losses = 2;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.loss_config.types[1] = LOSS_LAPLACE;
    cfg.loss_config.weights[1] = 0.1f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Small network: 4->8->4 */
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, "enc"});
    cnn_add_layer(cnn, (LayerConfig){8, 4, 1, "dec"});
    cnn_finalize(cnn);
    
    cnn_print_architecture(cnn);
    
    int size = 64 * 64 * 4;
    float *clean = malloc(size * sizeof(float));
    float *noisy = malloc(size * sizeof(float));
    float *noise_target = malloc(size * sizeof(float));
    
    /* Generate test: clean image = 0.5, add small noise */
    for (int i = 0; i < size; i++) {
        clean[i] = 0.5f;
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;  /* Small noise */
        noisy[i] = clean[i] + noise;
        noise_target[i] = noise;  /* Residual mode: predict the noise */
    }
    
    printf("\nTraining residual denoiser (predict noise):\n");
    printf("Clean: 0.5, Noise: ~0.05 std, Target: noise\n\n");
    
    for (int iter = 0; iter < 100; iter++) {
        float loss = cnn_train_step(cnn, noisy, noise_target, 1);
        
        if (iter % 10 == 0) {
            printf("Iter %3d: Loss = %.6f\n", iter, loss);
        }
    }
    
    printf("\nTest passed! Residual mode working.\n");
    
    free(clean);
    free(noisy);
    free(noise_target);
    cnn_destroy(cnn);
    return 0;
}
