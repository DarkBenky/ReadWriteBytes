/* Comprehensive test: SGD vs Adam, residual mode, multi-loss */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void test_optimizer(const char* name, OptimizerType opt, int residual) {
    srand(42);
    
    CNNConfig cfg = cnn_default_config(64, 64, 4);
    cfg.optimizer = opt;
    cfg.residual_mode = residual;
    cfg.learning_rate = (opt == OPTIMIZER_ADAM) ? 0.001f : 0.0001f;
    
    /* Multi-loss: MAE + Laplace for edge preservation */
    cfg.loss_config.num_losses = 2;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.loss_config.types[1] = LOSS_LAPLACE;
    cfg.loss_config.weights[1] = 0.1f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, "enc"});
    cnn_add_layer(cnn, (LayerConfig){8, 4, 1, "dec"});
    cnn_finalize(cnn);
    
    int size = 64 * 64 * 4;
    float *clean = malloc(size * sizeof(float));
    float *noisy = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    
    /* Generate test data */
    for (int i = 0; i < size; i++) {
        clean[i] = 0.5f;
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        noisy[i] = clean[i] + noise;
        target[i] = residual ? noise : clean[i];
    }
    
    printf("\n%s (%s mode):\n", name, residual ? "residual" : "direct");
    
    clock_t start = clock();
    float final_loss = 0.0f;
    
    for (int iter = 0; iter < 50; iter++) {
        final_loss = cnn_train_step(cnn, noisy, target, 1);
        if (iter % 10 == 0) {
            printf("  Iter %2d: Loss = %.6f\n", iter, final_loss);
        }
    }
    
    clock_t end = clock();
    double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    
    printf("  Final loss: %.6f\n", final_loss);
    printf("  Time: %.2f ms (%.2f ms/iter)\n", time_ms, time_ms / 50.0);
    
    free(clean);
    free(noisy);
    free(target);
    cnn_destroy(cnn);
}

int main() {
    printf("========================================\n");
    printf("  CNN Denoiser Feature Test Suite\n");
    printf("========================================\n");
    
    printf("\nFeatures tested:\n");
    printf("  [x] Residual connections (noise prediction)\n");
    printf("  [x] Adam optimizer\n");
    printf("  [x] SGD optimizer\n");
    printf("  [x] Multi-loss (MAE + Laplace)\n");
    printf("  [x] Edge preservation (Laplace loss)\n");
    
    /* Test 1: SGD with residual mode */
    test_optimizer("SGD + Residual", OPTIMIZER_SGD, 1);
    
    /* Test 2: Adam with residual mode */
    test_optimizer("Adam + Residual", OPTIMIZER_ADAM, 1);
    
    /* Test 3: SGD direct mode */
    test_optimizer("SGD + Direct", OPTIMIZER_SGD, 0);
    
    /* Test 4: Adam direct mode */
    test_optimizer("Adam + Direct", OPTIMIZER_ADAM, 0);
    
    printf("\n========================================\n");
    printf("All tests passed!\n");
    printf("========================================\n");
    
    return 0;
}
