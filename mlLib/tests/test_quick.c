#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    CNNConfig cfg = cnn_default_config(64, 64, 4);
    cfg.learning_rate = 0.01f;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.residual_mode = 0;
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Simple residual architecture */
    cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 8, 1, -1, -1, "noise1"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 8, 4, 0, -1, -1, "noise2"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
    cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 4, 0, -1, -1, "output"});
    cnn_finalize(cnn);
    
    int size = 64 * 64 * 4;
    float *noisy = malloc(size * sizeof(float));
    float *clean = malloc(size * sizeof(float));
    float *output = malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        noisy[i] = 0.8f;
        clean[i] = 0.5f;
    }
    
    printf("Training 100 steps...\n");
    for (int step = 0; step < 100; step++) {
        float loss = cnn_train_step(cnn, noisy, clean, 1);
        if (step % 20 == 0) {
            cnn_get_output(cnn, output);
            printf("Step %3d: Loss=%.6f, Output=%.6f (target=0.5)\n", step, loss, output[0]);
        }
    }
    
    cnn_get_output(cnn, output);
    float diff = fabs(output[0] - 0.5f);
    printf("\nFinal: Output=%.6f, Error=%.6f\n", output[0], diff);
    printf(diff < 0.05f ? "SUCCESS\n" : "NEEDS MORE TRAINING\n");
    
    free(noisy); free(clean); free(output);
    cnn_destroy(cnn);
    return 0;
}
