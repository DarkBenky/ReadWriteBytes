#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    CNNConfig cfg = cnn_default_config(128, 128, 4);
    cfg.auto_tune_workgroup = 1;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, "test"});
    cnn_finalize(cnn);
    
    int size = 128 * 128 * 4;
    float *input = malloc(size * sizeof(float));
    float *clean = malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        clean[i] = 0.5f;
        input[i] = 0.5f;
    }
    
    printf("Running first training step with auto-tuning...\n");
    cnn_train_step(cnn, input, clean, 1);
    
    printf("\nAuto-tuning complete!\n");
    
    free(input);
    free(clean);
    cnn_destroy(cnn);
    return 0;
}
