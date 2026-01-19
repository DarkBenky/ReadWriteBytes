/* Quick test to debug training */
#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    CNNConfig config = {
        .input_width = 64,
        .input_height = 64,
        .input_channels = 4,
        .output_channels = 4,
        .learning_rate = 0.00001f,
        .use_profiling = 0
    };

    LearningRateDecay lr_decay;
    learning_rate_decay_init(&lr_decay, 0.00001f, 0.95f, 1000);
    
    CNNDenoiser *cnn = cnn_create(config);
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, "test"});
    cnn_add_layer(cnn, (LayerConfig){8, 16, 1, "test2"});
    cnn_add_layer(cnn, (LayerConfig){16, 32, 1, "test2"});
    cnn_add_layer(cnn, (LayerConfig){32, 4, 0, "output"});
    cnn_finalize(cnn);

    cnn_print_architecture(cnn);
    
    int size = 64 * 64 * 4;
    float *input = malloc(size * sizeof(float));
    float *target = malloc(size * sizeof(float));
    
    /* Simple test: input is 0.5, target is 0.7 */
    for (int i = 0; i < size; i++) {
        input[i] = 0.5f;
        target[i] = 0.7f;
    }
    
    printf("Small network test (64x64x4):\n");
    printf("Input: all 0.5, Target: all 0.7\n\n");
    
    for (int iter = 0; iter < 1000; iter++) {
        float loss = cnn_train_step(cnn, input, target, 1);
        float adjusted_lr = learning_rate_decay_get(&lr_decay, iter);
        cnn_set_learning_rate(cnn, adjusted_lr);
        printf("Iter %d: Loss = %.6f, Learning Rate = %.8f\n", iter, loss, adjusted_lr);
    }
    
    free(input);
    free(target);
    cnn_destroy(cnn);
    return 0;
}
