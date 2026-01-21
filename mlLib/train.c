#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
    
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.002f;

    cfg.adam_beta1 = 0.95f;
    cfg.adam_beta2 = 0.999f;
    cfg.adam_epsilon = 1e-8f;
    
    cfg.loss_config.num_losses = 2;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.loss_config.types[1] = LOSS_LAPLACE;
    cfg.loss_config.weights[1] = 0.1f;
    
    cfg.residual_mode = 1;

    LearningRateDecay lr_decay;
    learning_rate_decay_init(&lr_decay, cfg.learning_rate, 0.9f, 1000);
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Encoder (compress) */
    cnn_add_layer(cnn, (LayerConfig){4, 8, 1, "encoder_1"});
    cnn_add_layer(cnn, (LayerConfig){8, 12, 1, "encoder_2"});
    cnn_add_layer(cnn, (LayerConfig){12, 16, 1, "encoder_3"});
    cnn_add_layer(cnn, (LayerConfig){16, 20, 1, "encoder_4"});
    cnn_add_layer(cnn, (LayerConfig){20, 24, 1, "encoder_5"});
    /* Decoder (expand) */
    cnn_add_layer(cnn, (LayerConfig){24, 20, 1, "decoder_1"});
    cnn_add_layer(cnn, (LayerConfig){20, 12, 1, "decoder_2"});
    cnn_add_layer(cnn, (LayerConfig){12, 4, 1, "decoder_3"});
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    

    DataLoader *loader = malloc(sizeof(DataLoader));
    fillDataLoader(loader, "/media/user/2TB Clear/imageData");

    ImageSample *sample = malloc(sizeof(ImageSample));
    
    printf("\nWarming up GPU (5 iterations)...\n");
    for (int i = 0; i < 5; i++) {
        getNextImagePair(loader, sample);
        cnn_train_step(cnn, sample->lowRes, sample->highRes, 1);
    }
    printf("Warmup complete.\n\n");
    

    cnn_reset_timing_stats(cnn);
    
    TimingStats cumulative_stats = {0};
    int timing_samples = 0;
    float accumulated_loss = 0.0f;
    for (int epoch = 0; epoch < 50; epoch++) {
        getNextImagePair(loader, sample);
        accumulated_loss += cnn_train_step(cnn, sample->lowRes, sample->highRes, 1);
        
        TimingStats stats;
        cnn_get_timing_stats(cnn, &stats);
        cumulative_stats.forward_time_ms += stats.forward_time_ms;
        cumulative_stats.backward_time_ms += stats.backward_time_ms;
        cumulative_stats.loss_time_ms += stats.loss_time_ms;
        cumulative_stats.update_time_ms += stats.update_time_ms;
        cumulative_stats.total_time_ms += stats.total_time_ms;
        timing_samples++;
        
        if (epoch % 10 == 0) {
            printf("   Epoch %3d: Loss = %.6f, LR = %.6f, Time = %.2f ms\n", epoch, accumulated_loss / 10, cnn_get_learning_rate(cnn), stats.total_time_ms);
            accumulated_loss = 0.0f;
        }

        float new_lr = learning_rate_decay_get(&lr_decay, epoch);
        cnn_set_learning_rate(cnn, new_lr);
    }

    printf("\n=== Training Performance (Averaged over %d iterations) ===\n", timing_samples);
    printf("Forward time:  %.3f ms (avg)\n", cumulative_stats.forward_time_ms / timing_samples);
    printf("Backward time: %.3f ms (avg)\n", cumulative_stats.backward_time_ms / timing_samples);
    printf("Loss time:     %.3f ms (avg)\n", cumulative_stats.loss_time_ms / timing_samples);
    printf("Update time:   %.3f ms (avg)\n", cumulative_stats.update_time_ms / timing_samples);
    printf("Total time:    %.3f ms (avg)\n", cumulative_stats.total_time_ms / timing_samples);
    printf("Throughput:    %.2f images/sec\n", 1000.0 / (cumulative_stats.total_time_ms / timing_samples));
    
    printf("\nSaving weights...\n");
    cnn_save_weights(cnn, "cnn_weights.bin");

    cnn_destroy(cnn);
    free(loader);
    free(sample);
    return 0;
}
