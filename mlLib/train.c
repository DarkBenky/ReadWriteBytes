#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_STEPS 100000
#define LOG_EVERY_N_STEPS 512
#define LOG_STEP 10


int main() {
    CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
    
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;  /* Higher LR for residual mode to escape zero-output trap */
    cfg.use_profiling = 1;

    cfg.adam_beta1 = 0.95f;
    cfg.adam_beta2 = 0.999f;
    cfg.adam_epsilon = 1e-8f;
    
    cfg.loss_config.num_losses = 3;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.55f;
    cfg.loss_config.types[1] = LOSS_MSE;
    cfg.loss_config.weights[1] = 1.05f;
    cfg.loss_config.types[2] = LOSS_COLOR_VARIANCE;
    cfg.loss_config.weights[2] = 0.20f;
    
    cfg.residual_mode = 0;

    LearningRateDecay lr_decay;
    learning_rate_decay_init(&lr_decay, cfg.learning_rate, 0.75f, MAX_STEPS);
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Architecture with skip connections */
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, -1, "encode_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 20, 1, -1, "encode_2"});
    cnn_add_layer(cnn, (LayerConfig){20, 24, 1, -1, "bottleneck"});
    cnn_add_layer(cnn, (LayerConfig){24, 20, 1, 1, "decode_1_skip1"});
    cnn_add_layer(cnn, (LayerConfig){20, 16, 1, 0, "decode_2_skip0"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 0, -1, "output"});
    
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);      

    DataLoader *loader = malloc(sizeof(DataLoader));
    fillDataLoader(loader, "/media/user/2TB Clear/imageData");

    ImageSample *sample = malloc(sizeof(ImageSample));
    float *prediction = malloc(IMAGE_SIZE * sizeof(float));
    float *prediction_interleaved = malloc(IMAGE_SIZE * sizeof(float));  /* For display */
    float *noise_target = malloc(IMAGE_SIZE * sizeof(float));  /* For residual mode: noise = noisy - clean */
    
    /* Buffers for converting interleaved input to planar format for GPU */
    float *input_planar = malloc(IMAGE_SIZE * sizeof(float));
    float *target_planar = malloc(IMAGE_SIZE * sizeof(float));
    
    printf("\nWarming up GPU (5 iterations)...\n");
    for (int i = 0; i < 5; i++) {
        getNextImagePair(loader, sample);
        interleavedToPlanar(sample->lowRes, input_planar, WIDTH, HEIGHT, 4);
        
        if (cfg.residual_mode) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                noise_target[j] = sample->lowRes[j] - sample->highRes[j];
            }
            interleavedToPlanar(noise_target, target_planar, WIDTH, HEIGHT, 4);
        } else {
            interleavedToPlanar(sample->highRes, target_planar, WIDTH, HEIGHT, 4);
        }
        
        cnn_train_step(cnn, input_planar, target_planar, 1);
    }
    printf("Warmup complete.\n\n");
    
    getNextImagePair(loader, sample);
    interleavedToPlanar(sample->lowRes, input_planar, WIDTH, HEIGHT, 4);
    
    if (cfg.residual_mode) {
        for (int i = 0; i < IMAGE_SIZE; i++) {
            noise_target[i] = sample->lowRes[i] - sample->highRes[i];
        }
        interleavedToPlanar(noise_target, target_planar, WIDTH, HEIGHT, 4);
    } else {
        interleavedToPlanar(sample->highRes, target_planar, WIDTH, HEIGHT, 4);
    }
    
    cnn_train_step(cnn, input_planar, target_planar, 1);    

    cnn_reset_timing_stats(cnn);
    
    TimingStats cumulative_stats = {0};
    int timing_samples = 0;
    float accumulated_loss = 0.0f;
    float best_loss = 1e10f;
    float mae_loss = 0.0f, mse_loss = 0.0f, laplace_loss = 0.0f, color_loss = 0.0f;
    float best_mae = 1e10f, best_mse = 1e10f, best_laplace = 1e10f, best_color = 1e10f;
    for (int step = 0; step < MAX_STEPS; step++) {
        getNextImagePair(loader, sample);
        
        /* Convert input from interleaved to planar format */
        interleavedToPlanar(sample->lowRes, input_planar, WIDTH, HEIGHT, 4);
        
        if (cfg.residual_mode) {
            for (int i = 0; i < IMAGE_SIZE; i++) {
                noise_target[i] = sample->lowRes[i] - sample->highRes[i];
            }
            interleavedToPlanar(noise_target, target_planar, WIDTH, HEIGHT, 4);
        } else {
            interleavedToPlanar(sample->highRes, target_planar, WIDTH, HEIGHT, 4);
        }
        
        accumulated_loss += cnn_train_step(cnn, input_planar, target_planar, 1);
        
        TimingStats stats;
        cnn_get_timing_stats(cnn, &stats);
        cumulative_stats.forward_time_ms += stats.forward_time_ms;
        cumulative_stats.backward_time_ms += stats.backward_time_ms;
        cumulative_stats.loss_time_ms += stats.loss_time_ms;
        cumulative_stats.update_time_ms += stats.update_time_ms;
        cumulative_stats.total_time_ms += stats.total_time_ms;
        timing_samples++;

        if (step % LOG_STEP == 0) {
            printf("Step %d: Forward %.2f ms, Backward %.2f ms, Loss %.2f ms, Update %.2f ms, Total %.2f ms\n",
                   step,
                   stats.forward_time_ms,
                   stats.backward_time_ms,
                   stats.loss_time_ms,
                   stats.update_time_ms,
                   stats.total_time_ms);
            printf("Current Loss: %.6f\n", accumulated_loss / (step + 1));
            cnn_get_individual_losses(cnn, &mae_loss, &mse_loss, &laplace_loss, &color_loss);
        }
        
        if (step % LOG_EVERY_N_STEPS == 0 && step > 0) {
            float avg_loss = accumulated_loss / LOG_EVERY_N_STEPS;
            printf("   Step %3d: Loss = %.6f, LR = %.6f, Time = %.2f ms\n, Step/s %.2f", step, avg_loss, cnn_get_learning_rate(cnn), stats.total_time_ms, 1000.0 / stats.total_time_ms);
            
            if (accumulated_loss < best_loss) {
                best_loss = accumulated_loss;
                printf("\n New best loss! \n");
                cnn_save_weights(cnn, "cnn_weights_best.bin");
            }
            
            printf("   Individual losses: MAE=%.4f MSE=%.4f Color=%.4f\n", mae_loss / (LOG_EVERY_N_STEPS / LOG_STEP), mse_loss / (LOG_EVERY_N_STEPS / LOG_STEP), color_loss / (LOG_EVERY_N_STEPS / LOG_STEP));
            
            send_metadata_to_python("http://127.0.0.1:5000/submitLoss", step, avg_loss, cnn_get_learning_rate(cnn), stats.total_time_ms, mae_loss, mse_loss, color_loss);
            accumulated_loss = 0.0f;
            
            cnn_get_output(cnn, prediction);
            planarToInterleaved(prediction, prediction_interleaved, WIDTH, HEIGHT, 4);
            
            size_t base64_size = ((WIDTH * HEIGHT * 3 + 2) / 3) * 4 + 1;
            size_t rgb_size = WIDTH * HEIGHT * 3;
            
            char *clean_b64 = malloc(base64_size);
            char *noisy_b64 = malloc(base64_size);
            char *pred_b64 = malloc(base64_size);
            
            unsigned char *clean_rgb = malloc(rgb_size);
            unsigned char *noisy_rgb = malloc(rgb_size);
            unsigned char *pred_rgb = malloc(rgb_size);
            
            imageToBase64_noalloc(sample->highRes, WIDTH, HEIGHT, clean_rgb, clean_b64);
            imageToBase64_noalloc(sample->lowRes, WIDTH, HEIGHT, noisy_rgb, noisy_b64);
            imageToBase64_noalloc(prediction_interleaved, WIDTH, HEIGHT, pred_rgb, pred_b64);
            
            send_images_to_python("http://127.0.0.1:5000/submitImage", noisy_b64, clean_b64, pred_b64, step);
            
            free(clean_b64);
            free(noisy_b64);
            free(pred_b64);
            free(clean_rgb);
            free(noisy_rgb);
            free(pred_rgb);
        }

        float new_lr = learning_rate_decay_get(&lr_decay, step);
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
    free(noise_target);
    free(prediction_interleaved);
    free(input_planar);
    free(target_planar);
    cnn_save_weights(cnn, "cnn_weights.bin");

    cnn_destroy(cnn);
    free(loader);
    free(sample);
    free(prediction);
    return 0;
}
