#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAX_STEPS 10000
#define LOG_EVERY_N_STEPS 10
#define LOG_STEP 10


int main() {
    CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
    
    cfg.max_batch_size = 32;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;
    cfg.use_profiling = 1;

    cfg.adam_beta1 = 0.95f;
    cfg.adam_beta2 = 0.999f;
    cfg.adam_epsilon = 1e-8f;
    
    cfg.loss_config.num_losses = 4;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 2.25f;    
    // cfg.loss_config.types[1] = LOSS_MSE;
    // cfg.loss_config.weights[1] = 0.75f;
    cfg.loss_config.types[1] = LOSS_COLOR_VARIANCE;
    cfg.loss_config.weights[1] = 0.0035f;
    cfg.loss_config.types[3] = LOSS_LAPLACE;
    cfg.loss_config.weights[3] = 0.5f;
    cfg.loss_config.types[2] = LOSS_SSIM;
    cfg.loss_config.weights[2] = 1.25f;
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
    
    /* Load pretrained baseline weights */
    printf("\nLoading pretrained weights from model/cnn_weights_baseline.bin...\n");
    if (cnn_load_weights(cnn, "model/cnn_weights_baseline.bin") == 0) {
        printf("Successfully loaded pretrained weights!\n");
    } else {
        printf("Warning: Could not load pretrained weights, starting from scratch\n");
    }      

    DataLoader *loader = malloc(sizeof(DataLoader));
    fillDataLoader(loader, "/media/user/2TB Clear/imageData");

    ImageSample *sample = malloc(sizeof(ImageSample));
    float *prediction = malloc(IMAGE_SIZE * sizeof(float));
    float *prediction_interleaved = malloc(IMAGE_SIZE * sizeof(float));  /* For display */
    float *noise_target = malloc(IMAGE_SIZE * sizeof(float));  /* For residual mode: noise = noisy - clean */
    
    /* Buffers for converting interleaved input to planar format for GPU */
    float *input_planar = malloc(IMAGE_SIZE * sizeof(float));
    float *target_planar = malloc(IMAGE_SIZE * sizeof(float));
    
    /* Batch training buffers - contiguous memory layout */
    const int BATCH_SIZE = 16;
    float *batch_input = malloc(BATCH_SIZE * IMAGE_SIZE * sizeof(float));
    float *batch_target = malloc(BATCH_SIZE * IMAGE_SIZE * sizeof(float));
    
    /* Save first sample in each batch for visualization */
    ImageSample *display_sample = malloc(sizeof(ImageSample));
    
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
    
    for (int step = 0; step < MAX_STEPS; step++) {
        /* Load batch of images */
        for (int b = 0; b < BATCH_SIZE; b++) {
            getNextImagePair(loader, sample);
            
            /* Save first sample for display */
            if (b == 0) {
                memcpy(display_sample, sample, sizeof(ImageSample));
            }
            
            /* Convert input from interleaved to planar format */
            /* Each image in batch is at offset b * IMAGE_SIZE */
            interleavedToPlanar(sample->lowRes, &batch_input[b * IMAGE_SIZE], WIDTH, HEIGHT, 4);
            
            if (cfg.residual_mode) {
                for (int i = 0; i < IMAGE_SIZE; i++) {
                    noise_target[i] = sample->lowRes[i] - sample->highRes[i];
                }
                interleavedToPlanar(noise_target, &batch_target[b * IMAGE_SIZE], WIDTH, HEIGHT, 4);
            } else {
                interleavedToPlanar(sample->highRes, &batch_target[b * IMAGE_SIZE], WIDTH, HEIGHT, 4);
            }
        }
        
        /* Train on batch */
        accumulated_loss += cnn_train_step(cnn, batch_input, batch_target, BATCH_SIZE);
        
        TimingStats stats;
        cnn_get_timing_stats(cnn, &stats);
        cumulative_stats.forward_time_ms += stats.forward_time_ms;
        cumulative_stats.backward_time_ms += stats.backward_time_ms;
        cumulative_stats.loss_time_ms += stats.loss_time_ms;
        cumulative_stats.update_time_ms += stats.update_time_ms;
        cumulative_stats.total_time_ms += stats.total_time_ms;
        timing_samples++;

        if (step % LOG_STEP == 0) {
            printf("Step %d (Batch %d): Forward %.2f ms, Backward %.2f ms, Loss %.2f ms, Update %.2f ms, Total %.2f ms\n",
                   step * BATCH_SIZE,
                   step,
                   stats.forward_time_ms,
                   stats.backward_time_ms,
                   stats.loss_time_ms,
                   stats.update_time_ms,
                   stats.total_time_ms);
            printf("Current Loss: %.6f (%.2f img/s)\n", 
                   accumulated_loss / (step + 1), 
                   BATCH_SIZE * 1000.0 / stats.total_time_ms);
        }
        
        if (step % LOG_EVERY_N_STEPS == 0 && step > 0) {
            float current_mae, current_mse, current_laplace, current_color, current_ssim;
            cnn_get_individual_losses(cnn, &current_mae, &current_mse, &current_laplace, &current_color, &current_ssim);
            
            /* Library returns unweighted losses, compute weighted versions for display */
            float weight_mae = 1.0f, weight_mse = 1.0f, weight_laplace = 1.0f, weight_color = 1.0f, weight_ssim = 1.0f;
            for (int i = 0; i < cfg.loss_config.num_losses; i++) {
                if (cfg.loss_config.types[i] == LOSS_MAE) weight_mae = cfg.loss_config.weights[i];
                else if (cfg.loss_config.types[i] == LOSS_MSE) weight_mse = cfg.loss_config.weights[i];
                else if (cfg.loss_config.types[i] == LOSS_LAPLACE) weight_laplace = cfg.loss_config.weights[i];
                else if (cfg.loss_config.types[i] == LOSS_COLOR_VARIANCE) weight_color = cfg.loss_config.weights[i];
                else if (cfg.loss_config.types[i] == LOSS_SSIM) weight_ssim = cfg.loss_config.weights[i];
            }
            
            /* Multiply by weights to get weighted contribution to total loss */
            float weighted_mae = current_mae * weight_mae;
            float weighted_mse = current_mse * weight_mse;
            float weighted_laplace = current_laplace * weight_laplace;
            float weighted_color = current_color * weight_color;
            float weighted_ssim = current_ssim * weight_ssim;
            
            float avg_loss = accumulated_loss / LOG_EVERY_N_STEPS;
            printf("\n=== Batch %3d (Images %d) ===\n", 
                   step, step * BATCH_SIZE);
            printf("   Avg Loss: %.6f, LR: %.6f, Time: %.2f ms/batch, Throughput: %.1f img/s\n", 
                   avg_loss, cnn_get_learning_rate(cnn), 
                   stats.total_time_ms, 
                   BATCH_SIZE * 1000.0 / stats.total_time_ms);
            
            /* DEBUG: Check input/target values for first image in last batch */
            float input_min = 1e9, input_max = -1e9, target_min = 1e9, target_max = -1e9;
            for (int i = 0; i < IMAGE_SIZE; i++) {
                if (batch_input[i] < input_min) input_min = batch_input[i];
                if (batch_input[i] > input_max) input_max = batch_input[i];
                if (batch_target[i] < target_min) target_min = batch_target[i];
                if (batch_target[i] > target_max) target_max = batch_target[i];
            }
            printf("[DEBUG] Input (noisy) range: [%.6f, %.6f]\n", input_min, input_max);
            printf("[DEBUG] Target (clean) range: [%.6f, %.6f]\n", target_min, target_max);
            
            /* Display: unweighted (weighted contribution) */
            printf("   Losses: MAE=%.4f(%.4f) MSE=%.4f(%.4f) Laplace=%.4f(%.4f) Color=%.4f(%.4f) SSIM=%.4f(%.4f)\n", 
                   current_mae, weighted_mae, current_mse, weighted_mse, 
                   current_laplace, weighted_laplace, current_color, weighted_color, 
                   current_ssim, weighted_ssim);
            
            if (avg_loss < best_loss) {
                best_loss = avg_loss;
                printf("   *** New best loss! Saving weights... ***\n");
                cnn_save_weights(cnn, "cnn_weights_best.bin");
            }
            
            /* Send unweighted losses to Python for better visualization */
            send_metadata_to_python("http://127.0.0.1:5000/submitLoss", step, avg_loss, cnn_get_learning_rate(cnn), stats.total_time_ms, current_mae, current_mse, current_color, current_laplace, current_ssim);
            accumulated_loss = 0.0f;
            
            /* Get output from first image in batch (batch training stores outputs in batch_output buffer) */
            cnn_get_batch_output(cnn, prediction, 0);
            
            /* DEBUG: Check prediction values (planar format) */
            float pred_min = 1e9, pred_max = -1e9;
            float pred_sum_r = 0, pred_sum_g = 0, pred_sum_b = 0, pred_sum_l = 0;
            for (int i = 0; i < WIDTH * HEIGHT; i++) {
                float r = prediction[i];
                float g = prediction[WIDTH * HEIGHT + i];
                float b = prediction[2 * WIDTH * HEIGHT + i];
                float l = prediction[3 * WIDTH * HEIGHT + i];
                
                if (r < pred_min) pred_min = r;
                if (r > pred_max) pred_max = r;
                if (g < pred_min) pred_min = g;
                if (g > pred_max) pred_max = g;
                if (b < pred_min) pred_min = b;
                if (b > pred_max) pred_max = b;
                
                pred_sum_r += r;
                pred_sum_g += g;
                pred_sum_b += b;
                pred_sum_l += l;
            }
            int num_pixels = WIDTH * HEIGHT;
            printf("[DEBUG] Prediction stats (planar): min=%.6f max=%.6f\n", pred_min, pred_max);
            printf("  Avg: R=%.6f G=%.6f B=%.6f L=%.6f\n", 
                   pred_sum_r/num_pixels, pred_sum_g/num_pixels, pred_sum_b/num_pixels, pred_sum_l/num_pixels);
            printf("  First pixel: R=%.6f G=%.6f B=%.6f L=%.6f\n",
                   prediction[0], prediction[WIDTH*HEIGHT], prediction[2*WIDTH*HEIGHT], prediction[3*WIDTH*HEIGHT]);
            
            planarToInterleaved(prediction, prediction_interleaved, WIDTH, HEIGHT, 4);
            
            /* DEBUG: Check after interleave */
            printf("[DEBUG] After interleave: first pixel RGBL = %.6f %.6f %.6f %.6f\n",
                   prediction_interleaved[0], prediction_interleaved[1], 
                   prediction_interleaved[2], prediction_interleaved[3]);
            
            /* DEBUG: Check input sample too */
            printf("[DEBUG] Input (noisy) first pixel: R=%.6f G=%.6f B=%.6f L=%.6f\n",
                   display_sample->lowRes[0], display_sample->lowRes[1], display_sample->lowRes[2], display_sample->lowRes[3]);
            printf("[DEBUG] Target (clean) first pixel: R=%.6f G=%.6f B=%.6f L=%.6f\n",
                   display_sample->highRes[0], display_sample->highRes[1], display_sample->highRes[2], display_sample->highRes[3]);
            
            size_t base64_size = ((WIDTH * HEIGHT * 3 + 2) / 3) * 4 + 1;
            size_t rgb_size = WIDTH * HEIGHT * 3;
            
            char *clean_b64 = malloc(base64_size);
            char *noisy_b64 = malloc(base64_size);
            char *pred_b64 = malloc(base64_size);
            
            unsigned char *clean_rgb = malloc(rgb_size);
            unsigned char *noisy_rgb = malloc(rgb_size);
            unsigned char *pred_rgb = malloc(rgb_size);
            
            imageToBase64_noalloc(display_sample->highRes, WIDTH, HEIGHT, clean_rgb, clean_b64);
            imageToBase64_noalloc(display_sample->lowRes, WIDTH, HEIGHT, noisy_rgb, noisy_b64);
            imageToBase64_noalloc(prediction_interleaved, WIDTH, HEIGHT, pred_rgb, pred_b64);
            
            /* DEBUG: Check RGB conversion */
            printf("[DEBUG] RGB conversion - first 10 bytes of pred_rgb: ");
            for (int i = 0; i < 10; i++) printf("%d ", pred_rgb[i]);
            printf("\n");
            
            /* Check if all bytes are zero */
            int nonzero_count = 0;
            for (int i = 0; i < rgb_size; i++) {
                if (pred_rgb[i] != 0) nonzero_count++;
            }
            printf("[DEBUG] pred_rgb has %d/%zu non-zero bytes (%.1f%%)\n", 
                   nonzero_count, rgb_size, 100.0 * nonzero_count / rgb_size);
            
            /* TEMPORARY: Scale prediction for visualization if values are too small */
            float scale_factor = 1.0f;
            if (pred_max < 0.1f && pred_max > 0.0f) {
                scale_factor = 0.5f / pred_max;  /* Scale max to 0.5 */
                printf("[VISUALIZATION] Scaling prediction by %.1fx for visibility (max=%.6f)\n", 
                       scale_factor, pred_max);
                
                for (int i = 0; i < IMAGE_SIZE; i++) {
                    prediction_interleaved[i] = fminf(prediction_interleaved[i] * scale_factor, 1.0f);
                }
                
                /* Re-convert to RGB with scaling */
                imageToBase64_noalloc(prediction_interleaved, WIDTH, HEIGHT, pred_rgb, pred_b64);
            }
            
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

    printf("\n=== Training Performance (Averaged over %d batches) ===\n", timing_samples);
    printf("Forward time:  %.3f ms (avg)\n", cumulative_stats.forward_time_ms / timing_samples);
    printf("Backward time: %.3f ms (avg)\n", cumulative_stats.backward_time_ms / timing_samples);
    printf("Loss time:     %.3f ms (avg)\n", cumulative_stats.loss_time_ms / timing_samples);
    printf("Update time:   %.3f ms (avg)\n", cumulative_stats.update_time_ms / timing_samples);
    printf("Total time:    %.3f ms (avg)\n", cumulative_stats.total_time_ms / timing_samples);
    printf("Throughput:    %.2f images/sec (batch_size=%d)\n", 
           BATCH_SIZE * 1000.0 / (cumulative_stats.total_time_ms / timing_samples), BATCH_SIZE);
    
    printf("\nSaving weights...\n");
    
    /* Cleanup batch buffers */
    free(batch_input);
    free(batch_target);
    free(display_sample);
    
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
