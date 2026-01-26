#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== Debug Output Test ===\n\n");
    
    CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
    cfg.max_batch_size = 4;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;
    cfg.use_profiling = 0;
    cfg.residual_mode = 0;
    
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    cnn_add_layer(cnn, (LayerConfig){4, 16, 1, -1, "encode_1"});
    cnn_add_layer(cnn, (LayerConfig){16, 20, 1, -1, "encode_2"});
    cnn_add_layer(cnn, (LayerConfig){20, 24, 1, -1, "bottleneck"});
    cnn_add_layer(cnn, (LayerConfig){24, 20, 1, 1, "decode_1_skip1"});
    cnn_add_layer(cnn, (LayerConfig){20, 16, 1, 0, "decode_2_skip0"});
    cnn_add_layer(cnn, (LayerConfig){16, 4, 0, -1, "output"});
    
    cnn_finalize(cnn);
    
    DataLoader *loader = malloc(sizeof(DataLoader));
    fillDataLoader(loader, "/media/user/2TB Clear/imageData");
    
    ImageSample *sample = malloc(sizeof(ImageSample));
    float *input_planar = malloc(IMAGE_SIZE * sizeof(float));
    float *target_planar = malloc(IMAGE_SIZE * sizeof(float));
    float *prediction = malloc(IMAGE_SIZE * sizeof(float));
    float *prediction_interleaved = malloc(IMAGE_SIZE * sizeof(float));
    
    printf("Loading test image...\n");
    getNextImagePair(loader, sample);
    
    printf("Input image stats (interleaved):\n");
    printf("  First pixel: R=%.3f G=%.3f B=%.3f L=%.3f\n", 
           sample->lowRes[0], sample->lowRes[1], sample->lowRes[2], sample->lowRes[3]);
    
    float sum_r = 0, sum_g = 0, sum_b = 0, sum_l = 0;
    for (int i = 0; i < IMAGE_SIZE; i += 4) {
        sum_r += sample->lowRes[i];
        sum_g += sample->lowRes[i+1];
        sum_b += sample->lowRes[i+2];
        sum_l += sample->lowRes[i+3];
    }
    int num_pixels = IMAGE_SIZE / 4;
    printf("  Average: R=%.3f G=%.3f B=%.3f L=%.3f\n\n", 
           sum_r/num_pixels, sum_g/num_pixels, sum_b/num_pixels, sum_l/num_pixels);
    
    interleavedToPlanar(sample->lowRes, input_planar, WIDTH, HEIGHT, 4);
    interleavedToPlanar(sample->highRes, target_planar, WIDTH, HEIGHT, 4);
    
    printf("Input image stats (planar):\n");
    printf("  Channel 0 (R) first pixel: %.3f\n", input_planar[0]);
    printf("  Channel 1 (G) first pixel: %.3f\n", input_planar[WIDTH*HEIGHT]);
    printf("  Channel 2 (B) first pixel: %.3f\n", input_planar[2*WIDTH*HEIGHT]);
    printf("  Channel 3 (L) first pixel: %.3f\n\n", input_planar[3*WIDTH*HEIGHT]);
    
    printf("Running single train step (includes forward+backward)...\n");
    float loss = cnn_train_step(cnn, input_planar, target_planar, 1);
    printf("Loss: %.6f\n\n", loss);
    
    printf("Reading single-image output buffer...\n");
    cnn_get_output(cnn, prediction);
    
    printf("\nSingle-image output (planar format):\n");
    printf("  Channel 0 (R) first pixel: %.6f\n", prediction[0]);
    printf("  Channel 1 (G) first pixel: %.6f\n", prediction[WIDTH*HEIGHT]);
    printf("  Channel 2 (B) first pixel: %.6f\n", prediction[2*WIDTH*HEIGHT]);
    printf("  Channel 3 (L) first pixel: %.6f\n", prediction[3*WIDTH*HEIGHT]);
    
    float min_val = 1e9, max_val = -1e9;
    sum_r = sum_g = sum_b = sum_l = 0;
    for (int i = 0; i < WIDTH*HEIGHT; i++) {
        float r = prediction[i];
        float g = prediction[WIDTH*HEIGHT + i];
        float b = prediction[2*WIDTH*HEIGHT + i];
        float l = prediction[3*WIDTH*HEIGHT + i];
        
        if (r < min_val) min_val = r;
        if (r > max_val) max_val = r;
        if (g < min_val) min_val = g;
        if (g > max_val) max_val = g;
        if (b < min_val) min_val = b;
        if (b > max_val) max_val = b;
        
        sum_r += r;
        sum_g += g;
        sum_b += b;
        sum_l += l;
    }
    
    printf("  Min value: %.6f, Max value: %.6f\n", min_val, max_val);
    printf("  Average: R=%.6f G=%.6f B=%.6f L=%.6f\n\n", 
           sum_r/num_pixels, sum_g/num_pixels, sum_b/num_pixels, sum_l/num_pixels);
    
    planarToInterleaved(prediction, prediction_interleaved, WIDTH, HEIGHT, 4);
    
    printf("After planarToInterleaved:\n");
    printf("  First pixel: R=%.6f G=%.6f B=%.6f L=%.6f\n", 
           prediction_interleaved[0], prediction_interleaved[1], 
           prediction_interleaved[2], prediction_interleaved[3]);
    
    printf("\n--- Testing BATCH training ---\n");
    const int BATCH_SIZE = 2;
    float *batch_input = malloc(BATCH_SIZE * IMAGE_SIZE * sizeof(float));
    float *batch_target = malloc(BATCH_SIZE * IMAGE_SIZE * sizeof(float));
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        getNextImagePair(loader, sample);
        interleavedToPlanar(sample->lowRes, &batch_input[b * IMAGE_SIZE], WIDTH, HEIGHT, 4);
        interleavedToPlanar(sample->highRes, &batch_target[b * IMAGE_SIZE], WIDTH, HEIGHT, 4);
    }
    
    printf("Running batch training (batch_size=%d)...\n", BATCH_SIZE);
    float batch_loss = cnn_train_step(cnn, batch_input, batch_target, BATCH_SIZE);
    printf("Batch loss: %.6f\n", batch_loss);
    
    for (int b = 0; b < BATCH_SIZE; b++) {
        printf("\nBatch image %d:\n", b);
        cnn_get_batch_output(cnn, prediction, b);
        
        printf("  First pixel (planar): R=%.6f G=%.6f B=%.6f L=%.6f\n", 
               prediction[0], prediction[WIDTH*HEIGHT], 
               prediction[2*WIDTH*HEIGHT], prediction[3*WIDTH*HEIGHT]);
        
        min_val = 1e9, max_val = -1e9;
        sum_r = sum_g = sum_b = sum_l = 0;
        for (int i = 0; i < WIDTH*HEIGHT; i++) {
            float r = prediction[i];
            float g = prediction[WIDTH*HEIGHT + i];
            float b = prediction[2*WIDTH*HEIGHT + i];
            
            if (r < min_val) min_val = r;
            if (r > max_val) max_val = r;
            if (g < min_val) min_val = g;
            if (g > max_val) max_val = g;
            if (b < min_val) min_val = b;
            if (b > max_val) max_val = b;
            
            sum_r += r;
            sum_g += g;
            sum_b += b;
            sum_l += prediction[3*WIDTH*HEIGHT + i];
        }
        
        printf("  Min: %.6f, Max: %.6f\n", min_val, max_val);
        printf("  Avg: R=%.6f G=%.6f B=%.6f L=%.6f\n", 
               sum_r/num_pixels, sum_g/num_pixels, sum_b/num_pixels, sum_l/num_pixels);
        
        planarToInterleaved(prediction, prediction_interleaved, WIDTH, HEIGHT, 4);
        printf("  After interleave: R=%.6f G=%.6f B=%.6f L=%.6f\n", 
               prediction_interleaved[0], prediction_interleaved[1],
               prediction_interleaved[2], prediction_interleaved[3]);
    }
    
    printf("\n=== Test Complete ===\n");
    
    free(batch_input);
    free(batch_target);
    free(prediction_interleaved);
    free(prediction);
    free(target_planar);
    free(input_planar);
    free(sample);
    free(loader);
    cnn_destroy(cnn);
    
    return 0;
}
