#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== Minimal Batch Forward Test ===\n\n");
    
    CNNConfig cfg = cnn_default_config(800, 600, 4);
    cfg.max_batch_size = 2;
    cfg.optimizer = OPTIMIZER_ADAM;
    cfg.learning_rate = 0.0001f;
    cfg.use_profiling = 0;
    cfg.residual_mode = 0;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    
    /* Single simple layer: 4 -> 4, linear output */
    cnn_add_layer(cnn, (LayerConfig){4, 4, 0, -1, "single_layer"});
    cnn_finalize(cnn);
    
    const int BATCH_SIZE = 2;
    const int IMG_SIZE = 800 * 600 * 4;
    float *batch_input = malloc(BATCH_SIZE * IMG_SIZE * sizeof(float));
    float *batch_output = malloc(IMG_SIZE * sizeof(float));
    
    /* Fill with simple test pattern */
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int i = 0; i < IMG_SIZE; i++) {
            batch_input[b * IMG_SIZE + i] = 0.5f;  /* Constant value */
        }
    }
    
    printf("Uploading batch input (batch_size=%d, img_size=%d)...\n", BATCH_SIZE, IMG_SIZE);
    clEnqueueWriteBuffer(cnn->queue, cnn->batch_input_buf, CL_TRUE, 0,
                        BATCH_SIZE * IMG_SIZE * sizeof(float), batch_input, 0, NULL, NULL);
    
    printf("Running batch forward...\n");
    ConvLayer *l = &cnn->layers[0];
    
    clSetKernelArg(cnn->k_batch_forward, 0, sizeof(cl_mem), &cnn->batch_input_buf);
    clSetKernelArg(cnn->k_batch_forward, 1, sizeof(cl_mem), &l->batch_output);
    clSetKernelArg(cnn->k_batch_forward, 2, sizeof(cl_mem), &l->weights);
    clSetKernelArg(cnn->k_batch_forward, 3, sizeof(cl_mem), &l->bias);
    clSetKernelArg(cnn->k_batch_forward, 4, sizeof(int), &BATCH_SIZE);
    clSetKernelArg(cnn->k_batch_forward, 5, sizeof(int), &l->cin4);
    clSetKernelArg(cnn->k_batch_forward, 6, sizeof(int), &l->cout);
    clSetKernelArg(cnn->k_batch_forward, 7, sizeof(int), &l->h);
    clSetKernelArg(cnn->k_batch_forward, 8, sizeof(int), &l->w);
    clSetKernelArg(cnn->k_batch_forward, 9, sizeof(int), &l->use_relu);
    
    size_t global[4] = {l->w, l->h, (l->cout + 3) / 4, BATCH_SIZE};
    printf("Global work size: [%zu, %zu, %zu, %zu]\n", global[0], global[1], global[2], global[3]);
    printf("Layer: cin=%d cin4=%d cout=%d h=%d w=%d use_relu=%d\n",
           l->cin, l->cin4, l->cout, l->h, l->w, l->use_relu);
    
    clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_forward, 4, NULL, global, NULL, 0, NULL, NULL);
    clFinish(cnn->queue);
    
    printf("\nReading outputs...\n");
    for (int b = 0; b < BATCH_SIZE; b++) {
        cnn_get_batch_output(cnn, batch_output, b);
        
        float sum = 0, min_val = 1e9, max_val = -1e9;
        for (int i = 0; i < IMG_SIZE; i++) {
            sum += batch_output[i];
            if (batch_output[i] < min_val) min_val = batch_output[i];
            if (batch_output[i] > max_val) max_val = batch_output[i];
        }
        
        printf("Batch %d: First 10 values: ", b);
        for (int i = 0; i < 10; i++) printf("%.6f ", batch_output[i]);
        printf("\n");
        printf("  Min=%.6f, Max=%.6f, Avg=%.6f\n", min_val, max_val, sum/IMG_SIZE);
    }
    
    printf("\n=== Test Complete ===\n");
    
    free(batch_output);
    free(batch_input);
    cnn_destroy(cnn);
    
    return 0;
}
