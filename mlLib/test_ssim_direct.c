#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TEST_WIDTH 64
#define TEST_HEIGHT 48
#define TEST_SIZE (TEST_WIDTH * TEST_HEIGHT * 4)

void test_ssim_kernel_directly(CNNDenoiser* cnn) {
    printf("\n=== Direct SSIM Kernel Test (Bypass Network) ===\n");
    
    int pixels = TEST_WIDTH * TEST_HEIGHT;
    int size = pixels * 4;
    
    float *data1 = malloc(size * sizeof(float));
    float *data2 = malloc(size * sizeof(float));
    float *grad_buf = malloc(size * sizeof(float));
    float *loss_buf = malloc(size * sizeof(float));
    
    printf("\nTest 1: Identical data\n");
    for (int i = 0; i < size; i++) {
        data1[i] = 0.5f;
        data2[i] = 0.5f;
    }
    
    cl_mem buf1 = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        size * sizeof(float), data1, NULL);
    cl_mem buf2 = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
        size * sizeof(float), data2, NULL);
    cl_mem buf_grad = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
    cl_mem buf_loss = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
    
    int batch = 1;
    clSetKernelArg(cnn->k_batch_ssim_loss, 0, sizeof(cl_mem), &buf1);
    clSetKernelArg(cnn->k_batch_ssim_loss, 1, sizeof(cl_mem), &buf2);
    clSetKernelArg(cnn->k_batch_ssim_loss, 2, sizeof(cl_mem), &buf_grad);
    clSetKernelArg(cnn->k_batch_ssim_loss, 3, sizeof(cl_mem), &buf_loss);
    clSetKernelArg(cnn->k_batch_ssim_loss, 4, sizeof(int), &batch);
    clSetKernelArg(cnn->k_batch_ssim_loss, 5, sizeof(int), &TEST_HEIGHT);
    clSetKernelArg(cnn->k_batch_ssim_loss, 6, sizeof(int), &TEST_WIDTH);
    
    size_t global[3] = {TEST_WIDTH, TEST_HEIGHT, 1};
    clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_ssim_loss, 3, NULL, global, NULL, 0, NULL, NULL);
    
    clEnqueueReadBuffer(cnn->queue, buf_loss, CL_TRUE, 0, size * sizeof(float), loss_buf, 0, NULL, NULL);
    
    float total_loss = 0.0f;
    for (int i = 0; i < pixels; i++) {
        total_loss += loss_buf[i];
    }
    float avg_loss = total_loss / pixels;
    
    printf("  Average SSIM loss: %.6f (should be ~0.0)\n", avg_loss);
    printf("  First 5 pixel losses: %.4f %.4f %.4f %.4f %.4f\n", 
        loss_buf[0], loss_buf[1], loss_buf[2], loss_buf[3], loss_buf[4]);
    
    printf("\nTest 2: Completely different data\n");
    for (int i = 0; i < size; i++) {
        data1[i] = 0.0f;
        data2[i] = 1.0f;
    }
    
    clEnqueueWriteBuffer(cnn->queue, buf1, CL_FALSE, 0, size * sizeof(float), data1, 0, NULL, NULL);
    clEnqueueWriteBuffer(cnn->queue, buf2, CL_FALSE, 0, size * sizeof(float), data2, 0, NULL, NULL);
    clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_ssim_loss, 3, NULL, global, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cnn->queue, buf_loss, CL_TRUE, 0, size * sizeof(float), loss_buf, 0, NULL, NULL);
    
    total_loss = 0.0f;
    for (int i = 0; i < pixels; i++) {
        total_loss += loss_buf[i];
    }
    avg_loss = total_loss / pixels;
    
    printf("  Average SSIM loss: %.6f (should be ~1.0)\n", avg_loss);
    
    printf("\nTest 3: Slightly different data\n");
    for (int i = 0; i < size; i++) {
        data1[i] = 0.5f;
        data2[i] = 0.51f;
    }
    
    clEnqueueWriteBuffer(cnn->queue, buf1, CL_FALSE, 0, size * sizeof(float), data1, 0, NULL, NULL);
    clEnqueueWriteBuffer(cnn->queue, buf2, CL_FALSE, 0, size * sizeof(float), data2, 0, NULL, NULL);
    clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_ssim_loss, 3, NULL, global, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(cnn->queue, buf_loss, CL_TRUE, 0, size * sizeof(float), loss_buf, 0, NULL, NULL);
    
    total_loss = 0.0f;
    for (int i = 0; i < pixels; i++) {
        total_loss += loss_buf[i];
    }
    avg_loss = total_loss / pixels;
    
    printf("  Average SSIM loss: %.6f (should be small, ~0.01-0.1)\n", avg_loss);
    
    clReleaseMemObject(buf1);
    clReleaseMemObject(buf2);
    clReleaseMemObject(buf_grad);
    clReleaseMemObject(buf_loss);
    
    free(data1);
    free(data2);
    free(grad_buf);
    free(loss_buf);
}

int main() {
    printf("=== SSIM Kernel Direct Test ===\n");
    printf("Testing kernel with %dx%d images\n", TEST_WIDTH, TEST_HEIGHT);
    
    CNNConfig cfg = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, 4);
    cfg.max_batch_size = 2;
    cfg.use_profiling = 0;
    
    CNNDenoiser* cnn = cnn_create(cfg);
    cnn_add_layer(cnn, (LayerConfig){4, 4, 0, -1, "passthrough"});
    cnn_finalize(cnn);
    
    if (!cnn->k_batch_ssim_loss) {
        printf("ERROR: SSIM kernel not created!\n");
        return 1;
    }
    
    test_ssim_kernel_directly(cnn);
    
    cnn_destroy(cnn);
    
    printf("\nDirect kernel test completed!\n");
    return 0;
}
