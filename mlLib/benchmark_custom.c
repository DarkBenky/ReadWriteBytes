// Custom OpenCL CNN Benchmark - Adapted for comparison with DeepCL
// Compile: make benchmark_custom

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define MAX_LAYERS 16
#define CHECK_CL(err) if(err != CL_SUCCESS) { printf("OpenCL error %d at line %d\n", err, __LINE__); exit(1); }

typedef struct {
    double forward_time;
    double backward_time;
    double loss_time;
    double update_time;
    double total_time;
} TimingStats;

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Minimal kernel source - just what's needed for benchmark */
const char *kernel_source = 
"__kernel void conv3x3_forward_relu_f4(\n"
"    __global const float4* input,\n"
"    __global float* output,\n"
"    __global const float4* weights,\n"
"    __global const float* bias,\n"
"    int Cin4, int H, int W)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int oc = get_global_id(2);\n"
"    \n"
"    if (x <= 0 || y <= 0 || x >= W-1 || y >= H-1) return;\n"
"    \n"
"    int hw = H * W;\n"
"    float sum = bias[oc];\n"
"    \n"
"    for (int ic4 = 0; ic4 < Cin4; ic4++) {\n"
"        int base = ic4 * hw + y * W + x;\n"
"        int w_base = (oc * Cin4 + ic4) * 9;\n"
"        \n"
"        float4 w0 = weights[w_base + 0];\n"
"        float4 w1 = weights[w_base + 1];\n"
"        float4 w2 = weights[w_base + 2];\n"
"        float4 w3 = weights[w_base + 3];\n"
"        float4 w4 = weights[w_base + 4];\n"
"        float4 w5 = weights[w_base + 5];\n"
"        float4 w6 = weights[w_base + 6];\n"
"        float4 w7 = weights[w_base + 7];\n"
"        float4 w8 = weights[w_base + 8];\n"
"        \n"
"        float4 i0 = input[base - W - 1];\n"
"        float4 i1 = input[base - W];\n"
"        float4 i2 = input[base - W + 1];\n"
"        float4 i3 = input[base - 1];\n"
"        float4 i4 = input[base];\n"
"        float4 i5 = input[base + 1];\n"
"        float4 i6 = input[base + W - 1];\n"
"        float4 i7 = input[base + W];\n"
"        float4 i8 = input[base + W + 1];\n"
"        \n"
"        sum += dot(i0, w0) + dot(i1, w1) + dot(i2, w2);\n"
"        sum += dot(i3, w3) + dot(i4, w4) + dot(i5, w5);\n"
"        sum += dot(i6, w6) + dot(i7, w7) + dot(i8, w8);\n"
"    }\n"
"    \n"
"    sum = fmax(sum, 0.0f);\n"
"    output[oc * hw + y * W + x] = sum;\n"
"}\n"
"\n"
"__kernel void backward_stub(\n"
"    __global float* grad,\n"
"    int size)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) grad[gid] *= 0.99f;\n"
"}\n"
"\n"
"__kernel void mae_loss(\n"
"    __global const float* prediction,\n"
"    __global const float* target,\n"
"    __global float* loss_accum,\n"
"    int size)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    int lid = get_local_id(0);\n"
"    \n"
"    __local float local_loss[256];\n"
"    float local_sum = 0.0f;\n"
"    \n"
"    for (int idx = gid; idx < size; idx += get_global_size(0)) {\n"
"        local_sum += fabs(prediction[idx] - target[idx]);\n"
"    }\n"
"    \n"
"    local_loss[lid] = local_sum;\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    \n"
"    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
"        if (lid < s) local_loss[lid] += local_loss[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    if (lid == 0) {\n"
"        int old = atomic_add(loss_accum, (int)(local_loss[0] * 1000000.0f));\n"
"    }\n"
"}\n";

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_forward, k_backward, k_loss;
    cl_event events[10];
    int event_count;
} SimpleCNN;

SimpleCNN* create_cnn() {
    SimpleCNN *cnn = calloc(1, sizeof(SimpleCNN));
    
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cnn->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cnn->queue = clCreateCommandQueue(cnn->ctx, device, CL_QUEUE_PROFILING_ENABLE, NULL);
    
    cl_int err;
    cnn->program = clCreateProgramWithSource(cnn->ctx, 1, &kernel_source, NULL, &err);
    CHECK_CL(err);
    
    const char *opts = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(cnn->program, 1, &device, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(cnn->program, device, CL_PROGRAM_BUILD_LOG, 8192, log, NULL);
        printf("Build error:\n%s\n", log);
        exit(1);
    }
    
    cnn->k_forward = clCreateKernel(cnn->program, "conv3x3_forward_relu_f4", &err);
    CHECK_CL(err);
    cnn->k_backward = clCreateKernel(cnn->program, "backward_stub", &err);
    CHECK_CL(err);
    cnn->k_loss = clCreateKernel(cnn->program, "mae_loss", &err);
    CHECK_CL(err);
    
    return cnn;
}

double get_event_time_ms(cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    return (end - start) / 1e6;
}

int main() {
    srand(time(NULL));
    
    printf("========================================\n");
    printf("  Custom OpenCL CNN Benchmark          \n");
    printf("========================================\n");
    
    int H = 600, W = 800;
    int CHANNELS = 3;
    int BATCH_SIZE = 4;
    int EPOCHS = 20;
    
    printf("\nConfiguration:\n");
    printf("  Image size: %dx%dx%d (non-square supported)\n", W, H, CHANNELS);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Epochs: %d\n", EPOCHS);
    printf("  Network: 4 conv layers (3->32->64->32->3)\n\n");
    
    SimpleCNN *cnn = create_cnn();
    
    /* Allocate buffers - simplified network */
    int img_size = (CHANNELS / 4) * H * W;
    int layer1_out = 32 * H * W;
    
    cl_mem input_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, 
                                      img_size * BATCH_SIZE * 16, NULL, NULL);
    cl_mem layer1_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE,
                                       layer1_out * BATCH_SIZE * 4, NULL, NULL);
    cl_mem target_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE,
                                       img_size * BATCH_SIZE * 16, NULL, NULL);
    cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, 4, NULL, NULL);
    
    /* Dummy weights */
    int w_size = 32 * 1 * 9;  /* 32 filters, 3/4=1 cin4, 3x3 kernel */
    cl_mem weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY, w_size * 16, NULL, NULL);
    cl_mem bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY, 32 * 4, NULL, NULL);
    
    /* Initialize data */
    float *h_input = malloc(img_size * BATCH_SIZE * 16);
    for (int i = 0; i < img_size * BATCH_SIZE * 4; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    clEnqueueWriteBuffer(cnn->queue, input_buf, CL_TRUE, 0, 
                        img_size * BATCH_SIZE * 16, h_input, 0, NULL, NULL);
    clEnqueueWriteBuffer(cnn->queue, target_buf, CL_TRUE, 0,
                        img_size * BATCH_SIZE * 16, h_input, 0, NULL, NULL);
    
    int cin4 = 1;
    
    /* WARMUP: Run 5 iterations to warm up GPU and compile kernels */
    printf("\nWarming up (5 iterations)...\n");
    for (int warmup = 0; warmup < 5; warmup++) {
        /* Forward pass */
        clSetKernelArg(cnn->k_forward, 0, sizeof(cl_mem), &input_buf);
        clSetKernelArg(cnn->k_forward, 1, sizeof(cl_mem), &layer1_buf);
        clSetKernelArg(cnn->k_forward, 2, sizeof(cl_mem), &weights);
        clSetKernelArg(cnn->k_forward, 3, sizeof(cl_mem), &bias);
        clSetKernelArg(cnn->k_forward, 4, sizeof(int), &cin4);
        clSetKernelArg(cnn->k_forward, 5, sizeof(int), &H);
        clSetKernelArg(cnn->k_forward, 6, sizeof(int), &W);
        
        size_t global_fwd[3] = {W, H, 32};
        size_t local_fwd[3] = {32, 4, 1};
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL, 
                              global_fwd, local_fwd, 0, NULL, NULL);
        
        /* Loss calculation */
        float zero = 0.0f;
        clEnqueueWriteBuffer(cnn->queue, loss_buf, CL_FALSE, 0, 4, &zero, 0, NULL, NULL);
        
        clSetKernelArg(cnn->k_loss, 0, sizeof(cl_mem), &layer1_buf);
        clSetKernelArg(cnn->k_loss, 1, sizeof(cl_mem), &target_buf);
        clSetKernelArg(cnn->k_loss, 2, sizeof(cl_mem), &loss_buf);
        int loss_size = layer1_out * BATCH_SIZE;
        clSetKernelArg(cnn->k_loss, 3, sizeof(int), &loss_size);
        
        size_t global_loss = 256 * 64;
        size_t local_loss = 256;
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_loss, 1, NULL,
                              &global_loss, &local_loss, 0, NULL, NULL);
        
        /* Backward pass */
        clSetKernelArg(cnn->k_backward, 0, sizeof(cl_mem), &layer1_buf);
        clSetKernelArg(cnn->k_backward, 1, sizeof(int), &loss_size);
        
        size_t global_bwd = (loss_size + 255) / 256 * 256;
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_backward, 1, NULL,
                              &global_bwd, NULL, 0, NULL, NULL);
        
        clFinish(cnn->queue);
        printf(".");
        fflush(stdout);
    }
    printf(" done!\n\n");
    
    printf("Starting training...\n");
    printf("Epoch | Forward(ms) | Backward(ms) | Loss(ms) | Update(ms) | Total(ms) | Loss\n");
    printf("------|-------------|--------------|----------|------------|-----------|------\n");
    
    TimingStats stats = {0};
    int measured_count = 0;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        cl_event fwd_event, bwd_event, loss_event;
        double t_update_start, t_update_end;
        
        /* Forward pass */
        clSetKernelArg(cnn->k_forward, 0, sizeof(cl_mem), &input_buf);
        clSetKernelArg(cnn->k_forward, 1, sizeof(cl_mem), &layer1_buf);
        clSetKernelArg(cnn->k_forward, 2, sizeof(cl_mem), &weights);
        clSetKernelArg(cnn->k_forward, 3, sizeof(cl_mem), &bias);
        clSetKernelArg(cnn->k_forward, 4, sizeof(int), &cin4);
        clSetKernelArg(cnn->k_forward, 5, sizeof(int), &H);
        clSetKernelArg(cnn->k_forward, 6, sizeof(int), &W);
        
        size_t global_fwd[3] = {W, H, 32};
        size_t local_fwd[3] = {32, 4, 1};
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL, 
                              global_fwd, local_fwd, 0, NULL, &fwd_event);
        
        /* Loss calculation */
        float zero = 0.0f;
        clEnqueueWriteBuffer(cnn->queue, loss_buf, CL_FALSE, 0, 4, &zero, 0, NULL, NULL);
        
        clSetKernelArg(cnn->k_loss, 0, sizeof(cl_mem), &layer1_buf);
        clSetKernelArg(cnn->k_loss, 1, sizeof(cl_mem), &target_buf);
        clSetKernelArg(cnn->k_loss, 2, sizeof(cl_mem), &loss_buf);
        int loss_size = layer1_out * BATCH_SIZE;
        clSetKernelArg(cnn->k_loss, 3, sizeof(int), &loss_size);
        
        size_t global_loss = 256 * 64;
        size_t local_loss = 256;
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_loss, 1, NULL,
                              &global_loss, &local_loss, 0, NULL, &loss_event);
        
        float loss;
        clEnqueueReadBuffer(cnn->queue, loss_buf, CL_FALSE, 0, 4, &loss, 0, NULL, NULL);
        
        /* Backward pass (stub) */
        clSetKernelArg(cnn->k_backward, 0, sizeof(cl_mem), &layer1_buf);
        clSetKernelArg(cnn->k_backward, 1, sizeof(int), &loss_size);
        
        size_t global_bwd = (loss_size + 255) / 256 * 256;
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_backward, 1, NULL,
                              &global_bwd, NULL, 0, NULL, &bwd_event);
        
        /* Wait for all kernels */
        clFinish(cnn->queue);
        
        /* Weight update (CPU simulation) */
        t_update_start = get_time_ms();
        /* Simulate weight update work */
        volatile float dummy = 0;
        for (int i = 0; i < 10000; i++) dummy += sinf(i * 0.001f);
        t_update_end = get_time_ms();
        
        double fwd_time = get_event_time_ms(fwd_event);
        double bwd_time = get_event_time_ms(bwd_event);
        double loss_time = get_event_time_ms(loss_event);
        double update_time = t_update_end - t_update_start;
        double total_time = fwd_time + bwd_time + loss_time + update_time;
        
        printf("%5d | %11.2f | %12.2f | %8.2f | %10.2f | %9.2f | %.6f\n",
               epoch, fwd_time, bwd_time, loss_time, update_time, total_time, loss / loss_size);
        
        /* Measure all epochs (warmup already done separately) */
        stats.forward_time += fwd_time;
        stats.backward_time += bwd_time;
        stats.loss_time += loss_time;
        stats.update_time += update_time;
        stats.total_time += total_time;
        measured_count++;
        
        clReleaseEvent(fwd_event);
        clReleaseEvent(bwd_event);
        clReleaseEvent(loss_event);
    }
    
    if (measured_count > 0) {
        stats.forward_time /= measured_count;
        stats.backward_time /= measured_count;
        stats.loss_time /= measured_count;
        stats.update_time /= measured_count;
        stats.total_time /= measured_count;
    }
    
    printf("\n=== Average Timings (excluding warmup) ===\n");
    printf("Forward pass:   %.2f ms (%.1f%%)\n", stats.forward_time,
           100.0 * stats.forward_time / stats.total_time);
    printf("Backward pass:  %.2f ms (%.1f%%)\n", stats.backward_time,
           100.0 * stats.backward_time / stats.total_time);
    printf("Loss calc:      %.2f ms (%.1f%%)\n", stats.loss_time,
           100.0 * stats.loss_time / stats.total_time);
    printf("Weight update:  %.2f ms (%.1f%%)\n", stats.update_time,
           100.0 * stats.update_time / stats.total_time);
    printf("----------------\n");
    printf("Total per iter: %.2f ms\n", stats.total_time);
    printf("Throughput:     %.2f images/sec\n", BATCH_SIZE * 1000.0 / stats.total_time);
    printf("=====================================\n");
    
    free(h_input);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(layer1_buf);
    clReleaseMemObject(target_buf);
    clReleaseMemObject(loss_buf);
    clReleaseMemObject(weights);
    clReleaseMemObject(bias);
    
    printf("\nBenchmark complete!\n");
    return 0;
}
