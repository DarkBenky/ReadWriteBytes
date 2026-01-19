/* Inference Benchmark - Forward pass only, no training */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define CHECK_CL(err) if(err != CL_SUCCESS) { printf("OpenCL error %d at line %d\n", err, __LINE__); exit(1); }

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Optimized: Loop unrolling for single input channel (Cin4 = 1) */
const char *kernel_source = 
"__kernel void conv3x3_inference(\n"
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
"    int base = y * W + x;\n"
"    int w_base = oc * 9;\n"
"    \n"
"    /* Fully unrolled for single input channel */\n"
"    float4 i0 = input[base - W - 1];\n"
"    float4 i1 = input[base - W];\n"
"    float4 i2 = input[base - W + 1];\n"
"    float4 i3 = input[base - 1];\n"
"    float4 i4 = input[base];\n"
"    float4 i5 = input[base + 1];\n"
"    float4 i6 = input[base + W - 1];\n"
"    float4 i7 = input[base + W];\n"
"    float4 i8 = input[base + W + 1];\n"
"    \n"
"    float4 w0 = weights[w_base + 0];\n"
"    float4 w1 = weights[w_base + 1];\n"
"    float4 w2 = weights[w_base + 2];\n"
"    float4 w3 = weights[w_base + 3];\n"
"    float4 w4 = weights[w_base + 4];\n"
"    float4 w5 = weights[w_base + 5];\n"
"    float4 w6 = weights[w_base + 6];\n"
"    float4 w7 = weights[w_base + 7];\n"
"    float4 w8 = weights[w_base + 8];\n"
"    \n"
"    float sum = bias[oc];\n"
"    sum += dot(i0, w0);\n"
"    sum += dot(i1, w1);\n"
"    sum += dot(i2, w2);\n"
"    sum += dot(i3, w3);\n"
"    sum += dot(i4, w4);\n"
"    sum += dot(i5, w5);\n"
"    sum += dot(i6, w6);\n"
"    sum += dot(i7, w7);\n"
"    sum += dot(i8, w8);\n"
"    \n"
"    output[oc * hw + y * W + x] = fmax(sum, 0.0f);\n"
"}\n";

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
} InferenceCNN;

InferenceCNN* create_inference_cnn() {
    InferenceCNN *cnn = calloc(1, sizeof(InferenceCNN));
    
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
    
    cnn->kernel = clCreateKernel(cnn->program, "conv3x3_inference", &err);
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
    printf("  CNN Inference Benchmark (Baseline)   \n");
    printf("========================================\n");
    
    int H = 600, W = 800;
    int CHANNELS = 4;  /* Using 4 for float4 alignment */
    int FILTERS = 32;
    int ITERATIONS = 100;
    
    printf("\nConfiguration:\n");
    printf("  Image size: %dx%dx%d\n", W, H, CHANNELS);
    printf("  Output filters: %d\n", FILTERS);
    printf("  Iterations: %d\n\n", ITERATIONS);
    
    InferenceCNN *cnn = create_inference_cnn();
    
    /* Allocate buffers */
    int img_size = (CHANNELS / 4) * H * W;
    int output_size = FILTERS * H * W;
    
    cl_mem input_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY, 
                                      img_size * 16, NULL, NULL);
    cl_mem output_buf = clCreateBuffer(cnn->ctx, CL_MEM_WRITE_ONLY,
                                       output_size * 4, NULL, NULL);
    
    /* Dummy weights and bias */
    int w_size = FILTERS * (CHANNELS / 4) * 9;
    cl_mem weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY, w_size * 16, NULL, NULL);
    cl_mem bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY, FILTERS * 4, NULL, NULL);
    
    /* Initialize input data */
    float *h_input = malloc(img_size * 16);
    for (int i = 0; i < img_size * 4; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    clEnqueueWriteBuffer(cnn->queue, input_buf, CL_TRUE, 0, 
                        img_size * 16, h_input, 0, NULL, NULL);
    
    int cin4 = CHANNELS / 4;
    
    /* Warmup */
    printf("Warming up (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        clSetKernelArg(cnn->kernel, 0, sizeof(cl_mem), &input_buf);
        clSetKernelArg(cnn->kernel, 1, sizeof(cl_mem), &output_buf);
        clSetKernelArg(cnn->kernel, 2, sizeof(cl_mem), &weights);
        clSetKernelArg(cnn->kernel, 3, sizeof(cl_mem), &bias);
        clSetKernelArg(cnn->kernel, 4, sizeof(int), &cin4);
        clSetKernelArg(cnn->kernel, 5, sizeof(int), &H);
        clSetKernelArg(cnn->kernel, 6, sizeof(int), &W);
        
        size_t global[3] = {W, H, FILTERS};
        size_t local[3] = {16, 8, 1};
        clEnqueueNDRangeKernel(cnn->queue, cnn->kernel, 3, NULL, 
                              global, local, 0, NULL, NULL);
        clFinish(cnn->queue);
        printf(".");
        fflush(stdout);
    }
    printf(" done!\n\n");
    
    /* Benchmark */
    printf("Running inference benchmark...\n");
    printf("Iter | Time (ms) | Throughput (img/sec)\n");
    printf("-----|-----------|---------------------\n");
    
    double total_time = 0;
    double min_time = 1e9, max_time = 0;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        cl_event event;
        
        clSetKernelArg(cnn->kernel, 0, sizeof(cl_mem), &input_buf);
        clSetKernelArg(cnn->kernel, 1, sizeof(cl_mem), &output_buf);
        clSetKernelArg(cnn->kernel, 2, sizeof(cl_mem), &weights);
        clSetKernelArg(cnn->kernel, 3, sizeof(cl_mem), &bias);
        clSetKernelArg(cnn->kernel, 4, sizeof(int), &cin4);
        clSetKernelArg(cnn->kernel, 5, sizeof(int), &H);
        clSetKernelArg(cnn->kernel, 6, sizeof(int), &W);
        
        size_t global[3] = {W, H, FILTERS};
        size_t local[3] = {16, 8, 1};
        clEnqueueNDRangeKernel(cnn->queue, cnn->kernel, 3, NULL, 
                              global, local, 0, NULL, &event);
        
        clFinish(cnn->queue);
        
        double kernel_time = get_event_time_ms(event);
        total_time += kernel_time;
        
        if (kernel_time < min_time) min_time = kernel_time;
        if (kernel_time > max_time) max_time = kernel_time;
        
        if (iter % 10 == 0 || iter == ITERATIONS - 1) {
            printf("%4d | %9.2f | %19.2f\n", 
                   iter, kernel_time, 1000.0 / kernel_time);
        }
        
        clReleaseEvent(event);
    }
    
    double avg_time = total_time / ITERATIONS;
    
    printf("\n=== Inference Performance ===\n");
    printf("Average time:   %.2f ms\n", avg_time);
    printf("Min time:       %.2f ms\n", min_time);
    printf("Max time:       %.2f ms\n", max_time);
    printf("Throughput:     %.2f images/sec\n", 1000.0 / avg_time);
    printf("============================\n");
    
    /* Cleanup */
    free(h_input);
    clReleaseMemObject(input_buf);
    clReleaseMemObject(output_buf);
    clReleaseMemObject(weights);
    clReleaseMemObject(bias);
    clReleaseKernel(cnn->kernel);
    clReleaseProgram(cnn->program);
    clReleaseCommandQueue(cnn->queue);
    clReleaseContext(cnn->ctx);
    free(cnn);
    
    printf("\nBenchmark complete!\n");
    return 0;
}
