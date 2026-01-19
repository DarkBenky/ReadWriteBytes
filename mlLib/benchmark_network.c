/* Multi-layer CNN inference benchmark - realistic denoising network */

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

/* Multi-output channel kernel - 2 outputs per thread */
const char *kernel_source = 
"__kernel void conv3x3_2out(\n"
"    __global const float4* input,\n"
"    __global float4* output,\n"
"    __global const float4* weights,\n"
"    __global const float* bias,\n"
"    int Cin4, int Cout, int H, int W)\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    int oc = get_global_id(2) * 2;\n"
"    \n"
"    if (x <= 0 || y <= 0 || x >= W-1 || y >= H-1 || oc >= Cout) return;\n"
"    \n"
"    int hw = H * W;\n"
"    \n"
"    float sum0 = bias[oc];\n"
"    float sum1 = (oc + 1 < Cout) ? bias[oc + 1] : 0.0f;\n"
"    \n"
"    for (int ic4 = 0; ic4 < Cin4; ic4++) {\n"
"        int base = ic4 * hw + y * W + x;\n"
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
"        int w_base0 = (oc * Cin4 + ic4) * 9;\n"
"        sum0 += dot(i0, weights[w_base0 + 0]);\n"
"        sum0 += dot(i1, weights[w_base0 + 1]);\n"
"        sum0 += dot(i2, weights[w_base0 + 2]);\n"
"        sum0 += dot(i3, weights[w_base0 + 3]);\n"
"        sum0 += dot(i4, weights[w_base0 + 4]);\n"
"        sum0 += dot(i5, weights[w_base0 + 5]);\n"
"        sum0 += dot(i6, weights[w_base0 + 6]);\n"
"        sum0 += dot(i7, weights[w_base0 + 7]);\n"
"        sum0 += dot(i8, weights[w_base0 + 8]);\n"
"        \n"
"        if (oc + 1 < Cout) {\n"
"            int w_base1 = ((oc + 1) * Cin4 + ic4) * 9;\n"
"            sum1 += dot(i0, weights[w_base1 + 0]);\n"
"            sum1 += dot(i1, weights[w_base1 + 1]);\n"
"            sum1 += dot(i2, weights[w_base1 + 2]);\n"
"            sum1 += dot(i3, weights[w_base1 + 3]);\n"
"            sum1 += dot(i4, weights[w_base1 + 4]);\n"
"            sum1 += dot(i5, weights[w_base1 + 5]);\n"
"            sum1 += dot(i6, weights[w_base1 + 6]);\n"
"            sum1 += dot(i7, weights[w_base1 + 7]);\n"
"            sum1 += dot(i8, weights[w_base1 + 8]);\n"
"        }\n"
"    }\n"
"    \n"
"    int out_base = oc / 4 * hw + y * W + x;\n"
"    int lane = oc % 4;\n"
"    \n"
"    float4 val = output[out_base];\n"
"    if (lane == 0) val.x = fmax(sum0, 0.0f);\n"
"    else if (lane == 1) val.y = fmax(sum0, 0.0f);\n"
"    else if (lane == 2) val.z = fmax(sum0, 0.0f);\n"
"    else val.w = fmax(sum0, 0.0f);\n"
"    output[out_base] = val;\n"
"    \n"
"    if (oc + 1 < Cout) {\n"
"        int lane1 = (oc + 1) % 4;\n"
"        float4 val1 = output[out_base + (lane1 < lane ? hw : 0)];\n"
"        if (lane1 == 0) val1.x = fmax(sum1, 0.0f);\n"
"        else if (lane1 == 1) val1.y = fmax(sum1, 0.0f);\n"
"        else if (lane1 == 2) val1.z = fmax(sum1, 0.0f);\n"
"        else val1.w = fmax(sum1, 0.0f);\n"
"        output[out_base + (lane1 < lane ? hw : 0)] = val1;\n"
"    }\n"
"}\n";

typedef struct {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
} NetworkCNN;

NetworkCNN* create_network() {
    NetworkCNN *net = calloc(1, sizeof(NetworkCNN));
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &net->device, NULL);
    
    net->ctx = clCreateContext(NULL, 1, &net->device, NULL, NULL, NULL);
    net->queue = clCreateCommandQueue(net->ctx, net->device, CL_QUEUE_PROFILING_ENABLE, NULL);
    
    cl_int err;
    net->program = clCreateProgramWithSource(net->ctx, 1, &kernel_source, NULL, &err);
    CHECK_CL(err);
    
    const char *opts = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(net->program, 1, &net->device, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[8192];
        clGetProgramBuildInfo(net->program, net->device, CL_PROGRAM_BUILD_LOG, 8192, log, NULL);
        printf("Build error:\n%s\n", log);
        exit(1);
    }
    
    net->kernel = clCreateKernel(net->program, "conv3x3_2out", &err);
    CHECK_CL(err);
    
    return net;
}

void run_layer(NetworkCNN *net, cl_mem input, cl_mem output, cl_mem weights, cl_mem bias,
               int cin, int cout, int H, int W, cl_event *event) {
    int cin4 = (cin + 3) / 4;
    
    clSetKernelArg(net->kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(net->kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(net->kernel, 2, sizeof(cl_mem), &weights);
    clSetKernelArg(net->kernel, 3, sizeof(cl_mem), &bias);
    clSetKernelArg(net->kernel, 4, sizeof(int), &cin4);
    clSetKernelArg(net->kernel, 5, sizeof(int), &cout);
    clSetKernelArg(net->kernel, 6, sizeof(int), &H);
    clSetKernelArg(net->kernel, 7, sizeof(int), &W);
    
    size_t global[3] = {W, H, (cout + 1) / 2};
    size_t local[3] = {16, 8, 1};
    
    clEnqueueNDRangeKernel(net->queue, net->kernel, 3, NULL, global, local, 0, NULL, event);
}

double get_event_time_ms(cl_event event) {
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    return (end - start) / 1e6;
}

int main() {
    srand(time(NULL));
    
    printf("================================================\n");
    printf("  Multi-Layer CNN Network Inference Benchmark  \n");
    printf("================================================\n");
    
    int H = 600, W = 800;
    int ITERATIONS = 100;
    
    /* Encoder-Decoder architecture: 3 -> 32 -> 64 -> 32 -> 3 */
    int layers[] = {3, 32, 64, 32, 3};
    int num_layers = 4;
    
    printf("\nNetwork Architecture:\n");
    printf("  Input: %dx%dx%d\n", W, H, layers[0]);
    for (int i = 0; i < num_layers; i++) {
        printf("  Layer %d: Conv3x3(%d -> %d) + ReLU\n", i+1, layers[i], layers[i+1]);
    }
    printf("  Iterations: %d\n\n", ITERATIONS);
    
    NetworkCNN *net = create_network();
    
    /* Allocate all buffers */
    cl_mem buffers[5];
    for (int i = 0; i < 5; i++) {
        int c4 = (layers[i] + 3) / 4;
        buffers[i] = clCreateBuffer(net->ctx, CL_MEM_READ_WRITE, c4 * H * W * 16, NULL, NULL);
    }
    
    /* Allocate weights and biases for each layer */
    cl_mem weights[4], bias[4];
    for (int i = 0; i < num_layers; i++) {
        int cin4 = (layers[i] + 3) / 4;
        int cout = layers[i+1];
        weights[i] = clCreateBuffer(net->ctx, CL_MEM_READ_ONLY, cout * cin4 * 9 * 16, NULL, NULL);
        bias[i] = clCreateBuffer(net->ctx, CL_MEM_READ_ONLY, cout * 4, NULL, NULL);
    }
    
    /* Initialize input */
    int c4_input = (layers[0] + 3) / 4;
    float *h_input = malloc(c4_input * H * W * 16);
    for (int i = 0; i < c4_input * H * W * 4; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    clEnqueueWriteBuffer(net->queue, buffers[0], CL_TRUE, 0, c4_input * H * W * 16, h_input, 0, NULL, NULL);
    
    /* Warmup */
    printf("Warming up (10 iterations)...\n");
    for (int i = 0; i < 10; i++) {
        for (int L = 0; L < num_layers; L++) {
            run_layer(net, buffers[L], buffers[L+1], weights[L], bias[L], 
                     layers[L], layers[L+1], H, W, NULL);
        }
        clFinish(net->queue);
        printf(".");
        fflush(stdout);
    }
    printf(" done!\n\n");
    
    /* Benchmark */
    printf("Running network inference benchmark...\n");
    printf("Iter | Total (ms) | Layer1 | Layer2 | Layer3 | Layer4 | FPS\n");
    printf("-----|------------|--------|--------|--------|--------|---------\n");
    
    double total_time = 0;
    double min_time = 1e9, max_time = 0;
    
    for (int iter = 0; iter < ITERATIONS; iter++) {
        cl_event events[4];
        
        for (int L = 0; L < num_layers; L++) {
            run_layer(net, buffers[L], buffers[L+1], weights[L], bias[L], 
                     layers[L], layers[L+1], H, W, &events[L]);
        }
        
        clFinish(net->queue);
        
        double layer_times[4];
        double iter_total = 0;
        for (int L = 0; L < num_layers; L++) {
            layer_times[L] = get_event_time_ms(events[L]);
            iter_total += layer_times[L];
            clReleaseEvent(events[L]);
        }
        
        total_time += iter_total;
        if (iter_total < min_time) min_time = iter_total;
        if (iter_total > max_time) max_time = iter_total;
        
        if (iter % 10 == 0 || iter == ITERATIONS - 1) {
            printf("%4d | %10.2f | %6.2f | %6.2f | %6.2f | %6.2f | %7.1f\n", 
                   iter, iter_total, layer_times[0], layer_times[1], 
                   layer_times[2], layer_times[3], 1000.0 / iter_total);
        }
    }
    
    double avg_time = total_time / ITERATIONS;
    
    printf("\n=== Network Performance ===\n");
    printf("Average inference:  %.2f ms\n", avg_time);
    printf("Min inference:      %.2f ms\n", min_time);
    printf("Max inference:      %.2f ms\n", max_time);
    printf("Throughput:         %.1f FPS\n", 1000.0 / avg_time);
    printf("===========================\n");
    
    /* Cleanup */
    free(h_input);
    for (int i = 0; i < 5; i++) clReleaseMemObject(buffers[i]);
    for (int i = 0; i < 4; i++) {
        clReleaseMemObject(weights[i]);
        clReleaseMemObject(bias[i]);
    }
    clReleaseKernel(net->kernel);
    clReleaseProgram(net->program);
    clReleaseCommandQueue(net->queue);
    clReleaseContext(net->ctx);
    free(net);
    
    printf("\nBenchmark complete!\n");
    return 0;
}
