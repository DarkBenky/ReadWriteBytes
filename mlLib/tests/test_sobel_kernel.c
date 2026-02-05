#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *sobel_kernel_src = "                                         \n"
"__kernel void test_sobel(                                               \n"
"    __global const float* input,                                        \n"
"    __global float* output,                                             \n"
"    int W, int H)                                                       \n"
"{                                                                       \n"
"    int x = get_global_id(0);                                          \n"
"    int y = get_global_id(1);                                          \n"
"    if (x >= W || y >= H) return;                                      \n"
"    if (x > 0 && y > 0 && x < W-1 && y < H-1) {                        \n"
"        int idx = y * W + x;                                           \n"
"        float sobel_x = -input[idx - W - 1] - 2.0f * input[idx - 1] - input[idx + W - 1]  \n"
"                      + input[idx - W + 1] + 2.0f * input[idx + 1] + input[idx + W + 1];  \n"
"        output[idx] = sobel_x;                                         \n"
"    } else {                                                           \n"
"        output[y * W + x] = 0.0f;                                      \n"
"    }                                                                  \n"
"}                                                                      \n";

int main() {
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, NULL);
    
    cl_program program = clCreateProgramWithSource(ctx, 1, &sobel_kernel_src, NULL, NULL);
    cl_int err = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build log:\n%s\n", log);
        return 1;
    }
    
    cl_kernel kernel = clCreateKernel(program, "test_sobel", NULL);
    
    int W = 8, H = 8;
    float *input = calloc(W * H, sizeof(float));
    float *output = calloc(W * H, sizeof(float));
    
    /* Create a step edge in the middle */
    for (int y = 0; y < H; y++) {
        for (int x = W/2; x < W; x++) {
            input[y * W + x] = 1.0f;
        }
    }
    
    cl_mem d_input = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    W * H * sizeof(float), input, NULL);
    cl_mem d_output = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, W * H * sizeof(float), NULL, NULL);
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_output);
    clSetKernelArg(kernel, 2, sizeof(int), &W);
    clSetKernelArg(kernel, 3, sizeof(int), &H);
    
    size_t global[2] = {W, H};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    clFinish(queue);
    
    clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, W * H * sizeof(float), output, 0, NULL, NULL);
    
    printf("Sobel X gradient (should show edge at x=4):\n");
    for (int y = 1; y < H-1; y++) {
        for (int x = 1; x < W-1; x++) {
            printf("%6.2f ", output[y * W + x]);
        }
        printf("\n");
    }
    
    free(input);
    free(output);
    clReleaseMemObject(d_input);
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    
    return 0;
}
