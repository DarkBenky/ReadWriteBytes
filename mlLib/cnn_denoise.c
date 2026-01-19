/* cnn_denoise.c - Implementation of OpenCL CNN Denoising Library */

#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_LAYERS 16
#define CHECK_CL(err, msg) if(err != CL_SUCCESS) { \
    fprintf(stderr, "OpenCL error %d at %s:%d - %s\n", err, __FILE__, __LINE__, msg); \
    return NULL; \
}

/* Internal layer representation */
typedef struct {
    int cin, cout, h, w, cin4;
    cl_mem weights, bias, output, grad_bias;
    float *h_weights, *h_bias, *h_grad_w, *h_grad_b;
    char name[64];
    int use_relu;
} ConvLayer;

/* Main CNN structure */
struct CNNDenoiser {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_forward, k_backward, k_weight_grad, k_mae_loss;
    
    CNNConfig config;
    int n_layers;
    ConvLayer layers[MAX_LAYERS];
    
    cl_mem input_buf, target_buf, grad_buf, temp_grad;
    
    TimingStats stats;
    int stats_count;
    int finalized;
};

/* Optimized OpenCL kernels - 4 outputs per thread */
static const char *kernel_source = 
"__kernel void conv3x3_forward_relu_f4(\n"
"    __global const float4* input, __global float* output,\n"
"    __global const float4* weights, __global const float* bias,\n"
"    int Cin4, int Cout, int H, int W)\n"
"{\n"
"    int x = get_global_id(0), y = get_global_id(1), oc = get_global_id(2) * 4;\n"
"    if (x <= 0 || y <= 0 || x >= W-1 || y >= H-1) return;\n"
"    \n"
"    int hw = H * W;\n"
"    \n"
"    float sum0 = (oc < Cout) ? bias[oc] : 0.0f;\n"
"    float sum1 = (oc + 1 < Cout) ? bias[oc + 1] : 0.0f;\n"
"    float sum2 = (oc + 2 < Cout) ? bias[oc + 2] : 0.0f;\n"
"    float sum3 = (oc + 3 < Cout) ? bias[oc + 3] : 0.0f;\n"
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
"        if (oc < Cout) {\n"
"            int wb = (oc * Cin4 + ic4) * 9;\n"
"            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];\n"
"            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];\n"
"            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];\n"
"            sum0 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);\n"
"        }\n"
"        if (oc + 1 < Cout) {\n"
"            int wb = ((oc+1) * Cin4 + ic4) * 9;\n"
"            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];\n"
"            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];\n"
"            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];\n"
"            sum1 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);\n"
"        }\n"
"        if (oc + 2 < Cout) {\n"
"            int wb = ((oc+2) * Cin4 + ic4) * 9;\n"
"            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];\n"
"            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];\n"
"            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];\n"
"            sum2 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);\n"
"        }\n"
"        if (oc + 3 < Cout) {\n"
"            int wb = ((oc+3) * Cin4 + ic4) * 9;\n"
"            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];\n"
"            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];\n"
"            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];\n"
"            sum3 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);\n"
"        }\n"
"    }\n"
"    \n"
"    if (oc < Cout) output[oc * hw + y * W + x] = fmax(sum0, 0.0f);\n"
"    if (oc + 1 < Cout) output[(oc + 1) * hw + y * W + x] = fmax(sum1, 0.0f);\n"
"    if (oc + 2 < Cout) output[(oc + 2) * hw + y * W + x] = fmax(sum2, 0.0f);\n"
"    if (oc + 3 < Cout) output[(oc + 3) * hw + y * W + x] = fmax(sum3, 0.0f);\n"
"}\n"
"\n"
"__kernel void conv3x3_backward_input_f4(\n"
"    __global const float* grad_out, __global const float* output,\n"
"    __global const float4* weights, __global float4* grad_in,\n"
"    int Cin4, int Cout, int H, int W, int use_relu)\n"
"{\n"
"    int x = get_global_id(0), y = get_global_id(1), ic4 = get_global_id(2);\n"
"    if (x <= 0 || y <= 0 || x >= W-1 || y >= H-1) return;\n"
"    \n"
"    int hw = H * W;\n"
"    float4 acc = (float4)(0.0f);\n"
"    \n"
"    for (int oc = 0; oc < Cout; oc++) {\n"
"        int oidx = oc * hw + y * W + x;\n"
"        float g = grad_out[oidx];\n"
"        if (use_relu && output[oidx] <= 0.0f) g = 0.0f;\n"
"        if (g == 0.0f) continue;\n"
"        \n"
"        int w_base = (oc * Cin4 + ic4) * 9;\n"
"        float4 w_sum = weights[w_base] + weights[w_base+1] + weights[w_base+2] +\n"
"                       weights[w_base+3] + weights[w_base+4] + weights[w_base+5] +\n"
"                       weights[w_base+6] + weights[w_base+7] + weights[w_base+8];\n"
"        acc += w_sum * g;\n"
"    }\n"
"    grad_in[ic4 * hw + y * W + x] = acc;\n"
"}\n"
"\n"
"__kernel void weight_grad_reduce(\n"
"    __global const float4* input, __global const float* grad_out,\n"
"    __global const float* output, __global float4* grad_w_vec,\n"
"    __global float* grad_b, int Cin4, int H, int W, int use_relu)\n"
"{\n"
"    int oc = get_global_id(0), ic4 = get_global_id(1), k = get_global_id(2);\n"
"    int hw = H * W, dy = (k / 3) - 1, dx = (k % 3) - 1;\n"
"    \n"
"    float4 sum = (float4)(0.0f);\n"
"    float bias_sum = 0.0f;\n"
"    \n"
"    for (int y = 1; y < H-1; y++) {\n"
"        for (int x = 1; x < W-1; x++) {\n"
"            int oidx = oc * hw + y * W + x;\n"
"            float g = grad_out[oidx];\n"
"            if (use_relu && output[oidx] <= 0.0f) g = 0.0f;\n"
"            if (g != 0.0f) {\n"
"                sum = fma(input[ic4 * hw + (y + dy) * W + (x + dx)], (float4)(g), sum);\n"
"                if (ic4 == 0 && k == 0) bias_sum += g;\n"
"            }\n"
"        }\n"
"    }\n"
"    grad_w_vec[(oc * Cin4 + ic4) * 9 + k] = sum;\n"
"    if (ic4 == 0 && k == 0) grad_b[oc] = bias_sum;\n"
"}\n"
"\n"
"__kernel void mae_loss_gradient(\n"
"    __global const float* prediction, __global const float* target,\n"
"    __global float* grad_out, __global float* loss_accum, int size)\n"
"{\n"
"    int gid = get_global_id(0), lid = get_local_id(0);\n"
"    __local float local_loss[256];\n"
"    float local_sum = 0.0f;\n"
"    \n"
"    for (int idx = gid; idx < size; idx += get_global_size(0)) {\n"
"        float diff = prediction[idx] - target[idx];\n"
"        grad_out[idx] = copysign(1.0f, diff);\n"
"        local_sum += fabs(diff);\n"
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
"    if (lid == 0) atomic_add_global(loss_accum, local_loss[0]);\n"
"}\n";

static void init_weights(float *w, int n) {
    float scale = sqrtf(2.0f / n);
    for (int i = 0; i < n; i++) {
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

CNNDenoiser* cnn_create(CNNConfig config) {
    CNNDenoiser *cnn = calloc(1, sizeof(CNNDenoiser));
    if (!cnn) return NULL;
    
    cnn->config = config;
    cnn->finalized = 0;
    
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cnn->ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(cnn);
        return NULL;
    }
    
    cl_command_queue_properties props = config.use_profiling ? CL_QUEUE_PROFILING_ENABLE : 0;
    cnn->queue = clCreateCommandQueue(cnn->ctx, device, props, &err);
    
    cnn->program = clCreateProgramWithSource(cnn->ctx, 1, &kernel_source, NULL, &err);
    const char *opts = "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(cnn->program, 0, NULL, opts, NULL, NULL);
    
    cnn->k_forward = clCreateKernel(cnn->program, "conv3x3_forward_relu_f4", &err);
    cnn->k_backward = clCreateKernel(cnn->program, "conv3x3_backward_input_f4", &err);
    cnn->k_weight_grad = clCreateKernel(cnn->program, "weight_grad_reduce", &err);
    cnn->k_mae_loss = clCreateKernel(cnn->program, "mae_loss_gradient", &err);
    
    return cnn;
}

int cnn_add_layer(CNNDenoiser *cnn, LayerConfig layer) {
    if (cnn->finalized) return -1;
    if (cnn->n_layers >= MAX_LAYERS) return -1;
    
    ConvLayer *l = &cnn->layers[cnn->n_layers++];
    l->cin = layer.cin;
    l->cout = layer.cout;
    l->use_relu = layer.use_relu;
    l->h = cnn->config.input_height;
    l->w = cnn->config.input_width;
    l->cin4 = layer.cin / 4;
    strncpy(l->name, layer.name, 63);
    
    int w_size = layer.cout * l->cin4 * 9;
    int out_size = layer.cout * l->h * l->w;
    
    posix_memalign((void**)&l->h_weights, 64, w_size * 16);
    posix_memalign((void**)&l->h_bias, 64, layer.cout * 4);
    posix_memalign((void**)&l->h_grad_w, 64, w_size * 16);
    posix_memalign((void**)&l->h_grad_b, 64, layer.cout * 4);
    
    memset(l->h_bias, 0, layer.cout * 4);
    init_weights(l->h_weights, w_size * 4);
    
    cl_int err;
    l->weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                w_size * 16, l->h_weights, &err);
    l->bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            layer.cout * 4, l->h_bias, &err);
    l->output = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, &err);
    l->grad_bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cout * 4, NULL, &err);
    
    return 0;
}

int cnn_finalize(CNNDenoiser *cnn) {
    int max_size = cnn->config.input_height * cnn->config.input_width * 
                   (cnn->config.input_channels > cnn->config.output_channels ? 
                    cnn->config.input_channels : cnn->config.output_channels);
    
    cnn->input_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    cnn->target_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    cnn->grad_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    
    int max_layer_params = 0;
    for (int i = 0; i < cnn->n_layers; i++) {
        int params = cnn->layers[i].cout * cnn->layers[i].cin4 * 9;
        if (params > max_layer_params) max_layer_params = params;
    }
    cnn->temp_grad = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_layer_params * 16, NULL, NULL);
    
    cnn->finalized = 1;
    return 0;
}

float cnn_train_step(CNNDenoiser *cnn, float* noisy_input, float* clean_target, int batch_size) {
    if (!cnn->finalized) return -1.0f;
    
    /* Upload data (simplified for single batch) */
    int input_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;
    clEnqueueWriteBuffer(cnn->queue, cnn->input_buf, CL_FALSE, 0, input_size * 4, 
                        noisy_input, 0, NULL, NULL);
    clEnqueueWriteBuffer(cnn->queue, cnn->target_buf, CL_FALSE, 0, input_size * 4,
                        clean_target, 0, NULL, NULL);
    
    /* Forward pass */
    cl_mem current = cnn->input_buf;
    for (int i = 0; i < cnn->n_layers; i++) {
        ConvLayer *l = &cnn->layers[i];
        
        clSetKernelArg(cnn->k_forward, 0, sizeof(cl_mem), &current);
        clSetKernelArg(cnn->k_forward, 1, sizeof(cl_mem), &l->output);
        clSetKernelArg(cnn->k_forward, 2, sizeof(cl_mem), &l->weights);
        clSetKernelArg(cnn->k_forward, 3, sizeof(cl_mem), &l->bias);
        clSetKernelArg(cnn->k_forward, 4, sizeof(int), &l->cin4);
        clSetKernelArg(cnn->k_forward, 5, sizeof(int), &l->cout);
        clSetKernelArg(cnn->k_forward, 6, sizeof(int), &l->h);
        clSetKernelArg(cnn->k_forward, 7, sizeof(int), &l->w);
        
        size_t global[3] = {l->w, l->h, (l->cout + 3) / 4};
        size_t local[3] = {16, 8, 1};
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL, global, local, 0, NULL, NULL);
        
        current = l->output;
    }
    
    /* Compute loss */
    int out_size = cnn->layers[cnn->n_layers - 1].cout * cnn->config.input_height * cnn->config.input_width;
    cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, 4, NULL, NULL);
    float zero = 0.0f;
    clEnqueueWriteBuffer(cnn->queue, loss_buf, CL_FALSE, 0, 4, &zero, 0, NULL, NULL);
    
    clSetKernelArg(cnn->k_mae_loss, 0, sizeof(cl_mem), &current);
    clSetKernelArg(cnn->k_mae_loss, 1, sizeof(cl_mem), &cnn->target_buf);
    clSetKernelArg(cnn->k_mae_loss, 2, sizeof(cl_mem), &cnn->grad_buf);
    clSetKernelArg(cnn->k_mae_loss, 3, sizeof(cl_mem), &loss_buf);
    clSetKernelArg(cnn->k_mae_loss, 4, sizeof(int), &out_size);
    
    size_t global_loss = 256 * 64;
    size_t local_loss = 256;
    clEnqueueNDRangeKernel(cnn->queue, cnn->k_mae_loss, 1, NULL, &global_loss, &local_loss, 0, NULL, NULL);
    
    float loss;
    clEnqueueReadBuffer(cnn->queue, loss_buf, CL_TRUE, 0, 4, &loss, 0, NULL, NULL);
    clReleaseMemObject(loss_buf);
    
    /* Backward pass and weight update would go here */
    /* Simplified for this example */
    
    clFinish(cnn->queue);
    return loss / out_size;
}

int cnn_get_num_parameters(CNNDenoiser *cnn) {
    int total = 0;
    for (int i = 0; i < cnn->n_layers; i++) {
        ConvLayer *l = &cnn->layers[i];
        total += l->cout * l->cin4 * 9 * 4;  /* weights */
        total += l->cout;                     /* biases */
    }
    return total;
}

void cnn_print_architecture(CNNDenoiser *cnn) {
    printf("\n=== CNN Architecture ===\n");
    printf("Input: %dx%dx%d\n", cnn->config.input_width, cnn->config.input_height, 
           cnn->config.input_channels);
    printf("Layers: %d\n", cnn->n_layers);
    for (int i = 0; i < cnn->n_layers; i++) {
        ConvLayer *l = &cnn->layers[i];
        printf("  [%d] %s: %d->%d channels, %s\n", i, l->name, l->cin, l->cout,
               l->use_relu ? "ReLU" : "Linear");
    }
    printf("Total parameters: %d\n", cnn_get_num_parameters(cnn));
    printf("========================\n\n");
}

void cnn_destroy(CNNDenoiser *cnn) {
    if (!cnn) return;
    
    for (int i = 0; i < cnn->n_layers; i++) {
        ConvLayer *l = &cnn->layers[i];
        free(l->h_weights);
        free(l->h_bias);
        free(l->h_grad_w);
        free(l->h_grad_b);
        clReleaseMemObject(l->weights);
        clReleaseMemObject(l->bias);
        clReleaseMemObject(l->output);
        clReleaseMemObject(l->grad_bias);
    }
    
    if (cnn->finalized) {
        clReleaseMemObject(cnn->input_buf);
        clReleaseMemObject(cnn->target_buf);
        clReleaseMemObject(cnn->grad_buf);
        clReleaseMemObject(cnn->temp_grad);
    }
    
    clReleaseKernel(cnn->k_forward);
    clReleaseKernel(cnn->k_backward);
    clReleaseKernel(cnn->k_weight_grad);
    clReleaseKernel(cnn->k_mae_loss);
    clReleaseProgram(cnn->program);
    clReleaseCommandQueue(cnn->queue);
    clReleaseContext(cnn->ctx);
    
    free(cnn);
}

void cnn_add_gaussian_noise(float* clean, float* noisy, int size, float sigma) {
    for (int i = 0; i < size; i++) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float noise = sigma * sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.283185f * u2);
        noisy[i] = clean[i] + noise;
    }
}

/* Helper: Convert RGB image to RGBA (RGB + Luminance) format for float4 processing */
void cnn_rgb_to_rgba_luminance(const unsigned char* rgb, float* rgba, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int rgb_idx = (y * width + x) * 3;
            int rgba_idx = (y * width + x) * 4;
            
            float r = rgb[rgb_idx + 0] / 255.0f;
            float g = rgb[rgb_idx + 1] / 255.0f;
            float b = rgb[rgb_idx + 2] / 255.0f;
            
            rgba[rgba_idx + 0] = r;
            rgba[rgba_idx + 1] = g;
            rgba[rgba_idx + 2] = b;
            rgba[rgba_idx + 3] = 0.299f * r + 0.587f * g + 0.114f * b;  /* Luminance */
        }
    }
}

/* Helper: Convert RGBA (RGB + Luminance) back to RGB image */
void cnn_rgba_luminance_to_rgb(const float* rgba, unsigned char* rgb, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int rgba_idx = (y * width + x) * 4;
            int rgb_idx = (y * width + x) * 3;
            
            rgb[rgb_idx + 0] = (unsigned char)(fminf(fmaxf(rgba[rgba_idx + 0], 0.0f), 1.0f) * 255.0f);
            rgb[rgb_idx + 1] = (unsigned char)(fminf(fmaxf(rgba[rgba_idx + 1], 0.0f), 1.0f) * 255.0f);
            rgb[rgb_idx + 2] = (unsigned char)(fminf(fmaxf(rgba[rgba_idx + 2], 0.0f), 1.0f) * 255.0f);
        }
    }
}

/* Helper: Prepare training batch - converts RGB to RGBA and adds noise */
int cnn_prepare_training_batch(const unsigned char* clean_rgb, unsigned char* noisy_rgb,
                                float* clean_rgba, float* noisy_rgba, 
                                int width, int height, float noise_sigma) {
    if (width != 800 || height != 600) {
        fprintf(stderr, "Error: Image must be 800x600 (got %dx%d)\n", width, height);
        return -1;
    }
    
    /* Convert clean RGB to RGBA */
    cnn_rgb_to_rgba_luminance(clean_rgb, clean_rgba, width, height);
    
    /* Add noise to RGBA */
    int rgba_size = width * height * 4;
    cnn_add_gaussian_noise(clean_rgba, noisy_rgba, rgba_size, noise_sigma);
    
    /* Clamp noisy values */
    for (int i = 0; i < rgba_size; i++) {
        noisy_rgba[i] = fminf(fmaxf(noisy_rgba[i], 0.0f), 1.0f);
    }
    
    /* Convert back to RGB for visualization if needed */
    if (noisy_rgb) {
        cnn_rgba_luminance_to_rgb(noisy_rgba, noisy_rgb, width, height);
    }
    
    return 0;
}

/* Helper: Simple inference from RGB image */
int cnn_inference_rgb(CNNDenoiser* cnn, const unsigned char* input_rgb, 
                      unsigned char* output_rgb, int width, int height) {
    if (!cnn || !cnn->finalized) return -1;
    if (width != 800 || height != 600) {
        fprintf(stderr, "Error: Image must be 800x600 (got %dx%d)\n", width, height);
        return -1;
    }
    
    int rgba_size = width * height * 4;
    float *input_rgba = malloc(rgba_size * sizeof(float));
    float *output_rgba = malloc(rgba_size * sizeof(float));
    
    /* Convert input RGB to RGBA */
    cnn_rgb_to_rgba_luminance(input_rgb, input_rgba, width, height);
    
    /* Run inference through network */
    clEnqueueWriteBuffer(cnn->queue, cnn->input_buf, CL_TRUE, 0, 
                        rgba_size * sizeof(float), input_rgba, 0, NULL, NULL);
    
    cl_mem current = cnn->input_buf;
    for (int i = 0; i < cnn->n_layers; i++) {
        ConvLayer *l = &cnn->layers[i];
        
        clSetKernelArg(cnn->k_forward, 0, sizeof(cl_mem), &current);
        clSetKernelArg(cnn->k_forward, 1, sizeof(cl_mem), &l->output);
        clSetKernelArg(cnn->k_forward, 2, sizeof(cl_mem), &l->weights);
        clSetKernelArg(cnn->k_forward, 3, sizeof(cl_mem), &l->bias);
        clSetKernelArg(cnn->k_forward, 4, sizeof(int), &l->cin4);
        clSetKernelArg(cnn->k_forward, 5, sizeof(int), &l->cout);
        clSetKernelArg(cnn->k_forward, 6, sizeof(int), &l->h);
        clSetKernelArg(cnn->k_forward, 7, sizeof(int), &l->w);
        
        size_t global[3] = {l->w, l->h, (l->cout + 3) / 4};
        size_t local[3] = {16, 8, 1};
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL, global, local, 0, NULL, NULL);
        
        current = l->output;
    }
    
    /* Read back result */
    clEnqueueReadBuffer(cnn->queue, current, CL_TRUE, 0, 
                       rgba_size * sizeof(float), output_rgba, 0, NULL, NULL);
    
    /* Convert output RGBA to RGB */
    cnn_rgba_luminance_to_rgb(output_rgba, output_rgb, width, height);
    
    free(input_rgba);
    free(output_rgba);
    
    return 0;
}

