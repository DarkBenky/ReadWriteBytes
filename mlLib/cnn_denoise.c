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
    cl_mem weights, bias, output, grad_bias, grad_weights, grad_input;
    cl_mem adam_m_w, adam_v_w, adam_m_b, adam_v_b;  /* Adam optimizer buffers */
    float *h_weights, *h_bias, *h_grad_w, *h_grad_b;
    char name[64];
    int use_relu;
} ConvLayer;

/* Main CNN structure */
struct CNNDenoiser {
    cl_context ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_forward, k_backward, k_weight_grad, k_mae_loss, k_sgd_update, k_adam_update;
    cl_kernel k_mse_loss, k_laplace_loss, k_add_weighted_grad;
    
    CNNConfig config;
    int n_layers;
    ConvLayer layers[MAX_LAYERS];
    int adam_t;  /* Adam timestep */
    
    cl_mem input_buf, target_buf, grad_buf, temp_grad, residual_buf;
    
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
"    if (lid == 0) loss_accum[get_group_id(0)] = local_loss[0];\n"
"}\n"
"\n"
"__kernel void sgd_update(\n"
"    __global float4* weights, __global float* bias,\n"
"    __global const float4* grad_w, __global const float* grad_b,\n"
"    float lr, int w_size, int b_size)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    \n"
"    if (gid < w_size) {\n"
"        float4 g = grad_w[gid] * lr;\n"
"        g = clamp(g, (float4)(-1.0f), (float4)(1.0f));\n"
"        weights[gid] -= g;\n"
"    }\n"
"    \n"
"    if (gid < b_size) {\n"
"        float g = clamp(grad_b[gid] * lr, -1.0f, 1.0f);\n"
"        bias[gid] -= g;\n"
"    }\n"
"}\n"
"\n"
"__kernel void adam_update(\n"
"    __global float4* weights, __global float* bias,\n"
"    __global const float4* grad_w, __global const float* grad_b,\n"
"    __global float4* m_w, __global float* m_b,\n"
"    __global float4* v_w, __global float* v_b,\n"
"    float lr, float beta1, float beta2, float epsilon, int t,\n"
"    int w_size, int b_size)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    float bias_correction1 = 1.0f - pown(beta1, t);\n"
"    float bias_correction2 = 1.0f - pown(beta2, t);\n"
"    float lr_t = lr * sqrt(bias_correction2) / bias_correction1;\n"
"    \n"
"    if (gid < w_size) {\n"
"        float4 g = clamp(grad_w[gid], (float4)(-1.0f), (float4)(1.0f));\n"
"        float4 m = beta1 * m_w[gid] + (1.0f - beta1) * g;\n"
"        float4 v = beta2 * v_w[gid] + (1.0f - beta2) * g * g;\n"
"        m_w[gid] = m;\n"
"        v_w[gid] = v;\n"
"        weights[gid] -= lr_t * m / (sqrt(v) + epsilon);\n"
"    }\n"
"    \n"
"    if (gid < b_size) {\n"
"        float g = clamp(grad_b[gid], -1.0f, 1.0f);\n"
"        float m = beta1 * m_b[gid] + (1.0f - beta1) * g;\n"
"        float v = beta2 * v_b[gid] + (1.0f - beta2) * g * g;\n"
"        m_b[gid] = m;\n"
"        v_b[gid] = v;\n"
"        bias[gid] -= lr_t * m / (sqrt(v) + epsilon);\n"
"    }\n"
"}\n"
"\n"
"__kernel void mse_loss_gradient(\n"
"    __global const float* output, __global const float* target,\n"
"    __global float* grad, __global float* loss_accum,\n"
"    int size, __local float* local_loss)\n"
"{\n"
"    int gid = get_global_id(0), lid = get_local_id(0);\n"
"    local_loss[lid] = 0.0f;\n"
"    \n"
"    if (gid < size) {\n"
"        float diff = output[gid] - target[gid];\n"
"        grad[gid] = 2.0f * diff;\n"
"        local_loss[lid] = diff * diff;\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    \n"
"    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
"        if (lid < s) local_loss[lid] += local_loss[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    if (lid == 0) loss_accum[get_group_id(0)] = local_loss[0];\n"
"}\n"
"\n"
"__kernel void laplace_loss_gradient(\n"
"    __global const float* output, __global const float* target,\n"
"    __global float* grad, __global float* loss_accum,\n"
"    int H, int W, int C, __local float* local_loss)\n"
"{\n"
"    int gid = get_global_id(0), lid = get_local_id(0);\n"
"    int size = H * W * C;\n"
"    local_loss[lid] = 0.0f;\n"
"    \n"
"    if (gid < size) {\n"
"        int x = (gid / C) % W;\n"
"        int y = (gid / C) / W;\n"
"        \n"
"        if (x > 0 && y > 0 && x < W-1 && y < H-1) {\n"
"            int c = gid % C;\n"
"            int idx = c * H * W + y * W + x;\n"
"            \n"
"            float lap_out = -4.0f * output[idx] +\n"
"                            output[idx - 1] + output[idx + 1] +\n"
"                            output[idx - W] + output[idx + W];\n"
"            \n"
"            float lap_tgt = -4.0f * target[idx] +\n"
"                            target[idx - 1] + target[idx + 1] +\n"
"                            target[idx - W] + target[idx + W];\n"
"            \n"
"            float diff = lap_out - lap_tgt;\n"
"            grad[idx] = (diff > 0.0f) ? 1.0f : -1.0f;\n"
"            local_loss[lid] = fabs(diff);\n"
"        }\n"
"    }\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    \n"
"    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {\n"
"        if (lid < s) local_loss[lid] += local_loss[lid + s];\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    if (lid == 0) loss_accum[get_group_id(0)] = local_loss[0];\n"
"}\n"
"\n"
"__kernel void add_weighted_grad(\n"
"    __global float* grad_accum, __global const float* grad_new,\n"
"    float weight, int size)\n"
"{\n"
"    int gid = get_global_id(0);\n"
"    if (gid < size) grad_accum[gid] += weight * grad_new[gid];\n"
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
    
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(cnn->program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(cnn->program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "OpenCL kernel build failed:\n%s\n", log);
        free(log);
        return NULL;
    }
    
    cnn->k_forward = clCreateKernel(cnn->program, "conv3x3_forward_relu_f4", &err);
    cnn->k_backward = clCreateKernel(cnn->program, "conv3x3_backward_input_f4", &err);
    cnn->k_weight_grad = clCreateKernel(cnn->program, "weight_grad_reduce", &err);
    cnn->k_mae_loss = clCreateKernel(cnn->program, "mae_loss_gradient", &err);
    cnn->k_mse_loss = clCreateKernel(cnn->program, "mse_loss_gradient", &err);
    cnn->k_laplace_loss = clCreateKernel(cnn->program, "laplace_loss_gradient", &err);
    cnn->k_sgd_update = clCreateKernel(cnn->program, "sgd_update", &err);
    cnn->k_adam_update = clCreateKernel(cnn->program, "adam_update", &err);
    cnn->k_add_weighted_grad = clCreateKernel(cnn->program, "add_weighted_grad", &err);
    
    cnn->adam_t = 0;
    
    return cnn;
}

CNNConfig cnn_default_config(int width, int height, int channels) {
    CNNConfig cfg;
    cfg.input_width = width;
    cfg.input_height = height;
    cfg.input_channels = channels;
    cfg.output_channels = channels;
    cfg.learning_rate = 0.00001f;
    cfg.use_profiling = 0;
    cfg.residual_mode = 0;
    cfg.optimizer = OPTIMIZER_SGD;
    cfg.loss_config.num_losses = 1;
    cfg.loss_config.types[0] = LOSS_MAE;
    cfg.loss_config.weights[0] = 1.0f;
    cfg.adam_beta1 = 0.9f;
    cfg.adam_beta2 = 0.999f;
    cfg.adam_epsilon = 1e-8f;
    return cfg;
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
    l->weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                w_size * 16, l->h_weights, &err);
    l->bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            layer.cout * 4, l->h_bias, &err);
    l->output = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, &err);
    l->grad_bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cout * 4, NULL, &err);
    l->grad_weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, w_size * 16, NULL, &err);
    l->grad_input = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cin * l->h * l->w * 4, NULL, &err);
    
    return 0;
}

int cnn_finalize(CNNDenoiser *cnn) {
    int max_size = cnn->config.input_height * cnn->config.input_width * 
                   (cnn->config.input_channels > cnn->config.output_channels ? 
                    cnn->config.input_channels : cnn->config.output_channels);
    
    cnn->input_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    cnn->target_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    cnn->grad_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    cnn->residual_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
    
    int max_layer_params = 0;
    for (int i = 0; i < cnn->n_layers; i++) {
        int params = cnn->layers[i].cout * cnn->layers[i].cin4 * 9;
        if (params > max_layer_params) max_layer_params = params;
    }
    cnn->temp_grad = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_layer_params * 16, NULL, NULL);
    
    /* Allocate Adam optimizer buffers if using Adam */
    if (cnn->config.optimizer == OPTIMIZER_ADAM) {
        for (int i = 0; i < cnn->n_layers; i++) {
            ConvLayer *l = &cnn->layers[i];
            int w_size = l->cout * l->cin4 * 9;
            l->adam_m_w = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, w_size * 16, NULL, NULL);
            l->adam_v_w = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, w_size * 16, NULL, NULL);
            l->adam_m_b = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, l->cout * 4, NULL, NULL);
            l->adam_v_b = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, l->cout * 4, NULL, NULL);
            
            /* Initialize to zero */
            float zero = 0.0f;
            clEnqueueFillBuffer(cnn->queue, l->adam_m_w, &zero, sizeof(float), 0, w_size * 16, 0, NULL, NULL);
            clEnqueueFillBuffer(cnn->queue, l->adam_v_w, &zero, sizeof(float), 0, w_size * 16, 0, NULL, NULL);
            clEnqueueFillBuffer(cnn->queue, l->adam_m_b, &zero, sizeof(float), 0, l->cout * 4, 0, NULL, NULL);
            clEnqueueFillBuffer(cnn->queue, l->adam_v_b, &zero, sizeof(float), 0, l->cout * 4, 0, NULL, NULL);
        }
    }
    
    cnn->finalized = 1;
    return 0;
}

float cnn_train_step(CNNDenoiser *cnn, float* noisy_input, float* clean_target, int batch_size) {
    if (!cnn->finalized) return -1.0f;
    
    int input_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;
    int hw = cnn->config.input_height * cnn->config.input_width;
    
    /* Upload input */
    clEnqueueWriteBuffer(cnn->queue, cnn->input_buf, CL_FALSE, 0, input_size * 4, 
                        noisy_input, 0, NULL, NULL);
    
    /* In residual mode, target is the noise (clean_target = noise), output = input - prediction
     * So we need to compute: target_for_network = input - clean_image
     * But user passes noise directly, so just use clean_target as-is */
    clEnqueueWriteBuffer(cnn->queue, cnn->target_buf, CL_FALSE, 0, input_size * 4,
                        clean_target, 0, NULL, NULL);
    
    /* ========== FORWARD PASS ========== */
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
    
    /* ========== COMPUTE LOSS & GRADIENT ========== */
    ConvLayer *last_layer = &cnn->layers[cnn->n_layers - 1];
    int out_size = last_layer->cout * hw;
    float total_loss = 0.0f;
    
    /* Zero out gradient buffer */
    float zero = 0.0f;
    clEnqueueFillBuffer(cnn->queue, cnn->grad_buf, &zero, sizeof(float), 0, out_size * 4, 0, NULL, NULL);
    
    /* Temporary buffer for individual loss gradients */
    cl_mem temp_grad_loss = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, NULL);
    
    /* Compute each loss and accumulate gradients */
    for (int loss_idx = 0; loss_idx < cnn->config.loss_config.num_losses; loss_idx++) {
        LossType loss_type = cnn->config.loss_config.types[loss_idx];
        float weight = cnn->config.loss_config.weights[loss_idx];
        
        if (loss_type == LOSS_MAE) {
            int num_workgroups = 64;
            cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, num_workgroups * 4, NULL, NULL);
            
            clSetKernelArg(cnn->k_mae_loss, 0, sizeof(cl_mem), &current);
            clSetKernelArg(cnn->k_mae_loss, 1, sizeof(cl_mem), &cnn->target_buf);
            clSetKernelArg(cnn->k_mae_loss, 2, sizeof(cl_mem), &temp_grad_loss);
            clSetKernelArg(cnn->k_mae_loss, 3, sizeof(cl_mem), &loss_buf);
            clSetKernelArg(cnn->k_mae_loss, 4, sizeof(int), &out_size);
            
            size_t global_loss = 256 * num_workgroups;
            size_t local_loss = 256;
            clEnqueueNDRangeKernel(cnn->queue, cnn->k_mae_loss, 1, NULL, &global_loss, &local_loss, 0, NULL, NULL);
            
            float loss_per_wg[64];
            clEnqueueReadBuffer(cnn->queue, loss_buf, CL_TRUE, 0, num_workgroups * 4, loss_per_wg, 0, NULL, NULL);
            
            float loss = 0.0f;
            for (int i = 0; i < num_workgroups; i++) loss += loss_per_wg[i];
            total_loss += weight * (loss / out_size);
            
            clReleaseMemObject(loss_buf);
        } else if (loss_type == LOSS_MSE) {
            int num_workgroups = 64;
            cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, num_workgroups * 4, NULL, NULL);
            
            clSetKernelArg(cnn->k_mse_loss, 0, sizeof(cl_mem), &current);
            clSetKernelArg(cnn->k_mse_loss, 1, sizeof(cl_mem), &cnn->target_buf);
            clSetKernelArg(cnn->k_mse_loss, 2, sizeof(cl_mem), &temp_grad_loss);
            clSetKernelArg(cnn->k_mse_loss, 3, sizeof(cl_mem), &loss_buf);
            clSetKernelArg(cnn->k_mse_loss, 4, sizeof(int), &out_size);
            
            size_t global_loss = 256 * num_workgroups;
            size_t local_loss = 256;
            clEnqueueNDRangeKernel(cnn->queue, cnn->k_mse_loss, 1, NULL, &global_loss, &local_loss, 0, NULL, NULL);
            
            float loss_per_wg[64];
            clEnqueueReadBuffer(cnn->queue, loss_buf, CL_TRUE, 0, num_workgroups * 4, loss_per_wg, 0, NULL, NULL);
            
            float loss = 0.0f;
            for (int i = 0; i < num_workgroups; i++) loss += loss_per_wg[i];
            total_loss += weight * (loss / out_size);
            
            clReleaseMemObject(loss_buf);
        } else if (loss_type == LOSS_LAPLACE) {
            int num_workgroups = 64;
            cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, num_workgroups * 4, NULL, NULL);
            float zero_loss = 0.0f;
            clEnqueueFillBuffer(cnn->queue, loss_buf, &zero_loss, sizeof(float), 0, num_workgroups * 4, 0, NULL, NULL);
            
            int H = cnn->config.input_height;
            int W = cnn->config.input_width;
            int C = last_layer->cout;
            
            clSetKernelArg(cnn->k_laplace_loss, 0, sizeof(cl_mem), &current);
            clSetKernelArg(cnn->k_laplace_loss, 1, sizeof(cl_mem), &cnn->target_buf);
            clSetKernelArg(cnn->k_laplace_loss, 2, sizeof(cl_mem), &temp_grad_loss);
            clSetKernelArg(cnn->k_laplace_loss, 3, sizeof(cl_mem), &loss_buf);
            clSetKernelArg(cnn->k_laplace_loss, 4, sizeof(int), &H);
            clSetKernelArg(cnn->k_laplace_loss, 5, sizeof(int), &W);
            clSetKernelArg(cnn->k_laplace_loss, 6, sizeof(int), &C);
            
            size_t global_loss = 256 * num_workgroups;
            size_t local_loss = 256;
            clEnqueueNDRangeKernel(cnn->queue, cnn->k_laplace_loss, 1, NULL, &global_loss, &local_loss, 0, NULL, NULL);
            
            float loss_per_wg[64];
            clEnqueueReadBuffer(cnn->queue, loss_buf, CL_TRUE, 0, num_workgroups * 4, loss_per_wg, 0, NULL, NULL);
            
            float loss = 0.0f;
            for (int i = 0; i < num_workgroups; i++) loss += loss_per_wg[i];
            total_loss += weight * (loss / out_size);
            
            clReleaseMemObject(loss_buf);
        }
        
        /* Add weighted gradient to accumulated gradient buffer */
        clSetKernelArg(cnn->k_add_weighted_grad, 0, sizeof(cl_mem), &cnn->grad_buf);
        clSetKernelArg(cnn->k_add_weighted_grad, 1, sizeof(cl_mem), &temp_grad_loss);
        clSetKernelArg(cnn->k_add_weighted_grad, 2, sizeof(float), &weight);
        clSetKernelArg(cnn->k_add_weighted_grad, 3, sizeof(int), &out_size);
        
        size_t global_add = ((out_size + 255) / 256) * 256;
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_add_weighted_grad, 1, NULL, &global_add, NULL, 0, NULL, NULL);
    }
    
    clReleaseMemObject(temp_grad_loss);
    
    /* ========== BACKWARD PASS ========== */
    cl_mem grad_current = cnn->grad_buf;
    
    /* First pass: Compute all gradients */
    for (int i = cnn->n_layers - 1; i >= 0; i--) {
        ConvLayer *l = &cnn->layers[i];
        cl_mem layer_input = (i == 0) ? cnn->input_buf : cnn->layers[i-1].output;
        
        clSetKernelArg(cnn->k_weight_grad, 0, sizeof(cl_mem), &layer_input);
        clSetKernelArg(cnn->k_weight_grad, 1, sizeof(cl_mem), &grad_current);
        clSetKernelArg(cnn->k_weight_grad, 2, sizeof(cl_mem), &l->output);
        clSetKernelArg(cnn->k_weight_grad, 3, sizeof(cl_mem), &l->grad_weights);
        clSetKernelArg(cnn->k_weight_grad, 4, sizeof(cl_mem), &l->grad_bias);
        clSetKernelArg(cnn->k_weight_grad, 5, sizeof(int), &l->cin4);
        clSetKernelArg(cnn->k_weight_grad, 6, sizeof(int), &l->h);
        clSetKernelArg(cnn->k_weight_grad, 7, sizeof(int), &l->w);
        clSetKernelArg(cnn->k_weight_grad, 8, sizeof(int), &l->use_relu);
        
        size_t grad_global[3] = {(size_t)l->cout, (size_t)l->cin4, 9};
        clEnqueueNDRangeKernel(cnn->queue, cnn->k_weight_grad, 3, NULL, grad_global, NULL, 0, NULL, NULL);
        
        if (i > 0) {
            int prev_cin4 = cnn->layers[i-1].cout / 4;
            
            clSetKernelArg(cnn->k_backward, 0, sizeof(cl_mem), &grad_current);
            clSetKernelArg(cnn->k_backward, 1, sizeof(cl_mem), &l->output);
            clSetKernelArg(cnn->k_backward, 2, sizeof(cl_mem), &l->weights);
            clSetKernelArg(cnn->k_backward, 3, sizeof(cl_mem), &l->grad_input);
            clSetKernelArg(cnn->k_backward, 4, sizeof(int), &prev_cin4);
            clSetKernelArg(cnn->k_backward, 5, sizeof(int), &l->cout);
            clSetKernelArg(cnn->k_backward, 6, sizeof(int), &l->h);
            clSetKernelArg(cnn->k_backward, 7, sizeof(int), &l->w);
            clSetKernelArg(cnn->k_backward, 8, sizeof(int), &l->use_relu);
            
            size_t back_global[3] = {(size_t)l->w, (size_t)l->h, (size_t)prev_cin4};
            size_t back_local[3] = {16, 8, 1};
            clEnqueueNDRangeKernel(cnn->queue, cnn->k_backward, 3, NULL, back_global, back_local, 0, NULL, NULL);
            
            grad_current = l->grad_input;
        }
    }
    
    /* ========== UPDATE WEIGHTS ========== */
    if (cnn->config.optimizer == OPTIMIZER_SGD) {
        for (int i = 0; i < cnn->n_layers; i++) {
            ConvLayer *l = &cnn->layers[i];
            int w_vec_size = l->cout * l->cin4 * 9;
            float lr = cnn->config.learning_rate;
            
            clSetKernelArg(cnn->k_sgd_update, 0, sizeof(cl_mem), &l->weights);
            clSetKernelArg(cnn->k_sgd_update, 1, sizeof(cl_mem), &l->bias);
            clSetKernelArg(cnn->k_sgd_update, 2, sizeof(cl_mem), &l->grad_weights);
            clSetKernelArg(cnn->k_sgd_update, 3, sizeof(cl_mem), &l->grad_bias);
            clSetKernelArg(cnn->k_sgd_update, 4, sizeof(float), &lr);
            clSetKernelArg(cnn->k_sgd_update, 5, sizeof(int), &w_vec_size);
            clSetKernelArg(cnn->k_sgd_update, 6, sizeof(int), &l->cout);
            
            size_t update_global = w_vec_size > l->cout ? w_vec_size : l->cout;
            clEnqueueNDRangeKernel(cnn->queue, cnn->k_sgd_update, 1, NULL, &update_global, NULL, 0, NULL, NULL);
        }
    } else if (cnn->config.optimizer == OPTIMIZER_ADAM) {
        cnn->adam_t++;
        for (int i = 0; i < cnn->n_layers; i++) {
            ConvLayer *l = &cnn->layers[i];
            int w_vec_size = l->cout * l->cin4 * 9;
            float lr = cnn->config.learning_rate;
            float beta1 = cnn->config.adam_beta1;
            float beta2 = cnn->config.adam_beta2;
            float eps = cnn->config.adam_epsilon;
            int t = cnn->adam_t;
            
            clSetKernelArg(cnn->k_adam_update, 0, sizeof(cl_mem), &l->weights);
            clSetKernelArg(cnn->k_adam_update, 1, sizeof(cl_mem), &l->bias);
            clSetKernelArg(cnn->k_adam_update, 2, sizeof(cl_mem), &l->grad_weights);
            clSetKernelArg(cnn->k_adam_update, 3, sizeof(cl_mem), &l->grad_bias);
            clSetKernelArg(cnn->k_adam_update, 4, sizeof(cl_mem), &l->adam_m_w);
            clSetKernelArg(cnn->k_adam_update, 5, sizeof(cl_mem), &l->adam_m_b);
            clSetKernelArg(cnn->k_adam_update, 6, sizeof(cl_mem), &l->adam_v_w);
            clSetKernelArg(cnn->k_adam_update, 7, sizeof(cl_mem), &l->adam_v_b);
            clSetKernelArg(cnn->k_adam_update, 8, sizeof(float), &lr);
            clSetKernelArg(cnn->k_adam_update, 9, sizeof(float), &beta1);
            clSetKernelArg(cnn->k_adam_update, 10, sizeof(float), &beta2);
            clSetKernelArg(cnn->k_adam_update, 11, sizeof(float), &eps);
            clSetKernelArg(cnn->k_adam_update, 12, sizeof(int), &t);
            clSetKernelArg(cnn->k_adam_update, 13, sizeof(int), &w_vec_size);
            clSetKernelArg(cnn->k_adam_update, 14, sizeof(int), &l->cout);
            
            size_t update_global = w_vec_size > l->cout ? w_vec_size : l->cout;
            clEnqueueNDRangeKernel(cnn->queue, cnn->k_adam_update, 1, NULL, &update_global, NULL, 0, NULL, NULL);
        }
    }
    
    clFinish(cnn->queue);
    return total_loss;
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
        
        if (cnn->config.optimizer == OPTIMIZER_ADAM) {
            clReleaseMemObject(l->adam_m_w);
            clReleaseMemObject(l->adam_v_w);
            clReleaseMemObject(l->adam_m_b);
            clReleaseMemObject(l->adam_v_b);
        }
    }
    
    if (cnn->finalized) {
        clReleaseMemObject(cnn->input_buf);
        clReleaseMemObject(cnn->target_buf);
        clReleaseMemObject(cnn->grad_buf);
        clReleaseMemObject(cnn->residual_buf);
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

void learning_rate_decay_init(LearningRateDecay* lr_decay, 
                              float initial_lr, float decay_rate, int decay_steps) {
    lr_decay->initial_lr = initial_lr;
    lr_decay->decay_rate = decay_rate;
    lr_decay->decay_steps = decay_steps;
    lr_decay->step = 0;
}

float learning_rate_decay_get(LearningRateDecay* lr_decay, int current_step) {
    lr_decay->step = current_step;
    return lr_decay->initial_lr * powf(lr_decay->decay_rate, 
                                       (float)(lr_decay->step) / lr_decay->decay_steps);
}

void cnn_set_learning_rate(CNNDenoiser* cnn, float learning_rate) {
    cnn->config.learning_rate = learning_rate;
}

float cnn_get_learning_rate(CNNDenoiser* cnn) {
    return cnn->config.learning_rate;
}
