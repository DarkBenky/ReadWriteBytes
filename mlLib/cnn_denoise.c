/* cnn_denoise.c - Implementation of OpenCL CNN Denoising Library */

#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <dirent.h>
#include <curl/curl.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define MAX_LAYERS 16
#define CHECK_CL(err, msg)                                                                \
	if (err != CL_SUCCESS) {                                                              \
		fprintf(stderr, "OpenCL error %d at %s:%d - %s\n", err, __FILE__, __LINE__, msg); \
		return NULL;                                                                      \
	}

/* Internal layer representation */
typedef struct {
	LayerType type; /* Layer type */
	int cin, cout, h, w, cin4;
	cl_mem weights, bias, output, grad_bias, grad_weights, grad_input;
	cl_mem batch_output;						   /* Batch output buffer [max_batch][cout][h][w] */
	cl_mem batch_grad_input;					   /* Batch gradient input buffer [max_batch][cin][h][w] */
	cl_mem batch_residual_saved;				   /* Batch residual saved buffer [max_batch][channels][h][w] */
	cl_mem adam_m_w, adam_v_w, adam_m_b, adam_v_b; /* Adam optimizer buffers */
	float *h_weights, *h_bias, *h_grad_w, *h_grad_b;
	char name[64];
	int use_relu;
	int skip_from;		   /* Layer index to skip from, -1 = no skip */
	int residual_from;	   /* For RESIDUAL_SUBTRACT: layer index that saved the input, -1 = use network input */
	cl_mem skip_input;	   /* Input from skip connection */
	cl_mem grad_output;	   /* Gradient w.r.t this layer's output (for skip connections) */
	cl_mem residual_saved; /* For RESIDUAL_INPUT: saved input buffer */
} ConvLayer;

/* Main CNN structure */
struct CNNDenoiser {
	cl_context ctx;
	cl_command_queue queue;
	cl_program program;
	cl_kernel k_forward, k_backward, k_weight_grad, k_mae_loss, k_sgd_update, k_adam_update;
	cl_kernel k_mse_loss, k_laplace_loss, k_add_weighted_grad, k_residual_subtract, k_negate;
	cl_kernel k_copy_buffer; /* Copy buffer for residual input layer */
	cl_kernel k_color_variance_loss;
	cl_kernel k_forward_residual; /* Fused forward + residual for last layer */
	cl_kernel k_add_skip;		  /* Add skip connection */
	cl_kernel k_add_skip_grad;	  /* Backprop through skip connection */

	/* Batch training kernels */
	cl_kernel k_batch_forward;
	cl_kernel k_batch_backward;
	cl_kernel k_batch_weight_grad;
	cl_kernel k_batch_mae_loss;
	cl_kernel k_batch_mse_loss;
	cl_kernel k_batch_laplace_loss;
	cl_kernel k_batch_color_loss;
	cl_kernel k_batch_ssim_loss;
	cl_kernel k_batch_sobel_loss;
	cl_kernel k_batch_clear_loss;
	cl_kernel k_batch_add_weighted_grad;
	cl_kernel k_batch_loss_reduce;

	/* Batch residual layer kernels */
	cl_kernel k_batch_residual_input;
	cl_kernel k_batch_residual_subtract;
	cl_kernel k_batch_residual_input_backward;
	cl_kernel k_batch_residual_subtract_backward;

	int batch_kernels_available;

	size_t optimal_local[3];
	int tuning_done;

	CNNConfig config;
	int n_layers;
	ConvLayer layers[MAX_LAYERS];
	int adam_t; /* Adam timestep */

	cl_mem input_buf, target_buf, grad_buf, temp_grad, residual_buf;

	/* Batch buffers (allocated if max_batch_size > 1) */
	cl_mem batch_input_buf;	 /* [max_batch][channels][h][w] */
	cl_mem batch_target_buf; /* [max_batch][channels][h][w] */
	cl_mem batch_loss_buf;	 /* [max_batch] - per-sample loss */
	cl_mem batch_grad_buf;	 /* [max_batch][channels][h][w] */

	TimingStats stats;
	int stats_count;
	int finalized;

	/* Individual loss tracking */
	float last_mae_loss;
	float last_mse_loss;
	float last_laplace_loss;
	float last_color_loss;
	float last_ssim_loss;
	float last_sobel_loss;
};

/* Optimized OpenCL kernels - 4 outputs per thread */
static const char *kernel_source =
	"__kernel void conv3x3_forward_relu_f4(\n"
	"    __global const float* input, __global float* output,\n"
	"    __global const float4* weights, __global const float* bias,\n"
	"    int Cin4, int Cout, int H, int W)\n"
	"{\n"
	"    int x = get_global_id(0), y = get_global_id(1), oc = get_global_id(2) * 4;\n"
	"    if (x >= W || y >= H) return;\n"
	"    \n"
	"    int hw = H * W;\n"
	"    \n"
	"    float sum0 = (oc < Cout) ? bias[oc] : 0.0f;\n"
	"    float sum1 = (oc + 1 < Cout) ? bias[oc + 1] : 0.0f;\n"
	"    float sum2 = (oc + 2 < Cout) ? bias[oc + 2] : 0.0f;\n"
	"    float sum3 = (oc + 3 < Cout) ? bias[oc + 3] : 0.0f;\n"
	"    \n"
	"    for (int ic4 = 0; ic4 < Cin4; ic4++) {\n"
	"        /* Clamp coordinates for replicate padding */\n"
	"        int y0 = max(y - 1, 0), y1 = y, y2 = min(y + 1, H - 1);\n"
	"        int x0 = max(x - 1, 0), x1 = x, x2 = min(x + 1, W - 1);\n"
	"        \n"
	"        /* Read 4 channels at each position from planar layout */\n"
	"        #define READ_PIXEL(py, px) (float4)(input[(ic4*4+0)*hw + (py)*W + (px)], \\\n"
	"                                             input[(ic4*4+1)*hw + (py)*W + (px)], \\\n"
	"                                             input[(ic4*4+2)*hw + (py)*W + (px)], \\\n"
	"                                             input[(ic4*4+3)*hw + (py)*W + (px)])\n"
	"        \n"
	"        float4 i0 = READ_PIXEL(y0, x0);\n"
	"        float4 i1 = READ_PIXEL(y0, x1);\n"
	"        float4 i2 = READ_PIXEL(y0, x2);\n"
	"        float4 i3 = READ_PIXEL(y1, x0);\n"
	"        float4 i4 = READ_PIXEL(y1, x1);\n"
	"        float4 i5 = READ_PIXEL(y1, x2);\n"
	"        float4 i6 = READ_PIXEL(y2, x0);\n"
	"        float4 i7 = READ_PIXEL(y2, x1);\n"
	"        float4 i8 = READ_PIXEL(y2, x2);\n"
	"        #undef READ_PIXEL\n"
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
	"    __global const float4* weights, __global float* grad_in,\n"
	"    int Cin4, int Cout, int H, int W, int use_relu)\n"
	"{\n"
	"    int x = get_global_id(0), y = get_global_id(1), ic4 = get_global_id(2);\n"
	"    if (x >= W || y >= H) return;\n"
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
	"    /* Accumulate to planar layout (use += for skip connections) */\n"
	"    int pixel_idx = y * W + x;\n"
	"    grad_in[(ic4*4 + 0)*hw + pixel_idx] += acc.s0;\n"
	"    grad_in[(ic4*4 + 1)*hw + pixel_idx] += acc.s1;\n"
	"    grad_in[(ic4*4 + 2)*hw + pixel_idx] += acc.s2;\n"
	"    grad_in[(ic4*4 + 3)*hw + pixel_idx] += acc.s3;\n"
	"}\n"
	"\n"
	"__kernel void weight_grad_reduce(\n"
	"    __global const float* input, __global const float* grad_out,\n"
	"    __global const float* output, __global float4* grad_w_vec,\n"
	"    __global float* grad_b, int Cin4, int H, int W, int use_relu)\n"
	"{\n"
	"    int oc = get_global_id(0), ic4 = get_global_id(1), k = get_global_id(2);\n"
	"    int hw = H * W, dy = (k / 3) - 1, dx = (k % 3) - 1;\n"
	"    \n"
	"    float4 sum = (float4)(0.0f);\n"
	"    float bias_sum = 0.0f;\n"
	"    \n"
	"    for (int y = 0; y < H; y++) {\n"
	"        for (int x = 0; x < W; x++) {\n"
	"            int oidx = oc * hw + y * W + x;\n"
	"            float g = grad_out[oidx];\n"
	"            if (use_relu && output[oidx] <= 0.0f) g = 0.0f;\n"
	"            if (g != 0.0f) {\n"
	"                /* Clamp input coordinates for padding */\n"
	"                int iy = clamp(y + dy, 0, H - 1);\n"
	"                int ix = clamp(x + dx, 0, W - 1);\n"
	"                int pixel_idx = iy * W + ix;\n"
	"                /* Read 4 channels from planar layout */\n"
	"                float4 input_val = (float4)(input[(ic4*4+0)*hw + pixel_idx],\n"
	"                                             input[(ic4*4+1)*hw + pixel_idx],\n"
	"                                             input[(ic4*4+2)*hw + pixel_idx],\n"
	"                                             input[(ic4*4+3)*hw + pixel_idx]);\n"
	"                sum = fma(input_val, (float4)(g), sum);\n"
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
	"    int pixels_per_channel = size / 4;\n"
	"    \n"
	"    for (int idx = gid; idx < size; idx += get_global_size(0)) {\n"
	"        int channel = idx / pixels_per_channel;\n"
	"        if (channel < 3) {\n"
	"            float diff = prediction[idx] - target[idx];\n"
	"            grad_out[idx] = copysign(1.0f, diff);\n"
	"            local_sum += fabs(diff);\n"
	"        } else {\n"
	"            grad_out[idx] = 0.0f;\n"
	"        }\n"
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
	"    int pixels_per_channel = size / 4;\n"
	"    \n"
	"    if (gid < size) {\n"
	"        int channel = gid / pixels_per_channel;\n"
	"        if (channel < 3) {\n"
	"            float diff = output[gid] - target[gid];\n"
	"            grad[gid] = 2.0f * diff;\n"
	"            local_loss[lid] = diff * diff;\n"
	"        } else {\n"
	"            grad[gid] = 0.0f;\n"
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
	"__kernel void color_variance_loss(\n"
	"    __global const float* output, __global const float* target,\n"
	"    __global float* grad, __global float* loss_accum,\n"
	"    int H, int W, __local float* local_loss)\n"
	"{\n"
	"    int gid = get_global_id(0), lid = get_local_id(0);\n"
	"    local_loss[lid] = 0.0f;\n"
	"    int pixels = H * W;\n"
	"    \n"
	"    if (gid < pixels) {\n"
	"        float out_r = output[gid];\n"
	"        float out_g = output[pixels + gid];\n"
	"        float out_b = output[2 * pixels + gid];\n"
	"        float tgt_r = target[gid];\n"
	"        float tgt_g = target[pixels + gid];\n"
	"        float tgt_b = target[2 * pixels + gid];\n"
	"        \n"
	"        /* Direction loss: 1 - dot(normalize(pred), normalize(target)) */\n"
	"        float pred_norm = sqrt(out_r*out_r + out_g*out_g + out_b*out_b) + 1e-6f;\n"
	"        float tgt_norm = sqrt(tgt_r*tgt_r + tgt_g*tgt_g + tgt_b*tgt_b) + 1e-6f;\n"
	"        \n"
	"        float pred_r_n = out_r / pred_norm;\n"
	"        float pred_g_n = out_g / pred_norm;\n"
	"        float pred_b_n = out_b / pred_norm;\n"
	"        float tgt_r_n = tgt_r / tgt_norm;\n"
	"        float tgt_g_n = tgt_g / tgt_norm;\n"
	"        float tgt_b_n = tgt_b / tgt_norm;\n"
	"        \n"
	"        float dot = pred_r_n * tgt_r_n + pred_g_n * tgt_g_n + pred_b_n * tgt_b_n;\n"
	"        float direction_loss = 1.0f - dot;\n"
	"        \n"
	"        /* Saturation loss: max(0, target_std - pred_std) */\n"
	"        float pred_mean = (out_r + out_g + out_b) / 3.0f;\n"
	"        float pred_var = ((out_r - pred_mean)*(out_r - pred_mean) +\n"
	"                         (out_g - pred_mean)*(out_g - pred_mean) +\n"
	"                         (out_b - pred_mean)*(out_b - pred_mean)) / 3.0f;\n"
	"        float pred_std = sqrt(pred_var + 1e-8f);\n"
	"        \n"
	"        float tgt_mean = (tgt_r + tgt_g + tgt_b) / 3.0f;\n"
	"        float tgt_var = ((tgt_r - tgt_mean)*(tgt_r - tgt_mean) +\n"
	"                        (tgt_g - tgt_mean)*(tgt_g - tgt_mean) +\n"
	"                        (tgt_b - tgt_mean)*(tgt_b - tgt_mean)) / 3.0f;\n"
	"        float tgt_std = sqrt(tgt_var + 1e-8f);\n"
	"        \n"
	"        float sat_diff = fmax(0.0f, tgt_std - pred_std);\n"
	"        float saturation_loss = sat_diff;\n"
	"        \n"
	"        /* Simplified color loss - stable gradients:\n"
	"         * Direction: penalize wrong color direction\n"
	"         * Saturation: penalize desaturation with moderate power */\n"
	"        float dir_term = direction_loss + 1.0f;\n"
	"        float dir_penalty = dir_term * dir_term;  /* squared */\n"
	"        \n"
	"        float sat_term = saturation_loss + 1.0f;\n"
	"        float sat_penalty = sat_term * sat_term * sat_term * sat_term;  /* ^4 */\n"
	"        \n"
	"        float total_loss = 2.0f * dir_penalty + 8.0f * sat_penalty;\n"
	"        local_loss[lid] = total_loss;\n"
	"        \n"
	"        /* Gradient computation */\n"
	"        /* d(direction_loss)/d(out_r) - moderate scaling */\n"
	"        float inv_pred_norm = 1.0f / pred_norm;\n"
	"        \n"
	"        float dir_grad_scale = 2.0f * 2.0f * dir_term;  /* 2 * 2 * (dir + 1) */\n"
	"        float grad_dir_r = dir_grad_scale * (-(tgt_r_n * inv_pred_norm - pred_r_n * dot * inv_pred_norm));\n"
	"        float grad_dir_g = dir_grad_scale * (-(tgt_g_n * inv_pred_norm - pred_g_n * dot * inv_pred_norm));\n"
	"        float grad_dir_b = dir_grad_scale * (-(tgt_b_n * inv_pred_norm - pred_b_n * dot * inv_pred_norm));\n"
	"        \n"
	"        /* d(saturation_loss)/d(out) - stable scaling */\n"
	"        float grad_sat_r = 0.0f, grad_sat_g = 0.0f, grad_sat_b = 0.0f;\n"
	"        \n"
	"        if (sat_diff > 0.0f) {\n"
	"            float sat_weight = 8.0f * 4.0f * sat_term * sat_term * sat_term;  /* 8 * 4 * (sat+1)^3 */\n"
	"            float inv_pred_std = -1.0f / (pred_std + 1e-8f);\n"
	"            float factor = inv_pred_std / 3.0f;\n"
	"            grad_sat_r = sat_weight * factor * (out_r - pred_mean);\n"
	"            grad_sat_g = sat_weight * factor * (out_g - pred_mean);\n"
	"            grad_sat_b = sat_weight * factor * (out_b - pred_mean);\n"
	"        }\n"
	"        \n"
	"        grad[gid] = grad_dir_r + grad_sat_r;\n"
	"        grad[pixels + gid] = grad_dir_g + grad_sat_g;\n"
	"        grad[2 * pixels + gid] = grad_dir_b + grad_sat_b;\n"
	"        grad[3 * pixels + gid] = 0.0f;\n"
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
	"}\n"
	"\n"
	"__kernel void residual_subtract(\n"
	"    __global const float* input, __global const float* prediction,\n"
	"    __global float* output, int size)\n"
	"{\n"
	"    int gid = get_global_id(0);\n"
	"    if (gid < size) {\n"
	"        output[gid] = input[gid] - prediction[gid];\n"
	"    }\n"
	"}\n"
	"\n"
	"__kernel void copy_buffer(\n"
	"    __global const float* input, __global float* output, int size)\n"
	"{\n"
	"    int gid = get_global_id(0);\n"
	"    if (gid < size) {\n"
	"        output[gid] = input[gid];\n"
	"    }\n"
	"}\n"
	"\n"
	"__kernel void negate(\n"
	"    __global const float* input, __global float* output, int size)\n"
	"{\n"
	"    int gid = get_global_id(0);\n"
	"    if (gid < size) {\n"
	"        output[gid] = -input[gid];\n"
	"    }\n"
	"}\n"
	"\n"
	"/* Add skip connection: output = layer_output + skip_input */\n"
	"__kernel void add_skip(\n"
	"    __global float* output, __global const float* skip_input, int size)\n"
	"{\n"
	"    int gid = get_global_id(0);\n"
	"    if (gid < size) {\n"
	"        output[gid] += skip_input[gid];\n"
	"    }\n"
	"}\n"
	"\n"
	"/* Backprop through skip: accumulate gradient to skip source */\n"
	"__kernel void add_skip_grad(\n"
	"    __global float* skip_grad, __global const float* current_grad, int size)\n"
	"{\n"
	"    int gid = get_global_id(0);\n"
	"    if (gid < size) {\n"
	"        skip_grad[gid] += current_grad[gid];\n"
	"    }\n"
	"}\n"
	"\n"
	"/* Fused final layer with residual: directly compute input - prediction */\n"
	"__kernel void conv3x3_forward_relu_residual_f4(\n"
	"    __global const float4* input, __global const float4* original_input,\n"
	"    __global float* output, __global const float4* weights,\n"
	"    __global const float* bias, int Cin4, int Cout, int H, int W)\n"
	"{\n"
	"    int x = get_global_id(0), y = get_global_id(1), oc = get_global_id(2) * 4;\n"
	"    if (x <= 0 || y <= 0 || x >= W-1 || y >= H-1) return;\n"
	"    \n"
	"    int hw = H * W;\n"
	"    int pixel_idx = y * W + x;\n"
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
	"    /* Apply ReLU and compute residual: original_input - prediction */\n"
	"    int oc4 = oc / 4;\n"
	"    float4 orig = original_input[oc4 * hw + pixel_idx];\n"
	"    \n"
	"    if (oc < Cout) output[oc * hw + pixel_idx] = orig.x - fmax(sum0, 0.0f);\n"
	"    if (oc + 1 < Cout) output[(oc + 1) * hw + pixel_idx] = orig.y - fmax(sum1, 0.0f);\n"
	"    if (oc + 2 < Cout) output[(oc + 2) * hw + pixel_idx] = orig.z - fmax(sum2, 0.0f);\n"
	"    if (oc + 3 < Cout) output[(oc + 3) * hw + pixel_idx] = orig.w - fmax(sum3, 0.0f);\n"
	"}\n";

static void cnn_auto_tune_workgroup(CNNDenoiser *cnn, int H, int W, int Cout) {
	if (cnn->tuning_done) return;

	printf("Auto-tuning work group sizes...\n");

	size_t test_configs[][3] = {
		{4, 4, 1}, {8, 8, 1}, {16, 16, 1}, {32, 32, 1}, {8, 4, 1}, {4, 8, 1}, {16, 8, 1}, {8, 16, 1}, {16, 4, 1}, {4, 16, 1}, {32, 16, 1}, {16, 32, 1}};
	int num_configs = sizeof(test_configs) / sizeof(test_configs[0]);

	double best_time = 1e9;
	int best_idx = 1;

	size_t global[3] = {W, H, (Cout + 3) / 4};
	int warmup_runs = 3;
	int bench_runs = 20;

	cl_device_id device;
	clGetCommandQueueInfo(cnn->queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);

	size_t max_wg_size;
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);

	for (int i = 0; i < num_configs; i++) {
		size_t *local = test_configs[i];

		if (local[0] * local[1] * local[2] > max_wg_size) continue;

		for (int w = 0; w < warmup_runs; w++) {
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL,
								   global, local, 0, NULL, NULL);
		}
		clFinish(cnn->queue);

		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);
		for (int r = 0; r < bench_runs; r++) {
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL,
								   global, local, 0, NULL, NULL);
		}
		clFinish(cnn->queue);
		clock_gettime(CLOCK_MONOTONIC, &end);

		double elapsed = (end.tv_sec - start.tv_sec) +
						 (end.tv_nsec - start.tv_nsec) * 1e-9;
		double avg_time = elapsed / bench_runs;

		printf("  [%zux%zux%zu]: %.1f μs\n", local[0], local[1], local[2], avg_time * 1e6);

		if (avg_time < best_time) {
			best_time = avg_time;
			best_idx = i;
		}
	}

	cnn->optimal_local[0] = test_configs[best_idx][0];
	cnn->optimal_local[1] = test_configs[best_idx][1];
	cnn->optimal_local[2] = test_configs[best_idx][2];
	cnn->tuning_done = 1;

	printf("Optimal: [%zux%zux%zu] (%.1f μs)\n",
		   cnn->optimal_local[0], cnn->optimal_local[1], cnn->optimal_local[2],
		   best_time * 1e6);
}

static void init_weights(float *w, int n) {
	float scale = sqrtf(2.0f / n);
	for (int i = 0; i < n; i++) {
		w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
	}
}

static void init_weights_linear(float *w, int n_in, int n_out) {
	/* Xavier/Glorot initialization for linear layers */
	float scale = sqrtf(6.0f / (n_in + n_out));
	for (int i = 0; i < n_in * n_out * 9; i++) {
		w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
	}
}

CNNDenoiser *cnn_create(CNNConfig config) {
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
	cnn->k_color_variance_loss = clCreateKernel(cnn->program, "color_variance_loss", &err);
	cnn->k_sgd_update = clCreateKernel(cnn->program, "sgd_update", &err);
	cnn->k_adam_update = clCreateKernel(cnn->program, "adam_update", &err);
	cnn->k_add_weighted_grad = clCreateKernel(cnn->program, "add_weighted_grad", &err);
	cnn->k_residual_subtract = clCreateKernel(cnn->program, "residual_subtract", &err);
	cnn->k_negate = clCreateKernel(cnn->program, "negate", &err);
	cnn->k_copy_buffer = clCreateKernel(cnn->program, "copy_buffer", &err);
	cnn->k_forward_residual = clCreateKernel(cnn->program, "conv3x3_forward_relu_residual_f4", &err);
	cnn->k_add_skip = clCreateKernel(cnn->program, "add_skip", &err);
	cnn->k_add_skip_grad = clCreateKernel(cnn->program, "add_skip_grad", &err);

	/* Load batch training kernels if max_batch_size > 1 */
	cnn->batch_kernels_available = 0;
	if (config.max_batch_size > 1) {
		FILE *batch_kernel_file = fopen("mlLib/batch_kernels.cl", "r");
		if (!batch_kernel_file) {
			batch_kernel_file = fopen("batch_kernels.cl", "r");
		}

		if (batch_kernel_file) {
			fseek(batch_kernel_file, 0, SEEK_END);
			size_t batch_src_size = ftell(batch_kernel_file);
			fseek(batch_kernel_file, 0, SEEK_SET);
			char *batch_src = malloc(batch_src_size + 1);
			fread(batch_src, 1, batch_src_size, batch_kernel_file);
			batch_src[batch_src_size] = '\0';
			fclose(batch_kernel_file);

			cl_program batch_program = clCreateProgramWithSource(cnn->ctx, 1,
																 (const char **)&batch_src, NULL, &err);
			free(batch_src);

			err = clBuildProgram(batch_program, 0, NULL, opts, NULL, NULL);
			if (err == CL_SUCCESS) {
				cnn->k_batch_forward = clCreateKernel(batch_program, "batch_conv3x3_forward_relu_f4", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_forward kernel: %d\n", err);

				cnn->k_batch_backward = clCreateKernel(batch_program, "batch_conv3x3_backward_input_f4", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_backward kernel: %d\n", err);

				cnn->k_batch_weight_grad = clCreateKernel(batch_program, "batch_weight_grad_reduce", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_weight_grad kernel: %d\n", err);

				cnn->k_batch_mae_loss = clCreateKernel(batch_program, "batch_mae_loss_gradient", &err);
				cnn->k_batch_mse_loss = clCreateKernel(batch_program, "batch_mse_loss_gradient", &err);
				cnn->k_batch_laplace_loss = clCreateKernel(batch_program, "batch_laplace_loss_gradient", &err);
				cnn->k_batch_color_loss = clCreateKernel(batch_program, "batch_color_variance_loss", &err);
				cnn->k_batch_ssim_loss = clCreateKernel(batch_program, "batch_ssim_loss_gradient", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_ssim_loss kernel: %d\n", err);
				cnn->k_batch_sobel_loss = clCreateKernel(batch_program, "batch_sobel_loss_gradient", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_sobel_loss kernel: %d\n", err);
				cnn->k_batch_clear_loss = clCreateKernel(batch_program, "batch_clear_loss_buffer", &err);
				cnn->k_batch_add_weighted_grad = clCreateKernel(batch_program, "batch_add_weighted_gradient", &err);
				cnn->k_batch_loss_reduce = clCreateKernel(batch_program, "batch_loss_reduce", &err);

				/* Residual layer kernels */
				cnn->k_batch_residual_input = clCreateKernel(batch_program, "batch_residual_input", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_residual_input kernel: %d\n", err);

				cnn->k_batch_residual_subtract = clCreateKernel(batch_program, "batch_residual_subtract", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_residual_subtract kernel: %d\n", err);

				cnn->k_batch_residual_input_backward = clCreateKernel(batch_program, "batch_residual_input_backward", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_residual_input_backward kernel: %d\n", err);

				cnn->k_batch_residual_subtract_backward = clCreateKernel(batch_program, "batch_residual_subtract_backward", &err);
				if (err != CL_SUCCESS) fprintf(stderr, "[WARN] Failed to create batch_residual_subtract_backward kernel: %d\n", err);

				cnn->batch_kernels_available = 1;
				printf("Batch training enabled (max_batch_size=%d)\n", config.max_batch_size);
			} else {
				size_t log_size;
				clGetProgramBuildInfo(batch_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
				char *log = malloc(log_size);
				clGetProgramBuildInfo(batch_program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
				fprintf(stderr, "Batch kernel build failed:\n%s\n", log);
				free(log);
				fprintf(stderr, "Batch training disabled, using single-image mode only\n");
			}
		} else {
			fprintf(stderr, "Warning: batch_kernels.cl not found, batch training disabled\n");
		}
	}

	cnn->adam_t = 0;
	cnn->tuning_done = 0;
	cnn->optimal_local[0] = 16;
	cnn->optimal_local[1] = 8;
	cnn->optimal_local[2] = 1;

	cnn->last_mae_loss = 0.0f;
	cnn->last_mse_loss = 0.0f;
	cnn->last_laplace_loss = 0.0f;
	cnn->last_color_loss = 0.0f;
	cnn->last_ssim_loss = 0.0f;
	cnn->last_sobel_loss = 0.0f;

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
	cfg.auto_tune_workgroup = 1;
	cfg.optimizer = OPTIMIZER_SGD;
	cfg.loss_config.num_losses = 1;
	cfg.loss_config.types[0] = LOSS_MAE;
	cfg.loss_config.weights[0] = 1.0f;
	cfg.adam_beta1 = 0.9f;
	cfg.adam_beta2 = 0.999f;
	cfg.adam_epsilon = 1e-8f;
	cfg.max_batch_size = 1; /* Default to single-image mode */
	return cfg;
}

int cnn_add_layer(CNNDenoiser *cnn, LayerConfig layer) {
	if (cnn->finalized) return -1;
	if (cnn->n_layers >= MAX_LAYERS) return -1;

	ConvLayer *l = &cnn->layers[cnn->n_layers++];
	l->type = layer.type;
	l->cin = layer.cin;
	l->cout = layer.cout;
	l->use_relu = layer.use_relu;
	l->skip_from = layer.skip_from;
	l->residual_from = layer.residual_from;
	l->h = cnn->config.input_height;
	l->w = cnn->config.input_width;
	l->cin4 = layer.cin / 4;
	strncpy(l->name, layer.name, 63);

	cl_int err;
	int out_size = layer.cout * l->h * l->w;

	/* Allocate output buffer for all layer types */
	l->output = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, &err);
	l->grad_output = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, &err);

	if (layer.type == LAYER_CONV) {
		/* Standard convolution layer - allocate weights and biases */
		int w_size = layer.cout * l->cin4 * 9;

		posix_memalign((void **)&l->h_weights, 64, w_size * 16);
		posix_memalign((void **)&l->h_bias, 64, layer.cout * 4);
		posix_memalign((void **)&l->h_grad_w, 64, w_size * 16);
		posix_memalign((void **)&l->h_grad_b, 64, layer.cout * 4);

		memset(l->h_bias, 0, layer.cout * 4);
		init_weights(l->h_weights, w_size * 4);

		l->weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
									w_size * 16, l->h_weights, &err);
		l->bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
								 layer.cout * 4, l->h_bias, &err);
		l->grad_bias = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cout * 4, NULL, &err);
		l->grad_weights = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, w_size * 16, NULL, &err);
		l->grad_input = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cin * l->h * l->w * 4, NULL, &err);

		/* Allocate skip_input buffer if skip connection exists */
		if (l->skip_from >= 0) {
			l->skip_input = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, &err);
		} else {
			l->skip_input = NULL;
		}

	} else if (layer.type == LAYER_RESIDUAL_INPUT) {
		/* Residual input layer - allocate buffer to save input */
		l->residual_saved = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, out_size * 4, NULL, &err);
		l->grad_input = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cin * l->h * l->w * 4, NULL, &err);

		/* No weights/biases needed */
		l->weights = NULL;
		l->bias = NULL;
		l->grad_weights = NULL;
		l->grad_bias = NULL;
		l->skip_input = NULL;
		l->h_weights = NULL;
		l->h_bias = NULL;
		l->h_grad_w = NULL;
		l->h_grad_b = NULL;

	} else if (layer.type == LAYER_RESIDUAL_SUBTRACT) {
		/* Residual subtract layer - no special buffers needed */
		l->grad_input = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer.cin * l->h * l->w * 4, NULL, &err);

		/* No weights/biases needed */
		l->weights = NULL;
		l->bias = NULL;
		l->grad_weights = NULL;
		l->grad_bias = NULL;
		l->skip_input = NULL;
		l->residual_saved = NULL;
		l->h_weights = NULL;
		l->h_bias = NULL;
		l->h_grad_w = NULL;
		l->h_grad_b = NULL;
	}

	return 0;
}

int cnn_finalize(CNNDenoiser *cnn) {
	int max_size = cnn->config.input_height * cnn->config.input_width *
				   (cnn->config.input_channels > cnn->config.output_channels ? cnn->config.input_channels : cnn->config.output_channels);

	cnn->input_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
	cnn->target_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
	cnn->grad_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);
	cnn->residual_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, max_size * 4, NULL, NULL);

	/* Allocate batch buffers if batch training enabled */
	if (cnn->config.max_batch_size > 1 && cnn->batch_kernels_available) {
		int img_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;
		size_t batch_size_bytes = cnn->config.max_batch_size * img_size * sizeof(float);

		cl_int err;
		cnn->batch_input_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, batch_size_bytes, NULL, &err);
		cnn->batch_target_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, batch_size_bytes, NULL, &err);
		cnn->batch_grad_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, batch_size_bytes, NULL, &err);
		/* Loss buffer needs to store per-pixel loss values for reduction */
		cnn->batch_loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, batch_size_bytes, NULL, &err);

		if (err != CL_SUCCESS) {
			fprintf(stderr, "Failed to allocate batch buffers (batch_size=%d, %.1f MB each)\n",
					cnn->config.max_batch_size, batch_size_bytes / (1024.0 * 1024.0));
			cnn->batch_kernels_available = 0;
		} else {
			printf("Batch buffers allocated: %d images × %.1f MB = %.1f MB total\n",
				   cnn->config.max_batch_size,
				   (img_size * sizeof(float)) / (1024.0 * 1024.0),
				   (batch_size_bytes * 4) / (1024.0 * 1024.0)); /* 4 = input+target+grad+loss */
		}
	}

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

			/* Only allocate Adam buffers for CONV layers */
			if (l->type != LAYER_CONV) {
				l->adam_m_w = NULL;
				l->adam_v_w = NULL;
				l->adam_m_b = NULL;
				l->adam_v_b = NULL;
				continue;
			}

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

	/* Allocate batch output buffers for each layer if batch training enabled */
	if (cnn->config.max_batch_size > 1 && cnn->batch_kernels_available) {
		for (int i = 0; i < cnn->n_layers; i++) {
			ConvLayer *l = &cnn->layers[i];
			size_t layer_output_size = cnn->config.max_batch_size * l->cout * l->h * l->w * sizeof(float);
			l->batch_output = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer_output_size, NULL, NULL);

			/* Also allocate batch-sized gradient buffer for backprop */
			size_t layer_input_size = cnn->config.max_batch_size * l->cin * l->h * l->w * sizeof(float);
			l->batch_grad_input = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer_input_size, NULL, NULL);

			/* Allocate batch residual_saved buffer for RESIDUAL_INPUT layers */
			if (l->type == LAYER_RESIDUAL_INPUT) {
				l->batch_residual_saved = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, layer_output_size, NULL, NULL);
			}
		}
	}

	/* Reinitialize output layer with Xavier init if linear and using residual mode */
	if (cnn->config.residual_mode && cnn->n_layers > 0) {
		ConvLayer *output_layer = &cnn->layers[cnn->n_layers - 1];
		if (!output_layer->use_relu) {
			int w_size = output_layer->cout * output_layer->cin4 * 9;
			init_weights_linear(output_layer->h_weights, output_layer->cin, output_layer->cout);
			clEnqueueWriteBuffer(cnn->queue, output_layer->weights, CL_TRUE, 0,
								 w_size * 16, output_layer->h_weights, 0, NULL, NULL);
		}
	}

	cnn->finalized = 1;
	return 0;
}

/* Batch training implementation - efficient GPU processing of multiple images */
static float cnn_train_step_batch(CNNDenoiser *cnn, float *noisy_input, float *clean_target, int batch_size) {
	struct timespec t_start, t_end, t_forward_start, t_loss_start, t_backward_start, t_update_start;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

	int img_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;
	int hw = cnn->config.input_height * cnn->config.input_width;

	/* Upload batch data */
	clEnqueueWriteBuffer(cnn->queue, cnn->batch_input_buf, CL_FALSE, 0,
						 batch_size * img_size * sizeof(float), noisy_input, 0, NULL, NULL);
	clEnqueueWriteBuffer(cnn->queue, cnn->batch_target_buf, CL_FALSE, 0,
						 batch_size * img_size * sizeof(float), clean_target, 0, NULL, NULL);

	/* Clear loss buffer */
	clSetKernelArg(cnn->k_batch_clear_loss, 0, sizeof(cl_mem), &cnn->batch_loss_buf);
	clSetKernelArg(cnn->k_batch_clear_loss, 1, sizeof(int), &batch_size);
	size_t clear_global = (batch_size + 255) / 256 * 256;
	clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_clear_loss, 1, NULL, &clear_global, NULL, 0, NULL, NULL);

	clock_gettime(CLOCK_MONOTONIC, &t_forward_start);

	/* Batch forward pass - process all images in parallel */
	cl_mem current = cnn->batch_input_buf;
	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];
		if (l->type == LAYER_RESIDUAL_INPUT) {
			/* Residual input layer - save and pass through */
			clSetKernelArg(cnn->k_batch_residual_input, 0, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_batch_residual_input, 1, sizeof(cl_mem), &l->batch_output);
			clSetKernelArg(cnn->k_batch_residual_input, 2, sizeof(cl_mem), &l->batch_residual_saved);
			clSetKernelArg(cnn->k_batch_residual_input, 3, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_residual_input, 4, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_batch_residual_input, 5, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_batch_residual_input, 6, sizeof(int), &l->w);

			size_t global[3] = {l->w, l->h, batch_size * l->cout};
			cl_int err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_residual_input, 3, NULL, global, NULL, 0, NULL, NULL);
			if (err != CL_SUCCESS) fprintf(stderr, "[ERROR] Batch residual_input layer %d failed: %d\n", i, err);
		} else if (l->type == LAYER_RESIDUAL_SUBTRACT) {
			/* Residual subtract layer - compute (saved - current) */
			cl_mem saved_input;
			if (l->residual_from >= 0 && l->residual_from < i) {
				if (cnn->layers[l->residual_from].type != LAYER_RESIDUAL_INPUT) {
					fprintf(stderr, "Error: Batch layer %d (%s) references layer %d for residual, but that layer is not RESIDUAL_INPUT\n",
							i, l->name, l->residual_from);
					saved_input = cnn->batch_input_buf;
				} else {
					saved_input = cnn->layers[l->residual_from].batch_residual_saved;
				}
			} else {
				saved_input = cnn->batch_input_buf;
			}

			clSetKernelArg(cnn->k_batch_residual_subtract, 0, sizeof(cl_mem), &saved_input);
			clSetKernelArg(cnn->k_batch_residual_subtract, 1, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_batch_residual_subtract, 2, sizeof(cl_mem), &l->batch_output);
			clSetKernelArg(cnn->k_batch_residual_subtract, 3, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_residual_subtract, 4, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_batch_residual_subtract, 5, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_batch_residual_subtract, 6, sizeof(int), &l->w);

			size_t global[3] = {l->w, l->h, batch_size * l->cout};
			cl_int err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_residual_subtract, 3, NULL, global, NULL, 0, NULL, NULL);
			if (err != CL_SUCCESS) fprintf(stderr, "[ERROR] Batch residual_subtract layer %d failed: %d\n", i, err);
		} else {
			/* Regular convolution layer */
			cl_int arg_err;
			arg_err = clSetKernelArg(cnn->k_batch_forward, 0, sizeof(cl_mem), &current);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 0: %d\n", i, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 1, sizeof(cl_mem), &l->batch_output);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 1: %d\n", i, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 2, sizeof(cl_mem), &l->weights);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 2: %d\n", i, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 3, sizeof(cl_mem), &l->bias);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 3: %d\n", i, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 4, sizeof(int), &batch_size);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 4 (batch_size=%d): %d\n", i, batch_size, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 5, sizeof(int), &l->cin4);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 5 (cin4=%d): %d\n", i, l->cin4, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 6, sizeof(int), &l->cout);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 6 (cout=%d): %d\n", i, l->cout, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 7, sizeof(int), &l->h);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 7 (h=%d): %d\n", i, l->h, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 8, sizeof(int), &l->w);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 8 (w=%d): %d\n", i, l->w, arg_err);

			arg_err = clSetKernelArg(cnn->k_batch_forward, 9, sizeof(int), &l->use_relu);
			if (arg_err != CL_SUCCESS) fprintf(stderr, "[ERROR] Layer %d arg 9 (use_relu=%d): %d\n", i, l->use_relu, arg_err);

			/* OpenCL 1.2 only supports 3D work sizes, so flatten batch into 3rd dimension */
			size_t global[3] = {l->w, l->h, (l->cout + 3) / 4 * batch_size};
			cl_int err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_forward, 3, NULL, global, NULL, 0, NULL, NULL);
			if (err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR] Batch forward layer %d failed: %d\n", i, err);
			}
		}

		current = l->batch_output;
	}
	clFinish(cnn->queue); /* Force sync to catch errors */

	/* Batch loss computation - support multiple weighted loss functions */
	ConvLayer *last = &cnn->layers[cnn->n_layers - 1];
	int size_per_img = last->cout * hw;
	int H = cnn->config.input_height;
	int W = cnn->config.input_width;
	int C = last->cout;

	/* Clear batch loss and gradient buffers */
	float zero = 0.0f;
	clEnqueueFillBuffer(cnn->queue, cnn->batch_loss_buf, &zero, sizeof(float),
						0, batch_size * sizeof(float), 0, NULL, NULL);
	clEnqueueFillBuffer(cnn->queue, cnn->batch_grad_buf, &zero, sizeof(float),
						0, batch_size * size_per_img * sizeof(float), 0, NULL, NULL);

	clock_gettime(CLOCK_MONOTONIC, &t_loss_start);

	/* DEBUG counter for both loss sections */
	static int debug_count = 0;

	/* Temporary gradient buffer for each loss component */
	cl_mem temp_grad_loss = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE,
										   batch_size * size_per_img * sizeof(float), NULL, NULL);

	/* Per-batch loss output buffer for reduction */
	cl_mem batch_loss_output = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE,
											  batch_size * sizeof(float), NULL, NULL);

	/* Pre-allocate host buffer for loss readback (reuse across all loss types) */
	float *batch_losses = malloc(batch_size * sizeof(float));

	float total_loss = 0.0f;

	/* Iterate through configured loss functions */
	for (int loss_idx = 0; loss_idx < cnn->config.loss_config.num_losses; loss_idx++) {
		LossType loss_type = cnn->config.loss_config.types[loss_idx];
		float weight = cnn->config.loss_config.weights[loss_idx];

		/* Clear temporary gradient buffer */
		clEnqueueFillBuffer(cnn->queue, temp_grad_loss, &zero, sizeof(float),
							0, batch_size * size_per_img * sizeof(float), 0, NULL, NULL);

		/* Clear per-pixel loss buffer */
		clEnqueueFillBuffer(cnn->queue, cnn->batch_loss_buf, &zero, sizeof(float),
							0, batch_size * size_per_img * sizeof(float), 0, NULL, NULL);

		/* Clear per-batch output buffer */
		clEnqueueFillBuffer(cnn->queue, batch_loss_output, &zero, sizeof(float),
							0, batch_size * sizeof(float), 0, NULL, NULL);

		if (loss_type == LOSS_MAE) {
			clSetKernelArg(cnn->k_batch_mae_loss, 0, sizeof(cl_mem), &last->batch_output);
			clSetKernelArg(cnn->k_batch_mae_loss, 1, sizeof(cl_mem), &cnn->batch_target_buf);
			clSetKernelArg(cnn->k_batch_mae_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_mae_loss, 3, sizeof(cl_mem), &cnn->batch_loss_buf);
			clSetKernelArg(cnn->k_batch_mae_loss, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_mae_loss, 5, sizeof(int), &size_per_img);

			size_t mae_global[2] = {batch_size, size_per_img};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_mae_loss, 2, NULL, mae_global, NULL, 0, NULL, NULL);

		} else if (loss_type == LOSS_MSE) {
			clSetKernelArg(cnn->k_batch_mse_loss, 0, sizeof(cl_mem), &last->batch_output);
			clSetKernelArg(cnn->k_batch_mse_loss, 1, sizeof(cl_mem), &cnn->batch_target_buf);
			clSetKernelArg(cnn->k_batch_mse_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_mse_loss, 3, sizeof(cl_mem), &cnn->batch_loss_buf);
			clSetKernelArg(cnn->k_batch_mse_loss, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_mse_loss, 5, sizeof(int), &size_per_img);

			size_t mse_global[2] = {batch_size, size_per_img};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_mse_loss, 2, NULL, mse_global, NULL, 0, NULL, NULL);

		} else if (loss_type == LOSS_LAPLACE) {
			clSetKernelArg(cnn->k_batch_laplace_loss, 0, sizeof(cl_mem), &last->batch_output);
			clSetKernelArg(cnn->k_batch_laplace_loss, 1, sizeof(cl_mem), &cnn->batch_target_buf);
			clSetKernelArg(cnn->k_batch_laplace_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_laplace_loss, 3, sizeof(cl_mem), &cnn->batch_loss_buf);
			clSetKernelArg(cnn->k_batch_laplace_loss, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_laplace_loss, 5, sizeof(int), &H);
			clSetKernelArg(cnn->k_batch_laplace_loss, 6, sizeof(int), &W);
			clSetKernelArg(cnn->k_batch_laplace_loss, 7, sizeof(int), &C);

			size_t laplace_global[3] = {W, H, batch_size * C};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_laplace_loss, 3, NULL, laplace_global, NULL, 0, NULL, NULL);

		} else if (loss_type == LOSS_COLOR_VARIANCE) {
			clSetKernelArg(cnn->k_batch_color_loss, 0, sizeof(cl_mem), &last->batch_output);
			clSetKernelArg(cnn->k_batch_color_loss, 1, sizeof(cl_mem), &cnn->batch_target_buf);
			clSetKernelArg(cnn->k_batch_color_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_color_loss, 3, sizeof(cl_mem), &cnn->batch_loss_buf);
			clSetKernelArg(cnn->k_batch_color_loss, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_color_loss, 5, sizeof(int), &H);
			clSetKernelArg(cnn->k_batch_color_loss, 6, sizeof(int), &W);

			size_t color_global[3] = {W, H, batch_size};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_color_loss, 3, NULL, color_global, NULL, 0, NULL, NULL);

		} else if (loss_type == LOSS_SSIM) {
			clSetKernelArg(cnn->k_batch_ssim_loss, 0, sizeof(cl_mem), &last->batch_output);
			clSetKernelArg(cnn->k_batch_ssim_loss, 1, sizeof(cl_mem), &cnn->batch_target_buf);
			clSetKernelArg(cnn->k_batch_ssim_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_ssim_loss, 3, sizeof(cl_mem), &cnn->batch_loss_buf);
			clSetKernelArg(cnn->k_batch_ssim_loss, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_ssim_loss, 5, sizeof(int), &H);
			clSetKernelArg(cnn->k_batch_ssim_loss, 6, sizeof(int), &W);

			size_t ssim_global[3] = {W, H, batch_size};
			cl_int ssim_err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_ssim_loss, 3, NULL, ssim_global, NULL, 0, NULL, NULL);
			if (ssim_err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR] SSIM kernel execution failed: %d\n", ssim_err);
			}

		} else if (loss_type == LOSS_SOBEL) {
			if (!cnn->k_batch_sobel_loss) {
				fprintf(stderr, "[ERROR] Sobel kernel is NULL!\n");
				continue;
			}
			clSetKernelArg(cnn->k_batch_sobel_loss, 0, sizeof(cl_mem), &last->batch_output);
			clSetKernelArg(cnn->k_batch_sobel_loss, 1, sizeof(cl_mem), &cnn->batch_target_buf);
			clSetKernelArg(cnn->k_batch_sobel_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_sobel_loss, 3, sizeof(cl_mem), &cnn->batch_loss_buf);
			clSetKernelArg(cnn->k_batch_sobel_loss, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_sobel_loss, 5, sizeof(int), &H);
			clSetKernelArg(cnn->k_batch_sobel_loss, 6, sizeof(int), &W);
			clSetKernelArg(cnn->k_batch_sobel_loss, 7, sizeof(int), &C);

			size_t sobel_global[3] = {W, H, batch_size * C};
			cl_int sobel_err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_sobel_loss, 3, NULL, sobel_global, NULL, 0, NULL, NULL);
			if (sobel_err != CL_SUCCESS) {
				fprintf(stderr, "[ERROR] Sobel kernel execution failed: %d\n", sobel_err);
			}
		}

		/* Reduce per-pixel losses to per-batch totals */
		clSetKernelArg(cnn->k_batch_loss_reduce, 0, sizeof(cl_mem), &cnn->batch_loss_buf);
		clSetKernelArg(cnn->k_batch_loss_reduce, 1, sizeof(cl_mem), &batch_loss_output);
		clSetKernelArg(cnn->k_batch_loss_reduce, 2, sizeof(int), &batch_size);
		clSetKernelArg(cnn->k_batch_loss_reduce, 3, sizeof(int), &size_per_img);
		clSetKernelArg(cnn->k_batch_loss_reduce, 4, 256 * sizeof(float), NULL); /* Local memory */

		size_t reduce_global = batch_size * 256; /* 256 work-items per batch */
		size_t reduce_local = 256;
		clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_loss_reduce, 1, NULL,
							   &reduce_global, &reduce_local, 0, NULL, NULL);

		/* Read back loss for this component - NON-BLOCKING read for performance */
		clEnqueueReadBuffer(cnn->queue, batch_loss_output, CL_FALSE, 0,
							batch_size * sizeof(float), batch_losses, 0, NULL, NULL);
		clEnqueueReadBuffer(cnn->queue, batch_loss_output, CL_FALSE, 0,
							batch_size * sizeof(float), batch_losses, 0, NULL, NULL);

		/* Add weighted gradient to accumulated gradient buffer (can run async) */
		clSetKernelArg(cnn->k_batch_add_weighted_grad, 0, sizeof(cl_mem), &cnn->batch_grad_buf);
		clSetKernelArg(cnn->k_batch_add_weighted_grad, 1, sizeof(cl_mem), &temp_grad_loss);
		clSetKernelArg(cnn->k_batch_add_weighted_grad, 2, sizeof(float), &weight);
		int total_size = batch_size * size_per_img;
		clSetKernelArg(cnn->k_batch_add_weighted_grad, 3, sizeof(int), &total_size);

		size_t add_global = ((total_size + 255) / 256) * 256;
		clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_add_weighted_grad, 1, NULL, &add_global, NULL, 0, NULL, NULL);

		/* Now sync to get loss value for accumulation */
		clFinish(cnn->queue);

		float component_loss = 0.0f;
		for (int i = 0; i < batch_size; i++) {
			component_loss += batch_losses[i];
		}

		/* Normalize loss based on type */
		float normalized_loss;
		if (loss_type == LOSS_COLOR_VARIANCE) {
			normalized_loss = component_loss / (batch_size * H * W);
		} else if (loss_type == LOSS_LAPLACE) {
			int rgb_pixels = (size_per_img / 4) * 3;
			normalized_loss = component_loss / (batch_size * rgb_pixels);
		} else {
			int rgb_pixels = (size_per_img / 4) * 3;
			normalized_loss = component_loss / (batch_size * rgb_pixels);
		}

		/* Store individual loss values */
		if (loss_type == LOSS_MAE) {
			cnn->last_mae_loss = normalized_loss;
		} else if (loss_type == LOSS_MSE) {
			cnn->last_mse_loss = normalized_loss;
		} else if (loss_type == LOSS_LAPLACE) {
			cnn->last_laplace_loss = normalized_loss;
		} else if (loss_type == LOSS_COLOR_VARIANCE) {
			cnn->last_color_loss = normalized_loss;
		} else if (loss_type == LOSS_SSIM) {
			cnn->last_ssim_loss = normalized_loss;
		} else if (loss_type == LOSS_SOBEL) {
			cnn->last_sobel_loss = normalized_loss;
		}

		total_loss += weight * normalized_loss;
	}

	free(batch_losses);
	clReleaseMemObject(temp_grad_loss);
	clReleaseMemObject(batch_loss_output);
	float avg_loss = total_loss / batch_size;

	clock_gettime(CLOCK_MONOTONIC, &t_backward_start);

	/* Batch backward pass */
	for (int i = cnn->n_layers - 1; i >= 0; i--) {
		ConvLayer *l = &cnn->layers[i];
		cl_mem layer_input = (i == 0) ? cnn->batch_input_buf : cnn->layers[i - 1].batch_output;

		/* Clear input gradient buffer */
		int in_size = l->cin * hw;
		float zero = 0.0f;
		clEnqueueFillBuffer(cnn->queue, l->batch_grad_input, &zero, sizeof(float),
							0, batch_size * in_size * sizeof(float), 0, NULL, NULL);

		/* For last layer, use batch_grad_buf (from loss computation)
		 * For other layers, use the next layer's batch_grad_input */
		cl_mem grad_source = (i == cnn->n_layers - 1) ? cnn->batch_grad_buf : cnn->layers[i + 1].batch_grad_input;

		if (l->type == LAYER_RESIDUAL_INPUT) {
			/* Residual input backward - pass gradient through */
			clSetKernelArg(cnn->k_batch_residual_input_backward, 0, sizeof(cl_mem), &grad_source);
			clSetKernelArg(cnn->k_batch_residual_input_backward, 1, sizeof(cl_mem), &l->batch_grad_input);
			clSetKernelArg(cnn->k_batch_residual_input_backward, 2, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_residual_input_backward, 3, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_batch_residual_input_backward, 4, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_batch_residual_input_backward, 5, sizeof(int), &l->w);

			size_t global[3] = {l->w, l->h, batch_size * l->cout};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_residual_input_backward, 3, NULL, global, NULL, 0, NULL, NULL);

		} else if (l->type == LAYER_RESIDUAL_SUBTRACT) {
			/* Residual subtract backward - split gradient */
			/* Find the RESIDUAL_INPUT layer this references */
			int residual_layer_idx = l->residual_from >= 0 ? l->residual_from : -1;
			cl_mem grad_saved_dest = (residual_layer_idx >= 0) ? cnn->layers[residual_layer_idx].batch_grad_input : cnn->batch_grad_buf;

			/* Clear gradient destination for saved input */
			clEnqueueFillBuffer(cnn->queue, grad_saved_dest, &zero, sizeof(float),
								0, batch_size * in_size * sizeof(float), 0, NULL, NULL);

			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 0, sizeof(cl_mem), &grad_source);
			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 1, sizeof(cl_mem), &grad_saved_dest);
			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 2, sizeof(cl_mem), &l->batch_grad_input);
			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 3, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 4, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 5, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_batch_residual_subtract_backward, 6, sizeof(int), &l->w);

			size_t global[3] = {l->w, l->h, batch_size * l->cout};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_residual_subtract_backward, 3, NULL, global, NULL, 0, NULL, NULL);

		} else {
			/* Regular convolution backward */
			clSetKernelArg(cnn->k_batch_backward, 0, sizeof(cl_mem), &grad_source);
			clSetKernelArg(cnn->k_batch_backward, 1, sizeof(cl_mem), &l->batch_output);
			clSetKernelArg(cnn->k_batch_backward, 2, sizeof(cl_mem), &l->weights);
			clSetKernelArg(cnn->k_batch_backward, 3, sizeof(cl_mem), &l->batch_grad_input);
			clSetKernelArg(cnn->k_batch_backward, 4, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_backward, 5, sizeof(int), &l->cin4);
			clSetKernelArg(cnn->k_batch_backward, 6, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_batch_backward, 7, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_batch_backward, 8, sizeof(int), &l->w);
			clSetKernelArg(cnn->k_batch_backward, 9, sizeof(int), &l->use_relu);

			/* Flatten batch and cin4 into 3rd dimension for OpenCL 1.2 compatibility */
			size_t back_global[3] = {l->w, l->h, l->cin4 * batch_size};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_backward, 3, NULL, back_global, NULL, 0, NULL, NULL);

			/* Weight gradients - accumulated across batch */
			clSetKernelArg(cnn->k_batch_weight_grad, 0, sizeof(cl_mem), &layer_input);
			clSetKernelArg(cnn->k_batch_weight_grad, 1, sizeof(cl_mem), &grad_source);
			clSetKernelArg(cnn->k_batch_weight_grad, 2, sizeof(cl_mem), &l->batch_output);
			clSetKernelArg(cnn->k_batch_weight_grad, 3, sizeof(cl_mem), &l->grad_weights);
			clSetKernelArg(cnn->k_batch_weight_grad, 4, sizeof(cl_mem), &l->grad_bias);
			clSetKernelArg(cnn->k_batch_weight_grad, 5, sizeof(int), &batch_size);
			clSetKernelArg(cnn->k_batch_weight_grad, 6, sizeof(int), &l->cin4);
			clSetKernelArg(cnn->k_batch_weight_grad, 7, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_batch_weight_grad, 8, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_batch_weight_grad, 9, sizeof(int), &l->w);
			clSetKernelArg(cnn->k_batch_weight_grad, 10, sizeof(int), &l->use_relu);

			size_t wgrad_global[3] = {l->cout, l->cin4, 9};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_weight_grad, 3, NULL, wgrad_global, NULL, 0, NULL, NULL);
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &t_update_start);

	/* Weight updates - same as single-image */
	cnn->adam_t++;
	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];
		int w_size = l->cout * l->cin4 * 9;

		if (cnn->config.optimizer == OPTIMIZER_ADAM) {
			clSetKernelArg(cnn->k_adam_update, 0, sizeof(cl_mem), &l->weights);
			clSetKernelArg(cnn->k_adam_update, 1, sizeof(cl_mem), &l->bias);
			clSetKernelArg(cnn->k_adam_update, 2, sizeof(cl_mem), &l->grad_weights);
			clSetKernelArg(cnn->k_adam_update, 3, sizeof(cl_mem), &l->grad_bias);
			clSetKernelArg(cnn->k_adam_update, 4, sizeof(cl_mem), &l->adam_m_w);
			clSetKernelArg(cnn->k_adam_update, 5, sizeof(cl_mem), &l->adam_m_b);
			clSetKernelArg(cnn->k_adam_update, 6, sizeof(cl_mem), &l->adam_v_w);
			clSetKernelArg(cnn->k_adam_update, 7, sizeof(cl_mem), &l->adam_v_b);
			clSetKernelArg(cnn->k_adam_update, 8, sizeof(float), &cnn->config.learning_rate);
			clSetKernelArg(cnn->k_adam_update, 9, sizeof(float), &cnn->config.adam_beta1);
			clSetKernelArg(cnn->k_adam_update, 10, sizeof(float), &cnn->config.adam_beta2);
			clSetKernelArg(cnn->k_adam_update, 11, sizeof(float), &cnn->config.adam_epsilon);
			clSetKernelArg(cnn->k_adam_update, 12, sizeof(int), &cnn->adam_t);
			clSetKernelArg(cnn->k_adam_update, 13, sizeof(int), &w_size);
			clSetKernelArg(cnn->k_adam_update, 14, sizeof(int), &l->cout);

			size_t update_global = ((w_size > l->cout ? w_size : l->cout) + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_adam_update, 1, NULL, &update_global, NULL, 0, NULL, NULL);
		} else {
			clSetKernelArg(cnn->k_sgd_update, 0, sizeof(cl_mem), &l->weights);
			clSetKernelArg(cnn->k_sgd_update, 1, sizeof(cl_mem), &l->bias);
			clSetKernelArg(cnn->k_sgd_update, 2, sizeof(cl_mem), &l->grad_weights);
			clSetKernelArg(cnn->k_sgd_update, 3, sizeof(cl_mem), &l->grad_bias);
			clSetKernelArg(cnn->k_sgd_update, 4, sizeof(float), &cnn->config.learning_rate);
			clSetKernelArg(cnn->k_sgd_update, 5, sizeof(int), &w_size);
			clSetKernelArg(cnn->k_sgd_update, 6, sizeof(int), &l->cout);

			size_t update_global = ((w_size > l->cout ? w_size : l->cout) + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_sgd_update, 1, NULL, &update_global, NULL, 0, NULL, NULL);
		}
	}

	clFinish(cnn->queue);
	clock_gettime(CLOCK_MONOTONIC, &t_end);

	/* Record timing stats */
	if (cnn->config.use_profiling) {
		cnn->stats.forward_time_ms = (t_loss_start.tv_sec - t_forward_start.tv_sec) * 1000.0 +
									 (t_loss_start.tv_nsec - t_forward_start.tv_nsec) / 1000000.0;
		cnn->stats.loss_time_ms = (t_backward_start.tv_sec - t_loss_start.tv_sec) * 1000.0 +
								  (t_backward_start.tv_nsec - t_loss_start.tv_nsec) / 1000000.0;
		cnn->stats.backward_time_ms = (t_update_start.tv_sec - t_backward_start.tv_sec) * 1000.0 +
									  (t_update_start.tv_nsec - t_backward_start.tv_nsec) / 1000000.0;
		cnn->stats.update_time_ms = (t_end.tv_sec - t_update_start.tv_sec) * 1000.0 +
									(t_end.tv_nsec - t_update_start.tv_nsec) / 1000000.0;
		cnn->stats.total_time_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
								   (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	}

	return avg_loss;
}

float cnn_train_step(CNNDenoiser *cnn, float *noisy_input, float *clean_target, int batch_size) {
	if (!cnn->finalized) return -1.0f;

	/* Dispatch to batch training if batch_size > 1 and batch kernels available */
	if (batch_size > 1 && cnn->batch_kernels_available) {
		if (batch_size > cnn->config.max_batch_size) {
			fprintf(stderr, "Error: batch_size %d exceeds max_batch_size %d\n",
					batch_size, cnn->config.max_batch_size);
			return -1.0f;
		}
		return cnn_train_step_batch(cnn, noisy_input, clean_target, batch_size);
	}

	/* Single-image training path (original implementation) */
	struct timespec t_start, t_end, t_forward_start, t_backward_start, t_loss_start, t_update_start;
	clock_gettime(CLOCK_MONOTONIC, &t_start);

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
	if (!cnn->tuning_done && cnn->config.auto_tune_workgroup && cnn->n_layers > 0) {
		ConvLayer *first = &cnn->layers[0];
		cnn_auto_tune_workgroup(cnn, first->h, first->w, first->cout);
	}

	clock_gettime(CLOCK_MONOTONIC, &t_forward_start);
	cl_mem current = cnn->input_buf;
	cl_event forward_events[32];
	memset(forward_events, 0, sizeof(forward_events));

	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];

		if (l->type == LAYER_CONV) {
			/* Standard convolution layer */
			clSetKernelArg(cnn->k_forward, 0, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_forward, 1, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_forward, 2, sizeof(cl_mem), &l->weights);
			clSetKernelArg(cnn->k_forward, 3, sizeof(cl_mem), &l->bias);
			clSetKernelArg(cnn->k_forward, 4, sizeof(int), &l->cin4);
			clSetKernelArg(cnn->k_forward, 5, sizeof(int), &l->cout);
			clSetKernelArg(cnn->k_forward, 6, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_forward, 7, sizeof(int), &l->w);

			size_t global[3] = {l->w, l->h, (l->cout + 3) / 4};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward, 3, NULL, global, NULL, 0, NULL, &forward_events[i]);

		} else if (l->type == LAYER_RESIDUAL_INPUT) {
			/* Residual input layer - save input and pass it through */
			int buffer_size = l->cout * l->h * l->w;

			/* Copy input to saved buffer */
			clSetKernelArg(cnn->k_copy_buffer, 0, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_copy_buffer, 1, sizeof(cl_mem), &l->residual_saved);
			clSetKernelArg(cnn->k_copy_buffer, 2, sizeof(int), &buffer_size);

			size_t global_copy = (buffer_size + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_copy_buffer, 1, NULL, &global_copy, NULL, 0, NULL, NULL);

			/* Copy input to output (pass through) */
			clSetKernelArg(cnn->k_copy_buffer, 0, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_copy_buffer, 1, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_copy_buffer, 2, sizeof(int), &buffer_size);

			clEnqueueNDRangeKernel(cnn->queue, cnn->k_copy_buffer, 1, NULL, &global_copy, NULL, 0, NULL, NULL);

		} else if (l->type == LAYER_RESIDUAL_SUBTRACT) {
			/* Residual subtract layer - compute (saved_input - current) */
			int buffer_size = l->cout * l->h * l->w;

			/* Get the saved input from the referenced layer */
			cl_mem saved_input;
			if (l->residual_from >= 0 && l->residual_from < i) {
				/* Validate that the referenced layer is RESIDUAL_INPUT */
				if (cnn->layers[l->residual_from].type != LAYER_RESIDUAL_INPUT) {
					fprintf(stderr, "Error: Layer %d (%s) references layer %d for residual, but that layer is not RESIDUAL_INPUT\n",
							i, l->name, l->residual_from);
					saved_input = cnn->input_buf; /* Fallback to network input */
				} else {
					saved_input = cnn->layers[l->residual_from].residual_saved;
				}
			} else {
				/* Use network input if no specific layer referenced */
				saved_input = cnn->input_buf;
			}

			/* Compute: output = saved_input - current (denoised = input - noise) */
			clSetKernelArg(cnn->k_residual_subtract, 0, sizeof(cl_mem), &saved_input);
			clSetKernelArg(cnn->k_residual_subtract, 1, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_residual_subtract, 2, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_residual_subtract, 3, sizeof(int), &buffer_size);

			size_t global_sub = (buffer_size + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_residual_subtract, 1, NULL, &global_sub, NULL, 0, NULL, NULL);
		}

		/* Add skip connection if specified (works for all layer types) */
		if (l->skip_from >= 0 && l->skip_from < i) {
			ConvLayer *skip_layer = &cnn->layers[l->skip_from];

			/* Validate channel compatibility */
			if (l->cout != skip_layer->cout) {
				fprintf(stderr, "Error: Skip connection channel mismatch! Layer %d has %d channels, skip source layer %d has %d channels\n",
						i, l->cout, l->skip_from, skip_layer->cout);
				continue;
			}

			int skip_size = l->cout * l->h * l->w;

			clSetKernelArg(cnn->k_add_skip, 0, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_add_skip, 1, sizeof(cl_mem), &skip_layer->output);
			clSetKernelArg(cnn->k_add_skip, 2, sizeof(int), &skip_size);

			size_t skip_global = (skip_size + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_add_skip, 1, NULL, &skip_global, NULL, 0, NULL, NULL);
		}

		current = l->output;
	}
	clFinish(cnn->queue);

	/* Measure GPU execution time using OpenCL event profiling if enabled */
	double gpu_forward_time_ms = 0.0;
	if (cnn->config.use_profiling) {
		cl_ulong first_start = 0, last_end = 0;
		int events_found = 0;
		for (int i = 0; i < cnn->n_layers; i++) {
			if (forward_events[i]) {
				cl_ulong start, end;
				clGetEventProfilingInfo(forward_events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
				clGetEventProfilingInfo(forward_events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
				if (events_found == 0) first_start = start;
				last_end = end;
				events_found++;
				clReleaseEvent(forward_events[i]);
			}
		}
		if (last_end > first_start) {
			gpu_forward_time_ms = (last_end - first_start) / 1e6;
		}
	} else {
		for (int i = 0; i < cnn->n_layers; i++) {
			if (forward_events[i]) clReleaseEvent(forward_events[i]);
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &t_loss_start);

	/* ========== COMPUTE LOSS & GRADIENT ========== */
	ConvLayer *last_layer = &cnn->layers[cnn->n_layers - 1];
	int out_size = last_layer->cout * hw;
	float total_loss = 0.0f;

	/* In residual mode: network output is the predicted noise, compare directly to noise target
	 * In direct mode: network output is the denoised image, compare to clean target
	 * Either way, loss is computed on the network's raw output */
	cl_mem loss_input = last_layer->output;

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

			clSetKernelArg(cnn->k_mae_loss, 0, sizeof(cl_mem), &loss_input);
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
			for (int i = 0; i < num_workgroups; i++)
				loss += loss_per_wg[i];
			int rgb_size = (out_size / 4) * 3;
			float mae_loss_normalized = loss / rgb_size;
			cnn->last_mae_loss = mae_loss_normalized;
			total_loss += weight * mae_loss_normalized;

			clReleaseMemObject(loss_buf);
		} else if (loss_type == LOSS_MSE) {
			int num_workgroups = 64;
			cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, num_workgroups * 4, NULL, NULL);

			clSetKernelArg(cnn->k_mse_loss, 0, sizeof(cl_mem), &loss_input);
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
			for (int i = 0; i < num_workgroups; i++)
				loss += loss_per_wg[i];
			int rgb_size = (out_size / 4) * 3;
			float mse_loss_normalized = loss / rgb_size;
			cnn->last_mse_loss = mse_loss_normalized;
			total_loss += weight * mse_loss_normalized;

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
			for (int i = 0; i < num_workgroups; i++)
				loss += loss_per_wg[i];
			float laplace_loss_normalized = loss / out_size;
			cnn->last_laplace_loss = laplace_loss_normalized;
			total_loss += weight * laplace_loss_normalized;

			clReleaseMemObject(loss_buf);
		} else if (loss_type == LOSS_COLOR_VARIANCE) {
			int num_workgroups = 64;
			cl_mem loss_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, num_workgroups * 4, NULL, NULL);
			float zero_loss = 0.0f;
			clEnqueueFillBuffer(cnn->queue, loss_buf, &zero_loss, sizeof(float), 0, num_workgroups * 4, 0, NULL, NULL);

			int H = cnn->config.input_height;
			int W = cnn->config.input_width;

			clSetKernelArg(cnn->k_color_variance_loss, 0, sizeof(cl_mem), &loss_input);
			clSetKernelArg(cnn->k_color_variance_loss, 1, sizeof(cl_mem), &cnn->target_buf);
			clSetKernelArg(cnn->k_color_variance_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_color_variance_loss, 3, sizeof(cl_mem), &loss_buf);
			clSetKernelArg(cnn->k_color_variance_loss, 4, sizeof(int), &H);
			clSetKernelArg(cnn->k_color_variance_loss, 5, sizeof(int), &W);
			clSetKernelArg(cnn->k_color_variance_loss, 6, 256 * sizeof(float), NULL);

			size_t global_loss = 256 * num_workgroups;
			size_t local_loss = 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_color_variance_loss, 1, NULL, &global_loss, &local_loss, 0, NULL, NULL);

			float loss_per_wg[64];
			clEnqueueReadBuffer(cnn->queue, loss_buf, CL_TRUE, 0, num_workgroups * 4, loss_per_wg, 0, NULL, NULL);

			float loss = 0.0f;
			for (int i = 0; i < num_workgroups; i++)
				loss += loss_per_wg[i];
			int pixels = H * W;
			float color_loss_normalized = loss / pixels;
			cnn->last_color_loss = color_loss_normalized;
			total_loss += weight * color_loss_normalized;

			clReleaseMemObject(loss_buf);
		} else if (loss_type == LOSS_SSIM) {
			int H = cnn->config.input_height;
			int W = cnn->config.input_width;
			int batch_size_one = 1;
			
			/* Create buffer for per-pixel loss values */
			cl_mem loss_pixel_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, hw * sizeof(float), NULL, NULL);

			clSetKernelArg(cnn->k_batch_ssim_loss, 0, sizeof(cl_mem), &loss_input);
			clSetKernelArg(cnn->k_batch_ssim_loss, 1, sizeof(cl_mem), &cnn->target_buf);
			clSetKernelArg(cnn->k_batch_ssim_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_ssim_loss, 3, sizeof(cl_mem), &loss_pixel_buf);
			clSetKernelArg(cnn->k_batch_ssim_loss, 4, sizeof(int), &batch_size_one);
			clSetKernelArg(cnn->k_batch_ssim_loss, 5, sizeof(int), &H);
			clSetKernelArg(cnn->k_batch_ssim_loss, 6, sizeof(int), &W);

			size_t ssim_global[3] = {W, H, 1};
			cl_int err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_ssim_loss, 3, NULL, ssim_global, NULL, 0, NULL, NULL);
			if (err != CL_SUCCESS) fprintf(stderr, "[ERROR] SSIM loss kernel failed: %d\n", err);

			/* Read back per-pixel loss and sum */
			float *pixel_losses = malloc(hw * sizeof(float));
			clEnqueueReadBuffer(cnn->queue, loss_pixel_buf, CL_TRUE, 0, hw * sizeof(float), pixel_losses, 0, NULL, NULL);
			
			float loss = 0.0f;
			for (int i = 0; i < hw; i++) {
				loss += pixel_losses[i];
			}
			int rgb_pixels = hw * 3;
			float ssim_loss_normalized = loss / rgb_pixels;
			cnn->last_ssim_loss = ssim_loss_normalized;
			total_loss += weight * ssim_loss_normalized;
			
			free(pixel_losses);
			clReleaseMemObject(loss_pixel_buf);
		} else if (loss_type == LOSS_SOBEL) {
			int H = cnn->config.input_height;
			int W = cnn->config.input_width;
			int C = last_layer->cout;
			int batch_size_one = 1;
			
			/* Create buffer for per-pixel loss values */
			cl_mem loss_pixel_buf = clCreateBuffer(cnn->ctx, CL_MEM_READ_WRITE, hw * C * sizeof(float), NULL, NULL);

			clSetKernelArg(cnn->k_batch_sobel_loss, 0, sizeof(cl_mem), &loss_input);
			clSetKernelArg(cnn->k_batch_sobel_loss, 1, sizeof(cl_mem), &cnn->target_buf);
			clSetKernelArg(cnn->k_batch_sobel_loss, 2, sizeof(cl_mem), &temp_grad_loss);
			clSetKernelArg(cnn->k_batch_sobel_loss, 3, sizeof(cl_mem), &loss_pixel_buf);
			clSetKernelArg(cnn->k_batch_sobel_loss, 4, sizeof(int), &batch_size_one);
			clSetKernelArg(cnn->k_batch_sobel_loss, 5, sizeof(int), &H);
			clSetKernelArg(cnn->k_batch_sobel_loss, 6, sizeof(int), &W);
			clSetKernelArg(cnn->k_batch_sobel_loss, 7, sizeof(int), &C);

			size_t sobel_global[3] = {W, H, C};
			cl_int err = clEnqueueNDRangeKernel(cnn->queue, cnn->k_batch_sobel_loss, 3, NULL, sobel_global, NULL, 0, NULL, NULL);
			if (err != CL_SUCCESS) fprintf(stderr, "[ERROR] Sobel loss kernel failed: %d\n", err);

			/* Read back per-pixel loss and sum (RGB only, skip luminance) */
			float *pixel_losses = malloc(hw * C * sizeof(float));
			clEnqueueReadBuffer(cnn->queue, loss_pixel_buf, CL_TRUE, 0, hw * C * sizeof(float), pixel_losses, 0, NULL, NULL);
			
			float loss = 0.0f;
			for (int c = 0; c < 3; c++) {  /* RGB only */
				for (int i = 0; i < hw; i++) {
					loss += pixel_losses[c * hw + i];
				}
			}
			int rgb_pixels = hw * 3;
			float sobel_loss_normalized = loss / rgb_pixels;
			cnn->last_sobel_loss = sobel_loss_normalized;
			total_loss += weight * sobel_loss_normalized;
			
			free(pixel_losses);
			clReleaseMemObject(loss_pixel_buf);
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
	clFinish(cnn->queue);
	clock_gettime(CLOCK_MONOTONIC, &t_backward_start);

	/* ========== BACKWARD PASS ========== */

	/* Initialize all layer output gradients to zero, then set last layer from loss */
	float zero_grad = 0.0f;
	for (int i = 0; i < cnn->n_layers; i++) {
		int layer_out_size = cnn->layers[i].cout * hw;
		clEnqueueFillBuffer(cnn->queue, cnn->layers[i].grad_output, &zero_grad, sizeof(float),
							0, layer_out_size * 4, 0, NULL, NULL);
	}

	/* Copy loss gradient to last layer's output gradient */
	clEnqueueCopyBuffer(cnn->queue, cnn->grad_buf, last_layer->grad_output,
						0, 0, out_size * 4, 0, NULL, NULL);

	/* Backward through layers */
	for (int i = cnn->n_layers - 1; i >= 0; i--) {
		ConvLayer *l = &cnn->layers[i];
		cl_mem layer_input = (i == 0) ? cnn->input_buf : cnn->layers[i - 1].output;

		/* STEP 1: Accumulate skip gradients if this layer is a skip source */
		for (int j = i + 1; j < cnn->n_layers; j++) {
			if (cnn->layers[j].skip_from == i && l->cout == cnn->layers[j].cout) {
				/* Layer j skips from layer i, so gradient flows back to i's output */
				int skip_size = l->cout * l->h * l->w;

				clSetKernelArg(cnn->k_add_skip_grad, 0, sizeof(cl_mem), &l->grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 1, sizeof(cl_mem), &cnn->layers[j].grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 2, sizeof(int), &skip_size);

				size_t skip_grad_global = (skip_size + 255) / 256 * 256;
				clEnqueueNDRangeKernel(cnn->queue, cnn->k_add_skip_grad, 1, NULL, &skip_grad_global, NULL, 0, NULL, NULL);
			}
		}

		/* STEP 2: Layer-type specific backward pass */
		if (l->type == LAYER_CONV) {
			/* Standard conv layer - compute weight and bias gradients */
			clSetKernelArg(cnn->k_weight_grad, 0, sizeof(cl_mem), &layer_input);
			clSetKernelArg(cnn->k_weight_grad, 1, sizeof(cl_mem), &l->grad_output);
			clSetKernelArg(cnn->k_weight_grad, 2, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_weight_grad, 3, sizeof(cl_mem), &l->grad_weights);
			clSetKernelArg(cnn->k_weight_grad, 4, sizeof(cl_mem), &l->grad_bias);
			clSetKernelArg(cnn->k_weight_grad, 5, sizeof(int), &l->cin4);
			clSetKernelArg(cnn->k_weight_grad, 6, sizeof(int), &l->h);
			clSetKernelArg(cnn->k_weight_grad, 7, sizeof(int), &l->w);
			clSetKernelArg(cnn->k_weight_grad, 8, sizeof(int), &l->use_relu);

			size_t grad_global[3] = {(size_t)l->cout, (size_t)l->cin4, 9};
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_weight_grad, 3, NULL, grad_global, NULL, 0, NULL, NULL);

			/* Backprop gradient to previous layer */
			if (i > 0) {
				int prev_cin4 = cnn->layers[i - 1].cout / 4;

				clSetKernelArg(cnn->k_backward, 0, sizeof(cl_mem), &l->grad_output);
				clSetKernelArg(cnn->k_backward, 1, sizeof(cl_mem), &l->output);
				clSetKernelArg(cnn->k_backward, 2, sizeof(cl_mem), &l->weights);
				clSetKernelArg(cnn->k_backward, 3, sizeof(cl_mem), &cnn->layers[i - 1].grad_output);
				clSetKernelArg(cnn->k_backward, 4, sizeof(int), &prev_cin4);
				clSetKernelArg(cnn->k_backward, 5, sizeof(int), &l->cout);
				clSetKernelArg(cnn->k_backward, 6, sizeof(int), &l->h);
				clSetKernelArg(cnn->k_backward, 7, sizeof(int), &l->w);
				clSetKernelArg(cnn->k_backward, 8, sizeof(int), &l->use_relu);

				size_t back_global[3] = {(size_t)l->w, (size_t)l->h, (size_t)prev_cin4};
				size_t back_local[3] = {16, 8, 1};
				clEnqueueNDRangeKernel(cnn->queue, cnn->k_backward, 3, NULL, back_global, back_local, 0, NULL, NULL);
			}

		} else if (l->type == LAYER_RESIDUAL_INPUT) {
			/* Residual input layer - gradient passes through to previous layer */
			/* Also need to accumulate gradient for layers that use this saved input */
			if (i > 0) {
				int buffer_size = l->cout * l->h * l->w;

				/* Copy gradient to previous layer */
				clSetKernelArg(cnn->k_add_skip_grad, 0, sizeof(cl_mem), &cnn->layers[i - 1].grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 1, sizeof(cl_mem), &l->grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 2, sizeof(int), &buffer_size);

				size_t grad_global = (buffer_size + 255) / 256 * 256;
				clEnqueueNDRangeKernel(cnn->queue, cnn->k_add_skip_grad, 1, NULL, &grad_global, NULL, 0, NULL, NULL);
			}

		} else if (l->type == LAYER_RESIDUAL_SUBTRACT) {
			/* Residual subtract: output = saved_input - noise_prediction
			 * d_loss/d_saved_input = d_loss/d_output * 1
			 * d_loss/d_noise_prediction = d_loss/d_output * (-1)
			 */
			int buffer_size = l->cout * l->h * l->w;

			/* Gradient w.r.t. noise prediction (previous layer) */
			if (i > 0) {
				/* Negate gradient and add to previous layer */
				clSetKernelArg(cnn->k_negate, 0, sizeof(cl_mem), &l->grad_output);
				clSetKernelArg(cnn->k_negate, 1, sizeof(cl_mem), &cnn->temp_grad);
				clSetKernelArg(cnn->k_negate, 2, sizeof(int), &buffer_size);

				size_t grad_global = (buffer_size + 255) / 256 * 256;
				clEnqueueNDRangeKernel(cnn->queue, cnn->k_negate, 1, NULL, &grad_global, NULL, 0, NULL, NULL);

				/* Add negated gradient to previous layer */
				clSetKernelArg(cnn->k_add_skip_grad, 0, sizeof(cl_mem), &cnn->layers[i - 1].grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 1, sizeof(cl_mem), &cnn->temp_grad);
				clSetKernelArg(cnn->k_add_skip_grad, 2, sizeof(int), &buffer_size);

				clEnqueueNDRangeKernel(cnn->queue, cnn->k_add_skip_grad, 1, NULL, &grad_global, NULL, 0, NULL, NULL);
			}

			/* Gradient w.r.t. saved input (flows back to residual_from layer) */
			if (l->residual_from >= 0 && l->residual_from < i) {
				/* Add gradient to the layer that saved the input */
				clSetKernelArg(cnn->k_add_skip_grad, 0, sizeof(cl_mem), &cnn->layers[l->residual_from].grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 1, sizeof(cl_mem), &l->grad_output);
				clSetKernelArg(cnn->k_add_skip_grad, 2, sizeof(int), &buffer_size);

				size_t grad_global = (buffer_size + 255) / 256 * 256;
				clEnqueueNDRangeKernel(cnn->queue, cnn->k_add_skip_grad, 1, NULL, &grad_global, NULL, 0, NULL, NULL);
			}
		}
	}
	clFinish(cnn->queue);
	clock_gettime(CLOCK_MONOTONIC, &t_update_start);

	/* ========== UPDATE WEIGHTS ========== */
	if (cnn->config.optimizer == OPTIMIZER_SGD) {
		for (int i = 0; i < cnn->n_layers; i++) {
			ConvLayer *l = &cnn->layers[i];

			/* Skip non-CONV layers - they don't have weights */
			if (l->type != LAYER_CONV) continue;

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

			/* Skip non-CONV layers - they don't have weights */
			if (l->type != LAYER_CONV) continue;

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
	clock_gettime(CLOCK_MONOTONIC, &t_end);

	/* Accumulate timing stats */
	double forward_time;
	if (cnn->config.use_profiling && gpu_forward_time_ms > 0.0) {
		forward_time = gpu_forward_time_ms; /* Use GPU-measured time for accuracy */
	} else {
		/* Fallback to wall-clock time */
		forward_time = (t_loss_start.tv_sec - t_forward_start.tv_sec) * 1000.0 +
					   (t_loss_start.tv_nsec - t_forward_start.tv_nsec) / 1e6;
	}
	double loss_time = (t_backward_start.tv_sec - t_loss_start.tv_sec) * 1000.0 +
					   (t_backward_start.tv_nsec - t_loss_start.tv_nsec) / 1e6;
	double backward_time = (t_update_start.tv_sec - t_backward_start.tv_sec) * 1000.0 +
						   (t_update_start.tv_nsec - t_backward_start.tv_nsec) / 1e6;
	double update_time = (t_end.tv_sec - t_update_start.tv_sec) * 1000.0 +
						 (t_end.tv_nsec - t_update_start.tv_nsec) / 1e6;
	double total_time = (t_end.tv_sec - t_start.tv_sec) * 1000.0 +
						(t_end.tv_nsec - t_start.tv_nsec) / 1e6;

	cnn->stats.forward_time_ms = forward_time;
	cnn->stats.backward_time_ms = backward_time;
	cnn->stats.loss_time_ms = loss_time;
	cnn->stats.update_time_ms = update_time;
	cnn->stats.total_time_ms = total_time;
	cnn->stats_count++;

	return total_loss;
}

int cnn_get_num_parameters(CNNDenoiser *cnn) {
	int total = 0;
	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];
		/* Only count parameters for CONV layers */
		if (l->type == LAYER_CONV) {
			total += l->cout * l->cin4 * 9 * 4; /* weights */
			total += l->cout;					/* biases */
		}
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
		const char *type_name;
		switch (l->type) {
		case LAYER_CONV:
			type_name = "Conv";
			break;
		case LAYER_RESIDUAL_INPUT:
			type_name = "ResInput";
			break;
		case LAYER_RESIDUAL_SUBTRACT:
			type_name = "ResSub";
			break;
		default:
			type_name = "Unknown";
			break;
		}

		if (l->type == LAYER_CONV) {
			printf("  [%d] %s (%s): %d->%d channels, %s", i, l->name, type_name, l->cin, l->cout,
				   l->use_relu ? "ReLU" : "Linear");
			if (l->skip_from >= 0) printf(" + skip[%d]", l->skip_from);
			printf("\n");
		} else if (l->type == LAYER_RESIDUAL_INPUT) {
			printf("  [%d] %s (%s): %d channels (save input)\n", i, l->name, type_name, l->cout);
		} else if (l->type == LAYER_RESIDUAL_SUBTRACT) {
			printf("  [%d] %s (%s): %d channels (input", i, l->name, type_name, l->cout);
			if (l->residual_from >= 0) printf("[%d]", l->residual_from);
			printf(" - noise)\n");
		}
	}
	printf("Total parameters: %d\n", cnn_get_num_parameters(cnn));
	printf("========================\n\n");
}

void cnn_get_timing_stats(CNNDenoiser *cnn, TimingStats *stats) {
	if (!cnn || !stats) return;
	*stats = cnn->stats;
}

void cnn_reset_timing_stats(CNNDenoiser *cnn) {
	if (!cnn) return;
	memset(&cnn->stats, 0, sizeof(TimingStats));
	cnn->stats_count = 0;
}

void cnn_destroy(CNNDenoiser *cnn) {
	if (!cnn) return;

	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];

		/* Free host memory and OpenCL buffers based on layer type */
		if (l->type == LAYER_CONV) {
			free(l->h_weights);
			free(l->h_bias);
			free(l->h_grad_w);
			free(l->h_grad_b);
			if (l->weights) clReleaseMemObject(l->weights);
			if (l->bias) clReleaseMemObject(l->bias);
			if (l->grad_bias) clReleaseMemObject(l->grad_bias);
			if (l->grad_weights) clReleaseMemObject(l->grad_weights);

			if (cnn->config.optimizer == OPTIMIZER_ADAM) {
				if (l->adam_m_w) clReleaseMemObject(l->adam_m_w);
				if (l->adam_v_w) clReleaseMemObject(l->adam_v_w);
				if (l->adam_m_b) clReleaseMemObject(l->adam_m_b);
				if (l->adam_v_b) clReleaseMemObject(l->adam_v_b);
			}
		} else if (l->type == LAYER_RESIDUAL_INPUT) {
			if (l->residual_saved) clReleaseMemObject(l->residual_saved);
		}

		/* Common cleanup for all layer types */
		if (l->output) clReleaseMemObject(l->output);
		if (l->grad_output) clReleaseMemObject(l->grad_output);
		if (l->grad_input) clReleaseMemObject(l->grad_input);
		if (l->skip_input) clReleaseMemObject(l->skip_input);
	}

	if (cnn->finalized) {
		clReleaseMemObject(cnn->input_buf);
		clReleaseMemObject(cnn->target_buf);
		clReleaseMemObject(cnn->grad_buf);
		clReleaseMemObject(cnn->residual_buf);
		clReleaseMemObject(cnn->temp_grad);

		/* Release batch buffers if allocated */
		if (cnn->config.max_batch_size > 1) {
			if (cnn->batch_input_buf) clReleaseMemObject(cnn->batch_input_buf);
			if (cnn->batch_target_buf) clReleaseMemObject(cnn->batch_target_buf);
			if (cnn->batch_loss_buf) clReleaseMemObject(cnn->batch_loss_buf);
			if (cnn->batch_grad_buf) clReleaseMemObject(cnn->batch_grad_buf);

			/* Release per-layer batch buffers */
			for (int i = 0; i < cnn->n_layers; i++) {
				ConvLayer *l = &cnn->layers[i];
				if (l->batch_output) clReleaseMemObject(l->batch_output);
				if (l->batch_grad_input) clReleaseMemObject(l->batch_grad_input);
				if (l->batch_residual_saved) clReleaseMemObject(l->batch_residual_saved);
			}

			/* Release batch kernels */
			if (cnn->k_batch_forward) clReleaseKernel(cnn->k_batch_forward);
			if (cnn->k_batch_backward) clReleaseKernel(cnn->k_batch_backward);
			if (cnn->k_batch_weight_grad) clReleaseKernel(cnn->k_batch_weight_grad);
			if (cnn->k_batch_loss_reduce) clReleaseKernel(cnn->k_batch_loss_reduce);
			if (cnn->k_batch_clear_loss) clReleaseKernel(cnn->k_batch_clear_loss);
			if (cnn->k_batch_add_weighted_grad) clReleaseKernel(cnn->k_batch_add_weighted_grad);
			if (cnn->k_batch_mae_loss) clReleaseKernel(cnn->k_batch_mae_loss);
			if (cnn->k_batch_mse_loss) clReleaseKernel(cnn->k_batch_mse_loss);
			if (cnn->k_batch_laplace_loss) clReleaseKernel(cnn->k_batch_laplace_loss);
			if (cnn->k_batch_color_loss) clReleaseKernel(cnn->k_batch_color_loss);
			if (cnn->k_batch_ssim_loss) clReleaseKernel(cnn->k_batch_ssim_loss);
			if (cnn->k_batch_sobel_loss) clReleaseKernel(cnn->k_batch_sobel_loss);
			if (cnn->k_batch_residual_input) clReleaseKernel(cnn->k_batch_residual_input);
			if (cnn->k_batch_residual_subtract) clReleaseKernel(cnn->k_batch_residual_subtract);
			if (cnn->k_batch_residual_input_backward) clReleaseKernel(cnn->k_batch_residual_input_backward);
			if (cnn->k_batch_residual_subtract_backward) clReleaseKernel(cnn->k_batch_residual_subtract_backward);
		}
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

int cnn_save_weights(CNNDenoiser *cnn, const char *filepath) {
	if (!cnn || !cnn->finalized) return -1;

	FILE *f = fopen(filepath, "wb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for writing\n", filepath);
		return -1;
	}

	/* Write header */
	int magic = 0x434E4E57; /* "CNNW" */
	fwrite(&magic, sizeof(int), 1, f);
	fwrite(&cnn->n_layers, sizeof(int), 1, f);
	fwrite(&cnn->config, sizeof(CNNConfig), 1, f);

	/* Write each layer's weights and biases */
	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];

		/* Write layer metadata */
		fwrite(&l->type, sizeof(int), 1, f);
		fwrite(&l->cin, sizeof(int), 1, f);
		fwrite(&l->cout, sizeof(int), 1, f);
		fwrite(&l->use_relu, sizeof(int), 1, f);
		fwrite(&l->skip_from, sizeof(int), 1, f);
		fwrite(&l->residual_from, sizeof(int), 1, f);
		fwrite(l->name, sizeof(char), 64, f);

		/* Only save weights for CONV layers */
		if (l->type == LAYER_CONV) {
			/* Download weights from GPU */
			int w_size = l->cout * l->cin4 * 9;
			clEnqueueReadBuffer(cnn->queue, l->weights, CL_TRUE, 0,
								w_size * 16, l->h_weights, 0, NULL, NULL);
			clEnqueueReadBuffer(cnn->queue, l->bias, CL_TRUE, 0,
								l->cout * 4, l->h_bias, 0, NULL, NULL);

			/* Write weights and biases */
			fwrite(l->h_weights, sizeof(float), w_size * 4, f);
			fwrite(l->h_bias, sizeof(float), l->cout, f);

			/* Write Adam optimizer state if using Adam */
			if (cnn->config.optimizer == OPTIMIZER_ADAM) {
				float *adam_m_w = malloc(w_size * 16);
				float *adam_v_w = malloc(w_size * 16);
				float *adam_m_b = malloc(l->cout * 4);
				float *adam_v_b = malloc(l->cout * 4);

				clEnqueueReadBuffer(cnn->queue, l->adam_m_w, CL_TRUE, 0, w_size * 16, adam_m_w, 0, NULL, NULL);
				clEnqueueReadBuffer(cnn->queue, l->adam_v_w, CL_TRUE, 0, w_size * 16, adam_v_w, 0, NULL, NULL);
				clEnqueueReadBuffer(cnn->queue, l->adam_m_b, CL_TRUE, 0, l->cout * 4, adam_m_b, 0, NULL, NULL);
				clEnqueueReadBuffer(cnn->queue, l->adam_v_b, CL_TRUE, 0, l->cout * 4, adam_v_b, 0, NULL, NULL);

				fwrite(adam_m_w, sizeof(float), w_size * 4, f);
				fwrite(adam_v_w, sizeof(float), w_size * 4, f);
				fwrite(adam_m_b, sizeof(float), l->cout, f);
				fwrite(adam_v_b, sizeof(float), l->cout, f);

				free(adam_m_w);
				free(adam_v_w);
				free(adam_m_b);
				free(adam_v_b);
			}
		}
	}

	/* Write Adam timestep */
	fwrite(&cnn->adam_t, sizeof(int), 1, f);

	fclose(f);
	printf("Saved network weights to %s\n", filepath);
	return 0;
}

int cnn_load_weights(CNNDenoiser *cnn, const char *filepath) {
	if (!cnn || !cnn->finalized) return -1;

	FILE *f = fopen(filepath, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for reading\n", filepath);
		return -1;
	}

	/* Read and verify header */
	int magic, n_layers;
	CNNConfig saved_config;

	fread(&magic, sizeof(int), 1, f);
	if (magic != 0x434E4E57) {
		fprintf(stderr, "Invalid file format\n");
		fclose(f);
		return -1;
	}

	fread(&n_layers, sizeof(int), 1, f);
	if (n_layers != cnn->n_layers) {
		fprintf(stderr, "Layer count mismatch: file has %d, network has %d\n", n_layers, cnn->n_layers);
		fclose(f);
		return -1;
	}

	fread(&saved_config, sizeof(CNNConfig), 1, f);

	/* Load each layer's weights */
	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];

		/* Read and verify layer metadata */
		int type, cin, cout, use_relu, skip_from, residual_from;
		char name[64];
		fread(&type, sizeof(int), 1, f);
		fread(&cin, sizeof(int), 1, f);
		fread(&cout, sizeof(int), 1, f);
		fread(&use_relu, sizeof(int), 1, f);
		fread(&skip_from, sizeof(int), 1, f);
		fread(&residual_from, sizeof(int), 1, f);
		fread(name, sizeof(char), 64, f);

		if (cin != l->cin || cout != l->cout || type != l->type) {
			fprintf(stderr, "Layer %d mismatch (cin=%d/%d, cout=%d/%d, type=%d/%d)\n",
					i, cin, l->cin, cout, l->cout, type, l->type);
			fclose(f);
			return -1;
		}

		/* Only load weights for CONV layers */
		if (l->type == LAYER_CONV) {
			/* Read weights and biases */
			int w_size = l->cout * l->cin4 * 9;
			fread(l->h_weights, sizeof(float), w_size * 4, f);
			fread(l->h_bias, sizeof(float), l->cout, f);

			/* Upload to GPU */
			clEnqueueWriteBuffer(cnn->queue, l->weights, CL_TRUE, 0,
								 w_size * 16, l->h_weights, 0, NULL, NULL);
			clEnqueueWriteBuffer(cnn->queue, l->bias, CL_TRUE, 0,
								 l->cout * 4, l->h_bias, 0, NULL, NULL);

			/* Read Adam optimizer state if present */
			if (saved_config.optimizer == OPTIMIZER_ADAM) {
				float *adam_m_w = malloc(w_size * 16);
				float *adam_v_w = malloc(w_size * 16);
				float *adam_m_b = malloc(l->cout * 4);
				float *adam_v_b = malloc(l->cout * 4);

				fread(adam_m_w, sizeof(float), w_size * 4, f);
				fread(adam_v_w, sizeof(float), w_size * 4, f);
				fread(adam_m_b, sizeof(float), l->cout, f);
				fread(adam_v_b, sizeof(float), l->cout, f);

				if (cnn->config.optimizer == OPTIMIZER_ADAM) {
					/* RESET Adam state for stable fine-tuning with new learning rate */
					printf("Resetting Adam momentum buffers for layer %d to zero for stable fine-tuning\n", i);
					memset(adam_m_w, 0, w_size * 16);
					memset(adam_v_w, 0, w_size * 16);
					memset(adam_m_b, 0, l->cout * 4);
					memset(adam_v_b, 0, l->cout * 4);

					clEnqueueWriteBuffer(cnn->queue, l->adam_m_w, CL_TRUE, 0, w_size * 16, adam_m_w, 0, NULL, NULL);
					clEnqueueWriteBuffer(cnn->queue, l->adam_v_w, CL_TRUE, 0, w_size * 16, adam_v_w, 0, NULL, NULL);
					clEnqueueWriteBuffer(cnn->queue, l->adam_m_b, CL_TRUE, 0, l->cout * 4, adam_m_b, 0, NULL, NULL);
					clEnqueueWriteBuffer(cnn->queue, l->adam_v_b, CL_TRUE, 0, l->cout * 4, adam_v_b, 0, NULL, NULL);
				}

				free(adam_m_w);
				free(adam_v_w);
				free(adam_m_b);
				free(adam_v_b);
			}
		}
	}

	/* Read Adam timestep */
	int saved_adam_t = 0;
	fread(&saved_adam_t, sizeof(int), 1, f);

	printf("Loaded adam_t=%d from file, but RESETTING to 0 for stable fine-tuning\n", saved_adam_t);
	cnn->adam_t = 0;

	fclose(f);
	printf("Loaded network weights from %s\n", filepath);
	return 0;
}

void cnn_add_gaussian_noise(float *clean, float *noisy, int size, float sigma) {
	for (int i = 0; i < size; i++) {
		float u1 = (float)rand() / RAND_MAX;
		float u2 = (float)rand() / RAND_MAX;
		float noise = sigma * sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(6.283185f * u2);
		noisy[i] = clean[i] + noise;
	}
}

/* Helper: Convert RGB image to RGBA (RGB + Luminance) format for float4 processing */
void cnn_load_rgba_luminance(const unsigned char *rgb, const unsigned char *lum, float *rgba, int width, int height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int rgb_idx = (y * width + x) * 3;
			int lum_idx = y * width + x;
			int rgba_idx = (y * width + x) * 4;

			rgba[rgba_idx + 0] = rgb[rgb_idx + 0] / 255.0f;
			rgba[rgba_idx + 1] = rgb[rgb_idx + 1] / 255.0f;
			rgba[rgba_idx + 2] = rgb[rgb_idx + 2] / 255.0f;
			rgba[rgba_idx + 3] = lum[lum_idx] / 255.0f;
		}
	}
}

void cnn_rgb_to_rgba_luminance(const unsigned char *rgb, float *rgba, int width, int height) {
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
			rgba[rgba_idx + 3] = 0.299f * r + 0.587f * g + 0.114f * b;
		}
	}
}

/* Helper: Convert RGBA (RGB + Luminance) back to RGB image */
void cnn_rgba_luminance_to_rgb(const float *rgba, unsigned char *rgb, int width, int height) {
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
int cnn_prepare_training_batch(const unsigned char *clean_rgb, const unsigned char *clean_lum,
							   unsigned char *noisy_rgb, unsigned char *noisy_lum,
							   float *clean_rgba, float *noisy_rgba,
							   int width, int height, float noise_sigma) {
	if (width != 800 || height != 600) {
		fprintf(stderr, "Error: Image must be 800x600 (got %dx%d)\n", width, height);
		return -1;
	}

	/* Convert clean RGB+Luminance to RGBA */
	cnn_load_rgba_luminance(clean_rgb, clean_lum, clean_rgba, width, height);

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
int cnn_inference_rgb(CNNDenoiser *cnn, const unsigned char *input_rgb,
					  unsigned char *output_rgb, int width, int height) {
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

void learning_rate_decay_init(LearningRateDecay *lr_decay,
							  float initial_lr, float decay_rate, int decay_steps) {
	lr_decay->initial_lr = initial_lr;
	lr_decay->decay_rate = decay_rate;
	lr_decay->decay_steps = decay_steps;
	lr_decay->step = 0;
}

float learning_rate_decay_get(LearningRateDecay *lr_decay, int current_step) {
	lr_decay->step = current_step;
	return lr_decay->initial_lr * powf(lr_decay->decay_rate,
									   (float)(lr_decay->step) / lr_decay->decay_steps);
}

void cnn_set_learning_rate(CNNDenoiser *cnn, float learning_rate) {
	cnn->config.learning_rate = learning_rate;
}

float cnn_get_learning_rate(CNNDenoiser *cnn) {
	return cnn->config.learning_rate;
}

void cnn_get_output(CNNDenoiser *cnn, float *output) {
	if (!cnn || !cnn->finalized) return;

	int input_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;
	ConvLayer *last_layer = &cnn->layers[cnn->n_layers - 1];

	/* In residual mode, need to compute: output = input - network_output */
	if (cnn->config.residual_mode) {
		/* DEBUG: Read raw values to see what's happening */
		float test_input[10], test_prediction[10], test_result[10];
		clEnqueueReadBuffer(cnn->queue, cnn->input_buf, CL_TRUE, 0, 40, test_input, 0, NULL, NULL);
		clEnqueueReadBuffer(cnn->queue, last_layer->output, CL_TRUE, 0, 40, test_prediction, 0, NULL, NULL);

		/* Run residual_subtract kernel: residual_buf = input - last_layer->output */
		clSetKernelArg(cnn->k_residual_subtract, 0, sizeof(cl_mem), &cnn->input_buf);
		clSetKernelArg(cnn->k_residual_subtract, 1, sizeof(cl_mem), &last_layer->output);
		clSetKernelArg(cnn->k_residual_subtract, 2, sizeof(cl_mem), &cnn->residual_buf);
		clSetKernelArg(cnn->k_residual_subtract, 3, sizeof(int), &input_size);

		size_t global = (input_size + 255) / 256 * 256;
		clEnqueueNDRangeKernel(cnn->queue, cnn->k_residual_subtract, 1, NULL, &global, NULL, 0, NULL, NULL);

		/* Read the computed residual output */
		clEnqueueReadBuffer(cnn->queue, cnn->residual_buf, CL_TRUE, 0,
							input_size * sizeof(float), output, 0, NULL, NULL);

		/* DEBUG: Check result */
		clEnqueueReadBuffer(cnn->queue, cnn->residual_buf, CL_TRUE, 0, 40, test_result, 0, NULL, NULL);
		printf("  [DEBUG cnn_get_output] input_buf[0:2]=%.3f,%.3f,%.3f prediction[0:2]=%.3f,%.3f,%.3f result[0:2]=%.3f,%.3f,%.3f\n",
			   test_input[0], test_input[1], test_input[2],
			   test_prediction[0], test_prediction[1], test_prediction[2],
			   test_result[0], test_result[1], test_result[2]);
	} else {
		/* Direct mode: just read the last layer output */
		clEnqueueReadBuffer(cnn->queue, last_layer->output, CL_TRUE, 0,
							input_size * sizeof(float), output, 0, NULL, NULL);
	}
}

/* Get batch output for first image in batch (for debugging batch training) */
void cnn_get_batch_output(CNNDenoiser *cnn, float *output, int batch_index) {
	if (!cnn || !cnn->finalized || batch_index < 0 || batch_index >= cnn->config.max_batch_size) return;

	int input_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;
	ConvLayer *last_layer = &cnn->layers[cnn->n_layers - 1];

	/* Read from batch_output buffer */
	size_t offset = batch_index * input_size * sizeof(float);
	clEnqueueReadBuffer(cnn->queue, last_layer->batch_output, CL_TRUE, offset,
						input_size * sizeof(float), output, 0, NULL, NULL);
}

int cnn_denoise(CNNDenoiser *cnn, float *noisy_input, float *denoised_output, int batch_size) {
	if (!cnn || !cnn->finalized) return -1;

	int input_size = cnn->config.input_height * cnn->config.input_width * cnn->config.input_channels;

	/* Upload input */
	clEnqueueWriteBuffer(cnn->queue, cnn->input_buf, CL_FALSE, 0, input_size * 4,
						 noisy_input, 0, NULL, NULL);

	/* Forward pass */
	cl_mem current = cnn->input_buf;
	int last_layer_idx = cnn->n_layers - 1;

	for (int i = 0; i < cnn->n_layers; i++) {
		ConvLayer *l = &cnn->layers[i];

		if (l->type == LAYER_CONV) {
			/* Standard convolution layer */
			/* Use fused residual kernel for last layer if old residual mode enabled */
			if (i == last_layer_idx && cnn->config.residual_mode) {
				clSetKernelArg(cnn->k_forward_residual, 0, sizeof(cl_mem), &current);
				clSetKernelArg(cnn->k_forward_residual, 1, sizeof(cl_mem), &cnn->input_buf);
				clSetKernelArg(cnn->k_forward_residual, 2, sizeof(cl_mem), &cnn->residual_buf);
				clSetKernelArg(cnn->k_forward_residual, 3, sizeof(cl_mem), &l->weights);
				clSetKernelArg(cnn->k_forward_residual, 4, sizeof(cl_mem), &l->bias);
				clSetKernelArg(cnn->k_forward_residual, 5, sizeof(int), &l->cin4);
				clSetKernelArg(cnn->k_forward_residual, 6, sizeof(int), &l->cout);
				clSetKernelArg(cnn->k_forward_residual, 7, sizeof(int), &l->h);
				clSetKernelArg(cnn->k_forward_residual, 8, sizeof(int), &l->w);

				size_t global[3] = {l->w, l->h, (l->cout + 3) / 4};
				size_t local[3] = {16, 8, 1};
				clEnqueueNDRangeKernel(cnn->queue, cnn->k_forward_residual, 3, NULL, global, local, 0, NULL, NULL);

				current = cnn->residual_buf;
			} else {
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

		} else if (l->type == LAYER_RESIDUAL_INPUT) {
			/* Residual input layer - save input and pass it through */
			int buffer_size = l->cout * l->h * l->w;

			/* Copy input to saved buffer */
			clSetKernelArg(cnn->k_copy_buffer, 0, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_copy_buffer, 1, sizeof(cl_mem), &l->residual_saved);
			clSetKernelArg(cnn->k_copy_buffer, 2, sizeof(int), &buffer_size);

			size_t global_copy = (buffer_size + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_copy_buffer, 1, NULL, &global_copy, NULL, 0, NULL, NULL);

			/* Copy input to output (pass through) */
			clSetKernelArg(cnn->k_copy_buffer, 0, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_copy_buffer, 1, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_copy_buffer, 2, sizeof(int), &buffer_size);

			clEnqueueNDRangeKernel(cnn->queue, cnn->k_copy_buffer, 1, NULL, &global_copy, NULL, 0, NULL, NULL);

			current = l->output;

		} else if (l->type == LAYER_RESIDUAL_SUBTRACT) {
			/* Residual subtract layer - compute (saved_input - current) */
			int buffer_size = l->cout * l->h * l->w;

			/* Get the saved input from the referenced layer */
			cl_mem saved_input;
			if (l->residual_from >= 0 && l->residual_from < i) {
				/* Validate that the referenced layer is RESIDUAL_INPUT */
				if (cnn->layers[l->residual_from].type != LAYER_RESIDUAL_INPUT) {
					fprintf(stderr, "Error: Layer %d (%s) references layer %d for residual, but that layer is not RESIDUAL_INPUT\n",
							i, l->name, l->residual_from);
					saved_input = cnn->input_buf; /* Fallback to network input */
				} else {
					saved_input = cnn->layers[l->residual_from].residual_saved;
				}
			} else {
				/* Use network input if no specific layer referenced */
				saved_input = cnn->input_buf;
			}

			/* Compute: output = saved_input - current (denoised = input - noise) */
			clSetKernelArg(cnn->k_residual_subtract, 0, sizeof(cl_mem), &saved_input);
			clSetKernelArg(cnn->k_residual_subtract, 1, sizeof(cl_mem), &current);
			clSetKernelArg(cnn->k_residual_subtract, 2, sizeof(cl_mem), &l->output);
			clSetKernelArg(cnn->k_residual_subtract, 3, sizeof(int), &buffer_size);

			size_t global_sub = (buffer_size + 255) / 256 * 256;
			clEnqueueNDRangeKernel(cnn->queue, cnn->k_residual_subtract, 1, NULL, &global_sub, NULL, 0, NULL, NULL);

			current = l->output;
		}
	}

	/* Read output (residual already computed if in residual mode) */
	if (cnn->config.residual_mode) {
		clEnqueueReadBuffer(cnn->queue, cnn->residual_buf, CL_TRUE, 0, input_size * 4, denoised_output, 0, NULL, NULL);
	} else {
		clEnqueueReadBuffer(cnn->queue, current, CL_TRUE, 0, input_size * 4, denoised_output, 0, NULL, NULL);
	}

	return 0;
}

/* Build list of all folder names on initialization */
void fillDataLoader(DataLoader *loader, char *folder_path) {
	strncpy(loader->folder_path, folder_path, 511);
	loader->folder_path[511] = '\0';
	loader->current_index = 0;

	printf("DataLoader initialized with path: %s\n", folder_path);
	printf("Images will be loaded on-demand from random folders\n");
}

/* Load a random image pair from a random folder */
void getNextImagePair(DataLoader *loader, ImageSample *sample) {
	DIR *dir = opendir(loader->folder_path);
	if (!dir) {
		fprintf(stderr, "Failed to open directory: %s\n", loader->folder_path);
		return;
	}

	/* Count total folders */
	struct dirent *entry;
	int total_folders = 0;
	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
			total_folders++;
		}
	}

	if (total_folders == 0) {
		fprintf(stderr, "No subdirectories found\n");
		closedir(dir);
		return;
	}

	/* Pick a random folder index */
	int target_idx = rand() % total_folders;
	rewinddir(dir);

	/* Find the selected folder */
	int current_idx = 0;
	char folder_name[256] = {0};
	while ((entry = readdir(dir)) != NULL) {
		if (entry->d_type != DT_DIR || strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
			continue;
		}
		if (current_idx == target_idx) {
			strncpy(folder_name, entry->d_name, 255);
			break;
		}
		current_idx++;
	}
	closedir(dir);

	if (folder_name[0] == '\0') {
		fprintf(stderr, "Failed to select random folder\n");
		return;
	}

	/* Load images from the selected folder */
	char path_buffer[512];
	char path_lum_buffer[512];

	/* Load noisy (low_res) images */
	snprintf(path_buffer, sizeof(path_buffer), "%s/%s/low_res.png", loader->folder_path, folder_name);
	snprintf(path_lum_buffer, sizeof(path_lum_buffer), "%s/%s/low_res_luminance.png", loader->folder_path, folder_name);

	int width, height, channels;
	unsigned char *noisy_rgb = stbi_load(path_buffer, &width, &height, &channels, 3);
	unsigned char *noisy_lum = stbi_load(path_lum_buffer, &width, &height, &channels, 1);

	static int first_load = 1;
	if (first_load) {
		if (noisy_rgb && noisy_lum) {
			printf("  Successfully loaded images: %dx%d from folder '%s'\n", width, height, folder_name);
			printf("  Sample RGB pixel values: R=%d G=%d B=%d\n", noisy_rgb[0], noisy_rgb[1], noisy_rgb[2]);
			printf("  Sample Luminance value: %d\n", noisy_lum[0]);
		} else {
			printf("  Failed to load images from folder '%s', using dummy data\n", folder_name);
		}
		first_load = 0;
	}

	if (!noisy_rgb || !noisy_lum) {
		fprintf(stderr, "Failed to load noisy images from folder: %s\n", folder_name);
		if (noisy_rgb) stbi_image_free(noisy_rgb);
		if (noisy_lum) stbi_image_free(noisy_lum);

		/* Fallback to dummy data */
		noisy_rgb = malloc(800 * 600 * 3);
		noisy_lum = malloc(800 * 600);
		memset(noisy_rgb, 128, 800 * 600 * 3);
		memset(noisy_lum, 128, 800 * 600);
		cnn_load_rgba_luminance(noisy_rgb, noisy_lum, sample->lowRes, 800, 600);
		free(noisy_rgb);
		free(noisy_lum);
	} else {
		cnn_load_rgba_luminance(noisy_rgb, noisy_lum, sample->lowRes, width, height);
		stbi_image_free(noisy_rgb);
		stbi_image_free(noisy_lum);
	}

	/* Load clean (high_res) images */
	snprintf(path_buffer, sizeof(path_buffer), "%s/%s/high_res.png", loader->folder_path, folder_name);
	snprintf(path_lum_buffer, sizeof(path_lum_buffer), "%s/%s/high_res_luminance.png", loader->folder_path, folder_name);

	unsigned char *clean_rgb = stbi_load(path_buffer, &width, &height, &channels, 3);
	unsigned char *clean_lum = stbi_load(path_lum_buffer, &width, &height, &channels, 1);

	/* Debug: Check if loaded images have color */
	static int debug_once = 1;
	if (debug_once && clean_rgb) {
		printf("\nDEBUG - Loaded clean image from: %s\n", path_buffer);
		printf("  First pixel RGB values: R=%d G=%d B=%d\n", clean_rgb[0], clean_rgb[1], clean_rgb[2]);
		printf("  Pixel 100 RGB values: R=%d G=%d B=%d\n", clean_rgb[300], clean_rgb[301], clean_rgb[302]);
		if (clean_rgb[0] == clean_rgb[1] && clean_rgb[1] == clean_rgb[2]) {
			printf("  WARNING: Clean image appears to be GRAYSCALE!\n");
		} else {
			printf("  Clean image has color variation.\n");
		}
		debug_once = 0;
	}

	if (!clean_rgb || !clean_lum) {
		fprintf(stderr, "Failed to load clean images from folder: %s\n", folder_name);
		if (clean_rgb) stbi_image_free(clean_rgb);
		if (clean_lum) stbi_image_free(clean_lum);

		/* Fallback to dummy data */
		clean_rgb = malloc(800 * 600 * 3);
		clean_lum = malloc(800 * 600);
		memset(clean_rgb, 128, 800 * 600 * 3);
		memset(clean_lum, 128, 800 * 600);
		cnn_load_rgba_luminance(clean_rgb, clean_lum, sample->highRes, 800, 600);
		free(clean_rgb);
		free(clean_lum);
	} else {
		cnn_load_rgba_luminance(clean_rgb, clean_lum, sample->highRes, width, height);
		stbi_image_free(clean_rgb);
		stbi_image_free(clean_lum);
	}

	loader->current_index++;
}

static const char base64_table[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t rgb_to_base64_noalloc(
	const unsigned char *data,
	size_t len,
	char *out) {
	size_t i = 0, j = 0;

	while (i < len) {
		uint32_t octet_a = i < len ? data[i++] : 0;
		uint32_t octet_b = i < len ? data[i++] : 0;
		uint32_t octet_c = i < len ? data[i++] : 0;

		uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;

		out[j++] = base64_table[(triple >> 18) & 0x3F];
		out[j++] = base64_table[(triple >> 12) & 0x3F];
		out[j++] = (i > len + 1) ? '=' : base64_table[(triple >> 6) & 0x3F];
		out[j++] = (i > len) ? '=' : base64_table[triple & 0x3F];
	}

	out[j] = '\0';
	return j;
}

void imageToBase64_noalloc(
	const float *image,
	int width,
	int height,
	unsigned char *rgb_buffer,
	char *base64_buffer) {
	int pixel_count = width * height;

	for (int i = 0; i < pixel_count; i++) {
		int rgba_idx = i * 4;
		int rgb_idx = i * 3;

		rgb_buffer[rgb_idx + 0] =
			(unsigned char)(fminf(fmaxf(image[rgba_idx + 0], 0.0f), 1.0f) * 255.0f);
		rgb_buffer[rgb_idx + 1] =
			(unsigned char)(fminf(fmaxf(image[rgba_idx + 1], 0.0f), 1.0f) * 255.0f);
		rgb_buffer[rgb_idx + 2] =
			(unsigned char)(fminf(fmaxf(image[rgba_idx + 2], 0.0f), 1.0f) * 255.0f);
	}

	rgb_to_base64_noalloc(
		rgb_buffer,
		(size_t)pixel_count * 3,
		base64_buffer);
}

void planarToInterleaved(
	const float *planar,
	float *interleaved,
	int width,
	int height,
	int channels) {
	int hw = height * width;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int pixel_idx = y * width + x;
			for (int c = 0; c < channels; c++) {
				interleaved[pixel_idx * channels + c] = planar[c * hw + pixel_idx];
			}
		}
	}
}

void interleavedToPlanar(
	const float *interleaved,
	float *planar,
	int width,
	int height,
	int channels) {
	int hw = height * width;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int pixel_idx = y * width + x;
			for (int c = 0; c < channels; c++) {
				planar[c * hw + pixel_idx] = interleaved[pixel_idx * channels + c];
			}
		}
	}
}

int send_images_to_python(
	const char *url,
	const char *input_img_b64,
	const char *original_img_b64,
	const char *prediction_img_b64,
	int step) {
	CURL *curl = curl_easy_init();
	if (!curl) return 0;

	struct curl_slist *headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");

	/* Estimate JSON size once (no reallocs) */
	size_t json_size =
		strlen(input_img_b64) +
		strlen(original_img_b64) +
		strlen(prediction_img_b64) +
		256;

	char *json = malloc(json_size);
	if (!json) {
		curl_easy_cleanup(curl);
		return 0;
	}

	snprintf(
		json,
		json_size,
		"{"
		"\"input_img\":\"%s\","
		"\"original_img\":\"%s\","
		"\"prediction_img\":\"%s\","
		"\"step\":%d"
		"}",
		input_img_b64,
		original_img_b64,
		prediction_img_b64,
		step);

	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(json));

	CURLcode res = curl_easy_perform(curl);

	free(json);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);

	return res == CURLE_OK;
}

void cnn_get_individual_losses(
	CNNDenoiser *cnn,
	float *mae_loss,
	float *mse_loss,
	float *laplace_loss,
	float *color_loss,
	float *ssim_loss,
	float *sobel_loss) {
	if (mae_loss) *mae_loss = cnn->last_mae_loss;
	if (mse_loss) *mse_loss = cnn->last_mse_loss;
	if (laplace_loss) *laplace_loss = cnn->last_laplace_loss;
	if (color_loss) *color_loss = cnn->last_color_loss;
	if (ssim_loss) *ssim_loss = cnn->last_ssim_loss;
	if (sobel_loss) *sobel_loss = cnn->last_sobel_loss;
}

void send_metadata_to_python(
	const char *url,
	int step,
	float loss,
	float learning_rate,
	float timeTookms,
	float forward_time,
	float mae_loss,
	float mse_loss,
	float color_loss,
	float laplacian_loss,
	float ssim_loss,
	float sobel_loss) {
	CURL *curl = curl_easy_init();
	if (!curl) return;

	struct curl_slist *headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");

	char json[640];
	snprintf(
		json,
		sizeof(json),
		"{"
		"\"step\":%d,"
		"\"loss\":%.6f,"
		"\"learning_rate\":%.6f,"
		"\"time\":%.6f,"
		"\"forward_time\":%.6f,"
		"\"mae_loss\":%.6f,"
		"\"mse_loss\":%.6f,"
		"\"color_loss\":%.6f,"
		"\"laplacian_loss\":%.6f,"
		"\"ssim_loss\":%.6f,"
		"\"sobel_loss\":%.6f"
		"}",
		step,
		loss,
		learning_rate,
		timeTookms,
		forward_time,
		mae_loss,
		mse_loss,
		color_loss,
		laplacian_loss,
		ssim_loss,
		sobel_loss);

	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(json));

	curl_easy_perform(curl);

	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
}