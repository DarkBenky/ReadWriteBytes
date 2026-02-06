/* batch_kernels.cl - OpenCL kernels for efficient batch training
 * Memory layout: [batch][channel][height][width] for optimal cache utilization
 */

/* Atomic add for floats using compare-and-swap (OpenCL 1.2 compatible)
 * Required because atom_add doesn't support floats in all OpenCL implementations
 */
inline void atomic_add_float(__global volatile float *addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr, 
                                      expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

/* Batch forward pass - process multiple samples in parallel
 * Adds batch dimension to the standard conv3x3 forward kernel
 * Each work item processes one pixel of one output channel for one batch sample
 */
__kernel void batch_conv3x3_forward_relu_f4(
    __global const float* input,        /* [batch][cin][h][w] planar */
    __global float* output,              /* [batch][cout][h][w] planar */
    __global const float4* weights,      /* [cout][cin4][9] */
    __global const float* bias,          /* [cout] */
    int batch_size, int Cin4, int Cout, int H, int W, int use_relu)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    /* Decode batch and output channel group from flattened 3rd dimension */
    int cout_groups = (Cout + 3) / 4;
    int batch = z / cout_groups;
    int oc = (z % cout_groups) * 4;  /* Process 4 output channels */
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int hw = H * W;
    int batch_offset = batch * Cin4 * 4 * hw;  /* Offset for this batch sample */
    int out_batch_offset = batch * Cout * hw;
    
    float sum0 = (oc < Cout) ? bias[oc] : 0.0f;
    float sum1 = (oc + 1 < Cout) ? bias[oc + 1] : 0.0f;
    float sum2 = (oc + 2 < Cout) ? bias[oc + 2] : 0.0f;
    float sum3 = (oc + 3 < Cout) ? bias[oc + 3] : 0.0f;
    
    for (int ic4 = 0; ic4 < Cin4; ic4++) {
        /* Clamp coordinates for replicate padding */
        int y0 = max(y - 1, 0), y1 = y, y2 = min(y + 1, H - 1);
        int x0 = max(x - 1, 0), x1 = x, x2 = min(x + 1, W - 1);
        
        /* Read 4 channels at each position from planar layout */
        #define READ_PIXEL(py, px) (float4)( \
            input[batch_offset + (ic4*4+0)*hw + (py)*W + (px)], \
            input[batch_offset + (ic4*4+1)*hw + (py)*W + (px)], \
            input[batch_offset + (ic4*4+2)*hw + (py)*W + (px)], \
            input[batch_offset + (ic4*4+3)*hw + (py)*W + (px)])
        
        float4 i0 = READ_PIXEL(y0, x0);
        float4 i1 = READ_PIXEL(y0, x1);
        float4 i2 = READ_PIXEL(y0, x2);
        float4 i3 = READ_PIXEL(y1, x0);
        float4 i4 = READ_PIXEL(y1, x1);
        float4 i5 = READ_PIXEL(y1, x2);
        float4 i6 = READ_PIXEL(y2, x0);
        float4 i7 = READ_PIXEL(y2, x1);
        float4 i8 = READ_PIXEL(y2, x2);
        #undef READ_PIXEL
        
        if (oc < Cout) {
            int wb = (oc * Cin4 + ic4) * 9;
            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];
            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];
            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];
            sum0 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + 
                    dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);
        }
        if (oc + 1 < Cout) {
            int wb = ((oc+1) * Cin4 + ic4) * 9;
            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];
            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];
            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];
            sum1 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + 
                    dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);
        }
        if (oc + 2 < Cout) {
            int wb = ((oc+2) * Cin4 + ic4) * 9;
            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];
            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];
            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];
            sum2 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + 
                    dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);
        }
        if (oc + 3 < Cout) {
            int wb = ((oc+3) * Cin4 + ic4) * 9;
            float4 w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2];
            float4 w3=weights[wb+3], w4=weights[wb+4], w5=weights[wb+5];
            float4 w6=weights[wb+6], w7=weights[wb+7], w8=weights[wb+8];
            sum3 += dot(i0,w0) + dot(i1,w1) + dot(i2,w2) + dot(i3,w3) + 
                    dot(i4,w4) + dot(i5,w5) + dot(i6,w6) + dot(i7,w7) + dot(i8,w8);
        }
    }
    
    /* Apply ReLU only if use_relu is set */
    if (oc < Cout) 
        output[out_batch_offset + oc * hw + y * W + x] = use_relu ? fmax(sum0, 0.0f) : sum0;
    if (oc + 1 < Cout) 
        output[out_batch_offset + (oc + 1) * hw + y * W + x] = use_relu ? fmax(sum1, 0.0f) : sum1;
    if (oc + 2 < Cout) 
        output[out_batch_offset + (oc + 2) * hw + y * W + x] = use_relu ? fmax(sum2, 0.0f) : sum2;
    if (oc + 3 < Cout) 
        output[out_batch_offset + (oc + 3) * hw + y * W + x] = use_relu ? fmax(sum3, 0.0f) : sum3;
}

/* Batch MAE loss computation - PASS 1: Compute per-pixel loss and gradient
 * No atomics, just write to buffer
 */
__kernel void batch_mae_loss_gradient(
    __global const float* prediction,    /* [batch][channels][h][w] */
    __global const float* target,        /* [batch][channels][h][w] */
    __global float* grad_out,            /* [batch][channels][h][w] */
    __global float* loss_buffer,         /* [batch * size_per_image] - temp buffer */
    int batch_size, int size_per_image)
{
    int batch = get_global_id(0);
    int idx = get_global_id(1);
    
    if (batch >= batch_size || idx >= size_per_image) return;
    
    int global_idx = batch * size_per_image + idx;
    int pixels_per_channel = size_per_image / 4;
    int channel = idx / pixels_per_channel;
    
    /* Only compute loss for RGB channels (0,1,2), skip luminance (3) */
    if (channel < 3) {
        float diff = prediction[global_idx] - target[global_idx];
        grad_out[global_idx] = copysign(1.0f, diff);
        loss_buffer[global_idx] = fabs(diff);
    } else {
        grad_out[global_idx] = 0.0f;
        loss_buffer[global_idx] = 0.0f;
    }
}

/* Batch loss reduction - PASS 2: Sum losses per batch
 * Efficient reduction with local memory
 */
__kernel void batch_loss_reduce(
    __global const float* loss_buffer,   /* [batch * size_per_image] */
    __global float* loss_per_batch,      /* [batch] - output */
    int batch_size,
    int size_per_image,
    __local float* local_sum)
{
    int batch = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    if (batch >= batch_size) return;
    
    int batch_offset = batch * size_per_image;
    
    /* Each work item sums a portion */
    float sum = 0.0f;
    for (int i = lid; i < size_per_image; i += local_size) {
        sum += loss_buffer[batch_offset + i];
    }
    local_sum[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Tree reduction in local memory */
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_sum[lid] += local_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    /* Write result */
    if (lid == 0) {
        loss_per_batch[batch] = local_sum[0];
    }
}

/* Batch backward pass - compute input gradients for all batch samples
 * Accumulates gradients using atomic operations
 */
__kernel void batch_conv3x3_backward_input_f4(
    __global const float* grad_out,      /* [batch][cout][h][w] */
    __global const float* output,        /* [batch][cout][h][w] */
    __global const float4* weights,      /* [cout][cin4][9] */
    __global float* grad_in,             /* [batch][cin][h][w] - output, accumulate */
    int batch_size, int Cin4, int Cout, int H, int W, int use_relu)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    /* Decode batch and input channel group from flattened 3rd dimension */
    int ic4 = z % Cin4;
    int batch = z / Cin4;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int hw = H * W;
    int out_batch_offset = batch * Cout * hw;
    int in_batch_offset = batch * Cin4 * 4 * hw;
    
    float4 acc = (float4)(0.0f);
    
    for (int oc = 0; oc < Cout; oc++) {
        int oidx = out_batch_offset + oc * hw + y * W + x;
        float g = grad_out[oidx];
        
        /* Apply ReLU gradient */
        if (use_relu && output[oidx] <= 0.0f) g = 0.0f;
        if (g == 0.0f) continue;
        
        /* Accumulate weighted gradients */
        int w_base = (oc * Cin4 + ic4) * 9;
        float4 w_sum = weights[w_base] + weights[w_base+1] + weights[w_base+2] +
                       weights[w_base+3] + weights[w_base+4] + weights[w_base+5] +
                       weights[w_base+6] + weights[w_base+7] + weights[w_base+8];
        acc += w_sum * g;
    }
    
    /* Write to planar layout with accumulation (for skip connections) */
    int pixel_idx = y * W + x;
    grad_in[in_batch_offset + (ic4*4 + 0)*hw + pixel_idx] += acc.s0;
    grad_in[in_batch_offset + (ic4*4 + 1)*hw + pixel_idx] += acc.s1;
    grad_in[in_batch_offset + (ic4*4 + 2)*hw + pixel_idx] += acc.s2;
    grad_in[in_batch_offset + (ic4*4 + 3)*hw + pixel_idx] += acc.s3;
}

/* Batch weight gradient computation - accumulate across all batch samples
 * This is where batch efficiency comes from: compute gradients once for all samples
 */
__kernel void batch_weight_grad_reduce(
    __global const float* input,         /* [batch][cin][h][w] */
    __global const float* grad_out,      /* [batch][cout][h][w] */
    __global const float* output,        /* [batch][cout][h][w] */
    __global float4* grad_w_vec,         /* [cout][cin4][9] - output, accumulate */
    __global float* grad_b,              /* [cout] - output, accumulate */
    int batch_size, int Cin4, int Cout, int H, int W, int use_relu)
{
    int oc = get_global_id(0);
    int ic4 = get_global_id(1);
    int k = get_global_id(2);
    
    if (oc >= Cout) return;
    
    int hw = H * W;
    int dy = (k / 3) - 1;
    int dx = (k % 3) - 1;
    
    float4 sum = (float4)(0.0f);
    float bias_sum = 0.0f;
    
    /* Accumulate gradients across all batch samples */
    for (int batch = 0; batch < batch_size; batch++) {
        int out_batch_offset = batch * Cout * hw;
        int in_batch_offset = batch * Cin4 * 4 * hw;
        
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int oidx = out_batch_offset + oc * hw + y * W + x;
                float g = grad_out[oidx];
                
                /* Apply ReLU gradient */
                if (use_relu && output[oidx] <= 0.0f) g = 0.0f;
                if (g != 0.0f) {
                    /* Clamp input coordinates */
                    int iy = clamp(y + dy, 0, H - 1);
                    int ix = clamp(x + dx, 0, W - 1);
                    int pixel_idx = iy * W + ix;
                    
                    /* Read 4 channels from planar layout */
                    float4 input_val = (float4)(
                        input[in_batch_offset + (ic4*4+0)*hw + pixel_idx],
                        input[in_batch_offset + (ic4*4+1)*hw + pixel_idx],
                        input[in_batch_offset + (ic4*4+2)*hw + pixel_idx],
                        input[in_batch_offset + (ic4*4+3)*hw + pixel_idx]);
                    
                    sum = fma(input_val, (float4)(g), sum);
                    
                    if (ic4 == 0 && k == 0) bias_sum += g;
                }
            }
        }
    }
    
    /* Average gradients across batch */
    float batch_scale = 1.0f / (float)batch_size;
    grad_w_vec[(oc * Cin4 + ic4) * 9 + k] = sum * batch_scale;
    if (ic4 == 0 && k == 0) grad_b[oc] = bias_sum * batch_scale;
}

/* Batch MSE loss - similar to MAE but with squared error */
__kernel void batch_mse_loss_gradient(
    __global const float* prediction,
    __global const float* target,
    __global float* grad_out,
    __global float* loss_per_batch,
    int batch_size, int size_per_image)
{
    int batch = get_global_id(0);
    int idx = get_global_id(1);
    
    if (batch >= batch_size || idx >= size_per_image) return;
    
    int global_idx = batch * size_per_image + idx;
    int pixels_per_channel = size_per_image / 4;
    int channel = idx / pixels_per_channel;
    
    if (channel < 3) {
        float diff = prediction[global_idx] - target[global_idx];
        grad_out[global_idx] = 2.0f * diff;
        loss_per_batch[global_idx] = diff * diff;
    } else {
        grad_out[global_idx] = 0.0f;
        loss_per_batch[global_idx] = 0.0f;
    }
}

/* Helper: accumulate weighted gradients from multiple loss functions
 * grad_total = grad_total + weight * grad_component
 */
__kernel void batch_add_weighted_gradient(
    __global float* grad_total,
    __global const float* grad_component,
    float weight,
    int total_size)
{
    int idx = get_global_id(0);
    if (idx < total_size) {
        grad_total[idx] += weight * grad_component[idx];
    }
}

/* Clear batch loss accumulator before reduction */
__kernel void batch_clear_loss_buffer(
    __global float* loss_per_batch,
    int batch_size)
{
    int idx = get_global_id(0);
    if (idx < batch_size) {
        loss_per_batch[idx] = 0.0f;
    }
}

/* Batch Laplace loss gradient - edge-preserving loss
 * Computes Laplacian operator on both output and target, compares them
 * Only operates on RGB channels (first 3), skips luminance (channel 4)
 */
__kernel void batch_laplace_loss_gradient(
    __global const float* output,        /* [batch][C][H][W] */
    __global const float* target,        /* [batch][C][H][W] */
    __global float* grad_out,            /* [batch][C][H][W] */
    __global float* loss_buffer,         /* [batch * C * H * W] loss per pixel */
    int batch_size,
    int H, int W, int C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int batch = get_global_id(2) / C;
    int c = get_global_id(2) % C;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int img_size = C * H * W;
    int channel_size = H * W;
    int global_idx = batch * img_size + c * channel_size + y * W + x;
    
    /* Skip luminance channel (channel 3), only process RGB (0,1,2) */
    if (c >= 3) {
        grad_out[global_idx] = 0.0f;
        loss_buffer[global_idx] = 0.0f;
        return;
    }
    
    /* Laplacian only computed for interior pixels */
    if (x > 0 && y > 0 && x < W-1 && y < H-1) {
        int idx = global_idx;
        
        /* Compute Laplacian: -4*center + left + right + up + down */
        float lap_out = -4.0f * output[idx] +
                        output[idx - 1] + output[idx + 1] +
                        output[idx - W] + output[idx + W];
        
        float lap_tgt = -4.0f * target[idx] +
                        target[idx - 1] + target[idx + 1] +
                        target[idx - W] + target[idx + W];
        
        float diff = lap_out - lap_tgt;
        
        /* MAE-style gradient for Laplacian */
        grad_out[idx] = (diff > 0.0f) ? 1.0f : -1.0f;
        
        /* Store loss per pixel for reduction */
        loss_buffer[idx] = fabs(diff);
    } else {
        /* Zero gradient and loss for border pixels */
        grad_out[global_idx] = 0.0f;
        loss_buffer[global_idx] = 0.0f;
    }
}

/* Batch Color Variance loss - color direction and saturation preservation
 * Only operates on RGB channels (first 3), skips luminance
 */
__kernel void batch_color_variance_loss(
    __global const float* output,        /* [batch][C][H][W] */
    __global const float* target,        /* [batch][C][H][W] */
    __global float* grad_out,            /* [batch][C][H][W] */
    __global float* loss_per_batch,      /* [batch] accumulator */
    int batch_size,
    int H, int W)
{
    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);
    int batch = get_global_id(2);
    
    if (pixel_x >= W || pixel_y >= H || batch >= batch_size) return;
    
    int pixels = H * W;
    int pixel_idx = pixel_y * W + pixel_x;
    int img_size = 4 * pixels;  /* 4 channels: RGB + luminance */
    int batch_offset = batch * img_size;
    
    /* Read RGB values (channels 0, 1, 2) */
    float out_r = output[batch_offset + pixel_idx];
    float out_g = output[batch_offset + pixels + pixel_idx];
    float out_b = output[batch_offset + 2 * pixels + pixel_idx];
    float tgt_r = target[batch_offset + pixel_idx];
    float tgt_g = target[batch_offset + pixels + pixel_idx];
    float tgt_b = target[batch_offset + 2 * pixels + pixel_idx];
    
    /* Direction loss: 1 - dot(normalize(pred), normalize(target)) */
    float pred_norm = sqrt(out_r*out_r + out_g*out_g + out_b*out_b) + 1e-6f;
    float tgt_norm = sqrt(tgt_r*tgt_r + tgt_g*tgt_g + tgt_b*tgt_b) + 1e-6f;
    
    float pred_r_n = out_r / pred_norm;
    float pred_g_n = out_g / pred_norm;
    float pred_b_n = out_b / pred_norm;
    float tgt_r_n = tgt_r / tgt_norm;
    float tgt_g_n = tgt_g / tgt_norm;
    float tgt_b_n = tgt_b / tgt_norm;
    
    float dot = pred_r_n * tgt_r_n + pred_g_n * tgt_g_n + pred_b_n * tgt_b_n;
    float direction_loss = 1.0f - dot;
    
    /* Saturation loss: max(0, target_std - pred_std) */
    float pred_mean = (out_r + out_g + out_b) / 3.0f;
    float pred_var = ((out_r - pred_mean)*(out_r - pred_mean) +
                     (out_g - pred_mean)*(out_g - pred_mean) +
                     (out_b - pred_mean)*(out_b - pred_mean)) / 3.0f;
    float pred_std = sqrt(pred_var + 1e-8f);
    
    float tgt_mean = (tgt_r + tgt_g + tgt_b) / 3.0f;
    float tgt_var = ((tgt_r - tgt_mean)*(tgt_r - tgt_mean) +
                    (tgt_g - tgt_mean)*(tgt_g - tgt_mean) +
                    (tgt_b - tgt_mean)*(tgt_b - tgt_mean)) / 3.0f;
    float tgt_std = sqrt(tgt_var + 1e-8f);
    
    float sat_diff = fmax(0.0f, tgt_std - pred_std);
    float saturation_loss = sat_diff;
    
    /* Combine losses with non-linear terms for stability */
    float dir_term = direction_loss + 1.0f;
    float dir_penalty = dir_term * dir_term;  /* squared */
    
    float sat_term = saturation_loss + 1.0f;
    float sat_penalty = sat_term * sat_term * sat_term * sat_term;  /* ^4 */
    
    float total_loss = 2.0f * dir_penalty + 8.0f * sat_penalty;
    
    /* Write per-pixel loss to buffer at channel 0 position (pixel-level loss, not per-channel) */
    int loss_idx = batch * img_size + pixel_y * W + pixel_x;  /* Channel 0 position */
    loss_per_batch[loss_idx] = total_loss;
    
    /* Zero out other channel positions to avoid counting them in reduction */
    loss_per_batch[batch * img_size + pixels + pixel_y * W + pixel_x] = 0.0f;  /* Channel 1 */
    loss_per_batch[batch * img_size + 2 * pixels + pixel_y * W + pixel_x] = 0.0f;  /* Channel 2 */
    loss_per_batch[batch * img_size + 3 * pixels + pixel_y * W + pixel_x] = 0.0f;  /* Channel 3 */
    
    /* Compute gradients */
    float inv_pred_norm = 1.0f / pred_norm;
    
    /* Direction gradient scaled by 2 * 2 * (dir + 1) */
    float dir_grad_scale = 2.0f * 2.0f * dir_term;
    float grad_dir_r = dir_grad_scale * (-(tgt_r_n * inv_pred_norm - pred_r_n * dot * inv_pred_norm));
    float grad_dir_g = dir_grad_scale * (-(tgt_g_n * inv_pred_norm - pred_g_n * dot * inv_pred_norm));
    float grad_dir_b = dir_grad_scale * (-(tgt_b_n * inv_pred_norm - pred_b_n * dot * inv_pred_norm));
    
    /* Saturation gradient */
    float grad_sat_r = 0.0f, grad_sat_g = 0.0f, grad_sat_b = 0.0f;
    
    if (sat_diff > 0.0f) {
        float sat_weight = 8.0f * 4.0f * sat_term * sat_term * sat_term;  /* 8 * 4 * (sat+1)^3 */
        float inv_pred_std = -1.0f / (pred_std + 1e-8f);
        float factor = inv_pred_std / 3.0f;
        grad_sat_r = sat_weight * factor * (out_r - pred_mean);
        grad_sat_g = sat_weight * factor * (out_g - pred_mean);
        grad_sat_b = sat_weight * factor * (out_b - pred_mean);
    }
    
    /* Write gradients for RGB channels */
    grad_out[batch_offset + pixel_idx] = grad_dir_r + grad_sat_r;
    grad_out[batch_offset + pixels + pixel_idx] = grad_dir_g + grad_sat_g;
    grad_out[batch_offset + 2 * pixels + pixel_idx] = grad_dir_b + grad_sat_b;
    
    /* Zero gradient for luminance channel */
    grad_out[batch_offset + 3 * pixels + pixel_idx] = 0.0f;
}

/* Batch SSIM Loss - Structural Similarity Index Measure
 * Computes SSIM in 7x7 windows, simplified gradient approximation
 * Only processes RGB channels (0,1,2), skips luminance (3)
 */
/* Batch Sobel Gradient loss - edge-preserving loss for sharp reconstructions
 * Computes Sobel gradients in X and Y directions and compares with target
 * Only operates on RGB channels (first 3), skips luminance
 */
__kernel void batch_sobel_loss_gradient(
    __global const float* output,        /* [batch][C][H][W] */
    __global const float* target,        /* [batch][C][H][W] */
    __global float* grad_out,            /* [batch][C][H][W] */
    __global float* loss_per_batch,      /* [batch] accumulator */
    int batch_size,
    int H, int W, int C)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int batch = get_global_id(2) / C;
    int c = get_global_id(2) % C;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    /* Skip luminance channel (channel 3), only process RGB (0,1,2) */
    if (c >= 3) {
        int img_size = C * H * W;
        int channel_size = H * W;
        int global_idx = batch * img_size + c * channel_size + y * W + x;
        grad_out[global_idx] = 0.0f;
        return;
    }
    
    /* Sobel only computed for interior pixels (need 1-pixel border) */
    if (x > 0 && y > 0 && x < W-1 && y < H-1) {
        int img_size = C * H * W;
        int channel_size = H * W;
        int idx = batch * img_size + c * channel_size + y * W + x;
        
        /* Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] */
        float sobel_x_out = -output[idx - W - 1] - 2.0f * output[idx - 1] - output[idx + W - 1]
                          + output[idx - W + 1] + 2.0f * output[idx + 1] + output[idx + W + 1];
        
        float sobel_x_tgt = -target[idx - W - 1] - 2.0f * target[idx - 1] - target[idx + W - 1]
                          + target[idx - W + 1] + 2.0f * target[idx + 1] + target[idx + W + 1];
        
        /* Sobel Y kernel: [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]] */
        float sobel_y_out = -output[idx - W - 1] - 2.0f * output[idx - W] - output[idx - W + 1]
                          + output[idx + W - 1] + 2.0f * output[idx + W] + output[idx + W + 1];
        
        float sobel_y_tgt = -target[idx - W - 1] - 2.0f * target[idx - W] - target[idx - W + 1]
                          + target[idx + W - 1] + 2.0f * target[idx + W] + target[idx + W + 1];
        
        float diff_x = sobel_x_out - sobel_x_tgt;
        float diff_y = sobel_y_out - sobel_y_tgt;
        
        /* MAE-style gradient for both directions */
        float grad_x = (diff_x > 0.0f) ? 1.0f : -1.0f;
        float grad_y = (diff_y > 0.0f) ? 1.0f : -1.0f;
        
        /* Combine gradients (simplified - derivatives affect multiple neighbors) */
        grad_out[idx] = grad_x + grad_y;
        
        /* Accumulate L1 loss from both gradient directions */
        loss_per_batch[idx] = fabs(diff_x) + fabs(diff_y);
    } else {
        /* Zero gradient and loss for border pixels */
        int img_size = C * H * W;
        int channel_size = H * W;
        int global_idx = batch * img_size + c * channel_size + y * W + x;
        grad_out[global_idx] = 0.0f;
        loss_per_batch[global_idx] = 0.0f;
    }
}

__kernel void batch_ssim_loss_gradient(
    __global const float* prediction,
    __global const float* target,
    __global float* grad_out,
    __global float* loss_buffer,
    int batch_size, int H, int W)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int batch = get_global_id(2);
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int pixels = H * W;
    int batch_offset = batch * 4 * pixels;
    int pixel_idx = y * W + x;
    
    const int window = 3;
    const float c1 = 0.0001f;
    const float c2 = 0.0009f;
    
    float total_loss = 0.0f;
    
    for (int c = 0; c < 3; c++) {
        int channel_offset = batch_offset + c * pixels;
        
        float sum_p = 0.0f, sum_t = 0.0f;
        float sum_pp = 0.0f, sum_tt = 0.0f, sum_pt = 0.0f;
        int count = 0;
        
        for (int dy = -window; dy <= window; dy++) {
            for (int dx = -window; dx <= window; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                    int idx = ny * W + nx;
                    float p = prediction[channel_offset + idx];
                    float t = target[channel_offset + idx];
                    sum_p += p;
                    sum_t += t;
                    sum_pp += p * p;
                    sum_tt += t * t;
                    sum_pt += p * t;
                    count++;
                }
            }
        }
        
        float mean_p = sum_p / count;
        float mean_t = sum_t / count;
        float var_p = sum_pp / count - mean_p * mean_p;
        float var_t = sum_tt / count - mean_t * mean_t;
        float cov_pt = sum_pt / count - mean_p * mean_t;
        
        float luminance = (2.0f * mean_p * mean_t + c1) / (mean_p * mean_p + mean_t * mean_t + c1);
        float contrast = (2.0f * sqrt(fmax(var_p, 0.0f)) * sqrt(fmax(var_t, 0.0f)) + c2) / (var_p + var_t + c2);
        float structure = (cov_pt + c2/2.0f) / (sqrt(fmax(var_p, 0.0f)) * sqrt(fmax(var_t, 0.0f)) + c2/2.0f);
        
        float ssim = luminance * contrast * structure;
        float loss = 1.0f - ssim;
        total_loss += loss;
        
        float p_val = prediction[channel_offset + pixel_idx];
        float t_val = target[channel_offset + pixel_idx];
        float grad_scale = 2.0f * (1.0f - ssim);
        
        grad_out[channel_offset + pixel_idx] = grad_scale * (p_val - t_val);
    }
    
    int lum_offset = batch_offset + 3 * pixels;
    grad_out[lum_offset + pixel_idx] = 0.0f;
    
    loss_buffer[batch_offset + pixel_idx] = total_loss / 3.0f;
}

/* ========== RESIDUAL LAYER SUPPORT ========== */

/* Batch residual input layer - save input for later subtraction
 * Simply copies input to both output and saved buffer
 */
__kernel void batch_residual_input(
    __global const float* input,         /* [batch][channels][h][w] */
    __global float* output,              /* [batch][channels][h][w] - pass through */
    __global float* saved,               /* [batch][channels][h][w] - saved for later */
    int batch_size, int channels, int H, int W)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);  /* batch * channels combined */
    
    int hw = H * W;
    int batch = z / channels;
    int c = z % channels;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int idx = batch * channels * hw + c * hw + y * W + x;
    
    /* Pass through and save */
    float val = input[idx];
    output[idx] = val;
    saved[idx] = val;
}

/* Batch residual subtract layer - compute (saved_input - current_input)
 * Typically used as: denoised = input - noise_prediction
 */
__kernel void batch_residual_subtract(
    __global const float* saved_input,   /* [batch][channels][h][w] - from RESIDUAL_INPUT layer */
    __global const float* noise_pred,    /* [batch][channels][h][w] - current layer input */
    __global float* output,              /* [batch][channels][h][w] - denoised result */
    int batch_size, int channels, int H, int W)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);  /* batch * channels combined */
    
    int hw = H * W;
    int batch = z / channels;
    int c = z % channels;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int idx = batch * channels * hw + c * hw + y * W + x;
    
    /* Subtract noise from saved input */
    output[idx] = saved_input[idx] - noise_pred[idx];
}

/* Batch backward for residual input - pass gradient through */
__kernel void batch_residual_input_backward(
    __global const float* grad_out,      /* [batch][channels][h][w] */
    __global float* grad_in,             /* [batch][channels][h][w] - accumulate */
    int batch_size, int channels, int H, int W)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int hw = H * W;
    int batch = z / channels;
    int c = z % channels;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int idx = batch * channels * hw + c * hw + y * W + x;
    
    /* Pass gradient through (identity operation) */
    grad_in[idx] = grad_out[idx];
}

/* Batch backward for residual subtract
 * d_loss/d_saved_input = d_loss/d_output * 1 (positive gradient)
 * d_loss/d_noise = d_loss/d_output * (-1) (negated gradient)
 */
__kernel void batch_residual_subtract_backward(
    __global const float* grad_out,          /* [batch][channels][h][w] - gradient from next layer */
    __global float* grad_saved,              /* [batch][channels][h][w] - gradient to saved input layer */
    __global float* grad_noise,              /* [batch][channels][h][w] - gradient to noise prediction */
    int batch_size, int channels, int H, int W)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    
    int hw = H * W;
    int batch = z / channels;
    int c = z % channels;
    
    if (x >= W || y >= H || batch >= batch_size) return;
    
    int idx = batch * channels * hw + c * hw + y * W + x;
    
    float grad = grad_out[idx];
    
    /* Gradient w.r.t. saved input (positive) */
    grad_saved[idx] = grad;
    
    /* Gradient w.r.t. noise prediction (negated) */
    grad_noise[idx] = -grad;
}

