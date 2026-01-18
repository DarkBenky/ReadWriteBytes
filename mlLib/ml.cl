
inline void atomic_add_f(__global float* addr, float val) {
    union {
        unsigned int intVal;
        float floatVal;
    } old, newVal;

    do {
        old.floatVal = *addr;
        newVal.floatVal = old.floatVal + val;
    } while (atomic_cmpxchg((__global unsigned int*)addr, old.intVal, newVal.intVal) != old.intVal);
}

float dotProduct(__global float* a, __global float* b, int size) {
    float result = 0.0f;
    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// Optimized convolution using loop unrolling and reduced branching
// For stride=1, kernel=3, with proper padding
__kernel void convForward3x3(
    __global float* restrict inputData,
    __global float* restrict kernelData,
    __global float* restrict outputData,
    __global float* restrict biasData,
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int outputHeight,
    int outputWidth,
    int outputChannels
    ) {
    
    int outY = get_global_id(0);
    int outX = get_global_id(1);
    int outC = get_global_id(2);

    if (outY >= outputHeight || outX >= outputWidth || outC >= outputChannels) return;

    float sum = biasData[outC];
    
    int inY_base = outY - 1;  // stride=1, padding=1
    int inX_base = outX - 1;
    
    // Manual unroll for 3x3 kernel - reduce loop overhead
    for (int inC = 0; inC < inputChannels; inC++) {
        __global float* kPtr = kernelData + (inC * outputChannels + outC);
        int kStride = inputChannels * outputChannels;
        
        #pragma unroll
        for (int ky = 0; ky < 3; ky++) {
            int inY = inY_base + ky;
            if (inY >= 0 && inY < inputHeight) {
                __global float* inRow = inputData + (inY * inputWidth + inX_base) * inputChannels + inC;
                int inStride = inputChannels;
                
                #pragma unroll
                for (int kx = 0; kx < 3; kx++) {
                    int inX = inX_base + kx;
                    if (inX >= 0 && inX < inputWidth) {
                        sum += inRow[kx * inStride] * kPtr[(ky * 3 + kx) * kStride];
                    }
                }
            }
        }
    }

    outputData[(outY * outputWidth + outX) * outputChannels + outC] = sum;
}

// Vectorized ReLU - process 4 elements at once
__kernel void reluForward4(
    __global float4* inputData,
    __global float4* outputData,
    int dataSize4  // dataSize / 4
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize4) return;

    float4 val = inputData[idx];
    outputData[idx] = fmax(val, (float4)(0.0f));
}

__kernel void convForward(
    __global float* inputData, // inputHeight * inputWidth * inputChannels
    __global float* kernelData, // kernelSize * kernelSize * inputChannels * outputChannels
    __global float* outputData, // outputHeight * outputWidth * outputChannels
    __global float* biasData, // outputChannels
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int kernelSize,
    int outputHeight,
    int outputWidth,
    int outputChannels, // number of filters / kernels
    int stride,
    int paddingHeight,
    int paddingWidth
    ) {
    
    int outY = get_global_id(0);
    int outX = get_global_id(1);
    int outC = get_global_id(2);

    if (outY >= outputHeight || outX >= outputWidth || outC >= outputChannels) {
        return;
    }

    float sum = biasData[outC];
    
    int inY_base = outY * stride - paddingHeight;
    int inX_base = outX * stride - paddingWidth;

    // Precompute bounds to avoid conditionals in inner loop
    int kY_start = max(0, -inY_base);
    int kY_end = min(kernelSize, inputHeight - inY_base);
    int kX_start = max(0, -inX_base);
    int kX_end = min(kernelSize, inputWidth - inX_base);

    for (int inC = 0; inC < inputChannels; inC++) {
        for (int kY = kY_start; kY < kY_end; kY++) {
            int inY = inY_base + kY;
            __global float* inputRow = inputData + (inY * inputWidth + inX_base) * inputChannels + inC;
            __global float* kernelRow = kernelData + ((kY * kernelSize + kX_start) * inputChannels + inC) * outputChannels + outC;
            
            for (int kX = kX_start; kX < kX_end; kX++) {
                sum += inputRow[kX * inputChannels] * kernelRow[(kX - kX_start) * inputChannels * outputChannels];
            }
        }
    }

    outputData[(outY * outputWidth + outX) * outputChannels + outC] = sum;
}

__kernel void convBackward(
    __global float* inputData, // inputHeight * inputWidth * inputChannels
    __global float* kernelData, // kernelSize * kernelSize * inputChannels * outputChannels
    __global float* outputGradData, // outputHeight * outputWidth * outputChannels
    __global float* inputGradData, // inputHeight * inputWidth * inputChannels
    __global float* kernelGradData, // kernelSize * kernelSize * inputChannels * outputChannels
    __global float* biasGradData, // outputChannels
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int kernelSize,
    int outputHeight,
    int outputWidth,
    int outputChannels, // number of filters / kernels
    int stride,
    int paddingHeight,
    int paddingWidth
    ) {
    
    int kY = get_global_id(0);
    int kX = get_global_id(1);
    int inC = get_global_id(2);
    int outC = get_global_id(3);

    if (kY >= kernelSize || kX >= kernelSize || inC >= inputChannels || outC >= outputChannels) {
        return;
    }

    float kernelGrad = 0.0f;
    float biasGrad = 0.0f;

    // Compute kernel gradient and bias gradient
    for (int outY = 0; outY < outputHeight; outY++) {
        for (int outX = 0; outX < outputWidth; outX++) {
            int inY = outY * stride + kY - paddingHeight;
            int inX = outX * stride + kX - paddingWidth;

            if (inY >= 0 && inY < inputHeight && inX >= 0 && inX < inputWidth) {
                float inputValue = inputData[(inY * inputWidth + inX) * inputChannels + inC];
                float outputGradValue = outputGradData[(outY * outputWidth + outX) * outputChannels + outC];
                kernelGrad += inputValue * outputGradValue;
                biasGrad += outputGradValue;
            }
        }
    }

    kernelGradData[((kY * kernelSize + kX) * inputChannels + inC) * outputChannels + outC] = kernelGrad;
    
    // Accumulate bias gradient (only for one kernel position to avoid duplication)
    if (kY == 0 && kX == 0 && inC == 0) {
        atomic_add_f(&biasGradData[outC], biasGrad);
    }
}

// Transposed convolution (deconvolution) for upscaling
__kernel void deconvForward(
    __global float* inputData,  // inputHeight * inputWidth * inputChannels
    __global float* kernelData, // kernelSize * kernelSize * inputChannels * outputChannels
    __global float* outputData, // outputHeight * outputWidth * outputChannels
    __global float* biasData,   // outputChannels
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int kernelSize,
    int outputHeight,
    int outputWidth,
    int outputChannels,
    int stride,
    int paddingHeight,
    int paddingWidth
    ) {
    
    int outY = get_global_id(0);
    int outX = get_global_id(1);
    int outC = get_global_id(2);
    
    if (outY >= outputHeight || outX >= outputWidth || outC >= outputChannels) {
        return;
    }
    
    float sum = biasData[outC];
    
    // Optimized: precompute bounds and pointer offsets
    __global float* kernelBase = kernelData + outC;
    int kY_start = max(0, (paddingHeight - outY + stride - 1) / stride);
    int kY_end = min(kernelSize, (paddingHeight + inputHeight - outY + stride - 1) / stride);
    int kX_start = max(0, (paddingWidth - outX + stride - 1) / stride);
    int kX_end = min(kernelSize, (paddingWidth + inputWidth - outX + stride - 1) / stride);
    
    for (int inC = 0; inC < inputChannels; inC++) {
        for (int kY = kY_start; kY < kY_end; kY++) {
            int offsetY = outY + paddingHeight - kY;
            if (offsetY % stride == 0) {
                int inY = offsetY / stride;
                __global float* inputRow = inputData + (inY * inputWidth) * inputChannels + inC;
                __global float* kernelRow = kernelBase + ((kY * kernelSize) * inputChannels + inC) * outputChannels;
                
                for (int kX = kX_start; kX < kX_end; kX++) {
                    int offsetX = outX + paddingWidth - kX;
                    if (offsetX % stride == 0) {
                        int inX = offsetX / stride;
                        sum += inputRow[inX * inputChannels] * kernelRow[kX * inputChannels * outputChannels];
                    }
                }
            }
        }
    }
    
    outputData[(outY * outputWidth + outX) * outputChannels + outC] = sum;
}

__kernel void deconvBackward(
    __global float* inputData,      // inputHeight * inputWidth * inputChannels
    __global float* kernelData,     // kernelSize * kernelSize * inputChannels * outputChannels
    __global float* outputGradData, // outputHeight * outputWidth * outputChannels
    __global float* inputGradData,  // inputHeight * inputWidth * inputChannels
    __global float* kernelGradData, // kernelSize * kernelSize * inputChannels * outputChannels
    __global float* biasGradData,   // outputChannels
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int kernelSize,
    int outputHeight,
    int outputWidth,
    int outputChannels,
    int stride,
    int paddingHeight,
    int paddingWidth
    ) {
    
    int kY = get_global_id(0);
    int kX = get_global_id(1);
    int inC = get_global_id(2);
    int outC = get_global_id(3);
    
    if (kY >= kernelSize || kX >= kernelSize || inC >= inputChannels || outC >= outputChannels) {
        return;
    }
    
    float kernelGrad = 0.0f;
    float biasGrad = 0.0f;
    
    for (int inY = 0; inY < inputHeight; inY++) {
        for (int inX = 0; inX < inputWidth; inX++) {
            int outY = inY * stride - paddingHeight + kY;
            int outX = inX * stride - paddingWidth + kX;
            
            if (outY >= 0 && outY < outputHeight && outX >= 0 && outX < outputWidth) {
                float inputValue = inputData[(inY * inputWidth + inX) * inputChannels + inC];
                float outputGradValue = outputGradData[(outY * outputWidth + outX) * outputChannels + outC];
                kernelGrad += inputValue * outputGradValue;
                biasGrad += outputGradValue;
            }
        }
    }
    
    kernelGradData[((kY * kernelSize + kX) * inputChannels + inC) * outputChannels + outC] = kernelGrad;
    
    if (kY == 0 && kX == 0 && inC == 0) {
        atomic_add_f(&biasGradData[outC], biasGrad);
    }
}

__kernel void maxPoolingForward(
    __global float* inputData, // inputHeight * inputWidth * inputChannels
    __global float* outputData, // outputHeight * outputWidth * inputChannels
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int poolSize,
    int stride,
    int outputHeight,
    int outputWidth
    ) {
    
    int outY = get_global_id(0);
    int outX = get_global_id(1);
    int inC = get_global_id(2);

    if (outY >= outputHeight || outX >= outputWidth || inC >= inputChannels) {
        return;
    }

    float maxVal = -FLT_MAX;

    for (int pY = 0; pY < poolSize; pY++) {
        for (int pX = 0; pX < poolSize; pX++) {
            int inY = outY * stride + pY;
            int inX = outX * stride + pX;

            if (inY < inputHeight && inX < inputWidth) {
                float inputValue = inputData[(inY * inputWidth + inX) * inputChannels + inC];
                if (inputValue > maxVal) {
                    maxVal = inputValue;
                }
            }
        }
    }

    outputData[(outY * outputWidth + outX) * inputChannels + inC] = maxVal;
}

__kernel void maxPoolingBackward(
    __global float* inputData, // inputHeight * inputWidth * inputChannels
    __global float* outputData, // outputHeight * outputWidth * inputChannels
    __global float* outputGradData, // outputHeight * outputWidth * inputChannels
    __global float* inputGradData, // inputHeight * inputWidth * inputChannels
    int inputHeight,
    int inputWidth,
    int inputChannels,
    int poolSize,
    int stride,
    int outputHeight,
    int outputWidth
    ) {
    
    int outY = get_global_id(0);
    int outX = get_global_id(1);
    int inC = get_global_id(2);

    if (outY >= outputHeight || outX >= outputWidth || inC >= inputChannels) {
        return;
    }

    float maxVal = -FLT_MAX;
    int maxInY = -1;
    int maxInX = -1;

    // Find the position of the max value in the pooling window
    for (int pY = 0; pY < poolSize; pY++) {
        for (int pX = 0; pX < poolSize; pX++) {
            int inY = outY * stride + pY;
            int inX = outX * stride + pX;

            if (inY < inputHeight && inX < inputWidth) {
                float inputValue = inputData[(inY * inputWidth + inX) * inputChannels + inC];
                if (inputValue > maxVal) {
                    maxVal = inputValue;
                    maxInY = inY;
                    maxInX = inX;
                }
            }
        }
    }

    // Propagate the gradient to the position of the max value
    if (maxInY != -1 && maxInX != -1) {
        atomic_add_f(&inputGradData[(maxInY * inputWidth + maxInX) * inputChannels + inC],
                     outputGradData[(outY * outputWidth + outX) * inputChannels + inC]);
    }
}

__kernel void reluForward(
    __global float* inputData,
    __global float* outputData,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize) {
        return;
    }

    float val = inputData[idx];
    outputData[idx] = val > 0.0f ? val : 0.0f;
}

__kernel void reluBackward(
    __global float* inputData,
    __global float* outputGradData,
    __global float* inputGradData,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize) {
        return;
    }

    float inputVal = inputData[idx];
    float gradVal = outputGradData[idx];
    inputGradData[idx] = inputVal > 0.0f ? gradVal : 0.0f;
}

__kernel void softmaxForward(
    __global float* inputData,
    __global float* outputData,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx != 0) {
        return; // Only first work item processes softmax
    }

    // Compute max for numerical stability
    float maxVal = inputData[0];
    for (int i = 1; i < dataSize; i++) {
        if (inputData[i] > maxVal) {
            maxVal = inputData[i];
        }
    }

    // Compute exponentials and sum
    float sumExp = 0.0f;
    for (int i = 0; i < dataSize; i++) {
        float expVal = exp(inputData[i] - maxVal);
        outputData[i] = expVal;
        sumExp += expVal;
    }

    // Normalize to get probabilities
    for (int i = 0; i < dataSize; i++) {
        outputData[i] /= sumExp;
    }
}

__kernel void softmaxBackward(
    __global float* outputData,
    __global float* outputGradData,
    __global float* inputGradData,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize) {
        return;
    }

    float softmaxVal = outputData[idx];
    float gradVal = outputGradData[idx];

    // Compute gradient for softmax
    inputGradData[idx] = softmaxVal * (gradVal - dotProduct(outputData, outputGradData, dataSize));
}

__kernel void meanAbsoluteError(
    __global float* predictions,
    __global float* targets,
    __global float* lossOutput,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize) {
        return;
    }

    float error = fabs(predictions[idx] - targets[idx]);
    atomic_add_f(lossOutput, error / dataSize);
}

__kernel void meanSquaredError(
    __global float* predictions,
    __global float* targets,
    __global float* lossOutput,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize) {
        return;
    }

    float error = predictions[idx] - targets[idx];
    atomic_add_f(lossOutput, (error * error) / dataSize);
}

__kernel void ssimLoss(
    __global float* img1,
    __global float* img2,
    __global float* lossOutput,
    int width,
    int height,
    int channels
    ) {
    
    // Each work item processes one 8x8 window
    int windowIdx = get_global_id(0);
    int windowSize = 8;
    int windowsPerRow = (width + windowSize - 1) / windowSize;
    int windowsPerCol = (height + windowSize - 1) / windowSize;
    int totalWindows = windowsPerRow * windowsPerCol * channels;
    
    if (windowIdx >= totalWindows) {
        return;
    }
    
    int c = windowIdx / (windowsPerRow * windowsPerCol);
    int windowPos = windowIdx % (windowsPerRow * windowsPerCol);
    int winY = (windowPos / windowsPerRow) * windowSize;
    int winX = (windowPos % windowsPerRow) * windowSize;
    
    // SSIM parameters
    const float C1 = 0.0001f;
    const float C2 = 0.0009f;
    
    // Compute means over window
    float mu1 = 0.0f, mu2 = 0.0f;
    int count = 0;
    
    for (int dy = 0; dy < windowSize && winY + dy < height; dy++) {
        for (int dx = 0; dx < windowSize && winX + dx < width; dx++) {
            int pixelIdx = ((winY + dy) * width + (winX + dx)) * channels + c;
            mu1 += img1[pixelIdx];
            mu2 += img2[pixelIdx];
            count++;
        }
    }
    
    if (count == 0) return;
    mu1 /= count;
    mu2 /= count;
    
    // Compute variances and covariance
    float sigma1_sq = 0.0f, sigma2_sq = 0.0f, sigma12 = 0.0f;
    
    for (int dy = 0; dy < windowSize && winY + dy < height; dy++) {
        for (int dx = 0; dx < windowSize && winX + dx < width; dx++) {
            int pixelIdx = ((winY + dy) * width + (winX + dx)) * channels + c;
            float diff1 = img1[pixelIdx] - mu1;
            float diff2 = img2[pixelIdx] - mu2;
            sigma1_sq += diff1 * diff1;
            sigma2_sq += diff2 * diff2;
            sigma12 += diff1 * diff2;
        }
    }
    
    sigma1_sq /= count;
    sigma2_sq /= count;
    sigma12 /= count;
    
    // Compute SSIM for this window
    float ssim_numerator = (2.0f * mu1 * mu2 + C1) * (2.0f * sigma12 + C2);
    float ssim_denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);
    float ssim = ssim_numerator / ssim_denominator;
    
    atomic_add_f(lossOutput, (1.0f - ssim) / totalWindows);
}

// Bit reversal for FFT
int bitReverse(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// 1D FFT using Cooley-Tukey (single work-item per row/col)
__kernel void fft1DKernel(
    __global float2* data,
    int n,
    int log2n,
    int inverse
    ) {
    
    int rowIdx = get_global_id(0);
    __global float2* row = data + rowIdx * n;
    
    // Bit-reversal permutation
    for (int i = 0; i < n; i++) {
        int j = bitReverse(i, log2n);
        if (i < j) {
            float2 temp = row[i];
            row[i] = row[j];
            row[j] = temp;
        }
    }
    
    // Cooley-Tukey iterative FFT
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        float angle = (inverse ? M_PI_F : -M_PI_F) / m2;
        float2 wm = (float2)(cos(angle), sin(angle));
        
        for (int k = 0; k < n; k += m) {
            float2 w = (float2)(1.0f, 0.0f);
            for (int j = 0; j < m2; j++) {
                int idx1 = k + j;
                int idx2 = k + j + m2;
                
                float2 t = (float2)(w.x * row[idx2].x - w.y * row[idx2].y,
                                    w.x * row[idx2].y + w.y * row[idx2].x);
                float2 u = row[idx1];
                
                row[idx1] = u + t;
                row[idx2] = u - t;
                
                float2 newW = (float2)(w.x * wm.x - w.y * wm.y, w.x * wm.y + w.y * wm.x);
                w = newW;
            }
        }
    }
    
    // Normalize for inverse FFT
    if (inverse) {
        for (int i = 0; i < n; i++) {
            row[i] /= (float)n;
        }
    }
}

// Transpose for 2D FFT
__kernel void transposeKernel(
    __global float2* input,
    __global float2* output,
    int width,
    int height
    ) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Gradient-based frequency loss (Laplacian for high-frequency content)
// Works with any image dimensions, no power-of-2 requirement
__kernel void gradientLoss(
    __global float* img1,
    __global float* img2,
    __global float* lossOutput,
    int width,
    int height,
    int channels
    ) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    
    // Skip border pixels
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1 || c >= channels) {
        return;
    }
    
    int idx = (y * width + x) * channels + c;
    int idxLeft = (y * width + (x - 1)) * channels + c;
    int idxRight = (y * width + (x + 1)) * channels + c;
    int idxUp = ((y - 1) * width + x) * channels + c;
    int idxDown = ((y + 1) * width + x) * channels + c;
    
    // Laplacian (high-frequency detector)
    float lap1 = 4.0f * img1[idx] - img1[idxLeft] - img1[idxRight] - img1[idxUp] - img1[idxDown];
    float lap2 = 4.0f * img2[idx] - img2[idxLeft] - img2[idxRight] - img2[idxUp] - img2[idxDown];
    
    float diff = lap1 - lap2;
    float error = diff * diff;
    
    atomic_add_f(lossOutput, error / ((width - 2) * (height - 2) * channels));
}

__kernel void fftLoss(
    __global float2* predictionsFFT,
    __global float2* targetsFFT,
    __global float* lossOutput,
    int dataSize
    ) {
    
    int idx = get_global_id(0);
    if (idx >= dataSize) {
        return;
    }

    float2 pred = predictionsFFT[idx];
    float2 target = targetsFFT[idx];

    float realDiff = pred.x - target.x;
    float imagDiff = pred.y - target.y;

    float error = realDiff * realDiff + imagDiff * imagDiff;
    atomic_add_f(lossOutput, error / dataSize);
}
__kernel void resize2DForward(
    __global float* inputData,
    __global float* outputData,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int channels
    ) {
    
    int outX = get_global_id(0);
    int outY = get_global_id(1);
    int c = get_global_id(2);

    if (outX >= outputWidth || outY >= outputHeight || c >= channels) {
        return;
    }

    // Bilinear interpolation for better quality
    float scaleX = (float)(inputWidth - 1) / (outputWidth - 1);
    float scaleY = (float)(inputHeight - 1) / (outputHeight - 1);
    
    float srcX = outX * scaleX;
    float srcY = outY * scaleY;
    
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = min(x0 + 1, inputWidth - 1);
    int y1 = min(y0 + 1, inputHeight - 1);
    
    float dx = srcX - x0;
    float dy = srcY - y0;
    
    float v00 = inputData[(y0 * inputWidth + x0) * channels + c];
    float v10 = inputData[(y0 * inputWidth + x1) * channels + c];
    float v01 = inputData[(y1 * inputWidth + x0) * channels + c];
    float v11 = inputData[(y1 * inputWidth + x1) * channels + c];
    
    float result = v00 * (1.0f - dx) * (1.0f - dy) +
                   v10 * dx * (1.0f - dy) +
                   v01 * (1.0f - dx) * dy +
                   v11 * dx * dy;

    outputData[(outY * outputWidth + outX) * channels + c] = result;
}

__kernel void resize2DBackward(
    __global float* outputGradData,
    __global float* inputGradData,
    int inputWidth,
    int inputHeight,
    int outputWidth,
    int outputHeight,
    int channels
    ) {
    
    int outX = get_global_id(0);
    int outY = get_global_id(1);
    int c = get_global_id(2);

    if (outX >= outputWidth || outY >= outputHeight || c >= channels) {
        return;
    }

    // Bilinear backward: distribute gradient to 4 source pixels
    float scaleX = (float)(inputWidth - 1) / (outputWidth - 1);
    float scaleY = (float)(inputHeight - 1) / (outputHeight - 1);
    
    float srcX = outX * scaleX;
    float srcY = outY * scaleY;
    
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = min(x0 + 1, inputWidth - 1);
    int y1 = min(y0 + 1, inputHeight - 1);
    
    float dx = srcX - x0;
    float dy = srcY - y0;
    
    float grad = outputGradData[(outY * outputWidth + outX) * channels + c];
    
    atomic_add_f(&inputGradData[(y0 * inputWidth + x0) * channels + c], grad * (1.0f - dx) * (1.0f - dy));
    atomic_add_f(&inputGradData[(y0 * inputWidth + x1) * channels + c], grad * dx * (1.0f - dy));
    atomic_add_f(&inputGradData[(y1 * inputWidth + x0) * channels + c], grad * (1.0f - dx) * dy);
    atomic_add_f(&inputGradData[(y1 * inputWidth + x1) * channels + c], grad * dx * dy);
}

// ============================================================================
// Combined loss kernel: MAE + SSIM + FFT (gradient-based) in one pass
// Weights: MAE=0.1, SSIM=0.5, FFT/Gradient=0.4
// ============================================================================
__kernel void combinedLossAndGradient(
    __global float* predictions,
    __global float* targets,
    __global float* gradient,     // output gradient
    __global float* lossOutput,   // scalar loss output
    int width,
    int height,
    int channels,
    float maeWeight,
    float ssimWeight,
    float fftWeight
    ) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int c = get_global_id(2);
    
    if (x >= width || y >= height || c >= channels) return;
    
    int idx = (y * width + x) * channels + c;
    int dataSize = width * height * channels;
    
    float pred = predictions[idx];
    float target = targets[idx];
    float diff = pred - target;
    
    // MAE component
    float maeError = fabs(diff);
    float maeGrad = (diff > 0.0f ? 1.0f : -1.0f) * maeWeight / dataSize;
    
    atomic_add_f(lossOutput, maeWeight * maeError / dataSize);
    
    // Gradient/Laplacian component (edge preservation)
    float lapGrad = 0.0f;
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int idxL = (y * width + (x - 1)) * channels + c;
        int idxR = (y * width + (x + 1)) * channels + c;
        int idxU = ((y - 1) * width + x) * channels + c;
        int idxD = ((y + 1) * width + x) * channels + c;
        
        float lap1 = 4.0f * pred - predictions[idxL] - predictions[idxR] - predictions[idxU] - predictions[idxD];
        float lap2 = 4.0f * target - targets[idxL] - targets[idxR] - targets[idxU] - targets[idxD];
        float lapDiff = lap1 - lap2;
        
        float fftError = lapDiff * lapDiff;
        atomic_add_f(lossOutput, fftWeight * fftError / ((width - 2) * (height - 2) * channels));
        
        // Laplacian gradient approximation
        lapGrad = fftWeight * 2.0f * lapDiff * 4.0f / ((width - 2) * (height - 2) * channels);
    }
    
    gradient[idx] = maeGrad + lapGrad;
}

// SSIM loss computed separately with windowing (more accurate)
__kernel void ssimLossAndGradient(
    __global float* predictions,
    __global float* targets,
    __global float* gradient,
    __global float* lossOutput,
    int width,
    int height,
    int channels,
    float ssimWeight
    ) {
    
    int windowIdx = get_global_id(0);
    int windowSize = 8;
    int windowsPerRow = (width + windowSize - 1) / windowSize;
    int windowsPerCol = (height + windowSize - 1) / windowSize;
    int totalWindows = windowsPerRow * windowsPerCol * channels;
    
    if (windowIdx >= totalWindows) return;
    
    int c = windowIdx / (windowsPerRow * windowsPerCol);
    int windowPos = windowIdx % (windowsPerRow * windowsPerCol);
    int winY = (windowPos / windowsPerRow) * windowSize;
    int winX = (windowPos % windowsPerRow) * windowSize;
    
    const float C1 = 0.0001f;
    const float C2 = 0.0009f;
    
    // Compute means
    float mu1 = 0.0f, mu2 = 0.0f;
    int count = 0;
    
    for (int dy = 0; dy < windowSize && winY + dy < height; dy++) {
        for (int dx = 0; dx < windowSize && winX + dx < width; dx++) {
            int idx = ((winY + dy) * width + (winX + dx)) * channels + c;
            mu1 += predictions[idx];
            mu2 += targets[idx];
            count++;
        }
    }
    
    if (count == 0) return;
    mu1 /= count;
    mu2 /= count;
    
    // Compute variances and covariance
    float sigma1_sq = 0.0f, sigma2_sq = 0.0f, sigma12 = 0.0f;
    
    for (int dy = 0; dy < windowSize && winY + dy < height; dy++) {
        for (int dx = 0; dx < windowSize && winX + dx < width; dx++) {
            int idx = ((winY + dy) * width + (winX + dx)) * channels + c;
            float d1 = predictions[idx] - mu1;
            float d2 = targets[idx] - mu2;
            sigma1_sq += d1 * d1;
            sigma2_sq += d2 * d2;
            sigma12 += d1 * d2;
        }
    }
    
    sigma1_sq /= count;
    sigma2_sq /= count;
    sigma12 /= count;
    
    float A = 2.0f * mu1 * mu2 + C1;
    float B = 2.0f * sigma12 + C2;
    float C = mu1 * mu1 + mu2 * mu2 + C1;
    float D = sigma1_sq + sigma2_sq + C2;
    
    float ssim = (A * B) / (C * D);
    atomic_add_f(lossOutput, ssimWeight * (1.0f - ssim) / totalWindows);
    
    // Compute SSIM gradient for each pixel in window
    float denom = C * D;
    float denom_sq = denom * denom;
    
    for (int dy = 0; dy < windowSize && winY + dy < height; dy++) {
        for (int dx = 0; dx < windowSize && winX + dx < width; dx++) {
            int idx = ((winY + dy) * width + (winX + dx)) * channels + c;
            float pred = predictions[idx];
            float target = targets[idx];
            
            // dSSIM/d_pred approximation
            float dA_dp = 2.0f * mu2 / count;
            float dC_dp = 2.0f * mu1 / count;
            float dB_dp = 2.0f * (target - mu2) / count;
            float dD_dp = 2.0f * (pred - mu1) / count;
            
            float num = A * B;
            float dNum_dp = dA_dp * B + A * dB_dp;
            float dDenom_dp = dC_dp * D + C * dD_dp;
            
            float dSSIM_dp = (dNum_dp * denom - num * dDenom_dp) / denom_sq;
            
            // Gradient is negative because loss = 1 - SSIM
            atomic_add_f(&gradient[idx], -ssimWeight * dSSIM_dp / totalWindows);
        }
    }
}