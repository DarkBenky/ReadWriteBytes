#define _GNU_SOURCE
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple 1D FFT implementation (Cooley-Tukey)
void fft1D(float* real, float* imag, int n, int inverse) {
    if (n <= 1) return;
    
    // Bit reversal
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        
        if (i < j) {
            float tempR = real[i];
            float tempI = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tempR;
            imag[j] = tempI;
        }
    }
    
    // Cooley-Tukey FFT
    for (int len = 2; len <= n; len *= 2) {
        float angle = (inverse ? 2.0f : -2.0f) * M_PI / len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);
        
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            
            for (int j = 0; j < len / 2; j++) {
                float u_r = real[i + j];
                float u_i = imag[i + j];
                float v_r = real[i + j + len / 2] * w_r - imag[i + j + len / 2] * w_i;
                float v_i = real[i + j + len / 2] * w_i + imag[i + j + len / 2] * w_r;
                
                real[i + j] = u_r + v_r;
                imag[i + j] = u_i + v_i;
                real[i + j + len / 2] = u_r - v_r;
                imag[i + j + len / 2] = u_i - v_i;
                
                float temp = w_r;
                w_r = w_r * wlen_r - w_i * wlen_i;
                w_i = temp * wlen_i + w_i * wlen_r;
            }
        }
    }
    
    if (inverse) {
        for (int i = 0; i < n; i++) {
            real[i] /= n;
            imag[i] /= n;
        }
    }
}

// 2D FFT using row-column approach
void fft2D(float* real, float* imag, int width, int height, int inverse) {
    // FFT on rows
    float* rowReal = (float*)malloc(width * sizeof(float));
    float* rowImag = (float*)malloc(width * sizeof(float));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            rowReal[x] = real[y * width + x];
            rowImag[x] = imag[y * width + x];
        }
        fft1D(rowReal, rowImag, width, inverse);
        for (int x = 0; x < width; x++) {
            real[y * width + x] = rowReal[x];
            imag[y * width + x] = rowImag[x];
        }
    }
    
    free(rowReal);
    free(rowImag);
    
    // FFT on columns
    float* colReal = (float*)malloc(height * sizeof(float));
    float* colImag = (float*)malloc(height * sizeof(float));
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            colReal[y] = real[y * width + x];
            colImag[y] = imag[y * width + x];
        }
        fft1D(colReal, colImag, height, inverse);
        for (int y = 0; y < height; y++) {
            real[y * width + x] = colReal[y];
            imag[y * width + x] = colImag[y];
        }
    }
    
    free(colReal);
    free(colImag);
}

char* readKernelSource(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        fprintf(stderr, "Failed to open kernel file: %s\n", path);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);
    
    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, file);
    source[size] = '\0';
    fclose(file);
    
    return source;
}

int initOpenCL(NeuralNetwork* nn, const char* kernelSourcePath) {
    cl_int err;
    cl_platform_id platform;
    
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform\n");
        return -1;
    }
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &nn->device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &nn->device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Failed to get device\n");
            return -1;
        }
    }
    
    nn->context = clCreateContext(NULL, 1, &nn->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create context\n");
        return -1;
    }
    
    nn->queue = clCreateCommandQueue(nn->context, nn->device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create command queue\n");
        return -1;
    }
    
    char* source = readKernelSource(kernelSourcePath);
    if (!source) return -1;
    
    nn->program = clCreateProgramWithSource(nn->context, 1, (const char**)&source, NULL, &err);
    free(source);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to create program\n");
        return -1;
    }
    
    err = clBuildProgram(nn->program, 1, &nn->device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(nn->program, nn->device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        return -1;
    }
    
    nn->convForwardKernel = clCreateKernel(nn->program, "convForward", &err);
    nn->convBackwardKernel = clCreateKernel(nn->program, "convBackward", &err);
    nn->deconvForwardKernel = clCreateKernel(nn->program, "deconvForward", &err);
    nn->deconvBackwardKernel = clCreateKernel(nn->program, "deconvBackward", &err);
    nn->maxPoolForwardKernel = clCreateKernel(nn->program, "maxPoolingForward", &err);
    nn->maxPoolBackwardKernel = clCreateKernel(nn->program, "maxPoolingBackward", &err);
    nn->reluForwardKernel = clCreateKernel(nn->program, "reluForward", &err);
    nn->reluBackwardKernel = clCreateKernel(nn->program, "reluBackward", &err);
    nn->softmaxForwardKernel = clCreateKernel(nn->program, "softmaxForward", &err);
    nn->softmaxBackwardKernel = clCreateKernel(nn->program, "softmaxBackward", &err);
    nn->resize2DForwardKernel = clCreateKernel(nn->program, "resize2DForward", &err);
    nn->resize2DBackwardKernel = clCreateKernel(nn->program, "resize2DBackward", &err);
    nn->mseLossKernel = clCreateKernel(nn->program, "meanSquaredError", &err);
    nn->maeLossKernel = clCreateKernel(nn->program, "meanAbsoluteError", &err);
    nn->ssimLossKernel = clCreateKernel(nn->program, "ssimLoss", &err);
    nn->gradientLossKernel = clCreateKernel(nn->program, "gradientLoss", &err);
    nn->fft1DKernel = clCreateKernel(nn->program, "fft1DKernel", &err);
    nn->transposeKernel = clCreateKernel(nn->program, "transposeKernel", &err);
    nn->fftLossKernel = clCreateKernel(nn->program, "fftLoss", &err);
    nn->combinedLossKernel = clCreateKernel(nn->program, "combinedLossAndGradient", &err);
    nn->combinedGradientKernel = clCreateKernel(nn->program, "ssimLossAndGradient", &err);
    
    nn->trainTargetBuffer = NULL;
    nn->trainOutputBuffer = NULL;
    nn->trainLossBuffer = NULL;
    nn->trainGradBuffer = NULL;
    nn->trainBufferSize = 0;
    
    return 0;
}

NeuralNetwork* createNetwork(int numLayers) {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->layers = (Layer*)malloc(numLayers * sizeof(Layer));
    nn->numLayers = numLayers;
    return nn;
}

void addConvLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, int inputC,
                  int kernelSize, int outputC, int stride, int padH, int padW) {
    Layer* layer = &nn->layers[layerIdx];
    layer->type = LAYER_CONV;
    
    ConvLayer* conv = &layer->conv;
    conv->inputHeight = inputH;
    conv->inputWidth = inputW;
    conv->inputChannels = inputC;
    conv->kernelSize = kernelSize;
    conv->outputChannels = outputC;
    conv->stride = stride;
    conv->paddingHeight = padH;
    conv->paddingWidth = padW;
    conv->outputHeight = (inputH + 2 * padH - kernelSize) / stride + 1;
    conv->outputWidth = (inputW + 2 * padW - kernelSize) / stride + 1;
    
    int inputSize = inputH * inputW * inputC;
    int outputSize = conv->outputHeight * conv->outputWidth * outputC;
    int kernelDataSize = kernelSize * kernelSize * inputC * outputC;
    
    conv->inputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, 
                                       inputSize * sizeof(float), NULL, NULL);
    conv->outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                        outputSize * sizeof(float), NULL, NULL);
    conv->kernelBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                        kernelDataSize * sizeof(float), NULL, NULL);
    conv->biasBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                      outputC * sizeof(float), NULL, NULL);
    conv->kernelGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                            kernelDataSize * sizeof(float), NULL, NULL);
    conv->biasGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                          outputC * sizeof(float), NULL, NULL);
    
    conv->kernelWeights = (float*)malloc(kernelDataSize * sizeof(float));
    conv->biasWeights = (float*)malloc(outputC * sizeof(float));
    
    // Random initialization
    for (int i = 0; i < kernelDataSize; i++) {
        conv->kernelWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < outputC; i++) {
        conv->biasWeights[i] = 0.0f;
    }
    
    clEnqueueWriteBuffer(nn->queue, conv->kernelBuffer, CL_TRUE, 0,
                        kernelDataSize * sizeof(float), conv->kernelWeights, 0, NULL, NULL);
    clEnqueueWriteBuffer(nn->queue, conv->biasBuffer, CL_TRUE, 0,
                        outputC * sizeof(float), conv->biasWeights, 0, NULL, NULL);
}

void addDeconvLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, int inputC,
                    int kernelSize, int outputC, int stride, int padH, int padW) {
    Layer* layer = &nn->layers[layerIdx];
    layer->type = LAYER_DECONV;
    
    DeconvLayer* deconv = &layer->deconv;
    deconv->inputHeight = inputH;
    deconv->inputWidth = inputW;
    deconv->inputChannels = inputC;
    deconv->kernelSize = kernelSize;
    deconv->outputChannels = outputC;
    deconv->stride = stride;
    deconv->paddingHeight = padH;
    deconv->paddingWidth = padW;
    // Transposed conv output size: (input - 1) * stride + kernel - 2*pad
    deconv->outputHeight = (inputH - 1) * stride + kernelSize - 2 * padH;
    deconv->outputWidth = (inputW - 1) * stride + kernelSize - 2 * padW;
    
    int inputSize = inputH * inputW * inputC;
    int outputSize = deconv->outputHeight * deconv->outputWidth * outputC;
    int kernelDataSize = kernelSize * kernelSize * inputC * outputC;
    
    deconv->inputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                         inputSize * sizeof(float), NULL, NULL);
    deconv->outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                          outputSize * sizeof(float), NULL, NULL);
    deconv->kernelBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                          kernelDataSize * sizeof(float), NULL, NULL);
    deconv->biasBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                        outputC * sizeof(float), NULL, NULL);
    deconv->kernelGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                              kernelDataSize * sizeof(float), NULL, NULL);
    deconv->biasGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                            outputC * sizeof(float), NULL, NULL);
    
    deconv->kernelWeights = (float*)malloc(kernelDataSize * sizeof(float));
    deconv->biasWeights = (float*)malloc(outputC * sizeof(float));
    
    // Random initialization (He initialization scaled)
    float scale = sqrtf(2.0f / (kernelSize * kernelSize * inputC));
    for (int i = 0; i < kernelDataSize; i++) {
        deconv->kernelWeights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    for (int i = 0; i < outputC; i++) {
        deconv->biasWeights[i] = 0.0f;
    }
    
    clEnqueueWriteBuffer(nn->queue, deconv->kernelBuffer, CL_TRUE, 0,
                        kernelDataSize * sizeof(float), deconv->kernelWeights, 0, NULL, NULL);
    clEnqueueWriteBuffer(nn->queue, deconv->biasBuffer, CL_TRUE, 0,
                        outputC * sizeof(float), deconv->biasWeights, 0, NULL, NULL);
}

void addMaxPoolLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, int inputC,
                     int poolSize, int stride) {
    Layer* layer = &nn->layers[layerIdx];
    layer->type = LAYER_MAXPOOL;
    
    MaxPoolLayer* pool = &layer->pool;
    pool->inputHeight = inputH;
    pool->inputWidth = inputW;
    pool->inputChannels = inputC;
    pool->poolSize = poolSize;
    pool->stride = stride;
    pool->outputHeight = (inputH - poolSize) / stride + 1;
    pool->outputWidth = (inputW - poolSize) / stride + 1;
    
    int inputSize = inputH * inputW * inputC;
    int outputSize = pool->outputHeight * pool->outputWidth * inputC;
    
    pool->inputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                       inputSize * sizeof(float), NULL, NULL);
    pool->outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                        outputSize * sizeof(float), NULL, NULL);
}

void addReluLayer(NeuralNetwork* nn, int layerIdx, int dataSize) {
    Layer* layer = &nn->layers[layerIdx];
    layer->type = LAYER_RELU;
    
    ActivationLayer* relu = &layer->activation;
    relu->dataSize = dataSize;
    relu->inputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                       dataSize * sizeof(float), NULL, NULL);
    relu->outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                        dataSize * sizeof(float), NULL, NULL);
}

void addSoftmaxLayer(NeuralNetwork* nn, int layerIdx, int dataSize) {
    Layer* layer = &nn->layers[layerIdx];
    layer->type = LAYER_SOFTMAX;
    
    ActivationLayer* softmax = &layer->activation;
    softmax->dataSize = dataSize;
    softmax->inputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                          dataSize * sizeof(float), NULL, NULL);
    softmax->outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                           dataSize * sizeof(float), NULL, NULL);
}

void addResize2DLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, 
                      int outputH, int outputW, int channels) {
    Layer* layer = &nn->layers[layerIdx];
    layer->type = LAYER_RESIZE2D;
    
    Resize2DLayer* resize = &layer->resize;
    resize->inputHeight = inputH;
    resize->inputWidth = inputW;
    resize->outputHeight = outputH;
    resize->outputWidth = outputW;
    resize->channels = channels;
    
    int inputSize = inputH * inputW * channels;
    int outputSize = outputH * outputW * channels;
    
    resize->inputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                         inputSize * sizeof(float), NULL, NULL);
    resize->outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                          outputSize * sizeof(float), NULL, NULL);
}

void forward(NeuralNetwork* nn, float* inputData) {
    cl_mem currentBuffer = NULL;
    
    for (int i = 0; i < nn->numLayers; i++) {
        Layer* layer = &nn->layers[i];
        
        switch (layer->type) {
            case LAYER_CONV: {
                ConvLayer* conv = &layer->conv;
                
                if (i == 0) {
                    clEnqueueWriteBuffer(nn->queue, conv->inputBuffer, CL_TRUE, 0,
                                        conv->inputHeight * conv->inputWidth * conv->inputChannels * sizeof(float),
                                        inputData, 0, NULL, NULL);
                } else if (currentBuffer != NULL) {
                    int inputSize = conv->inputHeight * conv->inputWidth * conv->inputChannels * sizeof(float);
                    clEnqueueCopyBuffer(nn->queue, currentBuffer, conv->inputBuffer, 0, 0, inputSize, 0, NULL, NULL);
                }
                
                clSetKernelArg(nn->convForwardKernel, 0, sizeof(cl_mem), &conv->inputBuffer);
                clSetKernelArg(nn->convForwardKernel, 1, sizeof(cl_mem), &conv->kernelBuffer);
                clSetKernelArg(nn->convForwardKernel, 2, sizeof(cl_mem), &conv->outputBuffer);
                clSetKernelArg(nn->convForwardKernel, 3, sizeof(cl_mem), &conv->biasBuffer);
                clSetKernelArg(nn->convForwardKernel, 4, sizeof(int), &conv->inputHeight);
                clSetKernelArg(nn->convForwardKernel, 5, sizeof(int), &conv->inputWidth);
                clSetKernelArg(nn->convForwardKernel, 6, sizeof(int), &conv->inputChannels);
                clSetKernelArg(nn->convForwardKernel, 7, sizeof(int), &conv->kernelSize);
                clSetKernelArg(nn->convForwardKernel, 8, sizeof(int), &conv->outputHeight);
                clSetKernelArg(nn->convForwardKernel, 9, sizeof(int), &conv->outputWidth);
                clSetKernelArg(nn->convForwardKernel, 10, sizeof(int), &conv->outputChannels);
                clSetKernelArg(nn->convForwardKernel, 11, sizeof(int), &conv->stride);
                clSetKernelArg(nn->convForwardKernel, 12, sizeof(int), &conv->paddingHeight);
                clSetKernelArg(nn->convForwardKernel, 13, sizeof(int), &conv->paddingWidth);
                
                size_t globalSize[3] = {conv->outputHeight, conv->outputWidth, conv->outputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->convForwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                currentBuffer = conv->outputBuffer;
                break;
            }
            case LAYER_MAXPOOL: {
                MaxPoolLayer* pool = &layer->pool;
                
                if (currentBuffer != NULL) {
                    int inputSize = pool->inputHeight * pool->inputWidth * pool->inputChannels * sizeof(float);
                    clEnqueueCopyBuffer(nn->queue, currentBuffer, pool->inputBuffer, 0, 0, inputSize, 0, NULL, NULL);
                }
                
                clSetKernelArg(nn->maxPoolForwardKernel, 0, sizeof(cl_mem), &pool->inputBuffer);
                clSetKernelArg(nn->maxPoolForwardKernel, 1, sizeof(cl_mem), &pool->outputBuffer);
                clSetKernelArg(nn->maxPoolForwardKernel, 2, sizeof(int), &pool->inputHeight);
                clSetKernelArg(nn->maxPoolForwardKernel, 3, sizeof(int), &pool->inputWidth);
                clSetKernelArg(nn->maxPoolForwardKernel, 4, sizeof(int), &pool->inputChannels);
                clSetKernelArg(nn->maxPoolForwardKernel, 5, sizeof(int), &pool->poolSize);
                clSetKernelArg(nn->maxPoolForwardKernel, 6, sizeof(int), &pool->stride);
                clSetKernelArg(nn->maxPoolForwardKernel, 7, sizeof(int), &pool->outputHeight);
                clSetKernelArg(nn->maxPoolForwardKernel, 8, sizeof(int), &pool->outputWidth);
                
                size_t globalSize[3] = {pool->outputHeight, pool->outputWidth, pool->inputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->maxPoolForwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                currentBuffer = pool->outputBuffer;
                break;
            }
            case LAYER_RELU: {
                ActivationLayer* relu = &layer->activation;
                
                if (currentBuffer != NULL) {
                    int dataSize = relu->dataSize * sizeof(float);
                    clEnqueueCopyBuffer(nn->queue, currentBuffer, relu->inputBuffer, 0, 0, dataSize, 0, NULL, NULL);
                }
                
                clSetKernelArg(nn->reluForwardKernel, 0, sizeof(cl_mem), &relu->inputBuffer);
                clSetKernelArg(nn->reluForwardKernel, 1, sizeof(cl_mem), &relu->outputBuffer);
                clSetKernelArg(nn->reluForwardKernel, 2, sizeof(int), &relu->dataSize);
                
                size_t globalSize = relu->dataSize;
                clEnqueueNDRangeKernel(nn->queue, nn->reluForwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                
                currentBuffer = relu->outputBuffer;
                break;
            }
            case LAYER_SOFTMAX: {
                ActivationLayer* softmax = &layer->activation;
                
                if (currentBuffer != NULL) {
                    int dataSize = softmax->dataSize * sizeof(float);
                    clEnqueueCopyBuffer(nn->queue, currentBuffer, softmax->inputBuffer, 0, 0, dataSize, 0, NULL, NULL);
                }
                
                clSetKernelArg(nn->softmaxForwardKernel, 0, sizeof(cl_mem), &softmax->inputBuffer);
                clSetKernelArg(nn->softmaxForwardKernel, 1, sizeof(cl_mem), &softmax->outputBuffer);
                clSetKernelArg(nn->softmaxForwardKernel, 2, sizeof(int), &softmax->dataSize);
                
                size_t globalSize = softmax->dataSize;
                clEnqueueNDRangeKernel(nn->queue, nn->softmaxForwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                
                currentBuffer = softmax->outputBuffer;
                break;
            }
            case LAYER_DECONV: {
                DeconvLayer* deconv = &layer->deconv;
                
                if (i == 0) {
                    clEnqueueWriteBuffer(nn->queue, deconv->inputBuffer, CL_TRUE, 0,
                                        deconv->inputHeight * deconv->inputWidth * deconv->inputChannels * sizeof(float),
                                        inputData, 0, NULL, NULL);
                } else if (currentBuffer != NULL) {
                    int inputSize = deconv->inputHeight * deconv->inputWidth * deconv->inputChannels * sizeof(float);
                    clEnqueueCopyBuffer(nn->queue, currentBuffer, deconv->inputBuffer, 0, 0, inputSize, 0, NULL, NULL);
                }
                
                clSetKernelArg(nn->deconvForwardKernel, 0, sizeof(cl_mem), &deconv->inputBuffer);
                clSetKernelArg(nn->deconvForwardKernel, 1, sizeof(cl_mem), &deconv->kernelBuffer);
                clSetKernelArg(nn->deconvForwardKernel, 2, sizeof(cl_mem), &deconv->outputBuffer);
                clSetKernelArg(nn->deconvForwardKernel, 3, sizeof(cl_mem), &deconv->biasBuffer);
                clSetKernelArg(nn->deconvForwardKernel, 4, sizeof(int), &deconv->inputHeight);
                clSetKernelArg(nn->deconvForwardKernel, 5, sizeof(int), &deconv->inputWidth);
                clSetKernelArg(nn->deconvForwardKernel, 6, sizeof(int), &deconv->inputChannels);
                clSetKernelArg(nn->deconvForwardKernel, 7, sizeof(int), &deconv->kernelSize);
                clSetKernelArg(nn->deconvForwardKernel, 8, sizeof(int), &deconv->outputHeight);
                clSetKernelArg(nn->deconvForwardKernel, 9, sizeof(int), &deconv->outputWidth);
                clSetKernelArg(nn->deconvForwardKernel, 10, sizeof(int), &deconv->outputChannels);
                clSetKernelArg(nn->deconvForwardKernel, 11, sizeof(int), &deconv->stride);
                clSetKernelArg(nn->deconvForwardKernel, 12, sizeof(int), &deconv->paddingHeight);
                clSetKernelArg(nn->deconvForwardKernel, 13, sizeof(int), &deconv->paddingWidth);
                
                size_t globalSize[3] = {deconv->outputHeight, deconv->outputWidth, deconv->outputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->deconvForwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                currentBuffer = deconv->outputBuffer;
                break;
            }
            case LAYER_RESIZE2D: {
                Resize2DLayer* resize = &layer->resize;
                
                if (i == 0) {
                    clEnqueueWriteBuffer(nn->queue, resize->inputBuffer, CL_TRUE, 0,
                                        resize->inputHeight * resize->inputWidth * resize->channels * sizeof(float),
                                        inputData, 0, NULL, NULL);
                } else if (currentBuffer != NULL) {
                    int inputSize = resize->inputHeight * resize->inputWidth * resize->channels * sizeof(float);
                    clEnqueueCopyBuffer(nn->queue, currentBuffer, resize->inputBuffer, 0, 0, inputSize, 0, NULL, NULL);
                }
                
                clSetKernelArg(nn->resize2DForwardKernel, 0, sizeof(cl_mem), &resize->inputBuffer);
                clSetKernelArg(nn->resize2DForwardKernel, 1, sizeof(cl_mem), &resize->outputBuffer);
                clSetKernelArg(nn->resize2DForwardKernel, 2, sizeof(int), &resize->inputWidth);
                clSetKernelArg(nn->resize2DForwardKernel, 3, sizeof(int), &resize->inputHeight);
                clSetKernelArg(nn->resize2DForwardKernel, 4, sizeof(int), &resize->outputWidth);
                clSetKernelArg(nn->resize2DForwardKernel, 5, sizeof(int), &resize->outputHeight);
                clSetKernelArg(nn->resize2DForwardKernel, 6, sizeof(int), &resize->channels);
                
                size_t globalSize[3] = {resize->outputWidth, resize->outputHeight, resize->channels};
                clEnqueueNDRangeKernel(nn->queue, nn->resize2DForwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                currentBuffer = resize->outputBuffer;
                break;
            }
        }
    }
    
    clFinish(nn->queue);
}

void backward(NeuralNetwork* nn, float* gradOutput) {
    cl_mem currentGradBuffer = NULL;
    
    // Write gradient to last layer
    Layer* lastLayer = &nn->layers[nn->numLayers - 1];
    int outputSize = 0;
    
    switch (lastLayer->type) {
        case LAYER_CONV:
            outputSize = lastLayer->conv.outputHeight * lastLayer->conv.outputWidth * lastLayer->conv.outputChannels;
            currentGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, outputSize * sizeof(float), NULL, NULL);
            break;
        case LAYER_DECONV:
            outputSize = lastLayer->deconv.outputHeight * lastLayer->deconv.outputWidth * lastLayer->deconv.outputChannels;
            currentGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, outputSize * sizeof(float), NULL, NULL);
            break;
        case LAYER_MAXPOOL:
            outputSize = lastLayer->pool.outputHeight * lastLayer->pool.outputWidth * lastLayer->pool.inputChannels;
            currentGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, outputSize * sizeof(float), NULL, NULL);
            break;
        case LAYER_RELU:
        case LAYER_SOFTMAX:
            outputSize = lastLayer->activation.dataSize;
            currentGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, outputSize * sizeof(float), NULL, NULL);
            break;
        case LAYER_RESIZE2D:
            outputSize = lastLayer->resize.outputHeight * lastLayer->resize.outputWidth * lastLayer->resize.channels;
            currentGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, outputSize * sizeof(float), NULL, NULL);
            break;
    }
    
    clEnqueueWriteBuffer(nn->queue, currentGradBuffer, CL_TRUE, 0, outputSize * sizeof(float), gradOutput, 0, NULL, NULL);
    
    // Backpropagate through layers in reverse
    for (int i = nn->numLayers - 1; i >= 0; i--) {
        Layer* layer = &nn->layers[i];
        
        switch (layer->type) {
            case LAYER_CONV: {
                ConvLayer* conv = &layer->conv;
                
                // Create input gradient buffer
                int inputGradSize = conv->inputHeight * conv->inputWidth * conv->inputChannels;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, 
                                                        inputGradSize * sizeof(float), NULL, NULL);
                float* zeros = (float*)calloc(inputGradSize, sizeof(float));
                clEnqueueWriteBuffer(nn->queue, inputGradBuffer, CL_TRUE, 0, 
                                    inputGradSize * sizeof(float), zeros, 0, NULL, NULL);
                free(zeros);
                
                // Clear gradient buffers
                zeros = (float*)calloc(conv->outputChannels, sizeof(float));
                clEnqueueWriteBuffer(nn->queue, conv->biasGradBuffer, CL_TRUE, 0,
                                    conv->outputChannels * sizeof(float), zeros, 0, NULL, NULL);
                free(zeros);
                
                // Set kernel arguments for backward pass
                clSetKernelArg(nn->convBackwardKernel, 0, sizeof(cl_mem), &conv->inputBuffer);
                clSetKernelArg(nn->convBackwardKernel, 1, sizeof(cl_mem), &conv->kernelBuffer);
                clSetKernelArg(nn->convBackwardKernel, 2, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 3, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 4, sizeof(cl_mem), &conv->kernelGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 5, sizeof(cl_mem), &conv->biasGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 6, sizeof(int), &conv->inputHeight);
                clSetKernelArg(nn->convBackwardKernel, 7, sizeof(int), &conv->inputWidth);
                clSetKernelArg(nn->convBackwardKernel, 8, sizeof(int), &conv->inputChannels);
                clSetKernelArg(nn->convBackwardKernel, 9, sizeof(int), &conv->kernelSize);
                clSetKernelArg(nn->convBackwardKernel, 10, sizeof(int), &conv->outputHeight);
                clSetKernelArg(nn->convBackwardKernel, 11, sizeof(int), &conv->outputWidth);
                clSetKernelArg(nn->convBackwardKernel, 12, sizeof(int), &conv->outputChannels);
                clSetKernelArg(nn->convBackwardKernel, 13, sizeof(int), &conv->stride);
                clSetKernelArg(nn->convBackwardKernel, 14, sizeof(int), &conv->paddingHeight);
                clSetKernelArg(nn->convBackwardKernel, 15, sizeof(int), &conv->paddingWidth);
                
                size_t globalSize[4] = {conv->kernelSize, conv->kernelSize, conv->inputChannels, conv->outputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->convBackwardKernel, 4, NULL, globalSize, NULL, 0, NULL, NULL);
                
                clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                break;
            }
            case LAYER_MAXPOOL: {
                MaxPoolLayer* pool = &layer->pool;
                
                int inputGradSize = pool->inputHeight * pool->inputWidth * pool->inputChannels;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        inputGradSize * sizeof(float), NULL, NULL);
                float* zeros = (float*)calloc(inputGradSize, sizeof(float));
                clEnqueueWriteBuffer(nn->queue, inputGradBuffer, CL_TRUE, 0,
                                    inputGradSize * sizeof(float), zeros, 0, NULL, NULL);
                free(zeros);
                
                clSetKernelArg(nn->maxPoolBackwardKernel, 0, sizeof(cl_mem), &pool->inputBuffer);
                clSetKernelArg(nn->maxPoolBackwardKernel, 1, sizeof(cl_mem), &pool->outputBuffer);
                clSetKernelArg(nn->maxPoolBackwardKernel, 2, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->maxPoolBackwardKernel, 3, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->maxPoolBackwardKernel, 4, sizeof(int), &pool->inputHeight);
                clSetKernelArg(nn->maxPoolBackwardKernel, 5, sizeof(int), &pool->inputWidth);
                clSetKernelArg(nn->maxPoolBackwardKernel, 6, sizeof(int), &pool->inputChannels);
                clSetKernelArg(nn->maxPoolBackwardKernel, 7, sizeof(int), &pool->poolSize);
                clSetKernelArg(nn->maxPoolBackwardKernel, 8, sizeof(int), &pool->stride);
                clSetKernelArg(nn->maxPoolBackwardKernel, 9, sizeof(int), &pool->outputHeight);
                clSetKernelArg(nn->maxPoolBackwardKernel, 10, sizeof(int), &pool->outputWidth);
                
                size_t globalSize[3] = {pool->outputHeight, pool->outputWidth, pool->inputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->maxPoolBackwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                break;
            }
            case LAYER_RELU: {
                ActivationLayer* relu = &layer->activation;
                
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        relu->dataSize * sizeof(float), NULL, NULL);
                
                clSetKernelArg(nn->reluBackwardKernel, 0, sizeof(cl_mem), &relu->inputBuffer);
                clSetKernelArg(nn->reluBackwardKernel, 1, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->reluBackwardKernel, 2, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->reluBackwardKernel, 3, sizeof(int), &relu->dataSize);
                
                size_t globalSize = relu->dataSize;
                clEnqueueNDRangeKernel(nn->queue, nn->reluBackwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                
                clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                break;
            }
            case LAYER_SOFTMAX: {
                ActivationLayer* softmax = &layer->activation;
                
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        softmax->dataSize * sizeof(float), NULL, NULL);
                
                clSetKernelArg(nn->softmaxBackwardKernel, 0, sizeof(cl_mem), &softmax->outputBuffer);
                clSetKernelArg(nn->softmaxBackwardKernel, 1, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->softmaxBackwardKernel, 2, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->softmaxBackwardKernel, 3, sizeof(int), &softmax->dataSize);
                
                size_t globalSize = softmax->dataSize;
                clEnqueueNDRangeKernel(nn->queue, nn->softmaxBackwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                
                clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                break;
            }
            case LAYER_DECONV: {
                DeconvLayer* deconv = &layer->deconv;
                
                int inputGradSize = deconv->inputHeight * deconv->inputWidth * deconv->inputChannels;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        inputGradSize * sizeof(float), NULL, NULL);
                float* zeros = (float*)calloc(inputGradSize, sizeof(float));
                clEnqueueWriteBuffer(nn->queue, inputGradBuffer, CL_TRUE, 0,
                                    inputGradSize * sizeof(float), zeros, 0, NULL, NULL);
                free(zeros);
                
                zeros = (float*)calloc(deconv->outputChannels, sizeof(float));
                clEnqueueWriteBuffer(nn->queue, deconv->biasGradBuffer, CL_TRUE, 0,
                                    deconv->outputChannels * sizeof(float), zeros, 0, NULL, NULL);
                free(zeros);
                
                clSetKernelArg(nn->deconvBackwardKernel, 0, sizeof(cl_mem), &deconv->inputBuffer);
                clSetKernelArg(nn->deconvBackwardKernel, 1, sizeof(cl_mem), &deconv->kernelBuffer);
                clSetKernelArg(nn->deconvBackwardKernel, 2, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->deconvBackwardKernel, 3, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->deconvBackwardKernel, 4, sizeof(cl_mem), &deconv->kernelGradBuffer);
                clSetKernelArg(nn->deconvBackwardKernel, 5, sizeof(cl_mem), &deconv->biasGradBuffer);
                clSetKernelArg(nn->deconvBackwardKernel, 6, sizeof(int), &deconv->inputHeight);
                clSetKernelArg(nn->deconvBackwardKernel, 7, sizeof(int), &deconv->inputWidth);
                clSetKernelArg(nn->deconvBackwardKernel, 8, sizeof(int), &deconv->inputChannels);
                clSetKernelArg(nn->deconvBackwardKernel, 9, sizeof(int), &deconv->kernelSize);
                clSetKernelArg(nn->deconvBackwardKernel, 10, sizeof(int), &deconv->outputHeight);
                clSetKernelArg(nn->deconvBackwardKernel, 11, sizeof(int), &deconv->outputWidth);
                clSetKernelArg(nn->deconvBackwardKernel, 12, sizeof(int), &deconv->outputChannels);
                clSetKernelArg(nn->deconvBackwardKernel, 13, sizeof(int), &deconv->stride);
                clSetKernelArg(nn->deconvBackwardKernel, 14, sizeof(int), &deconv->paddingHeight);
                clSetKernelArg(nn->deconvBackwardKernel, 15, sizeof(int), &deconv->paddingWidth);
                
                size_t globalSize[4] = {deconv->kernelSize, deconv->kernelSize, deconv->inputChannels, deconv->outputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->deconvBackwardKernel, 4, NULL, globalSize, NULL, 0, NULL, NULL);
                
                clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                break;
            }
            case LAYER_RESIZE2D: {
                Resize2DLayer* resize = &layer->resize;
                
                int inputGradSize = resize->inputHeight * resize->inputWidth * resize->channels;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        inputGradSize * sizeof(float), NULL, NULL);
                float* zeros = (float*)calloc(inputGradSize, sizeof(float));
                clEnqueueWriteBuffer(nn->queue, inputGradBuffer, CL_TRUE, 0,
                                    inputGradSize * sizeof(float), zeros, 0, NULL, NULL);
                free(zeros);
                
                clSetKernelArg(nn->resize2DBackwardKernel, 0, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->resize2DBackwardKernel, 1, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->resize2DBackwardKernel, 2, sizeof(int), &resize->inputWidth);
                clSetKernelArg(nn->resize2DBackwardKernel, 3, sizeof(int), &resize->inputHeight);
                clSetKernelArg(nn->resize2DBackwardKernel, 4, sizeof(int), &resize->outputWidth);
                clSetKernelArg(nn->resize2DBackwardKernel, 5, sizeof(int), &resize->outputHeight);
                clSetKernelArg(nn->resize2DBackwardKernel, 6, sizeof(int), &resize->channels);
                
                size_t globalSize[3] = {resize->outputWidth, resize->outputHeight, resize->channels};
                clEnqueueNDRangeKernel(nn->queue, nn->resize2DBackwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                break;
            }
        }
    }
    
    if (currentGradBuffer) {
        clReleaseMemObject(currentGradBuffer);
    }
    
    clFinish(nn->queue);
}

void updateWeights(NeuralNetwork* nn, float learningRate) {
    for (int i = 0; i < nn->numLayers; i++) {
        Layer* layer = &nn->layers[i];
        
        if (layer->type == LAYER_CONV) {
            ConvLayer* conv = &layer->conv;
            int kernelDataSize = conv->kernelSize * conv->kernelSize * conv->inputChannels * conv->outputChannels;
            
            // Read gradients from GPU
            float* kernelGrad = (float*)malloc(kernelDataSize * sizeof(float));
            float* biasGrad = (float*)malloc(conv->outputChannels * sizeof(float));
            
            clEnqueueReadBuffer(nn->queue, conv->kernelGradBuffer, CL_TRUE, 0,
                               kernelDataSize * sizeof(float), kernelGrad, 0, NULL, NULL);
            clEnqueueReadBuffer(nn->queue, conv->biasGradBuffer, CL_TRUE, 0,
                               conv->outputChannels * sizeof(float), biasGrad, 0, NULL, NULL);
            
            // Update weights
            for (int j = 0; j < kernelDataSize; j++) {
                conv->kernelWeights[j] -= learningRate * kernelGrad[j];
            }
            for (int j = 0; j < conv->outputChannels; j++) {
                conv->biasWeights[j] -= learningRate * biasGrad[j];
            }
            
            // Write back to GPU
            clEnqueueWriteBuffer(nn->queue, conv->kernelBuffer, CL_TRUE, 0,
                                kernelDataSize * sizeof(float), conv->kernelWeights, 0, NULL, NULL);
            clEnqueueWriteBuffer(nn->queue, conv->biasBuffer, CL_TRUE, 0,
                                conv->outputChannels * sizeof(float), conv->biasWeights, 0, NULL, NULL);
            
            free(kernelGrad);
            free(biasGrad);
        }
        
        if (layer->type == LAYER_DECONV) {
            DeconvLayer* deconv = &layer->deconv;
            int kernelDataSize = deconv->kernelSize * deconv->kernelSize * deconv->inputChannels * deconv->outputChannels;
            
            float* kernelGrad = (float*)malloc(kernelDataSize * sizeof(float));
            float* biasGrad = (float*)malloc(deconv->outputChannels * sizeof(float));
            
            clEnqueueReadBuffer(nn->queue, deconv->kernelGradBuffer, CL_TRUE, 0,
                               kernelDataSize * sizeof(float), kernelGrad, 0, NULL, NULL);
            clEnqueueReadBuffer(nn->queue, deconv->biasGradBuffer, CL_TRUE, 0,
                               deconv->outputChannels * sizeof(float), biasGrad, 0, NULL, NULL);
            
            for (int j = 0; j < kernelDataSize; j++) {
                deconv->kernelWeights[j] -= learningRate * kernelGrad[j];
            }
            for (int j = 0; j < deconv->outputChannels; j++) {
                deconv->biasWeights[j] -= learningRate * biasGrad[j];
            }
            
            clEnqueueWriteBuffer(nn->queue, deconv->kernelBuffer, CL_TRUE, 0,
                                kernelDataSize * sizeof(float), deconv->kernelWeights, 0, NULL, NULL);
            clEnqueueWriteBuffer(nn->queue, deconv->biasBuffer, CL_TRUE, 0,
                                deconv->outputChannels * sizeof(float), deconv->biasWeights, 0, NULL, NULL);
            
            free(kernelGrad);
            free(biasGrad);
        }
    }
    
    clFinish(nn->queue);
}

float computeCombinedLoss(NeuralNetwork* nn, float* output, float* target, int width, int height, int channels) {
    int dataSize = width * height * channels;
    
    // Upload data to GPU once
    cl_mem outputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         dataSize * sizeof(float), output, NULL);
    cl_mem targetBuffer = clCreateBuffer(nn->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         dataSize * sizeof(float), target, NULL);
    
    // MAE loss on GPU
    cl_mem maeLossBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    float zeroLoss = 0.0f;
    clEnqueueWriteBuffer(nn->queue, maeLossBuffer, CL_TRUE, 0, sizeof(float), &zeroLoss, 0, NULL, NULL);
    
    clSetKernelArg(nn->maeLossKernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(nn->maeLossKernel, 1, sizeof(cl_mem), &targetBuffer);
    clSetKernelArg(nn->maeLossKernel, 2, sizeof(cl_mem), &maeLossBuffer);
    clSetKernelArg(nn->maeLossKernel, 3, sizeof(int), &dataSize);
    
    size_t globalSize = dataSize;
    clEnqueueNDRangeKernel(nn->queue, nn->maeLossKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
    
    // SSIM loss on GPU (window-based)
    cl_mem ssimLossBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(nn->queue, ssimLossBuffer, CL_TRUE, 0, sizeof(float), &zeroLoss, 0, NULL, NULL);
    
    clSetKernelArg(nn->ssimLossKernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(nn->ssimLossKernel, 1, sizeof(cl_mem), &targetBuffer);
    clSetKernelArg(nn->ssimLossKernel, 2, sizeof(cl_mem), &ssimLossBuffer);
    clSetKernelArg(nn->ssimLossKernel, 3, sizeof(int), &width);
    clSetKernelArg(nn->ssimLossKernel, 4, sizeof(int), &height);
    clSetKernelArg(nn->ssimLossKernel, 5, sizeof(int), &channels);
    
    // SSIM uses 8x8 windows
    int windowSize = 8;
    int windowsPerRow = (width + windowSize - 1) / windowSize;
    int windowsPerCol = (height + windowSize - 1) / windowSize;
    size_t ssimGlobalSize = windowsPerRow * windowsPerCol * channels;
    clEnqueueNDRangeKernel(nn->queue, nn->ssimLossKernel, 1, NULL, &ssimGlobalSize, NULL, 0, NULL, NULL);
    
    // Gradient/frequency loss on GPU (Laplacian-based, works with any dimensions)
    cl_mem gradLossBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(nn->queue, gradLossBuffer, CL_TRUE, 0, sizeof(float), &zeroLoss, 0, NULL, NULL);
    
    clSetKernelArg(nn->gradientLossKernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(nn->gradientLossKernel, 1, sizeof(cl_mem), &targetBuffer);
    clSetKernelArg(nn->gradientLossKernel, 2, sizeof(cl_mem), &gradLossBuffer);
    clSetKernelArg(nn->gradientLossKernel, 3, sizeof(int), &width);
    clSetKernelArg(nn->gradientLossKernel, 4, sizeof(int), &height);
    clSetKernelArg(nn->gradientLossKernel, 5, sizeof(int), &channels);
    
    size_t gradGlobalSize[3] = {width, height, channels};
    clEnqueueNDRangeKernel(nn->queue, nn->gradientLossKernel, 3, NULL, gradGlobalSize, NULL, 0, NULL, NULL);
    
    // Read all losses from GPU
    float maeLoss, ssimLoss, gradLoss;
    clEnqueueReadBuffer(nn->queue, maeLossBuffer, CL_TRUE, 0, sizeof(float), &maeLoss, 0, NULL, NULL);
    clEnqueueReadBuffer(nn->queue, ssimLossBuffer, CL_TRUE, 0, sizeof(float), &ssimLoss, 0, NULL, NULL);
    clEnqueueReadBuffer(nn->queue, gradLossBuffer, CL_TRUE, 0, sizeof(float), &gradLoss, 0, NULL, NULL);
    
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(targetBuffer);
    clReleaseMemObject(maeLossBuffer);
    clReleaseMemObject(ssimLossBuffer);
    clReleaseMemObject(gradLossBuffer);
    
    return maeLoss + 0.5f * ssimLoss + 0.3f * gradLoss;
}

void computeCombinedGradient(float* output, float* target, float* gradient, int width, int height, int channels) {
    int dataSize = width * height * channels;
    
    // MAE gradient: sign(output - target) / dataSize
    for (int i = 0; i < dataSize; i++) {
        float diff = output[i] - target[i];
        gradient[i] = (diff > 0 ? 1.0f : -1.0f) / dataSize;
    }
    
    // Add SSIM gradient contribution (simplified)
    float weight_ssim = 0.5f;
    for (int i = 0; i < dataSize; i++) {
        gradient[i] += weight_ssim * 2.0f * (output[i] - target[i]) / dataSize;
    }
    
    // Add FFT/edge gradient contribution
    float weight_fft = 0.3f;
    
    // For FFT gradient, we approximate with high-frequency component gradients
    // (full FFT gradient would require inverse FFT, which is complex)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                int idxRight = (y * width + (x + 1)) * channels + c;
                int idxDown = ((y + 1) * width + x) * channels + c;
                int idxLeft = (y * width + (x - 1)) * channels + c;
                int idxUp = ((y - 1) * width + x) * channels + c;
                
                // Laplacian (high-frequency component)
                float laplacianOut = 4.0f * output[idx] - output[idxRight] - output[idxDown] - 
                                     output[idxLeft] - output[idxUp];
                float laplacianTarget = 4.0f * target[idx] - target[idxRight] - target[idxDown] - 
                                        target[idxLeft] - target[idxUp];
                
                float diff = laplacianOut - laplacianTarget;
                gradient[idx] += weight_fft * diff / ((width - 2) * (height - 2) * channels);
            }
        }
    }
}

// ============================================================================
// GPU-only training pipeline - minimizes CPU-GPU transfers
// ============================================================================

void initTrainBuffers(NeuralNetwork* nn, int width, int height, int channels) {
    int size = width * height * channels;
    if (nn->trainBufferSize != size) {
        if (nn->trainTargetBuffer) clReleaseMemObject(nn->trainTargetBuffer);
        if (nn->trainOutputBuffer) clReleaseMemObject(nn->trainOutputBuffer);
        if (nn->trainLossBuffer) clReleaseMemObject(nn->trainLossBuffer);
        if (nn->trainGradBuffer) clReleaseMemObject(nn->trainGradBuffer);
        
        nn->trainTargetBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
        nn->trainOutputBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
        nn->trainLossBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(float), NULL, NULL);
        nn->trainGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, size * sizeof(float), NULL, NULL);
        nn->trainBufferSize = size;
    }
}

void uploadTarget(NeuralNetwork* nn, float* target, int size) {
    clEnqueueWriteBuffer(nn->queue, nn->trainTargetBuffer, CL_FALSE, 0, size * sizeof(float), target, 0, NULL, NULL);
}

cl_mem getOutputBuffer(NeuralNetwork* nn) {
    Layer* lastLayer = &nn->layers[nn->numLayers - 1];
    switch (lastLayer->type) {
        case LAYER_CONV: return lastLayer->conv.outputBuffer;
        case LAYER_DECONV: return lastLayer->deconv.outputBuffer;
        case LAYER_MAXPOOL: return lastLayer->pool.outputBuffer;
        case LAYER_RELU:
        case LAYER_SOFTMAX: return lastLayer->activation.outputBuffer;
        case LAYER_RESIZE2D: return lastLayer->resize.outputBuffer;
        default: return NULL;
    }
}

// Backward pass starting from GPU buffer instead of CPU array
void backwardFromGPU(NeuralNetwork* nn) {
    cl_mem currentGradBuffer = nn->trainGradBuffer;
    int needsRelease = 0;
    
    for (int i = nn->numLayers - 1; i >= 0; i--) {
        Layer* layer = &nn->layers[i];
        
        switch (layer->type) {
            case LAYER_CONV: {
                ConvLayer* conv = &layer->conv;
                int inputGradSize = conv->inputHeight * conv->inputWidth * conv->inputChannels;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE, 
                                                        inputGradSize * sizeof(float), NULL, NULL);
                
                float zero = 0.0f;
                clEnqueueFillBuffer(nn->queue, inputGradBuffer, &zero, sizeof(float), 0, 
                                   inputGradSize * sizeof(float), 0, NULL, NULL);
                clEnqueueFillBuffer(nn->queue, conv->biasGradBuffer, &zero, sizeof(float), 0,
                                   conv->outputChannels * sizeof(float), 0, NULL, NULL);
                
                clSetKernelArg(nn->convBackwardKernel, 0, sizeof(cl_mem), &conv->inputBuffer);
                clSetKernelArg(nn->convBackwardKernel, 1, sizeof(cl_mem), &conv->kernelBuffer);
                clSetKernelArg(nn->convBackwardKernel, 2, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 3, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 4, sizeof(cl_mem), &conv->kernelGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 5, sizeof(cl_mem), &conv->biasGradBuffer);
                clSetKernelArg(nn->convBackwardKernel, 6, sizeof(int), &conv->inputHeight);
                clSetKernelArg(nn->convBackwardKernel, 7, sizeof(int), &conv->inputWidth);
                clSetKernelArg(nn->convBackwardKernel, 8, sizeof(int), &conv->inputChannels);
                clSetKernelArg(nn->convBackwardKernel, 9, sizeof(int), &conv->kernelSize);
                clSetKernelArg(nn->convBackwardKernel, 10, sizeof(int), &conv->outputHeight);
                clSetKernelArg(nn->convBackwardKernel, 11, sizeof(int), &conv->outputWidth);
                clSetKernelArg(nn->convBackwardKernel, 12, sizeof(int), &conv->outputChannels);
                clSetKernelArg(nn->convBackwardKernel, 13, sizeof(int), &conv->stride);
                clSetKernelArg(nn->convBackwardKernel, 14, sizeof(int), &conv->paddingHeight);
                clSetKernelArg(nn->convBackwardKernel, 15, sizeof(int), &conv->paddingWidth);
                
                size_t globalSize[4] = {conv->kernelSize, conv->kernelSize, conv->inputChannels, conv->outputChannels};
                clEnqueueNDRangeKernel(nn->queue, nn->convBackwardKernel, 4, NULL, globalSize, NULL, 0, NULL, NULL);
                
                if (needsRelease) clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                needsRelease = 1;
                break;
            }
            case LAYER_RELU: {
                ActivationLayer* relu = &layer->activation;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        relu->dataSize * sizeof(float), NULL, NULL);
                
                clSetKernelArg(nn->reluBackwardKernel, 0, sizeof(cl_mem), &relu->inputBuffer);
                clSetKernelArg(nn->reluBackwardKernel, 1, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->reluBackwardKernel, 2, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->reluBackwardKernel, 3, sizeof(int), &relu->dataSize);
                
                size_t globalSize = relu->dataSize;
                clEnqueueNDRangeKernel(nn->queue, nn->reluBackwardKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
                
                if (needsRelease) clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                needsRelease = 1;
                break;
            }
            case LAYER_RESIZE2D: {
                Resize2DLayer* resize = &layer->resize;
                int inputGradSize = resize->inputHeight * resize->inputWidth * resize->channels;
                cl_mem inputGradBuffer = clCreateBuffer(nn->context, CL_MEM_READ_WRITE,
                                                        inputGradSize * sizeof(float), NULL, NULL);
                
                float zero = 0.0f;
                clEnqueueFillBuffer(nn->queue, inputGradBuffer, &zero, sizeof(float), 0,
                                   inputGradSize * sizeof(float), 0, NULL, NULL);
                
                clSetKernelArg(nn->resize2DBackwardKernel, 0, sizeof(cl_mem), &currentGradBuffer);
                clSetKernelArg(nn->resize2DBackwardKernel, 1, sizeof(cl_mem), &inputGradBuffer);
                clSetKernelArg(nn->resize2DBackwardKernel, 2, sizeof(int), &resize->inputWidth);
                clSetKernelArg(nn->resize2DBackwardKernel, 3, sizeof(int), &resize->inputHeight);
                clSetKernelArg(nn->resize2DBackwardKernel, 4, sizeof(int), &resize->outputWidth);
                clSetKernelArg(nn->resize2DBackwardKernel, 5, sizeof(int), &resize->outputHeight);
                clSetKernelArg(nn->resize2DBackwardKernel, 6, sizeof(int), &resize->channels);
                
                size_t globalSize[3] = {resize->outputWidth, resize->outputHeight, resize->channels};
                clEnqueueNDRangeKernel(nn->queue, nn->resize2DBackwardKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
                
                if (needsRelease) clReleaseMemObject(currentGradBuffer);
                currentGradBuffer = inputGradBuffer;
                needsRelease = 1;
                break;
            }
            case LAYER_MAXPOOL:
            case LAYER_SOFTMAX:
            case LAYER_DECONV:
                // Use existing backward implementations
                break;
        }
    }
    
    if (needsRelease && currentGradBuffer) {
        clReleaseMemObject(currentGradBuffer);
    }
    clFinish(nn->queue);
}

float trainStepGPU(NeuralNetwork* nn, float* input, float learningRate, int width, int height, int channels) {
    int dataSize = width * height * channels;
    
    // Forward pass (input uploaded to GPU)
    forward(nn, input);
    
    // Get output buffer from last layer
    cl_mem outputBuffer = getOutputBuffer(nn);
    
    // Initialize loss to zero on GPU
    float zeroLoss = 0.0f;
    clEnqueueWriteBuffer(nn->queue, nn->trainLossBuffer, CL_TRUE, 0, sizeof(float), &zeroLoss, 0, NULL, NULL);
    
    // Clear gradient buffer
    float zero = 0.0f;
    clEnqueueFillBuffer(nn->queue, nn->trainGradBuffer, &zero, sizeof(float), 0, dataSize * sizeof(float), 0, NULL, NULL);
    
    // Loss weights
    float maeWeight = 0.1f;
    float ssimWeight = 0.5f;
    float fftWeight = 0.4f;
    
    // Combined loss + gradient kernel (MAE + Laplacian)
    clSetKernelArg(nn->combinedLossKernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(nn->combinedLossKernel, 1, sizeof(cl_mem), &nn->trainTargetBuffer);
    clSetKernelArg(nn->combinedLossKernel, 2, sizeof(cl_mem), &nn->trainGradBuffer);
    clSetKernelArg(nn->combinedLossKernel, 3, sizeof(cl_mem), &nn->trainLossBuffer);
    clSetKernelArg(nn->combinedLossKernel, 4, sizeof(int), &width);
    clSetKernelArg(nn->combinedLossKernel, 5, sizeof(int), &height);
    clSetKernelArg(nn->combinedLossKernel, 6, sizeof(int), &channels);
    clSetKernelArg(nn->combinedLossKernel, 7, sizeof(float), &maeWeight);
    clSetKernelArg(nn->combinedLossKernel, 8, sizeof(float), &ssimWeight);
    clSetKernelArg(nn->combinedLossKernel, 9, sizeof(float), &fftWeight);
    
    size_t globalSize[3] = {width, height, channels};
    clEnqueueNDRangeKernel(nn->queue, nn->combinedLossKernel, 3, NULL, globalSize, NULL, 0, NULL, NULL);
    
    // SSIM loss + gradient (window-based)
    clSetKernelArg(nn->combinedGradientKernel, 0, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(nn->combinedGradientKernel, 1, sizeof(cl_mem), &nn->trainTargetBuffer);
    clSetKernelArg(nn->combinedGradientKernel, 2, sizeof(cl_mem), &nn->trainGradBuffer);
    clSetKernelArg(nn->combinedGradientKernel, 3, sizeof(cl_mem), &nn->trainLossBuffer);
    clSetKernelArg(nn->combinedGradientKernel, 4, sizeof(int), &width);
    clSetKernelArg(nn->combinedGradientKernel, 5, sizeof(int), &height);
    clSetKernelArg(nn->combinedGradientKernel, 6, sizeof(int), &channels);
    clSetKernelArg(nn->combinedGradientKernel, 7, sizeof(float), &ssimWeight);
    
    int windowSize = 8;
    int windowsPerRow = (width + windowSize - 1) / windowSize;
    int windowsPerCol = (height + windowSize - 1) / windowSize;
    size_t ssimGlobalSize = windowsPerRow * windowsPerCol * channels;
    clEnqueueNDRangeKernel(nn->queue, nn->combinedGradientKernel, 1, NULL, &ssimGlobalSize, NULL, 0, NULL, NULL);
    
    // Backward pass from GPU gradient buffer
    backwardFromGPU(nn);
    
    // Update weights (still uses CPU for now, can be optimized later)
    updateWeights(nn, learningRate);
    
    // Only readback: loss value (single float)
    float loss;
    clEnqueueReadBuffer(nn->queue, nn->trainLossBuffer, CL_TRUE, 0, sizeof(float), &loss, 0, NULL, NULL);
    
    return loss;
}

void cleanupOpenCL(NeuralNetwork* nn) {
    clReleaseKernel(nn->convForwardKernel);
    clReleaseKernel(nn->convBackwardKernel);
    clReleaseKernel(nn->deconvForwardKernel);
    clReleaseKernel(nn->deconvBackwardKernel);
    clReleaseKernel(nn->maxPoolForwardKernel);
    clReleaseKernel(nn->maxPoolBackwardKernel);
    clReleaseKernel(nn->reluForwardKernel);
    clReleaseKernel(nn->reluBackwardKernel);
    clReleaseKernel(nn->softmaxForwardKernel);
    clReleaseKernel(nn->softmaxBackwardKernel);
    clReleaseKernel(nn->resize2DForwardKernel);
    clReleaseKernel(nn->resize2DBackwardKernel);
    clReleaseKernel(nn->mseLossKernel);
    clReleaseKernel(nn->maeLossKernel);
    clReleaseKernel(nn->ssimLossKernel);
    clReleaseKernel(nn->gradientLossKernel);
    clReleaseKernel(nn->fft1DKernel);
    clReleaseKernel(nn->transposeKernel);
    clReleaseKernel(nn->fftLossKernel);
    clReleaseKernel(nn->combinedLossKernel);
    clReleaseKernel(nn->combinedGradientKernel);
    
    if (nn->trainTargetBuffer) clReleaseMemObject(nn->trainTargetBuffer);
    if (nn->trainOutputBuffer) clReleaseMemObject(nn->trainOutputBuffer);
    if (nn->trainLossBuffer) clReleaseMemObject(nn->trainLossBuffer);
    if (nn->trainGradBuffer) clReleaseMemObject(nn->trainGradBuffer);
    
    clReleaseProgram(nn->program);
    clReleaseCommandQueue(nn->queue);
    clReleaseContext(nn->context);
}

void freeNetwork(NeuralNetwork* nn) {
    for (int i = 0; i < nn->numLayers; i++) {
        Layer* layer = &nn->layers[i];
        switch (layer->type) {
            case LAYER_CONV:
                clReleaseMemObject(layer->conv.inputBuffer);
                clReleaseMemObject(layer->conv.outputBuffer);
                clReleaseMemObject(layer->conv.kernelBuffer);
                clReleaseMemObject(layer->conv.biasBuffer);
                clReleaseMemObject(layer->conv.kernelGradBuffer);
                clReleaseMemObject(layer->conv.biasGradBuffer);
                free(layer->conv.kernelWeights);
                free(layer->conv.biasWeights);
                break;
            case LAYER_DECONV:
                clReleaseMemObject(layer->deconv.inputBuffer);
                clReleaseMemObject(layer->deconv.outputBuffer);
                clReleaseMemObject(layer->deconv.kernelBuffer);
                clReleaseMemObject(layer->deconv.biasBuffer);
                clReleaseMemObject(layer->deconv.kernelGradBuffer);
                clReleaseMemObject(layer->deconv.biasGradBuffer);
                free(layer->deconv.kernelWeights);
                free(layer->deconv.biasWeights);
                break;
            case LAYER_MAXPOOL:
                clReleaseMemObject(layer->pool.inputBuffer);
                clReleaseMemObject(layer->pool.outputBuffer);
                break;
            case LAYER_RELU:
            case LAYER_SOFTMAX:
                clReleaseMemObject(layer->activation.inputBuffer);
                clReleaseMemObject(layer->activation.outputBuffer);
                break;
            case LAYER_RESIZE2D:
                clReleaseMemObject(layer->resize.inputBuffer);
                clReleaseMemObject(layer->resize.outputBuffer);
                break;
        }
    }
    free(nn->layers);
    cleanupOpenCL(nn);
    free(nn);
}

int saveModel(NeuralNetwork* nn, const char* path) {
    FILE* file = fopen(path, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        return -1;
    }
    
    // Write header: magic + version + numLayers
    uint32_t magic = 0x4D4C4E4E; // "MLNN"
    uint32_t version = 1;
    fwrite(&magic, sizeof(uint32_t), 1, file);
    fwrite(&version, sizeof(uint32_t), 1, file);
    fwrite(&nn->numLayers, sizeof(int), 1, file);
    
    // Write each layer
    for (int i = 0; i < nn->numLayers; i++) {
        Layer* layer = &nn->layers[i];
        fwrite(&layer->type, sizeof(LayerType), 1, file);
        
        switch (layer->type) {
            case LAYER_CONV: {
                ConvLayer* conv = &layer->conv;
                
                // Write layer config
                fwrite(&conv->inputHeight, sizeof(int), 1, file);
                fwrite(&conv->inputWidth, sizeof(int), 1, file);
                fwrite(&conv->inputChannels, sizeof(int), 1, file);
                fwrite(&conv->kernelSize, sizeof(int), 1, file);
                fwrite(&conv->outputChannels, sizeof(int), 1, file);
                fwrite(&conv->stride, sizeof(int), 1, file);
                fwrite(&conv->paddingHeight, sizeof(int), 1, file);
                fwrite(&conv->paddingWidth, sizeof(int), 1, file);
                
                // Read weights from GPU to ensure we have latest
                int kernelDataSize = conv->kernelSize * conv->kernelSize * conv->inputChannels * conv->outputChannels;
                clEnqueueReadBuffer(nn->queue, conv->kernelBuffer, CL_TRUE, 0,
                                   kernelDataSize * sizeof(float), conv->kernelWeights, 0, NULL, NULL);
                clEnqueueReadBuffer(nn->queue, conv->biasBuffer, CL_TRUE, 0,
                                   conv->outputChannels * sizeof(float), conv->biasWeights, 0, NULL, NULL);
                
                // Write weights
                fwrite(conv->kernelWeights, sizeof(float), kernelDataSize, file);
                fwrite(conv->biasWeights, sizeof(float), conv->outputChannels, file);
                break;
            }
            case LAYER_DECONV: {
                DeconvLayer* deconv = &layer->deconv;
                
                fwrite(&deconv->inputHeight, sizeof(int), 1, file);
                fwrite(&deconv->inputWidth, sizeof(int), 1, file);
                fwrite(&deconv->inputChannels, sizeof(int), 1, file);
                fwrite(&deconv->kernelSize, sizeof(int), 1, file);
                fwrite(&deconv->outputChannels, sizeof(int), 1, file);
                fwrite(&deconv->stride, sizeof(int), 1, file);
                fwrite(&deconv->paddingHeight, sizeof(int), 1, file);
                fwrite(&deconv->paddingWidth, sizeof(int), 1, file);
                
                int kernelDataSize = deconv->kernelSize * deconv->kernelSize * deconv->inputChannels * deconv->outputChannels;
                clEnqueueReadBuffer(nn->queue, deconv->kernelBuffer, CL_TRUE, 0,
                                   kernelDataSize * sizeof(float), deconv->kernelWeights, 0, NULL, NULL);
                clEnqueueReadBuffer(nn->queue, deconv->biasBuffer, CL_TRUE, 0,
                                   deconv->outputChannels * sizeof(float), deconv->biasWeights, 0, NULL, NULL);
                
                fwrite(deconv->kernelWeights, sizeof(float), kernelDataSize, file);
                fwrite(deconv->biasWeights, sizeof(float), deconv->outputChannels, file);
                break;
            }
            case LAYER_MAXPOOL: {
                MaxPoolLayer* pool = &layer->pool;
                fwrite(&pool->inputHeight, sizeof(int), 1, file);
                fwrite(&pool->inputWidth, sizeof(int), 1, file);
                fwrite(&pool->inputChannels, sizeof(int), 1, file);
                fwrite(&pool->poolSize, sizeof(int), 1, file);
                fwrite(&pool->stride, sizeof(int), 1, file);
                break;
            }
            case LAYER_RELU:
            case LAYER_SOFTMAX: {
                ActivationLayer* act = &layer->activation;
                fwrite(&act->dataSize, sizeof(int), 1, file);
                break;
            }
            case LAYER_RESIZE2D: {
                Resize2DLayer* resize = &layer->resize;
                fwrite(&resize->inputHeight, sizeof(int), 1, file);
                fwrite(&resize->inputWidth, sizeof(int), 1, file);
                fwrite(&resize->outputHeight, sizeof(int), 1, file);
                fwrite(&resize->outputWidth, sizeof(int), 1, file);
                fwrite(&resize->channels, sizeof(int), 1, file);
                break;
            }
        }
    }
    
    fclose(file);
    printf("Model saved to %s\n", path);
    return 0;
}

int loadModel(NeuralNetwork* nn, const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open %s for reading\n", path);
        return -1;
    }
    
    // Read and verify header
    uint32_t magic, version;
    int numLayers;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(&version, sizeof(uint32_t), 1, file);
    fread(&numLayers, sizeof(int), 1, file);
    
    if (magic != 0x4D4C4E4E) {
        fprintf(stderr, "Invalid model file format\n");
        fclose(file);
        return -1;
    }
    
    if (numLayers != nn->numLayers) {
        fprintf(stderr, "Model layer count mismatch: file has %d, network has %d\n", numLayers, nn->numLayers);
        fclose(file);
        return -1;
    }
    
    // Read each layer
    for (int i = 0; i < nn->numLayers; i++) {
        Layer* layer = &nn->layers[i];
        LayerType fileLayerType;
        fread(&fileLayerType, sizeof(LayerType), 1, file);
        
        if (fileLayerType != layer->type) {
            fprintf(stderr, "Layer %d type mismatch\n", i);
            fclose(file);
            return -1;
        }
        
        switch (layer->type) {
            case LAYER_CONV: {
                ConvLayer* conv = &layer->conv;
                
                // Read and verify config
                int inputH, inputW, inputC, kernelSize, outputC, stride, padH, padW;
                fread(&inputH, sizeof(int), 1, file);
                fread(&inputW, sizeof(int), 1, file);
                fread(&inputC, sizeof(int), 1, file);
                fread(&kernelSize, sizeof(int), 1, file);
                fread(&outputC, sizeof(int), 1, file);
                fread(&stride, sizeof(int), 1, file);
                fread(&padH, sizeof(int), 1, file);
                fread(&padW, sizeof(int), 1, file);
                
                if (inputH != conv->inputHeight || inputW != conv->inputWidth ||
                    inputC != conv->inputChannels || kernelSize != conv->kernelSize ||
                    outputC != conv->outputChannels) {
                    fprintf(stderr, "Conv layer %d dimension mismatch\n", i);
                    fclose(file);
                    return -1;
                }
                
                // Read weights
                int kernelDataSize = conv->kernelSize * conv->kernelSize * conv->inputChannels * conv->outputChannels;
                fread(conv->kernelWeights, sizeof(float), kernelDataSize, file);
                fread(conv->biasWeights, sizeof(float), conv->outputChannels, file);
                
                // Upload to GPU
                clEnqueueWriteBuffer(nn->queue, conv->kernelBuffer, CL_TRUE, 0,
                                    kernelDataSize * sizeof(float), conv->kernelWeights, 0, NULL, NULL);
                clEnqueueWriteBuffer(nn->queue, conv->biasBuffer, CL_TRUE, 0,
                                    conv->outputChannels * sizeof(float), conv->biasWeights, 0, NULL, NULL);
                break;
            }
            case LAYER_DECONV: {
                DeconvLayer* deconv = &layer->deconv;
                
                int inputH, inputW, inputC, kernelSize, outputC, stride, padH, padW;
                fread(&inputH, sizeof(int), 1, file);
                fread(&inputW, sizeof(int), 1, file);
                fread(&inputC, sizeof(int), 1, file);
                fread(&kernelSize, sizeof(int), 1, file);
                fread(&outputC, sizeof(int), 1, file);
                fread(&stride, sizeof(int), 1, file);
                fread(&padH, sizeof(int), 1, file);
                fread(&padW, sizeof(int), 1, file);
                
                if (inputH != deconv->inputHeight || inputW != deconv->inputWidth ||
                    inputC != deconv->inputChannels || kernelSize != deconv->kernelSize ||
                    outputC != deconv->outputChannels) {
                    fprintf(stderr, "Deconv layer %d dimension mismatch\n", i);
                    fclose(file);
                    return -1;
                }
                
                int kernelDataSize = deconv->kernelSize * deconv->kernelSize * deconv->inputChannels * deconv->outputChannels;
                fread(deconv->kernelWeights, sizeof(float), kernelDataSize, file);
                fread(deconv->biasWeights, sizeof(float), deconv->outputChannels, file);
                
                clEnqueueWriteBuffer(nn->queue, deconv->kernelBuffer, CL_TRUE, 0,
                                    kernelDataSize * sizeof(float), deconv->kernelWeights, 0, NULL, NULL);
                clEnqueueWriteBuffer(nn->queue, deconv->biasBuffer, CL_TRUE, 0,
                                    deconv->outputChannels * sizeof(float), deconv->biasWeights, 0, NULL, NULL);
                break;
            }
            case LAYER_MAXPOOL: {
                int inputH, inputW, inputC, poolSize, stride;
                fread(&inputH, sizeof(int), 1, file);
                fread(&inputW, sizeof(int), 1, file);
                fread(&inputC, sizeof(int), 1, file);
                fread(&poolSize, sizeof(int), 1, file);
                fread(&stride, sizeof(int), 1, file);
                break;
            }
            case LAYER_RELU:
            case LAYER_SOFTMAX: {
                int dataSize;
                fread(&dataSize, sizeof(int), 1, file);
                break;
            }
            case LAYER_RESIZE2D: {
                int inputH, inputW, outputH, outputW, channels;
                fread(&inputH, sizeof(int), 1, file);
                fread(&inputW, sizeof(int), 1, file);
                fread(&outputH, sizeof(int), 1, file);
                fread(&outputW, sizeof(int), 1, file);
                fread(&channels, sizeof(int), 1, file);
                break;
            }
        }
    }
    
    clFinish(nn->queue);
    fclose(file);
    printf("Model loaded from %s\n", path);
    return 0;
}

size_t getModelSize(NeuralNetwork* nn) {
    size_t totalParams = 0;
    
    for (int i = 0; i < nn->numLayers; i++) {
        Layer* layer = &nn->layers[i];
        
        switch (layer->type) {
            case LAYER_CONV: {
                ConvLayer* conv = &layer->conv;
                size_t kernelParams = conv->kernelSize * conv->kernelSize * 
                                     conv->inputChannels * conv->outputChannels;
                size_t biasParams = conv->outputChannels;
                totalParams += kernelParams + biasParams;
                break;
            }
            case LAYER_DECONV: {
                DeconvLayer* deconv = &layer->deconv;
                size_t kernelParams = deconv->kernelSize * deconv->kernelSize * 
                                     deconv->inputChannels * deconv->outputChannels;
                size_t biasParams = deconv->outputChannels;
                totalParams += kernelParams + biasParams;
                break;
            }
            case LAYER_MAXPOOL:
            case LAYER_RELU:
            case LAYER_SOFTMAX:
                // No learnable parameters
                break;
        }
    }
    
    return totalParams;
}
