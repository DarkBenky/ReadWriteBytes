#ifndef ML_H
#define ML_H

#include <CL/cl.h>

typedef enum {
    LAYER_CONV,
    LAYER_DECONV,
    LAYER_MAXPOOL,
    LAYER_RELU,
    LAYER_SOFTMAX,
    LAYER_RESIZE2D
} LayerType;

typedef struct {
    int inputHeight;
    int inputWidth;
    int inputChannels;
    int kernelSize;
    int outputChannels;
    int stride;
    int paddingHeight;
    int paddingWidth;
    int outputHeight;
    int outputWidth;
    
    cl_mem inputBuffer;
    cl_mem kernelBuffer;
    cl_mem biasBuffer;
    cl_mem outputBuffer;
    cl_mem kernelGradBuffer;
    cl_mem biasGradBuffer;
    
    float* kernelWeights;
    float* biasWeights;
} ConvLayer;

typedef struct {
    int inputHeight;
    int inputWidth;
    int inputChannels;
    int kernelSize;
    int outputChannels;
    int stride;
    int paddingHeight;
    int paddingWidth;
    int outputHeight;
    int outputWidth;
    
    cl_mem inputBuffer;
    cl_mem kernelBuffer;
    cl_mem biasBuffer;
    cl_mem outputBuffer;
    cl_mem kernelGradBuffer;
    cl_mem biasGradBuffer;
    
    float* kernelWeights;
    float* biasWeights;
} DeconvLayer;

typedef struct {
    int inputHeight;
    int inputWidth;
    int inputChannels;
    int poolSize;
    int stride;
    int outputHeight;
    int outputWidth;
    
    cl_mem inputBuffer;
    cl_mem outputBuffer;
} MaxPoolLayer;

typedef struct {
    int dataSize;
    cl_mem inputBuffer;
    cl_mem outputBuffer;
} ActivationLayer;

typedef struct {
    int inputHeight;
    int inputWidth;
    int outputHeight;
    int outputWidth;
    int channels;
    
    cl_mem inputBuffer;
    cl_mem outputBuffer;
} Resize2DLayer;

typedef struct {
    LayerType type;
    union {
        ConvLayer conv;
        DeconvLayer deconv;
        MaxPoolLayer pool;
        ActivationLayer activation;
        Resize2DLayer resize;
    };
} Layer;

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_device_id device;
    
    cl_kernel convForwardKernel;
    cl_kernel convBackwardKernel;
    cl_kernel deconvForwardKernel;
    cl_kernel deconvBackwardKernel;
    cl_kernel maxPoolForwardKernel;
    cl_kernel maxPoolBackwardKernel;
    cl_kernel reluForwardKernel;
    cl_kernel reluBackwardKernel;
    cl_kernel softmaxForwardKernel;
    cl_kernel softmaxBackwardKernel;
    cl_kernel resize2DForwardKernel;
    cl_kernel resize2DBackwardKernel;
    cl_kernel mseLossKernel;
    cl_kernel maeLossKernel;
    cl_kernel ssimLossKernel;
    cl_kernel gradientLossKernel;
    cl_kernel fft1DKernel;
    cl_kernel transposeKernel;
    cl_kernel fftLossKernel;
    cl_kernel combinedLossKernel;
    cl_kernel combinedGradientKernel;
    
    // Persistent training buffers for GPU-only pipeline
    cl_mem trainTargetBuffer;
    cl_mem trainOutputBuffer;
    cl_mem trainLossBuffer;
    cl_mem trainGradBuffer;
    int trainBufferSize;
    
    Layer* layers;
    int numLayers;
} NeuralNetwork;

// OpenCL setup
int initOpenCL(NeuralNetwork* nn, const char* kernelSourcePath);
void cleanupOpenCL(NeuralNetwork* nn);

// Network operations
NeuralNetwork* createNetwork(int numLayers);
void addConvLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, int inputC, 
                  int kernelSize, int outputC, int stride, int padH, int padW);
void addDeconvLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, int inputC,
                    int kernelSize, int outputC, int stride, int padH, int padW);
void addMaxPoolLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, int inputC, 
                     int poolSize, int stride);
void addReluLayer(NeuralNetwork* nn, int layerIdx, int dataSize);
void addSoftmaxLayer(NeuralNetwork* nn, int layerIdx, int dataSize);
void addResize2DLayer(NeuralNetwork* nn, int layerIdx, int inputH, int inputW, 
                      int outputH, int outputW, int channels);

// Forward/backward
void forward(NeuralNetwork* nn, float* inputData);
void backward(NeuralNetwork* nn, float* gradOutput);
void updateWeights(NeuralNetwork* nn, float learningRate);
void updateWeightsGPU(NeuralNetwork* nn, float learningRate);

// Loss computation
float computeMSELoss(NeuralNetwork* nn, cl_mem predictions, cl_mem targets, int dataSize);
void computeMSEGradient(cl_mem predictions, cl_mem targets, float* gradient, int dataSize);
float computeCombinedLoss(NeuralNetwork* nn, float* output, float* target, int width, int height, int channels);
void computeCombinedGradient(float* output, float* target, float* gradient, int width, int height, int channels);

// GPU-only training pipeline (no CPU readbacks)
void initTrainBuffers(NeuralNetwork* nn, int width, int height, int channels);
void uploadTarget(NeuralNetwork* nn, float* target, int size);
float trainStepGPU(NeuralNetwork* nn, float* input, float learningRate, int width, int height, int channels);
void backwardFromGPU(NeuralNetwork* nn);
cl_mem getOutputBuffer(NeuralNetwork* nn);

// Model save/load
int saveModel(NeuralNetwork* nn, const char* path);
int loadModel(NeuralNetwork* nn, const char* path);

// Model utilities
size_t getModelSize(NeuralNetwork* nn);

// Cleanup
void freeNetwork(NeuralNetwork* nn);

#endif
