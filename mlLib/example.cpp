/* example.cpp - Example usage of the CNN Denoising Library
 * 
 * This demonstrates both C-style (your custom library) and C++ style (DeepCL)
 * approaches to building and training a CNN for image denoising.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* ============================================================================
 * EXAMPLE 1: Using Your Custom C Library (cnn_denoise.h)
 * ============================================================================
 * 
 * This library supports:
 * - Non-square images (e.g., 800x600)
 * - Optimized float4 vectorization
 * - Full control over architecture
 * - C-style API for easy integration
 */

#if 0  /* Uncomment this block to use the custom C library */

#include "cnn_denoise.h"

int main(int argc, char** argv) {
    srand(time(NULL));
    
    printf("=== Custom CNN Denoising Library Example ===\n\n");
    
    /* Step 1: Configure the network */
    CNNConfig config = {
        .input_width = 800,
        .input_height = 600,
        .input_channels = 4,      /* Must be multiple of 4 for vectorization */
        .output_channels = 4,
        .learning_rate = 0.001f,
        .use_profiling = 1
    };
    
    /* Step 2: Create the denoiser */
    CNNDenoiser* cnn = cnn_create(config);
    if (!cnn) {
        printf("Failed to create CNN\n");
        return 1;
    }
    
    /* Step 3: Build encoder-decoder architecture */
    /* Encoder: gradually increase channels while extracting features */
    cnn_add_layer(cnn, (LayerConfig){4, 32, 1, "encoder_1"});    /* 4 -> 32 channels, ReLU */
    cnn_add_layer(cnn, (LayerConfig){32, 64, 1, "encoder_2"});   /* 32 -> 64 channels, ReLU */
    cnn_add_layer(cnn, (LayerConfig){64, 128, 1, "encoder_3"});  /* 64 -> 128 channels, ReLU */
    
    /* Bottleneck: maximum channel compression for abstract features */
    cnn_add_layer(cnn, (LayerConfig){128, 256, 1, "bottleneck"}); /* 128 -> 256 channels, ReLU */
    
    /* Decoder: gradually decrease channels while reconstructing */
    cnn_add_layer(cnn, (LayerConfig){256, 128, 1, "decoder_1"});  /* 256 -> 128 channels, ReLU */
    cnn_add_layer(cnn, (LayerConfig){128, 64, 1, "decoder_2"});   /* 128 -> 64 channels, ReLU */
    cnn_add_layer(cnn, (LayerConfig){64, 32, 1, "decoder_3"});    /* 64 -> 32 channels, ReLU */
    
    /* Output: reconstruct clean image */
    cnn_add_layer(cnn, (LayerConfig){32, 4, 0, "output"});        /* 32 -> 4 channels, Linear */
    
    /* Step 4: Finalize (allocates GPU memory) */
    cnn_finalize(cnn);
    cnn_print_architecture(cnn);
    
    /* Step 5: Prepare training data */
    int img_size = 800 * 600 * 4;
    float* clean_image = (float*)malloc(img_size * sizeof(float));
    float* noisy_image = (float*)malloc(img_size * sizeof(float));
    
    /* Generate synthetic clean image */
    for (int i = 0; i < img_size; i++) {
        clean_image[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    /* Add Gaussian noise */
    cnn_add_gaussian_noise(clean_image, noisy_image, img_size, 0.1f);
    
    /* Step 6: Train the network */
    printf("Training for 10 epochs...\n");
    for (int epoch = 0; epoch < 10; epoch++) {
        float loss = cnn_train_step(cnn, noisy_image, clean_image, 1);
        printf("Epoch %d: Loss = %.6f\n", epoch, loss);
    }
    
    /* Step 7: Use for inference */
    float* denoised = (float*)malloc(img_size * sizeof(float));
    cnn_denoise(cnn, noisy_image, denoised, 1);
    
    printf("\nDenoising complete!\n");
    
    /* Cleanup */
    free(clean_image);
    free(noisy_image);
    free(denoised);
    cnn_destroy(cnn);
    
    return 0;
}

#endif  /* End of custom C library example */


/* ============================================================================
 * EXAMPLE 2: Using DeepCL (C++ Library)
 * ============================================================================
 * 
 * DeepCL provides:
 * - High-level C++ API
 * - Automatic kernel selection
 * - Built-in optimizations
 * - BUT: Requires square images only
 */

#if 1  /* Uncomment this block to use DeepCL */

#include "DeepCL.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"

int main(int argc, char** argv) {
    srand(time(NULL));
    
    printf("=== DeepCL Example (Simple MNIST-like) ===\n\n");
    
    /* Step 1: Initialize OpenCL */
    EasyCL* cl = EasyCL::createForFirstGpuOtherwiseCpu();
    
    /* Step 2: Create neural network */
    NeuralNet* net = new NeuralNet(cl);
    
    /* Step 3: Build a simple classifier network */
    /* Input: 28x28 grayscale images */
    net->addLayer(InputLayerMaker::instance()->numPlanes(1)->imageSize(28));
    
    /* Convolutional layer: 1 -> 8 filters, 5x5 kernel */
    net->addLayer(ConvolutionalMaker::instance()
                  ->numFilters(8)
                  ->filterSize(5)
                  ->padZeros()
                  ->biased());
    
    /* Activation: ReLU */
    net->addLayer(ActivationMaker::instance()->relu());
    
    /* Pooling: 2x2, reduces 28x28 -> 14x14 */
    net->addLayer(PoolingMaker::instance()->poolingSize(2));
    
    /* Fully connected: flatten to 10 classes */
    net->addLayer(FullyConnectedMaker::instance()
                  ->numPlanes(10)
                  ->imageSize(1)
                  ->biased());
    
    /* Loss function */
    net->addLayer(SquareLossMaker::instance());
    
    printf("Network created with %d layers\n", net->getNumLayers());
    
    /* Step 4: Prepare training data */
    int batchSize = 128;
    int inputSize = 28 * 28;
    int numClasses = 10;
    
    float* inputData = (float*)malloc(batchSize * inputSize * sizeof(float));
    float* labels = (float*)malloc(batchSize * numClasses * sizeof(float));
    
    /* Generate random training data */
    int i;
    for (i = 0; i < batchSize * inputSize; i++) {
        inputData[i] = (float)rand() / RAND_MAX;
    }
    
    /* Create one-hot labels */
    memset(labels, 0, batchSize * numClasses * sizeof(float));
    for (i = 0; i < batchSize; i++) {
        labels[i * numClasses + (i % numClasses)] = 1.0f;
    }
    
    /* Step 5: Train */
    net->setBatchSize(batchSize);
    
    printf("\nTraining for 5 epochs...\n");
    for (int epoch = 0; epoch < 5; epoch++) {
        /* Forward pass */
        net->forward(inputData);
        
        /* Compute loss */
        float loss = net->calcLoss(labels);
        
        /* Backward pass (updates weights) */
        net->backward(labels);
        
        printf("Epoch %d: Loss = %.6f\n", epoch, loss);
    }
    
    printf("\nTraining complete!\n");
    
    /* Cleanup */
    free(inputData);
    free(labels);
    delete net;
    delete cl;
    
    return 0;
}

#endif  /* End of DeepCL example */


/* ============================================================================
 * COMPARISON: Custom C Library vs DeepCL
 * ============================================================================
 * 
 * Custom C Library (cnn_denoise.h):
 *   PROS:
 *   - Supports non-square images (800x600)
 *   - Explicit control over every operation
 *   - Optimized vectorized kernels (float4)
 *   - C-style API (easy to integrate)
 *   - Potentially faster for specific use cases
 *   
 *   CONS:
 *   - Requires manual architecture building
 *   - Limited to convolution layers
 *   - Need to implement more layer types yourself
 * 
 * DeepCL:
 *   PROS:
 *   - Rich layer types (Conv, FC, Pooling, Activation)
 *   - Automatic kernel selection
 *   - Well-tested and documented
 *   - Easy to prototype
 *   
 *   CONS:
 *   - Requires square images only
 *   - Less control over low-level optimizations
 *   - C++ dependency
 *   - May have overhead from abstraction
 * 
 * RECOMMENDATION:
 * - Use DeepCL for: Rapid prototyping, research, standard architectures
 * - Use Custom Library for: Production denoising, non-square images, maximum performance
 * ============================================================================
 */
