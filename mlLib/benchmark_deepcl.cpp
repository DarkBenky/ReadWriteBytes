#include "DeepCL.h"
#include "net/NeuralNet.h"
#include "layer/Layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

typedef struct {
    double forward_time;
    double backward_time;
    double loss_time;
    double total_time;
} TimingStats;

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char** argv) {
    srand(time(NULL));
    
    printf("========================================\n");
    printf("  DeepCL CNN Denoising Benchmark       \n");
    printf("========================================\n");
    
    /* Configuration */
    int IMG_SIZE = 800;
    int CHANNELS = 3;
    int BATCH_SIZE = 4;
    int EPOCHS = 20;
    
    printf("\nConfiguration:\n");
    printf("  Image size: %dx%dx%d (square required by DeepCL)\n", IMG_SIZE, IMG_SIZE, CHANNELS);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Epochs: %d\n", EPOCHS);
    printf("  Target: ~5M parameters\n\n");
    
    EasyCL* cl = EasyCL::createForFirstGpuOtherwiseCpu();
    NeuralNet* net = new NeuralNet(cl);
    
    /* Encoder-Decoder architecture for denoising
     * Input: 800x800x3 -> bottleneck -> Output: 800x800x3
     */
    
    printf("Building network...\n");
    
    /* INPUT */
    net->addLayer(InputLayerMaker::instance()->numPlanes(CHANNELS)->imageSize(IMG_SIZE));
    
    /* ENCODER */
    /* 3 -> 32 channels */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    
    /* 32 -> 64 channels + pooling 800->400 */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(64)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    net->addLayer(PoolingMaker::instance()->poolingSize(2));
    
    /* 64 -> 128 channels + pooling 400->200 */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(128)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    net->addLayer(PoolingMaker::instance()->poolingSize(2));
    
    /* BOTTLENECK: 128 -> 256 channels */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(256)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    
    /* DECODER */
    /* 256 -> 128 channels */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(128)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    
    /* 128 -> 64 channels */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(64)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    
    /* 64 -> 32 channels */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(32)->filterSize(3)->padZeros()->biased());
    net->addLayer(ActivationMaker::instance()->relu());
    
    /* OUTPUT: 32 -> 3 channels */
    net->addLayer(ConvolutionalMaker::instance()->numFilters(CHANNELS)->filterSize(3)->padZeros()->biased());
    
    /* LOSS */
    net->addLayer(SquareLossMaker::instance());
    
    printf("Network built with %d layers\n", net->getNumLayers());
    
    /* Calculate final output size (pooling reduces dimensions) */
    int finalSize = IMG_SIZE / 4;  /* 2 pooling layers */
    
    printf("\nWARNING: Output will be %dx%d due to pooling layers\n", finalSize, finalSize);
    printf("DeepCL doesn't have built-in upsampling/transpose convolution\n\n");
    
    /* Allocate data */
    int inputDataSize = BATCH_SIZE * IMG_SIZE * IMG_SIZE * CHANNELS;
    int outputDataSize = BATCH_SIZE * finalSize * finalSize * CHANNELS;
    
    printf("Allocating %.2f MB for input, %.2f MB for output\n", 
           inputDataSize * 4 / 1024.0 / 1024.0,
           outputDataSize * 4 / 1024.0 / 1024.0);
    
    float* inputData = (float*)malloc(inputDataSize * sizeof(float));
    float* targetData = (float*)malloc(outputDataSize * sizeof(float));
    
    /* Generate random data */
    int i;
    for (i = 0; i < inputDataSize; i++) {
        inputData[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (i = 0; i < outputDataSize; i++) {
        targetData[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    net->setBatchSize(BATCH_SIZE);
    
    printf("\nWarming up (5 iterations)...\n");
    for (int warmup = 0; warmup < 5; warmup++) {
        net->forward(inputData);
        net->calcLoss(targetData);
        net->backward(targetData);
        printf(".");
        fflush(stdout);
    }
    printf(" done!\n\n");
    
    printf("Starting training...\n");
    printf("Epoch | Forward(ms) | Backward(ms) | Loss(ms) | Total(ms) | Loss Value\n");
    printf("------|-------------|--------------|----------|-----------|------------\n");
    
    /* Warmup */
    net->forward(inputData);
    net->backward(targetData);
    
    TimingStats stats = {0};
    int measured_count = 0;
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double t_start, t_end;
        double epoch_forward, epoch_backward, epoch_loss, epoch_total;
        
        /* Forward pass */
        t_start = get_time_ms();
        net->forward(inputData);
        t_end = get_time_ms();
        epoch_forward = t_end - t_start;
        
        /* Loss calculation */
        t_start = get_time_ms();
        float loss = net->calcLoss(targetData);
        t_end = get_time_ms();
        epoch_loss = t_end - t_start;
        
        /* Backward pass */
        t_start = get_time_ms();
        net->backward(targetData);
        t_end = get_time_ms();
        epoch_backward = t_end - t_start;
        
        epoch_total = epoch_forward + epoch_backward + epoch_loss;
        
        printf("%5d | %11.2f | %12.2f | %8.2f | %9.2f | %.6f\n",
               epoch, epoch_forward, epoch_backward, epoch_loss, epoch_total, loss);
        
        /* Measure all epochs (warmup already done separately) */
        stats.forward_time += epoch_forward;
        stats.backward_time += epoch_backward;
        stats.loss_time += epoch_loss;
        stats.total_time += epoch_total;
        measured_count++;
    }
    
    if (measured_count > 0) {
        stats.forward_time /= measured_count;
        stats.backward_time /= measured_count;
        stats.loss_time /= measured_count;
        stats.total_time /= measured_count;
    }
    
    printf("\n=== Average Timings (excluding warmup) ===\n");
    printf("Forward pass:   %.2f ms (%.1f%%)\n", stats.forward_time, 
           100.0 * stats.forward_time / stats.total_time);
    printf("Backward pass:  %.2f ms (%.1f%%)\n", stats.backward_time,
           100.0 * stats.backward_time / stats.total_time);
    printf("Loss calc:      %.2f ms (%.1f%%)\n", stats.loss_time,
           100.0 * stats.loss_time / stats.total_time);
    printf("----------------\n");
    printf("Total per iter: %.2f ms\n", stats.total_time);
    printf("Throughput:     %.2f images/sec\n", BATCH_SIZE * 1000.0 / stats.total_time);
    printf("=====================================\n");
    
    printf("\n=== Performance Breakdown ===\n");
    printf("Time per image: %.2f ms\n", stats.total_time / BATCH_SIZE);
    printf("Forward/image:  %.2f ms\n", stats.forward_time / BATCH_SIZE);
    printf("Backward/image: %.2f ms\n", stats.backward_time / BATCH_SIZE);
    
    free(inputData);
    free(targetData);
    delete net;
    delete cl;
    
    printf("\nBenchmark complete!\n");
    
    printf("\nNOTE: To compare with custom OpenCL:\n");
    printf("  Run your implementation with similar parameters\n");
    printf("  Compare timing breakdown for each phase\n");
    printf("  Your code supports 800x600, DeepCL requires 800x800\n");
    
    return 0;
}
