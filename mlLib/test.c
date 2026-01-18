#define _GNU_SOURCE
#include "ml.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define HIGH_RES_IMAGE_WIDTH 1200
#define HIGH_RES_IMAGE_HEIGHT 900
#define LOW_RES_IMAGE_WIDTH 800
#define LOW_RES_IMAGE_HEIGHT 600
#define IMAGES_IN_MEMORY 512
#define DATASET_PATH "/media/user/2TB/imageData"

struct Image {
    int width;
    int height;
    int channels;
    float* data;
};

struct Dataset {
    struct Image HighResImages[IMAGES_IN_MEMORY];
    struct Image LowResImages[IMAGES_IN_MEMORY];
    int currentIndex;
};

void fillDateSet(struct Dataset* dataset) {
    printf("fillDateSet: Starting...\n"); fflush(stdout);
    
    // Free existing images if already loaded
    for (int i = 0; i < IMAGES_IN_MEMORY; i++) {
        if (dataset->HighResImages[i].data) {
            free(dataset->HighResImages[i].data);
            dataset->HighResImages[i].data = NULL;
        }
        if (dataset->LowResImages[i].data) {
            free(dataset->LowResImages[i].data);
            dataset->LowResImages[i].data = NULL;
        }
    }
    
    printf("fillDateSet: Loading images...\n"); fflush(stdout);
    
    // Load images directly by index
    for (int i = 0; i < IMAGES_IN_MEMORY; i++) {
        printf("Loading image %d/%d\n", i + 1, IMAGES_IN_MEMORY); fflush(stdout);
        // Generate random image index (assuming numbered directories)
        int imageIdx = rand() % 10000 + 1;  // Adjust range as needed
        char highResPath[512];
        char lowResPath[512];
        
        snprintf(highResPath, sizeof(highResPath), "%s/image_%08d/high_res.png", DATASET_PATH, imageIdx);
        snprintf(lowResPath, sizeof(lowResPath), "%s/image_%08d/low_res.png", DATASET_PATH, imageIdx);
        
        // Load high res image
        int width, height, channels;
        unsigned char* highResData = stbi_load(highResPath, &width, &height, &channels, 3);
        if (!highResData) {
            fprintf(stderr, "Failed to load %s\n", highResPath);
            continue;
        } else {
            if (i == 0) {
                printf("First image: %dx%d, channels=%d\n", width, height, channels);
            }
            dataset->HighResImages[i].width = width;
            dataset->HighResImages[i].height = height;
            dataset->HighResImages[i].channels = 3;
            
            int dataSize = width * height * 3;
            dataset->HighResImages[i].data = (float*)malloc(dataSize * sizeof(float));
            if (!dataset->HighResImages[i].data) {
                fprintf(stderr, "Failed to allocate %d floats for high-res image\n", dataSize);
                stbi_image_free(highResData);
                continue;
            }
            
            // Normalize to [0, 1]
            for (int j = 0; j < dataSize; j++) {
                dataset->HighResImages[i].data[j] = highResData[j] / 255.0f;
            }
            stbi_image_free(highResData);
        }
        
        // Load low res image
        unsigned char* lowResData = stbi_load(lowResPath, &width, &height, &channels, 3);
        if (!lowResData) {
            fprintf(stderr, "Failed to load %s\n", lowResPath);
            // remove previously loaded high-res image
            free(dataset->HighResImages[i].data);
            dataset->HighResImages[i].data = NULL;
            continue;
        } else {
            dataset->LowResImages[i].width = width;
            dataset->LowResImages[i].height = height;
            dataset->LowResImages[i].channels = 3;
            
            int dataSize = width * height * 3;
            dataset->LowResImages[i].data = (float*)malloc(dataSize * sizeof(float));
            if (!dataset->LowResImages[i].data) {
                fprintf(stderr, "Failed to allocate %d floats for low-res image\n", dataSize);
                stbi_image_free(lowResData);
                continue;
            }
            
            // Normalize to [0, 1]
            for (int j = 0; j < dataSize; j++) {
                dataset->LowResImages[i].data[j] = lowResData[j] / 255.0f;
            }
            stbi_image_free(lowResData);
        }
    }
    
    dataset->currentIndex = 0;
    printf("fillDateSet: Completed loading %d images\n", IMAGES_IN_MEMORY);
    fflush(stdout);
}

void returnSample(struct Dataset* dataset, float** lowResImage, float** highResImage) {
    if (dataset->currentIndex >= IMAGES_IN_MEMORY) {
        // load more images into dataset
        fillDateSet(dataset);
    }
    *lowResImage = dataset->LowResImages[dataset->currentIndex].data;
    *highResImage = dataset->HighResImages[dataset->currentIndex].data;
    dataset->currentIndex++;
}

int main() {
    srand(time(NULL));
    
    printf("Building ~20M parameter super-resolution network...\n");
    printf("Architecture: Encoder -> Middle (high channels) -> Resize2D -> Decoder\n");
    fflush(stdout);
    
    // Architecture for ~20M params:
    // Encoder: 3->64->128->256->512 (downsampling via stride=2)
    // Middle: 512->768->768->512 (bottleneck with high channels)
    // Resize2D: bilinear upscale from 75x100 to 900x1200
    // Decoder: 512->256->128->64->32->3 (fix artifacts)
    
    int numLayers = 36;
    NeuralNetwork* nn = createNetwork(numLayers);
    
    printf("Network structure allocated\n");
    fflush(stdout);
    
    if (initOpenCL(nn, "ml.cl") != 0) {
        fprintf(stderr, "Failed to initialize OpenCL\n");
        return -1;
    }
    
    printf("OpenCL initialized\n");
    fflush(stdout);
    
    int layerIdx = 0;
    
    // ========== ENCODER ==========
    // Input: 600x800x3
    addConvLayer(nn, layerIdx++, 600, 800, 3, 3, 64, 1, 1, 1);    // 600x800x64
    addReluLayer(nn, layerIdx++, 600 * 800 * 64);                  // params: 1,792
    
    addConvLayer(nn, layerIdx++, 600, 800, 64, 3, 64, 1, 1, 1);   // 600x800x64
    addReluLayer(nn, layerIdx++, 600 * 800 * 64);                  // params: 36,928
    
    addConvLayer(nn, layerIdx++, 600, 800, 64, 3, 128, 2, 1, 1);  // 300x400x128 (stride=2)
    addReluLayer(nn, layerIdx++, 300 * 400 * 128);                 // params: 73,856
    
    addConvLayer(nn, layerIdx++, 300, 400, 128, 3, 128, 1, 1, 1); // 300x400x128
    addReluLayer(nn, layerIdx++, 300 * 400 * 128);                 // params: 147,584
    
    addConvLayer(nn, layerIdx++, 300, 400, 128, 3, 256, 2, 1, 1); // 150x200x256 (stride=2)
    addReluLayer(nn, layerIdx++, 150 * 200 * 256);                 // params: 295,168
    
    addConvLayer(nn, layerIdx++, 150, 200, 256, 3, 256, 1, 1, 1); // 150x200x256
    addReluLayer(nn, layerIdx++, 150 * 200 * 256);                 // params: 590,080
    
    addConvLayer(nn, layerIdx++, 150, 200, 256, 3, 512, 2, 1, 1); // 75x100x512 (stride=2)
    addReluLayer(nn, layerIdx++, 75 * 100 * 512);                  // params: 1,180,160
    
    // ========== MIDDLE (high channel bottleneck) ==========
    addConvLayer(nn, layerIdx++, 75, 100, 512, 3, 768, 1, 1, 1);  // 75x100x768
    addReluLayer(nn, layerIdx++, 75 * 100 * 768);                  // params: 3,539,712
    
    addConvLayer(nn, layerIdx++, 75, 100, 768, 3, 768, 1, 1, 1);  // 75x100x768
    addReluLayer(nn, layerIdx++, 75 * 100 * 768);                  // params: 5,309,184
    
    addConvLayer(nn, layerIdx++, 75, 100, 768, 3, 768, 1, 1, 1);  // 75x100x768
    addReluLayer(nn, layerIdx++, 75 * 100 * 768);                  // params: 5,309,184
    
    addConvLayer(nn, layerIdx++, 75, 100, 768, 3, 512, 1, 1, 1);  // 75x100x512
    addReluLayer(nn, layerIdx++, 75 * 100 * 512);                  // params: 3,539,456
    
    // ========== RESIZE2D (bilinear upscale) ==========
    addResize2DLayer(nn, layerIdx++, 75, 100, 900, 1200, 512);    // 900x1200x512 (no params)
    
    // ========== DECODER (fix interpolation artifacts) ==========
    addConvLayer(nn, layerIdx++, 900, 1200, 512, 3, 256, 1, 1, 1); // 900x1200x256
    addReluLayer(nn, layerIdx++, 900 * 1200 * 256);                // params: 1,179,904
    
    addConvLayer(nn, layerIdx++, 900, 1200, 256, 3, 128, 1, 1, 1); // 900x1200x128
    addReluLayer(nn, layerIdx++, 900 * 1200 * 128);                // params: 295,040
    
    addConvLayer(nn, layerIdx++, 900, 1200, 128, 3, 64, 1, 1, 1);  // 900x1200x64
    addReluLayer(nn, layerIdx++, 900 * 1200 * 64);                 // params: 73,792
    
    addConvLayer(nn, layerIdx++, 900, 1200, 64, 3, 32, 1, 1, 1);   // 900x1200x32
    addReluLayer(nn, layerIdx++, 900 * 1200 * 32);                 // params: 18,464
    
    addConvLayer(nn, layerIdx++, 900, 1200, 32, 3, 3, 1, 1, 1);    // 900x1200x3 (final RGB)
    // No ReLU on final layer - want full color range
    
    printf("Network created with %d layers (actual: %d)\n", numLayers, layerIdx);
    
    size_t modelSize = getModelSize(nn);
    printf("Total parameters: %zu (~%.1fM, %.2f MB)\n", 
           modelSize, modelSize / 1e6, modelSize * sizeof(float) / (1024.0 * 1024.0));
    
    // Try to load existing checkpoint
    printf("Attempting to load checkpoint...\n"); fflush(stdout);
    if (loadModel(nn, "model_best.bin") == 0) {
        printf("Resumed from checkpoint\n");
    } else {
        printf("Starting fresh training\n");
    }
    
    // Initialize dataset
    printf("Allocating dataset structure...\n"); fflush(stdout);
    struct Dataset* dataset = (struct Dataset*)calloc(1, sizeof(struct Dataset));
    if (!dataset) {
        fprintf(stderr, "Failed to allocate dataset\n");
        return -1;
    }
    printf("Loading dataset from %s...\n", DATASET_PATH);
    fflush(stdout);
    fillDateSet(dataset);
    printf("Dataset loaded successfully\n");
    fflush(stdout);
    
    // Training config
    float learningRate = 0.000175f;
    int numEpochs = 100;
    float bestLoss = 1e10f;
    
    // Output dimensions (network outputs exactly target size now)
    int targetWidth = HIGH_RES_IMAGE_WIDTH;   // 1200
    int targetHeight = HIGH_RES_IMAGE_HEIGHT; // 900
    int targetChannels = 3;
    int targetSize = targetWidth * targetHeight * targetChannels;
    
    // Initialize GPU training buffers (avoids allocations in training loop)
    initTrainBuffers(nn, targetWidth, targetHeight, targetChannels);
    
    // Open FIFO for wandb logging (non-blocking)
    FILE* wandb_pipe = NULL;
    const char* fifo_path = "/tmp/ml_metrics.fifo";
    // Try to open FIFO in non-blocking mode
    int fd = open(fifo_path, O_WRONLY | O_NONBLOCK);
    if (fd >= 0) {
        wandb_pipe = fdopen(fd, "w");
        if (wandb_pipe) {
            printf("Connected to wandb logger\n");
            setbuf(wandb_pipe, NULL); // Unbuffered for immediate logging
        }
    } else {
        printf("wandb logger not running (FIFO not available), skipping logging\n");
    }
    
    printf("Starting GPU-optimized training...\n");
    printf("Input: %dx%dx3 -> Output: %dx%dx3\n",
           LOW_RES_IMAGE_WIDTH, LOW_RES_IMAGE_HEIGHT,
           targetWidth, targetHeight);
    printf("Loss: MAE(0.1) + SSIM(0.5) + Gradient/FFT(0.4)\n");
    fflush(stdout);
    
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        int numSamples = 0;
        double totalForwardTime = 0.0;
        
        struct timespec epoch_start, epoch_end;
        clock_gettime(CLOCK_MONOTONIC, &epoch_start);
        
        for (int sample = 0; sample < IMAGES_IN_MEMORY; sample++) {
            float* lowRes = dataset->LowResImages[sample].data;
            float* highRes = dataset->HighResImages[sample].data;
            
            if (!lowRes || !highRes) continue;
            
            // Upload target to GPU (async, overlaps with forward pass)
            uploadTarget(nn, highRes, targetSize);
            
            // Time the forward pass (includes loss, backward, update)
            struct timespec forward_start, forward_end;
            clock_gettime(CLOCK_MONOTONIC, &forward_start);
            
            // GPU-only training step: forward -> loss+gradient -> backward -> update
            // Only reads back scalar loss value
            float loss = trainStepGPU(nn, lowRes, learningRate, targetWidth, targetHeight, targetChannels);
            
            clock_gettime(CLOCK_MONOTONIC, &forward_end);
            double forwardTime = (forward_end.tv_sec - forward_start.tv_sec) + 
                                (forward_end.tv_nsec - forward_start.tv_nsec) / 1e9;
            totalForwardTime += forwardTime;
            
            epochLoss += loss;
            numSamples++;
            
            if (sample % 50 == 0) {
                printf("  Sample %d/%d, Loss: %.6f, Forward FPS: %.1f\n", 
                       sample + 1, IMAGES_IN_MEMORY, loss, 1.0 / forwardTime);
                fflush(stdout);
            }
        }
        
        clock_gettime(CLOCK_MONOTONIC, &epoch_end);
        double epochTime = (epoch_end.tv_sec - epoch_start.tv_sec) + 
                          (epoch_end.tv_nsec - epoch_start.tv_nsec) / 1e9;
        
        float avgLoss = epochLoss / numSamples;
        double avgForwardTime = totalForwardTime / numSamples;
        double forwardFPS = 1.0 / avgForwardTime;
        
        printf("Epoch %d: Loss=%.6f (best=%.6f), Time=%.1fs, Samples/sec=%.1f, Forward FPS=%.1f\n", 
               epoch + 1, avgLoss, bestLoss, epochTime, numSamples / epochTime, forwardFPS);
        fflush(stdout);
        
        // Log to wandb if connected
        if (wandb_pipe) {
            fprintf(wandb_pipe, "{\"epoch\": %d, \"loss\": %.6f, \"best_loss\": %.6f, "
                   "\"epoch_time\": %.2f, \"samples_per_sec\": %.2f, \"forward_fps\": %.2f, "
                   "\"avg_forward_time_ms\": %.2f, \"learning_rate\": %.6f}\n",
                   epoch + 1, avgLoss, bestLoss, epochTime, numSamples / epochTime, 
                   forwardFPS, avgForwardTime * 1000.0, learningRate);
        }
        fflush(stdout);
        
        fflush(stdout);
        
        if (avgLoss < bestLoss) {
            bestLoss = avgLoss;
            saveModel(nn, "model_best.bin");
            printf("  -> New best model saved!\n");
        }
        
        if (epoch % 10 == 0) {
            char checkpointPath[64];
            snprintf(checkpointPath, sizeof(checkpointPath), "model_epoch_%d.bin", epoch);
            saveModel(nn, checkpointPath);
        }
        
    if (wandb_pipe) {
        fclose(wandb_pipe);
    }
    
        // Reload dataset for next epoch
        if (epoch < numEpochs - 1) {
            fillDateSet(dataset);
        }
    }
    
    // Cleanup
    for (int i = 0; i < IMAGES_IN_MEMORY; i++) {
        free(dataset->HighResImages[i].data);
        free(dataset->LowResImages[i].data);
    }
    free(dataset);
    
    freeNetwork(nn);
    printf("Training complete!\n");
    return 0;
}
