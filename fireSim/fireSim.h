#ifndef FIRE_SIM_H
#define FIRE_SIM_H

#include "../openGlShaders/gpuStruct.h"

// Initialize fire simulation - all parameters passed directly
int initOpenCLFireSim(
    struct OpenCLContext *ocl, 
    const char *kernelSource, 
    int screenWidth, 
    int screenHeight, 
    float *basePosition,      // 3 floats: x, y, z
    float *startingColor,     // 3 floats: r, g, b
    float *fireColor,         // 3 floats: r, g, b
    float *smokeColor,        // 3 floats: r, g, b
    float maxLifeTime
);

// Simulate one step - returns timing if needed
void simulateFireStep(
    struct OpenCLContext *ocl, 
    int numParticles, 
    float deltaTime, 
    float *kernelTime
);

// Render particles with view/projection matrices
void renderFireParticles(
    struct OpenCLContext *ocl, 
    int numParticles, 
    int screenWidth, 
    int screenHeight,
    float *viewMatrix,        // 16 floats
    float *projMatrix,        // 16 floats
    float particleSize, 
    float *kernelTime
);

// Download rendered buffer to CPU
void downloadFireRenderBuffer(
    struct OpenCLContext *ocl, 
    int screenWidth, 
    int screenHeight, 
    float *outputBuffer
);

// Update fire parameters at runtime
void updateFireBasePosition(struct OpenCLContext *ocl, float *newBasePosition);
void updateFireColors(struct OpenCLContext *ocl, float *startingColor, float *fireColor, float *smokeColor);
void updateFireMaxLifeTime(struct OpenCLContext *ocl, float maxLifeTime);

// Cleanup
void cleanupFireSim(struct OpenCLContext *ocl);

#endif
