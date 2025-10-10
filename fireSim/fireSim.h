#ifndef FIRE_SIM_H
#define FIRE_SIM_H

#include <CL/cl.h>

#define FIRE_PARTICLES 1000

struct Particles {
    float posX[FIRE_PARTICLES];
    float posY[FIRE_PARTICLES];
    float posZ[FIRE_PARTICLES];
    float velX[FIRE_PARTICLES];
    float velY[FIRE_PARTICLES];
    float velZ[FIRE_PARTICLES];
    float lifeTime[FIRE_PARTICLES];
    float basePos[3];
    float baseColor[3];
    float fireColor[3];
    float SmokeColor[3];
    float maxLifeTime;
    float particleSize;
};

struct OpenCLContextFireSim {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernelUpdateParticles;
    cl_kernel kernelRenderParticles;
    cl_kernel kernelBlurFire;
    cl_kernel kernelFindMaxDepth;
    cl_kernel kernelNormalizeDepth;
    cl_mem posX;
    cl_mem posY;
    cl_mem posZ;
    cl_mem velX;
    cl_mem velY;
    cl_mem velZ;
    cl_mem lifeTime;
    cl_mem buffer_color;
    cl_mem buffer_depth;
    cl_mem buffer_temp;
    cl_mem maxDepth;
};


int initOpenCLFireSim(struct OpenCLContextFireSim* cl, const char* kernelSource, int screenWidth, int screenHeight, struct Particles* particles, cl_context sharedContext, cl_device_id sharedDevice, cl_command_queue sharedQueue);

void stepFireSimulation(struct OpenCLContextFireSim* cl, struct Particles* particles, float deltaTime);

void renderFireParticles(struct OpenCLContextFireSim* cl, struct Particles* particles, 
                        int screenWidth, int screenHeight, float* viewMatrix, float* projMatrix);

void normalizeFireDepth(struct OpenCLContextFireSim* cl, int screenWidth, int screenHeight);

void cleanupFireSim(struct OpenCLContextFireSim* cl);

#endif
