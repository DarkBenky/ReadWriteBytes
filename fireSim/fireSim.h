#ifndef FIRE_SIM_H
#define FIRE_SIM_H

#include <CL/cl.h>

#define NUM_PARTICLES 1000

struct Particles {
    float posX[NUM_PARTICLES];
    float posY[NUM_PARTICLES];
    float posZ[NUM_PARTICLES];
    float velX[NUM_PARTICLES];
    float velY[NUM_PARTICLES];
    float velZ[NUM_PARTICLES];
    float lifeTime[NUM_PARTICLES];
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
};

void initFireSimulation(struct Particles* particles);

int initOpenCLFireSim(struct OpenCLContextFireSim* cl, const char* kernelSource);

void stepFireSimulation(struct OpenCLContextFireSim* cl, struct Particles* particles, float deltaTime);

void renderFireParticles(struct OpenCLContextFireSim* cl, struct Particles* particles, 
                        int screenWidth, int screenHeight, float* viewMatrix, float* projMatrix);

void cleanupFireSim(struct OpenCLContextFireSim* cl);

#endif
