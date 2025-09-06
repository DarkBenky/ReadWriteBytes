#ifndef PARTICLESIM_H
#define PARTICLESIM_H

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#define NUM_PARTICLES 50000
#define GRAVITY 10.0f
#define DAMPING 0.985f
#define gridResolutionAxis 32
#define gridResolution (gridResolutionAxis * gridResolutionAxis * gridResolutionAxis)
#define temperature 8.5f
#define pressure temperature * 0.1f
#define PARTICLE_RADIUS 4

struct PointSOA {
    float x[NUM_PARTICLES];
    float y[NUM_PARTICLES];
    float z[NUM_PARTICLES];
    float xVelocity[NUM_PARTICLES];
    float yVelocity[NUM_PARTICLES];
    float zVelocity[NUM_PARTICLES];
    float totalVelocity[NUM_PARTICLES];
    float bBoxMin[3];
    float bBoxMax[3];
    int gridParticleID[gridResolution][NUM_PARTICLES / 8]; // Assuming a max of NUM_PARTICLES/8 per cell
    int gridParticleCount[gridResolution];
    float gridParticleAvgPos[gridResolution][3];
    float gridCellGradientPressure[gridResolution][3];
    float gridCellBlurredGradientPressure[gridResolution][3];
};


void Step(struct PointSOA *particles, float deltaTime); // Simulation step
void updateGridData(struct PointSOA *particles); // Update grid data

#endif