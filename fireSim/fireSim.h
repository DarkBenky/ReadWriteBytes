#ifndef FIRE_SIM_H
#define FIRE_SIM_H
#define NUM_FIRE_PARTICLES 1000
#define G 9.81f

#include "../openGlShaders/gpuStruct.h"

struct FireSOA {
	float x[NUM_FIRE_PARTICLES];
	float y[NUM_FIRE_PARTICLES];
	float z[NUM_FIRE_PARTICLES];
	float xVelocity[NUM_FIRE_PARTICLES];
	float yVelocity[NUM_FIRE_PARTICLES];
	float zVelocity[NUM_FIRE_PARTICLES];
	float lifeTime[NUM_FIRE_PARTICLES];
	float basePosition[3];
	float startingColor[3];
	float fireColor[3];
	float smokeColor[3];
	float windDirection[3];
	float maxLifeTime;
	float buoyancy;
	float drag;
	float turbulence;
	float maxVelocity;
	float particlesSize;
	float maxDistance;
	float swirlIntensity;
	float swirlFrequency;
};

void InitializeFireParticles(struct FireSOA *particles);
void fireSimStep(struct FireSOA *particles, float deltaTime, float *timeTook);

#endif
