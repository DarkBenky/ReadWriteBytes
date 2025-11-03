#ifndef FLARE_H
#define FLARE_H

#include "../openGlShaders/gpuStruct.h"
#include <stdbool.h>
#include "../fireSim/fireSim.h"

struct Flare {
    struct FireSOA flareSim;
    float flareTemperature[NUM_FIRE_PARTICLES];
    float startingTemperature;
    float maxTemperature;
    float coolingRate;
    float lifeTimeRemaining;
    float burningRate;
    float maxLifeTime;
    float riseTimeAspect;
};

void InitializeFlare(struct Flare *flare);
void UpdateFlare(struct Flare *flare, float deltaTime);
void LunchFlare(struct Flare *flare, float position[3], float initialVelocity[3], float lunchDirection[3]);

#endif