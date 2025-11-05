#ifndef FLARE_H
#define FLARE_H

#include "../fireSim/fireSim.h"

void InitializeFlare(struct Flare *flare);
void UpdateFlare(struct Flare *flare, float deltaTime);
void LunchFlare(struct Flare *flare, float position[3], float initialVelocity[3], float lunchDirection[3]);

#endif