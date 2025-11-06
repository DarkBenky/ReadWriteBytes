#include "../fireSim/fireSim.h"
#include "flare.h"
#include <stdbool.h>
#include <string.h>

static float calculateTemperature(float riseTimeAspect, float lifeTimeRemaining, float maxLifeTime,
								  float maxTemperature, float coolingRate, float burnRate, float currentTemperature) {
	if (lifeTimeRemaining <= 0.0f) {
		return 0.0f;
	}

	float lifeRatio = lifeTimeRemaining / maxLifeTime;

	if (lifeRatio > riseTimeAspect) {
		float riseProgress = (lifeRatio - riseTimeAspect) / (1.0f - riseTimeAspect);
		currentTemperature += (maxTemperature - currentTemperature) * (1.0f - riseProgress);
	} else {
		currentTemperature -= (coolingRate * (1.0f - lifeRatio) + burnRate);
		if (currentTemperature < 0.0f) {
			currentTemperature = 0.0f;
		}
	}

	return currentTemperature;
}

void InitializeFlare(struct Flare *flare) {
	if (!flare) return;

	memset(flare, 0, sizeof(struct Flare));

	InitializeFireParticles(&flare->flareSim);

	flare->flareSim.basePosition[0] = 0.0f;
	flare->flareSim.basePosition[1] = 0.0f;
	flare->flareSim.basePosition[2] = 0.0f;

	flare->flareSim.startingColor[0] = 1.0f;
	flare->flareSim.startingColor[1] = 0.95f;
	flare->flareSim.startingColor[2] = 0.8f;

	flare->flareSim.fireColor[0] = 1.0f;
	flare->flareSim.fireColor[1] = 0.4f;
	flare->flareSim.fireColor[2] = 0.1f;

	flare->flareSim.smokeColor[0] = 0.3f;
	flare->flareSim.smokeColor[1] = 0.3f;
	flare->flareSim.smokeColor[2] = 0.3f;

	flare->flareSim.windDirection[0] = 0.0f;
	flare->flareSim.windDirection[1] = -5.0f;
	flare->flareSim.windDirection[2] = 0.0f;

	flare->flareSim.maxLifeTime = 8.0f;
	flare->flareSim.buoyancy = 2.0f;
	flare->flareSim.drag = 0.3f;
	flare->flareSim.turbulence = 1.5f;
	flare->flareSim.maxVelocity = 15.0f;
	flare->flareSim.particlesSize = 8.0f;
	flare->flareSim.maxDistance = 50.0f;
	flare->flareSim.swirlIntensity = 0.5f;
	flare->flareSim.swirlFrequency = 0.3f;

	flare->startingTemperature = 300.0f;
	flare->maxTemperature = 2200.0f;
	flare->coolingRate = 150.0f;
	flare->burningRate = 50.0f;
	flare->maxLifeTime = 8.0f;
	flare->lifeTimeRemaining = 0.0f;
	flare->riseTimeAspect = 0.15f;

	for (int i = 0; i < NUM_FIRE_PARTICLES; i++) {
		flare->flareTemperature[i] = flare->startingTemperature;
	}
}

void LunchFlare(struct Flare *flare, float *position, float *initialVelocity, float *lunchDirection) {
    if (!flare) return;

    // Set flare position
    flare->flareSim.basePosition[0] = position[0];
    flare->flareSim.basePosition[1] = position[1];
    flare->flareSim.basePosition[2] = position[2];

    // Initialize particles around the flare position
    for (int i = 0; i < NUM_FIRE_PARTICLES; i++) {
        flare->flareSim.x[i] = flare->flareSim.basePosition[0] + randRange(-1.0f, 1.0f) * 2.0f;
        flare->flareSim.y[i] = flare->flareSim.basePosition[1] + randRange(-1.0f, 1.0f) * 2.0f;
        flare->flareSim.z[i] = flare->flareSim.basePosition[2] + randRange(-1.0f, 1.0f) * 2.0f;

        flare->flareSim.xVelocity[i] = initialVelocity[0] + lunchDirection[0] * randRange(5.0f, 15.0f);
        flare->flareSim.yVelocity[i] = initialVelocity[1] + lunchDirection[1] * randRange(5.0f, 15.0f);
        flare->flareSim.zVelocity[i] = initialVelocity[2] + lunchDirection[2] * randRange(5.0f, 15.0f);

        flare->flareSim.lifeTime[i] = 0.0f;
        flare->flareTemperature[i] = flare->startingTemperature;
    }

    // Reset flare lifetime
    flare->lifeTimeRemaining = flare->maxLifeTime;
}

void UpdateFlare(struct Flare *flare, float deltaTime) {
	if (!flare || flare->lifeTimeRemaining <= 0.0f) return;

	flare->lifeTimeRemaining -= deltaTime;

	float timeTook = 0.0f;
	fireSimStep(&flare->flareSim, deltaTime, &timeTook);

	float averageVelocity[3] = {0.0f, 0.0f, 0.0f};

	for (int i = 0; i < NUM_FIRE_PARTICLES; i++) {
		float particleLifeRatio = flare->flareSim.lifeTime[i] / flare->flareSim.maxLifeTime;

		if (particleLifeRatio > 0.0f) {
			flare->flareTemperature[i] = calculateTemperature(
				flare->riseTimeAspect,
				flare->lifeTimeRemaining,
				flare->maxLifeTime,
				flare->maxTemperature,
				flare->coolingRate,
				flare->burningRate,
				flare->flareTemperature[i]);
			averageVelocity[0] += flare->flareSim.xVelocity[i];
			averageVelocity[1] += flare->flareSim.yVelocity[i];
			averageVelocity[2] += flare->flareSim.zVelocity[i];
		} else {
			flare->flareTemperature[i] = flare->startingTemperature;
		}
	}
	averageVelocity[0] /= NUM_FIRE_PARTICLES;
	averageVelocity[1] /= NUM_FIRE_PARTICLES;
	averageVelocity[2] /= NUM_FIRE_PARTICLES;

	flare->flareSim.basePosition[0] += averageVelocity[0] * deltaTime;
	flare->flareSim.basePosition[1] += averageVelocity[1] * deltaTime;
	flare->flareSim.basePosition[2] += averageVelocity[2] * deltaTime;
}