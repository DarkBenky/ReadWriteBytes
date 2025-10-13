#include "fireSim.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <CL/cl.h>
#define G 9.81f
#define min(a, b) ((a) < (b) ? (a) : (b))

#define SPEED_OF_SOUND 340.29f
#define SEA_LEVEL_DENSITY 1.225f
#define SCALE_HEIGHT 8500.0f

float randRange(float min, float max) {
	float scale = rand() / (float)RAND_MAX;
	return min + scale * (max - min);
}

float getAirDensity(float altitude) {
	if (altitude < 0.0f) altitude = 0.0f;
	return SEA_LEVEL_DENSITY * expf(-altitude / SCALE_HEIGHT);
}

float getMachDragMultiplier(float mach) {
	if (mach < 0.8f) {
		return 1.0f;
	} else if (mach < 1.2f) {
		float transsonic = (mach - 0.8f) / 0.4f;
		return 1.0f + transsonic * 3.0f;
	} else {
		return 4.0f + (mach - 1.2f) * 0.5f;
	}
}

void InitializeMissile(struct Missile *missile) {
	missile->position[0] = randRange(-250.0f, 250.f);
	missile->position[1] = randRange(100.0f, 500.f);
	missile->position[2] = randRange(-250.0f, 250.f);

	missile->velocity[0] = 0.0f;
	missile->velocity[1] = 0.0f;
	missile->velocity[2] = 0.0f;

	missile->targetDirection[0] = 1.0f;
	missile->targetDirection[1] = 0.0f;
	missile->targetDirection[2] = 0.0f;

	missile->bodyOrientation[0] = 1.0f;
	missile->bodyOrientation[1] = 0.0f;
	missile->bodyOrientation[2] = 0.0f;

	missile->angularVelocity[0] = 0.0f;
	missile->angularVelocity[1] = 0.0f;
	missile->angularVelocity[2] = 0.0f;

	missile->drag = randRange(0.012f, 0.025f);
	missile->inducedDragFactor = randRange(0.08f, 0.18f);
	missile->transsonicDragPeak = randRange(2.8f, 4.2f);
	missile->supersonicDragFactor = randRange(0.3f, 0.7f);
	missile->crossSectionArea = randRange(0.015f, 0.045f);
	missile->liftCoefficient = randRange(0.25f, 0.55f);
	missile->maxDynamicPressure = randRange(80000.0f, 150000.0f);
	missile->thrustVectoringEfficiency = randRange(0.75f, 0.95f);
	missile->momentOfInertia = randRange(1.8f, 4.5f);
	missile->controlAuthority = randRange(0.85f, 0.98f);
	missile->energyManagementFactor = randRange(0.6f, 0.9f);
	missile->minEnergyThreshold = randRange(0.3f, 0.5f);
	missile->optimalSpeed = randRange(600.0f, 900.0f);
	missile->dryMass = randRange(60.0f, 160.0f);
	missile->fuelMass = randRange(350.0f, 900.0f);
	missile->maxGPull = randRange(20.0f, 40.0f);
	missile->Isp = randRange(220.0f, 280.0f);
	missile->burning = 1;
	missile->burnRate = randRange(8.0f, 25.0f);
	missile->Q_spec = randRange(4.5e6f, 6.5e6f);

	missile->fireSim = malloc(sizeof(struct FireSOA));
	InitializeFireParticles(missile->fireSim);

	missile->fireSim->basePosition[0] = missile->position[0];
	missile->fireSim->basePosition[1] = missile->position[1];
	missile->fireSim->basePosition[2] = missile->position[2];

	float speed = sqrtf(missile->velocity[0] * missile->velocity[0] +
						missile->velocity[1] * missile->velocity[1] +
						missile->velocity[2] * missile->velocity[2]);

	if (speed > 0.1f) {
		missile->fireSim->windDirection[0] = -missile->velocity[0] * 10.0f;
		missile->fireSim->windDirection[1] = -missile->velocity[1] * 10.0f;
		missile->fireSim->windDirection[2] = -missile->velocity[2] * 10.0f;
	}
}

void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	float totalMass = missile->dryMass + missile->fuelMass;

	float thrust = 0.0f;
	if (missile->burning && missile->fuelMass > 0.0f) {
		float exhaustVelocity = missile->Isp * G;
		float massFlowRate = missile->burnRate;

		float fuelConsumed = massFlowRate * deltaTime;
		if (fuelConsumed > missile->fuelMass) {
			fuelConsumed = missile->fuelMass;
			missile->burning = 0;
		}
		missile->fuelMass -= fuelConsumed;

		thrust = massFlowRate * exhaustVelocity;
	}

	float speed = sqrtf(missile->velocity[0] * missile->velocity[0] +
						missile->velocity[1] * missile->velocity[1] +
						missile->velocity[2] * missile->velocity[2]);

	float altitude = missile->position[1];
	if (altitude < 0.0f) altitude = 0.0f;
	float airDensity = getAirDensity(altitude);
	float mach = speed / SPEED_OF_SOUND;
	float machDragMult = getMachDragMultiplier(mach);
	float dynamicPressure = 0.5f * airDensity * speed * speed;

	float kineticEnergy = 0.5f * totalMass * speed * speed;
	float potentialEnergy = totalMass * G * altitude;
	float totalEnergy = kineticEnergy + potentialEnergy;
	float maxPossibleKE = 0.5f * totalMass * missile->optimalSpeed * missile->optimalSpeed;
	float energyRatio = kineticEnergy / maxPossibleKE;

	float currentDir[3] = {0.0f, 0.0f, 0.0f};
	if (speed > 0.1f) {
		currentDir[0] = missile->velocity[0] / speed;
		currentDir[1] = missile->velocity[1] / speed;
		currentDir[2] = missile->velocity[2] / speed;
	}

	float errorDir[3] = {
		missile->targetDirection[0] - currentDir[0],
		missile->targetDirection[1] - currentDir[1],
		missile->targetDirection[2] - currentDir[2]};

	float errorMag = sqrtf(errorDir[0] * errorDir[0] +
						   errorDir[1] * errorDir[1] +
						   errorDir[2] * errorDir[2]);

	if (errorMag > 0.01f) {
		errorDir[0] /= errorMag;
		errorDir[1] /= errorMag;
		errorDir[2] /= errorMag;
	}

	float maxLateralAccel = missile->maxGPull * G;

	if (dynamicPressure > missile->maxDynamicPressure) {
		float qLimit = missile->maxDynamicPressure / dynamicPressure;
		maxLateralAccel *= qLimit;
	}

	float controlEffectiveness = (airDensity / SEA_LEVEL_DENSITY) * missile->controlAuthority;
	if (speed < 50.0f) {
		controlEffectiveness *= (speed / 50.0f);
	}
	maxLateralAccel *= controlEffectiveness;

	maxLateralAccel *= (airDensity / SEA_LEVEL_DENSITY);

	if (energyRatio < missile->minEnergyThreshold) {
		float energyLimitFactor = energyRatio / missile->minEnergyThreshold;
		maxLateralAccel *= (0.3f + 0.7f * energyLimitFactor);
	} else if (energyRatio < 0.7f) {
		float conservationFactor = missile->energyManagementFactor +
								   (1.0f - missile->energyManagementFactor) *
									   ((energyRatio - missile->minEnergyThreshold) / (0.7f - missile->minEnergyThreshold));
		maxLateralAccel *= conservationFactor;
	}

	float speedRatio = speed / missile->optimalSpeed;
	if (speedRatio < 0.5f) {
		maxLateralAccel *= (0.4f + 0.6f * (speedRatio / 0.5f));
	}

	float guidanceGain = 3.0f;

	float dotProduct = currentDir[0] * missile->targetDirection[0] +
					   currentDir[1] * missile->targetDirection[1] +
					   currentDir[2] * missile->targetDirection[2];
	float angleError = acosf(fmaxf(-1.0f, fminf(1.0f, dotProduct)));

	if (angleError < 0.2f && energyRatio > 0.8f) {
		guidanceGain = 2.0f;
	} else if (angleError > 1.0f && energyRatio < 0.5f) {
		guidanceGain = 1.5f;
	}

	float lateralAccel[3] = {
		errorDir[0] * maxLateralAccel * guidanceGain,
		errorDir[1] * maxLateralAccel * guidanceGain,
		errorDir[2] * maxLateralAccel * guidanceGain};

	float lateralAccelMag = sqrtf(lateralAccel[0] * lateralAccel[0] +
								  lateralAccel[1] * lateralAccel[1] +
								  lateralAccel[2] * lateralAccel[2]);
	if (lateralAccelMag > maxLateralAccel) {
		float scale = maxLateralAccel / lateralAccelMag;
		lateralAccel[0] *= scale;
		lateralAccel[1] *= scale;
		lateralAccel[2] *= scale;
	}

	float dragForce[3] = {
		-missile->drag * machDragMult * airDensity * missile->velocity[0] * speed,
		-missile->drag * machDragMult * airDensity * missile->velocity[1] * speed,
		-missile->drag * machDragMult * airDensity * missile->velocity[2] * speed};

	float inducedDragMag = missile->inducedDragFactor * lateralAccelMag / G * speed * airDensity;
	float inducedDragForce[3] = {
		-missile->velocity[0] * inducedDragMag / (speed + 0.001f),
		-missile->velocity[1] * inducedDragMag / (speed + 0.001f),
		-missile->velocity[2] * inducedDragMag / (speed + 0.001f)};

	float liftForce[3] = {0.0f, 0.0f, 0.0f};
	if (speed > 10.0f) {
		float dotProduct = currentDir[0] * missile->bodyOrientation[0] +
						   currentDir[1] * missile->bodyOrientation[1] +
						   currentDir[2] * missile->bodyOrientation[2];
		float aoa = acosf(fmaxf(-1.0f, fminf(1.0f, dotProduct)));

		if (aoa > 0.01f && aoa < 0.5f) {
			float liftMag = missile->liftCoefficient * dynamicPressure *
							missile->crossSectionArea * sinf(aoa);

			float liftDir[3];
			liftDir[0] = missile->bodyOrientation[1] * currentDir[2] -
						 missile->bodyOrientation[2] * currentDir[1];
			liftDir[1] = missile->bodyOrientation[2] * currentDir[0] -
						 missile->bodyOrientation[0] * currentDir[2];
			liftDir[2] = missile->bodyOrientation[0] * currentDir[1] -
						 missile->bodyOrientation[1] * currentDir[0];

			float liftDirMag = sqrtf(liftDir[0] * liftDir[0] +
									 liftDir[1] * liftDir[1] +
									 liftDir[2] * liftDir[2]);

			if (liftDirMag > 0.001f) {
				liftForce[0] = (liftDir[0] / liftDirMag) * liftMag;
				liftForce[1] = (liftDir[1] / liftDirMag) * liftMag;
				liftForce[2] = (liftDir[2] / liftDirMag) * liftMag;
			}
		}
	}

	float thrustAccel[3] = {0.0f, 0.0f, 0.0f};
	if (speed > 0.1f && thrust > 0.0f) {
		float thrustMag = thrust / totalMass;

		float altitudeFactor = airDensity / SEA_LEVEL_DENSITY;
		thrustMag *= (0.7f + 0.3f * altitudeFactor);

		float dotProduct = currentDir[0] * missile->targetDirection[0] +
						   currentDir[1] * missile->targetDirection[1] +
						   currentDir[2] * missile->targetDirection[2];
		float angleOfAttack = acosf(fmaxf(-1.0f, fminf(1.0f, dotProduct)));
		float thrustEfficiency = missile->thrustVectoringEfficiency +
								 (1.0f - missile->thrustVectoringEfficiency) * cosf(angleOfAttack);

		if (energyRatio < 0.6f && speedRatio < 0.8f) {
			float energyBoost = 1.0f + (1.0f - energyRatio) * 0.3f;
			thrustMag *= energyBoost;
		}

		thrustAccel[0] = currentDir[0] * thrustMag * thrustEfficiency;
		thrustAccel[1] = currentDir[1] * thrustMag * thrustEfficiency;
		thrustAccel[2] = currentDir[2] * thrustMag * thrustEfficiency;
	}

	float dragAccel[3] = {
		(dragForce[0] + inducedDragForce[0]) / totalMass,
		(dragForce[1] + inducedDragForce[1]) / totalMass,
		(dragForce[2] + inducedDragForce[2]) / totalMass};

	float liftAccel[3] = {
		liftForce[0] / totalMass,
		liftForce[1] / totalMass,
		liftForce[2] / totalMass};

	float gravityAccel[3] = {0.0f, -G, 0.0f};

	float totalAccel[3] = {
		thrustAccel[0] + dragAccel[0] + lateralAccel[0] + liftAccel[0] + gravityAccel[0],
		thrustAccel[1] + dragAccel[1] + lateralAccel[1] + liftAccel[1] + gravityAccel[1],
		thrustAccel[2] + dragAccel[2] + lateralAccel[2] + liftAccel[2] + gravityAccel[2]};

	missile->velocity[0] += totalAccel[0] * deltaTime;
	missile->velocity[1] += totalAccel[1] * deltaTime;
	missile->velocity[2] += totalAccel[2] * deltaTime;

	missile->position[0] += missile->velocity[0] * deltaTime;
	missile->position[1] += missile->velocity[1] * deltaTime;
	missile->position[2] += missile->velocity[2] * deltaTime;

	float orientationError[3] = {
		currentDir[0] - missile->bodyOrientation[0],
		currentDir[1] - missile->bodyOrientation[1],
		currentDir[2] - missile->bodyOrientation[2]};

	float turnRate = 5.0f * controlEffectiveness;
	missile->bodyOrientation[0] += orientationError[0] * turnRate * deltaTime;
	missile->bodyOrientation[1] += orientationError[1] * turnRate * deltaTime;
	missile->bodyOrientation[2] += orientationError[2] * turnRate * deltaTime;

	float bodyOrientMag = sqrtf(
		missile->bodyOrientation[0] * missile->bodyOrientation[0] +
		missile->bodyOrientation[1] * missile->bodyOrientation[1] +
		missile->bodyOrientation[2] * missile->bodyOrientation[2]);

	if (bodyOrientMag > 0.001f) {
		missile->bodyOrientation[0] /= bodyOrientMag;
		missile->bodyOrientation[1] /= bodyOrientMag;
		missile->bodyOrientation[2] /= bodyOrientMag;
	}

	if (missile->fireSim) {
		missile->fireSim->basePosition[0] = missile->position[0];
		missile->fireSim->basePosition[1] = missile->position[1];
		missile->fireSim->basePosition[2] = missile->position[2];

		float newSpeed = sqrtf(missile->velocity[0] * missile->velocity[0] +
							   missile->velocity[1] * missile->velocity[1] +
							   missile->velocity[2] * missile->velocity[2]);

		if (newSpeed > 0.1f) {
			missile->fireSim->windDirection[0] = -missile->velocity[0] * 10.0f;
			missile->fireSim->windDirection[1] = -missile->velocity[1] * 10.0f;
			missile->fireSim->windDirection[2] = -missile->velocity[2] * 10.0f;
		}

		float fireStepTime;
		fireSimStep(missile->fireSim, deltaTime, &fireStepTime);
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
						(end.tv_nsec - start.tv_nsec) / 1e6);
}

void setMissileTarget(struct Missile *missile, float targetPos[3]) {
	float dirToTarget[3] = {
		targetPos[0] - missile->position[0],
		targetPos[1] - missile->position[1],
		targetPos[2] - missile->position[2]};

	float mag = sqrtf(dirToTarget[0] * dirToTarget[0] +
					  dirToTarget[1] * dirToTarget[1] +
					  dirToTarget[2] * dirToTarget[2]);

	if (mag > 0.01f) {
		missile->targetDirection[0] = dirToTarget[0] / mag;
		missile->targetDirection[1] = dirToTarget[1] / mag;
		missile->targetDirection[2] = dirToTarget[2] / mag;
	}
}

void cleanupMissile(struct Missile *missile) {
	if (missile->fireSim) {
		free(missile->fireSim);
		missile->fireSim = NULL;
	}
}

void InitializeFireParticles(struct FireSOA *particles) {
	particles->buoyancy = 100.0f;
	particles->drag = 0.985f;
	particles->turbulence = 2.5f;
	particles->maxLifeTime = 0.5f;

	particles->startingColor[0] = randRange(0.0f, 0.25f);
	particles->startingColor[1] = randRange(0.0f, 0.25f);
	particles->startingColor[2] = randRange(0.0f, 0.25f);
	particles->fireColor[0] = randRange(0.0f, 0.25f);
	particles->fireColor[1] = randRange(0.0f, 0.25f);
	particles->fireColor[2] = randRange(0.0f, 0.25f);
	particles->smokeColor[0] = randRange(0.0f, 0.5f);
	particles->smokeColor[1] = randRange(0.0f, 0.5f);
	particles->smokeColor[2] = randRange(0.0f, 0.5f);

	particles->basePosition[0] = 100.0f;
	particles->basePosition[1] = 0.0f;
	particles->basePosition[2] = -350.0f;

	particles->maxVelocity = 0.0f;
	particles->maxDistance = 0.0f;
	particles->particlesSize = 6.0f;
	particles->windDirection[0] = 0.0f;
	particles->windDirection[1] = 0.0f;
	particles->windDirection[2] = 0.0f;
	particles->swirlIntensity = 1.0f;
	particles->swirlFrequency = 1.5f;

	for (int i = 0; i < NUM_FIRE_PARTICLES; i++) {
		particles->x[i] = particles->basePosition[0] + randRange(-1.0f, 1.0f) * 5.0f;
		particles->y[i] = particles->basePosition[1];
		particles->z[i] = particles->basePosition[2] + randRange(-1.0f, 1.0f) * 5.0f;
		particles->xVelocity[i] = randRange(-1.0f, 1.0f) * 5.0f;
		particles->yVelocity[i] = randRange(-1.0f, 1.0f) * 20.0f;
		particles->zVelocity[i] = randRange(-1.0f, 1.0f) * 5.0f;
		particles->lifeTime[i] = randRange(0.0f, particles->maxLifeTime);
	}
}

void fireSimStep(struct FireSOA *particles, float deltaTime, float *timeTook) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	for (int i = 0; i < NUM_FIRE_PARTICLES; i++) {
		float lifeRatio = particles->lifeTime[i] / particles->maxLifeTime;
		float inverseLife = 1.0f - (lifeRatio);
		float buoyancyForce = particles->buoyancy * inverseLife;
		particles->yVelocity[i] += (buoyancyForce - G) * deltaTime * lifeRatio;

		// apply drag
		particles->xVelocity[i] *= particles->drag;
		particles->yVelocity[i] *= particles->drag;
		particles->zVelocity[i] *= particles->drag;

		// apply turbulence
		float turbulence = inverseLife * particles->turbulence;
		particles->xVelocity[i] += randRange(-turbulence, turbulence) * deltaTime;
		particles->yVelocity[i] += randRange(-turbulence * 0.25, turbulence * 0.5) * deltaTime;
		particles->zVelocity[i] += randRange(-turbulence, turbulence) * deltaTime;

		// apply wind
		particles->xVelocity[i] += particles->windDirection[0] * deltaTime * lifeRatio;
		particles->yVelocity[i] += particles->windDirection[1] * deltaTime * lifeRatio;
		particles->zVelocity[i] += particles->windDirection[2] * deltaTime * lifeRatio;

		float swirl = particles->swirlIntensity * inverseLife;
		float angle = particles->swirlFrequency * lifeRatio * 2.0f * 3.14159f;
		particles->xVelocity[i] += swirl * cosf(angle) * deltaTime;
		particles->zVelocity[i] += swirl * sinf(angle) * deltaTime;

		// update positions
		particles->x[i] += particles->xVelocity[i] * deltaTime;
		particles->y[i] += particles->yVelocity[i] * deltaTime;
		particles->z[i] += particles->zVelocity[i] * deltaTime;

		// update lifetime
		particles->lifeTime[i] += deltaTime * randRange(0.5f, 1.5f);

		if (particles->lifeTime[i] >= particles->maxLifeTime) {
			// respawn particle
			particles->x[i] = particles->basePosition[0] + randRange(-1.0f, 1.0f) * 5.0f;
			particles->y[i] = particles->basePosition[1];
			particles->z[i] = particles->basePosition[2] + randRange(-1.0f, 1.0f) * 5.0f;
			particles->xVelocity[i] = randRange(-1.0f, 1.0f) * 5.0f;
			particles->yVelocity[i] = randRange(-1.0f, 1.0f) * 20.0f;
			particles->zVelocity[i] = randRange(-1.0f, 1.0f) * 5.0f;
			particles->lifeTime[i] = 0.0f;
		}
		float totalVelocity = particles->xVelocity[i] * particles->xVelocity[i] +
							  particles->yVelocity[i] * particles->yVelocity[i] +
							  particles->zVelocity[i] * particles->zVelocity[i];
		float totalDistance = (particles->x[i] - particles->basePosition[0]) * (particles->x[i] - particles->basePosition[0]) +
							  (particles->y[i] - particles->basePosition[1]) * (particles->y[i] - particles->basePosition[1]) +
							  (particles->z[i] - particles->basePosition[2]) * (particles->z[i] - particles->basePosition[2]);
		if (totalVelocity > particles->maxVelocity) {
			particles->maxVelocity = totalVelocity;
		}
		if (totalDistance > particles->maxDistance) {
			particles->maxDistance = totalDistance;
		}
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6);
	particles->maxVelocity = sqrtf(particles->maxVelocity);
	particles->maxDistance = sqrtf(particles->maxDistance);
}

void InitializeMissiles(struct Missiles *missiles, int count) {
	missiles->count = min(count, MAX_FIRE_SIMS);

	for (int i = 0; i < missiles->count; i++) {
		missiles->missiles[i] = malloc(sizeof(struct Missile));
		if (!missiles->missiles[i]) {
			printf("Failed to allocate memory for missile %d\n", i);
			missiles->count = i;
			return;
		}

		InitializeMissile(missiles->missiles[i]);

		float angle = (float)i / (float)missiles->count * 6.28318f;
		float radius = randRange(100.0f, 500.0f);

		missiles->missiles[i]->position[0] = cosf(angle) * radius;
		missiles->missiles[i]->position[1] = randRange(500.0f, 3000.0f);
		missiles->missiles[i]->position[2] = sinf(angle) * radius;

		float speed = randRange(150.0f, 400.0f);
		float pitchAngle = randRange(-0.3f, 0.3f);

		missiles->missiles[i]->velocity[0] = cosf(angle + 1.57f) * cosf(pitchAngle) * speed;
		missiles->missiles[i]->velocity[1] = sinf(pitchAngle) * speed;
		missiles->missiles[i]->velocity[2] = sinf(angle + 1.57f) * cosf(pitchAngle) * speed;

		missiles->missiles[i]->bodyOrientation[0] = missiles->missiles[i]->velocity[0] / speed;
		missiles->missiles[i]->bodyOrientation[1] = missiles->missiles[i]->velocity[1] / speed;
		missiles->missiles[i]->bodyOrientation[2] = missiles->missiles[i]->velocity[2] / speed;

		float targetAngle = angle + 3.14159f + randRange(-0.5f, 0.5f);
		float targetPitch = randRange(-0.2f, 0.2f);
		missiles->missiles[i]->targetDirection[0] = cosf(targetAngle) * cosf(targetPitch);
		missiles->missiles[i]->targetDirection[1] = sinf(targetPitch);
		missiles->missiles[i]->targetDirection[2] = sinf(targetAngle) * cosf(targetPitch);

		float len = sqrtf(
			missiles->missiles[i]->targetDirection[0] * missiles->missiles[i]->targetDirection[0] +
			missiles->missiles[i]->targetDirection[1] * missiles->missiles[i]->targetDirection[1] +
			missiles->missiles[i]->targetDirection[2] * missiles->missiles[i]->targetDirection[2]);

		if (len > 0.001f) {
			missiles->missiles[i]->targetDirection[0] /= len;
			missiles->missiles[i]->targetDirection[1] /= len;
			missiles->missiles[i]->targetDirection[2] /= len;
		}
	}
}

void UpdateAllMissiles(struct Missiles *missiles, float deltaTime) {
	for (int i = 0; i < missiles->count; i++) {
		float simTime;
		missileSimStep(missiles->missiles[i], deltaTime, &simTime);
	}
}

void CleanupMissiles(struct Missiles *missiles) {
	for (int i = 0; i < missiles->count; i++) {
		if (missiles->missiles[i]) {
			cleanupMissile(missiles->missiles[i]);
			free(missiles->missiles[i]);
			missiles->missiles[i] = NULL;
		}
	}
	missiles->count = 0;
}

#ifdef FIRE_BENCHMARK
#include <stdio.h>

int main(int argc, char **argv) {
	printf("=== Fire Particle Simulation Benchmark ===\n");
	printf("Particle count: %d\n\n", NUM_FIRE_PARTICLES);

	struct FireSOA particles;
	InitializeFireParticles(&particles);

	// Warm-up
	float warmupTime;
	for (int i = 0; i < 10; i++) {
		fireSimStep(&particles, 0.016f, &warmupTime);
	}

	// Benchmark parameters
	const int NUM_ITERATIONS = 1000;
	const float deltaTime = 0.016f; // 60 FPS target

	float totalTime = 0.0f;
	float minTime = 1e9f;
	float maxTime = 0.0f;

	printf("Running %d iterations...\n", NUM_ITERATIONS);

	struct timespec benchStart, benchEnd;
	clock_gettime(CLOCK_MONOTONIC, &benchStart);

	for (int i = 0; i < NUM_ITERATIONS; i++) {
		float stepTime;
		fireSimStep(&particles, deltaTime, &stepTime);

		totalTime += stepTime;
		if (stepTime < minTime) minTime = stepTime;
		if (stepTime > maxTime) maxTime = stepTime;

		// Progress indicator
		if ((i + 1) % 100 == 0) {
			printf("  %d/%d iterations complete\r", i + 1, NUM_ITERATIONS);
			fflush(stdout);
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &benchEnd);
	double wallTime = (benchEnd.tv_sec - benchStart.tv_sec) * 1000.0 +
					  (benchEnd.tv_nsec - benchStart.tv_nsec) / 1e6;

	printf("\n\n=== Results ===\n");
	printf("Total iterations: %d\n", NUM_ITERATIONS);
	printf("Wall clock time: %.2f ms\n", wallTime);
	printf("\nPer-step statistics:\n");
	printf("  Average: %.4f ms (%.1f FPS)\n", totalTime / NUM_ITERATIONS, 1000.0f / (totalTime / NUM_ITERATIONS));
	printf("  Minimum: %.4f ms (%.1f FPS)\n", minTime, 1000.0f / minTime);
	printf("  Maximum: %.4f ms (%.1f FPS)\n", maxTime, 1000.0f / maxTime);

	printf("\nParticle statistics:\n");
	printf("  Max velocity: %.2f units/s\n", particles.maxVelocity);
	printf("  Max distance: %.2f units\n", particles.maxDistance);

	// Performance metrics
	float avgTimePerParticle = (totalTime / NUM_ITERATIONS) / NUM_FIRE_PARTICLES;
	float particlesPerMs = NUM_FIRE_PARTICLES / (totalTime / NUM_ITERATIONS);

	printf("\nPerformance metrics:\n");
	printf("  Time per particle: %.6f ms\n", avgTimePerParticle);
	printf("  Particles per ms: %.2f\n", particlesPerMs);
	printf("  Particles per second: %.2f M\n", (particlesPerMs * 1000.0f) / 1e6);

	// Estimate max particles for 60 FPS
	float targetFrameTime = 16.667f; // 60 FPS
	int maxParticles60fps = (int)(targetFrameTime / avgTimePerParticle);

	printf("\nCapacity estimates (CPU only):\n");
	printf("  Max particles @ 60 FPS: ~%d\n", maxParticles60fps);
	printf("  Max particles @ 30 FPS: ~%d\n", maxParticles60fps * 2);
	printf("  Max particles @ 15 FPS: ~%d\n", maxParticles60fps * 4);

	return 0;
}
#endif
