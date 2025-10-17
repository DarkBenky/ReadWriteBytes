#include "fireSim.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
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

float getMachDragMultiplier(float mach, float transsonicPeak, float supersonicFactor) {
	if (mach < 0.8f) {
		return 1.0f;
	} else if (mach < 1.2f) {
		float transsonic = (mach - 0.8f) / 0.4f;
		return 1.0f + transsonic * (transsonicPeak - 1.0f);
	} else {
		return transsonicPeak + (mach - 1.2f) * supersonicFactor;
	}
}

void InitializeMissile(struct Missile *missile) {
	// Seeker initialization
	missile->seeker.seekerCamera.ray.origin[0] = 0.0f;
	missile->seeker.seekerCamera.ray.origin[1] = 0.0f;
	missile->seeker.seekerCamera.ray.origin[2] = 0.0f;
	missile->seeker.seekerCamera.ray.direction[0] = 1.0f;
	missile->seeker.seekerCamera.ray.direction[1] = 0.0f;
	missile->seeker.seekerCamera.ray.direction[2] = 0.0f;
	missile->seeker.seekerCamera.fov = 8.5f;
	missile->seeker.seekerFov = 60.0f;
	missile->seeker.seekerSteps = 1;
	missile->seeker.lockState = Lunching;

	// Core simulation
	missile->position[0] = randRange(-250.0f, 250.0f);
	missile->position[1] = randRange(100.0f, 1500.0f);
	missile->position[2] = randRange(-250.0f, 250.0f);

	missile->velocity[0] = 0.0f;
	missile->velocity[1] = 0.0f;
	missile->velocity[2] = 0.0f;

	missile->acceleration[0] = 0.0f;
	missile->acceleration[1] = 0.0f;
	missile->acceleration[2] = 0.0f;

	missile->targetPosition[0] = 0.0f;
	missile->targetPosition[1] = 0.0f;
	missile->targetPosition[2] = 0.0f;

	missile->targetDirection[0] = 1.0f;
	missile->targetDirection[1] = 0.0f;
	missile->targetDirection[2] = 0.0f;

	// Aerodynamic state
	missile->bodyOrientation[0] = 1.0f;
	missile->bodyOrientation[1] = 0.0f;
	missile->bodyOrientation[2] = 0.0f;

	missile->angularVelocity[0] = 0.0f;
	missile->angularVelocity[1] = 0.0f;
	missile->angularVelocity[2] = 0.0f;

	missile->angleOfAttack = 0.0f;
	missile->sideslipAngle = 0.0f;

	// Mass properties
	missile->dryMass = randRange(10.0f, 70.0f);
	missile->fuelMass = randRange(350.0f, 900.0f);
	missile->totalMass = missile->dryMass + missile->fuelMass;
	missile->momentOfInertia = randRange(1.8f, 7.5f);

	// Propulsion
	missile->thrust = 0.0f;
	missile->Isp = randRange(220.0f, 350.0f);
	missile->burning = 1;
	missile->burnRate = randRange(1.0f, 15.0f);
	missile->maxGimbalAngle = randRange(0.00f, 0.45f);
	missile->gimbalAngle[0] = 0.0f;
	missile->gimbalAngle[1] = 0.0f;

	// Aerodynamic coefficients
	missile->zeroLiftDrag = randRange(0.012f, 0.025f);
	missile->liftSlope = randRange(2.0f, 4.0f);
	missile->maxLiftCoeff = randRange(1.2f, 2.0f);
	missile->dragSlope = randRange(0.02f, 0.08f);
	missile->crossSectionArea = randRange(0.015f, 0.045f);
	missile->wingArea = randRange(0.12f, 0.85f);
	missile->aspectRatio = randRange(2.0f, 5.0f);
	missile->oswaldEfficiency = randRange(0.7f, 0.95f);

	// Control surfaces
	missile->finMaxDeflection = randRange(0.2f, 0.5f);
	for (int i = 0; i < 4; i++)
		missile->finDeflection[i] = 0.0f;
	missile->finEffectiveness = randRange(0.7f, 1.2f);
	missile->rollDamping = randRange(0.05f, 0.2f);

	// Performance limits
	missile->maxGPull = randRange(5.0f, 50.0f);
	missile->maxDynamicPressure = randRange(80000.0f, 250000.0f);
	missile->maxAoA = randRange(0.25f, 0.75f);
	missile->maxLoadFactor = randRange(30.0f, 60.0f);

	// Guidance & control
	missile->guidanceGain = randRange(1.5f, 7.5f);
	missile->controlAuthority = randRange(0.85f, 0.98f);
	missile->energyManagementFactor = randRange(0.6f, 0.95f);
	missile->optimalSpeed = randRange(300.0f, 900.0f);

	// Simulation
	missile->remainingTime = randRange(45.0f, 120.0f);
	missile->fireSim = malloc(sizeof(struct FireSOA));
	InitializeFireParticles(missile->fireSim);

	// Cached values
	missile->machNumber = 0.0f;
	missile->dynamicPressure = 0.0f;
	missile->prevLOS[0] = 0.0f;
	missile->prevLOS[1] = 0.0f;
	missile->prevLOS[2] = 0.0f;
}

void setMissileTarget(struct Missile *missile, float targetPos[3]) {
	// Set the target position (needed for proper PN guidance)
	missile->targetPosition[0] = targetPos[0];
	missile->targetPosition[1] = targetPos[1];
	missile->targetPosition[2] = targetPos[2];

	// Also calculate and set the initial target direction
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

		// Initialize previous LOS for proper PN guidance
		missile->prevLOS[0] = missile->targetDirection[0];
		missile->prevLOS[1] = missile->targetDirection[1];
		missile->prevLOS[2] = missile->targetDirection[2];
	}
}

void setMissileTargetDirection(struct Missile *missile, float targetDir[3], float *targetDist) {
	// Normalize the direction
	float mag = sqrtf(targetDir[0] * targetDir[0] +
					  targetDir[1] * targetDir[1] +
					  targetDir[2] * targetDir[2]);

	if (mag > 0.01f) {
		missile->targetDirection[0] = targetDir[0] / mag;
		missile->targetDirection[1] = targetDir[1] / mag;
		missile->targetDirection[2] = targetDir[2] / mag;

		// For direction-only guidance, create a "virtual" target position
		float virtualTargetDistance = 10000.0f; // 10km default virtual target

		if (targetDist != NULL) {
			virtualTargetDistance = *targetDist;
		}

		missile->targetPosition[0] = missile->position[0] + missile->targetDirection[0] * virtualTargetDistance;
		missile->targetPosition[1] = missile->position[1] + missile->targetDirection[1] * virtualTargetDistance;
		missile->targetPosition[2] = missile->position[2] + missile->targetDirection[2] * virtualTargetDistance;

		// Initialize previous LOS
		missile->prevLOS[0] = missile->targetDirection[0];
		missile->prevLOS[1] = missile->targetDirection[1];
		missile->prevLOS[2] = missile->targetDirection[2];
	}
}

void missileSeekStep(struct Missile *missile, bool fire, bool foundTarget, float targetDir[3]) {
	if (foundTarget) {
		missile->seeker.lockState = Tracking;
	}
	if (!foundTarget) {
		missile->seeker.lockState = Searching;
	}
	if (fire) {
		missile->seeker.lockState = Lunching;
	}

	if (missile->seeker.lockState == Searching) {
		// Convert FOV and gimbal limit from degrees to radians
		float fovRad = missile->seeker.seekerCamera.fov * (M_PI / 180.0f);
		float maxGimbalRad = missile->seeker.seekerFov * (M_PI / 180.0f) / 2.0f;

		// Get current camera ray direction
		float *rayDir = missile->seeker.seekerCamera.ray.direction;
		float *rayOrigin = missile->seeker.seekerCamera.ray.origin;

		// Calculate current angles relative to missile body (forward = +Z)
		float currentPitch = atan2f(rayDir[1], sqrtf(rayDir[0] * rayDir[0] + rayDir[2] * rayDir[2]));
		float currentYaw = atan2f(rayDir[0], rayDir[2]);

		// Move camera by one FOV step horizontally
		currentYaw += fovRad;

		// Check if we exceeded gimbal limit
		if (currentYaw > maxGimbalRad) {
			currentYaw = -maxGimbalRad; // Reset to left edge
			currentPitch += fovRad;		// Step down by one FOV

			// If exceeded vertical gimbal limit, reset to top
			if (currentPitch > maxGimbalRad) {
				currentPitch = -maxGimbalRad;
			}
		}

		// Convert angles back to direction vector
		float cosPitch = cosf(currentPitch);
		rayDir[0] = sinf(currentYaw) * cosPitch;
		rayDir[1] = sinf(currentPitch);
		rayDir[2] = cosf(currentYaw) * cosPitch;

		// Normalize direction vector
		float length = sqrtf(rayDir[0] * rayDir[0] + rayDir[1] * rayDir[1] + rayDir[2] * rayDir[2]);
		rayDir[0] /= length;
		rayDir[1] /= length;
		rayDir[2] /= length;

		// Update camera origin to missile position
		rayOrigin[0] = missile->position[0];
		rayOrigin[1] = missile->position[1];
		rayOrigin[2] = missile->position[2];

	} else if (missile->seeker.lockState == Tracking) {
		
		missile->seeker.seekerCamera.ray.direction[0] = targetDir[0];
		missile->seeker.seekerCamera.ray.direction[1] = targetDir[1];
		missile->seeker.seekerCamera.ray.direction[2] = targetDir[2];
		missile->targetDirection[0] = targetDir[0];
		missile->targetDirection[1] = targetDir[1];
		missile->targetDirection[2]	= targetDir[2];

		missile->prevLOS[0] = missile->targetPosition[0];
		missile->prevLOS[1] = missile->targetPosition[1];
		missile->prevLOS[2] = missile->targetPosition[2];

		// set missile target direction for PN guidance
		float virtualTargetDistance = 500.0f;
		missile->targetPosition[0] = missile->position[0] + missile->targetDirection[0] * virtualTargetDistance;
		missile->targetPosition[1] = missile->position[1] + missile->targetDirection[1] * virtualTargetDistance;
		missile->targetPosition[2] = missile->position[2] + missile->targetDirection[2] * virtualTargetDistance;


	} else if (missile->seeker.lockState == Lunching) {
		missile->seeker.seekerCamera.ray.direction[0] = missile->targetDirection[0];
		missile->seeker.seekerCamera.ray.direction[1] = missile->targetDirection[1];
		missile->seeker.seekerCamera.ray.direction[2] = missile->targetDirection[2];
		missile->seeker.seekerCamera.ray.origin[0] = missile->position[0];
		missile->seeker.seekerCamera.ray.origin[1] = missile->position[1];
		missile->seeker.seekerCamera.ray.origin[2] = missile->position[2];
		setMissileTargetDirection(missile, targetDir, NULL);
		missile->seeker.lockState = Tracking;
	}
	return;
}

void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook, bool *active, float *fireSimulationTime) {
	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);

	missile->remainingTime -= deltaTime;
	if (missile->remainingTime <= 0.0f) {
		*active = false;
		clock_gettime(CLOCK_MONOTONIC, &end);
		*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
							(end.tv_nsec - start.tv_nsec) / 1e6);
		return;
	}

	// Update mass
	missile->totalMass = missile->dryMass + missile->fuelMass;
	float speed = sqrtf(missile->velocity[0] * missile->velocity[0] +
						missile->velocity[1] * missile->velocity[1] +
						missile->velocity[2] * missile->velocity[2]);

	float altitude = fmaxf(0.0f, missile->position[1]);
	float airDensity = getAirDensity(altitude);
	missile->machNumber = speed / SPEED_OF_SOUND;
	missile->dynamicPressure = 0.5f * airDensity * speed * speed;

	// PROPULSION - PHYSICALLY CORRECT
	missile->thrust = 0.0f;
	if (missile->burning && missile->fuelMass > 0.0f) {
		float massFlowRate = missile->burnRate;
		float fuelConsumed = massFlowRate * deltaTime;

		if (fuelConsumed > missile->fuelMass) {
			fuelConsumed = missile->fuelMass;
			missile->burning = 0;
		}
		missile->fuelMass -= fuelConsumed;

		// Rocket equation: F = ṁ * Ve
		float exhaustVelocity = missile->Isp * G;
		missile->thrust = massFlowRate * exhaustVelocity;

		// Altitude compensation (simplified)
		float ambientPressure = airDensity * 287.0f * 288.0f;
		float pressureRatio = 1.0f - (ambientPressure / 101325.0f) * 0.15f;
		missile->thrust *= (1.0f + pressureRatio * 0.2f);
	}

	// AERODYNAMIC ANGLES - CORRECT
	float currentDir[3] = {0.0f, 0.0f, 0.0f};
	if (speed > 0.1f) {
		currentDir[0] = missile->velocity[0] / speed;
		currentDir[1] = missile->velocity[1] / speed;
		currentDir[2] = missile->velocity[2] / speed;
	} else {
		currentDir[0] = missile->bodyOrientation[0];
		currentDir[1] = missile->bodyOrientation[1];
		currentDir[2] = missile->bodyOrientation[2];
	}

	// Angle of attack (angle between body and velocity)
	float bodyDotVel = missile->bodyOrientation[0] * currentDir[0] +
					   missile->bodyOrientation[1] * currentDir[1] +
					   missile->bodyOrientation[2] * currentDir[2];
	missile->angleOfAttack = acosf(fmaxf(-1.0f, fminf(1.0f, bodyDotVel)));

	// PROPER PROPORTIONAL NAVIGATION - CORRECT IMPLEMENTATION
	float los[3] = {
		missile->targetPosition[0] - missile->position[0],
		missile->targetPosition[1] - missile->position[1],
		missile->targetPosition[2] - missile->position[2]};

	float losDistance = sqrtf(los[0] * los[0] + los[1] * los[1] + los[2] * los[2]);
	float losRate[3] = {0.0f, 0.0f, 0.0f};

	if (losDistance > 0.1f) {
		// Normalize LOS
		los[0] /= losDistance;
		los[1] /= losDistance;
		los[2] /= losDistance;

		// Calculate LOS rate properly
		if (deltaTime > 0.001f) {
			losRate[0] = (los[0] - missile->prevLOS[0]) / deltaTime;
			losRate[1] = (los[1] - missile->prevLOS[1]) / deltaTime;
			losRate[2] = (los[2] - missile->prevLOS[2]) / deltaTime;
		}

		// Store for next frame
		missile->prevLOS[0] = los[0];
		missile->prevLOS[1] = los[1];
		missile->prevLOS[2] = los[2];

		missile->targetDirection[0] = los[0];
		missile->targetDirection[1] = los[1];
		missile->targetDirection[2] = los[2];
	}

	// PROPER PROPORTIONAL NAVIGATION: a = N * Vc × Ω
	// Where × is cross product, Ω is LOS rate, Vc is closing velocity
	float closingSpeed = fmaxf(speed, 100.0f);
	float navGain = missile->guidanceGain;

	// Cross product: Vc × Ω
	float crossProduct[3] = {
		closingSpeed * losRate[2] - 0.0f * losRate[1], // Simplified: Vc along velocity
		0.0f * losRate[0] - closingSpeed * losRate[2],
		closingSpeed * losRate[1] - 0.0f * losRate[0]};

	float commandedAccel[3] = {
		navGain * crossProduct[0],
		navGain * crossProduct[1],
		navGain * crossProduct[2]};

	// Remove component along velocity to get pure lateral acceleration
	float parallelComp = commandedAccel[0] * currentDir[0] + commandedAccel[1] * currentDir[1] + commandedAccel[2] * currentDir[2];
	commandedAccel[0] -= parallelComp * currentDir[0];
	commandedAccel[1] -= parallelComp * currentDir[1];
	commandedAccel[2] -= parallelComp * currentDir[2];

	// AERODYNAMIC PERFORMANCE LIMITS - CORRECT
	float maxAvailableG = missile->maxGPull * G;

	// Dynamic pressure limits
	if (missile->dynamicPressure > missile->maxDynamicPressure) {
		float qFactor = missile->maxDynamicPressure / (missile->dynamicPressure + 0.001f);
		maxAvailableG *= qFactor;
	}

	// AoA limits
	float aoaFactor = 1.0f - fmaxf(0.0f, (missile->angleOfAttack - missile->maxAoA * 0.7f) / (missile->maxAoA * 0.3f));
	maxAvailableG *= fmaxf(0.1f, aoaFactor);

	// Control effectiveness
	float controlEffectiveness = (airDensity / SEA_LEVEL_DENSITY) * missile->controlAuthority;
	if (speed < 50.0f) {
		controlEffectiveness *= (speed / 50.0f);
	}
	maxAvailableG *= controlEffectiveness;

	// Limit commanded acceleration
	float commandedAccelMag = sqrtf(commandedAccel[0] * commandedAccel[0] +
									commandedAccel[1] * commandedAccel[1] +
									commandedAccel[2] * commandedAccel[2]);

	if (commandedAccelMag > maxAvailableG && commandedAccelMag > 0.001f) {
		float scale = maxAvailableG / commandedAccelMag;
		commandedAccel[0] *= scale;
		commandedAccel[1] *= scale;
		commandedAccel[2] *= scale;
		commandedAccelMag = maxAvailableG;
	}

	// SIMPLIFIED BUT PHYSICALLY CORRECT APPROACH:
	// We apply the guidance acceleration directly (simplification for real-time)
	// but calculate realistic aerodynamic forces that would produce this acceleration

	// Calculate what lift force would be needed to produce the commanded acceleration
	float requiredLift = commandedAccelMag * missile->totalMass;

	// Calculate what angle of attack would produce this lift
	float availableLift = missile->maxLiftCoeff * missile->dynamicPressure * missile->wingArea;
	float liftFraction = fminf(1.0f, requiredLift / (availableLift + 0.001f));

	// Set fin deflections based on required control (simplified)
	float finDeflection = liftFraction * missile->finMaxDeflection;
	for (int i = 0; i < 4; i++) {
		missile->finDeflection[i] = finDeflection;
	}

	// REALISTIC AERODYNAMIC FORCES - CORRECT
	// Base lift from current AoA
	float baseLiftCoeff = missile->liftSlope * missile->angleOfAttack;

	// Additional lift from fin control
	float controlLiftCoeff = finDeflection * missile->finEffectiveness;

	float totalLiftCoeff = baseLiftCoeff + controlLiftCoeff;
	totalLiftCoeff = fmaxf(-missile->maxLiftCoeff, fminf(missile->maxLiftCoeff, totalLiftCoeff));

	// Lift magnitude
	float liftMagnitude = totalLiftCoeff * missile->dynamicPressure * missile->wingArea;

	// CORRECT induced drag
	float inducedDragCoeff = (totalLiftCoeff * totalLiftCoeff) /
							 (3.14159f * missile->aspectRatio * missile->oswaldEfficiency);

	// Total drag (zero-lift + induced)
	float totalDragCoeff = missile->zeroLiftDrag + inducedDragCoeff;

	// Mach effects
	float mach = missile->machNumber;
	if (mach > 0.8f && mach < 1.2f) {
		totalDragCoeff *= (1.0f + (mach - 0.8f) * 2.0f);
	} else if (mach >= 1.2f) {
		totalDragCoeff *= (1.8f - (mach - 1.2f) * 0.3f);
	}

	float dragMagnitude = totalDragCoeff * missile->dynamicPressure * missile->crossSectionArea;

	// CORRECT lift direction (perpendicular to velocity)
	float liftDir[3];
	if (commandedAccelMag > 0.1f && speed > 1.0f) {
		// For simplicity, lift in the direction of commanded acceleration
		// This assumes the missile can instantly orient lift properly
		liftDir[0] = commandedAccel[0] / commandedAccelMag;
		liftDir[1] = commandedAccel[1] / commandedAccelMag;
		liftDir[2] = commandedAccel[2] / commandedAccelMag;

		// Ensure lift is perpendicular to velocity
		float parallel = liftDir[0] * currentDir[0] + liftDir[1] * currentDir[1] + liftDir[2] * currentDir[2];
		liftDir[0] -= parallel * currentDir[0];
		liftDir[1] -= parallel * currentDir[1];
		liftDir[2] -= parallel * currentDir[2];

		// Renormalize
		float liftDirMag = sqrtf(liftDir[0] * liftDir[0] + liftDir[1] * liftDir[1] + liftDir[2] * liftDir[2]);
		if (liftDirMag > 0.001f) {
			liftDir[0] /= liftDirMag;
			liftDir[1] /= liftDirMag;
			liftDir[2] /= liftDirMag;
		} else {
			// Fallback: arbitrary perpendicular direction
			liftDir[0] = -currentDir[1];
			liftDir[1] = currentDir[0];
			liftDir[2] = 0.0f;
		}
	} else {
		// Default lift direction
		liftDir[0] = -currentDir[1];
		liftDir[1] = currentDir[0];
		liftDir[2] = 0.0f;
	}

	float liftForce[3] = {
		liftDir[0] * liftMagnitude,
		liftDir[1] * liftMagnitude,
		liftDir[2] * liftMagnitude};

	float dragForce[3] = {
		-dragMagnitude * currentDir[0],
		-dragMagnitude * currentDir[1],
		-dragMagnitude * currentDir[2]};

	// CORRECT THRUST FORCE
	float thrustForce[3] = {0.0f, 0.0f, 0.0f};
	if (missile->thrust > 0.0f) {
		// Thrust follows body orientation with gimbal
		float thrustDir[3] = {
			missile->bodyOrientation[0],
			missile->bodyOrientation[1],
			missile->bodyOrientation[2]};

		// Apply gimbal if needed
		if (fabsf(missile->gimbalAngle[0]) > 0.001f || fabsf(missile->gimbalAngle[1]) > 0.001f) {
			// Simplified: blend toward velocity direction based on gimbal
			float gimbalStrength = sqrtf(missile->gimbalAngle[0] * missile->gimbalAngle[0] +
										 missile->gimbalAngle[1] * missile->gimbalAngle[1]);
			float blend = fminf(1.0f, gimbalStrength / missile->maxGimbalAngle);

			thrustDir[0] = thrustDir[0] * (1.0f - blend) + currentDir[0] * blend;
			thrustDir[1] = thrustDir[1] * (1.0f - blend) + currentDir[1] * blend;
			thrustDir[2] = thrustDir[2] * (1.0f - blend) + currentDir[2] * blend;

			// Normalize
			float thrustDirMag = sqrtf(thrustDir[0] * thrustDir[0] + thrustDir[1] * thrustDir[1] + thrustDir[2] * thrustDir[2]);
			thrustDir[0] /= thrustDirMag;
			thrustDir[1] /= thrustDirMag;
			thrustDir[2] /= thrustDirMag;
		}

		thrustForce[0] = thrustDir[0] * missile->thrust;
		thrustForce[1] = thrustDir[1] * missile->thrust;
		thrustForce[2] = thrustDir[2] * missile->thrust;
	}

	// GRAVITY FORCE
	float gravityForce[3] = {0.0f, -missile->totalMass * G, 0.0f};

	// CORRECT: Total forces from physics only
	float totalForce[3] = {
		thrustForce[0] + dragForce[0] + liftForce[0] + gravityForce[0],
		thrustForce[1] + dragForce[1] + liftForce[1] + gravityForce[1],
		thrustForce[2] + dragForce[2] + liftForce[2] + gravityForce[2]};

	// CORRECT: Add guidance acceleration as a PHYSICAL force (not magical)
	// This represents the actual lateral force from control surfaces
	float guidanceForce[3] = {
		commandedAccel[0] * missile->totalMass,
		commandedAccel[1] * missile->totalMass,
		commandedAccel[2] * missile->totalMass};

	totalForce[0] += guidanceForce[0];
	totalForce[1] += guidanceForce[1];
	totalForce[2] += guidanceForce[2];

	float totalAccel[3] = {
		totalForce[0] / missile->totalMass,
		totalForce[1] / missile->totalMass,
		totalForce[2] / missile->totalMass};

	// Store acceleration
	missile->acceleration[0] = totalAccel[0];
	missile->acceleration[1] = totalAccel[1];
	missile->acceleration[2] = totalAccel[2];

	// INTEGRATE MOTION - CORRECT
	missile->velocity[0] += totalAccel[0] * deltaTime;
	missile->velocity[1] += totalAccel[1] * deltaTime;
	missile->velocity[2] += totalAccel[2] * deltaTime;

	missile->position[0] += missile->velocity[0] * deltaTime;
	missile->position[1] += missile->velocity[1] * deltaTime;
	missile->position[2] += missile->velocity[2] * deltaTime;

	// BODY ORIENTATION - SIMPLIFIED BUT PHYSICAL
	// Missile tries to align with velocity + some lead for maneuvering
	float desiredOrientation[3];

	if (commandedAccelMag > 0.1f) {
		// Lead the velocity vector for better turning
		float leadFactor = 0.3f;
		desiredOrientation[0] = currentDir[0] + commandedAccel[0] * leadFactor;
		desiredOrientation[1] = currentDir[1] + commandedAccel[1] * leadFactor;
		desiredOrientation[2] = currentDir[2] + commandedAccel[2] * leadFactor;
	} else {
		desiredOrientation[0] = currentDir[0];
		desiredOrientation[1] = currentDir[1];
		desiredOrientation[2] = currentDir[2];
	}

	// Normalize desired orientation
	float desiredMag = sqrtf(desiredOrientation[0] * desiredOrientation[0] +
							 desiredOrientation[1] * desiredOrientation[1] +
							 desiredOrientation[2] * desiredOrientation[2]);
	if (desiredMag > 0.001f) {
		desiredOrientation[0] /= desiredMag;
		desiredOrientation[1] /= desiredMag;
		desiredOrientation[2] /= desiredMag;
	}

	// Smooth rotation toward desired orientation
	float rotationSpeed = 2.0f; // rad/s
	float maxRotation = rotationSpeed * deltaTime;

	float currentToDesired[3] = {
		desiredOrientation[0] - missile->bodyOrientation[0],
		desiredOrientation[1] - missile->bodyOrientation[1],
		desiredOrientation[2] - missile->bodyOrientation[2]};

	float rotationDist = sqrtf(currentToDesired[0] * currentToDesired[0] +
							   currentToDesired[1] * currentToDesired[1] +
							   currentToDesired[2] * currentToDesired[2]);

	if (rotationDist > maxRotation && rotationDist > 0.001f) {
		float scale = maxRotation / rotationDist;
		currentToDesired[0] *= scale;
		currentToDesired[1] *= scale;
		currentToDesired[2] *= scale;
	}

	missile->bodyOrientation[0] += currentToDesired[0];
	missile->bodyOrientation[1] += currentToDesired[1];
	missile->bodyOrientation[2] += currentToDesired[2];

	// Normalize
	float orientMag = sqrtf(missile->bodyOrientation[0] * missile->bodyOrientation[0] +
							missile->bodyOrientation[1] * missile->bodyOrientation[1] +
							missile->bodyOrientation[2] * missile->bodyOrientation[2]);
	if (orientMag > 0.001f) {
		missile->bodyOrientation[0] /= orientMag;
		missile->bodyOrientation[1] /= orientMag;
		missile->bodyOrientation[2] /= orientMag;
	}

	// Update fire simulation
	if (missile->fireSim) {
		missile->fireSim->basePosition[0] = missile->position[0];
		missile->fireSim->basePosition[1] = missile->position[1];
		missile->fireSim->basePosition[2] = missile->position[2];

		if (speed > 0.1f) {
			missile->fireSim->windDirection[0] = -missile->velocity[0] * 8.0f;
			missile->fireSim->windDirection[1] = -missile->velocity[1] * 8.0f;
			missile->fireSim->windDirection[2] = -missile->velocity[2] * 8.0f;
		}

		fireSimStep(missile->fireSim, deltaTime, fireSimulationTime);
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
						(end.tv_nsec - start.tv_nsec) / 1e6);
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
	particles->maxLifeTime = 1.25f;

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
	particles->particlesSize = 12.0f;
	particles->windDirection[0] = 0.0f;
	particles->windDirection[1] = 0.0f;
	particles->windDirection[2] = 0.0f;
	particles->swirlIntensity = 10.0f;
	particles->swirlFrequency = 10.5f;

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

void InitializeMissiles(struct Missiles *missiles, int count, struct Triangles *model) {
	missiles->count = min(count, MAX_FIRE_SIMS);
	missiles->missileModel = model;

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

		missiles->active[i] = true;
	}
}

void UpdateAllMissiles(struct Missiles *missiles, float deltaTime, float *simTime, float *fireSimulationTime) {
	for (int i = 0; i < missiles->count; i++) {
		if (missiles->active[i] && missiles->missiles[i]->seeker.lockState != Lunching) {
			missileSimStep(missiles->missiles[i], deltaTime, simTime, &missiles->active[i], fireSimulationTime);
		} else {
			missileSeekStep(missiles->missiles[i]);
		}
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

// old code for reference
// void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook, bool *active) {
// 	struct timespec start, end;
// 	clock_gettime(CLOCK_MONOTONIC, &start);

// 	missile->remainingTime -= deltaTime;
// 	if (missile->remainingTime <= 0.0f) {
// 		*active = false;
// 		clock_gettime(CLOCK_MONOTONIC, &end);
// 		*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
// 							(end.tv_nsec - start.tv_nsec) / 1e6);
// 		return;
// 	}

// 	float totalMass = missile->dryMass + missile->fuelMass;

// 	float thrust = 0.0f;
// 	if (missile->burning && missile->fuelMass > 0.0f) {
// 		float exhaustVelocity = missile->Isp * G;
// 		float massFlowRate = missile->burnRate;

// 		float fuelConsumed = massFlowRate * deltaTime;
// 		if (fuelConsumed > missile->fuelMass) {
// 			fuelConsumed = missile->fuelMass;
// 			missile->burning = 0;
// 		}
// 		missile->fuelMass -= fuelConsumed;

// 		thrust = massFlowRate * exhaustVelocity;

// 		float energyReleasedPerSecond = missile->Q_spec * massFlowRate;
// 		float theoreticalThrust = 2.0f * energyReleasedPerSecond / exhaustVelocity;
// 		float combustionEfficiency = fminf(1.0f, thrust / (theoreticalThrust + 0.001f));

// 		if (combustionEfficiency < 0.95f) {
// 			thrust *= 1.0f + (1.0f - combustionEfficiency) * 0.1f;
// 		}
// 	}

// 	float speed = sqrtf(missile->velocity[0] * missile->velocity[0] +
// 						missile->velocity[1] * missile->velocity[1] +
// 						missile->velocity[2] * missile->velocity[2]);

// 	float altitude = missile->position[1];
// 	if (altitude < 0.0f) altitude = 0.0f;
// 	float airDensity = getAirDensity(altitude);
// 	float mach = speed / SPEED_OF_SOUND;
// 	float machDragMult = getMachDragMultiplier(mach, missile->transsonicDragPeak, missile->supersonicDragFactor);
// 	float dynamicPressure = 0.5f * airDensity * speed * speed;

// 	float kineticEnergy = 0.5f * totalMass * speed * speed;
// 	float potentialEnergy = totalMass * G * altitude;
// 	float totalEnergy = kineticEnergy + potentialEnergy;
// 	float maxPossibleKE = 0.5f * totalMass * missile->optimalSpeed * missile->optimalSpeed;
// 	float energyRatio = kineticEnergy / maxPossibleKE;

// 	float currentDir[3] = {0.0f, 0.0f, 0.0f};
// 	if (speed > 0.1f) {
// 		currentDir[0] = missile->velocity[0] / speed;
// 		currentDir[1] = missile->velocity[1] / speed;
// 		currentDir[2] = missile->velocity[2] / speed;
// 	}

// 	float errorDir[3] = {
// 		missile->targetDirection[0] - currentDir[0],
// 		missile->targetDirection[1] - currentDir[1],
// 		missile->targetDirection[2] - currentDir[2]};

// 	float errorMag = sqrtf(errorDir[0] * errorDir[0] +
// 						   errorDir[1] * errorDir[1] +
// 						   errorDir[2] * errorDir[2]);

// 	if (errorMag > 0.01f) {
// 		errorDir[0] /= errorMag;
// 		errorDir[1] /= errorMag;
// 		errorDir[2] /= errorMag;
// 	}

// 	float maxLateralAccel = missile->maxGPull * G;

// 	if (dynamicPressure > missile->maxDynamicPressure) {
// 		float qLimit = missile->maxDynamicPressure / dynamicPressure;
// 		maxLateralAccel *= qLimit;
// 	}

// 	float controlEffectiveness = (airDensity / SEA_LEVEL_DENSITY) * missile->controlAuthority;
// 	if (speed < 50.0f) {
// 		controlEffectiveness *= (speed / 50.0f);
// 	}
// 	maxLateralAccel *= controlEffectiveness;

// 	maxLateralAccel *= (airDensity / SEA_LEVEL_DENSITY);

// 	if (energyRatio < missile->minEnergyThreshold) {
// 		float energyLimitFactor = energyRatio / missile->minEnergyThreshold;
// 		maxLateralAccel *= (0.3f + 0.7f * energyLimitFactor);
// 	} else if (energyRatio < 0.7f) {
// 		float conservationFactor = missile->energyManagementFactor +
// 								   (1.0f - missile->energyManagementFactor) *
// 									   ((energyRatio - missile->minEnergyThreshold) / (0.7f - missile->minEnergyThreshold));
// 		maxLateralAccel *= conservationFactor;
// 	}

// 	float speedRatio = speed / missile->optimalSpeed;
// 	if (speedRatio < 0.5f) {
// 		maxLateralAccel *= (0.4f + 0.6f * (speedRatio / 0.5f));
// 	}

// 	float guidanceGain = 3.0f;

// 	float dotProduct = currentDir[0] * missile->targetDirection[0] +
// 					   currentDir[1] * missile->targetDirection[1] +
// 					   currentDir[2] * missile->targetDirection[2];
// 	float angleError = acosf(fmaxf(-1.0f, fminf(1.0f, dotProduct)));

// 	if (angleError < 0.2f && energyRatio > 0.8f) {
// 		guidanceGain = 2.0f;
// 	} else if (angleError > 1.0f && energyRatio < 0.5f) {
// 		guidanceGain = 1.5f;
// 	}

// 	float lateralAccel[3] = {
// 		errorDir[0] * maxLateralAccel * guidanceGain,
// 		errorDir[1] * maxLateralAccel * guidanceGain,
// 		errorDir[2] * maxLateralAccel * guidanceGain};

// 	float lateralAccelMag = sqrtf(lateralAccel[0] * lateralAccel[0] +
// 								  lateralAccel[1] * lateralAccel[1] +
// 								  lateralAccel[2] * lateralAccel[2]);
// 	if (lateralAccelMag > maxLateralAccel) {
// 		float scale = maxLateralAccel / lateralAccelMag;
// 		lateralAccel[0] *= scale;
// 		lateralAccel[1] *= scale;
// 		lateralAccel[2] *= scale;
// 	}

// 	float dragForce[3] = {
// 		-missile->drag * machDragMult * airDensity * missile->velocity[0] * speed,
// 		-missile->drag * machDragMult * airDensity * missile->velocity[1] * speed,
// 		-missile->drag * machDragMult * airDensity * missile->velocity[2] * speed};

// 	float inducedDragMag = missile->inducedDragFactor * lateralAccelMag / G * speed * airDensity;
// 	float inducedDragForce[3] = {
// 		-missile->velocity[0] * inducedDragMag / (speed + 0.001f),
// 		-missile->velocity[1] * inducedDragMag / (speed + 0.001f),
// 		-missile->velocity[2] * inducedDragMag / (speed + 0.001f)};

// 	float liftForce[3] = {0.0f, 0.0f, 0.0f};
// 	if (speed > 10.0f) {
// 		float dotProduct = currentDir[0] * missile->bodyOrientation[0] +
// 						   currentDir[1] * missile->bodyOrientation[1] +
// 						   currentDir[2] * missile->bodyOrientation[2];
// 		float aoa = acosf(fmaxf(-1.0f, fminf(1.0f, dotProduct)));

// 		if (aoa > 0.01f && aoa < 0.5f) {
// 			float liftMag = missile->liftCoefficient * dynamicPressure *
// 							missile->crossSectionArea * sinf(aoa);

// 			float liftDir[3];
// 			liftDir[0] = missile->bodyOrientation[1] * currentDir[2] -
// 						 missile->bodyOrientation[2] * currentDir[1];
// 			liftDir[1] = missile->bodyOrientation[2] * currentDir[0] -
// 						 missile->bodyOrientation[0] * currentDir[2];
// 			liftDir[2] = missile->bodyOrientation[0] * currentDir[1] -
// 						 missile->bodyOrientation[1] * currentDir[0];

// 			float liftDirMag = sqrtf(liftDir[0] * liftDir[0] +
// 									 liftDir[1] * liftDir[1] +
// 									 liftDir[2] * liftDir[2]);

// 			if (liftDirMag > 0.001f) {
// 				liftForce[0] = (liftDir[0] / liftDirMag) * liftMag;
// 				liftForce[1] = (liftDir[1] / liftDirMag) * liftMag;
// 				liftForce[2] = (liftDir[2] / liftDirMag) * liftMag;
// 			}
// 		}
// 	}

// 	float thrustAccel[3] = {0.0f, 0.0f, 0.0f};
// 	if (speed > 0.1f && thrust > 0.0f) {
// 		float thrustMag = thrust / totalMass;

// 		float altitudeFactor = airDensity / SEA_LEVEL_DENSITY;
// 		thrustMag *= (0.7f + 0.3f * altitudeFactor);

// 		float dotProduct = currentDir[0] * missile->targetDirection[0] +
// 						   currentDir[1] * missile->targetDirection[1] +
// 						   currentDir[2] * missile->targetDirection[2];
// 		float angleOfAttack = acosf(fmaxf(-1.0f, fminf(1.0f, dotProduct)));
// 		float thrustEfficiency = missile->thrustVectoringEfficiency +
// 								 (1.0f - missile->thrustVectoringEfficiency) * cosf(angleOfAttack);

// 		if (energyRatio < 0.6f && speedRatio < 0.8f) {
// 			float energyBoost = 1.0f + (1.0f - energyRatio) * 0.3f;
// 			thrustMag *= energyBoost;
// 		}

// 		thrustAccel[0] = currentDir[0] * thrustMag * thrustEfficiency;
// 		thrustAccel[1] = currentDir[1] * thrustMag * thrustEfficiency;
// 		thrustAccel[2] = currentDir[2] * thrustMag * thrustEfficiency;
// 	}

// 	float dragAccel[3] = {
// 		(dragForce[0] + inducedDragForce[0]) / totalMass,
// 		(dragForce[1] + inducedDragForce[1]) / totalMass,
// 		(dragForce[2] + inducedDragForce[2]) / totalMass};

// 	float liftAccel[3] = {
// 		liftForce[0] / totalMass,
// 		liftForce[1] / totalMass,
// 		liftForce[2] / totalMass};

// 	float gravityAccel[3] = {0.0f, -G, 0.0f};

// 	float totalAccel[3] = {
// 		thrustAccel[0] + dragAccel[0] + lateralAccel[0] + liftAccel[0] + gravityAccel[0],
// 		thrustAccel[1] + dragAccel[1] + lateralAccel[1] + liftAccel[1] + gravityAccel[1],
// 		thrustAccel[2] + dragAccel[2] + lateralAccel[2] + liftAccel[2] + gravityAccel[2]};

// 	missile->velocity[0] += totalAccel[0] * deltaTime;
// 	missile->velocity[1] += totalAccel[1] * deltaTime;
// 	missile->velocity[2] += totalAccel[2] * deltaTime;

// 	missile->position[0] += missile->velocity[0] * deltaTime;
// 	missile->position[1] += missile->velocity[1] * deltaTime;
// 	missile->position[2] += missile->velocity[2] * deltaTime;

// 	float orientationError[3] = {
// 		currentDir[0] - missile->bodyOrientation[0],
// 		currentDir[1] - missile->bodyOrientation[1],
// 		currentDir[2] - missile->bodyOrientation[2]};

// 	float turnRate = 5.0f * controlEffectiveness;

// 	float torque[3] = {
// 		orientationError[0] * missile->controlAuthority * 100.0f,
// 		orientationError[1] * missile->controlAuthority * 100.0f,
// 		orientationError[2] * missile->controlAuthority * 100.0f};

// 	float angularAccel[3] = {
// 		torque[0] / missile->momentOfInertia,
// 		torque[1] / missile->momentOfInertia,
// 		torque[2] / missile->momentOfInertia};

// 	missile->angularVelocity[0] += angularAccel[0] * deltaTime;
// 	missile->angularVelocity[1] += angularAccel[1] * deltaTime;
// 	missile->angularVelocity[2] += angularAccel[2] * deltaTime;

// 	float angularDamping = 0.95f;
// 	missile->angularVelocity[0] *= angularDamping;
// 	missile->angularVelocity[1] *= angularDamping;
// 	missile->angularVelocity[2] *= angularDamping;

// 	missile->bodyOrientation[0] += missile->angularVelocity[0] * deltaTime;
// 	missile->bodyOrientation[1] += missile->angularVelocity[1] * deltaTime;
// 	missile->bodyOrientation[2] += missile->angularVelocity[2] * deltaTime;

// 	float bodyOrientMag = sqrtf(
// 		missile->bodyOrientation[0] * missile->bodyOrientation[0] +
// 		missile->bodyOrientation[1] * missile->bodyOrientation[1] +
// 		missile->bodyOrientation[2] * missile->bodyOrientation[2]);

// 	if (bodyOrientMag > 0.001f) {
// 		missile->bodyOrientation[0] /= bodyOrientMag;
// 		missile->bodyOrientation[1] /= bodyOrientMag;
// 		missile->bodyOrientation[2] /= bodyOrientMag;
// 	}

// 	if (missile->fireSim) {
// 		missile->fireSim->basePosition[0] = missile->position[0];
// 		missile->fireSim->basePosition[1] = missile->position[1];
// 		missile->fireSim->basePosition[2] = missile->position[2];

// 		float newSpeed = sqrtf(missile->velocity[0] * missile->velocity[0] +
// 							   missile->velocity[1] * missile->velocity[1] +
// 							   missile->velocity[2] * missile->velocity[2]);

// 		if (newSpeed > 0.1f) {
// 			missile->fireSim->windDirection[0] = -missile->velocity[0] * 10.0f;
// 			missile->fireSim->windDirection[1] = -missile->velocity[1] * 10.0f;
// 			missile->fireSim->windDirection[2] = -missile->velocity[2] * 10.0f;
// 		}

// 		float fireStepTime;
// 		fireSimStep(missile->fireSim, deltaTime, &fireStepTime);
// 	}

// 	clock_gettime(CLOCK_MONOTONIC, &end);
// 	*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
// 						(end.tv_nsec - start.tv_nsec) / 1e6);
// }

// void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook, bool *active) {
//     struct timespec start, end;
//     clock_gettime(CLOCK_MONOTONIC, &start);

//     missile->remainingTime -= deltaTime;
//     if (missile->remainingTime <= 0.0f) {
//         *active = false;
//         clock_gettime(CLOCK_MONOTONIC, &end);
//         *timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
//                             (end.tv_nsec - start.tv_nsec) / 1e6);
//         return;
//     }

//     // Update mass and cache values
//     missile->totalMass = missile->dryMass + missile->fuelMass;
//     float speed = sqrtf(missile->velocity[0] * missile->velocity[0] +
//                         missile->velocity[1] * missile->velocity[1] +
//                         missile->velocity[2] * missile->velocity[2]);

//     float altitude = fmaxf(0.0f, missile->position[1]);
//     float airDensity = getAirDensity(altitude);
//     missile->machNumber = speed / SPEED_OF_SOUND;
//     missile->dynamicPressure = 0.5f * airDensity * speed * speed;

//     // REALISTIC PROPULSION SYSTEM
//     missile->thrust = 0.0f;
//     if (missile->burning && missile->fuelMass > 0.0f) {
//         float massFlowRate = missile->burnRate;
//         float fuelConsumed = massFlowRate * deltaTime;

//         if (fuelConsumed > missile->fuelMass) {
//             fuelConsumed = missile->fuelMass;
//             missile->burning = 0;
//         }
//         missile->fuelMass -= fuelConsumed;

//         // Rocket thrust: F = ṁ * Ve + (Pe - Pamb) * Ae
//         float exhaustVelocity = missile->Isp * G;
//         missile->thrust = massFlowRate * exhaustVelocity;

//         // Altitude compensation (nozzle expansion)
//         float ambientPressure = airDensity * 287.0f * 288.0f; // P = ρRT approx
//         float seaLevelPressure = 101325.0f; // Standard sea level pressure
//         float pressureRatio = 1.0f - (ambientPressure / seaLevelPressure) * 0.2f;
//         missile->thrust *= (1.0f + pressureRatio * 0.25f); // Up to 25% thrust increase
//     }

//     // AERODYNAMIC ANGLES CALCULATION
//     float currentDir[3];
//     if (speed > 0.1f) {
//         currentDir[0] = missile->velocity[0] / speed;
//         currentDir[1] = missile->velocity[1] / speed;
//         currentDir[2] = missile->velocity[2] / speed;
//     } else {
//         currentDir[0] = missile->bodyOrientation[0];
//         currentDir[1] = missile->bodyOrientation[1];
//         currentDir[2] = missile->bodyOrientation[2];
//     }

//     // Calculate angle of attack and sideslip
//     float bodyDotVelocity = missile->bodyOrientation[0] * currentDir[0] +
//                            missile->bodyOrientation[1] * currentDir[1] +
//                            missile->bodyOrientation[2] * currentDir[2];
//     missile->angleOfAttack = acosf(fmaxf(-1.0f, fminf(1.0f, bodyDotVelocity)));

//     // Simple sideslip approximation
//     float sideDir[3] = {
//         missile->bodyOrientation[1] * currentDir[2] - missile->bodyOrientation[2] * currentDir[1],
//         missile->bodyOrientation[2] * currentDir[0] - missile->bodyOrientation[0] * currentDir[2],
//         missile->bodyOrientation[0] * currentDir[1] - missile->bodyOrientation[1] * currentDir[0]
//     };
//     missile->sideslipAngle = asinf(fmaxf(-1.0f, fminf(1.0f, sqrtf(sideDir[0]*sideDir[0] + sideDir[1]*sideDir[1] + sideDir[2]*sideDir[2]))));

//     // PROPORTIONAL NAVIGATION GUIDANCE
//     float losRate[3] = {0.0f, 0.0f, 0.0f};

//     // Calculate line-of-sight to target
//     float los[3] = {
//         missile->targetPosition[0] - missile->position[0],
//         missile->targetPosition[1] - missile->position[1],
//         missile->targetPosition[2] - missile->position[2]
//     };

//     float losDistance = sqrtf(los[0]*los[0] + los[1]*los[1] + los[2]*los[2]);
//     if (losDistance > 0.1f) {
//         los[0] /= losDistance;
//         los[1] /= losDistance;
//         los[2] /= losDistance;

//         // Simple LOS rate approximation
//         losRate[0] = los[0] - missile->targetDirection[0];
//         losRate[1] = los[1] - missile->targetDirection[1];
//         losRate[2] = los[2] - missile->targetDirection[2];

//         // Update target direction
//         missile->targetDirection[0] = los[0];
//         missile->targetDirection[1] = los[1];
//         missile->targetDirection[2] = los[2];
//     }

//     // Proportional navigation command
//     float closingSpeed = fmaxf(speed, 100.0f); // Minimum for guidance
//     float navGain = missile->guidanceGain * closingSpeed;

//     float commandedAccel[3] = {
//         navGain * losRate[0],
//         navGain * losRate[1],
//         navGain * losRate[2]
//     };

//     // Remove component along velocity to get pure lateral acceleration
//     float parallelComp = commandedAccel[0]*currentDir[0] + commandedAccel[1]*currentDir[1] + commandedAccel[2]*currentDir[2];
//     commandedAccel[0] -= parallelComp * currentDir[0];
//     commandedAccel[1] -= parallelComp * currentDir[1];
//     commandedAccel[2] -= parallelComp * currentDir[2];

//     // AERODYNAMIC PERFORMANCE LIMITS
//     float maxAvailableG = missile->maxGPull * G;

//     // Dynamic pressure limits
//     if (missile->dynamicPressure > missile->maxDynamicPressure) {
//         float qFactor = missile->maxDynamicPressure / (missile->dynamicPressure + 0.001f);
//         maxAvailableG *= qFactor * qFactor; // Quadratic scaling for structural limits
//     }

//     // AoA limits - stall prevention
//     float aoaLimit = fminf(missile->maxAoA, 0.35f); // ~20 degrees max
//     float aoaFactor = 1.0f - fmaxf(0.0f, (missile->angleOfAttack - aoaLimit * 0.7f) / (aoaLimit * 0.3f));
//     maxAvailableG *= fmaxf(0.1f, aoaFactor);

//     // Control effectiveness (air density and speed dependent)
//     float controlEffectiveness = (airDensity / SEA_LEVEL_DENSITY) * missile->controlAuthority;
//     if (speed < 50.0f) {
//         controlEffectiveness *= (speed / 50.0f);
//     }
//     maxAvailableG *= controlEffectiveness;

//     // Limit commanded acceleration
//     float commandedAccelMag = sqrtf(commandedAccel[0]*commandedAccel[0] +
//                                    commandedAccel[1]*commandedAccel[1] +
//                                    commandedAccel[2]*commandedAccel[2]);

//     if (commandedAccelMag > maxAvailableG && commandedAccelMag > 0.001f) {
//         float scale = maxAvailableG / commandedAccelMag;
//         commandedAccel[0] *= scale;
//         commandedAccel[1] *= scale;
//         commandedAccel[2] *= scale;
//         commandedAccelMag = maxAvailableG;
//     }

//     // FIN CONTROL SYSTEM
//     // Convert acceleration command to fin deflections
//     float lateralDir[3];
//     if (commandedAccelMag > 0.1f) {
//         lateralDir[0] = commandedAccel[0] / commandedAccelMag;
//         lateralDir[1] = commandedAccel[1] / commandedAccelMag;
//         lateralDir[2] = commandedAccel[2] / commandedAccelMag;
//     } else {
//         lateralDir[0] = lateralDir[1] = lateralDir[2] = 0.0f;
//     }

//     // Calculate required fin deflections
//     // For simplicity: 4 fins in + configuration, controlling pitch and yaw
//     float pitchDemand = lateralDir[1];  // Simplified
//     float yawDemand = lateralDir[0];    // Simplified

//     // Limit fin deflections
//     missile->finDeflection[0] = fmaxf(-missile->finMaxDeflection, fminf(missile->finMaxDeflection, pitchDemand + yawDemand));
//     missile->finDeflection[1] = fmaxf(-missile->finMaxDeflection, fminf(missile->finMaxDeflection, -pitchDemand + yawDemand));
//     missile->finDeflection[2] = fmaxf(-missile->finMaxDeflection, fminf(missile->finMaxDeflection, -pitchDemand - yawDemand));
//     missile->finDeflection[3] = fmaxf(-missile->finMaxDeflection, fminf(missile->finMaxDeflection, pitchDemand - yawDemand));

//     // THRUST VECTORING CONTROL
//     // Calculate gimbal angles based on acceleration demand
//     if (missile->thrust > 0.0f && missile->maxGimbalAngle > 0.001f) {
//         float tvGain = 0.5f; // Thrust vectoring gain

//         missile->gimbalAngle[0] = fmaxf(-missile->maxGimbalAngle,
//                                        fminf(missile->maxGimbalAngle, pitchDemand * tvGain));
//         missile->gimbalAngle[1] = fmaxf(-missile->maxGimbalAngle,
//                                        fminf(missile->maxGimbalAngle, yawDemand * tvGain));
//     } else {
//         missile->gimbalAngle[0] = missile->gimbalAngle[1] = 0.0f;
//     }

//     // REALISTIC AERODYNAMIC FORCES
//     // Lift coefficient calculation with fin contribution
//     float baseLiftCoeff = missile->liftSlope * missile->angleOfAttack;
//     float finLiftCoeff = missile->finEffectiveness * (missile->finDeflection[0] + missile->finDeflection[1] +
//                                                      missile->finDeflection[2] + missile->finDeflection[3]) / 4.0f;

//     float totalLiftCoeff = baseLiftCoeff + finLiftCoeff;
//     totalLiftCoeff = fmaxf(-missile->maxLiftCoeff, fminf(missile->maxLiftCoeff, totalLiftCoeff));

//     // Drag coefficient calculation
//     float inducedDragCoeff = (totalLiftCoeff * totalLiftCoeff) /
//                             (3.14159f * missile->aspectRatio * missile->oswaldEfficiency);
//     float totalDragCoeff = missile->zeroLiftDrag + inducedDragCoeff +
//                           missile->dragSlope * missile->angleOfAttack * missile->angleOfAttack;

//     // Mach effects on coefficients
//     float mach = missile->machNumber;
//     float machCorrection = 1.0f;
//     if (mach > 0.8f && mach < 1.2f) {
//         machCorrection = 1.0f + (mach - 0.8f) * 2.0f; // Transonic drag rise
//     } else if (mach >= 1.2f) {
//         machCorrection = 1.8f - (mach - 1.2f) * 0.3f; // Supersonic drag
//     }
//     totalDragCoeff *= machCorrection;

//     // Calculate forces
//     float liftMagnitude = totalLiftCoeff * missile->dynamicPressure * missile->wingArea;
//     float dragMagnitude = totalDragCoeff * missile->dynamicPressure * missile->crossSectionArea;

//     // Lift direction (perpendicular to velocity and body)
//     float liftDir[3];
//     if (speed > 1.0f) {
//         liftDir[0] = missile->bodyOrientation[1] * currentDir[2] - missile->bodyOrientation[2] * currentDir[1];
//         liftDir[1] = missile->bodyOrientation[2] * currentDir[0] - missile->bodyOrientation[0] * currentDir[2];
//         liftDir[2] = missile->bodyOrientation[0] * currentDir[1] - missile->bodyOrientation[1] * currentDir[0];

//         float liftDirMag = sqrtf(liftDir[0]*liftDir[0] + liftDir[1]*liftDir[1] + liftDir[2]*liftDir[2]);
//         if (liftDirMag > 0.001f) {
//             liftDir[0] /= liftDirMag;
//             liftDir[1] /= liftDirMag;
//             liftDir[2] /= liftDirMag;
//         }
//     } else {
//         liftDir[0] = liftDir[1] = liftDir[2] = 0.0f;
//     }

//     float liftForce[3] = {
//         liftDir[0] * liftMagnitude,
//         liftDir[1] * liftMagnitude,
//         liftDir[2] * liftMagnitude
//     };

//     float dragForce[3] = {
//         -dragMagnitude * currentDir[0],
//         -dragMagnitude * currentDir[1],
//         -dragMagnitude * currentDir[2]
//     };

//     // THRUST FORCE WITH VECTORING
//     float thrustForce[3] = {0.0f, 0.0f, 0.0f};
//     if (missile->thrust > 0.0f) {
//         // Apply gimbal angles to thrust direction
//         float thrustDir[3] = {
//             missile->bodyOrientation[0],
//             missile->bodyOrientation[1],
//             missile->bodyOrientation[2]
//         };

//         // Apply pitch gimbal (rotation around local X-axis)
//         float cosPitch = cosf(missile->gimbalAngle[0]);
//         float sinPitch = sinf(missile->gimbalAngle[0]);
//         float newY = thrustDir[1] * cosPitch - thrustDir[2] * sinPitch;
//         float newZ = thrustDir[1] * sinPitch + thrustDir[2] * cosPitch;
//         thrustDir[1] = newY;
//         thrustDir[2] = newZ;

//         // Apply yaw gimbal (rotation around local Y-axis)
//         float cosYaw = cosf(missile->gimbalAngle[1]);
//         float sinYaw = sinf(missile->gimbalAngle[1]);
//         float newX = thrustDir[0] * cosYaw + thrustDir[2] * sinYaw;
//         newZ = -thrustDir[0] * sinYaw + thrustDir[2] * cosYaw;
//         thrustDir[0] = newX;
//         thrustDir[2] = newZ;

//         thrustForce[0] = thrustDir[0] * missile->thrust;
//         thrustForce[1] = thrustDir[1] * missile->thrust;
//         thrustForce[2] = thrustDir[2] * missile->thrust;
//     }

//     // GRAVITY FORCE
//     float gravityForce[3] = {0.0f, -missile->totalMass * G, 0.0f};

//     // TOTAL FORCES AND ACCELERATION
//     float totalForce[3] = {
//         thrustForce[0] + dragForce[0] + liftForce[0] + gravityForce[0],
//         thrustForce[1] + dragForce[1] + liftForce[1] + gravityForce[1],
//         thrustForce[2] + dragForce[2] + liftForce[2] + gravityForce[2]
//     };

//     float totalAccel[3] = {
//         totalForce[0] / missile->totalMass,
//         totalForce[1] / missile->totalMass,
//         totalForce[2] / missile->totalMass
//     };

//     // Add guidance acceleration (simplified as direct force)
//     totalAccel[0] += commandedAccel[0];
//     totalAccel[1] += commandedAccel[1];
//     totalAccel[2] += commandedAccel[2];

//     // Store acceleration for guidance
//     missile->acceleration[0] = totalAccel[0];
//     missile->acceleration[1] = totalAccel[1];
//     missile->acceleration[2] = totalAccel[2];

//     // INTEGRATE MOTION (Verlet integration for better stability)
//     missile->velocity[0] += totalAccel[0] * deltaTime;
//     missile->velocity[1] += totalAccel[1] * deltaTime;
//     missile->velocity[2] += totalAccel[2] * deltaTime;

//     missile->position[0] += missile->velocity[0] * deltaTime;
//     missile->position[1] += missile->velocity[1] * deltaTime;
//     missile->position[2] += missile->velocity[2] * deltaTime;

//     // BODY ORIENTATION DYNAMICS
//     // Missile naturally aligns with velocity vector due to aerodynamics
//     float desiredOrientation[3] = {currentDir[0], currentDir[1], currentDir[2]};

//     // Add control input from fins
//     float finControlTorque = missile->finEffectiveness * missile->dynamicPressure * 0.1f;
//     float controlInput[3] = {
//         lateralDir[0] * finControlTorque,
//         lateralDir[1] * finControlTorque,
//         lateralDir[2] * finControlTorque
//     };

//     // Aerodynamic restoring torque (weathervaning)
//     float aoaTorque = -missile->angleOfAttack * missile->dynamicPressure * 0.05f;
//     float restoringTorque[3] = {
//         aoaTorque * sideDir[0],
//         aoaTorque * sideDir[1],
//         aoaTorque * sideDir[2]
//     };

//     // Damping torque
//     float dampingTorque[3] = {
//         -missile->angularVelocity[0] * missile->rollDamping,
//         -missile->angularVelocity[1] * missile->rollDamping,
//         -missile->angularVelocity[2] * missile->rollDamping
//     };

//     // Total torque
//     float totalTorque[3] = {
//         controlInput[0] + restoringTorque[0] + dampingTorque[0],
//         controlInput[1] + restoringTorque[1] + dampingTorque[1],
//         controlInput[2] + restoringTorque[2] + dampingTorque[2]
//     };

//     // Angular acceleration
//     float angularAccel[3] = {
//         totalTorque[0] / missile->momentOfInertia,
//         totalTorque[1] / missile->momentOfInertia,
//         totalTorque[2] / missile->momentOfInertia
//     };

//     // Integrate angular velocity and orientation
//     missile->angularVelocity[0] += angularAccel[0] * deltaTime;
//     missile->angularVelocity[1] += angularAccel[1] * deltaTime;
//     missile->angularVelocity[2] += angularAccel[2] * deltaTime;

//     // Apply orientation change
//     float rotation[3] = {
//         missile->angularVelocity[0] * deltaTime,
//         missile->angularVelocity[1] * deltaTime,
//         missile->angularVelocity[2] * deltaTime
//     };

//     // Simple rotation using small angle approximation
//     float rotationMag = sqrtf(rotation[0]*rotation[0] + rotation[1]*rotation[1] + rotation[2]*rotation[2]);
//     if (rotationMag > 0.001f) {
//         float axis[3] = {rotation[0]/rotationMag, rotation[1]/rotationMag, rotation[2]/rotationMag};
//         float sinHalf = sinf(rotationMag * 0.5f);
//         float cosHalf = cosf(rotationMag * 0.5f);

//         // Quaternion rotation (simplified)
//         float newOrientation[3] = {
//             missile->bodyOrientation[0] * cosHalf + (axis[1] * missile->bodyOrientation[2] - axis[2] * missile->bodyOrientation[1]) * sinHalf,
//             missile->bodyOrientation[1] * cosHalf + (axis[2] * missile->bodyOrientation[0] - axis[0] * missile->bodyOrientation[2]) * sinHalf,
//             missile->bodyOrientation[2] * cosHalf + (axis[0] * missile->bodyOrientation[1] - axis[1] * missile->bodyOrientation[0]) * sinHalf
//         };

//         missile->bodyOrientation[0] = newOrientation[0];
//         missile->bodyOrientation[1] = newOrientation[1];
//         missile->bodyOrientation[2] = newOrientation[2];

//         // Normalize
//         float mag = sqrtf(missile->bodyOrientation[0]*missile->bodyOrientation[0] +
//                          missile->bodyOrientation[1]*missile->bodyOrientation[1] +
//                          missile->bodyOrientation[2]*missile->bodyOrientation[2]);
//         if (mag > 0.001f) {
//             missile->bodyOrientation[0] /= mag;
//             missile->bodyOrientation[1] /= mag;
//             missile->bodyOrientation[2] /= mag;
//         }
//     }

//     // Update fire simulation
//     if (missile->fireSim) {
//         missile->fireSim->basePosition[0] = missile->position[0];
//         missile->fireSim->basePosition[1] = missile->position[1];
//         missile->fireSim->basePosition[2] = missile->position[2];

//         if (speed > 0.1f) {
//             missile->fireSim->windDirection[0] = -missile->velocity[0] * 5.0f;
//             missile->fireSim->windDirection[1] = -missile->velocity[1] * 5.0f;
//             missile->fireSim->windDirection[2] = -missile->velocity[2] * 5.0f;
//         }

//         float fireStepTime;
//         fireSimStep(missile->fireSim, deltaTime, &fireStepTime);
//     }

//     clock_gettime(CLOCK_MONOTONIC, &end);
//     *timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
//                         (end.tv_nsec - start.tv_nsec) / 1e6);
// }