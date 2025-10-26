#include "fireSim.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <CL/cl.h>
#define MAX_FLOAT 3.402823466e+38F
#define G 9.81f
#define SCALE 100.0f
#define min(a, b) ((a) < (b) ? (a) : (b))

// GPU: Render distance map (seeker POV)
//   ↓
// CPU: glReadPixels() - copy distance buffer to CPU
//   ↓
// CPU: findClosestObjectToViewCenter() - select best target
//   ↓
// CPU: Missile simulation uses target position

#define SPEED_OF_SOUND 340.29f
#define SEA_LEVEL_DENSITY 1.225f
#define SCALE_HEIGHT 8500.0f

static float dot3(const float a[3], const float b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void normalize3(float v[3]) {
	float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 0.0f) {
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
}

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

// Returns index of hottest object in seeker FOV
// centerWeight: 0.0-1.0, how much center alignment matters
// tempWeight: 0.0-1.0, how much temperature matters
// distWeight: 0.0-1.0, how much closeness matters
// (weights should sum to 1.0 for best results)
int findClosestObjectToViewCenter(
	const float pos[3],
	float dir[3],
	float FOV,
	const float *ObjX,
	const float *ObjY,
	const float *ObjZ,
	const float *tempObj,
	const float *seekerImageDistances,
	int numOfObj,
	float centerWeight,
	float tempWeight,
	float distWeight) {
	if (numOfObj <= 0) return -1;

	float viewDir[3] = {dir[0], dir[1], dir[2]};
	normalize3(viewDir);

	float fovHalfRad = (FOV / 2.0f) * (M_PI / 180.0f);
	float fovCosThreshold = cosf(fovHalfRad);

	int bestIdx = -1;
	float bestScore = -MAX_FLOAT;

	// Find max temp for normalization
	float maxTemp = 0.0f;
	for (int i = 0; i < numOfObj; i++) {
		if (tempObj[i] > maxTemp) maxTemp = tempObj[i];
	}
	if (maxTemp < 0.001f) maxTemp = 1.0f;

	for (int i = 0; i < numOfObj; i++) {
		float objPos[3] = {ObjX[i], ObjY[i], ObjZ[i]};
		float toObj[3] = {
			objPos[0] - pos[0],
			objPos[1] - pos[1],
			objPos[2] - pos[2]};

		float dist = sqrtf(toObj[0] * toObj[0] + toObj[1] * toObj[1] + toObj[2] * toObj[2]);

		if (dist < 0.001f) continue;

		float toObjNorm[3] = {toObj[0] / dist, toObj[1] / dist, toObj[2] / dist};
		float cosAngle = dot3(viewDir, toObjNorm);

		if (cosAngle < fovCosThreshold) continue;

		// Check if object is occluded by geometry using seeker depth buffer
		// Map object to seeker image coordinates
		float rightVec[3], upVec[3];
		// Create coordinate system from view direction
		if (fabsf(viewDir[1]) < 0.99f) {
			rightVec[0] = viewDir[2];
			rightVec[1] = 0.0f;
			rightVec[2] = -viewDir[0];
		} else {
			rightVec[0] = 1.0f;
			rightVec[1] = 0.0f;
			rightVec[2] = 0.0f;
		}
		normalize3(rightVec);
		upVec[0] = rightVec[1] * viewDir[2] - rightVec[2] * viewDir[1];
		upVec[1] = rightVec[2] * viewDir[0] - rightVec[0] * viewDir[2];
		upVec[2] = rightVec[0] * viewDir[1] - rightVec[1] * viewDir[0];

		float xProj = dot3(toObjNorm, rightVec);
		float yProj = dot3(toObjNorm, upVec);

		float angle = acosf(fmaxf(-1.0f, fminf(1.0f, cosAngle)));
		float pixelX = (xProj / tanf(fovHalfRad)) * 0.5f + 0.5f;
		float pixelY = (yProj / tanf(fovHalfRad)) * 0.5f + 0.5f;

		int px = (int)(pixelX * MISSILE_SEEKER_SIZE);
		int py = (int)(pixelY * MISSILE_SEEKER_SIZE);

		// Check if object is occluded
		if (px >= 0 && px < MISSILE_SEEKER_SIZE && py >= 0 && py < MISSILE_SEEKER_SIZE) {
			int pixelIdx = py * MISSILE_SEEKER_SIZE + px;
			float geometryDist = seekerImageDistances[pixelIdx];

			// Object is occluded if geometry is closer (with small tolerance)
			if (geometryDist > 0.0f && geometryDist < dist - 0.1f) {
				continue;
			}
		}

		// Angular score: 1.0 (centered) to 0.0 (edge of FOV)
		float angularScore = 1.0f - (angle / fovHalfRad);

		// Temperature with inverse square falloff (like real IR radiation)
		float tempAtSeeker = (tempObj[i] / maxTemp) / (dist * dist);
		float tempScore = tempAtSeeker;

		// Distance score: inverse so closer is better
		float distScore = 1.0f / (1.0f + dist);

		// Combined score (higher is better)
		float score = centerWeight * angularScore +
					  tempWeight * tempScore +
					  distWeight * distScore;

		if (score > bestScore) {
			bestScore = score;
			bestIdx = i;
		}
	}

	return bestIdx;
}

void InitializeMissile(struct Missile *missile) {
	// Seeker initialization
	missile->seeker.seekerCamera.ray.origin[0] = 0.0f;
	missile->seeker.seekerCamera.ray.origin[1] = 0.0f;
	missile->seeker.seekerCamera.ray.origin[2] = 0.0f;
	missile->seeker.seekerCamera.ray.direction[0] = 1.0f;
	missile->seeker.seekerCamera.ray.direction[1] = 0.0f;
	missile->seeker.seekerCamera.ray.direction[2] = 0.0f;
	missile->seeker.seekerCamera.fov = randRange(0.8f, 8.0f);
	missile->seeker.seekerFov = 45.0f;
	missile->seeker.lockState = Lunching;
	missile->seeker.searchMultiplayer = randRange(1.25f, 2.5f);
	missile->seeker.tiltSpeed = randRange(2.0f, 5.0f);

	// Core simulation
	missile->position[0] = randRange(-450.0f, 450.0f);
	missile->position[1] = randRange(250.0f, 1500.0f);
	missile->position[2] = randRange(-450.0f, 450.0f);

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
	missile->fuelMass = randRange(150.0f, 500.0f);
	missile->totalMass = missile->dryMass + missile->fuelMass;
	missile->momentOfInertia = randRange(1.8f, 12.5f);

	// Propulsion
	missile->thrust = 0.0f;
	missile->Isp = randRange(220.0f, 350.0f);
	missile->burning = 1;
	missile->burnRate = randRange(1.0f, 25.0f);
	missile->maxGimbalAngle = randRange(0.00f, 0.25f);
	missile->gimbalAngle[0] = 0.0f;
	missile->gimbalAngle[1] = 0.0f;

	// Aerodynamic coefficients
	missile->zeroLiftDrag = randRange(0.012f, 0.025f);
	missile->liftSlope = randRange(2.0f, 4.0f);
	missile->maxLiftCoeff = randRange(1.2f, 5.0f);
	missile->dragSlope = randRange(0.02f, 0.08f);
	missile->crossSectionArea = randRange(0.015f, 0.145f);
	missile->wingArea = randRange(0.12f, 1.85f);
	missile->aspectRatio = randRange(2.0f, 7.5f);
	missile->oswaldEfficiency = randRange(0.7f, 0.95f);

	// Control surfaces
	missile->finMaxDeflection = randRange(0.2f, 0.85f);
	for (int i = 0; i < 4; i++)
		missile->finDeflection[i] = 0.0f;
	missile->finEffectiveness = randRange(0.7f, 1.95f);
	missile->rollDamping = randRange(0.05f, 0.45f);

	// Performance limits
	missile->maxGPull = randRange(5.0f, 20.0f);
	missile->maxDynamicPressure = randRange(80000.0f, 250000.0f);
	missile->maxAoA = randRange(0.25f, 0.95f);
	missile->maxLoadFactor = randRange(30.0f, 90.0f);
	missile->maxSpeed = randRange(600.0f, 1200.0f);

	// Guidance & control
	missile->guidanceGain = randRange(1.5f, 2.5f);
	missile->controlAuthority = randRange(0.85f, 0.98f);
	missile->energyManagementFactor = randRange(0.45f, 0.95f);
	missile->optimalSpeed = randRange(300.0f, 600.0f);

	// Sensor parameters
	missile->searchConeAngle = randRange(15.0f, 35.0f) * (M_PI / 180.0f);
	missile->searchConeDepth = MAX_FLOAT;
	missile->trackingConeAngle = randRange(15.0f, 30.0f) * (M_PI / 180.0f);
	missile->trackingConeDepth = randRange(4000.0f, 8000.0f);
	missile->sensorFusionWeight = randRange(0.6f, 0.85f);
	missile->engineSignalSensitivity = randRange(0.8f, 1.5f);
	missile->velocitySignalSensitivity = randRange(0.7f, 1.3f);
	missile->minTrackConfidence = randRange(0.1f, 0.3f);

	// Simulation
	missile->remainingTime = randRange(60.0f, 120.0f);
	missile->fireSim = malloc(sizeof(struct FireSOA));
	InitializeFireParticles(missile->fireSim);

	// Cached values
	missile->machNumber = 0.0f;
	missile->dynamicPressure = 0.0f;
	missile->prevLOS[0] = 0.0f;
	missile->prevLOS[1] = 0.0f;
	missile->prevLOS[2] = 0.0f;

	missile->targetIdx = -1;
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
void updateSeekerPositions(struct Missiles *missiles) {
	int count = 0;
	for (int i = 0; i < missiles->count; i++) {
		if (missiles->active[i]) {
			struct Missile *missile = missiles->missiles[i];
			if (missile == NULL) continue;
			if (missiles->active[i] == true) {
				missiles->coneOriginsX[count] = missile->seeker.seekerCamera.ray.origin[0];
				missiles->coneOriginsY[count] = missile->seeker.seekerCamera.ray.origin[1];
				missiles->coneOriginsZ[count] = missile->seeker.seekerCamera.ray.origin[2];

				missiles->coneDirsX[count] = missile->seeker.seekerCamera.ray.direction[0];
				missiles->coneDirsY[count] = missile->seeker.seekerCamera.ray.direction[1];
				missiles->coneDirsZ[count] = missile->seeker.seekerCamera.ray.direction[2];

				float fov = missile->seeker.seekerFov;
				if (missile->seeker.lockState == Searching) {
					fov *= missile->seeker.searchMultiplayer;
				}

				missiles->coneFovs[count] = fov;
				missiles->coneMaxDistances[count] = 1000.0f;
				count++;
			}
		}
	}
	missiles->activeCount = count;
}

void missileSeekStep(struct Missile *missile, struct Missiles *allMissiles, bool fire, bool *active, float deltaTime, float *timeTook, float *fireSimulationTime, float lunchDir[3], float lunchPos[3]) {
	float *rayDir = missile->seeker.seekerCamera.ray.direction;
	float *rayOrigin = missile->seeker.seekerCamera.ray.origin;

	float seekerOffset = 15.0f;
	rayOrigin[0] = missile->position[0] + missile->bodyOrientation[0] * seekerOffset;
	rayOrigin[1] = missile->position[1] + missile->bodyOrientation[1] * seekerOffset;
	rayOrigin[2] = missile->position[2] + missile->bodyOrientation[2] * seekerOffset;

	if (*active) {
		float minDistance = MAX_FLOAT;
		int closePixels = 0;
		const float avoidanceThreshold = 200.0f;
		const float criticalThreshold = 40.0f;

		int regions[9] = {0};
		float regionMinDist[9];
		for (int i = 0; i < 9; i++)
			regionMinDist[i] = MAX_FLOAT;

		int gridSize = MISSILE_SEEKER_SIZE / 3;

		for (int y = 0; y < MISSILE_SEEKER_SIZE; y++) {
			for (int x = 0; x < MISSILE_SEEKER_SIZE; x++) {
				int i = y * MISSILE_SEEKER_SIZE + x;
				float dist = missile->seeker.seekerDepthMap[i];

				if (dist > 0.001f && dist < avoidanceThreshold) {
					closePixels++;
					if (dist < minDistance) minDistance = dist;

					int regionX = x / gridSize;
					int regionY = y / gridSize;
					if (regionX > 2) regionX = 2;
					if (regionY > 2) regionY = 2;
					int regionIdx = regionY * 3 + regionX;

					regions[regionIdx]++;
					if (dist < regionMinDist[regionIdx]) {
						regionMinDist[regionIdx] = dist;
					}
				}
			}
		}

		float groundClearance = missile->position[1];
		float minAltitude = 10.0f;
		bool nearGround = groundClearance < 25.0f;

		int leftThreat = regions[0] + regions[3] + regions[6];
		int rightThreat = regions[2] + regions[5] + regions[8];
		int topThreat = regions[0] + regions[1] + regions[2];
		int bottomThreat = regions[6] + regions[7] + regions[8];
		int centerThreat = regions[4];

		if ((closePixels > 5 && minDistance < avoidanceThreshold) || nearGround) {
			float avoidanceDir[3] = {0.0f, 0.0f, 0.0f};

			float forward[3] = {rayDir[0], rayDir[1], rayDir[2]};
			float right[3], up[3];

			if (fabsf(forward[1]) < 0.99f) {
				right[0] = forward[2];
				right[1] = 0.0f;
				right[2] = -forward[0];
			} else {
				right[0] = 1.0f;
				right[1] = 0.0f;
				right[2] = 0.0f;
			}
			normalize3(right);

			up[0] = right[1] * forward[2] - right[2] * forward[1];
			up[1] = right[2] * forward[0] - right[0] * forward[2];
			up[2] = right[0] * forward[1] - right[1] * forward[0];
			normalize3(up);

			float leftMinDist = fminf(fminf(regionMinDist[0], regionMinDist[3]), regionMinDist[6]);
			float rightMinDist = fminf(fminf(regionMinDist[2], regionMinDist[5]), regionMinDist[8]);
			float topMinDist = fminf(fminf(regionMinDist[0], regionMinDist[1]), regionMinDist[2]);
			float bottomMinDist = fminf(fminf(regionMinDist[6], regionMinDist[7]), regionMinDist[8]);

			if (leftThreat > 0 || rightThreat > 0) {
				float leftDanger = leftThreat * (1.0f - leftMinDist / avoidanceThreshold);
				float rightDanger = rightThreat * (1.0f - rightMinDist / avoidanceThreshold);

				if (leftDanger > rightDanger) {
					float bias = fminf(1.0f, (leftDanger - rightDanger) / (float)(leftThreat + 1));
					avoidanceDir[0] += right[0] * bias;
					avoidanceDir[1] += right[1] * bias;
					avoidanceDir[2] += right[2] * bias;
				} else {
					float bias = fminf(1.0f, (rightDanger - leftDanger) / (float)(rightThreat + 1));
					avoidanceDir[0] -= right[0] * bias;
					avoidanceDir[1] -= right[1] * bias;
					avoidanceDir[2] -= right[2] * bias;
				}
			}

			float upwardBias = 0.0f;

			if (groundClearance < minAltitude) {
				upwardBias = 2.0f * (1.0f - groundClearance / minAltitude);
			} else if (nearGround) {
				upwardBias = 0.5f * (1.0f - groundClearance / 100.0f);
			}

			if (bottomThreat > topThreat || minDistance < 80.0f) {
				float bottomDanger = bottomThreat * (1.0f - bottomMinDist / avoidanceThreshold);
				float topDanger = topThreat * (1.0f - topMinDist / avoidanceThreshold);
				float bias = fmaxf(upwardBias, (bottomDanger - topDanger) / (float)(bottomThreat + 1));
				avoidanceDir[0] += up[0] * bias * 2.0f;
				avoidanceDir[1] += up[1] * bias * 2.0f;
				avoidanceDir[2] += up[2] * bias * 2.0f;
			} else if (topThreat > bottomThreat && groundClearance > 150.0f) {
				float topDanger = topThreat * (1.0f - topMinDist / avoidanceThreshold);
				float bottomDanger = bottomThreat * (1.0f - bottomMinDist / avoidanceThreshold);
				float bias = fminf(0.3f, (topDanger - bottomDanger) / (float)(topThreat + 1));
				avoidanceDir[0] -= up[0] * bias;
				avoidanceDir[1] -= up[1] * bias;
				avoidanceDir[2] -= up[2] * bias;
			} else if (upwardBias > 0.0f) {
				avoidanceDir[0] += up[0] * upwardBias * 1.5f;
				avoidanceDir[1] += up[1] * upwardBias * 1.5f;
				avoidanceDir[2] += up[2] * upwardBias * 1.5f;
			}

			if (centerThreat > 10 && minDistance < 100.0f) {
				int bestEscape = -1;
				float bestScore = -1.0f;

				for (int i = 0; i < 9; i++) {
					if (i == 4) continue;
					float score = (avoidanceThreshold - regionMinDist[i]) / (float)(regions[i] + 1);
					if (score > bestScore) {
						bestScore = score;
						bestEscape = i;
					}
				}

				if (bestEscape >= 0) {
					int escapeX = (bestEscape % 3) - 1;
					int escapeY = (bestEscape / 3) - 1;

					avoidanceDir[0] += right[0] * escapeX * 0.5f;
					avoidanceDir[1] += right[1] * escapeX * 0.5f;
					avoidanceDir[2] += right[2] * escapeX * 0.5f;

					avoidanceDir[0] -= up[0] * escapeY * 0.5f;
					avoidanceDir[1] -= up[1] * escapeY * 0.5f;
					avoidanceDir[2] -= up[2] * escapeY * 0.5f;
				}
			}

			float avoidMag = sqrtf(avoidanceDir[0] * avoidanceDir[0] +
								   avoidanceDir[1] * avoidanceDir[1] +
								   avoidanceDir[2] * avoidanceDir[2]);

			if (avoidMag > 0.001f) {
				avoidanceDir[0] /= avoidMag;
				avoidanceDir[1] /= avoidMag;
				avoidanceDir[2] /= avoidMag;

				float urgency = 1.0f - (minDistance / avoidanceThreshold);
				urgency = urgency * urgency * urgency;

				if (nearGround) urgency = fmaxf(urgency, 0.6f);
				if (groundClearance < minAltitude) urgency = 1.0f;

				float avoidWeight = fminf(0.85f, urgency);
				float targetWeight = 1.0f - avoidWeight;

				missile->targetDirection[0] = missile->targetDirection[0] * targetWeight + avoidanceDir[0] * avoidWeight;
				missile->targetDirection[1] = missile->targetDirection[1] * targetWeight + avoidanceDir[1] * avoidWeight;
				missile->targetDirection[2] = missile->targetDirection[2] * targetWeight + avoidanceDir[2] * avoidWeight;

				float newMag = sqrtf(missile->targetDirection[0] * missile->targetDirection[0] +
									 missile->targetDirection[1] * missile->targetDirection[1] +
									 missile->targetDirection[2] * missile->targetDirection[2]);
				if (newMag > 0.001f) {
					missile->targetDirection[0] /= newMag;
					missile->targetDirection[1] /= newMag;
					missile->targetDirection[2] /= newMag;
				}

				if (missile->targetDirection[1] < 0.0f && groundClearance < 80.0f) {
					missile->targetDirection[1] = fmaxf(missile->targetDirection[1], 0.1f);
					float mag = sqrtf(missile->targetDirection[0] * missile->targetDirection[0] +
									  missile->targetDirection[1] * missile->targetDirection[1] +
									  missile->targetDirection[2] * missile->targetDirection[2]);
					missile->targetDirection[0] /= mag;
					missile->targetDirection[1] /= mag;
					missile->targetDirection[2] /= mag;
				}

				float avoidDistance = 600.0f;
				missile->targetPosition[0] = missile->position[0] + missile->targetDirection[0] * avoidDistance;
				missile->targetPosition[1] = missile->position[1] + missile->targetDirection[1] * avoidDistance;
				missile->targetPosition[2] = missile->position[2] + missile->targetDirection[2] * avoidDistance;
			}
		}

		if (centerThreat > 60 || minDistance < 15.0f) {
			*active = false;
			return;
		}

		// Check missile-to-missile proximity
		const float missileProximityThreshold = 20.0f;
		for (int i = 0; i < allMissiles->count; i++) {
			if (!allMissiles->active[i] || allMissiles->missiles[i] == missile) {
				continue;
			}

			struct Missile *otherMissile = allMissiles->missiles[i];
			float dx = missile->position[0] - otherMissile->position[0];
			float dy = missile->position[1] - otherMissile->position[1];
			float dz = missile->position[2] - otherMissile->position[2];
			float distSq = dx * dx + dy * dy + dz * dz;

			if (distSq < missileProximityThreshold * missileProximityThreshold) {
				*active = false;
				allMissiles->active[i] = false;
				return;
			}
		}
	}

	if (missile->seeker.lockState == Lunching) {
		if (fire) {
			missile->seeker.lockState = Searching;
			*active = true;
		}
		rayDir[0] = lunchDir[0];
		rayDir[1] = lunchDir[1];
		rayDir[2] = lunchDir[2];

	} else if (missile->seeker.lockState == Searching) {
		float fovHalfRad = (missile->seeker.seekerFov / 2.0f) * (M_PI / 180.0f);

		static float searchYaw = 0.0f;
		static float searchPitch = 0.0f;

		float searchStep = fovHalfRad * 0.25f;
		searchYaw += searchStep;

		if (searchYaw > fovHalfRad) {
			searchYaw = -fovHalfRad;
			searchPitch += searchStep;
			if (searchPitch > fovHalfRad) {
				searchPitch = -fovHalfRad;
			}
		}

		float cosPitch = cosf(searchPitch);
		float localDir[3];
		localDir[0] = sinf(searchYaw) * cosPitch;
		localDir[1] = sinf(searchPitch);
		localDir[2] = cosf(searchYaw) * cosPitch;

		// Transform local search direction to world space using body orientation
		float forward[3] = {missile->bodyOrientation[0], missile->bodyOrientation[1], missile->bodyOrientation[2]};
		float right[3], up[3];

		// Create right vector
		if (fabsf(forward[1]) < 0.99f) {
			right[0] = forward[2];
			right[1] = 0.0f;
			right[2] = -forward[0];
		} else {
			right[0] = 1.0f;
			right[1] = 0.0f;
			right[2] = 0.0f;
		}
		normalize3(right);

		// Create up vector (cross product: up = right × forward)
		up[0] = right[1] * forward[2] - right[2] * forward[1];
		up[1] = right[2] * forward[0] - right[0] * forward[2];
		up[2] = right[0] * forward[1] - right[1] * forward[0];
		normalize3(up);

		// Transform local direction to world space
		rayDir[0] = forward[0] * localDir[2] + right[0] * localDir[0] + up[0] * localDir[1];
		rayDir[1] = forward[1] * localDir[2] + right[1] * localDir[0] + up[1] * localDir[1];
		rayDir[2] = forward[2] * localDir[2] + right[2] * localDir[0] + up[2] * localDir[1];
		normalize3(rayDir);

		float objX[MAX_FIRE_SIMS], objY[MAX_FIRE_SIMS], objZ[MAX_FIRE_SIMS];
		int count = 0;

		for (int i = 0; i < allMissiles->count; i++) {
			if (i != allMissiles->count - 1 && allMissiles->active[i]) {
				objX[count] = allMissiles->missiles[i]->position[0];
				objY[count] = allMissiles->missiles[i]->position[1];
				objZ[count] = allMissiles->missiles[i]->position[2];
				count++;
			}
		}

		int bestIdx = findClosestObjectToViewCenter(
			rayOrigin, rayDir, missile->seeker.seekerFov * missile->seeker.searchMultiplayer,
			objX, objY, objZ, objX, missile->seeker.seekerDepthMap,
			count, 0.4f, 0.3f, 0.3f);

		if (bestIdx >= 0 && bestIdx < count) {
			missile->targetIdx = bestIdx;
			float toTarget[3] = {
				objX[bestIdx] - rayOrigin[0],
				objY[bestIdx] - rayOrigin[1],
				objZ[bestIdx] - rayOrigin[2]};
			float dist = sqrtf(toTarget[0] * toTarget[0] + toTarget[1] * toTarget[1] + toTarget[2] * toTarget[2]);

			if (dist > 0.1f) {
				rayDir[0] = toTarget[0] / dist;
				rayDir[1] = toTarget[1] / dist;
				rayDir[2] = toTarget[2] / dist;

				missile->targetPosition[0] = objX[bestIdx];
				missile->targetPosition[1] = objY[bestIdx];
				missile->targetPosition[2] = objZ[bestIdx];

				missile->seeker.lockState = Tracking;
			}
		}

	} else if (missile->seeker.lockState == Tracking) {
		float objX[MAX_FIRE_SIMS], objY[MAX_FIRE_SIMS], objZ[MAX_FIRE_SIMS];
		int count = 0;

		for (int i = 0; i < allMissiles->count; i++) {
			if (i != allMissiles->count - 1 && allMissiles->active[i]) {
				objX[count] = allMissiles->missiles[i]->position[0];
				objY[count] = allMissiles->missiles[i]->position[1];
				objZ[count] = allMissiles->missiles[i]->position[2];
				count++;
			}
		}

		int bestIdx = findClosestObjectToViewCenter(
			rayOrigin, rayDir, missile->seeker.seekerFov,
			objX, objY, objZ, objX, missile->seeker.seekerDepthMap,
			count, 0.5f, 0.3f, 0.2f);

		if (bestIdx >= 0 && bestIdx < count) {
			missile->targetIdx = bestIdx;
			float toTarget[3] = {
				objX[bestIdx] - rayOrigin[0],
				objY[bestIdx] - rayOrigin[1],
				objZ[bestIdx] - rayOrigin[2]};
			float dist = sqrtf(toTarget[0] * toTarget[0] + toTarget[1] * toTarget[1] + toTarget[2] * toTarget[2]);

			if (dist > 0.1f) {
				float maxTilt = missile->seeker.tiltSpeed * deltaTime;

				float targetDir[3] = {toTarget[0] / dist, toTarget[1] / dist, toTarget[2] / dist};

				float dot = rayDir[0] * targetDir[0] + rayDir[1] * targetDir[1] + rayDir[2] * targetDir[2];
				dot = fmaxf(-1.0f, fminf(1.0f, dot));

				float angle = acosf(dot);
				float t = (angle > 0.001f) ? fminf(maxTilt / angle, 1.0f) : 1.0f;

				rayDir[0] = rayDir[0] * (1.0f - t) + targetDir[0] * t;
				rayDir[1] = rayDir[1] * (1.0f - t) + targetDir[1] * t;
				rayDir[2] = rayDir[2] * (1.0f - t) + targetDir[2] * t;
				normalize3(rayDir);

				missile->targetPosition[0] = objX[bestIdx];
				missile->targetPosition[1] = objY[bestIdx];
				missile->targetPosition[2] = objZ[bestIdx];

				missile->targetDirection[0] = rayDir[0];
				missile->targetDirection[1] = rayDir[1];
				missile->targetDirection[2] = rayDir[2];

				missile->prevLOS[0] = rayDir[0];
				missile->prevLOS[1] = rayDir[1];
				missile->prevLOS[2] = rayDir[2];
			}
		} else {
			missile->seeker.lockState = Searching;
		}
	}

	missileSimStep(missile, deltaTime, timeTook, active, fireSimulationTime);
	updateSeekerPositions(allMissiles);
}

void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook, bool *active, float *fireSimulationTime) {
	if (!*active) {
		return;
	}

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
		los[0] /= losDistance;
		los[1] /= losDistance;
		los[2] /= losDistance;

		if (deltaTime > 0.001f) {
			losRate[0] = (los[0] - missile->prevLOS[0]) / deltaTime;
			losRate[1] = (los[1] - missile->prevLOS[1]) / deltaTime;
			losRate[2] = (los[2] - missile->prevLOS[2]) / deltaTime;
		}

		missile->prevLOS[0] = los[0];
		missile->prevLOS[1] = los[1];
		missile->prevLOS[2] = los[2];

		missile->targetDirection[0] = los[0];
		missile->targetDirection[1] = los[1];
		missile->targetDirection[2] = los[2];
	}

	float velDotVel = currentDir[0] * currentDir[0] + currentDir[1] * currentDir[1] + currentDir[2] * currentDir[2];
	float losDotVel = los[0] * currentDir[0] + los[1] * currentDir[1] + los[2] * currentDir[2];

	float velPerpComponent[3] = {
		los[0] - losDotVel * currentDir[0],
		los[1] - losDotVel * currentDir[1],
		los[2] - losDotVel * currentDir[2]};

	float velPerpMag = sqrtf(velPerpComponent[0] * velPerpComponent[0] +
							 velPerpComponent[1] * velPerpComponent[1] +
							 velPerpComponent[2] * velPerpComponent[2]);

	float crossProduct[3];
	if (velPerpMag > 0.001f) {
		float velPerp[3] = {
			velPerpComponent[0] / velPerpMag,
			velPerpComponent[1] / velPerpMag,
			velPerpComponent[2] / velPerpMag};

		crossProduct[0] = velPerp[1] * losRate[2] - velPerp[2] * losRate[1];
		crossProduct[1] = velPerp[2] * losRate[0] - velPerp[0] * losRate[2];
		crossProduct[2] = velPerp[0] * losRate[1] - velPerp[1] * losRate[0];
	} else {
		crossProduct[0] = 0.0f;
		crossProduct[1] = 0.0f;
		crossProduct[2] = 0.0f;
	}

	float closingSpeed = fmaxf(speed, 50.0f);
	float navGain = missile->guidanceGain;

	float commandedAccel[3] = {
		navGain * closingSpeed * crossProduct[0],
		navGain * closingSpeed * crossProduct[1],
		navGain * closingSpeed * crossProduct[2]};

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

	// Additional safety: scale down when already at high speed
	if (speed > missile->optimalSpeed) {
		float speedFactor = 1.0f - fminf(0.7f, (speed - missile->optimalSpeed) / (missile->maxSpeed - missile->optimalSpeed + 0.001f));
		commandedAccel[0] *= speedFactor;
		commandedAccel[1] *= speedFactor;
		commandedAccel[2] *= speedFactor;
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

	// Cap total acceleration magnitude to prevent runaway
	float totalAccelMag = sqrtf(totalAccel[0] * totalAccel[0] + totalAccel[1] * totalAccel[1] + totalAccel[2] * totalAccel[2]);
	float maxTotalAccel = maxAvailableG * 1.5f;
	if (totalAccelMag > maxTotalAccel) {
		float scale = maxTotalAccel / (totalAccelMag + 0.001f);
		totalAccel[0] *= scale;
		totalAccel[1] *= scale;
		totalAccel[2] *= scale;
	}

	// INTEGRATE MOTION - CORRECT
	missile->velocity[0] += totalAccel[0] * deltaTime;
	missile->velocity[1] += totalAccel[1] * deltaTime;
	missile->velocity[2] += totalAccel[2] * deltaTime;

	// Clamp velocity to maximum speed
	float currentSpeed = sqrtf(missile->velocity[0] * missile->velocity[0] +
							   missile->velocity[1] * missile->velocity[1] +
							   missile->velocity[2] * missile->velocity[2]);
	if (currentSpeed > missile->maxSpeed) {
		float speedRatio = missile->maxSpeed / (currentSpeed + 0.001f);
		missile->velocity[0] *= speedRatio;
		missile->velocity[1] *= speedRatio;
		missile->velocity[2] *= speedRatio;
	}

	missile->position[0] += missile->velocity[0] * SCALE * deltaTime;
	missile->position[1] += missile->velocity[1] * SCALE * deltaTime;
	missile->position[2] += missile->velocity[2] * SCALE * deltaTime;

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
			missile->fireSim->windDirection[0] = -missile->velocity[0] * 16.0f;
			missile->fireSim->windDirection[1] = -missile->velocity[1] * 16.0f;
			missile->fireSim->windDirection[2] = -missile->velocity[2] * 16.0f;
		}

		fireSimStep(missile->fireSim, deltaTime, fireSimulationTime);
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	*timeTook = (float)((end.tv_sec - start.tv_sec) * 1000.0 +
						(end.tv_nsec - start.tv_nsec) / 1e6);
}

void InitializeFireParticles(struct FireSOA *particles) {
	particles->buoyancy = 14.0f;
	particles->drag = 0.985f;
	particles->turbulence = 2.5f;
	particles->maxLifeTime = 0.15f;

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

		missiles->active[i] = false;
	}
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
