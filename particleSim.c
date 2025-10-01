#include <stdio.h>
#include <math.h>
#include <string.h>
// #include <time.h>
// #include <stdlib.h> // Added for malloc and free
#define DEBUG 1
#define NUM_PARTICLES 15000
#define GRAVITY 500.0f
#define DAMPING 0.985f
#define gridResolutionAxis 32
#define gridResolution (gridResolutionAxis * gridResolutionAxis * gridResolutionAxis)
#define temperature 10.0f
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
	int gridParticleID[gridResolution][NUM_PARTICLES / 8]; // Assuming a max of NUM_PARTICLES/4 per cell
	int gridParticleCount[gridResolution];
	float gridParticleAvgPos[gridResolution][3];
	float gridAverageVelocity[gridResolution][3];
	float gridCellGradientPressure[gridResolution][3];
	float gridCellBlurredGradientPressure[gridResolution][3];
#ifdef DEBUG
	float totalEnergy; // Total kinetic energy of the system
#endif
};

void updateGridData(struct PointSOA *particles) {
	// Reset BOTH grid counts AND average positions
	memset(particles->gridParticleCount, 0, sizeof(particles->gridParticleCount));
	memset(particles->gridParticleAvgPos, 0, sizeof(particles->gridParticleAvgPos));
	memset(particles->gridCellGradientPressure, 0, sizeof(particles->gridCellGradientPressure));

	// Calculate grid steps once
	float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
	float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
	float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;

	// Check for degenerate bounding box
	if (xStep <= 0.0f || yStep <= 0.0f || zStep <= 0.0f) {
		printf("Warning: Invalid bounding box dimensions\n");
		return;
	}

	// Precompute inverse steps
	float xStepInv = 1.0f / xStep;
	float yStepInv = 1.0f / yStep;
	float zStepInv = 1.0f / zStep;

	for (int i = 0; i < NUM_PARTICLES; i++) {
		// Calculate grid indices
		float xIndexFloat = (particles->x[i] - particles->bBoxMin[0]) * xStepInv;
		float yIndexFloat = (particles->y[i] - particles->bBoxMin[1]) * yStepInv;
		float zIndexFloat = (particles->z[i] - particles->bBoxMin[2]) * zStepInv;

		int xIndex = (int)xIndexFloat;
		int yIndex = (int)yIndexFloat;
		int zIndex = (int)zIndexFloat;

		// Clamp indices
		if (xIndex < 0)
			xIndex = 0;
		else if (xIndex >= gridResolutionAxis)
			xIndex = gridResolutionAxis - 1;
		if (yIndex < 0)
			yIndex = 0;
		else if (yIndex >= gridResolutionAxis)
			yIndex = gridResolutionAxis - 1;
		if (zIndex < 0)
			zIndex = 0;
		else if (zIndex >= gridResolutionAxis)
			zIndex = gridResolutionAxis - 1;

		int gridID = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
		int cellParticleCount = particles->gridParticleCount[gridID];

		// FIXED: Match actual array size
		if (cellParticleCount >= NUM_PARTICLES / 8) {
			continue;
		}

		particles->gridParticleID[gridID][cellParticleCount] = i;
		particles->gridParticleCount[gridID]++;

		// Accumulate positions
		particles->gridParticleAvgPos[gridID][0] += particles->x[i];
		particles->gridParticleAvgPos[gridID][1] += particles->y[i];
		particles->gridParticleAvgPos[gridID][2] += particles->z[i];
		// Accumulate velocities
		particles->gridAverageVelocity[gridID][0] += particles->xVelocity[i];
		particles->gridAverageVelocity[gridID][1] += particles->yVelocity[i];
		particles->gridAverageVelocity[gridID][2] += particles->zVelocity[i];
	}

	// Finalize average positions
	for (int gridID = 0; gridID < gridResolution; gridID++) {
		int count = particles->gridParticleCount[gridID];
		if (count > 0) {
			particles->gridParticleAvgPos[gridID][0] /= count;
			particles->gridParticleAvgPos[gridID][1] /= count;
			particles->gridParticleAvgPos[gridID][2] /= count;
			particles->gridAverageVelocity[gridID][0] /= count;
			particles->gridAverageVelocity[gridID][1] /= count;
			particles->gridAverageVelocity[gridID][2] /= count;
		}
	}
}

void ApplySurfaceTension(struct PointSOA *particles, float tensionCoeff, float deltaTime) {
	for (int cell = 0; cell < gridResolution; cell++) {
		int cnt = particles->gridParticleCount[cell];
		if (cnt < 2) continue;
		// normalized density: 0.0=empty, 1.0=max density
		float density = cnt / (float)((float)NUM_PARTICLES / 8);
		if (density >= 0.85f) continue; // interior
		float surfW = (1.0f - density); // strong at surface
		float cx = particles->gridParticleAvgPos[cell][0];
		float cy = particles->gridParticleAvgPos[cell][1];
		float cz = particles->gridParticleAvgPos[cell][2];
		float pull = tensionCoeff * surfW * deltaTime;
		for (int i = 0; i < cnt; i++) {
			int idx = particles->gridParticleID[cell][i];
			// pull each surface particle toward the cell center
			particles->xVelocity[idx] += pull * (cx - particles->x[idx]);
			particles->yVelocity[idx] += pull * (cy - particles->y[idx]);
			particles->zVelocity[idx] += pull * (cz - particles->z[idx]);
		}
	}
}

void ApplyViscosity(struct PointSOA *particles, float viscosityCoeff, float deltaTime) {
	for (int gridCellId = 0; gridCellId < gridResolution; gridCellId++) {
		int cellParticleCount = particles->gridParticleCount[gridCellId];
		if (cellParticleCount < 2) continue;

		float avgVelX = particles->gridAverageVelocity[gridCellId][0];
		float avgVelY = particles->gridAverageVelocity[gridCellId][1];
		float avgVelZ = particles->gridAverageVelocity[gridCellId][2];

		float densityFactor = (float)cellParticleCount / ((float)NUM_PARTICLES / (float)gridResolution);
		densityFactor = fminf(densityFactor, 1.0f);
		float viscosity = viscosityCoeff * densityFactor * deltaTime;
		viscosity = fminf(viscosity, 0.1f); // Cap maximum viscosity to prevent instability

		for (int p = 0; p < cellParticleCount; p++) {
			int particleIndex = particles->gridParticleID[gridCellId][p];
			particles->xVelocity[particleIndex] += (avgVelX - particles->xVelocity[particleIndex]) * viscosity;
			particles->yVelocity[particleIndex] += (avgVelY - particles->yVelocity[particleIndex]) * viscosity;
			particles->zVelocity[particleIndex] += (avgVelZ - particles->zVelocity[particleIndex]) * viscosity;
		}
	}
}

void CalculatePressureGradient(struct PointSOA *particles) {
	const int neighbors[6][3] = {
		{-1, 0, 0}, {1, 0, 0}, // left, right
		{0, -1, 0},
		{0, 1, 0}, // down, up
		{0, 0, -1},
		{0, 0, 1} // back, front
	};

	// First pass: Calculate pressure gradients
	for (int gridCellId = 0; gridCellId < gridResolution; gridCellId++) {
		int cellParticleCount = particles->gridParticleCount[gridCellId];
		if (cellParticleCount == 0) {
			// Clear gradient for empty cells
			particles->gridCellGradientPressure[gridCellId][0] = 0.0f;
			particles->gridCellGradientPressure[gridCellId][1] = 0.0f;
			particles->gridCellGradientPressure[gridCellId][2] = 0.0f;
			continue;
		}

		int zIndex = gridCellId / (gridResolutionAxis * gridResolutionAxis);
		int yIndex = (gridCellId / gridResolutionAxis) % gridResolutionAxis;
		int xIndex = gridCellId % gridResolutionAxis;

		float currentCellPressure = cellParticleCount * pressure;
		float currentCellAvgPosX = particles->gridParticleAvgPos[gridCellId][0];
		float currentCellAvgPosY = particles->gridParticleAvgPos[gridCellId][1];
		float currentCellAvgPosZ = particles->gridParticleAvgPos[gridCellId][2];
		float gradient[3] = {0.0f, 0.0f, 0.0f};

		for (int n = 0; n < 6; n++) {
			int neighborX = xIndex + neighbors[n][0];
			int neighborY = yIndex + neighbors[n][1];
			int neighborZ = zIndex + neighbors[n][2];

			// Check bounds
			if (neighborX < 0 || neighborX >= gridResolutionAxis ||
				neighborY < 0 || neighborY >= gridResolutionAxis ||
				neighborZ < 0 || neighborZ >= gridResolutionAxis) {
				continue;
			}

			int neighborGridID = neighborX + neighborY * gridResolutionAxis + neighborZ * gridResolutionAxis * gridResolutionAxis;
			int neighborParticleCount = particles->gridParticleCount[neighborGridID];

			// Use zero pressure for empty cells instead of skipping
			float neighborCellPressure = neighborParticleCount * pressure;
			float pressureDiff = neighborCellPressure - currentCellPressure;

			// float dx = particles->gridParticleAvgPos[neighborGridID][0] - currentCellAvgPosX;
			// float dy = particles->gridParticleAvgPos[neighborGridID][1] - currentCellAvgPosY;
			// float dz = particles->gridParticleAvgPos[neighborGridID][2] - currentCellAvgPosZ;
			float dx = (neighborX - xIndex) * ((particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis);
			float dy = (neighborY - yIndex) * ((particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis);
			float dz = (neighborZ - zIndex) * ((particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis);
			gradient[0] += pressureDiff * dx;
			gradient[1] += pressureDiff * dy;
			gradient[2] += pressureDiff * dz;
		}

		// Store the computed gradient
		particles->gridCellGradientPressure[gridCellId][0] = gradient[0];
		particles->gridCellGradientPressure[gridCellId][1] = gradient[1];
		particles->gridCellGradientPressure[gridCellId][2] = gradient[2];
	}

	// Second pass: Apply Gaussian blur to the pressure gradients
	const float kernel[3] = {0.25f, 0.5f, 0.25f}; // Fixed: removed extra 'f'

	for (int gridCellId = 0; gridCellId < gridResolution; gridCellId++) {
		int zIndex = gridCellId / (gridResolutionAxis * gridResolutionAxis);
		int yIndex = (gridCellId / gridResolutionAxis) % gridResolutionAxis;
		int xIndex = gridCellId % gridResolutionAxis;

		float blurredGradient[3] = {0.0f, 0.0f, 0.0f};

		for (int dz = -1; dz <= 1; dz++) {
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int neighborX = xIndex + dx;
					int neighborY = yIndex + dy;
					int neighborZ = zIndex + dz;

					// Check bounds
					if (neighborX < 0 || neighborX >= gridResolutionAxis ||
						neighborY < 0 || neighborY >= gridResolutionAxis ||
						neighborZ < 0 || neighborZ >= gridResolutionAxis) {
						continue;
					}

					int neighborGridID = neighborX + neighborY * gridResolutionAxis + neighborZ * gridResolutionAxis * gridResolutionAxis;
					float weight = kernel[dx + 1] * kernel[dy + 1] * kernel[dz + 1];

					blurredGradient[0] += particles->gridCellGradientPressure[neighborGridID][0] * weight;
					blurredGradient[1] += particles->gridCellGradientPressure[neighborGridID][1] * weight;
					blurredGradient[2] += particles->gridCellGradientPressure[neighborGridID][2] * weight;
				}
			}
		}
		// Store the blurred gradient
		particles->gridCellBlurredGradientPressure[gridCellId][0] = blurredGradient[0];
		particles->gridCellBlurredGradientPressure[gridCellId][1] = blurredGradient[1];
		particles->gridCellBlurredGradientPressure[gridCellId][2] = blurredGradient[2];
	}
}

void UpdateParticles(struct PointSOA *particles, float deltaTime) {
	for (int i = 0; i < NUM_PARTICLES; i++) {
		// Apply gravity
		particles->yVelocity[i] -= GRAVITY * deltaTime;

		// Update positions
		particles->x[i] += particles->xVelocity[i] * deltaTime;
		particles->y[i] += particles->yVelocity[i] * deltaTime;
		particles->z[i] += particles->zVelocity[i] * deltaTime;

		// Boundary constraints
		if (particles->x[i] < particles->bBoxMin[0]) {
			particles->x[i] = particles->bBoxMin[0];
			particles->xVelocity[i] *= -DAMPING; // Bounce with energy loss
		}
		if (particles->x[i] > particles->bBoxMax[0]) {
			particles->x[i] = particles->bBoxMax[0];
			particles->xVelocity[i] *= -DAMPING;
		}
		if (particles->y[i] < particles->bBoxMin[1]) {
			particles->y[i] = particles->bBoxMin[1];
			particles->yVelocity[i] *= -DAMPING;
		}
		if (particles->y[i] > particles->bBoxMax[1]) {
			particles->y[i] = particles->bBoxMax[1];
			particles->yVelocity[i] *= -DAMPING;
		}
		if (particles->z[i] < particles->bBoxMin[2]) {
			particles->z[i] = particles->bBoxMin[2];
			particles->zVelocity[i] *= -DAMPING;
		}
		if (particles->z[i] > particles->bBoxMax[2]) {
			particles->z[i] = particles->bBoxMax[2];
			particles->zVelocity[i] *= -DAMPING;
		}

		// Apply damping
		particles->xVelocity[i] *= DAMPING;
		particles->yVelocity[i] *= DAMPING;
		particles->zVelocity[i] *= DAMPING;

		// Update total velocity
		particles->totalVelocity[i] = sqrtf(particles->xVelocity[i] * particles->xVelocity[i] +
											particles->yVelocity[i] * particles->yVelocity[i] +
											particles->zVelocity[i] * particles->zVelocity[i]);
	}
}

void ApplyPressure(struct PointSOA *particles, float deltaTime) {
	// Calculate pressure gradients first
	CalculatePressureGradient(particles);

	for (int gridCellId = 0; gridCellId < gridResolution; gridCellId++) {
		// iterate through all particles in this cell
		int cellParticleCount = particles->gridParticleCount[gridCellId];
		if (cellParticleCount == 0) continue; // Skip empty cells
		float gradientX = particles->gridCellBlurredGradientPressure[gridCellId][0];
		float gradientY = particles->gridCellBlurredGradientPressure[gridCellId][1];
		float gradientZ = particles->gridCellBlurredGradientPressure[gridCellId][2];
		for (int p = 0; p < cellParticleCount; p++) {
			int particleIndex = particles->gridParticleID[gridCellId][p];
			// Apply pressure force to particle velocity
			// particles->xVelocity[particleIndex] -= gradientX * 0.01f; // Scaled down for stability
			// particles->yVelocity[particleIndex] -= gradientY * 0.01f;
			// particles->zVelocity[particleIndex] -= gradientZ * 0.01f;

			float disToCenterX = particles->x[particleIndex] - particles->gridParticleAvgPos[gridCellId][0];
			float disToCenterY = particles->y[particleIndex] - particles->gridParticleAvgPos[gridCellId][1];
			float disToCenterZ = particles->z[particleIndex] - particles->gridParticleAvgPos[gridCellId][2];

			// Calculate squared distance from center
			float distSq = disToCenterX * disToCenterX + disToCenterY * disToCenterY + disToCenterZ * disToCenterZ;

			// Inverse squared distance weighting: closer particles feel MORE pressure
			float pressureWeight = 1.0f / (distSq + 1.0f);

			// Apply gradient force weighted by proximity to center
			particles->xVelocity[particleIndex] -= gradientX * pressureWeight * 0.01f * deltaTime;
			particles->yVelocity[particleIndex] -= gradientY * pressureWeight * 0.01f * deltaTime;
			particles->zVelocity[particleIndex] -= gradientZ * pressureWeight * 0.01f * deltaTime;
		}
	}
}

void CollideParticlesInGrid(struct PointSOA *particles) {
	for (int gridCellId = 0; gridCellId < gridResolution; gridCellId++) {
		int cellParticleCount = particles->gridParticleCount[gridCellId];
		if (cellParticleCount < 2) continue; // No collisions possible

		for (int i = 0; i < cellParticleCount; i++) {
			int indexA = particles->gridParticleID[gridCellId][i];
			for (int j = i + 1; j < cellParticleCount; j++) {
				int indexB = particles->gridParticleID[gridCellId][j];

				float dx = particles->x[indexB] - particles->x[indexA];
				float dy = particles->y[indexB] - particles->y[indexA];
				float dz = particles->z[indexB] - particles->z[indexA];
				float distSq = dx * dx + dy * dy + dz * dz;

				float minDist = 2.0f * PARTICLE_RADIUS;
				float minDistSq = minDist * minDist;

				if (distSq < minDistSq && distSq > 0.0f) {
					float dist = sqrtf(distSq);
					float overlap = 0.5f * (minDist - dist);

					// Normalize the collision vector
					float nx = dx / dist;
					float ny = dy / dist;
					float nz = dz / dist;

					// Displace particles to resolve overlap
					particles->x[indexA] -= overlap * nx;
					particles->y[indexA] -= overlap * ny;
					particles->z[indexA] -= overlap * nz;

					particles->x[indexB] += overlap * nx;
					particles->y[indexB] += overlap * ny;
					particles->z[indexB] += overlap * nz;

					// Compute relative velocity
					float rvx = particles->xVelocity[indexB] - particles->xVelocity[indexA];
					float rvy = particles->yVelocity[indexB] - particles->yVelocity[indexA];
					float rvz = particles->zVelocity[indexB] - particles->zVelocity[indexA];

					// Compute relative velocity along the normal
					float velAlongNormal = rvx * nx + rvy * ny + rvz * nz;

					if (velAlongNormal > 0) continue; // Particles are moving apart

					// FIXED: Compute impulse scalar (was incomplete line)
					float restitution = 0.5f; // Coefficient of restitution
					float impulse = -(1.0f + restitution) * velAlongNormal;
					impulse /= 2.0f; // Assuming equal mass

					// Apply impulse to particle velocities
					particles->xVelocity[indexA] -= impulse * nx * DAMPING;
					particles->yVelocity[indexA] -= impulse * ny * DAMPING;
					particles->zVelocity[indexA] -= impulse * nz * DAMPING;
					particles->xVelocity[indexB] += impulse * nx * DAMPING;
					particles->yVelocity[indexB] += impulse * ny * DAMPING;
					particles->zVelocity[indexB] += impulse * nz * DAMPING;
				}
			}
		}
	}
}

void Step(struct PointSOA *particles, float deltaTime) {
	const int subSteps = 4;
	float subDeltaTime = deltaTime / subSteps;

	for (int step = 0; step < subSteps; step++) {
		updateGridData(particles);
		ApplyPressure(particles, subDeltaTime);
		CollideParticlesInGrid(particles);
		ApplyViscosity(particles, 0.01f, subDeltaTime);
		ApplySurfaceTension(particles, 500.0f, subDeltaTime);
		UpdateParticles(particles, subDeltaTime);
	}

#ifdef DEBUG
	// Print Total Energy in the system for debugging
	float totalEnergy = 0.0f;
	for (int i = 0; i < NUM_PARTICLES; i++) {
		totalEnergy += (particles->xVelocity[i] * particles->xVelocity[i] +
						particles->yVelocity[i] * particles->yVelocity[i] +
						particles->zVelocity[i] * particles->zVelocity[i]);
	}
	particles->totalEnergy = totalEnergy; // Kinetic energy
#endif
}

// Example usage and performance measurement
// clang -O3 particleSim.c -o sim -lm -march=native ; ./sim
// int main() {
// 	struct PointSOA *particles = malloc(sizeof(struct PointSOA));
// 	if (particles == NULL) {
// 		printf("Failed to allocate memory for particles.\n");
// 		return 1;
// 	}

// 	// Initialize bounding box
// 	particles->bBoxMin[0] = 0.0f;
// 	particles->bBoxMin[1] = 0.0f;
// 	particles->bBoxMin[2] = 0.0f;
// 	particles->bBoxMax[0] = 100.0f;
// 	particles->bBoxMax[1] = 100.0f;
// 	particles->bBoxMax[2] = 100.0f;

// 	// Initialize particles in a grid
// 	int particlesPerAxis = (int)cbrtf(NUM_PARTICLES);
// 	float spacing = (particles->bBoxMax[0] - particles->bBoxMin[0]) / particlesPerAxis;
// 	int index = 0;
// 	for (int x = 0; x < particlesPerAxis; x++) {
// 		for (int y = 0; y < particlesPerAxis; y++) {
// 			for (int z = 0; z < particlesPerAxis; z++) {
// 				if (index >= NUM_PARTICLES) break;
// 				particles->x[index] = particles->bBoxMin[0] + x * spacing + spacing * 0.5f;
// 				particles->y[index] = particles->bBoxMin[1] + y * spacing + spacing * 0.5f;
// 				particles->z[index] = particles->bBoxMin[2] + z * spacing + spacing * 0.5f;
// 				particles->xVelocity[index] = 0.0f;
// 				particles->yVelocity[index] = 0.0f;
// 				particles->zVelocity[index] = 0.0f;
// 				index++;
// 			}
// 		}
// 	}

// 	float deltaTime = 0.016f; // ~60 FPS

// 	struct timespec start, end;
// 	clock_gettime(CLOCK_MONOTONIC, &start);
// 	for (int frame = 0; frame < 100; frame++) {
// 		Step(particles, deltaTime);
// 	}
// 	clock_gettime(CLOCK_MONOTONIC, &end);
// 	double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
// 	printf("Simulated 100 frames in %.3f seconds\n", elapsed);
// 	printf("Average time per frame: %.3f ms\n", (elapsed / 100.0) * 1000.0);
// 	printf("Average TPS: %.2f\n", 100.0 / elapsed);
// 	printf("Number of particles: %d\n", NUM_PARTICLES);
// 	printf("Average particle simulation time: %.3f microseconds\n", (elapsed / (100.0 * NUM_PARTICLES)) * 1e6);

// 	free(particles); // Free the allocated memory
// 	return 0;
// }