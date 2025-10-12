#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
#define NUM_FIRE_PARTICLES 750
#define G 9.81f

float randRange(float min, float max) {
	float scale = rand() / (float)RAND_MAX;
	return min + scale * (max - min);
}

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

void InitializeFireParticles(struct FireSOA *particles) {
	srand((unsigned int)time(NULL));
	particles->buoyancy = 4000.0f;
	particles->drag = 0.985f;
	particles->turbulence = 2.5f;
	particles->maxLifeTime = 0.8f;

	particles->startingColor[0] = 1.0f;
	particles->startingColor[1] = 0.3f;
	particles->startingColor[2] = 0.2f; // Orange
	particles->fireColor[0] = 0.85f;
	particles->fireColor[1] = 0.0f;
	particles->fireColor[2] = 0.1f; // Red
	particles->smokeColor[0] = 0.1f / 3.0f;
	particles->smokeColor[1] = 0.15f / 3.0f;
	particles->smokeColor[2] = 0.25f / 3.0f; // Dark gray

	particles->basePosition[0] = 100.0f;
	particles->basePosition[1] = 0.0f;
	particles->basePosition[2] = -350.0f;

	particles->maxVelocity = 0.0f;
	particles->maxDistance = 0.0f;
	particles->particlesSize = 18.0f;
	particles->windDirection[0] = 0.55f;
	particles->windDirection[1] = 0.0f;
	particles->windDirection[2] = 0.22f;
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
