#define Capacity 1024
#define BUFFER_PERCENTAGE 0.2f
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../openGlShaders/gpuStruct.h"
#include "../mapGeneration/loadMap.h"

struct Volume {
	float BBoxMin[3];
	float BBoxMax[3];
	int count;
	float v1[Capacity * 3];
	float v2[Capacity * 3];
	float v3[Capacity * 3];
	float Roughness[Capacity];
	float Metallic[Capacity];
	float Emission[Capacity];
	float normals[Capacity * 3];
	float colors[Capacity * 3]; // RGB colors for each triangle
};

struct Cluster {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Volume volumes[8]; // 8 volumes per cluster forming a 3D grid
};

struct Block {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Cluster clusters[8];
};

struct Region {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Block blocks[8];
};

static void initializeBoundingBoxes(struct Region *region) {
	float inf = 1e30f;
	region->BBoxMin[0] = region->BBoxMin[1] = region->BBoxMin[2] = inf;
	region->BBoxMax[0] = region->BBoxMax[1] = region->BBoxMax[2] = -inf;

	for (int b = 0; b < 8; b++) {
		region->blocks[b].BBoxMin[0] = region->blocks[b].BBoxMin[1] = region->blocks[b].BBoxMin[2] = inf;
		region->blocks[b].BBoxMax[0] = region->blocks[b].BBoxMax[1] = region->blocks[b].BBoxMax[2] = -inf;

		for (int c = 0; c < 8; c++) {
			region->blocks[b].clusters[c].BBoxMin[0] = region->blocks[b].clusters[c].BBoxMin[1] = region->blocks[b].clusters[c].BBoxMin[2] = inf;
			region->blocks[b].clusters[c].BBoxMax[0] = region->blocks[b].clusters[c].BBoxMax[1] = region->blocks[b].clusters[c].BBoxMax[2] = -inf;

			for (int v = 0; v < 8; v++) {
				region->blocks[b].clusters[c].volumes[v].BBoxMin[0] = region->blocks[b].clusters[c].volumes[v].BBoxMin[1] = region->blocks[b].clusters[c].volumes[v].BBoxMin[2] = inf;
				region->blocks[b].clusters[c].volumes[v].BBoxMax[0] = region->blocks[b].clusters[c].volumes[v].BBoxMax[1] = region->blocks[b].clusters[c].volumes[v].BBoxMax[2] = -inf;
				region->blocks[b].clusters[c].volumes[v].count = 0;
			}
		}
	}
}

static void computeOctantIndex(float point[3], float center[3], int *idx) {
	*idx = ((point[0] >= center[0]) ? 1 : 0) |
		   ((point[1] >= center[1]) ? 2 : 0) |
		   ((point[2] >= center[2]) ? 4 : 0);
}

static void computeTriangleCenter(float v1[3], float v2[3], float v3[3], float center[3]) {
	center[0] = (v1[0] + v2[0] + v3[0]) / 3.0f;
	center[1] = (v1[1] + v2[1] + v3[1]) / 3.0f;
	center[2] = (v1[2] + v2[2] + v3[2]) / 3.0f;
}

static void clampPoint(float point[3], float minBounds[3], float maxBounds[3]) {
	for (int i = 0; i < 3; i++) {
		if (point[i] < minBounds[i]) point[i] = minBounds[i];
		if (point[i] > maxBounds[i]) point[i] = maxBounds[i];
	}
}

static void addTriangleToVolume(struct Volume *volume, float v1[3], float v2[3], float v3[3],
								float roughness, float metallic, float emission,
								float normal[3], float color[3]) {
	if (volume->count >= Capacity) return;

	int idx = volume->count;
	volume->v1[idx * 3 + 0] = v1[0];
	volume->v1[idx * 3 + 1] = v1[1];
	volume->v1[idx * 3 + 2] = v1[2];

	volume->v2[idx * 3 + 0] = v2[0];
	volume->v2[idx * 3 + 1] = v2[1];
	volume->v2[idx * 3 + 2] = v2[2];

	volume->v3[idx * 3 + 0] = v3[0];
	volume->v3[idx * 3 + 1] = v3[1];
	volume->v3[idx * 3 + 2] = v3[2];

	volume->Roughness[idx] = roughness;
	volume->Metallic[idx] = metallic;
	volume->Emission[idx] = emission;

	volume->normals[idx * 3 + 0] = normal[0];
	volume->normals[idx * 3 + 1] = normal[1];
	volume->normals[idx * 3 + 2] = normal[2];

	volume->colors[idx * 3 + 0] = color[0];
	volume->colors[idx * 3 + 1] = color[1];
	volume->colors[idx * 3 + 2] = color[2];

	updateBBox(v1[0], v1[1], v1[2], volume->BBoxMin, volume->BBoxMax);
	updateBBox(v2[0], v2[1], v2[2], volume->BBoxMin, volume->BBoxMax);
	updateBBox(v3[0], v3[1], v3[2], volume->BBoxMin, volume->BBoxMax);

	volume->count++;
}

enum Mode {
	APPEND_MODE,
	INIT_MODE,
};

void loadTriangles(struct Region *staticRegion, struct Triangles *sceneTriangles, enum Mode mode) {
	if (!staticRegion || !sceneTriangles) {
		return;
	}

	if (mode == INIT_MODE) {
		initializeBoundingBoxes(staticRegion);
		for (int b = 0; b < 8; b++) {
			for (int c = 0; c < 8; c++) {
				for (int v = 0; v < 8; v++) {
					staticRegion->blocks[b].clusters[c].volumes[v].count = 0;
				}
			}
		}
	}

	float regionMin[3] = {1e30f, 1e30f, 1e30f};
	float regionMax[3] = {-1e30f, -1e30f, -1e30f};

	// In APPEND_MODE, use existing region bounds instead of recalculating
	if (mode == APPEND_MODE) {
		regionMin[0] = staticRegion->BBoxMin[0];
		regionMin[1] = staticRegion->BBoxMin[1];
		regionMin[2] = staticRegion->BBoxMin[2];
		regionMax[0] = staticRegion->BBoxMax[0];
		regionMax[1] = staticRegion->BBoxMax[1];
		regionMax[2] = staticRegion->BBoxMax[2];
	} else {
		// In INIT_MODE, calculate bounds from triangles
		for (int i = 0; i < sceneTriangles->count && i < NUMBER_OF_TRIANGLES; i++) {
			float v1[3] = {sceneTriangles->v1[i * 3], sceneTriangles->v1[i * 3 + 1], sceneTriangles->v1[i * 3 + 2]};
			float v2[3] = {sceneTriangles->v2[i * 3], sceneTriangles->v2[i * 3 + 1], sceneTriangles->v2[i * 3 + 2]};
			float v3[3] = {sceneTriangles->v3[i * 3], sceneTriangles->v3[i * 3 + 1], sceneTriangles->v3[i * 3 + 2]};

			updateBBox(v1[0], v1[1], v1[2], regionMin, regionMax);
			updateBBox(v2[0], v2[1], v2[2], regionMin, regionMax);
			updateBBox(v3[0], v3[1], v3[2], regionMin, regionMax);
		}

		// Expand bounding box by buffer percentage in all 6 directions
		float sizeX = regionMax[0] - regionMin[0];
		float sizeY = regionMax[1] - regionMin[1];
		float sizeZ = regionMax[2] - regionMin[2];

		float bufferX = sizeX * BUFFER_PERCENTAGE;
		float bufferY = sizeY * BUFFER_PERCENTAGE;
		float bufferZ = sizeZ * BUFFER_PERCENTAGE;

		regionMin[0] -= bufferX;
		regionMin[1] -= bufferY;
		regionMin[2] -= bufferZ;
		regionMax[0] += bufferX;
		regionMax[1] += bufferY;
		regionMax[2] += bufferZ;
	}

	staticRegion->BBoxMin[0] = regionMin[0];
	staticRegion->BBoxMin[1] = regionMin[1];
	staticRegion->BBoxMin[2] = regionMin[2];
	staticRegion->BBoxMax[0] = regionMax[0];
	staticRegion->BBoxMax[1] = regionMax[1];
	staticRegion->BBoxMax[2] = regionMax[2];

	float regionCenter[3] = {
		(regionMin[0] + regionMax[0]) * 0.5f,
		(regionMin[1] + regionMax[1]) * 0.5f,
		(regionMin[2] + regionMax[2]) * 0.5f};

	for (int i = 0; i < sceneTriangles->count && i < NUMBER_OF_TRIANGLES; i++) {
		float v1[3] = {sceneTriangles->v1[i * 3], sceneTriangles->v1[i * 3 + 1], sceneTriangles->v1[i * 3 + 2]};
		float v2[3] = {sceneTriangles->v2[i * 3], sceneTriangles->v2[i * 3 + 1], sceneTriangles->v2[i * 3 + 2]};
		float v3[3] = {sceneTriangles->v3[i * 3], sceneTriangles->v3[i * 3 + 1], sceneTriangles->v3[i * 3 + 2]};
		float normal[3] = {sceneTriangles->normals[i * 3], sceneTriangles->normals[i * 3 + 1], sceneTriangles->normals[i * 3 + 2]};
		float color[3] = {sceneTriangles->colors[i * 3], sceneTriangles->colors[i * 3 + 1], sceneTriangles->colors[i * 3 + 2]};

		float triCenter[3];
		computeTriangleCenter(v1, v2, v3, triCenter);

		// In APPEND_MODE, clamp triangle center to existing bounds
		if (mode == APPEND_MODE) {
			clampPoint(triCenter, regionMin, regionMax);
		}

		int blockIdx;
		computeOctantIndex(triCenter, regionCenter, &blockIdx);
		struct Block *block = &staticRegion->blocks[blockIdx];

		float blockMin[3] = {
			(blockIdx & 1) ? regionCenter[0] : regionMin[0],
			(blockIdx & 2) ? regionCenter[1] : regionMin[1],
			(blockIdx & 4) ? regionCenter[2] : regionMin[2]};
		float blockMax[3] = {
			(blockIdx & 1) ? regionMax[0] : regionCenter[0],
			(blockIdx & 2) ? regionMax[1] : regionCenter[1],
			(blockIdx & 4) ? regionMax[2] : regionCenter[2]};
		float blockCenter[3] = {
			(blockMin[0] + blockMax[0]) * 0.5f,
			(blockMin[1] + blockMax[1]) * 0.5f,
			(blockMin[2] + blockMax[2]) * 0.5f};

		int clusterIdx;
		computeOctantIndex(triCenter, blockCenter, &clusterIdx);
		struct Cluster *cluster = &block->clusters[clusterIdx];

		float clusterMin[3] = {
			(clusterIdx & 1) ? blockCenter[0] : blockMin[0],
			(clusterIdx & 2) ? blockCenter[1] : blockMin[1],
			(clusterIdx & 4) ? blockCenter[2] : blockMin[2]};
		float clusterMax[3] = {
			(clusterIdx & 1) ? blockMax[0] : blockCenter[0],
			(clusterIdx & 2) ? blockMax[1] : blockCenter[1],
			(clusterIdx & 4) ? blockMax[2] : blockCenter[2]};
		float clusterCenter[3] = {
			(clusterMin[0] + clusterMax[0]) * 0.5f,
			(clusterMin[1] + clusterMax[1]) * 0.5f,
			(clusterMin[2] + clusterMax[2]) * 0.5f};

		int volumeIdx;
		computeOctantIndex(triCenter, clusterCenter, &volumeIdx);
		struct Volume *volume = &cluster->volumes[volumeIdx];

		addTriangleToVolume(volume, v1, v2, v3,
							sceneTriangles->Roughness[i],
							sceneTriangles->Metallic[i],
							sceneTriangles->Emission[i],
							normal, color);

		updateBBox(v1[0], v1[1], v1[2], cluster->BBoxMin, cluster->BBoxMax);
		updateBBox(v2[0], v2[1], v2[2], cluster->BBoxMin, cluster->BBoxMax);
		updateBBox(v3[0], v3[1], v3[2], cluster->BBoxMin, cluster->BBoxMax);

		updateBBox(v1[0], v1[1], v1[2], block->BBoxMin, block->BBoxMax);
		updateBBox(v2[0], v2[1], v2[2], block->BBoxMin, block->BBoxMax);
		updateBBox(v3[0], v3[1], v3[2], block->BBoxMin, block->BBoxMax);
	}
}

int randomInt(int min, int max) {
	return min + rand() % (max - min + 1);
}

int main() {
	int iterations = 1000;
	float timeStemps[iterations];

	// test performance of loading data
	struct Region *staticRegion = malloc(sizeof(struct Region));
	struct Triangles *sceneTriangles = malloc(sizeof(struct Triangles));
	sceneTriangles->count = 0;

	// Fill sceneTriangles with test data
	for (int iter = 0; iter < iterations; iter++) {
		sceneTriangles->count = 0;
		int numTriangles = randomInt(NUMBER_OF_TRIANGLES / 2, NUMBER_OF_TRIANGLES);
		for (int i = 0; i < numTriangles; i++) {
			sceneTriangles->v1[i * 3 + 0] = (float)randomInt(-1000, 1000);
			sceneTriangles->v1[i * 3 + 1] = (float)randomInt(-1000, 1000);
			sceneTriangles->v1[i * 3 + 2] = (float)randomInt(-1000, 1000);

			sceneTriangles->v2[i * 3 + 0] = (float)randomInt(-1000, 1000);
			sceneTriangles->v2[i * 3 + 1] = (float)randomInt(-1000, 1000);
			sceneTriangles->v2[i * 3 + 2] = (float)randomInt(-1000, 1000);

			sceneTriangles->v3[i * 3 + 0] = (float)randomInt(-1000, 1000);
			sceneTriangles->v3[i * 3 + 1] = (float)randomInt(-1000, 1000);
			sceneTriangles->v3[i * 3 + 2] = (float)randomInt(-1000, 1000);

			sceneTriangles->normals[i * 3 + 0] = (float)randomInt(-1, 1);
			sceneTriangles->normals[i * 3 + 1] = (float)randomInt(-1, 1);
			sceneTriangles->normals[i * 3 + 2] = (float)randomInt(-1, 1);

			sceneTriangles->colors[i * 3 + 0] = (float)randomInt(0, 255);
			sceneTriangles->colors[i * 3 + 1] = (float)randomInt(0, 255);
			sceneTriangles->colors[i * 3 + 2] = (float)randomInt(0, 255);

			sceneTriangles->Roughness[i] = (float)(rand() % 100) / 100.0f;
			sceneTriangles->Metallic[i] = (float)(rand() % 100) / 100.0f;
			sceneTriangles->Emission[i] = (float)(rand() % 100) / 100.0f;
			sceneTriangles->count++;
		}
		clock_t start = clock();
		loadTriangles(staticRegion, sceneTriangles, INIT_MODE);
		clock_t end = clock();
		timeStemps[iter] = (float)(end - start) / CLOCKS_PER_SEC * 1000.0f;
	}

	for (int i = 0; i < iterations - 1; i++) {
		for (int j = i + 1; j < iterations; j++) {
			if (timeStemps[i] > timeStemps[j]) {
				float temp = timeStemps[i];
				timeStemps[i] = timeStemps[j];
				timeStemps[j] = temp;
			}
		}
	}

	int medianIdx = iterations / 2;
	float median = (iterations % 2 == 0) ? (timeStemps[medianIdx - 1] + timeStemps[medianIdx]) / 2.0f : timeStemps[medianIdx];

	int p99Idx = (int)(iterations * 0.99f);
	if (p99Idx >= iterations) p99Idx = iterations - 1;
	float p99 = timeStemps[p99Idx];

	printf("Benchmark Results (n=%d):\n", iterations);
	printf("  Median: %.6f ms\n", median);
	printf("  P99:    %.6f ms\n", p99);
	printf("  Min:    %.6f ms\n", timeStemps[0]);
	printf("  Max:    %.6f ms\n", timeStemps[iterations - 1]);

	free(staticRegion);
	free(sceneTriangles);

	return 0;
}
