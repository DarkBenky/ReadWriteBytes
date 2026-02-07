#define Capacity 1024
#define BUFFER_PERCENTAGE 0.2f
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <CL/cl.h>
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

struct Scene {
	struct Region staticGeometry;
	struct Triangles _tempTriangles; // Temporary storage for converting models to triangles
	struct Region dynamicGeometry;
	cl_mem buffer_sceneRegion;
	cl_kernel rayTraceScene;
	cl_mem buffer_distances;				 // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_normals;					 // ScreenWidth * ScreenHeight * sizeof(float) * 3
	cl_mem buffer_screen_colors;			 // ScreenWidth * ScreenHeight * sizeof(float) * 3
};

struct HitInfo {
	int hit;
	float t;
	float hitPoint[3];
	float hitNormal[3];
	float color[3];
	float roughness;
	float metallic;
	float emission;
	int volumeIdx;
	int triangleIdx;
};

static int rayBoxIntersect(float rayOrigin[3], float rayDir[3],
						   float boxMin[3], float boxMax[3],
						   float *tMin, float *tMax) {
	const float epsilon = 1e-6f;
	float t1 = -1e30f;
	float t2 = 1e30f;

	for (int i = 0; i < 3; i++) {
		if (fabsf(rayDir[i]) < epsilon) {
			if (rayOrigin[i] < boxMin[i] || rayOrigin[i] > boxMax[i]) {
				return 0;
			}
		} else {
			float invD = 1.0f / rayDir[i];
			float t_near = (boxMin[i] - rayOrigin[i]) * invD;
			float t_far = (boxMax[i] - rayOrigin[i]) * invD;

			if (t_near > t_far) {
				float temp = t_near;
				t_near = t_far;
				t_far = temp;
			}

			t1 = (t_near > t1) ? t_near : t1;
			t2 = (t_far < t2) ? t_far : t2;

			if (t1 > t2) {
				return 0;
			}
		}
	}

	*tMin = t1;
	*tMax = t2;
	return (t2 >= 0.0f);
}

static int rayTriangleIntersect(float rayOrigin[3], float rayDir[3],
								float v1[3], float v2[3], float v3[3],
								float *t, float hitNormal[3]) {
	const float epsilon = 1e-6f;

	float edge1[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
	float edge2[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};

	float h[3] = {
		rayDir[1] * edge2[2] - rayDir[2] * edge2[1],
		rayDir[2] * edge2[0] - rayDir[0] * edge2[2],
		rayDir[0] * edge2[1] - rayDir[1] * edge2[0]};

	float a = edge1[0] * h[0] + edge1[1] * h[1] + edge1[2] * h[2];

	if (a > -epsilon && a < epsilon) {
		return 0;
	}

	float f = 1.0f / a;
	float s[3] = {rayOrigin[0] - v1[0], rayOrigin[1] - v1[1], rayOrigin[2] - v1[2]};
	float u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);

	if (u < 0.0f || u > 1.0f) {
		return 0;
	}

	float q[3] = {
		s[1] * edge1[2] - s[2] * edge1[1],
		s[2] * edge1[0] - s[0] * edge1[2],
		s[0] * edge1[1] - s[1] * edge1[0]};

	float v = f * (rayDir[0] * q[0] + rayDir[1] * q[1] + rayDir[2] * q[2]);

	if (v < 0.0f || u + v > 1.0f) {
		return 0;
	}

	float tValue = f * (edge2[0] * q[0] + edge2[1] * q[1] + edge2[2] * q[2]);

	if (tValue > epsilon) {
		*t = tValue;

		float normal[3] = {
			edge1[1] * edge2[2] - edge1[2] * edge2[1],
			edge1[2] * edge2[0] - edge1[0] * edge2[2],
			edge1[0] * edge2[1] - edge1[1] * edge2[0]};

		float len = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
		if (len > epsilon) {
			hitNormal[0] = normal[0] / len;
			hitNormal[1] = normal[1] / len;
			hitNormal[2] = normal[2] / len;
		}

		return 1;
	}

	return 0;
}

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

struct HitInfo intersectRay(struct Region *region, float pos[3], float dir[3]) {
	struct HitInfo result = {0};
	result.hit = 0;
	result.t = 1e30f;

	const float epsilon = 1e-6f;
	const float inf = 1e30f;

	float tMin, tMax;
	if (!rayBoxIntersect(pos, dir, region->BBoxMin, region->BBoxMax, &tMin, &tMax)) {
		return result;
	}

	for (int b = 0; b < 8; b++) {
		struct Block *block = &region->blocks[b];

		if (block->BBoxMin[0] >= inf || block->BBoxMin[1] >= inf || block->BBoxMin[2] >= inf) {
			continue;
		}

		if (!rayBoxIntersect(pos, dir, block->BBoxMin, block->BBoxMax, &tMin, &tMax)) {
			continue;
		}

		for (int c = 0; c < 8; c++) {
			struct Cluster *cluster = &block->clusters[c];

			if (cluster->BBoxMin[0] >= inf || cluster->BBoxMin[1] >= inf || cluster->BBoxMin[2] >= inf) {
				continue;
			}

			if (!rayBoxIntersect(pos, dir, cluster->BBoxMin, cluster->BBoxMax, &tMin, &tMax)) {
				continue;
			}

			for (int v = 0; v < 8; v++) {
				struct Volume *volume = &cluster->volumes[v];

				if (volume->count == 0) {
					continue;
				}

				if (volume->BBoxMin[0] >= inf || volume->BBoxMin[1] >= inf || volume->BBoxMin[2] >= inf) {
					continue;
				}

				if (!rayBoxIntersect(pos, dir, volume->BBoxMin, volume->BBoxMax, &tMin, &tMax)) {
					continue;
				}

				for (int i = 0; i < volume->count; i++) {
					float v1[3] = {volume->v1[i * 3], volume->v1[i * 3 + 1], volume->v1[i * 3 + 2]};
					float v2[3] = {volume->v2[i * 3], volume->v2[i * 3 + 1], volume->v2[i * 3 + 2]};
					float v3[3] = {volume->v3[i * 3], volume->v3[i * 3 + 1], volume->v3[i * 3 + 2]};

					float t = 0.0f;
					float normal[3] = {0, 0, 0};

					if (rayTriangleIntersect(pos, dir, v1, v2, v3, &t, normal)) {
						if (t > epsilon && t < result.t) {
							result.hit = 1;
							result.t = t;

							result.hitPoint[0] = pos[0] + dir[0] * t;
							result.hitPoint[1] = pos[1] + dir[1] * t;
							result.hitPoint[2] = pos[2] + dir[2] * t;

							result.hitNormal[0] = normal[0];
							result.hitNormal[1] = normal[1];
							result.hitNormal[2] = normal[2];

							result.color[0] = volume->colors[i * 3];
							result.color[1] = volume->colors[i * 3 + 1];
							result.color[2] = volume->colors[i * 3 + 2];

							result.roughness = volume->Roughness[i];
							result.metallic = volume->Metallic[i];
							result.emission = volume->Emission[i];

							result.volumeIdx = v;
							result.triangleIdx = i;
						}
					}
				}
			}
		}
	}

	return result;
}

int intersectAny(struct Region *region, float pos[3], float dir[3], float maxDist) {
	const float epsilon = 1e-6f;
	const float inf = 1e30f;

	float tMin, tMax;
	if (!rayBoxIntersect(pos, dir, region->BBoxMin, region->BBoxMax, &tMin, &tMax)) {
		return 0;
	}

	for (int b = 0; b < 8; b++) {
		struct Block *block = &region->blocks[b];

		if (block->BBoxMin[0] >= inf || block->BBoxMin[1] >= inf || block->BBoxMin[2] >= inf) {
			continue;
		}

		if (!rayBoxIntersect(pos, dir, block->BBoxMin, block->BBoxMax, &tMin, &tMax)) {
			continue;
		}

		for (int c = 0; c < 8; c++) {
			struct Cluster *cluster = &block->clusters[c];

			if (cluster->BBoxMin[0] >= inf || cluster->BBoxMin[1] >= inf || cluster->BBoxMin[2] >= inf) {
				continue;
			}

			if (!rayBoxIntersect(pos, dir, cluster->BBoxMin, cluster->BBoxMax, &tMin, &tMax)) {
				continue;
			}

			for (int v = 0; v < 8; v++) {
				struct Volume *volume = &cluster->volumes[v];

				if (volume->count == 0) {
					continue;
				}

				if (volume->BBoxMin[0] >= inf || volume->BBoxMin[1] >= inf || volume->BBoxMin[2] >= inf) {
					continue;
				}

				if (!rayBoxIntersect(pos, dir, volume->BBoxMin, volume->BBoxMax, &tMin, &tMax)) {
					continue;
				}

				for (int i = 0; i < volume->count; i++) {
					float v1[3] = {volume->v1[i * 3], volume->v1[i * 3 + 1], volume->v1[i * 3 + 2]};
					float v2[3] = {volume->v2[i * 3], volume->v2[i * 3 + 1], volume->v2[i * 3 + 2]};
					float v3[3] = {volume->v3[i * 3], volume->v3[i * 3 + 1], volume->v3[i * 3 + 2]};

					float t = 0.0f;
					float normal[3] = {0, 0, 0};

					if (rayTriangleIntersect(pos, dir, v1, v2, v3, &t, normal)) {
						if (t > epsilon && t <= maxDist) {
							return 1;
						}
					}
				}
			}
		}
	}

	return 0;
}

enum Mode {
	APPEND_MODE,
	INIT_MODE,
};

void convertModelToSceneTriangles(struct Triangles *model, float position[3], float direction[3], struct Triangles *sceneTriangles, enum Mode mode) {
	if (!model || !sceneTriangles) {
		return;
	}

	if (mode == INIT_MODE) {
		sceneTriangles->count = 0;
	}

	if (sceneTriangles->count + model->count > NUMBER_OF_TRIANGLES) {
		return;
	}

	int offset = sceneTriangles->count;

	float dirLen = sqrtf(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
	if (dirLen < 0.0001f) {
		dirLen = 1.0f;
	}
	float normDir[3] = {direction[0] / dirLen, direction[1] / dirLen, direction[2] / dirLen};

	float defaultDir[3] = {0.0f, 0.0f, 1.0f};

	float axis[3] = {
		defaultDir[1] * normDir[2] - defaultDir[2] * normDir[1],
		defaultDir[2] * normDir[0] - defaultDir[0] * normDir[2],
		defaultDir[0] * normDir[1] - defaultDir[1] * normDir[0]};

	float cosAngle = defaultDir[0] * normDir[0] + defaultDir[1] * normDir[1] + defaultDir[2] * normDir[2];
	float angle = acosf(cosAngle);
	float axisLen = sqrtf(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);

	float rotMatrix[3][3];
	if (axisLen > 0.0001f && fabsf(angle) > 0.0001f) {
		axis[0] /= axisLen;
		axis[1] /= axisLen;
		axis[2] /= axisLen;

		float c = cosf(angle);
		float s = sinf(angle);
		float t = 1.0f - c;

		rotMatrix[0][0] = t * axis[0] * axis[0] + c;
		rotMatrix[0][1] = t * axis[0] * axis[1] - s * axis[2];
		rotMatrix[0][2] = t * axis[0] * axis[2] + s * axis[1];

		rotMatrix[1][0] = t * axis[0] * axis[1] + s * axis[2];
		rotMatrix[1][1] = t * axis[1] * axis[1] + c;
		rotMatrix[1][2] = t * axis[1] * axis[2] - s * axis[0];

		rotMatrix[2][0] = t * axis[0] * axis[2] - s * axis[1];
		rotMatrix[2][1] = t * axis[1] * axis[2] + s * axis[0];
		rotMatrix[2][2] = t * axis[2] * axis[2] + c;
	} else {
		rotMatrix[0][0] = 1.0f;
		rotMatrix[0][1] = 0.0f;
		rotMatrix[0][2] = 0.0f;
		rotMatrix[1][0] = 0.0f;
		rotMatrix[1][1] = 1.0f;
		rotMatrix[1][2] = 0.0f;
		rotMatrix[2][0] = 0.0f;
		rotMatrix[2][1] = 0.0f;
		rotMatrix[2][2] = 1.0f;
	}

	for (int i = 0; i < model->count; i++) {
		int srcIdx = i * 3;
		int dstIdx = (offset + i) * 3;

		float v1[3] = {model->v1[srcIdx + 0], model->v1[srcIdx + 1], model->v1[srcIdx + 2]};
		sceneTriangles->v1[dstIdx + 0] = rotMatrix[0][0] * v1[0] + rotMatrix[0][1] * v1[1] + rotMatrix[0][2] * v1[2] + position[0];
		sceneTriangles->v1[dstIdx + 1] = rotMatrix[1][0] * v1[0] + rotMatrix[1][1] * v1[1] + rotMatrix[1][2] * v1[2] + position[1];
		sceneTriangles->v1[dstIdx + 2] = rotMatrix[2][0] * v1[0] + rotMatrix[2][1] * v1[1] + rotMatrix[2][2] * v1[2] + position[2];

		float v2[3] = {model->v2[srcIdx + 0], model->v2[srcIdx + 1], model->v2[srcIdx + 2]};
		sceneTriangles->v2[dstIdx + 0] = rotMatrix[0][0] * v2[0] + rotMatrix[0][1] * v2[1] + rotMatrix[0][2] * v2[2] + position[0];
		sceneTriangles->v2[dstIdx + 1] = rotMatrix[1][0] * v2[0] + rotMatrix[1][1] * v2[1] + rotMatrix[1][2] * v2[2] + position[1];
		sceneTriangles->v2[dstIdx + 2] = rotMatrix[2][0] * v2[0] + rotMatrix[2][1] * v2[1] + rotMatrix[2][2] * v2[2] + position[2];

		float v3[3] = {model->v3[srcIdx + 0], model->v3[srcIdx + 1], model->v3[srcIdx + 2]};
		sceneTriangles->v3[dstIdx + 0] = rotMatrix[0][0] * v3[0] + rotMatrix[0][1] * v3[1] + rotMatrix[0][2] * v3[2] + position[0];
		sceneTriangles->v3[dstIdx + 1] = rotMatrix[1][0] * v3[0] + rotMatrix[1][1] * v3[1] + rotMatrix[1][2] * v3[2] + position[1];
		sceneTriangles->v3[dstIdx + 2] = rotMatrix[2][0] * v3[0] + rotMatrix[2][1] * v3[1] + rotMatrix[2][2] * v3[2] + position[2];

		sceneTriangles->Roughness[offset + i] = model->Roughness[i];
		sceneTriangles->Metallic[offset + i] = model->Metallic[i];
		sceneTriangles->Emission[offset + i] = model->Emission[i];

		float n[3] = {model->normals[srcIdx + 0], model->normals[srcIdx + 1], model->normals[srcIdx + 2]};
		sceneTriangles->normals[dstIdx + 0] = rotMatrix[0][0] * n[0] + rotMatrix[0][1] * n[1] + rotMatrix[0][2] * n[2];
		sceneTriangles->normals[dstIdx + 1] = rotMatrix[1][0] * n[0] + rotMatrix[1][1] * n[1] + rotMatrix[1][2] * n[2];
		sceneTriangles->normals[dstIdx + 2] = rotMatrix[2][0] * n[0] + rotMatrix[2][1] * n[1] + rotMatrix[2][2] * n[2];

		sceneTriangles->colors[dstIdx + 0] = model->colors[srcIdx + 0];
		sceneTriangles->colors[dstIdx + 1] = model->colors[srcIdx + 1];
		sceneTriangles->colors[dstIdx + 2] = model->colors[srcIdx + 2];
	}
	sceneTriangles->count += model->count;
}

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

// PIPELINE:
// 1. initGeometry (once): Load static map geometry into both static and dynamic regions
// 2. Per-frame (or every N frames):
//    a. resetScene: Copy static geometry to dynamic region (clears previous frame's dynamic objects)
//    b. addToScene: Add models at new positions/directions to dynamic region (planes, missiles, etc.)
//    c. Render using dynamicGeometry
//
// This allows static map to persist while dynamic objects are repositioned each frame

void initGeometry(struct Scene *scene, struct Triangles *staticGeometry) {
	if (!scene || !staticGeometry) {
		printf("Invalid input to initGeometry\n");
		return;
	}
	loadTriangles(&scene->staticGeometry, staticGeometry, INIT_MODE);
	loadTriangles(&scene->dynamicGeometry, staticGeometry, INIT_MODE);
}

void addToScene(struct Scene *scene, struct Triangles *model, float position[3], float direction[3]) {
	if (!scene || !model) {
		printf("Invalid input to addToScene\n");
		return;
	}
	convertModelToSceneTriangles(model, position, direction, &scene->_tempTriangles, INIT_MODE);
	loadTriangles(&scene->dynamicGeometry, &scene->_tempTriangles, APPEND_MODE);
	// clear temp triangles after loading into scene
	scene->_tempTriangles.count = 0;
}

void resetScene(struct Scene *scene) {
	// when we reset the scene, we set dynamic geometry to be the same as static geometry
	if (!scene) {
		printf("Invalid input to resetScene\n");
		return;
	}
	memcopy(&scene->staticGeometry, &scene->dynamicGeometry, sizeof(struct Triangles));
}



int main() {
	int iterations = 1000;
	float timeStemps[iterations];
	float timeStempsModelCopy[iterations];

	struct Region *staticRegion = malloc(sizeof(struct Region));
	struct Triangles *sceneTriangles = malloc(sizeof(struct Triangles));
	struct Triangles *model = malloc(sizeof(struct Triangles));
	sceneTriangles->count = 0;
	model->count = 0;

	for (int i = 0; i < NUMBER_OF_TRIANGLES / 4; i++) {
		model->v1[i * 3 + 0] = (float)randomInt(-10, 10);
		model->v1[i * 3 + 1] = (float)randomInt(-10, 10);
		model->v1[i * 3 + 2] = (float)randomInt(-10, 10);

		model->v2[i * 3 + 0] = (float)randomInt(-10, 10);
		model->v2[i * 3 + 1] = (float)randomInt(-10, 10);
		model->v2[i * 3 + 2] = (float)randomInt(-10, 10);

		model->v3[i * 3 + 0] = (float)randomInt(-10, 10);
		model->v3[i * 3 + 1] = (float)randomInt(-10, 10);
		model->v3[i * 3 + 2] = (float)randomInt(-10, 10);

		model->normals[i * 3 + 0] = (float)randomInt(-1, 1);
		model->normals[i * 3 + 1] = (float)randomInt(-1, 1);
		model->normals[i * 3 + 2] = (float)randomInt(-1, 1);

		model->colors[i * 3 + 0] = (float)randomInt(0, 255);
		model->colors[i * 3 + 1] = (float)randomInt(0, 255);
		model->colors[i * 3 + 2] = (float)randomInt(0, 255);

		model->Roughness[i] = (float)(rand() % 100) / 100.0f;
		model->Metallic[i] = (float)(rand() % 100) / 100.0f;
		model->Emission[i] = (float)(rand() % 100) / 100.0f;
		model->count++;
	}

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

		float position[3] = {(float)randomInt(-100, 100), (float)randomInt(-100, 100), (float)randomInt(-100, 100)};
		float direction[3] = {(float)randomInt(-1, 1), (float)randomInt(-1, 1), (float)randomInt(-1, 1)};

		clock_t startModel = clock();
		convertModelToSceneTriangles(model, position, direction, sceneTriangles, APPEND_MODE);
		clock_t endModel = clock();
		timeStempsModelCopy[iter] = (float)(endModel - startModel) / CLOCKS_PER_SEC * 1000.0f;

		clock_t startLoad = clock();
		loadTriangles(staticRegion, sceneTriangles, INIT_MODE);
		clock_t endLoad = clock();
		timeStemps[iter] = (float)(endLoad - startLoad) / CLOCKS_PER_SEC * 1000.0f;
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

	for (int i = 0; i < iterations - 1; i++) {
		for (int j = i + 1; j < iterations; j++) {
			if (timeStempsModelCopy[i] > timeStempsModelCopy[j]) {
				float temp = timeStempsModelCopy[i];
				timeStempsModelCopy[i] = timeStempsModelCopy[j];
				timeStempsModelCopy[j] = temp;
			}
		}
	}

	int medianIdx = iterations / 2;
	float median = (iterations % 2 == 0) ? (timeStemps[medianIdx - 1] + timeStemps[medianIdx]) / 2.0f : timeStemps[medianIdx];
	float medianModel = (iterations % 2 == 0) ? (timeStempsModelCopy[medianIdx - 1] + timeStempsModelCopy[medianIdx]) / 2.0f : timeStempsModelCopy[medianIdx];

	int p99Idx = (int)(iterations * 0.99f);
	if (p99Idx >= iterations) p99Idx = iterations - 1;
	float p99 = timeStemps[p99Idx];
	float p99Model = timeStempsModelCopy[p99Idx];

	printf("=== loadTriangles Benchmark Results (n=%d) ===\n", iterations);
	printf("  Median: %.6f ms\n", median);
	printf("  P99:    %.6f ms\n", p99);
	printf("  Min:    %.6f ms\n", timeStemps[0]);
	printf("  Max:    %.6f ms\n", timeStemps[iterations - 1]);

	printf("\n=== convertModelToSceneTriangles Benchmark Results (n=%d) ===\n", iterations);
	printf("  Median: %.6f ms\n", medianModel);
	printf("  P99:    %.6f ms\n", p99Model);
	printf("  Min:    %.6f ms\n", timeStempsModelCopy[0]);
	printf("  Max:    %.6f ms\n", timeStempsModelCopy[iterations - 1]);

	printf("\n=== Ray Intersection Tests ===\n");

	sceneTriangles->count = 0;
	sceneTriangles->v1[0] = 0.0f;
	sceneTriangles->v1[1] = 0.0f;
	sceneTriangles->v1[2] = 10.0f;
	sceneTriangles->v2[0] = 10.0f;
	sceneTriangles->v2[1] = 0.0f;
	sceneTriangles->v2[2] = 10.0f;
	sceneTriangles->v3[0] = 5.0f;
	sceneTriangles->v3[1] = 10.0f;
	sceneTriangles->v3[2] = 10.0f;
	sceneTriangles->normals[0] = 0.0f;
	sceneTriangles->normals[1] = 0.0f;
	sceneTriangles->normals[2] = -1.0f;
	sceneTriangles->colors[0] = 255.0f;
	sceneTriangles->colors[1] = 0.0f;
	sceneTriangles->colors[2] = 0.0f;
	sceneTriangles->Roughness[0] = 0.5f;
	sceneTriangles->Metallic[0] = 0.1f;
	sceneTriangles->Emission[0] = 0.0f;
	sceneTriangles->count = 1;

	loadTriangles(staticRegion, sceneTriangles, INIT_MODE);

	float rayPos[3] = {5.0f, 3.0f, 0.0f};
	float rayDir[3] = {0.0f, 0.0f, 1.0f};

	printf("\nTest 1: intersectRay - Ray through triangle\n");
	struct HitInfo hit = intersectRay(staticRegion, rayPos, rayDir);
	if (hit.hit) {
		printf("  Hit at distance %.2f\n", hit.t);
		printf("  Hit point: (%.2f, %.2f, %.2f)\n", hit.hitPoint[0], hit.hitPoint[1], hit.hitPoint[2]);
		printf("  Normal: (%.2f, %.2f, %.2f)\n", hit.hitNormal[0], hit.hitNormal[1], hit.hitNormal[2]);
		printf("  Color: (%.0f, %.0f, %.0f)\n", hit.color[0], hit.color[1], hit.color[2]);
		printf("  Roughness: %.2f, Metallic: %.2f, Emission: %.2f\n", hit.roughness, hit.metallic, hit.emission);
	} else {
		printf("  No hit (unexpected)\n");
	}

	printf("\nTest 2: intersectRay - Ray missing triangle\n");
	float rayPos2[3] = {100.0f, 100.0f, 0.0f};
	float rayDir2[3] = {0.0f, 0.0f, 1.0f};
	struct HitInfo hit2 = intersectRay(staticRegion, rayPos2, rayDir2);
	if (hit2.hit) {
		printf("  Hit at distance %.2f (unexpected)\n", hit2.t);
	} else {
		printf("  No hit (expected)\n");
	}

	printf("\nTest 3: intersectAny - Ray within maxDist\n");
	float maxDist = 20.0f;
	int anyHit = intersectAny(staticRegion, rayPos, rayDir, maxDist);
	if (anyHit) {
		printf("  Object detected within %.2f units (expected)\n", maxDist);
	} else {
		printf("  No object detected (unexpected)\n");
	}

	printf("\nTest 4: intersectAny - Ray beyond maxDist\n");
	float shortDist = 5.0f;
	int anyHit2 = intersectAny(staticRegion, rayPos, rayDir, shortDist);
	if (anyHit2) {
		printf("  Object detected within %.2f units (unexpected)\n", shortDist);
	} else {
		printf("  No object detected within %.2f units (expected)\n", shortDist);
	}

	free(staticRegion);
	free(sceneTriangles);
	free(model);

	return 0;
}
