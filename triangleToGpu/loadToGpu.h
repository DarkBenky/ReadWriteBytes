#ifndef LOAD_TO_GPU_H
#define LOAD_TO_GPU_H

#define Capacity 1024
#define BUFFER_PERCENTAGE 0.2f

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>
#include "../openGlShaders/gpuStruct.h"
#include "../mapGeneration/loadMap.h"
#include "../utils/image.h"
#include "../utils/bbox.h"

/*
 * Usage Example:
 *
 * // Initialize Scene structure
 * struct Scene scene = {0};
 *
 * // Setup OpenCL context and queue (from your OpenCL initialization)
 * initGpuBuffers(&scene, context, queue, screenWidth, screenHeight);
 *
 * // Load geometry and skybox
 * loadSkyBoxForScene(&scene);
 * uploadSkyboxToGpu(&scene);
 * initGeometry(&scene, staticTriangles);
 *
 * // Initialize the ray tracing kernel
 * if (!initRayTraceKernel(&scene)) {
 *     printf("Failed to initialize ray trace kernel\n");
 *     return;
 * }
 *
 * // Set up camera and rendering parameters
 * float cameraPos[3] = {0.0f, 10.0f, -20.0f};
 * float cameraDir[3] = {0.0f, 0.0f, 1.0f};
 * float fov = 1.0f;
 * float sunDir[3] = {0.5f, 0.5f, 0.5f};
 * float sunColor[3] = {1.0f, 1.0f, 1.0f};
 * float sunIntensity = 1.5f;
 * int maxBounces = 3;
 *
 * // Launch the ray tracing kernel
 * launchRayTraceKernel(&scene, cameraPos, cameraDir, fov,
 *                      screenWidth, screenHeight,
 *                      sunDir, sunColor, sunIntensity, maxBounces);
 *
 * // Read back results from buffer_screen_colors, buffer_normals, buffer_distances
 * // Cleanup when done
 * cleanupGpuBuffers(&scene);
 */

enum Mode {
	APPEND_MODE,
	INIT_MODE,
};

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
	struct SkyBox SkyBox;
	cl_context context;
	cl_command_queue queue;
	cl_device_id device;
	cl_mem buffer_sceneRegion;
	cl_kernel rayTraceScene;
	cl_mem buffer_distances;	 // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_normals;		 // ScreenWidth * ScreenHeight * sizeof(float) * 3
	cl_mem buffer_screen_colors; // ScreenWidth * ScreenHeight * sizeof(float) * 3
	cl_mem buffer_skybox_top;
	cl_mem buffer_skybox_bottom;
	cl_mem buffer_skybox_left;
	cl_mem buffer_skybox_right;
	cl_mem buffer_skybox_front;
	cl_mem buffer_skybox_back;
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

struct HitInfo intersectRay(struct Region *region, float pos[3], float dir[3]);
int intersectAny(struct Region *region, float pos[3], float dir[3], float maxDist);
void convertModelToSceneTriangles(struct Triangles *model, float position[3], float direction[3], struct Triangles *sceneTriangles, enum Mode mode);
void loadTriangles(struct Region *staticRegion, struct Triangles *sceneTriangles, enum Mode mode);

void initGeometry(struct Scene *scene, struct Triangles *staticGeometry);
void addToScene(struct Scene *scene, struct Triangles *model, float position[3], float direction[3]);
void initGpuBuffers(struct Scene *scene, cl_context context, cl_command_queue queue, int screenWidth, int screenHeight);
void uploadSkyboxToGpu(struct Scene *scene);
void uploadToGpu(struct Scene *scene);
void cleanupGpuBuffers(struct Scene *scene);
void resetScene(struct Scene *scene);
bool loadSkyBoxForScene(struct Scene *scene);

bool initRayTraceKernel(struct Scene *scene);
void launchRayTraceKernel(struct Scene *scene,
						  struct Camera camera,
						  float fov,
						  int screenWidth,
						  int screenHeight,
						  float sunDir[3],
						  float sunColor[3],
						  float sunIntensity,
						  int maxBounces);

#endif
