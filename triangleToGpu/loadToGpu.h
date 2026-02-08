#ifndef LOAD_TO_GPU_H
#define LOAD_TO_GPU_H

#include <stdbool.h>
#include <CL/cl.h>

struct Scene;
struct Triangles;

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
						  float cameraPos[3], 
						  float cameraDir[3], 
						  float fov,
						  int screenWidth,
						  int screenHeight,
						  float sunDir[3],
						  float sunColor[3],
						  float sunIntensity,
						  int maxBounces);

#endif
