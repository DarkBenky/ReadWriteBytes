#ifndef LOAD_TO_GPU_H
#define LOAD_TO_GPU_H

#include <stdbool.h>
#include <CL/cl.h>

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
