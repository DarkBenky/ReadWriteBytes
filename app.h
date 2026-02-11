#ifndef APP_H
#define APP_H

#include <stdbool.h>
#include <time.h>

#include <GLFW/glfw3.h>

#include "fireSim/fireSim.h"
#include "mapGeneration/loadMap.h"
#include "particleSim.h"
#include "triangleToGpu/loadToGpu.h"

#define FrameCount 30

// Forward declaration to avoid pulling the ParticleIndexes definition into this header.
struct ParticleIndexes;

/*
 * RenderContext groups render-only pointers that reference AppState-owned data.
 * It does not own these pointers and is refreshed internally before rendering.
 */
struct RenderContext {
	struct PointSOA *particles;
	struct Camera *camera;
	struct TimePartition *timePartition;
	struct ParticleIndexes *particleIndexes;
	struct OpenCLContext *openCLContext;
	struct Triangles *triangles;
	struct SkyBox *skyBox;
	struct GPUTimings *gpuTimings;
	struct ImageFont *font;
	struct FireSOA *fireParticles;
	struct Missiles *missiles;
	struct IRSearchAndTrack *irst;
};

struct AppState {
	struct Triangles *missileModel;
	struct Missiles missiles;
	struct BVHLinear bvh;
	struct ImageFont font;
	struct SkyBox skyBox;
	struct Triangles *triangles;
	struct Camera camera;
	struct IRSearchAndTrack irst;
	struct ParticleIndexes *particleIndexes;
	struct PointSOA *particles;
	struct TimePartition *timePartition;
	struct Map terrain;
	struct MapGPU *mapGPU;
	struct GPUTimings gpuTimings;
	struct OpenCLContext ocl;
	GLFWwindow *window;
	struct FireSOA *fireParticles;
	struct Triangles *sceneTriangles;
	struct Scene *scene;
	struct RenderContext renderContext;
	bool paused;
	bool fireMissile;
	bool exit;
	int frameCount;
	float averageFPS[FrameCount];
	clock_t lastTime;
	float yaw;   // Input-driven camera yaw angle in degrees
	float pitch; // Input-driven camera pitch angle in degrees
};

// Allocates and initializes AppState resources; returns false on setup failures.
// initializeApp calls cleanupApp before returning false on failure.
bool initializeApp(struct AppState *state);
// Returns true when the app should stop running.
bool appShouldExit(const struct AppState *state);
// Runs a single frame update/render pass.
void runAppFrame(struct AppState *state);
// Releases resources owned by the AppState.
void cleanupApp(struct AppState *state);

#endif
