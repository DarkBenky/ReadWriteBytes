#ifndef APP_H
#define APP_H

#include <stdbool.h>
#include <time.h>

#include <GLFW/glfw3.h>

#include "fireSim/fireSim.h"
#include "mapGeneration/loadMap.h"
#include "particleSim.h"
#include "triangleToGpu/loadToGpu.h"

#ifndef FrameCount
#define FrameCount 30
#endif

struct ParticleIndexes;

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
	float yaw;
	float pitch;
};

bool initializeApp(struct AppState *state);
bool appShouldExit(const struct AppState *state);
void runAppFrame(struct AppState *state);
void cleanupApp(struct AppState *state);

#endif
