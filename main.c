#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>	   // for rand()
#include <unistd.h>	   // for usleep()
#include <math.h>	   // for sqrtf()
#include <time.h>	   // for time()
#include <string.h>	   // for memset()
#include <stdbool.h>   // for bool, true, false
#include <immintrin.h> // for AVX intrinsics
#include <omp.h>	   // for OpenMP
#include <CL/cl.h>	   // Add this line for OpenCL
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"
#include "particleSim.h"
#include <jpeglib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include "fireSim/fireSim.h"
#include "openGlShaders/gpuStruct.h"
#include "mapGeneration/loadMap.h"

void *SharedMem = NULL;
#define USE_GPU_LOD 1 // 1 = GPU LOD selection, 0 = CPU LOD selection
#define chartPosY 480 // Y position on screen for timing chart
#define chartPosX 700 // X position on screen for timing chart
#define MAX_TEXT_LENGTH 2048
// temporary buffer for saving text
char text[MAX_TEXT_LENGTH];
// Buffers for all text rendered on screen
uint32_t textColor[MAX_TEXT_LENGTH];
int posX[MAX_TEXT_LENGTH];
int posY[MAX_TEXT_LENGTH];
char textBuffer[MAX_TEXT_LENGTH];
int textBufferLen = 0;

#define WATER_SIMULATION 0
#define FLT_MAX 3.402823466e+38F
#define FLT_MIN 1.175494e-38F
#define RENDER_TRIAGES 1 // 1 = CALCULATE VERTEXES => PER PIXEL SHADING, 0 = RENDER PER TRIANGLE
#define ScreenWidth 800
#define ScreenHeight 600
#define SHM_NAME "/my_shared_mem"
#define SIZE ScreenWidth *ScreenHeight * 4 * 2 // Color + Normal
#define PARTICLE_RADIUS 4
#define FrameCount 30
#define NUM_THREADS 0
#define USE_GPU 1
#define NUMBER_OF_CUBES 100
pthread_t threads[NUM_THREADS];
#define GLFW_EXPOSE_NATIVE_X11
#define MoveMultiplier 100.25f
#define MouseSensitivity 0.25f
#define MAX_BLUR_PASSES 1
#define numFireParticles 10000
#define fireParticleSize 10.0f
#include <GLFW/glfw3native.h>
#include <CL/cl_gl.h>

#ifdef __linux__
#include <GL/glx.h>
#include <X11/Xlib.h>
#endif

#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <CL/cl_gl.h>

const char *clErrorString(cl_int err) {
	switch (err) {
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	default:
		return "UNKNOWN_OPENCL_ERROR";
	}
}

#define CHECK_CL(err, call)                             \
	do {                                                \
		if ((err) != CL_SUCCESS) {                      \
			fprintf(stderr, "%s => %s (%d)\n",          \
					(call), clErrorString(err), (err)); \
			exit(1);                                    \
		}                                               \
	} while (0)

#define CL_ERROR(err, call) CHECK_CL(err, call)

struct KeyState {
	bool keys[GLFW_KEY_LAST + 1];		 // Array to store state of all keys
	bool prevKeys[GLFW_KEY_LAST + 1];	 // Previous frame state for detecting press/release
	bool justPressed[GLFW_KEY_LAST + 1]; // Latched presses set in callback
};

struct MouseState {
	double x, y;		   // Current mouse position
	double prevX, prevY;   // Previous frame mouse position
	double deltaX, deltaY; // Change in position this frame
	bool leftButton;	   // Left mouse button state
	bool rightButton;	   // Right mouse button state
	bool prevLeftButton;   // Previous left button state
	bool prevRightButton;  // Previous right button state
	bool firstMouse;	   // Flag to handle first mouse movement
};

struct KeyState keyState = {0};
struct MouseState mouseState = {0};

char *renderModesName[] = {
	"Distance",
	"Velocity",
	"Opacity",
	"Normal",
	"Fluid",
	"Color",
	"Wireframe",
	"renderFireColor",
	"renderFireDepth",
	"renderFireNormal",
	"renderCompositedNormal",
	"renderCompositedColor",
	"renderCompositedDistance",
	"renderTemperatures",
	"renderDebugMode",
};

struct RawImage {
	unsigned char *data; // RGB pixel data
	int width, height, components;
};

struct ImageFont {
	int width;
	int height;
	char *data;
};

struct Triangle {
	float v1[3];	   // Vertex 1
	float v2[3];	   // Vertex 2
	float v3[3];	   // Vertex 3
	float normal[3];   // Normal vector
	float color[3];	   // RGB color
	float Roughness;   // Material roughness
	float Metallic;	   // Material metallic
	float Emission;	   // Material emission
	int TriangleIndex; // Index of the triangle
};

struct BVHNode {
	float BoundingBox[6]; // minX, minY, minZ, maxX, maxY, maxZ
	int LeftChild;		  // Index of left child node
	int RightChild;		  // Index of right child node
	int TriangleIndex;
};

struct BVHLinear {
	struct BVHNode *Nodes;		// Array of BVH nodes
	struct Triangle *Triangles; // Array of triangles
	int NodesCount;				// Number of nodes in the BVH
	int TrianglesCount;			// Number of triangles in the BVH
};

struct RawImage *load_jpeg(const char *filename) {
	FILE *f = fopen(filename, "rb");
	if (!f) return NULL;

	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, f);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);

	struct RawImage *img = malloc(sizeof(*img));
	img->width = cinfo.output_width;
	img->height = cinfo.output_height;
	img->components = cinfo.output_components; // usually 3 for RGB

	size_t rowbytes = img->width * img->components;
	img->data = malloc(img->height * rowbytes);

	JSAMPROW rowptr[1];
	while (cinfo.output_scanline < img->height) {
		rowptr[0] = img->data + rowbytes * cinfo.output_scanline;
		jpeg_read_scanlines(&cinfo, rowptr, 1);
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(f);

	return img;
}

struct SkyBox {
	struct RawImage *right;
	struct RawImage *left;
	struct RawImage *top;
	struct RawImage *bottom;
	struct RawImage *front;
	struct RawImage *back;
};

float *convertImageToFloat(struct RawImage *img) {
	if (!img) return NULL;

	float *data = malloc(img->width * img->height * 3 * sizeof(float));
	if (!data) return NULL;

	for (int i = 0; i < img->width * img->height * img->components; i += img->components) {
		int floatIdx = (i / img->components) * 3;
		data[floatIdx + 0] = (float)img->data[i + 0] / 255.0f; // R
		data[floatIdx + 1] = (float)img->data[i + 1] / 255.0f; // G
		data[floatIdx + 2] = (float)img->data[i + 2] / 255.0f; // B
	}

	return data;
}

bool loadSkyBox(struct SkyBox *skyBox) { // Changed from void to bool
	skyBox->right = load_jpeg("skybox/right.jpg");
	skyBox->left = load_jpeg("skybox/left.jpg");
	skyBox->top = load_jpeg("skybox/top.jpg");
	skyBox->bottom = load_jpeg("skybox/bottom.jpg");
	skyBox->front = load_jpeg("skybox/front.jpg");
	skyBox->back = load_jpeg("skybox/back.jpg");

	if (!skyBox->right || !skyBox->left || !skyBox->top ||
		!skyBox->bottom || !skyBox->front || !skyBox->back) {
		printf("Failed to load one or more skybox images\n");
		return false; // Return false on failure
	}

	return true; // Return true on success
}

void filterOverlapOpenCL(
	struct OpenCLContext *ocl,
	cl_mem inputBuffer,
	cl_mem inputDistance1,
	cl_mem inputDistance2,
	cl_mem outputBuffer,
	int mode // 0=RGB, 1=RGBA, 2=single channel
) {
	cl_int err;
	cl_int screenWidth = ScreenWidth;
	cl_int screenHeight = ScreenHeight;

	int size = ScreenWidth * ScreenHeight * 4 * sizeof(float);
	if (mode == 0)
		size = ScreenWidth * ScreenHeight * 3 * sizeof(float);
	else if (mode == 2)
		size = ScreenWidth * ScreenHeight * sizeof(float);

	// clear output buffer
	float zeroFloat = 0.0f;
	err = clEnqueueFillBuffer(ocl->queue, outputBuffer, &zeroFloat, sizeof(float), 0,
							  size, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error clearing output buffer: %d\n", err);
		return;
	}

	// Set kernel arguments
	err = clSetKernelArg(ocl->renderFireTemperature_kernel, 0, sizeof(cl_mem), &inputBuffer);
	err |= clSetKernelArg(ocl->renderFireTemperature_kernel, 1, sizeof(cl_mem), &inputDistance1);
	err |= clSetKernelArg(ocl->renderFireTemperature_kernel, 2, sizeof(cl_mem), &inputDistance2);
	err |= clSetKernelArg(ocl->renderFireTemperature_kernel, 3, sizeof(cl_mem), &outputBuffer);
	err |= clSetKernelArg(ocl->renderFireTemperature_kernel, 4, sizeof(cl_int), &screenWidth);
	err |= clSetKernelArg(ocl->renderFireTemperature_kernel, 5, sizeof(cl_int), &screenHeight);

	if (err != CL_SUCCESS) {
		printf("Error setting filterOverlap kernel args: %d\n", err);
		return;
	}

	// Execute kernel
	size_t global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->renderFireTemperature_kernel, 2, NULL,
								 global_size, NULL, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error executing filterOverlap kernel: %d\n", err);
		return;
	}

	clFinish(ocl->queue);
}

int launchOverlayImageOpenCL(
	struct OpenCLContext *ocl,
	cl_mem OutputBuffer, // target buffer (device) where overlay will be written
	cl_mem ImageBuffer,	 // source image buffer (device) laid out according to displayMode)
	int screenWidth,
	int screenHeight,
	int imageWidth,
	int imageHeight,
	int Outputmode,	 // 0=RGB, 1=RGBA, 2=Grayscale
	int displayMode, // 0=RGB, 1=RGBA, 2=Grayscale
	float *outGpuMs,
	int posX,
	int posY) {
	if (!ocl || !ocl->overlayImage_kernel) {
		fprintf(stderr, "Overlay kernel or context not initialized\n");
		return 0;
	}

	cl_int err;
	cl_event evt = NULL;

	// set kernel args
	err = clSetKernelArg(ocl->overlayImage_kernel, 0, sizeof(cl_mem), &OutputBuffer);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 1, sizeof(cl_mem), &ImageBuffer);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 2, sizeof(cl_int), &screenWidth);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 3, sizeof(cl_int), &screenHeight);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 4, sizeof(cl_int), &imageWidth);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 5, sizeof(cl_int), &imageHeight);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 6, sizeof(cl_int), &Outputmode);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 7, sizeof(cl_int), &displayMode);
	// new args for position
	err |= clSetKernelArg(ocl->overlayImage_kernel, 8, sizeof(cl_int), &posX);
	err |= clSetKernelArg(ocl->overlayImage_kernel, 9, sizeof(cl_int), &posY);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error setting OverlayImage kernel args: %s (%d)\n", clErrorString(err), err);
		return 0;
	}

	size_t global[2] = {(size_t)screenWidth, (size_t)screenHeight};

	err = clEnqueueNDRangeKernel(ocl->queue, ocl->overlayImage_kernel, 2, NULL, global, NULL, 0, NULL, &evt);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error enqueuing OverlayImage kernel: %s (%d)\n", clErrorString(err), err);
		return 0;
	}

	// Wait and optionally measure time
	clFinish(ocl->queue);

	if (outGpuMs != NULL) {
		cl_ulong t0 = 0, t1 = 0;
		if (evt) {
			clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
			clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
			*outGpuMs = (t1 - t0) * 1e-6f;
		} else {
			*outGpuMs = 0.0f;
		}
	}

	if (evt) clReleaseEvent(evt);
	return 1;
}

int renderDepthBuffer(
	struct OpenCLContext *ocl,
	cl_mem v1_buffer,	   // Device buffer: triangle vertex 1 data
	cl_mem v2_buffer,	   // Device buffer: triangle vertex 2 data
	cl_mem v3_buffer,	   // Device buffer: triangle vertex 3 data
	cl_mem normals_buffer, // Device buffer: triangle normals (can be NULL)
	cl_mem output_buffer,  // Device buffer: output depth buffer
	int triangle_count,	   // Number of triangles to render
	float cam_pos[3],	   // Camera position
	float cam_dir[3],	   // Camera direction
	float fov,			   // Field of view
	int screen_width,	   // Screen width
	int screen_height,	   // Screen height
	float *out_depth_cpu,  // CPU buffer to read back depth data (optional)
	float *outGpuMs,	   // Output GPU time in milliseconds (optional)
	int missileIdx) {
	if (!ocl || !ocl->renderDepth) {
		fprintf(stderr, "Depth kernel or context not initialized\n");
		return 0;
	}

	if (!v1_buffer || !v2_buffer || !v3_buffer || !output_buffer || triangle_count <= 0) {
		fprintf(stderr, "Invalid parameters for depth rendering\n");
		return 0;
	}

	cl_int err;
	cl_event evt = NULL;

	// clear output buffer
	float zeroFloat = 0.0f;
	err = clEnqueueFillBuffer(ocl->queue, output_buffer, &zeroFloat, sizeof(float), 0,
							  screen_width * screen_height * sizeof(float), 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error clearing depth buffer: %s (%d)\n", clErrorString(err), err);
		return 0;
	}

	// Prepare camera parameters
	cl_float3 cl_cam_pos = {cam_pos[0], cam_pos[1], cam_pos[2]};
	cl_float3 cl_cam_dir = {cam_dir[0], cam_dir[1], cam_dir[2]};
	cl_int cl_screen_width = screen_width;
	cl_int cl_screen_height = screen_height;
	cl_float cl_fov = fov;
	cl_int cl_triangle_count = triangle_count;
	cl_int cl_missile_idx = missileIdx;

	// Set kernel arguments
	err = clSetKernelArg(ocl->renderDepth, 0, sizeof(cl_mem), &v1_buffer);
	err |= clSetKernelArg(ocl->renderDepth, 1, sizeof(cl_mem), &v2_buffer);
	err |= clSetKernelArg(ocl->renderDepth, 2, sizeof(cl_mem), &v3_buffer);
	err |= clSetKernelArg(ocl->renderDepth, 3, sizeof(cl_mem), &normals_buffer);
	err |= clSetKernelArg(ocl->renderDepth, 4, sizeof(cl_mem), &output_buffer);
	err |= clSetKernelArg(ocl->renderDepth, 5, sizeof(cl_int), &cl_triangle_count);
	err |= clSetKernelArg(ocl->renderDepth, 6, sizeof(cl_float3), &cl_cam_pos);
	err |= clSetKernelArg(ocl->renderDepth, 7, sizeof(cl_float3), &cl_cam_dir);
	err |= clSetKernelArg(ocl->renderDepth, 8, sizeof(cl_float), &cl_fov);
	err |= clSetKernelArg(ocl->renderDepth, 9, sizeof(cl_int), &cl_screen_width);
	err |= clSetKernelArg(ocl->renderDepth, 10, sizeof(cl_int), &cl_screen_height);
	err |= clSetKernelArg(ocl->renderDepth, 11, sizeof(cl_int), &cl_missile_idx);
	err |= clSetKernelArg(ocl->renderDepth, 12, sizeof(cl_mem), &ocl->buffer_seeker_distances_atlas);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error setting renderDepthBufferFast kernel args: %s (%d)\n", clErrorString(err), err);
		return 0;
	}

	// Execute kernel
	size_t global_size = triangle_count;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->renderDepth, 1, NULL,
								 &global_size, NULL, 0, NULL, &evt);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error enqueuing renderDepthBufferFast kernel: %s (%d)\n", clErrorString(err), err);
		return 0;
	}

	// Read back depth buffer to CPU if requested (using mapped memory for faster transfer)
	if (out_depth_cpu != NULL) {
		size_t buffer_size = screen_width * screen_height * sizeof(float);

		float *mapped_ptr = clEnqueueMapBuffer(ocl->queue, output_buffer, CL_TRUE,
											   CL_MAP_READ, 0, buffer_size, 1, &evt, NULL, &err);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error mapping depth buffer: %s (%d)\n", clErrorString(err), err);
			if (evt) clReleaseEvent(evt);
			return 0;
		}

		memcpy(out_depth_cpu, mapped_ptr, buffer_size);

		err = clEnqueueUnmapMemObject(ocl->queue, output_buffer, mapped_ptr, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Error unmapping depth buffer: %s (%d)\n", clErrorString(err), err);
			if (evt) clReleaseEvent(evt);
			return 0;
		}
	} else {
		clFinish(ocl->queue);
	}

	// Calculate GPU time if requested
	if (outGpuMs != NULL) {
		cl_ulong t0 = 0, t1 = 0;
		if (evt) {
			clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
			clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
			*outGpuMs = (t1 - t0) * 1e-6f;
		} else {
			*outGpuMs = 0.0f;
		}
	}

	if (evt) clReleaseEvent(evt);
	return 1;
}

void renderAllMissileFires(struct OpenCLContext *ocl, struct Missiles *missiles, struct Camera *camera, float *timeTookMs) {
	if (!missiles || missiles->count == 0) return;

	cl_int err;
	float zeroFloat = 0.0f;

	cl_int screenWidth = ScreenWidth;
	cl_int screenHeight = ScreenHeight;

	// Clear fire rendering buffers
	err = clEnqueueFillBuffer(ocl->queue, ocl->FireScreenDistances, &zeroFloat, sizeof(float), 0,
							  ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->FireScreenNormals, &zeroFloat, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * sizeof(float) * 3, 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->FireScreenAlphas, &zeroFloat, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);

	cl_float3 backgroundColor = {0.0f, 0.0f, 0.0f};

	err |= clSetKernelArg(ocl->clearColorBuffer_kernel, 0, sizeof(cl_mem), &ocl->FireScreenColors);
	err |= clSetKernelArg(ocl->clearColorBuffer_kernel, 1, sizeof(cl_float3), &backgroundColor);
	err |= clSetKernelArg(ocl->clearColorBuffer_kernel, 2, sizeof(cl_int), &screenWidth);
	err |= clSetKernelArg(ocl->clearColorBuffer_kernel, 3, sizeof(cl_int), &screenHeight);

	size_t global_size[2] = {(size_t)screenWidth, (size_t)screenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->clearColorBuffer_kernel, 2, NULL,
								 global_size, NULL, 0, NULL, NULL);
	clFinish(ocl->queue);

	float totalFireTime = 0.0f;
	float totalMissileRenderTime = 0.0f;

	// Render each missile's body and fire
	for (int i = 0; i < missiles->count; i++) {
		struct Missile *missile = missiles->missiles[i];
		if (!missile || !missiles->active[i]) continue;

		// === RENDER MISSILE BODY ===
		cl_float3 missile_pos = {missile->position[0], missile->position[1], missile->position[2]};
		cl_float3 missile_orient = {missile->bodyOrientation[0], missile->bodyOrientation[1], missile->bodyOrientation[2]};
		cl_float missile_scale = 1.0f;

		cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
		cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
		cl_float fov = camera->fov;
		cl_int model_triangle_count = missiles->missileModel->count;

		// Set missile rendering kernel arguments
		err = clSetKernelArg(ocl->renderMissile_kernel, 0, sizeof(cl_mem), &ocl->buffer_missile_v1);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 1, sizeof(cl_mem), &ocl->buffer_missile_v2);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 2, sizeof(cl_mem), &ocl->buffer_missile_v3);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 3, sizeof(cl_mem), &ocl->buffer_missile_normals);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 4, sizeof(cl_mem), &ocl->missile_color_buffer);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 5, sizeof(cl_mem), &ocl->missile_roughness_buffer);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 6, sizeof(cl_mem), &ocl->missile_metallic_buffer);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 7, sizeof(cl_mem), &ocl->missile_emission_buffer);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 8, sizeof(cl_int), &model_triangle_count);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 9, sizeof(cl_float3), &missile_pos);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 10, sizeof(cl_float3), &missile_orient);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 11, sizeof(cl_float), &missile_scale);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 12, sizeof(cl_float3), &cam_pos);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 13, sizeof(cl_float3), &cam_dir);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 14, sizeof(cl_float), &fov);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 15, sizeof(cl_int), &screenWidth);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 16, sizeof(cl_int), &screenHeight);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 20, sizeof(cl_mem), &ocl->buffer_screen_material_roughness);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 21, sizeof(cl_mem), &ocl->buffer_screen_material_metallic);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 22, sizeof(cl_mem), &ocl->buffer_screen_material_emission);

		// normal screen mode
		err |= clSetKernelArg(ocl->renderMissile_kernel, 17, sizeof(cl_mem), &ocl->FireScreenDistances);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 18, sizeof(cl_mem), &ocl->FireScreenColors);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 19, sizeof(cl_mem), &ocl->FireScreenNormals);
		err |= clSetKernelArg(ocl->renderMissile_kernel, 23, sizeof(cl_mem), &ocl->FireScreenAlphas);

		if (err != CL_SUCCESS) {
			printf("Error setting missile render kernel args for missile %d: %d\n", i, err);
			continue;
		}

		// Execute missile rendering kernel
		cl_event missile_render_event;
		size_t missile_work_size = model_triangle_count;
		err = clEnqueueNDRangeKernel(ocl->queue, ocl->renderMissile_kernel, 1, NULL,
									 &missile_work_size, NULL, 0, NULL, &missile_render_event);

		if (err != CL_SUCCESS) {
			printf("Error executing missile render kernel for missile %d: %d\n", i, err);
			continue;
		}

		clFinish(ocl->queue);

		// Get missile rendering timing
		cl_ulong missile_start, missile_end;
		clGetEventProfilingInfo(missile_render_event, CL_PROFILING_COMMAND_START,
								sizeof(missile_start), &missile_start, NULL);
		clGetEventProfilingInfo(missile_render_event, CL_PROFILING_COMMAND_END,
								sizeof(missile_end), &missile_end, NULL);
		totalMissileRenderTime += (missile_end - missile_start) * 1e-6f;
		clReleaseEvent(missile_render_event);

		// === RENDER MISSILE FIRE ===
		if (!missile->fireSim) continue;

		float fireRenderTime = 0.0f;

		// Upload particle data for this missile's fire
		err = clEnqueueWriteBuffer(ocl->queue, ocl->posX, CL_FALSE, 0,
								   NUM_FIRE_PARTICLES * sizeof(float),
								   missile->fireSim->x, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl->queue, ocl->posY, CL_FALSE, 0,
									NUM_FIRE_PARTICLES * sizeof(float),
									missile->fireSim->y, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl->queue, ocl->posZ, CL_FALSE, 0,
									NUM_FIRE_PARTICLES * sizeof(float),
									missile->fireSim->z, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl->queue, ocl->velX, CL_FALSE, 0,
									NUM_FIRE_PARTICLES * sizeof(float),
									missile->fireSim->xVelocity, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl->queue, ocl->velY, CL_FALSE, 0,
									NUM_FIRE_PARTICLES * sizeof(float),
									missile->fireSim->yVelocity, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl->queue, ocl->velZ, CL_FALSE, 0,
									NUM_FIRE_PARTICLES * sizeof(float),
									missile->fireSim->zVelocity, 0, NULL, NULL);
		err |= clEnqueueWriteBuffer(ocl->queue, ocl->lifeTime, CL_TRUE, 0,
									NUM_FIRE_PARTICLES * sizeof(float),
									missile->fireSim->lifeTime, 0, NULL, NULL);

		if (err != CL_SUCCESS) {
			printf("Error uploading fire data for missile %d: %d\n", i, err);
			continue;
		}

		// Set fire rendering parameters
		cl_float3 baseColor = {missile->fireSim->startingColor[0],
							   missile->fireSim->startingColor[1],
							   missile->fireSim->startingColor[2]};
		cl_float3 fireColor = {missile->fireSim->fireColor[0],
							   missile->fireSim->fireColor[1],
							   missile->fireSim->fireColor[2]};
		cl_float3 smokeColor = {missile->fireSim->smokeColor[0],
								missile->fireSim->smokeColor[1],
								missile->fireSim->smokeColor[2]};

		cl_float maxLifeTime = missile->fireSim->maxLifeTime;
		cl_float maxVelocity = missile->fireSim->maxVelocity;
		cl_float maxDepth = missile->fireSim->maxDistance;
		cl_float3 camUp = {0.0f, 1.0f, 0.0f};
		cl_int numPoints = NUM_FIRE_PARTICLES;
		cl_int particleRadius = (int)(missile->fireSim->particlesSize * 100.0f);

		// Set fire kernel arguments
		err = clSetKernelArg(ocl->fire_render_kernel, 0, sizeof(cl_mem), &ocl->posX);
		err |= clSetKernelArg(ocl->fire_render_kernel, 1, sizeof(cl_mem), &ocl->posY);
		err |= clSetKernelArg(ocl->fire_render_kernel, 2, sizeof(cl_mem), &ocl->posZ);
		err |= clSetKernelArg(ocl->fire_render_kernel, 3, sizeof(cl_mem), &ocl->velX);
		err |= clSetKernelArg(ocl->fire_render_kernel, 4, sizeof(cl_mem), &ocl->velY);
		err |= clSetKernelArg(ocl->fire_render_kernel, 5, sizeof(cl_mem), &ocl->velZ);
		err |= clSetKernelArg(ocl->fire_render_kernel, 6, sizeof(cl_mem), &ocl->lifeTime);
		err |= clSetKernelArg(ocl->fire_render_kernel, 7, sizeof(cl_float3), &baseColor);
		err |= clSetKernelArg(ocl->fire_render_kernel, 8, sizeof(cl_float3), &fireColor);
		err |= clSetKernelArg(ocl->fire_render_kernel, 9, sizeof(cl_float3), &smokeColor);
		err |= clSetKernelArg(ocl->fire_render_kernel, 10, sizeof(cl_float), &maxLifeTime);
		err |= clSetKernelArg(ocl->fire_render_kernel, 11, sizeof(cl_float), &maxVelocity);
		err |= clSetKernelArg(ocl->fire_render_kernel, 12, sizeof(cl_float), &maxDepth);
		err |= clSetKernelArg(ocl->fire_render_kernel, 13, sizeof(cl_float3), &cam_pos);
		err |= clSetKernelArg(ocl->fire_render_kernel, 14, sizeof(cl_float3), &cam_dir);
		err |= clSetKernelArg(ocl->fire_render_kernel, 15, sizeof(cl_float3), &camUp);
		err |= clSetKernelArg(ocl->fire_render_kernel, 16, sizeof(cl_float), &fov);
		err |= clSetKernelArg(ocl->fire_render_kernel, 17, sizeof(cl_int), &screenWidth);
		err |= clSetKernelArg(ocl->fire_render_kernel, 18, sizeof(cl_int), &screenHeight);
		err |= clSetKernelArg(ocl->fire_render_kernel, 23, sizeof(cl_int), &numPoints);
		err |= clSetKernelArg(ocl->fire_render_kernel, 24, sizeof(cl_int), &particleRadius);

		// normal screen mode
		err |= clSetKernelArg(ocl->fire_render_kernel, 19, sizeof(cl_mem), &ocl->FireScreenDistances);
		err |= clSetKernelArg(ocl->fire_render_kernel, 20, sizeof(cl_mem), &ocl->FireScreenColors);
		err |= clSetKernelArg(ocl->fire_render_kernel, 21, sizeof(cl_mem), &ocl->FireScreenNormals);
		err |= clSetKernelArg(ocl->fire_render_kernel, 22, sizeof(cl_mem), &ocl->FireScreenAlphas);

		if (err != CL_SUCCESS) {
			printf("Error setting fire kernel args for missile %d: %d\n", i, err);
			continue;
		}

		// Execute fire rendering kernel
		cl_event fire_kernel_event;
		size_t work_size = NUM_FIRE_PARTICLES;
		err = clEnqueueNDRangeKernel(ocl->queue, ocl->fire_render_kernel, 1, NULL,
									 &work_size, NULL, 0, NULL, &fire_kernel_event);

		if (err != CL_SUCCESS) {
			printf("Error executing fire kernel for missile %d: %d\n", i, err);
			continue;
		}

		clFinish(ocl->queue);

		// Get fire timing
		cl_ulong fire_start, fire_end;
		clGetEventProfilingInfo(fire_kernel_event, CL_PROFILING_COMMAND_START,
								sizeof(fire_start), &fire_start, NULL);
		clGetEventProfilingInfo(fire_kernel_event, CL_PROFILING_COMMAND_END,
								sizeof(fire_end), &fire_end, NULL);
		fireRenderTime = (fire_end - fire_start) * 1e-6f;
		totalFireTime += fireRenderTime;

		clReleaseEvent(fire_kernel_event);
	}

	*timeTookMs = totalFireTime + totalMissileRenderTime;

	// Apply blur to all fires at once
	if (ocl->blur_fire_kernel != NULL) {
		const int NUM_BLUR_PASSES = 2;
		cl_int blurRadius = 3;
		cl_float sigmaColor = 0.8f;
		cl_float sigmaSpace = 2.5f;

		cl_mem srcColors = ocl->FireScreenColors;
		cl_mem srcDistances = ocl->FireScreenDistances;
		cl_mem srcAlphas = ocl->FireScreenAlphas;
		cl_mem dstColors = ocl->FireScreenColorsTemp;
		cl_mem dstDistances = ocl->FireScreenDistancesTemp;
		cl_mem dstAlphas = ocl->FireScreenAlphasTemp;

		for (int pass = 0; pass < NUM_BLUR_PASSES; pass++) {
			err = clSetKernelArg(ocl->blur_fire_kernel, 0, sizeof(cl_mem), &srcColors);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 1, sizeof(cl_mem), &srcDistances);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 2, sizeof(cl_mem), &srcAlphas);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 3, sizeof(cl_mem), &dstColors);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 4, sizeof(cl_mem), &dstDistances);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 5, sizeof(cl_mem), &dstAlphas);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 6, sizeof(cl_int), &screenWidth);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 7, sizeof(cl_int), &screenHeight);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 8, sizeof(cl_int), &blurRadius);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 9, sizeof(cl_float), &sigmaColor);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 10, sizeof(cl_float), &sigmaSpace);

			size_t blur_global_size[2] = {screenWidth, screenHeight};
			err = clEnqueueNDRangeKernel(ocl->queue, ocl->blur_fire_kernel, 2, NULL,
										 blur_global_size, NULL, 0, NULL, NULL);
			clFinish(ocl->queue);

			cl_mem tempColors = srcColors;
			cl_mem tempDistances = srcDistances;
			cl_mem tempAlphas = srcAlphas;
			srcColors = dstColors;
			srcDistances = dstDistances;
			srcAlphas = dstAlphas;
			dstColors = tempColors;
			dstDistances = tempDistances;
			dstAlphas = tempAlphas;
		}

		if (NUM_BLUR_PASSES % 2 == 1) {
			err = clEnqueueCopyBuffer(ocl->queue, ocl->FireScreenColorsTemp, ocl->FireScreenColors,
									  0, 0, screenWidth * screenHeight * 3 * sizeof(float),
									  0, NULL, NULL);
			err = clEnqueueCopyBuffer(ocl->queue, ocl->FireScreenDistancesTemp, ocl->FireScreenDistances,
									  0, 0, screenWidth * screenHeight * sizeof(float),
									  0, NULL, NULL);
			err = clEnqueueCopyBuffer(ocl->queue, ocl->FireScreenAlphasTemp, ocl->FireScreenAlphas,
									  0, 0, screenWidth * screenHeight * sizeof(float),
									  0, NULL, NULL);
			clFinish(ocl->queue);
		}
	}
}

float delay = 0.2f;
float timeSinceLastFire = 0.0f;
float firedMissileTime = 0.0f;
int firedMissileIdx = -1;

void missilesSimulation(struct OpenCLContext *ocl, struct Missiles *missiles, float *timeTookMs, bool *fire, struct Camera *camera, float deltaTime, struct Triangles *triangles, struct IRSearchAndTrack *irst) {
	float totalTimeTookMs = 0.0f;
	bool hasFired = false;

	timeSinceLastFire += deltaTime;
	bool canFire = (timeSinceLastFire >= delay);

	int missileCount = 0;

	for (int i = 0; i < missiles->count; i++) {
		struct Missile *missile = missiles->missiles[i];
		bool active = missiles->active[i];
		float tempTimeTookMs = 0.0f;
		bool shouldFireThisMissile = false;

		if (*fire && !active && !hasFired && canFire) {
			float spawnOffset = 25.0f;

			missile->position[0] = camera->ray.origin[0] + camera->ray.direction[0] * spawnOffset;
			missile->position[1] = camera->ray.origin[1] + camera->ray.direction[1] * spawnOffset;
			missile->position[2] = camera->ray.origin[2] + camera->ray.direction[2] * spawnOffset;

			missile->bodyOrientation[0] = camera->ray.direction[0];
			missile->bodyOrientation[1] = camera->ray.direction[1];
			missile->bodyOrientation[2] = camera->ray.direction[2];

			missile->velocity[0] = camera->ray.direction[0] * 17.5f;
			missile->velocity[1] = camera->ray.direction[1] * 17.5f;
			missile->velocity[2] = camera->ray.direction[2] * 17.5f;

			missile->targetDirection[0] = camera->ray.direction[0];
			missile->targetDirection[1] = camera->ray.direction[1];
			missile->targetDirection[2] = camera->ray.direction[2];

			float virtualTargetDist = 10.0f;
			if (irst->selectedTargetId >= 0) {
				missile->targetPosition[0] = irst->targetPositionX[irst->selectedTargetId];
				missile->targetPosition[1] = irst->targetPositionY[irst->selectedTargetId];
				missile->targetPosition[2] = irst->targetPositionZ[irst->selectedTargetId];
				irst->selectedTargetId = -1;

			} else {
				missile->targetPosition[0] = camera->ray.origin[0] + camera->ray.direction[0] * virtualTargetDist;
				missile->targetPosition[1] = camera->ray.origin[1] + camera->ray.direction[1] * virtualTargetDist;
				missile->targetPosition[2] = camera->ray.origin[2] + camera->ray.direction[2] * virtualTargetDist;
			}

			missile->prevLOS[0] = missile->targetDirection[0];
			missile->prevLOS[1] = missile->targetDirection[1];
			missile->prevLOS[2] = missile->targetDirection[2];

			missile->seeker.lockState = Lunching;
			missile->remainingTime = randRange(45.0f, 120.0f);

			missile->burning = 1;
			missile->fuelMass = missile->initialFuelMass;
			missile->totalMass = missile->dryMass + missile->fuelMass;

			shouldFireThisMissile = true;
			hasFired = true;
			timeSinceLastFire = 0.0f;
			firedMissileTime = 1.5f;
			firedMissileIdx = i;
			*fire = false;
		}

		float dx = missile->position[0] - camera->ray.origin[0];
		float dy = missile->position[1] - camera->ray.origin[1];
		float dz = missile->position[2] - camera->ray.origin[2];
		float distanceToCamera = sqrtf(dx * dx + dy * dy + dz * dz);

		if (active && distanceToCamera > 10.0f) {
			float seekerFovDegrees = missile->seeker.seekerCamera.fov;
			if (missile->seeker.lockState == Searching) {
				seekerFovDegrees = seekerFovDegrees * missile->seeker.searchMultiplayer;
			}

			// Convert degrees to radians and calculate tan(fov/2)
			// This matches the format used by the main camera (fov = tan(fov_angle/2))
			float fovRadians = seekerFovDegrees * (M_PI / 180.0f);
			float FOV = tanf(fovRadians / 2.0f);

			const float wideRadians = 270.f * (M_PI / 180.0f);
			const float wideFOV = tanf(wideRadians / 2.0f);

			float tempMS = 0.0f;

			renderDepthBuffer(ocl,
							  ocl->buffer_triangle_v1,
							  ocl->buffer_triangle_v2,
							  ocl->buffer_triangle_v3,
							  ocl->buffer_triangle_normals,
							  ocl->buffer_seeker_distances,
							  triangles->count,
							  missile->seeker.seekerCamera.ray.origin,
							  missile->seeker.seekerCamera.ray.direction,
							  FOV,
							  MISSILE_SEEKER_SIZE,
							  MISSILE_SEEKER_SIZE,
							  missile->seeker.seekerDepthMap,
							  &tempMS,
							  missileCount);

			float tempMS2 = 0.0f;

			renderDepthBuffer(ocl,
							  ocl->buffer_triangle_v1,
							  ocl->buffer_triangle_v2,
							  ocl->buffer_triangle_v3,
							  ocl->buffer_triangle_normals,
							  ocl->buffer_seeker_distances,
							  triangles->count,
							  missile->seeker.seekerCamera.ray.origin,
							  missile->bodyOrientation,
							  wideFOV,
							  MISSILE_SEEKER_SIZE,
							  MISSILE_SEEKER_SIZE,
							  missile->seeker.seekerDepthWideView,
							  &tempMS2,
							  missileCount);

			totalTimeTookMs += tempMS + tempMS2;

			missileCount++;
		}

		float tmp1 = 0.0f;
		float tmp2 = 0.0f;
		missileSeekStep(
			missile,
			missiles,
			shouldFireThisMissile,
			&missiles->active[i],
			deltaTime,
			&tmp1,
			&tmp2,
			camera->ray.direction,
			camera->ray.origin);
		totalTimeTookMs += tmp1 + tmp2;
	}
	*timeTookMs = totalTimeTookMs;
}

void compositeBuffersOpenCL(
	struct OpenCLContext *ocl,
	cl_mem inputColors1,
	cl_mem inputDistances1,
	cl_mem inputNormals1,
	cl_mem inputAlphas1,
	int useAlpha1,
	cl_mem inputColors2,
	cl_mem inputDistances2,
	cl_mem inputNormals2,
	cl_mem inputAlphas2,
	int useAlpha2,
	cl_mem outputColors,
	cl_mem outputDistances,
	cl_mem outputNormals,
	float *gpuTimeMs) {
	cl_int err;
	cl_event kernel_event;

	cl_int screenWidth = ScreenWidth;
	cl_int screenHeight = ScreenHeight;

	// Set all kernel arguments including alpha buffers
	err = clSetKernelArg(ocl->composite_kernel, 0, sizeof(cl_mem), &inputColors1);
	err |= clSetKernelArg(ocl->composite_kernel, 1, sizeof(cl_mem), &inputDistances1);
	err |= clSetKernelArg(ocl->composite_kernel, 2, sizeof(cl_mem), &inputNormals1);
	err |= clSetKernelArg(ocl->composite_kernel, 3, sizeof(cl_mem), &inputAlphas1);
	err |= clSetKernelArg(ocl->composite_kernel, 4, sizeof(cl_int), &useAlpha1);
	err |= clSetKernelArg(ocl->composite_kernel, 5, sizeof(cl_mem), &inputColors2);
	err |= clSetKernelArg(ocl->composite_kernel, 6, sizeof(cl_mem), &inputDistances2);
	err |= clSetKernelArg(ocl->composite_kernel, 7, sizeof(cl_mem), &inputNormals2);
	err |= clSetKernelArg(ocl->composite_kernel, 8, sizeof(cl_mem), &inputAlphas2);
	err |= clSetKernelArg(ocl->composite_kernel, 9, sizeof(cl_int), &useAlpha2);
	err |= clSetKernelArg(ocl->composite_kernel, 10, sizeof(cl_mem), &outputColors);
	err |= clSetKernelArg(ocl->composite_kernel, 11, sizeof(cl_mem), &outputDistances);
	err |= clSetKernelArg(ocl->composite_kernel, 12, sizeof(cl_mem), &outputNormals);
	err |= clSetKernelArg(ocl->composite_kernel, 13, sizeof(cl_int), &screenWidth);
	err |= clSetKernelArg(ocl->composite_kernel, 14, sizeof(cl_int), &screenHeight);

	if (err != CL_SUCCESS) {
		printf("Error setting composite kernel arguments: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->composite_kernel, 2, NULL,
								 global_work_size, NULL, 0, NULL, &kernel_event);

	if (err != CL_SUCCESS) {
		printf("Error executing composite kernel: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	clFinish(ocl->queue);

	if (gpuTimeMs != NULL) {
		cl_ulong start_time, end_time;
		err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
									  sizeof(start_time), &start_time, NULL);
		err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
									   sizeof(end_time), &end_time, NULL);

		if (err == CL_SUCCESS) {
			*gpuTimeMs = (end_time - start_time) * 1e-6f;
		} else {
			*gpuTimeMs = 0.0f;
		}
	}

	clReleaseEvent(kernel_event);
}

// TODO REWORK THIS TO MAKE IT MODULAR SO I CAN RENDER FIRE PARTICLES FROW WHAT EVER CAMERA
void renderFireParticles(struct OpenCLContext *ocl, struct FireSOA *fireParticles, struct Camera *camera, float *gpuTimeMs) {
	cl_int err;
	cl_event kernel_event, blur_event;

	err = clEnqueueWriteBuffer(ocl->queue, ocl->posX, CL_FALSE, 0,
							   NUM_FIRE_PARTICLES * sizeof(float), fireParticles->x, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->posY, CL_FALSE, 0,
								NUM_FIRE_PARTICLES * sizeof(float), fireParticles->y, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->posZ, CL_FALSE, 0,
								NUM_FIRE_PARTICLES * sizeof(float), fireParticles->z, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->velX, CL_FALSE, 0,
								NUM_FIRE_PARTICLES * sizeof(float), fireParticles->xVelocity, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->velY, CL_FALSE, 0,
								NUM_FIRE_PARTICLES * sizeof(float), fireParticles->yVelocity, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->velZ, CL_FALSE, 0,
								NUM_FIRE_PARTICLES * sizeof(float), fireParticles->zVelocity, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->lifeTime, CL_TRUE, 0,
								NUM_FIRE_PARTICLES * sizeof(float), fireParticles->lifeTime, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error uploading fire particle data: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	cl_int screenWidth = ScreenWidth;
	cl_int screenHeight = ScreenHeight;

	if (err != CL_SUCCESS) {
		printf("Error clearing fire buffers: %d\n", err);
	}

	if (err != CL_SUCCESS) {
		printf("Error clearing fire buffers: %d\n", err);
	}

	cl_float3 baseColor = {fireParticles->startingColor[0],
						   fireParticles->startingColor[1],
						   fireParticles->startingColor[2]};
	cl_float3 fireColor = {fireParticles->fireColor[0],
						   fireParticles->fireColor[1],
						   fireParticles->fireColor[2]};
	cl_float3 smokeColor = {fireParticles->smokeColor[0],
							fireParticles->smokeColor[1],
							fireParticles->smokeColor[2]};

	cl_float maxLifeTime = fireParticles->maxLifeTime;
	cl_float maxVelocity = fireParticles->maxVelocity;
	cl_float maxDepth = fireParticles->maxDistance;

	cl_float3 camPos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 camDir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float3 camUp = {0.0f, 1.0f, 0.0f};
	cl_float fov = camera->fov;

	cl_int numPoints = NUM_FIRE_PARTICLES;
	cl_int particleRadius = (int)(fireParticles->particlesSize * 100.0f);

	err = clSetKernelArg(ocl->fire_render_kernel, 0, sizeof(cl_mem), &ocl->posX);
	err |= clSetKernelArg(ocl->fire_render_kernel, 1, sizeof(cl_mem), &ocl->posY);
	err |= clSetKernelArg(ocl->fire_render_kernel, 2, sizeof(cl_mem), &ocl->posZ);
	err |= clSetKernelArg(ocl->fire_render_kernel, 3, sizeof(cl_mem), &ocl->velX);
	err |= clSetKernelArg(ocl->fire_render_kernel, 4, sizeof(cl_mem), &ocl->velY);
	err |= clSetKernelArg(ocl->fire_render_kernel, 5, sizeof(cl_mem), &ocl->velZ);
	err |= clSetKernelArg(ocl->fire_render_kernel, 6, sizeof(cl_mem), &ocl->lifeTime);
	err |= clSetKernelArg(ocl->fire_render_kernel, 7, sizeof(cl_float3), &baseColor);
	err |= clSetKernelArg(ocl->fire_render_kernel, 8, sizeof(cl_float3), &fireColor);
	err |= clSetKernelArg(ocl->fire_render_kernel, 9, sizeof(cl_float3), &smokeColor);
	err |= clSetKernelArg(ocl->fire_render_kernel, 10, sizeof(cl_float), &maxLifeTime);
	err |= clSetKernelArg(ocl->fire_render_kernel, 11, sizeof(cl_float), &maxVelocity);
	err |= clSetKernelArg(ocl->fire_render_kernel, 12, sizeof(cl_float), &maxDepth);
	err |= clSetKernelArg(ocl->fire_render_kernel, 13, sizeof(cl_float3), &camPos);
	err |= clSetKernelArg(ocl->fire_render_kernel, 14, sizeof(cl_float3), &camDir);
	err |= clSetKernelArg(ocl->fire_render_kernel, 15, sizeof(cl_float3), &camUp);
	err |= clSetKernelArg(ocl->fire_render_kernel, 16, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->fire_render_kernel, 17, sizeof(cl_int), &screenWidth);
	err |= clSetKernelArg(ocl->fire_render_kernel, 18, sizeof(cl_int), &screenHeight);
	err |= clSetKernelArg(ocl->fire_render_kernel, 19, sizeof(cl_mem), &ocl->FireScreenDistances);
	err |= clSetKernelArg(ocl->fire_render_kernel, 20, sizeof(cl_mem), &ocl->FireScreenColors);
	err |= clSetKernelArg(ocl->fire_render_kernel, 21, sizeof(cl_mem), &ocl->FireScreenNormals);
	err |= clSetKernelArg(ocl->fire_render_kernel, 22, sizeof(cl_mem), &ocl->FireScreenAlphas);
	err |= clSetKernelArg(ocl->fire_render_kernel, 23, sizeof(cl_int), &numPoints);
	err |= clSetKernelArg(ocl->fire_render_kernel, 24, sizeof(cl_int), &particleRadius);

	if (err != CL_SUCCESS) {
		printf("Error setting fire render kernel arguments: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	size_t global_work_size = NUM_FIRE_PARTICLES;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->fire_render_kernel, 1, NULL,
								 &global_work_size, NULL, 0, NULL, &kernel_event);

	if (err != CL_SUCCESS) {
		printf("Error executing fire render kernel: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	clFinish(ocl->queue);

	if (ocl->blur_fire_kernel != NULL) {
		const int NUM_BLUR_PASSES = 3;
		cl_int blurRadius = 3;
		cl_float sigmaColor = 0.8f;
		cl_float sigmaSpace = 2.5f;

		cl_mem srcColors = ocl->FireScreenColors;
		cl_mem srcDistances = ocl->FireScreenDistances;
		cl_mem srcAlphas = ocl->FireScreenAlphas;
		cl_mem dstColors = ocl->FireScreenColorsTemp;
		cl_mem dstDistances = ocl->FireScreenDistancesTemp;
		cl_mem dstAlphas = ocl->FireScreenAlphasTemp;

		for (int pass = 0; pass < NUM_BLUR_PASSES; pass++) {
			err = clSetKernelArg(ocl->blur_fire_kernel, 0, sizeof(cl_mem), &srcColors);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 1, sizeof(cl_mem), &srcDistances);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 2, sizeof(cl_mem), &srcAlphas);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 3, sizeof(cl_mem), &dstColors);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 4, sizeof(cl_mem), &dstDistances);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 5, sizeof(cl_mem), &dstAlphas);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 6, sizeof(cl_int), &screenWidth);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 7, sizeof(cl_int), &screenHeight);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 8, sizeof(cl_int), &blurRadius);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 9, sizeof(cl_float), &sigmaColor);
			err |= clSetKernelArg(ocl->blur_fire_kernel, 10, sizeof(cl_float), &sigmaSpace);

			if (err != CL_SUCCESS) {
				printf("Error setting blur kernel arguments (pass %d): %d\n", pass, err);
				break;
			}

			size_t blur_global_work_size[2] = {screenWidth, screenHeight};
			err = clEnqueueNDRangeKernel(ocl->queue, ocl->blur_fire_kernel, 2, NULL,
										 blur_global_work_size, NULL, 0, NULL, &blur_event);

			if (err != CL_SUCCESS) {
				printf("Error executing blur kernel (pass %d): %d\n", pass, err);
				break;
			}

			clFinish(ocl->queue);
			clReleaseEvent(blur_event);

			// Swap buffers for next pass
			cl_mem tempColors = srcColors;
			cl_mem tempDistances = srcDistances;
			cl_mem tempAlphas = srcAlphas;
			srcColors = dstColors;
			srcDistances = dstDistances;
			srcAlphas = dstAlphas;
			dstColors = tempColors;
			dstDistances = tempDistances;
			dstAlphas = tempAlphas;
		}

		// Copy back if odd number of passes
		if (NUM_BLUR_PASSES % 2 == 1) {
			err = clEnqueueCopyBuffer(ocl->queue, ocl->FireScreenColorsTemp, ocl->FireScreenColors,
									  0, 0, screenWidth * screenHeight * 3 * sizeof(float),
									  0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error copying final blurred colors: %d\n", err);
			}

			err = clEnqueueCopyBuffer(ocl->queue, ocl->FireScreenDistancesTemp, ocl->FireScreenDistances,
									  0, 0, screenWidth * screenHeight * sizeof(float),
									  0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error copying final blurred distances: %d\n", err);
			}

			err = clEnqueueCopyBuffer(ocl->queue, ocl->FireScreenAlphasTemp, ocl->FireScreenAlphas,
									  0, 0, screenWidth * screenHeight * sizeof(float),
									  0, NULL, NULL);
			if (err != CL_SUCCESS) {
				printf("Error copying final blurred alphas: %d\n", err);
			}

			clFinish(ocl->queue);
		}
	} else {
		printf("Blur kernel not available, skipping blur step\n");
		clFinish(ocl->queue);
	}

	if (gpuTimeMs != NULL) {
		cl_ulong start_time, end_time;
		err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
									  sizeof(start_time), &start_time, NULL);
		err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
									   sizeof(end_time), &end_time, NULL);

		if (err == CL_SUCCESS) {
			*gpuTimeMs = (end_time - start_time) * 1e-6f;
		} else {
			printf("Error getting fire render kernel profiling info: %d\n", err);
			*gpuTimeMs = 0.0f;
		}
	}

	clReleaseEvent(kernel_event);
}

void AddTriangle(struct Triangles *triangles,
				 float v1x, float v1y, float v1z,
				 float v2x, float v2y, float v2z,
				 float v3x, float v3y, float v3z,
				 float colorR, float colorG, float colorB, float Roughness, float Metallic, float Emission) {
	if (triangles->count >= NUMBER_OF_TRIANGLES) {
		printf("Maximum number of triangles reached\n");
		return;
	}

	int index = triangles->count * 3;
	triangles->v1[index] = v1x;
	triangles->v1[index + 1] = v1y;
	triangles->v1[index + 2] = v1z;

	triangles->v2[index] = v2x;
	triangles->v2[index + 1] = v2y;
	triangles->v2[index + 2] = v2z;

	triangles->v3[index] = v3x;
	triangles->v3[index + 1] = v3y;
	triangles->v3[index + 2] = v3z;

	// Calculate normal
	float ux = v2x - v1x;
	float uy = v2y - v1y;
	float uz = v2z - v1z;

	float vx = v3x - v1x;
	float vy = v3y - v1y;
	float vz = v3z - v1z;

	// Cross product for normal
	float nx = uy * vz - uz * vy;
	float ny = uz * vx - ux * vz;
	float nz = ux * vy - uy * vx;

	// Normalize the normal vector
	float length = sqrtf(nx * nx + ny * ny + nz * nz);
	if (length > 0.0f) {
		triangles->normals[index] = nx / length;
		triangles->normals[index + 1] = ny / length;
		triangles->normals[index + 2] = nz / length;
	} else {
		// Fallback for degenerate triangles
		triangles->normals[index] = 0.0f;
		triangles->normals[index + 1] = 1.0f;
		triangles->normals[index + 2] = 0.0f;
	}

	// Store color
	triangles->colors[index] = colorR;
	triangles->colors[index + 1] = colorG;
	triangles->colors[index + 2] = colorB;
	// Store material properties
	triangles->Roughness[triangles->count] = Roughness;
	triangles->Metallic[triangles->count] = Metallic;
	triangles->Emission[triangles->count] = Emission;

	triangles->count++;
}

int initializeSkyboxBuffers(struct OpenCLContext *ocl, struct SkyBox *skyBox) {
	cl_int err;

	// Convert images to float arrays
	float *top_data = convertImageToFloat(skyBox->top);
	float *bottom_data = convertImageToFloat(skyBox->bottom);
	float *left_data = convertImageToFloat(skyBox->left);
	float *right_data = convertImageToFloat(skyBox->right);
	float *front_data = convertImageToFloat(skyBox->front);
	float *back_data = convertImageToFloat(skyBox->back);

	if (!top_data || !bottom_data || !left_data || !right_data || !front_data || !back_data) {
		printf("Failed to convert skybox images to float arrays\n");
		return 0;
	}

	size_t image_size = skyBox->top->width * skyBox->top->height * 3 * sizeof(float);

	// Create and upload skybox buffers
	ocl->buffer_skybox_top = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											image_size, top_data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox top buffer: %d\n", err);
		free(top_data);
		free(bottom_data);
		free(left_data);
		free(right_data);
		free(front_data);
		free(back_data);
		return 0;
	}

	ocl->buffer_skybox_bottom = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											   image_size, bottom_data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox bottom buffer: %d\n", err);
		free(top_data);
		free(bottom_data);
		free(left_data);
		free(right_data);
		free(front_data);
		free(back_data);
		return 0;
	}

	ocl->buffer_skybox_left = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											 image_size, left_data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox left buffer: %d\n", err);
		free(top_data);
		free(bottom_data);
		free(left_data);
		free(right_data);
		free(front_data);
		free(back_data);
		return 0;
	}

	ocl->buffer_skybox_right = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											  image_size, right_data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox right buffer: %d\n", err);
		free(top_data);
		free(bottom_data);
		free(left_data);
		free(right_data);
		free(front_data);
		free(back_data);
		return 0;
	}

	ocl->buffer_skybox_front = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											  image_size, front_data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox front buffer: %d\n", err);
		free(top_data);
		free(bottom_data);
		free(left_data);
		free(right_data);
		free(front_data);
		free(back_data);
		return 0;
	}

	ocl->buffer_skybox_back = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
											 image_size, back_data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox back buffer: %d\n", err);
		free(top_data);
		free(bottom_data);
		free(left_data);
		free(right_data);
		free(front_data);
		free(back_data);
		return 0;
	}

	// Free temporary arrays
	free(top_data);
	free(bottom_data);
	free(left_data);
	free(right_data);
	free(front_data);
	free(back_data);

	printf("Skybox buffers initialized successfully\n");
	return 1;
}

void CreateBoardPlane(float centerX, float centerY, float centerZ, float size, int numberOfSquares, struct Triangles *triangles) {
	// Define two alternating colors for checkerboard pattern
	float color1R = 0.995f, color1G = 0.98f, color1B = 0.92f;		 // Light color (white-ish)
	float color2R = 0.05f, color2G = 0.01f, color2B = 0.01f;		 // Dark color (black-ish)
	float Metallic = 0.01f, Roughness = 0.955f, Emission = 0.0f;	 // Material properties
	float Metallic1 = 0.995f, Roughness1 = 0.02f, Emission1 = 0.02f; // Material properties

	for (int i = 0; i < numberOfSquares; i++) {
		for (int j = 0; j < numberOfSquares; j++) {
			// Calculate square position (centered around centerX, centerZ)
			float x1 = centerX + (i - numberOfSquares / 2.0f) * size;
			float y1 = centerY; // Keep Y constant for horizontal plane
			float z1 = centerZ + (j - numberOfSquares / 2.0f) * size;

			float x2 = x1 + size;
			float y2 = y1;
			float z2 = z1;

			float x3 = x1;
			float y3 = y1;
			float z3 = z1 + size;

			float x4 = x2;
			float y4 = y2;
			float z4 = z3;

			// Create checkerboard pattern
			bool isEvenSquare = ((i + j) % 2) == 0;

			float colorR, colorG, colorB, roughness, metallic, emission;
			if (isEvenSquare) {
				colorR = color1R;
				colorG = color1G;
				colorB = color1B;
				roughness = Roughness;
				metallic = Metallic;
				emission = Emission;
			} else {
				colorR = color2R;
				colorG = color2G;
				colorB = color2B;
				roughness = Roughness1;
				metallic = Metallic1;
				emission = Emission1;
			}

			// FIXED: Correct winding order for upward-facing normals (counter-clockwise from above)
			// Triangle 1: bottom-left, top-left, bottom-right (when viewed from above)
			AddTriangle(triangles, x1, y1, z1, // bottom-left
						x3, y3, z3,			   // top-left
						x2, y2, z2,			   // bottom-right
						colorR, colorG, colorB, roughness, metallic, emission);

			// Triangle 2: top-left, top-right, bottom-right (when viewed from above)
			AddTriangle(triangles, x3, y3, z3, // top-left
						x4, y4, z4,			   // top-right
						x2, y2, z2,			   // bottom-right
						colorR, colorG, colorB, roughness, metallic, emission);
		}
	}
}

void CreateCube(float centerX, float centerY, float centerZ, float size, struct Triangles *triangles, float colorR, float colorG, float colorB, float Roughness, float Metallic, float Emission) {
	float halfSize = size / 2.0f;

	// Define vertices of the cube
	float v1[3] = {centerX - halfSize, centerY - halfSize, centerZ - halfSize}; // min, min, min
	float v2[3] = {centerX + halfSize, centerY - halfSize, centerZ - halfSize}; // max, min, min
	float v3[3] = {centerX + halfSize, centerY + halfSize, centerZ - halfSize}; // max, max, min
	float v4[3] = {centerX - halfSize, centerY + halfSize, centerZ - halfSize}; // min, max, min
	float v5[3] = {centerX - halfSize, centerY - halfSize, centerZ + halfSize}; // min, min, max
	float v6[3] = {centerX + halfSize, centerY - halfSize, centerZ + halfSize}; // max, min, max
	float v7[3] = {centerX + halfSize, centerY + halfSize, centerZ + halfSize}; // max, max, max
	float v8[3] = {centerX - halfSize, centerY + halfSize, centerZ + halfSize}; // min, max, max

	// Front face (z = min) - normal pointing towards -Z
	AddTriangle(triangles, v1[0], v1[1], v1[2], v4[0], v4[1], v4[2], v2[0], v2[1], v2[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
	AddTriangle(triangles, v4[0], v4[1], v4[2], v3[0], v3[1], v3[2], v2[0], v2[1], v2[2], colorR, colorG, colorB, Roughness, Metallic, Emission);

	// Back face (z = max) - normal pointing towards +Z
	AddTriangle(triangles, v6[0], v6[1], v6[2], v7[0], v7[1], v7[2], v5[0], v5[1], v5[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
	AddTriangle(triangles, v7[0], v7[1], v7[2], v8[0], v8[1], v8[2], v5[0], v5[1], v5[2], colorR, colorG, colorB, Roughness, Metallic, Emission);

	// Left face (x = min) - normal pointing towards -X
	AddTriangle(triangles, v5[0], v5[1], v5[2], v8[0], v8[1], v8[2], v1[0], v1[1], v1[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
	AddTriangle(triangles, v8[0], v8[1], v8[2], v4[0], v4[1], v4[2], v1[0], v1[1], v1[2], colorR, colorG, colorB, Roughness, Metallic, Emission);

	// Right face (x = max) - normal pointing towards +X
	AddTriangle(triangles, v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], v6[0], v6[1], v6[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
	AddTriangle(triangles, v3[0], v3[1], v3[2], v7[0], v7[1], v7[2], v6[0], v6[1], v6[2], colorR, colorG, colorB, Roughness, Metallic, Emission);

	// Top face (y = max) - normal pointing towards +Y
	AddTriangle(triangles, v4[0], v4[1], v4[2], v8[0], v8[1], v8[2], v3[0], v3[1], v3[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
	AddTriangle(triangles, v8[0], v8[1], v8[2], v7[0], v7[1], v7[2], v3[0], v3[1], v3[2], colorR, colorG, colorB, Roughness, Metallic, Emission);

	// Bottom face (y = min) - normal pointing towards -Y
	AddTriangle(triangles, v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v5[0], v5[1], v5[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
	AddTriangle(triangles, v2[0], v2[1], v2[2], v6[0], v6[1], v6[2], v5[0], v5[1], v5[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
}

void renderSkyboxOpenCL(struct OpenCLContext *ocl, struct Camera *camera, struct SkyBox *skyBox, float *gpuTimeMs) {
	cl_int err;
	cl_event kernel_event; // Add this line

	// Set skybox kernel arguments
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int skybox_width = skyBox->top->width;
	cl_int skybox_height = skyBox->top->height;

	err = clSetKernelArg(ocl->skybox_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
	CHECK_CL(err, "clSetKernelArg skybox 0");
	err |= clSetKernelArg(ocl->skybox_kernel, 1, sizeof(cl_float3), &cam_pos);
	CHECK_CL(err, "clSetKernelArg skybox 1");
	err |= clSetKernelArg(ocl->skybox_kernel, 2, sizeof(cl_float3), &cam_dir);
	CHECK_CL(err, "clSetKernelArg skybox 2");
	err |= clSetKernelArg(ocl->skybox_kernel, 3, sizeof(cl_float), &fov);
	CHECK_CL(err, "clSetKernelArg skybox 3");
	err |= clSetKernelArg(ocl->skybox_kernel, 4, sizeof(cl_int), &screen_width);
	CHECK_CL(err, "clSetKernelArg skybox 4");
	err |= clSetKernelArg(ocl->skybox_kernel, 5, sizeof(cl_int), &screen_height);
	CHECK_CL(err, "clSetKernelArg skybox 5");
	err |= clSetKernelArg(ocl->skybox_kernel, 6, sizeof(cl_mem), &ocl->buffer_skybox_top);
	CHECK_CL(err, "clSetKernelArg skybox 6");
	err |= clSetKernelArg(ocl->skybox_kernel, 7, sizeof(cl_mem), &ocl->buffer_skybox_bottom);
	CHECK_CL(err, "clSetKernelArg skybox 7");
	err |= clSetKernelArg(ocl->skybox_kernel, 8, sizeof(cl_mem), &ocl->buffer_skybox_left);
	CHECK_CL(err, "clSetKernelArg skybox 8");
	err |= clSetKernelArg(ocl->skybox_kernel, 9, sizeof(cl_mem), &ocl->buffer_skybox_right);
	CHECK_CL(err, "clSetKernelArg skybox 9");
	err |= clSetKernelArg(ocl->skybox_kernel, 10, sizeof(cl_mem), &ocl->buffer_skybox_front);
	CHECK_CL(err, "clSetKernelArg skybox 10");
	err |= clSetKernelArg(ocl->skybox_kernel, 11, sizeof(cl_mem), &ocl->buffer_skybox_back);
	CHECK_CL(err, "clSetKernelArg skybox 11");
	err |= clSetKernelArg(ocl->skybox_kernel, 12, sizeof(cl_int), &skybox_width);
	CHECK_CL(err, "clSetKernelArg skybox 12");
	err |= clSetKernelArg(ocl->skybox_kernel, 13, sizeof(cl_int), &skybox_height);
	CHECK_CL(err, "clSetKernelArg skybox 13");

	if (err != CL_SUCCESS) {
		printf("Error setting skybox kernel arguments: %d\n", err);
		return;
	}

	// Execute skybox kernel with event for profiling
	size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->skybox_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing skybox kernel: %d\n", err);
		return;
	}

	clFinish(ocl->queue);

	// Get profiling info if gpuTimeMs is not NULL
	if (gpuTimeMs != NULL) {
		cl_ulong start_time, end_time;
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
		*gpuTimeMs = (end_time - start_time) * 1e-6; // convert ns to ms
	}

	clReleaseEvent(kernel_event); // Always release events to avoid leaks
}

void applyReflectionsOpenCL(struct OpenCLContext *ocl, struct Camera *camera, struct SkyBox *skyBox, float *gpuTimeMs, bool composite) {
	cl_int err;

	// Set kernel arguments for applyReflections
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int skybox_width = skyBox->top->width;
	cl_int skybox_height = skyBox->top->height;

	if (composite) {
		cl_int one = 1;
		err = clSetKernelArg(ocl->applyReflections_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		err |= clSetKernelArg(ocl->applyReflections_kernel, 1, sizeof(cl_mem), &ocl->CompositedScreenDistances);
		err |= clSetKernelArg(ocl->applyReflections_kernel, 2, sizeof(cl_mem), &ocl->CompositedScreenNormals);
		err |= clSetKernelArg(ocl->applyReflections_kernel, 19, sizeof(cl_int), &one);
	} else {
		cl_int zero = 0;
		err = clSetKernelArg(ocl->applyReflections_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		err |= clSetKernelArg(ocl->applyReflections_kernel, 1, sizeof(cl_mem), &ocl->buffer_distances);
		err |= clSetKernelArg(ocl->applyReflections_kernel, 2, sizeof(cl_mem), &ocl->buffer_normals);
		err |= clSetKernelArg(ocl->applyReflections_kernel, 19, sizeof(cl_int), &zero);
	}

	err |= clSetKernelArg(ocl->applyReflections_kernel, 3, sizeof(cl_mem), &ocl->buffer_screen_material_roughness);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 4, sizeof(cl_mem), &ocl->buffer_screen_material_metallic);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 5, sizeof(cl_mem), &ocl->buffer_screen_material_emission);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 6, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 7, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 8, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 9, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 10, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 11, sizeof(cl_mem), &ocl->buffer_skybox_top);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 12, sizeof(cl_mem), &ocl->buffer_skybox_bottom);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 13, sizeof(cl_mem), &ocl->buffer_skybox_left);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 14, sizeof(cl_mem), &ocl->buffer_skybox_right);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 15, sizeof(cl_mem), &ocl->buffer_skybox_front);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 16, sizeof(cl_mem), &ocl->buffer_skybox_back);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 17, sizeof(cl_int), &skybox_width);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 18, sizeof(cl_int), &skybox_height);

	if (err != CL_SUCCESS) {
		printf("Error setting applyReflections kernel arguments: %d\n", err);
		return;
	}

	// Execute reflection kernel
	size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
	cl_event kernel_event;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->applyReflections_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing applyReflections kernel: %d\n", err);
		return;
	}

	clFinish(ocl->queue);

	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
	*gpuTimeMs = (end_time - start_time) * 1e-6; // convert ns to ms

	clReleaseEvent(kernel_event); // Always release events to avoid leaks
}

void renderTrianglesOpenCL(struct OpenCLContext *ocl, struct Triangles *triangles, struct Camera *camera, float *gpuTimeMs) {
	if (triangles->count == 0) return;

	cl_int err;
	cl_event kernel_event;

	// Set camera parameters that change each frame
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;

	// Update camera arguments (indices 6-8)
	err = clSetKernelArg(ocl->triangle_kernel, 6, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->triangle_kernel, 7, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->triangle_kernel, 8, sizeof(cl_float), &fov);

	if (err != CL_SUCCESS) {
		printf("Error setting camera kernel arguments: %d\n", err);
		return;
	}

	// Execute triangle rendering kernel
	size_t global_work_size = triangles->count;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->triangle_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &kernel_event);
	// size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
	// err = clEnqueueNDRangeKernel(ocl->queue, ocl->triangle_kernel, 2, NULL, global_work_size, NULL, 0, NULL, &kernel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing triangle kernel: %d\n", err);
		return;
	}
	clFinish(ocl->queue);

	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
	*gpuTimeMs = (end_time - start_time) * 1e-6; // convert ns to ms

	clReleaseEvent(kernel_event);
}

void renderTrianglesOpenCL_TwoPass(struct OpenCLContext *ocl, struct Triangles *triangles, struct Camera *camera, float *gpuTimeMs) {
	if (triangles->count == 0) return;

	cl_int err;
	cl_event vertex_event, tile_event, pixel_event;

	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int num_triangles = triangles->count;

	err = clSetKernelArg(ocl->calculateVertex_kernel, 0, sizeof(cl_mem), &ocl->buffer_triangle_v1);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 1, sizeof(cl_mem), &ocl->buffer_triangle_v2);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 2, sizeof(cl_mem), &ocl->buffer_triangle_v3);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 3, sizeof(cl_mem), &ocl->buffer_triangle_normals);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 4, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 5, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 6, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 7, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 8, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 9, sizeof(cl_int), &num_triangles);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 10, sizeof(cl_mem), &ocl->buffer_projected_verts);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 11, sizeof(cl_mem), &ocl->buffer_triangle_bboxes);
	err |= clSetKernelArg(ocl->calculateVertex_kernel, 12, sizeof(cl_mem), &ocl->buffer_valid_triangles);

	if (err != CL_SUCCESS) {
		printf("Error setting vertex kernel arguments: %d\n", err);
		return;
	}

	size_t vertex_global_size = triangles->count;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->calculateVertex_kernel, 1, NULL, &vertex_global_size, NULL, 0, NULL, &vertex_event);
	if (err != CL_SUCCESS) {
		printf("Error executing vertex kernel: %d\n", err);
		return;
	}

	int numTilesX = (ScreenWidth + 15) / 16;
	int numTilesY = (ScreenHeight + 15) / 16;
	cl_int num_tiles_x = numTilesX;

	err = clSetKernelArg(ocl->tileCulling_kernel, 0, sizeof(cl_mem), &ocl->buffer_triangle_bboxes);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 1, sizeof(cl_mem), &ocl->buffer_valid_triangles);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 2, sizeof(cl_mem), &ocl->buffer_tileLists);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 3, sizeof(cl_mem), &ocl->buffer_tileListCounts);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 4, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 5, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 6, sizeof(cl_int), &num_triangles);
	err |= clSetKernelArg(ocl->tileCulling_kernel, 7, sizeof(cl_int), &num_tiles_x);

	if (err != CL_SUCCESS) {
		printf("Error setting tile culling arguments: %d\n", err);
		return;
	}

	size_t tile_global_size[2] = {numTilesX, numTilesY};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->tileCulling_kernel, 2, NULL, tile_global_size, NULL, 1, &vertex_event, &tile_event);
	if (err != CL_SUCCESS) {
		printf("Error executing tile culling kernel: %d\n", err);
		return;
	}

	err = clSetKernelArg(ocl->shadePixels_kernel, 0, sizeof(cl_mem), &ocl->buffer_projected_verts);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 1, sizeof(cl_mem), &ocl->buffer_triangle_bboxes);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 2, sizeof(cl_mem), &ocl->buffer_valid_triangles);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 3, sizeof(cl_mem), &ocl->buffer_tileLists);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 4, sizeof(cl_mem), &ocl->buffer_tileListCounts);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 5, sizeof(cl_mem), &ocl->buffer_screen_colors);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 6, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 7, sizeof(cl_mem), &ocl->buffer_normals);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 8, sizeof(cl_mem), &ocl->buffer_screen_material_roughness);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 9, sizeof(cl_mem), &ocl->buffer_screen_material_metallic);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 10, sizeof(cl_mem), &ocl->buffer_screen_material_emission);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 11, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 12, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 13, sizeof(cl_int), &num_tiles_x);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 14, sizeof(cl_mem), &ocl->buffer_triangle_colors);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 15, sizeof(cl_mem), &ocl->buffer_triangle_roughness);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 16, sizeof(cl_mem), &ocl->buffer_triangle_metallic);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 17, sizeof(cl_mem), &ocl->buffer_triangle_emission);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 18, sizeof(cl_mem), &ocl->buffer_triangle_normals);

	if (err != CL_SUCCESS) {
		printf("Error setting pixel shader arguments: %d\n", err);
		return;
	}

	size_t pixel_global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->shadePixels_kernel, 2, NULL, pixel_global_size, NULL, 1, &tile_event, &pixel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing pixel shader kernel: %d\n", err);
		return;
	}

	clFinish(ocl->queue);

	if (gpuTimeMs != NULL) {
		cl_ulong vertex_start, pixel_end;
		clGetEventProfilingInfo(vertex_event, CL_PROFILING_COMMAND_START, sizeof(vertex_start), &vertex_start, NULL);
		clGetEventProfilingInfo(pixel_event, CL_PROFILING_COMMAND_END, sizeof(pixel_end), &pixel_end, NULL);
		*gpuTimeMs = (pixel_end - vertex_start) * 1e-6f;
	}

	clReleaseEvent(vertex_event);
	clReleaseEvent(tile_event);
	clReleaseEvent(pixel_event);
}

struct TimePartition {
	float collisionTime;
	float applyPressureTime;
	float updateParticlesTime;
	float moveToBoxTime;
	float updateGridTime;
	float renderTime;
	float clearScreenTime;
	float projectParticlesTime;
	float drawCursorTime;
	float drawBoundingBoxTime;
	float saveScreenTime;
	float sortTime;
	float projectionTime;
	float renderDistanceVelocityTime;
	float renderOpacityTime;
	float readDataTime;
	float projectLightParticlesTime;
};

struct GPUTimings {
	float renderSkyBoxTime;
	float renderTrianglesTime;
	float applyReflectionsTime;
	float applyBlurTime;
	float readBackTime;
	float renderTextTime;
	float projectParticlesTime;
	float renderBoundingBoxTime;
	float antiAliasingTime;
	float fireSimulationTime;
	float fireRenderingTime;
	float fluidSimulationTime;
	float compositingTime;
	float missileSimulationTime;
	float missileFireSimulationTime;
	float missileRenderingTime;
};

float totalTime(struct GPUTimings *gpuTimings) {
	int numOfElements = sizeof(struct GPUTimings) / sizeof(float);
	float total = 0.0f;
	for (int i = 0; i < numOfElements; i++) {
		float *timePtr = (float *)gpuTimings + i;
		total += *timePtr;
	}
	return total;
}

void renderGPUTimings(struct OpenCLContext *ocl, struct GPUTimings *gpuTimings, cl_mem *renderBuffer) {
	// Timing chart parameters
	int chartPosXLocal = chartPosX;
	int chartPosYLocal = chartPosY;
	int chartWidth = 100;  // Width of timing chart
	int chartHeight = 100; // Height of timing chart
	int paddingY = 5;	   // Top padding

	// Find maximum time for normalization
	float maxTime = fmaxf(gpuTimings->renderSkyBoxTime,
						  fmaxf(gpuTimings->renderTrianglesTime,
								fmaxf(gpuTimings->applyReflectionsTime,
									  fmaxf(gpuTimings->applyBlurTime,
											fmaxf(gpuTimings->readBackTime, gpuTimings->renderTextTime)))));

	// Add some padding to max time
	maxTime *= 1.1f;
	if (maxTime < 0.1f) maxTime = 10.0f; // Minimum scale

	// Set kernel arguments
	cl_int err = 0;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	if (renderBuffer == NULL) {
		err |= clSetKernelArg(ocl->gpuTimings_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		CHECK_CL(err, "clSetKernelArg gpuTimings 0");
	} else {
		err |= clSetKernelArg(ocl->gpuTimings_kernel, 0, sizeof(cl_mem), renderBuffer);
		CHECK_CL(err, "clSetKernelArg gpuTimings 0 (custom buffer)");
	}
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 1, sizeof(cl_int), &screen_width);
	CHECK_CL(err, "clSetKernelArg gpuTimings 1");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 2, sizeof(cl_int), &screen_height);
	CHECK_CL(err, "clSetKernelArg gpuTimings 2");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 3, sizeof(cl_int), &chartWidth);
	CHECK_CL(err, "clSetKernelArg gpuTimings 3");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 4, sizeof(cl_int), &chartHeight);
	CHECK_CL(err, "clSetKernelArg gpuTimings 4");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 5, sizeof(cl_int), &chartPosXLocal);
	CHECK_CL(err, "clSetKernelArg gpuTimings 5");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 6, sizeof(cl_int), &chartPosYLocal);
	CHECK_CL(err, "clSetKernelArg gpuTimings 6");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 7, sizeof(cl_int), &paddingY);
	CHECK_CL(err, "clSetKernelArg gpuTimings 7");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 8, sizeof(cl_float), &gpuTimings->renderSkyBoxTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 8");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 9, sizeof(cl_float), &gpuTimings->renderTrianglesTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 9");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 10, sizeof(cl_float), &gpuTimings->applyReflectionsTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 10");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 11, sizeof(cl_float), &gpuTimings->applyBlurTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 11");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 12, sizeof(cl_float), &gpuTimings->readBackTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 12");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 13, sizeof(cl_float), &gpuTimings->renderTextTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 13");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 14, sizeof(cl_float), &gpuTimings->projectParticlesTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 14");
	err |= clSetKernelArg(ocl->gpuTimings_kernel, 15, sizeof(cl_float), &maxTime);
	CHECK_CL(err, "clSetKernelArg gpuTimings 15");

	if (err != CL_SUCCESS) {
		printf("Error setting gpuTimings kernel arguments: %d\n", err);
		return;
	}

	// Execute kernel
	size_t global_work_size[2] = {chartWidth, chartHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->gpuTimings_kernel, 2, NULL,
								 global_work_size, NULL, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error executing gpuTimings kernel: %d\n", err);
	}
}

// Function prototypes
void projectParticlesOpenCL(struct OpenCLContext *ocl, struct PointSOA *particles, struct Camera *camera, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font, struct FireSOA *fireParticles, struct Missiles *missiles);
void projectParticlesOpenCL_NoWater(struct OpenCLContext *ocl, struct Camera *camera, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font, struct FireSOA *fireParticles, struct Missiles *missiles, struct IRSearchAndTrack *irst);

struct Light {
	float x;
	float y;
	float z;
};

int readCameraData(struct Camera *camera) {
	// printf("Attempting to read camera data\n");
	FILE *file = fopen("camera.bin", "rb");
	if (!file) {
		// printf("Camera file not found, using default camera\n");
		return 0;
	}

	// Read camera position
	if (fread(&camera->ray.origin[0], sizeof(float), 1, file) != 1 ||	 // Position X
		fread(&camera->ray.origin[1], sizeof(float), 1, file) != 1 ||	 // Position Y
		fread(&camera->ray.origin[2], sizeof(float), 1, file) != 1 ||	 // Position Z
		fread(&camera->ray.direction[0], sizeof(float), 1, file) != 1 || // Direction X
		fread(&camera->ray.direction[1], sizeof(float), 1, file) != 1 || // Direction Y
		fread(&camera->ray.direction[2], sizeof(float), 1, file) != 1 || // Direction Z
		fread(&camera->fov, sizeof(float), 1, file) != 1 ||
		fread(&camera->renderMode, sizeof(uint8_t), 1, file) != 1) {

		fclose(file);
		// printf("Error reading camera data, using default camera\n");
		return 0; // Error reading, use default camera
	}

	fclose(file);
	// printf("Successfully read camera data\n");
	return 1; // Successfully read camera data
}

struct ParticleIndex {
	int index;
	float distance;
};

struct ParticleIndexes {
	struct ParticleIndex particleIndexes[NUM_PARTICLES];
};

// Comparison function for qsort
int compareParticlesByDistance(const void *a, const void *b) {
	const struct ParticleIndex *particleA = (const struct ParticleIndex *)a;
	const struct ParticleIndex *particleB = (const struct ParticleIndex *)b;

	// Sort from farthest to nearest (descending order)
	if (particleA->distance > particleB->distance) return 1;
	if (particleA->distance < particleB->distance) return -1;
	return 0;
}

float fastInvSqrt(float x) {
	union {
		float f;
		uint32_t i;
	} u = {x};
	u.i = 0x5f3759df - (u.i >> 1);
	float y = u.f;
	return y * (1.5f - 0.5f * x * y * y);
};

void render(struct PointSOA *particles, struct Camera *camera, struct TimePartition *timePartition, struct ParticleIndexes *particleIndexes, struct OpenCLContext *openCLContext, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font, struct FireSOA *fireParticles, struct Missiles *missiles, struct IRSearchAndTrack *irst) {
	if (USE_GPU == 1) {
#if WATER_SIMULATION == 1
		projectParticlesOpenCL(openCLContext, particles, camera, triangles, skyBox, gpuTimings, font, fireParticles, missiles);
#else
		projectParticlesOpenCL_NoWater(openCLContext, camera, triangles, skyBox, gpuTimings, font, fireParticles, missiles, irst);
#endif
	}
}

int uploadTriangleData(struct OpenCLContext *ocl, struct Triangles *triangles) {
	cl_int err;

	if (triangles->count > NUMBER_OF_TRIANGLES) {
		printf("Error: Triangle count (%d) exceeds maximum buffer size (%d)\n",
			   triangles->count, NUMBER_OF_TRIANGLES);
		return 0;
	}

	// Skip upload if triangle count hasn't changed (optimization for static scenes)
	if (ocl->last_uploaded_triangle_count == triangles->count) {
		return 1; // Data already uploaded, no need to re-upload
	}

	err = clFinish(ocl->queue);
	if (err != CL_SUCCESS) {
		printf("Error waiting for GPU to finish before uploading triangles: %d\n", err);
		return 0;
	}

	printf("Uploading triangle data: %d triangles (previous: %d)\n",
		   triangles->count, ocl->last_uploaded_triangle_count);

	// Upload all triangle data once
	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_v1, CL_TRUE, 0,
							   triangles->count * 3 * sizeof(float), triangles->v1, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle v1 buffer during init: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_v2, CL_TRUE, 0,
							   triangles->count * 3 * sizeof(float), triangles->v2, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle v2 buffer during init: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_v3, CL_TRUE, 0,
							   triangles->count * 3 * sizeof(float), triangles->v3, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle v3 buffer during init: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_normals, CL_TRUE, 0,
							   triangles->count * 3 * sizeof(float), triangles->normals, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle normals buffer during init: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_colors, CL_TRUE, 0,
							   triangles->count * 3 * sizeof(float), triangles->colors, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle colors buffer during init: %d\n", err);
		return 0;
	}

	// Upload triangle material properties once
	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_roughness, CL_TRUE, 0,
							   triangles->count * sizeof(float), triangles->Roughness, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle roughness buffer during init: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_metallic, CL_TRUE, 0,
							   triangles->count * sizeof(float), triangles->Metallic, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle metallic buffer during init: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_triangle_emission, CL_TRUE, 0,
							   triangles->count * sizeof(float), triangles->Emission, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing triangle emission buffer during init: %d\n", err);
		return 0;
	}

	// Update the last uploaded count
	ocl->last_uploaded_triangle_count = triangles->count;

	// CRITICAL: Update the triangle count kernel argument
	// The triangle_kernel uses this at argument index 11
	cl_int num_triangles = triangles->count;
	err = clSetKernelArg(ocl->triangle_kernel, 11, sizeof(cl_int), &num_triangles);
	if (err != CL_SUCCESS) {
		printf("Error updating triangle count kernel argument: %d\n", err);
		return 0;
	}

	printf("Triangle data uploaded successfully\n");
	return 1;
}

void renderTextOpenCL(struct OpenCLContext *ocl, struct ImageFont *font, float *gpuTimeMs, cl_mem *renderBuffer) {
	if (textBufferLen == 0) {
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	cl_int err;
	cl_event kernel_event; // Add event for profiling

	// update staging buffers
	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_text_posX, CL_TRUE, 0,
							   textBufferLen * sizeof(int), posX, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing posX: %d\n", err);
		return;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_text_posY, CL_TRUE, 0,
							   textBufferLen * sizeof(int), posY, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing posY: %d\n", err);
		return;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_text_chars, CL_TRUE, 0,
							   textBufferLen * sizeof(char), textBuffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing chars: %d\n", err);
		return;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_text_color, CL_TRUE, 0,
							   textBufferLen * sizeof(uint32_t), textColor, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing color: %d\n", err);
		return;
	}

	if (err != CL_SUCCESS) {
		printf("Error creating text rendering buffers: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	// Set kernel arguments
	cl_int fontSizeX = font->width;
	cl_int fontSizeY = font->height;
	cl_int spriteSizeX = 8;
	cl_int spriteSizeY = 8;
	cl_int screenWidth = ScreenWidth;
	cl_int screenHeight = ScreenHeight;
	cl_int numChars = textBufferLen;

	err = clSetKernelArg(ocl->renderText_kernel, 0, sizeof(cl_int), &fontSizeX);
	CHECK_CL(err, "clSetKernelArg fontSizeX");
	err |= clSetKernelArg(ocl->renderText_kernel, 1, sizeof(cl_int), &fontSizeY);
	CHECK_CL(err, "clSetKernelArg fontSizeY");
	err |= clSetKernelArg(ocl->renderText_kernel, 2, sizeof(cl_int), &spriteSizeX);
	CHECK_CL(err, "clSetKernelArg spriteSizeX");
	err |= clSetKernelArg(ocl->renderText_kernel, 3, sizeof(cl_int), &spriteSizeY);
	CHECK_CL(err, "clSetKernelArg spriteSizeY");

	if (renderBuffer == NULL) {
		err |= clSetKernelArg(ocl->renderText_kernel, 4, sizeof(cl_mem), &ocl->buffer_screen_colors);
		CHECK_CL(err, "clSetKernelArg screen colors");
	} else {
		err |= clSetKernelArg(ocl->renderText_kernel, 4, sizeof(cl_mem), renderBuffer);
		CHECK_CL(err, "clSetKernelArg custom render buffer");
	}

	err |= clSetKernelArg(ocl->renderText_kernel, 5, sizeof(cl_mem), &ocl->buffer_font_data);
	CHECK_CL(err, "clSetKernelArg font data");
	err |= clSetKernelArg(ocl->renderText_kernel, 6, sizeof(cl_int), &screenWidth);
	CHECK_CL(err, "clSetKernelArg screenWidth");
	err |= clSetKernelArg(ocl->renderText_kernel, 7, sizeof(cl_int), &screenHeight);
	CHECK_CL(err, "clSetKernelArg screenHeight");
	err |= clSetKernelArg(ocl->renderText_kernel, 8, sizeof(cl_mem), &ocl->buffer_text_posX);
	CHECK_CL(err, "clSetKernelArg text posX");
	err |= clSetKernelArg(ocl->renderText_kernel, 9, sizeof(cl_mem), &ocl->buffer_text_posY);
	CHECK_CL(err, "clSetKernelArg text posY");
	err |= clSetKernelArg(ocl->renderText_kernel, 10, sizeof(cl_mem), &ocl->buffer_text_chars);
	CHECK_CL(err, "clSetKernelArg text chars");
	err |= clSetKernelArg(ocl->renderText_kernel, 11, sizeof(cl_mem), &ocl->buffer_text_color);
	CHECK_CL(err, "clSetKernelArg text color");
	err |= clSetKernelArg(ocl->renderText_kernel, 12, sizeof(cl_int), &numChars);
	CHECK_CL(err, "clSetKernelArg numChars");

	if (err != CL_SUCCESS) {
		printf("Error setting RenderText kernel arguments: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	// Execute kernel with profiling event
	size_t global_work_size = textBufferLen;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->renderText_kernel, 1, NULL,
								 &global_work_size, NULL, 0, NULL, &kernel_event);

	if (err != CL_SUCCESS) {
		printf("Error executing RenderText kernel: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
	} else {
		// Wait for completion
		clFinish(ocl->queue);

		// Get profiling info if requested
		if (gpuTimeMs != NULL) {
			cl_ulong start_time, end_time;
			err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
										  sizeof(start_time), &start_time, NULL);
			err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
										   sizeof(end_time), &end_time, NULL);

			if (err == CL_SUCCESS) {
				*gpuTimeMs = (end_time - start_time) * 1e-6f; // convert ns to ms
			} else {
				*gpuTimeMs = 0.0f;
			}
		}

		clReleaseEvent(kernel_event);
	}

	// Reset text buffer length for next rendering
	textBufferLen = 0;
}

inline uint32_t colorToInt(uint8_t color[3]) {
	return ((uint32_t)color[0] << 16) | ((uint32_t)color[1] << 8) | (uint32_t)color[2];
}

void addTextOpenCL(struct OpenCLContext *ocl, struct ImageFont *font, const char *text, int startX, int startY, uint8_t color[3]) {
	if (!font->data || !text) return;

	int textLen = strlen(text);
	if (textLen == 0) return;

	// Check if we have enough space
	if (textBufferLen + textLen >= MAX_TEXT_LENGTH) {
		printf("Text buffer overflow: would exceed %d characters\n", MAX_TEXT_LENGTH);
		return;
	}

	// Track current position for line wrapping
	int currentX = startX;
	int currentY = startY;
	int charPosInLine = 0; // Position within current line

	// Copy characters and set positions
	for (int i = 0; i < textLen; i++) {
		char c = text[i];

		// Handle newline character
		if (c == '\n') {
			// Move to next line
			currentX = startX; // Reset to beginning X position
			currentY += 8;	   // Move down by 8 pixels
			charPosInLine = 0; // Reset character position in line
			continue;		   // Skip adding newline to buffer
		}

		// Only add printable ASCII characters
		if (c >= 32 && c <= 126) {
			textBuffer[textBufferLen] = c;
			posX[textBufferLen] = currentX;
			posY[textBufferLen] = currentY;
			textColor[textBufferLen] = colorToInt(color); // Convert RGB to int
			textBufferLen++;

			// Move to next character position
			currentX += 8; // 8 pixel spacing between characters
			charPosInLine++;
		}
	}
}

int setupStaticKernelArguments(struct OpenCLContext *ocl, struct Triangles *triangles, struct SkyBox *skyBox) {
	cl_int err;

	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int num_triangles = triangles->count;

	// Correct argument indices based on kernel signature:
	// renderTriangles(v1, v2, v3, normals, ScreenDistances, ScreenNormals, camPos, camDir, fov,
	//                screenWidth, screenHeight, numTriangles, TriangleColors, ScreenColors,
	//                roughness, metallic, emission)

	err = clSetKernelArg(ocl->triangle_kernel, 0, sizeof(cl_mem), &ocl->buffer_triangle_v1);
	err |= clSetKernelArg(ocl->triangle_kernel, 1, sizeof(cl_mem), &ocl->buffer_triangle_v2);
	err |= clSetKernelArg(ocl->triangle_kernel, 2, sizeof(cl_mem), &ocl->buffer_triangle_v3);
	err |= clSetKernelArg(ocl->triangle_kernel, 3, sizeof(cl_mem), &ocl->buffer_triangle_normals);
	err |= clSetKernelArg(ocl->triangle_kernel, 4, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->triangle_kernel, 5, sizeof(cl_mem), &ocl->buffer_normals);

	// Skip 6-8 (camera parameters - these change each frame)

	err |= clSetKernelArg(ocl->triangle_kernel, 9, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->triangle_kernel, 10, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->triangle_kernel, 11, sizeof(cl_int), &num_triangles);
	err |= clSetKernelArg(ocl->triangle_kernel, 12, sizeof(cl_mem), &ocl->buffer_triangle_colors);
	err |= clSetKernelArg(ocl->triangle_kernel, 13, sizeof(cl_mem), &ocl->buffer_screen_colors);
	err |= clSetKernelArg(ocl->triangle_kernel, 14, sizeof(cl_mem), &ocl->buffer_triangle_roughness);
	err |= clSetKernelArg(ocl->triangle_kernel, 15, sizeof(cl_mem), &ocl->buffer_triangle_metallic);
	err |= clSetKernelArg(ocl->triangle_kernel, 16, sizeof(cl_mem), &ocl->buffer_triangle_emission);
	// screen materials properties
	err |= clSetKernelArg(ocl->triangle_kernel, 17, sizeof(cl_mem), &ocl->buffer_screen_material_roughness);
	err |= clSetKernelArg(ocl->triangle_kernel, 18, sizeof(cl_mem), &ocl->buffer_screen_material_metallic);
	err |= clSetKernelArg(ocl->triangle_kernel, 19, sizeof(cl_mem), &ocl->buffer_screen_material_emission);

	if (err != CL_SUCCESS) {
		printf("Error setting static triangle kernel arguments: %d\n", err);
		return 0;
	}

	return 1;
}

void renderWireframeOpenCL(struct OpenCLContext *ocl, struct Triangles *triangles, struct Camera *camera, float *gpuTimeMs) {
	if (triangles->count == 0) return;

	cl_int err;
	cl_event kernel_event;

	// Set camera parameters that change each frame
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int num_triangles = triangles->count;
	cl_float3 wire_color = {1.0f, 1.0f, 1.0f}; // Red wireframe

	// Set wireframe kernel arguments
	err = clSetKernelArg(ocl->wireframe_kernel, 0, sizeof(cl_mem), &ocl->buffer_projected_verts);
	err |= clSetKernelArg(ocl->wireframe_kernel, 1, sizeof(cl_mem), &ocl->buffer_valid_triangles);
	err |= clSetKernelArg(ocl->wireframe_kernel, 2, sizeof(cl_mem), &ocl->buffer_screen_colors);
	err |= clSetKernelArg(ocl->wireframe_kernel, 3, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->wireframe_kernel, 4, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->wireframe_kernel, 5, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->wireframe_kernel, 6, sizeof(cl_int), &num_triangles);
	err |= clSetKernelArg(ocl->wireframe_kernel, 7, sizeof(cl_float3), &wire_color);

	if (err != CL_SUCCESS) {
		printf("Error setting wireframe kernel arguments: %d\n", err);
		return;
	}

	// Execute wireframe kernel
	size_t global_work_size = triangles->count;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->wireframe_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &kernel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing wireframe kernel: %d\n", err);
		return;
	}
	clFinish(ocl->queue);

	if (gpuTimeMs != NULL) {
		cl_ulong start_time, end_time;
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
		*gpuTimeMs = (end_time - start_time) * 1e-6; // convert ns to ms
	}

	clReleaseEvent(kernel_event);
}

void renderTerrainLOD_GPU(struct OpenCLContext *ocl, struct Camera *camera, float *gpuTimeMs) {
	cl_int err;
	cl_event kernel_event;

	// Set camera parameters
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;

	// Calculate total number of triangles across all LOD levels
	int totalTriangles = HIGH_RES_TRIANGLE_COUNT + MID_RES_TRIANGLE_COUNT + LOW_RES_TRIANGLE_COUNT;

	// Set kernel arguments
	err = clSetKernelArg(ocl->renderTerrainLOD_kernel, 0, sizeof(cl_mem), &ocl->struct_mapGpu);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 1, sizeof(cl_mem), &ocl->buffer_screen_colors);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 2, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 3, sizeof(cl_mem), &ocl->buffer_normals);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 4, sizeof(cl_mem), &ocl->buffer_screen_material_roughness);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 5, sizeof(cl_mem), &ocl->buffer_screen_material_metallic);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 6, sizeof(cl_mem), &ocl->buffer_screen_material_emission);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 7, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 8, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 9, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 10, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->renderTerrainLOD_kernel, 11, sizeof(cl_int), &screen_height);

	if (err != CL_SUCCESS) {
		printf("Error setting renderTerrainLOD kernel arguments: %d\n", err);
		return;
	}

	// Execute kernel - one work item per triangle
	size_t global_work_size = totalTriangles;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->renderTerrainLOD_kernel, 1, NULL,
								 &global_work_size, NULL, 0, NULL, &kernel_event);

	if (err != CL_SUCCESS) {
		printf("Error executing renderTerrainLOD kernel: %d\n", err);
		return;
	}

	clWaitForEvents(1, &kernel_event);

	// Get timing info
	if (gpuTimeMs) {
		cl_ulong start_time, end_time;
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
		*gpuTimeMs = (end_time - start_time) * 1e-6; // convert ns to ms
	}

	clReleaseEvent(kernel_event);
}

void renderTerrainDepthLOD_GPU(struct OpenCLContext *ocl, struct Camera *camera,
							   cl_mem output_buffer, int screenWidth, int screenHeight,
							   float *gpuTimeMs) {
	cl_int err;
	cl_event kernel_event;

	// Clear output buffer
	float zeroFloat = 0.0f;
	err = clEnqueueFillBuffer(ocl->queue, output_buffer, &zeroFloat, sizeof(float), 0,
							  screenWidth * screenHeight * sizeof(float), 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error clearing depth buffer: %d\n", err);
		return;
	}

	// Set camera parameters
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int cl_screen_width = screenWidth;
	cl_int cl_screen_height = screenHeight;

	// Calculate total number of triangles across all LOD levels
	int totalTriangles = HIGH_RES_TRIANGLE_COUNT + MID_RES_TRIANGLE_COUNT + LOW_RES_TRIANGLE_COUNT;

	// Set kernel arguments
	err = clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 0, sizeof(cl_mem), &ocl->struct_mapGpu);
	err |= clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 1, sizeof(cl_mem), &output_buffer);
	err |= clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 2, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 3, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 4, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 5, sizeof(cl_int), &cl_screen_width);
	err |= clSetKernelArg(ocl->renderTerrainDepthLOD_kernel, 6, sizeof(cl_int), &cl_screen_height);

	if (err != CL_SUCCESS) {
		printf("Error setting renderTerrainDepthLOD kernel arguments: %d\n", err);
		return;
	}

	// Execute kernel
	size_t global_work_size = totalTriangles;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->renderTerrainDepthLOD_kernel, 1, NULL,
								 &global_work_size, NULL, 0, NULL, &kernel_event);

	if (err != CL_SUCCESS) {
		printf("Error executing renderTerrainDepthLOD kernel: %d\n", err);
		return;
	}

	clWaitForEvents(1, &kernel_event);

	// Get timing info
	if (gpuTimeMs) {
		cl_ulong start_time, end_time;
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
		clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
		*gpuTimeMs = (end_time - start_time) * 1e-6; // convert ns to ms
	}

	clReleaseEvent(kernel_event);
}

int initializeOpenCLWithGL(struct OpenCLContext *ocl, struct Triangles *triangles,
						   struct SkyBox *skyBox, struct ImageFont *imageFont,
						   struct BVHLinear *bvh, GLFWwindow *window, struct Missiles *missiles, struct MapGPU *mapGPU) {
	cl_int err;

	// Make sure OpenGL context is current
	glfwMakeContextCurrent(window);

	// Get platform
	err = clGetPlatformIDs(1, &ocl->platform, NULL);
	if (err != CL_SUCCESS) {
		printf("Error getting OpenCL platform: %d\n", err);
		return 0;
	}

	// Get device
	err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_GPU, 1, &ocl->device, NULL);
	if (err != CL_SUCCESS) {
		// Fallback to CPU if GPU not available
		err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_CPU, 1, &ocl->device, NULL);
		if (err != CL_SUCCESS) {
			printf("Error getting OpenCL device: %d\n", err);
			return 0;
		}
	}

	// Create OpenGL texture that will be shared with OpenCL
	glGenTextures(1, &ocl->gl_texture);
	glBindTexture(GL_TEXTURE_2D, ocl->gl_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ScreenWidth, ScreenHeight, 0,
				 GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Create OpenGL texture for UI that will be shared with OpenCL
	glGenTextures(1, &ocl->gl_ui_texture);
	glBindTexture(GL_TEXTURE_2D, ocl->gl_ui_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ScreenWidth, ScreenHeight, 0,
				 GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

// Create OpenCL context with OpenGL sharing
#ifdef _WIN32
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)ocl->platform,
		0};
#elif defined(__APPLE__)
	CGLContextObj gl_context = CGLGetCurrentContext();
	CGLShareGroupObj share_group = CGLGetShareGroup(gl_context);
	cl_context_properties properties[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		(cl_context_properties)share_group,
		CL_CONTEXT_PLATFORM, (cl_context_properties)ocl->platform,
		0};
#else // Linux/X11
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
		CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
		CL_CONTEXT_PLATFORM, (cl_context_properties)ocl->platform,
		0};
#endif

	ocl->context = clCreateContext(properties, 1, &ocl->device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating OpenCL-GL context: %d\n", err);
		return 0;
	}

	// Create command queue with profiling (using modern API)
	cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	ocl->queue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, queue_props, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating OpenCL command queue: %d\n", err);
		return 0;
	}

	// Create OpenCL image from OpenGL texture
	ocl->cl_texture_buffer = clCreateFromGLTexture(ocl->context, CL_MEM_WRITE_ONLY,
												   GL_TEXTURE_2D, 0, ocl->gl_texture, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating CL image from GL texture: %d\n", err);
		return 0;
	}

	// Create OpenCL image from OpenGL texture for UI
	ocl->cl_ui_texture_buffer = clCreateFromGLTexture(ocl->context, CL_MEM_WRITE_ONLY,
													  GL_TEXTURE_2D, 0, ocl->gl_ui_texture, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating CL image from GL texture for UI: %d\n", err);
		return 0;
	}

	// Crete temporary buffer for rendering to screen before copying to texture
	ocl->cl_ui_texture_buffer_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
													ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating screen colors buffer: %d\n", err);
		return 0;
	}

	// Read kernel source
	FILE *file = fopen("openGlShaders/screenCordinates.cl", "r");
	if (!file) {
		printf("Error opening kernel file\n");
		return 0;
	}

	fseek(file, 0, SEEK_END);
	size_t source_size = ftell(file);
	fseek(file, 0, SEEK_SET);

	char *source = (char *)malloc(source_size + 1);
	fread(source, 1, source_size, file);
	source[source_size] = '\0';
	fclose(file);

	// Create program
	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char **)&source, &source_size, &err);
	free(source);
	if (err != CL_SUCCESS) {
		printf("Error creating OpenCL program: %d\n", err);
		return 0;
	}

	// Build program
	err = clBuildProgram(ocl->program, 1, &ocl->device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error building OpenCL program: %d\n", err);

		// Get build log
		size_t log_size;
		clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *log = (char *)malloc(log_size);
		clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("Build log: %s\n", log);
		free(log);
		return 0;
	}

	// Create all kernels
	ocl->renderMissile_kernel = clCreateKernel(ocl->program, "renderMissile", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile render kernel: %s\n", clErrorString(err));
		return 0;
	}

	ocl->renderDepth = clCreateKernel(ocl->program, "renderDepthBufferFast", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile render kernel: %s\n", clErrorString(err));
		return 0;
	}

	ocl->renderFireTemperature_kernel = clCreateKernel(ocl->program, "filterOverlap", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating renderFireTemperature kernel: %d\n", err);
		return 0;
	}

	ocl->composite_kernel = clCreateKernel(ocl->program, "compositeBuffers", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating composite kernel: %d\n", err);
		return 0;
	}

	ocl->antiAliasKernel = clCreateKernel(ocl->program, "antiAlias", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating antiAlias kernel: %d\n", err);
		return 0;
	}

	ocl->overlayImage_kernel = clCreateKernel(ocl->program, "OverlayImage", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating overlayImage kernel: %d\n", err);
		return 0;
	}

	ocl->fire_render_kernel = clCreateKernel(ocl->program, "renderFire", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating renderFireParticles kernel: %d\n", err);
		return 0;
	}

	ocl->blur_fire_kernel = clCreateKernel(ocl->program, "blurFire", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating blurFire kernel: %d\n", err);
		return 0;
	}

	ocl->clearColorBuffer_kernel = clCreateKernel(ocl->program, "clearColorBuffer", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating clearColorBuffer kernel: %d\n", err);
		return 0;
	}

	ocl->wireframe_kernel = clCreateKernel(ocl->program, "renderWireFrame", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating wireframe kernel: %d\n", err);
		return 0;
	}

	ocl->calculateVertex_kernel = clCreateKernel(ocl->program, "calculateVertexCoordinate", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating calculateVertexCoordinate kernel: %d\n", err);
		return 0;
	}

	ocl->tileCulling_kernel = clCreateKernel(ocl->program, "TileCulling", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating TileCulling kernel: %d\n", err);
		return 0;
	}

	ocl->drawBoundingBox_kernel = clCreateKernel(ocl->program, "drawBoundingBox", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating rayTrace kernel: %d\n", err);
		return 0;
	}

	ocl->shadePixels_kernel = clCreateKernel(ocl->program, "ShadePixels", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating ShadePixels kernel: %d\n", err);
		return 0;
	}

	ocl->renderText_kernel = clCreateKernel(ocl->program, "renderText", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating renderText kernel: %d\n", err);
		return 0;
	}

	ocl->gpuTimings_kernel = clCreateKernel(ocl->program, "gpuTimings", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating gpuTimings kernel: %d\n", err);
		return 0;
	}

	ocl->kernel = clCreateKernel(ocl->program, "project_points_to_screen", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating OpenCL kernel: %d\n", err);
		return 0;
	}

	ocl->blur_kernel = clCreateKernel(ocl->program, "blur_distances", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating blur kernel: %d\n", err);
		return 0;
	}

	ocl->applyReflections_kernel = clCreateKernel(ocl->program, "applyReflections", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating applyReflections kernel: %d\n", err);
		return 0;
	}

	ocl->skybox_kernel = clCreateKernel(ocl->program, "renderSkyBox", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating skybox kernel: %d\n", err);
		return 0;
	}

	ocl->normals_kernel = clCreateKernel(ocl->program, "calculate_normals_from_blurred_distances", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating normals kernel: %d\n", err);
		return 0;
	}

	ocl->triangle_kernel = clCreateKernel(ocl->program, "renderTriangles", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating triangle kernel: %d\n", err);
		return 0;
	}

	ocl->copyToTexture_kernel = clCreateKernel(ocl->program, "copyToGLTexture", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating copyToTexture kernel: %d\n", err);
		return 0;
	}

	ocl->composite_cones_kernel = clCreateKernel(ocl->program, "composite_cones", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating composite_cones kernel: %d\n", err);
		return 0;
	}

	ocl->renderTerrainLOD_kernel = clCreateKernel(ocl->program, "renderTerrainLOD", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating renderTerrainLOD kernel: %d\n", err);
		return 0;
	}

	ocl->renderTerrainDepthLOD_kernel = clCreateKernel(ocl->program, "renderTerrainDepthLOD", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating renderTerrainDepthLOD kernel: %d\n", err);
		return 0;
	}

#if WATER_SIMULATION == 1
	ocl->buffer_points = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
										NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating points buffer: %d\n", err);
		return 0;
	}
#else
	// Set to NULL when water simulation is disabled
	ocl->buffer_points = NULL;
#endif

	ocl->struct_mapGpu = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										sizeof(struct MapGPU), mapGPU, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating MapGPU struct buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_seeker_distances = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
												  MISSILE_SEEKER_SIZE * MISSILE_SEEKER_SIZE * sizeof(float), NULL, &err);

	if (err != CL_SUCCESS) {
		printf("Error creating seeker view buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_seeker_distances_atlas = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
														MISSILE_SEEKER_SIZE * MISSILE_SEEKER_SIZE * MAX_FIRE_SIMS * sizeof(float), NULL, &err);

	if (err != CL_SUCCESS) {
		printf("Error creating seeker view buffer: %d\n", err);
		return 0;
	}

	ocl->mapped_seeker_distances = NULL;

	size_t coneDataSize = MAX_FIRE_SIMS * sizeof(float);
	ocl->buffer_cone_originX = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone originX buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_originY = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone originY buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_originZ = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone originZ buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_dirX = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone dirX buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_dirY = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone dirY buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_dirZ = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone dirZ buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_fov = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone FOV buffer: %d\n", err);
		return 0;
	}
	ocl->buffer_cone_maxDist = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, coneDataSize, NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating cone maxDist buffer: %d\n", err);
		return 0;
	}

	// missile triangles buffer
	int maxMissileTriangles = missiles->missileModel->count;
	if (maxMissileTriangles == 0) {
		printf("Warning: No missile triangles loaded, skipping OpenCL buffer creation\n");
		return 0;
	}
	ocl->buffer_missile_v1 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											maxMissileTriangles * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile v1 buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_missile_v1, CL_TRUE, 0,
							   maxMissileTriangles * 3 * sizeof(float), missiles->missileModel->v1, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile v1 buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_missile_v2 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											maxMissileTriangles * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile v2 buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_missile_v2, CL_TRUE, 0,
							   maxMissileTriangles * 3 * sizeof(float), missiles->missileModel->v2, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile v2 buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_missile_v3 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											maxMissileTriangles * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile v3 buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_missile_v3, CL_TRUE, 0,
							   maxMissileTriangles * 3 * sizeof(float), missiles->missileModel->v3, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile v3 buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_missile_normals = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												 maxMissileTriangles * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile normals buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_missile_normals, CL_TRUE, 0,
							   maxMissileTriangles * 3 * sizeof(float), missiles->missileModel->normals, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile normals buffer: %d\n", err);
		return 0;
	}

	ocl->missile_color_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											   maxMissileTriangles * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile color buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->missile_color_buffer, CL_TRUE, 0,
							   maxMissileTriangles * 3 * sizeof(float), missiles->missileModel->colors, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile color buffer: %d\n", err);
		return 0;
	}

	ocl->missile_roughness_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												   maxMissileTriangles * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile roughness buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->missile_roughness_buffer, CL_TRUE, 0,
							   maxMissileTriangles * sizeof(float), missiles->missileModel->Roughness, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile roughness buffer: %d\n", err);
		return 0;
	}

	ocl->missile_metallic_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												  maxMissileTriangles * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile metallic buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->missile_metallic_buffer, CL_TRUE, 0,
							   maxMissileTriangles * sizeof(float), missiles->missileModel->Metallic, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile metallic buffer: %d\n", err);
		return 0;
	}

	ocl->missile_emission_buffer = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												  maxMissileTriangles * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating missile emission buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->missile_emission_buffer, CL_TRUE, 0,
							   maxMissileTriangles * sizeof(float), missiles->missileModel->Emission, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing missile emission buffer: %d\n", err);
		return 0;
	}

	ocl->FireScreenAlphas = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
										   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating screen alphas buffer: %d\n", err);
		return 0;
	}

	ocl->FireTemperature = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
										  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating screen alphas buffer: %d\n", err);
		return 0;
	}

	ocl->FireScreenAlphasTemp = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
											   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating screen alphas temp buffer: %d\n", err);
		return 0;
	}

	// BVH buffers
	ocl->buffer_bvh_nodes = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
										   bvh->NodesCount * sizeof(struct BVHNode), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating BVH nodes buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_bvh_triangles = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											   bvh->TrianglesCount * sizeof(struct Triangle), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating BVH triangles buffer: %d\n", err);
		return 0;
	}

	// Upload BVH data
	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_bvh_nodes, CL_TRUE, 0,
							   bvh->NodesCount * sizeof(struct BVHNode), bvh->Nodes, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing BVH nodes buffer: %d\n", err);
		return 0;
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_bvh_triangles, CL_TRUE, 0,
							   bvh->TrianglesCount * sizeof(struct Triangle), bvh->Triangles, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing BVH triangles buffer: %d\n", err);
		return 0;
	}

	// Create all other buffers (continuing with your existing buffer creation code)
	ocl->buffer_projected_verts = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												 triangles->count * 9 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating projected vertices buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_triangle_bboxes = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												 triangles->count * 4 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating triangle bboxes buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_valid_triangles = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												 triangles->count * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating valid triangles buffer: %d\n", err);
		return 0;
	}

	int numTilesX = (ScreenWidth + 15) / 16;
	int numTilesY = (ScreenHeight + 15) / 16;
	int numTiles = numTilesX * numTilesY;

	ocl->buffer_tileLists = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
										   numTiles * 512 * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating tile lists buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_tileListCounts = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												numTiles * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating tile list counts buffer: %d\n", err);
		return 0;
	}

	ocl->CompositedScreenColors = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
												 ScreenWidth * ScreenHeight * sizeof(float) * 3, NULL, &err);
	ocl->CompositedScreenDistances = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
													ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->CompositedScreenNormals = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
												  ScreenWidth * ScreenHeight * sizeof(float) * 3, NULL, &err);

	// Screen fire rendering buffers
	ocl->FireScreenColors = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
										   ScreenWidth * ScreenHeight * sizeof(float) * 3, NULL, &err);
	ocl->FireScreenDistances = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
											  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->FireScreenColorsTemp = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
											   ScreenWidth * ScreenHeight * sizeof(float) * 3, NULL, &err);
	ocl->FireScreenDistancesTemp = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
												  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->FireScreenNormals = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
											ScreenWidth * ScreenHeight * sizeof(float) * 3, NULL, &err);
	ocl->posX = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
							   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);
	ocl->posY = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
							   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);
	ocl->posZ = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
							   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);
	ocl->velX = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
							   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);
	ocl->velY = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
							   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);
	ocl->velZ = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
							   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);
	ocl->lifeTime = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
								   sizeof(float) * NUM_FIRE_PARTICLES, NULL, &err);

	// Screen material properties buffers
	ocl->buffer_screen_material_roughness = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
														   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_screen_material_metallic = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
														  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_screen_material_emission = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
														  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);

	// Continue with all other buffer creations from your existing code...
#if WATER_SIMULATION == 1
	ocl->buffer_velocities = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
#else
	ocl->buffer_velocities = NULL;
#endif
	ocl->buffer_distances = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
										   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_opacities = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
										   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_velocities_screen = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
												   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_normals = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
										 ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);
	ocl->buffer_distances_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_opacities_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);

	// Triangle buffers
	// Allocate buffers to maximum size to support dynamic LOD terrain loading
	ocl->buffer_triangle_roughness = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
													NUMBER_OF_TRIANGLES * sizeof(float), NULL, &err);
	ocl->buffer_triangle_metallic = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												   NUMBER_OF_TRIANGLES * sizeof(float), NULL, &err);
	ocl->buffer_triangle_emission = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												   NUMBER_OF_TRIANGLES * sizeof(float), NULL, &err);
	ocl->buffer_triangle_v1 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
											 NUMBER_OF_TRIANGLES * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_v2 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
											 NUMBER_OF_TRIANGLES * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_v3 = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
											 NUMBER_OF_TRIANGLES * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_normals = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												  NUMBER_OF_TRIANGLES * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_colors = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
												 NUMBER_OF_TRIANGLES * 3 * sizeof(float), NULL, &err);
	ocl->buffer_screen_colors = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
											   ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);

	// Font buffer
	size_t font_size = imageFont->width * imageFont->height * sizeof(char);
	ocl->buffer_font_data = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										   font_size, imageFont->data, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating font buffer: %d\n", err);
		return 0;
	}

	err = CL_SUCCESS;
	ocl->buffer_text_posX = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
										   MAX_TEXT_LENGTH * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating text_posX buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_text_posY = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
										   MAX_TEXT_LENGTH * sizeof(int), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating text_posY buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_text_chars = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											MAX_TEXT_LENGTH * sizeof(char), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating text_chars buffer: %d\n", err);
		return 0;
	}

	ocl->buffer_text_color = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											MAX_TEXT_LENGTH * sizeof(uint32_t), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating text_color buffer: %d\n", err);
		return 0;
	}

	// Initialize skybox buffers
	if (!initializeSkyboxBuffers(ocl, skyBox)) {
		printf("Failed to initialize skybox buffers during OpenCL init\n");
		return 0;
	}

	// Upload triangle data
	if (!uploadTriangleData(ocl, triangles)) {
		printf("Failed to upload triangle data during OpenCL init\n");
		return 0;
	}

	// Set static kernel arguments
	if (!setupStaticKernelArguments(ocl, triangles, skyBox)) {
		printf("Failed to set static kernel arguments during OpenCL init\n");
		return 0;
	}

	// Pre-allocate host memory buffers
#if WATER_SIMULATION == 1
	ocl->host_points_data = (float *)malloc(NUM_PARTICLES * 3 * sizeof(float));
	ocl->host_velocities_data = (float *)malloc(NUM_PARTICLES * 3 * sizeof(float));
#else
	ocl->host_points_data = NULL;
	ocl->host_velocities_data = NULL;
#endif
	ocl->host_distances_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
	ocl->host_opacities_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
	ocl->host_velocities_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
	ocl->host_normals_result = (float *)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));
	ocl->host_screen_colors_result = (float *)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));

	// Check for allocation failures
#if WATER_SIMULATION == 1
	if (!ocl->host_points_data || !ocl->host_velocities_data ||
		!ocl->host_distances_result || !ocl->host_opacities_result ||
		!ocl->host_velocities_result || !ocl->host_normals_result ||
		!ocl->host_screen_colors_result) {
		printf("Failed to allocate host memory for OpenCL\n");
		return 0;
	}
#else
	if (!ocl->host_distances_result || !ocl->host_opacities_result ||
		!ocl->host_velocities_result || !ocl->host_normals_result ||
		!ocl->host_screen_colors_result) {
		printf("Failed to allocate host memory for OpenCL\n");
		return 0;
	}
#endif

	// Initialize triangle tracking
	ocl->last_uploaded_triangle_count = 0;

	printf("OpenCL-GL interop initialized successfully\n");
	return 1;
}

void renderBoundingBox(struct OpenCLContext *ocl, struct Camera *camera, struct PointSOA *particles, float *gpuTimeMs) {
	cl_int err;
	cl_event kernel_event;

	// Set kernel arguments for drawing bounding box
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float3 cam_up = {0.0f, 1.0f, 0.0f}; // Standard up vector
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_float3 bbox_min = {particles->bBoxMin[0], particles->bBoxMin[1], particles->bBoxMin[2]};
	cl_float3 bbox_max = {particles->bBoxMax[0], particles->bBoxMax[1], particles->bBoxMax[2]};

	// Set all kernel arguments
	err = clSetKernelArg(ocl->drawBoundingBox_kernel, 0, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 1, sizeof(cl_mem), &ocl->buffer_opacities);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 2, sizeof(cl_mem), &ocl->buffer_velocities_screen);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 3, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 4, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 5, sizeof(cl_float3), &cam_up);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 6, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 7, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 8, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 9, sizeof(cl_float3), &bbox_min);
	err |= clSetKernelArg(ocl->drawBoundingBox_kernel, 10, sizeof(cl_float3), &bbox_max);

	if (err != CL_SUCCESS) {
		printf("Error setting drawBoundingBox kernel arguments: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	// Execute the kernel
	size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->drawBoundingBox_kernel, 2, NULL,
								 global_work_size, NULL, 0, NULL, &kernel_event);

	if (err != CL_SUCCESS) {
		printf("Error executing drawBoundingBox kernel: %d\n", err);
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	// Wait for completion
	clFinish(ocl->queue);

	// Get timing information if requested
	if (gpuTimeMs != NULL) {
		cl_ulong start_time, end_time;
		err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
									  sizeof(start_time), &start_time, NULL);
		err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
									   sizeof(end_time), &end_time, NULL);

		if (err == CL_SUCCESS) {
			*gpuTimeMs = (end_time - start_time) * 1e-6f; // Convert nanoseconds to milliseconds
		} else {
			*gpuTimeMs = 0.0f;
		}
	}

	// Clean up the event
	clReleaseEvent(kernel_event);
}

void renderConesOpenCL(
	struct OpenCLContext *ocl,
	struct Camera *camera,
	cl_mem colorBuffer,
	cl_mem depthBuffer,
	float *coneOriginsX,
	float *coneOriginsY,
	float *coneOriginsZ,
	float *coneDirsX,
	float *coneDirsY,
	float *coneDirsZ,
	float *coneFovs,
	int numCones,
	int seekerResolution,
	float *gpuTimeMs) {

	if (!ocl || !ocl->composite_cones_kernel) {
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	cl_int err;
	cl_event evt = NULL;

	size_t coneDataSize = numCones * sizeof(float);
	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_originX, CL_TRUE, 0, coneDataSize, coneOriginsX, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_originY, CL_TRUE, 0, coneDataSize, coneOriginsY, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_originZ, CL_TRUE, 0, coneDataSize, coneOriginsZ, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_dirX, CL_TRUE, 0, coneDataSize, coneDirsX, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_dirY, CL_TRUE, 0, coneDataSize, coneDirsY, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_dirZ, CL_TRUE, 0, coneDataSize, coneDirsZ, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->buffer_cone_fov, CL_TRUE, 0, coneDataSize, coneFovs, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}
	cl_int width = ScreenWidth;
	cl_int height = ScreenHeight;
	cl_float camPosX = camera->ray.origin[0];
	cl_float camPosY = camera->ray.origin[1];
	cl_float camPosZ = camera->ray.origin[2];
	cl_float camDirX = camera->ray.direction[0];
	cl_float camDirY = camera->ray.direction[1];
	cl_float camDirZ = camera->ray.direction[2];
	cl_float fov = camera->fov;
	cl_int coneCount = numCones;
	cl_int seekerRes = seekerResolution;

	err = clSetKernelArg(ocl->composite_cones_kernel, 0, sizeof(cl_mem), &colorBuffer);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 1, sizeof(cl_mem), &depthBuffer);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 2, sizeof(cl_int), &width);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 3, sizeof(cl_int), &height);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 4, sizeof(cl_float), &camPosX);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 5, sizeof(cl_float), &camPosY);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 6, sizeof(cl_float), &camPosZ);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 7, sizeof(cl_float), &camDirX);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 8, sizeof(cl_float), &camDirY);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 9, sizeof(cl_float), &camDirZ);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 10, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 11, sizeof(cl_mem), &ocl->buffer_cone_originX);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 12, sizeof(cl_mem), &ocl->buffer_cone_originY);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 13, sizeof(cl_mem), &ocl->buffer_cone_originZ);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 14, sizeof(cl_mem), &ocl->buffer_cone_dirX);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 15, sizeof(cl_mem), &ocl->buffer_cone_dirY);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 16, sizeof(cl_mem), &ocl->buffer_cone_dirZ);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 17, sizeof(cl_mem), &ocl->buffer_cone_fov);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 18, sizeof(cl_int), &coneCount);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 19, sizeof(cl_mem), &ocl->buffer_seeker_distances_atlas);
	err |= clSetKernelArg(ocl->composite_cones_kernel, 20, sizeof(cl_int), &seekerRes);

	if (err != CL_SUCCESS) {
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	size_t global[2] = {(size_t)ScreenWidth, (size_t)ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->composite_cones_kernel, 2, NULL, global, NULL, 0, NULL, &evt);

	if (err != CL_SUCCESS) {
		if (gpuTimeMs) *gpuTimeMs = 0.0f;
		return;
	}

	clFinish(ocl->queue);

	if (gpuTimeMs != NULL && evt) {
		cl_ulong t0 = 0, t1 = 0;
		clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
		clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
		*gpuTimeMs = (t1 - t0) * 1e-6f;
	}

	if (evt) clReleaseEvent(evt);
}

void antiAliasingOpenCL(struct OpenCLContext *ocl, struct GPUTimings *gpuTimings, struct Camera *camera) {
	cl_int err;
	cl_event kernel_event;

	cl_int mode = 0; // 0 = 3x float, 1 = 4x float, 2 = 1x float

	// Set the input buffer for antialiasing based on render mode
	switch (camera->renderMode) {
	case renderDistance:
		mode = 2;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_distances);
		break;
	case renderVelocity:
		mode = 2;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_velocities_screen);
		break;
	case renderOpacity:
		mode = 2;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_opacities);
		break;
	case renderNormal:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_normals);
		break;
	case renderFluid:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderColor:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderWireframe:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderFireColor:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->FireScreenColors);
		break;
	case renderFireDepth:
		mode = 2;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->FireScreenDistances);
		break;
	case renderFireNormal:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->FireScreenNormals);
		break;
	case renderCompositedColor:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		break;
	case renderCompositedDistance:
		mode = 2;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->CompositedScreenDistances);
		break;
	case renderCompositedNormal:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->CompositedScreenNormals);
		break;
	case renderTemperatures:
		mode = 2;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->FireTemperature);
		break;
	case renderDebugMode:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		break;
	case RENDER_MODE_COUNT:
	default:
		mode = 0;
		err = clSetKernelArg(ocl->antiAliasKernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	}

	if (err != CL_SUCCESS) {
		printf("Error setting antiAlias kernel argument 0: %d\n", err);
		if (gpuTimings) gpuTimings->antiAliasingTime = 0.0f;
		return;
	}

	// Set the remaining kernel arguments
	size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
	err = clSetKernelArg(ocl->antiAliasKernel, 1, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->antiAliasKernel, 2, sizeof(cl_int), &(cl_int){ScreenWidth});
	err |= clSetKernelArg(ocl->antiAliasKernel, 3, sizeof(cl_int), &(cl_int){ScreenHeight});
	err |= clSetKernelArg(ocl->antiAliasKernel, 4, sizeof(cl_int), &mode);
	err |= clSetKernelArg(ocl->antiAliasKernel, 5, sizeof(cl_mem), &ocl->buffer_normals);

	if (camera->advanceAntiAlias) {
		cl_int one = 1;
		err |= clSetKernelArg(ocl->antiAliasKernel, 6, sizeof(cl_int), &one);
	} else {
		cl_int zero = 0;
		err |= clSetKernelArg(ocl->antiAliasKernel, 6, sizeof(cl_int), &zero);
	}

	if (err != CL_SUCCESS) {
		printf("Error setting antiAlias kernel arguments: %d\n", err);
		if (gpuTimings) gpuTimings->antiAliasingTime = 0.0f;
		return;
	}

	// Execute kernel with timing event
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->antiAliasKernel, 2, NULL,
								 global_work_size, NULL, 0, NULL, &kernel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing antiAlias kernel: %d\n", err);
		if (gpuTimings) gpuTimings->antiAliasingTime = 0.0f;
		return;
	}

	clFinish(ocl->queue);

	// Get timing information
	if (gpuTimings != NULL) {
		cl_ulong start_time, end_time;
		err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START,
									  sizeof(start_time), &start_time, NULL);
		err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END,
									   sizeof(end_time), &end_time, NULL);

		if (err == CL_SUCCESS) {
			gpuTimings->antiAliasingTime = (end_time - start_time) * 1e-6f;
		} else {
			gpuTimings->antiAliasingTime = 0.0f;
		}
	}

	clReleaseEvent(kernel_event);
}

const int renderFromMissileSeeker = 1;
const int renderFromMainCamera = 0;

void projectParticlesOpenCL(struct OpenCLContext *ocl, struct PointSOA *particles, struct Camera *camera, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font, struct FireSOA *fireParticles, struct Missiles *missiles) {
	cl_int err;

	// Use pre-allocated buffers instead of malloc
	float *points_data = ocl->host_points_data;
	float *velocities_data = ocl->host_velocities_data;

	for (int i = 0; i < NUM_PARTICLES; i++) {
		points_data[i * 3 + 0] = particles->x[i];
		points_data[i * 3 + 1] = particles->y[i];
		points_data[i * 3 + 2] = particles->z[i];

		velocities_data[i * 3 + 0] = particles->xVelocity[i];
		velocities_data[i * 3 + 1] = particles->yVelocity[i];
		velocities_data[i * 3 + 2] = particles->zVelocity[i];
	}

	bool temp = false;
	float tempVec[3] = {0.0f, 0.0f, 0.0f};

	renderAllMissileFires(ocl, missiles, camera, &gpuTimings->missileRenderingTime);
	renderFireParticles(ocl, fireParticles, camera, &gpuTimings->fireRenderingTime);

	// Write data to GPU buffers
	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_points, CL_TRUE, 0,
							   NUM_PARTICLES * 3 * sizeof(float), points_data, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing points buffer: %d\n", err);
	}

	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_velocities, CL_TRUE, 0,
							   NUM_PARTICLES * 3 * sizeof(float), velocities_data, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error writing velocities buffer: %d\n", err);
	}

	// Clear screen buffers on GPU (INCLUDING NORMALS)
	float zero = 0.0f;
	err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_opacities, &zero, sizeof(float), 0,
							  ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_velocities_screen, &zero, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_normals, &zero, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * 3 * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_screen_colors, &zero, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * 3 * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_distances, &zero, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error clearing buffers: %d\n", err);
	}

	// *** RENDER SKYBOX FIRST (fills background) ***

	renderSkyboxOpenCL(ocl, camera, skyBox, &gpuTimings->renderSkyBoxTime);

	// === ADD TEXT TO RENDER ===
	uint8_t white[3] = {255, 255, 255};
	uint8_t yellow[3] = {255, 255, 0};
	uint8_t red[3] = {255, 0, 0};
	uint8_t green[3] = {0, 255, 0};

	int chart_pos_Y = chartPosY;
	float realFPS = totalTime(gpuTimings) > 0.001f ? (1000.0f / totalTime(gpuTimings)) : 0.0f;
	snprintf(text, sizeof(text), "Real FPS %.0f", realFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y - 15, yellow);

	float skyboxFPS = (gpuTimings->renderSkyBoxTime > 0.001f) ? (1000.0f / gpuTimings->renderSkyBoxTime) : 0.0f;
	snprintf(text, sizeof(text), "Skybox %.0f FPS", skyboxFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	// *** TRIANGLE RENDERING *** (keeping your existing triangle rendering logic)
	if (RENDER_TRIAGES == 0) {
		renderTrianglesOpenCL(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
	} else {
		if (camera->renderMode == 6) {
			// Wireframe rendering
			cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
			cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
			cl_float fov = camera->fov;
			cl_int screen_width = ScreenWidth;
			cl_int screen_height = ScreenHeight;
			cl_int num_triangles = triangles->count;

			// Set vertex calculation arguments
			err = clSetKernelArg(ocl->calculateVertex_kernel, 0, sizeof(cl_mem), &ocl->buffer_triangle_v1);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 1, sizeof(cl_mem), &ocl->buffer_triangle_v2);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 2, sizeof(cl_mem), &ocl->buffer_triangle_v3);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 3, sizeof(cl_mem), &ocl->buffer_triangle_normals);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 4, sizeof(cl_float3), &cam_pos);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 5, sizeof(cl_float3), &cam_dir);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 6, sizeof(cl_float), &fov);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 7, sizeof(cl_int), &screen_width);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 8, sizeof(cl_int), &screen_height);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 9, sizeof(cl_int), &num_triangles);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 10, sizeof(cl_mem), &ocl->buffer_projected_verts);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 11, sizeof(cl_mem), &ocl->buffer_triangle_bboxes);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 12, sizeof(cl_mem), &ocl->buffer_valid_triangles);

			size_t vertex_global_size = triangles->count;
			err = clEnqueueNDRangeKernel(ocl->queue, ocl->calculateVertex_kernel, 1, NULL, &vertex_global_size, NULL, 0, NULL, NULL);
			clFinish(ocl->queue);

#if USE_GPU_LOD == 1
			renderTerrainLOD_GPU(ocl, camera, &gpuTimings->renderTrianglesTime);
#else
			renderWireframeOpenCL(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
#endif
		} else {
#if USE_GPU_LOD == 1
			renderTerrainLOD_GPU(ocl, camera, &gpuTimings->renderTrianglesTime);
#else
			renderTrianglesOpenCL_TwoPass(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
#endif
		}
	}

	float trianglesFPS = (gpuTimings->renderTrianglesTime > 0.001f) ? (1000.0f / gpuTimings->renderTrianglesTime) : 0.0f;
	snprintf(text, sizeof(text), "Triangles %.0f FPS", trianglesFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	// *** Screen Space Projection Kernel and sky box ***
	applyReflectionsOpenCL(ocl, camera, skyBox, &gpuTimings->applyReflectionsTime, false);

	float reflectionsFPS = (gpuTimings->applyReflectionsTime > 0.001f) ? (1000.0f / gpuTimings->applyReflectionsTime) : 0.0f;
	snprintf(text, sizeof(text), "Reflections %.0f FPS", reflectionsFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float blurFPS = (gpuTimings->applyBlurTime > 0.001f) ? (1000.0f / gpuTimings->applyBlurTime) : 0.0f;
	snprintf(text, sizeof(text), "Blur %.0f FPS", blurFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float readbackFPS = (gpuTimings->readBackTime > 0.001f) ? (1000.0f / gpuTimings->readBackTime) : 0.0f;
	snprintf(text, sizeof(text), "Readback %.0f FPS", readbackFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float textFPS = (gpuTimings->renderTextTime > 0.001f) ? (1000.0f / gpuTimings->renderTextTime) : 0.0f;
	snprintf(text, sizeof(text), "Text %.0f FPS", textFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float projectParticlesFPS = (gpuTimings->projectParticlesTime > 0.001f) ? (1000.0f / gpuTimings->projectParticlesTime) : 0.0f;
	snprintf(text, sizeof(text), "Particles %.0f FPS", projectParticlesFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);

	snprintf(text, sizeof(text), "Render Mode: %s", renderModesName[camera->renderMode]);
	addTextOpenCL(ocl, font, text, 5, 5, white);

	if (camera->AntiAlias) {
		snprintf(text, sizeof(text), "AntiAlias Enabled (O)");
		addTextOpenCL(ocl, font, text, 5, 80, white);
		snprintf(text, sizeof(text), "Enabled");
		addTextOpenCL(ocl, font, text, 5, 95, green);
	} else {
		snprintf(text, sizeof(text), "AntiAlias Disabled (O)");
		addTextOpenCL(ocl, font, text, 5, 80, white);
		snprintf(text, sizeof(text), "Disabled");
		addTextOpenCL(ocl, font, text, 5, 95, red);
	}

	if (camera->advanceAntiAlias) {
		snprintf(text, sizeof(text), "Advance AntiAlias Enabled (I)");
		addTextOpenCL(ocl, font, text, 5, 110, white);
		snprintf(text, sizeof(text), "Enabled");
		addTextOpenCL(ocl, font, text, 5, 125, green);
	} else {
		snprintf(text, sizeof(text), "Advance AntiAlias Disabled (I)");
		addTextOpenCL(ocl, font, text, 5, 110, white);
		snprintf(text, sizeof(text), "Disabled");
		addTextOpenCL(ocl, font, text, 5, 125, red);
	}

	int missileUIy = 170;
	for (int i = 0; i < missiles->count; i++) {
		struct Missile *m = missiles->missiles[i];
		uint8_t stateColor[3] = {128, 128, 128};
		const char *stateName = "Idle";

		if (missiles->active[i]) {
			if (m->seeker.lockState == Tracking) {
				stateColor[0] = 0;
				stateColor[1] = 255;
				stateColor[2] = 0;
				stateName = "TRACKING";
			} else if (m->seeker.lockState == Searching) {
				stateColor[0] = 255;
				stateColor[1] = 255;
				stateColor[2] = 0;
				stateName = "SEARCHING";
			} else if (m->seeker.lockState == Lunching) {
				stateColor[0] = 255;
				stateColor[1] = 128;
				stateColor[2] = 0;
				stateName = "LAUNCHING";
			}

			float speed = sqrtf(m->velocity[0] * m->velocity[0] + m->velocity[1] * m->velocity[1] + m->velocity[2] * m->velocity[2]);
			float dist = sqrtf(m->targetPosition[0] * m->targetPosition[0] + m->targetPosition[1] * m->targetPosition[1] + m->targetPosition[2] * m->targetPosition[2]);

			snprintf(text, sizeof(text), "M%d: %s Spd:%.0f/%.0fm/s Fuel:%.0fkg Target Idx:%d", i, stateName, speed, m->maxSpeed, m->fuelMass, m->targetIdx);
			addTextOpenCL(ocl, font, text, 5, missileUIy, stateColor);

			missileUIy += 35;
		}
	}

#ifdef DEBUG
	snprintf(text, sizeof(text), "Cam Pos: %.1f %.1f %.1f", camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]);
	addTextOpenCL(ocl, font, text, 5, 20, white);
	snprintf(text, sizeof(text), "Cam Dir: %.2f %.2f %.2f", camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]);
	addTextOpenCL(ocl, font, text, 5, 35, white);
	snprintf(text, sizeof(text), "Cam FOV: %.1f", camera->fov);
	addTextOpenCL(ocl, font, text, 5, 50, white);
	snprintf(text, sizeof(text), "Total Energy: %.1f", particles->totalEnergy);
	addTextOpenCL(ocl, font, text, 5, 65, white);
#endif

#define renderUI_Separately 1
#if renderUI_Separately == 1
	// clear temp buffer
	err = clEnqueueFillBuffer(
		ocl->queue,
		ocl->cl_ui_texture_buffer_temp,
		&zero,
		sizeof(float), // <=16 and power of two
		0,
		ScreenWidth * ScreenHeight * 3 * sizeof(float),
		0, NULL, NULL);
	CL_ERROR(err, "Filling UI temp buffer");
	clFinish(ocl->queue);

	// use temp buffer for UI
	renderTextOpenCL(ocl, font, &gpuTimings->renderTextTime, &ocl->cl_ui_texture_buffer_temp);
	renderGPUTimings(ocl, gpuTimings, &ocl->cl_ui_texture_buffer_temp);
#else
	renderTextOpenCL(ocl, font, &gpuTimings->renderTextTime, NULL);
	renderGPUTimings(ocl, gpuTimings, NULL);
#endif

	// === PARTICLE PROJECTION (keeping your existing particle rendering) ===
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float3 cam_up = {0.0f, 1.0f, 0.0f};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int num_points = NUM_PARTICLES;
	cl_int particle_radius = PARTICLE_RADIUS * 100.0f;

	// calculate max particle velocity for normalization
	float max_velocity = 0.0f;
	for (int i = 0; i < NUM_PARTICLES; i++) {
		float vx = particles->xVelocity[i];
		float vy = particles->yVelocity[i];
		float vz = particles->zVelocity[i];
		float speed = vx * vx + vy * vy + vz * vz;
		if (speed > max_velocity) {
			max_velocity = speed;
		}
	}

	// Render Bounding Box of Particles Simulation area
	renderBoundingBox(ocl, camera, particles, &gpuTimings->renderBoundingBoxTime);

	err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), &ocl->buffer_points);
	err |= clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), &ocl->buffer_velocities);
	err |= clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->kernel, 3, sizeof(cl_mem), &ocl->buffer_opacities);
	err |= clSetKernelArg(ocl->kernel, 4, sizeof(cl_mem), &ocl->buffer_velocities_screen);
	err |= clSetKernelArg(ocl->kernel, 5, sizeof(cl_mem), &ocl->buffer_normals);
	err |= clSetKernelArg(ocl->kernel, 6, sizeof(cl_float3), &cam_pos);
	err |= clSetKernelArg(ocl->kernel, 7, sizeof(cl_float3), &cam_dir);
	err |= clSetKernelArg(ocl->kernel, 8, sizeof(cl_float3), &cam_up);
	err |= clSetKernelArg(ocl->kernel, 9, sizeof(cl_float), &fov);
	err |= clSetKernelArg(ocl->kernel, 10, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->kernel, 11, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->kernel, 12, sizeof(cl_int), &num_points);
	err |= clSetKernelArg(ocl->kernel, 13, sizeof(cl_int), &particle_radius);
	err |= clSetKernelArg(ocl->kernel, 14, sizeof(cl_float), &max_velocity);

	// TODO: Distance normalization will be applied later after rendering of triangles and other staff

	if (err != CL_SUCCESS) {
		printf("Error setting particle kernel arguments: %d\n", err);
	}

	// Execute particle kernel
	size_t global_work_size = NUM_PARTICLES;
	cl_event kernel_event;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel, 1, NULL, &global_work_size, NULL, 0, NULL, &kernel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing particle kernel: %d\n", err);
	}

	clFinish(ocl->queue);

	// Get particle kernel timing
	cl_ulong start_time, end_time;
	err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	err |= clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end_time), &end_time, NULL);
	if (err == CL_SUCCESS) {
		gpuTimings->projectParticlesTime = (end_time - start_time) * 1e-6f;
	}
	clReleaseEvent(kernel_event);

	// === BLUR PROCESSING (keeping your existing blur logic) ===
	// Move variable declarations to the top to avoid VLA issues
	cl_mem s_dist_src, s_dist_dst, s_opac_src, s_opac_dst;
	cl_int blur_kernel_size = 2;
	cl_float blur_sigma_range = 15.0f;
	cl_float blur_sigma_spatial = 2.5f;
	int blur_passes = MAX_BLUR_PASSES;

	cl_event blur_events[MAX_BLUR_PASSES];
	int event_count = 0;

	if (blur_passes > 0 && blur_passes <= MAX_BLUR_PASSES) {
		s_dist_src = ocl->buffer_distances;
		s_opac_src = ocl->buffer_opacities;

		for (int pass = 0; pass < blur_passes; ++pass) {
			if (pass % 2 == 0) {
				s_dist_dst = ocl->buffer_distances_temp;
				s_opac_dst = ocl->buffer_opacities_temp;
			} else {
				s_dist_dst = ocl->buffer_distances;
				s_opac_dst = ocl->buffer_opacities;
			}

			err = clSetKernelArg(ocl->blur_kernel, 0, sizeof(cl_mem), &s_dist_src);
			err |= clSetKernelArg(ocl->blur_kernel, 1, sizeof(cl_mem), &s_opac_src);
			err |= clSetKernelArg(ocl->blur_kernel, 2, sizeof(cl_mem), &s_dist_dst);
			err |= clSetKernelArg(ocl->blur_kernel, 3, sizeof(cl_mem), &s_opac_dst);
			err |= clSetKernelArg(ocl->blur_kernel, 4, sizeof(cl_int), &screen_width);
			err |= clSetKernelArg(ocl->blur_kernel, 5, sizeof(cl_int), &screen_height);
			err |= clSetKernelArg(ocl->blur_kernel, 6, sizeof(cl_int), &blur_kernel_size);
			err |= clSetKernelArg(ocl->blur_kernel, 7, sizeof(cl_float), &blur_sigma_range);
			err |= clSetKernelArg(ocl->blur_kernel, 8, sizeof(cl_float), &blur_sigma_spatial);

			if (err != CL_SUCCESS) {
				printf("Error setting blur kernel arguments for pass %d: %d\n", pass, err);
			}

			size_t blur_global_work_size[2] = {ScreenWidth, ScreenHeight};
			err = clEnqueueNDRangeKernel(ocl->queue, ocl->blur_kernel, 2, NULL,
										 blur_global_work_size, NULL, 0, NULL,
										 &blur_events[event_count]);
			if (err != CL_SUCCESS) {
				printf("Error executing blur kernel for pass %d: %d\n", pass, err);
			}

			event_count++;
			s_dist_src = s_dist_dst;
			s_opac_src = s_opac_dst;
		}

		clFinish(ocl->queue);

		if (event_count > 0) {
			cl_ulong first_start, last_end;
			err = clGetEventProfilingInfo(blur_events[0], CL_PROFILING_COMMAND_START,
										  sizeof(first_start), &first_start, NULL);
			err |= clGetEventProfilingInfo(blur_events[event_count - 1], CL_PROFILING_COMMAND_END,
										   sizeof(last_end), &last_end, NULL);
			if (err == CL_SUCCESS) {
				gpuTimings->applyBlurTime = (last_end - first_start) * 1e-6f;
			}

			for (int i = 0; i < event_count; i++) {
				clReleaseEvent(blur_events[i]);
			}
		}
	} else {
		// If no blur or too many passes, use original buffer
		s_dist_src = ocl->buffer_distances;
	}

	cl_mem final_blurred_distances_buf = s_dist_src;

	// === NORMALS CALCULATION ===
	err = clSetKernelArg(ocl->normals_kernel, 0, sizeof(cl_mem), &final_blurred_distances_buf);
	err |= clSetKernelArg(ocl->normals_kernel, 1, sizeof(cl_mem), &ocl->buffer_normals);
	err |= clSetKernelArg(ocl->normals_kernel, 2, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->normals_kernel, 3, sizeof(cl_int), &screen_height);

	size_t normals_global_work_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->normals_kernel, 2, NULL, normals_global_work_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error executing normals kernel: %d\n", err);
	}

	//  === COPY UI TO OPENGL UI TEXTURE ===

#if renderUI_Separately == 1
	float tmp = 0.0f;
	launchOverlayImageOpenCL(
		ocl,
		ocl->cl_ui_texture_buffer_temp,
		ocl->buffer_seeker_distances,
		ScreenWidth,
		ScreenHeight,
		MISSILE_SEEKER_SIZE,
		MISSILE_SEEKER_SIZE,
		0,
		2,
		&tmp,
		ScreenWidth - MISSILE_SEEKER_SIZE - 10,
		10);
	// Acquire UI texture
	err = clEnqueueAcquireGLObjects(ocl->queue, 1, &ocl->cl_ui_texture_buffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error acquiring UI GL texture: %d\n", err);
		return;
	}

	// Copy UI buffer to OpenGL UI texture
	err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->cl_ui_texture_buffer_temp);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 1, sizeof(cl_mem), &ocl->cl_ui_texture_buffer);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 2, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 3, sizeof(cl_int), &screen_height);
	cl_int mode_ui = 0; // 1 = 4x float for UI
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 4, sizeof(cl_int), &mode_ui);

	if (err != CL_SUCCESS) {
		printf("Error setting copyToTexture kernel args for UI: %d\n", err);
	}

	size_t ui_global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->copyToTexture_kernel, 2, NULL,
								 ui_global_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error executing copyToTexture kernel for UI: %d\n", err);
	}

	err = clEnqueueReleaseGLObjects(ocl->queue, 1, &ocl->cl_ui_texture_buffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error releasing UI GL texture: %d\n", err);
		return;
	}
	clFinish(ocl->queue);
#endif

	compositeBuffersOpenCL(
		ocl,
		ocl->buffer_screen_colors,
		ocl->buffer_distances,
		ocl->buffer_normals,
		ocl->FireScreenAlphas,
		0,
		ocl->FireScreenColors,
		ocl->FireScreenDistances,
		ocl->FireScreenNormals,
		ocl->FireScreenAlphas,
		1,
		ocl->CompositedScreenColors,
		ocl->CompositedScreenDistances,
		ocl->CompositedScreenNormals,
		&gpuTimings->compositingTime);

	filterOverlapOpenCL(
		ocl,
		ocl->FireScreenAlphas,
		ocl->FireScreenDistances,
		ocl->buffer_distances,
		ocl->FireTemperature, // reuse buffer
		2);

	if (camera->renderMode == renderCompositedColor || camera->renderMode == renderDebugMode) {
		applyReflectionsOpenCL(ocl, camera, skyBox, &gpuTimings->applyReflectionsTime, true);
	}

	if (camera->AntiAlias == true) {
		antiAliasingOpenCL(ocl, gpuTimings, camera);
	}

	if (camera->renderMode == renderDebugMode) {
		float temp = 0.0f;
		renderConesOpenCL(
			ocl,
			camera,
			ocl->CompositedScreenColors,
			ocl->buffer_distances,
			missiles->coneOriginsX,
			missiles->coneOriginsY,
			missiles->coneOriginsZ,
			missiles->coneDirsX,
			missiles->coneDirsY,
			missiles->coneDirsZ,
			missiles->coneFovs,
			missiles->activeCount,
			MISSILE_SEEKER_SIZE,
			&temp);
	}

	// === COPY FINAL RESULT TO OPENGL TEXTURE ===

	// === ACQUIRE OPENGL TEXTURE FOR OPENCL USE ===
	err = clEnqueueAcquireGLObjects(ocl->queue, 1, &ocl->cl_texture_buffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error acquiring GL texture: %d\n", err);
		return;
	}

	cl_int screen_width_arg = ScreenWidth;
	cl_int screen_height_arg = ScreenHeight;

	cl_int mode = 0; // 0 = 3x float, 1 = 4x float, 2 = 1x float

	switch (camera->renderMode) {
	case renderDistance:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_distances);
		break;
	case renderVelocity:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_velocities_screen);
		break;
	case renderOpacity:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_opacities);
		break;
	case renderNormal:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_normals);
		break;
	case renderFluid:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderColor:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderWireframe:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderFireColor:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireScreenColors);
		break;
	case renderFireDepth:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireScreenDistances);
		break;
	case renderFireNormal:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireScreenNormals);
		break;
	case renderCompositedColor:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		break;
	case renderCompositedDistance:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenDistances);
		break;
	case renderCompositedNormal:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenNormals);
		break;
	case renderTemperatures:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireTemperature);
		break;
	case renderDebugMode:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		break;
	case RENDER_MODE_COUNT:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	}

	err |= clSetKernelArg(ocl->copyToTexture_kernel, 1, sizeof(cl_mem), &ocl->cl_texture_buffer);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 2, sizeof(cl_int), &screen_width_arg);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 3, sizeof(cl_int), &screen_height_arg);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 4, sizeof(cl_int), &mode);

	if (err != CL_SUCCESS) {
		printf("Error setting copyToTexture kernel args: %d\n", err);
	}

	size_t copy_global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->copyToTexture_kernel, 2, NULL,
								 copy_global_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error executing copyToTexture kernel: %d\n", err);
	}
	clEnqueueReleaseGLObjects(ocl->queue, 1, &ocl->cl_texture_buffer, 0, NULL, NULL);
	clFinish(ocl->queue);
}

void projectParticlesOpenCL_NoWater(struct OpenCLContext *ocl, struct Camera *camera, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font, struct FireSOA *fireParticles, struct Missiles *missiles, struct IRSearchAndTrack *irst) {
	cl_int err;
	float zero = 0.0f;

	// Clear screen buffers on GPU
	err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_normals, &zero, sizeof(float), 0,
							  ScreenWidth * ScreenHeight * 3 * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_screen_colors, &zero, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * 3 * sizeof(float), 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_distances, &zero, sizeof(float), 0,
							   ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		printf("Error clearing buffers: %d\n", err);
	}

	// Render missile fires
	renderAllMissileFires(ocl, missiles, camera, &gpuTimings->missileRenderingTime);
	renderFireParticles(ocl, fireParticles, camera, &gpuTimings->fireRenderingTime);

	// Render skybox
	renderSkyboxOpenCL(ocl, camera, skyBox, &gpuTimings->renderSkyBoxTime);

	// Setup UI text
	uint8_t white[3] = {255, 255, 255};
	uint8_t yellow[3] = {255, 255, 0};
	uint8_t red[3] = {255, 0, 0};
	uint8_t green[3] = {0, 255, 0};

	// IRST Controls Help
	uint8_t gray[3] = {180, 180, 180};
	snprintf(text, sizeof(text), "IRST: TAB=Next | BACKSPACE=Clear | Targets: %d", irst->targetCount);
	addTextOpenCL(ocl, font, text, 10, 10, gray);

	int chart_pos_Y = chartPosY;
	float realFPS = totalTime(gpuTimings) > 0.001f ? (1000.0f / totalTime(gpuTimings)) : 0.0f;
	snprintf(text, sizeof(text), "Real FPS %.0f", realFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y - 15, yellow);

	for (int i = 0; i < irst->targetCount; i++) {
		// Calculate distance to target
		float dx = irst->targetPositionX[i] - camera->ray.origin[0];
		float dy = irst->targetPositionY[i] - camera->ray.origin[1];
		float dz = irst->targetPositionZ[i] - camera->ray.origin[2];
		float distance = sqrtf(dx * dx + dy * dy + dz * dz);

		// Color based on temperature (cool to hot)
		int r = 255, g = 0, b = 0;
		if (irst->targetTemperature[i] < 500.0f) {
			r = 0;
			g = 255;
			b = 0; // Green - cold
		} else if (irst->targetTemperature[i] < 1000.0f) {
			r = 255;
			g = 255;
			b = 0; // Yellow - warm
		} else {
			r = 255;
			g = 0;
			b = 0; // Red - hot
		}

		uint8_t color[3] = {r, g, b};

		// Selected target gets special treatment
		if (irst->selectedTargetId >= 0 && i == irst->selectedTargetId) {
			snprintf(text, sizeof(text), "[T%d]", i + 1);
			color[0] = 0;
			color[1] = 255;
			color[2] = 255; // Cyan for selected
		} else {
			snprintf(text, sizeof(text), "x");
		}
		addTextOpenCL(ocl, font, text, irst->targetScreenX[i], irst->targetScreenY[i], color);
	}

	// Display detailed info for selected target
	if (irst->selectedTargetId >= 0 && irst->selectedTargetId < irst->targetCount) {
		int idx = irst->selectedTargetId;
		float dx = irst->targetPositionX[idx] - camera->ray.origin[0];
		float dy = irst->targetPositionY[idx] - camera->ray.origin[1];
		float dz = irst->targetPositionZ[idx] - camera->ray.origin[2];
		float distance = sqrtf(dx * dx + dy * dy + dz * dz);

		uint8_t cyan[3] = {0, 255, 255};
		snprintf(text, sizeof(text), "TARGET %d SELECTED", idx + 1);
		addTextOpenCL(ocl, font, text, 10, 100, cyan);
		snprintf(text, sizeof(text), "Distance: %.0fm", distance);
		addTextOpenCL(ocl, font, text, 10, 115, white);
		snprintf(text, sizeof(text), "Temperature: %.0fK", irst->targetTemperature[idx]);
		addTextOpenCL(ocl, font, text, 10, 130, white);
		snprintf(text, sizeof(text), "Position: (%.0f, %.0f, %.0f)",
				 irst->targetPositionX[idx], irst->targetPositionY[idx], irst->targetPositionZ[idx]);
		addTextOpenCL(ocl, font, text, 10, 145, white);
	}

	float skyboxFPS = (gpuTimings->renderSkyBoxTime > 0.001f) ? (1000.0f / gpuTimings->renderSkyBoxTime) : 0.0f;
	snprintf(text, sizeof(text), "Skybox %.0f FPS", skyboxFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	// Render triangles
	if (RENDER_TRIAGES == 0) {
		renderTrianglesOpenCL(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
	} else {
		if (camera->renderMode == 6) {
			// Wireframe rendering
			cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
			cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
			cl_float fov = camera->fov;
			cl_int screen_width = ScreenWidth;
			cl_int screen_height = ScreenHeight;
			cl_int num_triangles = triangles->count;

			err = clSetKernelArg(ocl->calculateVertex_kernel, 0, sizeof(cl_mem), &ocl->buffer_triangle_v1);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 1, sizeof(cl_mem), &ocl->buffer_triangle_v2);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 2, sizeof(cl_mem), &ocl->buffer_triangle_v3);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 3, sizeof(cl_mem), &ocl->buffer_triangle_normals);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 4, sizeof(cl_float3), &cam_pos);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 5, sizeof(cl_float3), &cam_dir);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 6, sizeof(cl_float), &fov);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 7, sizeof(cl_int), &screen_width);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 8, sizeof(cl_int), &screen_height);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 9, sizeof(cl_int), &num_triangles);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 10, sizeof(cl_mem), &ocl->buffer_projected_verts);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 11, sizeof(cl_mem), &ocl->buffer_triangle_bboxes);
			err |= clSetKernelArg(ocl->calculateVertex_kernel, 12, sizeof(cl_mem), &ocl->buffer_valid_triangles);

			size_t vertex_global_size = triangles->count;
			err = clEnqueueNDRangeKernel(ocl->queue, ocl->calculateVertex_kernel, 1, NULL, &vertex_global_size, NULL, 0, NULL, NULL);
			clFinish(ocl->queue);

#if USE_GPU_LOD == 1
			renderTerrainLOD_GPU(ocl, camera, &gpuTimings->renderTrianglesTime);
#else
			renderWireframeOpenCL(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
#endif
		} else {
#if USE_GPU_LOD == 1
			renderTerrainLOD_GPU(ocl, camera, &gpuTimings->renderTrianglesTime);
#else
			renderTrianglesOpenCL_TwoPass(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
#endif
		}
	}

	float trianglesFPS = (gpuTimings->renderTrianglesTime > 0.001f) ? (1000.0f / gpuTimings->renderTrianglesTime) : 0.0f;
	snprintf(text, sizeof(text), "Triangles %.0f FPS", trianglesFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	// Apply reflections
	applyReflectionsOpenCL(ocl, camera, skyBox, &gpuTimings->applyReflectionsTime, false);

	float reflectionsFPS = (gpuTimings->applyReflectionsTime > 0.001f) ? (1000.0f / gpuTimings->applyReflectionsTime) : 0.0f;
	snprintf(text, sizeof(text), "Reflections %.0f FPS", reflectionsFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float blurFPS = (gpuTimings->applyBlurTime > 0.001f) ? (1000.0f / gpuTimings->applyBlurTime) : 0.0f;
	snprintf(text, sizeof(text), "Blur %.0f FPS", blurFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float readbackFPS = (gpuTimings->readBackTime > 0.001f) ? (1000.0f / gpuTimings->readBackTime) : 0.0f;
	snprintf(text, sizeof(text), "Readback %.0f FPS", readbackFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	float textFPS = (gpuTimings->renderTextTime > 0.001f) ? (1000.0f / gpuTimings->renderTextTime) : 0.0f;
	snprintf(text, sizeof(text), "Text %.0f FPS", textFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	snprintf(text, sizeof(text), "Render Mode: %s", renderModesName[camera->renderMode]);
	addTextOpenCL(ocl, font, text, 5, 5, white);

	if (camera->AntiAlias) {
		snprintf(text, sizeof(text), "AntiAlias Enabled (O)");
		addTextOpenCL(ocl, font, text, 5, 80, white);
		snprintf(text, sizeof(text), "Enabled");
		addTextOpenCL(ocl, font, text, 5, 95, green);
	} else {
		snprintf(text, sizeof(text), "AntiAlias Disabled (O)");
		addTextOpenCL(ocl, font, text, 5, 80, white);
		snprintf(text, sizeof(text), "Disabled");
		addTextOpenCL(ocl, font, text, 5, 95, red);
	}

	if (camera->advanceAntiAlias) {
		snprintf(text, sizeof(text), "Advance AntiAlias Enabled (I)");
		addTextOpenCL(ocl, font, text, 5, 110, white);
		snprintf(text, sizeof(text), "Enabled");
		addTextOpenCL(ocl, font, text, 5, 125, green);
	} else {
		snprintf(text, sizeof(text), "Advance AntiAlias Disabled (I)");
		addTextOpenCL(ocl, font, text, 5, 110, white);
		snprintf(text, sizeof(text), "Disabled");
		addTextOpenCL(ocl, font, text, 5, 125, red);
	}

	// Missile UI
	int missileUIy = 170;
	for (int i = 0; i < missiles->count; i++) {
		struct Missile *m = missiles->missiles[i];
		uint8_t stateColor[3] = {128, 128, 128};
		const char *stateName = "Idle";

		if (missiles->active[i]) {
			if (m->seeker.lockState == Tracking) {
				stateColor[0] = 0;
				stateColor[1] = 255;
				stateColor[2] = 0;
				stateName = "TRACKING";
			} else if (m->seeker.lockState == Searching) {
				stateColor[0] = 255;
				stateColor[1] = 255;
				stateColor[2] = 0;
				stateName = "SEARCHING";
			} else if (m->seeker.lockState == Lunching) {
				stateColor[0] = 255;
				stateColor[1] = 128;
				stateColor[2] = 0;
				stateName = "LAUNCHING";
			}

			float speed = sqrtf(m->velocity[0] * m->velocity[0] + m->velocity[1] * m->velocity[1] + m->velocity[2] * m->velocity[2]);

			snprintf(text, sizeof(text), "M%d: %s Spd:%.0f/%.0fm/s Fuel:%.0fkg Target Idx:%d", i, stateName, speed, m->maxSpeed, m->fuelMass, m->targetIdx);
			addTextOpenCL(ocl, font, text, 5, missileUIy, stateColor);

			missileUIy += 35;
		}
	}

#if renderUI_Separately == 1
	err = clEnqueueFillBuffer(
		ocl->queue,
		ocl->cl_ui_texture_buffer_temp,
		&zero,
		sizeof(float),
		0,
		ScreenWidth * ScreenHeight * 3 * sizeof(float),
		0, NULL, NULL);
	CL_ERROR(err, "Filling UI temp buffer");
	clFinish(ocl->queue);

	renderTextOpenCL(ocl, font, &gpuTimings->renderTextTime, &ocl->cl_ui_texture_buffer_temp);
	renderGPUTimings(ocl, gpuTimings, &ocl->cl_ui_texture_buffer_temp);
#else
	renderTextOpenCL(ocl, font, &gpuTimings->renderTextTime, NULL);
	renderGPUTimings(ocl, gpuTimings, NULL);
#endif

	// Skip particle rendering entirely when WATER_SIMULATION is 0

	// Composite fire buffers
	compositeBuffersOpenCL(
		ocl,
		ocl->buffer_screen_colors,
		ocl->buffer_distances,
		ocl->buffer_normals,
		ocl->FireScreenAlphas,
		0,
		ocl->FireScreenColors,
		ocl->FireScreenDistances,
		ocl->FireScreenNormals,
		ocl->FireScreenAlphas,
		1,
		ocl->CompositedScreenColors,
		ocl->CompositedScreenDistances,
		ocl->CompositedScreenNormals,
		&gpuTimings->compositingTime);

	filterOverlapOpenCL(
		ocl,
		ocl->FireScreenAlphas,
		ocl->FireScreenDistances,
		ocl->buffer_distances,
		ocl->FireTemperature,
		2);

	if (camera->renderMode == renderCompositedColor || camera->renderMode == renderDebugMode) {
		applyReflectionsOpenCL(ocl, camera, skyBox, &gpuTimings->applyReflectionsTime, true);
	}

	if (camera->AntiAlias == true) {
		antiAliasingOpenCL(ocl, gpuTimings, camera);
	}

	if (camera->renderMode == renderDebugMode) {
		float temp = 0.0f;
		renderConesOpenCL(
			ocl,
			camera,
			ocl->CompositedScreenColors,
			ocl->buffer_distances,
			missiles->coneOriginsX,
			missiles->coneOriginsY,
			missiles->coneOriginsZ,
			missiles->coneDirsX,
			missiles->coneDirsY,
			missiles->coneDirsZ,
			missiles->coneFovs,
			missiles->activeCount,
			MISSILE_SEEKER_SIZE,
			&temp);
	}

#if renderUI_Separately == 1
	float tmp = 0.0f;
	launchOverlayImageOpenCL(
		ocl,
		ocl->cl_ui_texture_buffer_temp,
		ocl->buffer_seeker_distances,
		ScreenWidth,
		ScreenHeight,
		MISSILE_SEEKER_SIZE,
		MISSILE_SEEKER_SIZE,
		0,
		2,
		&tmp,
		ScreenWidth - MISSILE_SEEKER_SIZE - 10,
		10);

	err = clEnqueueAcquireGLObjects(ocl->queue, 1, &ocl->cl_ui_texture_buffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error acquiring UI GL texture: %d\n", err);
		return;
	}

	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->cl_ui_texture_buffer_temp);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 1, sizeof(cl_mem), &ocl->cl_ui_texture_buffer);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 2, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 3, sizeof(cl_int), &screen_height);
	cl_int mode_ui = 0;
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 4, sizeof(cl_int), &mode_ui);

	if (err != CL_SUCCESS) {
		printf("Error setting copyToTexture kernel args for UI: %d\n", err);
	}

	size_t ui_global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->copyToTexture_kernel, 2, NULL,
								 ui_global_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error executing copyToTexture kernel for UI: %d\n", err);
	}

	err = clEnqueueReleaseGLObjects(ocl->queue, 1, &ocl->cl_ui_texture_buffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error releasing UI GL texture: %d\n", err);
		return;
	}
	clFinish(ocl->queue);
#endif

	// Copy final result to OpenGL texture
	err = clEnqueueAcquireGLObjects(ocl->queue, 1, &ocl->cl_texture_buffer, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error acquiring GL texture: %d\n", err);
		return;
	}

	cl_int screen_width_arg = ScreenWidth;
	cl_int screen_height_arg = ScreenHeight;
	cl_int mode = 0;

	switch (camera->renderMode) {
	case renderDistance:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_distances);
		break;
	case renderVelocity:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_velocities_screen);
		break;
	case renderOpacity:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_opacities);
		break;
	case renderNormal:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_normals);
		break;
	case renderFluid:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderColor:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderWireframe:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	case renderFireColor:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireScreenColors);
		break;
	case renderFireDepth:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireScreenDistances);
		break;
	case renderFireNormal:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireScreenNormals);
		break;
	case renderCompositedColor:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		break;
	case renderDebugMode:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenColors);
		break;
	case renderCompositedNormal:
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenNormals);
		break;
	case renderCompositedDistance:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->CompositedScreenDistances);
		break;
	case renderTemperatures:
		mode = 2;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->FireTemperature);
		break;
	case RENDER_MODE_COUNT:
		// Not a valid render mode, use default
		mode = 0;
		err = clSetKernelArg(ocl->copyToTexture_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
		break;
	}

	err |= clSetKernelArg(ocl->copyToTexture_kernel, 1, sizeof(cl_mem), &ocl->cl_texture_buffer);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 2, sizeof(cl_int), &screen_width_arg);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 3, sizeof(cl_int), &screen_height_arg);
	err |= clSetKernelArg(ocl->copyToTexture_kernel, 4, sizeof(cl_int), &mode);

	if (err != CL_SUCCESS) {
		printf("Error setting copyToTexture kernel args: %d\n", err);
	}

	size_t copy_global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->copyToTexture_kernel, 2, NULL,
								 copy_global_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error executing copyToTexture kernel: %d\n", err);
	}
	clEnqueueReleaseGLObjects(ocl->queue, 1, &ocl->cl_texture_buffer, 0, NULL, NULL);
	clFinish(ocl->queue);
}

void cleanupOpenCL(struct OpenCLContext *ocl) {
	// Free host memory (INCLUDING NORMALS)
	if (ocl->host_points_data) free(ocl->host_points_data);
	if (ocl->host_velocities_data) free(ocl->host_velocities_data);
	if (ocl->host_distances_result) free(ocl->host_distances_result);
	if (ocl->host_opacities_result) free(ocl->host_opacities_result);
	if (ocl->host_velocities_result) free(ocl->host_velocities_result);
	if (ocl->host_normals_result) free(ocl->host_normals_result);
	if (ocl->host_screen_colors_result) free(ocl->host_screen_colors_result);

	// Free OpenCL resources (INCLUDING NORMALS AND TRIANGLES)
	if (ocl->buffer_points) clReleaseMemObject(ocl->buffer_points);
	if (ocl->buffer_velocities) clReleaseMemObject(ocl->buffer_velocities);
	if (ocl->buffer_distances) clReleaseMemObject(ocl->buffer_distances);
	if (ocl->buffer_opacities) clReleaseMemObject(ocl->buffer_opacities);
	if (ocl->buffer_velocities_screen) clReleaseMemObject(ocl->buffer_velocities_screen);
	if (ocl->buffer_normals) clReleaseMemObject(ocl->buffer_normals);
	if (ocl->buffer_triangle_v1) clReleaseMemObject(ocl->buffer_triangle_v1);
	if (ocl->buffer_triangle_v2) clReleaseMemObject(ocl->buffer_triangle_v2);
	if (ocl->buffer_triangle_v3) clReleaseMemObject(ocl->buffer_triangle_v3);
	if (ocl->buffer_triangle_normals) clReleaseMemObject(ocl->buffer_triangle_normals);
	if (ocl->buffer_triangle_colors) clReleaseMemObject(ocl->buffer_triangle_colors);
	if (ocl->buffer_screen_colors) clReleaseMemObject(ocl->buffer_screen_colors);
	if (ocl->buffer_distances_temp) clReleaseMemObject(ocl->buffer_distances_temp);
	if (ocl->buffer_opacities_temp) clReleaseMemObject(ocl->buffer_opacities_temp);
	// triangle properties buffers
	if (ocl->buffer_triangle_roughness) clReleaseMemObject(ocl->buffer_triangle_roughness);
	if (ocl->buffer_triangle_metallic) clReleaseMemObject(ocl->buffer_triangle_metallic);
	if (ocl->buffer_triangle_emission) clReleaseMemObject(ocl->buffer_triangle_emission);
	if (ocl->buffer_projected_verts) clReleaseMemObject(ocl->buffer_projected_verts);
	if (ocl->buffer_triangle_bboxes) clReleaseMemObject(ocl->buffer_triangle_bboxes);
	if (ocl->buffer_valid_triangles) clReleaseMemObject(ocl->buffer_valid_triangles);
	if (ocl->buffer_tileLists) clReleaseMemObject(ocl->buffer_tileLists);
	if (ocl->buffer_tileListCounts) clReleaseMemObject(ocl->buffer_tileListCounts);

	if (ocl->buffer_skybox_top) clReleaseMemObject(ocl->buffer_skybox_top);
	if (ocl->buffer_skybox_bottom) clReleaseMemObject(ocl->buffer_skybox_bottom);
	if (ocl->buffer_skybox_left) clReleaseMemObject(ocl->buffer_skybox_left);
	if (ocl->buffer_skybox_right) clReleaseMemObject(ocl->buffer_skybox_right);
	if (ocl->buffer_skybox_front) clReleaseMemObject(ocl->buffer_skybox_front);
	if (ocl->buffer_skybox_back) clReleaseMemObject(ocl->buffer_skybox_back);

	if (ocl->buffer_seeker_distances) clReleaseMemObject(ocl->buffer_seeker_distances);
	ocl->mapped_seeker_distances = NULL;

	if (ocl->buffer_cone_originX) clReleaseMemObject(ocl->buffer_cone_originX);
	if (ocl->buffer_cone_originY) clReleaseMemObject(ocl->buffer_cone_originY);
	if (ocl->buffer_cone_originZ) clReleaseMemObject(ocl->buffer_cone_originZ);
	if (ocl->buffer_cone_dirX) clReleaseMemObject(ocl->buffer_cone_dirX);
	if (ocl->buffer_cone_dirY) clReleaseMemObject(ocl->buffer_cone_dirY);
	if (ocl->buffer_cone_dirZ) clReleaseMemObject(ocl->buffer_cone_dirZ);

	if (ocl->kernel) clReleaseKernel(ocl->kernel);
	if (ocl->skybox_kernel) clReleaseKernel(ocl->skybox_kernel);
	if (ocl->triangle_kernel) clReleaseKernel(ocl->triangle_kernel);
	if (ocl->blur_kernel) clReleaseKernel(ocl->blur_kernel);
	if (ocl->normals_kernel) clReleaseKernel(ocl->normals_kernel);
	if (ocl->applyReflections_kernel) clReleaseKernel(ocl->applyReflections_kernel);
	if (ocl->composite_cones_kernel) clReleaseKernel(ocl->composite_cones_kernel);
	if (ocl->calculateVertex_kernel) clReleaseKernel(ocl->calculateVertex_kernel);
	if (ocl->tileCulling_kernel) clReleaseKernel(ocl->tileCulling_kernel);
	if (ocl->shadePixels_kernel) clReleaseKernel(ocl->shadePixels_kernel);

	if (ocl->program) clReleaseProgram(ocl->program);
	if (ocl->queue) clReleaseCommandQueue(ocl->queue);
	if (ocl->context) clReleaseContext(ocl->context);

	// Release anti-aliasing kernel
	if (ocl->antiAliasKernel) clReleaseKernel(ocl->antiAliasKernel);
	if (ocl->kernel) clReleaseKernel(ocl->kernel);
	if (ocl->skybox_kernel) clReleaseKernel(ocl->skybox_kernel);
}

void my_file_reader(void *ctx, const char *filename, int is_mtl, const char *obj_filename, char **buf, size_t *len) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open file %s\n", filename);
		*buf = NULL;
		*len = 0;
		return;
	}

	fseek(file, 0, SEEK_END);
	*len = ftell(file);
	fseek(file, 0, SEEK_SET);

	*buf = (char *)malloc(*len + 1);
	if (*buf) {
		size_t read_size = fread(*buf, 1, *len, file);
		if (read_size != *len) {
			printf("Warning: Could not read entire file %s\n", filename);
		}
		(*buf)[*len] = '\0';
	} else {
		printf("Error: Could not allocate memory for file %s\n", filename);
		*len = 0;
	}

	fclose(file);
}

float rand_01() {
	return (float)rand() / (float)RAND_MAX;
}

void writeFileTriangles(const char *filename, struct Triangles *triangles) {
	FILE *file = fopen(filename, "wb");
	if (!file) {
		printf("Error: Could not open file %s for writing\n", filename);
		return;
	}

// Triangle size: 3 vertices (36) + normal (12) + roughness (4) + metallic (4) + emission (4) + colors (12) + index (4) = 76 bytes
#define SIZE_OF_TRIANGLE 76
	uint32_t fileSize = 8 + triangles->count * SIZE_OF_TRIANGLE; // 8 bytes header (file size + triangle count)
	uint32_t triangleStructSize = SIZE_OF_TRIANGLE;				 // Create a variable to hold the value

	fwrite(&fileSize, sizeof(uint32_t), 1, file);			// Write file size
	fwrite(&triangleStructSize, sizeof(uint32_t), 1, file); // Write triangle struct size (fixed)

	for (int i = 0; i < triangles->count; i++) {
		int idx = i * 3;

		// Write vertices (36 bytes)
		fwrite(&triangles->v1[idx], sizeof(float), 3, file); // 12 bytes
		fwrite(&triangles->v2[idx], sizeof(float), 3, file); // 12 bytes
		fwrite(&triangles->v3[idx], sizeof(float), 3, file); // 12 bytes

		// Write normals (12 bytes)
		fwrite(&triangles->normals[idx], sizeof(float), 3, file); // 12 bytes

		// Write material properties (12 bytes)
		fwrite(&triangles->Roughness[i], sizeof(float), 1, file); // 4 bytes
		fwrite(&triangles->Metallic[i], sizeof(float), 1, file);  // 4 bytes
		fwrite(&triangles->Emission[i], sizeof(float), 1, file);  // 4 bytes

		// Write colors (12 bytes)
		fwrite(&triangles->colors[idx], sizeof(float), 3, file); // 12 bytes

		// Write triangle index (4 bytes)
		uint32_t triangleIndex = i;
		fwrite(&triangleIndex, sizeof(uint32_t), 1, file); // 4 bytes
	}

	fclose(file);
	printf("Triangles written to %s successfully\n", filename);
	printf("File size: %u bytes\n", fileSize);
	printf("Triangle count: %d\n", triangles->count);
}

void updateBBox(float x, float y, float z, float minBB[3], float maxBB[3]) {
	if (x < minBB[0]) minBB[0] = x;
	if (y < minBB[1]) minBB[1] = y;
	if (z < minBB[2]) minBB[2] = z;

	if (x > maxBB[0]) maxBB[0] = x;
	if (y > maxBB[1]) maxBB[1] = y;
	if (z > maxBB[2]) maxBB[2] = z;
}

void readFileTriangles(const char *filename, struct Triangles *triangles, float scale) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open file %s for reading\n", filename);
		return;
	}

	uint32_t fileSize, triangleStructSize;
	fread(&fileSize, sizeof(uint32_t), 1, file);		   // Read file size
	fread(&triangleStructSize, sizeof(uint32_t), 1, file); // Read triangle struct size

	int triangleCount = (fileSize - 8) / triangleStructSize; // Calculate number of triangles
	printf("Reading %d triangles from %s with scale factor %.2f\n", triangleCount, filename, scale);

	// Check if we exceed the maximum number of triangles
	if (triangleCount > NUMBER_OF_TRIANGLES) {
		printf("Warning: File contains %d triangles, but maximum is %d. Only loading first %d triangles.\n",
			   triangleCount, NUMBER_OF_TRIANGLES, NUMBER_OF_TRIANGLES);
		triangleCount = NUMBER_OF_TRIANGLES;
	}

	triangles->count += triangleCount;

	float minBB[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxBB[3] = {FLT_MIN, FLT_MIN, FLT_MIN};

	// Read triangle data and apply scaling
	for (int i = 0; i < triangleCount; i++) {
		int idx = i * 3;

		// Read vertices and apply scale
		float v1[3], v2[3], v3[3];
		fread(v1, sizeof(float), 3, file);
		fread(v2, sizeof(float), 3, file);
		fread(v3, sizeof(float), 3, file);

		// Apply scaling to vertices
		triangles->v1[idx] = v1[0] * scale;
		triangles->v1[idx + 1] = v1[1] * scale;
		triangles->v1[idx + 2] = v1[2] * scale;

		triangles->v2[idx] = v2[0] * scale;
		triangles->v2[idx + 1] = v2[1] * scale;
		triangles->v2[idx + 2] = v2[2] * scale;

		triangles->v3[idx] = v3[0] * scale;
		triangles->v3[idx + 1] = v3[1] * scale;
		triangles->v3[idx + 2] = v3[2] * scale;

		updateBBox(triangles->v1[i * 3], triangles->v1[i * 3 + 1], triangles->v1[i * 3 + 2], minBB, maxBB);
		updateBBox(triangles->v2[i * 3], triangles->v2[i * 3 + 1], triangles->v2[i * 3 + 2], minBB, maxBB);
		updateBBox(triangles->v3[i * 3], triangles->v3[i * 3 + 1], triangles->v3[i * 3 + 2], minBB, maxBB);

		// Read normals (normals don't need scaling, they should remain unit vectors)
		fread(&triangles->normals[idx], sizeof(float), 3, file);

		// Read material properties (unchanged)
		fread(&triangles->Roughness[i], sizeof(float), 1, file);
		fread(&triangles->Metallic[i], sizeof(float), 1, file);
		fread(&triangles->Emission[i], sizeof(float), 1, file);

		// Read colors (unchanged)
		fread(&triangles->colors[idx], sizeof(float), 3, file);

		// Read triangle index (skip it since we don't use it)
		uint32_t triangleIndex;
		fread(&triangleIndex, sizeof(uint32_t), 1, file);
	}

	fclose(file);
	printf("Triangles read from %s successfully with scale factor %.2f\n", filename, scale);
	printf("File size: %u bytes\n", fileSize);
	printf("Triangle count: %d\n", triangles->count);

	printf("Bounding Box Min: (%.2f, %.2f, %.2f)\n", minBB[0], minBB[1], minBB[2]);
	printf("Bounding Box Max: (%.2f, %.2f, %.2f)\n", maxBB[0], maxBB[1], maxBB[2]);
	printf("Bounding Box Size: (%.2f, %.2f, %.2f)\n",
		   maxBB[0] - minBB[0],
		   maxBB[1] - minBB[1],
		   maxBB[2] - minBB[2]);
}

// Helper function to convert degrees to radians
float degreesToRadians(float degrees) {
	return degrees * M_PI / 180.0f;
}

// Add this helper function to create a rotation matrix from angles
void createRotationMatrix(float angleX, float angleY, float angleZ, float matrix[3][3]) {
	float cosX = cosf(angleX), sinX = sinf(angleX);
	float cosY = cosf(angleY), sinY = sinf(angleY);
	float cosZ = cosf(angleZ), sinZ = sinf(angleZ);

	// Combined rotation matrix (Z * Y * X order)
	matrix[0][0] = cosY * cosZ;
	matrix[0][1] = sinX * sinY * cosZ - cosX * sinZ;
	matrix[0][2] = cosX * sinY * cosZ + sinX * sinZ;

	matrix[1][0] = cosY * sinZ;
	matrix[1][1] = sinX * sinY * sinZ + cosX * cosZ;
	matrix[1][2] = cosX * sinY * sinZ - sinX * cosZ;

	matrix[2][0] = -sinY;
	matrix[2][1] = sinX * cosY;
	matrix[2][2] = cosX * cosY;
}

// Helper function to apply rotation matrix to a vector
void rotateVector(float v[3], float matrix[3][3], float result[3]) {
	result[0] = matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2];
	result[1] = matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2];
	result[2] = matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2];
}

void loadModelWithRotation(const char *filename, struct Triangles *triangles,
						   float scale, float translate[3],
						   float rotationXDeg, float rotationYDeg, float rotationZDeg) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open file %s for reading\n", filename);
		return;
	}

	uint32_t fileSize, triangleStructSize;
	fread(&fileSize, sizeof(uint32_t), 1, file);
	fread(&triangleStructSize, sizeof(uint32_t), 1, file);

	int triangleCount = (fileSize - 8) / triangleStructSize;
	printf("Loading %d triangles from %s\n", triangleCount, filename);
	printf("Scale: %.2f\n", scale);
	printf("Translation: (%.2f, %.2f, %.2f)\n", translate[0], translate[1], translate[2]);
	printf("Rotation: X=%.2f°, Y=%.2f°, Z=%.2f°\n", rotationXDeg, rotationYDeg, rotationZDeg);

	if (triangleCount > NUMBER_OF_TRIANGLES) {
		printf("Warning: File contains %d triangles, but maximum is %d. Only loading first %d triangles.\n",
			   triangleCount, NUMBER_OF_TRIANGLES, NUMBER_OF_TRIANGLES);
		triangleCount = NUMBER_OF_TRIANGLES;
	}

	triangles->count += triangleCount;

	float minBB[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxBB[3] = {FLT_MIN, FLT_MIN, FLT_MIN};

	// Convert degrees to radians
	float rotationXRad = degreesToRadians(rotationXDeg);
	float rotationYRad = degreesToRadians(rotationYDeg);
	float rotationZRad = degreesToRadians(rotationZDeg);

	// Create rotation matrix
	float rotMatrix[3][3];
	createRotationMatrix(rotationXRad, rotationYRad, rotationZRad, rotMatrix);

	for (int i = 0; i < triangleCount; i++) {
		int idx = i * 3;

		// Read vertices
		float v1[3], v2[3], v3[3], normal[3];
		fread(v1, sizeof(float), 3, file);
		fread(v2, sizeof(float), 3, file);
		fread(v3, sizeof(float), 3, file);

		// Apply transformations: Scale -> Rotate -> Translate

		// Vertex 1
		v1[0] *= scale;
		v1[1] *= scale;
		v1[2] *= scale;
		float v1_rotated[3];
		rotateVector(v1, rotMatrix, v1_rotated);
		triangles->v1[idx] = v1_rotated[0] + translate[0];
		triangles->v1[idx + 1] = v1_rotated[1] + translate[1];
		triangles->v1[idx + 2] = v1_rotated[2] + translate[2];

		// Vertex 2
		v2[0] *= scale;
		v2[1] *= scale;
		v2[2] *= scale;
		float v2_rotated[3];
		rotateVector(v2, rotMatrix, v2_rotated);
		triangles->v2[idx] = v2_rotated[0] + translate[0];
		triangles->v2[idx + 1] = v2_rotated[1] + translate[1];
		triangles->v2[idx + 2] = v2_rotated[2] + translate[2];

		// Vertex 3
		v3[0] *= scale;
		v3[1] *= scale;
		v3[2] *= scale;
		float v3_rotated[3];
		rotateVector(v3, rotMatrix, v3_rotated);
		triangles->v3[idx] = v3_rotated[0] + translate[0];
		triangles->v3[idx + 1] = v3_rotated[1] + translate[1];
		triangles->v3[idx + 2] = v3_rotated[2] + translate[2];

		// Update bounding box with transformed vertices
		updateBBox(triangles->v1[idx], triangles->v1[idx + 1], triangles->v1[idx + 2], minBB, maxBB);
		updateBBox(triangles->v2[idx], triangles->v2[idx + 1], triangles->v2[idx + 2], minBB, maxBB);
		updateBBox(triangles->v3[idx], triangles->v3[idx + 1], triangles->v3[idx + 2], minBB, maxBB);

		// Read and rotate normals (normals should only be rotated, not scaled or translated)
		fread(normal, sizeof(float), 3, file);
		float normal_rotated[3];
		rotateVector(normal, rotMatrix, normal_rotated);
		triangles->normals[idx] = normal_rotated[0];
		triangles->normals[idx + 1] = normal_rotated[1];
		triangles->normals[idx + 2] = normal_rotated[2];

		// Read material properties and colors (unchanged)
		fread(&triangles->Roughness[i], sizeof(float), 1, file);
		fread(&triangles->Metallic[i], sizeof(float), 1, file);
		fread(&triangles->Emission[i], sizeof(float), 1, file);
		fread(&triangles->colors[idx], sizeof(float), 3, file);

		// Skip triangle index
		uint32_t triangleIndex;
		fread(&triangleIndex, sizeof(uint32_t), 1, file);
	}

	fclose(file);

	printf("Model loaded successfully!\n");
	printf("Triangle count: %d\n", triangles->count);
	printf("Bounding Box Min: (%.2f, %.2f, %.2f)\n", minBB[0], minBB[1], minBB[2]);
	printf("Bounding Box Max: (%.2f, %.2f, %.2f)\n", maxBB[0], maxBB[1], maxBB[2]);
	printf("Bounding Box Size: (%.2f, %.2f, %.2f)\n",
		   maxBB[0] - minBB[0],
		   maxBB[1] - minBB[1],
		   maxBB[2] - minBB[2]);
}

void loadFont(struct ImageFont *font, const char *filename) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open font file %s\n", filename);
		return;
	}

	// Load first uint32 little-endian value as width
	fread(&font->width, sizeof(uint32_t), 1, file);
	// Load second uint32 little-endian value as height
	fread(&font->height, sizeof(uint32_t), 1, file);

	printf("Loading font: %dx%d pixels\n", font->width, font->height);

	// Allocate memory for font data
	int totalPixels = font->width * font->height;
	font->data = (char *)malloc(totalPixels * sizeof(char));
	if (!font->data) {
		printf("Error: Could not allocate memory for font data\n");
		fclose(file);
		return;
	}

	// Read the bit data (each bit is stored as a byte in your Python code)
	for (int i = 0; i < totalPixels; i++) {
		uint8_t bit;
		if (fread(&bit, sizeof(uint8_t), 1, file) != 1) {
			printf("Error: Could not read bit data at position %d\n", i);
			free(font->data);
			font->data = NULL;
			fclose(file);
			return;
		}
		font->data[i] = (char)bit; // Store as 0 or 1
	}

	fclose(file);
	printf("Font loaded successfully: %d pixels\n", totalPixels);
}

void ReadBVH(struct BVHLinear *bvh, const char *filename) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open BVH file %s\n", filename);
		return;
	}

	int NumberOfTriangles = 0;
	int NumberOfNodes = 0;

	// Load first uint32 little-endian value as Number of Nodes
	if (fread(&NumberOfNodes, sizeof(uint32_t), 1, file) != 1) {
		printf("Error: Could not read number of nodes\n");
		fclose(file);
		return;
	}

	bvh->NodesCount = NumberOfNodes;

	// Load second uint32 little-endian value as Number of Triangles
	if (fread(&NumberOfTriangles, sizeof(uint32_t), 1, file) != 1) {
		printf("Error: Could not read number of triangles\n");
		fclose(file);
		return;
	}

	bvh->TrianglesCount = NumberOfTriangles;

	printf("Reading BVH: %d nodes, %d triangles\n", NumberOfNodes, NumberOfTriangles);

	// Allocate memory for BVH nodes
	bvh->Nodes = (struct BVHNode *)malloc(NumberOfNodes * sizeof(struct BVHNode));
	if (!bvh->Nodes) {
		fclose(file);
		printf("Error: Could not allocate memory for BVH nodes\n");
		return;
	}

	// Allocate memory for triangles container
	bvh->Triangles = (struct Triangle *)malloc(sizeof(struct Triangle) * NumberOfTriangles);
	if (!bvh->Triangles) {
		printf("Error: Could not allocate memory for triangles container\n");
		free(bvh->Nodes);
		fclose(file);
		return;
	}

	// Read all BVH nodes
	for (int i = 0; i < NumberOfNodes; i++) {
		if (fread(&bvh->Nodes[i], sizeof(struct BVHNode), 1, file) != 1) {
			printf("Error: Could not read BVH node %d\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}
	}

	// Read all triangles
	for (int i = 0; i < NumberOfTriangles; i++) {
		struct Triangle *triangle = &bvh->Triangles[i];

		// Read vertex 1
		if (fread(triangle->v1, sizeof(float), 3, file) != 3) {
			printf("Error: Could not read triangle %d vertex 1\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		// Read vertex 2
		if (fread(triangle->v2, sizeof(float), 3, file) != 3) {
			printf("Error: Could not read triangle %d vertex 2\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		// Read vertex 3
		if (fread(triangle->v3, sizeof(float), 3, file) != 3) {
			printf("Error: Could not read triangle %d vertex 3\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		// Read normal vector
		if (fread(triangle->normal, sizeof(float), 3, file) != 3) {
			printf("Error: Could not read triangle %d normal\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		// Read color
		if (fread(triangle->color, sizeof(float), 3, file) != 3) {
			printf("Error: Could not read triangle %d color\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		// Read material properties
		if (fread(&triangle->Roughness, sizeof(float), 1, file) != 1) {
			printf("Error: Could not read triangle %d roughness\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		if (fread(&triangle->Metallic, sizeof(float), 1, file) != 1) {
			printf("Error: Could not read triangle %d metallic\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		if (fread(&triangle->Emission, sizeof(float), 1, file) != 1) {
			printf("Error: Could not read triangle %d emission\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}

		// Read triangle index
		if (fread(&triangle->TriangleIndex, sizeof(int), 1, file) != 1) {
			printf("Error: Could not read triangle %d index\n", i);
			free(bvh->Nodes);
			free(bvh->Triangles);
			fclose(file);
			return;
		}
	}

	fclose(file);
	printf("BVH loaded successfully: %d nodes, %d triangles\n", NumberOfNodes, NumberOfTriangles);
}

uint8_t readRenderMode(const char *filename) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open render mode file %s\n", filename);
		return 0; // Default to 0 if file cannot be read
	}

	uint8_t renderMode;
	if (fread(&renderMode, sizeof(uint8_t), 1, file) != 1) {
		printf("Error: Could not read render mode from file %s\n", filename);
		fclose(file);
		return 0; // Default to 0 if read fails
	}

	fclose(file);
	return renderMode;
}

bool isKeyPressed(int key) {
	// If the callback latched a press, consume and return true
	if (key >= 0 && key <= GLFW_KEY_LAST && keyState.justPressed[key]) {
		keyState.justPressed[key] = false;
		return true;
	}
	if (key >= 0 && key <= GLFW_KEY_LAST)
		return keyState.keys[key] && !keyState.prevKeys[key];
	return false;
}

bool isKeyReleased(int key) {
	return !keyState.keys[key] && keyState.prevKeys[key]; // Just released this frame
}

bool isKeyHeld(int key) {
	return keyState.keys[key]; // Currently held down
}

void updateKeyStates() {
	// Copy current state to previous state
	memcpy(keyState.prevKeys, keyState.keys, sizeof(keyState.keys));
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (key < 0 || key > GLFW_KEY_LAST) return; // Safety check

	if (action == GLFW_PRESS || action == GLFW_REPEAT) {
		keyState.keys[key] = true;
		if (action == GLFW_PRESS) keyState.justPressed[key] = true; // latch the press event
																	// printf("Key %d pressed/repeat\n", key);
	} else if (action == GLFW_RELEASE) {
		keyState.keys[key] = false;
		// printf("Key %d released\n", key);
	}
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
	if (mouseState.firstMouse) {
		mouseState.prevX = xpos;
		mouseState.prevY = ypos;
		mouseState.firstMouse = false;
	}

	mouseState.x = xpos;
	mouseState.y = ypos;

	// Calculate delta (change) in mouse position
	mouseState.deltaX = xpos - mouseState.prevX;
	mouseState.deltaY = ypos - mouseState.prevY;

	// printf("Mouse moved to: (%.1f, %.1f), Delta: (%.1f, %.1f)\n",
	// 	   xpos, ypos, mouseState.deltaX, mouseState.deltaY);
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		mouseState.leftButton = (action == GLFW_PRESS);
		if (action == GLFW_PRESS) {
			printf("Left mouse button pressed at (%.1f, %.1f)\n", mouseState.x, mouseState.y);
		} else if (action == GLFW_RELEASE) {
			printf("Left mouse button released at (%.1f, %.1f)\n", mouseState.x, mouseState.y);
		}
	} else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		mouseState.rightButton = (action == GLFW_PRESS);
		if (action == GLFW_PRESS) {
			printf("Right mouse button pressed at (%.1f, %.1f)\n", mouseState.x, mouseState.y);
		} else if (action == GLFW_RELEASE) {
			printf("Right mouse button released at (%.1f, %.1f)\n", mouseState.x, mouseState.y);
		}
	}
}

bool isMouseButtonPressed(int button) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		return mouseState.leftButton && !mouseState.prevLeftButton;
	} else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		return mouseState.rightButton && !mouseState.prevRightButton;
	}
	return false;
}

bool isMouseButtonReleased(int button) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		return !mouseState.leftButton && mouseState.prevLeftButton;
	} else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		return !mouseState.rightButton && mouseState.prevRightButton;
	}
	return false;
}

bool isMouseButtonHeld(int button) {
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		return mouseState.leftButton;
	} else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		return mouseState.rightButton;
	}
	return false;
}

void updateMouseStates() {
	mouseState.prevX = mouseState.x;
	mouseState.prevY = mouseState.y;
	mouseState.prevLeftButton = mouseState.leftButton;
	mouseState.prevRightButton = mouseState.rightButton;
}

void randomMissileMovement(struct Missiles *missiles, struct Camera *camera) {
	if (rand_01() < 0.25f) {
		for (int i = 0; i < missiles->count; i++) {
			if (missiles->active[i]) {
				float cameraPos[3] = {camera->ray.origin[0] + randRange(-200.0f, 200.0f),
									  camera->ray.origin[1] + randRange(0.0f, 200.0f),
									  camera->ray.origin[2] + randRange(-200.0f, 200.0f)};

				// float dist = 500.0f;

				// setMissileTargetDirection(missiles->missiles[i], camera->ray.direction, &dist);

				// setMissileTarget(missiles->missiles[i], cameraPos);  // DISABLED: Let seeker handle targeting
				// float dx = missiles->missiles[i]->position[0] - camera->ray.origin[0];
				// float dy = missiles->missiles[i]->position[1] - camera->ray.origin[1];
				// float dz = missiles->missiles[i]->position[2] - camera->ray.origin[2];
				// float distance = sqrtf(dx * dx + dy * dy + dz * dz);

				// // Normalize direction vector
				// float inv_dist = (distance > 0.001f) ? (1.0f / distance) : 0.0f;
				// float norm_dx = dx * inv_dist;
				// float norm_dy = dy * inv_dist;
				// float norm_dz = dz * inv_dist;

				// // Different behavior based on missile index for variety
				// int behavior_type = i % 4;

				// if (distance < 50.0f) {
				// 	// Too close - fly away with different patterns
				// 	switch (behavior_type) {
				// 	case 0: // Direct retreat
				// 		missiles->missiles[i]->targetDirection[0] = norm_dx;
				// 		missiles->missiles[i]->targetDirection[1] = norm_dy;
				// 		missiles->missiles[i]->targetDirection[2] = norm_dz;
				// 		break;
				// 	case 1: // Spiral retreat
				// 		missiles->missiles[i]->targetDirection[0] = norm_dx + sinf(glfwGetTime() * 2.0f + i) * 0.3f;
				// 		missiles->missiles[i]->targetDirection[1] = norm_dy + cosf(glfwGetTime() * 1.5f + i) * 0.2f;
				// 		missiles->missiles[i]->targetDirection[2] = norm_dz;
				// 		break;
				// 	case 2: // Side-step retreat
				// 		missiles->missiles[i]->targetDirection[0] = norm_dx + norm_dz * 0.4f;
				// 		missiles->missiles[i]->targetDirection[1] = norm_dy;
				// 		missiles->missiles[i]->targetDirection[2] = norm_dz - norm_dx * 0.4f;
				// 		break;
				// 	case 3: // Vertical dodge retreat
				// 		missiles->missiles[i]->targetDirection[0] = norm_dx;
				// 		missiles->missiles[i]->targetDirection[1] = norm_dy + ((i % 2 == 0) ? 0.5f : -0.5f);
				// 		missiles->missiles[i]->targetDirection[2] = norm_dz;
				// 		break;
				// 	}
				// } else if (distance > 200.0f) {
				// 	// Too far - approach with different patterns
				// 	switch (behavior_type) {
				// 	case 0: // Direct approach
				// 		missiles->missiles[i]->targetDirection[0] = -norm_dx;
				// 		missiles->missiles[i]->targetDirection[1] = -norm_dy;
				// 		missiles->missiles[i]->targetDirection[2] = -norm_dz;
				// 		break;
				// 	case 1: // Weaving approach
				// 		missiles->missiles[i]->targetDirection[0] = -norm_dx + sinf(glfwGetTime() * 3.0f + i) * 0.2f;
				// 		missiles->missiles[i]->targetDirection[1] = -norm_dy;
				// 		missiles->missiles[i]->targetDirection[2] = -norm_dz + cosf(glfwGetTime() * 3.0f + i) * 0.2f;
				// 		break;
				// 	case 2: // Arc approach
				// 	{
				// 		float angle = glfwGetTime() * 1.0f + i * 1.57f; // 90 degrees apart
				// 		missiles->missiles[i]->targetDirection[0] = -norm_dx + sinf(angle) * 0.3f;
				// 		missiles->missiles[i]->targetDirection[1] = -norm_dy + cosf(angle * 0.5f) * 0.2f;
				// 		missiles->missiles[i]->targetDirection[2] = -norm_dz + cosf(angle) * 0.3f;
				// 	} break;
				// 	case 3: // Bobbing approach
				// 		missiles->missiles[i]->targetDirection[0] = -norm_dx;
				// 		missiles->missiles[i]->targetDirection[1] = -norm_dy + sinf(glfwGetTime() * 4.0f + i) * 0.3f;
				// 		missiles->missiles[i]->targetDirection[2] = -norm_dz;
				// 		break;
				// 	}
				// } else {
				// 	// Good distance - complex orbital/patrol patterns
				// 	float time = glfwGetTime();
				// 	float phase = i * 0.785f; // 45 degrees apart

				// 	switch (behavior_type) {
				// 	case 0: // Circular orbit
				// 	{
				// 		float orbit_radius = 0.4f;
				// 		float orbit_speed = 1.5f;
				// 		missiles->missiles[i]->targetDirection[0] = sinf(time * orbit_speed + phase) * orbit_radius;
				// 		missiles->missiles[i]->targetDirection[1] = (rand_01() * 2.0f - 1.0f) * 0.1f;
				// 		missiles->missiles[i]->targetDirection[2] = cosf(time * orbit_speed + phase) * orbit_radius;
				// 	} break;
				// 	case 1: // Figure-8 pattern
				// 	{
				// 		float fig8_scale = 0.3f;
				// 		missiles->missiles[i]->targetDirection[0] = sinf(time * 2.0f + phase) * fig8_scale;
				// 		missiles->missiles[i]->targetDirection[1] = sinf(time * 4.0f + phase) * fig8_scale * 0.5f;
				// 		missiles->missiles[i]->targetDirection[2] = cosf(time * 2.0f + phase) * fig8_scale;
				// 	} break;
				// 	case 2: // Patrol pattern (back and forth)
				// 	{
				// 		float patrol_intensity = 0.4f;
				// 		missiles->missiles[i]->targetDirection[0] = sinf(time * 1.2f + phase) * patrol_intensity;
				// 		missiles->missiles[i]->targetDirection[1] = cosf(time * 0.8f + phase) * 0.2f;
				// 		missiles->missiles[i]->targetDirection[2] = cosf(time * 1.2f + phase) * patrol_intensity;
				// 	} break;
				// 	case 3: // Helix pattern
				// 	{
				// 		float helix_radius = 0.3f;
				// 		float helix_speed = 2.0f;
				// 		missiles->missiles[i]->targetDirection[0] = sinf(time * helix_speed + phase) * helix_radius;
				// 		missiles->missiles[i]->targetDirection[1] = sinf(time * helix_speed * 0.3f + phase) * 0.4f;
				// 		missiles->missiles[i]->targetDirection[2] = cosf(time * helix_speed + phase) * helix_radius;
				// 	} break;
				// 	}

				// 	// Add some random variation to make it less predictable
				// 	missiles->missiles[i]->targetDirection[0] += (rand_01() * 2.0f - 1.0f) * 0.05f;
				// 	missiles->missiles[i]->targetDirection[1] += (rand_01() * 2.0f - 1.0f) * 0.05f;
				// 	missiles->missiles[i]->targetDirection[2] += (rand_01() * 2.0f - 1.0f) * 0.05f;
				// }

				// // Normalize the final direction vector
				// float len = sqrtf(missiles->missiles[i]->targetDirection[0] * missiles->missiles[i]->targetDirection[0] +
				// 				  missiles->missiles[i]->targetDirection[1] * missiles->missiles[i]->targetDirection[1] +
				// 				  missiles->missiles[i]->targetDirection[2] * missiles->missiles[i]->targetDirection[2]);
				// if (len > 0.001f) {
				// 	float inv_len = 1.0f / len;
				// 	missiles->missiles[i]->targetDirection[0] *= inv_len;
				// 	missiles->missiles[i]->targetDirection[1] *= inv_len;
				// 	missiles->missiles[i]->targetDirection[2] *= inv_len;
				// }

				// // Optional: Add height preference to keep missiles at reasonable altitude
				// float preferred_height = 30.0f;
				// float height_diff = missiles->missiles[i]->position[1] - preferred_height;
				// if (fabs(height_diff) > 20.0f) {
				// 	missiles->missiles[i]->targetDirection[1] += (height_diff > 0) ? -0.2f : 0.2f;
				// }
			}
		}
	}
}

int main() {
	struct Triangles *missileModel = (struct Triangles *)malloc(sizeof(struct Triangles));
	if (!missileModel) {
		perror("Failed to allocate memory for triangles");
		return 1;
	}
	missileModel->count = 0;
	readFileTriangles("missile/r27.bin", missileModel, 25.0f);
	struct Missiles missiles;
	InitializeMissiles(&missiles, MAX_FIRE_SIMS, missileModel); // Create 8 missiles

	for (int i = 0; i < 4; i++) {
		missiles.missiles[i]->position[0] = (i - 1.5f) * 300.0f;
		missiles.missiles[i]->position[1] = 1500.0f + (i % 2) * 200.0f;
		missiles.missiles[i]->position[2] = 2000.0f;
		missiles.missiles[i]->velocity[0] = 0.0f;
		missiles.missiles[i]->velocity[1] = 0.0f;
		missiles.missiles[i]->velocity[2] = 0.0f;
		missiles.missiles[i]->burning = 1;
		missiles.missiles[i]->burningTemp = 2000.0f;
		missiles.active[i] = true;
	}

	// load BVH
	struct BVHLinear bvh;
	ReadBVH(&bvh, "parseObj/encoded.bvh");
	printf("BVH loaded with %d nodes and %d triangles\n", bvh.NodesCount, bvh.TrianglesCount);

	// load font
	struct ImageFont font;
	loadFont(&font, "fonts/fonts.bin");

	int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
	if (fd == -1) {
		perror("shm_open");
		return 1;
	}

	ftruncate(fd, SIZE); // Set size

	SharedMem = mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (SharedMem == MAP_FAILED) { // Fixed: was "ptr == MAP_FAILED"
		perror("mmap");
		return 1;
	}

	// srand(time(NULL));

	// load sky box texture
	struct SkyBox skyBox;
	if (!loadSkyBox(&skyBox)) {
		fprintf(stderr, "Failed to load skybox textures\n");
		return 1;
	}
	printf("SkyBox loaded successfully\n");

	struct Triangles *triangles = (struct Triangles *)malloc(sizeof(struct Triangles));
	if (!triangles) {
		perror("Failed to allocate memory for triangles");
		return 1;
	}
	triangles->count = 0;

	struct Camera camera;
	// Position camera near the center of the terrain map (tiles are 55000x55000)
	// Map is 8x8 tiles, so center is around 4*55000 = 220000 in each direction
	camera.ray.origin[0] = 2750.0f; // Center of first tile (tile 0,0)
	camera.ray.origin[1] = 500.0f;	// 5km above ground
	camera.ray.origin[2] = 2750.0f; // Center of first tile (tile 0,0)
	camera.ray.direction[0] = 0.0f;
	camera.ray.direction[1] = 0.0f;
	camera.ray.direction[2] = 1.0f;
	camera.renderMode = renderCompositedColor;
	camera.fov = 1.0f;
	camera.AntiAlias = false;
	camera.advanceAntiAlias = false;

	struct IRSearchAndTrack irst;
	InitializeIRST(&camera, &irst, ScreenWidth, ScreenHeight);

	// initialize the particles indexes
	struct ParticleIndexes *particleIndexes = (struct ParticleIndexes *)malloc(sizeof(struct ParticleIndexes));
	if (!particleIndexes) {
		perror("Failed to allocate memory for particle indexes");
		return 1;
	}

	struct PointSOA *particles = (struct PointSOA *)malloc(sizeof(struct PointSOA));
	if (!particles) {
		perror("Failed to allocate memory for particles");
		return 1;
	}

	if (WATER_SIMULATION == 1) {
		for (int i = 0; i < NUM_PARTICLES; i++) {
			particles->x[i] = (float)(rand() % 50 + 30);
			particles->y[i] = (float)(rand() % 50);
			particles->z[i] = (float)(rand() % 50 + 30);
			particles->xVelocity[i] = (float)(rand() % 10) / 100.0f;
			particles->yVelocity[i] = (float)(rand() % 10) / 100.0f;
			particles->zVelocity[i] = (float)(rand() % 10) / 100.0f;
		}

		particles->bBoxMin[0] = 0.0f;
		particles->bBoxMin[1] = 0.0f;
		particles->bBoxMin[2] = 0.0f;
		particles->bBoxMax[0] = 80.0f;
		particles->bBoxMax[1] = 80.0f;
		particles->bBoxMax[2] = 80.0f;

		updateGridData(particles);
	}

	int frameCount = 0;
	bool paused = false;
	bool fireMissile = false;

	float averageFPS[FrameCount];

	struct TimePartition *timePartition = (struct TimePartition *)malloc(sizeof(struct TimePartition));
	if (!timePartition) {
		perror("Failed to allocate memory for time partition");
		return 1;
	}

	clock_t lastTime = clock();

	printf("Triangles count after reading: %d\n", triangles->count);

	// for (int i = 0; i <= NUMBER_OF_CUBES; i++) {
	// 	float x = (float)(rand_01() * 500.0f);
	// 	float y = (float)(rand_01() * 500.0f);
	// 	float z = (float)(rand_01() * 500.0f);
	// 	float size = 25.0f;
	// 	float r = (float)rand_01();
	// 	float g = (float)rand_01();
	// 	float b = (float)rand_01();
	// 	float Roughness = (float)rand_01();
	// 	float Metallic = (float)rand_01();
	// 	// float Metallic = 0.0f; // Set Metallic to 1.0f for all cubes
	// 	float Emissive = (float)rand_01();
	// 	CreateCube(x, y, z, size, triangles, r, g, b, Metallic, Roughness, Emissive);
	// }

	float translate1[3] = {0.0f, 0.0f, 0.0f};

	// Initialize terrain map with all LOD levels
	struct Map terrain;
	printf("Loading terrain map...\n");
	init_terrain_map(
		"mapGeneration/highRes", // High resolution directory
		"mapGeneration/midRes",	 // Medium resolution directory
		"mapGeneration/lowRes",	 // Low resolution directory
		&terrain,
		150.0f,		// Scale
		translate1, // Base translation [x, y, z]
		90.0f,		// Rotation X (degrees)
		0.0f,		// Rotation Y (degrees)
		90.0f,		// Rotation Z (degrees)
		0.0f,		// Map position X
		0.0f,		// Map position Y
		0.0f		// Map position Z
	);

	printf("Terrain map initialized. Loading visible tiles based on camera...\n");
	loadCurrentMap(&terrain, &camera, triangles);
	printf("Loaded %d triangles from terrain\n", triangles->count);

	struct MapGPU *mapGPU = (struct MapGPU *)malloc(sizeof(struct MapGPU));
	if (!mapGPU) {
		perror("Failed to allocate memory for mapGPU");
		return 1;
	}
	initMapGPU(mapGPU, &terrain);

	// CreateBoardPlane(0.0f, -20.0f, 0.0f, 50.0f, 32, triangles);

	writeFileTriangles("parseObj/triangles.bin", triangles);

	struct GPUTimings gpuTimings;

	if (!glfwInit()) {
		fprintf(stderr, "Failed to init GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	GLFWwindow *window = glfwCreateWindow(ScreenWidth, ScreenHeight, "OpenCL-GL Interop", NULL, NULL);
	if (!window) {
		fprintf(stderr, "Failed to create window\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_callback);

	// Add mouse callbacks
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	mouseState.firstMouse = true;

	// Now initialize OpenCL with OpenGL sharing
	struct OpenCLContext ocl;
	int useOpenCL = initializeOpenCLWithGL(&ocl, triangles, &skyBox, &font, &bvh, window, &missiles, mapGPU);
	if (!useOpenCL) {
		printf("Failed to initialize OpenCL-GL interop, falling back to CPU\n");
	}

	bool exit = false;

	struct FireSOA *fireParticles = malloc(sizeof(struct FireSOA));
	if (!fireParticles) {
		perror("Failed to allocate memory for fire particles");
		return 1;
	}

	InitializeFireParticles(fireParticles);

	while (!glfwWindowShouldClose(window) && !exit) {
		// sleep(1);
		// Update key states at the start of each frame
		glfwPollEvents();

		float fx = camera.ray.direction[0];
		float fz = camera.ray.direction[2];
		// forward/back
		if (isKeyHeld(GLFW_KEY_W)) {
			camera.ray.origin[0] += fx * MoveMultiplier;
			camera.ray.origin[2] += fz * MoveMultiplier;
		}
		if (isKeyHeld(GLFW_KEY_S)) {
			camera.ray.origin[0] -= fx * MoveMultiplier;
			camera.ray.origin[2] -= fz * MoveMultiplier;
		}
		// strafing: cross(forward,up) = right
		float rx = fz;
		float rz = -fx;
		if (isKeyHeld(GLFW_KEY_A)) {
			camera.ray.origin[0] += rx * MoveMultiplier;
			camera.ray.origin[2] += rz * MoveMultiplier;
		}
		if (isKeyHeld(GLFW_KEY_D)) {
			camera.ray.origin[0] -= rx * MoveMultiplier;
			camera.ray.origin[2] -= rz * MoveMultiplier;
		}
		if (isKeyHeld(GLFW_KEY_Q)) {
			camera.ray.origin[1] -= MoveMultiplier; // Move down
		}
		if (isKeyHeld(GLFW_KEY_E)) {
			camera.ray.origin[1] += MoveMultiplier; // Move up
		}
		if (isKeyPressed(GLFW_KEY_R)) {
			enum RenderMode oldMode = camera.renderMode;
			camera.renderMode = (camera.renderMode + 1) % RENDER_MODE_COUNT;
		}
		if (isKeyPressed(GLFW_KEY_P)) {
			paused = !paused;
		}
		if (isKeyPressed(GLFW_KEY_F)) {
			fireMissile = true;
		}
		if (isKeyPressed(GLFW_KEY_O)) {
			camera.AntiAlias = !camera.AntiAlias;
		}
		if (isKeyPressed(GLFW_KEY_I)) {
			camera.advanceAntiAlias = !camera.advanceAntiAlias;
		}
		if (isKeyPressed(GLFW_KEY_TAB)) {
			IRSTSelectNextTarget(&irst);
		}
		if (isKeyPressed(GLFW_KEY_LEFT_SHIFT) && isKeyHeld(GLFW_KEY_TAB)) {
			IRSTSelectPreviousTarget(&irst);
		}
		if (isKeyPressed(GLFW_KEY_BACKSPACE)) {
			IRSTClearSelection(&irst);
		}
		if (isKeyPressed(GLFW_KEY_ESCAPE)) {
			exit = true;
		}

		static float yaw = 0.0f;
		static float pitch = 0.0f;

		if (isMouseButtonHeld(GLFW_MOUSE_BUTTON_LEFT)) {
			// Update angles
			yaw += mouseState.deltaX * MouseSensitivity;
			pitch -= mouseState.deltaY * MouseSensitivity;

			// Clamp pitch to prevent flipping
			if (pitch > 89.0f) pitch = 89.0f;
			if (pitch < -89.0f) pitch = -89.0f;

			// Convert angles to direction vector
			float pitchRad = pitch * M_PI / 180.0f;
			float yawRad = yaw * M_PI / 180.0f;

			camera.ray.direction[0] = cosf(pitchRad) * cosf(yawRad);
			camera.ray.direction[1] = sinf(pitchRad);
			camera.ray.direction[2] = cosf(pitchRad) * sinf(yawRad);

			mouseState.deltaX = 0.0;
			mouseState.deltaY = 0.0;
		}

		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);
		// Calculate delta step based on elapsed time since the last frame
		clock_t currentTime = clock();
		float dt = (float)(currentTime - lastTime) / (float)CLOCKS_PER_SEC; // Scale to a reasonable frame time
		float TPS = 1.0f / dt;
		// Cap dt to avoid instability for long delays (e.g., if paused)
		if (dt > 0.1f) dt = 0.08f;
		// dt = 0.1f;
		lastTime = currentTime;

		clock_t loopStartTime = clock();

		clock_t readDataTime = clock();
		clock_t endReadDataTime = clock();
		float dt1 = (float)(endReadDataTime - readDataTime) / (float)CLOCKS_PER_SEC;
		timePartition->readDataTime += dt1;

		// randomMissileMovement(&missiles, &camera);

		clock_t startGridTime = clock();
		if (!paused) {
			float tmp = 0.0f;
			missilesSimulation(&ocl, &missiles, &tmp, &fireMissile, &camera, 1 / 60.0f, triangles, &irst);
			if (WATER_SIMULATION == 1) {
				Step(particles, 1.0f / 60.0f, &gpuTimings.fluidSimulationTime);
			}
			fireSimStep(fireParticles, 1 / TPS, &gpuTimings.fireSimulationTime);
			firedMissileTime -= 1 / 60.0f;
			if (firedMissileTime < 0.0f) {
				firedMissileTime = 0.0f;
				firedMissileIdx = -1;
			}
		}
		clock_t endGridTime = clock();
		dt1 = (float)(endGridTime - startGridTime) / (float)CLOCKS_PER_SEC;
		timePartition->updateGridTime += dt1;

		clock_t afterUpdateTime = clock();
		float averageUpdateTime = (float)(afterUpdateTime - loopStartTime) / (float)CLOCKS_PER_SEC;

		clock_t startRenderTime = clock();

		// Render depth map for IRST using GPU LOD terrain (no CPU-side loading needed)
		float tempTimeTookMs = 0.0f;
		renderTerrainDepthLOD_GPU(&ocl, &irst.seekerCamera, ocl.buffer_seeker_distances,
								  MISSILE_SEEKER_SIZE, MISSILE_SEEKER_SIZE, &tempTimeTookMs);

		IRSearchAndTrackStep(&missiles, &irst, dt);
		render(particles, &camera, timePartition, particleIndexes, &ocl, triangles, &skyBox, &gpuTimings, &font, fireParticles, &missiles, &irst);
		clock_t endRenderTime = clock();
		clock_gettime(CLOCK_MONOTONIC, &end);
		dt1 = (float)(endRenderTime - startRenderTime) / (float)CLOCKS_PER_SEC;
		timePartition->renderTime += dt1;

		// === DISPLAY THE MAIN SCENE ===
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, ocl.gl_texture);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(-1.0f, 1.0f);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);

// === OVERLAY UI LAYER ===
#if renderUI_Separately == 1
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, ocl.gl_ui_texture);
		glBegin(GL_QUADS);
		glTexCoord2f(0.0f, 1.0f);
		glVertex2f(-1.0f, -1.0f);
		glTexCoord2f(1.0f, 1.0f);
		glVertex2f(1.0f, -1.0f);
		glTexCoord2f(1.0f, 0.0f);
		glVertex2f(1.0f, 1.0f);
		glTexCoord2f(0.0f, 0.0f);
		glVertex2f(-1.0f, 1.0f);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_BLEND);
		glDisable(GL_TEXTURE_2D);
#endif

		// Swap buffers and poll events
		glfwSwapBuffers(window);

		// Exit on ESC key
		if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			break;
		// Exit on window close
		if (glfwWindowShouldClose(window))
			break;

		// float currentFPS = 1.0f / dt1;

		double ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
		float currentFPS = 1000.0f / ms;

		if (frameCount < FrameCount) {
			averageFPS[frameCount] = currentFPS;
		}

		float averageRenderTime = (float)(endRenderTime - startRenderTime) / (float)CLOCKS_PER_SEC;

		if (0.01f > rand_01()) {
			printf("FPS: %.2f, TPS: %.2f, Update: %.02f s, Render: %0.2f s\n",
				   currentFPS, TPS,
				   (averageUpdateTime),
				   (averageRenderTime));
		}

		if (frameCount >= FrameCount) {
			frameCount = 0;
			FILE *fpsFile = fopen("average_fps.bin", "wb");
			if (fpsFile) {
				fwrite(averageFPS, sizeof(float), FrameCount, fpsFile);
				fclose(fpsFile);
			}
			FILE *timeFile = fopen("time_partition.bin", "wb");

			// average the time partition data
			timePartition->collisionTime /= FrameCount;
			timePartition->applyPressureTime /= FrameCount;
			timePartition->updateParticlesTime /= FrameCount;
			timePartition->moveToBoxTime /= FrameCount;
			timePartition->updateGridTime /= FrameCount;
			timePartition->renderTime /= FrameCount;
			timePartition->clearScreenTime /= FrameCount;
			timePartition->projectParticlesTime /= FrameCount;
			timePartition->drawCursorTime /= FrameCount;
			timePartition->drawBoundingBoxTime /= FrameCount;
			timePartition->saveScreenTime /= FrameCount;
			timePartition->sortTime /= FrameCount;
			timePartition->projectionTime /= FrameCount;
			timePartition->renderDistanceVelocityTime /= FrameCount;
			timePartition->renderOpacityTime /= FrameCount;
			timePartition->readDataTime /= FrameCount;
			// Write the averaged data to the file
			if (timeFile) {
				fwrite(timePartition, sizeof(struct TimePartition), 1, timeFile);
				fclose(timeFile);
			}
			// Reset the time partition data
			timePartition->collisionTime = 0;
			timePartition->applyPressureTime = 0;
			timePartition->updateParticlesTime = 0;
			timePartition->moveToBoxTime = 0;
			timePartition->updateGridTime = 0;
			timePartition->renderTime = 0;
			timePartition->clearScreenTime = 0;
			timePartition->projectParticlesTime = 0;
			timePartition->drawCursorTime = 0;
			timePartition->drawBoundingBoxTime = 0;
			timePartition->saveScreenTime = 0;
			timePartition->sortTime = 0;
			timePartition->projectionTime = 0;
			timePartition->renderDistanceVelocityTime = 0;
			timePartition->renderOpacityTime = 0;
			timePartition->readDataTime = 0;
		}
		frameCount++;

		updateMouseStates();
		updateKeyStates();
	}

	// Clean up
	printf("Cleaning up...\n");

	// Free terrain map
	free_map(&terrain);

	// Free triangles and other resources
	free(triangles);
	free(particles);
	free(particleIndexes);
	free(timePartition);
	free(missileModel);

	// Free fire particles
	free(fireParticles);

	// Cleanup OpenCL
	cleanupOpenCL(&ocl);

	// Cleanup GLFW
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
