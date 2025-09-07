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

void *SharedMem = NULL;

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
#define RENDER_TRIAGES 1 // 1 = CALCULATE VERTEXES => PER PIXEL SHADING, 0 = RENDER PER TRIANGLE
#define ScreenWidth 800
#define ScreenHeight 600
#define SHM_NAME "/my_shared_mem"
#define SIZE ScreenWidth *ScreenHeight * 4 * 2 // Color + Normal
#define PARTICLE_RADIUS 4
#define FrameCount 30
#define NUM_THREADS 0
#define USE_GPU 1
#define NUMBER_OF_TRIANGLES 10000
#define NUMBER_OF_CUBES 100
pthread_t threads[NUM_THREADS];
#define GLFW_EXPOSE_NATIVE_X11
#define MoveMultiplier 1.25f
#define MouseSensitivity 0.25f
#define MAX_BLUR_PASSES 1
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
	bool keys[GLFW_KEY_LAST + 1];	  // Array to store state of all keys
	bool prevKeys[GLFW_KEY_LAST + 1]; // Previous frame state for detecting press/release
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
	"Wireframe"};

enum RenderMode {
	renderDistance,
	renderVelocity,
	renderOpacity,
	renderNormal,
	renderFluid,
	renderColor,
	renderWireframe,
	RENDER_MODE_COUNT // Total number of render modes
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

struct Ray {
	float origin[3];
	float direction[3];
};

struct OpenCLContext {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	// kernels
	cl_kernel kernel; // project particles kernel
	cl_kernel blur_kernel;
	cl_kernel normals_kernel;
	cl_kernel triangle_kernel; // triangle kernel
	cl_kernel skybox_kernel;
	cl_kernel applyReflections_kernel;
	cl_kernel applyRayTracedReflections_kernel; // Apply ray-traced reflections
	cl_kernel gpuTimings_kernel;				// kernel for GPU timings
	cl_kernel renderText_kernel;				// kernel for rendering text
	cl_kernel calculateVertex_kernel;			// Vertex calculation kernel
	cl_kernel shadePixels_kernel;				// Pixel shading kernel
	cl_kernel wireframe_kernel;					// Wireframe rendering kernel
	cl_kernel copyToTexture_kernel;				// Copy buffer data to OpenGL texture kernel
	// buffers
	cl_mem buffer_points;
	cl_mem buffer_velocities;

	cl_mem buffer_distances_temp;
	cl_mem buffer_opacities_temp;
	cl_mem buffer_triangle_colors;
	cl_mem buffer_projected_verts; // Pre-calculated vertex coordinates
	cl_mem buffer_triangle_bboxes; // Pre-calculated bounding boxes
	cl_mem buffer_valid_triangles; // Pre-calculated validity flags

	// Add OpenGL interop members
	GLuint gl_texture;		  // OpenGL texture ID
	GLuint gl_ui_texture;	  // OpenGL texture ID for UI
	cl_mem cl_texture_buffer; // OpenCL image object from GL texture
	cl_mem cl_ui_texture_buffer;
	cl_mem cl_ui_texture_buffer_temp; // Temporary buffer for UI texture

	// rayTracing buffers
	cl_mem buffer_bvh_nodes;
	cl_mem buffer_bvh_triangles;

	// buffer for rendering text
	cl_mem buffer_font_data; // buffer for font data

	// Add triangle buffers
	cl_mem buffer_triangle_v1;
	cl_mem buffer_triangle_v2;
	cl_mem buffer_triangle_v3;
	cl_mem buffer_triangle_normals;

	// screen buffers
	cl_mem buffer_distances;				 // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_opacities;				 // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_velocities_screen;		 // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_normals;					 // ScreenWidth * ScreenHeight * sizeof(float) * 3
	cl_mem buffer_screen_colors;			 // ScreenWidth * ScreenHeight * sizeof(float) * 3
	cl_mem buffer_screen_material_roughness; // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_screen_material_metallic;	 // ScreenWidth * ScreenHeight * sizeof(float)
	cl_mem buffer_screen_material_emission;	 // ScreenWidth * ScreenHeight * sizeof(float)

	// Triangle properties
	cl_mem buffer_triangle_roughness;
	cl_mem buffer_triangle_metallic;
	cl_mem buffer_triangle_emission;

	// font buffers
	cl_mem buffer_text_posX;
	cl_mem buffer_text_posY;
	cl_mem buffer_text_chars;
	cl_mem buffer_text_color;

	// skybox buffers
	cl_mem buffer_skybox_top;
	cl_mem buffer_skybox_bottom;
	cl_mem buffer_skybox_left;
	cl_mem buffer_skybox_right;
	cl_mem buffer_skybox_front;
	cl_mem buffer_skybox_back;

	// Add pre-allocated host memory buffers
	float *host_points_data;
	float *host_velocities_data;
	float *host_distances_result;
	float *host_opacities_result;
	float *host_velocities_result;
	float *host_normals_result;
	float *host_screen_colors_result;
};

struct Camera {
	struct Ray ray;
	float fov;
	enum RenderMode renderMode;
};

struct Triangles {
	float v1[NUMBER_OF_TRIANGLES * 3];
	float v2[NUMBER_OF_TRIANGLES * 3];
	float v3[NUMBER_OF_TRIANGLES * 3];
	float Roughness[NUMBER_OF_TRIANGLES];
	float Metallic[NUMBER_OF_TRIANGLES];
	float Emission[NUMBER_OF_TRIANGLES];
	float normals[NUMBER_OF_TRIANGLES * 3];
	float colors[NUMBER_OF_TRIANGLES * 3]; // RGB colors for each triangle
	int count;
};

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

struct Screen {
	uint8_t distance[ScreenWidth][ScreenHeight];
	uint8_t velocity[ScreenWidth][ScreenHeight];
	uint8_t normalizedOpacity[ScreenWidth][ScreenHeight];
	uint8_t normalizedOpacityLight[ScreenWidth][ScreenHeight];
	uint8_t colors[ScreenWidth][ScreenHeight][3];
	uint16_t particleCount[ScreenWidth][ScreenHeight];
	uint16_t opacity[ScreenWidth][ScreenHeight];
	float normals[ScreenWidth][ScreenHeight][3]; // Normal vectors for lighting
};

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

void applyReflectionsOpenCL(struct OpenCLContext *ocl, struct Camera *camera, struct SkyBox *skyBox, float *gpuTimeMs) {
	cl_int err;

	// Set kernel arguments for applyReflections
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int skybox_width = skyBox->top->width;
	cl_int skybox_height = skyBox->top->height;

	err = clSetKernelArg(ocl->applyReflections_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 1, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->applyReflections_kernel, 2, sizeof(cl_mem), &ocl->buffer_normals);
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
	cl_event vertex_event, pixel_event;

	// === PASS 1: Calculate vertex coordinates ===
	cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
	cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
	cl_float fov = camera->fov;
	cl_int screen_width = ScreenWidth;
	cl_int screen_height = ScreenHeight;
	cl_int num_triangles = triangles->count;

	// Set arguments for vertex calculation kernel
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

	// Execute vertex calculation kernel
	size_t vertex_global_size = triangles->count;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->calculateVertex_kernel, 1, NULL, &vertex_global_size, NULL, 0, NULL, &vertex_event);
	if (err != CL_SUCCESS) {
		printf("Error executing vertex kernel: %d\n", err);
		return;
	}

	// === PASS 2: Shade pixels ===
	// Set arguments for pixel shading kernel
	err = clSetKernelArg(ocl->shadePixels_kernel, 0, sizeof(cl_mem), &ocl->buffer_projected_verts);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 1, sizeof(cl_mem), &ocl->buffer_triangle_bboxes);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 2, sizeof(cl_mem), &ocl->buffer_valid_triangles);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 3, sizeof(cl_mem), &ocl->buffer_screen_colors);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 4, sizeof(cl_mem), &ocl->buffer_distances);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 5, sizeof(cl_mem), &ocl->buffer_normals);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 6, sizeof(cl_int), &screen_width);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 7, sizeof(cl_int), &screen_height);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 8, sizeof(cl_int), &num_triangles);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 9, sizeof(cl_mem), &ocl->buffer_triangle_colors);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 10, sizeof(cl_mem), &ocl->buffer_triangle_roughness);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 11, sizeof(cl_mem), &ocl->buffer_triangle_metallic);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 12, sizeof(cl_mem), &ocl->buffer_triangle_emission);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 13, sizeof(cl_mem), &ocl->buffer_screen_material_roughness);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 14, sizeof(cl_mem), &ocl->buffer_screen_material_metallic);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 15, sizeof(cl_mem), &ocl->buffer_screen_material_emission);
	err |= clSetKernelArg(ocl->shadePixels_kernel, 16, sizeof(cl_mem), &ocl->buffer_triangle_normals);

	if (err != CL_SUCCESS) {
		printf("Error setting pixel shader arguments: %d\n", err);
		return;
	}

	// Execute pixel shading kernel (waits for vertex kernel completion)
	size_t pixel_global_size[2] = {ScreenWidth, ScreenHeight};
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->shadePixels_kernel, 2, NULL, pixel_global_size, NULL, 1, &vertex_event, &pixel_event);
	if (err != CL_SUCCESS) {
		printf("Error executing pixel shader kernel: %d\n", err);
		return;
	}

	clFinish(ocl->queue);

	// Calculate total GPU time if requested
	if (gpuTimeMs != NULL) {
		cl_ulong vertex_start, pixel_end;
		clGetEventProfilingInfo(vertex_event, CL_PROFILING_COMMAND_START, sizeof(vertex_start), &vertex_start, NULL);
		clGetEventProfilingInfo(pixel_event, CL_PROFILING_COMMAND_END, sizeof(pixel_end), &pixel_end, NULL);
		*gpuTimeMs = (pixel_end - vertex_start) * 1e-6f; // convert ns to ms
	}

	// Cleanup
	clReleaseEvent(vertex_event);
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
void projectParticlesOpenCL(struct OpenCLContext *ocl, struct PointSOA *particles, struct Camera *camera, struct Screen *screen, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font);

struct Light {
	float x;
	float y;
	float z;
};

struct Cursor {
	float x;
	float y;
	float z;
	float force;
	bool active;
};

void drawCursor(struct Screen *screen, struct Cursor *cursor, struct Camera *camera) {
	// Use same projection method as particles
	const float halfWidth = ScreenWidth * 0.5f;
	const float halfHeight = ScreenHeight * 0.5f;

	// Calculate view matrix vectors - same as in projectParticles
	float right[3], trueUp[3];
	float up[3] = {0, 1, 0};

	// Calculate right vector using cross product
	right[0] = camera->ray.direction[1] * up[2] - camera->ray.direction[2] * up[1];
	right[1] = camera->ray.direction[2] * up[0] - camera->ray.direction[0] * up[2];
	right[2] = camera->ray.direction[0] * up[1] - camera->ray.direction[1] * up[0];

	// Calculate true up vector
	trueUp[0] = right[1] * camera->ray.direction[2] - right[2] * camera->ray.direction[1];
	trueUp[1] = right[2] * camera->ray.direction[0] - right[0] * camera->ray.direction[2];
	trueUp[2] = right[0] * camera->ray.direction[1] - right[1] * camera->ray.direction[0];

	// Calculate vector from camera to cursor
	float x = cursor->x - camera->ray.origin[0];
	float y = cursor->y - camera->ray.origin[1];
	float z = cursor->z - camera->ray.origin[2];

	// Calculate dot product to check if cursor is in front of camera
	float dotProduct = x * camera->ray.direction[0] +
					   y * camera->ray.direction[1] +
					   z * camera->ray.direction[2];

	// Only draw cursor if it's in front of the camera
	if (dotProduct > 0) {
		float fovScale = 1.0f / (dotProduct * camera->fov);

		// Calculate screen position
		float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
		float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;

		int screenX = (int)(screenRight * halfWidth + halfWidth);
		int screenY = (int)(-screenUp * halfHeight + halfHeight);

		// Clamp to screen bounds
		screenX = (screenX < 0) ? 0 : (screenX >= ScreenWidth ? ScreenWidth - 1 : screenX);
		screenY = (screenY < 0) ? 0 : (screenY >= ScreenHeight ? ScreenHeight - 1 : screenY);

		// Draw horizontal line of cross
		const int cursorRadius = 5;
		for (int px = screenX - cursorRadius; px <= screenX + cursorRadius; px++) {
			if (px >= 0 && px < ScreenWidth) {
				screen->distance[px][screenY] = 255;
				screen->velocity[px][screenY] = 255;
				screen->normalizedOpacity[px][screenY] = 255;
			}
		}
		// Draw vertical line of cross (fixed: use screenX for column)
		for (int py = screenY - cursorRadius; py <= screenY + cursorRadius; py++) {
			if (py >= 0 && py < ScreenHeight) {
				// screen->distance[screenX][py] = 255;
				screen->velocity[screenX][py] = 255;
				screen->normalizedOpacity[screenX][py] = 255;
			}
		}
	}
}

void readCursorData(struct Cursor *cursor) {
	FILE *file = fopen("cursor.bin", "rb");
	if (!file) {
		printf("Cursor file not found, using default cursor\n");
		return;
	}

	if (fread(&cursor->x, sizeof(float), 1, file) != 1 ||
		fread(&cursor->y, sizeof(float), 1, file) != 1 ||
		fread(&cursor->z, sizeof(float), 1, file) != 1 ||
		fread(&cursor->active, sizeof(bool), 1, file) != 1 ||
		fread(&cursor->force, sizeof(float), 1, file) != 1) {

		fclose(file);
		return;
	}

	fclose(file);
}

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

void clearScreen(struct Screen *screen) {
	// Clear all screen buffers using memset - much faster than nested loops
	memset(screen->distance, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
	memset(screen->velocity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
	memset(screen->normalizedOpacity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
	memset(screen->opacity, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
	memset(screen->particleCount, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
	memset(screen->normals, 0, sizeof(float) * ScreenWidth * ScreenHeight * 3);
	memset(screen->colors, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight * 3);
}

void saveScreen(struct Screen *screen, const char *filename) {
	FILE *file = fopen(filename, "wb");
	if (!file) {
		perror("Failed to open file");
		return;
	}

	static uint8_t buffer[ScreenWidth * ScreenHeight * 3];

	int i = 0;
	for (int y = 0; y < ScreenHeight; y++) {
		for (int x = 0; x < ScreenWidth; x++) {
			buffer[i++] = screen->distance[x][y];		   // Changed from [x][y] to [y][x]
			buffer[i++] = screen->velocity[x][y];		   // Changed from [x][y] to [y][x]
			buffer[i++] = screen->normalizedOpacity[x][y]; // Changed from [x][y] to [y][x]
		}
	}

	fwrite(buffer, 1, ScreenWidth * ScreenHeight * 3, file);
	fclose(file);
}

void saveScreenColor(struct Screen *screen) {
	static uint8_t buffer[ScreenWidth * ScreenHeight * 4];

	// Single loop is more cache-friendly
	int totalPixels = ScreenWidth * ScreenHeight;
	for (int i = 0; i < totalPixels; i++) {
		int x = i % ScreenWidth;
		int y = i / ScreenWidth;
		int bufferIdx = i * 4;

		buffer[bufferIdx] = screen->colors[x][y][0];	 // R
		buffer[bufferIdx + 1] = screen->colors[x][y][1]; // G
		buffer[bufferIdx + 2] = screen->colors[x][y][2]; // B
		buffer[bufferIdx + 3] = 255;					 // Alpha
	}

	memcpy(SharedMem, buffer, ScreenWidth * ScreenHeight * 4);
}

void saveScreenNormal(struct Screen *screen) {
	static uint8_t buffer[ScreenWidth * ScreenHeight * 4];

	int totalPixels = ScreenWidth * ScreenHeight;
	for (int i = 0; i < totalPixels; i++) {
		int x = i % ScreenWidth;
		int y = i / ScreenWidth;
		int bufferIdx = i * 4;

		// Convert from [-1,1] to [0,255] range
		buffer[bufferIdx] = (uint8_t)((screen->normals[x][y][0] * 0.5f + 0.5f) * 255.0f);
		buffer[bufferIdx + 1] = (uint8_t)((screen->normals[x][y][1] * 0.5f + 0.5f) * 255.0f);
		buffer[bufferIdx + 2] = (uint8_t)((screen->normals[x][y][2] * 0.5f + 0.5f) * 255.0f);
		buffer[bufferIdx + 3] = 255; // Alpha
	}

	size_t offset = ScreenWidth * ScreenHeight * 4;
	memcpy((uint8_t *)SharedMem + offset, buffer, ScreenWidth * ScreenHeight * 4);
}

void render(struct Screen *screen, struct PointSOA *particles, struct Camera *camera, struct Cursor *cursor, struct TimePartition *timePartition, struct ParticleIndexes *particleIndexes, struct Camera *lightCamera, struct OpenCLContext *openCLContext, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font) {
	// printf("\n--- Starting render ---\n");
	int start = clock();
	clearScreen(screen);
	// clearScreen(lightScreen);
	int clearScreenTime = clock();

	float dt = (float)(clearScreenTime - start) / (float)CLOCKS_PER_SEC;
	timePartition->clearScreenTime += dt;
	// printf("Clear screen time: %f\n", dt);
	if (USE_GPU == 1) {
		projectParticlesOpenCL(openCLContext, particles, camera, screen, triangles, skyBox, gpuTimings, font);
		// save normal screen
		struct timespec start, end;
		clock_gettime(CLOCK_MONOTONIC, &start);
		saveScreenNormal(screen);
		clock_gettime(CLOCK_MONOTONIC, &end);
		double ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
		printf("Saved normals (shared mem) in %.3f ms\n", ms);
		clock_gettime(CLOCK_MONOTONIC, &start);
		saveScreenColor(screen);
		clock_gettime(CLOCK_MONOTONIC, &end);
		ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
		printf("Saved colors (shared mem) in %.3f ms\n", ms);
	}
	clock_t projectParticlesTime = clock();

	dt = (float)(projectParticlesTime - clearScreenTime) / (float)CLOCKS_PER_SEC;
	timePartition->projectParticlesTime += dt;
	// printf("Project particles time: %f\n", dt);

	// projectParticles(particles, lightCamera, lightScreen, timePartition, threadsData, particleIndexes);
	clock_t projectLightParticlesTime = clock();

	dt = (float)(projectLightParticlesTime - projectParticlesTime) / (float)CLOCKS_PER_SEC;
	timePartition->projectLightParticlesTime += dt;

	drawCursor(screen, cursor, camera);
	clock_t drawCursorTime = clock();

	dt = (float)(drawCursorTime - projectLightParticlesTime) / (float)CLOCKS_PER_SEC;
	timePartition->drawCursorTime += dt;
	// printf("Draw cursor time: %f\n", dt);

	clock_t drawBoundingBoxTime = clock();

	dt = (float)(drawBoundingBoxTime - drawCursorTime) / (float)CLOCKS_PER_SEC;
	timePartition->drawBoundingBoxTime += dt;
	// printf("Draw bounding box time: %f\n", dt);

	saveScreen(screen, "output.bin");
	// saveScreen(lightScreen, "light.bin");
	clock_t saveScreenTime = clock();
	dt = (float)(saveScreenTime - drawBoundingBoxTime) / (float)CLOCKS_PER_SEC;
	timePartition->saveScreenTime += dt;
	// printf("Save screen time: %f\n", dt);
}

void update_particles(struct PointSOA *particles, float dt, struct TimePartition *timePartition, struct Cursor *cursor) {
	clock_t start = clock();
	clock_t collideParticlesTime = clock();
	float dt_ = (float)(collideParticlesTime - start) / (float)CLOCKS_PER_SEC;
	timePartition->collisionTime += dt_;
	// printf("Collision time: %f\n", dt_);

	clock_t applyPressureTime = clock();
	dt_ = (float)(applyPressureTime - collideParticlesTime) / (float)CLOCKS_PER_SEC;
	timePartition->applyPressureTime += dt_;
	// printf("Apply pressure time: %f\n", dt_);

	clock_t updateParticlesTime = clock();
	dt_ = (float)(updateParticlesTime - applyPressureTime) / (float)CLOCKS_PER_SEC;
	timePartition->updateParticlesTime += dt_;
	// printf("Update particles time: %f\n", dt_);

	clock_t moveToBoxTime = clock();
	dt_ = (float)(moveToBoxTime - updateParticlesTime) / (float)CLOCKS_PER_SEC;
	timePartition->moveToBoxTime += dt_;
	// printf("Move to box time: %f\n", dt_);
}

// read the pause.bin
void readPauseData(bool *paused) {
	FILE *file = fopen("pause.bin", "rb");
	if (!file) {
		return; // If file doesn't exist, keep current pause state
	}

	// Read the boolean value from the file
	bool fileValue;
	if (fread(&fileValue, sizeof(bool), 1, file) == 1) {
		*paused = fileValue; // Update the pause state with the value from the file
	}

	fclose(file);
}

int uploadTriangleDataOnce(struct OpenCLContext *ocl, struct Triangles *triangles) {
	cl_int err;

	printf("Uploading triangle data once: %d triangles\n", triangles->count);

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

// int initializeOpenCL(struct OpenCLContext *ocl, struct Triangles *triangles, struct SkyBox *skyBox, struct ImageFont *imageFont, struct BVHLinear *bvh) {
// 	cl_int err;

// 	// Get platform
// 	err = clGetPlatformIDs(1, &ocl->platform, NULL);
// 	if (err != CL_SUCCESS) {
// 		printf("Error getting OpenCL platform: %d\n", err);
// 		return 0;
// 	}

// 	// Get device
// 	err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_GPU, 1, &ocl->device, NULL);
// 	if (err != CL_SUCCESS) {
// 		// Fallback to CPU if GPU not available
// 		err = clGetDeviceIDs(ocl->platform, CL_DEVICE_TYPE_CPU, 1, &ocl->device, NULL);
// 		if (err != CL_SUCCESS) {
// 			printf("Error getting OpenCL device: %d\n", err);
// 			return 0;
// 		}
// 	}

// 	// Create context
// 	ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating OpenCL context: %d\n", err);
// 		return 0;
// 	}

// 	// Create command queue WITH PROFILING ENABLED
// 	ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, CL_QUEUE_PROFILING_ENABLE, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating OpenCL command queue: %d\n", err);
// 		return 0;
// 	}

// 	// Read kernel source
// 	FILE *file = fopen("openGlShaders/screenCordinates.cl", "r");
// 	if (!file) {
// 		printf("Error opening kernel file\n");
// 		return 0;
// 	}

// 	fseek(file, 0, SEEK_END);
// 	size_t source_size = ftell(file);
// 	fseek(file, 0, SEEK_SET);

// 	char *source = (char *)malloc(source_size + 1);
// 	fread(source, 1, source_size, file);
// 	source[source_size] = '\0';
// 	fclose(file);

// 	// Create program
// 	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char **)&source, &source_size, &err);
// 	free(source);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating OpenCL program: %d\n", err);
// 		return 0;
// 	}

// 	// Build program
// 	err = clBuildProgram(ocl->program, 1, &ocl->device, NULL, NULL, NULL);
// 	if (err != CL_SUCCESS) {
// 		printf("Error building OpenCL program: %d\n", err);

// 		// Get build log
// 		size_t log_size;
// 		clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
// 		char *log = (char *)malloc(log_size);
// 		clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
// 		printf("Build log: %s\n", log);
// 		free(log);
// 		return 0;
// 	}

// 	ocl->wireframe_kernel = clCreateKernel(ocl->program, "renderWireFrame", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating wireframe kernel: %d\n", err);
// 		return 0;
// 	}

// 	// kernel for projecting vertices to screen
// 	ocl->calculateVertex_kernel = clCreateKernel(ocl->program, "calculateVertexCoordinate", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating calculateVertexCoordinate kernel: %d\n", err);
// 		return 0;
// 	}

// 	// kernel for ray-tracing
// 	ocl->applyRayTracedReflections_kernel = clCreateKernel(ocl->program, "applyRayTracedReflections", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating rayTrace kernel: %d\n", err);
// 		return 0;
// 	}

// 	// kernel for rendering particles
// 	ocl->shadePixels_kernel = clCreateKernel(ocl->program, "ShadePixels", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating ShadePixels kernel: %d\n", err);
// 		return 0;
// 	}

// 	// kernel for rendering text
// 	ocl->renderText_kernel = clCreateKernel(ocl->program, "renderText", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating renderText kernel: %d\n", err);
// 		return 0;
// 	}

// 	ocl->gpuTimings_kernel = clCreateKernel(ocl->program, "gpuTimings", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating gpuTimings kernel: %d\n", err);
// 		return 0;
// 	}

// 	// particle projection kernel
// 	ocl->kernel = clCreateKernel(ocl->program, "project_points_to_screen", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating OpenCL kernel: %d\n", err);
// 		return 0;
// 	}

// 	// kernels for blur-ing distances and opacities
// 	ocl->blur_kernel = clCreateKernel(ocl->program, "blur_distances", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating blur kernel: %d\n", err);
// 		return 0;
// 	}

// 	// Create applyReflections kernel
// 	ocl->applyReflections_kernel = clCreateKernel(ocl->program, "applyReflections", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating applyReflections kernel: %d\n", err);
// 		return 0;
// 	}

// 	// Create skybox kernel
// 	ocl->skybox_kernel = clCreateKernel(ocl->program, "renderSkyBox", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating skybox kernel: %d\n", err);
// 		return 0;
// 	}

// 	ocl->normals_kernel = clCreateKernel(ocl->program, "calculate_normals_from_blurred_distances", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating normals kernel: %d\n", err);
// 		return 0;
// 	}

// 	// Create triangle kernel
// 	ocl->triangle_kernel = clCreateKernel(ocl->program, "renderTriangles", &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle kernel: %d\n", err);
// 		return 0;
// 	}

// 	// Create particle buffers
// 	ocl->buffer_points = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 										NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating points buffer: %d\n", err);
// 		return 0;
// 	}

// 	// create buffer for BVH nodes
// 	ocl->buffer_bvh_nodes = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 										   bvh->NodesCount * sizeof(struct BVHNode), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating BVH nodes buffer: %d\n", err);
// 		return 0;
// 	}

// 	// create buffer for BVH triangles
// 	ocl->buffer_bvh_triangles = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 											   bvh->TrianglesCount * sizeof(struct Triangle), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating BVH triangles buffer: %d\n", err);
// 		return 0;
// 	}

// 	// upload BVH data
// 	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_bvh_nodes, CL_TRUE, 0,
// 							   bvh->NodesCount * sizeof(struct BVHNode), bvh->Nodes, 0, NULL, NULL);
// 	if (err != CL_SUCCESS) {
// 		printf("Error writing BVH nodes buffer: %d\n", err);
// 		return 0;
// 	}

// 	// buffer for triangle vertices
// 	ocl->buffer_triangle_v1 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 											 triangles->count * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle v1 buffer: %d\n", err);
// 		return 0;
// 	}

// 	// upload triangle data for bvh
// 	err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_bvh_triangles, CL_TRUE, 0,
// 							   bvh->TrianglesCount * sizeof(struct Triangle), bvh->Triangles, 0, NULL, NULL);
// 	if (err != CL_SUCCESS) {
// 		printf("Error writing BVH triangles buffer: %d\n", err);
// 		return 0;
// 	}

// 	// buffer for projected vertices of triangles to screen
// 	ocl->buffer_projected_verts = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												 triangles->count * 9 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating projected vertices buffer: %d\n", err);
// 		return 0;
// 	}

// 	// buffer for bounding boxes of triangles
// 	ocl->buffer_triangle_bboxes = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												 triangles->count * 4 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle bboxes buffer: %d\n", err);
// 		return 0;
// 	}

// 	// buffer for valid triangles
// 	ocl->buffer_valid_triangles = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												 triangles->count * sizeof(int), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating valid triangles buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_triangle_bboxes = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												 triangles->count * 4 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle bboxes buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_valid_triangles = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												 triangles->count * sizeof(int), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating valid triangles buffer: %d\n", err);
// 		return 0;
// 	}

// 	// crete screen material properties buffers
// 	ocl->buffer_screen_material_roughness = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 														   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating screen material roughness buffer: %d\n", err);
// 		return 0;
// 	}
// 	ocl->buffer_screen_material_metallic = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 														  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating screen material metallic buffer: %d\n", err);
// 		return 0;
// 	}
// 	ocl->buffer_screen_material_emission = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 														  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating screen material emission buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_velocities = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 											NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating velocities buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_distances = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 										   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating distances buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_opacities = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 										   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating opacities buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_velocities_screen = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 												   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating screen velocities buffer: %d\n", err);
// 		return 0;
// 	}

// 	// Create normals buffer
// 	ocl->buffer_normals = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
// 										 ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating normals buffer: %d\n", err);
// 		return 0;
// 	}

// 	// Create temporary buffers for blur pipeline
// 	ocl->buffer_distances_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating temp distances buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_opacities_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 												ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating temp opacities buffer: %d\n", err);
// 		return 0;
// 	}

// 	// crete buffers for triangle properties
// 	ocl->buffer_triangle_roughness = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 													triangles->count * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle roughness buffer: %d\n", err);
// 		return 0;
// 	}
// 	ocl->buffer_triangle_metallic = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 												   triangles->count * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle metalness buffer: %d\n", err);
// 		return 0;
// 	}
// 	ocl->buffer_triangle_emission = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 												   triangles->count * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle emission buffer: %d\n", err);
// 		return 0;
// 	}

// 	// Create triangle buffers (allocated once, reused every frame)
// 	ocl->buffer_triangle_v1 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 											 triangles->count * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle v1 buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_triangle_v2 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 											 triangles->count * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle v2 buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_triangle_v3 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 											 triangles->count * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle v3 buffer: %d\n", err);
// 		return 0;
// 	}

// 	ocl->buffer_triangle_normals = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 												  triangles->count * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle normals buffer: %d\n", err);
// 		return 0;
// 	}

// 	// ADD TRIANGLE COLORS BUFFER
// 	ocl->buffer_triangle_colors = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
// 												 triangles->count * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating triangle colors buffer: %d\n", err);
// 		return 0;
// 	}

// 	// ADD SCREEN COLORS BUFFER
// 	ocl->buffer_screen_colors = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
// 											   ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating screen colors buffer: %d\n", err);
// 		return 0;
// 	}

// 	// Buffer for font
// 	size_t font_size = imageFont->width * imageFont->height * sizeof(char);
// 	ocl->buffer_font_data = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
// 										   font_size, imageFont->data, &err);
// 	if (err != CL_SUCCESS) {
// 		printf("Error creating font buffer: %d\n", err);
// 		return 0;
// 	}
// 	printf("Font uploaded to GPU: %dx%d pixels (%zu bytes)\n", imageFont->width, imageFont->height, font_size);

// 	// *** INITIALIZE SKYBOX BUFFERS AND UPLOAD DATA ***
// 	if (!initializeSkyboxBuffers(ocl, skyBox)) {
// 		printf("Failed to initialize skybox buffers during OpenCL init\n");
// 		return 0;
// 	}

// 	// *** UPLOAD TRIANGLE DATA ONCE ***
// 	if (!uploadTriangleDataOnce(ocl, triangles)) {
// 		printf("Failed to upload triangle data during OpenCL init\n");
// 		return 0;
// 	}

// 	// Set static kernel arguments that never change
// 	if (!setupStaticKernelArguments(ocl, triangles, skyBox)) {
// 		printf("Failed to set static kernel arguments during OpenCL init\n");
// 		return 0;
// 	}

// 	// Pre-allocate host memory buffers
// 	ocl->host_points_data = (float *)malloc(NUM_PARTICLES * 3 * sizeof(float));
// 	ocl->host_velocities_data = (float *)malloc(NUM_PARTICLES * 3 * sizeof(float));
// 	ocl->host_distances_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
// 	ocl->host_opacities_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
// 	ocl->host_velocities_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
// 	ocl->host_normals_result = (float *)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));

// 	// ADD HOST MEMORY FOR SCREEN COLORS
// 	ocl->host_screen_colors_result = (float *)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));

// 	// Check for allocation failures
// 	if (!ocl->host_points_data || !ocl->host_velocities_data ||
// 		!ocl->host_distances_result || !ocl->host_opacities_result ||
// 		!ocl->host_velocities_result || !ocl->host_normals_result ||
// 		!ocl->host_screen_colors_result) {
// 		printf("Failed to allocate host memory for OpenCL\n");
// 		return 0;
// 	}

// 	printf("OpenCL initialized successfully with triangle support and colors\n");
// 	return 1;
// }

int initializeOpenCLWithGL(struct OpenCLContext *ocl, struct Triangles *triangles,
						   struct SkyBox *skyBox, struct ImageFont *imageFont,
						   struct BVHLinear *bvh, GLFWwindow *window) {
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

	// Create command queue with profiling
	ocl->queue = clCreateCommandQueue(ocl->context, ocl->device,
									  CL_QUEUE_PROFILING_ENABLE, &err);
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

	ocl->applyRayTracedReflections_kernel = clCreateKernel(ocl->program, "applyRayTracedReflections", &err);
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

	// Add the copyToTexture kernel
	ocl->copyToTexture_kernel = clCreateKernel(ocl->program, "copyToGLTexture", &err);
	if (err != CL_SUCCESS) {
		printf("Error creating copyToTexture kernel: %d\n", err);
		return 0;
	}

	// Create all buffers (same as your existing initializeOpenCL function)
	ocl->buffer_points = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
										NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		printf("Error creating points buffer: %d\n", err);
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

	// Screen material properties buffers
	ocl->buffer_screen_material_roughness = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
														   ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_screen_material_metallic = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
														  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
	ocl->buffer_screen_material_emission = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY,
														  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);

	// Continue with all other buffer creations from your existing code...
	ocl->buffer_velocities = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
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
	ocl->buffer_triangle_roughness = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
													triangles->count * sizeof(float), NULL, &err);
	ocl->buffer_triangle_metallic = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												   triangles->count * sizeof(float), NULL, &err);
	ocl->buffer_triangle_emission = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												   triangles->count * sizeof(float), NULL, &err);
	ocl->buffer_triangle_v1 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											 triangles->count * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_v2 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											 triangles->count * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_v3 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
											 triangles->count * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_normals = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												  triangles->count * 3 * sizeof(float), NULL, &err);
	ocl->buffer_triangle_colors = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY,
												 triangles->count * 3 * sizeof(float), NULL, &err);
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
	if (!uploadTriangleDataOnce(ocl, triangles)) {
		printf("Failed to upload triangle data during OpenCL init\n");
		return 0;
	}

	// Set static kernel arguments
	if (!setupStaticKernelArguments(ocl, triangles, skyBox)) {
		printf("Failed to set static kernel arguments during OpenCL init\n");
		return 0;
	}

	// Pre-allocate host memory buffers
	ocl->host_points_data = (float *)malloc(NUM_PARTICLES * 3 * sizeof(float));
	ocl->host_velocities_data = (float *)malloc(NUM_PARTICLES * 3 * sizeof(float));
	ocl->host_distances_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
	ocl->host_opacities_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
	ocl->host_velocities_result = (float *)malloc(ScreenWidth * ScreenHeight * sizeof(float));
	ocl->host_normals_result = (float *)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));
	ocl->host_screen_colors_result = (float *)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));

	// Check for allocation failures
	if (!ocl->host_points_data || !ocl->host_velocities_data ||
		!ocl->host_distances_result || !ocl->host_opacities_result ||
		!ocl->host_velocities_result || !ocl->host_normals_result ||
		!ocl->host_screen_colors_result) {
		printf("Failed to allocate host memory for OpenCL\n");
		return 0;
	}

	printf("OpenCL-GL interop initialized successfully\n");
	return 1;
}

void projectParticlesOpenCL(struct OpenCLContext *ocl, struct PointSOA *particles, struct Camera *camera, struct Screen *screen, struct Triangles *triangles, struct SkyBox *skyBox, struct GPUTimings *gpuTimings, struct ImageFont *font) {
	cl_int err;
	cl_event readback_events[5]; // Array to hold all readback events

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

			renderWireframeOpenCL(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
		} else {
			renderTrianglesOpenCL_TwoPass(ocl, triangles, camera, &gpuTimings->renderTrianglesTime);
		}
	}

	float trianglesFPS = (gpuTimings->renderTrianglesTime > 0.001f) ? (1000.0f / gpuTimings->renderTrianglesTime) : 0.0f;
	snprintf(text, sizeof(text), "Triangles %.0f FPS", trianglesFPS);
	addTextOpenCL(ocl, font, text, 545, chart_pos_Y, white);
	chart_pos_Y += 15;

	// *** Screen Space Projection Kernel and sky box ***
	applyReflectionsOpenCL(ocl, camera, skyBox, &gpuTimings->applyReflectionsTime);

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

	float *distances_result = ocl->host_distances_result;
	float *opacities_result = ocl->host_opacities_result;
	float *velocities_result = ocl->host_velocities_result;
	float *normals_result = ocl->host_normals_result;

	err = clEnqueueReadBuffer(ocl->queue, final_blurred_distances_buf, CL_FALSE, 0,
							  ScreenWidth * ScreenHeight * sizeof(float), distances_result, 0, NULL, &readback_events[0]);
	err |= clEnqueueReadBuffer(ocl->queue, s_opac_src, CL_FALSE, 0,
							   ScreenWidth * ScreenHeight * sizeof(float), opacities_result, 0, NULL, &readback_events[1]);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->buffer_velocities_screen, CL_FALSE, 0,
							   ScreenWidth * ScreenHeight * sizeof(float), velocities_result, 0, NULL, &readback_events[2]);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->buffer_normals, CL_FALSE, 0,
							   ScreenWidth * ScreenHeight * 3 * sizeof(float), normals_result, 0, NULL, &readback_events[3]);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->buffer_screen_colors, CL_FALSE, 0,
							   ScreenWidth * ScreenHeight * 3 * sizeof(float), ocl->host_screen_colors_result, 0, NULL, &readback_events[4]);

	clWaitForEvents(5, readback_events);

	// Get readback timing
	cl_ulong first_start, last_end;
	clGetEventProfilingInfo(readback_events[0], CL_PROFILING_COMMAND_START, sizeof(first_start), &first_start, NULL);
	clGetEventProfilingInfo(readback_events[4], CL_PROFILING_COMMAND_END, sizeof(last_end), &last_end, NULL);
	gpuTimings->readBackTime = (last_end - first_start) * 1e-6f;

	for (int i = 0; i < 5; i++) {
		clReleaseEvent(readback_events[i]);
	}

	memset(screen->distance, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
	memset(screen->velocity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
	memset(screen->normalizedOpacity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
	memset(screen->opacity, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
	memset(screen->normals, 0, sizeof(float) * ScreenWidth * ScreenHeight * 3);

	float maxDistance = 0.0f, maxVelocity = 0.0f, maxOpacity = 0.0f;
	for (int i = 0; i < ScreenWidth * ScreenHeight; i++) {
		if (distances_result[i] > 0.001f && distances_result[i] > maxDistance) {
			maxDistance = distances_result[i];
		}
		if (velocities_result[i] > maxVelocity) {
			maxVelocity = velocities_result[i];
		}
		if (opacities_result[i] > maxOpacity) {
			maxOpacity = opacities_result[i];
		}
	}

	for (int y = 0; y < ScreenHeight; y++) {
		for (int x = 0; x < ScreenWidth; x++) {
			int idx = y * ScreenWidth + x;

			if (opacities_result[idx] > 0.0f) {
				if (maxDistance > 0.0f) {
					screen->distance[x][y] = (uint8_t)(255 * (1.0f - distances_result[idx] / (maxDistance + 1.0f)));
				}
				if (maxVelocity > 0.0f) {
					screen->velocity[x][y] = (uint8_t)(255 * (velocities_result[idx] / (maxVelocity + 1.0f)));
				}
				if (maxOpacity > 0.0f) {
					screen->normalizedOpacity[x][y] = (uint8_t)(255 * (opacities_result[idx] / (maxOpacity + 1.0f)));
				}
			}

			// Copy normals
			float *nr = normals_result + idx * 3;
			screen->normals[x][y][0] = nr[0];
			screen->normals[x][y][1] = nr[1];
			screen->normals[x][y][2] = nr[2];

			// Copy screen colors
			float *color = ocl->host_screen_colors_result + idx * 3;
			screen->colors[x][y][0] = (uint8_t)(color[0] * 255.0f);
			screen->colors[x][y][1] = (uint8_t)(color[1] * 255.0f);
			screen->colors[x][y][2] = (uint8_t)(color[2] * 255.0f);
		}
	}
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

	// Add skybox buffer cleanup
	if (ocl->buffer_skybox_top) clReleaseMemObject(ocl->buffer_skybox_top);
	if (ocl->buffer_skybox_bottom) clReleaseMemObject(ocl->buffer_skybox_bottom);
	if (ocl->buffer_skybox_left) clReleaseMemObject(ocl->buffer_skybox_left);
	if (ocl->buffer_skybox_right) clReleaseMemObject(ocl->buffer_skybox_right);
	if (ocl->buffer_skybox_front) clReleaseMemObject(ocl->buffer_skybox_front);
	if (ocl->buffer_skybox_back) clReleaseMemObject(ocl->buffer_skybox_back);

	if (ocl->kernel) clReleaseKernel(ocl->kernel);
	if (ocl->skybox_kernel) clReleaseKernel(ocl->skybox_kernel);
	if (ocl->triangle_kernel) clReleaseKernel(ocl->triangle_kernel);
	if (ocl->blur_kernel) clReleaseKernel(ocl->blur_kernel);
	if (ocl->normals_kernel) clReleaseKernel(ocl->normals_kernel);
	if (ocl->applyReflections_kernel) clReleaseKernel(ocl->applyReflections_kernel);

	if (ocl->program) clReleaseProgram(ocl->program);
	if (ocl->queue) clReleaseCommandQueue(ocl->queue);
	if (ocl->context) clReleaseContext(ocl->context);
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
	return (float)rand() / RAND_MAX;
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
	return keyState.keys[key] && !keyState.prevKeys[key]; // Just pressed this frame
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

	if (action == GLFW_PRESS) {
		keyState.keys[key] = true;
		printf("Key %d pressed\n", key);
	} else if (action == GLFW_RELEASE) {
		keyState.keys[key] = false;
		printf("Key %d released\n", key);
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

	printf("Mouse moved to: (%.1f, %.1f), Delta: (%.1f, %.1f)\n",
		   xpos, ypos, mouseState.deltaX, mouseState.deltaY);
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

int main() {
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
	camera.ray.origin[0] = 50.0f;
	camera.ray.origin[1] = 50.0f;
	camera.ray.origin[2] = -50.0f;
	camera.ray.direction[0] = 0.0f;
	camera.ray.direction[1] = 0.0f;
	camera.ray.direction[2] = 1.0f;
	camera.fov = 1.0f;

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

	struct Screen *screen = (struct Screen *)malloc(sizeof(struct Screen));
	if (!screen) {
		perror("Failed to allocate memory for screen");
		free(particles);
		return 1;
	}

	for (int i = 0; i < NUM_PARTICLES; i++) {
		particles->x[i] = (float)(rand() % 50 + 30);
		particles->y[i] = (float)(rand() % 50);
		particles->z[i] = (float)(rand() % 50 + 30);
		particles->xVelocity[i] = (float)(rand() % 10) / 100.0f;
		particles->yVelocity[i] = (float)(rand() % 10) / 100.0f;
		particles->zVelocity[i] = (float)(rand() % 10) / 100.0f;
	}

	struct Screen *lightScreen = (struct Screen *)malloc(sizeof(struct Screen));
	if (!lightScreen) {
		perror("Failed to allocate memory for light screen");
		free(particles);
		free(screen);
		return 1;
	}
	// Initialize light screen buffers
	clearScreen(lightScreen);

	// initialize the cursor
	struct Cursor *cursor = (struct Cursor *)malloc(sizeof(struct Cursor));
	if (!cursor) {
		perror("Failed to allocate memory for cursor");
		free(particles);
		free(screen);
		return 1;
	}
	cursor->x = 0.0f;
	cursor->y = 0.0f;
	cursor->z = 0.0f;
	cursor->active = false;

	updateGridData(particles);

	particles->bBoxMin[0] = 0.0f;
	particles->bBoxMin[1] = 0.0f;
	particles->bBoxMin[2] = 0.0f;
	particles->bBoxMax[0] = 80.0f;
	particles->bBoxMax[1] = 80.0f;
	particles->bBoxMax[2] = 80.0f;

	struct Camera lightCamera;
	lightCamera.fov = 1.0f; // Set light camera FOV to match main camera
	lightCamera.ray.origin[0] = 50.142f;
	lightCamera.ray.origin[1] = 142.607f;
	lightCamera.ray.origin[2] = -62.493f;
	// Set light camera direction to point towards the particles
	lightCamera.ray.direction[0] = -0.148f;
	lightCamera.ray.direction[1] = -0.721f;
	lightCamera.ray.direction[2] = 0.680f; // Looking down the Z-axis

	clearScreen(screen);

	float averageFPS[FrameCount];
	int averageUpdateTime = 0;
	int averageRenderTime = 0;
	int frameCount = 0;
	bool paused = false;

	struct TimePartition *timePartition = (struct TimePartition *)malloc(sizeof(struct TimePartition));
	if (!timePartition) {
		perror("Failed to allocate memory for time partition");
		free(particles);
		free(screen);
		return 1;
	}

	clock_t lastTime = clock();

	// readFileTriangles("parseObj/encoded.bin", triangles, 25.0f);

	printf("Triangles count after reading: %d\n", triangles->count);

	for (int i = 0; i <= NUMBER_OF_CUBES; i++) {
		float x = (float)(rand_01() * 500.0f);
		float y = (float)(rand_01() * 500.0f);
		float z = (float)(rand_01() * 500.0f);
		float size = 25.0f;
		float r = (float)rand_01();
		float g = (float)rand_01();
		float b = (float)rand_01();
		float Roughness = (float)rand_01();
		float Metallic = (float)rand_01();
		// float Metallic = 0.0f; // Set Metallic to 1.0f for all cubes
		float Emissive = (float)rand_01();
		CreateCube(x, y, z, size, triangles, r, g, b, Metallic, Roughness, Emissive);
	}

	CreateBoardPlane(0.0f, -20.0f, 0.0f, 50.0f, 32, triangles);

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
	int useOpenCL = initializeOpenCLWithGL(&ocl, triangles, &skyBox, &font, &bvh, window);
	if (!useOpenCL) {
		printf("Failed to initialize OpenCL-GL interop, falling back to CPU\n");
	}

	camera.renderMode = renderColor;

	bool exit = false;

	while (!glfwWindowShouldClose(window) && !exit) {
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

		updateKeyStates();

		updateMouseStates();

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
		// readCameraData(&camera);
		// readCursorData(cursor);
		readPauseData(&paused);
		clock_t endReadDataTime = clock();
		float dt1 = (float)(endReadDataTime - readDataTime) / (float)CLOCKS_PER_SEC;
		timePartition->readDataTime += dt1;

		// Update the grid data and record timing
		// First, update the grid data regardless of pause state
		clock_t startGridTime = clock();
		Step(particles, 1.0f / 60.0f); // Always update grid data
		clock_t endGridTime = clock();
		dt1 = (float)(endGridTime - startGridTime) / (float)CLOCKS_PER_SEC;
		timePartition->updateGridTime += dt1;

		// Only update particles if NOT paused
		if (!paused) {
			update_particles(particles, dt, timePartition, cursor);
		} else {
			printf("Simulation paused, skipping update\n");
		}

		clock_t afterUpdateTime = clock();
		float averageUpdateTime = (float)(afterUpdateTime - loopStartTime) / (float)CLOCKS_PER_SEC;

		clock_t startRenderTime = clock();
		render(screen, particles, &camera, cursor, timePartition, particleIndexes, &lightCamera, &ocl, triangles, &skyBox, &gpuTimings, &font);
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

		printf("FPS: %.2f, TPS: %.2f, Update: %.02f s, Render: %0.2f s\n",
			   currentFPS, TPS,
			   (averageUpdateTime),
			   (averageRenderTime));

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
	}

	// Clean up
	free(particles);
	free(screen);

	return 0;
}
