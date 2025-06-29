#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>     // for rand()
#include <unistd.h>     // for usleep()
#include <math.h>       // for sqrtf()
#include <time.h>      // for time()
#include <string.h>     // for memset()
#include <stdbool.h>    // for bool, true, false
#include <immintrin.h> // for AVX intrinsics
#include <omp.h>        // for OpenMP
#include <pthread.h> // for pthreads
#include <CL/cl.h>     // Add this line for OpenCL
#define TINYOBJ_LOADER_C_IMPLEMENTATION
#include "tinyobj_loader_c.h"
#include <jpeglib.h>

#define NUM_PARTICLES 50000
#define GRAVITY 10.0f
#define DAMPING 0.985f
#define ScreenWidth 800
#define ScreenHeight 600
#define PARTICLE_RADIUS 4
#define gridResolutionAxis 32
#define gridResolution (gridResolutionAxis * gridResolutionAxis * gridResolutionAxis)
#define temperature 8.5f
#define pressure  temperature * 0.1f
#define FrameCount 30
#define NUM_THREADS 0
#define USE_GPU 1
#define NUMBER_OF_TRIANGLES 100000
#define NUMBER_OF_CUBES 200
pthread_t threads[NUM_THREADS];

struct RawImage {
    unsigned char *data; // RGB pixel data
    int width, height, components;
};

struct RawImage* load_jpeg(const char *filename) {
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

float* convertImageToFloat(struct RawImage *img) {
    if (!img) return NULL;
    
    float *data = malloc(img->width * img->height * 3 * sizeof(float));
    if (!data) return NULL;
    
    for (int i = 0; i < img->width * img->height * img->components; i += img->components) {
        int floatIdx = (i / img->components) * 3;
        data[floatIdx + 0] = (float)img->data[i + 0] / 255.0f;     // R
        data[floatIdx + 1] = (float)img->data[i + 1] / 255.0f;     // G
        data[floatIdx + 2] = (float)img->data[i + 2] / 255.0f;     // B
    }
    
    return data;
}

bool loadSkyBox(struct SkyBox *skyBox) {  // Changed from void to bool
    skyBox->right = load_jpeg("skybox/right.jpg");
    skyBox->left = load_jpeg("skybox/left.jpg");
    skyBox->top = load_jpeg("skybox/top.jpg");
    skyBox->bottom = load_jpeg("skybox/bottom.jpg");
    skyBox->front = load_jpeg("skybox/front.jpg");
    skyBox->back = load_jpeg("skybox/back.jpg");

    if (!skyBox->right || !skyBox->left || !skyBox->top || 
        !skyBox->bottom || !skyBox->front || !skyBox->back) {
        printf("Failed to load one or more skybox images\n");
        return false;  // Return false on failure
    }
    
    return true;  // Return true on success
}

struct ThreadSync {
    volatile int ready;
    volatile int done;
    volatile int collisionReady;
    volatile int renderReady;
} __attribute__((aligned(64))) threadSync[NUM_THREADS];

#define DISABLE_MP_PROJECT_PARTICLES 1
struct ThreadData* threadData[NUM_THREADS];
// volatile int ready[NUM_THREADS];
// volatile int done[NUM_THREADS];
static int threadIds[NUM_THREADS];

#define DISABLE_MP_COLLIDE_PARTICLES 1
struct ThreadDataCollideParticles {
    struct PointSOA *particles;
    int startGridId;  // Changed from startIdx/endIdx to gridId range
    int endGridId;
    pthread_t thread;
};

#define DISABLE_MP_RENDER_PARTICLES 1
// volatile int renderReady[NUM_THREADS];
// volatile int renderDone[NUM_THREADS];
struct RenderThreadData* renderThreadData[NUM_THREADS];
struct ThreadScreen* threadScreens[NUM_THREADS];


struct ThreadScreen {
    uint8_t distance[ScreenWidth][ScreenHeight];
    uint8_t velocity[ScreenWidth][ScreenHeight];
    uint16_t particleCount[ScreenWidth][ScreenHeight];
};


struct RenderThreadData {
    struct PointSOA *particles;
    struct ParticleIndexes *particleIndexes;
    struct Camera *camera;
    struct ThreadScreen *threadScreen;
    int startParticle;
    int endParticle;
    int validParticles;
    float maxVelocity;
    float maxDistance;
    int threadId;
};

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
    cl_kernel kernel;
    cl_kernel blur_kernel;
    cl_kernel normals_kernel;
    cl_kernel triangle_kernel;  // Add triangle kernel
    cl_kernel skybox_kernel;
    // buffers
    cl_mem buffer_points;
    cl_mem buffer_velocities;
    cl_mem buffer_distances;
    cl_mem buffer_opacities;
    cl_mem buffer_velocities_screen;
    cl_mem buffer_normals;
    cl_mem buffer_distances_temp;
    cl_mem buffer_opacities_temp;
    cl_mem buffer_triangle_colors;
    
    // Add triangle buffers
    cl_mem buffer_triangle_v1;
    cl_mem buffer_triangle_v2;
    cl_mem buffer_triangle_v3;
    cl_mem buffer_triangle_normals;
    cl_mem buffer_screen_colors;

    // Triangle properties
    cl_mem buffer_triangle_roughness;
    cl_mem buffer_triangle_metallic;
    cl_mem buffer_triangle_emission;

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
};

struct Triangles {
    float v1 [NUMBER_OF_TRIANGLES * 3];
    float v2 [NUMBER_OF_TRIANGLES * 3];
    float v3 [NUMBER_OF_TRIANGLES * 3];
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
        free(top_data); free(bottom_data); free(left_data); 
        free(right_data); free(front_data); free(back_data);
        return 0;
    }
    
    ocl->buffer_skybox_bottom = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                              image_size, bottom_data, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating skybox bottom buffer: %d\n", err);
        free(top_data); free(bottom_data); free(left_data); 
        free(right_data); free(front_data); free(back_data);
        return 0;
    }
    
    ocl->buffer_skybox_left = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                            image_size, left_data, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating skybox left buffer: %d\n", err);
        free(top_data); free(bottom_data); free(left_data); 
        free(right_data); free(front_data); free(back_data);
        return 0;
    }
    
    ocl->buffer_skybox_right = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             image_size, right_data, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating skybox right buffer: %d\n", err);
        free(top_data); free(bottom_data); free(left_data); 
        free(right_data); free(front_data); free(back_data);
        return 0;
    }
    
    ocl->buffer_skybox_front = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             image_size, front_data, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating skybox front buffer: %d\n", err);
        free(top_data); free(bottom_data); free(left_data); 
        free(right_data); free(front_data); free(back_data);
        return 0;
    }
    
    ocl->buffer_skybox_back = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                            image_size, back_data, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating skybox back buffer: %d\n", err);
        free(top_data); free(bottom_data); free(left_data); 
        free(right_data); free(front_data); free(back_data);
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
    float color1R = 0.9f, color1G = 0.9f, color1B = 0.9f; // Light color (white-ish)
    float color2R = 0.1f, color2G = 0.1f, color2B = 0.1f; // Dark color (black-ish)
    float Metallic = 0.95f, Roughness = 0.75f, Emission = 0.25f; // Material properties
    float Metallic1 = 0.10f, Roughness1 = 0.25f, Emission1 = 0.85f; // Material properties
    
    for (int i = 0; i < numberOfSquares; i++) {
        for (int j = 0; j < numberOfSquares; j++) {
            // Calculate square position (centered around centerX, centerZ)
            float x1 = centerX + (i - numberOfSquares / 2.0f) * size;
            float y1 = centerY;  // Keep Y constant for horizontal plane
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
            AddTriangle(triangles, x1, y1, z1,    // bottom-left
                                   x3, y3, z3,    // top-left  
                                   x2, y2, z2,    // bottom-right
                                   colorR, colorG, colorB, roughness, metallic, emission);
            
            // Triangle 2: top-left, top-right, bottom-right (when viewed from above)
            AddTriangle(triangles, x3, y3, z3,    // top-left
                                   x4, y4, z4,    // top-right
                                   x2, y2, z2,    // bottom-right
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

    AddTriangle(triangles, v1[0], v1[1], v1[2], v3[0], v3[1], v3[2], v2[0], v2[1], v2[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    AddTriangle(triangles, v1[0], v1[1], v1[2], v4[0], v4[1], v4[2], v3[0], v3[1], v3[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    
    AddTriangle(triangles, v5[0], v5[1], v5[2], v6[0], v6[1], v6[2], v7[0], v7[1], v7[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    AddTriangle(triangles, v5[0], v5[1], v5[2], v7[0], v7[1], v7[2], v8[0], v8[1], v8[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    
    AddTriangle(triangles, v1[0], v1[1], v1[2], v5[0], v5[1], v5[2], v8[0], v8[1], v8[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    AddTriangle(triangles, v1[0], v1[1], v1[2], v8[0], v8[1], v8[2], v4[0], v4[1], v4[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    
    AddTriangle(triangles, v2[0], v2[1], v2[2], v3[0], v3[1], v3[2], v7[0], v7[1], v7[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    AddTriangle(triangles, v2[0], v2[1], v2[2], v7[0], v7[1], v7[2], v6[0], v6[1], v6[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    
    AddTriangle(triangles, v4[0], v4[1], v4[2], v8[0], v8[1], v8[2], v7[0], v7[1], v7[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    AddTriangle(triangles, v4[0], v4[1], v4[2], v7[0], v7[1], v7[2], v3[0], v3[1], v3[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    
    AddTriangle(triangles, v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v6[0], v6[1], v6[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
    AddTriangle(triangles, v1[0], v1[1], v1[2], v6[0], v6[1], v6[2], v5[0], v5[1], v5[2], colorR, colorG, colorB, Roughness, Metallic, Emission);
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

void renderSkyboxOpenCL(struct OpenCLContext *ocl, struct Camera *camera, struct SkyBox *skyBox) {
    cl_int err;
    
    // Set skybox kernel arguments
    cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
    cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
    cl_float fov = camera->fov;
    cl_int screen_width = ScreenWidth;
    cl_int screen_height = ScreenHeight;
    cl_int skybox_width = skyBox->top->width;
    cl_int skybox_height = skyBox->top->height;
    
    err = clSetKernelArg(ocl->skybox_kernel, 0, sizeof(cl_mem), &ocl->buffer_screen_colors);
    err |= clSetKernelArg(ocl->skybox_kernel, 1, sizeof(cl_float3), &cam_pos);
    err |= clSetKernelArg(ocl->skybox_kernel, 2, sizeof(cl_float3), &cam_dir);
    err |= clSetKernelArg(ocl->skybox_kernel, 3, sizeof(cl_float), &fov);
    err |= clSetKernelArg(ocl->skybox_kernel, 4, sizeof(cl_int), &screen_width);
    err |= clSetKernelArg(ocl->skybox_kernel, 5, sizeof(cl_int), &screen_height);
    err |= clSetKernelArg(ocl->skybox_kernel, 6, sizeof(cl_mem), &ocl->buffer_skybox_top);
    err |= clSetKernelArg(ocl->skybox_kernel, 7, sizeof(cl_mem), &ocl->buffer_skybox_bottom);
    err |= clSetKernelArg(ocl->skybox_kernel, 8, sizeof(cl_mem), &ocl->buffer_skybox_left);
    err |= clSetKernelArg(ocl->skybox_kernel, 9, sizeof(cl_mem), &ocl->buffer_skybox_right);
    err |= clSetKernelArg(ocl->skybox_kernel, 10, sizeof(cl_mem), &ocl->buffer_skybox_front);
    err |= clSetKernelArg(ocl->skybox_kernel, 11, sizeof(cl_mem), &ocl->buffer_skybox_back);
    err |= clSetKernelArg(ocl->skybox_kernel, 12, sizeof(cl_int), &skybox_width);
    err |= clSetKernelArg(ocl->skybox_kernel, 13, sizeof(cl_int), &skybox_height);
    
    if (err != CL_SUCCESS) {
        printf("Error setting skybox kernel arguments: %d\n", err);
        return;
    }
    
    // Execute skybox kernel
    size_t global_work_size[2] = {ScreenWidth, ScreenHeight};
    err = clEnqueueNDRangeKernel(ocl->queue, ocl->skybox_kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing skybox kernel: %d\n", err);
        return;
    }

    clFinish(ocl->queue);
    float test_pixel[3];
    err = clEnqueueReadBuffer(ocl->queue, ocl->buffer_screen_colors, CL_TRUE, 0, 
                             3 * sizeof(float), test_pixel, 0, NULL, NULL);
    if (err == CL_SUCCESS) {
        printf("Skybox test pixel: R=%.3f G=%.3f B=%.3f\n", 
               test_pixel[0], test_pixel[1], test_pixel[2]);
    }
}

void renderTrianglesOpenCL(struct OpenCLContext *ocl, struct Triangles *triangles, struct Camera *camera, struct Screen *screen, struct SkyBox *skyBox) {
    if (triangles->count == 0) return;
    
    cl_int err;
    
    // Only set the camera parameters that change each frame
    cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
    cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
    cl_float fov = camera->fov;
    
    // Only update arguments 6-8 (camera data)
    err = clSetKernelArg(ocl->triangle_kernel, 6, sizeof(cl_float3), &cam_pos);
    err |= clSetKernelArg(ocl->triangle_kernel, 7, sizeof(cl_float3), &cam_dir);
    err |= clSetKernelArg(ocl->triangle_kernel, 8, sizeof(cl_float), &fov);
    
    if (err != CL_SUCCESS) {
        printf("Error setting camera kernel arguments: %d\n", err);
        return;
    }
    
    // Execute triangle rendering kernel
    size_t global_work_size = triangles->count;
    err = clEnqueueNDRangeKernel(ocl->queue, ocl->triangle_kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing triangle kernel: %d\n", err);
        return;
    }
    clFinish(ocl->queue);
}

// Initialize render thread system
void initializeRenderThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        threadSync[i].renderReady = 0;
        
        // Allocate thread screen buffer
        threadScreens[i] = (struct ThreadScreen*)malloc(sizeof(struct ThreadScreen));
        if (!threadScreens[i]) {
            printf("Failed to allocate thread screen for thread %d\n", i);
            exit(1);
        }
        
        // Allocate render thread data
        renderThreadData[i] = (struct RenderThreadData*)malloc(sizeof(struct RenderThreadData));
        if (!renderThreadData[i]) {
            printf("Failed to allocate render thread data for thread %d\n", i);
            exit(1);
        }
        renderThreadData[i]->threadId = i;
        renderThreadData[i]->threadScreen = threadScreens[i];
    }
}

// volatile int collisionReady[NUM_THREADS];
// volatile int collisionDone[NUM_THREADS];
struct ThreadDataCollideParticles* collisionThreadData[NUM_THREADS];

void initializeCollisionThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        threadSync[i].collisionReady = 0;
        collisionThreadData[i] = (struct ThreadDataCollideParticles*)malloc(sizeof(struct ThreadDataCollideParticles));
        if (!collisionThreadData[i]) {
            printf("Failed to allocate collision thread data for thread %d\n", i);
            exit(1);
        }
    }
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

// Function prototypes
void calculateParticleScreenCoordinates(struct ThreadData *threadData);
void *threadFunction(void *arg);
void CollideParticlesInGridInThread(struct ThreadDataCollideParticles *data);
void renderParticlesToBuffer(struct RenderThreadData *data);
void projectParticlesOpenCL(struct OpenCLContext *ocl, struct PointSOA *particles, struct Camera *camera, struct Screen *screen, struct Triangles *triangles, struct SkyBox *skyBox);

void createThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        threadSync[i].ready = 0;
        threadSync[i].done = 0;
        threadSync[i].collisionReady = 0;
        threadSync[i].renderReady = 0;
        threadIds[i] = i;
        pthread_create(&threads[i], NULL, threadFunction, &threadIds[i]);
    }
}


void *threadFunction(void *arg) {
    int threadId = *(int *)arg;
    
    while (!threadSync[threadId].done) {
        // Wait for any type of work
        while (!threadSync[threadId].ready && 
               !threadSync[threadId].collisionReady && 
               !threadSync[threadId].renderReady && 
               !threadSync[threadId].done) {
            #if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
            #else
                usleep(100); 
            #endif
        }
        
        if (threadSync[threadId].done) {
            break;
        }
        
        // Handle projection work
        if (DISABLE_MP_PROJECT_PARTICLES == 1) {
            if (threadSync[threadId].ready) {
                calculateParticleScreenCoordinates(threadData[threadId]);
                threadSync[threadId].ready = 0;
            }
        }
        
        // Handle collision work
        if (threadSync[threadId].collisionReady) {
            CollideParticlesInGridInThread(collisionThreadData[threadId]);
            threadSync[threadId].collisionReady = 0;
        }

        if (threadSync[threadId].renderReady) {
            renderParticlesToBuffer(renderThreadData[threadId]);
            threadSync[threadId].renderReady = 0;
        }
    }
    
    return NULL;
}


void joinThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        threadSync[i].done = 1;
        pthread_join(threads[i], NULL);
        
        // Free collision thread data
        if (collisionThreadData[i]) {
            free(collisionThreadData[i]);
        }

         // Free thread screen buffers
        if (threadScreens[i]) {
            free(threadScreens[i]);
        }
    }
}

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


struct PointSOA {
    float   x[NUM_PARTICLES];
    float   y[NUM_PARTICLES];
    float   z[NUM_PARTICLES];
    float   xVelocity[NUM_PARTICLES];
    float   yVelocity[NUM_PARTICLES];
    float   zVelocity[NUM_PARTICLES];
    float   totalVelocity[NUM_PARTICLES];
    // float   distance[NUM_PARTICLES];
    float   bBoxMin[3];
    float   bBoxMax[3];
    int     gridID[NUM_PARTICLES];
    int     startIndex[gridResolution];
    int     numberOfParticle[gridResolution];
    int     screenX[NUM_PARTICLES];
    int     screenY[NUM_PARTICLES];
};

struct ThreadsData {
    struct ThreadData *threadData[NUM_THREADS]; // array of thread data
};

struct ThreadData {
    struct Camera *camera; // pointer to the camera
    float *xPtrStart;      // pointer to start of the x array
    float *yPtrStart;      // pointer to start of the y array
    float *zPtrStart;      // pointer to start of the z array
    float *xVelocityPtrStart;    // pointer to start of the xVelocity array
    float *yVelocityPtrStart;    // pointer to start of the yVelocity array
    float *zVelocityPtrStart;    // pointer to start of the zVelocity array
    float *totalVelocityPtrStart; // pointer to start of the totalVelocity array
    int *screenXPtrStart;      // pointer to start of the screenX array (changed from float* to int*)
    int *screenYPtrStart;      // pointer to start of the screenY array (changed from float* to int*)
    int length;            // length of the arrays
    float maxVelocity; // maximum velocity for normalization
    pthread_t thread;  // Add thread handle
};

void asignDataToThreads(struct PointSOA *particles, struct Camera *camera, int ValidParticles) {
    // Use dynamic chunk sizes for better load balancing
    // Last thread may get slightly more/fewer particles
    clock_t startTime = clock();
    int baseChunkSize = ValidParticles / NUM_THREADS;
    int remainder = ValidParticles % NUM_THREADS;
    
    int startIdx = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        // Calculate chunk size - distribute remainder particles one per thread
        int chunkSize = baseChunkSize + (i < remainder ? 1 : 0);
        
        threadData[i]->camera = camera;
        threadData[i]->xPtrStart = &particles->x[startIdx];
        threadData[i]->yPtrStart = &particles->y[startIdx];
        threadData[i]->zPtrStart = &particles->z[startIdx];
        threadData[i]->xVelocityPtrStart = &particles->xVelocity[startIdx];
        threadData[i]->yVelocityPtrStart = &particles->yVelocity[startIdx];
        threadData[i]->zVelocityPtrStart = &particles->zVelocity[startIdx];
        threadData[i]->totalVelocityPtrStart = &particles->totalVelocity[startIdx];
        threadData[i]->screenXPtrStart = &particles->screenX[startIdx];
        threadData[i]->screenYPtrStart = &particles->screenY[startIdx];
        threadData[i]->length = chunkSize;
        threadData[i]->maxVelocity = 0.0f;
        
        startIdx += chunkSize;
    }
    printf("Time taken to assign data to threads: %f seconds\n", (float)(clock() - startTime) / CLOCKS_PER_SEC);
}

void drawBoundingBox(struct Screen *screen, float bBoxMax[3], float bBoxMin[3], struct Camera *camera) {
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

    // Define the 8 corners of the bounding box
    float corners[8][3] = {
        {bBoxMin[0], bBoxMin[1], bBoxMin[2]}, // 0: min, min, min
        {bBoxMax[0], bBoxMin[1], bBoxMin[2]}, // 1: max, min, min
        {bBoxMin[0], bBoxMax[1], bBoxMin[2]}, // 2: min, max, min
        {bBoxMax[0], bBoxMax[1], bBoxMin[2]}, // 3: max, max, min
        {bBoxMin[0], bBoxMin[1], bBoxMax[2]}, // 4: min, min, max
        {bBoxMax[0], bBoxMin[1], bBoxMax[2]}, // 5: max, min, max
        {bBoxMin[0], bBoxMax[1], bBoxMax[2]}, // 6: min, max, max
        {bBoxMax[0], bBoxMax[1], bBoxMax[2]}  // 7: max, max, max
    };
    
    // Project corners to screen space
    int screenPoints[8][2];
    bool inFront[8];
    
    for (int i = 0; i < 8; i++) {
        // Vector from camera to corner
        float x = corners[i][0] - camera->ray.origin[0];
        float y = corners[i][1] - camera->ray.origin[1];
        float z = corners[i][2] - camera->ray.origin[2];
        
        // Check if corner is in front of camera
        float dotProduct = x * camera->ray.direction[0] + 
                           y * camera->ray.direction[1] + 
                           z * camera->ray.direction[2];
        
        inFront[i] = (dotProduct > 0);
        
        if (inFront[i]) {
            // Project point to screen space
            float fovScale = 1.0f / (dotProduct * camera->fov);
            float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
            float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;
            
            screenPoints[i][0] = (int)(screenRight * halfWidth + halfWidth);
            screenPoints[i][1] = (int)(-screenUp * halfHeight + halfHeight);
            
            // Clamp to screen bounds
            screenPoints[i][0] = screenPoints[i][0] < 0 ? 0 : (screenPoints[i][0] >= ScreenWidth ? ScreenWidth - 1 : screenPoints[i][0]);
            screenPoints[i][1] = screenPoints[i][1] < 0 ? 0 : (screenPoints[i][1] >= ScreenHeight ? ScreenHeight - 1 : screenPoints[i][1]);
        }
    }
    
    // Define edges by vertex indices
    int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0}, // Bottom face
        {4, 5}, {5, 7}, {7, 6}, {6, 4}, // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Connecting edges
    };
    
    // Draw edges
    for (int i = 0; i < 12; i++) {
        int v1 = edges[i][0];
        int v2 = edges[i][1];
        
        // Only draw edge if both vertices are in front of camera
        if (inFront[v1] && inFront[v2]) {
            int x0 = screenPoints[v1][0];
            int y0 = screenPoints[v1][1];
            int x1 = screenPoints[v2][0];
            int y1 = screenPoints[v2][1];
            
            int dx = abs(x1 - x0);
            int dy = -abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx + dy;
            int e2;
            
            while (1) {
                // Draw pixel if it's within screen bounds
                if (x0 >= 0 && x0 < ScreenWidth && y0 >= 0 && y0 < ScreenHeight) {
                    // screen->distance[x0][y0] = 255; // Distance
                    screen->velocity[x0][y0] = 255; // Velocity
                    screen->normalizedOpacity[x0][y0] = 255; // Normalized Opacity
                }
                
                if (x0 == x1 && y0 == y1) break;
                e2 = 2 * err;
                if (e2 >= dy) {
                    if (x0 == x1) break;
                    err += dy;
                    x0 += sx;
                }
                if (e2 <= dx) {
                    if (y0 == y1) break;
                    err += dx;
                    y0 += sy;
                }
            }
        }
    }
}

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
        fread(&cursor->active, sizeof(bool), 1, file) != 1|| 
        fread(&cursor->force, sizeof(float), 1, file) != 1) {
        
        fclose(file);
        return; 
    }
    
    fclose(file);
}

void CollideParticlesInGridInThread(struct ThreadDataCollideParticles *data) {
    struct PointSOA *particles = data->particles;
    
    // Constants as AVX vectors
    const __m256 particleRadiusVec = _mm256_set1_ps(PARTICLE_RADIUS);
    const __m256 minDistVec = _mm256_set1_ps(3.0f * PARTICLE_RADIUS);
    const __m256 minDistSquaredVec = _mm256_mul_ps(minDistVec, minDistVec);
    const __m256 halfVec = _mm256_set1_ps(0.5f);
    const __m256 dampingVec = _mm256_set1_ps(DAMPING);
    const __m256 correctionFactorVec = _mm256_set1_ps(0.01f);
    const __m256 zeroVec = _mm256_setzero_ps();

    // Process assigned grid cells
    for (int gridId = data->startGridId; gridId < data->endGridId; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        int endIdx = startIdx + particles->numberOfParticle[gridId];

        // Check collisions between all particles in this cell
        for (int i = startIdx; i < endIdx; i++) {
            // Load particle i data once
            __m256 xi = _mm256_set1_ps(particles->x[i]);
            __m256 yi = _mm256_set1_ps(particles->y[i]);
            __m256 zi = _mm256_set1_ps(particles->z[i]);
            __m256 vxi = _mm256_set1_ps(particles->xVelocity[i]);
            __m256 vyi = _mm256_set1_ps(particles->yVelocity[i]);
            __m256 vzi = _mm256_set1_ps(particles->zVelocity[i]);
            
            // Process 8 particles at a time where possible
            for (int j = i + 1; j < endIdx; j += 8) {
                int remaining = endIdx - j;
                if (remaining >= 8) {
                    // Load 8 particles at once
                    __m256 xj = _mm256_loadu_ps(&particles->x[j]);
                    __m256 yj = _mm256_loadu_ps(&particles->y[j]);
                    __m256 zj = _mm256_loadu_ps(&particles->z[j]);
                    
                    // Calculate distance vectors
                    __m256 dx = _mm256_sub_ps(xj, xi);
                    __m256 dy = _mm256_sub_ps(yj, yi);
                    __m256 dz = _mm256_sub_ps(zj, zi);
                    
                    // Calculate squared distance
                    __m256 distSquared = _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_mul_ps(dx, dx),
                            _mm256_mul_ps(dy, dy)
                        ),
                        _mm256_mul_ps(dz, dz)
                    );
                    
                    // Compare with minimum distance squared
                    __m256 collisionMask = _mm256_cmp_ps(distSquared, minDistSquaredVec, _CMP_LT_OQ);
                    
                    // Skip to next batch if no collisions
                    if (_mm256_testz_ps(collisionMask, collisionMask)) {
                        continue;
                    }
                    
                    // Get mask as integer for processing individual lanes
                    int mask = _mm256_movemask_ps(collisionMask);
                    
                    // Process collisions for each set bit in the mask
                    for (int k = 0; k < 8; k++) {
                        if (mask & (1 << k)) {
                            int jIdx = j + k;
                            
                            // Recalculate with scalar math for this specific pair
                            float dx_s = particles->x[jIdx] - particles->x[i];
                            float dy_s = particles->y[jIdx] - particles->y[i];
                            float dz_s = particles->z[jIdx] - particles->z[i];
                            
                            float distSquared_s = dx_s*dx_s + dy_s*dy_s + dz_s*dz_s;
                            float minDist = 3.0f * PARTICLE_RADIUS;
                            
                            // Double-check collision (should always be true due to mask)
                            if (distSquared_s < minDist * minDist) {
                                // Calculate collision normal
                                float dist = sqrtf(distSquared_s);
                                float nx = dx_s / dist;
                                float ny = dy_s / dist;
                                float nz = dz_s / dist;
                                
                                // Calculate relative velocity along normal
                                float vx = particles->xVelocity[jIdx] - particles->xVelocity[i];
                                float vy = particles->yVelocity[jIdx] - particles->yVelocity[i];
                                float vz = particles->zVelocity[jIdx] - particles->zVelocity[i];
                                
                                float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                                
                                // Only collide if particles are moving toward each other
                                if (velocityAlongNormal < 0) {
                                    // Calculate impulse scalar (assuming equal mass)
                                    float impulse = -2.0f * velocityAlongNormal;
                                    
                                    // Apply impulse
                                    particles->xVelocity[i] -= impulse * nx * 0.5f;
                                    particles->yVelocity[i] -= impulse * ny * 0.5f;
                                    particles->zVelocity[i] -= impulse * nz * 0.5f;
                                    
                                    particles->xVelocity[jIdx] += impulse * nx * 0.5f;
                                    particles->yVelocity[jIdx] += impulse * ny * 0.5f;
                                    particles->zVelocity[jIdx] += impulse * nz * 0.5f;
                                    
                                    // Add some damping to collision
                                    particles->xVelocity[i] *= DAMPING;
                                    particles->yVelocity[i] *= DAMPING;
                                    particles->zVelocity[i] *= DAMPING;
                                    
                                    particles->xVelocity[jIdx] *= DAMPING;
                                    particles->yVelocity[jIdx] *= DAMPING;
                                    particles->zVelocity[jIdx] *= DAMPING;
                                    
                                    // Push particles apart to avoid sticking
                                    float correction = (minDist - dist) * 0.01f;
                                    particles->x[i] -= nx * correction;
                                    particles->y[i] -= ny * correction;
                                    particles->z[i] -= nz * correction;
                                    
                                    particles->x[jIdx] += nx * correction;
                                    particles->y[jIdx] += ny * correction;
                                    particles->z[jIdx] += nz * correction;
                                    
                                    // Update cached values for particle i
                                    xi = _mm256_set1_ps(particles->x[i]);
                                    yi = _mm256_set1_ps(particles->y[i]);
                                    zi = _mm256_set1_ps(particles->z[i]);
                                    vxi = _mm256_set1_ps(particles->xVelocity[i]);
                                    vyi = _mm256_set1_ps(particles->yVelocity[i]);
                                    vzi = _mm256_set1_ps(particles->zVelocity[i]);
                                }
                            }
                        }
                    }
                } else {
                    // Handle remaining particles with scalar code
                    for (int jIdx = j; jIdx < endIdx; jIdx++) {
                        float dx = particles->x[jIdx] - particles->x[i];
                        float dy = particles->y[jIdx] - particles->y[i];
                        float dz = particles->z[jIdx] - particles->z[i];
                        
                        float distSquared = dx*dx + dy*dy + dz*dz;
                        float minDist = 3.0f * PARTICLE_RADIUS;
                        
                        if (distSquared < minDist * minDist) {
                            float dist = sqrtf(distSquared);
                            float nx = dx / dist;
                            float ny = dy / dist;
                            float nz = dz / dist;
                            
                            float vx = particles->xVelocity[jIdx] - particles->xVelocity[i];
                            float vy = particles->yVelocity[jIdx] - particles->yVelocity[i];
                            float vz = particles->zVelocity[jIdx] - particles->zVelocity[i];
                            
                            float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                            
                            if (velocityAlongNormal < 0) {
                                float impulse = -2.0f * velocityAlongNormal;
                                
                                particles->xVelocity[i] -= impulse * nx * 0.5f;
                                particles->yVelocity[i] -= impulse * ny * 0.5f;
                                particles->zVelocity[i] -= impulse * nz * 0.5f;
                                
                                particles->xVelocity[jIdx] += impulse * nx * 0.5f;
                                particles->yVelocity[jIdx] += impulse * ny * 0.5f;
                                particles->zVelocity[jIdx] += impulse * nz * 0.5f;
                                
                                particles->xVelocity[i] *= DAMPING;
                                particles->yVelocity[i] *= DAMPING;
                                particles->zVelocity[i] *= DAMPING;
                                
                                particles->xVelocity[jIdx] *= DAMPING;
                                particles->yVelocity[jIdx] *= DAMPING;
                                particles->zVelocity[jIdx] *= DAMPING;
                                
                                float correction = (minDist - dist) * 0.01f;
                                particles->x[i] -= nx * correction;
                                particles->y[i] -= ny * correction;
                                particles->z[i] -= nz * correction;
                                
                                particles->x[jIdx] += nx * correction;
                                particles->y[jIdx] += ny * correction;
                                particles->z[jIdx] += nz * correction;
                            }
                        }
                    }
                }
            }
        }
    }
}

void CollideParticlesInGrid(struct PointSOA *particles) {
    if (DISABLE_MP_COLLIDE_PARTICLES == 0 ) {
        clock_t startTime = clock();
        int gridCellsPerThread = gridResolution / NUM_THREADS;
        int remainder = gridResolution % NUM_THREADS;
        
        // Assign work to existing threads
        int startGridId = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            int cellsForThisThread = gridCellsPerThread + (i < remainder ? 1 : 0);
            
            collisionThreadData[i]->particles = particles;
            collisionThreadData[i]->startGridId = startGridId;
            collisionThreadData[i]->endGridId = startGridId + cellsForThisThread;
            
            startGridId += cellsForThisThread;
        }
        
        // Signal threads to start collision processing
        for (int i = 0; i < NUM_THREADS; i++) {
            threadSync[i].collisionReady = 1;
        }
        
        // Wait for all threads to complete collision work
        for (int i = 0; i < NUM_THREADS; i++) {
            while (threadSync[i].collisionReady) {
                #if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                #else
                    usleep(100);
                #endif
            }
        }
    }
    else {
        // Constants as AVX vectors
        clock_t startTime = clock();
        const __m256 particleRadiusVec = _mm256_set1_ps(PARTICLE_RADIUS);
        const __m256 minDistVec = _mm256_set1_ps(3.0f * PARTICLE_RADIUS);
        const __m256 minDistSquaredVec = _mm256_mul_ps(minDistVec, minDistVec);
        const __m256 halfVec = _mm256_set1_ps(0.5f);
        const __m256 dampingVec = _mm256_set1_ps(DAMPING);
        const __m256 correctionFactorVec = _mm256_set1_ps(0.01f);
        const __m256 zeroVec = _mm256_setzero_ps();
        
        for (int gridId = 0; gridId < gridResolution; gridId++) {
            int startIdx = particles->startIndex[gridId];
            if (startIdx == -1) continue; // No particles in this grid cell
            int endIdx = startIdx + particles->numberOfParticle[gridId];
            
            // Check collisions between all particles in this cell
            for (int i = startIdx; i < endIdx; i++) {
                // Load particle i data once
                __m256 xi = _mm256_set1_ps(particles->x[i]);
                __m256 yi = _mm256_set1_ps(particles->y[i]);
                __m256 zi = _mm256_set1_ps(particles->z[i]);
                __m256 vxi = _mm256_set1_ps(particles->xVelocity[i]);
                __m256 vyi = _mm256_set1_ps(particles->yVelocity[i]);
                __m256 vzi = _mm256_set1_ps(particles->zVelocity[i]);
                
                // Process 8 particles at a time where possible
                for (int j = i + 1; j < endIdx; j += 8) {
                    int remaining = endIdx - j;
                    if (remaining >= 8) {
                        // Load 8 particles at once
                        __m256 xj = _mm256_loadu_ps(&particles->x[j]);
                        __m256 yj = _mm256_loadu_ps(&particles->y[j]);
                        __m256 zj = _mm256_loadu_ps(&particles->z[j]);
                        
                        // Calculate distance vectors
                        __m256 dx = _mm256_sub_ps(xj, xi);
                        __m256 dy = _mm256_sub_ps(yj, yi);
                        __m256 dz = _mm256_sub_ps(zj, zi);
                        
                        // Calculate squared distance
                        __m256 distSquared = _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_mul_ps(dx, dx),
                                _mm256_mul_ps(dy, dy)
                            ),
                            _mm256_mul_ps(dz, dz)
                        );
                        
                        // Compare with minimum distance squared
                        __m256 collisionMask = _mm256_cmp_ps(distSquared, minDistSquaredVec, _CMP_LT_OQ);
                        
                        // Skip to next batch if no collisions
                        if (_mm256_testz_ps(collisionMask, collisionMask)) {
                            continue;
                        }
                        
                        // Get mask as integer for processing individual lanes
                        int mask = _mm256_movemask_ps(collisionMask);
                        
                        // Process collisions for each set bit in the mask
                        for (int k = 0; k < 8; k++) {
                            if (mask & (1 << k)) {
                                int jIdx = j + k;
                                
                                // Recalculate with scalar math for this specific pair
                                float dx_s = particles->x[jIdx] - particles->x[i];
                                float dy_s = particles->y[jIdx] - particles->y[i];
                                float dz_s = particles->z[jIdx] - particles->z[i];
                                
                                float distSquared_s = dx_s*dx_s + dy_s*dy_s + dz_s*dz_s;
                                float minDist = 3.0f * PARTICLE_RADIUS;
                                
                                // Double-check collision (should always be true due to mask)
                                if (distSquared_s < minDist * minDist) {
                                    // Calculate collision normal
                                    float dist = sqrtf(distSquared_s);
                                    float nx = dx_s / dist;
                                    float ny = dy_s / dist;
                                    float nz = dz_s / dist;
                                    
                                    // Calculate relative velocity along normal
                                    float vx = particles->xVelocity[jIdx] - particles->xVelocity[i];
                                    float vy = particles->yVelocity[jIdx] - particles->yVelocity[i];
                                    float vz = particles->zVelocity[jIdx] - particles->zVelocity[i];
                                    
                                    float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                                    
                                    // Only collide if particles are moving toward each other
                                    if (velocityAlongNormal < 0) {
                                        // Calculate impulse scalar (assuming equal mass)
                                        float impulse = -2.0f * velocityAlongNormal;
                                        
                                        // Apply impulse
                                        particles->xVelocity[i] -= impulse * nx * 0.5f;
                                        particles->yVelocity[i] -= impulse * ny * 0.5f;
                                        particles->zVelocity[i] -= impulse * nz * 0.5f;
                                        
                                        particles->xVelocity[jIdx] += impulse * nx * 0.5f;
                                        particles->yVelocity[jIdx] += impulse * ny * 0.5f;
                                        particles->zVelocity[jIdx] += impulse * nz * 0.5f;
                                        
                                        // Add some damping to collision
                                        particles->xVelocity[i] *= DAMPING;
                                        particles->yVelocity[i] *= DAMPING;
                                        particles->zVelocity[i] *= DAMPING;
                                        
                                        particles->xVelocity[jIdx] *= DAMPING;
                                        particles->yVelocity[jIdx] *= DAMPING;
                                        particles->zVelocity[jIdx] *= DAMPING;
                                        
                                        // Push particles apart to avoid sticking
                                        float correction = (minDist - dist) * 0.01f;
                                        particles->x[i] -= nx * correction;
                                        particles->y[i] -= ny * correction;
                                        particles->z[i] -= nz * correction;
                                        
                                        particles->x[jIdx] += nx * correction;
                                        particles->y[jIdx] += ny * correction;
                                        particles->z[jIdx] += nz * correction;
                                        
                                        // Update cached values for particle i
                                        xi = _mm256_set1_ps(particles->x[i]);
                                        yi = _mm256_set1_ps(particles->y[i]);
                                        zi = _mm256_set1_ps(particles->z[i]);
                                        vxi = _mm256_set1_ps(particles->xVelocity[i]);
                                        vyi = _mm256_set1_ps(particles->yVelocity[i]);
                                        vzi = _mm256_set1_ps(particles->zVelocity[i]);
                                    }
                                }
                            }
                        }
                    } else {
                        // Handle remaining particles with scalar code
                        for (int jIdx = j; jIdx < endIdx; jIdx++) {
                            float dx = particles->x[jIdx] - particles->x[i];
                            float dy = particles->y[jIdx] - particles->y[i];
                            float dz = particles->z[jIdx] - particles->z[i];
                            
                            float distSquared = dx*dx + dy*dy + dz*dz;
                            float minDist = 3.0f * PARTICLE_RADIUS;
                            
                            if (distSquared < minDist * minDist) {
                                float dist = sqrtf(distSquared);
                                float nx = dx / dist;
                                float ny = dy / dist;
                                float nz = dz / dist;
                                
                                float vx = particles->xVelocity[jIdx] - particles->xVelocity[i];
                                float vy = particles->yVelocity[jIdx] - particles->yVelocity[i];
                                float vz = particles->zVelocity[jIdx] - particles->zVelocity[i];
                                
                                float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                                
                                if (velocityAlongNormal < 0) {
                                    float impulse = -2.0f * velocityAlongNormal;
                                    
                                    particles->xVelocity[i] -= impulse * nx * 0.5f;
                                    particles->yVelocity[i] -= impulse * ny * 0.5f;
                                    particles->zVelocity[i] -= impulse * nz * 0.5f;
                                    
                                    particles->xVelocity[jIdx] += impulse * nx * 0.5f;
                                    particles->yVelocity[jIdx] += impulse * ny * 0.5f;
                                    particles->zVelocity[jIdx] += impulse * nz * 0.5f;
                                    
                                    particles->xVelocity[i] *= DAMPING;
                                    particles->yVelocity[i] *= DAMPING;
                                    particles->zVelocity[i] *= DAMPING;
                                    
                                    particles->xVelocity[jIdx] *= DAMPING;
                                    particles->yVelocity[jIdx] *= DAMPING;
                                    particles->zVelocity[jIdx] *= DAMPING;
                                    
                                    float correction = (minDist - dist) * 0.01f;
                                    particles->x[i] -= nx * correction;
                                    particles->y[i] -= ny * correction;
                                    particles->z[i] -= nz * correction;
                                    
                                    particles->x[jIdx] += nx * correction;
                                    particles->y[jIdx] += ny * correction;
                                    particles->z[jIdx] += nz * correction;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void swapParticles(struct PointSOA *particles, int i, int j) {
    // Swap positions
    float tempX = particles->x[i];
    float tempY = particles->y[i];
    float tempZ = particles->z[i];
    particles->x[i] = particles->x[j];
    particles->y[i] = particles->y[j];
    particles->z[i] = particles->z[j];
    particles->x[j] = tempX;
    particles->y[j] = tempY;
    particles->z[j] = tempZ;
    
    // Swap velocities
    float tempVX = particles->xVelocity[i];
    float tempVY = particles->yVelocity[i];
    float tempVZ = particles->zVelocity[i];
    particles->xVelocity[i] = particles->xVelocity[j];
    particles->yVelocity[i] = particles->yVelocity[j];
    particles->zVelocity[i] = particles->zVelocity[j];
    particles->xVelocity[j] = tempVX;
    particles->yVelocity[j] = tempVY;
    particles->zVelocity[j] = tempVZ;
    
    // Swap grid IDs
    int tempID = particles->gridID[i];
    particles->gridID[i] = particles->gridID[j];
    particles->gridID[j] = tempID;
}

int positionToGridId(struct PointSOA *particles, float position[3]) {
    // Calculate grid ID based on position
    int xIndex = (int)((position[0] - particles->bBoxMin[0]) / (particles->bBoxMax[0] - particles->bBoxMin[0]) * gridResolutionAxis);
    int yIndex = (int)((position[1] - particles->bBoxMin[1]) / (particles->bBoxMax[1] - particles->bBoxMin[1]) * gridResolutionAxis);
    int zIndex = (int)((position[2] - particles->bBoxMin[2]) / (particles->bBoxMax[2] - particles->bBoxMin[2]) * gridResolutionAxis);
    
    // Clamp indices to grid resolution
    if (xIndex < 0) xIndex = 0;
    if (xIndex >= gridResolutionAxis) xIndex = gridResolutionAxis - 1;
    if (yIndex < 0) yIndex = 0;
    if (yIndex >= gridResolutionAxis) yIndex = gridResolutionAxis - 1;
    if (zIndex < 0) zIndex = 0;
    if (zIndex >= gridResolutionAxis) zIndex = gridResolutionAxis - 1;
    // Convert 3D indices to 1D grid ID
    return xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
}

void CalculateCenterOfCell(struct PointSOA *particles, int index, float centerOfCell[3]) {
    int xIndex = index % gridResolutionAxis;
    int yIndex = (index / gridResolutionAxis) % gridResolutionAxis;
    int zIndex = index / (gridResolutionAxis * gridResolutionAxis);
    
    float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
    float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
    float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
    centerOfCell[0] = particles->bBoxMin[0] + (xIndex + 0.5f) * xStep;
    centerOfCell[1] = particles->bBoxMin[1] + (yIndex + 0.5f) * yStep;
    centerOfCell[2] = particles->bBoxMin[2] + (zIndex + 0.5f) * zStep;
}

// void ApplyPressure(struct PointSOA *particles) {
//     // Process each cell
//     for (int gridId = 0; gridId < gridResolution; gridId++) {
//         int startIdx = particles->startIndex[gridId];
//         if (startIdx == -1) continue; // No particles in this grid cell
        
//         int endIdx = startIdx + particles->numberOfParticle[gridId];
//         float currentPressure = pressure * particles->numberOfParticle[gridId];
        
//         // Get 3D grid coordinates
//         int x = gridId % gridResolutionAxis;
//         int y = (gridId / gridResolutionAxis) % gridResolutionAxis;
//         int z = gridId / (gridResolutionAxis * gridResolutionAxis);
        
//         // Process each particle in this cell
//         for (int i = startIdx; i < endIdx; i++) {
//             float netForceX = 0.0f;
//             float netForceY = 0.0f;
//             float netForceZ = 0.0f;
            
//             // Check all 6 face-adjacent neighbors
//             const int neighbors[6][3] = {
//                 {-1, 0, 0}, {1, 0, 0},  // left, right
//                 {0, -1, 0}, {0, 1, 0},  // down, up
//                 {0, 0, -1}, {0, 0, 1}   // back, front
//             };
            
//             // Consider all neighbors for pressure gradient
//             for (int j = 0; j < 6; j++) {
//                 int nx = x + neighbors[j][0];
//                 int ny = y + neighbors[j][1];
//                 int nz = z + neighbors[j][2];
                
//                 // Skip out-of-bounds neighbors
//                 if (nx < 0 || nx >= gridResolutionAxis ||
//                     ny < 0 || ny >= gridResolutionAxis ||
//                     nz < 0 || nz >= gridResolutionAxis) {
//                     continue;
//                 }
                
//                 int neighborGridId = nx + ny * gridResolutionAxis + nz * gridResolutionAxis * gridResolutionAxis;
//                 // Calculate neighbor pressure
//                 float neighborPressure = pressure * particles->numberOfParticle[neighborGridId];
                
//                 // Calculate pressure difference
//                 float pressureDiff = currentPressure - neighborPressure;
                
//                 // Only push if there's pressure gradient
//                 if (pressureDiff > 0.0f) {
//                     // Add force component in direction of this neighbor
//                     netForceX += neighbors[j][0] * pressureDiff;
//                     netForceY += neighbors[j][1] * pressureDiff;
//                     netForceZ += neighbors[j][2] * pressureDiff;
//                 }
//             }
            
//             // Apply the net force if it's significant
//             float forceMagnitude = sqrtf(netForceX*netForceX + netForceY*netForceY + netForceZ*netForceZ);
//             if (forceMagnitude > 0.1f) {
//                 // Normalize force direction
//                 float invMag = 1.0f / forceMagnitude;
//                 netForceX *= invMag;
//                 netForceY *= invMag;
//                 netForceZ *= invMag;
                
//                 // Scale by current cell pressure
//                 float forceScale = currentPressure * 0.2f;
                
//                 // Apply to particle velocity
//                 particles->xVelocity[i] += netForceX * forceScale;
//                 particles->yVelocity[i] += netForceY * forceScale;
//                 particles->zVelocity[i] += netForceZ * forceScale;
//             }
//         }
//     }
// }

void ApplyPressure(struct PointSOA *particles) {
    // Constants as AVX vectors for vectorized operations
    const __m256 pressureVec = _mm256_set1_ps(pressure);
    const __m256 forceScaleFactorVec = _mm256_set1_ps(0.2f);
    const __m256 thresholdVec = _mm256_set1_ps(0.1f);
    const __m256 zeroVec = _mm256_setzero_ps();
    
    // Define neighbor directions once
    const int neighbors[6][3] = {
        {-1, 0, 0}, {1, 0, 0},  // left, right
        {0, -1, 0}, {0, 1, 0},  // down, up
        {0, 0, -1}, {0, 0, 1}   // back, front
    };
    
    // Process each cell
    for (int gridId = 0; gridId < gridResolution; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        
        int endIdx = startIdx + particles->numberOfParticle[gridId];
        float currentPressure = pressure * particles->numberOfParticle[gridId];
        __m256 currentPressureVec = _mm256_set1_ps(currentPressure);
        
        // Get 3D grid coordinates
        int x = gridId % gridResolutionAxis;
        int y = (gridId / gridResolutionAxis) % gridResolutionAxis;
        int z = gridId / (gridResolutionAxis * gridResolutionAxis);
        
        // Pre-compute and cache neighbor information
        float neighborPressures[6];
        float directionX[6], directionY[6], directionZ[6];
        int validNeighbors = 0;
        
        for (int j = 0; j < 6; j++) {
            int nx = x + neighbors[j][0];
            int ny = y + neighbors[j][1];
            int nz = z + neighbors[j][2];
            
            // Skip out-of-bounds neighbors
            if (nx < 0 || nx >= gridResolutionAxis ||
                ny < 0 || ny >= gridResolutionAxis ||
                nz < 0 || nz >= gridResolutionAxis) {
                continue;
            }
            
            int neighborGridId = nx + ny * gridResolutionAxis + nz * gridResolutionAxis * gridResolutionAxis;
            float neighborPressure = pressure * particles->numberOfParticle[neighborGridId];
            
            // Store neighbor info for reuse with all particles
            neighborPressures[validNeighbors] = neighborPressure;
            directionX[validNeighbors] = (float)neighbors[j][0];
            directionY[validNeighbors] = (float)neighbors[j][1];
            directionZ[validNeighbors] = (float)neighbors[j][2];
            validNeighbors++;
        }
        
        // Process particles in this cell in batches of 8
        int i;
        for (i = startIdx; i <= endIdx - 8; i += 8) {
            // Process 8 particles at once with AVX
            __m256 netForceXVec = _mm256_setzero_ps();
            __m256 netForceYVec = _mm256_setzero_ps();
            __m256 netForceZVec = _mm256_setzero_ps();
            
            // Consider all valid neighbors
            for (int j = 0; j < validNeighbors; j++) {
                // Calculate pressure difference
                __m256 neighborPressureVec = _mm256_set1_ps(neighborPressures[j]);
                __m256 pressureDiffVec = _mm256_sub_ps(currentPressureVec, neighborPressureVec);
                
                // Create mask for positive pressure differences
                __m256 positiveMask = _mm256_cmp_ps(pressureDiffVec, zeroVec, _CMP_GT_OQ);
                
                // Skip if no positive pressure differences
                if (_mm256_testz_ps(positiveMask, positiveMask)) {
                    continue;
                }
                
                // Apply direction * pressure difference with mask
                __m256 scaledDiffVec = _mm256_and_ps(pressureDiffVec, positiveMask);
                __m256 dirXVec = _mm256_set1_ps(directionX[j]);
                __m256 dirYVec = _mm256_set1_ps(directionY[j]);
                __m256 dirZVec = _mm256_set1_ps(directionZ[j]);
                
                netForceXVec = _mm256_add_ps(netForceXVec, _mm256_mul_ps(dirXVec, scaledDiffVec));
                netForceYVec = _mm256_add_ps(netForceYVec, _mm256_mul_ps(dirYVec, scaledDiffVec));
                netForceZVec = _mm256_add_ps(netForceZVec, _mm256_mul_ps(dirZVec, scaledDiffVec));
            }
            
            // Calculate force magnitude squared
            __m256 forceMagSquaredVec = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(netForceXVec, netForceXVec),
                    _mm256_mul_ps(netForceYVec, netForceYVec)
                ),
                _mm256_mul_ps(netForceZVec, netForceZVec)
            );
            
            // Calculate sqrt for magnitude
            __m256 forceMagVec = _mm256_sqrt_ps(forceMagSquaredVec);
            
            // Create mask for significant forces
            __m256 forceMask = _mm256_cmp_ps(forceMagVec, thresholdVec, _CMP_GT_OQ);
            
            // Skip if no particles have significant force
            if (_mm256_testz_ps(forceMask, forceMask)) {
                continue;
            }
            
            // Calculate safe inverse magnitude (avoid division by zero)
            __m256 safeForceVec = _mm256_max_ps(forceMagVec, _mm256_set1_ps(0.0001f));
            __m256 invMagVec = _mm256_div_ps(_mm256_set1_ps(1.0f), safeForceVec);
            
            // Normalize force vectors
            __m256 normForceXVec = _mm256_mul_ps(netForceXVec, invMagVec);
            __m256 normForceYVec = _mm256_mul_ps(netForceYVec, invMagVec);
            __m256 normForceZVec = _mm256_mul_ps(netForceZVec, invMagVec);
            
            // Scale by current pressure
            __m256 forceScaleVec = _mm256_mul_ps(currentPressureVec, forceScaleFactorVec);
            __m256 finalForceXVec = _mm256_mul_ps(normForceXVec, forceScaleVec);
            __m256 finalForceYVec = _mm256_mul_ps(normForceYVec, forceScaleVec);
            __m256 finalForceZVec = _mm256_mul_ps(normForceZVec, forceScaleVec);
            
            // Apply force mask to only affect particles with significant force
            finalForceXVec = _mm256_and_ps(finalForceXVec, forceMask);
            finalForceYVec = _mm256_and_ps(finalForceYVec, forceMask);
            finalForceZVec = _mm256_and_ps(finalForceZVec, forceMask);
            
            // Load current velocities
            __m256 xVelVec = _mm256_loadu_ps(&particles->xVelocity[i]);
            __m256 yVelVec = _mm256_loadu_ps(&particles->yVelocity[i]);
            __m256 zVelVec = _mm256_loadu_ps(&particles->zVelocity[i]);
            
            // Update velocities and store
            _mm256_storeu_ps(&particles->xVelocity[i], _mm256_add_ps(xVelVec, finalForceXVec));
            _mm256_storeu_ps(&particles->yVelocity[i], _mm256_add_ps(yVelVec, finalForceYVec));
            _mm256_storeu_ps(&particles->zVelocity[i], _mm256_add_ps(zVelVec, finalForceZVec));
        }
        
        // Handle remaining particles with scalar code
        for (; i < endIdx; i++) {
            float netForceX = 0.0f;
            float netForceY = 0.0f;
            float netForceZ = 0.0f;
            
            // Consider all neighbors for pressure gradient
            for (int j = 0; j < validNeighbors; j++) {
                float pressureDiff = currentPressure - neighborPressures[j];
                
                // Only push if there's pressure gradient
                if (pressureDiff > 0.0f) {
                    netForceX += directionX[j] * pressureDiff;
                    netForceY += directionY[j] * pressureDiff;
                    netForceZ += directionZ[j] * pressureDiff;
                }
            }
            
            // Apply the net force if it's significant
            float forceMagnitude = sqrtf(netForceX*netForceX + netForceY*netForceY + netForceZ*netForceZ);
            if (forceMagnitude > 0.1f) {
                // Normalize force direction
                float invMag = 1.0f / forceMagnitude;
                netForceX *= invMag;
                netForceY *= invMag;
                netForceZ *= invMag;
                
                // Scale by current cell pressure
                float forceScale = currentPressure * 0.2f;
                
                // Apply to particle velocity
                particles->xVelocity[i] += netForceX * forceScale;
                particles->yVelocity[i] += netForceY * forceScale;
                particles->zVelocity[i] += netForceZ * forceScale;
            }
        }
    }
}

// First, optimize the move_to_grid function for direct calculation
// void move_to_grid(struct PointSOA *particles, int index) {
//     float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
//     float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
//     float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
//     // Calculate grid positions directly with clamping
//     int xIndex = (int)((particles->x[index] - particles->bBoxMin[0]) / xStep);
//     int yIndex = (int)((particles->y[index] - particles->bBoxMin[1]) / yStep);
//     int zIndex = (int)((particles->z[index] - particles->bBoxMin[2]) / zStep);
    
//     // Clamp indices
//     xIndex = (xIndex < 0) ? 0 : ((xIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : xIndex);
//     yIndex = (yIndex < 0) ? 0 : ((yIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : yIndex);
//     zIndex = (zIndex < 0) ? 0 : ((zIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : zIndex);
    
//     // Convert 3D coordinates to a single index
//     particles->gridID[index] = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
// }

// Use radix sort for better performance with integer keys
// void radixSortParticles(struct PointSOA *particles, int n) {
//     // Find the maximum number to know the number of digits
//     int maxVal = 0;
//     for (int i = 0; i < n; i++) {
//         if (particles->gridID[i] > maxVal) {
//             maxVal = particles->gridID[i];
//         }
//     }
    
//     // Temporary arrays for sorting
//     float* tempX = (float*)malloc(n * sizeof(float));
//     float* tempY = (float*)malloc(n * sizeof(float));
//     float* tempZ = (float*)malloc(n * sizeof(float));
//     float* tempVX = (float*)malloc(n * sizeof(float));
//     float* tempVY = (float*)malloc(n * sizeof(float));
//     float* tempVZ = (float*)malloc(n * sizeof(float));
//     int* tempGridID = (int*)malloc(n * sizeof(int));
    
//     // Do counting sort for every digit
//     for (int exp = 1; maxVal / exp > 0; exp *= 10) {
//         int count[10] = {0};
        
//         // Count occurrences of each digit at current place value
//         for (int i = 0; i < n; i++) {
//             count[(particles->gridID[i] / exp) % 10]++;
//         }
        
//         // Compute cumulative count (positions)
//         for (int i = 1; i < 10; i++) {
//             count[i] += count[i - 1];
//         }
        
//         // Build the output array in reverse order to maintain stability
//         for (int i = n - 1; i >= 0; i--) {
//             int digit = (particles->gridID[i] / exp) % 10;
//             int pos = --count[digit];
            
//             // Copy the data to temporary arrays
//             tempX[pos] = particles->x[i];
//             tempY[pos] = particles->y[i];
//             tempZ[pos] = particles->z[i];
//             tempVX[pos] = particles->xVelocity[i];
//             tempVY[pos] = particles->yVelocity[i];
//             tempVZ[pos] = particles->zVelocity[i];
//             tempGridID[pos] = particles->gridID[i];
//         }
        
//         // Copy back to original arrays
//         for (int i = 0; i < n; i++) {
//             particles->x[i] = tempX[i];
//             particles->y[i] = tempY[i];
//             particles->z[i] = tempZ[i];
//             particles->xVelocity[i] = tempVX[i];
//             particles->yVelocity[i] = tempVY[i];
//             particles->zVelocity[i] = tempVZ[i];
//             particles->gridID[i] = tempGridID[i];
//         }
//     }
    
//     // Free temporary arrays
//     free(tempX);
//     free(tempY);
//     free(tempZ);
//     free(tempVX);
//     free(tempVY);
//     free(tempVZ);
//     free(tempGridID);
// }

void radixSortParticles(struct PointSOA *particles, int n) {
    // Early exit for small arrays
    if (n <= 1) return;
    
    // Find the maximum number to know the number of digits - vectorized
    int maxVal = 0;
    int i;
    
    // Process 8 values at once with AVX
    __m256i maxVec = _mm256_setzero_si256();
    for (i = 0; i <= n - 8; i += 8) {
        __m256i gridVec = _mm256_loadu_si256((__m256i*)&particles->gridID[i]);
        maxVec = _mm256_max_epi32(maxVec, gridVec);
    }
    
    // Extract max from vector
    int maxArray[8] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i*)maxArray, maxVec);
    for (int j = 0; j < 8; j++) {
        if (maxArray[j] > maxVal) maxVal = maxArray[j];
    }
    
    // Handle remaining elements
    for (; i < n; i++) {
        if (particles->gridID[i] > maxVal) {
            maxVal = particles->gridID[i];
        }
    }
    
    // Early exit if all values are the same
    if (maxVal == 0) return;
    
    // Calculate number of bits needed instead of decimal digits
    int numBits = 0;
    int temp = maxVal;
    while (temp > 0) {
        numBits++;
        temp >>= 1;
    }
    
    // Use stack allocation for small arrays, heap for large
    const int STACK_THRESHOLD = 8192; // 8KB threshold
    bool useStack = (n * sizeof(float) * 7 < STACK_THRESHOLD);
    
    float *tempX, *tempY, *tempZ, *tempVX, *tempVY, *tempVZ;
    int *tempGridID;
    
    if (useStack && n <= STACK_THRESHOLD / (sizeof(float) * 7)) {
        // Use stack allocation for better cache performance
        float stackX[STACK_THRESHOLD / (sizeof(float) * 7)];
        float stackY[STACK_THRESHOLD / (sizeof(float) * 7)];
        float stackZ[STACK_THRESHOLD / (sizeof(float) * 7)];
        float stackVX[STACK_THRESHOLD / (sizeof(float) * 7)];
        float stackVY[STACK_THRESHOLD / (sizeof(float) * 7)];
        float stackVZ[STACK_THRESHOLD / (sizeof(float) * 7)];
        int stackGridID[STACK_THRESHOLD / (sizeof(float) * 7)];
        
        tempX = stackX;
        tempY = stackY;
        tempZ = stackZ;
        tempVX = stackVX;
        tempVY = stackVY;
        tempVZ = stackVZ;
        tempGridID = stackGridID;
    } else {
        // Allocate all temporary arrays at once for better memory locality
        size_t totalSize = n * (6 * sizeof(float) + sizeof(int));
        char *tempBuffer = (char*)malloc(totalSize);
        if (!tempBuffer) {
            printf("Failed to allocate memory for radix sort\n");
            return;
        }
        
        tempX = (float*)tempBuffer;
        tempY = (float*)(tempBuffer + n * sizeof(float));
        tempZ = (float*)(tempBuffer + 2 * n * sizeof(float));
        tempVX = (float*)(tempBuffer + 3 * n * sizeof(float));
        tempVY = (float*)(tempBuffer + 4 * n * sizeof(float));
        tempVZ = (float*)(tempBuffer + 5 * n * sizeof(float));
        tempGridID = (int*)(tempBuffer + 6 * n * sizeof(float));
    }
    
    // Use 8-bit radix for better performance - define as enum to avoid VLA warning
    enum { RADIX_BITS = 8, RADIX_SIZE = 256 };
    
    // Calculate number of passes needed
    int numPasses = (numBits + RADIX_BITS - 1) / RADIX_BITS;
    
    for (int pass = 0; pass < numPasses; pass++) {
        int shift = pass * RADIX_BITS;
        
        // Skip this pass if all values have 0 in this position
        bool hasNonZero = false;
        for (int i = 0; i < n && !hasNonZero; i++) {
            if ((particles->gridID[i] >> shift) & (RADIX_SIZE - 1)) {
                hasNonZero = true;
            }
        }
        if (!hasNonZero) continue;
        
        // Fixed-size count array - no VLA warning
        int count[RADIX_SIZE];
        memset(count, 0, sizeof(count));
        
        // Count occurrences - vectorized counting
        for (int i = 0; i < n; i++) {
            int digit = (particles->gridID[i] >> shift) & (RADIX_SIZE - 1);
            count[digit]++;
        }
        
        // Compute cumulative count (positions)
        for (int i = 1; i < RADIX_SIZE; i++) {
            count[i] += count[i - 1];
        }
        
        // Build the output array in reverse order to maintain stability
        for (int i = n - 1; i >= 0; i--) {
            int digit = (particles->gridID[i] >> shift) & (RADIX_SIZE - 1);
            int pos = --count[digit];
            
            // Copy data - this could be optimized with SIMD for larger chunks
            tempX[pos] = particles->x[i];
            tempY[pos] = particles->y[i];
            tempZ[pos] = particles->z[i];
            tempVX[pos] = particles->xVelocity[i];
            tempVY[pos] = particles->yVelocity[i];
            tempVZ[pos] = particles->zVelocity[i];
            tempGridID[pos] = particles->gridID[i];
        }
        
        // Copy back to original arrays using memcpy for better performance
        memcpy(particles->x, tempX, n * sizeof(float));
        memcpy(particles->y, tempY, n * sizeof(float));
        memcpy(particles->z, tempZ, n * sizeof(float));
        memcpy(particles->xVelocity, tempVX, n * sizeof(float));
        memcpy(particles->yVelocity, tempVY, n * sizeof(float));
        memcpy(particles->zVelocity, tempVZ, n * sizeof(float));
        memcpy(particles->gridID, tempGridID, n * sizeof(int));
    }
    
    // Free memory only if we used heap allocation
    if (!useStack) {
        free(tempX); // This frees the entire allocated block
    }
}

// Earlier in the file, add this AVX-optimized version:
void updateGridData(struct PointSOA *particles) {
    // Calculate grid steps once
    float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
    float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
    float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
    // Prepare constant AVX registers
    __m256 xStepVec = _mm256_set1_ps(1.0f / xStep);
    __m256 yStepVec = _mm256_set1_ps(1.0f / yStep);
    __m256 zStepVec = _mm256_set1_ps(1.0f / zStep);
    __m256 xMinVec = _mm256_set1_ps(particles->bBoxMin[0]);
    __m256 yMinVec = _mm256_set1_ps(particles->bBoxMin[1]);
    __m256 zMinVec = _mm256_set1_ps(particles->bBoxMin[2]);
    __m256 zeroVec = _mm256_setzero_ps();
    __m256 gridMaxVec = _mm256_set1_ps((float)(gridResolutionAxis - 1));
    __m256i gridResVec = _mm256_set1_epi32(gridResolutionAxis);
    __m256i gridResSquaredVec = _mm256_set1_epi32(gridResolutionAxis * gridResolutionAxis);
    
    // Process particles in chunks of 8
    int alignedCount = NUM_PARTICLES & ~7; // Round down to multiple of 8
    
    for (int i = 0; i < alignedCount; i += 8) {
        // Load particle positions
        __m256 xVec = _mm256_loadu_ps(&particles->x[i]);
        __m256 yVec = _mm256_loadu_ps(&particles->y[i]);
        __m256 zVec = _mm256_loadu_ps(&particles->z[i]);
        
        // Calculate grid indices
        __m256 xIndexFloat = _mm256_mul_ps(_mm256_sub_ps(xVec, xMinVec), xStepVec);
        __m256 yIndexFloat = _mm256_mul_ps(_mm256_sub_ps(yVec, yMinVec), yStepVec);
        __m256 zIndexFloat = _mm256_mul_ps(_mm256_sub_ps(zVec, zMinVec), zStepVec);
        
        // Convert float indices to integers
        __m256i xIndexVec = _mm256_cvttps_epi32(xIndexFloat);
        __m256i yIndexVec = _mm256_cvttps_epi32(yIndexFloat);
        __m256i zIndexVec = _mm256_cvttps_epi32(zIndexFloat);
        
        // Apply clamp to x indices (min = 0, max = gridResolutionAxis - 1)
        __m256i xClamped = _mm256_max_epi32(_mm256_min_epi32(xIndexVec, _mm256_cvttps_epi32(gridMaxVec)), 
                                           _mm256_setzero_si256());
        
        // Apply clamp to y indices
        __m256i yClamped = _mm256_max_epi32(_mm256_min_epi32(yIndexVec, _mm256_cvttps_epi32(gridMaxVec)), 
                                           _mm256_setzero_si256());
        
        // Apply clamp to z indices
        __m256i zClamped = _mm256_max_epi32(_mm256_min_epi32(zIndexVec, _mm256_cvttps_epi32(gridMaxVec)), 
                                           _mm256_setzero_si256());
        
        // Calculate grid IDs: xIndex + yIndex * gridRes + zIndex * gridRes * gridRes
        __m256i yContrib = _mm256_mullo_epi32(yClamped, gridResVec);
        __m256i zContrib = _mm256_mullo_epi32(zClamped, gridResSquaredVec);
        __m256i gridIDs = _mm256_add_epi32(_mm256_add_epi32(xClamped, yContrib), zContrib);
        
        // Store grid IDs
        int results[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)results, gridIDs);
        
        // Copy results to particles array
        for (int j = 0; j < 8; j++) {
            particles->gridID[i + j] = results[j];
        }
    }
    
    // Handle remaining particles
    for (int i = alignedCount; i < NUM_PARTICLES; i++) {
        // Calculate grid positions directly with clamping
        int xIndex = (int)((particles->x[i] - particles->bBoxMin[0]) / xStep);
        int yIndex = (int)((particles->y[i] - particles->bBoxMin[1]) / yStep);
        int zIndex = (int)((particles->z[i] - particles->bBoxMin[2]) / zStep);
        
        // Clamp indices
        xIndex = (xIndex < 0) ? 0 : ((xIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : xIndex);
        yIndex = (yIndex < 0) ? 0 : ((yIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : yIndex);
        zIndex = (zIndex < 0) ? 0 : ((zIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : zIndex);
        
        // Convert 3D coordinates to a single index
        particles->gridID[i] = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
    }
    
    // Reset grid cell start indices
    memset(particles->startIndex, -1, sizeof(int) * gridResolution);
    memset(particles->numberOfParticle, 0, sizeof(int) * gridResolution);
    
    // Sort particles by grid ID using radix sort (much faster than insertion sort)
    radixSortParticles(particles, NUM_PARTICLES);
    // printf("Radix sort time: %f seconds\n", (double)(clock() - start) / CLOCKS_PER_SEC);
    
    // Update start indices and count particles in a single pass
    int currentGridID = -1;
    
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int gridID = particles->gridID[i];
        
        // When we encounter a new grid ID, mark its start index
        if (gridID != currentGridID) {
            particles->startIndex[gridID] = i;
            currentGridID = gridID;
        }
        
        // Increment count for this grid cell
        particles->numberOfParticle[gridID]++;
    }
}

void update_particle_apply_gravity(struct PointSOA *particles, float dt) {
    // Prepare constants as AVX vectors
    __m256 dtVec = _mm256_set1_ps(dt);
    __m256 dampingVec = _mm256_set1_ps(DAMPING);
    __m256 gravityVec = _mm256_set1_ps(GRAVITY);
    
    // Process particles in chunks of 8
    int alignedCount = NUM_PARTICLES & ~7; // Round down to multiple of 8
    
    for (int i = 0; i < alignedCount; i += 8) {
        // Load current positions
        __m256 xVec = _mm256_loadu_ps(&particles->x[i]);
        __m256 yVec = _mm256_loadu_ps(&particles->y[i]);
        __m256 zVec = _mm256_loadu_ps(&particles->z[i]);
        
        // Load velocities
        __m256 xVelocityVec = _mm256_loadu_ps(&particles->xVelocity[i]);
        __m256 yVelocityVec = _mm256_loadu_ps(&particles->yVelocity[i]);
        __m256 zVelocityVec = _mm256_loadu_ps(&particles->zVelocity[i]);
        
        // Apply gravity to velocity
        yVelocityVec = _mm256_sub_ps(yVelocityVec, gravityVec);
        
        // Calculate position updates: velocity * dt * damping
        __m256 dtDampingVec = _mm256_mul_ps(dtVec, dampingVec);
        __m256 xOffsetVec = _mm256_mul_ps(_mm256_mul_ps(xVelocityVec, dtVec), dampingVec);
        __m256 yOffsetVec = _mm256_mul_ps(_mm256_mul_ps(yVelocityVec, dtVec), dampingVec);
        __m256 zOffsetVec = _mm256_mul_ps(_mm256_mul_ps(zVelocityVec, dtVec), dampingVec);
        
        // Update positions
        xVec = _mm256_add_ps(xVec, xOffsetVec);
        yVec = _mm256_add_ps(yVec, yOffsetVec);
        zVec = _mm256_add_ps(zVec, zOffsetVec);
        
        // Store updated positions
        _mm256_storeu_ps(&particles->x[i], xVec);
        _mm256_storeu_ps(&particles->y[i], yVec);
        _mm256_storeu_ps(&particles->z[i], zVec);
        
        // Store updated velocities
        _mm256_storeu_ps(&particles->xVelocity[i], xVelocityVec);
        _mm256_storeu_ps(&particles->yVelocity[i], yVelocityVec);
        _mm256_storeu_ps(&particles->zVelocity[i], zVelocityVec);
    }
    
    // Handle remaining particles
    for (int i = alignedCount; i < NUM_PARTICLES; i++) {
        // Apply gravity
        particles->yVelocity[i] -= GRAVITY;
        
        // Update positions
        particles->x[i] += particles->xVelocity[i] * dt * DAMPING;
        particles->y[i] += particles->yVelocity[i] * dt * DAMPING;
        particles->z[i] += particles->zVelocity[i] * dt * DAMPING;
    }
}

void move_to_box(struct PointSOA *particles, float bBoxMin[3], float bBoxMax[3]) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (particles->x[i] < bBoxMin[0]) {
            particles->x[i] = bBoxMin[0];
            // change velocity to opposite direction
            particles->xVelocity[i] = -particles->xVelocity[i] * DAMPING;
        } else if (particles->x[i] > bBoxMax[0]) {
            particles->x[i] = bBoxMax[0];
            // change velocity to opposite direction
            particles->xVelocity[i] = -particles->xVelocity[i] * DAMPING;
        }
        if (particles->y[i] < bBoxMin[1]) {
            particles->y[i] = bBoxMin[1];
            // change velocity to opposite direction
            particles->yVelocity[i] = -particles->yVelocity[i] * DAMPING;
        } else if (particles->y[i] > bBoxMax[1]) {
            particles->y[i] = bBoxMax[1];
            // change velocity to opposite direction
            particles->yVelocity[i] = -particles->yVelocity[i] * DAMPING;
        }
        if (particles->z[i] < bBoxMin[2]) {
            particles->z[i] = bBoxMin[2];
            // change velocity to opposite direction
            particles->zVelocity[i] = -particles->zVelocity[i] * DAMPING;
        } else if (particles->z[i] > bBoxMax[2]) {
            particles->z[i] = bBoxMax[2];
            // change velocity to opposite direction
            particles->zVelocity[i] = -particles->zVelocity[i] * DAMPING;
        }
    }
}

int readCameraData(struct Camera *camera) {
    // printf("Attempting to read camera data\n");
    FILE *file = fopen("camera.bin", "rb");
    if (!file) {
        // printf("Camera file not found, using default camera\n");
        return 0;
    }
    
    // Read camera position
    if (fread(&camera->ray.origin[0], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.origin[1], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.origin[2], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.direction[0], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.direction[1], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.direction[2], sizeof(float), 1, file) != 1 ||
        fread(&camera->fov, sizeof(float), 1, file) != 1) {
        
        fclose(file);
        // printf("Error reading camera data, using default camera\n");
        return 0;  // Error reading, use default camera
    }
    
    fclose(file);
    // printf("Successfully read camera data\n");
    return 1;  // Successfully read camera data
}

// Add this quick sort helper function
void quickSortParticles(struct PointSOA *particles, int low, int high, float *distances) {
    if (low < high) {
        float pivot = distances[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (distances[j] <= pivot) {
                i++;
                swapParticles(particles, i, j);
                float temp = distances[i];
                distances[i] = distances[j];
                distances[j] = temp;
            }
        }
        swapParticles(particles, i + 1, high);
        float temp = distances[i + 1];
        distances[i + 1] = distances[high];
        distances[high] = temp;
        
        int partition = i + 1;
        quickSortParticles(particles, low, partition - 1, distances);
        quickSortParticles(particles, partition + 1, high, distances);
    }
}


void calculateParticleScreenCoordinates(struct ThreadData *data) {
    struct Camera *camera = data->camera;
    float *xPtr = data->xPtrStart;
    float *yPtr = data->yPtrStart;
    float *zPtr = data->zPtrStart;
    float *xVelocityPtr = data->xVelocityPtrStart;
    float *yVelocityPtr = data->yVelocityPtrStart;
    float *zVelocityPtr = data->zVelocityPtrStart;
    float *totalVelocityPtr = data->totalVelocityPtrStart;
    int *screenXPtr = (int *)data->screenXPtrStart;
    int *screenYPtr = (int *)data->screenYPtrStart;
    int length = data->length;
    float maxVelocity = data->maxVelocity;

    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;

    // Calculate view matrix vectors
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
    // Cache camera origin for faster access
    const float camX = camera->ray.origin[0];
    const float camY = camera->ray.origin[1];
    const float camZ = camera->ray.origin[2];
    const float camDirX = camera->ray.direction[0];
    const float camDirY = camera->ray.direction[1];
    const float camDirZ = camera->ray.direction[2];

    // Calculate the screen coordinates
    for (int i = 0; i < length; i++) {
        float x = xPtr[i] - camX;
        float y = yPtr[i] - camY;
        float z = zPtr[i] - camZ;

        float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
        float fovScale = 1.0f / (dotProduct * camera->fov);

        float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
        float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;

        int screenX = (int)(screenRight * halfWidth + halfWidth);
        int screenY = (int)(-screenUp * halfHeight + halfHeight);
        
        if (screenX < 0 || screenX >= ScreenWidth || screenY < 0 || screenY >= ScreenHeight) continue;
        
        float totalVelocity = xVelocityPtr[i] * xVelocityPtr[i] + yVelocityPtr[i] * yVelocityPtr[i] + zVelocityPtr[i] * zVelocityPtr[i];

        screenXPtr[i] = screenX;
        screenYPtr[i] = screenY;
        totalVelocityPtr[i] = totalVelocity;

        if (totalVelocity > data->maxVelocity) {
            data->maxVelocity = totalVelocity;    // Update the thread's maxVelocity
        }
    }
}

void heapifyParticles(struct PointSOA *particles, float *distances, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    
    if (left < n && distances[left] > distances[largest])
        largest = left;
    
    if (right < n && distances[right] > distances[largest])
        largest = right;
    
    if (largest != i) {
        // Swap distances
        float temp = distances[i];
        distances[i] = distances[largest];
        distances[largest] = temp;
        
        // Swap particles
        swapParticles(particles, i, largest);
        
        heapifyParticles(particles, distances, n, largest);
    }
}

void heapSortParticles(struct PointSOA *particles, float *distances, int n) {
    // Build heap
    for (int i = n / 2 - 1; i >= 0; i--)
        heapifyParticles(particles, distances, n, i);
    
    // Extract elements from heap
    for (int i = n - 1; i > 0; i--) {
        // Swap distances
        float temp = distances[0];
        distances[0] = distances[i];
        distances[i] = temp;
        
        // Swap particles
        swapParticles(particles, 0, i);
        
        heapifyParticles(particles, distances, i, 0);
    }
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
    union { float f; uint32_t i; } u = { x };
    u.i = 0x5f3759df - (u.i >> 1);
    float y = u.f;
    return y * (1.5f - 0.5f * x * y * y);
};

void renderParticlesToBuffer(struct RenderThreadData *data) {
    struct ThreadScreen *screen = data->threadScreen;
    struct PointSOA *particles = data->particles;
    struct ParticleIndexes *particleIndexes = data->particleIndexes;
    struct Camera *camera = data->camera;
    
    // Clear thread's screen buffer
    memset(screen->distance, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->velocity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->particleCount, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
    
    // Pre-calculate camera vectors and constants (same as before)
    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;
    
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

    const float camX = camera->ray.origin[0];
    const float camY = camera->ray.origin[1];
    const float camZ = camera->ray.origin[2];
    
    // Render assigned particle range back-to-front
    for (int i = data->endParticle - 1; i >= data->startParticle; i--) {
        int index = particleIndexes->particleIndexes[i].index;
        
        int screenX = particles->screenX[index];
        int screenY = particles->screenY[index];

        // Fast distance approximation using squared distance
        float distSquared = particleIndexes->particleIndexes[i].distance;
        float invDist = fastInvSqrt(distSquared);

        int particleRadius = (int)(PARTICLE_RADIUS * 100.0f * invDist);
        particleRadius = particleRadius < 1 ? 1 : (particleRadius > PARTICLE_RADIUS * 3 ? PARTICLE_RADIUS * 3 : particleRadius);

        // Calculate drawing bounds
        int minX = screenX - particleRadius;
        int maxX = screenX + particleRadius;
        int minY = screenY - particleRadius;
        int maxY = screenY + particleRadius;
        
        // Quick bounds check
        if (minX < 0 || maxX >= ScreenWidth || minY < 0 || maxY >= ScreenHeight) continue;
        
        // Calculate particle color once
        uint8_t normalizedVelocity = (uint8_t)(255 * (1.0f - (particles->totalVelocity[index] / data->maxVelocity)));
        uint8_t distanceNormalized = (uint8_t)(255 * (1.0f - (distSquared / data->maxDistance)));

        // Draw filled circle and count particles per pixel
        int radiusSquared = particleRadius * particleRadius;
        for (int dy = -particleRadius; dy <= particleRadius; dy++) {
            int py = screenY + dy;
            if (py < 0 || py >= ScreenHeight) continue;
            
            int dy2 = dy * dy;
            int maxDx = (int)sqrtf((float)(radiusSquared - dy2));
            
            int startX = screenX - maxDx;
            int endX = screenX + maxDx;
            
            if (startX < 0) startX = 0;
            if (endX >= ScreenWidth) endX = ScreenWidth - 1;
            
            for (int px = startX; px <= endX; px++) {
                // For depth compositing, only update if this particle is closer or first
                if (screen->particleCount[px][py] == 0 || distanceNormalized > screen->distance[px][py]) {
                    screen->distance[px][py] = distanceNormalized;
                    screen->velocity[px][py] = normalizedVelocity;
                }
                // Count this particle at this pixel
                screen->particleCount[px][py]++;
            }
        }
    }
}

void compositeThreadBuffers(struct Screen *finalScreen) {
    uint16_t globalMaxParticleCount = 0;
    
    // Single pass: accumulate particle counts and find max distance/velocity per pixel
    for (int x = 0; x < ScreenWidth; x++) {
        for (int y = 0; y < ScreenHeight; y++) {
            uint32_t totalParticleCount = 0;
            uint8_t maxDistance = 0;
            uint8_t maxDistanceVelocity = 0;
            
            // Accumulate from all threads
            for (int t = 0; t < NUM_THREADS; t++) {
                uint16_t particleCount = threadScreens[t]->particleCount[x][y];
                if (particleCount > 0) {
                    totalParticleCount += particleCount;
                    
                    // Use distance/velocity from thread with farthest particle
                    if (threadScreens[t]->distance[x][y] > maxDistance) {
                        maxDistance = threadScreens[t]->distance[x][y];
                        maxDistanceVelocity = threadScreens[t]->velocity[x][y];
                    }
                }
            }
            
            if (totalParticleCount > 0) {
                // Store results
                finalScreen->distance[x][y] = maxDistance;
                finalScreen->velocity[x][y] = maxDistanceVelocity;
                
                // Clamp and store opacity
                uint16_t clampedCount = (totalParticleCount > 65535) ? 65535 : (uint16_t)totalParticleCount;
                finalScreen->opacity[x][y] = clampedCount;
                
                // Track global maximum
                if (clampedCount > globalMaxParticleCount) {
                    globalMaxParticleCount = clampedCount;
                }
            }
        }
    }
    
    // Early exit if no particles
    if (globalMaxParticleCount == 0) return;
    
    // Pre-calculate normalization factor
    float invLogMaxParticleCount = 1.0f / logf((float)globalMaxParticleCount + 1.0f);
    
    // Second pass: calculate normalized opacity only where needed
    for (int x = 0; x < ScreenWidth; x++) {
        for (int y = 0; y < ScreenHeight; y++) {
            uint16_t opacity = finalScreen->opacity[x][y];
            if (opacity > 0) {
                float logParticleCount = logf((float)opacity + 1.0f) * invLogMaxParticleCount;
                finalScreen->normalizedOpacity[x][y] = (uint8_t)(255.0f * logParticleCount);
            }
        }
    }
}

void projectParticles(struct PointSOA *particles, struct Camera *camera, struct Screen *screen, struct TimePartition *timePartition, struct ThreadsData *threadsData, struct ParticleIndexes *particleIndexes) {
    // Pre-calculate camera vectors and constants
    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;
    
    // Calculate view matrix vectors
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

    // Cache camera origin for faster access
    const float camX = camera->ray.origin[0];
    const float camY = camera->ray.origin[1];
    const float camZ = camera->ray.origin[2];
    const float camDirX = camera->ray.direction[0];
    const float camDirY = camera->ray.direction[1];
    const float camDirZ = camera->ray.direction[2];
    
    // Calculate distances once
    int validParticles = 0;
    float maxDistance = 0.0f;
    
    clock_t start = clock();
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x = particles->x[i] - camX;
        float y = particles->y[i] - camY;
        float z = particles->z[i] - camZ;
        
        float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
                         
        if (dotProduct > 0) { // Only include particles in front of camera
            float distSquared = x*x + y*y + z*z;
            // particles->distance[validParticles] = distSquared;
            particleIndexes->particleIndexes[validParticles].index = i;
            particleIndexes->particleIndexes[validParticles].distance = distSquared;
            
            // Track maximum distance directly during first pass
            if (distSquared > maxDistance) {
                maxDistance = distSquared;
            }
            validParticles++;
        }
    }
    
    // Skip sort if no particles are visible
    if (validParticles == 0) return;

    // heapSortParticles(particles, distances, validParticles);

    // Use qsort for simplicity
    qsort(particleIndexes->particleIndexes, validParticles, sizeof(struct ParticleIndex), compareParticlesByDistance);

    clock_t endSortTime = clock();
    float dt = (float)(endSortTime - start) / (float)CLOCKS_PER_SEC;
    timePartition->sortTime += dt;
    float maxVelocity = 0.0f;
    
    // Project particles to screen coordinates using sorted indices
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int index = particleIndexes->particleIndexes[i].index;
        float x = particles->x[index] - camX;
        float y = particles->y[index] - camY;
        float z = particles->z[index] - camZ;

        float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
        float fovScale = 1.0f / (dotProduct * camera->fov);

        float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
        float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;

        int screenX = (int)(screenRight * halfWidth + halfWidth);
        int screenY = (int)(-screenUp * halfHeight + halfHeight);
        
        if (screenX < 0 || screenX >= ScreenWidth || screenY < 0 || screenY >= ScreenHeight) continue;

        // Calculate velocity
        float vx = particles->xVelocity[index];
        float vy = particles->yVelocity[index];
        float vz = particles->zVelocity[index];
        float totalVelocity = vx*vx + vy*vy + vz*vz;

        if (totalVelocity > maxVelocity) {
            maxVelocity = totalVelocity;
        }

        particles->screenX[index] = screenX;
        particles->screenY[index] = screenY;
        particles->totalVelocity[index] = totalVelocity;
    }
    // printf("Single thread projection time: %f\n", (float)(clock() - starProjectTimeSingle) / CLOCKS_PER_SEC);


    clock_t endProjectTime = clock();
    dt = (float)(endProjectTime - endSortTime) / (float)CLOCKS_PER_SEC;
    // printf("Projection time: %f\n", dt);
    timePartition->projectionTime += dt;

    if (DISABLE_MP_RENDER_PARTICLES == 1) {
        float maxOpacity = 0;
        // Process particles back-to-front (furthest to nearest)
        for (int i = validParticles - 1; i >= 0; i--) {    
            int index = particleIndexes->particleIndexes[i].index;
            
            int screenX = particles->screenX[index];
            int screenY = particles->screenY[index];

            // Fast distance approximation using squared distance
            float distSquared = particleIndexes->particleIndexes[i].distance;
            float invDist = fastInvSqrt(distSquared);

            int particleRadius = (int)(PARTICLE_RADIUS * 100.0f * invDist);
            
            // Apply bounds without branching
            particleRadius = particleRadius < 1 ? 1 : (particleRadius > PARTICLE_RADIUS * 3 ? PARTICLE_RADIUS * 3 : particleRadius);

            // Calculate drawing bounds
            int minX = screenX - particleRadius;
            int maxX = screenX + particleRadius;
            int minY = screenY - particleRadius;
            int maxY = screenY + particleRadius;
            
            // Quick bounds check
            if (minX < 0 || maxX >= ScreenWidth || minY < 0 || maxY >= ScreenHeight) continue;
            
            // Calculate particle color once
            uint8_t NormalizedVelocity = (uint8_t)(255 * (1.0f - (particles->totalVelocity[index] / maxVelocity)));
            uint8_t distanceNormalized = (uint8_t)(255 * (1.0f - (distSquared / maxDistance)));

            // Track maximum values
            float currentOpacity = (float)(screen->opacity[screenX][screenY]) + 1.0f;
            if (currentOpacity > maxOpacity) maxOpacity = currentOpacity;

            int radiusSquared = particleRadius * particleRadius;
            for (int dy = -particleRadius; dy <= particleRadius; dy++) {
                int py = screenY + dy;
                if (py < 0 || py >= ScreenHeight) continue;
                
                int dy2 = dy * dy;
                int maxDx = (int)sqrtf((float)(radiusSquared - dy2)); // Calculate span once per row
                
                int startX = screenX - maxDx;
                int endX = screenX + maxDx;
                
                // Clamp to screen bounds
                if (startX < 0) startX = 0;
                if (endX >= ScreenWidth) endX = ScreenWidth - 1;
                
                // Fill the entire span at once
                // NOTE: this is rendering it as 2d not as spherical particles
                // for (int px = startX; px <= endX; px++) {
                //     screen->distance[px][py] = distanceNormalized;
                //     screen->velocity[px][py] = NormalizedVelocity;
                //     screen->opacity[px][py]++;
                // }
                for (int px = startX; px <= endX; px++) {
                    int dx = px - screenX;
                    int dy = py - screenY;
                    int r2 = dx*dx + dy*dy;
                    
                    // Calculate sphere depth offset at this pixel
                    float dz = sqrtf((float)(radiusSquared - r2));
                    
                    // The surface depth is the center depth minus the z-offset (closer to camera)
                    // Assuming camera looks along +Z, closer objects have smaller depth values
                    float surfaceDepth = distSquared - (dz * dz * 1000.0f); // Scale factor for depth variation
                    
                    // Ensure surface depth doesn't go negative
                    if (surfaceDepth < 0.0f) surfaceDepth = 0.0f;
                    
                    // Convert surface depth to normalized distance
                    uint8_t surfaceDistanceNormalized = (uint8_t)(255 * (1.0f - (surfaceDepth / maxDistance)));
                    
                    screen->distance[px][py] = surfaceDistanceNormalized;
                    screen->velocity[px][py] = NormalizedVelocity;
                    screen->opacity[px][py]++;
                }
            }
        }
        clock_t renderDistanceVelocity = clock();
        dt = (float)(renderDistanceVelocity - endProjectTime) / (float)CLOCKS_PER_SEC;
        timePartition->renderDistanceVelocityTime += dt;

         // Pre-calculate normalization factors to avoid repeated calculations
        float invMaxOpacity = 1.0f / (float)maxOpacity;
        
        // Normalize opacity and velocity in a single pass
        for (int px = 0; px < ScreenWidth; px++) {
            for (int py = 0; py < ScreenHeight; py++) {
                uint16_t opacity = screen->opacity[px][py];
                if (opacity != 0) {
                    float linearOpacity = (float)opacity * invMaxOpacity;
                    screen->normalizedOpacity[px][py] = (uint8_t)(255 * linearOpacity);
                }
            }
        }
        clock_t renderOpacityTime = clock();
        dt = (float)(renderOpacityTime - renderDistanceVelocity) / (float)CLOCKS_PER_SEC;
        timePartition->renderOpacityTime += dt;
        // printf("Render opacity time: %f\n", dt);
    } else {
    // Multithreaded rendering of particles
        clock_t renderStart = clock();
        
        // Divide particles among threads
        int particlesPerThread = validParticles / NUM_THREADS;
        int remainder = validParticles % NUM_THREADS;
        
        int startParticle = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            int particlesForThisThread = particlesPerThread + (i < remainder ? 1 : 0);
            
            // Set up render thread data
            renderThreadData[i]->particles = particles;
            renderThreadData[i]->particleIndexes = particleIndexes;
            renderThreadData[i]->camera = camera;
            renderThreadData[i]->startParticle = startParticle;
            renderThreadData[i]->endParticle = startParticle + particlesForThisThread;
            renderThreadData[i]->validParticles = validParticles;
            renderThreadData[i]->maxVelocity = maxVelocity;
            renderThreadData[i]->maxDistance = maxDistance;
            
            startParticle += particlesForThisThread;
        }
        
        // Signal all threads to start rendering
        for (int i = 0; i < NUM_THREADS; i++) {
            threadSync[i].renderReady = 1;
        }

        // Wait for all threads to complete rendering
        for (int i = 0; i < NUM_THREADS; i++) {
            while (threadSync[i].renderReady) {
                #if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
                #else
                    usleep(100);
                #endif
            }
        }

        clock_t renderDistanceVelocity = clock();
        dt = (float)(renderDistanceVelocity - renderStart) / (float)CLOCKS_PER_SEC;
        timePartition->renderDistanceVelocityTime += dt;
        // printf("Render distance and velocity time: %f\n", dt);


        // Composite all thread buffers into final screen
        clock_t compositeStart = clock();
        compositeThreadBuffers(screen);
        clock_t compositeEnd = clock();
        
        dt = (float)(compositeEnd - compositeStart) / (float)CLOCKS_PER_SEC;
        timePartition->renderOpacityTime += dt;
    }
}


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
            buffer[i++] = screen->distance[x][y];  // Changed from [x][y] to [y][x]
            buffer[i++] = screen->velocity[x][y];  // Changed from [x][y] to [y][x]
            buffer[i++] = screen->normalizedOpacity[x][y];  // Changed from [x][y] to [y][x]
        }
    }

    fwrite(buffer, 1, ScreenWidth * ScreenHeight * 3, file);
    fclose(file);
}

void saveScreenNormal(struct Screen *screen, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    static uint8_t buffer[ScreenWidth * ScreenHeight * 4];

    int i = 0;
    for (int y = 0; y < ScreenHeight; y++) {
        for (int x = 0; x < ScreenWidth; x++) {
            // Convert from [-1,1] to [0,255] range
            buffer[i++] = (uint8_t)((screen->normals[x][y][0] * 0.5f + 0.5f) * 255.0f);
            buffer[i++] = (uint8_t)((screen->normals[x][y][1] * 0.5f + 0.5f) * 255.0f);
            buffer[i++] = (uint8_t)((screen->normals[x][y][2] * 0.5f + 0.5f) * 255.0f);
            buffer[i++] = (uint8_t)(255); // Normalize opacity to [0,255]
        }
    }

    fwrite(buffer, 1, ScreenWidth * ScreenHeight * 4, file);
    fclose(file);
}

void saveScreenColor(struct Screen *screen, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    static uint8_t buffer[ScreenWidth * ScreenHeight * 4];

    int i = 0;
    for (int y = 0; y < ScreenHeight; y++) {
        for (int x = 0; x < ScreenWidth; x++) {
            // No conversion needed since colors is already uint8_t
            buffer[i++] = screen->colors[x][y][0];  // R
            buffer[i++] = screen->colors[x][y][1];  // G
            buffer[i++] = screen->colors[x][y][2];  // B
            buffer[i++] = 255; // Alpha
        }
    }

    fwrite(buffer, 1, ScreenWidth * ScreenHeight * 4, file);
    fclose(file);
}

void render(struct Screen *screen, struct PointSOA *particles, struct Camera *camera, struct Cursor *cursor, struct TimePartition *timePartition, struct ThreadsData *threadsData, struct ParticleIndexes *particleIndexes, struct Camera *lightCamera, struct Screen *lightScreen, struct OpenCLContext *openCLContext, struct Triangles *triangles, struct SkyBox *skyBox) {
    // printf("\n--- Starting render ---\n");
    int start = clock();
    clearScreen(screen);
    clearScreen(lightScreen);
    int clearScreenTime = clock();

    float dt = (float)(clearScreenTime - start) / (float)CLOCKS_PER_SEC;
    timePartition->clearScreenTime += dt;
    // printf("Clear screen time: %f\n", dt);
    if (USE_GPU == 1) {
        projectParticlesOpenCL(openCLContext, particles, camera, screen, triangles, skyBox);
        // save normal screen
        saveScreenNormal(screen, "normal.bin");
        saveScreenColor(screen, "color.bin");
    } else {
        projectParticles(particles, camera, screen, timePartition, threadsData, particleIndexes);
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

    drawBoundingBox(screen, particles->bBoxMax, particles->bBoxMin, camera);
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

void addForce(struct PointSOA *particles, struct Cursor *cursor) {
    if (!cursor->active) return;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float dx = particles->x[i] - cursor->x;
        float dy = particles->y[i] - cursor->y;
        float dz = particles->z[i] - cursor->z;
        float distSquared = dx*dx + dy*dy + dz*dz;
        float dist = sqrtf(distSquared);
        
        // Prevent extreme forces at very close distances
        const float minDistance = 2.0f;  // Minimum effective distance
        if (dist < minDistance) {
            dist = minDistance;
        }
        
        // Apply force with the minimum distance protection
        float forceFactor = cursor->force / dist;
        particles->xVelocity[i] += dx * forceFactor;
        particles->yVelocity[i] += dy * forceFactor;
        particles->zVelocity[i] += dz * forceFactor;
    }
}


void update_particles(struct PointSOA *particles, float dt, struct TimePartition *timePartition, struct Cursor *cursor) {
    clock_t start = clock();
    CollideParticlesInGrid(particles);
    clock_t collideParticlesTime = clock();
    float dt_ = (float)(collideParticlesTime - start) / (float)CLOCKS_PER_SEC;
    timePartition->collisionTime += dt_;
    // printf("Collision time: %f\n", dt_);
    
    ApplyPressure(particles);
    clock_t applyPressureTime = clock();
    dt_ = (float)(applyPressureTime - collideParticlesTime) / (float)CLOCKS_PER_SEC;
    timePartition->applyPressureTime += dt_;
    // printf("Apply pressure time: %f\n", dt_);
    
    addForce(particles, cursor);

    update_particle_apply_gravity(particles,dt);
    
    clock_t updateParticlesTime = clock();
    dt_ = (float)(updateParticlesTime - applyPressureTime) / (float)CLOCKS_PER_SEC;
    timePartition->updateParticlesTime += dt_;
    // printf("Update particles time: %f\n", dt_);
    
    move_to_box(particles, particles->bBoxMin, particles->bBoxMax);
    clock_t moveToBoxTime = clock();
    dt_ = (float)(moveToBoxTime - updateParticlesTime) / (float)CLOCKS_PER_SEC;
    timePartition->moveToBoxTime += dt_;
    // printf("Move to box time: %f\n", dt_);
}

// read the pause.bin
void readPauseData(bool *paused) {
    FILE *file = fopen("pause.bin", "rb");
    if (!file) {
        return;  // If file doesn't exist, keep current pause state
    }
    
    // Read the boolean value from the file
    bool fileValue;
    if (fread(&fileValue, sizeof(bool), 1, file) == 1) {
        *paused = fileValue;  // Update the pause state with the value from the file
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


int setupStaticKernelArguments(struct OpenCLContext *ocl, struct Triangles *triangles, struct SkyBox *skyBox) {
    cl_int err;
    
    // Set all the static arguments that never change
    cl_int screen_width = ScreenWidth;
    cl_int screen_height = ScreenHeight;
    cl_int num_triangles = triangles->count;
    cl_int skybox_width = skyBox->top->width;
    cl_int skybox_height = skyBox->top->height;
    
    // Set static buffer arguments (0-5, 9-24)
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
    
    // Skybox arguments (14-21)
    err |= clSetKernelArg(ocl->triangle_kernel, 14, sizeof(cl_mem), &ocl->buffer_skybox_top);
    err |= clSetKernelArg(ocl->triangle_kernel, 15, sizeof(cl_mem), &ocl->buffer_skybox_bottom);
    err |= clSetKernelArg(ocl->triangle_kernel, 16, sizeof(cl_mem), &ocl->buffer_skybox_left);
    err |= clSetKernelArg(ocl->triangle_kernel, 17, sizeof(cl_mem), &ocl->buffer_skybox_right);
    err |= clSetKernelArg(ocl->triangle_kernel, 18, sizeof(cl_mem), &ocl->buffer_skybox_front);
    err |= clSetKernelArg(ocl->triangle_kernel, 19, sizeof(cl_mem), &ocl->buffer_skybox_back);
    err |= clSetKernelArg(ocl->triangle_kernel, 20, sizeof(cl_int), &skybox_width);
    err |= clSetKernelArg(ocl->triangle_kernel, 21, sizeof(cl_int), &skybox_height);
    
    // Triangle material properties (22-24)
    err |= clSetKernelArg(ocl->triangle_kernel, 22, sizeof(cl_mem), &ocl->buffer_triangle_roughness);
    err |= clSetKernelArg(ocl->triangle_kernel, 23, sizeof(cl_mem), &ocl->buffer_triangle_metallic);
    err |= clSetKernelArg(ocl->triangle_kernel, 24, sizeof(cl_mem), &ocl->buffer_triangle_emission);
    
    if (err != CL_SUCCESS) {
        printf("Error setting static triangle kernel arguments: %d\n", err);
        return 0;
    }
    
    printf("Static kernel arguments set successfully\n");
    return 1;
}

int initializeOpenCL(struct OpenCLContext *ocl, struct Triangles *triangles, struct SkyBox *skyBox) {
    cl_int err;
    
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
    
    // Create context
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL context: %d\n", err);
        return 0;
    }
    
    // Create command queue
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL command queue: %d\n", err);
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
    
    char *source = (char*)malloc(source_size + 1);
    fread(source, 1, source_size, file);
    source[source_size] = '\0';
    fclose(file);
    
    // Create program
    ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &source_size, &err);
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
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log: %s\n", log);
        free(log);
        return 0;
    }
    
    // Create particle projection kernel
    ocl->kernel = clCreateKernel(ocl->program, "project_points_to_screen", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating OpenCL kernel: %d\n", err);
        return 0;
    }

    // Create additional kernels
    ocl->blur_kernel = clCreateKernel(ocl->program, "blur_distances", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating blur kernel: %d\n", err);
        return 0;
    }

    // Create skybox kernel
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
    
    // Create triangle kernel
    ocl->triangle_kernel = clCreateKernel(ocl->program, "renderTriangles", &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle kernel: %d\n", err);
        return 0;
    }
    
    // Create particle buffers
    ocl->buffer_points = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                       NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating points buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_velocities = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                           NUM_PARTICLES * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating velocities buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_distances = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, 
                                          ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating distances buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_opacities = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, 
                                          ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating opacities buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_velocities_screen = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, 
                                                  ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating screen velocities buffer: %d\n", err);
        return 0;
    }
    
    // Create normals buffer
    ocl->buffer_normals = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY, 
                                        ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating normals buffer: %d\n", err);
        return 0;
    }
    
    // Create temporary buffers for blur pipeline
    ocl->buffer_distances_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, 
                                               ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating temp distances buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_opacities_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, 
                                               ScreenWidth * ScreenHeight * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating temp opacities buffer: %d\n", err);
        return 0;
    }

    // crete buffers for triangle properties
    ocl->buffer_triangle_roughness = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                           triangles->count * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle roughness buffer: %d\n", err);
        return 0;
    }
    ocl->buffer_triangle_metallic = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                           triangles->count * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle metalness buffer: %d\n", err);
        return 0;
    }
    ocl->buffer_triangle_emission = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                           triangles->count * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle emission buffer: %d\n", err);
        return 0;
    }
    
    // Create triangle buffers (allocated once, reused every frame)
    ocl->buffer_triangle_v1 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                           triangles->count * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle v1 buffer: %d\n", err);
        return 0;
    }

    ocl->buffer_triangle_v2 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                           triangles->count * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle v2 buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_triangle_v3 = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                            triangles->count * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle v3 buffer: %d\n", err);
        return 0;
    }
    
    ocl->buffer_triangle_normals = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                            triangles->count * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle normals buffer: %d\n", err);
        return 0;
    }
    
    // ADD TRIANGLE COLORS BUFFER
    ocl->buffer_triangle_colors = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, 
                                            triangles->count * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating triangle colors buffer: %d\n", err);
        return 0;
    }
    
    // ADD SCREEN COLORS BUFFER
    ocl->buffer_screen_colors = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, 
                                          ScreenWidth * ScreenHeight * 3 * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Error creating screen colors buffer: %d\n", err);
        return 0;
    }

    // *** INITIALIZE SKYBOX BUFFERS AND UPLOAD DATA ***
    if (!initializeSkyboxBuffers(ocl, skyBox)) {
        printf("Failed to initialize skybox buffers during OpenCL init\n");
        return 0;
    }

    // *** UPLOAD TRIANGLE DATA ONCE ***
    if (!uploadTriangleDataOnce(ocl, triangles)) {
        printf("Failed to upload triangle data during OpenCL init\n");
        return 0;
    }

    // Set static kernel arguments that never change
    if (!setupStaticKernelArguments(ocl, triangles, skyBox)) {
        printf("Failed to set static kernel arguments during OpenCL init\n");
        return 0;
    }
    
    // Pre-allocate host memory buffers
    ocl->host_points_data = (float*)malloc(NUM_PARTICLES * 3 * sizeof(float));
    ocl->host_velocities_data = (float*)malloc(NUM_PARTICLES * 3 * sizeof(float));
    ocl->host_distances_result = (float*)malloc(ScreenWidth * ScreenHeight * sizeof(float));
    ocl->host_opacities_result = (float*)malloc(ScreenWidth * ScreenHeight * sizeof(float));
    ocl->host_velocities_result = (float*)malloc(ScreenWidth * ScreenHeight * sizeof(float));
    ocl->host_normals_result = (float*)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));
    
    // ADD HOST MEMORY FOR SCREEN COLORS
    ocl->host_screen_colors_result = (float*)malloc(ScreenWidth * ScreenHeight * 3 * sizeof(float));
    
    // Check for allocation failures
    if (!ocl->host_points_data || !ocl->host_velocities_data || 
        !ocl->host_distances_result || !ocl->host_opacities_result || 
        !ocl->host_velocities_result || !ocl->host_normals_result || 
        !ocl->host_screen_colors_result) {
        printf("Failed to allocate host memory for OpenCL\n");
        return 0;
    }
    
    printf("OpenCL initialized successfully with triangle support and colors\n");
    return 1;
}

void projectParticlesOpenCL(struct OpenCLContext *ocl, struct PointSOA *particles, struct Camera *camera, struct Screen *screen, struct Triangles *triangles, struct SkyBox *skyBox) {
    cl_int err;  // ADD THIS LINE - it's missing!
    
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
        return;
    }
    
    err = clEnqueueWriteBuffer(ocl->queue, ocl->buffer_velocities, CL_TRUE, 0, 
                              NUM_PARTICLES * 3 * sizeof(float), velocities_data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error writing velocities buffer: %d\n", err);
        return;
    }
    
    // Clear screen buffers on GPU (INCLUDING NORMALS)
    float zero = 0.0f;
    
    err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_opacities, &zero, sizeof(float), 0, 
                             ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error clearing opacities buffer: %d\n", err);
        return;
    }
    
    err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_velocities_screen, &zero, sizeof(float), 0, 
                             ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error clearing screen velocities buffer: %d\n", err);
        return;
    }
    
    // ADD NORMALS BUFFER CLEARING
    err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_normals, &zero, sizeof(float), 0, 
                             ScreenWidth * ScreenHeight * 3 * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error clearing normals buffer: %d\n", err);
        return;
    }

    err = clEnqueueFillBuffer(ocl->queue,ocl->buffer_screen_colors,&zero,sizeof(float),0,
                            ScreenWidth * ScreenHeight * 3 * sizeof(float),0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error clearing screen colors: %d\n", err);
        return;
    }

    err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_distances, &zero, sizeof(float), 0, 
                             ScreenWidth * ScreenHeight * sizeof(float), 0, NULL, NULL);
    err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_screen_colors, &zero, sizeof(float), 0, 
                              ScreenWidth * ScreenHeight * 3 * sizeof(float), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error clearing buffers: %d\n", err);
        return;
    }

    // *** RENDER SKYBOX FIRST (fills background) ***
    renderSkyboxOpenCL(ocl, camera, skyBox); // You'll need to pass skyBox as parameter
    // *** TRIANGLE RENDERING ***
    renderTrianglesOpenCL(ocl, triangles, camera, screen, skyBox);
    
    // Set kernel arguments (FIXED ARGUMENT INDICES)
    cl_float3 cam_pos = {camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]};
    cl_float3 cam_dir = {camera->ray.direction[0], camera->ray.direction[1], camera->ray.direction[2]};
    cl_float3 cam_up = {0.0f, 1.0f, 0.0f};
    cl_float fov = camera->fov;
    cl_int screen_width = ScreenWidth;
    cl_int screen_height = ScreenHeight;
    cl_int num_points = NUM_PARTICLES;
    cl_int particle_radius = PARTICLE_RADIUS * 100.0f;
    
    err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), &ocl->buffer_points);
    err |= clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), &ocl->buffer_velocities);
    err |= clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), &ocl->buffer_distances);
    err |= clSetKernelArg(ocl->kernel, 3, sizeof(cl_mem), &ocl->buffer_opacities);
    err |= clSetKernelArg(ocl->kernel, 4, sizeof(cl_mem), &ocl->buffer_velocities_screen);
    err |= clSetKernelArg(ocl->kernel, 5, sizeof(cl_mem), &ocl->buffer_normals);        // FIXED: normals is arg 5
    err |= clSetKernelArg(ocl->kernel, 6, sizeof(cl_float3), &cam_pos);                // shifted +1
    err |= clSetKernelArg(ocl->kernel, 7, sizeof(cl_float3), &cam_dir);                // shifted +1
    err |= clSetKernelArg(ocl->kernel, 8, sizeof(cl_float3), &cam_up);                 // shifted +1
    err |= clSetKernelArg(ocl->kernel, 9, sizeof(cl_float), &fov);                     // shifted +1
    err |= clSetKernelArg(ocl->kernel, 10, sizeof(cl_int), &screen_width);             // shifted +1
    err |= clSetKernelArg(ocl->kernel, 11, sizeof(cl_int), &screen_height);            // shifted +1
    err |= clSetKernelArg(ocl->kernel, 12, sizeof(cl_int), &num_points);               // shifted +1
    err |= clSetKernelArg(ocl->kernel, 13, sizeof(cl_int), &particle_radius);          // FIXED: was duplicate arg 12, now arg 13
    
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arguments: %d\n", err);
        return;
    }
    
    // Execute kernel
    size_t global_work_size = NUM_PARTICLES;
    err = clEnqueueNDRangeKernel(ocl->queue, ocl->kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing kernel: %d\n", err);
        return;
    }

    // --- Multi-Pass Blur Section ---
    cl_mem s_dist_src, s_dist_dst, s_opac_src, s_opac_dst; // s_ for stage
    cl_mem temp_buf_holder; // For swapping

    // Parameters for blur kernel - adjust these to reduce outlines and get desired smoothness
    cl_int blur_kernel_size = 2;      // Half-width (e.g., 2 for 5x5 window). Try 2 or 3.
    cl_float blur_sigma_range = 15.0f; // Sigma for depth differences. Try values like 10.0, 15.0, 20.0.
    cl_float blur_sigma_spatial = 2.5f; // Sigma for spatial distance. Try 1.5, 2.0, 2.5.

    int blur_passes = 1; // Number of blur passes. 2-3 is usually sufficient for good smoothing.
    if (blur_passes > 0) {
        // Initial source for the first blur pass is the output of the projection kernel
        s_dist_src = ocl->buffer_distances;
        s_opac_src = ocl->buffer_opacities;

        for (int pass = 0; pass < blur_passes; ++pass) {
            // Determine destination buffer for this pass to achieve ping-pong
            // Pass 0 (1st pass): src=original, dst=temp
            // Pass 1 (2nd pass): src=temp,    dst=original
            // Pass 2 (3rd pass): src=original, dst=temp
            if (pass % 2 == 0) { // Output to _temp buffers
                s_dist_dst = ocl->buffer_distances_temp;
                s_opac_dst = ocl->buffer_opacities_temp;
            } else { // Output to original buffers (which now act as temp for this pass)
                s_dist_dst = ocl->buffer_distances;
                s_opac_dst = ocl->buffer_opacities;
            }

            // Set blur kernel arguments for the current pass
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
                return;
            }

            // Execute blur kernel
            size_t blur_global_work_size[2] = {ScreenWidth, ScreenHeight};
            err = clEnqueueNDRangeKernel(ocl->queue, ocl->blur_kernel, 2, NULL, blur_global_work_size, NULL, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                printf("Error executing blur kernel for pass %d: %d\n", pass, err);
                return;
            }
            
            // The output of this pass (s_dist_dst, s_opac_dst) becomes the input for the next
            s_dist_src = s_dist_dst;
            s_opac_src = s_opac_dst;
        }
    } else {
        // No blur passes, the "final" blurred data is just the direct output of projection
        s_dist_src = ocl->buffer_distances; // Or s_dist_dst, doesn't matter if passes = 0
        s_opac_src = ocl->buffer_opacities;
    }

    cl_mem final_blurred_distances_buf = s_dist_src;
    cl_mem final_blurred_opacities_buf = s_opac_src;

    // --- Normals Calculation Kernel ---
    // The normals kernel uses the output of the blur stage (final_blurred_distances_buf)
    err = clSetKernelArg(ocl->normals_kernel, 0, sizeof(cl_mem), &final_blurred_distances_buf); // Input: blurred distances
    if (err != CL_SUCCESS) { printf("Error setting normals_kernel arg 0: %d\n", err); return; }
    
    err = clSetKernelArg(ocl->normals_kernel, 1, sizeof(cl_mem), &ocl->buffer_normals);      // Output: screen normals
    if (err != CL_SUCCESS) { printf("Error setting normals_kernel arg 1: %d\n", err); return; }

    err = clSetKernelArg(ocl->normals_kernel, 2, sizeof(cl_int), &screen_width);
    if (err != CL_SUCCESS) { printf("Error setting normals_kernel arg 2: %d\n", err); return; }

    err = clSetKernelArg(ocl->normals_kernel, 3, sizeof(cl_int), &screen_height);
    if (err != CL_SUCCESS) { printf("Error setting normals_kernel arg 3: %d\n", err); return; }

    // Execute normals kernel
    size_t normals_global_work_size[2] = {ScreenWidth, ScreenHeight};
    err = clEnqueueNDRangeKernel(ocl->queue, ocl->normals_kernel, 2, NULL, normals_global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing normals kernel: %d\n", err);
        return;
    }
    
    // Use pre-allocated result buffers (INCLUDING NORMALS)
    float *distances_result = ocl->host_distances_result;
    float *opacities_result = ocl->host_opacities_result;
    float *velocities_result = ocl->host_velocities_result;
    float *normals_result = ocl->host_normals_result;
    
     // FIX: Read blurred distances instead of original distances
    err = clEnqueueReadBuffer(ocl->queue, final_blurred_distances_buf, CL_TRUE, 0, 
                             ScreenWidth * ScreenHeight * sizeof(float), distances_result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading blurred distances buffer: %d\n", err);
        return;
    }
    
   // FIX: Read blurred opacities instead of original opacities
    err = clEnqueueReadBuffer(ocl->queue, final_blurred_opacities_buf, CL_TRUE, 0, 
                             ScreenWidth * ScreenHeight * sizeof(float), opacities_result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading blurred opacities buffer: %d\n", err);
        return;
    }
    
    err = clEnqueueReadBuffer(ocl->queue, ocl->buffer_velocities_screen, CL_TRUE, 0, 
                             ScreenWidth * ScreenHeight * sizeof(float), velocities_result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading velocities buffer: %d\n", err);
        return;
    }
    
    // ADD NORMALS READBACK
    err = clEnqueueReadBuffer(ocl->queue, ocl->buffer_normals, CL_TRUE, 0, 
                             ScreenWidth * ScreenHeight * 3 * sizeof(float), normals_result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading normals buffer: %d\n", err);
        return;
    }

    // READ BACK SCREEN COLORS
    err = clEnqueueReadBuffer(ocl->queue, ocl->buffer_screen_colors, CL_TRUE, 0, 
                             ScreenWidth * ScreenHeight * 3 * sizeof(float), ocl->host_screen_colors_result, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading screen colors buffer: %d\n", err);
        return;
    }
    
    // Clear the screen arrays first (INCLUDING NORMALS)
    memset(screen->distance, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->velocity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->normalizedOpacity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->opacity, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
    memset(screen->normals, 0, sizeof(float) * ScreenWidth * ScreenHeight * 3);         // ADD NORMALS CLEAR
    // memset(screen->colors, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight * 3);  // ADD THIS
    
    // Find max values for normalization (skip zero values)
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
    
    // Convert results to screen format with proper normalization (INCLUDING NORMALS)
    for (int y = 0; y < ScreenHeight; y++) {
        for (int x = 0; x < ScreenWidth; x++) {
            int idx = y * ScreenWidth + x;
            
            // Only process pixels that have particles
            if (opacities_result[idx] > 0.0f) {
                // Distance: closer = higher value (like your C code)
                if (maxDistance > 0.0f) {
                    screen->distance[x][y] = (uint8_t)(255 * (1.0f - distances_result[idx] / (maxDistance + 1.0f )));
                }
                
                // Velocity: normalized linearly
                if (maxVelocity > 0.0f) {
                    screen->velocity[x][y] = (uint8_t)(255 * (velocities_result[idx] / (maxVelocity + 1.0f ) ));
                }
                
                // Linear opacity normalization
                if (maxOpacity > 0.0f) {
                    screen->normalizedOpacity[x][y] = (uint8_t)(255 * (opacities_result[idx] / (maxOpacity +1.0f ) ));
                }
            }
            
            // Copy normals (always copy, even for pixels without particles)
            float *nr = normals_result + idx * 3;
            screen->normals[x][y][0] = nr[0];
            screen->normals[x][y][1] = nr[1];
            screen->normals[x][y][2] = nr[2];

            // Copy screen colors (always copy, even for pixels without particles)
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
    
    *buf = (char*)malloc(*len + 1);
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

int main() {
    // load sky box texture
    struct SkyBox skyBox;
    if (!loadSkyBox(&skyBox)) {
        fprintf(stderr, "Failed to load skybox textures\n");
        return 1;
    }
    printf("SkyBox loaded successfully\n");

    // tinyobj_attrib_t attrib;  // Fix typo: was inyobj_attrib_t
    // tinyobj_shape_t* shapes = NULL;
    // size_t num_shapes;
    // tinyobj_material_t* materials = NULL;
    // size_t num_materials;

    // // Initialize attribute structure
    // tinyobj_attrib_init(&attrib);

    // // Load and triangulate .obj file with correct parameters
    // int result = tinyobj_parse_obj(
    //     &attrib, &shapes, &num_shapes, &materials, &num_materials,
    //     "model.obj", my_file_reader, NULL, TINYOBJ_FLAG_TRIANGULATE
    // );
    // if (result != TINYOBJ_SUCCESS) {
    //     fprintf(stderr, "Error loading OBJ\n");
    //     return 1;
    // }

    struct Triangles *triangles = (struct Triangles *)malloc(sizeof(struct Triangles));
    if (!triangles) {
        perror("Failed to allocate memory for triangles");
        return 1;
    }
    triangles->count = 0;

    // // Extract triangles from loaded OBJ data
    // for (size_t i = 0; i < num_shapes; ++i) {
    //     // Access face data directly from attrib structure
    //     size_t face_offset = shapes[i].face_offset;
    //     size_t num_faces = shapes[i].length / 3; // Each triangle has 3 vertices
        
    //     // Process each face (which is already triangulated)
    //     for (size_t f = 0; f < num_faces; ++f) {
    //         if (triangles->count >= NUMBER_OF_TRIANGLES) {
    //             printf("Warning: Model has more triangles than buffer size (%d)\n", NUMBER_OF_TRIANGLES);
    //             break;
    //         }
            
    //         // Get the three vertices of this triangle
    //         float v1[3], v2[3], v3[3];
            
    //         for (size_t v = 0; v < 3; ++v) {
    //             tinyobj_vertex_index_t idx = attrib.faces[face_offset + f * 3 + v];
                
    //             // Extract vertex coordinates
    //             float vx = attrib.vertices[3 * idx.v_idx + 0];
    //             float vy = attrib.vertices[3 * idx.v_idx + 1];
    //             float vz = attrib.vertices[3 * idx.v_idx + 2];
                
    //             if (v == 0) {
    //                 v1[0] = vx; v1[1] = vy; v1[2] = vz;
    //             } else if (v == 1) {
    //                 v2[0] = vx; v2[1] = vy; v2[2] = vz;
    //             } else {
    //                 v3[0] = vx; v3[1] = vy; v3[2] = vz;
    //             }
    //         }
            
    //         // Add triangle to your structure
    //         AddTriangle(triangles, 
    //                    v1[0], v1[1], v1[2],
    //                    v2[0], v2[1], v2[2],
    //                    v3[0], v3[1], v3[2],
    //                    rand(), rand(), rand()); // Normals will be recalculated later
    //     }
        
    //     // Break if we've reached the triangle limit
    //     if (triangles->count >= NUMBER_OF_TRIANGLES) {
    //         break;
    //     }
    // }

    // printf("Loaded %d triangles from OBJ file\n", triangles->count);

    // const float scaleFactor = 10.0f;

    // for (int i = 0; i < triangles->count; i++) {
    //     int idx = i * 3;
    //     triangles->v1[idx + 0] *= scaleFactor;
    //     triangles->v1[idx + 1] *= scaleFactor;
    //     triangles->v1[idx + 2] *= scaleFactor;
    //     triangles->v2[idx + 0] *= scaleFactor;
    //     triangles->v2[idx + 1] *= scaleFactor;
    //     triangles->v2[idx + 2] *= scaleFactor;
    //     triangles->v3[idx + 0] *= scaleFactor;
    //     triangles->v3[idx + 1] *= scaleFactor;
    //     triangles->v3[idx + 2] *= scaleFactor;

    //     // Optionally, recalc normals if needed:
    //     float ux = triangles->v2[idx + 0] - triangles->v1[idx + 0];
    //     float uy = triangles->v2[idx + 1] - triangles->v1[idx + 1];
    //     float uz = triangles->v2[idx + 2] - triangles->v1[idx + 2];
    //     float vx = triangles->v3[idx + 0] - triangles->v1[idx + 0];
    //     float vy = triangles->v3[idx + 1] - triangles->v1[idx + 1];
    //     float vz = triangles->v3[idx + 2] - triangles->v1[idx + 2];
    //     triangles->normals[idx + 0] = uy * vz - uz * vy;
    //     triangles->normals[idx + 1] = uz * vx - ux * vz;
    //     triangles->normals[idx + 2] = ux * vy - uy * vx;
    // }

    

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

    // Initialize collision thread system
    // initializeCollisionThreads();
    // // Initialize render thread system
    // initializeRenderThreads();

    // Initialize threadData array before creating threads
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i] = (struct ThreadData *)malloc(sizeof(struct ThreadData));
        if (!threadData[i]) {
            perror("Failed to allocate memory for thread data");
            return 1;
        }
        // Initialize basic fields
        threadData[i]->maxVelocity = 0.0f;
    }
    createThreads();

    struct PointSOA *particles = (struct PointSOA *)malloc(sizeof(struct PointSOA));
    if (!particles) {
        perror("Failed to allocate memory for particles");
        return 1;
    }


    // crete threadsData
    struct ThreadsData *threadsData = (struct ThreadsData *)malloc(sizeof(struct ThreadsData));
    if (!threadsData) {
        perror("Failed to allocate memory for threads data");
        free(particles);
        return 1;
    }
    
    // Initialize each thread data pointer
    for (int i = 0; i < NUM_THREADS; i++) {
        threadsData->threadData[i] = (struct ThreadData *)malloc(sizeof(struct ThreadData));
        if (!threadsData->threadData[i]) {
            perror("Failed to allocate memory for thread data");
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(threadsData->threadData[j]);
            }
            free(threadsData);
            free(particles);
            return 1;
        }
    }

    asignDataToThreads(particles, &camera, NUM_PARTICLES);

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
    CreateBoardPlane(0.0f, -20.0f, 0.0f, 50.0f, 32, triangles);

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
        float Emissive = (float)rand_01();
        CreateCube(x, y, z, size, triangles, r, g, b, Metallic, Roughness, Emissive);
    }


    struct OpenCLContext ocl;
    int useOpenCL = initializeOpenCL(&ocl, triangles, &skyBox);
    if (!useOpenCL) {
        printf("Failed to initialize OpenCL, falling back to CPU\n");
    }

    

    while (1) {
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
        readCameraData(&camera);
        readCursorData(cursor);
        readPauseData(&paused);
        clock_t endReadDataTime = clock();
        float dt1 = (float)(endReadDataTime - readDataTime) / (float)CLOCKS_PER_SEC;
        timePartition->readDataTime += dt1;
        
        // Update the grid data and record timing
        // First, update the grid data regardless of pause state
        clock_t startGridTime = clock();
        updateGridData(particles);
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
        render(screen, particles, &camera, cursor, timePartition, threadsData, particleIndexes, &lightCamera, lightScreen, &ocl, triangles, &skyBox);
        clock_t endRenderTime = clock();
        dt1 = (float)(endRenderTime - startRenderTime) / (float)CLOCKS_PER_SEC;
        timePartition->renderTime += dt1;

    
        float currentFPS = 1.0f / dt1;
        
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
