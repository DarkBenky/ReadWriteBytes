#ifndef OPEN_CL_STRUCT_H
#define OPEN_CL_STRUCT_H
#define NUMBER_OF_TRIANGLES 10000

#include <CL/cl.h>
#include <GL/gl.h>
#include <stdbool.h>

struct OpenCLContext {
	cl_platform_id platform;
	cl_device_id device;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	// kernels
	cl_kernel antiAliasKernel;
	cl_kernel drawBoundingBox_kernel;
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
	cl_kernel fire_sim_kernel;				// Fire simulation kernel
	cl_program fire_program;              // Fire simulation program
	cl_kernel fire_render_kernel;         // Fire render kernel
	cl_kernel blur_fire_kernel;          // Fire blur kernel
	cl_kernel clearColorBuffer_kernel;   // Clear color buffer kernel
	cl_kernel composite_kernel;          // Compositing kernel
	cl_kernel renderMissile_kernel;      // Missile rendering kernel
	cl_kernel renderFireTemperature_kernel; // Fire temperature rendering kernel
	cl_kernel overlayImage_kernel;        // Overlay image kernel
	cl_kernel renderDepth;
	// buffers
	cl_mem buffer_seeker_distances;
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
	// triangle buffers for missile model
	cl_mem buffer_missile_v1;
	cl_mem buffer_missile_v2;
	cl_mem buffer_missile_v3;
	cl_mem buffer_missile_normals;
	cl_mem missile_color_buffer;
	cl_mem missile_roughness_buffer;
	cl_mem missile_metallic_buffer;
	cl_mem missile_emission_buffer;	

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
	// fire rendering
	cl_mem posX;
	cl_mem posY;
	cl_mem posZ;
	cl_mem velX;
	cl_mem velY;
	cl_mem velZ;
	cl_mem lifeTime;
    cl_mem FireScreenDistances;
	cl_mem FireScreenDistancesTemp;
    cl_mem FireScreenColors;
	cl_mem FireScreenColorsTemp;
    cl_mem FireScreenNormals;
	cl_mem FireScreenAlphas;
	cl_mem FireScreenAlphasTemp;
	cl_mem FireTemperature;
	// compositing buffers
	cl_mem CompositedScreenColors;
	cl_mem CompositedScreenNormals;
	cl_mem CompositedScreenDistances;
	

	

	// Add pre-allocated host memory buffers
	float *host_points_data;
	float *host_velocities_data;
	float *host_distances_result;
	float *host_opacities_result;
	float *host_velocities_result;
	float *host_normals_result;
	float *host_screen_colors_result;
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

enum RenderMode {
	renderDistance,
	renderVelocity,
	renderOpacity,
	renderNormal,
	renderFluid,
	renderColor,
	renderWireframe,
	renderFireColor,
	renderFireDepth,
	renderFireNormal,
	renderCompositedNormal,
	renderCompositedColor,
	renderCompositedDistance,
	renderTemperatures,
	RENDER_MODE_COUNT // Total number of render modes
};

struct Ray {
	float origin[3];
	float direction[3];
};

struct Camera {
	struct Ray ray;
	float fov;
	enum RenderMode renderMode;
	bool AntiAlias;
	bool advanceAntiAlias;
};

#endif
