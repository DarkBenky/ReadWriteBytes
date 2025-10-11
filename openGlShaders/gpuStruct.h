#ifndef OPEN_CL_STRUCT_H
#define OPEN_CL_STRUCT_H

#include <CL/cl.h>
#include <GL/gl.h>

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
	// fire simulation particles buffers
	cl_mem posX;
	cl_mem posY;
	cl_mem posZ;
	cl_mem velX;
	cl_mem velY;
	cl_mem velZ;
	cl_mem lifeTime;
	cl_mem maxDepth;
    cl_mem basePosition;
    cl_mem staringColor;
    cl_mem fireColor;
    cl_mem smokeColor;
    cl_mem maxLifeTime;

	// rendering buffers for fire simulation
	cl_mem buffer_color;
	cl_mem buffer_depth;
	cl_mem buffer_temp;

	// Add pre-allocated host memory buffers
	float *host_points_data;
	float *host_velocities_data;
	float *host_distances_result;
	float *host_opacities_result;
	float *host_velocities_result;
	float *host_normals_result;
	float *host_screen_colors_result;
};
#endif
