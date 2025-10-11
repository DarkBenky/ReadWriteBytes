#include "../openGlShaders/gpuStruct.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

#define NUM_FIRE_PARTICLES 2500
#define RAND_MAXF (float)RAND_MAX

int initOpenCLFireSim(struct OpenCLContext *ocl, const char *kernelSource, int screenWidth, int screenHeight, float *basePosition, float *startingColor, float *fireColor, float *smokeColor, float maxLifeTime) {
	cl_int err;

	// Compile the kernel source
	const char *sources[] = {kernelSource};
	size_t sourceSizes[] = {strlen(kernelSource)};

	cl_program fireProgram = clCreateProgramWithSource(ocl->context, 1, sources, sourceSizes, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create fire program: %d\n", err);
		return -1;
	}

	// Build the program
	err = clBuildProgram(fireProgram, 1, &ocl->device, "-cl-fast-relaxed-math", NULL, NULL);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to build fire program: %d\n", err);

		// Print build log
		size_t logSize;
		clGetProgramBuildInfo(fireProgram, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		char *log = (char *)malloc(logSize);
		clGetProgramBuildInfo(fireProgram, ocl->device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
		fprintf(stderr, "Build log:\n%s\n", log);
		free(log);

		clReleaseProgram(fireProgram);
		return -1;
	}

	// Create kernels
	ocl->fire_sim_kernel = clCreateKernel(fireProgram, "fireSim", &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create fireSim kernel: %d\n", err);
		clReleaseProgram(fireProgram);
		return -1;
	}
	// Create renderParticles kernel from the same program
	ocl->fire_render_kernel = clCreateKernel(fireProgram, "renderParticles", &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create renderParticles kernel: %d\n", err);
		clReleaseKernel(ocl->fire_sim_kernel);
		clReleaseProgram(fireProgram);
		return -1;
	}
	// Store program to release later
	ocl->fire_program = fireProgram;

	// Define number of particles
	int numParticles = 10000;

	// Create particle buffers
	ocl->posX = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create posX buffer\n");
		return -1;
	}

	ocl->posY = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create posY buffer\n");
		return -1;
	}

	ocl->posZ = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create posZ buffer\n");
		return -1;
	}

	ocl->velX = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create velX buffer\n");
		return -1;
	}

	ocl->velY = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create velY buffer\n");
		return -1;
	}

	ocl->velZ = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create velZ buffer\n");
		return -1;
	}

	ocl->lifeTime = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float) * numParticles, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create lifeTime buffer\n");
		return -1;
	}

	// Create rendering buffers
	ocl->buffer_color = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
									   sizeof(float) * screenWidth * screenHeight * 3, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create color buffer\n");
		return -1;
	}

	ocl->buffer_depth = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
									   sizeof(float) * screenWidth * screenHeight, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create depth buffer\n");
		return -1;
	}

	ocl->buffer_temp = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,
									  sizeof(float) * screenWidth * screenHeight * 3, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create temp buffer\n");
		return -1;
	}

	ocl->maxDepth = clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create maxDepth buffer\n");
		return -1;
	}

	// Store color parameters as buffers
	ocl->basePosition = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									   sizeof(float) * 3, basePosition, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create basePosition buffer\n");
		return -1;
	}

	ocl->staringColor = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									   sizeof(float) * 3, startingColor, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create startingColor buffer\n");
		return -1;
	}

	ocl->fireColor = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									sizeof(float) * 3, fireColor, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create fireColor buffer\n");
		return -1;
	}

	ocl->smokeColor = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									 sizeof(float) * 3, smokeColor, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create smokeColor buffer\n");
		return -1;
	}

	ocl->maxLifeTime = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
									  sizeof(float), &maxLifeTime, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to create maxLifeTime buffer\n");
		return -1;
	}

	// Initialize particles with random values
	float *posX_host = (float *)malloc(sizeof(float) * numParticles);
	float *posY_host = (float *)malloc(sizeof(float) * numParticles);
	float *posZ_host = (float *)malloc(sizeof(float) * numParticles);
	float *velX_host = (float *)malloc(sizeof(float) * numParticles);
	float *velY_host = (float *)malloc(sizeof(float) * numParticles);
	float *velZ_host = (float *)malloc(sizeof(float) * numParticles);
	float *lifeTime_host = (float *)malloc(sizeof(float) * numParticles);

	for (int i = 0; i < numParticles; i++) {
		posX_host[i] = basePosition[0] + ((float)rand() / RAND_MAXF - 0.5f) * 0.8f;
		posY_host[i] = basePosition[1] + ((float)rand() / RAND_MAXF) * 0.3f;
		posZ_host[i] = basePosition[2] + ((float)rand() / RAND_MAXF - 0.5f) * 0.8f;
		velX_host[i] = ((float)rand() / RAND_MAXF - 0.5f) * 1.6f;
		velY_host[i] = ((float)rand() / RAND_MAXF) * 3.0f + 2.5f;
		velZ_host[i] = ((float)rand() / RAND_MAXF - 0.5f) * 1.6f;
		lifeTime_host[i] = ((float)rand() / RAND_MAXF) * 0.2f;
	}

	// Upload initial particle data to device
	err = clEnqueueWriteBuffer(ocl->queue, ocl->posX, CL_TRUE, 0,
							   sizeof(float) * numParticles, posX_host, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->posY, CL_TRUE, 0,
								sizeof(float) * numParticles, posY_host, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->posZ, CL_TRUE, 0,
								sizeof(float) * numParticles, posZ_host, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->velX, CL_TRUE, 0,
								sizeof(float) * numParticles, velX_host, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->velY, CL_TRUE, 0,
								sizeof(float) * numParticles, velY_host, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->velZ, CL_TRUE, 0,
								sizeof(float) * numParticles, velZ_host, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(ocl->queue, ocl->lifeTime, CL_TRUE, 0,
								sizeof(float) * numParticles, lifeTime_host, 0, NULL, NULL);

	free(posX_host);
	free(posY_host);
	free(posZ_host);
	free(velX_host);
	free(velY_host);
	free(velZ_host);
	free(lifeTime_host);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to upload initial particle data: %d\n", err);
		return -1;
	}

	// Initialize color and depth buffers to zero
	float zero = 0.0f;
	err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_color, &zero, sizeof(float),
							  0, sizeof(float) * screenWidth * screenHeight * 3, 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_depth, &zero, sizeof(float),
							   0, sizeof(float) * screenWidth * screenHeight, 0, NULL, NULL);

	clFinish(ocl->queue);
	// Program will be released in cleanupFireSim
	printf("Fire simulation initialized: %d particles, %dx%d resolution\n",
		   numParticles, screenWidth, screenHeight);

	return 0;
}

void simulateFireStep(struct OpenCLContext *ocl, int numParticles, float deltaTime, float *kernelTime) {
	cl_int err;
	cl_event event;

	// Read base position and max lifetime from buffers
	float basePos[3];
	float maxLifeTime;

	err = clEnqueueReadBuffer(ocl->queue, ocl->basePosition, CL_TRUE, 0,
							  sizeof(float) * 3, basePos, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->maxLifeTime, CL_TRUE, 0,
							   sizeof(float), &maxLifeTime, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to read fire parameters: %d\n", err);
		return;
	}

	cl_float3 basePosVec = {{basePos[0], basePos[1], basePos[2]}};

	// Set kernel arguments
	err = clSetKernelArg(ocl->fire_sim_kernel, 0, sizeof(cl_mem), &ocl->posX);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 1, sizeof(cl_mem), &ocl->posY);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 2, sizeof(cl_mem), &ocl->posZ);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 3, sizeof(cl_mem), &ocl->velX);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 4, sizeof(cl_mem), &ocl->velY);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 5, sizeof(cl_mem), &ocl->velZ);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 6, sizeof(cl_mem), &ocl->lifeTime);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 7, sizeof(cl_float3), &basePosVec);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 8, sizeof(float), &maxLifeTime);
	err |= clSetKernelArg(ocl->fire_sim_kernel, 9, sizeof(float), &deltaTime);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to set fireSim kernel arguments: %d\n", err);
		return;
	}

	// Execute kernel
	size_t globalSize = numParticles;
	err = clEnqueueNDRangeKernel(ocl->queue, ocl->fire_sim_kernel, 1, NULL,
								 &globalSize, NULL, 0, NULL, &event);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to execute fireSim kernel: %d\n", err);
		return;
	}

	// Wait for completion and get timing
	clWaitForEvents(1, &event);

	if (kernelTime != NULL) {
		cl_ulong start, end;
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
		*kernelTime = (end - start) / 1000000.0f; // Convert to milliseconds
	}

	clReleaseEvent(event);
}

void renderFireParticles(struct OpenCLContext *ocl, int numParticles, int screenWidth, int screenHeight,
						 float *viewMatrix, float *projMatrix, float particleSize, float *kernelTime) {
	cl_int err;
	cl_event events[5];
	int eventCount = 0;

	// Ensure renderParticles kernel is initialized
	if (ocl->fire_render_kernel == NULL) {
		fprintf(stderr, "Render kernel not initialized\n");
		return;
	}

	// Read color parameters from buffers
	float startingColor[3], fireColor[3], smokeColor[3], maxLifeTime;

	err = clEnqueueReadBuffer(ocl->queue, ocl->staringColor, CL_TRUE, 0,
							  sizeof(float) * 3, startingColor, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->fireColor, CL_TRUE, 0,
							   sizeof(float) * 3, fireColor, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->smokeColor, CL_TRUE, 0,
							   sizeof(float) * 3, smokeColor, 0, NULL, NULL);
	err |= clEnqueueReadBuffer(ocl->queue, ocl->maxLifeTime, CL_TRUE, 0,
							   sizeof(float), &maxLifeTime, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to read color parameters: %d\n", err);
		return;
	}

	cl_float3 startingColorVec = {{startingColor[0], startingColor[1], startingColor[2]}};
	cl_float3 fireColorVec = {{fireColor[0], fireColor[1], fireColor[2]}};
	cl_float3 smokeColorVec = {{smokeColor[0], smokeColor[1], smokeColor[2]}};

	// Convert matrices to cl_float16
	cl_float16 viewMat, projMat;
	memcpy(&viewMat, viewMatrix, sizeof(float) * 16);
	memcpy(&projMat, projMatrix, sizeof(float) * 16);

	// Clear color and depth buffers
	float zero = 0.0f;
	err = clEnqueueFillBuffer(ocl->queue, ocl->buffer_color, &zero, sizeof(float),
							  0, sizeof(float) * screenWidth * screenHeight * 3, 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->buffer_depth, &zero, sizeof(float),
							   0, sizeof(float) * screenWidth * screenHeight, 0, NULL, NULL);
	err |= clEnqueueFillBuffer(ocl->queue, ocl->maxDepth, &zero, sizeof(float),
							   0, sizeof(float), 0, NULL, NULL);

	// Use renderParticles kernel
	cl_kernel renderKernel = ocl->fire_render_kernel;

	// Set render kernel arguments
	err = clSetKernelArg(renderKernel, 0, sizeof(cl_mem), &ocl->posX);
	err |= clSetKernelArg(renderKernel, 1, sizeof(cl_mem), &ocl->posY);
	err |= clSetKernelArg(renderKernel, 2, sizeof(cl_mem), &ocl->posZ);
	err |= clSetKernelArg(renderKernel, 3, sizeof(cl_mem), &ocl->lifeTime);
	err |= clSetKernelArg(renderKernel, 4, sizeof(cl_mem), &ocl->buffer_color);
	err |= clSetKernelArg(renderKernel, 5, sizeof(cl_mem), &ocl->buffer_depth);
	err |= clSetKernelArg(renderKernel, 6, sizeof(cl_float3), &startingColorVec);
	err |= clSetKernelArg(renderKernel, 7, sizeof(cl_float3), &fireColorVec);
	err |= clSetKernelArg(renderKernel, 8, sizeof(cl_float3), &smokeColorVec);
	err |= clSetKernelArg(renderKernel, 9, sizeof(float), &maxLifeTime);
	err |= clSetKernelArg(renderKernel, 10, sizeof(float), &particleSize);
	err |= clSetKernelArg(renderKernel, 11, sizeof(int), &screenWidth);
	err |= clSetKernelArg(renderKernel, 12, sizeof(int), &screenHeight);
	err |= clSetKernelArg(renderKernel, 13, sizeof(cl_float16), &viewMat);
	err |= clSetKernelArg(renderKernel, 14, sizeof(cl_float16), &projMat);
	err |= clSetKernelArg(renderKernel, 15, sizeof(cl_mem), &ocl->maxDepth);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to set renderParticles kernel arguments: %d\n", err);
		return;
	}

	// Execute render kernel over each particle
	size_t globalSize = (size_t)numParticles;
	err = clEnqueueNDRangeKernel(ocl->queue, renderKernel, 1, NULL,
								 &globalSize, NULL, 0, NULL, &events[eventCount++]);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to execute renderParticles kernel: %d\n", err);
		return;
	}

	// Apply blur
	if (ocl->blur_kernel != NULL) {
		int pass = 0;
		err = clSetKernelArg(ocl->blur_kernel, 0, sizeof(cl_mem), &ocl->buffer_color);
		err |= clSetKernelArg(ocl->blur_kernel, 1, sizeof(cl_mem), &ocl->buffer_temp);
		err |= clSetKernelArg(ocl->blur_kernel, 2, sizeof(int), &screenWidth);
		err |= clSetKernelArg(ocl->blur_kernel, 3, sizeof(int), &screenHeight);
		err |= clSetKernelArg(ocl->blur_kernel, 4, sizeof(int), &pass);

		size_t globalSize2D[2] = {screenWidth, screenHeight};
		err = clEnqueueNDRangeKernel(ocl->queue, ocl->blur_kernel, 2, NULL,
									 globalSize2D, NULL, 0, NULL, &events[eventCount++]);

		// Second pass
		pass = 1;
		err |= clSetKernelArg(ocl->blur_kernel, 4, sizeof(int), &pass);
		err = clEnqueueNDRangeKernel(ocl->queue, ocl->blur_kernel, 2, NULL,
									 globalSize2D, NULL, 0, NULL, &events[eventCount++]);
	}

	// Wait for all operations to complete
	clWaitForEvents(eventCount, events);

	// Calculate total time
	if (kernelTime != NULL) {
		*kernelTime = 0.0f;
		for (int i = 0; i < eventCount; i++) {
			cl_ulong start, end;
			clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
			clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
			*kernelTime += (end - start) / 1000000.0f; // Convert to milliseconds
		}
	}

	// Release events
	for (int i = 0; i < eventCount; i++) {
		clReleaseEvent(events[i]);
	}
}

void downloadFireRenderBuffer(struct OpenCLContext *ocl, int screenWidth, int screenHeight, float *outputBuffer) {
	cl_int err;

	err = clEnqueueReadBuffer(ocl->queue, ocl->buffer_color, CL_TRUE, 0,
							  sizeof(float) * screenWidth * screenHeight * 3,
							  outputBuffer, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to download fire render buffer: %d\n", err);
	}
}

void updateFireBasePosition(struct OpenCLContext *ocl, float *newBasePosition) {
	cl_int err;

	err = clEnqueueWriteBuffer(ocl->queue, ocl->basePosition, CL_TRUE, 0,
							   sizeof(float) * 3, newBasePosition, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to update fire base position: %d\n", err);
	}
}

void updateFireColors(struct OpenCLContext *ocl, float *startingColor, float *fireColor, float *smokeColor) {
	cl_int err;

	if (startingColor != NULL) {
		err = clEnqueueWriteBuffer(ocl->queue, ocl->staringColor, CL_TRUE, 0,
								   sizeof(float) * 3, startingColor, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Failed to update starting color: %d\n", err);
		}
	}

	if (fireColor != NULL) {
		err = clEnqueueWriteBuffer(ocl->queue, ocl->fireColor, CL_TRUE, 0,
								   sizeof(float) * 3, fireColor, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Failed to update fire color: %d\n", err);
		}
	}

	if (smokeColor != NULL) {
		err = clEnqueueWriteBuffer(ocl->queue, ocl->smokeColor, CL_TRUE, 0,
								   sizeof(float) * 3, smokeColor, 0, NULL, NULL);
		if (err != CL_SUCCESS) {
			fprintf(stderr, "Failed to update smoke color: %d\n", err);
		}
	}
}

void updateFireMaxLifeTime(struct OpenCLContext *ocl, float maxLifeTime) {
	cl_int err;

	err = clEnqueueWriteBuffer(ocl->queue, ocl->maxLifeTime, CL_TRUE, 0,
							   sizeof(float), &maxLifeTime, 0, NULL, NULL);

	if (err != CL_SUCCESS) {
		fprintf(stderr, "Failed to update max lifetime: %d\n", err);
	}
}

void cleanupFireSim(struct OpenCLContext *ocl) {
	if (ocl->posX) clReleaseMemObject(ocl->posX);
	if (ocl->posY) clReleaseMemObject(ocl->posY);
	if (ocl->posZ) clReleaseMemObject(ocl->posZ);
	if (ocl->velX) clReleaseMemObject(ocl->velX);
	if (ocl->velY) clReleaseMemObject(ocl->velY);
	if (ocl->velZ) clReleaseMemObject(ocl->velZ);
	if (ocl->lifeTime) clReleaseMemObject(ocl->lifeTime);
	if (ocl->buffer_color) clReleaseMemObject(ocl->buffer_color);
	if (ocl->buffer_depth) clReleaseMemObject(ocl->buffer_depth);
	if (ocl->buffer_temp) clReleaseMemObject(ocl->buffer_temp);
	if (ocl->maxDepth) clReleaseMemObject(ocl->maxDepth);
	if (ocl->basePosition) clReleaseMemObject(ocl->basePosition);
	if (ocl->staringColor) clReleaseMemObject(ocl->staringColor);
	if (ocl->fireColor) clReleaseMemObject(ocl->fireColor);
	if (ocl->smokeColor) clReleaseMemObject(ocl->smokeColor);
	if (ocl->maxLifeTime) clReleaseMemObject(ocl->maxLifeTime);
	if (ocl->fire_sim_kernel) clReleaseKernel(ocl->fire_sim_kernel);

	printf("Fire simulation cleaned up\n");
	// Release fire simulation kernels and program
	if (ocl->fire_sim_kernel) clReleaseKernel(ocl->fire_sim_kernel);
	if (ocl->fire_render_kernel) clReleaseKernel(ocl->fire_render_kernel);
	if (ocl->fire_program) clReleaseProgram(ocl->fire_program);
}
