#include "fireSim.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void initFireSimulation(struct Particles *particles) {
	for (int i = 0; i < NUM_PARTICLES; i++) {
		particles->posX[i] = particles->basePos[0] + (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
		particles->posY[i] = particles->basePos[1];
		particles->posZ[i] = particles->basePos[2] + (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
		particles->velX[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
		particles->velY[i] = rand() / (float)RAND_MAX * 3.0f + 2.0f;
		particles->velZ[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
		particles->lifeTime[i] = rand() / (float)RAND_MAX * particles->maxLifeTime;
	}
}

void renderFireParticles(struct OpenCLContextFireSim *cl, struct Particles *particles,
						 int screenWidth, int screenHeight, float *viewMatrix, float *projMatrix) {

	cl_float3 baseColor = {{particles->baseColor[0], particles->baseColor[1], particles->baseColor[2]}};
	cl_float3 fireColor = {{particles->fireColor[0], particles->fireColor[1], particles->fireColor[2]}};
	cl_float3 smokeColor = {{particles->SmokeColor[0], particles->SmokeColor[1], particles->SmokeColor[2]}};
	cl_float16 viewMat, projMat;

	for (int i = 0; i < 16; i++) {
		viewMat.s[i] = viewMatrix[i];
		projMat.s[i] = projMatrix[i];
	}

	size_t globalSize = NUM_PARTICLES;

	clSetKernelArg(cl->kernelRenderParticles, 0, sizeof(cl_mem), &cl->posX);
	clSetKernelArg(cl->kernelRenderParticles, 1, sizeof(cl_mem), &cl->posY);
	clSetKernelArg(cl->kernelRenderParticles, 2, sizeof(cl_mem), &cl->posZ);
	clSetKernelArg(cl->kernelRenderParticles, 3, sizeof(cl_mem), &cl->lifeTime);
	clSetKernelArg(cl->kernelRenderParticles, 4, sizeof(cl_mem), &cl->buffer_color);
	clSetKernelArg(cl->kernelRenderParticles, 5, sizeof(cl_mem), &cl->buffer_depth);
	clSetKernelArg(cl->kernelRenderParticles, 6, sizeof(cl_float3), &baseColor);
	clSetKernelArg(cl->kernelRenderParticles, 7, sizeof(cl_float3), &fireColor);
	clSetKernelArg(cl->kernelRenderParticles, 8, sizeof(cl_float3), &smokeColor);
	clSetKernelArg(cl->kernelRenderParticles, 9, sizeof(float), &particles->maxLifeTime);
	clSetKernelArg(cl->kernelRenderParticles, 10, sizeof(float), &particles->particleSize);
	clSetKernelArg(cl->kernelRenderParticles, 11, sizeof(int), &screenWidth);
	clSetKernelArg(cl->kernelRenderParticles, 12, sizeof(int), &screenHeight);
	clSetKernelArg(cl->kernelRenderParticles, 13, sizeof(cl_float16), &viewMat);
	clSetKernelArg(cl->kernelRenderParticles, 14, sizeof(cl_float16), &projMat);

	clEnqueueNDRangeKernel(cl->queue, cl->kernelRenderParticles, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

	size_t globalSize2D[2] = {screenWidth, screenHeight};

	clSetKernelArg(cl->kernelBlurFire, 0, sizeof(cl_mem), &cl->buffer_color);
	clSetKernelArg(cl->kernelBlurFire, 1, sizeof(cl_mem), &cl->buffer_temp);
	clSetKernelArg(cl->kernelBlurFire, 2, sizeof(int), &screenWidth);
	clSetKernelArg(cl->kernelBlurFire, 3, sizeof(int), &screenHeight);

	int pass = 0;
	clSetKernelArg(cl->kernelBlurFire, 4, sizeof(int), &pass);
	clEnqueueNDRangeKernel(cl->queue, cl->kernelBlurFire, 2, NULL, globalSize2D, NULL, 0, NULL, NULL);

	pass = 1;
	clSetKernelArg(cl->kernelBlurFire, 4, sizeof(int), &pass);
	clEnqueueNDRangeKernel(cl->queue, cl->kernelBlurFire, 2, NULL, globalSize2D, NULL, 0, NULL, NULL);

	clFinish(cl->queue);
}

int initOpenCLFireSim(struct OpenCLContextFireSim *cl, const char *kernelSource) {
	cl_int err;

	err = clGetPlatformIDs(1, &cl->platform, NULL);
	if (err != CL_SUCCESS) return -1;

	err = clGetDeviceIDs(cl->platform, CL_DEVICE_TYPE_GPU, 1, &cl->device, NULL);
	if (err != CL_SUCCESS) return -1;

	cl->context = clCreateContext(NULL, 1, &cl->device, NULL, NULL, &err);
	if (err != CL_SUCCESS) return -1;

	cl->queue = clCreateCommandQueue(cl->context, cl->device, 0, &err);
	if (err != CL_SUCCESS) return -1;

	size_t sourceLen = strlen(kernelSource);
	cl->program = clCreateProgramWithSource(cl->context, 1, &kernelSource, &sourceLen, &err);
	if (err != CL_SUCCESS) return -1;

	err = clBuildProgram(cl->program, 1, &cl->device, NULL, NULL, NULL);
	if (err != CL_SUCCESS) return -1;

	cl->kernelUpdateParticles = clCreateKernel(cl->program, "fireSim", &err);
	if (err != CL_SUCCESS) return -1;

	cl->kernelRenderParticles = clCreateKernel(cl->program, "renderParticles", &err);
	if (err != CL_SUCCESS) return -1;

	cl->kernelBlurFire = clCreateKernel(cl->program, "blurFire", &err);
	if (err != CL_SUCCESS) return -1;

	cl->posX = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);
	cl->posY = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);
	cl->posZ = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);
	cl->velX = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);
	cl->velY = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);
	cl->velZ = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);
	cl->lifeTime = clCreateBuffer(cl->context, CL_MEM_READ_WRITE, sizeof(float) * NUM_PARTICLES, NULL, &err);

	return 0;
}

void stepFireSimulation(struct OpenCLContextFireSim *cl, struct Particles *particles, float deltaTime) {
	cl_float3 basePos = {{particles->basePos[0], particles->basePos[1], particles->basePos[2]}};

	clEnqueueWriteBuffer(cl->queue, cl->posX, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->posX, 0, NULL, NULL);
	clEnqueueWriteBuffer(cl->queue, cl->posY, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->posY, 0, NULL, NULL);
	clEnqueueWriteBuffer(cl->queue, cl->posZ, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->posZ, 0, NULL, NULL);
	clEnqueueWriteBuffer(cl->queue, cl->velX, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->velX, 0, NULL, NULL);
	clEnqueueWriteBuffer(cl->queue, cl->velY, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->velY, 0, NULL, NULL);
	clEnqueueWriteBuffer(cl->queue, cl->velZ, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->velZ, 0, NULL, NULL);
	clEnqueueWriteBuffer(cl->queue, cl->lifeTime, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->lifeTime, 0, NULL, NULL);

	clSetKernelArg(cl->kernelUpdateParticles, 0, sizeof(cl_mem), &cl->posX);
	clSetKernelArg(cl->kernelUpdateParticles, 1, sizeof(cl_mem), &cl->posY);
	clSetKernelArg(cl->kernelUpdateParticles, 2, sizeof(cl_mem), &cl->posZ);
	clSetKernelArg(cl->kernelUpdateParticles, 3, sizeof(cl_mem), &cl->velX);
	clSetKernelArg(cl->kernelUpdateParticles, 4, sizeof(cl_mem), &cl->velY);
	clSetKernelArg(cl->kernelUpdateParticles, 5, sizeof(cl_mem), &cl->velZ);
	clSetKernelArg(cl->kernelUpdateParticles, 6, sizeof(cl_mem), &cl->lifeTime);
	clSetKernelArg(cl->kernelUpdateParticles, 7, sizeof(cl_float3), &basePos);
	clSetKernelArg(cl->kernelUpdateParticles, 8, sizeof(float), &particles->maxLifeTime);
	clSetKernelArg(cl->kernelUpdateParticles, 9, sizeof(float), &deltaTime);

	size_t globalSize = NUM_PARTICLES;
	clEnqueueNDRangeKernel(cl->queue, cl->kernelUpdateParticles, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

	clEnqueueReadBuffer(cl->queue, cl->posX, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->posX, 0, NULL, NULL);
	clEnqueueReadBuffer(cl->queue, cl->posY, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->posY, 0, NULL, NULL);
	clEnqueueReadBuffer(cl->queue, cl->posZ, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->posZ, 0, NULL, NULL);
	clEnqueueReadBuffer(cl->queue, cl->velX, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->velX, 0, NULL, NULL);
	clEnqueueReadBuffer(cl->queue, cl->velY, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->velY, 0, NULL, NULL);
	clEnqueueReadBuffer(cl->queue, cl->velZ, CL_FALSE, 0, sizeof(float) * NUM_PARTICLES, particles->velZ, 0, NULL, NULL);
	clEnqueueReadBuffer(cl->queue, cl->lifeTime, CL_TRUE, 0, sizeof(float) * NUM_PARTICLES, particles->lifeTime, 0, NULL, NULL);
}

void cleanupFireSim(struct OpenCLContextFireSim *cl) {
	if (cl->posX) clReleaseMemObject(cl->posX);
	if (cl->posY) clReleaseMemObject(cl->posY);
	if (cl->posZ) clReleaseMemObject(cl->posZ);
	if (cl->velX) clReleaseMemObject(cl->velX);
	if (cl->velY) clReleaseMemObject(cl->velY);
	if (cl->velZ) clReleaseMemObject(cl->velZ);
	if (cl->lifeTime) clReleaseMemObject(cl->lifeTime);
	if (cl->buffer_color) clReleaseMemObject(cl->buffer_color);
	if (cl->buffer_depth) clReleaseMemObject(cl->buffer_depth);
	if (cl->buffer_temp) clReleaseMemObject(cl->buffer_temp);
	if (cl->kernelUpdateParticles) clReleaseKernel(cl->kernelUpdateParticles);
	if (cl->kernelRenderParticles) clReleaseKernel(cl->kernelRenderParticles);
	if (cl->kernelBlurFire) clReleaseKernel(cl->kernelBlurFire);
	if (cl->program) clReleaseProgram(cl->program);
	if (cl->queue) clReleaseCommandQueue(cl->queue);
	if (cl->context) clReleaseContext(cl->context);
}