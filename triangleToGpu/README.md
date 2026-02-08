# Ray Tracing GPU Kernel Interface

This directory contains the C interface for launching the ray tracing kernel implemented in `rayTrace.cl`.

## Files

- `rayTrace.cl` - OpenCL kernel implementing path tracing with PBR materials
- `loadToGpu.c` - C implementation of scene management and kernel launching
- `loadToGpu.h` - Header file with function declarations and usage example
- `makefile` - Build configuration

## Key Functions

### Kernel Initialization
```c
bool initRayTraceKernel(struct Scene *scene);
```
Loads and compiles the ray tracing kernel from `rayTrace.cl`. Must be called once after initializing GPU buffers.

### Kernel Execution
```c
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
```

Launches the ray tracing kernel with the specified camera and lighting parameters.

**Parameters:**
- `scene` - Scene structure containing geometry and GPU buffers
- `cameraPos` - Camera position in world space [x, y, z]
- `cameraDir` - Camera direction vector (normalized) [x, y, z]
- `fov` - Field of view multiplier
- `screenWidth` - Output image width in pixels
- `screenHeight` - Output image height in pixels
- `sunDir` - Sun light direction (normalized) [x, y, z]
- `sunColor` - Sun color [R, G, B] (0.0 to 1.0)
- `sunIntensity` - Sun light intensity multiplier
- `maxBounces` - Maximum number of ray bounces for path tracing

## Building

```bash
make
```

Requirements:
- OpenCL development headers (`opencl-headers`, `ocl-icd-opencl-dev`)
- JPEG library (`libjpeg-dev`)
- Clang compiler

## Usage

See the usage example in `loadToGpu.h` for a complete workflow.

## Features

The ray tracing kernel implements:
- BVH-based hierarchical scene traversal (Region → Block → Cluster → Volume)
- Physically-Based Rendering (PBR) with metallic/roughness workflow
- Path tracing with multiple bounces
- Direct lighting with shadow rays
- Skybox environment mapping
- Cosine-weighted hemisphere sampling for diffuse materials
- GGX microfacet BRDF for specular materials
- Fresnel reflections
