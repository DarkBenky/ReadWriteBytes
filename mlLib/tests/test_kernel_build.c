#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    
    FILE *f = fopen("batch_kernels.cl", "r");
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *src = malloc(len + 1);
    fread(src, 1, len, f);
    src[len] = 0;
    fclose(f);
    
    printf("Building batch kernels...\n");
    cl_program prog = clCreateProgramWithSource(ctx, 1, (const char**)&src, NULL, NULL);
    cl_int err = clBuildProgram(prog, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    
    if (err != CL_SUCCESS) {
        printf("Build failed: %d\n", err);
        size_t log_size;
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("%s\n", log);
        return 1;
    }
    
    printf("Build successful!\n");
    return 0;
}
