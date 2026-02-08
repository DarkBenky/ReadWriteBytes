#ifndef IMAGE_H
#define IMAGE_H

#include "../openGlShaders/gpuStruct.h"

struct RawImage *load_jpeg(const char *filename);
float *convertImageToFloat(struct RawImage *img);

#endif
