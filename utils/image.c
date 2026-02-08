#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

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
	img->components = cinfo.output_components;

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

float *convertImageToFloat(struct RawImage *img) {
	if (!img) return NULL;

	float *data = malloc(img->width * img->height * 3 * sizeof(float));
	if (!data) return NULL;

	for (int i = 0; i < img->width * img->height * img->components; i += img->components) {
		int floatIdx = (i / img->components) * 3;
		data[floatIdx + 0] = (float)img->data[i + 0] / 255.0f;
		data[floatIdx + 1] = (float)img->data[i + 1] / 255.0f;
		data[floatIdx + 2] = (float)img->data[i + 2] / 255.0f;
	}

	return data;
}
