#include "planeSim.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void normalize(float v[3]) {
	float len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	if (len > 1e-6f) {
		v[0] /= len;
		v[1] /= len;
		v[2] /= len;
	}
}

static float dot(float a[3], float b[3]) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void cross(float a[3], float b[3], float result[3]) {
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = a[2] * b[0] - a[0] * b[2];
	result[2] = a[0] * b[1] - a[1] * b[0];
}

static float clamp(float value, float min, float max) {
	if (value < min) return min;
	if (value > max) return max;
	return value;
}

