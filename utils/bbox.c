#include "bbox.h"

void updateBBox(float x, float y, float z, float minBB[3], float maxBB[3]) {
	if (x < minBB[0]) minBB[0] = x;
	if (y < minBB[1]) minBB[1] = y;
	if (z < minBB[2]) minBB[2] = z;
	if (x > maxBB[0]) maxBB[0] = x;
	if (y > maxBB[1]) maxBB[1] = y;
	if (z > maxBB[2]) maxBB[2] = z;
}
