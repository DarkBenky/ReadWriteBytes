#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include "../fireSim/fireSim.h"
#include "loadMap.h"

static float degreesToRadians(float degrees) {
	return degrees * M_PI / 180.0f;
}

static void updateBBox(float x, float y, float z, float minBB[3], float maxBB[3]) {
	if (x < minBB[0]) minBB[0] = x;
	if (y < minBB[1]) minBB[1] = y;
	if (z < minBB[2]) minBB[2] = z;
	if (x > maxBB[0]) maxBB[0] = x;
	if (y > maxBB[1]) maxBB[1] = y;
	if (z > maxBB[2]) maxBB[2] = z;
}

static void createRotationMatrix(float rx, float ry, float rz, float matrix[3][3]) {
	float cosX = cosf(rx), sinX = sinf(rx);
	float cosY = cosf(ry), sinY = sinf(ry);
	float cosZ = cosf(rz), sinZ = sinf(rz);

	// Combined rotation matrix (Z * Y * X)
	matrix[0][0] = cosY * cosZ;
	matrix[0][1] = cosY * sinZ;
	matrix[0][2] = -sinY;

	matrix[1][0] = sinX * sinY * cosZ - cosX * sinZ;
	matrix[1][1] = sinX * sinY * sinZ + cosX * cosZ;
	matrix[1][2] = sinX * cosY;

	matrix[2][0] = cosX * sinY * cosZ + sinX * sinZ;
	matrix[2][1] = cosX * sinY * sinZ - sinX * cosZ;
	matrix[2][2] = cosX * cosY;
}

static void rotateVector(float v[3], float matrix[3][3], float result[3]) {
	result[0] = matrix[0][0] * v[0] + matrix[0][1] * v[1] + matrix[0][2] * v[2];
	result[1] = matrix[1][0] * v[0] + matrix[1][1] * v[1] + matrix[1][2] * v[2];
	result[2] = matrix[2][0] * v[0] + matrix[2][1] * v[1] + matrix[2][2] * v[2];
}

// Calculate distance from camera to tile center
static float calculate_tile_distance(struct Camera *camera, struct Map *map, int tile_x, int tile_y) {
	// Calculate tile center in world space accounting for map position
	float tile_center_x = map->posX + tile_x * map->tileSizeX + map->tileSizeX * 0.5f;
	float tile_center_z = map->posZ + tile_y * map->tileSizeZ + map->tileSizeZ * 0.5f;

	float dx = camera->ray.origin[0] - tile_center_x;
	float dz = camera->ray.origin[2] - tile_center_z;

	return sqrtf(dx * dx + dz * dz);
}

// Check if tile is behind the camera (backface culling)
static int is_tile_behind_camera(struct Camera *camera, struct Map *map, int tile_x, int tile_y) {
	// Calculate tile center in world space
	float tile_center_x = map->posX + tile_x * map->tileSizeX + map->tileSizeX * 0.5f;
	float tile_center_z = map->posZ + tile_y * map->tileSizeZ + map->tileSizeZ * 0.5f;

	// Vector from camera to tile center
	float to_tile_x = tile_center_x - camera->ray.origin[0];
	float to_tile_z = tile_center_z - camera->ray.origin[2];

	// Dot product with camera forward direction
	// If negative, tile is behind camera
	float dot = to_tile_x * camera->ray.direction[0] + to_tile_z * camera->ray.direction[2];

	return dot < 0.0f;
}

static LODLevel get_required_lod(float distance) {
	if (distance <= LOD_HIGH_DISTANCE) {
		return LOD_HIGH;
	} else if (distance <= LOD_MED_DISTANCE) {
		return LOD_MEDIUM;
	} else if (distance <= LOD_LOW_DISTANCE) {
		return LOD_LOW;
	}
	return LOD_NONE;
}

static int loadModelWithRotation(const char *filename, struct Triangles *triangles,
								 float scale, float translate[3],
								 float rotationXDeg, float rotationYDeg, float rotationZDeg, float pos[3], float *sizeX, float *sizeY, float *sizeZ) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		printf("Error: Could not open file %s for reading\n", filename);
		return 0;
	}

	uint32_t fileSize, triangleStructSize;
	fread(&fileSize, sizeof(uint32_t), 1, file);
	fread(&triangleStructSize, sizeof(uint32_t), 1, file);

	int triangleCount = (fileSize - 8) / triangleStructSize;

	// Check if we have space
	int space_available = NUMBER_OF_TRIANGLES - triangles->count;

	if (triangleCount > space_available) {
		printf("Warning: File contains %d triangles, but only %d slots available. Loading %d triangles.\n",
			   triangleCount, space_available, space_available);
		triangleCount = space_available;
	}

	if (triangleCount == 0) {
		fclose(file);
		return 0;
	}

	int start_idx = triangles->count;
	triangles->count += triangleCount;

	float minBB[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
	float maxBB[3] = {FLT_MIN, FLT_MIN, FLT_MIN};

	// Convert degrees to radians
	float rotationXRad = degreesToRadians(rotationXDeg);
	float rotationYRad = degreesToRadians(rotationYDeg);
	float rotationZRad = degreesToRadians(rotationZDeg);

	// Create rotation matrix
	float rotMatrix[3][3];
	createRotationMatrix(rotationXRad, rotationYRad, rotationZRad, rotMatrix);

	for (int i = 0; i < triangleCount; i++) {
		int idx = (start_idx + i) * 3;
		int mat_idx = start_idx + i;

		// Read vertices
		float v1[3], v2[3], v3[3], normal[3];
		fread(v1, sizeof(float), 3, file);
		fread(v2, sizeof(float), 3, file);
		fread(v3, sizeof(float), 3, file);

		// Apply transformations: Scale -> Rotate -> Translate

		// Vertex 1
		v1[0] *= scale;
		v1[1] *= scale;
		v1[2] *= scale;
		float v1_rotated[3];
		rotateVector(v1, rotMatrix, v1_rotated);
		triangles->v1[idx] = v1_rotated[0] + translate[0];
		triangles->v1[idx + 1] = v1_rotated[1] + translate[1];
		triangles->v1[idx + 2] = v1_rotated[2] + translate[2];

		// Vertex 2
		v2[0] *= scale;
		v2[1] *= scale;
		v2[2] *= scale;
		float v2_rotated[3];
		rotateVector(v2, rotMatrix, v2_rotated);
		triangles->v2[idx] = v2_rotated[0] + translate[0];
		triangles->v2[idx + 1] = v2_rotated[1] + translate[1];
		triangles->v2[idx + 2] = v2_rotated[2] + translate[2];

		// Vertex 3
		v3[0] *= scale;
		v3[1] *= scale;
		v3[2] *= scale;
		float v3_rotated[3];
		rotateVector(v3, rotMatrix, v3_rotated);
		triangles->v3[idx] = v3_rotated[0] + translate[0];
		triangles->v3[idx + 1] = v3_rotated[1] + translate[1];
		triangles->v3[idx + 2] = v3_rotated[2] + translate[2];

		// Update bounding box with transformed vertices
		updateBBox(triangles->v1[idx], triangles->v1[idx + 1], triangles->v1[idx + 2], minBB, maxBB);
		updateBBox(triangles->v2[idx], triangles->v2[idx + 1], triangles->v2[idx + 2], minBB, maxBB);
		updateBBox(triangles->v3[idx], triangles->v3[idx + 1], triangles->v3[idx + 2], minBB, maxBB);

		// Read and rotate normals (normals should only be rotated, not scaled or translated)
		fread(normal, sizeof(float), 3, file);
		float normal_rotated[3];
		rotateVector(normal, rotMatrix, normal_rotated);
		triangles->normals[idx] = normal_rotated[0];
		triangles->normals[idx + 1] = normal_rotated[1];
		triangles->normals[idx + 2] = normal_rotated[2];

		// Read material properties and colors (unchanged)
		fread(&triangles->Roughness[mat_idx], sizeof(float), 1, file);
		fread(&triangles->Metallic[mat_idx], sizeof(float), 1, file);
		fread(&triangles->Emission[mat_idx], sizeof(float), 1, file);
		fread(&triangles->colors[idx], sizeof(float), 3, file);

		// Skip triangle index
		uint32_t triangleIndex;
		fread(&triangleIndex, sizeof(uint32_t), 1, file);
	}

	fclose(file);

	// Calculate bounding box size
	*sizeX = maxBB[0] - minBB[0];
	*sizeY = maxBB[1] - minBB[1];
	*sizeZ = maxBB[2] - minBB[2];

	return triangleCount;
}

static int parse_chunk_coords(const char *filename, int *x, int *y) {
	// Expected format: terrain_chunk_x_N_y_M_.bin
	const char *x_ptr = strstr(filename, "_x_");
	const char *y_ptr = strstr(filename, "_y_");

	if (!x_ptr || !y_ptr) {
		return 0; // Not a valid chunk file
	}

	if (sscanf(x_ptr, "_x_%d_y_%d_", x, y) == 2) {
		return 1; // Success
	}

	return 0; // Failed to parse
}

void init_terrain_map(char *dir_high, char *dir_med, char *dir_low, struct Map *map,
					  float scale, float translate[3],
					  float rotXDeg, float rotYDeg, float rotZDeg, float posX, float posY, float posZ) {
	map->posX = posX;
	map->posY = posY;
	map->posZ = posZ;

	DIR *dir = opendir(dir_high);
	if (!dir) {
		printf("Failed to open map directory: %s\n", dir_high);
		return;
	}

	struct dirent *entry;

	// First pass: find dimensions from high-res directory
	int min_x = INT_MAX, min_y = INT_MAX;
	int max_x = INT_MIN, max_y = INT_MIN;
	int chunk_count = 0;

	while ((entry = readdir(dir)) != NULL) {
		if (strstr(entry->d_name, ".bin")) {
			int x, y;
			if (parse_chunk_coords(entry->d_name, &x, &y)) {
				if (x < min_x) min_x = x;
				if (x > max_x) max_x = x;
				if (y < min_y) min_y = y;
				if (y > max_y) max_y = y;
				chunk_count++;
			}
		}
	}

	if (chunk_count == 0) {
		printf("No terrain chunks found in %s\n", dir_high);
		closedir(dir);
		return;
	}

	// Store directory paths
	strncpy(map->dir_high, dir_high, sizeof(map->dir_high) - 1);
	strncpy(map->dir_med, dir_med, sizeof(map->dir_med) - 1);
	strncpy(map->dir_low, dir_low, sizeof(map->dir_low) - 1);

	map->tilesX = min_x;
	map->tilesY = min_y;

	// Allocate temporary tile on heap (it's too large for stack - ~54MB)
	struct MapTile *tempTile = (struct MapTile *)calloc(1, sizeof(struct MapTile));
	if (!tempTile) {
		printf("Failed to allocate memory for temporary tile\n");
		closedir(dir);
		return;
	}
	tempTile->terrainHigh.count = 0;

	float tempSizeX, tempSizeY, tempSizeZ;
	char filepath[512];
	snprintf(filepath, sizeof(filepath), "%s/terrain_chunk_x_%d_y_%d_.bin", dir_high, min_x, min_y);
	loadModelWithRotation(filepath, &tempTile->terrainHigh,
						  scale, translate,
						  rotXDeg, rotYDeg, rotZDeg, (float[]){posX, posY, posZ}, &tempSizeX, &tempSizeY, &tempSizeZ);

	map->tileSizeX = tempSizeX;
	map->tileSizeY = tempSizeY;
	map->tileSizeZ = tempSizeZ;

	// Free temporary tile
	free(tempTile);

	// Allocate tiles array
	int width = max_x - min_x + 1;
	int height = max_y - min_y + 1;
	map->tiles = (struct MapTile *)calloc(width * height, sizeof(struct MapTile));
	if (!map->tiles) {
		printf("Failed to allocate memory for map tiles\n");
		closedir(dir);
		return;
	}

	map->mapSizeX = width * tempSizeX;
	map->mapSizeY = tempSizeY;
	map->mapSizeZ = height * tempSizeZ;

	printf("Loading terrain map: %d x %d tiles (total: %d)\n", width, height, width * height);
	printf("Tile size: %.2f x %.2f x %.2f\n", tempSizeX, tempSizeY, tempSizeZ);

	// Load all tiles
	for (int tile_x = min_x; tile_x <= max_x; tile_x++) {
		for (int tile_y = min_y; tile_y <= max_y; tile_y++) {
			int idx = (tile_x - min_x) * height + (tile_y - min_y);
			struct MapTile *tile = &map->tiles[idx];

			tile->x = tile_x;
			tile->y = tile_y;
			tile->current_lod = LOD_NONE;
			tile->is_loaded = 0;

			// Calculate tile offset in grid space (relative to min tile)
			// Each tile occupies tempSizeX x tempSizeZ in the XZ plane
			float gridOffsetX = (tile_x - min_x) * tempSizeX;
			float gridOffsetY = 0.0f;
			float gridOffsetZ = (tile_y - min_y) * tempSizeZ;

			// Tile position = base translation + grid offset
			// Note: We don't rotate the grid offset, as the vertices inside each tile
			// will be rotated by loadModelWithRotation
			float tileTranslate[3] = {
				translate[0] + gridOffsetX,
				translate[1] + gridOffsetY,
				translate[2] + gridOffsetZ};

			// Build file paths
			snprintf(tile->filepath_high, sizeof(tile->filepath_high),
					 "%s/terrain_chunk_x_%d_y_%d_.bin", dir_high, tile_x, tile_y);
			snprintf(tile->filepath_med, sizeof(tile->filepath_med),
					 "%s/terrain_midres_chunk_x_%d_y_%d_.bin", dir_med, tile_x, tile_y);
			snprintf(tile->filepath_low, sizeof(tile->filepath_low),
					 "%s/terrain_lowres_chunk_x_%d_y_%d_.bin", dir_low, tile_x, tile_y);
			snprintf(tile->name, sizeof(tile->name), "chunk_%d_%d", tile_x, tile_y);

			// Initialize triangle buffers
			tile->terrainHigh.count = 0;
			tile->terrainMed.count = 0;
			tile->terrainLow.count = 0;

			// Load all LOD levels
			float sizeX, sizeY, sizeZ;

			// Load high resolution
			int loaded = loadModelWithRotation(tile->filepath_high, &tile->terrainHigh,
											   scale, tileTranslate,
											   rotXDeg, rotYDeg, rotZDeg,
											   (float[]){posX, posY, posZ},
											   &sizeX, &sizeY, &sizeZ);
			if (loaded > 0) {
				printf("  Loaded HIGH LOD for tile (%d, %d): %d triangles\n", tile_x, tile_y, loaded);
			}

			// Load medium resolution
			loaded = loadModelWithRotation(tile->filepath_med, &tile->terrainMed,
										   scale, tileTranslate,
										   rotXDeg, rotYDeg, rotZDeg,
										   (float[]){posX, posY, posZ},
										   &sizeX, &sizeY, &sizeZ);
			if (loaded > 0) {
				printf("  Loaded MED LOD for tile (%d, %d): %d triangles\n", tile_x, tile_y, loaded);
			}

			// Load low resolution
			loaded = loadModelWithRotation(tile->filepath_low, &tile->terrainLow,
										   scale, tileTranslate,
										   rotXDeg, rotYDeg, rotZDeg,
										   (float[]){posX, posY, posZ},
										   &sizeX, &sizeY, &sizeZ);
			if (loaded > 0) {
				printf("  Loaded LOW LOD for tile (%d, %d): %d triangles\n", tile_x, tile_y, loaded);
				tile->is_loaded = 1;
			}
		}
	}

	closedir(dir);
	printf("Terrain map initialization complete\n");
}

void loadCurrentMap(struct Map *map, struct Camera *camera, struct Triangles *sceneTriangles) {
	if (!map || !map->tiles || !camera || !sceneTriangles) {
		return;
	}

	// Reset scene triangles count
	sceneTriangles->count = 0;

	// Calculate map dimensions
	int min_x = map->tilesX;
	int min_y = map->tilesY;
	int width = (int)(map->mapSizeX / map->tileSizeX);
	int height = (int)(map->mapSizeZ / map->tileSizeZ);

	// Iterate through all tiles and load based on distance
	for (int tile_x = min_x; tile_x < min_x + width; tile_x++) {
		for (int tile_y = min_y; tile_y < min_y + height; tile_y++) {
			struct MapTile *tile = get_tile(map, tile_x, tile_y);
			if (!tile || !tile->is_loaded) {
				continue;
			}

			// Skip tiles behind the camera
			if (is_tile_behind_camera(camera, map, tile_x, tile_y)) {
				continue;
			}

			// Calculate distance from camera to tile
			float distance = calculate_tile_distance(camera, map, tile_x, tile_y);

			// Determine required LOD
			LODLevel required_lod = get_required_lod(distance);
			tile->current_lod = required_lod;

			// Select appropriate triangle data based on LOD
			struct Triangles *source = NULL;
			switch (required_lod) {
			case LOD_HIGH:
				source = &tile->terrainHigh;
				break;
			case LOD_MEDIUM:
				source = &tile->terrainMed;
				break;
			case LOD_LOW:
				source = &tile->terrainLow;
				break;
			case LOD_NONE:
			default:
				continue; // Skip this tile
			}

			// Check if we have space in scene buffer
			if (sceneTriangles->count + source->count > NUMBER_OF_TRIANGLES) {
				printf("Warning: Scene triangle buffer full (%d/%d), skipping tile [%d,%d] with %d triangles (LOD: %d)\n",
					   sceneTriangles->count, NUMBER_OF_TRIANGLES, tile_x, tile_y, source->count, required_lod);
				return;
			}

			// Copy triangle data to scene buffer
			int start_idx = sceneTriangles->count;
			for (int i = 0; i < source->count; i++) {
				int src_idx = i * 3;
				int dst_idx = (start_idx + i) * 3;

				// Copy vertices
				sceneTriangles->v1[dst_idx] = source->v1[src_idx];
				sceneTriangles->v1[dst_idx + 1] = source->v1[src_idx + 1];
				sceneTriangles->v1[dst_idx + 2] = source->v1[src_idx + 2];

				sceneTriangles->v2[dst_idx] = source->v2[src_idx];
				sceneTriangles->v2[dst_idx + 1] = source->v2[src_idx + 1];
				sceneTriangles->v2[dst_idx + 2] = source->v2[src_idx + 2];

				sceneTriangles->v3[dst_idx] = source->v3[src_idx];
				sceneTriangles->v3[dst_idx + 1] = source->v3[src_idx + 1];
				sceneTriangles->v3[dst_idx + 2] = source->v3[src_idx + 2];

				// Copy normals
				sceneTriangles->normals[dst_idx] = source->normals[src_idx];
				sceneTriangles->normals[dst_idx + 1] = source->normals[src_idx + 1];
				sceneTriangles->normals[dst_idx + 2] = source->normals[src_idx + 2];

				// Copy colors
				sceneTriangles->colors[dst_idx] = source->colors[src_idx];
				sceneTriangles->colors[dst_idx + 1] = source->colors[src_idx + 1];
				sceneTriangles->colors[dst_idx + 2] = source->colors[src_idx + 2];

				// Copy material properties
				int src_mat_idx = i;
				int dst_mat_idx = start_idx + i;
				sceneTriangles->Roughness[dst_mat_idx] = source->Roughness[src_mat_idx];
				sceneTriangles->Metallic[dst_mat_idx] = source->Metallic[src_mat_idx];
				sceneTriangles->Emission[dst_mat_idx] = source->Emission[src_mat_idx];
			}

			sceneTriangles->count += source->count;
		}
	}
}

struct MapTile *get_tile(struct Map *map, int world_x, int world_y) {
	if (!map || !map->tiles) {
		return NULL;
	}

	int min_x = map->tilesX;
	int min_y = map->tilesY;
	int width = (int)(map->mapSizeX / map->tileSizeX);
	int height = (int)(map->mapSizeZ / map->tileSizeZ);

	// Check bounds
	if (world_x < min_x || world_x >= min_x + width ||
		world_y < min_y || world_y >= min_y + height) {
		return NULL;
	}

	// Calculate index in tiles array
	int idx = (world_x - min_x) * height + (world_y - min_y);
	return &map->tiles[idx];
}

void free_map(struct Map *map) {
	if (!map) {
		return;
	}

	if (map->tiles) {
		free(map->tiles);
		map->tiles = NULL;
	}

	map->tilesX = 0;
	map->tilesY = 0;
	map->mapSizeX = 0;
	map->mapSizeY = 0;
	map->mapSizeZ = 0;
	map->tileSizeX = 0;
	map->tileSizeY = 0;
	map->tileSizeZ = 0;
}