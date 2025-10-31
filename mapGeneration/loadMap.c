#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include "../fireSim/fireSim.h"

// LOD configuration
#define LOD_HIGH_DISTANCE 50.0f    // Load high-res within this distance
#define LOD_MED_DISTANCE 150.0f    // Load medium-res within this distance
#define LOD_LOW_DISTANCE 300.0f    // Load low-res within this distance

typedef enum {
    LOD_NONE = 0,
    LOD_LOW = 1,
    LOD_MEDIUM = 2,
    LOD_HIGH = 3
} LODLevel;

struct MapTile {
    int x;
    int y;
    char name[64];
    char filepath_high[512];
    char filepath_med[512];
    char filepath_low[512];
    struct Triangles *terrain;
    LODLevel current_lod;
    int is_loaded;
};

struct Map {
    struct MapTile **tiles;  // 2D array
    int mapSizeX;
    int mapSizeY;
    int min_x;
    int min_y;
    char dir_high[512];
    char dir_med[512];
    char dir_low[512];
};


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
static float calculate_tile_distance(struct Camera *camera, int tile_x, int tile_y, float tile_size) {
    // Assuming tiles are positioned at (tile_x * tile_size, 0, tile_y * tile_size)
    float tile_center_x = tile_x * tile_size + tile_size * 0.5f;
    float tile_center_z = tile_y * tile_size + tile_size * 0.5f;
    
    float dx = camera->ray.origin[0] - tile_center_x;
    float dz = camera->ray.origin[2] - tile_center_z;
    
    return sqrtf(dx * dx + dz * dz);
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
                                  float rotationXDeg, float rotationYDeg, float rotationZDeg,
                                  int max_triangles_to_add) {
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
    if (max_triangles_to_add > 0 && max_triangles_to_add < space_available) {
        space_available = max_triangles_to_add;
    }
    
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
    return triangleCount;
}

static int parse_chunk_coords(const char* filename, int* x, int* y) {
    // Expected format: terrain_chunk_x_N_y_M_.bin
    const char* x_ptr = strstr(filename, "_x_");
    const char* y_ptr = strstr(filename, "_y_");
    
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
                      float rotXDeg, float rotYDeg, float rotZDeg) {
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

    // Calculate map dimensions
    map->mapSizeX = max_x - min_x + 1;
    map->mapSizeY = max_y - min_y + 1;
    map->min_x = min_x;
    map->min_y = min_y;

    printf("Map dimensions: %dx%d (x: %d to %d, y: %d to %d)\n",
           map->mapSizeX, map->mapSizeY,
           min_x, max_x, min_y, max_y);
    printf("Found %d terrain chunks\n", chunk_count);
    printf("Global transform - Scale: %.2f, Translate: (%.2f, %.2f, %.2f), Rotation: (%.2f°, %.2f°, %.2f°)\n",
           scale, translate[0], translate[1], translate[2], rotXDeg, rotYDeg, rotZDeg);

    // Allocate 2D array
    map->tiles = calloc(map->mapSizeY, sizeof(struct MapTile*));
    if (!map->tiles) {
        printf("Failed to allocate memory for map tiles\n");
        closedir(dir);
        return;
    }
    
    for (int i = 0; i < map->mapSizeY; i++) {
        map->tiles[i] = calloc(map->mapSizeX, sizeof(struct MapTile));
        if (!map->tiles[i]) {
            printf("Failed to allocate memory for map tile row %d\n", i);
            closedir(dir);
            return;
        }
    }

    // Initialize tile metadata
    rewinddir(dir);
    
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, ".bin")) {
            int x, y;
            if (parse_chunk_coords(entry->d_name, &x, &y)) {
                // Convert to array indices
                int arr_x = x - min_x;
                int arr_y = y - min_y;
                
                struct MapTile *tile = &map->tiles[arr_y][arr_x];
                
                // Store tile info
                tile->x = x;
                tile->y = y;
                strncpy(tile->name, entry->d_name, sizeof(tile->name) - 1);
                snprintf(tile->filepath_high, sizeof(tile->filepath_high), "%s/%s", dir_high, entry->d_name);
                snprintf(tile->filepath_med, sizeof(tile->filepath_med), "%s/%s", dir_med, entry->d_name);
                snprintf(tile->filepath_low, sizeof(tile->filepath_low), "%s/%s", dir_low, entry->d_name);
                
                tile->terrain = NULL;
                tile->current_lod = LOD_NONE;
                tile->is_loaded = 0;
            }
        }
    }

    closedir(dir);
    printf("Terrain map initialized!\n");
}

static int load_tile(struct MapTile *tile, LODLevel lod, float scale, float translate[3],
                     float rotXDeg, float rotYDeg, float rotZDeg) {
    const char *filepath = NULL;
    
    switch (lod) {
        case LOD_HIGH:
            filepath = tile->filepath_high;
            break;
        case LOD_MEDIUM:
            filepath = tile->filepath_med;
            break;
        case LOD_LOW:
            filepath = tile->filepath_low;
            break;
        default:
            return 0;
    }
    
    // Allocate if needed
    if (!tile->terrain) {
        tile->terrain = malloc(sizeof(struct Triangles));
        if (!tile->terrain) {
            printf("Failed to allocate Triangles for tile (%d, %d)\n", tile->x, tile->y);
            return 0;
        }
        tile->terrain->count = 0;
    } else {
        // Reset count for reload
        tile->terrain->count = 0;
    }
    
    int loaded = loadModelWithRotation(filepath, tile->terrain, 
                                       scale, translate, rotXDeg, rotYDeg, rotZDeg, -1);
    
    if (loaded > 0) {
        tile->current_lod = lod;
        tile->is_loaded = 1;
        return 1;
    }
    
    return 0;
}

// Unload a tile
static void unload_tile(struct MapTile *tile) {
    if (tile->terrain) {
        tile->terrain->count = 0;
    }
    tile->current_lod = LOD_NONE;
    tile->is_loaded = 0;
}

void loadCurrentMap(struct Map *map, struct Camera *camera, struct Triangles *sceneTriangles) {
    if (!map || !map->tiles || !camera || !sceneTriangles) {
        return;
    }
    
    // Reset scene triangles
    sceneTriangles->count = 0;
    
    printf("\n=== Loading terrain based on camera position ===\n");
    printf("Camera position: (%.2f, %.2f, %.2f)\n", 
           camera->ray.origin[0], camera->ray.origin[1], camera->ray.origin[2]);
    
    int tiles_loaded = 0;
    int triangles_loaded = 0;
    
    // Iterate through all tiles
    for (int y = 0; y < map->mapSizeY; y++) {
        for (int x = 0; x < map->mapSizeX; x++) {
            struct MapTile *tile = &map->tiles[y][x];
            
            // Calculate distance from camera to tile
            float distance = calculate_tile_distance(camera, tile->x, tile->y, tile_size);
            
            // Determine required LOD
            LODLevel required_lod = get_required_lod(distance);
            
            // If tile should not be loaded, unload it if it is
            if (required_lod == LOD_NONE) {
                if (tile->is_loaded) {
                    unload_tile(tile);
                }
                continue;
            }
            
            // If tile needs different LOD, reload it
            if (!tile->is_loaded || tile->current_lod != required_lod) {
                if (tile->is_loaded) {
                    unload_tile(tile);
                }
                
                printf("Loading tile (%d, %d) at distance %.2f with LOD %d\n", 
                       tile->x, tile->y, distance, required_lod);
                
                load_tile(tile, required_lod, scale, translate, rotXDeg, rotYDeg, rotZDeg);
            }
            
            // Copy tile triangles to scene
            if (tile->is_loaded && tile->terrain && tile->terrain->count > 0) {
                int space_left = NUMBER_OF_TRIANGLES - sceneTriangles->count;
                int to_copy = tile->terrain->count;
                
                if (to_copy > space_left) {
                    printf("Warning: Not enough space for all triangles. Need %d, have %d\n", 
                           to_copy, space_left);
                    to_copy = space_left;
                }
                
                if (to_copy > 0) {
                    int dest_start = sceneTriangles->count;
                    
                    // Copy vertex data
                    memcpy(&sceneTriangles->v1[dest_start * 3], 
                           tile->terrain->v1, 
                           to_copy * 3 * sizeof(float));
                    memcpy(&sceneTriangles->v2[dest_start * 3], 
                           tile->terrain->v2, 
                           to_copy * 3 * sizeof(float));
                    memcpy(&sceneTriangles->v3[dest_start * 3], 
                           tile->terrain->v3, 
                           to_copy * 3 * sizeof(float));
                    
                    // Copy normals and colors
                    memcpy(&sceneTriangles->normals[dest_start * 3], 
                           tile->terrain->normals, 
                           to_copy * 3 * sizeof(float));
                    memcpy(&sceneTriangles->colors[dest_start * 3], 
                           tile->terrain->colors, 
                           to_copy * 3 * sizeof(float));
                    
                    // Copy material properties
                    memcpy(&sceneTriangles->Roughness[dest_start], 
                           tile->terrain->Roughness, 
                           to_copy * sizeof(float));
                    memcpy(&sceneTriangles->Metallic[dest_start], 
                           tile->terrain->Metallic, 
                           to_copy * sizeof(float));
                    memcpy(&sceneTriangles->Emission[dest_start], 
                           tile->terrain->Emission, 
                           to_copy * sizeof(float));
                    
                    sceneTriangles->count += to_copy;
                    triangles_loaded += to_copy;
                    tiles_loaded++;
                }
            }
        }
    }
    
    printf("=== Terrain loading complete ===\n");
    printf("Tiles loaded: %d\n", tiles_loaded);
    printf("Total triangles in scene: %d\n", sceneTriangles->count);
}

// Helper function to get a specific tile
struct MapTile* get_tile(struct Map *map, int world_x, int world_y) {
    int arr_x = world_x - map->min_x;
    int arr_y = world_y - map->min_y;
    
    if (arr_x < 0 || arr_x >= map->mapSizeX || 
        arr_y < 0 || arr_y >= map->mapSizeY) {
        return NULL;
    }
    
    return &map->tiles[arr_y][arr_x];
}

// Free the map
void free_map(struct Map *map) {
    if (!map || !map->tiles) return;
    
    for (int i = 0; i < map->mapSizeY; i++) {
        if (map->tiles[i]) {
            for (int j = 0; j < map->mapSizeX; j++) {
                if (map->tiles[i][j].terrain) {
                    free(map->tiles[i][j].terrain);
                }
            }
            free(map->tiles[i]);
        }
    }
    free(map->tiles);
    map->tiles = NULL;
}