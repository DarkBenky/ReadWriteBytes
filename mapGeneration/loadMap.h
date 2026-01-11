#ifndef TERRAIN_LOADER_H
#define TERRAIN_LOADER_H

#define HIGH_RES_TRIANGLE_COUNT 1305332
#define MID_RES_TRIANGLE_COUNT 329922
#define LOW_RES_TRIANGLE_COUNT 18106
#define CHUNK_COUNT 256

#include "../fireSim/fireSim.h"

#define LOD_HIGH_DISTANCE 20000.0f
#define LOD_MED_DISTANCE 35000.0f
#define LOD_LOW_DISTANCE 50000.0f

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
    struct Triangles terrainHigh;
    char filepath_med[512];
    struct Triangles terrainMed;
    char filepath_low[512];
    struct Triangles terrainLow;
    LODLevel current_lod;
    int is_loaded;
};

struct Map {
    struct MapTile *tiles;
    float posX;
    float posY;
    float posZ;
    float tileSizeX;
    float tileSizeY;
    float tileSizeZ;
    int tilesX;
    int tilesY;
    float mapSizeX;
    float mapSizeY;
    float mapSizeZ;
    char dir_high[512];
    char dir_med[512];
    char dir_low[512];
};

struct MapGPU {
    int numberOfTiles;
    float posX;
    float posY;
    float posZ;
    float tileSizeX;
    float tileSizeY;
    float tileSizeZ;
    int tilesX;
    int tilesY;
    float mapSizeX;
    float mapSizeY;
    float mapSizeZ;
    int chunkStartHigh[CHUNK_COUNT];
    int chunkStartMed[CHUNK_COUNT];
    int chunkStartLow[CHUNK_COUNT];

    float chunkHighTrianglesData[HIGH_RES_TRIANGLE_COUNT * 9];
    float chunkHighRoughnessData[HIGH_RES_TRIANGLE_COUNT];
    float chunkHighMetallicData[HIGH_RES_TRIANGLE_COUNT];
    float chunkHighEmissionData[HIGH_RES_TRIANGLE_COUNT];
    float chunkHighNormalsData[HIGH_RES_TRIANGLE_COUNT * 3];
    float chunkHighColorsData[HIGH_RES_TRIANGLE_COUNT * 3];

    float chunkMedTrianglesData[MID_RES_TRIANGLE_COUNT * 9];
    float chunkMedRoughnessData[MID_RES_TRIANGLE_COUNT];
    float chunkMedMetallicData[MID_RES_TRIANGLE_COUNT];
    float chunkMedEmissionData[MID_RES_TRIANGLE_COUNT];
    float chunkMedNormalsData[MID_RES_TRIANGLE_COUNT * 3];
    float chunkMedColorsData[MID_RES_TRIANGLE_COUNT * 3];

    float chunkLowTrianglesData[LOW_RES_TRIANGLE_COUNT * 9];
    float chunkLowRoughnessData[LOW_RES_TRIANGLE_COUNT];
    float chunkLowMetallicData[LOW_RES_TRIANGLE_COUNT];
    float chunkLowEmissionData[LOW_RES_TRIANGLE_COUNT];
    float chunkLowNormalsData[LOW_RES_TRIANGLE_COUNT * 3];
    float chunkLowColorsData[LOW_RES_TRIANGLE_COUNT * 3];

};

void init_terrain_map(char *dir_high, char *dir_med, char *dir_low, struct Map *map,
                      float scale, float translate[3], 
                      float rotXDeg, float rotYDeg, float rotZDeg, float posX, float posY, float posZ);
void loadCurrentMap(struct Map *map, struct Camera *camera, struct Triangles *sceneTriangles);
struct MapTile* get_tile(struct Map *map, int world_x, int world_y);
void free_map(struct Map *map);
void initMapGPU(struct MapGPU *mapGpu, struct Map *map);

#endif