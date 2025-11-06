#ifndef TERRAIN_LOADER_H
#define TERRAIN_LOADER_H

#include "../fireSim/fireSim.h"

#define LOD_HIGH_DISTANCE 20000.0f
#define LOD_MED_DISTANCE 35000.0f
#define LOD_LOW_DISTANCE 50000.0f

change tringles on gpu

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

void init_terrain_map(char *dir_high, char *dir_med, char *dir_low, struct Map *map,
                      float scale, float translate[3], 
                      float rotXDeg, float rotYDeg, float rotZDeg, float posX, float posY, float posZ);
void loadCurrentMap(struct Map *map, struct Camera *camera, struct Triangles *sceneTriangles);
struct MapTile* get_tile(struct Map *map, int world_x, int world_y);
void free_map(struct Map *map);

#endif