#ifndef TERRAIN_LOADER_H
#define TERRAIN_LOADER_H

#include "../fireSim/fireSim.h"

#define LOD_HIGH_DISTANCE 5000.0f
#define LOD_MED_DISTANCE 15000.0f
#define LOD_LOW_DISTANCE 30000.0f

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
    struct MapTile **tiles;
    int mapSizeX;
    int mapSizeY;
    int min_x;
    int min_y;
    char dir_high[512];
    char dir_med[512];
    char dir_low[512];
};

void init_terrain_map(char *dir_high, char *dir_med, char *dir_low, struct Map *map,
                      float scale, float translate[3], 
                      float rotXDeg, float rotYDeg, float rotZDeg);
void loadCurrentMap(struct Map *map, struct Camera *camera, struct Triangles *sceneTriangles);
struct MapTile* get_tile(struct Map *map, int world_x, int world_y);
void free_map(struct Map *map);

#endif