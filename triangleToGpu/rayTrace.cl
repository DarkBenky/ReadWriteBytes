#define Capacity 1024

struct HitInfo {
	int hit;
	float t;
	float hitPoint[3];
	float hitNormal[3];
	float color[3];
	float roughness;
	float metallic;
	float emission;
	int volumeIdx;
	int triangleIdx;
};

struct Volume {
	float BBoxMin[3];
	float BBoxMax[3];
	int count;
	float v1[Capacity * 3];
	float v2[Capacity * 3];
	float v3[Capacity * 3];
	float Roughness[Capacity];
	float Metallic[Capacity];
	float Emission[Capacity];
	float normals[Capacity * 3];
	float colors[Capacity * 3]; // RGB colors for each triangle
};

struct Cluster {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Volume volumes[8]; // 8 volumes per cluster forming a 3D grid
};

struct Block {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Cluster clusters[8];
};

struct Region {
	float BBoxMin[3];
	float BBoxMax[3];
	struct Block blocks[8];
};