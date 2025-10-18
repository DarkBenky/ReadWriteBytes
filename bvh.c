#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define FLT_MAX 3.402823466e+38F

struct Triangle {
	float v1[3];	   // Vertex 1
	float v2[3];	   // Vertex 2
	float v3[3];	   // Vertex 3
	float normal[3];   // Normal vector
	float color[3];	   // RGB color
	float Roughness;   // Material roughness
	float Metallic;	   // Material metallic
	float Emission;	   // Material emission
	int TriangleIndex; // Index of the triangle
};

struct BVHNode {
	float BoundingBox[6]; // minX, minY, minZ, maxX, maxY, maxZ
	int LeftChild;		  // Index of left child node
	int RightChild;		  // Index of right child node
	int TriangleIndex;
};

struct BVHLinear {
	struct BVHNode *Nodes;		// Array of BVH nodes
	struct Triangle *Triangles; // Array of triangles
	int NodeCapacity;			// Capacity of the nodes array
	int NodesCount;				// Number of nodes in the BVH
	int TrianglesCount;			// Number of triangles in the BVH
};

void InitializeBVH(struct BVHLinear *bvh, int initialTriangleCapacity) {
	bvh->NodeCapacity = initialTriangleCapacity * 2; // Need more nodes than triangles
	bvh->Nodes = malloc(sizeof(struct BVHNode) * bvh->NodeCapacity);
	bvh->Triangles = malloc(sizeof(struct Triangle) * initialTriangleCapacity);
	bvh->NodesCount = 0;
	bvh->TrianglesCount = 0;
}

void addNode(struct BVHLinear *bvh, struct BVHNode node) {
	if (bvh->NodesCount >= bvh->NodeCapacity) {
		bvh->NodeCapacity *= 2;
		bvh->Nodes = realloc(bvh->Nodes, sizeof(struct BVHNode) * bvh->NodeCapacity);
	}
	bvh->Nodes[bvh->NodesCount++] = node;
}

float calSAH(float bbox[6], int numTriangles) {
	float dx = bbox[3] - bbox[0];
	float dy = bbox[4] - bbox[1];
	float dz = bbox[5] - bbox[2];
	float surfaceArea = 2.0f * (dx * dy + dy * dz + dz * dx);
	return surfaceArea * numTriangles;
}

void computeBoundingBox(struct Triangle *triangles, int *indices, int count, float bbox[6]) {
	bbox[0] = bbox[1] = bbox[2] = FLT_MAX;
	bbox[3] = bbox[4] = bbox[5] = -FLT_MAX;

	for (int i = 0; i < count; i++) {
		struct Triangle *tri = &triangles[indices[i]];
		for (int j = 0; j < 3; j++) {
			if (tri->v1[j] < bbox[j]) bbox[j] = tri->v1[j];
			if (tri->v2[j] < bbox[j]) bbox[j] = tri->v2[j];
			if (tri->v3[j] < bbox[j]) bbox[j] = tri->v3[j];
			if (tri->v1[j] > bbox[j + 3]) bbox[j + 3] = tri->v1[j];
			if (tri->v2[j] > bbox[j + 3]) bbox[j + 3] = tri->v2[j];
			if (tri->v3[j] > bbox[j + 3]) bbox[j + 3] = tri->v3[j];
		}
	}
}

float getTriangleCentroid(struct Triangle *tri, int axis) {
	float centroid = (tri->v1[axis] + tri->v2[axis] + tri->v3[axis]) / 3.0f;
	return centroid;
}

int partition(struct Triangle *triangles, int *indices, int left, int right, int axis, float pivot) {
	int i = left;
	for (int j = left; j <= right; j++) {
		if (getTriangleCentroid(&triangles[indices[j]], axis) < pivot) {
			int temp = indices[i];
			indices[i] = indices[j];
			indices[j] = temp;
			i++;
		}
	}
	return i;
}

int buildBVHRecursive(struct BVHLinear *bvh, struct Triangle *triangles, int *indices, int count) {
	struct BVHNode node;
	node.LeftChild = -1;
	node.RightChild = -1;
	node.TriangleIndex = -1;

	// Compute bounding box for this node
	computeBoundingBox(triangles, indices, count, node.BoundingBox);

	// If only one triangle, make it a leaf
	if (count == 1) {
		node.TriangleIndex = indices[0];
		int nodeIndex = bvh->NodesCount;
		addNode(bvh, node);
		return nodeIndex;
	}

	// Find the best axis to split on (longest axis)
	float dx = node.BoundingBox[3] - node.BoundingBox[0];
	float dy = node.BoundingBox[4] - node.BoundingBox[1];
	float dz = node.BoundingBox[5] - node.BoundingBox[2];

	int splitAxis = 0;
	if (dy > dx && dy > dz)
		splitAxis = 1;
	else if (dz > dx && dz > dy)
		splitAxis = 2;

	// Split at the middle of the bounding box
	float splitPos = (node.BoundingBox[splitAxis] + node.BoundingBox[splitAxis + 3]) / 2.0f;

	// Partition triangles
	int mid = partition(triangles, indices, 0, count - 1, splitAxis, splitPos);

	// If partition failed (all triangles on one side), split in half
	if (mid == 0 || mid == count) {
		mid = count / 2;
	}

	int nodeIndex = bvh->NodesCount;
	addNode(bvh, node); // Add node first to reserve its index

	// Recursively build left and right children
	if (mid > 0) {
		bvh->Nodes[nodeIndex].LeftChild = buildBVHRecursive(bvh, triangles, indices, mid);
	}
	if (count - mid > 0) {
		bvh->Nodes[nodeIndex].RightChild = buildBVHRecursive(bvh, triangles, indices + mid, count - mid);
	}

	return nodeIndex;
}

void BuildBVH(struct Triangle *triangles, int triangleCount, struct BVHLinear *bvh) {
	InitializeBVH(bvh, triangleCount);

	// Copy triangles to bvh
	for (int i = 0; i < triangleCount; i++) {
		bvh->Triangles[bvh->TrianglesCount++] = triangles[i];
	}

	// Create array of triangle indices
	int *indices = malloc(sizeof(int) * triangleCount);
	for (int i = 0; i < triangleCount; i++) {
		indices[i] = i;
	}

	// Build BVH recursively using internal copy of triangles
	if (triangleCount > 0) {
		buildBVHRecursive(bvh, bvh->Triangles, indices, triangleCount);
	}

	free(indices);

	printf("BVH built with %d nodes for %d triangles\n", bvh->NodesCount, bvh->TrianglesCount);
}

void FreeBVH(struct BVHLinear *bvh) {
	if (bvh->Nodes) {
		free(bvh->Nodes);
		bvh->Nodes = NULL;
	}
	if (bvh->Triangles) {
		free(bvh->Triangles);
		bvh->Triangles = NULL;
	}
	bvh->NodesCount = 0;
	bvh->TrianglesCount = 0;
	bvh->NodeCapacity = 0;
}
