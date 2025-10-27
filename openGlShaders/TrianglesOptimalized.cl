// ============================================================================
// OPTIMIZED OPENCL RENDERER - COMPLETE PIPELINE
// ============================================================================
//
// PIPELINE USAGE:
// 1. calculateVertexCoordinate  - Projects triangles, culls backfaces
// 2. TileCulling                - Bins triangles into screen tiles
// 3. ShadePixels                - Rasterizes only relevant triangles per tile
//
// HOST SETUP:
// - Create buffers for all inputs/outputs
// - Set TILE_SIZE (16 or 32 recommended)
// - numTilesX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE
// - numTilesY = (screenHeight + TILE_SIZE - 1) / TILE_SIZE
// - MAX_TRIS_PER_TILE = 512 (adjust based on scene complexity)
//
// BUFFER SIZES:
// - tileLists: numTilesX * numTilesY * MAX_TRIS_PER_TILE * sizeof(int)
// - tileListCounts: numTilesX * numTilesY * sizeof(int)
// ============================================================================

#define TILE_SIZE 16
#define MAX_TRIS_PER_TILE 512

// ============================================================================
// KERNEL 1: Project and Cull Triangles
// ============================================================================
__kernel void calculateVertexCoordinate(
    // Input buffers
    __global const float* v1,
    __global const float* v2,
    __global const float* v3,
    __global const float* normals,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    const int numTriangles,
    // Output buffers
    __global float* projectedVerts,      // 9 floats per triangle
    __global float* bboxes,              // 4 floats per triangle
    __global int* validTriangles         // 1 = valid, 0 = culled
) { 
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;

    validTriangles[triangleId] = 0;

    // Load vertices and normal
    float3 vertex1 = vload3(triangleId, v1);
    float3 vertex2 = vload3(triangleId, v2);
    float3 vertex3 = vload3(triangleId, v3);
    float3 faceNormal = normalize(vload3(triangleId, normals));

    // Camera basis
    float3 forward = normalize(camDir);
    float3 up = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, up));
    up = cross(right, forward);

    // Backface culling
    float3 center = (vertex1 + vertex2 + vertex3) * 0.33333f;
    float3 toCamera = camPos - center;
    if (dot(faceNormal, toCamera) <= 0.0f) return;

    // Check all vertices are in front of camera first (reduce divergence)
    float3 rel1 = vertex1 - camPos;
    float3 rel2 = vertex2 - camPos;
    float3 rel3 = vertex3 - camPos;
    
    float depth1 = dot(rel1, forward);
    float depth2 = dot(rel2, forward);
    float depth3 = dot(rel3, forward);
    
    if (depth1 <= 0.01f || depth2 <= 0.01f || depth3 <= 0.01f) return;

    // Project all three vertices
    float invFov = 1.0f / fov;
    float halfWidth = screenWidth * 0.5f;
    float halfHeight = screenHeight * 0.5f;
    
    float scale1 = invFov / depth1;
    float x1 = dot(rel1, right) * scale1 * halfWidth + halfWidth;
    float y1 = -dot(rel1, up) * scale1 * halfHeight + halfHeight;
    
    float scale2 = invFov / depth2;
    float x2 = dot(rel2, right) * scale2 * halfWidth + halfWidth;
    float y2 = -dot(rel2, up) * scale2 * halfHeight + halfHeight;
    
    float scale3 = invFov / depth3;
    float x3 = dot(rel3, right) * scale3 * halfWidth + halfWidth;
    float y3 = -dot(rel3, up) * scale3 * halfHeight + halfHeight;

    // Check for degenerate triangles
    float area = fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    if (area < 1.0f) return;

    // Compute bounding box
    float minX = min(min(x1, x2), x3);
    float maxX = max(max(x1, x2), x3);
    float minY = min(min(y1, y2), y3);
    float maxY = max(max(y1, y2), y3);

    // Store projected vertices
    int base = triangleId * 9;
    projectedVerts[base + 0] = x1;
    projectedVerts[base + 1] = y1;
    projectedVerts[base + 2] = depth1;
    projectedVerts[base + 3] = x2;
    projectedVerts[base + 4] = y2;
    projectedVerts[base + 5] = depth2;
    projectedVerts[base + 6] = x3;
    projectedVerts[base + 7] = y3;
    projectedVerts[base + 8] = depth3;

    // Store bounding box (clamped to screen)
    bboxes[triangleId * 4 + 0] = clamp(minX, 0.0f, (float)screenWidth);
    bboxes[triangleId * 4 + 1] = clamp(maxX, 0.0f, (float)screenWidth);
    bboxes[triangleId * 4 + 2] = clamp(minY, 0.0f, (float)screenHeight);
    bboxes[triangleId * 4 + 3] = clamp(maxY, 0.0f, (float)screenHeight);

    validTriangles[triangleId] = 1;
}

// ============================================================================
// KERNEL 2: Tile-Based Triangle Binning
// ============================================================================
__kernel void TileCulling(
    __global const float* bboxes,
    __global const int* validTriangles,
    __global int* tileLists,           // [numTiles][MAX_TRIS_PER_TILE]
    __global int* tileListCounts,      // [numTiles]
    const int screenWidth,
    const int screenHeight,
    const int numTriangles,
    const int numTilesX
) {
    int tileX = get_global_id(0);
    int tileY = get_global_id(1);
    int numTilesY = get_global_size(1);
    
    if (tileX >= numTilesX || tileY >= numTilesY) return;
    
    int tileIdx = tileY * numTilesX + tileX;
    
    // Tile boundaries in pixels
    float tileMinX = tileX * TILE_SIZE;
    float tileMaxX = min((tileX + 1) * TILE_SIZE, screenWidth);
    float tileMinY = tileY * TILE_SIZE;
    float tileMaxY = min((tileY + 1) * TILE_SIZE, screenHeight);
    
    // Bin triangles into this tile
    int count = 0;
    int baseOffset = tileIdx * MAX_TRIS_PER_TILE;
    
    for (int t = 0; t < numTriangles; t++) {
        if (validTriangles[t] == 0) continue;
        
        // Check if triangle bbox overlaps tile
        int bi = t * 4;
        float triMinX = bboxes[bi];
        float triMaxX = bboxes[bi + 1];
        float triMinY = bboxes[bi + 2];
        float triMaxY = bboxes[bi + 3];
        
        if (triMaxX >= tileMinX && triMinX <= tileMaxX &&
            triMaxY >= tileMinY && triMinY <= tileMaxY) {
            
            if (count < MAX_TRIS_PER_TILE) {
                tileLists[baseOffset + count] = t;
                count++;
            }
        }
    }
    
    tileListCounts[tileIdx] = count;
}

// ============================================================================
// KERNEL 3: Rasterize Pixels Using Tile Lists
// ============================================================================
__kernel void ShadePixels(
    __global const float* projectedVerts,
    __global const float* bboxes,
    __global const int* validTriangles,
    __global const int* tileLists,
    __global const int* tileListCounts,
    
    __global float* ScreenColors,
    __global float* ScreenDistances,
    __global float* ScreenNormals,
    __global float* ScreenMaterialRoughness,
    __global float* ScreenMaterialMetallic,
    __global float* ScreenMaterialEmission,

    const int screenWidth,
    const int screenHeight,
    const int numTilesX,

    __global const float* TriangleColors,
    __global const float* roughness,
    __global const float* metallic,
    __global const float* emission,
    __global const float* normals
) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= screenWidth || py >= screenHeight) return;

    const float cx = (float)px + 0.5f;
    const float cy = (float)py + 0.5f;
    const int idx = py * screenWidth + px;

    // Determine which tile this pixel belongs to
    int tileX = px / TILE_SIZE;
    int tileY = py / TILE_SIZE;
    int tileIdx = tileY * numTilesX + tileX;
    
    // Get triangle list for this tile
    int numTrisInTile = tileListCounts[tileIdx];
    int tileListBase = tileIdx * MAX_TRIS_PER_TILE;

    // Start with existing depth or infinity
    float bestDepth = ScreenDistances[idx] > 0.0f 
        ? ScreenDistances[idx] : INFINITY;
    int bestTri = -1;

    // Test only triangles in this tile
    for (int i = 0; i < numTrisInTile; i++) {
        int t = tileLists[tileListBase + i];
        
        // Early depth rejection
        int ov = t * 9;
        float minZ = min(min(projectedVerts[ov + 2], 
                            projectedVerts[ov + 5]), 
                            projectedVerts[ov + 8]);
        if (minZ >= bestDepth) continue;

        // Bbox test (still useful for sub-tile culling)
        int bi = t * 4;
        float minX = bboxes[bi];
        float maxX = bboxes[bi + 1];
        float minY = bboxes[bi + 2];
        float maxY = bboxes[bi + 3];
        if (cx < minX || cx > maxX || cy < minY || cy > maxY) continue;

        // Load projected vertices
        float2 p0 = (float2)(projectedVerts[ov], projectedVerts[ov + 1]);
        float2 p1 = (float2)(projectedVerts[ov + 3], projectedVerts[ov + 4]);
        float2 p2 = (float2)(projectedVerts[ov + 6], projectedVerts[ov + 7]);
        float z0 = projectedVerts[ov + 2];
        float z1 = projectedVerts[ov + 5];
        float z2 = projectedVerts[ov + 8];

        // Half-space edge function test
        float2 v0 = p1 - p0;
        float2 v1 = p2 - p0;
        float2 vp = (float2)(cx, cy) - p0;
        
        float denom = v0.x * v1.y - v1.x * v0.y;
        if (fabs(denom) < 1e-6f) continue;
        
        float invDenom = 1.0f / denom;
        float u = (vp.x * v1.y - v1.x * vp.y) * invDenom;
        float v = (v0.x * vp.y - vp.x * v0.y) * invDenom;
        
        if (u < 0.0f || v < 0.0f || u + v > 1.0f) continue;

        // Interpolate depth
        float w = 1.0f - u - v;
        float depth = w * z0 + u * z1 + v * z2;
        
        if (depth < bestDepth) {
            bestDepth = depth;
            bestTri = t;
        }
    }

    // Write result if we found a closer triangle
    if (bestTri >= 0 && (ScreenDistances[idx] == 0.0f || bestDepth < ScreenDistances[idx])) {
        ScreenDistances[idx] = bestDepth;

        // Load and store normal
        float3 N = normalize(vload3(bestTri, normals));
        vstore3(N, idx, ScreenNormals);

        // Load color and apply simple lighting
        float3 C = vload3(bestTri, TriangleColors);
        float3 lightDir = normalize((float3)(0.3f, 0.7f, 0.5f));
        float lighting = max(0.65f, dot(N, lightDir));
        float3 finalColor = clamp(C * lighting + C * emission[bestTri], 0.0f, 1.0f);
        vstore3(finalColor, idx, ScreenColors);

        // Store material properties
        ScreenMaterialRoughness[idx] = roughness[bestTri];
        ScreenMaterialMetallic[idx] = metallic[bestTri];
        ScreenMaterialEmission[idx] = emission[bestTri];
    }
}
