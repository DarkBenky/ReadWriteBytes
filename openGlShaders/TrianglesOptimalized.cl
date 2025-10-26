// === CONSTANTS ===
#define TILE_SIZE 16
#define MAX_TRIANGLES_PER_TILE 256
#define FMAX 1e9f

// === KERNEL 1: Project triangles and compute bounding boxes ===
__kernel void calculateVertexCoordinate(
    // Input buffers - using float4 for better memory alignment
    __global const float4* vertices,      // 3 per triangle (v0, v1, v2)
    __global const float4* normals,
    const float16 cameraMatrix,           // Precomputed camera transform
    const float2 fovScale,                // fovScale.x = 1.0f / fov, fovScale.y = fov
    const int screenWidth,
    const int screenHeight,
    const int numTriangles,
    // Output buffers
    __global float4* projectedVerts,      // 3 per triangle (x,y,z)
    __global float4* bboxes,              // minX, maxX, minY, maxY
    __global int* validTriangles
) {
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;

    // Initialize as invalid
    validTriangles[triangleId] = 0;

    // Load triangle vertices
    float4 v0 = vertices[triangleId * 3];
    float4 v1 = vertices[triangleId * 3 + 1];
    float4 v2 = vertices[triangleId * 3 + 2];
    float4 normal = normals[triangleId];

    // Extract camera parameters from matrix
    float3 camPos = cameraMatrix.s012;
    float3 camRight = cameraMatrix.s345;
    float3 camUp = cameraMatrix.s678;
    float3 camForward = cameraMatrix.s9ab;

    // Fast backface culling
    float3 center = (v0.xyz + v1.xyz + v2.xyz) * (1.0f / 3.0f);
    float3 viewDir = normalize(camPos - center);
    if (dot(normal.xyz, viewDir) <= 0.0f) {
        return;
    }

    // Project vertices to screen space
    float3 rel0 = v0.xyz - camPos;
    float3 rel1 = v1.xyz - camPos;
    float3 rel2 = v2.xyz - camPos;

    float depth0 = dot(rel0, camForward);
    float depth1 = dot(rel1, camForward);
    float depth2 = dot(rel2, camForward);

    // Cull triangles behind camera
    if (depth0 <= 0.01f || depth1 <= 0.01f || depth2 <= 0.01f) {
        return;
    }

    // Perspective projection
    float invDepth0 = 1.0f / (depth0 * fovScale.y);
    float invDepth1 = 1.0f / (depth1 * fovScale.y);
    float invDepth2 = 1.0f / (depth2 * fovScale.y);

    float2 screen0, screen1, screen2;
    
    screen0.x = (dot(rel0, camRight) * invDepth0) * screenWidth * 0.5f + screenWidth * 0.5f;
    screen0.y = (-dot(rel0, camUp) * invDepth0) * screenHeight * 0.5f + screenHeight * 0.5f;
    
    screen1.x = (dot(rel1, camRight) * invDepth1) * screenWidth * 0.5f + screenWidth * 0.5f;
    screen1.y = (-dot(rel1, camUp) * invDepth1) * screenHeight * 0.5f + screenHeight * 0.5f;
    
    screen2.x = (dot(rel2, camRight) * invDepth2) * screenWidth * 0.5f + screenWidth * 0.5f;
    screen2.y = (-dot(rel2, camUp) * invDepth2) * screenHeight * 0.5f + screenHeight * 0.5f;

    // Compute bounding box
    float minX = min(min(screen0.x, screen1.x), screen2.x);
    float maxX = max(max(screen0.x, screen1.x), screen2.x);
    float minY = min(min(screen0.y, screen1.y), screen2.y);
    float maxY = max(max(screen0.y, screen1.y), screen2.y);

    // Check for degenerate triangles
    float area = fabs((screen1.x - screen0.x) * (screen2.y - screen0.y) - 
                     (screen2.x - screen0.x) * (screen1.y - screen0.y)) * 0.5f;
    if (area < 0.5f) {
        return;
    }

    // Store results
    projectedVerts[triangleId * 3] = (float4)(screen0, depth0, 0.0f);
    projectedVerts[triangleId * 3 + 1] = (float4)(screen1, depth1, 0.0f);
    projectedVerts[triangleId * 3 + 2] = (float4)(screen2, depth2, 0.0f);
    
    // Clamp bbox to screen bounds and store
    bboxes[triangleId] = (float4)(
        max(0.0f, min((float)screenWidth, minX)),
        max(0.0f, min((float)screenWidth, maxX)),
        max(0.0f, min((float)screenHeight, minY)),
        max(0.0f, min((float)screenHeight, maxY))
    );
    
    validTriangles[triangleId] = 1;
}

// === KERNEL 2: Bin triangles into tiles ===
__kernel void binTriangles(
    __global const float4* bboxes,
    __global const int* validTriangles,
    __global int* triangleBins,
    __global atomic_int* binCounts,
    const int screenWidth,
    const int screenHeight,
    const int numTriangles
) {
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles || validTriangles[triangleId] == 0) return;
    
    float4 bbox = bboxes[triangleId];
    float minX = bbox.x, maxX = bbox.y;
    float minY = bbox.z, maxY = bbox.w;
    
    // Calculate tile ranges
    int tileMinX = (int)(minX / TILE_SIZE);
    int tileMaxX = (int)(maxX / TILE_SIZE);
    int tileMinY = (int)(minY / TILE_SIZE);
    int tileMaxY = (int)(maxY / TILE_SIZE);
    
    int tilesX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
    
    // Add triangle to all overlapping tiles
    for (int ty = tileMinY; ty <= tileMaxY; ty++) {
        for (int tx = tileMinX; tx <= tileMaxX; tx++) {
            int binIndex = ty * tilesX + tx;
            int index = atomic_fetch_add(&binCounts[binIndex], 1);
            if (index < MAX_TRIANGLES_PER_TILE) {
                triangleBins[binIndex * MAX_TRIANGLES_PER_TILE + index] = triangleId;
            }
        }
    }
}

// === KERNEL 3: Tile-based rasterization ===
__kernel void ShadePixels_Tiled(
    __global const float4* projectedVerts,
    __global const int* triangleBins,
    __global const int* binCounts,
    __global float4* ScreenColors,
    __global float* ScreenDistances,
    __global float4* ScreenNormals,
    const int screenWidth,
    const int screenHeight,
    const int numTriangles,
    __global const float4* TriangleColors,
    __global const float* roughness,
    __global const float* metallic,
    __global const float* emission,
    __global float* ScreenMaterialRoughness,
    __global float* ScreenMaterialMetallic,
    __global float* ScreenMaterialEmission,
    __global const float4* normals,
    __local int* localTriangles,
    __local float2* localVertices,
    __local float* localDepths
) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    int localId = get_local_id(1) * get_local_size(0) + get_local_id(0);
    int localSize = get_local_size(0) * get_local_size(1);
    
    if (px >= screenWidth || py >= screenHeight) return;
    
    const float cx = (float)px + 0.5f;
    const float cy = (float)py + 0.5f;
    const int pixelIdx = py * screenWidth + px;
    
    // Initialize with existing depth or far plane
    float bestDepth = ScreenDistances[pixelIdx] > 0.0f ? ScreenDistances[pixelIdx] : FMAX;
    int bestTri = -1;
    float3 bestBary = (float3)(0.0f);
    
    // Determine tile and load triangle list
    int tileX = px / TILE_SIZE;
    int tileY = py / TILE_SIZE;
    int tilesX = (screenWidth + TILE_SIZE - 1) / TILE_SIZE;
    int binIndex = tileY * tilesX + tileX;
    
    int triangleCount = min(binCounts[binIndex], MAX_TRIANGLES_PER_TILE);
    __global const int* binTriangles = &triangleBins[binIndex * MAX_TRIANGLES_PER_TILE];
    
    // Load triangles for this tile into local memory in parallel
    for (int i = localId; i < triangleCount; i += localSize) {
        int t = binTriangles[i];
        localTriangles[i] = t;
        
        // Pre-load vertex data for faster access
        int base = t * 3;
        float4 v0 = projectedVerts[base];
        float4 v1 = projectedVerts[base + 1];
        float4 v2 = projectedVerts[base + 2];
        
        localVertices[i * 6] = v0.xy;
        localVertices[i * 6 + 1] = v1.xy;
        localVertices[i * 6 + 2] = v2.xy;
        localDepths[i * 3] = v0.z;
        localDepths[i * 3 + 1] = v1.z;
        localDepths[i * 3 + 2] = v2.z;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Test each triangle in the tile
    for (int i = 0; i < triangleCount; i++) {
        int t = localTriangles[i];
        
        // Early depth test using triangle's minimum depth
        float minTriDepth = min(min(localDepths[i * 3], localDepths[i * 3 + 1]), localDepths[i * 3 + 2]);
        if (minTriDepth >= bestDepth) continue;
        
        // Load vertices from local memory
        float2 p0 = localVertices[i * 6];
        float2 p1 = localVertices[i * 6 + 1];
        float2 p2 = localVertices[i * 6 + 2];
        
        // Barycentric coordinates test
        float2 v0 = p1 - p0;
        float2 v1 = p2 - p0;
        float2 v2 = (float2)(cx, cy) - p0;
        
        float denom = v0.x * v1.y - v1.x * v0.y;
        if (fabs(denom) < 1e-6f) continue;
        
        float invDenom = 1.0f / denom;
        float u = (v2.x * v1.y - v1.x * v2.y) * invDenom;
        float v = (v0.x * v2.y - v2.x * v0.y) * invDenom;
        
        if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
            // Inside triangle - compute depth
            float w = 1.0f - u - v;
            float depth = w * localDepths[i * 3] + u * localDepths[i * 3 + 1] + v * localDepths[i * 3 + 2];
            
            if (depth < bestDepth && depth > 0.01f) {
                bestDepth = depth;
                bestTri = t;
                bestBary = (float3)(w, u, v);
            }
        }
    }
    
    // Write results
    if (bestTri >= 0) {
        ScreenDistances[pixelIdx] = bestDepth;
        
        // Interpolate normal
        float4 n0 = normals[bestTri * 3];
        float4 n1 = normals[bestTri * 3 + 1];
        float4 n2 = normals[bestTri * 3 + 2];
        float3 normal = normalize(n0.xyz * bestBary.x + n1.xyz * bestBary.y + n2.xyz * bestBary.z);
        
        // Simple lighting
        float4 color = TriangleColors[bestTri];
        float light = max(0.65f, dot(normal, normalize((float3)(0.3f, 0.7f, 0.5f))));
        float3 finalColor = color.xyz * light + color.xyz * emission[bestTri];
        
        ScreenColors[pixelIdx] = (float4)(finalColor, 1.0f);
        ScreenNormals[pixelIdx] = (float4)(normal, 0.0f);
        ScreenMaterialRoughness[pixelIdx] = roughness[bestTri];
        ScreenMaterialMetallic[pixelIdx] = metallic[bestTri];
        ScreenMaterialEmission[pixelIdx] = emission[bestTri];
    }
}
