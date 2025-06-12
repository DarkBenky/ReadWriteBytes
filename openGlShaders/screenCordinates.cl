// IMPROVED FLUID RENDERING PIPELINE:
// 1. Project particles to screen-space z-buffer (distances + velocities + basic opacity)
// 2. Apply bilateral Gaussian blur to distances (preserves depth discontinuities)
// 3. Calculate smooth normals from blurred distance field using gradients
// 4. Apply thickness estimation based on opacity accumulation
// 5. Enhanced fluid surface reconstruction with proper depth testing
// 6. Optional: Foam/bubble generation in high-velocity regions

__kernel void renderTriangles(
    __global const float* v1,
    __global const float* v2,
    __global const float* v3,
    __global const float* normals,
    __global float *ScreenDistances,
    __global float *ScreenNormals,      // 3 floats per pixel now!
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    const int numTriangles
) {
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;

    // Get triangle vertices
    float3 vertex1 = (float3)(v1[triangleId * 3], v1[triangleId * 3 + 1], v1[triangleId * 3 + 2]);
    float3 vertex2 = (float3)(v2[triangleId * 3], v2[triangleId * 3 + 1], v2[triangleId * 3 + 2]);
    float3 vertex3 = (float3)(v3[triangleId * 3], v3[triangleId * 3 + 1], v3[triangleId * 3 + 2]);
    
    // Get triangle normal
    float3 normal = (float3)(normals[triangleId * 3], normals[triangleId * 3 + 1], normals[triangleId * 3 + 2]);

    // Compute camera basis
    float3 forward = normalize(camDir);
    float3 camUp = (float3)(0.0f, 1.0f, 0.0f); // Assume Y-up
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);

    // Project vertices to screen space
    float3 screenPos1, screenPos2, screenPos3;
    
    // Project vertex1
    float3 rel1 = vertex1 - camPos;
    float depth1 = dot(rel1, forward);
    if (depth1 <= 0.001f) return;
    float fovScale1 = 1.0f / (depth1 * fov);
    screenPos1.x = dot(rel1, right) * fovScale1 * screenWidth * 0.5f + screenWidth * 0.5f;
    screenPos1.y = -dot(rel1, up) * fovScale1 * screenHeight * 0.5f + screenHeight * 0.5f;
    screenPos1.z = depth1;

    // Project vertex2
    float3 rel2 = vertex2 - camPos;
    float depth2 = dot(rel2, forward);
    if (depth2 <= 0.001f) return;
    float fovScale2 = 1.0f / (depth2 * fov);
    screenPos2.x = dot(rel2, right) * fovScale2 * screenWidth * 0.5f + screenWidth * 0.5f;
    screenPos2.y = -dot(rel2, up) * fovScale2 * screenHeight * 0.5f + screenHeight * 0.5f;
    screenPos2.z = depth2;

    // Project vertex3
    float3 rel3 = vertex3 - camPos;
    float depth3 = dot(rel3, forward);
    if (depth3 <= 0.001f) return;
    float fovScale3 = 1.0f / (depth3 * fov);
    screenPos3.x = dot(rel3, right) * fovScale3 * screenWidth * 0.5f + screenWidth * 0.5f;
    screenPos3.y = -dot(rel3, up) * fovScale3 * screenHeight * 0.5f + screenHeight * 0.5f;
    screenPos3.z = depth3;

    // Calculate bounding box
    int minX = max(0, (int)min(min(screenPos1.x, screenPos2.x), screenPos3.x));
    int maxX = min(screenWidth - 1, (int)max(max(screenPos1.x, screenPos2.x), screenPos3.x));
    int minY = max(0, (int)min(min(screenPos1.y, screenPos2.y), screenPos3.y));
    int maxY = min(screenHeight - 1, (int)max(max(screenPos1.y, screenPos2.y), screenPos3.y));

    // Rasterize triangle
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            // Barycentric coordinates
            float2 p = (float2)(x + 0.5f, y + 0.5f);
            float2 v0 = screenPos3.xy - screenPos1.xy;
            float2 v1 = screenPos2.xy - screenPos1.xy;
            float2 v2 = p - screenPos1.xy;

            float dot00 = dot(v0, v0);
            float dot01 = dot(v0, v1);
            float dot02 = dot(v0, v2);
            float dot11 = dot(v1, v1);
            float dot12 = dot(v1, v2);

            float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

            // Check if point is in triangle
            if (u >= 0 && v >= 0 && u + v <= 1) {
                float w = 1.0f - u - v;
                
                // Interpolate depth
                float depth = w * screenPos1.z + u * screenPos2.z + v * screenPos3.z;
                
                int pixelIndex = y * screenWidth + x;
                
                // Depth test
                if (ScreenDistances[pixelIndex] == 0.0f || depth < ScreenDistances[pixelIndex]) {
                    ScreenDistances[pixelIndex] = depth;
                    
                    // Store normal (3 floats per pixel)
                    int normalIndex = pixelIndex * 3;
                    ScreenNormals[normalIndex] = normal.x;
                    ScreenNormals[normalIndex + 1] = normal.y;
                    ScreenNormals[normalIndex + 2] = normal.z;
                }
            }
        }
    }
}


// 3. Calculate smooth normals from blurred distance field using gradients
__kernel void calculate_normals_from_blurred_distances(
    __global const float *BlurredDistances,
    __global float *ScreenNormals,
    const int screenWidth,
    const int screenHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;

    int index = y * screenWidth + x;
    int baseIndex = index * 3;

    if (BlurredDistances[index] <= 0.001f) { // Use a small epsilon
        ScreenNormals[baseIndex]     = 0.0f;
        ScreenNormals[baseIndex + 1] = 0.0f;
        ScreenNormals[baseIndex + 2] = 0.0f;
        return; // Skip normal calculation for this pixel
    }
    
    // Initialize normal components
    float3 normal = (float3)(0.0f, 0.0f, 0.0f);
    
    // Gradient calculation using central differences
    if (x > 0 && x < screenWidth - 1 && y > 0 && y < screenHeight - 1) {
        float left = BlurredDistances[index - 1];
        float right = BlurredDistances[index + 1];
        float up = BlurredDistances[index - screenWidth];
        float down = BlurredDistances[index + screenWidth];

        // Calculate gradients
        normal.x = left - right; // X gradient
        normal.y = up - down;     // Y gradient
        normal.z = 2.0f;          // Z component is constant for depth

        // Normalize the normal vector
        float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (length > 0.0f) {
            normal /= length;
        }
    } else {
        // Default to a flat normal if out of bounds
        normal = (float3)(0.0f, 0.0f, 1.0f);
    }

    ScreenNormals[baseIndex]     = normal.x;
    ScreenNormals[baseIndex + 1] = normal.y;
    ScreenNormals[baseIndex + 2] = normal.z;
}

// 2. Apply bilateral Gaussian blur to distances (preserves depth discontinuities)
__kernel void blur_distances(
    __global const float *ScreenDistances,
    __global const float *ScreenOpacities,
    __global float *BlurredDistances,
    __global float *BlurredOpacities,
    const int screenWidth,
    const int screenHeight,
    const int kernelSize,      // e.g., 2 or 3 for a 5x5 or 7x7 window
    const float sigmaRange,    // Sigma for depth/value differences, e.g., 5.0 or 10.0
    const float sigmaSpatial   // Sigma for spatial distance, e.g., 2.0 or 3.0
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= screenWidth || y >= screenHeight) return;

    int centerIndex = y * screenWidth + x;
    float centerDistance = ScreenDistances[centerIndex];
    // float centerOpacity = ScreenOpacities[centerIndex]; // Not used for bilateral weight on opacity

    float sumWeightedDistances = 0.0f;
    float sumWeightedOpacities = 0.0f;
    float totalWeightDistances = 0.0f;
    float totalWeightOpacities = 0.0f; // Opacity can use a simpler Gaussian weight

    // Iterate over the kernel window
    for (int j = -kernelSize; j <= kernelSize; j++) { // dy
        for (int i = -kernelSize; i <= kernelSize; i++) { // dx
            int nx = x + i;
            int ny = y + j;

            // Check bounds
            if (nx >= 0 && nx < screenWidth && ny >= 0 && ny < screenHeight) {
                int neighborIndex = ny * screenWidth + nx;
                float neighborDistance = ScreenDistances[neighborIndex];
                float neighborOpacity = ScreenOpacities[neighborIndex];

                if (neighborDistance <= 0.001f) {
                    continue;
                }

                // Spatial Gaussian weight (common for both distance and opacity)
                float spatialWeight = exp(-((float)(i * i + j * j)) / (2.0f * sigmaSpatial * sigmaSpatial));

                // Range/Value Gaussian weight for distances (bilateral part)
                float distanceDifference = centerDistance - neighborDistance;
                float rangeWeight = exp(-((distanceDifference * distanceDifference)) / (2.0f * sigmaRange * sigmaRange));
                
                float weightForDistance = spatialWeight * rangeWeight;
                sumWeightedDistances += neighborDistance * weightForDistance;
                totalWeightDistances += weightForDistance;

                // For opacity, we can do a simple Gaussian blur or also bilateral.
                // Here, let's do a simple Gaussian blur for opacity using only spatialWeight.
                // If you want bilateral on opacity too, calculate a rangeWeight for opacity.
                sumWeightedOpacities += neighborOpacity * spatialWeight;
                totalWeightOpacities += spatialWeight;
            }
        }
    }

    if (totalWeightDistances > 0.0f) {
        BlurredDistances[centerIndex] = sumWeightedDistances / totalWeightDistances;
    } else {
        BlurredDistances[centerIndex] = centerDistance; // Or ScreenDistances[centerIndex]
    }

    if (totalWeightOpacities > 0.0f) {
        BlurredOpacities[centerIndex] = sumWeightedOpacities / totalWeightOpacities;
    } else {
        BlurredOpacities[centerIndex] = ScreenOpacities[centerIndex]; // Or 0.0f
    }
}


// 1. Project particles to screen-space z-buffer (distances + velocities + basic opacity)
__kernel void project_points_to_screen(
    __global const float* points,
    __global const float* velocities,
    __global float *ScreenDistances,
    __global float *ScreenOpacities,
    __global float *ScreenVelocities,
    __global float *ScreenNormals,      // 3 floats per pixel now!
    const float3 camPos,
    const float3 camDir,
    const float3 camUp,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    const int numPoints,
    const int ParticleRadius
) {
    int i = get_global_id(0);
    if (i >= numPoints) return;

    // manually unpack
    float3 point = (float3)( points[3*i+0],
                             points[3*i+1],
                             points[3*i+2] );
    float3 vel   = (float3)( velocities[3*i+0],
                             velocities[3*i+1],
                             velocities[3*i+2] );

    // Compute camera basis
    float3 forward = normalize(camDir);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward); // Ensure orthogonality

    // Compute Screen space coordinates
    
    // Transform point relative to camera
    float3 relativePoint = point - camPos;
    
    // Project to camera space
    float dotProduct = dot(relativePoint, forward);
    if (dotProduct <= 0.001f) return; // Behind camera or too close
    
    float fovScale = 1.0f / (dotProduct * fov);
    
    float screenRight = dot(relativePoint, right) * fovScale;
    float screenUp = dot(relativePoint, up) * fovScale;
    
    // Convert to screen coordinates
    float halfWidth = screenWidth * 0.5f;
    float halfHeight = screenHeight * 0.5f;
    
    int screenX = (int)(screenRight * halfWidth + halfWidth);
    int screenY = (int)(-screenUp * halfHeight + halfHeight);
    
    // Bounds check
    if (screenX < 0 || screenX >= screenWidth || screenY < 0 || screenY >= screenHeight) return;
    
    // Calculate screen index
    int screenIndex = screenY * screenWidth + screenX;

    float distance = length(relativePoint);
    
    // calculate radius based on particle distance
    float particleRadiusBasedOnDistance = (float)ParticleRadius / dotProduct;
    int radiusInt = max(1, (int)particleRadiusBasedOnDistance); // Ensure minimum radius of 1
    int radiusSquared = radiusInt * radiusInt;

    for (int dy = -radiusInt; dy <= radiusInt; dy++) {
        int offsetY = screenY + dy;
        if (offsetY < 0 || offsetY >= screenHeight) continue;
        
        int dy2 = dy * dy;
        if (dy2 > radiusSquared) continue; // Safety check
        
        int maxDx = (int)sqrt((float)(radiusSquared - dy2));
        
        for (int dx = -maxDx; dx <= maxDx; dx++) {
            int offsetX = screenX + dx;
            if (offsetX < 0 || offsetX >= screenWidth) continue;
            
            int offsetIndex = offsetY * screenWidth + offsetX;
            
            // Calculate distance from center of particle
            int r2 = dx*dx + dy*dy;
            if (r2 > radiusSquared) continue; // Skip pixels outside circle
            
            // Calculate proper sphere depth offset
            float normalizedR2 = (float)r2 / (float)radiusSquared;
            float sphereDepth  = sqrt(max(0.0f,1.0f - normalizedR2));
            float depthOffset = sphereDepth * particleRadiusBasedOnDistance;
            
            // Surface depth is center depth minus z-offset (closer to camera)
            float surfaceDistance = distance - depthOffset;
            
            // Ensure surface distance doesn't go negative
            surfaceDistance = max(0.001f, surfaceDistance);
            
            // Update ScreenDistances with spherical surface depth
            if (ScreenDistances[offsetIndex] == 0 || surfaceDistance < ScreenDistances[offsetIndex]) {
                ScreenDistances[offsetIndex] = surfaceDistance;
                ScreenVelocities[offsetIndex] = length(vel);
                 // your normal in local sphere‐space:
                float nx = dx / (float)radiusInt;
                float ny = dy / (float)radiusInt;
                float nz = sphereDepth;
                float3 normal = normalize((float3)(nx, ny, nz));

                // pack 3 floats per pixel:
                int base = offsetIndex*3;
                ScreenNormals[base+0] = normal.x;
                ScreenNormals[base+1] = normal.y;
                ScreenNormals[base+2] = normal.z;
            }

            float maxFloat = 1000000.0f; // Arbitrary large value for opacity cap

            // Screen Opacity should accumulate
            if (ScreenOpacities[offsetIndex] < maxFloat) { 
                ScreenOpacities[offsetIndex] += 0.1f; // Increment opacity
            }

           
        }
    }
}
