// TODO: Fix Projection of 3D points to screen coordinates

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
