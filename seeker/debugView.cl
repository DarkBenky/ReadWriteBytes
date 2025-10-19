__kernel void raytrace_cones(
    __global float* imageBufferColor,
    __global float* imageBufferDepth,
    int imageWidth,
    int imageHeight,
    float cameraPosX,
    float cameraPosY,
    float cameraPosZ,
    float cameraDirX,
    float cameraDirY,
    float cameraDirZ,
    float cameraFOV,
    __global float* coneOriginX,
    __global float* coneOriginY,
    __global float* coneOriginZ,
    __global float* coneDirX,
    __global float* coneDirY,
    __global float* coneDirZ,
    int numberOfCones,
    __global float* maxDistance
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= imageWidth || y >= imageHeight) return;
    
    int pixelIdx = y * imageWidth + x;
    
    // Calculate normalized device coordinates
    float ndcX = (2.0f * x / (float)imageWidth - 1.0f);
    float ndcY = (1.0f - 2.0f * y / (float)imageHeight);
    float aspectRatio = (float)imageWidth / (float)imageHeight;
    
    // Calculate camera basis vectors
    float camDirLen = sqrt(cameraDirX * cameraDirX + 
                           cameraDirY * cameraDirY + 
                           cameraDirZ * cameraDirZ);
    float camDirNormX = cameraDirX / camDirLen;
    float camDirNormY = cameraDirY / camDirLen;
    float camDirNormZ = cameraDirZ / camDirLen;
    
    // Create arbitrary up vector
    float upX = 0.0f, upY = 1.0f, upZ = 0.0f;
    if (fabs(camDirNormY) > 0.99f) {
        upX = 1.0f; upY = 0.0f; upZ = 0.0f;
    }
    
    // Right vector = normalize(cross(camDir, up))
    float rightX = camDirNormY * upZ - camDirNormZ * upY;
    float rightY = camDirNormZ * upX - camDirNormX * upZ;
    float rightZ = camDirNormX * upY - camDirNormY * upX;
    float rightLen = sqrt(rightX * rightX + rightY * rightY + rightZ * rightZ);
    rightX /= rightLen;
    rightY /= rightLen;
    rightZ /= rightLen;
    
    // Up vector = normalize(cross(right, camDir))
    upX = rightY * camDirNormZ - rightZ * camDirNormY;
    upY = rightZ * camDirNormX - rightX * camDirNormZ;
    upZ = rightX * camDirNormY - rightY * camDirNormX;
    
    // Calculate ray direction
    float tanFov = tan(cameraFOV * 0.5f * M_PI_F / 180.0f);
    float rayDirX = camDirNormX + rightX * ndcX * aspectRatio * tanFov + upX * ndcY * tanFov;
    float rayDirY = camDirNormY + rightY * ndcX * aspectRatio * tanFov + upY * ndcY * tanFov;
    float rayDirZ = camDirNormZ + rightZ * ndcX * aspectRatio * tanFov + upZ * ndcY * tanFov;
    
    // Normalize ray direction
    float rayDirLen = sqrt(rayDirX * rayDirX + rayDirY * rayDirY + rayDirZ * rayDirZ);
    rayDirX /= rayDirLen;
    rayDirY /= rayDirLen;
    rayDirZ /= rayDirLen;
    
    // Ray tracing
    float closestT = maxDistance[0];
    int hitCone = -1;
    
    // Test intersection with each cone
    for (int i = 0; i < numberOfCones; i++) {
        float coX = coneOriginX[i];
        float coY = coneOriginY[i];
        float coZ = coneOriginZ[i];
        float cdX = coneDirX[i];
        float cdY = coneDirY[i];
        float cdZ = coneDirZ[i];
        
        // Normalize cone direction
        float cdLen = sqrt(cdX * cdX + cdY * cdY + cdZ * cdZ);
        cdX /= cdLen;
        cdY /= cdLen;
        cdZ /= cdLen;
        
        // Cone parameters (45 degree half-angle, height = 2.0)
        float coneAngle = 0.785398f; // 45 degrees in radians
        float cosAngle = cos(coneAngle);
        float cos2Angle = cosAngle * cosAngle;
        float coneHeight = 2.0f;
        
        // Ray-cone intersection
        float deltaX = cameraPosX - coX;
        float deltaY = cameraPosY - coY;
        float deltaZ = cameraPosZ - coZ;
        
        float adp = rayDirX * cdX + rayDirY * cdY + rayDirZ * cdZ;
        float cop = deltaX * cdX + deltaY * cdY + deltaZ * cdZ;
        float dop = deltaX * rayDirX + deltaY * rayDirY + deltaZ * rayDirZ;
        
        float a = adp * adp - cos2Angle;
        float b = 2.0f * (adp * cop - dop * cos2Angle);
        float c = cop * cop - dot(deltaX, deltaY, deltaZ, deltaX, deltaY, deltaZ) * cos2Angle;
        
        float discriminant = b * b - 4.0f * a * c;
        
        if (discriminant >= 0.0f && fabs(a) > 1e-6f) {
            float sqrtDisc = sqrt(discriminant);
            float t1 = (-b - sqrtDisc) / (2.0f * a);
            float t2 = (-b + sqrtDisc) / (2.0f * a);
            
            float t = (t1 > 0.0001f) ? t1 : t2;
            
            if (t > 0.0001f && t < closestT) {
                // Check if hit point is within cone height
                float hitX = cameraPosX + rayDirX * t;
                float hitY = cameraPosY + rayDirY * t;
                float hitZ = cameraPosZ + rayDirZ * t;
                
                float hitDistAlongAxis = (hitX - coX) * cdX + 
                                        (hitY - coY) * cdY + 
                                        (hitZ - coZ) * cdZ;
                
                if (hitDistAlongAxis >= 0.0f && hitDistAlongAxis <= coneHeight) {
                    closestT = t;
                    hitCone = i;
                }
            }
        }
    }
    
    // Write output
    if (hitCone >= 0) {
        // Color based on cone index
        float hue = (float)hitCone / (float)numberOfCones;
        imageBufferColor[pixelIdx * 3 + 0] = hue;
        imageBufferColor[pixelIdx * 3 + 1] = 0.7f;
        imageBufferColor[pixelIdx * 3 + 2] = 0.9f;
        imageBufferDepth[pixelIdx] = closestT;
    } else {
        // Background
        imageBufferColor[pixelIdx * 3 + 0] = 0.1f;
        imageBufferColor[pixelIdx * 3 + 1] = 0.1f;
        imageBufferColor[pixelIdx * 3 + 2] = 0.15f;
        imageBufferDepth[pixelIdx] = maxDistance[0];
    }
}

// Helper function for dot product
inline float dot(float x1, float y1, float z1, float x2, float y2, float z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}
