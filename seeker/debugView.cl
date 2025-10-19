// Usage: Render 3D geometry -> Pass buffers to this kernel -> Composites radar cones over scene
// Desaturates background to make cones stand out, overwrites pixels where cone is closer

__kernel void composite_cones(
    __global float* imageBufferColor,      // RGB input/output, 3 floats per pixel
    __global float* imageBufferDepth,      // Depth input/output, 1 float per pixel
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
    
    // Desaturate background (convert to grayscale, keep 40% saturation)
    float r = imageBufferColor[pixelIdx * 3 + 0];
    float g = imageBufferColor[pixelIdx * 3 + 1];
    float b = imageBufferColor[pixelIdx * 3 + 2];
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    imageBufferColor[pixelIdx * 3 + 0] = gray * 0.6f + r * 0.4f;
    imageBufferColor[pixelIdx * 3 + 1] = gray * 0.6f + g * 0.4f;
    imageBufferColor[pixelIdx * 3 + 2] = gray * 0.6f + b * 0.4f;
    
    // Read existing depth
    float existingDepth = imageBufferDepth[pixelIdx];
    
    // NDC coords
    float ndcX = (2.0f * x / (float)imageWidth - 1.0f);
    float ndcY = (1.0f - 2.0f * y / (float)imageHeight);
    float aspectRatio = (float)imageWidth / (float)imageHeight;
    
    // Camera basis
    float camLen = sqrt(cameraDirX * cameraDirX + cameraDirY * cameraDirY + cameraDirZ * cameraDirZ);
    float cdx = cameraDirX / camLen;
    float cdy = cameraDirY / camLen;
    float cdz = cameraDirZ / camLen;
    
    float upx = 0.0f, upy = 1.0f, upz = 0.0f;
    if (fabs(cdy) > 0.99f) { upx = 1.0f; upy = 0.0f; upz = 0.0f; }
    
    // Right = cross(camDir, up)
    float rx = cdy * upz - cdz * upy;
    float ry = cdz * upx - cdx * upz;
    float rz = cdx * upy - cdy * upx;
    float rlen = sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rlen; ry /= rlen; rz /= rlen;
    
    // Up = cross(right, camDir)
    upx = ry * cdz - rz * cdy;
    upy = rz * cdx - rx * cdz;
    upz = rx * cdy - ry * cdx;
    
    // Ray direction
    float tanFov = tan(cameraFOV * 0.5f * M_PI_F / 180.0f);
    float rdx = cdx + rx * ndcX * aspectRatio * tanFov + upx * ndcY * tanFov;
    float rdy = cdy + ry * ndcX * aspectRatio * tanFov + upy * ndcY * tanFov;
    float rdz = cdz + rz * ndcX * aspectRatio * tanFov + upz * ndcY * tanFov;
    float rdLen = sqrt(rdx * rdx + rdy * rdy + rdz * rdz);
    rdx /= rdLen; rdy /= rdLen; rdz /= rdLen;
    
    // Find closest cone intersection
    float closestT = existingDepth;
    int hitCone = -1;
    
    for (int i = 0; i < numberOfCones; i++) {
        float cox = coneOriginX[i];
        float coy = coneOriginY[i];
        float coz = coneOriginZ[i];
        float cvx = coneDirX[i];
        float cvy = coneDirY[i];
        float cvz = coneDirZ[i];
        
        float cvLen = sqrt(cvx * cvx + cvy * cvy + cvz * cvz);
        cvx /= cvLen; cvy /= cvLen; cvz /= cvLen;
        
        float coneAngle = 0.785398f; // 45 deg
        float cosA = cos(coneAngle);
        float cos2A = cosA * cosA;
        float coneHeight = 2.0f;
        
        float dx = cameraPosX - cox;
        float dy = cameraPosY - coy;
        float dz = cameraPosZ - coz;
        
        float adp = rdx * cvx + rdy * cvy + rdz * cvz;
        float cop = dx * cvx + dy * cvy + dz * cvz;
        float dop = dx * rdx + dy * rdy + dz * rdz;
        
        float a = adp * adp - cos2A;
        float b = 2.0f * (adp * cop - dop * cos2A);
        float c = cop * cop - (dx * dx + dy * dy + dz * dz) * cos2A;
        
        float disc = b * b - 4.0f * a * c;
        
        if (disc >= 0.0f && fabs(a) > 1e-6f) {
            float sqrtD = sqrt(disc);
            float t1 = (-b - sqrtD) / (2.0f * a);
            float t2 = (-b + sqrtD) / (2.0f * a);
            float t = (t1 > 0.0001f) ? t1 : t2;
            
            if (t > 0.0001f && t < closestT) {
                float hx = cameraPosX + rdx * t;
                float hy = cameraPosY + rdy * t;
                float hz = cameraPosZ + rdz * t;
                float hDist = (hx - cox) * cvx + (hy - coy) * cvy + (hz - coz) * cvz;
                
                if (hDist >= 0.0f && hDist <= coneHeight) {
                    closestT = t;
                    hitCone = i;
                }
            }
        }
    }
    
    // Only overwrite if cone is closer
    if (hitCone >= 0) {
        float hue = (float)hitCone / (float)numberOfCones;
        imageBufferColor[pixelIdx * 3 + 0] = 0.2f + hue * 0.8f;
        imageBufferColor[pixelIdx * 3 + 1] = 0.8f;
        imageBufferColor[pixelIdx * 3 + 2] = 0.3f;
        imageBufferDepth[pixelIdx] = closestT;
    }
}
