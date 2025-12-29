#define FMAX 3.402823466e+38F
#define TILE_SIZE 16
#define MAX_TRIS_PER_TILE 512

#define HIGH_RES_TRIANGLE_COUNT 1305332
#define MID_RES_TRIANGLE_COUNT 329922
#define LOW_RES_TRIANGLE_COUNT 18106
#define CHUNK_COUNT 256
#define LOD_HIGH_DISTANCE 10000.0f
#define LOD_MED_DISTANCE 25000.0f
#define LOD_LOW_DISTANCE 50000.0f

struct MapGPU {
    int numberOfTiles;
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
    int chunkStartHigh[CHUNK_COUNT];
    int chunkStartMed[CHUNK_COUNT];
    int chunkStartLow[CHUNK_COUNT];

    float chunkHighTrianglesData[HIGH_RES_TRIANGLE_COUNT * 9];
    float chunkHighRoughnessData[HIGH_RES_TRIANGLE_COUNT];
    float chunkHighMetallicData[HIGH_RES_TRIANGLE_COUNT];
    float chunkHighEmissionData[HIGH_RES_TRIANGLE_COUNT];
    float chunkHighNormalsData[HIGH_RES_TRIANGLE_COUNT * 3];
    float chunkHighColorsData[HIGH_RES_TRIANGLE_COUNT * 3];

    float chunkMedTrianglesData[MID_RES_TRIANGLE_COUNT * 9];
    float chunkMedRoughnessData[MID_RES_TRIANGLE_COUNT];
    float chunkMedMetallicData[MID_RES_TRIANGLE_COUNT];
    float chunkMedEmissionData[MID_RES_TRIANGLE_COUNT];
    float chunkMedNormalsData[MID_RES_TRIANGLE_COUNT * 3];
    float chunkMedColorsData[MID_RES_TRIANGLE_COUNT * 3];

    float chunkLowTrianglesData[LOW_RES_TRIANGLE_COUNT * 9];
    float chunkLowRoughnessData[LOW_RES_TRIANGLE_COUNT];
    float chunkLowMetallicData[LOW_RES_TRIANGLE_COUNT];
    float chunkLowEmissionData[LOW_RES_TRIANGLE_COUNT];
    float chunkLowNormalsData[LOW_RES_TRIANGLE_COUNT * 3];
    float chunkLowColorsData[LOW_RES_TRIANGLE_COUNT * 3];

};

float fract(float x) {
    return x - floor(x);
}

void renderFont(
    const int fontSizeX,     // Total font texture width
    const int fontSizeY,     // Total font texture height  
    const int spriteSizeX,   // Individual character width (e.g., 8)
    const int spriteSizeY,   // Individual character height (e.g., 8)
    const char character,
    __global float *ScreenColors,
    __global const char *FontData,
    const int screenWidth,
    const int screenHeight,
    const int posX,          // Screen position to render character
    const int posY,          // Screen position to render character
    const float3 color       // Color for the text
) {
    int ascii_code = (int)character;
    int idx = ascii_code - 32; // ASCII offset for printable characters

    int cols = (fontSizeX / spriteSizeX);
    int rows = (fontSizeY / spriteSizeY);

    // Calculate character position in font texture
    int charCol = idx % cols;
    int charRow = idx / cols;
    
    int fontStartX = charCol * spriteSizeX;
    int fontStartY = charRow * spriteSizeY;

    // Render character pixel by pixel
    for (int charY = 0; charY < spriteSizeY; charY++) {
        for (int charX = 0; charX < spriteSizeX; charX++) {
            // Font texture coordinates
            int fontX = fontStartX + charX;
            int fontY = fontStartY + charY;
            
            // Screen coordinates
            int screenX = posX + charX;
            int screenY = posY + charY;

            // Bounds checking
            if (screenX < 0 || screenX >= screenWidth || 
                screenY < 0 || screenY >= screenHeight) {
                continue; // Skip out of bounds pixels
            }
            
            if (fontX >= fontSizeX || fontY >= fontSizeY) {
                continue; // Skip invalid font coordinates
            }

            // Read pixel from font data
            int fontPixelIndex = fontY * fontSizeX + fontX;
            char fontPixel = FontData[fontPixelIndex];
            
            // Only render if font pixel is "on" (1 = foreground, 0 = background)
            if (fontPixel == 0) {
                int screenPixelIndex = screenY * screenWidth + screenX;
                int colorIndex = screenPixelIndex * 3;

                // Set the color for this pixel
                ScreenColors[colorIndex]     = color.x; // R
                ScreenColors[colorIndex + 1] = color.y; // G  
                ScreenColors[colorIndex + 2] = color.z; // B
            }
        }
    }
}

__kernel void renderText(
    const int fontSizeX,     // Total font texture width
    const int fontSizeY,     // Total font texture height  
    const int spriteSizeX,   // Individual character width (e.g., 8)
    const int spriteSizeY,   // Individual character height (e.g., 8)
    __global float *ScreenColors,
    __global const char *FontData,
    const int screenWidth,
    const int screenHeight,
    __global const int *posX,          // Screen position to render character
    __global const int *posY  ,         // Screen position to render character
    __global const char *character,
    __global const uint *color, // Color for the text
    const int NumberOfCharacters
) {
    int globalId = get_global_id(0);
    
    if (globalId >= NumberOfCharacters) return; // Out of bounds check
    
    char currentChar = character[globalId];
    
    // Calculate position for this character
    int posXValue = posX[globalId];
    int posYValue = posY[globalId];

    // convert color from uint to float3
    uint colorInt = color[globalId];
    float3 colorFloat;
    colorFloat.x = ((colorInt >> 16) & 0xFF) / 255.0f; // R
    colorFloat.y = ((colorInt >> 8) & 0xFF) / 255.0f;  // G
    colorFloat.z = (colorInt & 0xFF) / 255.0f;         // B
    
    // Render the character at the specified position
    renderFont(fontSizeX, fontSizeY, spriteSizeX, spriteSizeY, currentChar,
               ScreenColors, FontData, screenWidth, screenHeight, posXValue, posYValue, colorFloat);
}

__kernel void gpuTimings(
    __global float *ScreenColors,
    const int screenWidth,
    const int screenHeight,
    const int SizeX,
    const int SizeY,
    const int PosX,
    const int PosY,
    const int PaddingY,
    const float renderSkyBoxTime,
    const float renderTrianglesTime,
    const float applyReflectionsTime,
    const float applyBlurTime,
    const float readBackTime,
    const float renderTextTime, // New parameter for text rendering time
    const float projectParticlesTime, // New parameter for particle projection time
    const float maxTime
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= SizeX || y >= SizeY) return;
    
    int pixelIndex = (y + PosY) * screenWidth + (x + PosX);
    
    // Check if we're within screen bounds
    if ((x + PosX) >= screenWidth || (y + PosY) >= screenHeight) return;
    
    // **IMPORTANT: Only render if there's timing data to show**
    // Skip rendering if all timing values are zero or very small
    if (renderSkyBoxTime < 0.001f && renderTrianglesTime < 0.001f && 
        applyReflectionsTime < 0.001f && applyBlurTime < 0.001f && 
        readBackTime < 0.001f) {
        return; // Don't modify pixels if no timing data
    }
    
    // Create horizontal bar chart
    float barHeight = (float)SizeY / 7.0f;
    int barIndex = y / (int)barHeight;
    float barProgress = (float)x / (float)SizeX;
    
    // **ONLY render bars, don't change background**
    float timeValue = 0.0f;
    float normalizedTime = 0.0f;
    bool shouldRender = false;
    float3 color = (float3)(0.0f, 0.0f, 0.0f);
    
    switch(barIndex) {
        case 0: // SkyBox time (Red)
            timeValue = renderSkyBoxTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.8f, 0.2f, 0.2f);
                shouldRender = true;
            }
            break;
        case 1: // Triangles time (Green)
            timeValue = renderTrianglesTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.2f, 0.8f, 0.2f);
                shouldRender = true;
            }
            break;
        case 2: // Reflections time (Blue)
            timeValue = applyReflectionsTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.2f, 0.2f, 0.8f);
                shouldRender = true;
            }
            break;
        case 3: // Blur time (Yellow)
            timeValue = applyBlurTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.8f, 0.8f, 0.2f);
                shouldRender = true;
            }
            break;
        case 4: // ReadBack time (Magenta)
            timeValue = readBackTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.8f, 0.2f, 0.8f);
                shouldRender = true;
            }
            break;
        case 5: // Render Text time (Cyan)
            timeValue = renderTextTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.2f, 0.8f, 0.8f);
                shouldRender = true;
            }
            break;
        case 6: // Project Particles time (Orange)
            timeValue = projectParticlesTime;
            normalizedTime = timeValue / maxTime;
            if (barProgress <= normalizedTime && timeValue > 0.001f) {
                color = (float3)(0.8f, 0.5f, 0.2f);
                shouldRender = true;
            }
            break;
    }
    
    // Add bar separators (thin lines between bars)
    if (y % (int)barHeight == 0 && y > 0) {
        color = (float3)(0.3f, 0.3f, 0.3f); // Darker gray separator
        shouldRender = true;
    }
    
    if (shouldRender) {
        ScreenColors[pixelIndex * 3]     = color.x;
        ScreenColors[pixelIndex * 3 + 1] = color.y;
        ScreenColors[pixelIndex * 3 + 2] = color.z;
    }
}

__kernel void renderSkyBox(
    __global float *ScreenColors,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    __global const float *SkyBoxTop, // 3 floats for RGB
    __global const float *SkyBoxBottom, // 3 floats for RGB
    __global const float *SkyBoxLeft, // 3 floats for RGB
    __global const float *SkyBoxRight, // 3 floats for RGB
    __global const float *SkyBoxFront, // 3 floats for RGB
    __global const float *SkyBoxBack, // 3 floats for RGB
    const int skyBoxWidth,
    const int skyBoxHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    int pixelIndex = y * screenWidth + x;
    
    // Compute camera basis
    float3 forward = normalize(camDir);
    float3 camUp = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);
    
    // Convert screen coordinates to normalized device coordinates
    float ndcX = (x + 0.5f) / screenWidth * 2.0f - 1.0f;
    float ndcY = -((y + 0.5f) / screenHeight * 2.0f - 1.0f);  // FLIP Y HERE
    
    // Create ray direction in world space
    float3 rayDir = normalize(forward + ndcX * right * fov + ndcY * up * fov);
    
    // Sample skybox based on ray direction
    float3 skyboxColor = (float3)(0.5f, 0.7f, 1.0f); // Default sky blue
    
    // Determine which face of the skybox to sample
    float3 absDir = fabs(rayDir);
    float maxComponent = max(max(absDir.x, absDir.y), absDir.z);
    
    float2 uv;
    __global const float *selectedFace;
    
    if (maxComponent == absDir.x) {
        // Left or Right face
        if (rayDir.x > 0) {
            // Right face (+X)
            uv.x = (-rayDir.z / rayDir.x + 1.0f) * 0.5f;
            uv.y = (-rayDir.y / rayDir.x + 1.0f) * 0.5f;
            selectedFace = SkyBoxRight;
        } else {
            // Left face (-X)
            uv.x = (rayDir.z / (-rayDir.x) + 1.0f) * 0.5f;
            uv.y = (-rayDir.y / (-rayDir.x) + 1.0f) * 0.5f;
            selectedFace = SkyBoxLeft;
        }
    } else if (maxComponent == absDir.y) {
        // Top or Bottom face
        if (rayDir.y > 0) {
            // Top face (+Y)
            uv.x = (rayDir.x / rayDir.y + 1.0f) * 0.5f;
            uv.y = (rayDir.z / rayDir.y + 1.0f) * 0.5f;
            selectedFace = SkyBoxTop;
        } else {
            // Bottom face (-Y)
            uv.x = (rayDir.x / (-rayDir.y) + 1.0f) * 0.5f;
            uv.y = (-rayDir.z / (-rayDir.y) + 1.0f) * 0.5f;
            selectedFace = SkyBoxBottom;
        }
    } else {
        // Front or Back face
        if (rayDir.z > 0) {
            // Front face (+Z)
            uv.x = (rayDir.x / rayDir.z + 1.0f) * 0.5f;
            uv.y = (-rayDir.y / rayDir.z + 1.0f) * 0.5f;
            selectedFace = SkyBoxFront;
        } else {
            // Back face (-Z)
            uv.x = (-rayDir.x / (-rayDir.z) + 1.0f) * 0.5f;
            uv.y = (-rayDir.y / (-rayDir.z) + 1.0f) * 0.5f;
            selectedFace = SkyBoxBack;
        }
    }
    
    // Clamp UV coordinates
    uv = clamp(uv, 0.0f, 1.0f);
    
    // Sample the texture with bilinear filtering
    int texX = (int)(uv.x * (skyBoxWidth - 1));
    int texY = (int)(uv.y * (skyBoxHeight - 1));
    int texIndex = (texY * skyBoxWidth + texX) * 3;
    
    if (selectedFace != NULL) {
        skyboxColor.x = selectedFace[texIndex];
        skyboxColor.y = selectedFace[texIndex + 1];
        skyboxColor.z = selectedFace[texIndex + 2];
    }
    
    // Apply atmospheric perspective and time-of-day effects
    float altitude = rayDir.y; // -1 to 1, where 1 is straight up
    
    // Horizon fade effect
    float horizonFade = smoothstep(-0.1f, 0.3f, altitude);
    
    // Sun/moon position (you can make this dynamic)
    float3 sunDir = normalize((float3)(-0.2f, 0.6f, -0.8f));
    float sunDot = max(0.0f, dot(rayDir, sunDir));
    
    // Sun glow effect
    float sunGlow = pow(sunDot, 50.0f) * 2.0f + pow(sunDot, 5.0f) * 0.5f;
    float3 sunColor = (float3)(1.0f, 0.9f, 0.7f);
    
    // Atmospheric scattering approximation
    float3 atmosColor = mix((float3)(0.8f, 0.9f, 1.0f), (float3)(1.0f, 0.7f, 0.4f), (1.0f - altitude) * 0.5f);
    
    // Combine skybox with atmospheric effects
    skyboxColor = mix(skyboxColor, atmosColor, 0.3f * (1.0f - horizonFade));
    skyboxColor += sunGlow * sunColor;
    
    // Store skybox color
    int colorIndex = pixelIndex * 3;
    ScreenColors[colorIndex] = clamp(skyboxColor.x, 0.0f, 1.0f);
    ScreenColors[colorIndex + 1] = clamp(skyboxColor.y, 0.0f, 1.0f);
    ScreenColors[colorIndex + 2] = clamp(skyboxColor.z, 0.0f, 1.0f);
}

// Helper function to sample skybox color for a given ray direction
float3 sampleSkybox(
    const float3 rayDir,
    __global const float *SkyBoxTop,
    __global const float *SkyBoxBottom, 
    __global const float *SkyBoxLeft,
    __global const float *SkyBoxRight,
    __global const float *SkyBoxFront,
    __global const float *SkyBoxBack,
    const int skyBoxWidth,
    const int skyBoxHeight
) {
    // Normalize ray direction
    float3 dir = normalize(rayDir);
    
    // Determine which face of the skybox to sample
    float3 absDir = fabs(dir);
    float maxComponent = max(max(absDir.x, absDir.y), absDir.z);
    
    // Add safety check for very small components
    const float epsilon = 1e-6f;
    
    float2 uv;
    __global const float *selectedFace = NULL;
    
    if (maxComponent == absDir.x && fabs(dir.x) > epsilon) {
        // Left or Right face
        if (dir.x > 0) {
            // Right face (+X)
            uv.x = (-dir.z / dir.x + 1.0f) * 0.5f;
            uv.y = (-dir.y / dir.x + 1.0f) * 0.5f;
            selectedFace = SkyBoxRight;
        } else {
            // Left face (-X)  
            uv.x = (dir.z / (-dir.x) + 1.0f) * 0.5f;
            uv.y = (-dir.y / (-dir.x) + 1.0f) * 0.5f;
            selectedFace = SkyBoxLeft;
        }
    } else if (maxComponent == absDir.y && fabs(dir.y) > epsilon) {
        // Top or Bottom face
        if (dir.y > 0) {
            // Top face (+Y)
            uv.x = (dir.x / dir.y + 1.0f) * 0.5f;
            uv.y = (dir.z / dir.y + 1.0f) * 0.5f;
            selectedFace = SkyBoxTop;
        } else {
            // Bottom face (-Y)
            uv.x = (dir.x / (-dir.y) + 1.0f) * 0.5f;
            uv.y = (-dir.z / (-dir.y) + 1.0f) * 0.5f;
            selectedFace = SkyBoxBottom;
        }
    } else if (fabs(dir.z) > epsilon) {
        // Front or Back face
        if (dir.z > 0) {
            // Front face (+Z)
            uv.x = (dir.x / dir.z + 1.0f) * 0.5f;
            uv.y = (-dir.y / dir.z + 1.0f) * 0.5f;
            selectedFace = SkyBoxFront;
        } else {
            // Back face (-Z)
            uv.x = (-dir.x / (-dir.z) + 1.0f) * 0.5f;
            uv.y = (-dir.y / (-dir.z) + 1.0f) * 0.5f;
            selectedFace = SkyBoxBack;
        }
    }
    
    // Clamp UV coordinates to valid range
    uv = clamp(uv, 0.0f, 1.0f);
    
    // Default fallback color
    float3 skyboxColor = (float3)(0.5f, 0.7f, 1.0f);
    
    // Sample the texture with bounds checking
    if (selectedFace != NULL && skyBoxWidth > 0 && skyBoxHeight > 0) {
        int texX = clamp((int)(uv.x * (skyBoxWidth - 1)), 0, skyBoxWidth - 1);
        int texY = clamp((int)(uv.y * (skyBoxHeight - 1)), 0, skyBoxHeight - 1);
        int texIndex = (texY * skyBoxWidth + texX) * 3;
        
        skyboxColor.x = selectedFace[texIndex];
        skyboxColor.y = selectedFace[texIndex + 1];
        skyboxColor.z = selectedFace[texIndex + 2];
    }
    
    return skyboxColor;
}

float3 sampleScreenSpaceReflectionFiltered(
    __global const float* ScreenColors,
    __global const float* ScreenDistances,
    const float3 rayOrigin,
    const float3 rayDirection,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    const float maxDistance,
    const int maxSteps,
    const float stepSize
) {
    float3 fallbackColor = (float3)(0.0f, 0.0f, 0.0f);
    
    float3 forward = normalize(camDir);
    float3 camUp = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);
    
    float3 currentPos = rayOrigin;
    float distanceTraveled = 0.0f;
    
    // FIX 1: Start with a small offset to avoid self-intersection
    currentPos += rayDirection * stepSize * 0.5f;
    
    for (int step = 0; step < maxSteps; step++) {
        currentPos += rayDirection * stepSize;
        distanceTraveled += stepSize;
        
        if (distanceTraveled > maxDistance) {
            break;
        }
        
        float3 relativePos = currentPos - camPos;
        float depth = dot(relativePos, forward);
        
        // FIX 2: Better depth bounds checking
        if (depth <= 0.01f || depth > maxDistance) {
            continue;
        }
        
        float fovScale = 1.0f / (depth * fov);
        float screenRight = dot(relativePos, right) * fovScale;
        float screenUpward = dot(relativePos, up) * fovScale;
        
        float halfWidth = screenWidth * 0.5f;
        float halfHeight = screenHeight * 0.5f;
        
        float screenX = screenRight * halfWidth + halfWidth;
        float screenY = -screenUpward * halfHeight + halfHeight;
        
        // FIX 3: Add margin to screen bounds to avoid edge artifacts
        if (screenX < 1.0f || screenX >= (screenWidth - 1.0f) || 
            screenY < 1.0f || screenY >= (screenHeight - 1.0f)) {
            continue;
        }
        
        // Get integer coordinates for depth test
        int pixelX = (int)screenX;
        int pixelY = (int)screenY;
        int pixelIndex = pixelY * screenWidth + pixelX;
        
        // FIX 4: Bounds check for pixelIndex
        if (pixelIndex < 0 || pixelIndex >= screenWidth * screenHeight) {
            continue;
        }
        
        float sceneDepth = ScreenDistances[pixelIndex];
        
        // FIX 5: Better depth comparison with adaptive threshold
        float depthThreshold = stepSize * 1.5f + depth * 0.001f; // Adaptive threshold
        float depthDifference = depth - sceneDepth;
        
        // FIX 6: Check if we've hit something and it's in front of our ray
        if (sceneDepth > 0.01f && depthDifference > 0.0f && depthDifference < depthThreshold) {
            // FIX 7: Improved bilinear filtering with bounds checking
            float fx = screenX - pixelX;
            float fy = screenY - pixelY;
            
            // Sample 4 neighboring pixels with bounds checking
            int x0 = clamp(pixelX, 0, screenWidth - 1);
            int x1 = clamp(pixelX + 1, 0, screenWidth - 1);
            int y0 = clamp(pixelY, 0, screenHeight - 1);
            int y1 = clamp(pixelY + 1, 0, screenHeight - 1);
            
            int idx00 = (y0 * screenWidth + x0) * 3;
            int idx10 = (y0 * screenWidth + x1) * 3;
            int idx01 = (y1 * screenWidth + x0) * 3;
            int idx11 = (y1 * screenWidth + x1) * 3;
            
            // FIX 8: Check all sample indices are valid
            if (idx00 >= 0 && idx11 < screenWidth * screenHeight * 3) {
                // Interpolate colors
                float3 color00 = (float3)(ScreenColors[idx00], ScreenColors[idx00+1], ScreenColors[idx00+2]);
                float3 color10 = (float3)(ScreenColors[idx10], ScreenColors[idx10+1], ScreenColors[idx10+2]);
                float3 color01 = (float3)(ScreenColors[idx01], ScreenColors[idx01+1], ScreenColors[idx01+2]);
                float3 color11 = (float3)(ScreenColors[idx11], ScreenColors[idx11+1], ScreenColors[idx11+2]);
                
                float3 colorTop = mix(color00, color10, fx);
                float3 colorBottom = mix(color01, color11, fx);
                float3 finalColor = mix(colorTop, colorBottom, fy);
                
                // FIX 9: Ensure we return a valid color (not black)
                if (length(finalColor) > 0.01f) {
                    return finalColor;
                }
            }
        }
    }
    
    return fallbackColor;
}

float3 reflectVector(const float3 I, const float3 N) {
    return I - 2.0f * dot(N, I) * N;
}

__kernel void applyReflections(
    __global float *ScreenColors,
    __global const float *ScreenDistances,
    __global const float *ScreenNormals,
    __global const float *ScreenMaterialRoughness,
    __global const float *ScreenMaterialMetallic,
    __global const float *ScreenMaterialEmission,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    __global const float *SkyBoxTop,
    __global const float *SkyBoxBottom,
    __global const float *SkyBoxLeft,
    __global const float *SkyBoxRight,
    __global const float *SkyBoxFront,
    __global const float *SkyBoxBack,
    const int skyBoxWidth,
    const int skyBoxHeight,
    const int conservativeMode  // 0 = use materials, 1 = conservative reflections
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    int pixelIndex = y * screenWidth + x;
    float depth = ScreenDistances[pixelIndex];
    
    if (depth <= 0.001f) return;
    
    int normalIndex = pixelIndex * 3;
    float3 normal = (float3)(ScreenNormals[normalIndex], 
                             ScreenNormals[normalIndex + 1], 
                             ScreenNormals[normalIndex + 2]);
    
    float3 forward = normalize(camDir);
    float3 camUp = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);
    
    float ndcX = (x + 0.5f) / screenWidth * 2.0f - 1.0f;
    float ndcY = -((y + 0.5f) / screenHeight * 2.0f - 1.0f);
    
    float3 rayDir = normalize(forward + ndcX * right * fov + ndcY * up * fov);
    float3 worldPos = camPos + rayDir * depth;
    
    float3 viewDir = normalize(camPos - worldPos);
    float3 reflectedDir = reflectVector(-viewDir, normalize(normal));
    
    // Sample reflections
    float3 screenSpaceReflection = sampleScreenSpaceReflectionFiltered(
        ScreenColors, ScreenDistances, worldPos, reflectedDir, camPos, camDir, fov,
        screenWidth, screenHeight, 
        min(500.0f, depth * 50.0f),
        512,
        max(0.5f, depth * 0.1f)
    );
    
    float3 skyboxReflection = sampleSkybox(reflectedDir, SkyBoxTop, SkyBoxBottom,
                                           SkyBoxLeft, SkyBoxRight,
                                           SkyBoxFront, SkyBoxBack,
                                           skyBoxWidth, skyBoxHeight);
    
    float screenReflectionStrength = length(screenSpaceReflection);
    float3 environmentReflection = (screenReflectionStrength > 0.01f) ? 
                                     screenSpaceReflection : skyboxReflection;
    
    int colorIndex = pixelIndex * 3;
    float3 baseColor = (float3)(ScreenColors[colorIndex], 
                                ScreenColors[colorIndex + 1], 
                                ScreenColors[colorIndex + 2]);
    
    float3 finalColor;
    
    if (conservativeMode == 1) {
        // Conservative mode - ignore materials, use simple Fresnel-based reflection
        float fresnel = 0.04f + (1.0f - 0.04f) * pow(1.0f - max(0.0f, dot(normal, viewDir)), 5.0f);
        
        // Conservative reflection strength (subtle)
        float reflectionFactor = clamp(fresnel * 0.3f, 0.0f, 0.1f);  // Max 30% reflection
        
        finalColor = mix(baseColor, environmentReflection, reflectionFactor);
    } else {
        // Material-based mode (original behavior)
        float roughness = ScreenMaterialRoughness[pixelIndex]; 
        float metallic  = ScreenMaterialMetallic[pixelIndex];
        float emission  = ScreenMaterialEmission[pixelIndex];
        
        float fresnel = 0.04f + (1.0f - 0.04f) * pow(1.0f - max(0.0f, dot(normal, viewDir)), 5.0f);
        fresnel = mix(fresnel, 1.0f, metallic);
        
        float metallicBoost = mix(1.0f, 2.0f, metallic);
        float reflectionFactor = clamp(fresnel * (1.0f - roughness) * (1.0f - emission) * metallicBoost, 0.0f, 1.0f);
        
        finalColor = mix(baseColor, environmentReflection, reflectionFactor);
    }
    
    ScreenColors[colorIndex]     = clamp(finalColor.x, 0.0f, 1.0f);
    ScreenColors[colorIndex + 1] = clamp(finalColor.y, 0.0f, 1.0f);
    ScreenColors[colorIndex + 2] = clamp(finalColor.z, 0.0f, 1.0f);
}

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

            // Proper bounds check
            if (nx >= 0 && nx < screenWidth && ny >= 0 && ny < screenHeight) {
                int neighborIndex = ny * screenWidth + nx;
                
                // Ensure neighborIndex is within bounds
                if (neighborIndex >= 0 && neighborIndex < screenWidth * screenHeight) {
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

__kernel void drawBoundingBox(
    __global float *ScreenDistances,
    __global float *ScreenOpacities,
    __global float *ScreenVelocities,
    const float3 camPos,
    const float3 camDir,
    const float3 camUp,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    const float3 bBoxMin,
    const float3 bBoxMax
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    // Camera basis vectors (precompute once per thread)
    float3 forward = normalize(camDir);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);
    
    // Convert screen coordinates to NDC
    float ndcX = (x + 0.5f) / screenWidth * 2.0f - 1.0f;
    float ndcY = -((y + 0.5f) / screenHeight * 2.0f - 1.0f);
    
    // Create ray direction
    float3 rayDir = normalize(forward + ndcX * right * fov + ndcY * up * fov);
    
    // Ray-AABB intersection using slab method
    float3 invRayDir = 1.0f / rayDir;
    float3 t1 = (bBoxMin - camPos) * invRayDir;
    float3 t2 = (bBoxMax - camPos) * invRayDir;
    
    // Ensure t1 <= t2 for all components
    float3 tMin = fmin(t1, t2);
    float3 tMax = fmax(t1, t2);
    
    // Find intersection interval
    float tNear = fmax(fmax(tMin.x, tMin.y), tMin.z);
    float tFar = fmin(fmin(tMax.x, tMax.y), tMax.z);
    
    // Check if ray intersects the bounding box
    if (tNear <= tFar && tFar > 0.001f) {
        // Use the near intersection point (closest to camera)
        float t = (tNear > 0.001f) ? tNear : tFar;
        
        int pixelIndex = y * screenWidth + x;
        
        // Only update if this is closer than existing geometry
        if (ScreenDistances[pixelIndex] == 0.0f || t < ScreenDistances[pixelIndex]) {
            // Calculate which face we hit to determine color intensity
            float3 hitPoint = camPos + rayDir * t;
            float3 center = (bBoxMin + bBoxMax) * 0.5f;
            float3 size = bBoxMax - bBoxMin;
            float3 localPos = (hitPoint - center) / size; // Normalize to [-0.5, 0.5]
            
            // Determine which face (for edge highlighting)
            float3 absLocal = fabs(localPos);
            float maxComp = fmax(fmax(absLocal.x, absLocal.y), absLocal.z);
            
            // Check if we're near an edge (for wireframe effect)
            float edgeThickness = 0.02f; // Adjust for thicker/thinner edges
            bool isEdge = false;
            
            if (maxComp == absLocal.x) {
                // Hit X face, check Y and Z edges
                isEdge = (fabs(absLocal.y) > 0.5f - edgeThickness) || 
                         (fabs(absLocal.z) > 0.5f - edgeThickness);
            } else if (maxComp == absLocal.y) {
                // Hit Y face, check X and Z edges
                isEdge = (fabs(absLocal.x) > 0.5f - edgeThickness) || 
                         (fabs(absLocal.z) > 0.5f - edgeThickness);
            } else {
                // Hit Z face, check X and Y edges
                isEdge = (fabs(absLocal.x) > 0.5f - edgeThickness) || 
                         (fabs(absLocal.y) > 0.5f - edgeThickness);
            }
            
            // Set distance and opacity based on whether it's an edge
            if (isEdge) {
                ScreenDistances[pixelIndex] = t;
                ScreenOpacities[pixelIndex] = 1.0f;
                ScreenVelocities[pixelIndex] = 1.0f;
            }
        }
    }
}

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
    const int ParticleRadius,
    const float maxParticleVelocity
    // const float maxParticleDistance
) {
    int i = get_global_id(0);
    if (i >= numPoints) return;

    // manually unpack
    float3 point = (float3)( points[3*i+0],
                             points[3*i+1],
                             points[3*i+2] );

    // normalize velocity to 0,1 range based on maxParticleVelocity
    float velNormalized =   (velocities[3*i+0] * velocities[3*i+0] + velocities[3*i+1] * velocities[3*i+1] + velocities[3*i+2] * velocities[3*i+2]) / maxParticleVelocity;
                        

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
                ScreenVelocities[offsetIndex] = velNormalized;
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

            if (ScreenOpacities[offsetIndex] < maxFloat) { 
                ScreenOpacities[offsetIndex] += 0.1f; // Increment opacity
            }
        }
    }
}

float3 calculateVertexNormal(
    const float3 vertex,
    const int currentTriangleId,
    __global const float* v1,
    __global const float* v2, 
    __global const float* v3,
    __global const float* normals,
    const int numTriangles,
    const float threshold
) {
    float3 accumulatedNormal = (float3)(0.0f, 0.0f, 0.0f);
    int normalCount = 0;
    
    // Search through all triangles to find ones that share this vertex
    for (int triId = 0; triId < numTriangles; triId++) {
        int vertexIndex = triId * 3;
        float3 tri_v1 = (float3)(v1[vertexIndex], v1[vertexIndex + 1], v1[vertexIndex + 2]);
        float3 tri_v2 = (float3)(v2[vertexIndex], v2[vertexIndex + 1], v2[vertexIndex + 2]);
        float3 tri_v3 = (float3)(v3[vertexIndex], v3[vertexIndex + 1], v3[vertexIndex + 2]);
        
        // Check if this triangle shares the vertex (within threshold)
        bool sharesVertex = false;
        if (distance(vertex, tri_v1) < threshold || 
            distance(vertex, tri_v2) < threshold || 
            distance(vertex, tri_v3) < threshold) {
            sharesVertex = true;
        }
        
        if (sharesVertex) {
            // Add this triangle's normal to the accumulation
            float3 triNormal = (float3)(normals[triId * 3], 
                                       normals[triId * 3 + 1], 
                                       normals[triId * 3 + 2]);
            accumulatedNormal += triNormal;
            normalCount++;
        }
        
        // Limit search to avoid performance issues
        if (normalCount >= 8) break;
    }
    
    // Average the normals and normalize
    if (normalCount > 0) {
        accumulatedNormal /= (float)normalCount;
        return normalize(accumulatedNormal);
    } else {
        // Fallback to face normal if no adjacent triangles found
        float3 faceNormal = (float3)(normals[currentTriangleId * 3], 
                                    normals[currentTriangleId * 3 + 1], 
                                    normals[currentTriangleId * 3 + 2]);
        return normalize(faceNormal);
    }
}

int scaledValue(float x) {
    float a = 1.2f;
    float b = 5.0f - a;
    float c = 0.014f;
    return (int)a + b / (1.0f + c * x);
}

__kernel void calculateVertexCoordinate(
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
    __global float* projectedVerts,
    __global float* bboxes,
    __global int* validTriangles
) { 
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;

    validTriangles[triangleId] = 0;

    float3 vertex1 = vload3(triangleId, v1);
    float3 vertex2 = vload3(triangleId, v2);
    float3 vertex3 = vload3(triangleId, v3);
    float3 faceNormal = normalize(vload3(triangleId, normals));

    float3 forward = normalize(camDir);
    float3 up = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, up));
    up = cross(right, forward);

    float3 center = (vertex1 + vertex2 + vertex3) * 0.33333f;
    float3 toCamera = camPos - center;
    if (dot(faceNormal, toCamera) <= 0.0f) return;

    float3 rel1 = vertex1 - camPos;
    float3 rel2 = vertex2 - camPos;
    float3 rel3 = vertex3 - camPos;
    
    float depth1 = dot(rel1, forward);
    float depth2 = dot(rel2, forward);
    float depth3 = dot(rel3, forward);
    
    if (depth1 <= 0.01f || depth2 <= 0.01f || depth3 <= 0.01f) return;

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

    float area = fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    if (area < 1.0f) return;

    float minX = min(min(x1, x2), x3);
    float maxX = max(max(x1, x2), x3);
    float minY = min(min(y1, y2), y3);
    float maxY = max(max(y1, y2), y3);

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

    bboxes[triangleId * 4 + 0] = clamp(minX, 0.0f, (float)screenWidth);
    bboxes[triangleId * 4 + 1] = clamp(maxX, 0.0f, (float)screenWidth);
    bboxes[triangleId * 4 + 2] = clamp(minY, 0.0f, (float)screenHeight);
    bboxes[triangleId * 4 + 3] = clamp(maxY, 0.0f, (float)screenHeight);

    validTriangles[triangleId] = 1;
}

__kernel void TileCulling(
    __global const float* bboxes,
    __global const int* validTriangles,
    __global int* tileLists,
    __global int* tileListCounts,
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
    
    float tileMinX = tileX * TILE_SIZE;
    float tileMaxX = min((tileX + 1) * TILE_SIZE, screenWidth);
    float tileMinY = tileY * TILE_SIZE;
    float tileMaxY = min((tileY + 1) * TILE_SIZE, screenHeight);
    
    int count = 0;
    int baseOffset = tileIdx * MAX_TRIS_PER_TILE;
    
    for (int t = 0; t < numTriangles; t++) {
        if (validTriangles[t] == 0) continue;
        
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

    int tileX = px / TILE_SIZE;
    int tileY = py / TILE_SIZE;
    int tileIdx = tileY * numTilesX + tileX;
    
    int numTrisInTile = tileListCounts[tileIdx];
    int tileListBase = tileIdx * MAX_TRIS_PER_TILE;

    float bestDepth = ScreenDistances[idx] > 0.0f 
        ? ScreenDistances[idx] : INFINITY;
    int bestTri = -1;

    for (int i = 0; i < numTrisInTile; i++) {
        int t = tileLists[tileListBase + i];
        
        int ov = t * 9;
        float minZ = min(min(projectedVerts[ov + 2], 
                            projectedVerts[ov + 5]), 
                            projectedVerts[ov + 8]);
        if (minZ >= bestDepth) continue;

        int bi = t * 4;
        float minX = bboxes[bi];
        float maxX = bboxes[bi + 1];
        float minY = bboxes[bi + 2];
        float maxY = bboxes[bi + 3];
        if (cx < minX || cx > maxX || cy < minY || cy > maxY) continue;

        float2 p0 = (float2)(projectedVerts[ov], projectedVerts[ov + 1]);
        float2 p1 = (float2)(projectedVerts[ov + 3], projectedVerts[ov + 4]);
        float2 p2 = (float2)(projectedVerts[ov + 6], projectedVerts[ov + 7]);
        float z0 = projectedVerts[ov + 2];
        float z1 = projectedVerts[ov + 5];
        float z2 = projectedVerts[ov + 8];

        float2 v0 = p1 - p0;
        float2 v1 = p2 - p0;
        float2 vp = (float2)(cx, cy) - p0;
        
        float denom = v0.x * v1.y - v1.x * v0.y;
        if (fabs(denom) < 1e-6f) continue;
        
        float invDenom = 1.0f / denom;
        float u = (vp.x * v1.y - v1.x * vp.y) * invDenom;
        float v = (v0.x * vp.y - vp.x * v0.y) * invDenom;
        
        if (u < 0.0f || v < 0.0f || u + v > 1.0f) continue;

        float w = 1.0f - u - v;
        float depth = w * z0 + u * z1 + v * z2;
        
        if (depth < bestDepth) {
            bestDepth = depth;
            bestTri = t;
        }
    }

    if (bestTri >= 0 && (ScreenDistances[idx] == 0.0f || bestDepth < ScreenDistances[idx])) {
        ScreenDistances[idx] = bestDepth;

        float3 N = normalize(vload3(bestTri, normals));
        vstore3(N, idx, ScreenNormals);

        float3 C = vload3(bestTri, TriangleColors);
        float3 lightDir = normalize((float3)(0.3f, 0.7f, 0.5f));
        float lighting = max(0.65f, dot(N, lightDir));
        float3 finalColor = clamp(C * lighting + C * emission[bestTri], 0.0f, 1.0f);
        vstore3(finalColor, idx, ScreenColors);

        ScreenMaterialRoughness[idx] = roughness[bestTri];
        ScreenMaterialMetallic[idx] = metallic[bestTri];
        ScreenMaterialEmission[idx] = emission[bestTri];
    }
}

void drawLineWireframe(
    const float2 start,
    const float2 end,
    const float startDepth,
    const float endDepth,
    __global float* ScreenColors,
    __global float* ScreenDistances,
    const int screenWidth,
    const int screenHeight,
    const float3 wireColor,
    const int lineWidth
) {
    // OPTIMIZATION 1: Early exit for zero-length lines
    if (distance(start, end) < 0.5f) return;
    
    // OPTIMIZATION 2: Pre-calculate bounds to avoid repeated bounds checking
    int minX = max(0, (int)(min(start.x, end.x) - lineWidth));
    int maxX = min(screenWidth - 1, (int)(max(start.x, end.x) + lineWidth));
    int minY = max(0, (int)(min(start.y, end.y) - lineWidth));
    int maxY = min(screenHeight - 1, (int)(max(start.y, end.y) + lineWidth));
    
    // Early exit if line is completely outside screen
    if (minX > maxX || minY > maxY) return;
    
    // Bresenham's line algorithm for the center line
    int x0 = (int)start.x;
    int y0 = (int)start.y;
    int x1 = (int)end.x;
    int y1 = (int)end.y;
    
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int x_inc = (x0 < x1) ? 1 : -1;
    int y_inc = (y0 < y1) ? 1 : -1;
    int error = dx - dy;
    
    int x = x0;
    int y = y0;
    
    // OPTIMIZATION 3: Pre-calculate values that don't change in the loop
    float totalDistance = distance(start, end);
    float invTotalDistance = (totalDistance > 0.0f) ? (1.0f / totalDistance) : 0.0f;
    int halfWidth = lineWidth / 2;
    int halfWidthSquared = halfWidth * halfWidth;
    
    // OPTIMIZATION 4: Cache color values to avoid repeated memory access
    float colorR = wireColor.x;
    float colorG = wireColor.y;
    float colorB = wireColor.z;
    
    while (true) {
        // OPTIMIZATION 5: Early bounds check for current position
        if (x < minX || x > maxX || y < minY || y > maxY) {
            goto next_step; // Skip pixel drawing but continue line
        }
        
        // OPTIMIZATION 6: Pre-calculate depth interpolation for this point
        float t = distance((float2)(x, y), start) * invTotalDistance;
        t = clamp(t, 0.0f, 1.0f); // Ensure t is in valid range
        float depth = mix(startDepth, endDepth, t);
        
        // OPTIMIZATION 7: Optimized square drawing with early bounds checking
        int startY = max(minY, y - halfWidth);
        int endY = min(maxY, y + halfWidth);
        int startX = max(minX, x - halfWidth);
        int endX = min(maxX, x + halfWidth);
        
        // OPTIMIZATION 8: Use linear indexing and minimize array accesses
        for (int py = startY; py <= endY; py++) {
            int rowBase = py * screenWidth; // Pre-calculate row offset
            int dy_offset = py - y;
            int dy2 = dy_offset * dy_offset;
            
            // OPTIMIZATION 9: Early exit if outside circle (for circular brush)
            if (dy2 > halfWidthSquared) continue;
            
            for (int px = startX; px <= endX; px++) {
                int dx_offset = px - x;
                int dx2 = dx_offset * dx_offset;
                
                // OPTIMIZATION 10: Circle check (optional - remove if you want square brush)
                if (dx2 + dy2 > halfWidthSquared) continue;
                
                int pixelIndex = rowBase + px; // Use pre-calculated row offset
                
                // OPTIMIZATION 11: Single depth test with early exit
                if (ScreenDistances[pixelIndex] != 0.0f && depth >= ScreenDistances[pixelIndex]) {
                    continue;
                }
                
                // OPTIMIZATION 12: Update depth and color in single pass
                ScreenDistances[pixelIndex] = depth;
                
                int colorIndex = pixelIndex * 3;
                ScreenColors[colorIndex] = colorR;
                ScreenColors[colorIndex + 1] = colorG;
                ScreenColors[colorIndex + 2] = colorB;
            }
        }
        
        next_step:
        // Check if we've reached the end
        if (x == x1 && y == y1) break;
        
        // OPTIMIZATION 13: Optimized Bresenham step
        int error2 = error << 1; // Bit shift instead of multiplication
        if (error2 > -dy) {
            error -= dy;
            x += x_inc;
        }
        if (error2 < dx) {
            error += dx;
            y += y_inc;
        }
    }
}

__kernel void renderWireFrame(
    __global const float* projectedVerts,
    __global const int* validTriangles,
    __global float* ScreenColors,
    __global float* ScreenDistances,
    const int screenWidth,
    const int screenHeight,
    const int numTriangles,
    const float3 wireColor
) {
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;
    
    // Check if triangle is valid
    if (validTriangles[triangleId] == 0) return;
    
    // Load projected vertices
    int vertBase = triangleId * 9;
    float3 v1 = vload3(0, &projectedVerts[vertBase]);
    float3 v2 = vload3(0, &projectedVerts[vertBase + 3]);
    float3 v3 = vload3(0, &projectedVerts[vertBase + 6]);
    
    // Draw three edges of the triangle
    drawLineWireframe(v1.xy, v2.xy, v1.z, v2.z, ScreenColors, ScreenDistances, 
                      screenWidth, screenHeight, wireColor, 1);
    drawLineWireframe(v2.xy, v3.xy, v2.z, v3.z, ScreenColors, ScreenDistances, 
                      screenWidth, screenHeight, wireColor, 1);
    drawLineWireframe(v3.xy, v1.xy, v3.z, v1.z, ScreenColors, ScreenDistances, 
                      screenWidth, screenHeight, wireColor, 1);
}

__kernel void renderTriangles(
    __global const float* v1,
    __global const float* v2,
    __global const float* v3,
    __global const float* normals,
    __global float *ScreenDistances,
    __global float *ScreenNormals,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    const int numTriangles,
    __global const float *TriangleColors,
    __global float *ScreenColors,
    __global const float* roughness,
    __global const float* metallic,
    __global const float* emission,
    __global float *ScreenMaterialRoughness,
    __global float *ScreenMaterialMetallic,
    __global float *ScreenMaterialEmission
) {
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;

    // Precompute indices and load data once
    int b3 = triangleId * 3;
    float3 p0 = vload3(0, v1 + b3);
    float3 p1 = vload3(0, v2 + b3);
    float3 p2 = vload3(0, v3 + b3);
    float3 fn = vload3(0, normals + b3);
    float3 tc = vload3(0, TriangleColors + b3);
    float Rg = roughness[triangleId];
    float Mt = metallic[triangleId];
    float Em = emission[triangleId];

    // Back-face culling
    float3 center = (p0 + p1 + p2) * (1.0f/3.0f);
    if (dot(fn, normalize(camPos - center)) <= 0.0f) return;

    // Camera basis
    float3 F = normalize(camDir);
    float3 U = (float3)(0,1,0);
    float3 R = normalize(cross(F,U));
    U = cross(R,F);

    // Constants
    float invF = 1.0f / fov;
    float halfW = screenWidth * 0.5f, halfH = screenHeight * 0.5f;

    // Compute depths and screen projections
    float3 r0 = p0 - camPos, r1 = p1 - camPos, r2 = p2 - camPos;
    float d0 = dot(r0,F), d1 = dot(r1,F), d2 = dot(r2,F);
    float minD = fmin(fmin(d0,d1),d2);
    if (minD <= 0.001f) return;
    float s0 = invF/d0, s1 = invF/d1, s2 = invF/d2;
    float3 sp0 = (float3)(dot(r0,R)*halfW*s0 + halfW, -dot(r0,U)*halfH*s0 + halfH, d0);
    float3 sp1 = (float3)(dot(r1,R)*halfW*s1 + halfW, -dot(r1,U)*halfH*s1 + halfH, d1);
    float3 sp2 = (float3)(dot(r2,R)*halfW*s2 + halfW, -dot(r2,U)*halfH*s2 + halfH, d2);

    // Frustum culling & bounding box
    float minXf = fmin(fmin(sp0.x,sp1.x),sp2.x),
          maxXf = fmax(fmax(sp0.x,sp1.x),sp2.x),
          minYf = fmin(fmin(sp0.y,sp1.y),sp2.y),
          maxYf = fmax(fmax(sp0.y,sp1.y),sp2.y);
    if (maxXf < 0 || minXf >= screenWidth || maxYf < 0 || minYf >= screenHeight) return;
    int x0 = max(0,(int)minXf), x1 = min(screenWidth-1,(int)maxXf);
    int y0 = max(0,(int)minYf), y1 = min(screenHeight-1,(int)maxYf);

    // Small triangle culling
    float area = fabs((sp1.x-sp0.x)*(sp2.y-sp0.y) - (sp2.x-sp0.x)*(sp1.y-sp0.y)) * 0.5f;
    if (area < 0.5f) return;

    // Precompute barycentric constants
    float2 e1 = sp2.xy - sp0.xy, e2 = sp1.xy - sp0.xy;
    float d00 = dot(e1,e1), d01 = dot(e1,e2), d11 = dot(e2,e2);
    float invDen = 1.0f / (d00*d11 - d01*d01);

    // Rasterize
    for (int y = y0; y <= y1; ++y) {
        float cy = (y + 0.5f) - sp0.y;
        int row = y * screenWidth;
        for (int x = x0; x <= x1; ++x) {
            float cx = (x + 0.5f) - sp0.x;
            float l0 = e1.x*cx + e1.y*cy;
            float l1 = e2.x*cx + e2.y*cy;
            float u = (d11*l0 - d01*l1) * invDen;
            float v = (d00*l1 - d01*l0) * invDen;
            if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
                int idx = row + x;
                float w = 1.0f - u - v;
                float depth = w*sp0.z + u*sp1.z + v*sp2.z;
                float prev = ScreenDistances[idx];
                if (prev != 0.0f && depth >= prev) continue;
                ScreenDistances[idx] = depth;

                int i3 = idx * 3;
                float3 nrm = normalize(fn);
                ScreenNormals[i3  ] = nrm.x;
                ScreenNormals[i3+1] = nrm.y;
                ScreenNormals[i3+2] = nrm.z;

                float intensity = max(0.65f, dot(nrm, (float3)(0.3f,0.7f,0.5f)));
                float3 finalCol = clamp(tc * intensity + tc * Em, 0.0f, 1.0f);
                ScreenColors[i3  ] = finalCol.x;
                ScreenColors[i3+1] = finalCol.y;
                ScreenColors[i3+2] = finalCol.z;

                ScreenMaterialRoughness[idx] = Rg;
                ScreenMaterialMetallic[idx]  = Mt;
                ScreenMaterialEmission[idx]  = Em;
            }
        }
    }
}

__kernel void copyToGLTexture(
    __global float* screen_colors,
    __write_only image2d_t gl_texture,
    int screen_width,
    int screen_height,
    int mode // 0 = 3x float, 1 = 4x float, 2 = 1x float
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screen_width || y >= screen_height) return;
    
    if (mode == 0) {
        // 3x float mode
        int idx = (y * screen_width + x) * 3; // Index for a float array
        float3 color = (float3)(screen_colors[idx], screen_colors[idx+1], screen_colors[idx+2]);
        if (length(color) == 0.0f) { 
            // write transparent black for zero color
            write_imagef(gl_texture, (int2)(x, y), (float4)(0.0f, 0.0f, 0.0f, 0.0f));
            return;
        }
        write_imagef(gl_texture, (int2)(x, y), (float4)(color.x, color.y, color.z, 1.0f));
        return;
    } else if (mode == 1) {
        // 4x float mode
        int idx = (y * screen_width + x) * 4; // Index for a float array
        float4 color = (float4)(screen_colors[idx], screen_colors[idx+1], screen_colors[idx+2], screen_colors[idx+3]);
        write_imagef(gl_texture, (int2)(x, y), color);
        return;
    } else if (mode == 2) {
        // 1x float mode
        int idx = y * screen_width + x; // Index for a single float
        float gray = screen_colors[idx];
        write_imagef(gl_texture, (int2)(x, y), (float4)(gray, gray, gray, 1.0f));
        return;
    }
}

typedef struct {
    float edge_strength;
    float2 edge_direction;
    bool is_silhouette;
    float luminance_contrast;
    float alpha_variance;
} EdgeInfo;

EdgeInfo analyzeEdge(
    const int x,
    const int y,
    __global const float* input_colors,
    __global const float* input_distances,
    __global const float* input_normals,
    const int screen_width,
    const int screen_height,
    const int mode,
    const float center_distance,
    const float3 center_normal,
    const float4 center_color_full
) {
    EdgeInfo edge_info;
    edge_info.edge_strength = 0.0f;
    edge_info.edge_direction = (float2)(0.0f, 0.0f);
    edge_info.is_silhouette = false;
    edge_info.luminance_contrast = 0.0f;
    edge_info.alpha_variance = 0.0f;
    
    const float normal_threshold = 0.92f;
    const float depth_base_threshold = 0.02f;
    const float depth_scale_factor = 0.008f;
    const float luminance_threshold = 0.15f;
    const float alpha_threshold = 0.1f;
    
    float center_luminance = dot(center_color_full.xyz, (float3)(0.299f, 0.587f, 0.114f));
    
    float2 gradient_normal = (float2)(0.0f, 0.0f);
    float2 gradient_depth = (float2)(0.0f, 0.0f);
    float2 gradient_luminance = (float2)(0.0f, 0.0f);
    float max_alpha_diff = 0.0f;
    float max_luminance_diff = 0.0f;
    
    int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    int offsets_x[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int offsets_y[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    
    for (int i = 0; i < 9; i++) {
        int nx = x + offsets_x[i];
        int ny = y + offsets_y[i];
        
        if (nx >= 0 && nx < screen_width && ny >= 0 && ny < screen_height) {
            int neighbor_idx = ny * screen_width + nx;
            float neighbor_distance = input_distances[neighbor_idx];
            
            if (neighbor_distance > 0.001f) {
                float3 neighbor_normal = vload3(0, &input_normals[neighbor_idx * 3]);
                float4 neighbor_color_full;
                
                if (mode == 0) {
                    float3 neighbor_color = vload3(0, &input_colors[neighbor_idx * 3]);
                    neighbor_color_full = (float4)(neighbor_color.x, neighbor_color.y, neighbor_color.z, 1.0f);
                } else if (mode == 1) {
                    neighbor_color_full = vload4(0, &input_colors[neighbor_idx * 4]);
                } else {
                    float gray = input_colors[neighbor_idx];
                    neighbor_color_full = (float4)(gray, gray, gray, 1.0f);
                }
                
                if (length(neighbor_normal) > 0.001f) {
                    neighbor_normal = normalize(neighbor_normal);
                    
                    float normal_similarity = dot(center_normal, neighbor_normal);
                    float depth_diff = fabs(neighbor_distance - center_distance);
                    float neighbor_luminance = dot(neighbor_color_full.xyz, (float3)(0.299f, 0.587f, 0.114f));
                    float luminance_diff = fabs(center_luminance - neighbor_luminance);
                    float alpha_diff = fabs(center_color_full.w - neighbor_color_full.w);
                    
                    if (normal_similarity < normal_threshold) {
                        edge_info.is_silhouette = true;
                        edge_info.edge_strength = max(edge_info.edge_strength, 1.0f - normal_similarity);
                    }
                    
                    float adaptive_depth_threshold = depth_base_threshold + center_distance * depth_scale_factor;
                    if (depth_diff > adaptive_depth_threshold) {
                        edge_info.edge_strength = max(edge_info.edge_strength, 
                                                    min(1.0f, depth_diff / adaptive_depth_threshold));
                    }
                    
                    if (luminance_diff > luminance_threshold) {
                        edge_info.edge_strength = max(edge_info.edge_strength, 
                                                    min(1.0f, luminance_diff / luminance_threshold));
                        max_luminance_diff = max(max_luminance_diff, luminance_diff);
                    }
                    
                    if (alpha_diff > alpha_threshold) {
                        max_alpha_diff = max(max_alpha_diff, alpha_diff);
                    }
                    
                    float sobel_weight_x = (float)sobel_x[i] / 8.0f;
                    float sobel_weight_y = (float)sobel_y[i] / 8.0f;
                    
                    gradient_depth += (float2)(sobel_weight_x, sobel_weight_y) * depth_diff;
                    gradient_luminance += (float2)(sobel_weight_x, sobel_weight_y) * luminance_diff;
                    
                    float normal_diff = 1.0f - normal_similarity;
                    gradient_normal += (float2)(sobel_weight_x, sobel_weight_y) * normal_diff;
                }
            } else {
                edge_info.edge_strength = 1.0f;
                edge_info.is_silhouette = true;
            }
        }
    }
    
    float2 combined_gradient = gradient_normal + gradient_depth * 0.5f + gradient_luminance * 0.3f;
    edge_info.edge_direction = length(combined_gradient) > 0.001f ? normalize(combined_gradient) : (float2)(0.0f, 0.0f);
    edge_info.luminance_contrast = max_luminance_diff;
    edge_info.alpha_variance = max_alpha_diff;
    
    return edge_info;
}

bool detectEdge(
    const int x,
    const int y,
    __global const float* input_colors,
    __global const float* input_distances,
    __global const float* input_normals,
    const int screen_width,
    const int screen_height,
    const int mode,
    const float center_distance,
    const float3 center_normal,
    const float4 center_color_full
) {
    const float normal_threshold = 0.92f;
    const float depth_base_threshold = 0.02f;
    const float depth_scale_factor = 0.008f;
    const float luminance_threshold = 0.08f;
    const float color_threshold = 0.12f;
    const float alpha_threshold = 0.05f;
    
    float center_luminance = dot(center_color_full.xyz, (float3)(0.299f, 0.587f, 0.114f));
    
    int neighbors[8] = {-1, 0, 1, 0, 0, -1, 0, 1};
    
    for (int i = 0; i < 8; i += 2) {
        int nx = x + neighbors[i];
        int ny = y + neighbors[i + 1];
        
        if (nx >= 0 && nx < screen_width && ny >= 0 && ny < screen_height) {
            int neighbor_idx = ny * screen_width + nx;
            float neighbor_distance = input_distances[neighbor_idx];
            
            if (neighbor_distance > 0.001f) {
                float3 neighbor_normal = vload3(0, &input_normals[neighbor_idx * 3]);
                
                if (length(neighbor_normal) > 0.001f) {
                    neighbor_normal = normalize(neighbor_normal);
                    
                    float4 neighbor_color_full;
                    if (mode == 0) {
                        float3 neighbor_color = vload3(0, &input_colors[neighbor_idx * 3]);
                        neighbor_color_full = (float4)(neighbor_color.x, neighbor_color.y, neighbor_color.z, 1.0f);
                    } else if (mode == 1) {
                        neighbor_color_full = vload4(0, &input_colors[neighbor_idx * 4]);
                    } else {
                        float gray = input_colors[neighbor_idx];
                        neighbor_color_full = (float4)(gray, gray, gray, 1.0f);
                    }
                    
                    float normal_similarity = dot(center_normal, neighbor_normal);
                    if (normal_similarity < normal_threshold) {
                        return true;
                    }
                    
                    float depth_diff = fabs(neighbor_distance - center_distance);
                    float adaptive_depth_threshold = depth_base_threshold + 
                                                   center_distance * depth_scale_factor;
                    
                    if (depth_diff > adaptive_depth_threshold) {
                        return true;
                    }
                    
                    float neighbor_luminance = dot(neighbor_color_full.xyz, (float3)(0.299f, 0.587f, 0.114f));
                    float luminance_diff = fabs(center_luminance - neighbor_luminance);
                    
                    if (luminance_diff > luminance_threshold) {
                        return true;
                    }
                    
                    float3 color_diff = center_color_full.xyz - neighbor_color_full.xyz;
                    float color_distance = length(color_diff);
                    
                    if (color_distance > color_threshold) {
                        return true;
                    }
                    
                    float alpha_diff = fabs(center_color_full.w - neighbor_color_full.w);
                    
                    if (alpha_diff > alpha_threshold) {
                        return true;
                    }
                }
            } else {
                return true;
            }
        }
    }
    
    return false;
}

float4 sampleSubpixel(
    const int x,
    const int y,
    __global const float* input_colors,
    const int screen_width,
    const int screen_height,
    const int mode,
    const float2 offset
) {
    float fx = (float)x + offset.x;
    float fy = (float)y + offset.y;
    
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = min(x0 + 1, screen_width - 1);
    int y1 = min(y0 + 1, screen_height - 1);
    
    x0 = max(x0, 0);
    y0 = max(y0, 0);
    
    float wx = fx - (float)x0;
    float wy = fy - (float)y0;
    
    float4 c00, c10, c01, c11;
    
    if (mode == 0) {
        float3 color00 = vload3(0, &input_colors[(y0 * screen_width + x0) * 3]);
        float3 color10 = vload3(0, &input_colors[(y0 * screen_width + x1) * 3]);
        float3 color01 = vload3(0, &input_colors[(y1 * screen_width + x0) * 3]);
        float3 color11 = vload3(0, &input_colors[(y1 * screen_width + x1) * 3]);
        c00 = (float4)(color00.x, color00.y, color00.z, 1.0f);
        c10 = (float4)(color10.x, color10.y, color10.z, 1.0f);
        c01 = (float4)(color01.x, color01.y, color01.z, 1.0f);
        c11 = (float4)(color11.x, color11.y, color11.z, 1.0f);
    } else if (mode == 1) {
        c00 = vload4(0, &input_colors[(y0 * screen_width + x0) * 4]);
        c10 = vload4(0, &input_colors[(y0 * screen_width + x1) * 4]);
        c01 = vload4(0, &input_colors[(y1 * screen_width + x0) * 4]);
        c11 = vload4(0, &input_colors[(y1 * screen_width + x1) * 4]);
    } else {
        float g00 = input_colors[y0 * screen_width + x0];
        float g10 = input_colors[y0 * screen_width + x1];
        float g01 = input_colors[y1 * screen_width + x0];
        float g11 = input_colors[y1 * screen_width + x1];
        c00 = (float4)(g00, g00, g00, 1.0f);
        c10 = (float4)(g10, g10, g10, 1.0f);
        c01 = (float4)(g01, g01, g01, 1.0f);
        c11 = (float4)(g11, g11, g11, 1.0f);
    }
    
    float4 top = mix(c00, c10, wx);
    float4 bottom = mix(c01, c11, wx);
    return mix(top, bottom, wy);
}

void performAdvancedEdgeSmoothing(
    const int x,
    const int y,
    __global float* input_colors,
    __global float* input_distances,
    __global float* input_normals,
    const int screen_width,
    const int screen_height,
    const int mode,
    const float4 center_color_full,
    const float center_distance,
    const float3 center_normal,
    const EdgeInfo edge_info
) {
    float4 accum_color = center_color_full * 4.0f;
    float3 accum_normal = center_normal * 4.0f;
    float accum_distance = center_distance * 4.0f;
    float total_weight = 4.0f;
    
    int kernel_size = edge_info.is_silhouette ? 2 : 1;
    float edge_strength_factor = 1.0f + edge_info.edge_strength * 0.5f;
    
    float2 perpendicular_dir = (float2)(-edge_info.edge_direction.y, edge_info.edge_direction.x);
    
    float subpixel_offsets[12] = {
        -0.375f, -0.125f,  0.125f, -0.375f,  0.375f, -0.125f,
        -0.125f,  0.125f,  0.125f,  0.125f,  0.375f,  0.375f
    };
    
    for (int sample = 0; sample < 6; sample++) {
        float2 sample_offset = (float2)(subpixel_offsets[sample * 2], subpixel_offsets[sample * 2 + 1]);
        
        if (edge_info.edge_strength > 0.3f) {
            sample_offset += perpendicular_dir * (sample - 2.5f) * 0.2f * edge_info.edge_strength;
        }
        
        float4 subpixel_color = sampleSubpixel(x, y, input_colors, screen_width, screen_height, mode, sample_offset);
        float subpixel_weight = 0.8f + 0.4f * cos((float)sample * 1.047f);
        
        accum_color += subpixel_color * subpixel_weight;
        total_weight += subpixel_weight;
    }
    
    for (int dy = -kernel_size; dy <= kernel_size; dy++) {
        for (int dx = -kernel_size; dx <= kernel_size; dx++) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < screen_width && ny >= 0 && ny < screen_height) {
                int neighbor_idx = ny * screen_width + nx;
                float neighbor_distance = input_distances[neighbor_idx];
                
                if (neighbor_distance > 0.001f) {
                    float3 neighbor_normal = vload3(0, &input_normals[neighbor_idx * 3]);
                    
                    if (length(neighbor_normal) > 0.001f) {
                        neighbor_normal = normalize(neighbor_normal);
                        
                        float4 neighbor_color_full;
                        if (mode == 0) {
                            float3 neighbor_color = vload3(0, &input_colors[neighbor_idx * 3]);
                            neighbor_color_full = (float4)(neighbor_color.x, neighbor_color.y, neighbor_color.z, 1.0f);
                        } else if (mode == 1) {
                            neighbor_color_full = vload4(0, &input_colors[neighbor_idx * 4]);
                        } else {
                            float gray = input_colors[neighbor_idx];
                            neighbor_color_full = (float4)(gray, gray, gray, 1.0f);
                        }
                        
                        float normal_similarity = dot(center_normal, neighbor_normal);
                        float depth_diff = fabs(neighbor_distance - center_distance);
                        float max_depth_diff = center_distance * 0.08f + 0.04f;
                        
                        float color_distance = length(neighbor_color_full.xyz - center_color_full.xyz);
                        float alpha_diff = fabs(neighbor_color_full.w - center_color_full.w);
                        
                        float similarity_threshold = edge_info.is_silhouette ? 0.2f : 0.6f;
                        float color_threshold = edge_info.luminance_contrast > 0.2f ? 0.8f : 0.4f;
                        
                        if (normal_similarity > similarity_threshold && 
                            depth_diff < max_depth_diff &&
                            color_distance < color_threshold &&
                            alpha_diff < 0.3f) {
                            
                            float spatial_weight = 1.0f / (1.0f + (float)(dx*dx + dy*dy));
                            float similarity_weight = normal_similarity * normal_similarity;
                            float color_weight = 1.0f / (1.0f + color_distance * 2.0f);
                            float alpha_weight = 1.0f / (1.0f + alpha_diff * 5.0f);
                            
                            float final_weight = spatial_weight * similarity_weight * color_weight * alpha_weight * edge_strength_factor;
                            
                            if (edge_info.is_silhouette && normal_similarity < 0.7f) {
                                final_weight *= 0.3f;
                            }
                            
                            accum_color += neighbor_color_full * final_weight;
                            accum_normal += neighbor_normal * final_weight;
                            accum_distance += neighbor_distance * final_weight;
                            total_weight += final_weight;
                        }
                    }
                }
            }
        }
    }
    
    if (total_weight > 0.1f) {
        accum_color /= total_weight;
        accum_normal = normalize(accum_normal / total_weight);
        accum_distance /= total_weight;
        
        accum_color = clamp(accum_color, 0.0f, 4.0f);
        
        if (edge_info.luminance_contrast > 0.1f) {
            float contrast_boost = 1.0f + edge_info.luminance_contrast * 0.1f;
            accum_color.xyz = mix(accum_color.xyz, center_color_full.xyz, 0.2f) * contrast_boost;
        }
        
        if (mode == 0) {
            int color_idx = (y * screen_width + x) * 3;
            vstore3(accum_color.xyz, 0, &input_colors[color_idx]);
        } else if (mode == 1) {
            int color_idx = (y * screen_width + x) * 4;
            vstore4(accum_color, 0, &input_colors[color_idx]);
        } else if (mode == 2) {
            int color_idx = y * screen_width + x;
            input_colors[color_idx] = (accum_color.x + accum_color.y + accum_color.z) / 3.0f;
        }
        
        int normal_idx = (y * screen_width + x) * 3;
        vstore3(accum_normal, 0, &input_normals[normal_idx]);
        input_distances[y * screen_width + x] = accum_distance;
    }
}

void performEdgeSmoothing(
    const int x,
    const int y,
    __global float* input_colors,
    __global float* input_distances,
    __global float* input_normals,
    const int screen_width,
    const int screen_height,
    const int mode,
    const float3 center_color,
    const float center_distance,
    const float3 center_normal
) {
    float3 accum_color = center_color * 2.0f;
    float3 accum_normal = center_normal * 2.0f;
    float accum_distance = center_distance * 2.0f;
    float total_weight = 2.0f;
    
    int offsets[16] = {-1, -1, 0, -1, 1, -1, -1, 0, 1, 0, -1, 1, 0, 1, 1, 1};
    float weights[8] = {0.7f, 1.0f, 0.7f, 1.0f, 1.0f, 0.7f, 1.0f, 0.7f};
    
    for (int i = 0; i < 16; i += 2) {
        int nx = x + offsets[i];
        int ny = y + offsets[i + 1];
        
        if (nx >= 0 && nx < screen_width && ny >= 0 && ny < screen_height) {
            int neighbor_idx = ny * screen_width + nx;
            float neighbor_distance = input_distances[neighbor_idx];
            
            if (neighbor_distance > 0.001f) {
                float3 neighbor_normal = vload3(0, &input_normals[neighbor_idx * 3]);
                
                if (length(neighbor_normal) > 0.001f) {
                    neighbor_normal = normalize(neighbor_normal);
                    
                    float normal_similarity = dot(center_normal, neighbor_normal);
                    float depth_diff = fabs(neighbor_distance - center_distance);
                    float max_depth_diff = center_distance * 0.05f + 0.03f;
                    
                    if (normal_similarity > 0.4f && depth_diff < max_depth_diff) {
                        float weight = weights[i / 2] * normal_similarity;
                        
                        if (mode == 0) {
                            float3 neighbor_color = vload3(0, &input_colors[neighbor_idx * 3]);
                            accum_color += neighbor_color * weight;
                        } else if (mode == 1) {
                            float4 neighbor_color4 = vload4(0, &input_colors[neighbor_idx * 4]);
                            accum_color += neighbor_color4.xyz * weight;
                        } else if (mode == 2) {
                            float neighbor_gray = input_colors[neighbor_idx];
                            accum_color += (float3)(neighbor_gray, neighbor_gray, neighbor_gray) * weight;
                        }
                        
                        accum_normal += neighbor_normal * weight;
                        accum_distance += neighbor_distance * weight;
                        total_weight += weight;
                    }
                }
            }
        }
    }
    
    if (total_weight > 0.1f) {
        accum_color /= total_weight;
        accum_normal = normalize(accum_normal / total_weight);
        accum_distance /= total_weight;
        
        if (mode == 0) {
            int color_idx = (y * screen_width + x) * 3;
            vstore3(accum_color, 0, &input_colors[color_idx]);
        } else if (mode == 1) {
            int color_idx = (y * screen_width + x) * 4;
            float4 orig_color4 = vload4(0, &input_colors[color_idx]);
            vstore4((float4)(accum_color.x, accum_color.y, accum_color.z, orig_color4.w), 0, &input_colors[color_idx]);
        } else if (mode == 2) {
            int color_idx = y * screen_width + x;
            input_colors[color_idx] = (accum_color.x + accum_color.y + accum_color.z) / 3.0f;
        }
        
        int normal_idx = (y * screen_width + x) * 3;
        vstore3(accum_normal, 0, &input_normals[normal_idx]);
        input_distances[y * screen_width + x] = accum_distance;
    }
}

__kernel void antiAlias(
    __global float* input_colors,
    __global float* input_distances,
    int screen_width,
    int screen_height,
    int mode,
    __global float* input_normals,
    int const use_advanced
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screen_width || y >= screen_height) return;
    
    int pixel_idx = y * screen_width + x;
    float center_distance = input_distances[pixel_idx];
    
    if (center_distance <= 0.001f) return;
    
    float3 center_normal = vload3(0, &input_normals[pixel_idx * 3]);
    if (length(center_normal) < 0.001f) return;
    
    center_normal = normalize(center_normal);
    
    float3 center_color;
    if (mode == 0) {
        center_color = vload3(0, &input_colors[pixel_idx * 3]);
    } else if (mode == 1) {
        float4 color4 = vload4(0, &input_colors[pixel_idx * 4]);
        center_color = color4.xyz;
    } else if (mode == 2) {
        float gray = input_colors[pixel_idx];
        center_color = (float3)(gray, gray, gray);
    }
    
    if (length(center_color) < 0.001f) return;
    
    float4 center_color_full;
    if (mode == 0) {
        center_color_full = (float4)(center_color.x, center_color.y, center_color.z, 1.0f);
    } else if (mode == 1) {
        float4 color4 = vload4(0, &input_colors[pixel_idx * 4]);
        center_color_full = color4;
    } else {
        float gray = input_colors[pixel_idx];
        center_color_full = (float4)(gray, gray, gray, 1.0f);
    }

    bool is_edge = detectEdge(x, y, input_colors, input_distances, input_normals, 
                             screen_width, screen_height, mode, center_distance, center_normal, center_color_full);

    if (!is_edge) return;
    
    if (use_advanced != 0) {
        EdgeInfo edge_info = analyzeEdge(x, y, input_colors, input_distances, input_normals, 
                                        screen_width, screen_height, mode, center_distance, center_normal, center_color_full);
        performAdvancedEdgeSmoothing(x, y, input_colors, input_distances, input_normals, 
                                    screen_width, screen_height, mode, center_color_full, center_distance, center_normal, edge_info);
    } else {
        performEdgeSmoothing(x, y, input_colors, input_distances, input_normals, 
                            screen_width, screen_height, mode, center_color, center_distance, center_normal);
    }
}

__kernel void renderFire(
    __global const float* posX,
    __global const float* posY,
    __global const float* posZ,
    __global const float* velX,
    __global const float* velY,
    __global const float* velZ,
    __global const float* lifeTime,
    
    const float3 baseColor,
    const float3 fireColor,
    const float3 smokeColor,
    
    const float maxLifeTime,
    const float maxVelocity,
    const float maxDepth,
    
    const float3 camPos,
    const float3 camDir,
    const float3 camUp,
    const float fov,
    
    const int screenWidth,
    const int screenHeight,
    
    __global float *ScreenDistances,
    __global float *ScreenColors,
    __global float *ScreenNormals,
    __global float *ScreenAlphas,
    
    const int numPoints,
    const int ParticleRadius
) {
    int i = get_global_id(0);
    if (i >= numPoints) return;

    float3 point = (float3)(posX[i], posY[i], posZ[i]);
    float3 velocity = (float3)(velX[i], velY[i], velZ[i]);
    float velMagnitude = length(velocity);
    float velNormalized = min(1.0f, velMagnitude / sqrt(maxVelocity));
    float lifeRatio = lifeTime[i] / maxLifeTime;
    
    float3 forward = normalize(camDir);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);

    float3 relativePoint = point - camPos;
    float dotProduct = dot(relativePoint, forward);
    if (dotProduct <= 0.001f) return;
    
    float fovScale = 1.0f / (dotProduct * fov);
    float screenRight = dot(relativePoint, right) * fovScale;
    float screenUp = dot(relativePoint, up) * fovScale;
    
    float halfWidth = screenWidth * 0.5f;
    float halfHeight = screenHeight * 0.5f;
    
    int screenX = (int)(screenRight * halfWidth + halfWidth);
    int screenY = (int)(-screenUp * halfHeight + halfHeight);
    
    if (screenX < 0 || screenX >= screenWidth || screenY < 0 || screenY >= screenHeight) return;

    float distance = length(relativePoint);
    
    float randomSeed = (float)i * 0.12345f;
    float random1 = fract(sin(randomSeed * 12.9898f) * 43758.5453f);
    float random2 = fract(sin(randomSeed * 78.233f) * 43758.5453f);
    float random3 = fract(sin(randomSeed * 45.678f) * 43758.5453f);
    float random4 = fract(sin(randomSeed * 91.234f) * 43758.5453f);
    
    float baseRandomSize = 0.7f + random4 * 0.6f;
    
    float sizeMultiplier = baseRandomSize;
    if (lifeRatio > 0.4f) {
        float smokePhase = (lifeRatio - 0.4f) / 0.6f;
        sizeMultiplier = baseRandomSize * (1.0f + smokePhase * 2.0f);
    }
    
    float particleRadiusBasedOnDistance = (float)ParticleRadius * sizeMultiplier / dotProduct;
    int radiusInt = max(1, (int)particleRadiusBasedOnDistance);
    int radiusSquared = radiusInt * radiusInt;
    
    float3 particleColor;
    float emissionBoost;
    
    float3 hotWhite = (float3)(1.5f, 1.4f, 1.0f);
    float3 brightOrange = baseColor * 1.3f;
    float3 darkRed = fireColor;
    float3 darkSmoke = smokeColor;

    float decayRate = 3.5f;
    float decayFactor = exp(-decayRate * lifeRatio);

    if (lifeRatio < 0.15f) {
        float t = 1.0f - exp(-8.0f * lifeRatio / 0.15f);
        particleColor = mix(hotWhite, brightOrange, t);
        emissionBoost = 3.5f * decayFactor + 0.5f;
    } else if (lifeRatio < 0.45f) {
        float normalizedLife = (lifeRatio - 0.15f) / 0.3f;
        float t = 1.0f - exp(-4.0f * normalizedLife);
        particleColor = mix(brightOrange, darkRed, t);
        emissionBoost = (2.8f - 1.5f * normalizedLife) * decayFactor + 0.3f;
    } else {
        float normalizedLife = (lifeRatio - 0.45f) / 0.55f;
        float t = 1.0f - exp(-2.5f * normalizedLife);
        particleColor = mix(darkRed, darkSmoke, t);
        emissionBoost = 1.0f * exp(-5.0f * normalizedLife) + 0.1f;
    }
    
    float3 colorVariation = (float3)(
        random1 * 0.3f,
        random2 * 0.2f - 0.1f,
        random3 * 0.15f - 0.1f
    );
    particleColor = clamp(particleColor + colorVariation, 0.0f, 3.0f);
    particleColor *= (1.0f + velNormalized * 0.5f);
    
    // Calculate base opacity
    float opacity;
    if (lifeRatio < 0.08f) {
        opacity = lifeRatio / 0.08f;
    } else if (lifeRatio > 0.7f) {
        opacity = (1.0f - lifeRatio) / 0.3f;
    } else {
        opacity = 1.0f;
    }
    
    if (lifeRatio > 0.5f) {
        float smokeAmount = (lifeRatio - 0.5f) / 0.5f;
        opacity *= (1.0f - smokeAmount * 0.8f);
        emissionBoost *= (1.0f - smokeAmount * 0.9f);
    }

    for (int dy = -radiusInt; dy <= radiusInt; dy++) {
        int offsetY = screenY + dy;
        if (offsetY < 0 || offsetY >= screenHeight) continue;
        
        int dy2 = dy * dy;
        if (dy2 > radiusSquared) continue;
        
        int maxDx = (int)sqrt((float)(radiusSquared - dy2));
        
        for (int dx = -maxDx; dx <= maxDx; dx++) {
            int offsetX = screenX + dx;
            if (offsetX < 0 || offsetX >= screenWidth) continue;
            
            int offsetIndex = offsetY * screenWidth + offsetX;
            int r2 = dx*dx + dy*dy;
            if (r2 > radiusSquared) continue;
            
            float normalizedR2 = (float)r2 / (float)radiusSquared;
            float sphereDepth = sqrt(max(0.0f, 1.0f - normalizedR2));
            float depthOffset = sphereDepth * particleRadiusBasedOnDistance;
            float surfaceDistance = max(0.001f, distance - depthOffset);
            
            float edgeFalloff = pow(sphereDepth, 0.4f);
            float pixelOpacity = opacity * edgeFalloff;
            
            float centerGlow = pow(1.0f - normalizedR2, 0.5f);
            
            float existingDepth = ScreenDistances[offsetIndex];
            
            int colorBase = offsetIndex * 3;
            float3 existingColor = vload3(0, &ScreenColors[colorBase]);
            
            float finalEmission = emissionBoost * pixelOpacity * centerGlow;
            float3 emittedColor = particleColor * finalEmission;
            float3 newColor = clamp(existingColor + emittedColor, 0.0f, 5.0f);
            
            vstore3(newColor, 0, &ScreenColors[colorBase]);
            
            // Store alpha value (max of existing and new)
            float existingAlpha = ScreenAlphas[offsetIndex];
            ScreenAlphas[offsetIndex] = max(existingAlpha, pixelOpacity * centerGlow);
            
            if (existingDepth == 0.0f || surfaceDistance < existingDepth) {
                ScreenDistances[offsetIndex] = surfaceDistance;
                
                float3 normal = normalize((float3)(dx / (float)radiusInt, dy / (float)radiusInt, sphereDepth));
                vstore3(normal, 0, &ScreenNormals[offsetIndex * 3]);
            }
        }
    }
}

__kernel void blurFire(
    __global const float* InputColors,
    __global const float* InputDistances,
    __global const float* InputAlphas,
    __global float* OutputColors,
    __global float* OutputDistances,
    __global float* OutputAlphas,
    const int screenWidth,
    const int screenHeight,
    const int blurRadius,
    const float sigmaColor,
    const float sigmaSpace
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    int centerIdx = y * screenWidth + x;
    float centerDepth = InputDistances[centerIdx];
    float centerAlpha = InputAlphas[centerIdx];
    
    // Skip blurring if alpha is negative (solid object like missile)
    if (centerAlpha < 0.0f) {
        int colorBase = centerIdx * 3;
        OutputColors[colorBase + 0] = InputColors[colorBase + 0];
        OutputColors[colorBase + 1] = InputColors[colorBase + 1];
        OutputColors[colorBase + 2] = InputColors[colorBase + 2];
        OutputDistances[centerIdx] = centerDepth;
        OutputAlphas[centerIdx] = centerAlpha;
        return;
    }
    
    if (centerDepth <= 0.001f) {
        int colorBase = centerIdx * 3;
        OutputColors[colorBase + 0] = InputColors[colorBase + 0];
        OutputColors[colorBase + 1] = InputColors[colorBase + 1];
        OutputColors[colorBase + 2] = InputColors[colorBase + 2];
        OutputDistances[centerIdx] = 0.0f;
        OutputAlphas[centerIdx] = 0.0f;
        return;
    }
    
    float3 centerColor = vload3(0, &InputColors[centerIdx * 3]);
    float centerBrightness = (centerColor.x + centerColor.y + centerColor.z) / 3.0f;
    
    float brightnessThreshold = 0.5f;
    float blurStrength;
    if (centerBrightness > brightnessThreshold) {
        blurStrength = 1.0f - (centerBrightness - brightnessThreshold) / (5.0f - brightnessThreshold);
        blurStrength = clamp(blurStrength, 0.1f, 1.0f);
    } else {
        blurStrength = 1.0f;
    }
    
    float3 accumulatedColor = (float3)(0.0f, 0.0f, 0.0f);
    float accumulatedAlpha = 0.0f;
    float totalWeight = 0.0f;
    float sigma = (float)blurRadius / 2.0f;
    
    int effectiveRadius = (int)((float)blurRadius * blurStrength);
    effectiveRadius = max(1, effectiveRadius);
    
    for (int dy = -effectiveRadius; dy <= effectiveRadius; dy++) {
        for (int dx = -effectiveRadius; dx <= effectiveRadius; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx < 0 || nx >= screenWidth || ny < 0 || ny >= screenHeight) continue;
            
            int neighborIdx = ny * screenWidth + nx;
            float neighborDepth = InputDistances[neighborIdx];
            float neighborAlpha = InputAlphas[neighborIdx];
            
            // Skip solid objects (negative alpha) in blur sampling
            if (neighborDepth <= 0.001f || neighborAlpha < 0.0f) continue;
            
            float3 neighborColor = vload3(0, &InputColors[neighborIdx * 3]);
            float neighborBrightness = (neighborColor.x + neighborColor.y + neighborColor.z) / 3.0f;
            
            float brightnessDiff = fabs(centerBrightness - neighborBrightness);
            float brightnessWeight = exp(-brightnessDiff * brightnessDiff / 0.5f);
            
            float dist = (float)(dx * dx + dy * dy);
            float spatialWeight = exp(-dist / (2.0f * sigma * sigma));
            
            float alphaDiff = fabs(centerAlpha - neighborAlpha);
            float alphaWeight = exp(-alphaDiff * alphaDiff / (2.0f * sigmaColor * sigmaColor));
            
            float weight = spatialWeight * brightnessWeight * alphaWeight;
            
            accumulatedColor += neighborColor * weight;
            accumulatedAlpha += neighborAlpha * weight;
            totalWeight += weight;
        }
    }
    
    if (totalWeight > 0.001f) {
        float3 blurredColor = accumulatedColor / totalWeight;
        float blurredAlpha = accumulatedAlpha / totalWeight;
        
        float blendFactor = 1.0f - blurStrength;
        float3 finalColor = mix(blurredColor, centerColor, blendFactor);
        float finalAlpha = mix(blurredAlpha, centerAlpha, blendFactor);
        
        finalColor = clamp(finalColor, 0.0f, 5.0f);
        finalAlpha = clamp(finalAlpha, 0.0f, 1.0f);
        
        vstore3(finalColor, 0, &OutputColors[centerIdx * 3]);
        OutputDistances[centerIdx] = centerDepth;
        OutputAlphas[centerIdx] = finalAlpha;
    } else {
        vstore3(centerColor, 0, &OutputColors[centerIdx * 3]);
        OutputDistances[centerIdx] = centerDepth;
        OutputAlphas[centerIdx] = centerAlpha;
    }
}

__kernel void clearColorBuffer(
    __global float* colors,
    const float3 backgroundColor,
    const int width,
    const int height
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 3;
    colors[idx + 0] = backgroundColor.x;
    colors[idx + 1] = backgroundColor.y;
    colors[idx + 2] = backgroundColor.z;
}

__kernel void compositeBuffers(
    // Input buffer 1
    __global const float* InputColors1,
    __global const float* InputDistances1,
    __global const float* InputNormals1,
    __global const float* InputAlphas1,
    const int useAlpha1,
    
    // Input buffer 2
    __global const float* InputColors2,
    __global const float* InputDistances2,
    __global const float* InputNormals2,
    __global const float* InputAlphas2,
    const int useAlpha2,
    
    // Output buffers
    __global float* OutputColors,
    __global float* OutputDistances,
    __global float* OutputNormals,
    
    const int screenWidth,
    const int screenHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    int pixelIdx = y * screenWidth + x;
    int colorIdx = pixelIdx * 3;
    
    float depth1 = InputDistances1[pixelIdx];
    float depth2 = InputDistances2[pixelIdx];
    float alpha1 = useAlpha1 ? InputAlphas1[pixelIdx] : 1.0f;
    float alpha2 = useAlpha2 ? InputAlphas2[pixelIdx] : 1.0f;
    
    // Store original alpha signs for special handling
    bool alpha1_is_solid = (alpha1 < 0.0f);
    bool alpha2_is_solid = (alpha2 < 0.0f);
    
    // Treat negative alpha values as fully opaque (solid objects like missiles)
    if (alpha1_is_solid) alpha1 = 1.0f;
    if (alpha2_is_solid) alpha2 = 1.0f;
    
    float3 color1 = vload3(0, &InputColors1[colorIdx]);
    float3 color2 = vload3(0, &InputColors2[colorIdx]);
    float3 normal1 = vload3(0, &InputNormals1[colorIdx]);
    float3 normal2 = vload3(0, &InputNormals2[colorIdx]);
    
    bool valid1 = (depth1 > 0.001f);
    bool valid2 = (depth2 > 0.001f);
    
    float3 finalColor;
    float finalDepth;
    float3 finalNormal;
    
    // Case 1: Both buffers empty
    if (!valid1 && !valid2) {
        if (length(color1) > 0.001f) {
            vstore3(color1, 0, &OutputColors[colorIdx]);
            OutputDistances[pixelIdx] = 0.0f;
            vstore3((float3)(0.0f, 0.0f, 0.0f), 0, &OutputNormals[colorIdx]);
        } else {
            vstore3((float3)(0.0f, 0.0f, 0.0f), 0, &OutputColors[colorIdx]);
            OutputDistances[pixelIdx] = 0.0f;
            vstore3((float3)(0.0f, 0.0f, 0.0f), 0, &OutputNormals[colorIdx]);
        }
        return;
    }
    
    // Case 2: Only buffer 1 has data
    if (valid1 && !valid2) {
        if (useAlpha1 && alpha1 < 0.999f && !alpha1_is_solid) {
            finalColor = mix(color2, color1, alpha1);
            finalDepth = depth1;
            finalNormal = normal1;
        } else {
            finalColor = color1;
            finalDepth = depth1;
            finalNormal = normal1;
        }
    }
    // Case 3: Only buffer 2 has data
    else if (!valid1 && valid2) {
        if (useAlpha2 && alpha2 < 0.999f && !alpha2_is_solid) {
            finalColor = mix(color1, color2, alpha2);
            finalDepth = depth2;
            finalNormal = normal2;
        } else {
            finalColor = color2;
            finalDepth = depth2;
            finalNormal = normal2;
        }
    }
    // Case 4: Both buffers have data
    else {
        if (depth1 < depth2) {
            // Buffer 1 is in front
            if (useAlpha1 && alpha1 < 0.999f && !alpha1_is_solid) {
                finalColor = mix(color2, color1, alpha1);
                finalDepth = depth1;
                finalNormal = mix(normal2, normal1, alpha1);
            } else {
                finalColor = color1;
                finalDepth = depth1;
                finalNormal = normal1;
            }
        } else {
            // Buffer 2 is in front
            if (useAlpha2 && alpha2 < 0.999f && !alpha2_is_solid) {
                finalColor = mix(color1, color2, alpha2);
                finalDepth = depth2;
                finalNormal = mix(normal1, normal2, alpha2);
            } else {
                finalColor = color2;
                finalDepth = depth2;
                finalNormal = normal2;
            }
        }
    }
    
    // Write final composited result
    vstore3(finalColor, 0, &OutputColors[colorIdx]);
    OutputDistances[pixelIdx] = finalDepth;
    vstore3(finalNormal, 0, &OutputNormals[colorIdx]);
}

__kernel void renderMissile(
    // Missile model data
    __global const float* model_v1,
    __global const float* model_v2,
    __global const float* model_v3,
    __global const float* model_normals,
    __global const float* model_colors,
    __global const float* model_roughness,
    __global const float* model_metallic,
    __global const float* model_emission,
    const int model_triangle_count,
    
    // Missile instance data
    const float3 missile_position,
    const float3 missile_orientation,  // bodyOrientation[3] - forward direction
    const float missile_scale,
    
    // Camera and screen parameters
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight,
    
    // Output buffers
    __global float* ScreenDistances,
    __global float* ScreenColors,
    __global float* ScreenNormals,
    __global float* ScreenMaterialRoughness,
    __global float* ScreenMaterialMetallic,
    __global float* ScreenMaterialEmission,
    __global float *ScreenAlphas
) {
    int triangleId = get_global_id(0);
    if (triangleId >= model_triangle_count) return;

    // Load triangle vertices from model
    int idx = triangleId * 3;
    float3 model_p1 = (float3)(model_v1[idx], model_v1[idx + 1], model_v1[idx + 2]);
    float3 model_p2 = (float3)(model_v2[idx], model_v2[idx + 1], model_v2[idx + 2]);
    float3 model_p3 = (float3)(model_v3[idx], model_v3[idx + 1], model_v3[idx + 2]);
    float3 model_normal = (float3)(model_normals[idx], model_normals[idx + 1], model_normals[idx + 2]);

    // Create rotation matrix to align missile with bodyOrientation
    // Assume model's default forward is +X axis
    float3 model_forward = (float3)(1.0f, 0.0f, 0.0f);
    float3 target_forward = normalize(missile_orientation);
    
    // Calculate rotation axis and angle
    float3 rotation_axis = cross(model_forward, target_forward);
    float rotation_angle = acos(clamp(dot(model_forward, target_forward), -1.0f, 1.0f));
    
    // Build rotation matrix using Rodrigues' rotation formula
    // Represent rotation matrix as three float3 vectors (rows)
    float3 row0, row1, row2;
    if (length(rotation_axis) > 0.001f) {
        rotation_axis = normalize(rotation_axis);
        float c = cos(rotation_angle);
        float s = sin(rotation_angle);
        float t = 1.0f - c;
        float x = rotation_axis.x;
        float y = rotation_axis.y;
        float z = rotation_axis.z;
        
        // Construct 3x3 rotation matrix rows
        row0 = (float3)(t * x * x + c,      t * x * y - s * z,    t * x * z + s * y);
        row1 = (float3)(t * x * y + s * z,  t * y * y + c,        t * y * z - s * x);
        row2 = (float3)(t * x * z - s * y,  t * y * z + s * x,    t * z * z + c);
        
        // Transform vertices to world space
        float3 world_p1 = (float3)(
            dot(row0, model_p1),
            dot(row1, model_p1),
            dot(row2, model_p1)
        ) * missile_scale + missile_position;
        float3 world_p2 = (float3)(
            dot(row0, model_p2),
            dot(row1, model_p2),
            dot(row2, model_p2)
        ) * missile_scale + missile_position;
        float3 world_p3 = (float3)(
            dot(row0, model_p3),
            dot(row1, model_p3),
            dot(row2, model_p3)
        ) * missile_scale + missile_position;
        
        // Rotate normal (no translation, no scale)
        float3 world_normal = normalize((float3)(
            dot(row0, model_normal),
            dot(row1, model_normal),
            dot(row2, model_normal)
        ));
        
        // Backface culling
        float3 tri_center = (world_p1 + world_p2 + world_p3) / 3.0f;
        float3 to_camera = normalize(camPos - tri_center);
        if (dot(world_normal, to_camera) <= 0.0f) return;
        
        // Camera basis
        float3 forward = normalize(camDir);
        float3 up = (float3)(0.0f, 1.0f, 0.0f);
        float3 right = normalize(cross(forward, up));
        up = cross(right, forward);
        
        // Project vertices to screen space
        float3 vertices[3] = {world_p1, world_p2, world_p3};
        float3 projected[3];
        float minX = 1e9f, maxX = -1e9f, minY = 1e9f, maxY = -1e9f;
        
        for (int i = 0; i < 3; i++) {
            float3 rel = vertices[i] - camPos;
            float depth = dot(rel, forward);
            
            if (depth <= 0.01f) return;
            
            float scale = 1.0f / (depth * fov);
            float x = dot(rel, right) * scale;
            float y = dot(rel, up) * scale;
            
            float sx = x * screenWidth * 0.5f + screenWidth * 0.5f;
            float sy = -y * screenHeight * 0.5f + screenHeight * 0.5f;
            
            projected[i] = (float3)(sx, sy, depth);
            
            minX = fmin(minX, sx);
            maxX = fmax(maxX, sx);
            minY = fmin(minY, sy);
            maxY = fmax(maxY, sy);
        }
        
        // Check triangle area
        float area = fabs((projected[1].x - projected[0].x) * (projected[2].y - projected[0].y) - 
                         (projected[2].x - projected[0].x) * (projected[1].y - projected[0].y)) * 0.5f;
        if (area < 0.5f) return;
        
        // Clamp bounding box
        int x0 = max(0, (int)minX);
        int x1 = min(screenWidth - 1, (int)maxX);
        int y0 = max(0, (int)minY);
        int y1 = min(screenHeight - 1, (int)maxY);
        
        // Barycentric setup
        float2 v0 = projected[1].xy - projected[0].xy;
        float2 v1 = projected[2].xy - projected[0].xy;
        float d00 = dot(v0, v0);
        float d01 = dot(v0, v1);
        float d11 = dot(v1, v1);
        float invDenom = 1.0f / (d00 * d11 - d01 * d01);
        
        // Rasterize
        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                float2 p = (float2)(x + 0.5f, y + 0.5f) - projected[0].xy;
                float d20 = dot(p, v0);
                float d21 = dot(p, v1);
                
                float v = (d11 * d20 - d01 * d21) * invDenom;
                float w = (d00 * d21 - d01 * d20) * invDenom;
                float u = 1.0f - v - w;
                
                if (u >= -0.001f && v >= -0.001f && w >= -0.001f) {
                    float depth = u * projected[0].z + v * projected[1].z + w * projected[2].z;
                    
                    int pixelIdx = y * screenWidth + x;
                    
                    if (ScreenDistances[pixelIdx] == 0.0f || depth < ScreenDistances[pixelIdx]) {
                        ScreenDistances[pixelIdx] = depth;
                        
                        // Store normal
                        int normalIdx = pixelIdx * 3;
                        ScreenNormals[normalIdx] = world_normal.x;
                        ScreenNormals[normalIdx + 1] = world_normal.y;
                        ScreenNormals[normalIdx + 2] = world_normal.z;
                        
                        // Store color with simple lighting
                        float3 color = (float3)(model_colors[idx], model_colors[idx + 1], model_colors[idx + 2]);
                        float lighting = max(0.3f, dot(world_normal, normalize((float3)(0.3f, 0.7f, 0.5f))));
                        float3 final_color = color * lighting;
                        
                        int colorIdx = pixelIdx * 3;
                        ScreenColors[colorIdx] = clamp(final_color.x, 0.0f, 1.0f);
                        ScreenColors[colorIdx + 1] = clamp(final_color.y, 0.0f, 1.0f);
                        ScreenColors[colorIdx + 2] = clamp(final_color.z, 0.0f, 1.0f);
                        
                        // Store material properties
                        ScreenMaterialRoughness[pixelIdx] = model_roughness[triangleId];
                        ScreenMaterialMetallic[pixelIdx] = model_metallic[triangleId];
                        ScreenMaterialEmission[pixelIdx] = model_emission[triangleId];
                        ScreenAlphas[pixelIdx] = -1.0f;
                    }
                }
            }
        }
    }
}


__kernel void filterOverlap(
    __global const float* InputBuffer1,
    __global const float* InputDistance1,
    __global const float* InputDistance2,
    __global float* OutputBuffer,
    const int screenWidth,
    const int screenHeight
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    int pixelIdx = y * screenWidth + x;
    if (InputDistance1[pixelIdx] > InputDistance2[pixelIdx] && InputDistance2[pixelIdx] > 0.001f && InputDistance1[pixelIdx] > 0.001f) {
        OutputBuffer[pixelIdx] = 0.0f; // Set to black
    } else {
        OutputBuffer[pixelIdx] = InputBuffer1[pixelIdx];
    }
}

__kernel void OverlayImage(
    __global float* OutputBuffer,
    __global float* Image,
    const int screenWidth,
    const int screenHeight,
    const int imageWidth,
    const int imageHeight,
    const int Outputmode, // 0=RGB, 1=RGBA, 2=Grayscale
    const int displayMode, // 0=RGB, 1=RGBA, 2=Grayscale
    const int posX,
    const int posY
) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= screenWidth || y >= screenHeight) return;

    // Use posX and posY directly as the offset
    int offsetX = posX;
    int offsetY = posY;

    int imgX = x - offsetX;
    int imgY = y - offsetY;

    // nothing to overlay if outside image bounds
    if (imgX < 0 || imgX >= imageWidth || imgY < 0 || imgY >= imageHeight) return;

    int screenPixel = y * screenWidth + x;
    int imagePixel  = imgY * imageWidth + imgX;

    // load source pixel from Image according to displayMode (ignore alpha entirely)
    float srcR = 0.0f, srcG = 0.0f, srcB = 0.0f;
    if (displayMode == 0) { // RGB
        int i = imagePixel * 3;
        srcR = Image[i + 0];
        srcG = Image[i + 1];
        srcB = Image[i + 2];
    } else if (displayMode == 1) { // RGBA - ignore A
        int i = imagePixel * 4;
        srcR = Image[i + 0];
        srcG = Image[i + 1];
        srcB = Image[i + 2];
    } else { // Grayscale
        int i = imagePixel;
        float g = Image[i];
        srcR = g; srcG = g; srcB = g;
    }

    // clamp source components
    srcR = clamp(srcR, 0.0f, 1.0f);
    srcG = clamp(srcG, 0.0f, 1.0f);
    srcB = clamp(srcB, 0.0f, 1.0f);

    // Write directly into OutputBuffer (no alpha blending — overlay/replace)
    if (Outputmode == 0) { // RGB output (3 floats per pixel)
        int o = screenPixel * 3;
        OutputBuffer[o + 0] = srcR;
        OutputBuffer[o + 1] = srcG;
        OutputBuffer[o + 2] = srcB;
    } else if (Outputmode == 1) { // RGBA output (4 floats per pixel) -> set A = 1.0
        int o = screenPixel * 4;
        OutputBuffer[o + 0] = srcR;
        OutputBuffer[o + 1] = srcG;
        OutputBuffer[o + 2] = srcB;
        OutputBuffer[o + 3] = 1.0f;
    } else { // Grayscale output (1 float per pixel) -> convert to luminance
        int o = screenPixel;
        float lum = dot((float3)(srcR, srcG, srcB), (float3)(0.299f, 0.587f, 0.114f));
        OutputBuffer[o] = clamp(lum, 0.0f, 1.0f);
    }
}



float3 screenToDirection(
    int x, int y,
    const int screenWidth,
    const int screenHeight,
    const float3 camDir,
    const float fov
) {
    float aspectRatio = (float)screenWidth / (float)screenHeight;
    float tanHalfFov = tan(fov * 0.5f);
    
    // Normalized device coordinates (-1 to 1)
    float ndcX = (2.0f * x) / screenWidth - 1.0f;
    float ndcY = 1.0f - (2.0f * y) / screenHeight;
    
    // Calculate camera right and up vectors
    float3 worldUp = (float3)(0.0f, 1.0f, 0.0f);
    float3 camRight = normalize(cross(camDir, worldUp));
    float3 camUp = normalize(cross(camRight, camDir));
    
    // Calculate ray direction (normalized vector pointing to hotspot)
    float3 rayDir = normalize(
        camDir + 
        camRight * ndcX * aspectRatio * tanHalfFov +
        camUp * ndcY * tanHalfFov
    );
    
    return rayDir;
}

__kernel void renderDepthBufferFast(
    __global const float *v1,          // Triangle vertex 1 (3 floats per triangle)
    __global const float *v2,          // Triangle vertex 2
    __global const float *v3,          // Triangle vertex 3
    __global const float *normals,     // Triangle normals (unused but kept for compatibility)
    __global float *ScreenDistances,   // Output: depth buffer
    const int triangleCount,           // Number of triangles
    const float3 camPos,               // Camera position
    const float3 camDir,               // Camera direction
    const float fov,                   // Field of view
    const int screenWidth,             // Screen width
    const int screenHeight,           // Screen height
    const int idxOffset,
    __global float *ScreenDistancesAtlas
) {
    int triangleIdx = get_global_id(0);
    if (triangleIdx >= triangleCount) return;

    int startId = idxOffset * screenWidth * screenHeight; // staring index for atlas to save all depth buffers

    // Get triangle vertices
    int idx = triangleIdx * 3;
    float3 v1_pos = (float3)(v1[idx], v1[idx + 1], v1[idx + 2]);
    float3 v2_pos = (float3)(v2[idx], v2[idx + 1], v2[idx + 2]);
    float3 v3_pos = (float3)(v3[idx], v3[idx + 1], v3[idx + 2]);

    // Back-face culling (match renderTriangles)
    float3 fn = vload3(0, normals + idx);
    float3 center = (v1_pos + v2_pos + v3_pos) * (1.0f/3.0f);
    if (dot(fn, normalize(camPos - center)) <= 0.0f) return;

    // Camera basis (match renderTriangles)
    float3 F = normalize(camDir);
    float3 U = (float3)(0,1,0);
    float3 R = normalize(cross(F,U));
    U = cross(R,F);

    // Constants (match renderTriangles)
    float invF = 1.0f / fov;
    float halfW = screenWidth * 0.5f, halfH = screenHeight * 0.5f;

    // Project vertices to screen space (match renderTriangles exactly)
    float3 r0 = v1_pos - camPos, r1 = v2_pos - camPos, r2 = v3_pos - camPos;
    float d0 = dot(r0,F), d1 = dot(r1,F), d2 = dot(r2,F);
    float minD = fmin(fmin(d0,d1),d2);
    if (minD <= 0.001f) return;
    
    float s0 = invF/d0, s1 = invF/d1, s2 = invF/d2;
    float3 sp0 = (float3)(dot(r0,R)*halfW*s0 + halfW, -dot(r0,U)*halfH*s0 + halfH, d0);
    float3 sp1 = (float3)(dot(r1,R)*halfW*s1 + halfW, -dot(r1,U)*halfH*s1 + halfH, d1);
    float3 sp2 = (float3)(dot(r2,R)*halfW*s2 + halfW, -dot(r2,U)*halfH*s2 + halfH, d2);

    // Bounding box
    float minXf = fmin(fmin(sp0.x,sp1.x),sp2.x),
          maxXf = fmax(fmax(sp0.x,sp1.x),sp2.x),
          minYf = fmin(fmin(sp0.y,sp1.y),sp2.y),
          maxYf = fmax(fmax(sp0.y,sp1.y),sp2.y);
    if (maxXf < 0 || minXf >= screenWidth || maxYf < 0 || minYf >= screenHeight) return;
    
    int x0 = max(0,(int)minXf), x1 = min(screenWidth-1,(int)maxXf);
    int y0 = max(0,(int)minYf), y1 = min(screenHeight-1,(int)maxYf);

    // Precompute barycentric constants (match renderTriangles)
    float2 e1 = sp2.xy - sp0.xy, e2 = sp1.xy - sp0.xy;
    float d00 = dot(e1,e1), d01 = dot(e1,e2), d11 = dot(e2,e2);
    float invDen = 1.0f / (d00*d11 - d01*d01);

    // Rasterize triangle
    for (int y = y0; y <= y1; ++y) {
        float cy = (y + 0.5f) - sp0.y;
        int row = y * screenWidth;
        for (int x = x0; x <= x1; ++x) {
            float cx = (x + 0.5f) - sp0.x;
            float l0 = e1.x*cx + e1.y*cy;
            float l1 = e2.x*cx + e2.y*cy;
            float u = (d11*l0 - d01*l1) * invDen;
            float v = (d00*l1 - d01*l0) * invDen;
            if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
                int pixelIdx = row + x;
                float w = 1.0f - u - v;
                float depth = w*sp0.z + u*sp1.z + v*sp2.z;
                
                // Update ScreenDistances
                float prev = ScreenDistances[pixelIdx];
                if (prev == 0.0f || depth < prev) {
                    ScreenDistances[pixelIdx] = depth;
                }
                
                // Update ScreenDistancesAtlas
                int atlasIdx = startId + pixelIdx;
                float prevAtlas = ScreenDistancesAtlas[atlasIdx];
                if (prevAtlas == 0.0f || depth < prevAtlas) {
                    ScreenDistancesAtlas[atlasIdx] = depth;
                }
            }
        }
    }
}

// GPU-based LOD terrain depth-only rendering kernel (for IRST seeker)
__kernel void renderTerrainDepthLOD(
    __global const struct MapGPU *mapData,
    __global float *ScreenDistances,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight
) {
    int triangleId = get_global_id(0);
    
    // Determine which chunk and LOD level this triangle belongs to
    int chunkIdx = 0;
    int localTriangleId = triangleId;
    int lodLevel = 3;
    
    // Find which chunk and LOD this triangle belongs to
    for (int i = 0; i < CHUNK_COUNT; i++) {
        int highStart = mapData->chunkStartHigh[i];
        int medStart = mapData->chunkStartMed[i];
        int lowStart = mapData->chunkStartLow[i];
        
        int highCount = (i < CHUNK_COUNT - 1) ? 
            (mapData->chunkStartHigh[i + 1] - highStart) / 9 : 
            (HIGH_RES_TRIANGLE_COUNT * 9 - highStart) / 9;
        int medCount = (i < CHUNK_COUNT - 1) ? 
            (mapData->chunkStartMed[i + 1] - medStart) / 9 : 
            (MID_RES_TRIANGLE_COUNT * 9 - medStart) / 9;
        int lowCount = (i < CHUNK_COUNT - 1) ? 
            (mapData->chunkStartLow[i + 1] - lowStart) / 9 : 
            (LOW_RES_TRIANGLE_COUNT * 9 - lowStart) / 9;
        
        if (localTriangleId < highCount) {
            chunkIdx = i;
            lodLevel = 3;
            break;
        }
        localTriangleId -= highCount;
        
        if (localTriangleId < medCount) {
            chunkIdx = i;
            lodLevel = 2;
            break;
        }
        localTriangleId -= medCount;
        
        if (localTriangleId < lowCount) {
            chunkIdx = i;
            lodLevel = 1;
            break;
        }
        localTriangleId -= lowCount;
    }
    
    // Calculate chunk center and distance
    int tilesPerRow = (int)(mapData->mapSizeZ / mapData->tileSizeZ);
    int chunkX = chunkIdx / tilesPerRow;
    int chunkZ = chunkIdx % tilesPerRow;
    
    float chunkCenterX = mapData->posX + chunkX * mapData->tileSizeX + mapData->tileSizeX * 0.5f;
    float chunkCenterZ = mapData->posZ + chunkZ * mapData->tileSizeZ + mapData->tileSizeZ * 0.5f;
    
    float dx = camPos.x - chunkCenterX;
    float dz = camPos.z - chunkCenterZ;
    float distanceToChunk = sqrt(dx * dx + dz * dz);
    
    // Determine required LOD based on distance
    int requiredLOD = 0;
    if (distanceToChunk <= LOD_HIGH_DISTANCE) {
        requiredLOD = 3;
    } else if (distanceToChunk <= LOD_MED_DISTANCE) {
        requiredLOD = 2;
    } else if (distanceToChunk <= LOD_LOW_DISTANCE) {
        requiredLOD = 1;
    }
    
    if (lodLevel != requiredLOD) return;
    
    // Load triangle vertices
    float3 p0, p1, p2;
    int dataOffset;
    
    if (lodLevel == 3) {
        dataOffset = mapData->chunkStartHigh[chunkIdx] + localTriangleId * 9;
        p0 = (float3)(mapData->chunkHighTrianglesData[dataOffset + 0],
                      mapData->chunkHighTrianglesData[dataOffset + 1],
                      mapData->chunkHighTrianglesData[dataOffset + 2]);
        p1 = (float3)(mapData->chunkHighTrianglesData[dataOffset + 3],
                      mapData->chunkHighTrianglesData[dataOffset + 4],
                      mapData->chunkHighTrianglesData[dataOffset + 5]);
        p2 = (float3)(mapData->chunkHighTrianglesData[dataOffset + 6],
                      mapData->chunkHighTrianglesData[dataOffset + 7],
                      mapData->chunkHighTrianglesData[dataOffset + 8]);
    } else if (lodLevel == 2) {
        dataOffset = mapData->chunkStartMed[chunkIdx] + localTriangleId * 9;
        p0 = (float3)(mapData->chunkMedTrianglesData[dataOffset + 0],
                      mapData->chunkMedTrianglesData[dataOffset + 1],
                      mapData->chunkMedTrianglesData[dataOffset + 2]);
        p1 = (float3)(mapData->chunkMedTrianglesData[dataOffset + 3],
                      mapData->chunkMedTrianglesData[dataOffset + 4],
                      mapData->chunkMedTrianglesData[dataOffset + 5]);
        p2 = (float3)(mapData->chunkMedTrianglesData[dataOffset + 6],
                      mapData->chunkMedTrianglesData[dataOffset + 7],
                      mapData->chunkMedTrianglesData[dataOffset + 8]);
    } else {
        dataOffset = mapData->chunkStartLow[chunkIdx] + localTriangleId * 9;
        p0 = (float3)(mapData->chunkLowTrianglesData[dataOffset + 0],
                      mapData->chunkLowTrianglesData[dataOffset + 1],
                      mapData->chunkLowTrianglesData[dataOffset + 2]);
        p1 = (float3)(mapData->chunkLowTrianglesData[dataOffset + 3],
                      mapData->chunkLowTrianglesData[dataOffset + 4],
                      mapData->chunkLowTrianglesData[dataOffset + 5]);
        p2 = (float3)(mapData->chunkLowTrianglesData[dataOffset + 6],
                      mapData->chunkLowTrianglesData[dataOffset + 7],
                      mapData->chunkLowTrianglesData[dataOffset + 8]);
    }
    
    // Back-face culling
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float3 fn = normalize(cross(edge1, edge2));
    float3 center = (p0 + p1 + p2) * (1.0f/3.0f);
    if (dot(fn, normalize(camPos - center)) <= 0.0f) return;
    
    // Camera basis
    float3 F = normalize(camDir);
    float3 U = (float3)(0,1,0);
    float3 R = normalize(cross(F,U));
    U = cross(R,F);
    
    float invF = 1.0f / fov;
    float halfW = screenWidth * 0.5f, halfH = screenHeight * 0.5f;
    
    // Project vertices
    float3 r0 = p0 - camPos, r1 = p1 - camPos, r2 = p2 - camPos;
    float d0 = dot(r0,F), d1 = dot(r1,F), d2 = dot(r2,F);
    float minD = fmin(fmin(d0,d1),d2);
    if (minD <= 0.001f) return;
    
    float s0 = invF/d0, s1 = invF/d1, s2 = invF/d2;
    float3 sp0 = (float3)(dot(r0,R)*halfW*s0 + halfW, -dot(r0,U)*halfH*s0 + halfH, d0);
    float3 sp1 = (float3)(dot(r1,R)*halfW*s1 + halfW, -dot(r1,U)*halfH*s1 + halfH, d1);
    float3 sp2 = (float3)(dot(r2,R)*halfW*s2 + halfW, -dot(r2,U)*halfH*s2 + halfH, d2);
    
    // Frustum culling
    float minXf = fmin(fmin(sp0.x,sp1.x),sp2.x),
          maxXf = fmax(fmax(sp0.x,sp1.x),sp2.x),
          minYf = fmin(fmin(sp0.y,sp1.y),sp2.y),
          maxYf = fmax(fmax(sp0.y,sp1.y),sp2.y);
    if (maxXf < 0 || minXf >= screenWidth || maxYf < 0 || minYf >= screenHeight) return;
    
    int x0 = max(0,(int)minXf), x1 = min(screenWidth-1,(int)maxXf);
    int y0 = max(0,(int)minYf), y1 = min(screenHeight-1,(int)maxYf);
    
    // Barycentric setup
    float2 e1 = sp2.xy - sp0.xy, e2 = sp1.xy - sp0.xy;
    float d00 = dot(e1,e1), d01 = dot(e1,e2), d11 = dot(e2,e2);
    float invDen = 1.0f / (d00*d11 - d01*d01);
    
    // Rasterize (depth only)
    for (int y = y0; y <= y1; ++y) {
        float cy = (y + 0.5f) - sp0.y;
        int row = y * screenWidth;
        for (int x = x0; x <= x1; ++x) {
            float cx = (x + 0.5f) - sp0.x;
            float l0 = e1.x*cx + e1.y*cy;
            float l1 = e2.x*cx + e2.y*cy;
            float u = (d11*l0 - d01*l1) * invDen;
            float v = (d00*l1 - d01*l0) * invDen;
            if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
                int idx = row + x;
                float w = 1.0f - u - v;
                float depth = w*sp0.z + u*sp1.z + v*sp2.z;
                float prev = ScreenDistances[idx];
                if (prev == 0.0f || depth < prev) {
                    ScreenDistances[idx] = depth;
                }
            }
        }
    }
}

// Helper function: find ray-cone intersection
// Returns the distance along the ray, or -1 if no intersection
float findConeIntersection(
    float rayOx, float rayOy, float rayOz,
    float rayDx, float rayDy, float rayDz,
    float coneOx, float coneOy, float coneOz,
    float coneDx, float coneDy, float coneDz,
    float cosAngle)
{
    // Vector from cone origin to ray origin
    float cox = rayOx - coneOx;
    float coy = rayOy - coneOy;
    float coz = rayOz - coneOz;

    float dv = rayDx * coneDx + rayDy * coneDy + rayDz * coneDz;
    float cov = cox * coneDx + coy * coneDy + coz * coneDz;
    
    // Quadratic equation coefficients for ray-cone intersection
    float a = dv * dv - cosAngle * cosAngle;
    float b = 2.0f * (dv * cov - (rayDx * cox + rayDy * coy + rayDz * coz) * cosAngle * cosAngle);
    float c = cov * cov - (cox * cox + coy * coy + coz * coz) * cosAngle * cosAngle;

    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) return -1.0f;

    float sqrtDisc = sqrt(discriminant);
    float t1 = (-b - sqrtDisc) / (2.0f * a);
    float t2 = (-b + sqrtDisc) / (2.0f * a);

    // Check both intersections and pick the closest valid one
    // We only need to check if intersection is in front of cone origin
    for (int i = 0; i < 2; i++) {
        float t = (i == 0) ? t1 : t2;
        if (t <= 0.0f) continue;
        
        // Check if intersection point is in front of cone origin (positive along axis)
        float px = rayOx + rayDx * t;
        float py = rayOy + rayDy * t;
        float pz = rayOz + rayDz * t;
        
        float toPx = px - coneOx;
        float toPy = py - coneOy;
        float toPz = pz - coneOz;
        
        float distAlongAxis = toPx * coneDx + toPy * coneDy + toPz * coneDz;
        
        // Only accept if in front of cone origin
        if (distAlongAxis >= 0.0f) {
            return t;
        }
    }

    return -1.0f;
}

// Project world point into seeker's view and check occlusion
int isPointVisibleToSeeker(
    float px, float py, float pz,
    float seekerX, float seekerY, float seekerZ,
    float seekerDx, float seekerDy, float seekerDz,
    float seekerFovDegrees,
    int coneIndex,
    int seekerResolution,
    __global float* seekerDepthMaps)
{
    // Vector from seeker to point
    float toPx = px - seekerX;
    float toPy = py - seekerY;
    float toPz = pz - seekerZ;
    
    float distToPoint = sqrt(toPx * toPx + toPy * toPy + toPz * toPz);
    if (distToPoint < 0.001f) return 1;
    
    // Normalize
    toPx /= distToPoint;
    toPy /= distToPoint;
    toPz /= distToPoint;
    
    // Check if point is within seeker's FOV cone
    float dotWithSeeker = toPx * seekerDx + toPy * seekerDy + toPz * seekerDz;
    float seekerConeAngle = (seekerFovDegrees / 2.0f) * M_PI_F / 180.0f;
    float cosSeeker = cos(seekerConeAngle);
    
    if (dotWithSeeker < cosSeeker) return 0; // Outside seeker FOV
    
    // Build seeker camera basis (same method as main camera)
    float supx = 0.0f, supy = 1.0f, supz = 0.0f;
    if (fabs(seekerDy) > 0.99f) { supx = 1.0f; supy = 0.0f; supz = 0.0f; }
    
    // Right = cross(seekerDir, up)
    float srx = seekerDy * supz - seekerDz * supy;
    float sry = seekerDz * supx - seekerDx * supz;
    float srz = seekerDx * supy - seekerDy * supx;
    float srlen = sqrt(srx * srx + sry * sry + srz * srz);
    srx /= srlen; sry /= srlen; srz /= srlen;
    
    // Up = cross(right, seekerDir)
    supx = sry * seekerDz - srz * seekerDy;
    supy = srz * seekerDx - srx * seekerDz;
    supz = srx * seekerDy - sry * seekerDx;
    
    // Project point into seeker's view space
    float seekerFovTan = tan(seekerConeAngle);
    float ndcX = (toPx * srx + toPy * sry + toPz * srz) / (seekerFovTan * dotWithSeeker);
    float ndcY = -(toPx * supx + toPy * supy + toPz * supz) / (seekerFovTan * dotWithSeeker);
    
    // Convert to texture coordinates (using seekerResolution)
    float texU = ndcX * 0.5f + 0.5f;
    float texV = ndcY * 0.5f + 0.5f;
    
    if (texU < 0.0f || texU >= 1.0f || texV < 0.0f || texV >= 1.0f) return 0;
    
    int texX = (int)(texU * (float)seekerResolution);
    int texY = (int)(texV * (float)seekerResolution);
    texX = max(0, min(seekerResolution - 1, texX));
    texY = max(0, min(seekerResolution - 1, texY));
    
    // Check seeker's depth map
    int seekerPixelIdx = coneIndex * seekerResolution * seekerResolution + texY * seekerResolution + texX;
    float seekerDepth = seekerDepthMaps[seekerPixelIdx];
    
    // Point is visible if:
    // 1. Seeker sees sky (depth = 0) at this direction, OR
    // 2. Point is at approximately the same depth the seeker sees (within tolerance)
    if (seekerDepth < 0.001f) return 1; // Seeker sees sky here
    
    // Allow some tolerance for depth comparison (5%)
    return (distToPoint <= seekerDepth * 1.05f) ? 1 : 0;
}

__kernel void composite_cones(
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
    __global float* coneFov,
    int numberOfCones,
    __global float* seekerDepthMaps,  // Depth map per seeker
    int seekerResolution  // Resolution of seeker depth maps (e.g., 96)
)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= imageWidth || y >= imageHeight) return;

    int pixelIdx = y * imageWidth + x;

    // Desaturate background
    float r = imageBufferColor[pixelIdx * 3 + 0];
    float g = imageBufferColor[pixelIdx * 3 + 1];
    float b = imageBufferColor[pixelIdx * 3 + 2];
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    imageBufferColor[pixelIdx * 3 + 0] = gray * 0.9f + r * 0.1f;
    imageBufferColor[pixelIdx * 3 + 1] = gray * 0.9f + g * 0.1f;
    imageBufferColor[pixelIdx * 3 + 2] = gray * 0.9f + b * 0.1f;

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
    float rdx = cdx + rx * ndcX * aspectRatio * cameraFOV + upx * ndcY * cameraFOV;
    float rdy = cdy + ry * ndcX * aspectRatio * cameraFOV + upy * ndcY * cameraFOV;
    float rdz = cdz + rz * ndcX * aspectRatio * cameraFOV + upz * ndcY * cameraFOV;
    float rdLen = sqrt(rdx * rdx + rdy * rdy + rdz * rdz);
    rdx /= rdLen; rdy /= rdLen; rdz /= rdLen;

    // Find closest visible cone
    int hitCone = -1;
    float bestBlendFactor = 0.0f;
    float closestT = 1e10f;

    for (int i = 0; i < numberOfCones; i++) {
        float cox = coneOriginX[i];
        float coy = coneOriginY[i];
        float coz = coneOriginZ[i];
        float cvx = coneDirX[i];
        float cvy = coneDirY[i];
        float cvz = coneDirZ[i];

        float cvLen = sqrt(cvx * cvx + cvy * cvy + cvz * cvz);
        cvx /= cvLen; cvy /= cvLen; cvz /= cvLen;

        float coneFovDegrees = coneFov[i];
        float coneAngle = (coneFovDegrees / 2.0f) * M_PI_F / 180.0f;
        float cosA = cos(coneAngle);

        // Find where camera ray intersects this cone (infinite cone)
        float t = findConeIntersection(
            cameraPosX, cameraPosY, cameraPosZ, rdx, rdy, rdz,
            cox, coy, coz, cvx, cvy, cvz, cosA);

        if (t <= 0.0f) continue; // No intersection

        // The cone is limited by geometry:
        // - If there's existing geometry (depth > 0), cone stops there
        // - If no geometry (depth = 0, sky), cone extends infinitely
        float maxConeDepth = (existingDepth > 0.001f) ? existingDepth : 1e10f;
        
        // Check if intersection is before the geometry
        if (t >= maxConeDepth) continue;

        // Get intersection point
        float hitX = cameraPosX + rdx * t;
        float hitY = cameraPosY + rdy * t;
        float hitZ = cameraPosZ + rdz * t;

        // Check if this point is visible to the seeker (not occluded in seeker's view)
        if (!isPointVisibleToSeeker(
            hitX, hitY, hitZ,
            cox, coy, coz, cvx, cvy, cvz,
            coneFovDegrees, i, seekerResolution, seekerDepthMaps)) {
            continue; // Occluded from seeker's perspective
        }

        // This is a valid, visible cone intersection
        if (t < closestT) {
            closestT = t;
            hitCone = i;

            // Calculate blend factor based on distance from cone axis
            float toHitX = hitX - cox;
            float toHitY = hitY - coy;
            float toHitZ = hitZ - coz;
            
            float distAlongAxis = toHitX * cvx + toHitY * cvy + toHitZ * cvz;
            
            // Project onto axis to find perpendicular distance
            float projX = cvx * distAlongAxis;
            float projY = cvy * distAlongAxis;
            float projZ = cvz * distAlongAxis;
            
            float perpX = toHitX - projX;
            float perpY = toHitY - projY;
            float perpZ = toHitZ - projZ;
            
            float distFromAxis = sqrt(perpX * perpX + perpY * perpY + perpZ * perpZ);
            float maxRadius = distAlongAxis * tan(coneAngle);
            
            if (maxRadius > 0.001f) {
                float radialFactor = 1.0f - (distFromAxis / maxRadius);
                radialFactor = max(0.0f, min(1.0f, radialFactor));
                
                // Stronger at center, weaker at edges
                bestBlendFactor = 0.2f + radialFactor * 0.6f;
            } else {
                bestBlendFactor = 0.5f;
            }
        }
    }

    // Apply cone visualization
    if (hitCone >= 0 && bestBlendFactor > 0.001f) {
        float hue = (float)hitCone / (float)max(numberOfCones, 1);
        
        float multiplayer = 3.0f;

        // Vibrant color based on cone index
        float coneR = 0.1f + hue * 0.9f * multiplayer;
        float coneG = 0.9f - hue * 0.4f * multiplayer;
        float coneB = 0.2f + (1.0f - hue) * 0.6f * multiplayer;

        imageBufferColor[pixelIdx * 3 + 0] = imageBufferColor[pixelIdx * 3 + 0] * (1.0f - bestBlendFactor) + coneR * bestBlendFactor;
        imageBufferColor[pixelIdx * 3 + 1] = imageBufferColor[pixelIdx * 3 + 1] * (1.0f - bestBlendFactor) + coneG * bestBlendFactor;
        imageBufferColor[pixelIdx * 3 + 2] = imageBufferColor[pixelIdx * 3 + 2] * (1.0f - bestBlendFactor) + coneB * bestBlendFactor;
    }
}

// GPU-based LOD terrain rendering kernel
__kernel void renderTerrainLOD(
    __global const struct MapGPU *mapData,
    __global float *ScreenColors,
    __global float *ScreenDistances,
    __global float *ScreenNormals,
    __global float *ScreenMaterialRoughness,
    __global float *ScreenMaterialMetallic,
    __global float *ScreenMaterialEmission,
    const float3 camPos,
    const float3 camDir,
    const float fov,
    const int screenWidth,
    const int screenHeight
) {
    int triangleId = get_global_id(0);
    
    // Determine which chunk and LOD level this triangle belongs to
    int chunkIdx = 0;
    int localTriangleId = triangleId;
    int lodLevel = 3; // Start with high LOD
    
    // Find which chunk and LOD this triangle belongs to
    for (int i = 0; i < CHUNK_COUNT; i++) {
        int highStart = mapData->chunkStartHigh[i];
        int medStart = mapData->chunkStartMed[i];
        int lowStart = mapData->chunkStartLow[i];
        
        // Calculate chunk bounds (number of triangles)
        int highCount = (i < CHUNK_COUNT - 1) ? 
            (mapData->chunkStartHigh[i + 1] - highStart) / 9 : 
            (HIGH_RES_TRIANGLE_COUNT * 9 - highStart) / 9;
        int medCount = (i < CHUNK_COUNT - 1) ? 
            (mapData->chunkStartMed[i + 1] - medStart) / 9 : 
            (MID_RES_TRIANGLE_COUNT * 9 - medStart) / 9;
        int lowCount = (i < CHUNK_COUNT - 1) ? 
            (mapData->chunkStartLow[i + 1] - lowStart) / 9 : 
            (LOW_RES_TRIANGLE_COUNT * 9 - lowStart) / 9;
        
        // Check if triangle is in this chunk's high LOD
        if (localTriangleId < highCount) {
            chunkIdx = i;
            lodLevel = 3;
            break;
        }
        localTriangleId -= highCount;
        
        // Check medium LOD
        if (localTriangleId < medCount) {
            chunkIdx = i;
            lodLevel = 2;
            break;
        }
        localTriangleId -= medCount;
        
        // Check low LOD
        if (localTriangleId < lowCount) {
            chunkIdx = i;
            lodLevel = 1;
            break;
        }
        localTriangleId -= lowCount;
    }
    
    // Calculate chunk center position
    // CPU uses: idx = (world_x - min_x) * height + (world_y - min_y)
    // So to reverse: given idx, we need to find world_x and world_y
    int tilesPerCol = (int)(mapData->mapSizeZ / mapData->tileSizeZ);
    int chunkX = chunkIdx / tilesPerCol;  // Row index
    int chunkY = chunkIdx % tilesPerCol;  // Column index
    
    float chunkCenterX = mapData->posX + chunkX * mapData->tileSizeX + mapData->tileSizeX * 0.5f;
    float chunkCenterZ = mapData->posZ + chunkY * mapData->tileSizeZ + mapData->tileSizeZ * 0.5f;
    
    // Calculate distance from camera to chunk center
    float dx = camPos.x - chunkCenterX;
    float dz = camPos.z - chunkCenterZ;
    float distanceToChunk = sqrt(dx * dx + dz * dz);
    
    // Determine required LOD based on distance
    int requiredLOD = 0; // LOD_NONE
    if (distanceToChunk <= LOD_HIGH_DISTANCE) {
        requiredLOD = 3; // LOD_HIGH
    } else if (distanceToChunk <= LOD_MED_DISTANCE) {
        requiredLOD = 2; // LOD_MEDIUM
    } else if (distanceToChunk <= LOD_LOW_DISTANCE) {
        requiredLOD = 1; // LOD_LOW
    }
    
    // Skip this triangle if it doesn't match the required LOD
    if (lodLevel != requiredLOD) {
        return;
    }
    
    // Load triangle vertices based on LOD level
    float3 p0, p1, p2, normal, color;
    float roughness, metallic, emission;
    int dataOffset, materialOffset;
    
    if (lodLevel == 3) {
        dataOffset = mapData->chunkStartHigh[chunkIdx] + localTriangleId * 9;
        materialOffset = mapData->chunkStartHigh[chunkIdx] / 9 + localTriangleId;
        
        p0 = (float3)(mapData->chunkHighTrianglesData[dataOffset + 0],
                      mapData->chunkHighTrianglesData[dataOffset + 1],
                      mapData->chunkHighTrianglesData[dataOffset + 2]);
        p1 = (float3)(mapData->chunkHighTrianglesData[dataOffset + 3],
                      mapData->chunkHighTrianglesData[dataOffset + 4],
                      mapData->chunkHighTrianglesData[dataOffset + 5]);
        p2 = (float3)(mapData->chunkHighTrianglesData[dataOffset + 6],
                      mapData->chunkHighTrianglesData[dataOffset + 7],
                      mapData->chunkHighTrianglesData[dataOffset + 8]);
        
        // Load color
        color = (float3)(mapData->chunkHighColorsData[materialOffset * 3 + 0],
                        mapData->chunkHighColorsData[materialOffset * 3 + 1],
                        mapData->chunkHighColorsData[materialOffset * 3 + 2]);
        
        // Load normal
        normal = (float3)(mapData->chunkHighNormalsData[materialOffset * 3 + 0],
                         mapData->chunkHighNormalsData[materialOffset * 3 + 1],
                         mapData->chunkHighNormalsData[materialOffset * 3 + 2]);
        
        // Load material properties
        roughness = mapData->chunkHighRoughnessData[materialOffset];
        metallic = mapData->chunkHighMetallicData[materialOffset];
        emission = mapData->chunkHighEmissionData[materialOffset];
        
    } else if (lodLevel == 2) {
        dataOffset = mapData->chunkStartMed[chunkIdx] + localTriangleId * 9;
        materialOffset = mapData->chunkStartMed[chunkIdx] / 9 + localTriangleId;
        
        p0 = (float3)(mapData->chunkMedTrianglesData[dataOffset + 0],
                      mapData->chunkMedTrianglesData[dataOffset + 1],
                      mapData->chunkMedTrianglesData[dataOffset + 2]);
        p1 = (float3)(mapData->chunkMedTrianglesData[dataOffset + 3],
                      mapData->chunkMedTrianglesData[dataOffset + 4],
                      mapData->chunkMedTrianglesData[dataOffset + 5]);
        p2 = (float3)(mapData->chunkMedTrianglesData[dataOffset + 6],
                      mapData->chunkMedTrianglesData[dataOffset + 7],
                      mapData->chunkMedTrianglesData[dataOffset + 8]);
        
        // Load color
        color = (float3)(mapData->chunkMedColorsData[materialOffset * 3 + 0],
                        mapData->chunkMedColorsData[materialOffset * 3 + 1],
                        mapData->chunkMedColorsData[materialOffset * 3 + 2]);
        
        // Load normal
        normal = (float3)(mapData->chunkMedNormalsData[materialOffset * 3 + 0],
                         mapData->chunkMedNormalsData[materialOffset * 3 + 1],
                         mapData->chunkMedNormalsData[materialOffset * 3 + 2]);
        
        // Load material properties
        roughness = mapData->chunkMedRoughnessData[materialOffset];
        metallic = mapData->chunkMedMetallicData[materialOffset];
        emission = mapData->chunkMedEmissionData[materialOffset];
        
    } else {
        dataOffset = mapData->chunkStartLow[chunkIdx] + localTriangleId * 9;
        materialOffset = mapData->chunkStartLow[chunkIdx] / 9 + localTriangleId;
        
        p0 = (float3)(mapData->chunkLowTrianglesData[dataOffset + 0],
                      mapData->chunkLowTrianglesData[dataOffset + 1],
                      mapData->chunkLowTrianglesData[dataOffset + 2]);
        p1 = (float3)(mapData->chunkLowTrianglesData[dataOffset + 3],
                      mapData->chunkLowTrianglesData[dataOffset + 4],
                      mapData->chunkLowTrianglesData[dataOffset + 5]);
        p2 = (float3)(mapData->chunkLowTrianglesData[dataOffset + 6],
                      mapData->chunkLowTrianglesData[dataOffset + 7],
                      mapData->chunkLowTrianglesData[dataOffset + 8]);
        
        // Load color
        color = (float3)(mapData->chunkLowColorsData[materialOffset * 3 + 0],
                        mapData->chunkLowColorsData[materialOffset * 3 + 1],
                        mapData->chunkLowColorsData[materialOffset * 3 + 2]);
        
        // Load normal
        normal = (float3)(mapData->chunkLowNormalsData[materialOffset * 3 + 0],
                         mapData->chunkLowNormalsData[materialOffset * 3 + 1],
                         mapData->chunkLowNormalsData[materialOffset * 3 + 2]);
        
        // Load material properties
        roughness = mapData->chunkLowRoughnessData[materialOffset];
        metallic = mapData->chunkLowMetallicData[materialOffset];
        emission = mapData->chunkLowEmissionData[materialOffset];
    }
    
    // Calculate face normal from geometry (for back-face culling)
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float3 fn = normalize(cross(edge1, edge2));
    
    // Back-face culling
    float3 center = (p0 + p1 + p2) * (1.0f/3.0f);
    if (dot(fn, normalize(camPos - center)) <= 0.0f) return;
    
    // Camera basis
    float3 F = normalize(camDir);
    float3 U = (float3)(0,1,0);
    float3 R = normalize(cross(F,U));
    U = cross(R,F);
    
    // Constants
    float invF = 1.0f / fov;
    float halfW = screenWidth * 0.5f, halfH = screenHeight * 0.5f;
    
    // Compute depths and screen projections
    float3 r0 = p0 - camPos, r1 = p1 - camPos, r2 = p2 - camPos;
    float d0 = dot(r0,F), d1 = dot(r1,F), d2 = dot(r2,F);
    float minD = fmin(fmin(d0,d1),d2);
    if (minD <= 0.001f) return;
    
    float s0 = invF/d0, s1 = invF/d1, s2 = invF/d2;
    float3 sp0 = (float3)(dot(r0,R)*halfW*s0 + halfW, -dot(r0,U)*halfH*s0 + halfH, d0);
    float3 sp1 = (float3)(dot(r1,R)*halfW*s1 + halfW, -dot(r1,U)*halfH*s1 + halfH, d1);
    float3 sp2 = (float3)(dot(r2,R)*halfW*s2 + halfW, -dot(r2,U)*halfH*s2 + halfH, d2);
    
    // Frustum culling & bounding box
    float minXf = fmin(fmin(sp0.x,sp1.x),sp2.x),
          maxXf = fmax(fmax(sp0.x,sp1.x),sp2.x),
          minYf = fmin(fmin(sp0.y,sp1.y),sp2.y),
          maxYf = fmax(fmax(sp0.y,sp1.y),sp2.y);
    if (maxXf < 0 || minXf >= screenWidth || maxYf < 0 || minYf >= screenHeight) return;
    
    int x0 = max(0,(int)minXf), x1 = min(screenWidth-1,(int)maxXf);
    int y0 = max(0,(int)minYf), y1 = min(screenHeight-1,(int)maxYf);
    
    // Small triangle culling
    float area = fabs((sp1.x-sp0.x)*(sp2.y-sp0.y) - (sp2.x-sp0.x)*(sp1.y-sp0.y)) * 0.5f;
    if (area < 0.5f) return;
    
    // Precompute barycentric constants
    float2 e1 = sp2.xy - sp0.xy, e2 = sp1.xy - sp0.xy;
    float d00 = dot(e1,e1), d01 = dot(e1,e2), d11 = dot(e2,e2);
    float invDen = 1.0f / (d00*d11 - d01*d01);
    
    // Use the stored normal (normalized)
    float3 nrm = normalize(normal);
    
    // Rasterize
    for (int y = y0; y <= y1; ++y) {
        float cy = (y + 0.5f) - sp0.y;
        int row = y * screenWidth;
        for (int x = x0; x <= x1; ++x) {
            float cx = (x + 0.5f) - sp0.x;
            float l0 = e1.x*cx + e1.y*cy;
            float l1 = e2.x*cx + e2.y*cy;
            float u = (d11*l0 - d01*l1) * invDen;
            float v = (d00*l1 - d01*l0) * invDen;
            if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f) {
                int idx = row + x;
                float w = 1.0f - u - v;
                float depth = w*sp0.z + u*sp1.z + v*sp2.z;
                float prev = ScreenDistances[idx];
                if (prev != 0.0f && depth >= prev) continue;
                ScreenDistances[idx] = depth;
                
                int i3 = idx * 3;
                ScreenNormals[i3  ] = nrm.x;
                ScreenNormals[i3+1] = nrm.y;
                ScreenNormals[i3+2] = nrm.z;
                
                // Apply basic lighting to the stored color
                float intensity = max(0.3f, dot(nrm, normalize((float3)(0.3f, 0.7f, 0.5f))));
                float3 finalCol = clamp(color * intensity, 0.0f, 1.0f);
                ScreenColors[i3  ] = finalCol.x;
                ScreenColors[i3+1] = finalCol.y;
                ScreenColors[i3+2] = finalCol.z;
                
                // Store material properties
                ScreenMaterialRoughness[idx] = roughness;
                ScreenMaterialMetallic[idx]  = metallic;
                ScreenMaterialEmission[idx]  = emission;
            }
        }
    }
}
