#define FLT_MAX 3.402823466e+38F

typedef struct {
    float   BoundingBox[6]; // minX, minY, minZ, maxX, maxY, maxZ
    int     LeftChild;        // Index of left child node -1 => no child
    int     RightChild;       // Index of right child node -1 => no child
    int     TriangleIndex;    // -1 => internal node
} BVHNode;

typedef struct{
    float3  v1; // Vertex 1
    float3  v2; // Vertex 2
    float3  v3; // Vertex 3
    float3  normal; // Normal vector
    float3  color; // RGB color
    float   Roughness; // Material roughness
    float   Metallic; // Material metallic
    float   Emission; // Material emission
    int     TriangleIndex; // Index of the triangle
} Triangle;

typedef struct  {
    __global BVHNode  *Nodes; // Array of BVH nodes
    __global Triangle *Triangles; // Array of triangles
    int NodesCount; // Number of nodes in the BVH
    int TrianglesCount; // Number of triangles in the BVH
} BVHLinear;

typedef struct {
    float3  PointOfIntersection;
    float3  NormalAtIntersection;
    float3  ColorAtIntersection;
    float   Distance; 
    int     TriangleIndex;
    bool    Hit;
} IntersectionTriangle;

typedef struct {
    int     TriangleIndex; // Index of the triangle if this is a leaf node, -1 otherwise
    bool    IsLeaf; // True if this is a leaf node, false if it has children
    bool    IsHit; // True if this bounding box was hit by the ray
} IntersectionBoundingBox;

typedef struct {
    float3 Position;
    float3 Direction;
} Ray;

float3 reflectVector(float3 incident, float3 normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

IntersectionBoundingBox intersectBoundingBox(
    Ray ray, 
    __global const BVHNode *bvhNodes, 
    int nodeIndex
) {
    IntersectionBoundingBox result;
    result.IsHit = false;
    result.IsLeaf = false;
    result.TriangleIndex = -1;

    const BVHNode node = bvhNodes[nodeIndex];

    // FIX: Add safety checks for division by zero
    float tMin, tMax;
    if (fabs(ray.Direction.x) > 1e-6f) {
        tMin = (node.BoundingBox[0] - ray.Position.x) / ray.Direction.x;
        tMax = (node.BoundingBox[3] - ray.Position.x) / ray.Direction.x;
        if (tMin > tMax) {
            float temp = tMin; tMin = tMax; tMax = temp;
        }
    } else {
        // Ray is parallel to X planes
        if (ray.Position.x < node.BoundingBox[0] || ray.Position.x > node.BoundingBox[3]) {
            return result;
        }
        tMin = -FLT_MAX;
        tMax = FLT_MAX;
    }

    // Similar fixes for Y and Z axes
    float tyMin, tyMax;
    if (fabs(ray.Direction.y) > 1e-6f) {
        tyMin = (node.BoundingBox[1] - ray.Position.y) / ray.Direction.y;
        tyMax = (node.BoundingBox[4] - ray.Position.y) / ray.Direction.y;
        if (tyMin > tyMax) {
            float temp = tyMin; tyMin = tyMax; tyMax = temp;
        }
    } else {
        if (ray.Position.y < node.BoundingBox[1] || ray.Position.y > node.BoundingBox[4]) {
            return result;
        }
        tyMin = -FLT_MAX;
        tyMax = FLT_MAX;
    }

    if ((tMin > tyMax) || (tyMin > tMax)) return result;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;

    float tzMin, tzMax;
    if (fabs(ray.Direction.z) > 1e-6f) {
        tzMin = (node.BoundingBox[2] - ray.Position.z) / ray.Direction.z;
        tzMax = (node.BoundingBox[5] - ray.Position.z) / ray.Direction.z;
        if (tzMin > tzMax) {
            float temp = tzMin; tzMin = tzMax; tzMax = temp;
        }
    } else {
        if (ray.Position.z < node.BoundingBox[2] || ray.Position.z > node.BoundingBox[5]) {
            return result;
        }
        tzMin = -FLT_MAX;
        tzMax = FLT_MAX;
    }

    if ((tMin > tzMax) || (tzMin > tMax)) return result;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;

    result.IsHit = true;
    if (node.TriangleIndex != -1) {
        result.IsLeaf = true;
        result.TriangleIndex = node.TriangleIndex;
    }

    return result;
}

IntersectionTriangle intersectTriangle(
    Ray ray, 
    __global const Triangle *triangles, 
    int triangleIndex
) {
    IntersectionTriangle result;
    result.Hit = false;
    result.TriangleIndex = -1;

    const Triangle triangle = triangles[triangleIndex];

    // Möller–Trumbore intersection algorithm
    float3 edge1 = triangle.v2 - triangle.v1;
    float3 edge2 = triangle.v3 - triangle.v1;
    float3 h = cross(ray.Direction, edge2);
    float a = dot(edge1, h);

    if (fabs(a) < 1e-6f) {
        return result; // Ray is parallel to the triangle
    }

    float f = 1.0f / a;
    float3 s = ray.Position - triangle.v1;
    float u = f * dot(s, h);

    if (u < 0.0f || u > 1.0f) {
        return result; // Not hit
    }

    float3 q = cross(s, edge1);
    float v = f * dot(ray.Direction, q);

    if (v < 0.0f || u + v > 1.0f) {
        return result; // Not hit
    }

    // Calculate t to find the intersection point
    float t = f * dot(edge2, q);
    
    if (t < 0.0f) {
        return result; // Not hit
    }

    // Hit detected
    result.Hit = true;
    result.TriangleIndex = triangleIndex;
    result.PointOfIntersection = ray.Position + ray.Direction * t;
    
    // Calculate normal at intersection
    result.NormalAtIntersection = triangle.normal;
    
    // Set color at intersection
    result.ColorAtIntersection = triangle.color;
    
    // Set distance from ray origin to intersection point
    result.Distance = t;

    return result;
}


float3 Trace(Ray ray, __global const BVHLinear *bvh, int maxDepth) {
    float3 incomingLight = (float3)(0.0f, 0.0f, 0.0f);
    float3 rayColor = (float3)(1.0f, 1.0f, 1.0f);
    
    for (int depth = 0; depth < maxDepth; depth++) {
        IntersectionTriangle hit;
        hit.Hit = false;
        hit.Distance = FLT_MAX;
        hit.TriangleIndex = -1;
        
        // Traverse BVH to find closest intersection
        int stack[64]; // Stack for BVH traversal
        int stackPtr = 0;
        stack[stackPtr++] = 0; // Start with root node
        
        while (stackPtr > 0 && stackPtr < 64) {
            int nodeIndex = stack[--stackPtr];
            
            if (nodeIndex >= bvh->NodesCount) continue;
            
            // Test intersection with bounding box
            IntersectionBoundingBox boxHit = intersectBoundingBox(ray, bvh->Nodes, nodeIndex);
            
            if (!boxHit.IsHit) continue;
            
            if (boxHit.IsLeaf) {
                // Leaf node - test triangle intersection
                if (boxHit.TriangleIndex >= 0 && boxHit.TriangleIndex < bvh->TrianglesCount) {
                    IntersectionTriangle triHit = intersectTriangle(ray, bvh->Triangles, boxHit.TriangleIndex);
                    
                    if (triHit.Hit && triHit.Distance < hit.Distance && triHit.Distance > 0.001f) {
                        hit = triHit;
                    }
                }
            } else {
                // Internal node - add children to stack
                const BVHNode node = bvh->Nodes[nodeIndex];
                if (node.LeftChild >= 0 && stackPtr < 63) {
                    stack[stackPtr++] = node.LeftChild;
                }
                if (node.RightChild >= 0 && stackPtr < 63) {
                    stack[stackPtr++] = node.RightChild;
                }
            }
        }
        
        // Process intersection result
        if (!hit.Hit) {
            // No intersection - return accumulated light
            break;
        }
        
        // Add emission from hit surface
        if (hit.TriangleIndex >= 0 && hit.TriangleIndex < bvh->TrianglesCount) {
            const Triangle hitTriangle = bvh->Triangles[hit.TriangleIndex];
            float3 emission = hitTriangle.color * hitTriangle.Emission;
            incomingLight += rayColor * emission;
            
            // Simple diffuse reflection for next bounce
            float3 randomDir = normalize((float3)(
                sin((float)depth * 12.9898f) * 0.5f + 0.5f,
                cos((float)depth * 78.233f) * 0.5f + 0.5f,
                sin((float)depth * 37.719f) * 0.5f + 0.5f
            ));
            
            // Ensure reflection is in correct hemisphere
            if (dot(randomDir, hit.NormalAtIntersection) < 0.0f) {
                randomDir = -randomDir;
            }
            
            // Update ray for next bounce
            ray.Position = hit.PointOfIntersection + hit.NormalAtIntersection * 0.001f;
            ray.Direction = normalize(randomDir);
            
            // Attenuate ray color
            rayColor *= hit.ColorAtIntersection * (1.0f - hitTriangle.Metallic);
            
            // Russian roulette termination
            float maxComponent = max(max(rayColor.x, rayColor.y), rayColor.z);
            if (maxComponent < 0.1f) break;
        } else {
            break;
        }
    }
    
    return incomingLight;
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

// FIX 10: Also update the reflection usage in applyReflections
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
    const int skyBoxHeight
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
    
    // FIX 11: Better parameters for screen space reflection
    float3 screenSpaceReflection = sampleScreenSpaceReflectionFiltered(
        ScreenColors, ScreenDistances, worldPos, reflectedDir, camPos, camDir, fov,
        screenWidth, screenHeight, 
        min(500.0f, depth * 50.0f),  // Adaptive max distance
        512,                           // Reduced steps for better performance
        max(0.5f, depth * 0.1f)      // Adaptive step size
    );
    
    float3 skyboxReflection = sampleSkybox(reflectedDir, SkyBoxTop, SkyBoxBottom,
                                           SkyBoxLeft, SkyBoxRight,
                                           SkyBoxFront, SkyBoxBack,
                                           skyBoxWidth, skyBoxHeight);
    
    // FIX 12: Better fallback logic
    float screenReflectionStrength = length(screenSpaceReflection);
    float3 environmentReflection = (screenReflectionStrength > 0.01f) ? 
                                     screenSpaceReflection : skyboxReflection;
    
    float roughness = ScreenMaterialRoughness[pixelIndex]; 
    float metallic  = ScreenMaterialMetallic[pixelIndex];
    float emission  = ScreenMaterialEmission[pixelIndex];
    
    float fresnel = 0.04f + (1.0f - 0.04f) * pow(1.0f - max(0.0f, dot(normal, viewDir)), 5.0f);
    fresnel = mix(fresnel, 1.0f, metallic);
    
    // // FIX 13: Limit reflection strength to avoid too strong reflections
    // float reflectionFactor = clamp(fresnel * (1.0f - roughness) * (1.0f - emission), 0.0f, 0.8f);

    float metallicBoost = mix(1.0f, 2.0f, metallic); // Metals get 2x reflection strength
    float reflectionFactor = clamp(fresnel * (1.0f - roughness) * (1.0f - emission) * metallicBoost, 0.0f, 1.0f);
    
    int colorIndex = pixelIndex * 3;
    float3 baseColor = (float3)(ScreenColors[colorIndex], 
                                ScreenColors[colorIndex + 1], 
                                ScreenColors[colorIndex + 2]);
    
    float3 finalColor = mix(baseColor, environmentReflection, reflectionFactor);
    
    ScreenColors[colorIndex]     = clamp(finalColor.x, 0.0f, 1.0f);
    ScreenColors[colorIndex + 1] = clamp(finalColor.y, 0.0f, 1.0f);
    ScreenColors[colorIndex + 2] = clamp(finalColor.z, 0.0f, 1.0f);
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

// Helper function to calculate vertex normals from adjacent triangles
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
    // input buffers
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
    // output buffers
    __global float* projectedVerts,      // 9 floats per triangle (x,y,z for each of v1, v2, v3)
    __global float* bboxes,              // 4 floats per triangle: minX, maxX, minY, maxY
    __global int* validTriangles         // 1 = valid, 0 = culled
) { 
    int triangleId = get_global_id(0);
    if (triangleId >= numTriangles) return;

    // === Initialize outputs to invalid state ===
    validTriangles[triangleId] = 0;

    // === 1. Get triangle vertices and normal ===
    float3 vertex1 = (float3)(v1[triangleId * 3], v1[triangleId * 3 + 1], v1[triangleId * 3 + 2]);
    float3 vertex2 = (float3)(v2[triangleId * 3], v2[triangleId * 3 + 1], v2[triangleId * 3 + 2]);
    float3 vertex3 = (float3)(v3[triangleId * 3], v3[triangleId * 3 + 1], v3[triangleId * 3 + 2]);
    float3 faceNormal = normalize((float3)(normals[triangleId * 3], normals[triangleId * 3 + 1], normals[triangleId * 3 + 2]));

    // === 2. Camera basis ===
    float3 forward = normalize(camDir);
    float3 up = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, up));
    up = cross(right, forward);

    // === 3. Backface culling ===
    float3 center = (vertex1 + vertex2 + vertex3) / 3.0f;
    float3 toCamera = normalize(camPos - center);
    if (dot(faceNormal, toCamera) <= 0.0f) {
        return;
    }

    // === 4. Project to screen space ===
    float3 vertices[3] = {vertex1, vertex2, vertex3};
    float3 projected[3];
    float minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;

    for (int i = 0; i < 3; i++) {
        float3 rel = vertices[i] - camPos;
        float depth = dot(rel, forward);

        // Cull triangles behind camera
        if (depth <= 0.01f) {
            return;
        }

        float scale = 1.0f / (depth * fov);
        float x = dot(rel, right) * scale;
        float y = dot(rel, up) * scale;

        // Convert to pixel coordinates
        float sx = x * screenWidth * 0.5f + screenWidth * 0.5f;
        float sy = -y * screenHeight * 0.5f + screenHeight * 0.5f;

        projected[i] = (float3)(sx, sy, depth);

        // Track bounding box
        minX = fmin(minX, sx);
        maxX = fmax(maxX, sx);
        minY = fmin(minY, sy);
        maxY = fmax(maxY, sy);
    }

    // === 4.5. Check for degenerate triangles ===
    float triangleArea = fabs((projected[1].x - projected[0].x) * (projected[2].y - projected[0].y) - 
                             (projected[2].x - projected[0].x) * (projected[1].y - projected[0].y)) * 0.5f;
    if (triangleArea < 0.5f) {
        return;
    }

    // === 5. Store projected vertices ===
    for (int i = 0; i < 3; i++) {
        int base = triangleId * 9 + i * 3;
        projectedVerts[base + 0] = projected[i].x;
        projectedVerts[base + 1] = projected[i].y;
        projectedVerts[base + 2] = projected[i].z;
    }

    // === 6. Store bounding box ===
    bboxes[triangleId * 4 + 0] = fmax(0.0f, fmin((float)screenWidth, minX));
    bboxes[triangleId * 4 + 1] = fmax(0.0f, fmin((float)screenWidth, maxX));
    bboxes[triangleId * 4 + 2] = fmax(0.0f, fmin((float)screenHeight, minY));
    bboxes[triangleId * 4 + 3] = fmax(0.0f, fmin((float)screenHeight, maxY));

    validTriangles[triangleId] = 1;
}

__kernel void ShadePixels(
    __global const float* projectedVerts,
    __global const float* bboxes,
    __global const int* validTriangles,
    __global float* ScreenColors,
    __global float* ScreenDistances,
    __global float* ScreenNormals,

    const int screenWidth,
    const int screenHeight,
    const int numTriangles,

    __global const float* TriangleColors,
    __global const float* roughness,
    __global const float* metallic,
    __global const float* emission,

    __global float* ScreenMaterialRoughness,
    __global float* ScreenMaterialMetallic,
    __global float* ScreenMaterialEmission,
    
    // Original triangle data for normals
    __global const float* normals
) {
    int pixelX = get_global_id(0);
    int pixelY = get_global_id(1);
    if (pixelX >= screenWidth || pixelY >= screenHeight) return;
    
    int pixelIndex = pixelY * screenWidth + pixelX;
    float2 pixelPos = (float2)(pixelX + 0.5f, pixelY + 0.5f);
    
    float closestDepth = FLT_MAX;
    int closestTriangle = -1;
    
    // **OPTIMIZATION 1: Process triangles in chunks for better cache locality**
    const int CHUNK_SIZE = 64;
    
    for (int chunkStart = 0; chunkStart < numTriangles; chunkStart += CHUNK_SIZE) {
        int chunkEnd = min(chunkStart + CHUNK_SIZE, numTriangles);
        
        // **OPTIMIZATION 2: Pre-filter valid triangles in this chunk**
        int validCount = 0;
        int validIds[64]; // Fixed size array
        
        for (int i = chunkStart; i < chunkEnd; i++) {
            if (validTriangles[i] != 0) {
                // **OPTIMIZATION 3: Quick bounding box pre-filter**
                float4 bbox = (float4)(bboxes[i * 4 + 0], bboxes[i * 4 + 1], 
                                      bboxes[i * 4 + 2], bboxes[i * 4 + 3]);
                
                if (pixelPos.x >= bbox.x && pixelPos.x <= bbox.y && 
                    pixelPos.y >= bbox.z && pixelPos.y <= bbox.w) {
                    validIds[validCount++] = i;
                    if (validCount >= 64) break; // Prevent overflow
                }
            }
        }
        
        // **OPTIMIZATION 4: Only process triangles that passed bbox test**
        for (int vi = 0; vi < validCount; vi++) {
            int triangleId = validIds[vi];
            
            // **OPTIMIZATION 5: Load vertices as float3 directly**
            int vertBase = triangleId * 9;
            float3 v1_proj = vload3(0, &projectedVerts[vertBase]);
            float3 v2_proj = vload3(0, &projectedVerts[vertBase + 3]);
            float3 v3_proj = vload3(0, &projectedVerts[vertBase + 6]);
            
            // **OPTIMIZATION 6: Optimized barycentric using edge function**
            float2 v0 = v2_proj.xy - v1_proj.xy;
            float2 v1 = v3_proj.xy - v1_proj.xy;
            float2 v2 = pixelPos - v1_proj.xy;
            
            float denom = v0.x * v1.y - v1.x * v0.y;
            
            // **OPTIMIZATION 7: Early exit on degenerate triangles**
            if (fabs(denom) < 1e-6f) continue;
            
            float invDenom = 1.0f / denom;
            float u = (v2.x * v1.y - v1.x * v2.y) * invDenom;
            float v = (v0.x * v2.y - v2.x * v0.y) * invDenom;
            
            // **OPTIMIZATION 8: Single bounds check with early exit**
            if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
                float w = 1.0f - u - v;
                float depth = w * v1_proj.z + u * v2_proj.z + v * v3_proj.z;
                
                // **OPTIMIZATION 9: Early depth rejection**
                if (depth < closestDepth) {
                    closestDepth = depth;
                    closestTriangle = triangleId;
                }
            }
        }
        
        // **OPTIMIZATION 10: Early termination for very close triangles**
        if (closestDepth < 0.1f) break;
    }
    
    // **OPTIMIZATION 11: Single write pass at the end**
    if (closestTriangle >= 0 && (ScreenDistances[pixelIndex] == 0.0f || closestDepth < ScreenDistances[pixelIndex])) {
        ScreenDistances[pixelIndex] = closestDepth;
        
        // **OPTIMIZATION 12: Vectorized memory loads**
        int materialBase = closestTriangle * 3;
        float3 faceNormal = vload3(0, &normals[materialBase]);
        float3 triangleColor = vload3(0, &TriangleColors[materialBase]);
        
        float3 interpolatedNormal = normalize(faceNormal);
        
        // **OPTIMIZATION 13: Vectorized lighting calculation**
        float3 simpleLight = (float3)(0.3f, 0.7f, 0.5f);
        float lightIntensity = max(0.65f, dot(interpolatedNormal, simpleLight));
        float3 finalColor = triangleColor * lightIntensity;
        
        float emissionValue = emission[closestTriangle];
        finalColor = mad(triangleColor, emissionValue, finalColor); // Use mad for efficiency
        finalColor = clamp(finalColor, 0.0f, 1.0f);
        
        // **OPTIMIZATION 14: Coalesced memory writes**
        vstore3(interpolatedNormal, pixelIndex, ScreenNormals);
        vstore3(finalColor, pixelIndex, ScreenColors);
        
        // Store material properties
        ScreenMaterialRoughness[pixelIndex] = roughness[closestTriangle];
        ScreenMaterialMetallic[pixelIndex] = metallic[closestTriangle];
        ScreenMaterialEmission[pixelIndex] = emissionValue;
    }
}

// // Updated renderTriangles kernel with improved vertex normal calculation
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

    // Get material properties for this triangle
    float Roughness = roughness[triangleId];
    float Metallic = metallic[triangleId];
    float Emission = emission[triangleId];

    // Add bounds checking for array access
    int vertexIndex = triangleId * 3;
    if (vertexIndex + 2 >= numTriangles * 3) return;

    // Get triangle vertices
    float3 vertex1 = (float3)(v1[vertexIndex], v1[vertexIndex + 1], v1[vertexIndex + 2]);
    float3 vertex2 = (float3)(v2[triangleId * 3], v2[triangleId * 3 + 1], v2[triangleId * 3 + 2]);
    float3 vertex3 = (float3)(v3[triangleId * 3], v3[triangleId * 3 + 1], v3[triangleId * 3 + 2]);
    
    // Get triangle face normal
    float3 faceNormal = (float3)(normals[triangleId * 3], normals[triangleId * 3 + 1], normals[triangleId * 3 + 2]);
    float3 triangleColor = (float3)(TriangleColors[triangleId * 3], 
                                   TriangleColors[triangleId * 3 + 1], 
                                   TriangleColors[triangleId * 3 + 2]);

    float3 vertexNormal1 = normalize(faceNormal);
    float3 vertexNormal2 = normalize(faceNormal);
    float3 vertexNormal3 = normalize(faceNormal);

    // === OPTIMIZATION 1: BACK-FACE CULLING ===
    float3 triangleCenter = (vertex1 + vertex2 + vertex3) / 3.0f;
    float3 viewDirection = normalize(camPos - triangleCenter);
    
    // Fixed: For back-face culling, cull when normal points away from viewer
    if (dot(faceNormal, viewDirection) <= 0.0f) {
        return; // Back-face culling
    }

    // Compute camera basis
    float3 forward = normalize(camDir);
    float3 camUp = (float3)(0.0f, 1.0f, 0.0f);
    float3 right = normalize(cross(forward, camUp));
    float3 up = cross(right, forward);

    // === OPTIMIZATION 2: EARLY DEPTH REJECTION ===
    float3 rel1 = vertex1 - camPos;
    float3 rel2 = vertex2 - camPos;
    float3 rel3 = vertex3 - camPos;
    
    float depth1 = dot(rel1, forward);
    float depth2 = dot(rel2, forward);
    float depth3 = dot(rel3, forward);
    
    if (depth1 <= 0.001f && depth2 <= 0.001f && depth3 <= 0.001f) return;
    if (depth1 <= 0.001f || depth2 <= 0.001f || depth3 <= 0.001f) return;

    // Project vertices to screen space
    float3 screenPos1, screenPos2, screenPos3;
    
    float fovScale1 = 1.0f / (depth1 * fov);
    screenPos1.x = dot(rel1, right) * fovScale1 * screenWidth * 0.5f + screenWidth * 0.5f;
    screenPos1.y = -dot(rel1, up) * fovScale1 * screenHeight * 0.5f + screenHeight * 0.5f;
    screenPos1.z = depth1;

    float fovScale2 = 1.0f / (depth2 * fov);
    screenPos2.x = dot(rel2, right) * fovScale2 * screenWidth * 0.5f + screenWidth * 0.5f;
    screenPos2.y = -dot(rel2, up) * fovScale2 * screenHeight * 0.5f + screenHeight * 0.5f;
    screenPos2.z = depth2;

    float fovScale3 = 1.0f / (depth3 * fov);
    screenPos3.x = dot(rel3, right) * fovScale3 * screenWidth * 0.5f + screenWidth * 0.5f;
    screenPos3.y = -dot(rel3, up) * fovScale3 * screenHeight * 0.5f + screenHeight * 0.5f;
    screenPos3.z = depth3;

    // === OPTIMIZATION 3: FRUSTUM CULLING ===
    float minX_f = min(min(screenPos1.x, screenPos2.x), screenPos3.x);
    float maxX_f = max(max(screenPos1.x, screenPos2.x), screenPos3.x);
    float minY_f = min(min(screenPos1.y, screenPos2.y), screenPos3.y);
    float maxY_f = max(max(screenPos1.y, screenPos2.y), screenPos3.y);
    
    if (maxX_f < 0 || minX_f >= screenWidth || maxY_f < 0 || minY_f >= screenHeight) {
        return;
    }

    int minX = max(0, (int)minX_f);
    int maxX = min(screenWidth - 1, (int)maxX_f);
    int minY = max(0, (int)minY_f);
    int maxY = min(screenHeight - 1, (int)maxY_f);

    // === OPTIMIZATION 4: SMALL TRIANGLE CULLING ===
    float triangleArea = fabs((screenPos2.x - screenPos1.x) * (screenPos3.y - screenPos1.y) - 
                            (screenPos3.x - screenPos1.x) * (screenPos2.y - screenPos1.y)) * 0.5f;
    if (triangleArea < 0.5f) {
        return;
    }

    // === OPTIMIZATION 5: PRECOMPUTE BARYCENTRIC CONSTANTS ===
    float2 edge1 = screenPos3.xy - screenPos1.xy;
    float2 edge2 = screenPos2.xy - screenPos1.xy;
    
    float dot00 = dot(edge1, edge1);
    float dot01 = dot(edge1, edge2);
    float dot11 = dot(edge2, edge2);
    
    float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    
    if (isinf(invDenom) || isnan(invDenom)) {
        return;
    }

    // Rasterize triangle
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float2 pixelPos = (float2)(x + 0.5f, y + 0.5f) - screenPos1.xy;
            
            float dot02 = dot(edge1, pixelPos);
            float dot12 = dot(edge2, pixelPos);

            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

            if (u >= 0 && v >= 0 && u + v <= 1) {
                float w = 1.0f - u - v;
                
                float depth = w * screenPos1.z + u * screenPos2.z + v * screenPos3.z;
                
                int pixelIndex = y * screenWidth + x;
                
                if (ScreenDistances[pixelIndex] != 0.0f && depth >= ScreenDistances[pixelIndex]) {
                    continue;
                }
                
                ScreenDistances[pixelIndex] = depth;
                
                // IMPROVED: Interpolate the calculated vertex normals using barycentric coordinates
                float3 interpolatedNormal = normalize(w * vertexNormal1 + u * vertexNormal2 + v * vertexNormal3);
                
                // Store interpolated normal
                int normalIndex = pixelIndex * 3;
                ScreenNormals[normalIndex] = interpolatedNormal.x;
                ScreenNormals[normalIndex + 1] = interpolatedNormal.y;
                ScreenNormals[normalIndex + 2] = interpolatedNormal.z;
                
                // === SIMPLE SHADING ===
                // Just use a simple normal-based shading for depth perception
                float3 simpleLight = normalize((float3)(0.3f, 0.7f, 0.5f)); // Soft light direction
                float lightIntensity = max(0.65f, dot(interpolatedNormal, simpleLight)); // Clamp to avoid too dark areas
                
                float3 finalColor = triangleColor * lightIntensity;
                
                // Add emission if present
                float3 emissionColor = triangleColor * Emission;
                finalColor += emissionColor;
                
                // Clamp color values
                finalColor = clamp(finalColor, 0.0f, 1.0f);
                
                // Store color
                int colorIndex = pixelIndex * 3;
                ScreenColors[colorIndex] = finalColor.x;
                ScreenColors[colorIndex + 1] = finalColor.y;
                ScreenColors[colorIndex + 2] = finalColor.z;

                // Store material properties
                ScreenMaterialRoughness[pixelIndex] = Roughness;
                ScreenMaterialMetallic[pixelIndex] = Metallic;
                ScreenMaterialEmission[pixelIndex] = Emission;
            }
        }
    }
}