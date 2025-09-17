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


float fract(float x) {
    return x - floor(x);
}

// Add this helper function for better random number generation
float hash(float seed) {
    return fract(sin(seed * 12.9898f) * 43758.5453f);
}

float3 generateRoughnessBiasedDirection(
    float3 normal, 
    float3 perfectReflection, 
    float roughness, 
    float randomSeed
) {
    // Generate random numbers for spherical coordinates
    float r1 = hash(randomSeed * 73.156f);
    float r2 = hash(randomSeed * 47.832f);
    
    // Convert roughness to cone angle (more roughness = wider cone)
    float maxAngle = roughness * 1.57079632679f; // roughness * PI/2 (max 90 degrees)
    
    // Generate random direction within cone around perfect reflection
    float cosTheta = cos(r1 * maxAngle);
    float sinTheta = sin(r1 * maxAngle);
    float phi = r2 * 6.28318530718f; // 2 * PI
    
    // Create local coordinate system around perfect reflection
    float3 up = (fabs(perfectReflection.z) < 0.999f) ? 
                (float3)(0.0f, 0.0f, 1.0f) : (float3)(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(up, perfectReflection));
    float3 bitangent = cross(perfectReflection, tangent);
    
    // Generate direction in cone
    float3 randomDir = sinTheta * cos(phi) * tangent + 
                       sinTheta * sin(phi) * bitangent + 
                       cosTheta * perfectReflection;
    
    // For very rough surfaces, blend with diffuse (Lambertian) reflection
    if (roughness > 0.5f) {
        // Generate diffuse direction
        float3 diffuseDir = normalize(normal + (float3)(
            hash(randomSeed * 91.234f) * 2.0f - 1.0f,
            hash(randomSeed * 67.891f) * 2.0f - 1.0f,
            hash(randomSeed * 123.456f) * 2.0f - 1.0f
        ));
        
        // Ensure diffuse direction is in correct hemisphere
        if (dot(diffuseDir, normal) < 0.0f) {
            diffuseDir = -diffuseDir;
        }
        
        // Blend between specular and diffuse based on roughness
        float blendFactor = (roughness - 0.5f) * 2.0f; // 0 to 1 for roughness 0.5 to 1.0
        randomDir = normalize(mix(randomDir, diffuseDir, blendFactor));
    }
    
    return normalize(randomDir);
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
        int stack[32]; // Reduced stack size
        int stackPtr = 0;
        stack[stackPtr++] = 0;
        
        while (stackPtr > 0 && stackPtr < 32) {
            int nodeIndex = stack[--stackPtr];
            
            if (nodeIndex >= bvh->NodesCount || nodeIndex < 0) continue;
            
            IntersectionBoundingBox boxHit = intersectBoundingBox(ray, bvh->Nodes, nodeIndex);
            
            if (!boxHit.IsHit) continue;
            
            if (boxHit.IsLeaf) {
                if (boxHit.TriangleIndex >= 0 && boxHit.TriangleIndex < bvh->TrianglesCount) {
                    IntersectionTriangle triHit = intersectTriangle(ray, bvh->Triangles, boxHit.TriangleIndex);
                    
                    if (triHit.Hit && triHit.Distance < hit.Distance && triHit.Distance > 0.001f) {
                        hit = triHit;
                    }
                }
            } else {
                const BVHNode node = bvh->Nodes[nodeIndex];
                if (node.LeftChild >= 0 && stackPtr < 31) {
                    stack[stackPtr++] = node.LeftChild;
                }
                if (node.RightChild >= 0 && stackPtr < 31) {
                    stack[stackPtr++] = node.RightChild;
                }
            }
        }
        
        if (!hit.Hit) break;
        
        if (hit.TriangleIndex >= 0 && hit.TriangleIndex < bvh->TrianglesCount) {
            const Triangle hitTriangle = bvh->Triangles[hit.TriangleIndex];
            float3 emission = hitTriangle.color * hitTriangle.Emission;
            float roughness = hitTriangle.Roughness;
            float metallic = hitTriangle.Metallic;
            
            incomingLight += rayColor * emission;
            
            // Calculate perfect reflection direction
            float3 incidentDir = normalize(ray.Direction);
            float3 normal = normalize(hit.NormalAtIntersection);
            float3 perfectReflection = reflectVector(incidentDir, normal);
            
            // Generate roughness-based random seed
            float randomSeed = (float)depth + dot(hit.PointOfIntersection, (float3)(12.9898f, 78.233f, 37.719f));
            
            // Generate direction based on roughness
            float3 newDirection;
            if (metallic > 0.5f) {
                // For metals: use roughness-biased reflection
                newDirection = generateRoughnessBiasedDirection(normal, perfectReflection, roughness, randomSeed);
            } else {
                // For dielectrics: mix between diffuse and specular based on roughness
                if (roughness > 0.8f) {
                    // Very rough surface - mostly diffuse
                    float3 diffuseDir = normalize(normal + (float3)(
                        hash(randomSeed * 91.234f) * 2.0f - 1.0f,
                        hash(randomSeed * 67.891f) * 2.0f - 1.0f,
                        hash(randomSeed * 123.456f) * 2.0f - 1.0f
                    ));
                    
                    if (dot(diffuseDir, normal) < 0.0f) {
                        diffuseDir = -diffuseDir;
                    }
                    newDirection = diffuseDir;
                } else {
                    // Smooth to moderately rough - use roughness-biased reflection
                    newDirection = generateRoughnessBiasedDirection(normal, perfectReflection, roughness, randomSeed);
                }
            }
            
            // Update ray for next bounce
            ray.Position = hit.PointOfIntersection + normal * 0.001f;
            ray.Direction = normalize(newDirection);
            
            // Attenuate ray color based on material properties
            float3 baseReflectance = mix(hit.ColorAtIntersection, (float3)(0.04f, 0.04f, 0.04f), metallic);
            
            // Fresnel calculation
            float cosTheta = max(0.0f, dot(-incidentDir, normal));
            float3 F0 = mix((float3)(0.04f, 0.04f, 0.04f), hit.ColorAtIntersection, metallic);
            float3 fresnel = F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
            
            // Energy conservation: less diffuse for rough metals
            float diffuseWeight = (1.0f - metallic) * (1.0f - roughness * 0.5f);
            float3 diffuseColor = hit.ColorAtIntersection * diffuseWeight;
            float3 specularColor = fresnel;
            
            rayColor *= (diffuseColor + specularColor) * (1.0f + roughness * 0.2f); // Slight energy boost for rough surfaces
            
            // Russian roulette termination with roughness consideration
            float maxComponent = max(max(rayColor.x, rayColor.y), rayColor.z);
            float survivalProbability = min(0.95f, maxComponent * (1.0f + roughness * 0.3f));
            
            if (hash(randomSeed * 151.847f) > survivalProbability) break;
            
            rayColor /= survivalProbability; // Unbiased estimator
        } else {
            break;
        }
    }
    
    return incomingLight;
}

__kernel void applyRayTracedReflections(
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

    float3 rayTracedColor = Trace(
        (Ray){worldPos, rayDir}, 
        (__global const BVHLinear *)ScreenColors, // Assuming ScreenColors contains BVH data
        3 // Max depth for ray tracing
    );

    // Mix ScreenColors with ray-traced reflections
    int colorIndex = pixelIndex * 3;
    float3 currentColor = (float3)(ScreenColors[colorIndex], 
                                   ScreenColors[colorIndex + 1], 
                                   ScreenColors[colorIndex + 2]);
    float3 finalColor = mix(currentColor, rayTracedColor, 0.5f); // 50% reflection mix
    
    ScreenColors[colorIndex]     = clamp(finalColor.x, 0.0f, 1.0f);
    ScreenColors[colorIndex + 1] = clamp(finalColor.y, 0.0f, 1.0f);
    ScreenColors[colorIndex + 2] = clamp(finalColor.z, 0.0f, 1.0f);
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
    
    __global const float* normals
) {
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= screenWidth || py >= screenHeight) return;

    const float cx = (float)px + 0.5f;
    const float cy = (float)py + 0.5f;
    const int idx = py * screenWidth + px;

    // start with existing depth (or FLT_MAX if zero)
    float bestDepth = ScreenDistances[idx] > 0.0f
        ? ScreenDistances[idx] : FLT_MAX;
    int   bestTri   = -1;

    const int CHUNK = 64;
    for (int s = 0; s < numTriangles; s += CHUNK) {
        int e = min(s + CHUNK, numTriangles);
        for (int t = s; t < e; ++t) {
            if (validTriangles[t] == 0) continue;

            // bbox test
            int bi = t * 4;
            float minX = bboxes[bi  ], maxX = bboxes[bi+1];
            float minY = bboxes[bi+2], maxY = bboxes[bi+3];
            if (cx < minX || cx > maxX || cy < minY || cy > maxY)
                continue;

            // load projected verts
            int ov = t * 9;
            float2 p0 = (float2)(projectedVerts[ov], projectedVerts[ov+1]);
            float2 p1 = (float2)(projectedVerts[ov+3], projectedVerts[ov+4]);
            float2 p2 = (float2)(projectedVerts[ov+6], projectedVerts[ov+7]);
            float   z0 = projectedVerts[ov+2];
            float   z1 = projectedVerts[ov+5];
            float   z2 = projectedVerts[ov+8];

            // edge‐function / barycentric
            float2 v0 = p1 - p0;
            float2 v1 = p2 - p0;
            float2 v2 = (float2)(cx, cy) - p0;
            float  denom = v0.x * v1.y - v1.x * v0.y;
            if (fabs(denom) < 1e-6f) continue;
            float invDen = 1.0f / denom;
            float u = (v2.x * v1.y - v1.x * v2.y) * invDen;
            float v = (v0.x * v2.y - v2.x * v0.y) * invDen;
            if (u < 0.0f || v < 0.0f || u + v > 1.0f) continue;

            // interpolate depth
            float w = 1.0f - u - v;
            float d = w*z0 + u*z1 + v*z2;
            if (d < bestDepth) {
                bestDepth = d;
                bestTri   = t;
            }
        }
        if (bestDepth < 0.1f) break;
    }

    // commit result
    if (bestTri >= 0 && (ScreenDistances[idx] == 0.0f || bestDepth < ScreenDistances[idx])) {
        ScreenDistances[idx] = bestDepth;

        // normal + color load
        int mb = bestTri * 3;
        float3 N = normalize(vload3(0, normals + mb));
        vstore3(N, idx, ScreenNormals);

        float3 C = vload3(0, TriangleColors + mb);
        float  L = max(0.65f, dot(N, (float3)(0.3f,0.7f,0.5f)));
        float3 col = clamp(C * L + C * emission[bestTri], 0.0f, 1.0f);
        vstore3(col, idx, ScreenColors);

        ScreenMaterialRoughness[idx] = roughness[bestTri];
        ScreenMaterialMetallic [idx] = metallic [bestTri];
        ScreenMaterialEmission [idx] = emission[bestTri];
    }
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

__kernel void antiAlias(
    __global float* input_colors,
    __global float* input_distances,
    int screen_width,
    int screen_height
    // int mode // 0 = 3x float, 1 = 4x float, 2 = 1x float TODO
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screen_width || y >= screen_height) return;
    
    int idx = (y * screen_width + x) * 3; // Index for a float array
    float3 center_color = (float3)(input_colors[idx], input_colors[idx+1], input_colors[idx+2]);
    float center_distance = input_distances[y * screen_width + x];
    
    if (length(center_color) == 0.0f) {
        // If center pixel is black, no need to process
        return;
    }
    
    float3 accum_color = center_color;
    int count = 1;
    
    // Check 8 neighboring pixels
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue; // Skip center pixel
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < screen_width && ny >= 0 && ny < screen_height) {
                int nidx = (ny * screen_width + nx) * 3;
                float3 neighbor_color = (float3)(input_colors[nidx], input_colors[nidx+1], input_colors[nidx+2]);
                float neighbor_distance = input_distances[ny * screen_width + nx];
                
                // Only consider neighbors with similar distance
                if (length(neighbor_color) > 0.0f && fabs(neighbor_distance - center_distance) < 0.01f) {
                    accum_color += neighbor_color;
                    count++;
                }
            }
        }
    }
    
    // Average the accumulated color
    if (count > 1) {
        accum_color /= (float)count;
        input_colors[idx]   = accum_color.x;
        input_colors[idx+1] = accum_color.y;
        input_colors[idx+2] = accum_color.z;
    }
}