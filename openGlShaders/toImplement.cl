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