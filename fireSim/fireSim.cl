uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >>  6u);
    x += (x <<  3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

float rand(uint seed) {
    return hash(seed) / 4294967296.0f;
}

float randRange(uint seed, float min, float max) {
    return min + (max - min) * rand(seed);
}

__kernel void fireSim(
    __global float* posX,
    __global float* posY,
    __global float* posZ,
    __global float* velX,
    __global float* velY,
    __global float* velZ,
    __global float* lifeTime,
    const float3 basePos,
    const float maxLifeTime,
    const float deltaTime
) 
{
    int id = get_global_id(0);
    
    uint timeSeed = (uint)(lifeTime[id] * 1000.0f) + id * 12345u;
    
    float ageFactor = lifeTime[id] / maxLifeTime;
    float invAgeFactor = 1.0f - ageFactor;
    
    float buoyancy = 8.0f * invAgeFactor;
    velY[id] += buoyancy * deltaTime;
    
    float drag = 0.985f;
    velX[id] *= drag;
    velY[id] *= drag;
    velZ[id] *= drag;
    
    float turbulence = invAgeFactor * 1.5f;
    velX[id] += randRange(timeSeed + 100, -turbulence, turbulence) * deltaTime;
    velY[id] += randRange(timeSeed + 200, -turbulence * 0.3f, turbulence * 0.5f) * deltaTime;
    velZ[id] += randRange(timeSeed + 300, -turbulence, turbulence) * deltaTime;
    
    velX[id] += 1.0f * invAgeFactor * deltaTime;
    
    float swirl = invAgeFactor * 0.8f;
    float angle = lifeTime[id] * 3.0f + id * 0.1f;
    velX[id] += cos(angle) * swirl * deltaTime;
    velZ[id] += sin(angle) * swirl * deltaTime;
    
    posX[id] += velX[id] * deltaTime;
    posY[id] += velY[id] * deltaTime;
    posZ[id] += velZ[id] * deltaTime;
    
    lifeTime[id] += deltaTime;
    
    if (lifeTime[id] > maxLifeTime) {
        uint spawnSeed = id * 987654u;
        
        posX[id] = basePos.x + randRange(spawnSeed + 10, -0.4f, 0.4f);
        posY[id] = basePos.y + randRange(spawnSeed + 11, -0.1f, 0.3f);
        posZ[id] = basePos.z + randRange(spawnSeed + 12, -0.4f, 0.4f);
        
        velX[id] = randRange(spawnSeed + 13, -0.8f, 0.8f);
        velY[id] = randRange(spawnSeed + 14, 2.5f, 5.5f);
        velZ[id] = randRange(spawnSeed + 15, -0.8f, 0.8f);
        
        lifeTime[id] = randRange(spawnSeed + 16, 0.0f, 0.2f);
    }
}

__kernel void renderParticles(
    __global const float* posX,
    __global const float* posY,
    __global const float* posZ,
    __global const float* lifeTime,
    __global float* colorBuffer,
    __global float* depthBuffer,
    const float3 baseColor,
    const float3 fireColor,
    const float3 smokeColor,
    const float maxLifeTime,
    const float particleSize,
    const int screenWidth,
    const int screenHeight,
    const float16 viewMatrix,
    const float16 projMatrix
) {
    int id = get_global_id(0);
    
    float3 worldPos = (float3)(posX[id], posY[id], posZ[id]);
    
    float4 viewPos = (float4)(
        dot(worldPos, viewMatrix.s012) + viewMatrix.s3,
        dot(worldPos, viewMatrix.s456) + viewMatrix.s7,
        dot(worldPos, viewMatrix.s89a) + viewMatrix.sb,
        1.0f
    );
    
    if (viewPos.z <= 0.0f) return;
    
    float4 clipPos = (float4)(
        dot(viewPos, (float4)(projMatrix.s0, projMatrix.s4, projMatrix.s8, projMatrix.sc)),
        dot(viewPos, (float4)(projMatrix.s1, projMatrix.s5, projMatrix.s9, projMatrix.sd)),
        dot(viewPos, (float4)(projMatrix.s2, projMatrix.s6, projMatrix.sa, projMatrix.se)),
        dot(viewPos, (float4)(projMatrix.s3, projMatrix.s7, projMatrix.sb, projMatrix.sf))
    );
    
    if (clipPos.w <= 0.0f) return;
    
    float2 screenPos = (float2)(
        (clipPos.x / clipPos.w + 1.0f) * 0.5f * screenWidth,
        (1.0f - clipPos.y / clipPos.w) * 0.5f * screenHeight
    );
    
    float depth = clipPos.z / clipPos.w;
    float ageFactor = lifeTime[id] / maxLifeTime;
    
    float3 color;
    if (ageFactor < 0.3f) {
        color = mix(baseColor, fireColor, ageFactor / 0.3f);
    } else if (ageFactor < 0.7f) {
        color = mix(fireColor, smokeColor, (ageFactor - 0.3f) / 0.4f);
    } else {
        color = smokeColor * (1.0f - ageFactor);
    }
    
    float alpha = (1.0f - ageFactor) * 0.8f;
    float size = particleSize * (1.0f + ageFactor * 0.5f);
    
    int minX = max(0, (int)(screenPos.x - size));
    int maxX = min(screenWidth - 1, (int)(screenPos.x + size));
    int minY = max(0, (int)(screenPos.y - size));
    int maxY = min(screenHeight - 1, (int)(screenPos.y + size));
    
    for (int y = minY; y <= maxY; y++) {
        for (int x = minX; x <= maxX; x++) {
            float dx = x - screenPos.x;
            float dy = y - screenPos.y;
            float dist = sqrt(dx * dx + dy * dy);
            
            if (dist <= size) {
                int idx = (y * screenWidth + x) * 3;
                int depthIdx = y * screenWidth + x;
                
                if (depth < depthBuffer[depthIdx] || depthBuffer[depthIdx] == 0.0f) {
                    float falloff = 1.0f - (dist / size);
                    falloff = falloff * falloff;
                    float contribution = alpha * falloff * 0.1f;
                    
                    colorBuffer[idx] += color.x * contribution;
                    colorBuffer[idx + 1] += color.y * contribution;
                    colorBuffer[idx + 2] += color.z * contribution;
                    
                    depthBuffer[depthIdx] = depth;
                }
            }
        }
    }
}

__kernel void blurFire(
    __global float* colorBuffer,
    __global float* tempBuffer,
    const int screenWidth,
    const int screenHeight,
    const int pass
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= screenWidth || y >= screenHeight) return;
    
    int idx = (y * screenWidth + x) * 3;
    
    if (pass == 0) {
        float3 color = (float3)(0.0f);
        float weights[5] = {0.06f, 0.24f, 0.4f, 0.24f, 0.06f};
        
        for (int i = -2; i <= 2; i++) {
            int nx = clamp(x + i, 0, screenWidth - 1);
            int nidx = (y * screenWidth + nx) * 3;
            color.x += colorBuffer[nidx] * weights[i + 2];
            color.y += colorBuffer[nidx + 1] * weights[i + 2];
            color.z += colorBuffer[nidx + 2] * weights[i + 2];
        }
        
        tempBuffer[idx] = color.x;
        tempBuffer[idx + 1] = color.y;
        tempBuffer[idx + 2] = color.z;
    } else {
        float3 color = (float3)(0.0f);
        float weights[5] = {0.06f, 0.24f, 0.4f, 0.24f, 0.06f};
        
        for (int i = -2; i <= 2; i++) {
            int ny = clamp(y + i, 0, screenHeight - 1);
            int nidx = (ny * screenWidth + x) * 3;
            color.x += tempBuffer[nidx] * weights[i + 2];
            color.y += tempBuffer[nidx + 1] * weights[i + 2];
            color.z += tempBuffer[nidx + 2] * weights[i + 2];
        }
        
        colorBuffer[idx] = color.x;
        colorBuffer[idx + 1] = color.y;
        colorBuffer[idx + 2] = color.z;
    }
}