#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>     // for rand()
#include <unistd.h>     // for usleep()
#include <math.h>       // for sqrtf()
#include <time.h>      // for time()
#include <string.h>     // for memset()
#include <stdbool.h>    // for bool, true, false
#include <immintrin.h> // for AVX intrinsics
#include <omp.h>        // for OpenMP
#include <pthread.h> // for pthreads

#define NUM_PARTICLES 150000
#define GRAVITY 10.0f
#define DAMPING 0.985f
#define ScreenWidth 800
#define ScreenHeight 600
#define PARTICLE_RADIUS 4
#define gridResolutionAxis 32
#define gridResolution (gridResolutionAxis * gridResolutionAxis * gridResolutionAxis)
#define temperature 10.0f
#define pressure  temperature * 0.1f
#define FrameCount 30
#define NUM_THREADS 16
pthread_t threads[NUM_THREADS];
struct ThreadData* threadData[NUM_THREADS];
volatile int ready[NUM_THREADS];
volatile int done[NUM_THREADS];
static int threadIds[NUM_THREADS];

// Function prototypes
void calculateParticleScreenCoordinates(struct ThreadData *threadData);
void *threadFunction(void *arg);

void createThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        ready[i] = 0;
        done[i] = 0;
        threadIds[i] = i;  // Use the static array instead of malloc
        pthread_create(&threads[i], NULL, threadFunction, &threadIds[i]);
    }
}

void *threadFunction(void *arg) {
    int threadId = *(int *)arg;
    
    while (!done[threadId]) {
        // Wait for signal to process data
        while (!ready[threadId] && !done[threadId]) {
            // Remove sleep or use a more efficient wait strategy
            // Consider using compiler intrinsics for a brief pause
            #if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
            #else
                // Shorter sleep time for non-x86 platforms
                usleep(100); 
            #endif
        }
        
        if (done[threadId]) {
            break;
        }
        
        // Process data
        calculateParticleScreenCoordinates(threadData[threadId]);
        
        // Signal that processing is complete
        ready[threadId] = 0;
    }
    
    return NULL;
}


void joinThreads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        done[i] = 1;
        pthread_join(threads[i], NULL);
    }
}


struct Screen {
    uint8_t distance[ScreenWidth][ScreenHeight];
    uint8_t velocity[ScreenWidth][ScreenHeight];
    uint8_t normalizedOpacity[ScreenWidth][ScreenHeight];
    uint16_t opacity[ScreenWidth][ScreenHeight];
};


struct Ray {
    float origin[3];
    float direction[3];
};

struct Camera {
    struct Ray ray;
    float fov;
};

struct Cursor {
    float x;
    float y;
    float z;
    float force;
    bool active;
};

struct TimePartition {
    float collisionTime;
    float applyPressureTime;
    float updateParticlesTime;
    float moveToBoxTime;
    float updateGridTime;
    float renderTime;
    float clearScreenTime;
    float projectParticlesTime;
    float drawCursorTime;
    float drawBoundingBoxTime;
    float saveScreenTime;
    float sortTime;
    float projectionTime;
    float renderDistanceVelocityTime;
    float renderOpacityTime;
};


struct PointSOA {
    float   x[NUM_PARTICLES];
    float   y[NUM_PARTICLES];
    float   z[NUM_PARTICLES];
    float   xVelocity[NUM_PARTICLES];
    float   yVelocity[NUM_PARTICLES];
    float   zVelocity[NUM_PARTICLES];
    float   totalVelocity[NUM_PARTICLES];
    float   distance[NUM_PARTICLES];
    float   bBoxMin[3];
    float   bBoxMax[3];
    int     gridID[NUM_PARTICLES];
    int     startIndex[gridResolution];
    int     numberOfParticle[gridResolution];
    int     screenX[NUM_PARTICLES];
    int     screenY[NUM_PARTICLES];
};

struct ThreadsData {
    struct ThreadData *threadData[NUM_THREADS]; // array of thread data
};

struct ThreadData {
    struct Camera *camera; // pointer to the camera
    float *xPtrStart;      // pointer to start of the x array
    float *yPtrStart;      // pointer to start of the y array
    float *zPtrStart;      // pointer to start of the z array
    float *xVelocityPtrStart;    // pointer to start of the xVelocity array
    float *yVelocityPtrStart;    // pointer to start of the yVelocity array
    float *zVelocityPtrStart;    // pointer to start of the zVelocity array
    float *totalVelocityPtrStart; // pointer to start of the totalVelocity array
    int *screenXPtrStart;      // pointer to start of the screenX array (changed from float* to int*)
    int *screenYPtrStart;      // pointer to start of the screenY array (changed from float* to int*)
    int length;            // length of the arrays
    float maxVelocity; // maximum velocity for normalization
    pthread_t thread;  // Add thread handle
};

void asignDataToThreads(struct PointSOA *particles, struct Camera *camera, int ValidParticles) {
    // Use dynamic chunk sizes for better load balancing
    // Last thread may get slightly more/fewer particles
    int baseChunkSize = ValidParticles / NUM_THREADS;
    int remainder = ValidParticles % NUM_THREADS;
    
    int startIdx = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        // Calculate chunk size - distribute remainder particles one per thread
        int chunkSize = baseChunkSize + (i < remainder ? 1 : 0);
        
        threadData[i]->camera = camera;
        threadData[i]->xPtrStart = &particles->x[startIdx];
        threadData[i]->yPtrStart = &particles->y[startIdx];
        threadData[i]->zPtrStart = &particles->z[startIdx];
        threadData[i]->xVelocityPtrStart = &particles->xVelocity[startIdx];
        threadData[i]->yVelocityPtrStart = &particles->yVelocity[startIdx];
        threadData[i]->zVelocityPtrStart = &particles->zVelocity[startIdx];
        threadData[i]->totalVelocityPtrStart = &particles->totalVelocity[startIdx];
        threadData[i]->screenXPtrStart = &particles->screenX[startIdx];
        threadData[i]->screenYPtrStart = &particles->screenY[startIdx];
        threadData[i]->length = chunkSize;
        threadData[i]->maxVelocity = 0.0f;
        
        startIdx += chunkSize;
    }
}

void drawBoundingBox(struct Screen *screen, float bBoxMax[3], float bBoxMin[3], struct Camera *camera) {
    // Use same projection method as particles
    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;
    
    // Calculate view matrix vectors - same as in projectParticles
    float right[3], trueUp[3];
    float up[3] = {0, 1, 0};
    
    // Calculate right vector using cross product
    right[0] = camera->ray.direction[1] * up[2] - camera->ray.direction[2] * up[1];
    right[1] = camera->ray.direction[2] * up[0] - camera->ray.direction[0] * up[2];
    right[2] = camera->ray.direction[0] * up[1] - camera->ray.direction[1] * up[0];
    
    // Calculate true up vector
    trueUp[0] = right[1] * camera->ray.direction[2] - right[2] * camera->ray.direction[1];
    trueUp[1] = right[2] * camera->ray.direction[0] - right[0] * camera->ray.direction[2];
    trueUp[2] = right[0] * camera->ray.direction[1] - right[1] * camera->ray.direction[0];
    
    // Define the 8 corners of the bounding box
    float corners[8][3] = {
        {bBoxMin[0], bBoxMin[1], bBoxMin[2]}, // 0: min, min, min
        {bBoxMax[0], bBoxMin[1], bBoxMin[2]}, // 1: max, min, min
        {bBoxMin[0], bBoxMax[1], bBoxMin[2]}, // 2: min, max, min
        {bBoxMax[0], bBoxMax[1], bBoxMin[2]}, // 3: max, max, min
        {bBoxMin[0], bBoxMin[1], bBoxMax[2]}, // 4: min, min, max
        {bBoxMax[0], bBoxMin[1], bBoxMax[2]}, // 5: max, min, max
        {bBoxMin[0], bBoxMax[1], bBoxMax[2]}, // 6: min, max, max
        {bBoxMax[0], bBoxMax[1], bBoxMax[2]}  // 7: max, max, max
    };
    
    // Project corners to screen space
    int screenPoints[8][2];
    bool inFront[8];
    
    for (int i = 0; i < 8; i++) {
        // Vector from camera to corner
        float x = corners[i][0] - camera->ray.origin[0];
        float y = corners[i][1] - camera->ray.origin[1];
        float z = corners[i][2] - camera->ray.origin[2];
        
        // Check if corner is in front of camera
        float dotProduct = x * camera->ray.direction[0] + 
                           y * camera->ray.direction[1] + 
                           z * camera->ray.direction[2];
        
        inFront[i] = (dotProduct > 0);
        
        if (inFront[i]) {
            // Project point to screen space
            float fovScale = 1.0f / (dotProduct * camera->fov);
            float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
            float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;
            
            screenPoints[i][0] = (int)(screenRight * halfWidth + halfWidth);
            screenPoints[i][1] = (int)(-screenUp * halfHeight + halfHeight);
            
            // Clamp to screen bounds
            screenPoints[i][0] = screenPoints[i][0] < 0 ? 0 : (screenPoints[i][0] >= ScreenWidth ? ScreenWidth - 1 : screenPoints[i][0]);
            screenPoints[i][1] = screenPoints[i][1] < 0 ? 0 : (screenPoints[i][1] >= ScreenHeight ? ScreenHeight - 1 : screenPoints[i][1]);
        }
    }
    
    // Define edges by vertex indices
    int edges[12][2] = {
        {0, 1}, {1, 3}, {3, 2}, {2, 0}, // Bottom face
        {4, 5}, {5, 7}, {7, 6}, {6, 4}, // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Connecting edges
    };
    
    // Draw edges
    for (int i = 0; i < 12; i++) {
        int v1 = edges[i][0];
        int v2 = edges[i][1];
        
        // Only draw edge if both vertices are in front of camera
        if (inFront[v1] && inFront[v2]) {
            int x0 = screenPoints[v1][0];
            int y0 = screenPoints[v1][1];
            int x1 = screenPoints[v2][0];
            int y1 = screenPoints[v2][1];
            
            int dx = abs(x1 - x0);
            int dy = -abs(y1 - y0);
            int sx = x0 < x1 ? 1 : -1;
            int sy = y0 < y1 ? 1 : -1;
            int err = dx + dy;
            int e2;
            
            while (1) {

                // struct Screen {
                //     uint8_t distance[ScreenWidth][ScreenHeight];
                //     uint8_t velocity[ScreenWidth][ScreenHeight];
                //     uint8_t normalizedOpacity[ScreenWidth][ScreenHeight];
                //     uint16_t opacity[ScreenWidth][ScreenHeight];
                // };
                // Draw pixel if it's within screen bounds
                if (x0 >= 0 && x0 < ScreenWidth && y0 >= 0 && y0 < ScreenHeight) {
                    // screen->distance[x0][y0] = 255; // Distance
                    screen->velocity[x0][y0] = 255; // Velocity
                    screen->normalizedOpacity[x0][y0] = 255; // Normalized Opacity
                }
                
                if (x0 == x1 && y0 == y1) break;
                e2 = 2 * err;
                if (e2 >= dy) {
                    if (x0 == x1) break;
                    err += dy;
                    x0 += sx;
                }
                if (e2 <= dx) {
                    if (y0 == y1) break;
                    err += dx;
                    y0 += sy;
                }
            }
        }
    }
}

void drawCursor(struct Screen *screen, struct Cursor *cursor, struct Camera *camera) {
    // Use same projection method as particles
    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;
    
    // Calculate view matrix vectors - same as in projectParticles
    float right[3], trueUp[3];
    float up[3] = {0, 1, 0};
    
    // Calculate right vector using cross product
    right[0] = camera->ray.direction[1] * up[2] - camera->ray.direction[2] * up[1];
    right[1] = camera->ray.direction[2] * up[0] - camera->ray.direction[0] * up[2];
    right[2] = camera->ray.direction[0] * up[1] - camera->ray.direction[1] * up[0];
    
    // Calculate true up vector
    trueUp[0] = right[1] * camera->ray.direction[2] - right[2] * camera->ray.direction[1];
    trueUp[1] = right[2] * camera->ray.direction[0] - right[0] * camera->ray.direction[2];
    trueUp[2] = right[0] * camera->ray.direction[1] - right[1] * camera->ray.direction[0];

    // Calculate vector from camera to cursor
    float x = cursor->x - camera->ray.origin[0];
    float y = cursor->y - camera->ray.origin[1];
    float z = cursor->z - camera->ray.origin[2];
    
    // Calculate dot product to check if cursor is in front of camera
    float dotProduct = x * camera->ray.direction[0] + 
                     y * camera->ray.direction[1] + 
                     z * camera->ray.direction[2];
    
    // Only draw cursor if it's in front of the camera
    if (dotProduct > 0) {
        float fovScale = 1.0f / (dotProduct * camera->fov);
        
        // Calculate screen position
        float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
        float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;
        
        int screenX = (int)(screenRight * halfWidth + halfWidth);
        int screenY = (int)(-screenUp * halfHeight + halfHeight);
        
        // Clamp to screen bounds
        screenX = (screenX < 0) ? 0 : (screenX >= ScreenWidth ? ScreenWidth - 1 : screenX);
        screenY = (screenY < 0) ? 0 : (screenY >= ScreenHeight ? ScreenHeight - 1 : screenY);
        
        // Draw horizontal line of cross
        const int cursorRadius = 5;
        for (int px = screenX - cursorRadius; px <= screenX + cursorRadius; px++) {
            if (px >= 0 && px < ScreenWidth) {
                screen->distance[px][screenY] = 255;
                screen->velocity[px][screenY] = 255;
                screen->normalizedOpacity[px][screenY] = 255;
            }
        }
        // Draw vertical line of cross (fixed: use screenX for column)
        for (int py = screenY - cursorRadius; py <= screenY + cursorRadius; py++) {
            if (py >= 0 && py < ScreenHeight) {
                // screen->distance[screenX][py] = 255;
                screen->velocity[screenX][py] = 255;
                screen->normalizedOpacity[screenX][py] = 255;
            }
        }
    }
}

void readCursorData(struct Cursor *cursor) {
    FILE *file = fopen("cursor.bin", "rb");
    if (!file) {
        printf("Cursor file not found, using default cursor\n");
        return;
    }
    
    if (fread(&cursor->x, sizeof(float), 1, file) != 1 ||
        fread(&cursor->y, sizeof(float), 1, file) != 1 ||
        fread(&cursor->z, sizeof(float), 1, file) != 1 ||
        fread(&cursor->active, sizeof(bool), 1, file) != 1|| 
        fread(&cursor->force, sizeof(float), 1, file) != 1) {
        
        fclose(file);
        return; 
    }
    
    fclose(file);
}


// void CollideParticlesInGrid(struct PointSOA *particles) {
//     for (int gridId = 0; gridId < gridResolution; gridId++) {
//         int startIdx = particles->startIndex[gridId];
//         if (startIdx == -1) continue; // No particles in this grid cell
//         int endIdx = startIdx + particles->numberOfParticle[gridId];
        
//         // Check collisions between all particles in this cell
//         for (int i = startIdx; i < endIdx; i++) {
//             for (int j = i + 1; j < endIdx; j++) {
//                 // Calculate distance between particles
//                 float dx = particles->x[j] - particles->x[i];
//                 float dy = particles->y[j] - particles->y[i];
//                 float dz = particles->z[j] - particles->z[i];
                
//                 float distSquared = dx*dx + dy*dy + dz*dz;
//                 float minDist = 3.0f * PARTICLE_RADIUS;
                
//                 // If particles are colliding (distance < 2*radius)
//                 if (distSquared < minDist * minDist) {
//                     // Calculate collision normal
//                     float dist = sqrtf(distSquared);
//                     float nx = dx / dist;
//                     float ny = dy / dist;
//                     float nz = dz / dist;
                    
//                     // Calculate relative velocity along normal
//                     float vx = particles->xVelocity[j] - particles->xVelocity[i];
//                     float vy = particles->yVelocity[j] - particles->yVelocity[i];
//                     float vz = particles->zVelocity[j] - particles->zVelocity[i];
                    
//                     float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                    
//                     // Only collide if particles are moving toward each other
//                     if (velocityAlongNormal < 0) {
//                         // Calculate impulse scalar (assuming equal mass)
//                         float impulse = -2.0f * velocityAlongNormal;
                        
//                         // Apply impulse
//                         particles->xVelocity[i] -= impulse * nx * 0.5f;
//                         particles->yVelocity[i] -= impulse * ny * 0.5f;
//                         particles->zVelocity[i] -= impulse * nz * 0.5f;
                        
//                         particles->xVelocity[j] += impulse * nx * 0.5f;
//                         particles->yVelocity[j] += impulse * ny * 0.5f;
//                         particles->zVelocity[j] += impulse * nz * 0.5f;
                        
//                         // Add some damping to collision
//                         particles->xVelocity[i] *= DAMPING;
//                         particles->yVelocity[i] *= DAMPING;
//                         particles->zVelocity[i] *= DAMPING;
                        
//                         particles->xVelocity[j] *= DAMPING;
//                         particles->yVelocity[j] *= DAMPING;
//                         particles->zVelocity[j] *= DAMPING;
                        
//                         // Push particles apart to avoid sticking
//                         float correction = (minDist - dist) * 0.01f;
//                         particles->x[i] -= nx * correction;
//                         particles->y[i] -= ny * correction;
//                         particles->z[i] -= nz * correction;
                        
//                         particles->x[j] += nx * correction;
//                         particles->y[j] += ny * correction;
//                         particles->z[j] += nz * correction;
//                     }
//                 }
//             }
//         }
//     }
// }

void CollideParticlesInGrid(struct PointSOA *particles) {
    // Constants as AVX vectors
    const __m256 particleRadiusVec = _mm256_set1_ps(PARTICLE_RADIUS);
    const __m256 minDistVec = _mm256_set1_ps(3.0f * PARTICLE_RADIUS);
    const __m256 minDistSquaredVec = _mm256_mul_ps(minDistVec, minDistVec);
    const __m256 halfVec = _mm256_set1_ps(0.5f);
    const __m256 dampingVec = _mm256_set1_ps(DAMPING);
    const __m256 correctionFactorVec = _mm256_set1_ps(0.01f);
    const __m256 zeroVec = _mm256_setzero_ps();
    
    for (int gridId = 0; gridId < gridResolution; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        int endIdx = startIdx + particles->numberOfParticle[gridId];
        
        // Check collisions between all particles in this cell
        for (int i = startIdx; i < endIdx; i++) {
            // Load particle i data once
            __m256 xi = _mm256_set1_ps(particles->x[i]);
            __m256 yi = _mm256_set1_ps(particles->y[i]);
            __m256 zi = _mm256_set1_ps(particles->z[i]);
            __m256 vxi = _mm256_set1_ps(particles->xVelocity[i]);
            __m256 vyi = _mm256_set1_ps(particles->yVelocity[i]);
            __m256 vzi = _mm256_set1_ps(particles->zVelocity[i]);
            
            // Process 8 particles at a time where possible
            for (int j = i + 1; j < endIdx; j += 8) {
                int remaining = endIdx - j;
                if (remaining >= 8) {
                    // Load 8 particles at once
                    __m256 xj = _mm256_loadu_ps(&particles->x[j]);
                    __m256 yj = _mm256_loadu_ps(&particles->y[j]);
                    __m256 zj = _mm256_loadu_ps(&particles->z[j]);
                    
                    // Calculate distance vectors
                    __m256 dx = _mm256_sub_ps(xj, xi);
                    __m256 dy = _mm256_sub_ps(yj, yi);
                    __m256 dz = _mm256_sub_ps(zj, zi);
                    
                    // Calculate squared distance
                    __m256 distSquared = _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_mul_ps(dx, dx),
                            _mm256_mul_ps(dy, dy)
                        ),
                        _mm256_mul_ps(dz, dz)
                    );
                    
                    // Compare with minimum distance squared
                    __m256 collisionMask = _mm256_cmp_ps(distSquared, minDistSquaredVec, _CMP_LT_OQ);
                    
                    // Skip to next batch if no collisions
                    if (_mm256_testz_ps(collisionMask, collisionMask)) {
                        continue;
                    }
                    
                    // Get mask as integer for processing individual lanes
                    int mask = _mm256_movemask_ps(collisionMask);
                    
                    // Process collisions for each set bit in the mask
                    for (int k = 0; k < 8; k++) {
                        if (mask & (1 << k)) {
                            int jIdx = j + k;
                            
                            // Recalculate with scalar math for this specific pair
                            float dx_s = particles->x[jIdx] - particles->x[i];
                            float dy_s = particles->y[jIdx] - particles->y[i];
                            float dz_s = particles->z[jIdx] - particles->z[i];
                            
                            float distSquared_s = dx_s*dx_s + dy_s*dy_s + dz_s*dz_s;
                            float minDist = 3.0f * PARTICLE_RADIUS;
                            
                            // Double-check collision (should always be true due to mask)
                            if (distSquared_s < minDist * minDist) {
                                // Calculate collision normal
                                float dist = sqrtf(distSquared_s);
                                float nx = dx_s / dist;
                                float ny = dy_s / dist;
                                float nz = dz_s / dist;
                                
                                // Calculate relative velocity along normal
                                float vx = particles->xVelocity[jIdx] - particles->xVelocity[i];
                                float vy = particles->yVelocity[jIdx] - particles->yVelocity[i];
                                float vz = particles->zVelocity[jIdx] - particles->zVelocity[i];
                                
                                float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                                
                                // Only collide if particles are moving toward each other
                                if (velocityAlongNormal < 0) {
                                    // Calculate impulse scalar (assuming equal mass)
                                    float impulse = -2.0f * velocityAlongNormal;
                                    
                                    // Apply impulse
                                    particles->xVelocity[i] -= impulse * nx * 0.5f;
                                    particles->yVelocity[i] -= impulse * ny * 0.5f;
                                    particles->zVelocity[i] -= impulse * nz * 0.5f;
                                    
                                    particles->xVelocity[jIdx] += impulse * nx * 0.5f;
                                    particles->yVelocity[jIdx] += impulse * ny * 0.5f;
                                    particles->zVelocity[jIdx] += impulse * nz * 0.5f;
                                    
                                    // Add some damping to collision
                                    particles->xVelocity[i] *= DAMPING;
                                    particles->yVelocity[i] *= DAMPING;
                                    particles->zVelocity[i] *= DAMPING;
                                    
                                    particles->xVelocity[jIdx] *= DAMPING;
                                    particles->yVelocity[jIdx] *= DAMPING;
                                    particles->zVelocity[jIdx] *= DAMPING;
                                    
                                    // Push particles apart to avoid sticking
                                    float correction = (minDist - dist) * 0.01f;
                                    particles->x[i] -= nx * correction;
                                    particles->y[i] -= ny * correction;
                                    particles->z[i] -= nz * correction;
                                    
                                    particles->x[jIdx] += nx * correction;
                                    particles->y[jIdx] += ny * correction;
                                    particles->z[jIdx] += nz * correction;
                                    
                                    // Update cached values for particle i
                                    xi = _mm256_set1_ps(particles->x[i]);
                                    yi = _mm256_set1_ps(particles->y[i]);
                                    zi = _mm256_set1_ps(particles->z[i]);
                                    vxi = _mm256_set1_ps(particles->xVelocity[i]);
                                    vyi = _mm256_set1_ps(particles->yVelocity[i]);
                                    vzi = _mm256_set1_ps(particles->zVelocity[i]);
                                }
                            }
                        }
                    }
                } else {
                    // Handle remaining particles with scalar code
                    for (int jIdx = j; jIdx < endIdx; jIdx++) {
                        float dx = particles->x[jIdx] - particles->x[i];
                        float dy = particles->y[jIdx] - particles->y[i];
                        float dz = particles->z[jIdx] - particles->z[i];
                        
                        float distSquared = dx*dx + dy*dy + dz*dz;
                        float minDist = 3.0f * PARTICLE_RADIUS;
                        
                        if (distSquared < minDist * minDist) {
                            float dist = sqrtf(distSquared);
                            float nx = dx / dist;
                            float ny = dy / dist;
                            float nz = dz / dist;
                            
                            float vx = particles->xVelocity[jIdx] - particles->xVelocity[i];
                            float vy = particles->yVelocity[jIdx] - particles->yVelocity[i];
                            float vz = particles->zVelocity[jIdx] - particles->zVelocity[i];
                            
                            float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                            
                            if (velocityAlongNormal < 0) {
                                float impulse = -2.0f * velocityAlongNormal;
                                
                                particles->xVelocity[i] -= impulse * nx * 0.5f;
                                particles->yVelocity[i] -= impulse * ny * 0.5f;
                                particles->zVelocity[i] -= impulse * nz * 0.5f;
                                
                                particles->xVelocity[jIdx] += impulse * nx * 0.5f;
                                particles->yVelocity[jIdx] += impulse * ny * 0.5f;
                                particles->zVelocity[jIdx] += impulse * nz * 0.5f;
                                
                                particles->xVelocity[i] *= DAMPING;
                                particles->yVelocity[i] *= DAMPING;
                                particles->zVelocity[i] *= DAMPING;
                                
                                particles->xVelocity[jIdx] *= DAMPING;
                                particles->yVelocity[jIdx] *= DAMPING;
                                particles->zVelocity[jIdx] *= DAMPING;
                                
                                float correction = (minDist - dist) * 0.01f;
                                particles->x[i] -= nx * correction;
                                particles->y[i] -= ny * correction;
                                particles->z[i] -= nz * correction;
                                
                                particles->x[jIdx] += nx * correction;
                                particles->y[jIdx] += ny * correction;
                                particles->z[jIdx] += nz * correction;
                            }
                        }
                    }
                }
            }
        }
    }
}

void swapParticles(struct PointSOA *particles, int i, int j) {
    // Swap positions
    float tempX = particles->x[i];
    float tempY = particles->y[i];
    float tempZ = particles->z[i];
    particles->x[i] = particles->x[j];
    particles->y[i] = particles->y[j];
    particles->z[i] = particles->z[j];
    particles->x[j] = tempX;
    particles->y[j] = tempY;
    particles->z[j] = tempZ;
    
    // Swap velocities
    float tempVX = particles->xVelocity[i];
    float tempVY = particles->yVelocity[i];
    float tempVZ = particles->zVelocity[i];
    particles->xVelocity[i] = particles->xVelocity[j];
    particles->yVelocity[i] = particles->yVelocity[j];
    particles->zVelocity[i] = particles->zVelocity[j];
    particles->xVelocity[j] = tempVX;
    particles->yVelocity[j] = tempVY;
    particles->zVelocity[j] = tempVZ;
    
    // Swap grid IDs
    int tempID = particles->gridID[i];
    particles->gridID[i] = particles->gridID[j];
    particles->gridID[j] = tempID;
}

int positionToGridId(struct PointSOA *particles, float position[3]) {
    // Calculate grid ID based on position
    int xIndex = (int)((position[0] - particles->bBoxMin[0]) / (particles->bBoxMax[0] - particles->bBoxMin[0]) * gridResolutionAxis);
    int yIndex = (int)((position[1] - particles->bBoxMin[1]) / (particles->bBoxMax[1] - particles->bBoxMin[1]) * gridResolutionAxis);
    int zIndex = (int)((position[2] - particles->bBoxMin[2]) / (particles->bBoxMax[2] - particles->bBoxMin[2]) * gridResolutionAxis);
    
    // Clamp indices to grid resolution
    if (xIndex < 0) xIndex = 0;
    if (xIndex >= gridResolutionAxis) xIndex = gridResolutionAxis - 1;
    if (yIndex < 0) yIndex = 0;
    if (yIndex >= gridResolutionAxis) yIndex = gridResolutionAxis - 1;
    if (zIndex < 0) zIndex = 0;
    if (zIndex >= gridResolutionAxis) zIndex = gridResolutionAxis - 1;
    // Convert 3D indices to 1D grid ID
    return xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
}

void CalculateCenterOfCell(struct PointSOA *particles, int index, float centerOfCell[3]) {
    int xIndex = index % gridResolutionAxis;
    int yIndex = (index / gridResolutionAxis) % gridResolutionAxis;
    int zIndex = index / (gridResolutionAxis * gridResolutionAxis);
    
    float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
    float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
    float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
    centerOfCell[0] = particles->bBoxMin[0] + (xIndex + 0.5f) * xStep;
    centerOfCell[1] = particles->bBoxMin[1] + (yIndex + 0.5f) * yStep;
    centerOfCell[2] = particles->bBoxMin[2] + (zIndex + 0.5f) * zStep;
}

// void ApplyPressure(struct PointSOA *particles) {
//     // Process each cell
//     for (int gridId = 0; gridId < gridResolution; gridId++) {
//         int startIdx = particles->startIndex[gridId];
//         if (startIdx == -1) continue; // No particles in this grid cell
        
//         int endIdx = startIdx + particles->numberOfParticle[gridId];
//         float currentPressure = pressure * particles->numberOfParticle[gridId];
        
//         // Get 3D grid coordinates
//         int x = gridId % gridResolutionAxis;
//         int y = (gridId / gridResolutionAxis) % gridResolutionAxis;
//         int z = gridId / (gridResolutionAxis * gridResolutionAxis);
        
//         // Process each particle in this cell
//         for (int i = startIdx; i < endIdx; i++) {
//             float netForceX = 0.0f;
//             float netForceY = 0.0f;
//             float netForceZ = 0.0f;
            
//             // Check all 6 face-adjacent neighbors
//             const int neighbors[6][3] = {
//                 {-1, 0, 0}, {1, 0, 0},  // left, right
//                 {0, -1, 0}, {0, 1, 0},  // down, up
//                 {0, 0, -1}, {0, 0, 1}   // back, front
//             };
            
//             // Consider all neighbors for pressure gradient
//             for (int j = 0; j < 6; j++) {
//                 int nx = x + neighbors[j][0];
//                 int ny = y + neighbors[j][1];
//                 int nz = z + neighbors[j][2];
                
//                 // Skip out-of-bounds neighbors
//                 if (nx < 0 || nx >= gridResolutionAxis ||
//                     ny < 0 || ny >= gridResolutionAxis ||
//                     nz < 0 || nz >= gridResolutionAxis) {
//                     continue;
//                 }
                
//                 int neighborGridId = nx + ny * gridResolutionAxis + nz * gridResolutionAxis * gridResolutionAxis;
//                 // Calculate neighbor pressure
//                 float neighborPressure = pressure * particles->numberOfParticle[neighborGridId];
                
//                 // Calculate pressure difference
//                 float pressureDiff = currentPressure - neighborPressure;
                
//                 // Only push if there's pressure gradient
//                 if (pressureDiff > 0.0f) {
//                     // Add force component in direction of this neighbor
//                     netForceX += neighbors[j][0] * pressureDiff;
//                     netForceY += neighbors[j][1] * pressureDiff;
//                     netForceZ += neighbors[j][2] * pressureDiff;
//                 }
//             }
            
//             // Apply the net force if it's significant
//             float forceMagnitude = sqrtf(netForceX*netForceX + netForceY*netForceY + netForceZ*netForceZ);
//             if (forceMagnitude > 0.1f) {
//                 // Normalize force direction
//                 float invMag = 1.0f / forceMagnitude;
//                 netForceX *= invMag;
//                 netForceY *= invMag;
//                 netForceZ *= invMag;
                
//                 // Scale by current cell pressure
//                 float forceScale = currentPressure * 0.2f;
                
//                 // Apply to particle velocity
//                 particles->xVelocity[i] += netForceX * forceScale;
//                 particles->yVelocity[i] += netForceY * forceScale;
//                 particles->zVelocity[i] += netForceZ * forceScale;
//             }
//         }
//     }
// }

void ApplyPressure(struct PointSOA *particles) {
    // Constants as AVX vectors for vectorized operations
    const __m256 pressureVec = _mm256_set1_ps(pressure);
    const __m256 forceScaleFactorVec = _mm256_set1_ps(0.2f);
    const __m256 thresholdVec = _mm256_set1_ps(0.1f);
    const __m256 zeroVec = _mm256_setzero_ps();
    
    // Define neighbor directions once
    const int neighbors[6][3] = {
        {-1, 0, 0}, {1, 0, 0},  // left, right
        {0, -1, 0}, {0, 1, 0},  // down, up
        {0, 0, -1}, {0, 0, 1}   // back, front
    };
    
    // Process each cell
    for (int gridId = 0; gridId < gridResolution; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        
        int endIdx = startIdx + particles->numberOfParticle[gridId];
        float currentPressure = pressure * particles->numberOfParticle[gridId];
        __m256 currentPressureVec = _mm256_set1_ps(currentPressure);
        
        // Get 3D grid coordinates
        int x = gridId % gridResolutionAxis;
        int y = (gridId / gridResolutionAxis) % gridResolutionAxis;
        int z = gridId / (gridResolutionAxis * gridResolutionAxis);
        
        // Pre-compute and cache neighbor information
        float neighborPressures[6];
        float directionX[6], directionY[6], directionZ[6];
        int validNeighbors = 0;
        
        for (int j = 0; j < 6; j++) {
            int nx = x + neighbors[j][0];
            int ny = y + neighbors[j][1];
            int nz = z + neighbors[j][2];
            
            // Skip out-of-bounds neighbors
            if (nx < 0 || nx >= gridResolutionAxis ||
                ny < 0 || ny >= gridResolutionAxis ||
                nz < 0 || nz >= gridResolutionAxis) {
                continue;
            }
            
            int neighborGridId = nx + ny * gridResolutionAxis + nz * gridResolutionAxis * gridResolutionAxis;
            float neighborPressure = pressure * particles->numberOfParticle[neighborGridId];
            
            // Store neighbor info for reuse with all particles
            neighborPressures[validNeighbors] = neighborPressure;
            directionX[validNeighbors] = (float)neighbors[j][0];
            directionY[validNeighbors] = (float)neighbors[j][1];
            directionZ[validNeighbors] = (float)neighbors[j][2];
            validNeighbors++;
        }
        
        // Process particles in this cell in batches of 8
        int i;
        for (i = startIdx; i <= endIdx - 8; i += 8) {
            // Process 8 particles at once with AVX
            __m256 netForceXVec = _mm256_setzero_ps();
            __m256 netForceYVec = _mm256_setzero_ps();
            __m256 netForceZVec = _mm256_setzero_ps();
            
            // Consider all valid neighbors
            for (int j = 0; j < validNeighbors; j++) {
                // Calculate pressure difference
                __m256 neighborPressureVec = _mm256_set1_ps(neighborPressures[j]);
                __m256 pressureDiffVec = _mm256_sub_ps(currentPressureVec, neighborPressureVec);
                
                // Create mask for positive pressure differences
                __m256 positiveMask = _mm256_cmp_ps(pressureDiffVec, zeroVec, _CMP_GT_OQ);
                
                // Skip if no positive pressure differences
                if (_mm256_testz_ps(positiveMask, positiveMask)) {
                    continue;
                }
                
                // Apply direction * pressure difference with mask
                __m256 scaledDiffVec = _mm256_and_ps(pressureDiffVec, positiveMask);
                __m256 dirXVec = _mm256_set1_ps(directionX[j]);
                __m256 dirYVec = _mm256_set1_ps(directionY[j]);
                __m256 dirZVec = _mm256_set1_ps(directionZ[j]);
                
                netForceXVec = _mm256_add_ps(netForceXVec, _mm256_mul_ps(dirXVec, scaledDiffVec));
                netForceYVec = _mm256_add_ps(netForceYVec, _mm256_mul_ps(dirYVec, scaledDiffVec));
                netForceZVec = _mm256_add_ps(netForceZVec, _mm256_mul_ps(dirZVec, scaledDiffVec));
            }
            
            // Calculate force magnitude squared
            __m256 forceMagSquaredVec = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(netForceXVec, netForceXVec),
                    _mm256_mul_ps(netForceYVec, netForceYVec)
                ),
                _mm256_mul_ps(netForceZVec, netForceZVec)
            );
            
            // Calculate sqrt for magnitude
            __m256 forceMagVec = _mm256_sqrt_ps(forceMagSquaredVec);
            
            // Create mask for significant forces
            __m256 forceMask = _mm256_cmp_ps(forceMagVec, thresholdVec, _CMP_GT_OQ);
            
            // Skip if no particles have significant force
            if (_mm256_testz_ps(forceMask, forceMask)) {
                continue;
            }
            
            // Calculate safe inverse magnitude (avoid division by zero)
            __m256 safeForceVec = _mm256_max_ps(forceMagVec, _mm256_set1_ps(0.0001f));
            __m256 invMagVec = _mm256_div_ps(_mm256_set1_ps(1.0f), safeForceVec);
            
            // Normalize force vectors
            __m256 normForceXVec = _mm256_mul_ps(netForceXVec, invMagVec);
            __m256 normForceYVec = _mm256_mul_ps(netForceYVec, invMagVec);
            __m256 normForceZVec = _mm256_mul_ps(netForceZVec, invMagVec);
            
            // Scale by current pressure
            __m256 forceScaleVec = _mm256_mul_ps(currentPressureVec, forceScaleFactorVec);
            __m256 finalForceXVec = _mm256_mul_ps(normForceXVec, forceScaleVec);
            __m256 finalForceYVec = _mm256_mul_ps(normForceYVec, forceScaleVec);
            __m256 finalForceZVec = _mm256_mul_ps(normForceZVec, forceScaleVec);
            
            // Apply force mask to only affect particles with significant force
            finalForceXVec = _mm256_and_ps(finalForceXVec, forceMask);
            finalForceYVec = _mm256_and_ps(finalForceYVec, forceMask);
            finalForceZVec = _mm256_and_ps(finalForceZVec, forceMask);
            
            // Load current velocities
            __m256 xVelVec = _mm256_loadu_ps(&particles->xVelocity[i]);
            __m256 yVelVec = _mm256_loadu_ps(&particles->yVelocity[i]);
            __m256 zVelVec = _mm256_loadu_ps(&particles->zVelocity[i]);
            
            // Update velocities and store
            _mm256_storeu_ps(&particles->xVelocity[i], _mm256_add_ps(xVelVec, finalForceXVec));
            _mm256_storeu_ps(&particles->yVelocity[i], _mm256_add_ps(yVelVec, finalForceYVec));
            _mm256_storeu_ps(&particles->zVelocity[i], _mm256_add_ps(zVelVec, finalForceZVec));
        }
        
        // Handle remaining particles with scalar code
        for (; i < endIdx; i++) {
            float netForceX = 0.0f;
            float netForceY = 0.0f;
            float netForceZ = 0.0f;
            
            // Consider all neighbors for pressure gradient
            for (int j = 0; j < validNeighbors; j++) {
                float pressureDiff = currentPressure - neighborPressures[j];
                
                // Only push if there's pressure gradient
                if (pressureDiff > 0.0f) {
                    netForceX += directionX[j] * pressureDiff;
                    netForceY += directionY[j] * pressureDiff;
                    netForceZ += directionZ[j] * pressureDiff;
                }
            }
            
            // Apply the net force if it's significant
            float forceMagnitude = sqrtf(netForceX*netForceX + netForceY*netForceY + netForceZ*netForceZ);
            if (forceMagnitude > 0.1f) {
                // Normalize force direction
                float invMag = 1.0f / forceMagnitude;
                netForceX *= invMag;
                netForceY *= invMag;
                netForceZ *= invMag;
                
                // Scale by current cell pressure
                float forceScale = currentPressure * 0.2f;
                
                // Apply to particle velocity
                particles->xVelocity[i] += netForceX * forceScale;
                particles->yVelocity[i] += netForceY * forceScale;
                particles->zVelocity[i] += netForceZ * forceScale;
            }
        }
    }
}

// First, optimize the move_to_grid function for direct calculation
// void move_to_grid(struct PointSOA *particles, int index) {
//     float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
//     float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
//     float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
//     // Calculate grid positions directly with clamping
//     int xIndex = (int)((particles->x[index] - particles->bBoxMin[0]) / xStep);
//     int yIndex = (int)((particles->y[index] - particles->bBoxMin[1]) / yStep);
//     int zIndex = (int)((particles->z[index] - particles->bBoxMin[2]) / zStep);
    
//     // Clamp indices
//     xIndex = (xIndex < 0) ? 0 : ((xIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : xIndex);
//     yIndex = (yIndex < 0) ? 0 : ((yIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : yIndex);
//     zIndex = (zIndex < 0) ? 0 : ((zIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : zIndex);
    
//     // Convert 3D coordinates to a single index
//     particles->gridID[index] = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
// }

// Use radix sort for better performance with integer keys
void radixSortParticles(struct PointSOA *particles, int n) {
    // Find the maximum number to know the number of digits
    int maxVal = 0;
    for (int i = 0; i < n; i++) {
        if (particles->gridID[i] > maxVal) {
            maxVal = particles->gridID[i];
        }
    }
    
    // Temporary arrays for sorting
    float* tempX = (float*)malloc(n * sizeof(float));
    float* tempY = (float*)malloc(n * sizeof(float));
    float* tempZ = (float*)malloc(n * sizeof(float));
    float* tempVX = (float*)malloc(n * sizeof(float));
    float* tempVY = (float*)malloc(n * sizeof(float));
    float* tempVZ = (float*)malloc(n * sizeof(float));
    int* tempGridID = (int*)malloc(n * sizeof(int));
    
    // Do counting sort for every digit
    for (int exp = 1; maxVal / exp > 0; exp *= 10) {
        int count[10] = {0};
        
        // Count occurrences of each digit at current place value
        for (int i = 0; i < n; i++) {
            count[(particles->gridID[i] / exp) % 10]++;
        }
        
        // Compute cumulative count (positions)
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        
        // Build the output array in reverse order to maintain stability
        for (int i = n - 1; i >= 0; i--) {
            int digit = (particles->gridID[i] / exp) % 10;
            int pos = --count[digit];
            
            // Copy the data to temporary arrays
            tempX[pos] = particles->x[i];
            tempY[pos] = particles->y[i];
            tempZ[pos] = particles->z[i];
            tempVX[pos] = particles->xVelocity[i];
            tempVY[pos] = particles->yVelocity[i];
            tempVZ[pos] = particles->zVelocity[i];
            tempGridID[pos] = particles->gridID[i];
        }
        
        // Copy back to original arrays
        for (int i = 0; i < n; i++) {
            particles->x[i] = tempX[i];
            particles->y[i] = tempY[i];
            particles->z[i] = tempZ[i];
            particles->xVelocity[i] = tempVX[i];
            particles->yVelocity[i] = tempVY[i];
            particles->zVelocity[i] = tempVZ[i];
            particles->gridID[i] = tempGridID[i];
        }
    }
    
    // Free temporary arrays
    free(tempX);
    free(tempY);
    free(tempZ);
    free(tempVX);
    free(tempVY);
    free(tempVZ);
    free(tempGridID);
}

// Earlier in the file, add this AVX-optimized version:
void updateGridData(struct PointSOA *particles) {
    // Calculate grid steps once
    float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
    float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
    float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
    // Prepare constant AVX registers
    __m256 xStepVec = _mm256_set1_ps(1.0f / xStep);
    __m256 yStepVec = _mm256_set1_ps(1.0f / yStep);
    __m256 zStepVec = _mm256_set1_ps(1.0f / zStep);
    __m256 xMinVec = _mm256_set1_ps(particles->bBoxMin[0]);
    __m256 yMinVec = _mm256_set1_ps(particles->bBoxMin[1]);
    __m256 zMinVec = _mm256_set1_ps(particles->bBoxMin[2]);
    __m256 zeroVec = _mm256_setzero_ps();
    __m256 gridMaxVec = _mm256_set1_ps((float)(gridResolutionAxis - 1));
    __m256i gridResVec = _mm256_set1_epi32(gridResolutionAxis);
    __m256i gridResSquaredVec = _mm256_set1_epi32(gridResolutionAxis * gridResolutionAxis);
    
    // Process particles in chunks of 8
    int alignedCount = NUM_PARTICLES & ~7; // Round down to multiple of 8
    
    for (int i = 0; i < alignedCount; i += 8) {
        // Load particle positions
        __m256 xVec = _mm256_loadu_ps(&particles->x[i]);
        __m256 yVec = _mm256_loadu_ps(&particles->y[i]);
        __m256 zVec = _mm256_loadu_ps(&particles->z[i]);
        
        // Calculate grid indices
        __m256 xIndexFloat = _mm256_mul_ps(_mm256_sub_ps(xVec, xMinVec), xStepVec);
        __m256 yIndexFloat = _mm256_mul_ps(_mm256_sub_ps(yVec, yMinVec), yStepVec);
        __m256 zIndexFloat = _mm256_mul_ps(_mm256_sub_ps(zVec, zMinVec), zStepVec);
        
        // Convert float indices to integers
        __m256i xIndexVec = _mm256_cvttps_epi32(xIndexFloat);
        __m256i yIndexVec = _mm256_cvttps_epi32(yIndexFloat);
        __m256i zIndexVec = _mm256_cvttps_epi32(zIndexFloat);
        
        // Apply clamp to x indices (min = 0, max = gridResolutionAxis - 1)
        __m256i xClamped = _mm256_max_epi32(_mm256_min_epi32(xIndexVec, _mm256_cvttps_epi32(gridMaxVec)), 
                                           _mm256_setzero_si256());
        
        // Apply clamp to y indices
        __m256i yClamped = _mm256_max_epi32(_mm256_min_epi32(yIndexVec, _mm256_cvttps_epi32(gridMaxVec)), 
                                           _mm256_setzero_si256());
        
        // Apply clamp to z indices
        __m256i zClamped = _mm256_max_epi32(_mm256_min_epi32(zIndexVec, _mm256_cvttps_epi32(gridMaxVec)), 
                                           _mm256_setzero_si256());
        
        // Calculate grid IDs: xIndex + yIndex * gridRes + zIndex * gridRes * gridRes
        __m256i yContrib = _mm256_mullo_epi32(yClamped, gridResVec);
        __m256i zContrib = _mm256_mullo_epi32(zClamped, gridResSquaredVec);
        __m256i gridIDs = _mm256_add_epi32(_mm256_add_epi32(xClamped, yContrib), zContrib);
        
        // Store grid IDs
        int results[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)results, gridIDs);
        
        // Copy results to particles array
        for (int j = 0; j < 8; j++) {
            particles->gridID[i + j] = results[j];
        }
    }
    
    // Handle remaining particles
    for (int i = alignedCount; i < NUM_PARTICLES; i++) {
        // Calculate grid positions directly with clamping
        int xIndex = (int)((particles->x[i] - particles->bBoxMin[0]) / xStep);
        int yIndex = (int)((particles->y[i] - particles->bBoxMin[1]) / yStep);
        int zIndex = (int)((particles->z[i] - particles->bBoxMin[2]) / zStep);
        
        // Clamp indices
        xIndex = (xIndex < 0) ? 0 : ((xIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : xIndex);
        yIndex = (yIndex < 0) ? 0 : ((yIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : yIndex);
        zIndex = (zIndex < 0) ? 0 : ((zIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : zIndex);
        
        // Convert 3D coordinates to a single index
        particles->gridID[i] = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
    }
    
    // Reset grid cell start indices
    memset(particles->startIndex, -1, sizeof(int) * gridResolution);
    memset(particles->numberOfParticle, 0, sizeof(int) * gridResolution);
    
    // Sort particles by grid ID using radix sort (much faster than insertion sort)
    radixSortParticles(particles, NUM_PARTICLES);
    
    // Update start indices and count particles in a single pass
    int currentGridID = -1;
    
    for (int i = 0; i < NUM_PARTICLES; i++) {
        int gridID = particles->gridID[i];
        
        // When we encounter a new grid ID, mark its start index
        if (gridID != currentGridID) {
            particles->startIndex[gridID] = i;
            currentGridID = gridID;
        }
        
        // Increment count for this grid cell
        particles->numberOfParticle[gridID]++;
    }
}

void update_particle_apply_gravity(struct PointSOA *particles, float dt) {
    // Prepare constants as AVX vectors
    __m256 dtVec = _mm256_set1_ps(dt);
    __m256 dampingVec = _mm256_set1_ps(DAMPING);
    __m256 gravityVec = _mm256_set1_ps(GRAVITY);
    
    // Process particles in chunks of 8
    int alignedCount = NUM_PARTICLES & ~7; // Round down to multiple of 8
    
    for (int i = 0; i < alignedCount; i += 8) {
        // Load current positions
        __m256 xVec = _mm256_loadu_ps(&particles->x[i]);
        __m256 yVec = _mm256_loadu_ps(&particles->y[i]);
        __m256 zVec = _mm256_loadu_ps(&particles->z[i]);
        
        // Load velocities
        __m256 xVelocityVec = _mm256_loadu_ps(&particles->xVelocity[i]);
        __m256 yVelocityVec = _mm256_loadu_ps(&particles->yVelocity[i]);
        __m256 zVelocityVec = _mm256_loadu_ps(&particles->zVelocity[i]);
        
        // Apply gravity to velocity
        yVelocityVec = _mm256_sub_ps(yVelocityVec, gravityVec);
        
        // Calculate position updates: velocity * dt * damping
        __m256 dtDampingVec = _mm256_mul_ps(dtVec, dampingVec);
        __m256 xOffsetVec = _mm256_mul_ps(_mm256_mul_ps(xVelocityVec, dtVec), dampingVec);
        __m256 yOffsetVec = _mm256_mul_ps(_mm256_mul_ps(yVelocityVec, dtVec), dampingVec);
        __m256 zOffsetVec = _mm256_mul_ps(_mm256_mul_ps(zVelocityVec, dtVec), dampingVec);
        
        // Update positions
        xVec = _mm256_add_ps(xVec, xOffsetVec);
        yVec = _mm256_add_ps(yVec, yOffsetVec);
        zVec = _mm256_add_ps(zVec, zOffsetVec);
        
        // Store updated positions
        _mm256_storeu_ps(&particles->x[i], xVec);
        _mm256_storeu_ps(&particles->y[i], yVec);
        _mm256_storeu_ps(&particles->z[i], zVec);
        
        // Store updated velocities
        _mm256_storeu_ps(&particles->xVelocity[i], xVelocityVec);
        _mm256_storeu_ps(&particles->yVelocity[i], yVelocityVec);
        _mm256_storeu_ps(&particles->zVelocity[i], zVelocityVec);
    }
    
    // Handle remaining particles
    for (int i = alignedCount; i < NUM_PARTICLES; i++) {
        // Apply gravity
        particles->yVelocity[i] -= GRAVITY;
        
        // Update positions
        particles->x[i] += particles->xVelocity[i] * dt * DAMPING;
        particles->y[i] += particles->yVelocity[i] * dt * DAMPING;
        particles->z[i] += particles->zVelocity[i] * dt * DAMPING;
    }
}

void move_to_box(struct PointSOA *particles, float bBoxMin[3], float bBoxMax[3]) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        if (particles->x[i] < bBoxMin[0]) {
            particles->x[i] = bBoxMin[0];
            // change velocity to opposite direction
            particles->xVelocity[i] = -particles->xVelocity[i] * DAMPING;
        } else if (particles->x[i] > bBoxMax[0]) {
            particles->x[i] = bBoxMax[0];
            // change velocity to opposite direction
            particles->xVelocity[i] = -particles->xVelocity[i] * DAMPING;
        }
        if (particles->y[i] < bBoxMin[1]) {
            particles->y[i] = bBoxMin[1];
            // change velocity to opposite direction
            particles->yVelocity[i] = -particles->yVelocity[i] * DAMPING;
        } else if (particles->y[i] > bBoxMax[1]) {
            particles->y[i] = bBoxMax[1];
            // change velocity to opposite direction
            particles->yVelocity[i] = -particles->yVelocity[i] * DAMPING;
        }
        if (particles->z[i] < bBoxMin[2]) {
            particles->z[i] = bBoxMin[2];
            // change velocity to opposite direction
            particles->zVelocity[i] = -particles->zVelocity[i] * DAMPING;
        } else if (particles->z[i] > bBoxMax[2]) {
            particles->z[i] = bBoxMax[2];
            // change velocity to opposite direction
            particles->zVelocity[i] = -particles->zVelocity[i] * DAMPING;
        }
    }
}

int readCameraData(struct Camera *camera) {
    // printf("Attempting to read camera data\n");
    FILE *file = fopen("camera.bin", "rb");
    if (!file) {
        // printf("Camera file not found, using default camera\n");
        return 0;
    }
    
    // Read camera position
    if (fread(&camera->ray.origin[0], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.origin[1], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.origin[2], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.direction[0], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.direction[1], sizeof(float), 1, file) != 1 ||
        fread(&camera->ray.direction[2], sizeof(float), 1, file) != 1 ||
        fread(&camera->fov, sizeof(float), 1, file) != 1) {
        
        fclose(file);
        // printf("Error reading camera data, using default camera\n");
        return 0;  // Error reading, use default camera
    }
    
    fclose(file);
    // printf("Successfully read camera data\n");
    return 1;  // Successfully read camera data
}

// Add this quick sort helper function
void quickSortParticles(struct PointSOA *particles, int low, int high, float *distances) {
    if (low < high) {
        float pivot = distances[high];
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (distances[j] <= pivot) {
                i++;
                swapParticles(particles, i, j);
                float temp = distances[i];
                distances[i] = distances[j];
                distances[j] = temp;
            }
        }
        swapParticles(particles, i + 1, high);
        float temp = distances[i + 1];
        distances[i + 1] = distances[high];
        distances[high] = temp;
        
        int partition = i + 1;
        quickSortParticles(particles, low, partition - 1, distances);
        quickSortParticles(particles, partition + 1, high, distances);
    }
}


void calculateParticleScreenCoordinates(struct ThreadData *data) {
    struct Camera *camera = data->camera;
    float *xPtr = data->xPtrStart;
    float *yPtr = data->yPtrStart;
    float *zPtr = data->zPtrStart;
    float *xVelocityPtr = data->xVelocityPtrStart;
    float *yVelocityPtr = data->yVelocityPtrStart;
    float *zVelocityPtr = data->zVelocityPtrStart;
    float *totalVelocityPtr = data->totalVelocityPtrStart;
    int *screenXPtr = (int *)data->screenXPtrStart;
    int *screenYPtr = (int *)data->screenYPtrStart;
    int length = data->length;
    float maxVelocity = data->maxVelocity;

    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;

    // Calculate view matrix vectors
    float right[3], trueUp[3];
    float up[3] = {0, 1, 0};
    // Calculate right vector using cross product
    right[0] = camera->ray.direction[1] * up[2] - camera->ray.direction[2] * up[1];
    right[1] = camera->ray.direction[2] * up[0] - camera->ray.direction[0] * up[2];
    right[2] = camera->ray.direction[0] * up[1] - camera->ray.direction[1] * up[0];
    // Calculate true up vector
    trueUp[0] = right[1] * camera->ray.direction[2] - right[2] * camera->ray.direction[1];
    trueUp[1] = right[2] * camera->ray.direction[0] - right[0] * camera->ray.direction[2];
    trueUp[2] = right[0] * camera->ray.direction[1] - right[1] * camera->ray.direction[0];
    // Cache camera origin for faster access
    const float camX = camera->ray.origin[0];
    const float camY = camera->ray.origin[1];
    const float camZ = camera->ray.origin[2];
    const float camDirX = camera->ray.direction[0];
    const float camDirY = camera->ray.direction[1];
    const float camDirZ = camera->ray.direction[2];

    // Calculate the screen coordinates
    for (int i = 0; i < length; i++) {
        float x = xPtr[i] - camX;
        float y = yPtr[i] - camY;
        float z = zPtr[i] - camZ;

        float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
        float fovScale = 1.0f / (dotProduct * camera->fov);

        float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
        float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;

        int screenX = (int)(screenRight * halfWidth + halfWidth);
        int screenY = (int)(-screenUp * halfHeight + halfHeight);

        if (screenX < 0 || screenX >= ScreenWidth || screenY < 0 || screenY >= ScreenHeight) continue;
        
        float totalVelocity = xVelocityPtr[i] * xVelocityPtr[i] + yVelocityPtr[i] * yVelocityPtr[i] + zVelocityPtr[i] * zVelocityPtr[i];

        screenXPtr[i] = screenX;
        screenYPtr[i] = screenY;
        totalVelocityPtr[i] = totalVelocity;

        if (totalVelocity > data->maxVelocity) {
            data->maxVelocity = totalVelocity;    // Update the thread's maxVelocity
        }
    }
   
}

void projectParticles(struct PointSOA *particles, struct Camera *camera, struct Screen *screen, struct TimePartition *timePartition, struct ThreadsData *threadsData) {
    // Pre-calculate camera vectors and constants
    const float halfWidth = ScreenWidth * 0.5f;
    const float halfHeight = ScreenHeight * 0.5f;
    
    // Calculate view matrix vectors
    float right[3], trueUp[3];
    float up[3] = {0, 1, 0};
    
    // Calculate right vector using cross product
    right[0] = camera->ray.direction[1] * up[2] - camera->ray.direction[2] * up[1];
    right[1] = camera->ray.direction[2] * up[0] - camera->ray.direction[0] * up[2];
    right[2] = camera->ray.direction[0] * up[1] - camera->ray.direction[1] * up[0];
    
    // Calculate true up vector
    trueUp[0] = right[1] * camera->ray.direction[2] - right[2] * camera->ray.direction[1];
    trueUp[1] = right[2] * camera->ray.direction[0] - right[0] * camera->ray.direction[2];
    trueUp[2] = right[0] * camera->ray.direction[1] - right[1] * camera->ray.direction[0];

    // Cache camera origin for faster access
    const float camX = camera->ray.origin[0];
    const float camY = camera->ray.origin[1];
    const float camZ = camera->ray.origin[2];
    const float camDirX = camera->ray.direction[0];
    const float camDirY = camera->ray.direction[1];
    const float camDirZ = camera->ray.direction[2];
    
    // Calculate distances once
    int validParticles = 0;
    float maxDistance = 0.0f;
    
    clock_t start = clock();
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x = particles->x[i] - camX;
        float y = particles->y[i] - camY;
        float z = particles->z[i] - camZ;
        
        float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
                         
        if (dotProduct > 0) { // Only include particles in front of camera
            float distSquared = x*x + y*y + z*z;
            particles->distance[validParticles] = distSquared;
            
            // Track maximum distance directly during first pass
            if (distSquared > maxDistance) {
                maxDistance = distSquared;
            }
            
            if (validParticles != i) {
                swapParticles(particles, validParticles, i);
            }
            validParticles++;
        }
    }
    
    // Skip sort if no particles are visible
    if (validParticles == 0) return;
    
    // Sort particles using quick sort
    quickSortParticles(particles, 0, validParticles - 1, particles->distance);
    clock_t endSortTime = clock();
    timePartition->sortTime = (((endSortTime - start) / CLOCKS_PER_SEC) * 0.25f + timePartition->sortTime * 0.75f);
   
    // Add maxVelocity declaration here
    // float maxVelocity = 0.0f;
    
    // clock_t startProjectTimeMP = clock();
    // asignDataToThreads(particles, camera, validParticles);
    // // create threads
    // for (int i = 0; i < NUM_THREADS; i++) {
    //     // create thread and store the handle in the ThreadData struct
    //     pthread_create(&threadsData->threadData[i]->thread, NULL, (void *(*)(void *))calculateParticleScreenCoordinates, threadsData->threadData[i]);
    // }
    // // wait for threads to finish
    // for (int i = 0; i < NUM_THREADS; i++) {
    //     pthread_join(threadsData->threadData[i]->thread, NULL);
    // }
    // // calculate the maximum velocity from all threads
    // for (int i = 0; i < NUM_THREADS; i++) {
    //     if (threadsData->threadData[i]->maxVelocity > maxVelocity) {
    //         maxVelocity = threadsData->threadData[i]->maxVelocity;
    //     }
    // }
    // printf("Multithreaded projection time: %f\n", (float)(clock() - startProjectTimeMP) / CLOCKS_PER_SEC);
    
    float maxVelocity = 0.0f;
    
    // clock_t startProjectTimeMP = clock();
    
    // Assign data to the global thread data structures
    asignDataToThreads(particles, camera, validParticles);
    
    // Reset max velocity for each thread
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i]->maxVelocity = 0.0f;
    }
    
    // Signal threads to start processing
    for (int i = 0; i < NUM_THREADS; i++) {
        ready[i] = 1;
    }
    
    // Wait until all threads have processed their data
    for (int i = 0; i < NUM_THREADS; i++) {
        while (ready[i]) {
            // Busy wait or use a short sleep
            usleep(100); // Sleep for 100 microseconds
        }
    }
    
    // Calculate the maximum velocity from all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        if (threadData[i]->maxVelocity > maxVelocity) {
            maxVelocity = threadData[i]->maxVelocity;
        }
    }
    
    // printf("Multithreaded projection time: %f\n", (float)(clock() - startProjectTimeMP) / CLOCKS_PER_SEC);


    // clock_t starProjectTimeSingle = clock();
    // for (int i = 0; i < validParticles - 1; i++) {
    //     float x = particles->x[i] - camX;
    //     float y = particles->y[i] - camY;
    //     float z = particles->z[i] - camZ;

    //     float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
    //     float fovScale = 1.0f / (dotProduct * camera->fov);

    //     float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
    //     float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;

    //     int screenX = (int)(screenRight * halfWidth + halfWidth);
    //     int screenY = (int)(-screenUp * halfHeight + halfHeight);

    //     float vx = particles->xVelocity[i];
    //     float vy = particles->yVelocity[i];
    //     float vz = particles->zVelocity[i];

        
    //     if (screenX < 0 || screenX >= ScreenWidth || screenY < 0 || screenY >= ScreenHeight) continue;

    //     float totalVelocity = vx*vx + vy*vy + vz*vz;

    //     if (totalVelocity > maxVelocity) {
    //         maxVelocity = totalVelocity;
    //     }

    //     particles->screenX[i] = screenX;
    //     particles->screenY[i] = screenY;
    //     particles->totalVelocity[i] = totalVelocity;
    // }
    // printf("Single thread projection time: %f\n", (float)(clock() - starProjectTimeSingle) / CLOCKS_PER_SEC);


    clock_t endProjectTime = clock();
    timePartition->projectionTime = ((endProjectTime - endSortTime) / CLOCKS_PER_SEC) * 0.25f + timePartition->projectionTime * 0.75f;

    float maxOpacity = 0;
    // Process particles back-to-front (furthest to nearest)
    for (int i = validParticles - 1; i >= 0; i--) {       
        int screenX = particles->screenX[i];
        int screenY = particles->screenY[i];

        // Fast distance approximation using squared distance
        float distSquared = particles->distance[i];
        
        // Use inverse square root approximation if available for speed
        float invDist = 1.0f / (sqrtf(distSquared) + 10.0f);
        int particleRadius = (int)(PARTICLE_RADIUS * 100.0f * invDist);
        
        // Apply bounds without branching
        particleRadius = particleRadius < 1 ? 1 : (particleRadius > PARTICLE_RADIUS * 3 ? PARTICLE_RADIUS * 3 : particleRadius);

        // Calculate drawing bounds
        int minX = screenX - particleRadius;
        int maxX = screenX + particleRadius;
        int minY = screenY - particleRadius;
        int maxY = screenY + particleRadius;
        
        // Quick bounds check
        if (minX < 0 || maxX >= ScreenWidth || minY < 0 || maxY >= ScreenHeight) continue;
        
        // Calculate particle color once
        uint8_t NormalizedVelocity = (uint8_t)(255 * (1.0f - (particles->totalVelocity[i] / maxVelocity)));
        uint8_t distanceNormalized = (uint8_t)(255 * (1.0f - (distSquared / maxDistance)));

        // Track maximum values
        float currentOpacity = (float)(screen->opacity[screenX][screenY]) + 1.0f;
        if (currentOpacity > maxOpacity) maxOpacity = currentOpacity;

        for (int px = minX; px <= maxX; px++) {
            for (int py = minY; py <= maxY; py++) {
                screen->distance[px][py] = distanceNormalized;
                screen->velocity[px][py] = NormalizedVelocity;
                screen->opacity[px][py]++;
            }
        }
    }

    clock_t renderDistanceVelocity = clock();
    timePartition->renderDistanceVelocityTime = ((renderDistanceVelocity - endProjectTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->renderDistanceVelocityTime * 0.75f;
    

    // Pre-calculate normalization factors to avoid repeated calculations
    float invLogMaxOpacity = 1.0f / logf(maxOpacity + 1.0f);
    float invLogMaxVelocity = 1.0f / logf(maxVelocity + 1.0f);
    
    // Normalize opacity and velocity in a single pass
    for (int px = 0; px < ScreenWidth; px++) {
        for (int py = 0; py < ScreenHeight; py++) {
            uint16_t opacity = screen->opacity[px][py];
            if (opacity != 0) {
                float logOpacity = logf((float)opacity + 1.0f) * invLogMaxOpacity;
                screen->normalizedOpacity[px][py] = (uint8_t)(255 * (1.0f - logOpacity));
            }
        }
    }
    clock_t renderOpacityTime = clock();
    timePartition->renderOpacityTime = ((renderOpacityTime - renderDistanceVelocity)  / CLOCKS_PER_SEC) * 0.25f + timePartition->renderOpacityTime * 0.75f;
}


void clearScreen(struct Screen *screen) {
    // Clear all screen buffers using memset - much faster than nested loops
    memset(screen->distance, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->velocity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->normalizedOpacity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->opacity, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
}

void saveScreen(struct Screen *screen, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    
    static uint8_t buffer[ScreenWidth * 3];
    
    // Process each row
    for (int y = 0; y < ScreenHeight; y++) {
        // Pack row data into the buffer for faster writes
        for (int x = 0; x < ScreenWidth; x++) {
            buffer[x*3]     = screen->distance[x][y];
            buffer[x*3 + 1] = screen->velocity[x][y];
            buffer[x*3 + 2] = screen->normalizedOpacity[x][y];
        }
        
        // Write entire row at once
        fwrite(buffer, 1, ScreenWidth * 3, file);
    }
    
    fclose(file);
}

void render(struct Screen *screen, struct PointSOA *particles, struct Camera *camera, struct Cursor *cursor, struct TimePartition *timePartition, struct ThreadsData *threadsData) {
    // printf("\n--- Starting render ---\n");
    int start = clock();
    clearScreen(screen);
    int clearScreenTime = clock();
    timePartition->clearScreenTime = ((clearScreenTime - start)  / CLOCKS_PER_SEC) * 0.25f + timePartition->clearScreenTime * 0.75f;
    projectParticles(particles, camera, screen, timePartition, threadsData);
    int projectParticlesTime = clock();
    timePartition->projectParticlesTime = ((projectParticlesTime - clearScreenTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->projectParticlesTime * 0.75f;
    drawCursor(screen, cursor, camera);
    int drawCursorTime = clock();
    timePartition->drawCursorTime = ((drawCursorTime - projectParticlesTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->drawCursorTime * 0.75f;
    drawBoundingBox(screen, particles->bBoxMax, particles->bBoxMin, camera);
    int drawBoundingBoxTime = clock();
    timePartition->drawBoundingBoxTime = ((drawBoundingBoxTime - drawCursorTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->drawBoundingBoxTime * 0.75f;
    saveScreen(screen, "output.bin");
    int saveScreenTime = clock();
    timePartition->saveScreenTime = ((saveScreenTime - drawBoundingBoxTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->saveScreenTime * 0.75f;
}

void addForce(struct PointSOA *particles, struct Cursor *cursor) {
    if (!cursor->active) return;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float dx = particles->x[i] - cursor->x;
        float dy = particles->y[i] - cursor->y;
        float dz = particles->z[i] - cursor->z;
        float distSquared = dx*dx + dy*dy + dz*dz;
        float dist = sqrtf(distSquared);
        
        // Prevent extreme forces at very close distances
        const float minDistance = 2.0f;  // Minimum effective distance
        if (dist < minDistance) {
            dist = minDistance;
        }
        
        // Apply force with the minimum distance protection
        float forceFactor = cursor->force / dist;
        particles->xVelocity[i] += dx * forceFactor;
        particles->yVelocity[i] += dy * forceFactor;
        particles->zVelocity[i] += dz * forceFactor;
    }
}


void update_particles(struct PointSOA *particles, float dt, struct TimePartition *timePartition, struct Cursor *cursor) {
    clock_t start = clock();
    CollideParticlesInGrid(particles);
    clock_t collideParticlesTime = clock();
    timePartition->collisionTime = ((collideParticlesTime - start)  / CLOCKS_PER_SEC) * 0.25f + timePartition->collisionTime * 0.75f;
    
    ApplyPressure(particles);
    clock_t applyPressureTime = clock();
    timePartition->applyPressureTime = ((applyPressureTime - collideParticlesTime)  / CLOCKS_PER_SEC) * 0.25 + timePartition->applyPressureTime * 0.75f;
    
    addForce(particles, cursor);

    update_particle_apply_gravity(particles,dt);
    
    clock_t updateParticlesTime = clock();
    timePartition->updateParticlesTime = ((updateParticlesTime - applyPressureTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->updateParticlesTime * 0.75f;
    
    move_to_box(particles, particles->bBoxMin, particles->bBoxMax);
    clock_t moveToBoxTime = clock();
    timePartition->moveToBoxTime = (((moveToBoxTime - updateParticlesTime)  / CLOCKS_PER_SEC) * 0.25f + timePartition->moveToBoxTime * 0.75f);
}

// read the pause.bin
void readPauseData(bool *paused) {
    FILE *file = fopen("pause.bin", "rb");
    if (!file) {
        return;  // If file doesn't exist, keep current pause state
    }
    
    // Read the boolean value from the file
    bool fileValue;
    if (fread(&fileValue, sizeof(bool), 1, file) == 1) {
        *paused = fileValue;  // Update the pause state with the value from the file
    }
    
    fclose(file);
}


int main() {
    struct Camera camera;
    camera.ray.origin[0] = 50.0f;
    camera.ray.origin[1] = 50.0f;
    camera.ray.origin[2] = -50.0f;
    camera.ray.direction[0] = 0.0f;
    camera.ray.direction[1] = 0.0f;
    camera.ray.direction[2] = 1.0f;
    camera.fov = 1.0f;


    // Initialize threadData array before creating threads
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i] = (struct ThreadData *)malloc(sizeof(struct ThreadData));
        if (!threadData[i]) {
            perror("Failed to allocate memory for thread data");
            return 1;
        }
        // Initialize basic fields
        threadData[i]->maxVelocity = 0.0f;
    }
    createThreads();

    struct PointSOA *particles = (struct PointSOA *)malloc(sizeof(struct PointSOA));
    if (!particles) {
        perror("Failed to allocate memory for particles");
        return 1;
    }


    // crete threadsData
    struct ThreadsData *threadsData = (struct ThreadsData *)malloc(sizeof(struct ThreadsData));
    if (!threadsData) {
        perror("Failed to allocate memory for threads data");
        free(particles);
        return 1;
    }
    
    // Initialize each thread data pointer
    for (int i = 0; i < NUM_THREADS; i++) {
        threadsData->threadData[i] = (struct ThreadData *)malloc(sizeof(struct ThreadData));
        if (!threadsData->threadData[i]) {
            perror("Failed to allocate memory for thread data");
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(threadsData->threadData[j]);
            }
            free(threadsData);
            free(particles);
            return 1;
        }
    }

    asignDataToThreads(particles, &camera, NUM_PARTICLES);

    struct Screen *screen = (struct Screen *)malloc(sizeof(struct Screen));
    if (!screen) {
        perror("Failed to allocate memory for screen");
        free(particles);
        return 1;
    }

    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles->x[i] = (float)(rand() % 100);
        particles->y[i] = (float)(rand() % 100);
        particles->z[i] = (float)(rand() % 100);
        particles->xVelocity[i] = (float)(rand() % 10) / 10.0f;
        particles->yVelocity[i] = (float)(rand() % 10) / 10.0f;
        particles->zVelocity[i] = (float)(rand() % 10) / 10.0f;
    }

    // initialize the cursor
    struct Cursor *cursor = (struct Cursor *)malloc(sizeof(struct Cursor));
    if (!cursor) {
        perror("Failed to allocate memory for cursor");
        free(particles);
        free(screen);
        return 1;
    }
    cursor->x = 0.0f;
    cursor->y = 0.0f;
    cursor->z = 0.0f;
    cursor->active = false;

    updateGridData(particles);


    particles->bBoxMin[0] = 0.0f;
    particles->bBoxMin[1] = 0.0f;
    particles->bBoxMin[2] = 0.0f;
    particles->bBoxMax[0] = 150.0f;
    particles->bBoxMax[1] = 150.0f;
    particles->bBoxMax[2] = 150.0f;

    clearScreen(screen);

    float averageFPS[FrameCount];
    int averageUpdateTime = 0;
    int averageRenderTime = 0;
    int frameCount = 0;
    bool paused = false;

    struct TimePartition *timePartition = (struct TimePartition *)malloc(sizeof(struct TimePartition));
    if (!timePartition) {
        perror("Failed to allocate memory for time partition");
        free(particles);
        free(screen);
        return 1;
    }

    
    clock_t lastTime = clock();
    while (1) {
        // Calculate delta step based on elapsed time since the last frame
        clock_t currentTime = clock();
        float dt = (float)(currentTime - lastTime) / CLOCKS_PER_SEC;
        // Cap dt to avoid instability for long delays (e.g., if paused)
        if (dt > 0.1f) dt = 0.1f;
        lastTime = currentTime;
        
        int loopStartTime = clock();
        
        readCameraData(&camera);
        readCursorData(cursor);
        readPauseData(&paused);
        
        // Update the grid data and record timing
        // First, update the grid data regardless of pause state
        int startGridTime = clock();
        updateGridData(particles);
        int endGridTime = clock();
        timePartition->updateGridTime = ((endGridTime - startGridTime) / CLOCKS_PER_SEC) * 0.25f +
                                        timePartition->updateGridTime * 0.75f;

        // Only update particles if NOT paused
        if (!paused) {
            update_particles(particles, dt, timePartition, cursor);
        } else {
            printf("Simulation paused, skipping update\n");
        }
        
        int afterUpdateTime = clock();
        float averageUpdateTime = (afterUpdateTime - loopStartTime) / CLOCKS_PER_SEC;
        
        int startRenderTime = clock();
        render(screen, particles, &camera, cursor, timePartition, threadsData);
        int endRenderTime = clock();
        timePartition->renderTime = (float)((endRenderTime - startRenderTime)  / CLOCKS_PER_SEC) * 0.25f +
                                    timePartition->renderTime * 0.75f;


        float frameTime = (float)(endRenderTime - startRenderTime) / CLOCKS_PER_SEC;
        float currentFPS = 1.0f / frameTime;
        
        if (frameCount < FrameCount) {
            averageFPS[frameCount] = currentFPS;
        }
        
        float averageRenderTime = (endRenderTime - startRenderTime) / CLOCKS_PER_SEC;
        
        printf("FPS: %.2f, dt: %.4f, Update: %f ms, Render: %f ms\n", 
               currentFPS, dt,
               (averageUpdateTime),
               (averageRenderTime));
        
        if (frameCount >= FrameCount) {
            frameCount = 0;
            FILE *fpsFile = fopen("average_fps.bin", "wb");
            if (fpsFile) {
                fwrite(averageFPS, sizeof(float), FrameCount, fpsFile);
                fclose(fpsFile);
            }
            FILE *timeFile = fopen("time_partition.bin", "wb");
            if (timeFile) {
                fwrite(timePartition, sizeof(struct TimePartition), 1, timeFile);
                fclose(timeFile);
            }
        }
        frameCount++;
    }
    
    // Clean up
    free(particles);
    free(screen);
    
    return 0;
}
