#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>     // for rand()
#include <unistd.h>     // for usleep()
#include <math.h>       // for sqrtf()
#include <time.h>      // for time()
#include <string.h>     // for memset()
#include <stdbool.h>    // for bool, true, false

#define NUM_PARTICLES 100000
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

struct PointSOA {
    float   x[NUM_PARTICLES];
    float   y[NUM_PARTICLES];
    float   z[NUM_PARTICLES];
    float   xVelocity[NUM_PARTICLES];
    float   yVelocity[NUM_PARTICLES];
    float   zVelocity[NUM_PARTICLES];
    float   distance[NUM_PARTICLES];
    float   bBoxMin[3];
    float   bBoxMax[3];
    int     gridID[NUM_PARTICLES];
    int     startIndex[gridResolution];
    int     numberOfParticle[gridResolution];
};

struct Screen {
    uint8_t distance[ScreenWidth][ScreenHeight];
    uint8_t velocity[ScreenWidth][ScreenHeight];
    uint8_t normalizedOpacity[ScreenWidth][ScreenHeight];
    uint16_t opacity[ScreenWidth][ScreenHeight];
    uint16_t unNormalizedVelocity[ScreenWidth][ScreenHeight];
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
    int collisionTime;
    int applyPressureTime;
    int updateParticlesTime;
    int moveToBoxTime;
    int updateGridTime;
    int renderTime;
    int clearScreenTime;
    int projectParticlesTime;
    int drawCursorTime;
    int drawBoundingBoxTime;
    int saveScreenTime;
};

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
                    screen->distance[x0][y0] = 255; // Distance
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
                screen->distance[screenX][py] = 255;
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


void CollideParticlesInGrid(struct PointSOA *particles) {
    for (int gridId = 0; gridId < gridResolution; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        int endIdx = startIdx + particles->numberOfParticle[gridId];
        
        // Check collisions between all particles in this cell
        for (int i = startIdx; i < endIdx; i++) {
            for (int j = i + 1; j < endIdx; j++) {
                // Calculate distance between particles
                float dx = particles->x[j] - particles->x[i];
                float dy = particles->y[j] - particles->y[i];
                float dz = particles->z[j] - particles->z[i];
                
                float distSquared = dx*dx + dy*dy + dz*dz;
                float minDist = 3.0f * PARTICLE_RADIUS;
                
                // If particles are colliding (distance < 2*radius)
                if (distSquared < minDist * minDist) {
                    // Calculate collision normal
                    float dist = sqrtf(distSquared);
                    float nx = dx / dist;
                    float ny = dy / dist;
                    float nz = dz / dist;
                    
                    // Calculate relative velocity along normal
                    float vx = particles->xVelocity[j] - particles->xVelocity[i];
                    float vy = particles->yVelocity[j] - particles->yVelocity[i];
                    float vz = particles->zVelocity[j] - particles->zVelocity[i];
                    
                    float velocityAlongNormal = vx*nx + vy*ny + vz*nz;
                    
                    // Only collide if particles are moving toward each other
                    if (velocityAlongNormal < 0) {
                        // Calculate impulse scalar (assuming equal mass)
                        float impulse = -2.0f * velocityAlongNormal;
                        
                        // Apply impulse
                        particles->xVelocity[i] -= impulse * nx * 0.5f;
                        particles->yVelocity[i] -= impulse * ny * 0.5f;
                        particles->zVelocity[i] -= impulse * nz * 0.5f;
                        
                        particles->xVelocity[j] += impulse * nx * 0.5f;
                        particles->yVelocity[j] += impulse * ny * 0.5f;
                        particles->zVelocity[j] += impulse * nz * 0.5f;
                        
                        // Add some damping to collision
                        particles->xVelocity[i] *= DAMPING;
                        particles->yVelocity[i] *= DAMPING;
                        particles->zVelocity[i] *= DAMPING;
                        
                        particles->xVelocity[j] *= DAMPING;
                        particles->yVelocity[j] *= DAMPING;
                        particles->zVelocity[j] *= DAMPING;
                        
                        // Push particles apart to avoid sticking
                        float correction = (minDist - dist) * 0.01f;
                        particles->x[i] -= nx * correction;
                        particles->y[i] -= ny * correction;
                        particles->z[i] -= nz * correction;
                        
                        particles->x[j] += nx * correction;
                        particles->y[j] += ny * correction;
                        particles->z[j] += nz * correction;
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

void ApplyPressure(struct PointSOA *particles) {
    // Process each cell
    for (int gridId = 0; gridId < gridResolution; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        
        int endIdx = startIdx + particles->numberOfParticle[gridId];
        float currentPressure = pressure * particles->numberOfParticle[gridId];
        
        // Get 3D grid coordinates
        int x = gridId % gridResolutionAxis;
        int y = (gridId / gridResolutionAxis) % gridResolutionAxis;
        int z = gridId / (gridResolutionAxis * gridResolutionAxis);
        
        // Process each particle in this cell
        for (int i = startIdx; i < endIdx; i++) {
            float netForceX = 0.0f;
            float netForceY = 0.0f;
            float netForceZ = 0.0f;
            
            // Check all 6 face-adjacent neighbors
            const int neighbors[6][3] = {
                {-1, 0, 0}, {1, 0, 0},  // left, right
                {0, -1, 0}, {0, 1, 0},  // down, up
                {0, 0, -1}, {0, 0, 1}   // back, front
            };
            
            // Consider all neighbors for pressure gradient
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
                // Calculate neighbor pressure
                float neighborPressure = pressure * particles->numberOfParticle[neighborGridId];
                
                // Calculate pressure difference
                float pressureDiff = currentPressure - neighborPressure;
                
                // Only push if there's pressure gradient
                if (pressureDiff > 0.0f) {
                    // Add force component in direction of this neighbor
                    netForceX += neighbors[j][0] * pressureDiff;
                    netForceY += neighbors[j][1] * pressureDiff;
                    netForceZ += neighbors[j][2] * pressureDiff;
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
void move_to_grid(struct PointSOA *particles, int index) {
    float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
    float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
    float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
    // Calculate grid positions directly with clamping
    int xIndex = (int)((particles->x[index] - particles->bBoxMin[0]) / xStep);
    int yIndex = (int)((particles->y[index] - particles->bBoxMin[1]) / yStep);
    int zIndex = (int)((particles->z[index] - particles->bBoxMin[2]) / zStep);
    
    // Clamp indices
    xIndex = (xIndex < 0) ? 0 : ((xIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : xIndex);
    yIndex = (yIndex < 0) ? 0 : ((yIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : yIndex);
    zIndex = (zIndex < 0) ? 0 : ((zIndex >= gridResolutionAxis) ? gridResolutionAxis - 1 : zIndex);
    
    // Convert 3D coordinates to a single index
    particles->gridID[index] = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
}

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

void updateGridData(struct PointSOA *particles) {
    // Assign each particle to its grid cell
    for (int i = 0; i < NUM_PARTICLES; i++) {
        move_to_grid(particles, i);
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

void update_particle(struct PointSOA *particles, int index, float dt) {
    particles->x[index] += particles->xVelocity[index] * dt * DAMPING;
    particles->y[index] += particles->yVelocity[index] * dt * DAMPING;
    particles->z[index] += particles->zVelocity[index] * dt * DAMPING;
}

void add_gravity(struct PointSOA *particles, int index) {
    particles->yVelocity[index] -= GRAVITY;
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



void projectParticles(struct PointSOA *particles, struct Camera *camera, struct Screen *screen) {
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
    
    // Initialize tracking variables
    float maxOpacity = 0;
    float maxVelocity = 0;

    // Process particles back-to-front (furthest to nearest)
    for (int i = validParticles - 1; i >= 0; i--) {
        // Reuse calculations from distance computation
        float x = particles->x[i] - camX;
        float y = particles->y[i] - camY;
        float z = particles->z[i] - camZ;

        float dotProduct = x * camDirX + y * camDirY + z * camDirZ;
        float fovScale = 1.0f / (dotProduct * camera->fov);
        
        // Project onto screen
        float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
        float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;
        
        int screenX = (int)(screenRight * halfWidth + halfWidth);
        int screenY = (int)(-screenUp * halfHeight + halfHeight);

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
        float vx = particles->xVelocity[i];
        float vy = particles->yVelocity[i];
        float vz = particles->zVelocity[i];
        float sumVelocity = vx*vx + vy*vy + vz*vz;
        uint8_t distanceNormalized = (uint8_t)(255 * (1.0f - (distSquared / maxDistance)));

        // Track maximum values
        float currentOpacity = (float)(screen->opacity[screenX][screenY]) + 1.0f;
        if (currentOpacity > maxOpacity) maxOpacity = currentOpacity;
        if (sumVelocity > maxVelocity) maxVelocity = sumVelocity;

        // Draw particle with tight loop bounds
        uint16_t sumVelocityInt = (uint16_t)sumVelocity;
        for (int px = minX; px <= maxX; px++) {
            for (int py = minY; py <= maxY; py++) {
                screen->distance[px][py] = distanceNormalized;
                screen->unNormalizedVelocity[px][py] = sumVelocityInt;
                screen->opacity[px][py]++;
            }
        }
    }
    
    // Pre-calculate normalization factors to avoid repeated calculations
    float invLogMaxOpacity = 1.0f / logf(maxOpacity + 1.0f);
    float invLogMaxVelocity = 1.0f / logf(maxVelocity + 1.0f);
    
    // Normalize opacity and velocity in a single pass
    for (int px = 0; px < ScreenWidth; px++) {
        for (int py = 0; py < ScreenHeight; py++) {
            uint16_t opacity = screen->opacity[px][py];
            uint16_t velocity = screen->unNormalizedVelocity[px][py];
            
            if (opacity != 0) {
                float logOpacity = logf((float)opacity + 1.0f) * invLogMaxOpacity;
                screen->normalizedOpacity[px][py] = (uint8_t)(255 * (1.0f - logOpacity));
            }
            
            if (velocity != 0) {
                float logVelocity = logf((float)velocity + 1.0f) * invLogMaxVelocity;
                screen->velocity[px][py] = (uint8_t)(255 * (1.0f - logVelocity));
            }
        }
    }
}


void clearScreen(struct Screen *screen) {
    // Clear all screen buffers using memset - much faster than nested loops
    memset(screen->distance, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->velocity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->normalizedOpacity, 0, sizeof(uint8_t) * ScreenWidth * ScreenHeight);
    memset(screen->opacity, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
    memset(screen->unNormalizedVelocity, 0, sizeof(uint16_t) * ScreenWidth * ScreenHeight);
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

void render(struct Screen *screen, struct PointSOA *particles, struct Camera *camera, struct Cursor *cursor, struct TimePartition *timePartition) {
    // printf("\n--- Starting render ---\n");
    int start = clock();
    clearScreen(screen);
    int clearScreenTime = clock();
    timePartition->clearScreenTime = (int)((clearScreenTime - start) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->clearScreenTime * 0.75f;
    projectParticles(particles, camera, screen);
    int projectParticlesTime = clock();
    timePartition->projectParticlesTime = (int)((projectParticlesTime - clearScreenTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->projectParticlesTime * 0.75f;
    drawCursor(screen, cursor, camera);
    int drawCursorTime = clock();
    timePartition->drawCursorTime = (int)((drawCursorTime - projectParticlesTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->drawCursorTime * 0.75f;
    drawBoundingBox(screen, particles->bBoxMax, particles->bBoxMin, camera);
    int drawBoundingBoxTime = clock();
    timePartition->drawBoundingBoxTime = (int)((drawBoundingBoxTime - drawCursorTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->drawBoundingBoxTime * 0.75f;
    saveScreen(screen, "output.bin");
    int saveScreenTime = clock();
    timePartition->saveScreenTime = (int)((saveScreenTime - drawBoundingBoxTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->saveScreenTime * 0.75f;
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
    timePartition->collisionTime = (int)((collideParticlesTime - start) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->collisionTime * 0.75f;
    
    ApplyPressure(particles);
    clock_t applyPressureTime = clock();
    timePartition->applyPressureTime = (int)((applyPressureTime - collideParticlesTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25 + timePartition->applyPressureTime * 0.75f;
    
    addForce(particles, cursor);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        add_gravity(particles, i);
        update_particle(particles, i, dt);
    }
    clock_t updateParticlesTime = clock();
    timePartition->updateParticlesTime = (int)((updateParticlesTime - applyPressureTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->updateParticlesTime * 0.75f;
    
    move_to_box(particles, particles->bBoxMin, particles->bBoxMax);
    clock_t moveToBoxTime = clock();
    timePartition->moveToBoxTime = (int)((moveToBoxTime - updateParticlesTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f + timePartition->moveToBoxTime * 0.75f;
}


int main() {

    struct PointSOA *particles = (struct PointSOA *)malloc(sizeof(struct PointSOA));
    if (!particles) {
        perror("Failed to allocate memory for particles");
        return 1;
    }


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


    struct Camera camera;
    camera.ray.origin[0] = 50.0f;
    camera.ray.origin[1] = 50.0f;
    camera.ray.origin[2] = -50.0f;
    camera.ray.direction[0] = 0.0f;
    camera.ray.direction[1] = 0.0f;
    camera.ray.direction[2] = 1.0f;
    camera.fov = 1.0f;


    clearScreen(screen);

    float dt = 0.08f; // 60 FPS
    float averageFPS[FrameCount];
    int averageUpdateTime = 0;
    int averageRenderTime = 0;
    int frameCount = 0;

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
        
        // Update the grid data and record timing
        int startGridTime = clock();
        updateGridData(particles);
        int endGridTime = clock();
        timePartition->updateGridTime = (int)((endGridTime - startGridTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f +
                                        timePartition->updateGridTime * 0.75f;
        
        // Update particles using the dynamic time step
        update_particles(particles, dt, timePartition, cursor);
        
        int afterUpdateTime = clock();
        averageUpdateTime = (afterUpdateTime - loopStartTime) * 0.1f + averageUpdateTime * 0.9f;
        
        int startRenderTime = clock();
        render(screen, particles, &camera, cursor, timePartition);
        int endRenderTime = clock();
        timePartition->renderTime = (int)((endRenderTime - startRenderTime) * 1000.0f / CLOCKS_PER_SEC) * 0.25f +
                                    timePartition->renderTime * 0.75f;
        
        // Calculate frame time and FPS
        float frameTime = (float)(endRenderTime - loopStartTime) / CLOCKS_PER_SEC;
        float currentFPS = 1.0f / frameTime;
        
        if (frameCount < FrameCount) {
            averageFPS[frameCount] = currentFPS;
        }
        
        averageRenderTime = (endRenderTime - startRenderTime) * 0.25f + averageRenderTime * 0.75f;
        
        printf("FPS: %.2f, dt: %.4f, Update: %d ms, Render: %d ms\n", 
               currentFPS, dt,
               (int)(averageUpdateTime * 1000.0f / CLOCKS_PER_SEC),
               (int)(averageRenderTime * 1000.0f / CLOCKS_PER_SEC));
        
        // No sleeping here as the simulation runs continuously with the dynamic dt
        
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
