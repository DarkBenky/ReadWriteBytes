#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>     // for rand()
#include <unistd.h>     // for usleep()
#include <math.h>       // for sqrtf()
#include <time.h>      // for time()
#include <string.h>     // for memset()

#define NUM_PARTICLES 200000
#define GRAVITY 10.0f
#define DAMPING 0.9f
#define ScreenWidth 800
#define ScreenHeight 600
#define PARTICLE_RADIUS 3
#define MAX_SPEED 100.0f
#define gridResolutionAxis 32
#define gridResolution (gridResolutionAxis * gridResolutionAxis * gridResolutionAxis)
#define temperature 1.0f
#define pressure  temperature * 0.01f
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
    uint8_t data[ScreenWidth][ScreenHeight][4]; // RGBA
};

struct Ray {
    float origin[3];
    float direction[3];
};

struct Camera {
    struct Ray ray;
    float fov;
};


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
                float minDist = 2.0f * PARTICLE_RADIUS;
                
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
                        float correction = (minDist - dist) * 0.5f;
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
    for (int gridId = 0; gridId < gridResolution; gridId++) {
        int startIdx = particles->startIndex[gridId];
        if (startIdx == -1) continue; // No particles in this grid cell
        int endIdx = startIdx + particles->numberOfParticle[gridId];
        // calculate center of the given cell
        float centerOfCell[3];
        CalculateCenterOfCell(particles, gridId, centerOfCell);
        float baselineForce = pressure * particles->numberOfParticle[gridId];
        for (int i = startIdx; i < endIdx; i++) {
            // calculate distance from center of cell
            float dx = particles->x[i] - centerOfCell[0];
            float dy = particles->y[i] - centerOfCell[1];
            float dz = particles->z[i] - centerOfCell[2];
            float distSquared = dx*dx + dy*dy + dz*dz;
            float dist = sqrtf(distSquared);
            if (dist > 0) {
                // apply force in the direction of the center of the cell
                particles->xVelocity[i] += baselineForce * dx / dist;
                particles->yVelocity[i] += baselineForce * dy / dist;
                particles->zVelocity[i] += baselineForce * dz / dist;
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
    particles->x[index] += particles->xVelocity[index] * dt;
    particles->y[index] += particles->yVelocity[index] * dt;
    particles->z[index] += particles->zVelocity[index] * dt;
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

    
    // Calculate distances once
    int validParticles = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x = particles->x[i] - camera->ray.origin[0];
        float y = particles->y[i] - camera->ray.origin[1];
        float z = particles->z[i] - camera->ray.origin[2];
        
        float dotProduct = x * camera->ray.direction[0] + 
                         y * camera->ray.direction[1] + 
                         z * camera->ray.direction[2];
                         
        if (dotProduct > 0) { // Only include particles in front of camera
            particles->distance[validParticles] = x*x + y*y + z*z;
            if (validParticles != i) {
                swapParticles(particles, validParticles, i);
            }
            validParticles++;
        }
    }
    
    // Sort particles using quick sort
    quickSortParticles(particles, 0, validParticles - 1, particles->distance);
    
    for (int i = 0; i < validParticles; i++) {
        float x = particles->x[i] - camera->ray.origin[0];
        float y = particles->y[i] - camera->ray.origin[1];
        float z = particles->z[i] - camera->ray.origin[2];

        float dotProduct = x * camera->ray.direction[0] + 
                         y * camera->ray.direction[1] + 
                         z * camera->ray.direction[2];
        
        float fovScale = 1.0f / (dotProduct * camera->fov);
        
        // Project onto screen
        float screenRight = (x * right[0] + y * right[1] + z * right[2]) * fovScale;
        float screenUp = (x * trueUp[0] + y * trueUp[1] + z * trueUp[2]) * fovScale;
        
        int screenX = (int)(screenRight * halfWidth + halfWidth);
        int screenY = (int)(-screenUp * halfHeight + halfHeight);

        // Fast distance approximation using squared distance
        float distSquared = particles->distance[i];
        int particleRadius = (int)(PARTICLE_RADIUS * (100.0f / (sqrtf(distSquared) + 10.0f)));
        particleRadius = particleRadius < 1 ? 1 : (particleRadius > PARTICLE_RADIUS * 3 ? PARTICLE_RADIUS * 3 : particleRadius);

        // Calculate particle color once
        float sumVelocity = sqrtf(particles->xVelocity[i] * particles->xVelocity[i] +
                                particles->yVelocity[i] * particles->yVelocity[i] +
                                particles->zVelocity[i] * particles->zVelocity[i]);
        
        float normalizedVelocity = sumVelocity / MAX_SPEED;
        if (normalizedVelocity > 1.0f) normalizedVelocity = 1.0f;
        
        uint8_t r, g, b;
        if (normalizedVelocity < 0.5f) {
            float t = normalizedVelocity * 2.0f;
            r = (uint8_t)(128 * (1.0f - t));
            g = 0;
            b = 255;
        } else {
            float t = (normalizedVelocity - 0.5f) * 2.0f;
            r = (uint8_t)(255 * t);
            g = (uint8_t)(64 * t);
            b = (uint8_t)(255 * (1.0f - t));
        }

        // Draw particle
        int minX = screenX - particleRadius;
        int maxX = screenX + particleRadius;
        int minY = screenY - particleRadius;
        int maxY = screenY + particleRadius;
        
        // Clamp to screen bounds
        minX = minX < 0 ? 0 : minX;
        maxX = maxX >= ScreenWidth ? ScreenWidth - 1 : maxX;
        minY = minY < 0 ? 0 : minY;
        maxY = maxY >= ScreenHeight ? ScreenHeight - 1 : maxY;

        for (int px = minX; px <= maxX; px++) {
            for (int py = minY; py <= maxY; py++) {
                screen->data[px][py][0] = r;
                screen->data[px][py][1] = g;
                screen->data[px][py][2] = b;
                screen->data[px][py][3] = 255;
            }
        }
    }
}


void clearScreen(struct Screen *screen) {
    // printf("Starting clearScreen\n");
    for (int x = 0; x < ScreenWidth; x++) {
        for (int y = 0; y < ScreenHeight; y++) {
            screen->data[x][y][0] = 0;
            screen->data[x][y][1] = 0;
            screen->data[x][y][2] = 0;
            screen->data[x][y][3] = 0; // Alpha
        }
    }
    // printf("Finished clearScreen\n");
}

void saveScreen(struct Screen *screen, const char *filename) {
    // printf("Starting saveScreen: %s\n", filename);
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        return;
    }
    // DO NOT USE PPM HEADER FOR BINARY FILES
    for (int y = 0; y < ScreenHeight; y++) {
        for (int x = 0; x < ScreenWidth; x++) {
            fputc(screen->data[x][y][0], file);
            fputc(screen->data[x][y][1], file);
            fputc(screen->data[x][y][2], file);
            fputc(screen->data[x][y][3], file); // Alpha
        }
        if (y % 100 == 0) {
            // printf("Saving row %d of %d\n", y, ScreenHeight);
        }
    }
    fclose(file);
    // printf("Finished saveScreen\n");
}

void render(struct Screen *screen, struct PointSOA *particles, struct Camera *camera) {
    // printf("\n--- Starting render ---\n");
    clearScreen(screen);
    projectParticles(particles, camera, screen);
    saveScreen(screen, "output.bin");
    // printf("--- Finished render ---\n");
}

void update_particles(struct PointSOA *particles, float dt) {
    // printf("Starting update_particles with dt=%f\n", dt);
    // clock_t start, collideParticlesTime, applyPressureTime, updateParticlesTime, moveToBoxTime;
    
    // start = clock();
    CollideParticlesInGrid(particles);
    // collideParticlesTime = clock();
    // printf("CollideParticlesInGrid took %d ms\n", (int)((collideParticlesTime - start) * 1000.0f / CLOCKS_PER_SEC));
    
    ApplyPressure(particles);
    // applyPressureTime = clock();
    // printf("ApplyPressure took %d ms\n", (int)((applyPressureTime - collideParticlesTime) * 1000.0f / CLOCKS_PER_SEC));
    
    for (int i = 0; i < NUM_PARTICLES; i++) {
        add_gravity(particles, i);
        update_particle(particles, i, dt);
    }
    // updateParticlesTime = clock();
    // printf("update_particle took %d ms\n", (int)((updateParticlesTime - applyPressureTime) * 1000.0f / CLOCKS_PER_SEC));
    
    move_to_box(particles, particles->bBoxMin, particles->bBoxMax);
    // moveToBoxTime = clock();
    // printf("move_to_box took %d ms\n", (int)((moveToBoxTime - updateParticlesTime) * 1000.0f / CLOCKS_PER_SEC));
}


int main() {
    // Allocate large structures on the heap instead of stack
    struct PointSOA *particles = (struct PointSOA *)malloc(sizeof(struct PointSOA));
    if (!particles) {
        perror("Failed to allocate memory for particles");
        return 1;
    }
    // printf("Successfully allocated particles structure\n");

    struct Screen *screen = (struct Screen *)malloc(sizeof(struct Screen));
    if (!screen) {
        perror("Failed to allocate memory for screen");
        free(particles);
        return 1;
    }
    // printf("Successfully allocated screen structure\n");
    
    // initialize particles
    // printf("Initializing %d particles\n", NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles->x[i] = (float)(rand() % 100);
        particles->y[i] = (float)(rand() % 100);
        particles->z[i] = (float)(rand() % 100);
        particles->xVelocity[i] = (float)(rand() % 10) / 10.0f;
        particles->yVelocity[i] = (float)(rand() % 10) / 10.0f;
        particles->zVelocity[i] = (float)(rand() % 10) / 10.0f;
    }

    // printf("Updating Particles\n");
    // Update grid data
    updateGridData(particles);

    // Set bounding box
    // printf("Setting bounding box\n");
    particles->bBoxMin[0] = 0.0f;
    particles->bBoxMin[1] = 0.0f;
    particles->bBoxMin[2] = 0.0f;
    particles->bBoxMax[0] = 100.0f;
    particles->bBoxMax[1] = 100.0f;
    particles->bBoxMax[2] = 100.0f;

    // initialize camera
    // printf("Initializing camera\n");
    struct Camera camera;
    camera.ray.origin[0] = 50.0f;
    camera.ray.origin[1] = 50.0f;
    camera.ray.origin[2] = -50.0f;
    camera.ray.direction[0] = 0.0f;
    camera.ray.direction[1] = 0.0f;
    camera.ray.direction[2] = 1.0f;
    camera.fov = 1.0f;

    // initialize screen
    // printf("Initializing screen\n");
    clearScreen(screen);

    float dt = 0.016f; // 60 FPS
    float averageFPS[FrameCount];
    int averageUpdateTime = 0;
    int averageRenderTime = 0;
    int frameCount = 0;

    
    while (1) {
            int startTime = clock();
            readCameraData(&camera);
    
            // Update The Grid Data
            int startGridTime = clock();
            updateGridData(particles);
            int endGridTime = clock();
            printf("Grid update took %d ms\n", (int)((endGridTime - startGridTime) * 1000.0f / CLOCKS_PER_SEC));
        
            
            // update particles
            update_particles(particles, dt);
    
            int endTime = clock();
            averageUpdateTime = (endTime - startTime) * 0.1f + averageUpdateTime * 0.9f;
    
            int startRenderTime = clock();
            render(screen, particles, &camera);
            int endRenderTime = clock();
            
            // Calculate total frame time in seconds
            float frameTime = (float)(endRenderTime - startTime) / CLOCKS_PER_SEC;
            float currentFPS = 1.0f / frameTime;
            
            // Store FPS before incrementing frameCount
            if (frameCount < FrameCount) {
                averageFPS[frameCount] = currentFPS;
            }
            
            averageRenderTime = (endRenderTime - startRenderTime) * 0.25f + averageRenderTime * 0.75f;
            
            printf("FPS: %.2f, Update: %d ms, Render: %d ms\n", 
                   currentFPS, 
                   (int)(averageUpdateTime * 1000.0f / CLOCKS_PER_SEC),
                   (int)(averageRenderTime * 1000.0f / CLOCKS_PER_SEC));
    
            // Sleep if we have remaining time in the frame
            int remainingTime = (int)(dt * 1000) - (int)(frameTime * 1000);
            if (remainingTime > 0) {
                usleep(remainingTime * 1000);
            }

            if (frameCount >= FrameCount) {
                frameCount = 0;
                FILE *fpsFile = fopen("average_fps.bin", "wb");
                if (fpsFile) {
                    fwrite(averageFPS, sizeof(float), 300, fpsFile);
                    fclose(fpsFile);
                }
            }
            frameCount++;
    }
    
    // Clean up
    free(particles);
    free(screen);
    
    return 0;
}
