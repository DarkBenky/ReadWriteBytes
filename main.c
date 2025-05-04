#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>     // for rand()
#include <unistd.h>     // for usleep()
#include <math.h>       // for sqrtf()

#define NUM_PARTICLES 1000
#define GRAVITY 9.81f
#define DAMPING 0.9f
#define ScreenWidth 800
#define ScreenHeight 600
#define PARTICLE_RADIUS 3
#define MAX_SPEED 35.0f
#define gridResolutionAxis 240
#define gridResolution (gridResolutionAxis * gridResolutionAxis * gridResolutionAxis)

struct PointSOA {
    float   x[NUM_PARTICLES];
    float   y[NUM_PARTICLES];
    float   z[NUM_PARTICLES];
    float   xVelocity[NUM_PARTICLES];
    float   yVelocity[NUM_PARTICLES];
    float   zVelocity[NUM_PARTICLES];
    float   bBoxMin[3];
    float   bBoxMax[3];
    int     gridID[NUM_PARTICLES];
    int     startIndex[gridResolution];
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

void move_to_grid(struct PointSOA *particles, int index) {
    float xStep = (particles->bBoxMax[0] - particles->bBoxMin[0]) / gridResolutionAxis;
    float yStep = (particles->bBoxMax[1] - particles->bBoxMin[1]) / gridResolutionAxis;
    float zStep = (particles->bBoxMax[2] - particles->bBoxMin[2]) / gridResolutionAxis;
    
    int xIndex = 0, yIndex = 0, zIndex = 0;
    
    // Find x grid position
    for (int i = 0; i < gridResolutionAxis; i++) {
        if (particles->x[index] < particles->bBoxMin[0] + xStep * (i + 1)) {
            xIndex = i;
            break;
        }
    }
    
    // Find y grid position
    for (int i = 0; i < gridResolutionAxis; i++) {
        if (particles->y[index] < particles->bBoxMin[1] + yStep * (i + 1)) {
            yIndex = i;
            break;
        }
    }
    
    // Find z grid position
    for (int i = 0; i < gridResolutionAxis; i++) {
        if (particles->z[index] < particles->bBoxMin[2] + zStep * (i + 1)) {
            zIndex = i;
            break;
        }
    }
    
    // Convert 3D coordinates to a single index
    particles->gridID[index] = xIndex + yIndex * gridResolutionAxis + zIndex * gridResolutionAxis * gridResolutionAxis;
}

void updateGridData(struct PointSOA *particles) {

    for (int i = 0; i < NUM_PARTICLES; i++) {
        move_to_grid(particles, i);
    }
    

    for (int i = 0; i < gridResolution; i++) {
        particles->startIndex[i] = -1;
    }
    

    for (int i = 1; i < NUM_PARTICLES; i++) {
        int j = i;
        while (j > 0 && particles->gridID[j-1] > particles->gridID[j]) {
            swapParticles(particles, j, j-1);
            j--;
        }
    }

    for (int i = 0; i < NUM_PARTICLES; i++) {
        int gridID = particles->gridID[i];
        if (i == 0 || particles->gridID[i-1] != gridID) {
            particles->startIndex[gridID] = i;
        }
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

void projectParticles(struct PointSOA *particles, struct Camera *camera, struct Screen *screen) {
    // printf("Starting projectParticles\n");
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float x = particles->x[i] - camera->ray.origin[0];
        float y = particles->y[i] - camera->ray.origin[1];
        float z = particles->z[i] - camera->ray.origin[2];

        // Project along the camera direction vector
        float dotProduct = x * camera->ray.direction[0] + 
                          y * camera->ray.direction[1] + 
                          z * camera->ray.direction[2];
                          
        // Skip particles behind the camera
        if (dotProduct <= 0) continue;

        // Apply perspective projection
        float fovScale = 1.0f / (dotProduct * camera->fov);
        
        // Calculate screen coordinates based on camera orientation
        // Create a simple projection plane perpendicular to camera direction
        float up[3] = {0, 1, 0};  // Simplified up vector
        
        // Calculate right vector (cross product of direction and up)
        float right[3];
        right[0] = camera->ray.direction[1] * up[2] - camera->ray.direction[2] * up[1];
        right[1] = camera->ray.direction[2] * up[0] - camera->ray.direction[0] * up[2];
        right[2] = camera->ray.direction[0] * up[1] - camera->ray.direction[1] * up[0];
        
        // Recalculate up vector to be perpendicular to right and direction
        float trueUp[3];
        trueUp[0] = right[1] * camera->ray.direction[2] - right[2] * camera->ray.direction[1];
        trueUp[1] = right[2] * camera->ray.direction[0] - right[0] * camera->ray.direction[2];
        trueUp[2] = right[0] * camera->ray.direction[1] - right[1] * camera->ray.direction[0];
        
        // Project onto right and up vectors to get screen coordinates
        float screenRight = x * right[0] + y * right[1] + z * right[2];
        float screenUp = x * trueUp[0] + y * trueUp[1] + z * trueUp[2];
        
        int screenX = (int)((screenRight * fovScale) * 0.5f * ScreenWidth + ScreenWidth / 2);
        int screenY = (int)((-screenUp * fovScale) * 0.5f * ScreenHeight + ScreenHeight / 2);

        if (i % 100 == 0) {
            // printf("Particle %d: screenX=%d, screenY=%d\n", i, screenX, screenY);
        }

        // Calculate particle size based on distance
        float distance = sqrtf(x*x + y*y + z*z);
        int particleRadius = (int)(PARTICLE_RADIUS * (100.0f / (distance + 10.0f)));
        
        // Ensure a minimum and maximum size
        if (particleRadius < 1) particleRadius = 1;
        if (particleRadius > PARTICLE_RADIUS * 3) particleRadius = PARTICLE_RADIUS * 3;

        // Draw particle based on radius and the velocity
        for (int dx = -particleRadius; dx <= particleRadius; dx++) {
            for (int dy = -particleRadius; dy <= particleRadius; dy++) {
                int px = screenX + dx;
                int py = screenY + dy;
                if (px >= 0 && px < ScreenWidth && py >= 0 && py < ScreenHeight) {
                    // Calculate the total velocity (speed) of the particle
                    float sumVelocity = sqrtf(particles->xVelocity[i] * particles->xVelocity[i] +
                                            particles->yVelocity[i] * particles->yVelocity[i] +
                                            particles->zVelocity[i] * particles->zVelocity[i]);
                    
                    // Normalize velocity to MAX_SPEED for color calculations
                    float normalizedVelocity = sumVelocity / MAX_SPEED;
                    if (normalizedVelocity > 1.0f) normalizedVelocity = 1.0f;
                    
                    // Create a color gradient based on velocity:
                    // Slow: Blue -> Medium: Green -> Fast: Red
                    uint8_t r, g, b;
                    
                    if (normalizedVelocity < 0.5f) {
                        // Blue to Green transition (slow to medium)
                        float t = normalizedVelocity * 2.0f; // Scale to 0-1 range
                        r = 0;
                        g = (uint8_t)(255 * t);
                        b = (uint8_t)(255 * (1.0f - t));
                    } else {
                        // Green to Red transition (medium to fast)
                        float t = (normalizedVelocity - 0.5f) * 2.0f; // Scale to 0-1 range
                        r = (uint8_t)(255 * t);
                        g = (uint8_t)(255 * (1.0f - t));
                        b = 0;
                    }
                    
                    // Add velocity direction influence
                    // X velocity affects red, Y affects green, Z affects blue
                    float dirInfluence = 0.3f; // How much direction affects color
                    
                    // Replace fminf with a simple conditional
                    float temp;
                    
                    if (particles->xVelocity[i] > 0) {
                        temp = r + 255 * dirInfluence * (particles->xVelocity[i] / MAX_SPEED);
                        r = (temp > 255) ? 255 : (uint8_t)temp;
                    }
                    if (particles->yVelocity[i] > 0) {
                        temp = g + 255 * dirInfluence * (particles->yVelocity[i] / MAX_SPEED);
                        g = (temp > 255) ? 255 : (uint8_t)temp;
                    }
                    if (particles->zVelocity[i] > 0) {
                        temp = b + 255 * dirInfluence * (particles->zVelocity[i] / MAX_SPEED);
                        b = (temp > 255) ? 255 : (uint8_t)temp;
                    }
                    
                    // Add distance-based color modulation
                    float distanceFactor = 1.0f - (distance / 150.0f);
                    if (distanceFactor < 0.2f) distanceFactor = 0.2f;
                    
                    r = (uint8_t)(r * distanceFactor);
                    g = (uint8_t)(g * distanceFactor);
                    b = (uint8_t)(b * distanceFactor);
                    
                    // Add glow effect for fast particles
                    if (normalizedVelocity > 0.8f) {
                        float glowFactor = (normalizedVelocity - 0.8f) * 5.0f; // Scale to 0-1
                        
                        temp = r + (255 - r) * glowFactor;
                        r = (temp > 255) ? 255 : (uint8_t)temp;
                        
                        temp = g + (255 - g) * glowFactor * 0.7f;
                        g = (temp > 255) ? 255 : (uint8_t)temp;
                        
                        temp = b + (255 - b) * glowFactor * 0.5f;
                        b = (temp > 255) ? 255 : (uint8_t)temp;
                    }
                    
                    screen->data[px][py][0] = r;
                    screen->data[px][py][1] = g;
                    screen->data[px][py][2] = b;
                    screen->data[px][py][3] = 255; // Alpha
                }
            }
        }
    }
    // printf("Finished projectParticles\n");
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
    for (int i = 0; i < NUM_PARTICLES; i++) {
        add_gravity(particles, i);
        update_particle(particles, i, dt);
        move_to_box(particles, particles->bBoxMin, particles->bBoxMax);
    }
    // printf("Finished update_particles\n");
}


int main() {
    // printf("Starting particle simulation\n");
    
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

    // printf("Starting main loop\n");
    float dt = 0.016f; // 60 FPS
    int frameCount = 0;
    int maxFrames = 1000; // Limit frames to avoid infinite loop during testing
    
    while (frameCount < maxFrames) {
        // printf("Frame %d\n", frameCount);
        // Update camera from file if available
        readCameraData(&camera);
        
        // update particles
        update_particles(particles, dt);

        // render particles
        render(screen, particles, &camera);

        // wait for a short time (simulate frame delay)
        // printf("Sleeping for 16ms\n");
        usleep(16000); // 16 ms for 60 FPS
        
        frameCount++;
    }
    
    // Clean up
    free(particles);
    free(screen);
    
    return 0;
}