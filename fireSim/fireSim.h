#ifndef FIRE_SIM_H
#define FIRE_SIM_H

#define NUM_FIRE_PARTICLES 250
#define MISSILE_SEEKER_SIZE 96
#define IRST_TRACKING_LIMIT 16
#define MAX_FIRE_SIMS 16
#define G 9.81f
#define MAX_FLOAT 3.402823466e+38F

#include "../openGlShaders/gpuStruct.h"
#include <stdbool.h>

struct FireSOA {
    float x[NUM_FIRE_PARTICLES];
    float y[NUM_FIRE_PARTICLES];
    float z[NUM_FIRE_PARTICLES];
    float xVelocity[NUM_FIRE_PARTICLES];
    float yVelocity[NUM_FIRE_PARTICLES];
    float zVelocity[NUM_FIRE_PARTICLES];
    float lifeTime[NUM_FIRE_PARTICLES];
    float basePosition[3];        // Spawn point for particles
    float startingColor[3];       // Initial particle color (RGB 0-1)
    float fireColor[3];           // Hot flame color (RGB 0-1)
    float smokeColor[3];          // Cool smoke color (RGB 0-1)
    float windDirection[3];       // Environmental wind vector
    float maxLifeTime;            // Particle lifetime before respawn (s)
    float buoyancy;               // Upward force from heat (m/s²)
    float drag;                   // Air resistance coefficient (0=none, 1=full stop)
    float turbulence;             // Random motion intensity
    float maxVelocity;            // Peak velocity reached (m/s)
    float particlesSize;          // Render size of particles
    float maxDistance;            // Farthest distance from base (m)
    float swirlIntensity;         // Circular motion strength
    float swirlFrequency;         // Rotation speed (Hz)
};


enum MissileLock{
    Searching,
    Tracking,
    Lunching
};

struct Seeker {
    float seekerDepthMap[MISSILE_SEEKER_SIZE * MISSILE_SEEKER_SIZE];
    struct Camera seekerCamera;
    float seekerFov;
    float searchMultiplayer;
    float tiltSpeed;
    enum  MissileLock lockState;
    float searchYaw;        // Individual search yaw angle for this missile
    float searchPitch;      // Individual search pitch angle for this missile
};

struct IRSearchAndTrack {
    struct Camera *mainRenderingCamera; // Reference to main scene camera
    struct Camera seekerCamera;
    float seekerFov;
    float seekerDepthMap[MISSILE_SEEKER_SIZE * MISSILE_SEEKER_SIZE];
    float tiltSpeed;
    float searchYaw;        // Individual search yaw angle for this missile
    float searchPitch;      // Individual search pitch angle for this missile
    float targetScreenX[IRST_TRACKING_LIMIT];
    float targetScreenY[IRST_TRACKING_LIMIT];
    float targetTemperature[IRST_TRACKING_LIMIT];
    float targetPositionX[IRST_TRACKING_LIMIT];
    float targetPositionY[IRST_TRACKING_LIMIT];
    float targetPositionZ[IRST_TRACKING_LIMIT];
    int selectedTargetId;
    int targetCount;
    int mainScreenWidth;
    int mainScreenHeight;
    struct Missile *lockedTarget;
    int lockedTargetId;
    float lockTime;
    
    // Multi-scan tracking state
    struct Missile *trackedTargets[IRST_TRACKING_LIMIT]; // Pointers to missiles being tracked
    int scanCount[IRST_TRACKING_LIMIT];                   // Number of consecutive successful scans per target
    int requiredScans;                                     // Number of scans needed before displaying target
};

struct Missile {
    // Core simulation
    struct Seeker seeker;                 // Target detection and tracking system
    float position[3];                    // World coordinates (x, y, z) in meters
    float velocity[3];                    // Velocity vector (m/s)
    float acceleration[3];                // Current acceleration vector (m/s²)
    float targetPosition[3];              // Absolute target position in world space
    float targetDirection[3];             // Normalized direction to target (unit vector)
    
    // Aerodynamic state
    float bodyOrientation[3];             // Missile's forward direction (unit vector)
    float angularVelocity[3];             // Rotation rates around axes (rad/s)
    float angleOfAttack;                  // Angle between body and velocity vector (radians)
    float sideslipAngle;                  // Side-slip angle for crosswind effects (radians)
    
    // Mass properties  
    float dryMass;                        // Empty missile mass without fuel (kg)
    float fuelMass;                       // Remaining propellant mass (kg)
    float totalMass;                      // Current total mass (dry + fuel) (kg)
    float momentOfInertia;                // Resistance to rotational acceleration (kg·m²)
    
    // Propulsion
    float thrust;                         // Current thrust output (Newtons)
    float Isp;                            // Specific impulse - fuel efficiency (seconds)
    int burning;                          // Engine state: 1=active, 0=shutdown
    float burnRate;                       // Fuel consumption rate (kg/s)
    float maxGimbalAngle;                 // Maximum nozzle deflection angle (radians)
    float gimbalAngle[2];                 // Current pitch/yaw gimbal angles (radians)
    float burningTemp;                    // Temperature of burning propellant (Kelvin)
    float heatAspect[6];                  // Heat radiation from each face of the missile
    float initialFuelMass;                // Stored initial fuel mass for burned-mass calculations
    
    // Aerodynamic coefficients
    float zeroLiftDrag;                   // Base drag coefficient at zero lift (Cd0)
    float liftSlope;                      // Lift curve slope - how much lift per AoA (per radian)
    float maxLiftCoeff;                   // Maximum achievable lift coefficient (Cl_max)
    float dragSlope;                      // Drag increase with AoA squared
    float crossSectionArea;               // Frontal area for drag calculations (m²)
    float wingArea;                       // Wing/fin reference area for lift (m²)
    float aspectRatio;                    // Wing aspect ratio (span²/area)
    float oswaldEfficiency;               // Wing efficiency factor (0.7-0.95 typical)
    
    // Control surfaces
    float finMaxDeflection;               // Maximum fin deflection angle (radians)
    float finDeflection[4];               // Individual fin angles [rad] for 4-fin configuration
    float finEffectiveness;               // Fin control power (0-1, higher = more responsive)
    float rollDamping;                    // Natural roll damping coefficient
    
    // Performance limits
    float maxGPull;                       // Maximum lateral acceleration capability (g-forces)
    float maxDynamicPressure;             // Structural limit for dynamic pressure (Pascals)
    float maxAoA;                         // Maximum safe angle of attack (radians)
    // Example: Higher maxAoA (e.g., 0.52 rad/30°) allows more aggressive turns but risks stall
    float maxLoadFactor;                  // Maximum structural g-load
    float maxSpeed;                       // Maximum velocity limit (m/s)
    
    // Guidance & control
    float guidanceGain;                   // Proportional navigation gain constant
    // Example: Higher gain (4-6) = more aggressive interception, lower (2-3) = smoother pursuit
    float controlAuthority;               // Overall control system effectiveness (0-1)
    float energyManagementFactor;         // Energy conservation vs maneuver tradeoff (0-1)
    // Example: Lower value (0.6) = aggressive turns, higher (0.9) = energy conservation
    float optimalSpeed;                   // Design cruise speed for best performance (m/s)
    
    // Sensor parameters
    float searchConeAngle;                // Search mode cone width (radians)
    float searchConeDepth;                // Search mode detection range (meters)
    float trackingConeAngle;              // Tracking mode cone width (radians)
    float trackingConeDepth;              // Tracking mode detection range (meters)
    float sensorFusionWeight;             // How much to trust sensor data vs image (0-1)
    float engineSignalSensitivity;        // Multiplier for engine thrust detection
    float velocitySignalSensitivity;      // Multiplier for velocity signal detection
    float minTrackConfidence;             // Minimum confidence to maintain tracking (0-1)
    
    // Simulation
    float remainingTime;                  // Time until self-destruct (seconds)
    struct FireSOA *fireSim;              // Exhaust plume and visual effects
    
    // Cached values (updated each frame)
    float machNumber;                     // Current Mach number (speed/speed_of_sound)
    float dynamicPressure;                // Current dynamic pressure 0.5*ρ*v² (Pa)
    float prevLOS[3];                     // Previous line-of-sight vector for guidance
    int targetIdx;                       // Index of current target in missile list
};

struct Missiles {
    struct Missile *missiles[MAX_FIRE_SIMS];
    struct Triangles *missileModel;
    float coneOriginsX[MAX_FIRE_SIMS];
    float coneOriginsY[MAX_FIRE_SIMS];
    float coneOriginsZ[MAX_FIRE_SIMS];
    float coneDirsX[MAX_FIRE_SIMS];
    float coneDirsY[MAX_FIRE_SIMS];
    float coneDirsZ[MAX_FIRE_SIMS];
    float coneFovs[MAX_FIRE_SIMS];
    float coneMaxDistances[MAX_FIRE_SIMS];
    bool active[MAX_FIRE_SIMS];
    int activeCount;
    int count;
};

void InitializeFireParticles(struct FireSOA *particles);
void fireSimStep(struct FireSOA *particles, float deltaTime, float *timeTook);
void updateSeekerPositions(struct Missiles *missiles);
void InitializeMissile(struct Missile *missile);
void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook, bool *active, float *fireSimulationTime);
void setMissileTarget(struct Missile *missile, float targetPos[3]);
void setMissileTargetDirection(struct Missile *missile, float targetDir[3], float *targetDist);
void cleanupMissile(struct Missile *missile);

void InitializeMissiles(struct Missiles *missiles, int count, struct Triangles *missileModel);
void CleanupMissiles(struct Missiles *missiles);
void missileSeekStep(struct Missile *missile, struct Missiles *allMissiles, bool fire, bool *active, float deltaTime, float *timeTook, float *fireSimulationTime, float lunchDir[3], float lunchPos[3]);
float randRange(float min, float max);

void scanConeForTargets(struct Missile *missile, struct Missiles *allMissiles, float coneAngle, float coneDepth);
void fuseSensorData(struct Missile *missile, float *fusedTargetPos, float *fusedConfidence);

void InitializeIRST(struct Camera *mainRenderingCamera, struct IRSearchAndTrack *irst, int screenWidth, int screenHeight);
void IRSearchAndTrackStep(struct Missiles *allMissiles, struct IRSearchAndTrack *irst, float deltaTime);
void IRSTSelectNextTarget(struct IRSearchAndTrack *irst);
void IRSTSelectPreviousTarget(struct IRSearchAndTrack *irst);
void IRSTClearSelection(struct IRSearchAndTrack *irst);

#endif
