#ifndef FIRE_SIM_H
#define FIRE_SIM_H

#define NUM_FIRE_PARTICLES 750
#define MAX_FIRE_SIMS 128
#define G 9.81f

#include "../openGlShaders/gpuStruct.h"
#include <stdbool.h>

// struct FireSOA {
//     float x[NUM_FIRE_PARTICLES];
//     float y[NUM_FIRE_PARTICLES];
//     float z[NUM_FIRE_PARTICLES];
//     float xVelocity[NUM_FIRE_PARTICLES];
//     float yVelocity[NUM_FIRE_PARTICLES];
//     float zVelocity[NUM_FIRE_PARTICLES];
//     float lifeTime[NUM_FIRE_PARTICLES];
//     float basePosition[3];        // Spawn point for particles
//     float startingColor[3];       // Initial particle color (RGB 0-1)
//     float fireColor[3];           // Hot flame color (RGB 0-1)
//     float smokeColor[3];          // Cool smoke color (RGB 0-1)
//     float windDirection[3];       // Environmental wind vector
//     float maxLifeTime;            // Particle lifetime before respawn (s)
//     float buoyancy;               // Upward force from heat (m/s²)
//     float drag;                   // Air resistance coefficient (0=none, 1=full stop)
//     float turbulence;             // Random motion intensity
//     float maxVelocity;            // Peak velocity reached (m/s)
//     float particlesSize;          // Render size of particles
//     float maxDistance;            // Farthest distance from base (m)
//     float swirlIntensity;         // Circular motion strength
//     float swirlFrequency;         // Rotation speed (Hz)
// };

struct Missile {
    // Core simulation
    struct Seeker seeker;                 // Target seeking system (must keep)
    float position[3];                    // Current position (m)
    float velocity[3];                    // Current velocity (m/s)
    float acceleration[3];                // NEW: Current acceleration (m/s²) for guidance
    float targetPosition[3];              // NEW: Target position in world space
    float targetDirection[3];             // Desired flight direction (unit vector)
    
    // Aerodynamic state
    float bodyOrientation[3];             // Physical body alignment (unit vector)
    float angularVelocity[3];             // Rotation rate (rad/s)
    float angleOfAttack;                  // NEW: Current AoA (radians)
    float sideslipAngle;                  // NEW: Current sideslip (radians)
    
    // Mass properties
    float dryMass;                        // Empty weight (kg)
    float fuelMass;                       // Remaining propellant (kg)
    float totalMass;                      // NEW: Cached total mass
    float momentOfInertia;                // Resistance to rotation (kg·m²)
    
    // Propulsion
    float thrust;                         // NEW: Current thrust (N)
    float Isp;                            // Specific impulse (s)
    int burning;                          // Engine state
    float burnRate;                       // Fuel consumption (kg/s)
    float maxGimbalAngle;                 // NEW: Thrust vectoring limit (radians)
    float gimbalAngle[2];                 // NEW: Current gimbal angles (pitch, yaw)
    
    // Aerodynamic coefficients
    float zeroLiftDrag;                   // NEW: Cd0 - zero lift drag coefficient
    float liftSlope;                      // NEW: dCl/dα (per radian)
    float maxLiftCoeff;                   // NEW: Maximum lift coefficient
    float dragSlope;                      // NEW: dCd/dα² (drag due to AoA)
    float crossSectionArea;               // Frontal area (m²)
    float wingArea;                       // NEW: Wing/fin reference area (m²)
    float aspectRatio;                    // NEW: Wing aspect ratio
    float oswaldEfficiency;               // NEW: Wing efficiency factor (0.8-0.95)
    
    // Control surfaces
    float finMaxDeflection;               // NEW: Maximum fin deflection (radians)
    float finDeflection[4];               // NEW: Individual fin deflections (rad)
    float finEffectiveness;               // NEW: Fin control power
    float rollDamping;                    // NEW: Natural roll damping
    
    // Performance limits
    float maxGPull;                       // Maximum lateral acceleration (g)
    float maxDynamicPressure;             // Structural limit (Pa)
    float maxAoA;                         // NEW: Maximum angle of attack (rad)
    float maxLoadFactor;                  // NEW: Maximum structural g-load
    
    // Guidance & control
    float guidanceGain;                   // Proportional navigation gain
    float controlAuthority;               // Control surface effectiveness
    float energyManagementFactor;         // Energy conservation
    float optimalSpeed;                   // Target cruise speed (m/s)
    
    // Simulation
    float remainingTime;                  // Remaining simulation time (s)
    struct FireSOA *fireSim;              // Exhaust plume particles (must keep)
    
    // NEW: Cached values for efficiency
    float machNumber;                     // Current Mach number
    float dynamicPressure;                // Current dynamic pressure (Pa)
};

enum MissileLock{
    Searching,
    Tracking,
};

struct Seeker {
    struct Camera seekerCamera;
    float seekerFov; // Seeker gimbal limit (degrees)
    int seekerSteps; // Number of steps preferred in one simulation step
    enum  MissileLock lockState;

};

struct Missile {
    struct Seeker seeker;
    float position[3];                    // Current position in world space (m)
    float velocity[3];                    // Current velocity vector (m/s)
    float targetDirection[3];             // Desired flight direction (unit vector)
    float bodyOrientation[3];             // Physical body alignment (unit vector)
    float angularVelocity[3];             // Rotation rate (rad/s)
    float drag;                           // Base drag coefficient (lower=more aerodynamic)
    float inducedDragFactor;              // Additional drag from maneuvering (0=none, 0.2=high)
    float transsonicDragPeak;             // Drag multiplier at Mach 0.8-1.2 (1=none, 4=severe)
    float supersonicDragFactor;           // Drag increase above Mach 1.2 per Mach (0.3=low, 0.7=high)
    float crossSectionArea;               // Frontal area for drag/lift (m²)
    float liftCoefficient;                // Body lift generation efficiency (0.25=low, 0.55=high)
    float maxDynamicPressure;             // Structural limit (Pa, 80k=weak, 150k=strong)
    float thrustVectoringEfficiency;      // Thrust effectiveness at angle (0.75=poor, 0.95=excellent)
    float momentOfInertia;                // Resistance to rotation (kg·m², lower=more agile)
    float controlAuthority;               // Control surface effectiveness (0.85=limited, 0.98=excellent)
    float energyManagementFactor;         // Conservation aggressiveness (0.6=aggressive turns, 0.9=conservative)
    float minEnergyThreshold;             // Energy ratio to start limiting maneuvers (0.3=late, 0.5=early)
    float optimalSpeed;                   // Target cruise speed (m/s)
    float dryMass;                        // Empty weight (kg)
    float fuelMass;                       // Remaining propellant (kg)
    float maxGPull;                       // Maximum lateral acceleration (g-force)
    float Isp;                            // Specific impulse efficiency (s, higher=better)
    int burning;                          // Engine state: 1=on, 0=off
    float burnRate;                       // Fuel consumption rate (kg/s)
    float Q_spec;                         // Energy per kg of fuel (J/kg)
    float remainingTime;                  // Remaining simulation time (s)
    struct FireSOA *fireSim;              // Exhaust plume particles
};


struct Missiles {
    struct Missile *missiles[MAX_FIRE_SIMS];
    struct Triangles *missileModel;
    bool active[MAX_FIRE_SIMS];
    int count;
};

void InitializeFireParticles(struct FireSOA *particles);
void fireSimStep(struct FireSOA *particles, float deltaTime, float *timeTook);

void InitializeMissile(struct Missile *missile);
void missileSimStep(struct Missile *missile, float deltaTime, float *timeTook, bool *active);
void setMissileTarget(struct Missile *missile, float targetPos[3]);
void cleanupMissile(struct Missile *missile);

void InitializeMissiles(struct Missiles *missiles, int count, struct Triangles *missileModel);
void UpdateAllMissiles(struct Missiles *missiles, float deltaTime);
void CleanupMissiles(struct Missiles *missiles);

#endif
