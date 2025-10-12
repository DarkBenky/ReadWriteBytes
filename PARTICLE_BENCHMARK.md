# Particle Simulation Benchmark

Standalone benchmark utility for testing the CPU-based particle physics simulation performance with fluid dynamics, pressure, viscosity, and collision detection.

## Building

```bash
make -f Makefile.particleBench
```

## Running

```bash
./particleSim_benchmark
```

Or build and run in one step:

```bash
make -f Makefile.particleBench run
```

## What it Tests

This benchmark simulates a complete fluid dynamics system including:

- **Gravity and Physics**: Realistic particle motion
- **Grid-based Spatial Partitioning**: 32x32x32 grid for efficient neighbor queries
- **Pressure System**: Pressure gradients with Gaussian blur smoothing
- **Collision Detection**: Particle-particle collisions within grid cells
- **Viscosity**: Fluid resistance to flow
- **Surface Tension**: Cohesion forces at fluid boundaries
- **Sub-stepping**: 4 sub-steps per frame for stability

## Output

The benchmark provides comprehensive metrics:

- **Per-frame statistics**: Average, minimum, and maximum frame times
- **FPS calculations**: Real-world achievable frame rates
- **Physics statistics**: Total kinetic energy tracking
- **Performance metrics**: Time per particle, throughput
- **Capacity estimates**: Maximum particles at 60 FPS and 30 FPS
- **Grid statistics**: Cell occupancy and distribution

## Customization

To test with different particle counts, modify in `particleSim.c`:

```c
#define NUM_PARTICLES 15000  // Change this value
```

To adjust grid resolution:

```c
#define gridResolutionAxis 32  // Change grid size (32x32x32 = 32768 cells)
```

Then recompile:

```bash
make -f Makefile.particleBench clean && make -f Makefile.particleBench run
```

## Example Output

```
=== Particle Simulation Benchmark ===
Particle count: 15000
Grid resolution: 32x32x32 (32768 cells)

Warming up...
Running 100 frames...
  100/100 frames complete

=== Results ===
Total frames: 100
Wall clock time: 1210.60 ms

Per-frame statistics:
  Average: 12.1047 ms (82.6 FPS)
  Minimum: 11.0906 ms (90.2 FPS)
  Maximum: 14.3910 ms (69.5 FPS)

Physics statistics:
  Average total energy: 73743136.00

Performance metrics:
  Time per particle: 0.000807 ms
  Particles per ms: 1239.18
  Particles per second: 1.24 M

Capacity estimates (CPU only):
  Max particles @ 60 FPS: ~20653
  Max particles @ 30 FPS: ~41305

Grid statistics:
  Non-empty cells: 7240 / 32768 (22.1%)
  Max particles per cell: 11
  Average particles per non-empty cell: 2.07
```

## Performance Analysis

### Current Results (15,000 particles)
- **82.6 FPS average** - Good real-time performance
- **1.24 M particles/second** throughput
- **22.1% grid occupancy** - Efficient spatial partitioning

### Capacity Estimates
- **~20K particles @ 60 FPS** - Suitable for real-time fluid simulation
- **~41K particles @ 30 FPS** - Suitable for high-quality offline rendering

## Optimization Notes

This benchmark tests **CPU-only** simulation. Performance characteristics:

- **Grid resolution**: 32³ = 32,768 cells provides good balance
- **Sub-stepping**: 4 sub-steps per frame ensures stability
- **Collision detection**: O(n²) within cells, but O(n) across all particles due to grid
- **Memory**: ~2.5 MB per PointSOA structure

## Comparison with Fire Simulation

| Metric | Particle Sim | Fire Sim | Ratio |
|--------|--------------|----------|-------|
| Particles | 15,000 | 750 | 20x |
| Throughput | 1.24 M/s | 15.57 M/s | 0.08x |
| Time/particle | 0.807 μs | 0.064 μs | 12.6x |

The particle simulation is ~13x more expensive per particle due to:
- Grid updates
- Pressure calculations
- Collision detection
- Viscosity and surface tension
- Multiple sub-steps (4x)

## GPU Acceleration Potential

With OpenCL/GPU acceleration, this simulation could potentially achieve:
- **10-50x speedup** for pressure and viscosity calculations
- **5-10x speedup** for collision detection (with spatial hashing)
- **500K-1M particles @ 60 FPS** (estimated)
