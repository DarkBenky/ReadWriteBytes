# Benchmark Quick Reference

This workspace contains two standalone benchmarks for performance testing.

## Fire Simulation Benchmark

**Location:** `fireSim/`

**Build & Run:**
```bash
cd fireSim
make run
```

**What it tests:**
- Fire particle physics (buoyancy, drag, turbulence, wind)
- 750 particles with lifetime management
- Particle respawning system

**Results:** ~15.57 M particles/second, ~260K particles @ 60 FPS

---

## Particle Simulation Benchmark

**Location:** Root directory

**Build & Run:**
```bash
make -f Makefile.particleBench run
```

**What it tests:**
- Full fluid dynamics system
- 15,000 particles with grid-based spatial partitioning
- Pressure, viscosity, surface tension, collisions
- 4 sub-steps per frame

**Results:** ~1.24 M particles/second, ~20K particles @ 60 FPS

---

## Comparison

| Feature | Fire Sim | Particle Sim |
|---------|----------|--------------|
| Particles | 750 | 15,000 |
| Complexity | Simple | Complex |
| Per-particle cost | 0.064 μs | 0.807 μs |
| Throughput | 15.57 M/s | 1.24 M/s |
| 60 FPS capacity | ~260K | ~20K |
| Use case | Visual effects | Fluid simulation |

---

## Quick Commands

```bash
# Fire benchmark
cd fireSim && make clean && make run

# Particle benchmark
make -f Makefile.particleBench clean && make -f Makefile.particleBench run

# Main application
make clean && make && ./main
```

## Customization

**Fire particles:** Edit `NUM_FIRE_PARTICLES` in `fireSim/fireSim.c`

**Particle count:** Edit `NUM_PARTICLES` in `particleSim.c`

**Grid resolution:** Edit `gridResolutionAxis` in `particleSim.c`
