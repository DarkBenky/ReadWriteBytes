# Fire Simulation Benchmark

This is a standalone benchmark utility for testing the CPU-based fire particle simulation performance.

## Building

```bash
make
```

This compiles `fireSim.c` with the `FIRE_BENCHMARK` flag, creating a standalone benchmark executable.

## Running

```bash
./fireSim_benchmark
```

Or build and run in one step:

```bash
make run
```

## Output

The benchmark provides:

- **Per-step statistics**: Average, minimum, and maximum frame times
- **FPS calculations**: Theoretical maximum frame rates
- **Particle statistics**: Max velocity and distance traveled
- **Performance metrics**: Time per particle, throughput
- **Capacity estimates**: How many particles you can simulate at different target frame rates

## Customization

To test with different particle counts, modify the `NUM_FIRE_PARTICLES` define in `fireSim.c`:

```c
#define NUM_FIRE_PARTICLES 750  // Change this value
```

Then recompile:

```bash
make clean && make run
```

## Example Output

```
=== Fire Particle Simulation Benchmark ===
Particle count: 750

Running 1000 iterations...
  1000/1000 iterations complete

=== Results ===
Total iterations: 1000
Wall clock time: 48.28 ms

Per-step statistics:
  Average: 0.0482 ms (20756.2 FPS)
  Minimum: 0.0457 ms (21867.0 FPS)
  Maximum: 0.0831 ms (12040.2 FPS)

Particle statistics:
  Max velocity: 416.74 units/s
  Max distance: 187.88 units

Performance metrics:
  Time per particle: 0.000064 ms
  Particles per ms: 15567.12
  Particles per second: 15.57 M

Capacity estimates (CPU only):
  Max particles @ 60 FPS: ~259457
  Max particles @ 30 FPS: ~518914
  Max particles @ 15 FPS: ~1037828
```

## Notes

- This benchmark tests **CPU-only** simulation (no GPU rendering)
- Results will vary based on CPU speed and compiler optimizations
- The benchmark includes a warm-up phase to stabilize measurements
- Uses `-O3 -march=native` for optimal performance
