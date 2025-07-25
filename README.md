# Binary Frame Read/Write Performance Test

This experiment tests the performance of writing and reading frame data as binary files using Go. The test simulates rendering a frame (800x600x3 bytes) and measures the read/write speeds and frame rates.

## Test Specifications
- Frame dimensions: 800x600 pixels
- Color depth: 3 bytes per pixel (RGB)
- Total frame size: 1,440,000 bytes (~1.44MB)

## Performance Results

### Write Performance
- Speed: 636.30 MB/s
- Frame write time: 2,263,080 ns (~2.26ms)
- Write FPS: 441.88 frames/second

### Read Performance
- Speed: 388.02 MB/s
- Frame read time: 3,711,145 ns (~3.71ms)
- Read FPS: 269.46 frames/second

## Conclusion
The results show that writing frames is faster than reading them, with write operations capable of handling ~442 FPS while read operations can handle ~269 FPS. This suggests that the system could theoretically handle real-time frame processing for applications requiring up to 60 FPS.

## TODO

- [ ] Fix BVH Lineation
- [ ] Trace Ray Function prototype ![trace diagram](trace.png)
  - [ ] Finish raytracing kernel
    - [ ] Initialize kernel
      - [ ] BVH buffer ...
- [ ] convert timing to this format

  ```c
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  saveScreenNormal(screen, "normal.bin");
  clock_gettime(CLOCK_MONOTONIC, &end);
  double ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
  ```
  
- [ ] (Not For Now) Render in C code do not save to file
  - [ ] (To Do) Write to shared memory - share image as shared memory
- [ ] Wire Frame
- [ ] Better screen space reflections (Ray Tracing)
  - [ ] add shadows (with ray tracing)
- [ ] (Look Into) Fix .obj file parsing
- [ ] Update timimg (CPU)
- [ ] add emission and bloom
- [ ] render fluid in c (open gl) and openCL ([link](https://tympanus.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/))

### Add Open GL to the go code to project particles on GPU directly
- Render headlessly the image and save it in the to drive so the go code can read it
- [ ] implement

### Add better timing
- [ ] add timing
- [ ] test timing why i get so weird numbers

### Threading and Performance [ ***BackLog*** ]
- [ ] Implement simulation and rendering in different threads
  - Render thread should run at a fixed rate (e.g., 24 FPS)
  - Lock simulation thread when scene is being rendered
  - Run simulation as fast as possible, but adjust step size based on TPS (higher TPS = smaller simulation steps)
- [ ] why pragma opm is not helping

### Graphics Enhancement
- [ ] Accelerate rendering with CUDA
- [ ] add screens based fluid rendering

### DONE
- [X] Fix hole in in middle of screen space reflections ![example](hole.png)
- [X] Add chart for gpu timing
- [X] Add Timing for rendering of particles
- [X] Add Text Color
- [X] New Triangle Renderer ([link](https://chatgpt.com/c/6878486d-ee08-8004-b21e-31c714a8479f))
- [X] Add text support
- [X] GPU Timeing
  - [X] renderSkyBoxTime;
  - [X] renderTrianglesTime;
  - [X] applyReflectionsTime;
  - [X] applyBlurTime;
  - [X] readBackTime;
- [X] Triangles are static, we can upload them once and reuse them
- [X] Screen Space Reflections ( general purpose function where you provide image ray and it will return color )
- [X] add go code to parse obj files
- [X] add skybox directly to the render triangles shader so i can sample the sky box to get realistic reflections of sky
- [X] implement multithreading (project particles in diffrent threads)
- [x] rework it to work based on grid
- [x] particle grid based on the Sebastian's video https://m.youtube.com/watch?v=pLwYMecqOxY
- [X] more realistic attraction and repulsion
- [X] optimized the sorting of the particles
- [X] Update the MP code
- [X] Encode more data to the images for better fluid rendering
- [X] Build Rasterize - 3D
- [X] Idea we can render based on distance buffer we don't need to sort
- [X] add sky box and the ground
  - [X] get sky box textures and load it
  - [X] crete check board ground
