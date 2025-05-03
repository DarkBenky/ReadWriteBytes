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

- [ ] rework it to work based on grid
- [ ] particle grid based on the Sebastian's video https://m.youtube.com/watch?v=pLwYMecqOxY
