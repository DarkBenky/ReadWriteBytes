CC = clang
CFLAGS = -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -fomit-frame-pointer -fno-stack-protector -fno-math-errno -ffinite-math-only -fno-signed-zeros -fno-trapping-math -freciprocal-math -DNDEBUG
LDFLAGS = -flto -lm -lOpenCL -ljpeg

all: loadToGpu

loadToGpu: loadToGpu.c ../mapGeneration/loadMap.c ../mapGeneration/loadMap.h ../openGlShaders/gpuStruct.h ../utils/bbox.c ../utils/image.c
	$(CC) $(CFLAGS) loadToGpu.c ../mapGeneration/loadMap.c ../utils/image.c ../utils/bbox.c -o loadToGpu $(LDFLAGS)

run: loadToGpu
	./loadToGpu

clean:
	rm -f loadToGpu

.PHONY: all run clean
