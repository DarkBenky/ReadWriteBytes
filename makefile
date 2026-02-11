CC = clang
CFLAGS = -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -fomit-frame-pointer -fno-stack-protector -fno-math-errno -ffinite-math-only -fno-signed-zeros -fno-trapping-math -freciprocal-math -DNDEBUG
LDFLAGS = -flto -lm -lOpenCL -ljpeg -lglfw -lGL

all: main

main: app.c main.c particleSim.c fireSim/fireSim.c flares/flare.c tinyobj_loader_c.h particleSim.h fireSim/fireSim.h flares/flare.h openGlShaders/gpuStruct.h mapGeneration/loadMap.c mapGeneration/loadMap.h utils/bbox.c utils/bbox.h utils/image.c utils/image.h app.h
	$(CC) $(CFLAGS) app.c main.c particleSim.c fireSim/fireSim.c flares/flare.c mapGeneration/loadMap.c utils/bbox.c utils/image.c -o main $(LDFLAGS)

clean:
	rm -f main

.PHONY: all clean
