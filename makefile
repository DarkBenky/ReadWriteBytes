CC = clang
CFLAGS = -O3 -march=native
LDFLAGS = -lm -lOpenCL -ljpeg -lglfw -lGL -g

all: main

main: main.c particleSim.c fireSim/fireSim.c tinyobj_loader_c.h particleSim.h fireSim/fireSim.h openGlShaders/gpuStruct.h mapGeneration/loadMap.c mapGeneration/loadMap.h
	$(CC) $(CFLAGS) main.c particleSim.c fireSim/fireSim.c mapGeneration/loadMap.c -o main $(LDFLAGS)

clean:
	rm -f main

.PHONY: all clean
