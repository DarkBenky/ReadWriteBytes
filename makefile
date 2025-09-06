CC = clang
CFLAGS = -O3 -march=native
LDFLAGS = -lm -lOpenCL -ljpeg -lglfw -lGL -g

all: main

main: main.c particleSim.c tinyobj_loader_c.h particleSim.h
	$(CC) $(CFLAGS) main.c particleSim.c -o main $(LDFLAGS)

clean:
	rm -f main

.PHONY: all clean