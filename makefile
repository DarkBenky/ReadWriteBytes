CC = clang
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm -lOpenCL -ljpeg

all: main

main: main.c tinyobj_loader_c.h
	$(CC) $(CFLAGS) main.c -o main $(LDFLAGS)

clean:
	rm -f main

.PHONY: all clean