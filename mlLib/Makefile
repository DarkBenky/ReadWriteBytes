CC = gcc
CFLAGS = -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -fomit-frame-pointer -fno-stack-protector -fno-math-errno -ffinite-math-only -fno-signed-zeros -fno-trapping-math -freciprocal-math -DNDEBUG
LDFLAGS = -flto -lOpenCL -lm

all: test

test: test.c ml.c ml.h
	$(CC) $(CFLAGS) test.c ml.c -o test $(LDFLAGS)

clean:
	rm -f test

run: test
	./test

.PHONY: all clean run
