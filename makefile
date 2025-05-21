CC = clang
CFLAGS = -O3 -march=native -fopenmp
LDFLAGS = -lm

all: main

main: main.c
    $(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
    rm -f main