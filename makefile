CC = clang
CFLAGS = -O3 -march=native
LDFLAGS = -lm

all: main

main: main.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f main