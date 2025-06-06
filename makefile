CC = clang
# -g for debugging
CFLAGS = -O3 -march=native
LDFLAGS = -lm -lOpenCL

all: main

main: main.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f main
.PHONY: clean