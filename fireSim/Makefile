CC = gcc
CFLAGS = -O3 -march=native -Wall -DFIRE_BENCHMARK
LDFLAGS = -lm

TARGET = fireSim_benchmark
SRC = fireSim.c ../mapGeneration/loadMap.c ../flares/flare.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
