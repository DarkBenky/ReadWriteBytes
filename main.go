package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"time"
)

const (
	tests = 1_000_000_000_000_000
)

func CalculateReadWriteSpeed(startTime, endTime time.Time) float64 {
	numBytes := 800 * 600 * 3
	duration := endTime.Sub(startTime).Seconds()
	speed := float64(numBytes) / duration
	return speed
}

func WriteBytesToFile(filename string, data *[800][600][3]uint8) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Convert 3D array to byte slice
	bytes := make([]byte, 800*600*3)
	for x := range 800 {
		for y := range 600 {
			offset := (x*600 + y) * 3
			bytes[offset] = data[x][y][0]
			bytes[offset+1] = data[x][y][1]
			bytes[offset+2] = data[x][y][2]
		}
	}

	_, err = file.Write(bytes)
	return err
}

func ReadBytesFromFile(filename string, buffer *[800][600][3]uint8) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// read the bytes from the file
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return err
	}
	if len(data) != 800*600*3 {
		return fmt.Errorf("unexpected data size: got %d, want %d", len(data), 800*600*3)
	}
	// Convert byte slice back to 3D array
	for x := range 800 {
		for y := range 600 {
			offset := (x*600 + y) * 3
			buffer[x][y][0] = data[offset]
			buffer[x][y][1] = data[offset+1]
			buffer[x][y][2] = data[offset+2]
		}
	}
	return nil
}

func main() {
	// create a 3D array of bytes
	var data [800][600][3]uint8
	for x := range 800 {
		for y := range 600 {
			data[x][y][0] = uint8(x % 256)
			data[x][y][1] = uint8(y % 256)
			data[x][y][2] = uint8((x + y) % 256)
		}
	}

	// Write operation
	startTime := time.Now()
	err := WriteBytesToFile("test.bin", &data)
	if err != nil {
		log.Fatal(err)
	}
	endTime := time.Now()
	duration := endTime.Sub(startTime).Seconds()
	speed := CalculateReadWriteSpeed(startTime, endTime)
	fmt.Printf("Write speed: %.2f MB/s\n", speed/1_000_000)
	fmt.Printf("Write time: %d ns\n", endTime.Sub(startTime).Nanoseconds())
	fmt.Printf("Write FPS: %.2f\n", 1/duration)

	// Read operation
	startTime = time.Now()
	err = ReadBytesFromFile("test.bin", &data)
	if err != nil {
		log.Fatal(err)
	}
	endTime = time.Now()
	duration = endTime.Sub(startTime).Seconds()
	speed = CalculateReadWriteSpeed(startTime, endTime)
	fmt.Printf("Read speed: %.2f MB/s\n", speed/1_000_000)
	fmt.Printf("Read time: %d ns\n", endTime.Sub(startTime).Nanoseconds())
	fmt.Printf("Read FPS: %.2f\n", 1/duration)
}
