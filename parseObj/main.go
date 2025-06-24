package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"unsafe"
)

type Vertex struct {
	X, Y, Z float32
}

type Triangle struct {
	Vertex1, Vertex2, Vertex3, Normal Vertex
	AdditionalFields                  []float32
}

type FileObject struct {
	// File Header
	FileSize             uint32
	HeaderSize           uint32
	TriangleMetadataSize uint32

	// Triangle Metadata (part of header)
	AdditionalFieldsCount uint32
	TriangleStructSize    uint32
	FieldNames            []byte // Will be null-terminated when written to file

	// Triangle Data
	Triangles []Triangle
}

func cross(a, b, c Vertex) float32 {
	return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X)
}

func pointInTriangle(a, b, c, p Vertex) bool {
	d1 := cross(p, a, b)
	d2 := cross(p, b, c)
	d3 := cross(p, c, a)

	hasNeg := (d1 < 0) || (d2 < 0) || (d3 < 0)
	hasPos := (d1 > 0) || (d2 > 0) || (d3 > 0)

	return !(hasNeg && hasPos)
}

func isEar(vertices []Vertex, prev, curr, next int) bool {
	n := len(vertices)
	a := vertices[prev%n]
	b := vertices[curr%n]
	c := vertices[next%n]

	if cross(a, b, c) <= 0 {
		return false
	}

	for i := range n {
		if i == prev || i == curr || i == next {
			continue
		}
		if pointInTriangle(a, b, c, vertices[i]) {
			return false
		}
	}
	return true
}

func polygonArea(vertices []Vertex) float32 {
	n := len(vertices)
	if n < 3 {
		return 0
	}
	area := float32(0)
	for i := range n {
		j := (i + 1) % n
		area += vertices[i].X * vertices[j].Y
		area -= vertices[j].X * vertices[i].Y
	}
	return area / 2.0
}

func ensureCounterClockwise(vertices []Vertex) []Vertex {
	if polygonArea(vertices) < 0 {
		result := make([]Vertex, len(vertices))
		for i := range vertices {
			result[i] = vertices[len(vertices)-1-i]
		}
		return result
	}
	return vertices
}

func Normalize(v Vertex) Vertex {
	length := v.X*v.X + v.Y*v.Y + v.Z*v.Z
	if length == 0 {
		return Vertex{0, 0, 0}
	}
	invLength := 1.0 / float32(length)
	return Vertex{v.X * invLength, v.Y * invLength, v.Z * invLength}
}

func Triangulate(v []Vertex) []Triangle {
	if len(v) < 3 {
		return nil
	}
	if len(v) == 3 {
		return []Triangle{{Vertex1: v[0], Vertex2: v[1], Vertex3: v[2]}}
	}

	vertices := ensureCounterClockwise(v)
	n := len(vertices)

	indices := make([]int, n)
	for i := range n {
		indices[i] = i
	}

	var triangles []Triangle

	for len(indices) > 3 {
		earFound := false

		for i := 0; i < len(indices); i++ {
			prev := (i - 1 + len(indices)) % len(indices)
			curr := i
			next := (i + 1) % len(indices)

			if isEar(vertices, indices[prev], indices[curr], indices[next]) {
				triangle := Triangle{
					Vertex1: vertices[indices[prev]],
					Vertex2: vertices[indices[curr]],
					Vertex3: vertices[indices[next]],
					Normal: Normalize(Vertex{
						X: vertices[indices[curr]].Y - vertices[indices[prev]].Y,
						Y: vertices[indices[curr]].X - vertices[indices[prev]].X,
						Z: vertices[indices[curr]].Z - vertices[indices[prev]].Z,
					}),
				}
				triangles = append(triangles, triangle)

				newIndices := make([]int, len(indices)-1)
				copy(newIndices[:curr], indices[:curr])
				copy(newIndices[curr:], indices[curr+1:])
				indices = newIndices

				earFound = true
				break
			}
		}

		if !earFound {
			break
		}
	}

	if len(indices) == 3 {
		triangle := Triangle{
			Vertex1: vertices[indices[0]],
			Vertex2: vertices[indices[1]],
			Vertex3: vertices[indices[2]],
		}
		triangles = append(triangles, triangle)
	}

	return triangles
}

func convertStringToCString(s []string) []byte {
	size := 0
	for _, str := range s {
		size += len(str) + 1 // +1 for null terminator
	}
	result := make([]byte, size)
	for i, str := range s {
		copy(result[i*len(str):], str)
		result[i*len(str)+len(str)] = 0 // null terminator
	}
	return result
}

func parseObjFile(filename string, additionFieldsNames []string) (*FileObject, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var vertices []Vertex
	var faces [][]int
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		switch parts[0] {
		case "v":
			if len(parts) >= 4 {
				x, err1 := strconv.ParseFloat(parts[1], 32)
				y, err2 := strconv.ParseFloat(parts[2], 32)
				z, err3 := strconv.ParseFloat(parts[3], 32)
				if err1 == nil && err2 == nil && err3 == nil {
					vertices = append(vertices, Vertex{
						X: float32(x),
						Y: float32(y),
						Z: float32(z),
					})
				}
			}
		case "f":
			if len(parts) >= 4 {
				var faceIndices []int
				for i := 1; i < len(parts); i++ {
					indexStr := strings.Split(parts[i], "/")[0]
					index, err := strconv.Atoi(indexStr)
					if err == nil {
						faceIndices = append(faceIndices, index-1)
					}
				}
				if len(faceIndices) >= 3 {
					faces = append(faces, faceIndices)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	var allTriangles []Triangle
	for _, face := range faces {
		if len(face) == 3 {
			if face[0] >= 0 && face[0] < len(vertices) &&
				face[1] >= 0 && face[1] < len(vertices) &&
				face[2] >= 0 && face[2] < len(vertices) {
				triangle := Triangle{
					Vertex1: vertices[face[0]],
					Vertex2: vertices[face[1]],
					Vertex3: vertices[face[2]],
				}
				allTriangles = append(allTriangles, triangle)
			}
		} else if len(face) > 3 {
			faceVertices := make([]Vertex, len(face))
			valid := true
			for i, idx := range face {
				if idx >= 0 && idx < len(vertices) {
					faceVertices[i] = vertices[idx]
				} else {
					valid = false
					break
				}
			}
			if valid {
				triangles := Triangulate(faceVertices)
				allTriangles = append(allTriangles, triangles...)
			}
		}
	}
	fileObj := &FileObject{}
	if additionFieldsNames == nil {
		fileObj = &FileObject{
			Triangles:             allTriangles,
			AdditionalFieldsCount: uint32(0),
		}
	} else {
		cStrings := convertStringToCString(additionFieldsNames)
		fileObj = &FileObject{
			Triangles:             allTriangles,
			AdditionalFieldsCount: uint32(len(additionFieldsNames)),
			FieldNames:            cStrings,
		}
	}
	return fileObj, nil
}

func uint32ToBytes(value uint32) []byte {
	return []byte{
		byte(value & 0xFF),
		byte((value >> 8) & 0xFF),
		byte((value >> 16) & 0xFF),
		byte((value >> 24) & 0xFF),
	}
}

func bytesToUint32(b []byte) uint32 {
	if len(b) < 4 {
		return 0
	}
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

func float32ToBytes(value float32) []byte {
	bits := uint32(*(*uint32)(unsafe.Pointer(&value)))
	return []byte{
		byte(bits & 0xFF),
		byte((bits >> 8) & 0xFF),
		byte((bits >> 16) & 0xFF),
		byte((bits >> 24) & 0xFF),
	}
}

func bytesToFloat32(b []byte) float32 {
	if len(b) < 4 {
		return 0.0
	}
	bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
	return *(*float32)(unsafe.Pointer(&bits))
}

func writeFile(filename string, obj *FileObject) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	w := bufio.NewWriter(file)

	triangleStructSize := uint32(48 + int(obj.AdditionalFieldsCount)*4)
	triangleMetadataSize := uint32(4 + len(obj.FieldNames) + 4)
	headerSize := uint32(4) + triangleMetadataSize
	fileSize := uint32(4+4) + headerSize + uint32(len(obj.Triangles))*triangleStructSize

	w.Write(uint32ToBytes(fileSize))             // File Size
	w.Write(uint32ToBytes(headerSize))           // Header Size
	w.Write(uint32ToBytes(triangleMetadataSize)) // Triangle Metadata Size

	w.Write(uint32ToBytes(obj.AdditionalFieldsCount)) // Additional Fields Count
	w.Write(obj.FieldNames)                           // Field Names
	w.Write(uint32ToBytes(triangleStructSize))        // Triangle Struct Size

	for _, tri := range obj.Triangles {
		// vertices + normal (48 bytes)
		w.Write(float32ToBytes(tri.Vertex1.X))
		w.Write(float32ToBytes(tri.Vertex1.Y))
		w.Write(float32ToBytes(tri.Vertex1.Z))
		w.Write(float32ToBytes(tri.Vertex2.X))
		w.Write(float32ToBytes(tri.Vertex2.Y))
		w.Write(float32ToBytes(tri.Vertex2.Z))
		w.Write(float32ToBytes(tri.Vertex3.X))
		w.Write(float32ToBytes(tri.Vertex3.Y))
		w.Write(float32ToBytes(tri.Vertex3.Z))
		// Normal
		w.Write(float32ToBytes(tri.Normal.X))
		w.Write(float32ToBytes(tri.Normal.Y))
		w.Write(float32ToBytes(tri.Normal.Z))
		// Additional fields
		for _, f := range tri.AdditionalFields {
			w.Write(float32ToBytes(f))
		}
	}

	return w.Flush()
}

func getFileSize(filename string) (uint32, error) {
	file, err := os.Stat(filename)
	if err != nil {
		return 0, err
	}
	return uint32(file.Size()), nil
}

func getEncodedFileSize(filename string) (uint32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	// get first 4 bytes
	var sizeBytes [4]byte
	_, err = file.Read(sizeBytes[:])
	if err != nil {
		return 0, err
	}
	return bytesToUint32(sizeBytes[:]), nil
}

func main() {
	obj, err := parseObjFile("test.obj", nil)
	if err != nil {
		panic(err)
	}
	err = writeFile("output.bin", obj)
	if err != nil {
		panic(err)
	}

	realFileSize, err := getFileSize("output.bin")
	if err != nil {
		panic(err)
	}
	encodedFileSize, err := getEncodedFileSize("output.bin")
	if err != nil {
		panic(err)
	}
	fmt.Println("Encoded file size:", encodedFileSize)
	fmt.Println("Real file size:", realFileSize)

	println("OBJ file parsed and written successfully.")
	println("File size:", obj.FileSize)
	println("Header size:", obj.HeaderSize)
	println("Number of triangles:", len(obj.Triangles))
}
