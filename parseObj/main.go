package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"unsafe"
)

type Vertex struct {
	X, Y, Z float32
}

type Triangle struct {
	Vertex1, Vertex2, Vertex3, Normal Vertex
	Roughness                         float32
	Metallic                          float32
	Emission                          float32
	Color                             [3]float32 // RGB color
	index                             int32      // Index of the triangle in the original OBJ file
}

type FileObject struct {
	// File Header
	FileSize           uint32
	TriangleStructSize uint32
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
	triangleIndex := int32(0)

	for _, face := range faces {
		if len(face) == 3 {
			// Simple triangle case
			if face[0] >= 0 && face[0] < len(vertices) &&
				face[1] >= 0 && face[1] < len(vertices) &&
				face[2] >= 0 && face[2] < len(vertices) {
				triangle := Triangle{
					Vertex1: vertices[face[0]],
					Vertex2: vertices[face[1]],
					Vertex3: vertices[face[2]],
					index:   triangleIndex, // Correct assignment
				}
				allTriangles = append(allTriangles, triangle)
				triangleIndex++ // Increment after each triangle
			}
		} else if len(face) > 3 {
			// Polygon case - triangulate
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
				for _, tri := range triangles {
					triangle := Triangle{
						Vertex1: tri.Vertex1,
						Vertex2: tri.Vertex2,
						Vertex3: tri.Vertex3,
						Normal:  tri.Normal,
						index:   triangleIndex, // Correct assignment
					}
					allTriangles = append(allTriangles, triangle)
					triangleIndex++
				}
			}
		}
	}
	fileObj := &FileObject{}
	if additionFieldsNames == nil {
		fileObj = &FileObject{
			Triangles: allTriangles,
		}
	} else {
		fileObj = &FileObject{
			Triangles: allTriangles,
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

func writeFile(filename string, obj *FileObject) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	w := bufio.NewWriter(file)

	triangleStructSize := uint32(unsafe.Sizeof(Triangle{}))
	fileSize := uint32(4+4) + uint32(len(obj.Triangles))*triangleStructSize

	w.Write(uint32ToBytes(fileSize))           // File Size
	w.Write(uint32ToBytes(triangleStructSize)) // Triangle Struct Size

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
		w.Write(float32ToBytes(tri.Roughness))
		w.Write(float32ToBytes(tri.Metallic))
		w.Write(float32ToBytes(tri.Emission))
		w.Write(float32ToBytes(tri.Color[0]))
		w.Write(float32ToBytes(tri.Color[1]))
		w.Write(float32ToBytes(tri.Color[2]))
		// Triangle index
		w.Write(uint32ToBytes(uint32(tri.index)))
	}

	return w.Flush()
}

func readFile(filename string) (*FileObject, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var fileObj FileObject
	header := make([]byte, 8) // 4 bytes for file size + 4
	headerSize := 8
	if _, err := file.Read(header); err != nil {
		return nil, err
	}
	fileObj.FileSize = bytesToUint32(header[:4])
	fileObj.TriangleStructSize = bytesToUint32(header[4:8])
	numberOfTriangles := (fileObj.FileSize - uint32(headerSize)) / fileObj.TriangleStructSize
	println("Number of triangles:", numberOfTriangles)
	fileObj.Triangles = make([]Triangle, numberOfTriangles)
	triangleSize := int(fileObj.TriangleStructSize)
	for i := 0; i < int(numberOfTriangles); i++ {
		triangleData := make([]byte, triangleSize)
		if _, err := file.Read(triangleData); err != nil {
			return nil, err
		}
		tri := &fileObj.Triangles[i]
		tri.Vertex1.X = *(*float32)(unsafe.Pointer(&triangleData[0]))
		tri.Vertex1.Y = *(*float32)(unsafe.Pointer(&triangleData[4]))
		tri.Vertex1.Z = *(*float32)(unsafe.Pointer(&triangleData[8]))
		tri.Vertex2.X = *(*float32)(unsafe.Pointer(&triangleData[12]))
		tri.Vertex2.Y = *(*float32)(unsafe.Pointer(&triangleData[16]))
		tri.Vertex2.Z = *(*float32)(unsafe.Pointer(&triangleData[20]))
		tri.Vertex3.X = *(*float32)(unsafe.Pointer(&triangleData[24]))
		tri.Vertex3.Y = *(*float32)(unsafe.Pointer(&triangleData[28]))
		tri.Vertex3.Z = *(*float32)(unsafe.Pointer(&triangleData[32]))
		tri.Normal.X = *(*float32)(unsafe.Pointer(&triangleData[36]))
		tri.Normal.Y = *(*float32)(unsafe.Pointer(&triangleData[40]))
		tri.Normal.Z = *(*float32)(unsafe.Pointer(&triangleData[44]))
		tri.Roughness = *(*float32)(unsafe.Pointer(&triangleData[48]))
		tri.Metallic = *(*float32)(unsafe.Pointer(&triangleData[52]))
		tri.Emission = *(*float32)(unsafe.Pointer(&triangleData[56]))
		tri.Color[0] = *(*float32)(unsafe.Pointer(&triangleData[60]))
		tri.Color[1] = *(*float32)(unsafe.Pointer(&triangleData[64]))
		tri.Color[2] = *(*float32)(unsafe.Pointer(&triangleData[68]))
		tri.index = int32(bytesToUint32(triangleData[72:76]))
	}
	return &fileObj, nil
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

type BVHNode struct {
	BoundingBox   [6]float32 // minX, minY, minZ, maxX, maxY, maxZ
	LeftIndex     int32      // Index of left child in linearized array, -1 if leaf
	RightIndex    int32      // Index of right child in linearized array, -1 if leaf
	TriangleIndex int32      // -1 if not a leaf node, otherwise the index of the triangle
}

func CalculateBoundingBox(triangles []Triangle) [6]float32 {
	if len(triangles) == 0 {
		return [6]float32{0, 0, 0, 0, 0, 0}
	}

	minX, minY, minZ := triangles[0].Vertex1.X, triangles[0].Vertex1.Y, triangles[0].Vertex1.Z
	maxX, maxY, maxZ := triangles[0].Vertex1.X, triangles[0].Vertex1.Y, triangles[0].Vertex1.Z

	for _, tri := range triangles {
		vertices := []Vertex{tri.Vertex1, tri.Vertex2, tri.Vertex3}
		for _, v := range vertices {
			minX = min(minX, v.X)
			minY = min(minY, v.Y)
			minZ = min(minZ, v.Z)
			maxX = max(maxX, v.X)
			maxY = max(maxY, v.Y)
			maxZ = max(maxZ, v.Z)
		}
	}

	return [6]float32{minX, minY, minZ, maxX, maxY, maxZ}
}

func CalculateSAH(triangles []Triangle, bbox [6]float32) float32 {
	if len(triangles) == 0 {
		return 0
	}

	width := bbox[3] - bbox[0]  // maxX - minX
	height := bbox[4] - bbox[1] // maxY - minY
	depth := bbox[5] - bbox[2]  // maxZ - minZ

	// Surface area * number of triangles gives the SAH cost
	area := 2 * (width*height + width*depth + height*depth)
	return area * float32(len(triangles))
}

type BVHBuildNode struct {
	Left, Right   *BVHBuildNode
	BoundingBox   [6]float32
	Triangles     []Triangle
	TriangleIndex int32 // Only used for leaf nodes
	IsLeaf        bool
}

func BuildBVHRecursive(triangles []Triangle) *BVHBuildNode {
	if len(triangles) == 0 {
		return nil
	}

	node := &BVHBuildNode{
		BoundingBox: CalculateBoundingBox(triangles),
		Triangles:   triangles,
	}

	// Leaf case - single triangle
	if len(triangles) == 1 {
		node.IsLeaf = true
		node.TriangleIndex = triangles[0].index

		// Ensure normal is calculated if it isn't already
		if triangles[0].Normal.X == 0 && triangles[0].Normal.Y == 0 && triangles[0].Normal.Z == 0 {
			v1, v2, v3 := triangles[0].Vertex1, triangles[0].Vertex2, triangles[0].Vertex3

			// Calculate two edges
			edge1 := Vertex{v2.X - v1.X, v2.Y - v1.Y, v2.Z - v1.Z}
			edge2 := Vertex{v3.X - v1.X, v3.Y - v1.Y, v3.Z - v1.Z}

			// Cross product to get normal
			normal := Vertex{
				edge1.Y*edge2.Z - edge1.Z*edge2.Y,
				edge1.Z*edge2.X - edge1.X*edge2.Z,
				edge1.X*edge2.Y - edge1.Y*edge2.X,
			}

			triangles[0].Normal = Normalize(normal)
		}

		return node
	}

	// Find the longest axis to split along
	bbox := node.BoundingBox
	extentX := bbox[3] - bbox[0]
	extentY := bbox[4] - bbox[1]
	extentZ := bbox[5] - bbox[2]

	axis := 0 // X-axis by default
	if extentY > extentX && extentY > extentZ {
		axis = 1 // Y-axis
	} else if extentZ > extentX && extentZ > extentY {
		axis = 2 // Z-axis
	}

	// Sort triangles based on their centroids along the chosen axis
	sortedTriangles := make([]Triangle, len(triangles))
	copy(sortedTriangles, triangles)

	sort.Slice(sortedTriangles, func(i, j int) bool {
		var centroidI, centroidJ float32

		if axis == 0 { // X-axis
			centroidI = (sortedTriangles[i].Vertex1.X + sortedTriangles[i].Vertex2.X + sortedTriangles[i].Vertex3.X) / 3.0
			centroidJ = (sortedTriangles[j].Vertex1.X + sortedTriangles[j].Vertex2.X + sortedTriangles[j].Vertex3.X) / 3.0
		} else if axis == 1 { // Y-axis
			centroidI = (sortedTriangles[i].Vertex1.Y + sortedTriangles[i].Vertex2.Y + sortedTriangles[i].Vertex3.Y) / 3.0
			centroidJ = (sortedTriangles[j].Vertex1.Y + sortedTriangles[j].Vertex2.Y + sortedTriangles[j].Vertex3.Y) / 3.0
		} else { // Z-axis
			centroidI = (sortedTriangles[i].Vertex1.Z + sortedTriangles[i].Vertex2.Z + sortedTriangles[i].Vertex3.Z) / 3.0
			centroidJ = (sortedTriangles[j].Vertex1.Z + sortedTriangles[j].Vertex2.Z + sortedTriangles[j].Vertex3.Z) / 3.0
		}

		return centroidI < centroidJ
	})

	// Find best split using SAH
	bestCost := float32(math.MaxFloat32)
	bestSplit := len(sortedTriangles) / 2 // Default mid-point split

	// Try different splits and find the one with lowest SAH cost
	for i := 1; i < len(sortedTriangles); i++ {
		leftTris := sortedTriangles[:i]
		rightTris := sortedTriangles[i:]

		leftBox := CalculateBoundingBox(leftTris)
		rightBox := CalculateBoundingBox(rightTris)

		leftSAH := CalculateSAH(leftTris, leftBox)
		rightSAH := CalculateSAH(rightTris, rightBox)

		totalCost := leftSAH + rightSAH

		if totalCost < bestCost {
			bestCost = totalCost
			bestSplit = i
		}
	}

	// Create children using the best split
	leftTris := sortedTriangles[:bestSplit]
	rightTris := sortedTriangles[bestSplit:]

	if len(leftTris) == 0 || len(rightTris) == 0 {
		// SAH failed to find a good split, fall back to median
		mid := len(sortedTriangles) / 2
		leftTris = sortedTriangles[:mid]
		rightTris = sortedTriangles[mid:]
	}

	node.Left = BuildBVHRecursive(leftTris)
	node.Right = BuildBVHRecursive(rightTris)
	node.IsLeaf = false
	node.TriangleIndex = -1 // Mark as non-leaf

	return node
}

type BVHLinear struct {
	Nodes     []BVHNode
	Triangles []Triangle
}

// Linearize the BVH tree into a flat array
func (bvh *BVHLinear) BuildLinearBVH(triangles []Triangle) {
	if len(triangles) == 0 {
		return
	}

	// Build the BVH tree recursively
	root := BuildBVHRecursive(triangles)
	if root == nil {
		return
	}

	// Linearize the tree
	bvh.Nodes = make([]BVHNode, 0)
	bvh.Triangles = make([]Triangle, len(triangles))
	copy(bvh.Triangles, triangles)

	// Recursive function to flatten tree
	var flattenTree func(*BVHBuildNode) int32
	flattenTree = func(node *BVHBuildNode) int32 {
		if node == nil {
			return -1
		}

		// Current index in nodes array
		nodeIndex := int32(len(bvh.Nodes))

		// Create linear node
		linearNode := BVHNode{
			BoundingBox:   node.BoundingBox,
			LeftIndex:     -1,
			RightIndex:    -1,
			TriangleIndex: -1, // Default to internal node
		}

		if node.IsLeaf {
			linearNode.TriangleIndex = node.TriangleIndex
		}

		// Add node to array
		bvh.Nodes = append(bvh.Nodes, linearNode)

		// Process children and update indices
		if !node.IsLeaf {
			linearNode.LeftIndex = flattenTree(node.Left)
			linearNode.RightIndex = flattenTree(node.Right)
			bvh.Nodes[nodeIndex] = linearNode // Update with correct child indices
		}

		return nodeIndex
	}

	flattenTree(root)
}

// WriteBVHToFile writes the linearized BVH to a binary file
func (bvh *BVHLinear) WriteBVHToFile(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	w := bufio.NewWriter(file)

	// Write header
	// 1. Magic number for BVH file type (4 bytes)
	w.Write([]byte("BVH1")) // BVH format version 1

	// 2. Number of nodes (4 bytes)
	w.Write(uint32ToBytes(uint32(len(bvh.Nodes))))

	// 3. Number of triangles (4 bytes)
	w.Write(uint32ToBytes(uint32(len(bvh.Triangles))))

	// 4. Root node index (4 bytes) - always 0 for our implementation
	w.Write(uint32ToBytes(0))

	// Write all nodes
	for _, node := range bvh.Nodes {
		// Bounding box (24 bytes: 6 float32s)
		for i := 0; i < 6; i++ {
			w.Write(float32ToBytes(node.BoundingBox[i]))
		}

		// Child indices (8 bytes: 2 int32s)
		w.Write(uint32ToBytes(uint32(node.LeftIndex)))
		w.Write(uint32ToBytes(uint32(node.RightIndex)))

		// Triangle index (4 bytes) - negative if not a leaf
		w.Write(uint32ToBytes(uint32(node.TriangleIndex)))
	}

	// Write triangle indices
	// This is useful if the BVH needs to reference the original triangles
	for _, tri := range bvh.Triangles {
		w.Write(uint32ToBytes(uint32(tri.index)))
	}

	return w.Flush()
}

// ReadBVHFromFile reads a linearized BVH from a binary file
func ReadBVHFromFile(filename string) (*BVHLinear, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read header
	header := make([]byte, 16) // 4 bytes magic + 4 bytes node count + 4 bytes triangle count + 4 bytes root index
	if _, err := file.Read(header); err != nil {
		return nil, err
	}

	// Check magic number
	if string(header[0:4]) != "BVH1" {
		return nil, fmt.Errorf("invalid BVH file format")
	}

	nodeCount := bytesToUint32(header[4:8])
	triangleCount := bytesToUint32(header[8:12])
	// root index at header[12:16] is not used here since we always use index 0

	bvh := &BVHLinear{
		Nodes:     make([]BVHNode, nodeCount),
		Triangles: make([]Triangle, triangleCount),
	}

	// Node size: 24 bytes for bounding box + 8 bytes for child indices + 4 bytes for triangle index = 36 bytes
	nodeSize := 36

	// Read all nodes
	for i := range nodeCount {
		nodeData := make([]byte, nodeSize)
		if _, err := file.Read(nodeData); err != nil {
			return nil, err
		}

		node := &bvh.Nodes[i]

		// Read bounding box (24 bytes)
		for j := 0; j < 6; j++ {
			bits := bytesToUint32(nodeData[j*4 : j*4+4])
			node.BoundingBox[j] = *(*float32)(unsafe.Pointer(&bits))
		}

		// Read child indices (8 bytes)
		node.LeftIndex = int32(bytesToUint32(nodeData[24:28]))
		node.RightIndex = int32(bytesToUint32(nodeData[28:32]))

		// Read triangle index (4 bytes)
		node.TriangleIndex = int32(bytesToUint32(nodeData[32:36]))
	}

	// Read triangle indices
	// Note: We only read the indices here, not the actual triangles
	// The actual triangles would need to be loaded separately
	triIndices := make([]byte, triangleCount*4)
	if _, err := file.Read(triIndices); err != nil {
		return nil, err
	}

	for i := range triangleCount {
		index := bytesToUint32(triIndices[i*4 : i*4+4])
		bvh.Triangles[i].index = int32(index)
	}

	return bvh, nil
}

// Helper function to check if a node is a leaf
func isLeafNode(node BVHNode) bool {
	return node.TriangleIndex >= 0
}

func main() {
	obj, err := parseObjFile("test.obj", nil)
	if err != nil {
		panic(err)
	}

	obj, err = readFile("triangles.bin")
	if err != nil {
		panic(err)
	}

	// Build BVH
	bvhLinear := &BVHLinear{}
	bvhLinear.BuildLinearBVH(obj.Triangles)
	fmt.Printf("Built BVH with %d nodes\n", len(bvhLinear.Nodes))

	// Write BVH to binary file
	err = bvhLinear.WriteBVHToFile("output.bvh")
	if err != nil {
		panic(err)
	}
	fmt.Println("BVH written to output.bvh")

	// Original file writing
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
	println("Number of triangles:", len(obj.Triangles))
}
