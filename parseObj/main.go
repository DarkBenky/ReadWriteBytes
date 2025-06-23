package main

type Vertex struct {
	X, Y, Z float32
}

type Triangle struct {
	Vertex1, Vertex2, Vertex3 Vertex
	AdditionalFields          []float32
}

type FileObject struct {
	// File Header
	FileSize   uint32
	HeaderSize uint32

	// Triangle Metadata (part of header)
	AdditionalFieldsCount uint32
	TriangleStructSize    uint32
	FieldNames            []string // Will be null-terminated when written to file

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

func main() {

}
