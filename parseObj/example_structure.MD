# Binary File Format Specification

## File Header Structure

| Field | Type | Size | Description |
|-------|------|------|-------------|
| File Size | `uint32` | 4 bytes | Total size of the entire file |
| Triangle Struct Size | `uint32` | 4 bytes | Size in bytes of each triangle structure |

## Triangle Data Structure

Each triangle contains:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Vertex 1 | `float32[3]` | 12 bytes | X, Y, Z coordinates |
| Vertex 2 | `float32[3]` | 12 bytes | X, Y, Z coordinates |
| Vertex 3 | `float32[3]` | 12 bytes | X, Y, Z coordinates |
| Normal   | `float32[3]` | 12 bytes | Normal vector X, Y, Z |
| Roughness | `float32` | 4 bytes | Surface roughness value |
| Metallic | `float32` | 4 bytes | Metallic property value |
| Emission | `float32` | 4 bytes | Emission intensity value |
| Color | `float32[3]` | 12 bytes | RGB color values |
| Index | `int32` | 4 bytes | Triangle index in original OBJ file |

## File Layout

```text
[File Size: 4 bytes]
[Triangle Struct Size: 4 bytes]
[Triangle 1 Data: 48 bytes (vertices + normal only)]
[Triangle 2 Data: 48 bytes (vertices + normal only)]
...
[Triangle N Data: 48 bytes (vertices + normal only)]
```

## Go Struct Definition

```go
type Triangle struct {
    Vertex1, Vertex2, Vertex3, Normal Vertex  // 48 bytes (4 * 12)
    Roughness                         float32  // 4 bytes
    Metallic                          float32  // 4 bytes
    Emission                          float32  // 4 bytes
    Color                             [3]float32  // 12 bytes
    index                             int32    // 4 bytes
}

type Vertex struct {
    X, Y, Z float32  // 12 bytes total
}
```

## Notes

- All integers are little-endian
- Full triangle struct size: 68 bytes
- Actually written to file: 48 bytes per triangle (vertices + normal only)
- Coordinates use right-handed coordinate system
- Triangle index tracks original position in OBJ file
- Color is stored as RGB float32 values
- Additional material properties (roughness, metallic, emission) are included in struct but not written to file
