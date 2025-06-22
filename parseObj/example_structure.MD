# Binary File Format Specification

## File Header Structure

| Field | Type | Size | Description |
|-------|------|------|-------------|
| File Size | `uint32` | 4 bytes | Total size of the entire file |
| Header Size | `uint32` | 4 bytes | Size of the header section |

## Triangle Metadata

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Additional Fields Count | `uint32` | 4 bytes | Number of additional fields per triangle |
| Field Names | `string[]` | Variable | Names of additional fields (null-terminated strings) |
| Triangle Struct Size | `uint32` | 4 bytes | Size in bytes of each triangle structure |

## Triangle Data Structure

Each triangle contains:

| Field | Type | Size | Description |
|-------|------|------|-------------|
| Vertex 1 | `float32[3]` | 12 bytes | X, Y, Z coordinates |
| Vertex 2 | `float32[3]` | 12 bytes | X, Y, Z coordinates |
| Vertex 3 | `float32[3]` | 12 bytes | X, Y, Z coordinates |
| Additional Fields | `float32[]` | Variable | Custom fields as defined in header |

## File Layout

```
[File Size: 4 bytes]
[Header Size: 4 bytes]
[Additional Fields Count: 4 bytes]
[Field Names: Variable length]
[Triangle Struct Size: 4 bytes]
[Triangle 1 Data: Triangle Struct Size bytes]
[Triangle 2 Data: Triangle Struct Size bytes]
...
[Triangle N Data: Triangle Struct Size bytes]
```

## Notes

- All integers are little-endian
- Strings are null-terminated UTF-8
- Coordinates use right-handed coordinate system
