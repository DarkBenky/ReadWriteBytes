# BVH Binary File Format Specification

## File Header Structure (16 bytes)

| Field | Type | Size | Description |
|---|---|---|---|
| Magic Number | `char[4]` | 4 bytes | Identifier for the file format, should be "BVH1". |
| Node Count | `uint32` | 4 bytes | Total number of nodes in the BVH data section. |
| Triangle Count | `uint32` | 4 bytes | Total number of triangles referenced by the BVH. |
| Root Node Index | `uint32` | 4 bytes | Index of the root node in the node array (typically 0). |

## BVH Node Structure (36 bytes)

This structure is repeated for each node specified by `Node Count` in the header.

| Field | Type | Size | Description |
|---|---|---|---|
| Bounding Box | `float32[6]` | 24 bytes | Min X, Y, Z and Max X, Y, Z coordinates. |
| Left Child Index | `int32` | 4 bytes | Index of the left child node. -1 for a leaf node. |
| Right Child Index | `int32` | 4 bytes | Index of the right child node. -1 for a leaf node. |
| Triangle Index | `int32` | 4 bytes | Index of the triangle if it's a leaf node. -1 for an internal node. |

## Triangle Index Structure (4 bytes)

This structure is repeated for each triangle specified by `Triangle Count` in the header.

| Field | Type | Size | Description |
|---|---|---|---|
| Original Index | `int32` | 4 bytes | The original index of the triangle from the source file. |

## File Layout

```
[File Header: 16 bytes]
[BVH Node 1 Data: 36 bytes]
[BVH Node 2 Data: 36 bytes]
...
[BVH Node N Data: 36 bytes]
[Triangle Index 1 Data: 4 bytes]
[Triangle Index 2 Data: 4 bytes]
...
[Triangle Index M Data: 4 bytes]
```

## Notes

- All integer and float values are stored in little-endian format.
- The total file size is the size of the header (16 bytes) + (`Node Count` * 36 bytes) + (`Triangle Count`