# V4D Version 1 specification

The initial prototype is deliberately simple.

## Header

| Field | Type | Meaning |
|---|---|---|
| magic | 4 bytes | `V4D1` |
| version | uint16 | format version |
| grid_x | uint16 | voxel grid width |
| grid_y | uint16 | voxel grid height |
| grid_z | uint16 | voxel grid depth |
| frame_count | uint32 | number of frames |
| fps | float32 | nominal frame rate |
| voxel_size | float32 | world-space voxel size |

## Frame

Each frame stores:

| Field | Type |
|---|---|
| timestamp | float32 |
| voxel_count | uint32 |
| voxel records | `voxel_count × 6 bytes` |

Each voxel record is:

`x, y, z, r, g, b`

where each component is an unsigned byte.

## Future V4D versions

Possible additions:
- explicit frame index table
- byte offsets for random access
- keyframes
- delta frames
- optional compression blocks
- representation type (voxel/gaussian/hybrid)
- metadata for coordinate conventions and bounds
