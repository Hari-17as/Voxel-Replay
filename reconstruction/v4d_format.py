"""Version 1 V4D binary format.

Header:
    magic        4 bytes   ASCII V4D1
    version      uint16
    grid_x       uint16
    grid_y       uint16
    grid_z       uint16
    frame_count  uint32
    fps          float32
    voxel_size   float32

Each frame:
    timestamp    float32
    voxel_count  uint32
    voxel_count records of:
        x,y,z,r,g,b = 6 unsigned bytes

This is intentionally simple and random-access-friendly at the frame level.
A later V4D version can add an explicit frame index and keyframe/delta blocks.
"""

from dataclasses import dataclass
import struct
from pathlib import Path

MAGIC = b"V4D1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHHHIff")
FRAME_HEADER_STRUCT = struct.Struct("<fI")
VOXEL_STRUCT = struct.Struct("<6B")


@dataclass
class V4DHeader:
    grid_x: int
    grid_y: int
    grid_z: int
    frame_count: int
    fps: float
    voxel_size: float


def write_v4d(path, header: V4DHeader, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        f.write(
            HEADER_STRUCT.pack(
                MAGIC,
                VERSION,
                header.grid_x,
                header.grid_y,
                header.grid_z,
                header.frame_count,
                header.fps,
                header.voxel_size,
            )
        )

        for timestamp, voxels in frames:
            f.write(FRAME_HEADER_STRUCT.pack(float(timestamp), len(voxels)))
            for voxel in voxels:
                f.write(VOXEL_STRUCT.pack(*voxel))


def read_header(path):
    with Path(path).open("rb") as f:
        raw = f.read(HEADER_STRUCT.size)

    if len(raw) != HEADER_STRUCT.size:
        raise ValueError("File is too small to contain a V4D header.")

    magic, version, gx, gy, gz, frame_count, fps, voxel_size = HEADER_STRUCT.unpack(raw)

    if magic != MAGIC:
        raise ValueError("Not a V4D1 file.")
    if version != VERSION:
        raise ValueError(f"Unsupported V4D version: {version}")

    return V4DHeader(gx, gy, gz, frame_count, fps, voxel_size)


def read_all_frames(path):
    path = Path(path)

    with path.open("rb") as f:
        header_raw = f.read(HEADER_STRUCT.size)
        magic, version, gx, gy, gz, frame_count, fps, voxel_size = HEADER_STRUCT.unpack(header_raw)

        if magic != MAGIC or version != VERSION:
            raise ValueError("Unsupported V4D file.")

        frames = []

        for _ in range(frame_count):
            raw = f.read(FRAME_HEADER_STRUCT.size)
            if len(raw) != FRAME_HEADER_STRUCT.size:
                raise ValueError("Unexpected end of file in frame header.")

            timestamp, voxel_count = FRAME_HEADER_STRUCT.unpack(raw)

            voxels = []
            for _ in range(voxel_count):
                raw = f.read(VOXEL_STRUCT.size)
                if len(raw) != VOXEL_STRUCT.size:
                    raise ValueError("Unexpected end of file in voxel data.")
                voxels.append(VOXEL_STRUCT.unpack(raw))

            frames.append((timestamp, voxels))

    return V4DHeader(gx, gy, gz, frame_count, fps, voxel_size), frames
