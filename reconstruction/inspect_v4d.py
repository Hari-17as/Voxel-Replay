"""Print basic information about a V4D file."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4d_format import read_header, read_all_frames


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_v4d.py ..\\exports\\demo.v4d")
        raise SystemExit(1)

    path = Path(sys.argv[1])
    header, frames = read_all_frames(path)

    total_voxels = sum(len(voxels) for _, voxels in frames)

    print(f"File: {path}")
    print(f"Grid: {header.grid_x} x {header.grid_y} x {header.grid_z}")
    print(f"Frames: {header.frame_count}")
    print(f"FPS: {header.fps}")
    print(f"Voxel size: {header.voxel_size}")
    print(f"Total stored voxel records: {total_voxels}")
    print(f"File size: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
