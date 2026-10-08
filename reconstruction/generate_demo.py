"""Generate a tiny animated synthetic Voxel Replay scene."""

from pathlib import Path
import sys

# Allow running this file directly from reconstruction/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4d_format import V4DHeader, write_v4d

GRID_X = 32
GRID_Y = 24
GRID_Z = 24
FRAME_COUNT = 60
FPS = 15.0
VOXEL_SIZE = 0.1


def create_frame(frame_index):
    """Create a moving colored cube plus a small second object."""
    voxels = []

    # Move the main cube left-to-right.
    center_x = 5 + int((GRID_X - 10) * frame_index / (FRAME_COUNT - 1))
    center_y = 12
    center_z = 12

    for x in range(center_x - 2, center_x + 3):
        for y in range(center_y - 2, center_y + 3):
            for z in range(center_z - 2, center_z + 3):
                if 0 <= x < GRID_X and 0 <= y < GRID_Y and 0 <= z < GRID_Z:
                    # Red main cube.
                    voxels.append((x, y, z, 240, 70, 70))

    # Small blue cube moving vertically.
    blue_x = 24
    blue_y = 5 + int(8 * (0.5 + 0.5 * __import__("math").sin(frame_index * 0.2)))
    blue_z = 7

    for x in range(blue_x - 1, blue_x + 2):
        for y in range(blue_y - 1, blue_y + 2):
            for z in range(blue_z - 1, blue_z + 2):
                if 0 <= x < GRID_X and 0 <= y < GRID_Y and 0 <= z < GRID_Z:
                    voxels.append((x, y, z, 70, 150, 245))

    return voxels


def main():
    project_root = Path(__file__).resolve().parent.parent
    output = project_root / "exports" / "demo.v4d"

    header = V4DHeader(
        grid_x=GRID_X,
        grid_y=GRID_Y,
        grid_z=GRID_Z,
        frame_count=FRAME_COUNT,
        fps=FPS,
        voxel_size=VOXEL_SIZE,
    )

    frames = []
    for frame_index in range(FRAME_COUNT):
        timestamp = frame_index / FPS
        frames.append((timestamp, create_frame(frame_index)))

    write_v4d(output, header, frames)

    print("Voxel Replay synthetic demo")
    print("---------------------------")
    print(f"Created: {output}")
    print(f"Grid: {GRID_X} x {GRID_Y} x {GRID_Z}")
    print(f"Frames: {FRAME_COUNT}")
    print(f"FPS: {FPS}")
    print(f"Voxel size: {VOXEL_SIZE}")


if __name__ == "__main__":
    main()
