"""Reconstruct the synthetic multi-camera capture into V4D.

Uses cam0/cam1/cam2 for reconstruction and leaves 'heldout' untouched.
Run from reconstruction:
    python reconstruct_capture.py
"""
from pathlib import Path
import json
import cv2
import numpy as np

from silhouette_carving import load_cameras, carve_visual_hull
from v4d_format import V4DHeader, write_v4d

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "data" / "capture"
OUT = ROOT / "exports" / "reconstructed.v4d"

def voxel_rgb(x, y, z):
    # Stable synthetic color for the reconstruction baseline.
    if x < 0:
        return (235, 90, 110)
    return (90, 160, 235)

def main():
    meta = json.loads((CAP / "capture.json").read_text(encoding="utf-8"))
    all_cameras = load_cameras(CAP / "cameras.json")
    use_names = ["cam0", "cam1", "cam2"]
    cameras = {name: all_cameras[name] for name in use_names}

    bounds_min = np.asarray(meta["bounds_min"], dtype=np.float64)
    bounds_max = np.asarray(meta["bounds_max"], dtype=np.float64)
    grid_shape = (64, 44, 48)

    frames = []
    for f in range(meta["frames"]):
        masks = {}
        for name in use_names:
            path = CAP / "masks" / name / f"{f:04d}.png"
            masks[name] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        points = carve_visual_hull(
            masks, cameras, bounds_min, bounds_max, grid_shape
        )

        # Quantize world points into 0..255 grid coordinates for V4D v1.
        normalized = (points - bounds_min) / (bounds_max - bounds_min)
        coords = np.rint(normalized * (np.asarray(grid_shape) - 1)).astype(np.int32)
        coords = np.clip(coords, 0, np.asarray(grid_shape) - 1)

        voxels = []
        seen = set()
        for x, y, z in coords:
            key = (int(x), int(y), int(z))
            if key in seen:
                continue
            seen.add(key)
            r, g, b = voxel_rgb(
                bounds_min[0] + x / (grid_shape[0]-1) * (bounds_max[0]-bounds_min[0]),
                bounds_min[1] + y / (grid_shape[1]-1) * (bounds_max[1]-bounds_min[1]),
                bounds_min[2] + z / (grid_shape[2]-1) * (bounds_max[2]-bounds_min[2]),
            )
            voxels.append((int(x), int(y), int(z), r, g, b))

        frames.append((f / meta["fps"], voxels))
        print(f"frame {f+1:02d}/{meta['frames']}: {len(voxels):5d} voxels")

    header = V4DHeader(
        grid_x=grid_shape[0],
        grid_y=grid_shape[1],
        grid_z=grid_shape[2],
        frame_count=meta["frames"],
        fps=float(meta["fps"]),
        voxel_size=float(np.mean((bounds_max - bounds_min) / (np.asarray(grid_shape)-1))),
    )
    write_v4d(OUT, header, frames)
    print()
    print("RECONSTRUCTION COMPLETE")
    print(f"V4D: {OUT}")
    print(f"Used cameras: {', '.join(use_names)}")
    print("Held-out camera was excluded from reconstruction.")

if __name__ == "__main__":
    main()
