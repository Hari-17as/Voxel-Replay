"""Small, readable visual-hull / silhouette-carving prototype.

Inputs:
- calibrated cameras
- binary foreground masks for one synchronized time step
- bounded voxel volume

This implementation is intentionally simple and CPU-based first.
It is the correctness reference before any GPU optimization.

Camera JSON format is documented in data/calibration/cameras.example.json.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_width: int
    image_height: int


def load_cameras(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras = {}

    for name, item in data["cameras"].items():
        cameras[name] = Camera(
            K=np.asarray(item["K"], dtype=np.float64),
            R=np.asarray(item["R"], dtype=np.float64),
            t=np.asarray(item["t"], dtype=np.float64).reshape(3, 1),
            image_width=int(item["image_width"]),
            image_height=int(item["image_height"]),
        )

    return cameras


def project_points(points_xyz, camera):
    """Project Nx3 world points to Nx2 pixel coordinates."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    cam = (camera.R @ pts.T) + camera.t
    z = cam[2]

    safe_z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = camera.K[0, 0] * cam[0] / safe_z + camera.K[0, 2]
    v = camera.K[1, 1] * cam[1] / safe_z + camera.K[1, 2]

    return np.column_stack([u, v]), z


def carve_visual_hull(masks, cameras, bounds_min, bounds_max, grid_shape):
    """Return occupied voxel centers that project inside every mask.

    bounds_min/max are XYZ world coordinates.
    masks maps camera name -> uint8 mask where foreground > 0.
    """
    gx, gy, gz = grid_shape

    xs = np.linspace(bounds_min[0], bounds_max[0], gx)
    ys = np.linspace(bounds_min[1], bounds_max[1], gy)
    zs = np.linspace(bounds_min[2], bounds_max[2], gz)

    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    keep = np.ones(len(points), dtype=bool)

    for name, camera in cameras.items():
        projected, depth = project_points(points, camera)

        u = np.rint(projected[:, 0]).astype(int)
        v = np.rint(projected[:, 1]).astype(int)

        inside = (
            (depth > 0)
            & (u >= 0)
            & (u < camera.image_width)
            & (v >= 0)
            & (v < camera.image_height)
        )

        visible_foreground = np.zeros(len(points), dtype=bool)
        valid_indices = np.where(inside)[0]

        if len(valid_indices):
            mask = masks[name]
            visible_foreground[valid_indices] = mask[
                v[valid_indices], u[valid_indices]
            ] > 0

        keep &= visible_foreground

    return points[keep]


def save_points_npz(path, points):
    np.savez_compressed(path, points=np.asarray(points, dtype=np.float32))


if __name__ == "__main__":
    print("This module provides carve_visual_hull().")
    print("Use it from a reconstruction script once real calibrated masks exist.")
