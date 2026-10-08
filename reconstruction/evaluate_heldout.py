"""Held-out camera evaluation for the synthetic capture.

For each frame, projects reconstructed voxel centers into the held-out
camera and compares the predicted silhouette against the real held-out mask.
"""
from pathlib import Path
import json
import cv2
import numpy as np

from silhouette_carving import load_cameras, project_points
from v4d_format import read_all_frames

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "data" / "capture"
V4D = ROOT / "exports" / "reconstructed.v4d"

def main():
    meta = json.loads((CAP / "capture.json").read_text(encoding="utf-8"))
    cameras = load_cameras(CAP / "cameras.json")
    cam = cameras["heldout"]
    header, frames = read_all_frames(V4D)

    scores = []
    for i, (_, voxels) in enumerate(frames):
        if not voxels:
            scores.append(0.0)
            continue
        pts = np.array([
            [
                meta["bounds_min"][0] + v[0]/(header.grid_x-1)*(meta["bounds_max"][0]-meta["bounds_min"][0]),
                meta["bounds_min"][1] + v[1]/(header.grid_y-1)*(meta["bounds_max"][1]-meta["bounds_min"][1]),
                meta["bounds_min"][2] + v[2]/(header.grid_z-1)*(meta["bounds_max"][2]-meta["bounds_min"][2]),
            ]
            for v in voxels
        ], dtype=np.float64)
        uv, depth = project_points(pts, cam)
        pred = np.zeros((cam.image_height, cam.image_width), dtype=np.uint8)
        good = (
            (depth > 0)
            & (uv[:,0] >= 0) & (uv[:,0] < cam.image_width)
            & (uv[:,1] >= 0) & (uv[:,1] < cam.image_height)
        )
        xy = np.rint(uv[good]).astype(int)
        pred[xy[:,1], xy[:,0]] = 255

        # Dilate projected voxel points into a silhouette-like footprint.
        pred = cv2.dilate(pred, np.ones((9,9), np.uint8), iterations=1)
        real = cv2.imread(str(CAP / "masks" / "heldout" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        real = (real > 0).astype(np.uint8)
        p = (pred > 0).astype(np.uint8)
        inter = np.logical_and(p, real).sum()
        union = np.logical_or(p, real).sum()
        iou = float(inter / union) if union else 1.0
        scores.append(iou)
        print(f"frame {i+1:02d}/{len(frames)}: held-out silhouette IoU = {iou:.4f}")

    print()
    print(f"Mean held-out silhouette IoU: {np.mean(scores):.4f}")

if __name__ == "__main__":
    main()
