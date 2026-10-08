"""Generate a small offline multi-camera dynamic capture for Voxel Replay.

This is a synthetic stand-in for real synchronized cameras. It creates:
- 4 calibrated cameras
- RGB frames
- binary foreground masks
- one held-out camera

Run from reconstruction:
    python build_capture.py
"""
from pathlib import Path
import json
import math
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "capture"

W, H = 320, 240
FPS = 15
FRAMES = 30
CAMERAS = ["cam0", "cam1", "cam2", "heldout"]

# World bounds used by reconstruction.
BOUNDS_MIN = np.array([-1.6, -1.1, -0.4], dtype=np.float32)
BOUNDS_MAX = np.array([ 1.6,  1.1,  1.8], dtype=np.float32)

def look_at(position, target=np.array([0.0, 0.0, 0.6], dtype=np.float64)):
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Camera coordinates: x=right, y=up, z=forward
    R = np.vstack([right, up, forward])
    t = -R @ position.reshape(3, 1)
    return R, t

def project(points, K, R, t):
    pts = (R @ points.T) + t
    z = pts[2]
    good = z > 1e-6
    u = K[0,0] * pts[0] / np.maximum(z, 1e-6) + K[0,2]
    v = K[1,1] * pts[1] / np.maximum(z, 1e-6) + K[1,2]
    return np.column_stack([u, v]), z, good

def cube_vertices(center, size):
    cx, cy, cz = center
    sx, sy, sz = np.asarray(size) / 2.0
    return np.array([
        [cx+dx, cy+dy, cz+dz]
        for dx in (-sx, sx)
        for dy in (-sy, sy)
        for dz in (-sz, sz)
    ], dtype=np.float64)

def draw_object(img, mask, center, size, color, K, R, t):
    verts = cube_vertices(center, size)
    uv, depth, good = project(verts, K, R, t)
    if not np.all(good):
        return
    pts = np.rint(uv).astype(np.int32)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if x1 < 0 or y1 < 0 or x0 >= W or y0 >= H:
        return
    x0, x1 = max(0, x0), min(W-1, x1)
    y0, y1 = max(0, y0), min(H-1, y1)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rgb").mkdir(exist_ok=True)
    (OUT / "masks").mkdir(exist_ok=True)

    fx = fy = 280.0
    K = [[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]]

    positions = {
        "cam0": [-3.6, -2.8, 2.2],
        "cam1": [ 3.6, -2.8, 2.2],
        "cam2": [ 0.0,  3.8, 2.5],
        "heldout": [0.0, -4.2, 2.0],
    }

    cameras = {}
    for name, pos in positions.items():
        R, t = look_at(pos)
        cameras[name] = {
            "K": K,
            "R": R.tolist(),
            "t": t.reshape(-1).tolist(),
            "image_width": W,
            "image_height": H,
        }

    (OUT / "cameras.json").write_text(
        json.dumps({"cameras": cameras}, indent=2), encoding="utf-8"
    )

    for cam in CAMERAS:
        (OUT / "rgb" / cam).mkdir(exist_ok=True)
        (OUT / "masks" / cam).mkdir(exist_ok=True)

    for f in range(FRAMES):
        t = f / FPS
        # Two moving colored cubes make the scene dynamic.
        c1 = (-0.85 + 1.7 * (f/(FRAMES-1)), -0.25, 0.65 + 0.12*math.sin(t*2))
        c2 = (0.55, 0.35*math.sin(t*2.2), 0.85 + 0.35*math.cos(t*1.6))
        objects = [
            (c1, (0.55, 0.55, 0.75), (40, 80, 220)),
            (c2, (0.45, 0.45, 0.55), (220, 100, 40)),
        ]
        for cam in CAMERAS:
            info = cameras[cam]
            K_np = np.asarray(info["K"], dtype=np.float64)
            R = np.asarray(info["R"], dtype=np.float64)
            tt = np.asarray(info["t"], dtype=np.float64).reshape(3,1)
            img = np.full((H, W, 3), 18, dtype=np.uint8)
            mask = np.zeros((H, W), dtype=np.uint8)
            for center, size, color in objects:
                draw_object(img, mask, center, size, color, K_np, R, tt)
            cv2.imwrite(str(OUT / "rgb" / cam / f"{f:04d}.png"), img)
            cv2.imwrite(str(OUT / "masks" / cam / f"{f:04d}.png"), mask)

    meta = {
        "frames": FRAMES, "fps": FPS, "width": W, "height": H,
        "cameras": CAMERAS, "heldout_camera": "heldout",
        "bounds_min": BOUNDS_MIN.tolist(), "bounds_max": BOUNDS_MAX.tolist(),
    }
    (OUT / "capture.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Synthetic multi-camera capture created.")
    print(f"Frames: {FRAMES}, FPS: {FPS}, cameras: {', '.join(CAMERAS)}")
    print(f"Held-out camera: heldout")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()
