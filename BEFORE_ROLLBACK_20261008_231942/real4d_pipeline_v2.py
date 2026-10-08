"""
VOXEL REPLAY - HIGH-FIDELITY 4D HUMAN RECONSTRUCTION PIPELINE
Generates both:
1. Watertight Marching Cubes triangle mesh (.vmesh + .ply sequence)
2. Backward-compatible voxel representation (.v4d)
"""

import sys
import os
import time
import math
import json
import struct
import shutil
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import measure

# ------------------------------------------------------------
# ENVIRONMENT & PATH CONFIGURATION
# ------------------------------------------------------------
ROOT = Path(r"C:\Users\hari1\voxel-replay")
VIDEO_ROOT = ROOT / "data" / "videos"
MASK_ROOT = ROOT / "data" / "real_masks_universal"
CALIB_FILE = ROOT / "calibration.json"

EXPORT_DIR = ROOT / "exports"
MESH_FRAME_DIR = EXPORT_DIR / "mesh_frames"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
MESH_FRAME_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_V4D = EXPORT_DIR / "reconstructed_color.v4d"
OUTPUT_VMESH = EXPORT_DIR / "reconstructed_mesh.vmesh"

PLAYER_PUBLIC = ROOT / "player" / "public"
PLAYER_OUTPUT_V4D = PLAYER_PUBLIC / "reconstructed_color.v4d"
PLAYER_OUTPUT_VMESH = PLAYER_PUBLIC / "reconstructed_mesh.vmesh"
PLAYER_PUBLIC.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# RECONSTRUCTION HYPERPARAMETERS
# ------------------------------------------------------------
CAMERAS = ["cam0", "cam1", "cam2"]
MAX_FRAMES = 90
TARGET_FPS = 15.0

# 3D bounding volume encompassing human motion envelope (in meters)
GLOBAL_BOUNDS_MIN = np.array([-1.1, -1.1, 0.0], dtype=np.float32)
GLOBAL_BOUNDS_MAX = np.array([1.1, 1.1, 2.2], dtype=np.float32)

# Two-tier visual hull grid resolutions
COARSE_RES = np.array([48, 48, 64], dtype=np.int32)
FINE_RES = np.array([96, 96, 128], dtype=np.int32)

# V4D standard grid resolution (retains original format dimensions)
GRID_X = 128
GRID_Y = 128
GRID_Z = 192

# Silhouette consensus: 3 = strict intersection; 2 = majority vote
MIN_VIEWS_CONSENSUS = 3

# YOLO model selection
YOLO_MODEL_PATH = ROOT / "yolo11n-seg.pt"
if not YOLO_MODEL_PATH.exists():
    YOLO_MODEL_PATH = ROOT / "yolov8n-seg.pt"

try:
    from ultralytics import YOLO
except ImportError as err:
    raise RuntimeError(
        "Ultralytics is not installed in current venv. "
        "Run: .venv\\Scripts\\pip.exe install ultralytics"
    ) from err

# ------------------------------------------------------------
# CAMERA CALIBRATION & GEOMETRY
# ------------------------------------------------------------
def look_at(position, target=np.array([0.0, 0.0, 1.05], dtype=np.float32)):
    pos = np.asarray(position, dtype=np.float32)
    forward = target - pos
    norm_f = np.linalg.norm(forward)
    forward = forward / (norm_f + 1e-8)

    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_world)
    norm_r = np.linalg.norm(right)
    right = right / (norm_r + 1e-8)

    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-8)

    R = np.vstack([right, up, forward])
    t = -R @ pos.reshape(3, 1)
    return R, t

def get_cameras(video_infos):
    if CALIB_FILE.exists():
        print("CAMERA CALIBRATION MODE:\n- calibrated")
        with open(CALIB_FILE, "r") as f:
            data = json.load(f)
        cameras = {}
        for cam_name in CAMERAS:
            K = np.array(data[cam_name]["K"], dtype=np.float32)
            R = np.array(data[cam_name]["R"], dtype=np.float32)
            t = np.array(data[cam_name]["t"], dtype=np.float32).reshape(3, 1)
            cameras[cam_name] = (K, R, t)
        return cameras

    print("CAMERA CALIBRATION MODE:\n- approximate")
    print("WARNING: Camera poses were not explicitly calibrated; reconstruction geometry is approximate.")

    positions = {
        "cam0": np.array([-2.6, -2.6, 1.45], dtype=np.float32),
        "cam1": np.array([ 2.6, -2.6, 1.45], dtype=np.float32),
        "cam2": np.array([ 0.0,  3.2, 1.50], dtype=np.float32)
    }

    cameras = {}
    for name in CAMERAS:
        w = video_infos[name]["width"]
        h = video_infos[name]["height"]
        focal = 0.95 * max(w, h)
        K = np.array([
            [focal, 0.0,   w * 0.5],
            [0.0,   focal, h * 0.5],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)
        R, t = look_at(positions[name])
        cameras[name] = (K, R, t)

    return cameras

def project_points(pts, camera):
    K, R, t = camera
    cam_pts = (R @ pts.T) + t
    z = cam_pts[2]
    valid_z = z > 1e-4

    safe_z = np.where(valid_z, z, 1e-4)
    u = (K[0, 0] * (cam_pts[0] / safe_z)) + K[0, 2]
    v = (K[1, 1] * (cam_pts[1] / safe_z)) + K[1, 2]

    # Calculate unit ray directions from optical center for lighting/angle weights
    C = -R.T @ t
    rays = pts - C.T
    norm_rays = np.linalg.norm(rays, axis=1, keepdims=True) + 1e-8
    rays = rays / norm_rays

    return np.column_stack([u, v]), z, rays, valid_z

# ------------------------------------------------------------
# TEMPORAL SYNCHRONIZATION
# ------------------------------------------------------------
def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream: {path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0 or math.isnan(fps):
        fps = 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps if count > 0 else 0.0
    cap.release()

    return {"width": w, "height": h, "fps": fps, "frames": count, "duration": duration}

def read_frame_at_time(cap, timestamp, video_fps, total_frames):
    target_idx = int(round(timestamp * video_fps))
    target_idx = max(0, min(total_frames - 1, target_idx))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    return frame if ok else None

# ------------------------------------------------------------
# SILHOUETTE SEGMENTATION WITH TEMPORAL CACHING
# ------------------------------------------------------------
class SilhouetteExtractor:
    def __init__(self, model_path):
        self.model = YOLO(str(model_path))
        self.cache = {}
        self.failed_frames = {c: 0 for c in CAMERAS}

    def process(self, frame, cam_name, frame_idx):
        h, w = frame.shape[:2]
        results = self.model.predict(
            frame,
            imgsz=768,
            conf=0.25,
            classes=[0],
            verbose=False
        )[0]

        best_mask = None
        if results.masks is not None and len(results.masks.data) > 0:
            boxes = results.boxes
            masks = results.masks.data.cpu().numpy()
            best_area = 0

            for i, box in enumerate(boxes):
                if int(box.cls[0].item()) != 0:
                    continue
                m = masks[i]
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                bin_m = (m > 0.45).astype(np.uint8) * 255
                area = np.count_nonzero(bin_m)
                if area > best_area:
                    best_area = area
                    best_mask = bin_m

        if best_mask is None or np.count_nonzero(best_mask) < 200:
            self.failed_frames[cam_name] += 1
            if cam_name in self.cache:
                return self.cache[cam_name].copy()
            return np.zeros((h, w), dtype=np.uint8)

        # Morphological filter: Fill interior holes without losing extremities
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Retain only the single largest connected human component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        self.cache[cam_name] = cleaned
        return cleaned

# ------------------------------------------------------------
# COARSE-TO-FINE VISUAL HULL
# ------------------------------------------------------------
def query_occupancy(pts, masks, cameras, min_views=3):
    in_view_count = np.zeros(len(pts), dtype=np.int16)

    for cam_name in CAMERAS:
        mask = masks[cam_name]
        camera = cameras[cam_name]
        h, w = mask.shape[:2]

        proj, z, _, valid_z = project_points(pts, camera)
        u = np.rint(proj[:, 0]).astype(np.int32)
        v = np.rint(proj[:, 1]).astype(np.int32)

        valid = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.where(valid)[0]

        if len(idx) > 0:
            fg = mask[v[idx], u[idx]] > 128
            in_view_count[idx[fg]] += 1

    return in_view_count >= min_views

def reconstruct_dense_volume(masks, cameras):
    # Stage 1: Coarse Grid Evaluation
    xs = np.linspace(GLOBAL_BOUNDS_MIN[0], GLOBAL_BOUNDS_MAX[0], COARSE_RES[0], dtype=np.float32)
    ys = np.linspace(GLOBAL_BOUNDS_MIN[1], GLOBAL_BOUNDS_MAX[1], COARSE_RES[1], dtype=np.float32)
    zs = np.linspace(GLOBAL_BOUNDS_MIN[2], GLOBAL_BOUNDS_MAX[2], COARSE_RES[2], dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    coarse_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    coarse_occ = query_occupancy(coarse_pts, masks, cameras, min_views=MIN_VIEWS_CONSENSUS)
    occupied_idx = np.where(coarse_occ)[0]

    if len(occupied_idx) < 10:
        # Fallback to majority voting if strict consensus yielded insufficient volume
        coarse_occ = query_occupancy(coarse_pts, masks, cameras, min_views=2)
        occupied_idx = np.where(coarse_occ)[0]

    if len(occupied_idx) < 10:
        return None, None, None

    occupied_pts = coarse_pts[occupied_idx]
    box_min = np.clip(occupied_pts.min(axis=0) - 0.12, GLOBAL_BOUNDS_MIN, GLOBAL_BOUNDS_MAX)
    box_max = np.clip(occupied_pts.max(axis=0) + 0.12, GLOBAL_BOUNDS_MIN, GLOBAL_BOUNDS_MAX)

    # Stage 2: Fine Grid bounded within localized AABB
    fxs = np.linspace(box_min[0], box_max[0], FINE_RES[0], dtype=np.float32)
    fys = np.linspace(box_min[1], box_max[1], FINE_RES[1], dtype=np.float32)
    fzs = np.linspace(box_min[2], box_max[2], FINE_RES[2], dtype=np.float32)

    spacing = np.array([
        (box_max[0] - box_min[0]) / (FINE_RES[0] - 1),
        (box_max[1] - box_min[1]) / (FINE_RES[1] - 1),
        (box_max[2] - box_min[2]) / (FINE_RES[2] - 1)
    ], dtype=np.float32)

    FX, FY, FZ = np.meshgrid(fxs, fys, fzs, indexing="ij")
    fine_pts = np.column_stack([FX.ravel(), FY.ravel(), FZ.ravel()])

    fine_occ = query_occupancy(fine_pts, masks, cameras, min_views=MIN_VIEWS_CONSENSUS)
    volume = fine_occ.reshape(FINE_RES).astype(np.float32)

    # Topological Gaussian regularization to smooth voxel staircase artifacts
    volume = ndi.gaussian_filter(volume, sigma=0.85)
    return volume, box_min, spacing

# ------------------------------------------------------------
# MARCHING CUBES & MESH REFINEMENT
# ------------------------------------------------------------
def extract_clean_surface(volume, box_min, spacing):
    if volume is None or np.max(volume) < 0.35:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32), np.empty((0, 3), dtype=np.float32)

    # Isosurface extraction at 0.40 density
    verts, faces, normals, _ = measure.marching_cubes(
        volume,
        level=0.40,
        spacing=(spacing[0], spacing[1], spacing[2]),
        method='lewiner'
    )

    verts = verts.astype(np.float32) + box_min.astype(np.float32)
    faces = faces.astype(np.uint32)

    if len(faces) == 0:
        return verts, faces, normals

    # Disconnected component removal: keep largest body component
    adjacency = {}
    for f in faces:
        for i in range(3):
            v0, v1 = f[i], f[(i + 1) % 3]
            adjacency.setdefault(v0, set()).add(v1)
            adjacency.setdefault(v1, set()).add(v0)

    visited = np.zeros(len(verts), dtype=bool)
    components = []

    for v_idx in range(len(verts)):
        if visited[v_idx]:
            continue
        comp = []
        queue = [v_idx]
        visited[v_idx] = True

        while queue:
            curr = queue.pop()
            comp.append(curr)
            for neighbor in adjacency.get(curr, []):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(comp)

    if components:
        largest_comp = max(components, key=len)
        if len(largest_comp) < len(verts):
            keep_mask = np.zeros(len(verts), dtype=bool)
            keep_mask[largest_comp] = True
            remap = np.full(len(verts), -1, dtype=np.int32)
            remap[largest_comp] = np.arange(len(largest_comp))

            verts = verts[largest_comp]
            normals = normals[largest_comp]

            valid_faces = keep_mask[faces[:, 0]] & keep_mask[faces[:, 1]] & keep_mask[faces[:, 2]]
            faces = remap[faces[valid_faces]].astype(np.uint32)

    # Compute outward surface unit normals
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    normals = (normals / norm_len).astype(np.float32)

    return verts, faces, normals

# ------------------------------------------------------------
# MULTI-VIEW COLOR FUSION
# ------------------------------------------------------------
def blend_vertex_colors(verts, normals, frames, masks, cameras):
    if len(verts) == 0:
        return np.empty((0, 3), dtype=np.uint8)

    accum_colors = np.zeros((len(verts), 3), dtype=np.float32)
    accum_weights = np.zeros(len(verts), dtype=np.float32)

    for cam_name in CAMERAS:
        frame = frames[cam_name]
        mask = masks[cam_name]
        camera = cameras[cam_name]
        h, w = frame.shape[:2]

        proj, z, rays, valid_z = project_points(verts, camera)
        u = np.rint(proj[:, 0]).astype(np.int32)
        v = np.rint(proj[:, 1]).astype(np.int32)

        valid = valid_z & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue

        fg = mask[v[idx], u[idx]] > 128
        idx = idx[fg]
        if len(idx) == 0:
            continue

        # Cosine alignment weight: surface normal facing the camera lens
        cos_theta = np.clip(-np.sum(rays[idx] * normals[idx], axis=1), 0.05, 1.0)
        weight = cos_theta / (np.sqrt(z[idx]) + 1e-4)

        bgr_samples = frame[v[idx], u[idx]].astype(np.float32)
        rgb_samples = bgr_samples[:, ::-1]

        accum_colors[idx] += rgb_samples * weight[:, None]
        accum_weights[idx] += weight

    valid_colored = accum_weights > 1e-5
    final_rgb = np.full((len(verts), 3), 160, dtype=np.uint8)

    if np.any(valid_colored):
        final_rgb[valid_colored] = np.clip(
            accum_colors[valid_colored] / accum_weights[valid_colored, None],
            0,
            255
        ).astype(np.uint8)

    return final_rgb

# ------------------------------------------------------------
# EXPORTERS (VMESH, PLY, V4D)
# ------------------------------------------------------------
def write_ply_file(path, verts, faces, colors):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, c in zip(verts, colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def write_vmesh_stream(path, mesh_frames, fps, bounds_min, bounds_max):
    """
    Compact binary format for high-speed streaming into Three.js BufferGeometry:
    Header: Magic('VMES'), Version(uint16), FrameCount(uint32), FPS(float32), Bounds(6*float32)
    Per Frame: Timestamp(float32), VertCount(uint32), IndexCount(uint32)
               Verts(N*3*float32), Normals(N*3*float32), Colors(N*3*uint8), Indices(I*uint32)
    """
    with open(path, "wb") as f:
        f.write(b"VMES")
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", len(mesh_frames)))
        f.write(struct.pack("<f", float(fps)))
        f.write(struct.pack("<6f", *bounds_min, *bounds_max))

        for frame in mesh_frames:
            ts = frame["timestamp"]
            verts = frame["verts"]
            faces = frame["faces"]
            normals = frame["normals"]
            colors = frame["colors"]

            indices = faces.ravel()
            f.write(struct.pack("<f", float(ts)))
            f.write(struct.pack("<I", len(verts)))
            f.write(struct.pack("<I", len(indices)))

            if len(verts) > 0:
                f.write(verts.astype(np.float32).tobytes())
                f.write(normals.astype(np.float32).tobytes())
                f.write(colors.astype(np.uint8).tobytes())
                f.write(indices.astype(np.uint32).tobytes())

def write_v4d_backward_compatible(path, v4d_frames, fps):
    voxel_size = float(np.mean(
        (GLOBAL_BOUNDS_MAX - GLOBAL_BOUNDS_MIN) /
        np.array([GRID_X - 1, GRID_Y - 1, GRID_Z - 1], dtype=np.float32)
    ))

    with open(path, "wb") as f:
        f.write(b"V4D1")
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<HHH", GRID_X, GRID_Y, GRID_Z))
        f.write(struct.pack("<I", len(v4d_frames)))
        f.write(struct.pack("<f", float(fps)))
        f.write(struct.pack("<f", voxel_size))

        for ts, voxels in v4d_frames:
            f.write(struct.pack("<f", float(ts)))
            f.write(struct.pack("<I", len(voxels)))
            if len(voxels) > 0:
                f.write(voxels.astype(np.uint8).tobytes())

def mesh_to_v4d_voxels(verts, colors):
    if len(verts) == 0:
        return np.empty((0, 6), dtype=np.uint8)

    norm_coords = (verts - GLOBAL_BOUNDS_MIN) / (GLOBAL_BOUNDS_MAX - GLOBAL_BOUNDS_MIN)
    grid_scale = np.array([GRID_X - 1, GRID_Y - 1, GRID_Z - 1], dtype=np.float32)
    coords = np.clip(np.rint(norm_coords * grid_scale), 0, 255).astype(np.uint8)

    # Unique voxels to prevent duplicate alpha buildup
    composite = np.column_stack([coords, colors])
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return composite[unique_idx]

# ------------------------------------------------------------
# MAIN EXECUTION ROUTINE
# ------------------------------------------------------------
def run_pipeline(progress_callback=None):
    start_time = time.time()
    print("=" * 70)
    print("VOXEL REPLAY: HIGH-FIDELITY 3-CAMERA RECONSTRUCTION")
    print("=" * 70)

    # Verify input videos
    video_paths = [VIDEO_ROOT / f"{c}.mp4" for c in CAMERAS]
    for p in video_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input video: {p}")

    infos = {c: inspect_video(p) for c, p in zip(CAMERAS, video_paths)}
    common_duration = min(infos[c]["duration"] for c in CAMERAS)
    if common_duration <= 0.05:
        raise ValueError("Invalid video stream: Zero or negative synchronized duration.")

    frame_count = min(MAX_FRAMES, max(2, int(math.floor(common_duration * TARGET_FPS))))
    output_fps = (frame_count - 1) / common_duration if common_duration > 0 else TARGET_FPS

    print(f"Cameras: {', '.join(CAMERAS)}")
    print(f"Synchronized Timeline: {common_duration:.2f}s | Frame Count: {frame_count} | Target FPS: {output_fps:.2f}")

    cameras = get_cameras(infos)
    seg_extractor = SilhouetteExtractor(YOLO_MODEL_PATH)
    captures = {c: cv2.VideoCapture(str(VIDEO_ROOT / f"{c}.mp4")) for c in CAMERAS}

    mesh_frames = []
    v4d_frames = []

    for f_idx in range(frame_count):
        ts = 0.0 if frame_count == 1 else (f_idx / (frame_count - 1)) * common_duration
        percent = int(((f_idx + 1) / frame_count) * 100)

        if progress_callback:
            progress_callback(stage="reconstructing", progress=percent, current_frame=f_idx + 1, total=frame_count)

        print(f"[{f_idx + 1:02d}/{frame_count:02d}] t={ts:.2f}s | ", end="", flush=True)

        frames = {}
        masks = {}
        for c in CAMERAS:
            fr = read_frame_at_time(captures[c], ts, infos[c]["fps"], infos[c]["frames"])
            if fr is None:
                raise RuntimeError(f"Could not read camera frame for {c} at time {ts:.3f}s")
            frames[c] = fr
            masks[c] = seg_extractor.process(fr, c, f_idx)

        # Step 1: Coarse-to-fine visual hull volume
        volume, box_min, spacing = reconstruct_dense_volume(masks, cameras)

        # Step 2: Surface mesh extraction via Marching Cubes
        if volume is not None:
            verts, faces, normals = extract_clean_surface(volume, box_min, spacing)
            colors = blend_vertex_colors(verts, normals, frames, masks, cameras)
        else:
            verts = np.empty((0, 3), dtype=np.float32)
            faces = np.empty((0, 3), dtype=np.uint32)
            normals = np.empty((0, 3), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.uint8)

        print(f"Mesh: {len(verts):,} v, {len(faces):,} f | ", end="", flush=True)

        # Step 3: Export PLY for local DCC inspection
        ply_path = MESH_FRAME_DIR / f"frame_{f_idx:04d}.ply"
        write_ply_file(ply_path, verts, faces, colors)

        # Step 4: Cache for VMESH binary
        mesh_frames.append({
            "timestamp": ts,
            "verts": verts,
            "faces": faces,
            "normals": normals,
            "colors": colors
        })

        # Step 5: Generate backward-compatible V4D voxels
        voxels = mesh_to_v4d_voxels(verts, colors)
        v4d_frames.append((ts, voxels))
        print(f"Voxels: {len(voxels):,}")

    for cap in captures.values():
        cap.release()

    if progress_callback:
        progress_callback(stage="packaging", progress=98, current_frame=frame_count, total=frame_count)

    print("\nPackaging animations...")
    # Write VMESH
    write_vmesh_stream(OUTPUT_VMESH, mesh_frames, output_fps, GLOBAL_BOUNDS_MIN, GLOBAL_BOUNDS_MAX)
    shutil.copy2(OUTPUT_VMESH, PLAYER_OUTPUT_VMESH)

    # Write V4D
    write_v4d_backward_compatible(OUTPUT_V4D, v4d_frames, output_fps)
    shutil.copy2(OUTPUT_V4D, PLAYER_OUTPUT_V4D)

    elapsed = time.time() - start_time
    print("=" * 70)
    print("RECONSTRUCTION COMPLETED SUCCESSFULLY")
    print(f"Total Processing Duration : {elapsed / 60.0:.2f} minutes")
    print(f"Dynamic Mesh Animation    : {OUTPUT_VMESH} -> {PLAYER_OUTPUT_VMESH}")
    print(f"Compatible V4D Output     : {OUTPUT_V4D} -> {PLAYER_OUTPUT_V4D}")
    print(f"Exported PLY Sequences    : {MESH_FRAME_DIR}")
    for c, count in seg_extractor.failed_frames.items():
        if count > 0:
            print(f"Notice: {c} had {count} temporal mask interpolations.")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()