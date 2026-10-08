"""
VOXEL REPLAY - SINGLE PERSON, REAL-COLOUR, FRAME-BY-FRAME 4D RECONSTRUCTION

3 portrait videos of ONE person  ->  person-only masks  ->  strict multi-view
visual hull  ->  3D ghost removal  ->  real video colours  ->  V4D file.

Output (same V4D format as before, same two paths):
    exports/reconstructed_color.v4d
    player/public/reconstructed_color.v4d

Run:
    C:\\Users\\hari1\\voxel-replay\\.venv\\Scripts\\python.exe reconstruct_color.py
"""

from pathlib import Path
import shutil
import struct
import time

import cv2
import numpy as np
from scipy import ndimage as ndi   # installed together with ultralytics


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(r"C:\Users\hari1\voxel-replay")
VIDEO_DIR = ROOT / "data" / "videos"
EXPORT_DIR = ROOT / "exports"
PLAYER_DIR = ROOT / "player" / "public"
MODEL_PATH = ROOT / "models" / "yolo11n-seg.pt"
OUTPUT_V4D = EXPORT_DIR / "reconstructed_color.v4d"
PLAYER_V4D = PLAYER_DIR / "reconstructed_color.v4d"

CAMERA_FILES = ["cam0.mp4", "cam1.mp4", "cam2.mp4"]


# ============================================================
# SETTINGS
# ============================================================

STEP = 2                                # use every 2nd source frame

# 3D grid (must stay the same size your player already expects)
GRID_X, GRID_Y, GRID_Z = 72, 72, 108

# CUBIC voxels. The old code used a different size on every axis but wrote a
# single voxel_size into the file, so the player stretched the person.
# 0.0225 m  ->  X/Y span 1.60 m, Z span 2.41 m.
# If outstretched arms ever get clipped at the sides, raise GRID_X and GRID_Y
# (for example to 96) - nothing else needs to change.
VOXEL_SIZE = 0.0225

XMIN = -(GRID_X - 1) * VOXEL_SIZE / 2.0
YMIN = -(GRID_Y - 1) * VOXEL_SIZE / 2.0
ZMIN = 0.0                              # feet are at z = 0, head points to +z

# ---- camera model (3 cameras on a circle around the person) ----------------
PERSON_HEIGHT_M = 1.70                  # only sets the overall scale
CAMERA_DISTANCE_M = 4.0
CAMERA_HEIGHT_M = 0.9
# Camera angle around the person (degrees). Same layout as your old positions:
# cam0 behind, cam1 right, cam2 left, roughly 120 degrees apart.
NOMINAL_AZIMUTH_DEG = [90.0, -30.0, -150.0]
REFINE_AZIMUTH = True                   # fine-tune cam1/cam2 angles from the videos
AZIMUTH_SEARCH_DEG = 40
CALIBRATION_FRAMES = 12

# ---- segmentation ----------------------------------------------------------
YOLO_IMAGE_SIZE = 768
YOLO_CONFIDENCE = 0.25
MIN_PERSON_AREA_FRACTION = 0.01         # person must cover >= 1 % of the frame
AREA_JUMP_LOW, AREA_JUMP_HIGH = 0.5, 1.8   # reject sudden mask size glitches

# ---- visual hull / ghost removal -------------------------------------------
HULL_MASK_DILATE_PX = 3                 # tolerate small mask errors
MERGE_ITER = 3                          # voxels; joins a hand to its body
TEMPORAL_ITER = 4                       # voxels; follow the same body over time
GUARD_ITER = 6                          # used only when a camera drops out
MIN_BODY_VOXELS = 300

# ---- colour ----------------------------------------------------------------
VISIBILITY_TOLERANCE_M = 0.07

SAVE_DEBUG_IMAGES = True

# ---- V4D format (unchanged) ------------------------------------------------
MAGIC = b"V4D1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sHHHHIff")
FRAME_HEADER_STRUCT = struct.Struct("<fI")

STRUCT26 = ndi.generate_binary_structure(3, 3)


# ============================================================
# CAMERA MODEL
# ============================================================

def make_camera(azimuth_deg, f, cx, cy, width, height):
    a = np.radians(azimuth_deg)
    pos = np.array(
        [CAMERA_DISTANCE_M * np.cos(a),
         CAMERA_DISTANCE_M * np.sin(a),
         CAMERA_HEIGHT_M],
        dtype=np.float64,
    )
    forward = np.array([-np.cos(a), -np.sin(a), 0.0])
    right = np.array([forward[1], -forward[0], 0.0])
    up = np.cross(right, forward)
    rot = np.stack([right, up, forward], axis=0)
    return {"pos": pos, "R": rot, "f": f, "cx": cx, "cy": cy,
            "w": width, "h": height, "az": azimuth_deg}


def project(points, cam):
    c = (points - cam["pos"]) @ cam["R"].T
    z = c[:, 2]
    zs = np.maximum(z, 1e-6)
    px = cam["cx"] + cam["f"] * c[:, 0] / zs
    py = cam["cy"] - cam["f"] * c[:, 1] / zs      # image y grows downward
    return px, py, z


def grid_points(nx, ny, nz):
    xs = np.linspace(XMIN, -XMIN, nx)
    ys = np.linspace(YMIN, -YMIN, ny)
    zs = np.linspace(ZMIN, ZMIN + (GRID_Z - 1) * VOXEL_SIZE, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)


# ============================================================
# SEGMENTATION  (one person, correct image coordinates)
# ============================================================

def load_segmenter():
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    def candidates(frame):
        h, w = frame.shape[:2]
        res = model.predict(frame, imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE,
                            classes=[0], verbose=False)[0]
        out = []
        if res.masks is None:
            return out
        # masks.xy = polygons already in ORIGINAL frame pixels. (The old code
        # resized the padded network mask, which shifted/distorted silhouettes.)
        for poly in res.masks.xy:
            if len(poly) < 3:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            out.append(m)
        return out

    return candidates


def clean_mask(mask):
    """Close small holes, keep the body plus anything attached/very close."""
    h = mask.shape[0]
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    gap = max(3, int(0.02 * h))
    grown = cv2.dilate(m, np.ones((2 * gap + 1, 2 * gap + 1), np.uint8))
    n, lab = cv2.connectedComponents(grown, connectivity=8)
    if n <= 2:
        return m
    counts = [int(((lab == i) & (m > 0)).sum()) for i in range(1, n)]
    best = 1 + int(np.argmax(counts))
    return (m * (lab == best)).astype(np.uint8)


class PersonTracker:
    """Follows ONE person in one camera and rejects glitches."""

    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.center = None
        self.areas = []
        self.rejected = 0

    def __call__(self, frame):
        h, w = frame.shape[:2]
        cands = [m for m in self.segmenter(frame)
                 if m.sum() >= MIN_PERSON_AREA_FRACTION * h * w]
        if not cands:
            return None

        def score(m):
            area = float(m.sum())
            if self.center is None:
                return area
            ys, xs = np.nonzero(m)
            d = np.hypot(xs.mean() - self.center[0], ys.mean() - self.center[1])
            return area * np.exp(-d / (0.25 * max(h, w)))

        main = max(cands, key=score)
        union = main.copy()
        for m in cands:
            if m is main:
                continue
            overlap = (m & main).sum() / max(1, min(m.sum(), main.sum()))
            if overlap > 0.3:               # same person detected twice
                union |= m
        mask = clean_mask(union)
        area = int(mask.sum())
        if area < MIN_PERSON_AREA_FRACTION * h * w:
            return None

        if len(self.areas) >= 3 and self.rejected < 5:
            med = float(np.median(self.areas[-15:]))
            if area < AREA_JUMP_LOW * med or area > AREA_JUMP_HIGH * med:
                self.rejected += 1
                return None
        self.rejected = 0
        ys, xs = np.nonzero(mask)
        self.center = (float(xs.mean()), float(ys.mean()))
        self.areas.append(area)
        return mask


# ============================================================
# CALIBRATION FROM THE VIDEOS THEMSELVES
# ============================================================

def open_caps():
    caps, info = [], []
    for name in CAMERA_FILES:
        path = VIDEO_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing video: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        caps.append(cap)
        info.append({
            "name": name,
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        })
    return caps, info


def sample_masks(segmenter, common_frames):
    """Run the segmenter on a few frames of every camera."""
    caps, info = open_caps()
    trackers = [PersonTracker(segmenter) for _ in caps]
    idxs = np.linspace(0.05 * common_frames, 0.95 * common_frames - 1,
                       CALIBRATION_FRAMES).astype(int)
    masks = [[] for _ in caps]
    for i in idxs:
        row = []
        for cap, tr in zip(caps, trackers):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            row.append(tr(frame) if ok else None)
        if all(m is not None for m in row):
            for c in range(len(caps)):
                masks[c].append(row[c])
    for cap in caps:
        cap.release()
    return masks, info


def intrinsics_from_masks(masks, w, h, name):
    tops, bottoms, cxs = [], [], []
    for m in masks:
        ys, xs = np.nonzero(m)
        tops.append(ys.min())
        bottoms.append(ys.max())
        cxs.append(xs.mean())
        if (xs.max() - xs.min()) > 1.15 * (ys.max() - ys.min()):
            print(f"  WARNING {name}: person looks wider than tall - "
                  f"is this video rotated sideways?")
    top, bottom, cx = float(np.median(tops)), float(np.median(bottoms)), float(np.median(cxs))
    f = (bottom - top) * CAMERA_DISTANCE_M / PERSON_HEIGHT_M
    cy = bottom - f * CAMERA_HEIGHT_M / CAMERA_DISTANCE_M     # feet row -> z = 0
    if bottom >= h - 3:
        print(f"  WARNING {name}: feet touch the bottom edge of the frame; "
              f"scale/height may be slightly off")
    return {"f": f, "cx": cx, "cy": cy, "width": w, "height": h}


def refine_azimuths(base, masks):
    """Search camera angles whose silhouettes agree best (hull re-projection)."""
    S = 0.25
    small = [[cv2.resize(m, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
              for m in masks[c]] for c in range(3)]
    K = len(small[0])
    pts = grid_points(36, 36, 54)
    N = len(pts)
    cache = {}
    kern = np.ones((7, 7), np.uint8)

    def view(c, az):
        key = (c, round(az, 3))
        if key in cache:
            return cache[key]
        cam = make_camera(az, **base[c])
        px, py, z = project(pts, cam)
        px, py = px * S, py * S
        hs, ws = small[c][0].shape
        ok = (z > 0.05) & (px >= 0) & (px < ws) & (py >= 0) & (py < hs)
        pxi = np.clip(px.astype(np.int32), 0, ws - 1)
        pyi = np.clip(py.astype(np.int32), 0, hs - 1)
        inside = np.zeros((K, N), bool)
        for k in range(K):
            inside[k] = ok & (small[c][k][pyi, pxi] > 0)
        cache[key] = (pxi, pyi, inside)
        return cache[key]

    def score(azs):
        views = [view(c, azs[c]) for c in range(3)]
        hull = views[0][2] & views[1][2] & views[2][2]
        total = 0.0
        for k in range(K):
            hk = hull[k]
            if hk.sum() < 30:
                continue
            for c in range(3):
                ms = small[c][k]
                img = np.zeros(ms.shape, np.uint8)
                img[views[c][1][hk], views[c][0][hk]] = 1
                img = cv2.dilate(img, kern)
                inter = int((img & ms).sum())
                union = int((img | ms).sum())
                total += inter / max(union, 1)
        return total / (K * 3)

    n = NOMINAL_AZIMUTH_DEG
    best, best_s = list(n), score(n)
    R = AZIMUTH_SEARCH_DEG
    for d1 in range(-R, R + 1, 5):
        for d2 in range(-R, R + 1, 5):
            azs = [n[0], n[1] + d1, n[2] + d2]
            s = score(azs)
            if s > best_s:
                best, best_s = azs, s
    centre = list(best)
    for d1 in range(-4, 5):
        for d2 in range(-4, 5):
            azs = [centre[0], centre[1] + d1, centre[2] + d2]
            s = score(azs)
            if s > best_s:
                best, best_s = azs, s
    return best, best_s


# ============================================================
# 3D GHOST REMOVAL
# ============================================================

def keep_main_body(vol, prev):
    """Keep ONE body (plus anything within a few voxels of it)."""
    if not vol.any():
        return vol
    merged = ndi.binary_dilation(vol, STRUCT26, iterations=MERGE_ITER)
    lab, n = ndi.label(merged, structure=STRUCT26)
    idx = np.arange(1, n + 1)
    sizes = np.asarray(ndi.sum(vol.astype(np.uint8), lab, idx))
    best = int(np.argmax(sizes))
    if prev is not None and n > 1:
        pd = ndi.binary_dilation(prev, STRUCT26, iterations=TEMPORAL_ITER)
        overlap = np.asarray(ndi.sum(pd.astype(np.uint8), lab, idx))
        cands = np.flatnonzero(sizes >= 0.4 * sizes.max())
        best = int(cands[np.argmax(overlap[cands])])
    out = vol & (lab == best + 1)
    if int(out.sum()) < MIN_BODY_VOXELS:
        return np.zeros_like(vol)
    return out


# ============================================================
# REAL COLOUR (visibility aware, no invented colours)
# ============================================================

def colorize(vol, frames, masks, cams, valid_cams):
    flat = vol.reshape(-1)
    idx = np.flatnonzero(flat)
    if len(idx) == 0:
        return np.zeros((0, 6), np.uint8)

    pts = VOXEL_POINTS[idx]
    smooth = ndi.gaussian_filter(vol.astype(np.float32), 1.2)
    grad = np.stack(np.gradient(smooth), axis=-1).reshape(-1, 3)[idx]
    normals = -grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-9)

    acc = np.zeros((len(idx), 3), np.float64)
    wsum = np.zeros(len(idx), np.float64)

    for c in valid_cams:
        cam = cams[c]
        px, py, z = cam["px"][idx], cam["py"][idx], cam["z"][idx]
        cell = cam["cell"]
        zbuf = np.full((cam["h"] // cell + 2, cam["w"] // cell + 2), np.inf)
        np.minimum.at(zbuf, (py // cell, px // cell), z)
        visible = z <= zbuf[py // cell, px // cell] + VISIBILITY_TOLERANCE_M

        er = max(2, cell // 2)
        emask = cv2.erode(masks[c], np.ones((2 * er + 1, 2 * er + 1), np.uint8))
        inside = emask[py, px] > 0

        to_cam = cam["pos"] - pts
        to_cam /= (np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9)
        facing = np.clip(np.einsum("ij,ij->i", normals, to_cam), 0.0, 1.0) ** 2
        weight = facing * visible * inside

        k = max(3, cell | 1)
        smooth_img = cv2.blur(frames[c], (k, k))
        rgb = smooth_img[py, px][:, ::-1].astype(np.float64)   # BGR -> RGB
        acc += rgb * weight[:, None]
        wsum += weight

    has = wsum > 1e-3
    coloured = np.zeros(vol.shape, bool)
    colour_vol = np.zeros(vol.shape + (3,), np.uint8)
    cflat = coloured.reshape(-1)
    colour_flat = colour_vol.reshape(-1, 3)
    cflat[idx[has]] = True
    colour_flat[idx[has]] = np.clip(acc[has] / wsum[has, None], 0, 255).astype(np.uint8)

    if not has.all():
        if not has.any():
            return np.zeros((0, 6), np.uint8)
        # Hidden / interior voxels borrow the colour of the nearest voxel that
        # really was seen by a camera (still a real video pixel colour).
        _, near = ndi.distance_transform_edt(~coloured, return_indices=True)
        missing = idx[~has]
        mx, my, mz = np.unravel_index(missing, vol.shape)
        colour_flat[missing] = colour_vol[near[0][mx, my, mz],
                                          near[1][mx, my, mz],
                                          near[2][mx, my, mz]]

    ix, iy, iz = np.unravel_index(idx, vol.shape)
    return np.column_stack([ix, iy, iz, colour_flat[idx]]).astype(np.uint8)


# ============================================================
# V4D I/O
# ============================================================

def write_v4d(path, frames, fps):
    with open(path, "wb") as f:
        f.write(HEADER_STRUCT.pack(MAGIC, VERSION, GRID_X, GRID_Y, GRID_Z,
                                   len(frames), float(fps), float(VOXEL_SIZE)))
        for i, vox in enumerate(frames):
            f.write(FRAME_HEADER_STRUCT.pack(float(i / fps), int(len(vox))))
            if len(vox):
                f.write(np.ascontiguousarray(vox, dtype=np.uint8).tobytes())


def verify_v4d(path):
    data = Path(path).read_bytes()
    magic, ver, gx, gy, gz, nf, fps, vs = HEADER_STRUCT.unpack_from(data, 0)
    pos = HEADER_STRUCT.size
    total = 0
    for _ in range(nf):
        _, cnt = FRAME_HEADER_STRUCT.unpack_from(data, pos)
        pos += FRAME_HEADER_STRUCT.size + 6 * cnt
        total += cnt
    assert magic == MAGIC and pos == len(data), "V4D file check failed"
    return nf, total


def save_debug_images(snapshot, cams):
    frames, masks, idx = snapshot
    pts = VOXEL_POINTS[idx]
    for c, cam in enumerate(cams):
        img = frames[c].copy()
        cs, _ = cv2.findContours(masks[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 255, 0), 2)
        px, py, _ = project(pts, cam)
        for x, y in zip(px[::6].astype(int), py[::6].astype(int)):
            if 0 <= x < cam["w"] and 0 <= y < cam["h"]:
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.imwrite(str(EXPORT_DIR / f"debug_cam{c}.png"), img)


# ============================================================
# MAIN
# ============================================================

VOXEL_POINTS = None


def main(segmenter=None):
    global VOXEL_POINTS
    t0 = time.time()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("VOXEL REPLAY - SINGLE PERSON REAL 4D RECONSTRUCTION")
    print("=" * 78)

    if segmenter is None:
        print("Loading YOLO segmentation model...")
        segmenter = load_segmenter()

    caps, info = open_caps()
    for i in info:
        print(f"{i['name']:10s} {i['w']:4d} x {i['h']:<4d} "
              f"{i['fps']:6.2f} FPS {i['frames']:5d} frames")
    for cap in caps:
        cap.release()

    common = min(i["frames"] for i in info)
    src_fps = min(i["fps"] for i in info)
    out_fps = src_fps / STEP
    print(f"Common frames {common}, output about {common // STEP} frames at {out_fps:.2f} FPS")

    # ---------------- calibration ----------------
    print("\nCalibrating cameras from the videos...")
    cal_masks, _ = sample_masks(segmenter, common)
    if min(len(m) for m in cal_masks) < 3:
        raise RuntimeError("Person was not found in all 3 cameras often enough "
                           "to calibrate. Check the videos / YOLO model path.")
    base = [intrinsics_from_masks(cal_masks[c], info[c]["w"], info[c]["h"],
                                  info[c]["name"]) for c in range(3)]
    azimuths = list(NOMINAL_AZIMUTH_DEG)
    if REFINE_AZIMUTH:
        azimuths, agree = refine_azimuths(base, cal_masks)
        print(f"Camera angles (deg): {[round(a, 1) for a in azimuths]}  "
              f"silhouette agreement {agree:.2f}")
        if agree < 0.5:
            print("  WARNING: silhouettes agree poorly - the cameras may not be "
                  "arranged like cam0 behind / cam1 right / cam2 left.")
    cams = [make_camera(azimuths[c], **base[c]) for c in range(3)]

    # ---------------- precompute projections (cameras are fixed) ------------
    VOXEL_POINTS = grid_points(GRID_X, GRID_Y, GRID_Z)
    for cam in cams:
        px, py, z = project(VOXEL_POINTS, cam)
        ok = (z > 0.05) & (px >= 0) & (px < cam["w"]) & (py >= 0) & (py < cam["h"])
        cam["ok"] = ok
        cam["px"] = np.clip(px.astype(np.int32), 0, cam["w"] - 1)
        cam["py"] = np.clip(py.astype(np.int32), 0, cam["h"] - 1)
        cam["z"] = z.astype(np.float32)
        cam["cell"] = max(2, int(round(cam["f"] * VOXEL_SIZE / CAMERA_DISTANCE_M)))
    grid_shape = (GRID_X, GRID_Y, GRID_Z)

    # ---------------- main reconstruction loop ----------------
    caps, _ = open_caps()
    trackers = [PersonTracker(segmenter) for _ in range(3)]
    all_frames, prev_vol, snapshot = [], None, None
    out_index = 0

    for src in range(common):
        if src % STEP:
            if not all([cap.grab() for cap in caps]):
                break
            continue
        frames = []
        for cap in caps:
            ok, fr = cap.read()
            frames.append(fr if ok else None)
        if any(fr is None for fr in frames):
            break

        masks = [trackers[c](frames[c]) for c in range(3)]
        valid = [c for c in range(3) if masks[c] is not None]
        print(f"[{out_index + 1:03d}] src {src:4d}  cameras with person: {len(valid)}/3", end="")

        if len(valid) < 2:
            print("  -> too few views, repeating previous frame")
            all_frames.append(all_frames[-1] if all_frames else np.zeros((0, 6), np.uint8))
            out_index += 1
            continue

        vol = np.ones(len(VOXEL_POINTS), bool)
        k = 2 * HULL_MASK_DILATE_PX + 1
        for c in valid:
            grown = cv2.dilate(masks[c], np.ones((k, k), np.uint8))
            vol &= cams[c]["ok"] & (grown[cams[c]["py"], cams[c]["px"]] > 0)
        vol = vol.reshape(grid_shape)

        if len(valid) < 3 and prev_vol is not None:
            vol &= ndi.binary_dilation(prev_vol, STRUCT26, iterations=GUARD_ITER)

        vol = keep_main_body(vol, prev_vol)
        if not vol.any():
            print("  -> empty hull, repeating previous frame")
            all_frames.append(all_frames[-1] if all_frames else np.zeros((0, 6), np.uint8))
            out_index += 1
            continue

        voxels = colorize(vol, frames, masks, cams, valid)
        all_frames.append(voxels)
        prev_vol = vol
        out_index += 1
        print(f"  voxels {len(voxels):6,d}")

        if snapshot is None and len(valid) == 3:
            snapshot = (frames, masks, np.flatnonzero(vol.reshape(-1)))

    for cap in caps:
        cap.release()

    if not all_frames:
        raise RuntimeError("No frames were reconstructed.")
    if SAVE_DEBUG_IMAGES and snapshot is not None:
        save_debug_images(snapshot, cams)
        print(f"\nDebug images: {EXPORT_DIR}\\debug_cam0.png / cam1 / cam2 "
              f"(green = person mask, red dots = reconstructed voxels)")

    # ---------------- write ----------------
    write_v4d(OUTPUT_V4D, all_frames, out_fps)
    shutil.copy2(OUTPUT_V4D, PLAYER_V4D)
    nf, total = verify_v4d(OUTPUT_V4D)

    counts = [len(f) for f in all_frames if len(f)]
    print("\n" + "=" * 78)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 78)
    print(f"Frames          : {nf}")
    print(f"FPS             : {out_fps:.3f}")
    print(f"Grid            : {GRID_X} x {GRID_Y} x {GRID_Z}  (voxel {VOXEL_SIZE} m)")
    if counts:
        print(f"Min voxels      : {min(counts):,}")
        print(f"Max voxels      : {max(counts):,}")
        print(f"Average voxels  : {np.mean(counts):,.0f}")
    print(f"File size       : {OUTPUT_V4D.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Time            : {(time.time() - t0) / 60:.2f} minutes")
    print(f"\nV4D           : {OUTPUT_V4D}")
    print(f"PLAYER UPDATED: {PLAYER_V4D}")
    print("=" * 78)
    print("ONE PERSON / REAL RGB / MULTI-VIEW / TEMPORAL 4D")
    print("=" * 78)


if __name__ == "__main__":
    main()