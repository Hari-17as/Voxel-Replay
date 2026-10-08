"""
VOXEL REPLAY - V4  (smooth, gap-free, textured human mesh)

What this does differently from V2/V3 (which only recolored cubes):

  1. Re-carves the body at 2-3x HIGHER resolution straight from your real masks
     (continuous signed-distance visual hull)  -> no cubes, no stair-steps.
  2. Cleans the hull: closes gaps, keeps the largest body, fills inner holes.
  3. Builds a smooth signed-distance field + marching cubes -> real surface.
  4. Taubin smoothing (smooths without shrinking the body).
  5. Colors every mesh vertex from the FULL-RES video frames with
     occlusion test + surface-normal weighting + mask-edge rejection,
     fills unseen areas by diffusion, blends seams.
  6. Writes:
        exports/mesh_frames/frame_XXXX.ply   (smooth colored mesh per frame)
        exports/mesh_frames/viewer.html      (three.js player, smooth shading)
        exports/reconstructed_color_V4.v4d   (gap-free cleaned voxels for your old player)

Install once:
    pip install numpy opencv-python scipy scikit-image

View the result:
    cd exports\\mesh_frames
    python -m http.server 8000
    open http://localhost:8000/viewer.html
"""

import json
import os
import shutil
import struct
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(os.environ.get("VOXEL_ROOT", r"C:\Users\hari1\voxel-replay"))

INPUT = ROOT / "exports" / "reconstructed_color.v4d"
OUTPUT = ROOT / "exports" / "reconstructed_color_V4.v4d"
PLAYER_OUTPUT = ROOT / "player" / "public" / "reconstructed_color.v4d"
MESH_DIR = ROOT / "exports" / "mesh_frames"

VIDEO_ROOT = ROOT / "data" / "videos"
MASK_ROOT = ROOT / "data" / "real_masks_v2"

CAMERAS = ["cam0", "cam1", "cam2"]
SOURCE_FPS = 23.94

# which frames to process (set FRAME_END=1 for a quick single-frame test)
FRAME_START = 0
FRAME_END = None          # None = all
FRAME_STEP = 1

# ---- shape quality ----
UPSCALE = 2               # 2 = good/fast, 3 = best (slower, bigger files)
CAND_DILATE = 2           # search radius around old voxels (coarse voxels)
CONSENSUS_INDEX = 0       # 0 = all cameras must agree, 1 = allow one camera to be wrong
HULL_BIAS = 1.0           # pixels; + grows body, - shrinks (fixes calibration error)
MASK_CLOSE = 7            # px, closes tiny holes in masks
KEEP_FRACTION = 0.03      # keep body parts >= 3% of the biggest piece (head/hands), drop floating noise
SDF_SIGMA = 1.2           # smoothness of surface in fine voxels (1.0 sharp .. 2.0 very smooth)
TAUBIN_ITERS = 12         # mesh smoothing passes
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53

# ---- color quality ----
VIEW_POWER = 3.0          # higher = trust cameras facing the surface more (sharper)
EDGE_MIN_DIST = 3.0       # px inside mask required
MIN_COS = 0.10
COLOR_SMOOTH_ITERS = 2    # blends camera seams on the surface
FRAME_BLUR = 3
ZBUF_SCALE = 2

# ---- outputs ----
EXPORT_MESH = True
WRITE_VIEWER = True
SHELL_THICKNESS = 2       # thickness of cleaned voxel shell for the old V4D player

PAD_C = CAND_DILATE + 2
PAD_F = 2
BIG = 50.0

BOUNDS_MIN = np.array([-1.25, -1.25, 0.0], dtype=np.float32)
BOUNDS_MAX = np.array([1.25, 1.25, 2.25], dtype=np.float32)
RANGE = BOUNDS_MAX - BOUNDS_MIN

CAMERA_POSITIONS = {
    "cam0": np.array([0.0, -4.2, 1.65], dtype=np.float32),
    "cam1": np.array([4.2, 0.0, 1.65], dtype=np.float32),
    "cam2": np.array([0.0, 4.2, 1.65], dtype=np.float32),
}


# ============================================================
# CAMERA (identical convention to your original code)
# ============================================================

def look_at(position):
    target = np.array([0.0, 0.0, 1.10], dtype=np.float32)
    forward = target - position
    forward /= np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    R = np.vstack([right, up, forward])
    t = -R @ position.reshape(3, 1)
    return R, t


def make_camera(name, width, height):
    focal = 0.95 * max(width, height)
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                 dtype=np.float32)
    R, t = look_at(CAMERA_POSITIONS[name])
    return K, R, t


def project(points, camera):
    K, R, t = camera
    cam = (R @ points.T.astype(np.float32)) + t
    z = cam[2]
    safe = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * cam[0] / safe + K[0, 2]
    v = K[1, 1] * cam[1] / safe + K[1, 2]
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


# ============================================================
# V4D IO
# ============================================================

def read_v4d(path):
    with open(path, "rb") as f:
        if f.read(4) != b"V4D1":
            raise RuntimeError("Invalid V4D file")
        version = struct.unpack("<H", f.read(2))[0]
        gx, gy, gz = struct.unpack("<HHH", f.read(6))
        frame_count = struct.unpack("<I", f.read(4))[0]
        fps = struct.unpack("<f", f.read(4))[0]
        voxel_size = struct.unpack("<f", f.read(4))[0]
        frames = []
        for _ in range(frame_count):
            ts = struct.unpack("<f", f.read(4))[0]
            count = struct.unpack("<I", f.read(4))[0]
            raw = f.read(count * 6)
            vox = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 6).copy()
            frames.append((ts, vox))
    return version, gx, gy, gz, frame_count, fps, voxel_size, frames


def write_v4d(path, version, gx, gy, gz, fps, voxel_size, frames):
    with open(path, "wb") as f:
        f.write(b"V4D1")
        f.write(struct.pack("<H", version))
        f.write(struct.pack("<HHH", gx, gy, gz))
        f.write(struct.pack("<I", len(frames)))
        f.write(struct.pack("<f", fps))
        f.write(struct.pack("<f", voxel_size))
        for ts, vox in frames:
            f.write(struct.pack("<f", float(ts)))
            f.write(struct.pack("<I", len(vox)))
            f.write(vox.astype(np.uint8).tobytes())


# ============================================================
# VIDEO
# ============================================================

class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pos = 0
        self.last_idx = -1
        self.last = None

    def get(self, idx):
        if idx == self.last_idx:
            return self.last
        if idx < self.pos or idx - self.pos > 30:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        while self.pos < idx:
            self.cap.grab()
            self.pos += 1
        ok, frame = self.cap.read()
        self.pos += 1
        if not ok:
            raise RuntimeError(f"Cannot read frame {idx}")
        self.last_idx, self.last = idx, frame
        return frame

    def release(self):
        self.cap.release()


# ============================================================
# SAMPLING HELPERS
# ============================================================

def sample_image(img, u, v, border_mode, border_value=0, chunk=30000):
    """Bilinear sampling with cv2.remap, chunked (remap dislikes huge 1-D maps)."""
    n = len(u)
    tail = img.shape[2:]
    out = np.empty((n,) + tail, dtype=img.dtype)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        r = cv2.remap(
            img,
            np.ascontiguousarray(u[s:e].reshape(-1, 1), dtype=np.float32),
            np.ascontiguousarray(v[s:e].reshape(-1, 1), dtype=np.float32),
            cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=border_value,
        )
        out[s:e] = r.reshape((e - s,) + tail)
    return out


def build_zbuffer(u, v, z, inside, width, height, s):
    zw, zh = width // s + 2, height // s + 2
    zb = np.full((zh, zw), 1e9, dtype=np.float32)
    idx = np.where(inside)[0]
    if len(idx) == 0:
        return zb
    order = idx[np.argsort(-z[idx])]
    zb[(v[order] / s).astype(np.int32), (u[order] / s).astype(np.int32)] = z[order]
    return cv2.erode(zb, np.ones((3, 3), np.uint8))


# ============================================================
# MASK -> SIGNED DISTANCE IMAGE
# ============================================================

def prepare_mask(mask):
    m = (mask > 127).astype(np.uint8) * 255
    if MASK_CLOSE > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_CLOSE, MASK_CLOSE))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.medianBlur(m, 5)
    inside = (m > 127).astype(np.uint8)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3)
    return (d_in - d_out).astype(np.float32), d_in.astype(np.float32)


# ============================================================
# STEP 1+2: HIGH-RES HULL  ->  CLEAN BINARY BODY
# ============================================================

def carve_body(voxels, grid, cameras, sdf_imgs):
    U = UPSCALE
    coords = voxels[:, :3].astype(np.int32)

    lo_c = np.maximum(coords.min(axis=0) - PAD_C, 0)
    hi_c = np.minimum(coords.max(axis=0) + PAD_C + 1, grid)      # exclusive
    size_c = hi_c - lo_c

    occ_c = np.zeros(tuple(size_c.tolist()), dtype=bool)
    l = coords - lo_c
    occ_c[l[:, 0], l[:, 1], l[:, 2]] = True
    occ_c = ndi.binary_dilation(occ_c, iterations=CAND_DILATE)

    # fine lattice: coarse voxel centres sit exactly on fine index c*U
    n_f = (size_c - 1) * U + 1
    ax = []
    for a in range(3):
        F = np.arange(n_f[a])                       # local fine index
        c = np.floor(F / U + 0.5).astype(np.int32)  # local coarse index
        ax.append(np.clip(c, 0, size_c[a] - 1))
    cand = occ_c[np.ix_(ax[0], ax[1], ax[2])]

    idx = np.argwhere(cand)
    if len(idx) == 0:
        return None
    step_f = (RANGE / (grid - 1) / U).astype(np.float32)
    world = BOUNDS_MIN + (idx + lo_c * U).astype(np.float32) * step_f

    vals = np.full((len(idx), len(CAMERAS)), -BIG, dtype=np.float32)
    for k, name in enumerate(CAMERAS):
        u, v, z = project(world, cameras[name])
        sdf = sdf_imgs[name]
        s = sample_image(sdf, u, v, cv2.BORDER_CONSTANT, -BIG)
        s[z <= 0] = -BIG
        vals[:, k] = s
    vals.sort(axis=1)
    fv = vals[:, min(CONSENSUS_INDEX, len(CAMERAS) - 1)] + HULL_BIAS

    f = np.full(tuple(n_f.tolist()), -BIG, dtype=np.float32)
    f[idx[:, 0], idx[:, 1], idx[:, 2]] = fv
    binary = f > 0

    # clean: close gaps, keep biggest body, fill inner holes
    binary = np.pad(binary, PAD_F)
    binary = ndi.binary_closing(binary, iterations=2)
    lab, n = ndi.label(binary)
    if n == 0:
        return None
    if n > 1:
        sizes = ndi.sum(binary, lab, range(1, n + 1))
        keep = np.where(sizes >= KEEP_FRACTION * sizes.max())[0] + 1
        binary = np.isin(lab, keep)
    binary = ndi.binary_fill_holes(binary)

    return binary, lo_c, hi_c, step_f


# ============================================================
# STEP 3+4: SURFACE + SMOOTHING
# ============================================================

def binary_to_mesh(binary, lo_c, grid):
    U = UPSCALE
    d_in = ndi.distance_transform_edt(binary)
    d_out = ndi.distance_transform_edt(~binary)
    sdf = ndi.gaussian_filter((d_in - d_out).astype(np.float32), SDF_SIGMA)
    if sdf.max() <= 0 or sdf.min() >= 0:
        return None

    verts, faces, _, _ = marching_cubes(sdf, level=0.0)
    verts = verts - PAD_F + lo_c * U                      # global fine index
    step_f = (RANGE / (grid - 1) / U).astype(np.float64)
    world = BOUNDS_MIN.astype(np.float64) + verts * step_f
    faces = faces.astype(np.int64)

    # make normals point outward (positive signed volume)
    a, b, c = world[faces[:, 0]], world[faces[:, 1]], world[faces[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    if vol < 0:
        faces = faces[:, ::-1].copy()
    return world, faces


def smoother(faces, n):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return diags(1.0 / deg) @ A


def taubin(verts, W):
    v = verts.copy()
    for _ in range(TAUBIN_ITERS):
        v = v + TAUBIN_LAMBDA * (W @ v - v)
        v = v + TAUBIN_MU * (W @ v - v)
    return v


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        for c in range(3):
            vn[:, c] += np.bincount(faces[:, k], weights=fn[:, c], minlength=len(verts))
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(ln, 1e-12)


# ============================================================
# STEP 5: COLOR THE MESH FROM THE VIDEO
# ============================================================

def color_mesh(verts, normals, W, cameras, vframes, d_imgs, tol):
    n = len(verts)
    acc = np.zeros((n, 3), dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    pts = verts.astype(np.float32)

    for name in CAMERAS:
        frame = vframes[name]
        d_img = d_imgs[name]
        h, w = d_img.shape

        u, v, z = project(pts, cameras[name])
        inside = (z > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if not inside.any():
            continue

        zb = build_zbuffer(u, v, z, inside, w, h, ZBUF_SCALE)
        ui = np.clip(np.rint(u).astype(np.int32), 0, w - 1)
        vi = np.clip(np.rint(v).astype(np.int32), 0, h - 1)
        visible = z <= zb[vi // ZBUF_SCALE, ui // ZBUF_SCALE] + tol

        d = d_img[vi, ui]
        view = CAMERA_POSITIONS[name] - pts
        view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
        cosang = np.sum(normals * view, axis=1)

        good = inside & visible & (d >= EDGE_MIN_DIST) & (cosang > MIN_COS)
        if not good.any():
            continue

        wt = np.zeros(n, dtype=np.float32)
        wt[good] = (cosang[good] ** VIEW_POWER) * (np.minimum(d[good], 10.0) / 10.0)

        col = sample_image(frame, u, v, cv2.BORDER_REPLICATE)
        col = col[:, ::-1].astype(np.float32)           # BGR -> RGB
        acc += col * wt[:, None]
        wsum += wt

    known = wsum > 1e-6
    col = np.zeros((n, 3), dtype=np.float32)
    col[known] = acc[known] / wsum[known, None]

    # fill never-seen areas by diffusing colors over the surface
    for _ in range(60):
        if known.all():
            break
        kf = known.astype(np.float32)
        avg = W @ (col * kf[:, None])
        cnt = W @ kf
        new = (~known) & (cnt > 1e-6)
        if not new.any():
            break
        col[new] = avg[new] / cnt[new, None]
        known |= new
    col[~known] = 128.0

    # blend camera seams
    for _ in range(COLOR_SMOOTH_ITERS):
        col = 0.5 * col + 0.5 * (W @ col)

    return np.clip(col, 0, 255).astype(np.uint8)


# ============================================================
# EXPORTS
# ============================================================

def write_ply(path, verts, normals, colors, faces):
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    va = np.empty(len(verts), dtype=vdt)
    va["x"], va["y"], va["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    va["nx"], va["ny"], va["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    va["r"], va["g"], va["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    fa = np.empty(len(faces), dtype=fdt)
    fa["n"] = 3
    fa["i"] = faces.astype(np.int32)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(va)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(fa)}\n"
        "property list uchar int vertex_indices\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(va.tobytes())
        f.write(fa.tobytes())


def clean_voxels(binary, lo_c, hi_c, grid, verts, colors):
    """Gap-free voxel shell for the old V4D player, colored from the mesh."""
    U = UPSCALE
    size_c = hi_c - lo_c
    sl = tuple(slice(PAD_F, PAD_F + (int(size_c[a]) - 1) * U + 1, U) for a in range(3))
    inside_c = binary[sl]
    shell = inside_c & ~ndi.binary_erosion(inside_c, iterations=SHELL_THICKNESS)
    pos = np.argwhere(shell)
    if len(pos) == 0:
        return np.zeros((0, 6), dtype=np.uint8)
    coords = pos + lo_c
    world = BOUNDS_MIN + coords.astype(np.float32) / (grid - 1) * RANGE
    _, nn = cKDTree(verts).query(world)
    out = np.empty((len(coords), 6), dtype=np.uint8)
    out[:, :3] = coords.astype(np.uint8)
    out[:, 3:] = colors[nn]
    return out


VIEWER_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Voxel Replay - Mesh Viewer</title>
<style>html,body{margin:0;height:100%;background:#14161b;overflow:hidden}
#hud{position:fixed;left:12px;top:10px;color:#cfd3dc;font:13px system-ui}</style>
<script type="importmap">{"imports":{
"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body><div id="hud">Loading...</div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {PLYLoader} from 'three/addons/loaders/PLYLoader.js';

const hud = document.getElementById('hud');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161b);
const camera = new THREE.PerspectiveCamera(35, innerWidth/innerHeight, 0.05, 50);
camera.up.set(0,0,1);
camera.position.set(0.0,-4.2,1.5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0,0,1.0);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(2,-3,4); scene.add(key);
const rim = new THREE.DirectionalLight(0xaaccff, 0.8); rim.position.set(-3,3,2); scene.add(rim);

const grid = new THREE.GridHelper(4, 16, 0x333844, 0x22262f);
grid.rotation.x = Math.PI/2; scene.add(grid);

const mat = new THREE.MeshStandardMaterial({vertexColors:true, roughness:0.85, metalness:0.0});
const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
scene.add(mesh);

const info = await (await fetch('frames.json')).json();
const loader = new PLYLoader();
const geos = [];
const c = new THREE.Color();
for (let i = 0; i < info.files.length; i++) {
  const g = await loader.loadAsync(info.files[i]);
  const col = g.attributes.color;
  for (let k = 0; k < col.count; k++) {
    c.fromBufferAttribute(col, k).convertSRGBToLinear();
    col.setXYZ(k, c.r, c.g, c.b);
  }
  geos.push(g);
  hud.textContent = 'Loading ' + (i+1) + '/' + info.files.length;
  if (i === 0) mesh.geometry = g;
}

let t0 = performance.now();
function loop(now) {
  const i = Math.floor(((now - t0) / 1000) * info.fps) % geos.length;
  mesh.geometry = geos[i];
  hud.textContent = 'frame ' + (i+1) + '/' + geos.length + '  (drag = orbit, wheel = zoom)';
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script></body></html>
"""


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print(" V4 SMOOTH HUMAN MESH RECONSTRUCTION")
    print("=" * 60 + "\n")

    print("Reading:", INPUT)
    version, gx, gy, gz, frame_count, fps, voxel_size, frames = read_v4d(INPUT)
    grid = np.array([gx, gy, gz], dtype=np.int32)
    print("Frames:", frame_count, "| Grid:", gx, gy, gz)

    readers, cameras = {}, {}
    for name in CAMERAS:
        r = VideoReader(VIDEO_ROOT / f"{name}.mp4")
        readers[name] = r
        cameras[name] = make_camera(name, r.width, r.height)
        print(name, ":", r.width, "x", r.height, "|", round(r.fps, 2), "FPS")
    max_src = min(r.count for r in readers.values()) - 1

    tol = float(np.max(RANGE / (grid - 1))) * 3.0
    blur_k = FRAME_BLUR if FRAME_BLUR % 2 == 1 else FRAME_BLUR + 1

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    end = frame_count if FRAME_END is None else min(FRAME_END, frame_count)
    todo = list(range(FRAME_START, end, FRAME_STEP))

    out_frames, mesh_files = [], []
    warned_mask = False
    t_start = time.time()

    for n_done, i in enumerate(todo):
        timestamp, voxels = frames[i]
        t0 = time.time()

        if len(voxels) == 0:
            out_frames.append((timestamp, voxels))
            continue

        try:
            src = min(int(round(timestamp * SOURCE_FPS)), max_src)

            vframes, sdf_imgs, d_imgs = {}, {}, {}
            for name in CAMERAS:
                frame = readers[name].get(src)
                vframes[name] = (cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
                                 if blur_k > 1 else frame)
                mp = MASK_ROOT / name / f"{src:04d}.png"
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
                if mask is None:
                    if not warned_mask:
                        print("  WARNING: mask missing, shape will be less accurate:", mp)
                        warned_mask = True
                    mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
                elif mask.shape != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                sdf_imgs[name], d_imgs[name] = prepare_mask(mask)

            body = carve_body(voxels, grid, cameras, sdf_imgs)
            if body is None:
                raise RuntimeError("empty body after carving")
            binary, lo_c, hi_c, _ = body

            m = binary_to_mesh(binary, lo_c, grid)
            if m is None:
                raise RuntimeError("no surface found")
            verts, faces = m

            W = smoother(faces, len(verts))
            verts = taubin(verts, W)
            normals = vertex_normals(verts, faces)

            colors = color_mesh(verts, normals, W, cameras, vframes, d_imgs, tol)

            if EXPORT_MESH:
                name = f"frame_{i:04d}.ply"
                write_ply(MESH_DIR / name, verts.astype(np.float32),
                          normals.astype(np.float32), colors, faces)
                mesh_files.append(name)

            vox = clean_voxels(binary, lo_c, hi_c, grid, verts, colors)
            out_frames.append((timestamp, vox))

            print(f"[{n_done + 1:03d}/{len(todo):03d}] frame {i:03d} | "
                  f"{len(verts)} verts | {len(faces)} tris | "
                  f"{len(vox)} voxels | {time.time() - t0:.1f}s")

        except Exception:
            print(f"[{n_done + 1:03d}/{len(todo):03d}] frame {i:03d} FAILED, keeping original")
            traceback.print_exc()
            out_frames.append((timestamp, voxels))

    for r in readers.values():
        r.release()

    if WRITE_VIEWER and mesh_files:
        (MESH_DIR / "frames.json").write_text(
            json.dumps({"fps": float(fps) / FRAME_STEP if fps > 0 else 24.0,
                        "files": mesh_files}))
        (MESH_DIR / "viewer.html").write_text(VIEWER_HTML, encoding="utf-8")

    print("\nWriting cleaned V4D...")
    write_v4d(OUTPUT, version, gx, gy, gz, fps, voxel_size, out_frames)
    if PLAYER_OUTPUT.parent.exists():
        shutil.copy2(OUTPUT, PLAYER_OUTPUT)
        print("Player updated:", PLAYER_OUTPUT)

    print("\n" + "=" * 60)
    print(f" DONE in {(time.time() - t_start) / 60:.1f} min")
    print("=" * 60)
    print("V4D  :", OUTPUT)
    if mesh_files:
        print("Mesh :", MESH_DIR)
        print("View : cd", MESH_DIR, "&& python -m http.server 8000")
        print("       then open http://localhost:8000/viewer.html")
    print()


if __name__ == "__main__":
    main()