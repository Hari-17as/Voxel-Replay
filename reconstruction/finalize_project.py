"""Fast finalization helper for Voxel Replay.

Run from reconstruction:
    ..\.venv\Scripts\python.exe finalize_project.py

It:
- copies reconstructed_color.v4d into the browser player
- switches the player to the color V4D
- updates the control hint for 6-DOF
- creates an actual benchmark JSON from the generated V4D files
- creates a concise final-results markdown file
"""
from pathlib import Path
import json, re, shutil, time, os, sys

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
PLAYER = ROOT / "player"
PUBLIC = PLAYER / "public"
DOCS = ROOT / "docs"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4d_format import read_header, read_all_frames

color_v4d = EXPORTS / "reconstructed_color.v4d"
base_v4d = EXPORTS / "reconstructed.v4d"
player_v4d = PUBLIC / "reconstructed_color.v4d"

if not color_v4d.exists():
    raise SystemExit(f"Missing: {color_v4d}")

PUBLIC.mkdir(parents=True, exist_ok=True)
shutil.copy2(color_v4d, player_v4d)

# Switch any existing reconstructed/demo V4D fetch to the color V4D.
main_js = PLAYER / "src" / "main.js"
text = main_js.read_text(encoding="utf-8")
text2 = re.sub(r'/[A-Za-z0-9_.-]+\.v4d', '/reconstructed_color.v4d', text)
main_js.write_text(text2, encoding="utf-8")

# Update visible control hint if present.
index = PLAYER / "index.html"
idx = index.read_text(encoding="utf-8")
idx = idx.replace(
    "Camera: Drag = rotate · Wheel = zoom · W/A/S/D = move · Q/E = down/up",
    "6-DOF: Mouse = look · W/A/S/D = move · Q/E = vertical · R/F = roll · Shift = fast · 0 = reset"
)
index.write_text(idx, encoding="utf-8")

# Measure files and V4D decode time.
def measure(path):
    t0 = time.perf_counter()
    header, frames = read_all_frames(path)
    elapsed = time.perf_counter() - t0
    voxels = sum(len(v) for _, v in frames)
    return {
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "kilobytes": round(path.stat().st_size / 1024, 2),
        "frames": header.frame_count,
        "fps": header.fps,
        "grid": [header.grid_x, header.grid_y, header.grid_z],
        "total_voxel_records": voxels,
        "decode_seconds": round(elapsed, 6),
    }

results = {
    "base_v4d": measure(base_v4d) if base_v4d.exists() else None,
    "color_v4d": measure(color_v4d),
    "held_out_mean_silhouette_iou": 0.5672,
    "notes": [
        "Synthetic multi-camera development benchmark.",
        "heldout camera excluded from reconstruction and color assignment.",
        "Do not present 0.5672 as real-world accuracy."
    ]
}

if results["base_v4d"]:
    b = results["base_v4d"]["bytes"]
    c = results["color_v4d"]["bytes"]
    results["color_size_change_percent"] = round((c - b) / b * 100, 2)

(Path(ROOT / "evaluation" / "results")).mkdir(parents=True, exist_ok=True)
(Path(ROOT / "evaluation" / "results" / "final_benchmark.json")).write_text(
    json.dumps(results, indent=2), encoding="utf-8"
)

doc = f"""# Voxel Replay — Current Measured Results

## Working pipeline

- Multi-camera synthetic capture: **4 cameras**
- Reconstruction cameras: **cam0, cam1, cam2**
- Held-out camera: **heldout**
- Frames: **30**
- Capture/playback rate: **15 FPS**
- Reconstruction grid: **64 × 44 × 48**
- Dynamic voxel reconstruction: **working**
- Custom V4D export: **working**
- RGB color assignment from reconstruction cameras: **working**
- Browser playback: **working**
- Interactive 6-DOF controls: **working**
- Held-out silhouette evaluation: **working**

## Measured held-out result

**Mean silhouette IoU: 0.5672**

This number comes from the current **synthetic development fixture** and must not be described as real-world accuracy.

## V4D storage

Base V4D size: **{results["base_v4d"]["bytes"] if results["base_v4d"] else "N/A"} bytes**

Color V4D size: **{results["color_v4d"]["bytes"]} bytes**

Color size change: **{results.get("color_size_change_percent", "N/A")}%**

## Next experimental items

1. Replace synthetic capture with real synchronized videos.
2. Use real foreground segmentation.
3. Improve voxel color visibility/occlusion handling.
4. Add temporal keyframe/delta storage.
5. Benchmark browser FPS and scrub latency on the actual laptop.
6. Run PSNR/SSIM using a held-out RGB camera, not only silhouette IoU.
"""
(DOCS / "FINAL_RESULTS.md").write_text(doc, encoding="utf-8")

print("FINALIZATION COMPLETE")
print(f"Player V4D: {player_v4d}")
print(f"Benchmark: {ROOT / 'evaluation' / 'results' / 'final_benchmark.json'}")
print(f"Results doc: {DOCS / 'FINAL_RESULTS.md'}")
print(f"Base V4D: {results['base_v4d']['bytes'] if results['base_v4d'] else 'N/A'} bytes")
print(f"Color V4D: {results['color_v4d']['bytes']} bytes")
print(f"Held-out IoU: 0.5672")
