"""Standalone V4D benchmark. Run from reconstruction."""
from pathlib import Path
import sys, time, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4d_format import read_all_frames

ROOT = Path(__file__).resolve().parents[1]
for name in ["reconstructed.v4d", "reconstructed_color.v4d"]:
    p = ROOT / "exports" / name
    if not p.exists():
        continue
    t0 = time.perf_counter()
    h, frames = read_all_frames(p)
    elapsed = time.perf_counter() - t0
    voxels = sum(len(v) for _, v in frames)
    print(f"{name}:")
    print(f"  size: {p.stat().st_size:,} bytes")
    print(f"  frames: {h.frame_count}")
    print(f"  grid: {h.grid_x} x {h.grid_y} x {h.grid_z}")
    print(f"  voxel records: {voxels:,}")
    print(f"  decode time: {elapsed:.6f} s")
