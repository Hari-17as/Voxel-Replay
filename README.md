# Voxel Replay

A practical prototype for a traversable dynamic-scene format (4D scene reconstruction).

## What this repository contains

- `reconstruction/` — Python synthetic generator, `.v4d` writer/reader, silhouette-carving prototype, and evaluation utilities.
- `player/` — Vite + Three.js browser player with playback, scrubbing, and free camera movement.
- `data/` — expected structure for synchronized multi-view videos, masks, and calibration.
- `exports/` — generated `.v4d` files (generated locally; large outputs should not be committed).
- `evaluation/` — evaluation scripts and results.
- `docs/` — project/presentation notes.

## Important scope

The current first milestone is a **synthetic format/player demo**. It is not a real reconstruction result.

The real reconstruction pipeline requires synchronized multi-view footage, camera calibration, and foreground masks. The repository includes a small silhouette-carving implementation that can be tested once those inputs exist.

## Offline workflow

After the required Python packages and npm packages have been installed once, the project can run locally without cloud services.

Do not commit:
- `.venv/`
- `player/node_modules/`
- large raw videos
- generated `.v4d` files unless intentionally selected
- private datasets

## Quick start

### Python

From the project root with the virtual environment activated:

```cmd
cd reconstruction
python generate_demo.py
cd ..
```

This creates:

```text
exports\demo.v4d
```

### Player

In another CMD window:

```cmd
cd C:\Users\hari1\voxel-replay\player
npm install
npm run dev
```

Open the displayed localhost URL.

## Development order

1. Synthetic `.v4d` generation.
2. Browser `.v4d` decoding.
3. Batched voxel rendering.
4. Play/pause/scrubbing.
5. 6-DOF camera.
6. Calibration + silhouette carving.
7. Color assignment.
8. Sparse representation.
9. Keyframes/deltas.
10. Held-out-camera evaluation with PSNR/SSIM.
11. Laptop benchmarks.
12. Optional GPU/AI/Gaussian experiments.

## Reproducibility

Record the actual:
- laptop CPU/RAM/GPU/driver
- browser version
- voxel resolution
- number of cameras
- frame rate
- reconstruction settings
- file size
- load time
- FPS/frame time
- memory/VRAM when measured
- PSNR/SSIM

Never put invented benchmark values in the final presentation.
