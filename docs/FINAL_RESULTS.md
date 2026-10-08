# Voxel Replay — Current Measured Results

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

Base V4D size: **1055268 bytes**

Color V4D size: **1055268 bytes**

Color size change: **0.0%**

## Next experimental items

1. Replace synthetic capture with real synchronized videos.
2. Use real foreground segmentation.
3. Improve voxel color visibility/occlusion handling.
4. Add temporal keyframe/delta storage.
5. Benchmark browser FPS and scrub latency on the actual laptop.
6. Run PSNR/SSIM using a held-out RGB camera, not only silhouette IoU.
