# Suggested 8-slide presentation

## 1. Problem
Normal video gives a fixed viewpoint.
Voxel Replay aims to make dynamic scenes traversable.

## 2. Idea
Synchronized multi-view video -> time-varying 3D -> custom V4D -> browser player.

## 3. Reconstruction
Calibration + foreground masks + silhouette carving.

## 4. V4D format
Header + timeline/frame data + sparse occupied voxels.
Explain future keyframe/delta compression.

## 5. Interactive player
Play/pause, timeline, free camera movement, different viewpoints.

## 6. Evaluation
Hold one camera out of reconstruction.
Render from its pose.
Compare using PSNR and SSIM.

## 7. Performance
Show actual measured:
- voxel resolution
- FPS/frame time
- load time
- file size
- memory/VRAM

## 8. Limitations and future work
Visual-hull concavity limits.
Appearance limits.
Future GPU optimization and Gaussian-based representations.
