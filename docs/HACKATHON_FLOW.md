# Hackathon flow

## Demo story

1. Show synchronized multi-view footage.
2. Explain calibration and foreground masks.
3. Reconstruct a bounded 3D voxel volume at each timestamp.
4. Store the time-varying scene in `.v4d`.
5. Open the `.v4d` in the browser player.
6. Play, pause, scrub, and move the camera.
7. Show held-out-camera rendering.
8. Report measured PSNR/SSIM.
9. Report measured file size/load time/FPS/memory.
10. Explain limitations and future Gaussian-based representation.

## Claims to avoid

Do not claim:
- perfect geometry
- complete hidden-surface recovery
- guaranteed real-time performance at arbitrary resolution
- specific PSNR/SSIM/FPS without measurement
- Gaussian Splatting results before actually running it

## Optional advanced branch

If the baseline is stable, investigate GPU-accelerated segmentation/reconstruction and Gaussian Splatting. Keep the baseline available as the reliable demo.
