"""Compute PSNR and SSIM between matching image folders.

Usage:
    python evaluate_images.py real reconstructed

Both folders must contain matching image filenames.
"""

from pathlib import Path
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def main():
    if len(sys.argv) != 3:
        print("Usage: python evaluate_images.py real reconstructed")
        raise SystemExit(1)

    real_dir = Path(sys.argv[1])
    pred_dir = Path(sys.argv[2])

    files = sorted(
        p for p in real_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    if not files:
        raise RuntimeError("No images found in real directory.")

    psnr_values = []
    ssim_values = []

    for real_path in files:
        pred_path = pred_dir / real_path.name
        if not pred_path.exists():
            print(f"Skipping missing prediction: {pred_path.name}")
            continue

        real = cv2.imread(str(real_path), cv2.IMREAD_COLOR)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_COLOR)

        if real is None or pred is None:
            print(f"Skipping unreadable pair: {real_path.name}")
            continue

        if real.shape != pred.shape:
            pred = cv2.resize(pred, (real.shape[1], real.shape[0]))

        psnr_values.append(psnr(real, pred))

        real_gray = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY)
        pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

        ssim_values.append(
            structural_similarity(real_gray, pred_gray, data_range=255)
        )

    if not psnr_values:
        raise RuntimeError("No matching image pairs were evaluated.")

    print(f"Images evaluated: {len(psnr_values)}")
    print(f"Mean PSNR: {np.mean(psnr_values):.4f} dB")
    print(f"Mean SSIM: {np.mean(ssim_values):.4f}")


if __name__ == "__main__":
    main()
