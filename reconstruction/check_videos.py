import cv2
import glob
import os

files = glob.glob(r"..\data\videos\*.mp4")

print()
print("========== REAL VIDEO CHECK ==========")
print()

if not files:
    print("NO MP4 FILES FOUND")
else:
    for path in sorted(files):
        cap = cv2.VideoCapture(path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frames / fps if fps > 0 else 0

        print(os.path.basename(path))
        print("  Resolution :", width, "x", height)
        print("  FPS        :", round(fps, 2))
        print("  Frames     :", frames)
        print("  Duration   :", round(duration, 2), "seconds")
        print()

        cap.release()

print("======================================")