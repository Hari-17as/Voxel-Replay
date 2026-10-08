from flask import Flask, request, render_template_string, send_from_directory
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "ai_uploads")
OUTPUT_DIR = os.path.join(BASE, "ai_outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading YOLO segmentation model...")
model = YOLO("yolo11n-seg.pt")
print("YOLO READY!")

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Voxel Replay - AI Vision</title>
<style>
body {
    margin:0;
    background:#07111f;
    color:white;
    font-family:Arial;
}
.container {
    max-width:1100px;
    margin:40px auto;
    padding:30px;
}
h1 {
    color:#42e8ff;
    font-size:38px;
}
.card {
    background:#101d30;
    padding:25px;
    border-radius:18px;
    margin-top:20px;
}
input {
    padding:12px;
}
button {
    padding:13px 25px;
    background:#19cfff;
    border:0;
    border-radius:10px;
    font-weight:bold;
    cursor:pointer;
}
.grid {
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:20px;
    margin-top:25px;
}
img {
    width:100%;
    border-radius:12px;
}
.label {
    color:#42e8ff;
    font-size:18px;
    margin-bottom:8px;
}
</style>
</head>

<body>
<div class="container">

<h1>Voxel Replay — AI Vision Analyzer</h1>

<div class="card">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="video" accept="video/*" required>
<button type="submit">ANALYZE VIDEO</button>
</form>
</div>

{% if results %}
<div class="card">
<h2>Computer Vision Results</h2>
<p>Frames analyzed: {{ count }}</p>

{% for r in results %}
<div class="grid">

<div>
<div class="label">Original Frame {{ loop.index }}</div>
<img src="/outputs/{{ r.original }}">
</div>

<div>
<div class="label">YOLO Detection + Segmentation</div>
<img src="/outputs/{{ r.annotated }}">
</div>

<div>
<div class="label">Segmentation Mask</div>
<img src="/outputs/{{ r.mask }}">
</div>

</div>
{% endfor %}

</div>
{% endif %}

</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        video = request.files.get("video")

        if not video:
            return "No video uploaded"

        job = str(uuid.uuid4())[:8]

        video_path = os.path.join(UPLOAD_DIR, job + ".mp4")
        video.save(video_path)

        cap = cv2.VideoCapture(video_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            return "Could not read video"

        positions = [
            int(total * 0.15),
            int(total * 0.35),
            int(total * 0.55),
            int(total * 0.75),
            int(total * 0.90)
        ]

        for i, pos in enumerate(positions):

            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()

            if not ok:
                continue

            original_name = f"{job}_{i}_original.jpg"
            annotated_name = f"{job}_{i}_ai.jpg"
            mask_name = f"{job}_{i}_mask.png"

            original_path = os.path.join(OUTPUT_DIR, original_name)
            annotated_path = os.path.join(OUTPUT_DIR, annotated_name)
            mask_path = os.path.join(OUTPUT_DIR, mask_name)

            cv2.imwrite(original_path, frame)

            prediction = model(frame, verbose=False)[0]

            annotated = prediction.plot()
            cv2.imwrite(annotated_path, annotated)

            mask = frame.copy()

            if prediction.masks is not None:

                mask_data = prediction.masks.data.cpu().numpy()

                combined = mask_data.max(axis=0)

                combined = (combined * 255).astype("uint8")

                combined = cv2.resize(
                    combined,
                    (frame.shape[1], frame.shape[0])
                )

                cv2.imwrite(mask_path, combined)

            else:
                blank = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )
                blank[:] = 0
                cv2.imwrite(mask_path, blank)

            results.append({
                "original": original_name,
                "annotated": annotated_name,
                "mask": mask_name
            })

        cap.release()

    return render_template_string(
        HTML,
        results=results,
        count=len(results)
    )


@app.route("/outputs/<filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )