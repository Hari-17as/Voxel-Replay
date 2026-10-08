"""
VOXEL REPLAY - ASYNCHRONOUS RECONSTRUCTION & UPLOAD API SERVER
"""

import sys
import os
import shutil
import threading
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

ROOT = Path(r"C:\Users\hari1\voxel-replay")
sys.path.insert(0, str(ROOT / "V2_HIGH_QUALITY" / "reconstruction"))

from real4d_pipeline_v2 import run_pipeline

app = Flask(__name__)
CORS(app)

VIDEO_DIR = ROOT / "data" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

MASK_DIR = ROOT / "data" / "real_masks_universal"

# Thread-safe global execution tracker
STATE_LOCK = threading.Lock()
job_state = {
    "processing": False,
    "stage": "idle",
    "progress": 0,
    "current_frame": 0,
    "total_frames": 0,
    "error": None,
    "outputs": {
        "vmesh": "/reconstructed_mesh.vmesh",
        "v4d": "/reconstructed_color.v4d"
    }
}

def pipeline_progress_bridge(stage, progress, current_frame, total):
    with STATE_LOCK:
        job_state["stage"] = stage
        job_state["progress"] = progress
        job_state["current_frame"] = current_frame
        job_state["total_frames"] = total

def background_reconstruction_task():
    try:
        pipeline_progress_bridge(stage="initializing", progress=2, current_frame=0, total=0)
        run_pipeline(progress_callback=pipeline_progress_bridge)
        with STATE_LOCK:
            job_state["processing"] = False
            job_state["stage"] = "completed"
            job_state["progress"] = 100
            job_state["error"] = None
    except Exception as e:
        with STATE_LOCK:
            job_state["processing"] = False
            job_state["stage"] = "failed"
            job_state["error"] = str(e)
            print(f"[RECONSTRUCTION ERROR] {e}", file=sys.stderr)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "hardware": "NVIDIA RTX 4050 / CUDA Active",
        "service": "Voxel Replay 4D Mesh Pipeline"
    }), 200

@app.route("/status", methods=["GET"])
def status():
    with STATE_LOCK:
        return jsonify(job_state), 200

@app.route("/upload", methods=["POST"])
def upload():
    with STATE_LOCK:
        if job_state["processing"]:
            return jsonify({
                "status": "error",
                "message": "A reconstruction task is already in progress. Check /status."
            }), 409

    for cam in ["cam0", "cam1", "cam2"]:
        if cam not in request.files:
            return jsonify({
                "status": "error",
                "message": f"Missing multipart payload stream: {cam}"
            }), 400

    # Purge existing stale masks
    if MASK_DIR.exists():
        try:
            shutil.rmtree(MASK_DIR)
        except Exception:
            pass
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    # Ingest and normalize uploaded video files
    for cam in ["cam0", "cam1", "cam2"]:
        file_obj = request.files[cam]
        dest_path = VIDEO_DIR / f"{cam}.mp4"
        file_obj.save(str(dest_path))

    with STATE_LOCK:
        job_state["processing"] = True
        job_state["stage"] = "uploaded"
        job_state["progress"] = 0
        job_state["current_frame"] = 0
        job_state["total_frames"] = 0
        job_state["error"] = None

    worker = threading.Thread(target=background_reconstruction_task, daemon=True)
    worker.start()

    return jsonify({
        "status": "accepted",
        "message": "Videos synchronized and ingested. Reconstruction launched.",
        "status_endpoint": "/status"
    }), 202

if __name__ == "__main__":
    print("Voxel Replay Upload Server online at http://127.0.0.1:8000")
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)