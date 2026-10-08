from pathlib import Path
import json
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(
    r"C:\Users\hari1\voxel-replay"
)

VIDEOS = ROOT / "data" / "videos"

OUTPUT = (
    ROOT
    / "player"
    / "public"
    / "motion.json"
)


# ============================================================
# INPUT VIDEOS
# ============================================================

CAMERAS = [
    "cam0",
    "cam1",
    "cam2"
]


# ============================================================
# PROCESSING
# ============================================================

STEP = 2

MODEL_NAME = "yolo11n-pose.pt"


# ============================================================
# YOLO COCO KEYPOINTS
# ============================================================
#
# 0  nose
# 1  left_eye
# 2  right_eye
# 3  left_ear
# 4  right_ear
# 5  left_shoulder
# 6  right_shoulder
# 7  left_elbow
# 8  right_elbow
# 9  left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle
#
# ============================================================


# ============================================================
# USEFUL BODY JOINTS
# ============================================================

JOINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]


# ============================================================
# READ VIDEO
# ============================================================

def open_video(name):

    path = VIDEOS / f"{name}.mp4"

    if not path.exists():

        raise RuntimeError(
            f"VIDEO NOT FOUND:\n{path}"
        )

    cap = cv2.VideoCapture(
        str(path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"COULD NOT OPEN:\n{path}"
        )

    frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    return cap, frames, fps, width, height


# ============================================================
# NORMALIZE PERSON
# ============================================================

def normalize_pose(keypoints):

    kp = np.asarray(
        keypoints,
        dtype=np.float32
    )

    if kp.shape[0] < 17:

        return None

    # --------------------------------------------------------
    # SHOULDER CENTER
    # --------------------------------------------------------

    shoulder_center = (
        kp[5] +
        kp[6]
    ) / 2.0

    # --------------------------------------------------------
    # HIP CENTER
    # --------------------------------------------------------

    hip_center = (
        kp[11] +
        kp[12]
    ) / 2.0

    # --------------------------------------------------------
    # BODY CENTER
    # --------------------------------------------------------

    center = (
        shoulder_center +
        hip_center
    ) / 2.0

    kp = kp - center

    # --------------------------------------------------------
    # BODY HEIGHT
    # --------------------------------------------------------

    top = kp[0, 1]

    bottom = max(
        kp[15, 1],
        kp[16, 1]
    )

    body_height = (
        abs(bottom - top)
    )

    if body_height < 1e-6:

        return None

    kp /= body_height

    return kp


# ============================================================
# SELECT MAIN PERSON
# ============================================================

def get_main_person(result):

    if (
        result.keypoints is None
        or
        len(result.keypoints.data) == 0
    ):

        return None

    keypoints = (
        result
        .keypoints
        .data
        .cpu()
        .numpy()
    )

    if len(keypoints) == 0:

        return None

    # Select person with largest visible body.

    best_index = 0

    best_score = -1

    for i, person in enumerate(
        keypoints
    ):

        visible = (
            person[:, 2] > 0.35
        )

        score = int(
            visible.sum()
        )

        if score > best_score:

            best_score = score

            best_index = i

    if best_score < 5:

        return None

    return keypoints[
        best_index
    ]


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video(
    model,
    name
):

    print("")
    print(
        "Processing:",
        name
    )

    cap, frame_count, fps, width, height = (
        open_video(name)
    )

    print(
        f"Resolution: "
        f"{width}x{height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {frame_count}"
    )

    motion = []

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:

            break

        if (
            frame_index % STEP
            != 0
        ):

            frame_index += 1

            continue

        result = model(
            frame,
            imgsz=640,
            conf=0.20,
            verbose=False
        )[0]

        person = get_main_person(
            result
        )

        if person is None:

            normalized = None

        else:

            normalized = normalize_pose(
                person
            )

        if normalized is None:

            motion.append(None)

        else:

            # Only keep the 17 COCO points.

            motion.append(
                normalized[
                    :, :2
                ].tolist()
            )

        if (
            len(motion) % 20
            == 0
        ):

            print(
                f"  frames processed: "
                f"{len(motion)}"
            )

        frame_index += 1

    cap.release()

    print(
        "Extracted:",
        len(motion),
        "motion frames"
    )

    return {
        "name": name,
        "fps": fps / STEP,
        "frames": motion
    }


# ============================================================
# FILL MISSING FRAMES
# ============================================================

def fill_missing(
    frames
):

    if not frames:

        return []

    result = list(frames)

    # Find first valid frame.

    first = None

    for frame in result:

        if frame is not None:

            first = frame

            break

    if first is None:

        raise RuntimeError(
            "No valid human pose detected."
        )

    # Fill beginning.

    for i in range(
        len(result)
    ):

        if result[i] is not None:

            break

        result[i] = first

    # Forward fill.

    previous = first

    for i in range(
        len(result)
    ):

        if result[i] is None:

            result[i] = previous

        else:

            previous = result[i]

    return result


# ============================================================
# RESAMPLE MOTION
# ============================================================

def resample(
    frames,
    target_count
):

    if len(frames) == 0:

        return []

    if len(frames) == target_count:

        return frames

    source = np.asarray(
        frames,
        dtype=np.float32
    )

    old_count = len(source)

    result = np.zeros(
        (
            target_count,
            17,
            2
        ),
        dtype=np.float32
    )

    old_positions = np.linspace(
        0,
        1,
        old_count
    )

    new_positions = np.linspace(
        0,
        1,
        target_count
    )

    for joint in range(17):

        for axis in range(2):

            result[
                :,
                joint,
                axis
            ] = np.interp(
                new_positions,
                old_positions,
                source[
                    :,
                    joint,
                    axis
                ]
            )

    return result


# ============================================================
# SMOOTH MOTION
# ============================================================

def smooth_motion(
    motion,
    strength=0.35
):

    arr = np.asarray(
        motion,
        dtype=np.float32
    )

    if len(arr) < 3:

        return arr

    output = arr.copy()

    for i in range(
        1,
        len(arr) - 1
    ):

        output[i] = (
            arr[i] * (1.0 - strength)
            +
            (
                arr[i - 1]
                +
                arr[i + 1]
            )
            * (strength / 2.0)
        )

    return output


# ============================================================
# COMBINE THREE PEOPLE
# ============================================================

def combine_motion(
    motions
):

    lengths = [
        len(m)
        for m in motions
    ]

    target_count = min(
        lengths
    )

    print("")
    print(
        "Common animation frames:",
        target_count
    )

    normalized = []

    for motion in motions:

        filled = fill_missing(
            motion
        )

        resampled = resample(
            filled,
            target_count
        )

        normalized.append(
            np.asarray(
                resampled,
                dtype=np.float32
            )
        )

    # --------------------------------------------------------
    # AVERAGE BODY MOVEMENT
    # --------------------------------------------------------

    combined = np.mean(
        np.stack(
            normalized,
            axis=0
        ),
        axis=0
    )

    # --------------------------------------------------------
    # SMOOTH
    # --------------------------------------------------------

    combined = smooth_motion(
        combined,
        strength=0.30
    )

    return combined


# ============================================================
# CONVERT 2D MOTION TO SIMPLE 3D
# ============================================================

def create_3d_motion(
    motion
):

    motion = np.asarray(
        motion,
        dtype=np.float32
    )

    frames = []

    for frame in motion:

        points = []

        for x, y in frame:

            # Screen coordinates:
            #
            # x -> horizontal
            # y -> vertical
            #
            # Convert to character coordinates.

            X = float(x)

            Y = float(-y)

            # Approximate depth.
            #
            # The videos are different people and do not
            # provide true 3D depth, so depth is estimated
            # from body structure.

            Z = 0.0

            points.append(
                [
                    X,
                    Y,
                    Z
                ]
            )

        # ----------------------------------------------------
        # ADD DEPTH FROM LIMB POSITION
        # ----------------------------------------------------

        # Arms.

        for a, b in [
            (7, 8),
            (9, 10),
            (13, 14),
            (15, 16)
        ]:

            if (
                a < len(points)
                and
                b < len(points)
            ):

                dx = (
                    points[b][0]
                    -
                    points[a][0]
                )

                points[a][2] = (
                    -dx * 0.25
                )

                points[b][2] = (
                    dx * 0.25
                )

        frames.append(
            points
        )

    return frames


# ============================================================
# WRITE MOTION JSON
# ============================================================

def write_output(
    motion,
    fps
):

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data = {

        "format":
            "VOXEL-REPLAY-MOTION-1",

        "description":
            "Single canonical character motion generated from three different-person videos.",

        "source_videos": [
            "cam0.mp4",
            "cam1.mp4",
            "cam2.mp4"
        ],

        "fps":
            float(fps),

        "joint_names": [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ],

        "frames":
            motion
    }

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            separators=(
                ",",
                ":"
            )
        )

    print("")
    print(
        "Motion saved:"
    )

    print(
        OUTPUT
    )

    print(
        "Frames:",
        len(motion)
    )

    print(
        "FPS:",
        fps
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print(
        "========================================"
    )
    print(
        " SINGLE CHARACTER MOTION PIPELINE"
    )
    print(
        "========================================"
    )
    print("")

    print(
        "Videos:",
        VIDEOS
    )

    print(
        "Output:",
        OUTPUT
    )

    print("")

    # --------------------------------------------------------
    # LOAD POSE MODEL
    # --------------------------------------------------------

    print(
        "Loading YOLO pose model..."
    )

    model = YOLO(
        MODEL_NAME
    )

    print(
        "Pose model loaded."
    )

    # --------------------------------------------------------
    # EXTRACT THREE VIDEOS
    # --------------------------------------------------------

    results = []

    for name in CAMERAS:

        result = extract_video(
            model,
            name
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # COMBINE MOVEMENT
    # --------------------------------------------------------

    motions = [
        result["frames"]
        for result in results
    ]

    combined = combine_motion(
        motions
    )

    # --------------------------------------------------------
    # 3D MOTION
    # --------------------------------------------------------

    motion_3d = create_3d_motion(
        combined
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = min(
        result["fps"]
        for result in results
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    write_output(
        motion_3d,
        fps
    )

    print("")
    print(
        "========================================"
    )
    print(
        " DONE"
    )
    print(
        "========================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()