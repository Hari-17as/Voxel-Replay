from pathlib import Path
import sys
import shutil

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy import ndimage


# ============================================================
# IMPORT V4D FORMAT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAIN_RECONSTRUCTION = PROJECT_ROOT / "reconstruction"

sys.path.insert(
    0,
    str(MAIN_RECONSTRUCTION)
)

from v4d_format import V4DHeader, write_v4d


# ============================================================
# PATHS
# ============================================================

ROOT = PROJECT_ROOT

VIDEOS = ROOT / "data" / "videos"

OUT = ROOT / "exports" / "reconstructed_color.v4d"

MASK_ROOT = ROOT / "data" / "real_masks_presentation"


# ============================================================
# CAMERAS
# ============================================================

CAM_NAMES = [
    "cam0",
    "cam1",
    "cam2"
]


# Process every second frame.
STEP = 2


# ============================================================
# VOXEL GRID
# ============================================================

GRID = (
    64,
    64,
    96
)


# ============================================================
# WORLD BOUNDS
# ============================================================

BOUNDS_MIN = np.array(
    [
        -1.30,
        -1.30,
        0.00
    ],
    dtype=np.float64
)

BOUNDS_MAX = np.array(
    [
        1.30,
        1.30,
        2.35
    ],
    dtype=np.float64
)


# ============================================================
# CAMERA LOOK-AT
# ============================================================

def look_at(
    position,
    target=np.array(
        [0.0, 0.0, 1.15],
        dtype=np.float64
    )
):

    position = np.asarray(
        position,
        dtype=np.float64
    )

    target = np.asarray(
        target,
        dtype=np.float64
    )

    forward = target - position

    forward /= (
        np.linalg.norm(forward) + 1e-9
    )

    world_up = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64
    )

    right = np.cross(
        forward,
        world_up
    )

    if np.linalg.norm(right) < 1e-8:

        right = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64
        )

    right /= (
        np.linalg.norm(right) + 1e-9
    )

    up = np.cross(
        right,
        forward
    )

    up /= (
        np.linalg.norm(up) + 1e-9
    )

    R = np.vstack(
        [
            right,
            up,
            forward
        ]
    )

    t = -R @ position.reshape(
        3,
        1
    )

    return R, t


# ============================================================
# CAMERA MODEL
# ============================================================

def camera_for(
    name,
    width,
    height
):

    positions = {

        "cam0":
            np.array(
                [
                    0.0,
                    -4.2,
                    1.60
                ],
                dtype=np.float64
            ),

        "cam1":
            np.array(
                [
                    4.2,
                    0.0,
                    1.60
                ],
                dtype=np.float64
            ),

        "cam2":
            np.array(
                [
                    0.0,
                    4.2,
                    1.60
                ],
                dtype=np.float64
            )
    }

    position = positions[name]

    R, t = look_at(
        position
    )

    focal = (
        0.95 *
        max(
            width,
            height
        )
    )

    K = np.array(
        [
            [
                focal,
                0,
                width / 2
            ],
            [
                0,
                focal,
                height / 2
            ],
            [
                0,
                0,
                1
            ]
        ],
        dtype=np.float64
    )

    return {
        "K": K,
        "R": R,
        "t": t,
        "w": width,
        "h": height
    }


# ============================================================
# 3D TO 2D PROJECTION
# ============================================================

def project(
    points,
    camera
):

    camera_points = (
        camera["R"] @ points.T
    ) + camera["t"]

    z = camera_points[2]

    safe_z = np.where(
        np.abs(z) < 1e-8,
        1e-8,
        z
    )

    u = (
        camera["K"][0, 0]
        *
        camera_points[0]
        /
        safe_z
    ) + camera["K"][0, 2]

    v = (
        camera["K"][1, 1]
        *
        camera_points[1]
        /
        safe_z
    ) + camera["K"][1, 2]

    return (
        np.column_stack(
            [
                u,
                v
            ]
        ),
        z
    )


# ============================================================
# READ VIDEO FRAME
# ============================================================

def read_frame(
    capture,
    index
):

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        index
    )

    ok, frame = capture.read()

    if not ok:

        return None

    return frame


# ============================================================
# PERSON MASK
# ============================================================

def person_mask(
    model,
    frame
):

    result = model(
        frame,
        imgsz=960,
        conf=0.18,
        classes=[0],
        verbose=False
    )[0]

    if (
        result.masks is None
        or
        len(result.masks.data) == 0
    ):

        return np.zeros(
            frame.shape[:2],
            dtype=np.uint8
        )

    masks = (
        result
        .masks
        .data
        .cpu()
        .numpy()
    )

    areas = []

    for mask in masks:

        areas.append(
            float(
                (
                    mask > 0.45
                ).sum()
            )
        )

    best_index = int(
        np.argmax(areas)
    )

    mask = (
        masks[best_index]
        >
        0.45
    ).astype(
        np.uint8
    ) * 255

    mask = cv2.resize(
        mask,
        (
            frame.shape[1],
            frame.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # Close small holes.

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2
    )

    # Remove small noise.

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    # Small expansion to protect thin limbs.

    expand_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    mask = cv2.dilate(
        mask,
        expand_kernel,
        iterations=1
    )

    return mask


# ============================================================
# VOXEL GRID
# ============================================================

def voxel_grid():

    gx, gy, gz = GRID

    xs = np.linspace(
        BOUNDS_MIN[0],
        BOUNDS_MAX[0],
        gx
    )

    ys = np.linspace(
        BOUNDS_MIN[1],
        BOUNDS_MAX[1],
        gy
    )

    zs = np.linspace(
        BOUNDS_MIN[2],
        BOUNDS_MAX[2],
        gz
    )

    xx, yy, zz = np.meshgrid(
        xs,
        ys,
        zs,
        indexing="ij"
    )

    return np.column_stack(
        [
            xx.ravel(),
            yy.ravel(),
            zz.ravel()
        ]
    )


# ============================================================
# VISUAL HULL
# ============================================================

def carve(
    points,
    cameras,
    masks
):

    votes = np.zeros(
        len(points),
        dtype=np.uint8
    )

    for name, camera in cameras.items():

        uv, z = project(
            points,
            camera
        )

        u = np.rint(
            uv[:, 0]
        ).astype(
            np.int32
        )

        v = np.rint(
            uv[:, 1]
        ).astype(
            np.int32
        )

        inside = (
            (z > 0)
            &
            (u >= 0)
            &
            (u < camera["w"])
            &
            (v >= 0)
            &
            (v < camera["h"])
        )

        indices = np.where(
            inside
        )[0]

        if len(indices) == 0:

            continue

        visible = (
            masks[name][
                v[indices],
                u[indices]
            ]
            >
            0
        )

        votes[
            indices
        ] += visible.astype(
            np.uint8
        )

    # Majority hull.
    #
    # Two of three cameras must agree.

    keep = votes >= 2

    return points[keep]


# ============================================================
# 3D BODY CLEANUP
# ============================================================

def improve_volume(
    points
):

    if len(points) == 0:

        return points

    gx, gy, gz = GRID

    normalized = (
        points - BOUNDS_MIN
    ) / (
        BOUNDS_MAX -
        BOUNDS_MIN
    )

    coordinates = np.rint(
        normalized
        *
        (
            np.asarray(GRID)
            -
            1
        )
    ).astype(
        np.int32
    )

    coordinates = np.clip(
        coordinates,
        0,
        np.asarray(GRID) - 1
    )

    volume = np.zeros(
        GRID,
        dtype=bool
    )

    volume[
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2]
    ] = True

    # 3D closing.

    volume = ndimage.binary_closing(
        volume,
        structure=np.ones(
            (3, 3, 3),
            dtype=bool
        ),
        iterations=1
    )

    # Fill internal holes.

    volume = ndimage.binary_fill_holes(
        volume
    )

    # Remove tiny disconnected components.

    labels, count = ndimage.label(
        volume
    )

    if count > 0:

        sizes = np.bincount(
            labels.ravel()
        )

        cleaned = np.zeros_like(
            volume
        )

        for label_id in range(
            1,
            len(sizes)
        ):

            if sizes[label_id] >= 8:

                cleaned |= (
                    labels
                    ==
                    label_id
                )

        volume = cleaned

    # Convert back to world coordinates.

    result = np.argwhere(
        volume
    )

    xs, ys, zs = result.T

    xs = (
        xs
        /
        (gx - 1)
    ) * (
        BOUNDS_MAX[0]
        -
        BOUNDS_MIN[0]
    ) + BOUNDS_MIN[0]

    ys = (
        ys
        /
        (gy - 1)
    ) * (
        BOUNDS_MAX[1]
        -
        BOUNDS_MIN[1]
    ) + BOUNDS_MIN[1]

    zs = (
        zs
        /
        (gz - 1)
    ) * (
        BOUNDS_MAX[2]
        -
        BOUNDS_MIN[2]
    ) + BOUNDS_MIN[2]

    return np.column_stack(
        [
            xs,
            ys,
            zs
        ]
    )


# ============================================================
# COLOR
# ============================================================

def color_voxels(
    points,
    cameras,
    frames
):

    sums = np.zeros(
        (
            len(points),
            3
        ),
        dtype=np.float64
    )

    counts = np.zeros(
        len(points),
        dtype=np.uint8
    )

    for name, camera in cameras.items():

        image = frames[name]

        uv, z = project(
            points,
            camera
        )

        u = np.rint(
            uv[:, 0]
        ).astype(
            np.int32
        )

        v = np.rint(
            uv[:, 1]
        ).astype(
            np.int32
        )

        valid = (
            (z > 0)
            &
            (u >= 0)
            &
            (u < camera["w"])
            &
            (v >= 0)
            &
            (v < camera["h"])
        )

        indices = np.where(
            valid
        )[0]

        if len(indices) == 0:

            continue

        # IMPORTANT:
        # Work only with the valid projected pixels.

        valid_u = u[indices]

        valid_v = v[indices]

        pixels = image[
            valid_v,
            valid_u
        ]

        b = pixels[
            :, 0
        ].astype(
            np.int16
        )

        g = pixels[
            :, 1
        ].astype(
            np.int16
        )

        r = pixels[
            :, 2
        ].astype(
            np.int16
        )

        brightness = (
            r.astype(np.float32)
            +
            g.astype(np.float32)
            +
            b.astype(np.float32)
        ) / 3.0

        chroma = (
            np.maximum.reduce(
                [
                    r,
                    g,
                    b
                ]
            )
            -
            np.minimum.reduce(
                [
                    r,
                    g,
                    b
                ]
            )
        )

        foreground_like = (
            (chroma > 10)
            |
            (brightness < 170)
        )

        selected_indices = indices[
            foreground_like
        ]

        if len(
            selected_indices
        ) == 0:

            continue

        # FIX:
        # Use valid_v/valid_u rather than the full
        # v/u arrays.

        selected_pixels = image[
            valid_v[foreground_like],
            valid_u[foreground_like]
        ]

        selected_rgb = (
            selected_pixels[
                :,
                ::-1
            ].astype(
                np.float64
            )
        )

        sums[
            selected_indices
        ] += selected_rgb

        counts[
            selected_indices
        ] += 1

    rgb = np.zeros(
        (
            len(points),
            3
        ),
        dtype=np.uint8
    )

    good = counts > 0

    if good.any():

        rgb[good] = np.rint(
            sums[good]
            /
            counts[
                good,
                None
            ]
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

    # Dark warm fallback.
    # No white ghost colour.

    rgb[~good] = np.array(
        [
            70,
            55,
            48
        ],
        dtype=np.uint8
    )

    return rgb


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("========================================")
    print(" PRESENTATION 3D HUMAN RECONSTRUCTION")
    print("========================================")
    print("")

    print(
        "Project:",
        ROOT
    )

    print(
        "Videos:",
        VIDEOS
    )

    print(
        "Output:",
        OUT
    )

    print("")

    # --------------------------------------------------------
    # CHECK VIDEO DIRECTORY
    # --------------------------------------------------------

    if not VIDEOS.exists():

        raise SystemExit(
            f"VIDEO FOLDER NOT FOUND: {VIDEOS}"
        )

    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO segmentation..."
    )

    model = YOLO(
        "yolo11n-seg.pt"
    )

    cuda_available = (
        torch.cuda.is_available()
    )

    print(
        "Device:",
        "NVIDIA CUDA"
        if cuda_available
        else "CPU"
    )

    # --------------------------------------------------------
    # OPEN VIDEOS
    # --------------------------------------------------------

    captures = {}

    metadata = {}

    for name in CAM_NAMES:

        path = VIDEOS / (
            name + ".mp4"
        )

        if not path.exists():

            raise SystemExit(
                f"Missing video: {path}"
            )

        capture = cv2.VideoCapture(
            str(path)
        )

        if not capture.isOpened():

            raise SystemExit(
                f"Could not open: {path}"
            )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        captures[name] = capture

        metadata[name] = (
            frame_count,
            fps,
            width,
            height
        )

        print(
            f"{name}: "
            f"{width}x{height} "
            f"{fps:.2f} FPS "
            f"{frame_count} frames"
        )

    # --------------------------------------------------------
    # COMMON FRAME COUNT
    # --------------------------------------------------------

    total_frames = min(
        data[0]
        for data in metadata.values()
    )

    # Use the slowest camera FPS for timing.

    fps = min(
        data[1]
        for data in metadata.values()
    )

    frame_indices = list(
        range(
            0,
            total_frames,
            STEP
        )
    )

    # --------------------------------------------------------
    # CAMERAS
    # --------------------------------------------------------

    cameras = {}

    for name in CAM_NAMES:

        cameras[name] = camera_for(
            name,
            metadata[name][2],
            metadata[name][3]
        )

    # --------------------------------------------------------
    # VOXEL CANDIDATES
    # --------------------------------------------------------

    grid = voxel_grid()

    print("")

    print(
        "Grid:",
        GRID
    )

    print(
        "Candidate voxels:",
        len(grid)
    )

    print(
        "Output frames:",
        len(frame_indices)
    )

    print("")

    MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    frames_out = []

    previous_points = None

    # ========================================================
    # PROCESS FRAMES
    # ========================================================

    for output_index, source_index in enumerate(
        frame_indices
    ):

        masks = {}

        color_frames = {}

        # ----------------------------------------------------
        # READ THREE CAMERAS
        # ----------------------------------------------------

        for name in CAM_NAMES:

            frame = read_frame(
                captures[name],
                source_index
            )

            if frame is None:

                raise SystemExit(
                    f"Failed frame "
                    f"{source_index} "
                    f"in {name}"
                )

            color_frames[name] = frame

            mask_path = (
                MASK_ROOT
                /
                name
                /
                f"{source_index:04d}.png"
            )

            mask_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            if mask_path.exists():

                mask = cv2.imread(
                    str(mask_path),
                    cv2.IMREAD_GRAYSCALE
                )

            else:

                mask = person_mask(
                    model,
                    frame
                )

                cv2.imwrite(
                    str(mask_path),
                    mask
                )

            masks[name] = mask

        # ----------------------------------------------------
        # VISUAL HULL
        # ----------------------------------------------------

        points = carve(
            grid,
            cameras,
            masks
        )

        # ----------------------------------------------------
        # 3D CLEANUP
        # ----------------------------------------------------

        points = improve_volume(
            points
        )

        # ----------------------------------------------------
        # TEMPORAL STABILITY
        # ----------------------------------------------------

        if (
            previous_points is not None
            and
            len(points)
            <
            len(previous_points) * 0.55
        ):

            print(
                "  temporal stabilization applied"
            )

            points = previous_points

        # ----------------------------------------------------
        # COLOR
        # ----------------------------------------------------

        if len(points) > 0:

            colors = color_voxels(
                points,
                cameras,
                color_frames
            )

            normalized = (
                points
                -
                BOUNDS_MIN
            ) / (
                BOUNDS_MAX
                -
                BOUNDS_MIN
            )

            coordinates = np.rint(
                normalized
                *
                (
                    np.asarray(GRID)
                    -
                    1
                )
            ).astype(
                np.int32
            )

            coordinates = np.clip(
                coordinates,
                0,
                np.asarray(GRID) - 1
            )

            voxels = []

            for coordinate, color in zip(
                coordinates,
                colors
            ):

                x, y, z = coordinate

                r, g, b = color

                voxels.append(
                    (
                        int(x),
                        int(y),
                        int(z),
                        int(r),
                        int(g),
                        int(b)
                    )
                )

        else:

            voxels = []

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp = (
            source_index
            /
            fps
        )

        frames_out.append(
            (
                timestamp,
                voxels
            )
        )

        previous_points = points

        print(
            f"{output_index + 1:03d}/"
            f"{len(frame_indices):03d} "
            f"t={timestamp:5.2f}s "
            f"voxels={len(voxels)}"
        )

    # ========================================================
    # CLOSE VIDEOS
    # ========================================================

    for capture in captures.values():

        capture.release()

    # ========================================================
    # V4D HEADER
    # ========================================================

    voxel_size = float(
        np.mean(
            (
                BOUNDS_MAX
                -
                BOUNDS_MIN
            )
            /
            (
                np.asarray(GRID)
                -
                1
            )
        )
    )

    output_fps = (
        fps
        /
        STEP
    )

    header = V4DHeader(
        GRID[0],
        GRID[1],
        GRID[2],
        len(frames_out),
        output_fps,
        voxel_size
    )

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    write_v4d(
        OUT,
        header,
        frames_out
    )

    # ========================================================
    # COPY TO PLAYER
    # ========================================================

    player_output = (
        ROOT
        /
        "player"
        /
        "public"
        /
        "reconstructed_color.v4d"
    )

    player_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        OUT,
        player_output
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("")
    print("========================================")
    print(" RECONSTRUCTION COMPLETE")
    print("========================================")
    print("")

    print(
        "V4D:",
        OUT
    )

    print(
        "Player:",
        player_output
    )

    print(
        "Frames:",
        len(frames_out)
    )

    print(
        "FPS:",
        output_fps
    )

    print(
        "Grid:",
        GRID
    )

    print(
        "Voxel size:",
        voxel_size
    )

    print("")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()