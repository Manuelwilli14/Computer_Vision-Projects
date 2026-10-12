"""
Import a Stray Scanner recording (iPhone / iPad with LiDAR) into the project format.

Stray Scanner recording (input):
    rgb.mp4            color video (1920x1440)
    depth/000000.png   16-bit depth in millimeters (256x192), one per video frame
    confidence/        depth confidence per pixel (0 = low, 1 = medium, 2 = high)
    odometry.csv       per frame: camera position (m), rotation (quaternion), fx, fy, cx, cy

Dataset folder (output), usable by pipeline.py --data_dir:
    color/000000.jpg   color image, upright, same size as the depth image
    depth/000000.png   16-bit depth in millimeters (0 = no depth / low confidence)
    camera.json        intrinsics of the images + depth scale
    poses.txt          one camera-to-world 4x4 matrix per frame (16 values per line),
                       camera axes x right, y down, z forward; world y axis = up

Frames are rotated upright when the phone was held in portrait (detected from the
camera poses), so that SegFormer sees the street the right way up.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# Rotation of the image -> (cv2 rotation code, rotation of the camera axes: new = R @ old)
ROTATIONS = {
    "none": (None, np.eye(3)),
    "clockwise": (cv2.ROTATE_90_CLOCKWISE, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
    "counterclockwise": (cv2.ROTATE_90_COUNTERCLOCKWISE, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),
    "180": (cv2.ROTATE_180, np.diag([-1, -1, 1])),
}


def read_odometry(path):
    #Rows of odometry.csv: timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...
    odo = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=range(13))
    poses = []
    for x, y, z, qx, qy, qz, qw in odo[:, 2:9]:
        pose = np.eye(4)
        pose[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
        pose[:3, 3] = [x, y, z]
        poses.append(pose)
    return np.array(poses), odo[:, 9:13]  # camera-to-world poses, (fx, fy, cx, cy)


def detect_rotation(poses):
    """
    Image rotation that puts the scene upright: the world "up" axis (y) seen from the
    camera must point to the top of the image (camera axis -y).
    """
    up = np.median([pose[:3, :3].T @ [0, 1, 0] for pose in poses], axis=0)
    if abs(up[1]) >= abs(up[0]):
        return "none" if up[1] < 0 else "180"
    return "clockwise" if up[0] < 0 else "counterclockwise"


def rotate_intrinsics(fx, fy, cx, cy, width, height, rotation):
    #Intrinsics of the rotated image (width, height = size before rotation)
    if rotation == "clockwise":
        return fy, fx, height - 1 - cy, cx
    if rotation == "counterclockwise":
        return fy, fx, cy, width - 1 - cx
    if rotation == "180":
        return fx, fy, width - 1 - cx, height - 1 - cy
    return fx, fy, cx, cy


def import_stray(stray_dir, output_dir, step=10, min_confidence=1, width=640):
    stray_dir, output_dir = Path(stray_dir), Path(output_dir)
    (output_dir / "color").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)

    poses, intrinsics = read_odometry(stray_dir / "odometry.csv")
    video = cv2.VideoCapture(str(stray_dir / "rgb.mp4"))
    video_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = round(width * video_h / video_w)  # same aspect ratio as the video (4:3)

    n_depth = len(list((stray_dir / "depth").glob("*.png")))
    frames = range(0, min(len(poses), n_depth), step)
    rotation = detect_rotation(poses[list(frames)])
    cv2_rotation, axes_rotation = ROTATIONS[rotation]
    print(f"{stray_dir.name}: {len(frames)} frames kept (1 out of {step}), rotation: {rotation}")

    saved_poses = []
    frame_idx = -1
    for i in frames:
        # Read the video up to frame i (the video is decoded sequentially)
        while frame_idx < i:
            ok, color = video.read()
            frame_idx += 1
            if not ok:
                break
        if not ok:
            print(f"Video ends at frame {frame_idx}: stopping")
            break

        depth = cv2.imread(str(stray_dir / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(stray_dir / "confidence" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        depth[confidence < min_confidence] = 0  # LiDAR depth is extrapolated (unreliable) there

        color = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
        # nearest: no interpolation between foreground and background depths
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        if cv2_rotation is not None:
            color = cv2.rotate(color, cv2_rotation)
            depth = cv2.rotate(depth, cv2_rotation)

        name = f"{len(saved_poses):06d}"
        cv2.imwrite(str(output_dir / "color" / f"{name}.jpg"), color, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(output_dir / "depth" / f"{name}.png"), depth)

        # Camera axes of the rotated image: p_new = R p_old -> pose_new = pose_old @ R^T
        rotated = np.eye(4)
        rotated[:3, :3] = axes_rotation.T
        saved_poses.append(poses[i] @ rotated)

    # Intrinsics (median over the kept frames, they vary slightly with autofocus),
    # rescaled from the video resolution to the output resolution, then rotated
    fx, fy, cx, cy = np.median(intrinsics[list(frames)], axis=0)
    scale = width / video_w
    fx, fy, cx, cy = fx * scale, fy * scale, (cx + 0.5) * scale - 0.5, (cy + 0.5) * scale - 0.5
    fx, fy, cx, cy = rotate_intrinsics(fx, fy, cx, cy, width, height, rotation)
    out_w, out_h = (height, width) if rotation in ("clockwise", "counterclockwise") else (width, height)

    camera = {"width": out_w, "height": out_h, "fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": 1000.0}
    (output_dir / "camera.json").write_text(json.dumps(camera, indent=2))
    np.savetxt(output_dir / "poses.txt", np.array(saved_poses).reshape(-1, 16), fmt="%.6f")

    print(f"Dataset written to {output_dir}: {len(saved_poses)} frames {out_w}x{out_h}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import a Stray Scanner (iPhone LiDAR) recording")
    parser.add_argument("stray_dir", help="Stray Scanner recording folder (contains rgb.mp4, depth/, odometry.csv)")
    parser.add_argument("output_dir", help="Dataset folder to create (e.g. data/my_street)")
    parser.add_argument("--step", type=int, default=10, help="Keep 1 frame out of STEP (the video is 60 fps)")
    parser.add_argument("--min-confidence", type=int, default=1, choices=[0, 1, 2],
                        help="Drop depth pixels below this LiDAR confidence (0 = keep all)")
    parser.add_argument("--width", type=int, default=640, help="Width of the images before rotation")
    args = parser.parse_args()

    import_stray(args.stray_dir, args.output_dir, args.step, args.min_confidence, args.width)
