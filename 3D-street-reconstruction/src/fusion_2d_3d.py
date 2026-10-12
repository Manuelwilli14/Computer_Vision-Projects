"""
2D -> 3D Fusion: projects 2D semantic segmentation masks onto the reconstructed
3D point cloud to obtain a 3D scene where each point is labeled
(road, car, building, vegetation, sky, other).

Principle:
- For each RGB-D frame used during reconstruction, we have:
    - the depth image (used to generate the 3D points)
    - the corresponding 2D semantic mask (computed via semantic_segmentation.py),
      with the same size as the depth image
- Each pixel (u, v) with a valid depth is back-projected into a 3D point that
  keeps the label of pixel (u, v) in the mask.
- The frames are placed in the scene like in reconstruction.py (camera poses of the
  dataset, or ICP), then the points are downsampled by voxel with a majority vote
  on the labels.
- Each point is colored according to its super-category for the final visualization.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from config import SUPER_CATEGORIES, SUPER_CATEGORY_COLORS, DEPTH_SCALE, DEPTH_TRUNC
from reconstruction import CAMERA_TO_DISPLAY, get_intrinsics, list_images, load_camera, register_and_merge

SUPER_CAT_LIST = list(SUPER_CATEGORIES.keys())
# Row i = RGB color (0-1) of the super-category with ID i
SUPER_CAT_PALETTE = np.array([SUPER_CATEGORY_COLORS[c] for c in SUPER_CAT_LIST]) / 255.0


def project_labels_to_points(depth_path, label_map, intrinsic=None, depth_scale=DEPTH_SCALE, pose=None):
    """
    Back-projects the pixels of a frame that have a valid depth and returns their
    3D points (N, 3) with their super-category IDs (N,) read from label_map.

    label_map : array (H, W) of integers (indices into SUPER_CAT_LIST), generated
                by semantic_segmentation.segment_image and saved as .npy
    pose      : camera-to-world 4x4 matrix (see reconstruction.rgbd_to_pointcloud)
    """
    depth_np = np.asarray(o3d.io.read_image(str(depth_path))) / depth_scale

    h, w = depth_np.shape
    if label_map.shape != (h, w):
        raise ValueError(
            f"Label map {label_map.shape} does not match depth image {(h, w)} for {depth_path}. "
            "Re-run semantic_segmentation.py to regenerate the .labels.npy files."
        )
    if intrinsic is None:
        intrinsic = get_intrinsics(w, h)

    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    # Back-project every valid pixel (u, v) at once
    v, u = np.nonzero((depth_np > 0) & (depth_np <= DEPTH_TRUNC))
    z = depth_np[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=1)

    # same transformation as in reconstruction.rgbd_to_pointcloud
    transform = CAMERA_TO_DISPLAY if pose is None else pose
    points = points @ transform[:3, :3].T + transform[:3, 3]
    return points, label_map[v, u]


def voxel_down_sample_labels(points, labels, voxel_size):
    """
    Voxel downsampling of labeled points: one point per voxel, at the mean position of
    its points, with the MAJORITY label. (Open3D's voxel_down_sample would average the
    class colors, giving colors that belong to no class.)
    """
    if len(points) == 0:
        return points, labels

    voxels = np.floor(points / voxel_size).astype(np.int64)
    voxels -= voxels.min(axis=0)
    dims = voxels.max(axis=0) + 1
    keys = (voxels[:, 0] * dims[1] + voxels[:, 1]) * dims[2] + voxels[:, 2]
    _, voxel_ids, counts = np.unique(keys, return_inverse=True, return_counts=True)

    centers = np.stack(
        [np.bincount(voxel_ids, weights=points[:, k]) for k in range(3)], axis=1
    ) / counts[:, None]
    n_classes = len(SUPER_CAT_LIST)
    votes = np.bincount(voxel_ids * n_classes + labels, minlength=len(counts) * n_classes)
    return centers, votes.reshape(-1, n_classes).argmax(axis=1)


def labels_to_pointcloud(points, labels):
    #Point cloud colored by super-category
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(SUPER_CAT_PALETTE[labels])
    return pcd


def fuse_sequence(color_dir, depth_dir, label_dir, output_path, voxel_size=0.03):
    """
    Builds a fused semantic point cloud for an entire sequence.

    label_dir must contain one .labels.npy file per frame, with the same
    base name as the color images (output of semantic_segmentation.segment_image).
    The frames are placed like in reconstruction.build_scene_from_rgbd_sequence:
    with the camera poses of the dataset when available, otherwise with ICP.
    """
    color_files = list_images(color_dir)
    depth_files = list_images(depth_dir)
    if len(color_files) != len(depth_files):
        raise ValueError(
            f"Inconsistent number of files: {len(color_files)} color, {len(depth_files)} depth"
        )

    # Each color image is paired with its labels by name (other files in label_dir are ignored)
    label_files = [Path(label_dir) / f"{f.stem}.labels.npy" for f in color_files]
    missing = [f.name for f in label_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} label file(s) missing in {label_dir} (e.g. {missing[0]}): "
            "run semantic_segmentation.py on every frame first"
        )

    intrinsic, depth_scale, poses = load_camera(Path(color_dir).parent)
    if poses is not None and len(poses) != len(color_files):
        raise ValueError(f"{len(poses)} poses in poses.txt for {len(color_files)} images")

    frames = []  # (points, labels) of each frame
    for i, (color_f, depth_f, label_f) in enumerate(zip(color_files, depth_files, label_files)):
        pose = None if poses is None else poses[i]
        points, labels = project_labels_to_points(depth_f, np.load(label_f), intrinsic, depth_scale, pose)
        frames.append(voxel_down_sample_labels(points, labels, voxel_size))
        print(f"Semantic frame added: {color_f.name} -> {len(frames[-1][0])} points")

    if poses is None:
        # Same ICP alignment as reconstruction.build_scene_from_rgbd_sequence
        # (ICP only uses geometry; it moves the cloud of each frame in place)
        pointclouds = [labels_to_pointcloud(points, labels) for points, labels in frames]
        register_and_merge(pointclouds, voxel_size)
        frames = [(np.array(pcd.points), labels) for pcd, (_, labels) in zip(pointclouds, frames)]

    points = np.concatenate([points for points, _ in frames])
    labels = np.concatenate([labels for _, labels in frames])
    if len(frames) > 1:
        # Overlapping frames give several points for the same surface: keep one per voxel
        points, labels = voxel_down_sample_labels(points, labels, voxel_size)

    scene = labels_to_pointcloud(points, labels)
    scene, _ = scene.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud(str(output_path), scene)
    print(f"Semantic point cloud saved: {output_path} ({len(scene.points)} points)")

    print_class_statistics(scene)
    return scene


def print_class_statistics(pcd):
    """Prints the percentage of points per semantic class."""
    colors = np.asarray(pcd.colors)
    total = len(colors)
    print("\nSemantic distribution of the point cloud:")
    for super_cat, rgb in SUPER_CATEGORY_COLORS.items():
        target = np.array(rgb) / 255.0
        mask = np.all(np.isclose(colors, target, atol=1e-3), axis=1)
        pct = 100 * mask.sum() / total if total > 0 else 0
        print(f"  - {super_cat:12s}: {pct:5.1f}%  ({mask.sum()} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D->3D fusion of semantic labels")
    parser.add_argument("--color", required=True, help="Color images directory")
    parser.add_argument("--depth", required=True, help="Depth images directory")
    parser.add_argument("--labels", required=True, help="Directory of .labels.npy files")
    parser.add_argument("--output", required=True, help="Output .ply file")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args()

    fuse_sequence(args.color, args.depth, args.labels, args.output, args.voxel_size)
