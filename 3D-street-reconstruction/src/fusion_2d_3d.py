"""
2D -> 3D Fusion: projects 2D semantic segmentation masks onto the reconstructed
3D point cloud to obtain a 3D scene where each point is labeled
(road, car, building, vegetation, sky, other).

Principle:
- For each RGB-D frame used during reconstruction, we have:
    - the color image (and therefore pixel coordinates (u, v))
    - the depth image (used to generate the 3D points)
    - the corresponding 2D semantic mask (computed via semantic_segmentation.py)
- Since create_from_rgbd_image generates points in the same order as pixels
  (ignoring zero-depth pixels), we can recover for each 3D point the original
  pixel (u, v) and therefore its semantic label.
- Each point is colored according to its super-category for the final visualization.
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from config import SUPER_CATEGORIES, SUPER_CATEGORY_COLORS, DEPTH_SCALE, DEPTH_TRUNC
from reconstruction import get_intrinsics

SUPER_CAT_LIST = list(SUPER_CATEGORIES.keys())


def project_labels_to_pointcloud(color_path, depth_path, label_map, intrinsic=None):
    """
    Recreates the point cloud for a frame and assigns, point by point,
    the color of the corresponding super-category (from label_map).

    label_map : array (H, W) of integers (indices into SUPER_CAT_LIST), generated
                by semantic_segmentation.segment_image and saved as .npy
    """
    color_raw = o3d.io.read_image(str(color_path))
    depth_raw = o3d.io.read_image(str(depth_path))
    depth_np = np.asarray(depth_raw)

    h, w = depth_np.shape
    if intrinsic is None:
        intrinsic = get_intrinsics(w, h)

    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    points = []
    colors = []

    for v in range(h):
        for u in range(w):
            z = depth_np[v, u] / DEPTH_SCALE
            if z <= 0 or z > DEPTH_TRUNC:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

            label_id = label_map[v, u]
            super_cat = SUPER_CAT_LIST[label_id]
            color = np.array(SUPER_CATEGORY_COLORS[super_cat]) / 255.0
            colors.append(color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # same transformation as in reconstruction.rgbd_to_pointcloud
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def fuse_sequence(color_dir, depth_dir, label_dir, output_path, voxel_size=0.03):
    """
    Builds a fused semantic point cloud for an entire sequence.

    label_dir must contain one .labels.npy file per frame, with the same
    base name as the color/depth images (output of
    semantic_segmentation.segment_image).
    """
    color_files = sorted(Path(color_dir).glob("*"))
    depth_files = sorted(Path(depth_dir).glob("*"))
    label_files = sorted(Path(label_dir).glob("*.labels.npy"))

    if not (len(color_files) == len(depth_files) == len(label_files)):
        raise ValueError(
            f"Inconsistent number of files: "
            f"{len(color_files)} color, {len(depth_files)} depth, "
            f"{len(label_files)} labels"
        )

    pointclouds = []
    for color_f, depth_f, label_f in zip(color_files, depth_files, label_files):
        label_map = np.load(label_f)
        pcd = project_labels_to_pointcloud(color_f, depth_f, label_map)
        pcd = pcd.voxel_down_sample(voxel_size)
        pointclouds.append(pcd)
        print(f"Semantic frame added: {color_f.name} -> {len(pcd.points)} points")

    scene = pointclouds[0]
    for pcd in pointclouds[1:]:
        scene += pcd

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
