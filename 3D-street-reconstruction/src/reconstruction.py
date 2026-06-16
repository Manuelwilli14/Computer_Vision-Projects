"""
3D reconstruction of a street scene using Open3D.

Two supported modes:
- RGB-D: based on aligned (color, depth) pairs + camera poses
   -> TSDF integration (open3d.pipelines.integration) or point cloud merging
- Multi-view without depth: sparse point cloud via point matching
   (simplified mode; for best results, use COLMAP beforehand and then
   load the dense point cloud directly here)

Output: a .ply file (point cloud or mesh)
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2

from config import DEFAULT_INTRINSICS, DEPTH_SCALE, DEPTH_TRUNC


def get_intrinsics(width=None, height=None):
    w = width or DEFAULT_INTRINSICS["width"]
    h = height or DEFAULT_INTRINSICS["height"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h,
        DEFAULT_INTRINSICS["fx"], DEFAULT_INTRINSICS["fy"],
        DEFAULT_INTRINSICS["cx"], DEFAULT_INTRINSICS["cy"],
    )
    return intrinsic


def rgbd_to_pointcloud(color_path, depth_path, intrinsic=None):
    #Convert a pair (color, depth) into a 3D point cloud
    color_raw = o3d.io.read_image(str(color_path))
    depth_raw = o3d.io.read_image(str(depth_path))

    if intrinsic is None:
        color_np = np.asarray(color_raw)
        h, w = color_np.shape[:2]
        intrinsic = get_intrinsics(w, h)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    #Open3D reference point: Rotate to obtain a standard “world” orientation
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def register_and_merge(pointclouds, voxel_size=0.05):
    """
    Roughly align and merge multiple point clouds using ICP.
    pointclouds: a list of o3d.geometry.PointCloud objects (already in nearby coordinate systems)
    """
    if len(pointclouds) == 1:
        return pointclouds[0]

    merged = pointclouds[0]
    for i in range(1, len(pointclouds)):
        source = pointclouds[i].voxel_down_sample(voxel_size)
        target = merged.voxel_down_sample(voxel_size)

        source.estimate_normals()
        target.estimate_normals()

        reg = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance=voxel_size * 2,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        pointclouds[i].transform(reg.transformation)
        merged = merged + pointclouds[i]

    return merged


def build_scene_from_rgbd_sequence(color_dir, depth_dir, voxel_size=0.03):
    """
    Build a 3D point cloud of the complete scene from a
    sequence of color+depth images. The files must have
    the same name in color_dir and depth_dir, sorted alphabetically
    """
    color_files = sorted(Path(color_dir).glob("*"))
    depth_files = sorted(Path(depth_dir).glob("*"))

    if len(color_files) != len(depth_files):
        raise ValueError("The number of color and depth images do not match")

    pointclouds = []
    for color_f, depth_f in zip(color_files, depth_files):
        pcd = rgbd_to_pointcloud(color_f, depth_f)
        pcd = pcd.voxel_down_sample(voxel_size)
        pointclouds.append(pcd)
        print(f"Frame ajoutée : {color_f.name} -> {len(pcd.points)} points")

    scene = register_and_merge(pointclouds, voxel_size)
    scene, _ = scene.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return scene


def reconstruct_mesh(pointcloud, depth=9):
    #Surface Reconstruction (Poisson) from a Point Cloud
    pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pointcloud.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pointcloud, depth=depth
    )

    #Removes low-density areas (artifacts)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Reconstruction of a Street Scene (Open3D)")
    parser.add_argument("--color", required=True, help="Color Images Folder")
    parser.add_argument("--depth", required=True, help="Depth Image Folder")
    parser.add_argument("--output", required=True, help="Output .ply file")
    parser.add_argument("--mesh", action="store_true", help="Also generates a Poisson mesh")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args()

    scene_pcd = build_scene_from_rgbd_sequence(args.color, args.depth, args.voxel_size)
    o3d.io.write_point_cloud(args.output, scene_pcd)
    print(f"Point cloud saved : {args.output} ({len(scene_pcd.points)} points)")

    if args.mesh:
        mesh = reconstruct_mesh(scene_pcd)
        mesh_path = str(Path(args.output).with_suffix(".mesh.ply"))
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Mesh saved : {mesh_path}")
