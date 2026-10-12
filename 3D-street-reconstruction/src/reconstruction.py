"""
3D reconstruction of a street scene using Open3D, from a sequence of aligned
(color, depth) pairs.

Each frame is back-projected into a point cloud, then the frames are placed in a
common coordinate system:
- with the camera poses of the dataset (poses.txt, e.g. recorded by an iPhone with
  Stray Scanner, see import_stray.py), when available
- otherwise by aligning them with ICP

Output: a .ply file (point cloud or mesh)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


from config import DEFAULT_INTRINSICS, DEPTH_SCALE, DEPTH_TRUNC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Open3D camera axes (x right, y down, z forward) -> standard "world" orientation (y up),
# used when the dataset has no camera poses
CAMERA_TO_DISPLAY = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def list_images(folder):
    #Sorted image files of a folder (skips .gitkeep and other non-image files)
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_camera(dataset_dir):
    """
    Calibration of a dataset folder (the folder that contains color/ and depth/):
    - camera.json : intrinsics of the images + depth scale (e.g. written by import_stray.py)
    - poses.txt   : one camera-to-world 4x4 matrix per frame (16 values per line)
    Both are optional: without them, config.DEFAULT_INTRINSICS and config.DEPTH_SCALE
    are used, and the frames are aligned with ICP.

    Returns (intrinsic or None, depth_scale, poses or None)
    """
    dataset_dir = Path(dataset_dir)
    intrinsic, depth_scale, poses = None, DEPTH_SCALE, None

    camera_file = dataset_dir / "camera.json"
    if camera_file.exists():
        cam = json.loads(camera_file.read_text())
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            cam["width"], cam["height"], cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        )
        depth_scale = cam.get("depth_scale", DEPTH_SCALE)

    poses_file = dataset_dir / "poses.txt"
    if poses_file.exists():
        poses = np.loadtxt(poses_file).reshape(-1, 4, 4)

    return intrinsic, depth_scale, poses


def get_intrinsics(width=None, height=None):
    w = width or DEFAULT_INTRINSICS["width"]
    h = height or DEFAULT_INTRINSICS["height"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        w, h,
        DEFAULT_INTRINSICS["fx"], DEFAULT_INTRINSICS["fy"],
        DEFAULT_INTRINSICS["cx"], DEFAULT_INTRINSICS["cy"],
    )
    return intrinsic


def rgbd_to_pointcloud(color_path, depth_path, intrinsic=None, depth_scale=DEPTH_SCALE, pose=None):
    """
    Convert a pair (color, depth) into a 3D point cloud.
    pose: camera-to-world 4x4 matrix that places the frame in the scene. Without pose,
    the cloud is only rotated to a standard "world" orientation (y up).
    """
    color_raw = o3d.io.read_image(str(color_path))
    depth_raw = o3d.io.read_image(str(depth_path))

    if intrinsic is None:
        color_np = np.asarray(color_raw)
        h, w = color_np.shape[:2]
        intrinsic = get_intrinsics(w, h)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,
        depth_scale=depth_scale,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pcd.transform(CAMERA_TO_DISPLAY if pose is None else pose)
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
    the same name in color_dir and depth_dir, sorted alphabetically.
    The frames are placed with the camera poses of the dataset when available
    (see load_camera), otherwise aligned with ICP.
    """
    color_files = list_images(color_dir)
    depth_files = list_images(depth_dir)

    if not color_files:
        raise FileNotFoundError(f"No images found in {color_dir}")
    if len(color_files) != len(depth_files):
        raise ValueError("The number of color and depth images do not match")

    intrinsic, depth_scale, poses = load_camera(Path(color_dir).parent)
    if poses is not None and len(poses) != len(color_files):
        raise ValueError(f"{len(poses)} poses in poses.txt for {len(color_files)} images")

    pointclouds = []
    for i, (color_f, depth_f) in enumerate(zip(color_files, depth_files)):
        pose = None if poses is None else poses[i]
        pcd = rgbd_to_pointcloud(color_f, depth_f, intrinsic, depth_scale, pose)
        pcd = pcd.voxel_down_sample(voxel_size)
        pointclouds.append(pcd)
        print(f"Frame added: {color_f.name} -> {len(pcd.points)} points")

    if poses is None:
        scene = register_and_merge(pointclouds, voxel_size)
    else:
        # The frames are already in the same coordinate system (recorded camera poses)
        scene = o3d.geometry.PointCloud()
        for pcd in pointclouds:
            scene += pcd

    # Overlapping frames give several points for the same surface: keep one per voxel
    if len(pointclouds) > 1:
        scene = scene.voxel_down_sample(voxel_size)
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
