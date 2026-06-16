"""
Reconstruction 3D d'une scène de rue avec Open3D.

Deux modes supportés :
1. RGB-D : à partir de paires (couleur, profondeur) alignées + poses caméra
   -> intégration TSDF (open3d.pipelines.integration) ou fusion de nuages de points
2. Multi-vue sans profondeur : nuage de points épars via correspondances de points
   (mode simplifié, pour de meilleurs résultats utiliser COLMAP en amont puis
   charger directement le nuage de points dense ici)

Sortie : un fichier .ply (nuage de points ou maillage)
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
    """Convertit une paire (couleur, profondeur) en nuage de points 3D."""
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
    # Repère Open3D : retourner pour avoir une orientation "monde" standard
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def register_and_merge(pointclouds, voxel_size=0.05):
    """
    Aligne grossièrement et fusionne plusieurs nuages de points avec ICP.
    pointclouds : liste de o3d.geometry.PointCloud (déjà dans des repères proches)
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
    Construit un nuage de points 3D de la scène complète à partir d'une
    séquence d'images couleur+profondeur. Les fichiers doivent porter
    le même nom dans color_dir et depth_dir, triés par ordre alphabétique.
    """
    color_files = sorted(Path(color_dir).glob("*"))
    depth_files = sorted(Path(depth_dir).glob("*"))

    if len(color_files) != len(depth_files):
        raise ValueError("Le nombre d'images couleur et de profondeur ne correspond pas")

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
    """Reconstruction de surface (Poisson) à partir du nuage de points."""
    pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pointcloud.orient_normals_consistent_tangent_plane(30)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pointcloud, depth=depth
    )

    # Supprime les zones de faible densité (artefacts)
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction 3D d'une scène de rue (Open3D)")
    parser.add_argument("--color", required=True, help="Dossier des images couleur")
    parser.add_argument("--depth", required=True, help="Dossier des images de profondeur")
    parser.add_argument("--output", required=True, help="Fichier .ply de sortie")
    parser.add_argument("--mesh", action="store_true", help="Génère également un maillage Poisson")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args()

    scene_pcd = build_scene_from_rgbd_sequence(args.color, args.depth, args.voxel_size)
    o3d.io.write_point_cloud(args.output, scene_pcd)
    print(f"Nuage de points sauvegardé : {args.output} ({len(scene_pcd.points)} points)")

    if args.mesh:
        mesh = reconstruct_mesh(scene_pcd)
        mesh_path = str(Path(args.output).with_suffix(".mesh.ply"))
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Maillage sauvegardé : {mesh_path}")
