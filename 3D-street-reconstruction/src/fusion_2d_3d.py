"""
Fusion 2D -> 3D : projette les masques de segmentation sémantique 2D sur le
nuage de points 3D reconstruit, pour obtenir une scène 3D où chaque point est
étiqueté (route, voiture, bâtiment, végétation, ciel, autre).

Principe :
- Pour chaque frame RGB-D utilisée lors de la reconstruction, on dispose :
    - de l'image couleur (et donc des coordonnées pixel (u, v))
    - de l'image de profondeur (qui a servi à générer les points 3D)
    - du masque sémantique 2D correspondant (calculé via semantic_segmentation.py)
- Comme create_from_rgbd_image génère les points dans le même ordre que les
  pixels (en ignorant les pixels de profondeur nulle), on peut retrouver pour
  chaque point 3D le pixel (u, v) d'origine et donc son label sémantique.
- On colore chaque point selon sa super-catégorie pour la visualisation finale.
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
    Recrée le nuage de points pour une frame et lui attribue, point par point,
    la couleur de la super-catégorie correspondante (issue de label_map).

    label_map : array (H, W) d'entiers (indices dans SUPER_CAT_LIST), généré
                par semantic_segmentation.segment_image puis sauvegardé en .npy
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

    # même transformation que dans reconstruction.rgbd_to_pointcloud
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def fuse_sequence(color_dir, depth_dir, label_dir, output_path, voxel_size=0.03):
    """
    Construit un nuage de points sémantique fusionné pour toute une séquence.

    label_dir doit contenir un fichier .labels.npy par frame, avec le même
    nom de base que les images couleur/profondeur (sortie de
    semantic_segmentation.segment_image).
    """
    color_files = sorted(Path(color_dir).glob("*"))
    depth_files = sorted(Path(depth_dir).glob("*"))
    label_files = sorted(Path(label_dir).glob("*.labels.npy"))

    if not (len(color_files) == len(depth_files) == len(label_files)):
        raise ValueError(
            f"Nombre de fichiers incohérent : "
            f"{len(color_files)} couleur, {len(depth_files)} profondeur, "
            f"{len(label_files)} labels"
        )

    pointclouds = []
    for color_f, depth_f, label_f in zip(color_files, depth_files, label_files):
        label_map = np.load(label_f)
        pcd = project_labels_to_pointcloud(color_f, depth_f, label_map)
        pcd = pcd.voxel_down_sample(voxel_size)
        pointclouds.append(pcd)
        print(f"Frame sémantique ajoutée : {color_f.name} -> {len(pcd.points)} points")

    scene = pointclouds[0]
    for pcd in pointclouds[1:]:
        scene += pcd

    scene, _ = scene.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.io.write_point_cloud(str(output_path), scene)
    print(f"Nuage de points sémantique sauvegardé : {output_path} ({len(scene.points)} points)")

    print_class_statistics(scene)
    return scene


def print_class_statistics(pcd):
    """Affiche le pourcentage de points par classe sémantique."""
    colors = np.asarray(pcd.colors)
    total = len(colors)
    print("\nRépartition sémantique du nuage de points :")
    for super_cat, rgb in SUPER_CATEGORY_COLORS.items():
        target = np.array(rgb) / 255.0
        mask = np.all(np.isclose(colors, target, atol=1e-3), axis=1)
        pct = 100 * mask.sum() / total if total > 0 else 0
        print(f"  - {super_cat:12s}: {pct:5.1f}%  ({mask.sum()} points)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion 2D->3D des labels sémantiques")
    parser.add_argument("--color", required=True, help="Dossier des images couleur")
    parser.add_argument("--depth", required=True, help="Dossier des images de profondeur")
    parser.add_argument("--labels", required=True, help="Dossier des fichiers .labels.npy")
    parser.add_argument("--output", required=True, help="Fichier .ply de sortie")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args()

    fuse_sequence(args.color, args.depth, args.labels, args.output, args.voxel_size)
