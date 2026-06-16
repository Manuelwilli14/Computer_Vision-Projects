"""
Visualisation d'un nuage de points / maillage 3D (résultat de la
reconstruction ou de la fusion sémantique) avec Open3D.
"""

import argparse
import sys

import open3d as o3d

from config import SUPER_CATEGORY_COLORS


def print_legend():
    print("\nLégende des couleurs (super-catégories) :")
    for name, rgb in SUPER_CATEGORY_COLORS.items():
        print(f"  {name:12s}: RGB{rgb}")


def visualize(path):
    if path.endswith(".ply"):
        try:
            geometry = o3d.io.read_point_cloud(path)
            if len(geometry.points) == 0:
                geometry = o3d.io.read_triangle_mesh(path)
                geometry.compute_vertex_normals()
        except Exception:
            geometry = o3d.io.read_triangle_mesh(path)
            geometry.compute_vertex_normals()
    else:
        raise ValueError("Format de fichier non supporté (utiliser .ply)")

    print_legend()
    o3d.visualization.draw_geometries(
        [geometry],
        window_name="Reconstruction 3D + Segmentation Sémantique",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise un nuage de points ou maillage .ply")
    parser.add_argument("path", help="Chemin du fichier .ply à visualiser")
    args = parser.parse_args()

    visualize(args.path)
