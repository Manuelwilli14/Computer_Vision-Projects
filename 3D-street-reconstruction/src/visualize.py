"""
Visualization of a 3D point cloud / mesh (result of reconstruction or
semantic fusion) with Open3D.
"""

import argparse
import sys

import open3d as o3d

from config import SUPER_CATEGORY_COLORS


def print_legend():
    print("\nColor legend (super-categories):")
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
        raise ValueError("Unsupported file format (use .ply)")

    print_legend()
    o3d.visualization.draw_geometries(
        [geometry],
        window_name="3D Reconstruction + Semantic Segmentation",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a .ply point cloud or mesh")
    parser.add_argument("path", help="Path to the .ply file to visualize")
    args = parser.parse_args()

    visualize(args.path)
