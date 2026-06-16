"""
Full pipeline: 2D semantic segmentation -> 3D reconstruction -> 2D/3D fusion.

Expected data structure:
data/
├── color/   (RGB images: 0001.png, 0002.png, ...)
└── depth/   (aligned 16-bit depth images: 0001.png, 0002.png, ...)

Outputs generated in outputs/:
├── segmentations/      (colored masks + .labels.npy per frame)
├── scene.ply           (raw 3D point cloud)
├── scene.mesh.ply      (Poisson mesh, optional)
└── scene_semantic.ply  (3D point cloud with semantic colors)
"""

import argparse
from pathlib import Path

from semantic_segmentation import segment_image
from reconstruction import build_scene_from_rgbd_sequence, reconstruct_mesh
from fusion_2d_3d import fuse_sequence
import open3d as o3d


def run_pipeline(data_dir, output_dir, use_sam=True, make_mesh=False, voxel_size=0.03):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    color_dir = data_dir / "color"
    depth_dir = data_dir / "depth"
    seg_dir = output_dir / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    color_files = sorted(color_dir.glob("*"))
    if not color_files:
        raise FileNotFoundError(f"No images found in {color_dir}")

    # 1. Semantic segmentation of each frame
    print("=== Step 1/3: 2D Semantic Segmentation ===")
    for color_f in color_files:
        out_path = seg_dir / f"{color_f.stem}.png"
        print(f"Segmenting {color_f.name}...")
        segment_image(color_f, out_path, use_sam=use_sam)

    # 2. Raw 3D reconstruction
    print("\n=== Step 2/3: 3D Reconstruction ===")
    scene_pcd = build_scene_from_rgbd_sequence(color_dir, depth_dir, voxel_size)
    scene_path = output_dir / "scene.ply"
    o3d.io.write_point_cloud(str(scene_path), scene_pcd)
    print(f"Point cloud saved: {scene_path} ({len(scene_pcd.points)} points)")

    if make_mesh:
        mesh = reconstruct_mesh(scene_pcd)
        mesh_path = output_dir / "scene.mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Mesh saved: {mesh_path}")

    # 3. 2D -> 3D semantic fusion
    print("\n=== Step 3/3: 2D -> 3D Semantic Fusion ===")
    semantic_path = output_dir / "scene_semantic.ply"
    fuse_sequence(color_dir, depth_dir, seg_dir, semantic_path, voxel_size)

    print("\nPipeline complete.")
    print(f"  - Raw point cloud      : {scene_path}")
    print(f"  - Semantic point cloud : {semantic_path}")
    if make_mesh:
        print(f"  - Mesh                 : {output_dir / 'scene.mesh.ply'}")
    print("\nVisualize with:")
    print(f"  python visualize.py {semantic_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full pipeline: 3D reconstruction + semantic segmentation")
    parser.add_argument("--data_dir", required=True, help="Directory containing color/ and depth/")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM refinement (faster)")
    parser.add_argument("--mesh", action="store_true", help="Also generate a Poisson mesh")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args()

    run_pipeline(
        args.data_dir, args.output_dir,
        use_sam=not args.no_sam,
        make_mesh=args.mesh,
        voxel_size=args.voxel_size,
    )
