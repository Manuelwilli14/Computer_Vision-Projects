"""
Pipeline complet : segmentation sémantique 2D -> reconstruction 3D -> fusion 2D/3D.

Structure de données attendue :
data/
├── color/   (images RGB : 0001.png, 0002.png, ...)
└── depth/   (images de profondeur 16-bit alignées : 0001.png, 0002.png, ...)

Sorties générées dans outputs/ :
├── segmentations/      (masques colorés + .labels.npy par frame)
├── scene.ply           (nuage de points 3D brut)
├── scene.mesh.ply       (maillage Poisson, optionnel)
└── scene_semantic.ply   (nuage de points 3D avec couleurs sémantiques)
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
        raise FileNotFoundError(f"Aucune image trouvée dans {color_dir}")

    # 1. Segmentation sémantique de chaque frame
    print("=== Étape 1/3 : Segmentation sémantique 2D ===")
    for color_f in color_files:
        out_path = seg_dir / f"{color_f.stem}.png"
        print(f"Segmentation de {color_f.name}...")
        segment_image(color_f, out_path, use_sam=use_sam)

    # 2. Reconstruction 3D brute
    print("\n=== Étape 2/3 : Reconstruction 3D ===")
    scene_pcd = build_scene_from_rgbd_sequence(color_dir, depth_dir, voxel_size)
    scene_path = output_dir / "scene.ply"
    o3d.io.write_point_cloud(str(scene_path), scene_pcd)
    print(f"Nuage de points sauvegardé : {scene_path} ({len(scene_pcd.points)} points)")

    if make_mesh:
        mesh = reconstruct_mesh(scene_pcd)
        mesh_path = output_dir / "scene.mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Maillage sauvegardé : {mesh_path}")

    # 3. Fusion sémantique 2D -> 3D
    print("\n=== Étape 3/3 : Fusion sémantique 2D -> 3D ===")
    semantic_path = output_dir / "scene_semantic.ply"
    fuse_sequence(color_dir, depth_dir, seg_dir, semantic_path, voxel_size)

    print("\nPipeline terminé.")
    print(f"  - Nuage de points brut     : {scene_path}")
    print(f"  - Nuage de points sémantique : {semantic_path}")
    if make_mesh:
        print(f"  - Maillage                  : {output_dir / 'scene.mesh.ply'}")
    print("\nVisualisez avec :")
    print(f"  python visualize.py {semantic_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet de reconstruction 3D + segmentation sémantique")
    parser.add_argument("--data_dir", required=True, help="Dossier contenant color/ et depth/")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie")
    parser.add_argument("--no-sam", action="store_true", help="Désactive le raffinement SAM (plus rapide)")
    parser.add_argument("--mesh", action="store_true", help="Génère aussi un maillage Poisson")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    args = parser.parse_args()

    run_pipeline(
        args.data_dir, args.output_dir,
        use_sam=not args.no_sam,
        make_mesh=args.mesh,
        voxel_size=args.voxel_size,
    )
