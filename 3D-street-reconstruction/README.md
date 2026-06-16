# 3D Reconstruction + Semantic Segmentation of a Street Scene

A complete pipeline combining **computer vision**, **deep learning**, and **3D geometry** to reconstruct a street scene from images and semantically identify roads, cars, buildings, and vegetation.

## Objectives

- Reconstruct a 3D point cloud of a street from multiple images (Structure from Motion / RGB-D)
- Semantically segment each image (roads, cars, buildings, vegetation, sky...)
- Project 2D segmentation masks onto the 3D point cloud to obtain a **semantically annotated 3D scene**
- Visualize the result with Open3D

## Tech Stack

| Component | Role |
|---|---|
| **Open3D** | 3D reconstruction, point clouds, mesh, visualization |
| **PyTorch** | Deep learning backend |
| **Segment Anything (SAM)** | Instance/mask segmentation |
| **DeepLabV3 / Cityscapes (torchvision)** | Semantic segmentation with predefined classes (road, car, building, vegetation) |
| **OpenCV** | Image processing, camera calibration |

## Pipeline Architecture

```
RGB-D / multi-view images
        │
        ├──► [1] 2D Semantic Segmentation (DeepLabV3 + SAM)
        │         → masks: road, car, building, vegetation, sky...
        │
        ├──► [2] 3D Reconstruction (Open3D)
        │         → point cloud / mesh via RGB-D or SfM
        │
        └──► [3] 2D→3D Fusion
                  → each 3D point receives a semantic label
                  → point cloud colored by class
                  → .ply export for visualization
```

## Project Structure

```
3d-street-reconstruction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          # instructions for obtaining/placing data
├── src/
│   ├── config.py
│   ├── semantic_segmentation.py   # DeepLabV3 (Cityscapes) + SAM
│   ├── reconstruction.py          # Open3D RGB-D / SfM → point cloud
│   ├── fusion_2d_3d.py            # projection of labels onto 3D point cloud
│   ├── visualize.py               # Open3D display
│   └── pipeline.py                # full orchestration
├── notebooks/
│   └── demo.ipynb
└── outputs/
    └── .gitkeep
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the SAM checkpoint (ViT-B model, ~375 MB):
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

## Quick Start

```bash
# 1. Semantic segmentation of an image
python src/semantic_segmentation.py --image data/sample_street.jpg --output outputs/seg_mask.png

# 2. 3D reconstruction from RGB-D (color + depth)
python src/reconstruction.py --color data/color/ --depth data/depth/ --output outputs/scene.ply

# 3. 2D → 3D semantic fusion
python src/fusion_2d_3d.py --pointcloud outputs/scene.ply --seg_dir outputs/segmentations/ --output outputs/scene_semantic.ply

# 4. Full pipeline
python src/pipeline.py --data_dir data/ --output_dir outputs/

# 5. Visualization
python src/visualize.py outputs/scene_semantic.ply
```

## Detected Semantic Classes

Based on Cityscapes classes (grouped):

| Class | RGB Color |
|---|---|
| Road | (128, 64, 128) |
| Car | (0, 0, 142) |
| Building | (70, 70, 70) |
| Vegetation | (107, 142, 35) |
| Sky | (70, 130, 180) |
| Sidewalk | (244, 35, 232) |
| Other | (0, 0, 0) |

## Test Data

The `data/` folder contains instructions for obtaining public RGB-D sequences (e.g. KITTI, TUM RGB-D, or your own photos taken with a depth sensor / smartphone LiDAR).

## Limitations & Future Work

- RGB-D reconstruction requires aligned depth images (LiDAR/Kinect sensor, or monocular depth estimation via MiDaS/Depth Anything)
- For reconstruction without a depth sensor, integrate an SfM/MVS step (e.g. COLMAP) before Open3D
- SAM segments without semantic labels by default: this project combines SAM (precise masks) with DeepLabV3 (class labels) to obtain reliable semantic masks
- Fine-tuning with a model specifically trained on Cityscapes/Mapillary would improve performance in urban environments


