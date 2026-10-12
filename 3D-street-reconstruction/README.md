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
| **SegFormer-B0 / Cityscapes (Hugging Face transformers)** | Semantic segmentation with the 19 Cityscapes classes (road, car, building, vegetation...) |
| **Segment Anything (SAM)** | Optional: precise masks to refine the class edges |
| **OpenCV** | Image processing, camera calibration |

## Pipeline Architecture

```
RGB-D / multi-view images
        │
        ├──► [1] 2D Semantic Segmentation (SegFormer + optional SAM)
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
│   ├── color/             # RGB images
│   ├── depth/             # aligned 16-bit depth images (same count, same order)
│   └── stray_xxx/         # datasets imported from Stray Scanner (same layout + camera.json, poses.txt)
├── src/
│   ├── config.py
│   ├── import_stray.py            # Stray Scanner (iPhone LiDAR) recording → dataset folder
│   ├── semantic_segmentation.py   # SegFormer-B0 (Cityscapes) + SAM
│   ├── reconstruction.py          # Open3D RGB-D → point cloud (camera poses or ICP)
│   ├── fusion_2d_3d.py            # projection of labels onto 3D point cloud
│   ├── visualize.py               # Open3D display
│   └── pipeline.py                # full orchestration
├── notebooks/
│   └── demoReconstruction.ipynb
└── outputs/
    └── .gitkeep
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

The SegFormer-B0 weights (~15 MB) are downloaded automatically from Hugging Face on the
first run, then loaded from the local cache (`~/.cache/huggingface`). It runs on CPU in
about 1 s per image.

Optional — download the SAM checkpoint (ViT-B model, ~375 MB) to refine the class edges.
Without it, the SAM step is skipped. SAM is slow without a GPU (`--no-sam` to disable it):
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

## Quick Start

```bash
# 1. Semantic segmentation of an image (add --no-sam to skip SAM)
python src/semantic_segmentation.py --image data/color/0001.png --output outputs/segmentations/0001.png

# 2. 3D reconstruction from RGB-D (color + depth)
python src/reconstruction.py --color data/color/ --depth data/depth/ --output outputs/scene.ply

# 3. 2D → 3D semantic fusion (needs one .labels.npy per frame, produced by step 1)
python src/fusion_2d_3d.py --color data/color/ --depth data/depth/ --labels outputs/segmentations/ --output outputs/scene_semantic.ply

# 4. Full pipeline
python src/pipeline.py --data_dir data/ --output_dir outputs/

# 5. Visualization
python src/visualize.py outputs/scene_semantic.ply
```

> **Depth scale:** set `DEPTH_SCALE` in `src/config.py` to match your sensor
> (5000 for TUM RGB-D, the sample data; 1000 for RealSense / Azure Kinect).
> A dataset folder can override it, and the camera intrinsics, with a `camera.json` file (see below).

## Your Own Street: iPhone / iPad LiDAR (Stray Scanner)

1. Record the street with the free [Stray Scanner](https://apps.apple.com/app/id1557051662) app
   (iPhone / iPad **Pro** with LiDAR). Stay within ~5 m of what you film (LiDAR range) and move slowly.
2. Copy the recording folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`...) to your computer.
3. Import it, then run the pipeline on the created dataset folder:

```bash
python src/import_stray.py path/to/recording data/my_street      # --step 10: keep 1 frame out of 10
python src/pipeline.py --data_dir data/my_street --output_dir outputs/my_street --no-sam
```

The import keeps 1 frame out of N, puts the frames upright (portrait recordings), drops the low-confidence
LiDAR depth, and writes two extra files to the dataset folder:

| File | Content |
|---|---|
| `camera.json` | intrinsics of the images (`width`, `height`, `fx`, `fy`, `cx`, `cy`) + `depth_scale` (1000: millimeters) |
| `poses.txt` | camera-to-world 4x4 matrix of each frame (16 values per line), recorded by the iPhone (ARKit) |

When `poses.txt` exists, the reconstruction and the fusion place the frames with these poses
instead of aligning them with ICP.

## Detected Semantic Classes

The 19 Cityscapes classes predicted by SegFormer are grouped into 6 supercategories
(see `SUPER_CATEGORIES` in `src/config.py`):

| Supercategory | Cityscapes classes | RGB Color |
|---|---|---|
| road | road, sidewalk, terrain | (128, 64, 128) |
| vehicle | car, truck, bus, train, motorcycle, bicycle, rider | (0, 0, 142) |
| building | building, wall, fence | (70, 70, 70) |
| vegetation | vegetation | (107, 142, 35) |
| sky | sky | (70, 130, 180) |
| other | pole, traffic light, traffic sign, person | (0, 0, 0) |

## Test Data

Put an RGB-D sequence in `data/color/` and `data/depth/` (public datasets: KITTI, TUM RGB-D, or your own
images taken with a depth sensor / smartphone LiDAR). The sample frame comes from TUM RGB-D, which is
an **indoor** dataset: it is fine to test the 3D reconstruction, but the street classes of SegFormer
are not meaningful on it.

## Limitations & Future Work

- RGB-D reconstruction requires aligned depth images (LiDAR/Kinect sensor, or monocular depth estimation via MiDaS/Depth Anything)
- For reconstruction without a depth sensor, integrate an SfM/MVS step (e.g. COLMAP) before Open3D
- SegFormer is trained on Cityscapes (car-mounted camera, European streets): it works best on similar street views
- SegFormer-B0 is the lightest version (runs on CPU); for better accuracy, set `SEGFORMER_MODEL` in `src/config.py` to a larger version (e.g. B2), at the cost of a slower inference
- SAM segments without semantic labels: this project combines SAM (precise masks) with SegFormer (class labels, majority vote per mask) to obtain sharper semantic masks


