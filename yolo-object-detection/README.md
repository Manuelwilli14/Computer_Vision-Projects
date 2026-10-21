# 🎯 YOLO Object Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

A complete object detection project using **YOLOv8** — supporting images, videos, and real-time webcam inference.

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Examples](#-examples)
- [Training Custom Model](#-training-a-custom-model)
- [Results](#-results)
- [License](#-license)

---

## ✨ Features

- 📸 **Image detection** — run inference on single images or batches
- 🎬 **Video detection** — process video files with annotated output
- 📷 **Real-time webcam** — live detection from your camera
- 🏋️ **Custom training** — train on your own dataset (YOLO format)
- 📊 **Evaluation metrics** — mAP, precision, recall
- 🖥️ **CLI + Python API** — use from terminal or import as a module

---

## 📁 Project Structure

```
yolo-object-detection/
│
├── src/
│   ├── detect.py          # Main detection script
│   ├── train.py           # Model training
│   ├── evaluate.py        # Evaluation & metrics
│   └── utils.py           # Helper functions
│
├── scripts/
│   ├── download_model.py  # Download pretrained weights
│   └── prepare_dataset.py # Dataset preparation utility
│
├── notebooks/
│   └── demo.ipynb         # Interactive Jupyter demo
│
├── data/
│   ├── images/            # Input images
│   └── videos/            # Input videos
│
├── models/                # Pretrained / trained weights (.pt)
├── results/               # Output detections
│
├── requirements.txt
├── config.yaml            # Detection configuration
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manuelwilli14/Computer_Vision-Projects.git 
cd Computer_Vision-Projects/yolo-object-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download a pretrained model

```bash
python scripts/download_model.py --model yolov8n
```

Available models: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`  
(n=nano, s=small, m=medium, l=large, x=extra-large)

---

## 🚀 Usage

### Detect on an image

```bash
python src/detect.py --source data/images/test.jpg --model models/yolov8n.pt
```

### Detect on a video

```bash
python src/detect.py --source data/videos/test.mp4 --model models/yolov8n.pt
```

### Real-time webcam detection

```bash
python src/detect.py --source 0 --model models/yolov8n.pt
```

### Batch detection on a folder

```bash
python src/detect.py --source data/images/ --model models/yolov8n.pt --save
```

### Full options

```
python src/detect.py --help

Arguments:
  --source       Path to image, video, folder, or webcam (0)
  --model        Path to YOLO model weights (.pt)
  --conf         Confidence threshold (default: 0.25)
  --iou          IoU threshold for NMS (default: 0.45)
  --classes      Filter by class IDs (e.g. --classes 0 2 3)
  --save         Save output to results/
  --show         Display results in window
  --device       Device: 'cpu', '0' (GPU), 'mps' (Apple Silicon)
```

---

## 🖼️ Examples

| Input | Output |
|-------|--------|
| Street scene | People, cars, bicycles detected |
| Indoor photo | Chairs, tables, laptops detected |
| Webcam stream | Real-time bounding boxes + FPS |

---

## 🏋️ Training a Custom Model

### 1. Prepare your dataset

Your dataset must follow the YOLO format:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each label file is a `.txt` with one row per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to `[0, 1]`.

### 2. Configure training

Edit `config.yaml`:

```yaml
model: yolov8n.pt        # Base model
data: dataset/data.yaml  # Dataset config
epochs: 100
imgsz: 640
batch: 16
device: 0                # GPU (use 'cpu' if no GPU)
```

### 3. Start training

```bash
python src/train.py --config config.yaml
```

Training results are saved to `runs/train/`.

---

## 📊 Evaluation

```bash
python src/evaluate.py --model models/best.pt --data dataset/data.yaml
```

Outputs: **mAP@0.5**, **mAP@0.5:0.95**, precision, recall, confusion matrix.

---

## 📦 Requirements

| Package | Version |
|---------|---------|
| ultralytics | ≥ 8.0 |
| opencv-python | ≥ 4.8 |
| torch | ≥ 2.0 |
| numpy | ≥ 1.24 |
| Pillow | ≥ 9.0 |
| PyYAML | ≥ 6.0 |
| tqdm | ≥ 4.65 |
| matplotlib | ≥ 3.7 |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- [OpenCV](https://opencv.org/)
