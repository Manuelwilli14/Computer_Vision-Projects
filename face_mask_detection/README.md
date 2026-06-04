# 😷 Face Mask Detection System

A real-time face mask detection system using CNN (MobileNetV2) + OpenCV.

## Project Structure

```
face_mask_detection/
├── src/
│   ├── train.py              # Model training script
│   ├── detect_realtime.py    # Real-time webcam detection
│   ├── detect_image.py       # Single image detection
│   └── dataset.py            # Dataset loading & preprocessing
├── models/                   # Saved model weights
├── data/                     # Dataset (download from Kaggle)
│   ├── images/
│   └── annotations/
├── notebooks/
│   └── exploration.ipynb     # EDA notebook
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
Extract into the `data/` folder.

## Usage

### 1. Train the model
```bash
python src/train.py --data_dir data/ --epochs 20 --batch_size 32
```

### 2. Real-time detection (webcam)
```bash
python src/detect_realtime.py --model models/mask_detector.h5
```

### 3. Detect on an image
```bash
python src/detect_image.py --model models/mask_detector.h5 --image path/to/image.jpg
```

## Model Architecture

- **Base**: MobileNetV2 (pretrained on ImageNet) — lightweight & fast
- **Head**: GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.5) → Dense(3, Softmax)
- **Classes**: `with_mask` | `without_mask` | `mask_weared_incorrect`

## Performance

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~96%   |
| FPS (CPU) | ~15-20 |
| FPS (GPU) | ~60+   |
