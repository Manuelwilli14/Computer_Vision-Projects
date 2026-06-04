"""
dataset.py — Load and preprocess the Face Mask Detection dataset.

The Kaggle dataset contains:
  - images/      : JPEG images
  - annotations/ : Pascal VOC XML files with bounding boxes + labels
                   Labels: with_mask | without_mask | mask_weared_incorrect
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = (224, 224)   # MobileNetV2 input size
LABEL_MAP    = {
    "with_mask":              0,
    "without_mask":           1,
    "mask_weared_incorrect":  2,
}
LABEL_NAMES  = ["With Mask", "Without Mask", "Incorrect Mask"]


# ── XML Parsing ───────────────────────────────────────────────────────────────
def parse_annotation(xml_path: str) -> list[dict]:
    """
    Parse a Pascal VOC annotation XML file.
    Returns a list of dicts: {xmin, ymin, xmax, ymax, label}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        objects.append({
            "label": label,
            "xmin":  int(float(bndbox.find("xmin").text)),
            "ymin":  int(float(bndbox.find("ymin").text)),
            "xmax":  int(float(bndbox.find("xmax").text)),
            "ymax":  int(float(bndbox.find("ymax").text)),
        })
    return objects


# ── Face-crop extraction ───────────────────────────────────────────────────────
def extract_faces(data_dir: str, img_size: tuple = IMAGE_SIZE) -> tuple:
    """
    Walk through images + annotations, crop each annotated face,
    resize it, and return (X, y) numpy arrays.

    Args:
        data_dir : root folder containing 'images/' and 'annotations/'
        img_size : target (H, W) for resizing

    Returns:
        X : float32 array of shape (N, H, W, 3), values in [0, 1]
        y : int array of shape (N,)
    """
    images_dir      = Path(data_dir) / "images"
    annotations_dir = Path(data_dir) / "annotations"

    X, y = [], []

    xml_files = sorted(annotations_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No XML annotations found in {annotations_dir}. "
            "Did you extract the Kaggle dataset correctly?"
        )

    print(f"[dataset] Found {len(xml_files)} annotation files.")

    for xml_path in tqdm(xml_files, desc="Loading faces"):
        # Derive image path (try .png then .jpg)
        stem = xml_path.stem
        img_path = images_dir / f"{stem}.png"
        if not img_path.exists():
            img_path = images_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        for obj in parse_annotation(str(xml_path)):
            label = obj["label"]
            if label not in LABEL_MAP:
                continue

            # Clamp bounding box to image dimensions
            xmin = max(0, obj["xmin"])
            ymin = max(0, obj["ymin"])
            xmax = min(w, obj["xmax"])
            ymax = min(h, obj["ymax"])

            if xmax <= xmin or ymax <= ymin:
                continue

            face = img_rgb[ymin:ymax, xmin:xmax]
            face = cv2.resize(face, img_size)
            X.append(face)
            y.append(LABEL_MAP[label])

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y, dtype="int32")
    print(f"[dataset] Total face crops: {len(X)}")
    return X, y


# ── Train / Val / Test split ──────────────────────────────────────────────────
def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size:  float = 0.15,
    test_size: float = 0.15,
    seed:      int   = 42,
) -> tuple:
    """
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=seed, stratify=y
    )
    ratio = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=ratio, random_state=seed, stratify=y_tmp
    )
    print(
        f"[dataset] Split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── tf.data pipeline ──────────────────────────────────────────────────────────
def augment(image, label):
    """Light augmentation applied only during training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def make_tf_datasets(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    batch_size: int = 32,
    num_classes: int = 3,
) -> tuple:
    """
    Build optimised tf.data.Dataset objects for train / val / test.
    Labels are one-hot encoded.
    """
    def to_ds(X, y, shuffle=False, augmentation=False):
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.map(
            lambda x, lbl: (x, tf.one_hot(lbl, num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if augmentation:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=1024)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = to_ds(X_train, y_train, shuffle=True,  augmentation=True)
    val_ds   = to_ds(X_val,   y_val,   shuffle=False, augmentation=False)
    test_ds  = to_ds(X_test,  y_test,  shuffle=False, augmentation=False)
    return train_ds, val_ds, test_ds
