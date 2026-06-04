"""
detect_image.py — Detect face masks in a single image.

Usage:
    python src/detect_image.py --model models/mask_detector.h5 --image photo.jpg
    python src/detect_image.py --model models/mask_detector.h5 --image photo.jpg --output result.jpg
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ── Reuse constants from detect_realtime ──────────────────────────────────────
LABEL_NAMES  = ["With Mask", "Without Mask", "Incorrect Mask"]
LABEL_COLORS_BGR = [(0, 200, 0), (0, 0, 220), (0, 140, 255)]
IMAGE_SIZE   = (224, 224)

FACE_PROTO  = "models/face_detector/deploy.prototxt"
FACE_MODEL  = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
FACE_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
FACE_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


def download_face_detector():
    Path("models/face_detector").mkdir(parents=True, exist_ok=True)
    if not Path(FACE_PROTO).exists():
        urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)
    if not Path(FACE_MODEL).exists():
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)


def detect_and_annotate(
    image_path: str,
    mask_model: tf.keras.Model,
    face_net,
    face_conf: float = 0.5,
) -> np.ndarray:
    """
    Load image, detect faces, classify masks, return annotated BGR image.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < face_conf:
            continue

        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))

        if x2 <= x1 or y2 <= y1:
            continue

        face      = frame[y1:y2, x1:x2]
        face_rgb  = cv2.cvtColor(cv2.resize(face, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        inp       = np.expand_dims(face_rgb.astype("float32") / 255.0, 0)
        preds     = mask_model.predict(inp, verbose=0)[0]
        label_idx = int(np.argmax(preds))
        label_conf = float(preds[label_idx])

        color = LABEL_COLORS_BGR[label_idx]
        label_text = f"{LABEL_NAMES[label_idx]}: {label_conf * 100:.1f}%"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        results.append({
            "label": LABEL_NAMES[label_idx],
            "confidence": label_conf,
            "box": (x1, y1, x2, y2),
        })

    # Print summary
    print(f"\n[detect] Found {len(results)} face(s):")
    for r in results:
        print(f"  • {r['label']} ({r['confidence']*100:.1f}%)  box={r['box']}")

    return frame


def main(args):
    download_face_detector()

    print("[detect] Loading mask model…")
    mask_model = tf.keras.models.load_model(args.model)

    print("[detect] Loading face detector…")
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

    annotated = detect_and_annotate(args.image, mask_model, face_net, args.face_conf)

    if args.output:
        cv2.imwrite(args.output, annotated)
        print(f"[detect] Result saved → {args.output}")
    else:
        # Display with matplotlib (works headless too)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Face Mask Detection Result")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect face masks in an image")
    parser.add_argument("--model",     required=True, help="Path to .h5 model")
    parser.add_argument("--image",     required=True, help="Input image path")
    parser.add_argument("--output",    default=None,  help="Output image path (optional)")
    parser.add_argument("--face_conf", type=float, default=0.5,
                        help="Minimum face confidence (default: 0.5)")
    args = parser.parse_args()
    main(args)
