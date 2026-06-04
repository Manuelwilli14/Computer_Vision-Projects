"""
detect_realtime.py — Real-time face mask detection via webcam.

Pipeline:
  1. Capture frame from webcam
  2. Detect faces with OpenCV DNN (Caffe face detector)
  3. Crop & preprocess each face
  4. Classify with our trained MobileNetV2 model
  5. Overlay coloured bounding boxes + labels + FPS

Usage:
    python src/detect_realtime.py --model models/mask_detector.h5

Controls:
    Q  — quit
    S  — save current frame as screenshot
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# ── Labels & colours ──────────────────────────────────────────────────────────
LABEL_NAMES  = ["With Mask", "Without Mask", "Incorrect Mask"]
# BGR colours: green, red, orange
LABEL_COLORS = [(0, 200, 0), (0, 0, 220), (0, 140, 255)]

IMAGE_SIZE   = (224, 224)

# ── Face detector assets (OpenCV Caffe model) ─────────────────────────────────
FACE_PROTO  = "models/face_detector/deploy.prototxt"
FACE_MODEL  = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

FACE_PROTO_URL  = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
FACE_MODEL_URL  = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


def download_face_detector():
    """Download OpenCV's Caffe face detector if not present."""
    Path("models/face_detector").mkdir(parents=True, exist_ok=True)
    if not Path(FACE_PROTO).exists():
        print("[detect] Downloading face detector prototxt…")
        urllib.request.urlretrieve(FACE_PROTO_URL, FACE_PROTO)
    if not Path(FACE_MODEL).exists():
        print("[detect] Downloading face detector caffemodel…")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL)
    print("[detect] Face detector ready.")


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize, convert to RGB, normalise → (1, 224, 224, 3) float32."""
    face = cv2.resize(face_bgr, IMAGE_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    return np.expand_dims(face, axis=0)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_prediction(frame, box, label_idx, confidence):
    """Draw bounding box + label on frame."""
    x1, y1, x2, y2 = box
    color           = LABEL_COLORS[label_idx]
    label_text      = f"{LABEL_NAMES[label_idx]}: {confidence * 100:.1f}%"

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

    # Label text
    cv2.putText(
        frame, label_text,
        (x1 + 3, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2,
    )


def draw_fps(frame, fps: float):
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 0), 2,
    )


def draw_stats(frame, counts: dict):
    """Draw small legend with per-class counts."""
    y = 60
    for idx, name in enumerate(LABEL_NAMES):
        text  = f"{name}: {counts.get(idx, 0)}"
        color = LABEL_COLORS[idx]
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28


# ── Main detection loop ───────────────────────────────────────────────────────
def run(args):
    download_face_detector()

    # Load models
    print("[detect] Loading mask classifier…")
    mask_model = tf.keras.models.load_model(args.model)

    print("[detect] Loading face detector…")
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    # Optionally boost resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[detect] Press Q to quit, S to save screenshot.")

    frame_count  = 0
    fps_avg      = 0.0
    t_prev       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # ── Face detection ────────────────────────────────────────────────────
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        counts = {}

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < args.face_conf:
                continue

            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))

            if x2 <= x1 or y2 <= y1:
                continue

            face       = frame[y1:y2, x1:x2]
            inp        = preprocess_face(face)
            preds      = mask_model.predict(inp, verbose=0)[0]
            label_idx  = int(np.argmax(preds))
            conf_mask  = float(preds[label_idx])

            counts[label_idx] = counts.get(label_idx, 0) + 1
            draw_prediction(frame, (x1, y1, x2, y2), label_idx, conf_mask)

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_count += 1
        now = time.time()
        if (now - t_prev) >= 1.0:
            fps_avg  = frame_count / (now - t_prev)
            frame_count = 0
            t_prev   = now

        draw_fps(frame, fps_avg)
        draw_stats(frame, counts)

        cv2.imshow("Face Mask Detection  [Q=quit  S=save]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[detect] Screenshot saved → {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[detect] Stream closed.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time face mask detection")
    parser.add_argument("--model",     default="models/mask_detector.h5",
                        help="Path to trained .h5 model")
    parser.add_argument("--camera",    type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--face_conf", type=float, default=0.5,
                        help="Minimum face detection confidence (default: 0.5)")
    args = parser.parse_args()
    run(args)
