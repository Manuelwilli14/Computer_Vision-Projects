"""
train.py — Build, train, and evaluate the face mask detection model.

Architecture : MobileNetV2 (ImageNet weights, frozen) + custom head
               → GlobalAveragePooling2D
               → Dense(128, relu) + BatchNorm + Dropout(0.5)
               → Dense(3, softmax)

Usage:
    python src/train.py --data_dir data/ --epochs 20 --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    LABEL_NAMES,
    IMAGE_SIZE,
    extract_faces,
    split_dataset,
    make_tf_datasets,
)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(num_classes: int = 3, img_size: tuple = IMAGE_SIZE) -> Model:
    """
    Transfer-learning model:
      1. MobileNetV2 backbone (frozen initially)
      2. Custom classification head
    """
    base = MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze backbone for phase 1

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base


# ── Fine-tune ─────────────────────────────────────────────────────────────────
def unfreeze_top_layers(base_model: Model, num_layers: int = 30) -> None:
    """Unfreeze the top N layers of the base model for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    print(f"[train] Fine-tuning: last {num_layers} backbone layers unfrozen.")


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_history(history, history_ft=None, save_path: str = "models/training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for metric, ax in zip(["accuracy", "loss"], axes):
        ax.plot(history.history[metric],       label="Train (phase 1)")
        ax.plot(history.history[f"val_{metric}"], label="Val (phase 1)")
        if history_ft:
            offset = len(history.history[metric])
            epochs_ft = range(offset, offset + len(history_ft.history[metric]))
            ax.plot(epochs_ft, history_ft.history[metric],       label="Train (fine-tune)", linestyle="--")
            ax.plot(epochs_ft, history_ft.history[f"val_{metric}"], label="Val (fine-tune)",   linestyle="--")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"[train] History plot saved → {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str = "models/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print(f"[train] Confusion matrix saved → {save_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    # 1. Data
    print("\n=== Step 1 — Loading dataset ===")
    X, y = extract_faces(args.data_dir, img_size=IMAGE_SIZE)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    train_ds, val_ds, test_ds = make_tf_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size,
    )

    # Class weights to handle imbalance
    counts      = np.bincount(y_train)
    total       = y_train.shape[0]
    class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print(f"[train] Class weights: {class_weight}")

    # 2. Build model
    print("\n=== Step 2 — Building model ===")
    model, base = build_model(num_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Phase 1 — Train head only
    print("\n=== Step 3 — Phase 1: Training head ===")
    callbacks_p1 = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint("models/mask_detector_best.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir="logs/phase1"),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks_p1,
    )

    # 4. Phase 2 — Fine-tune top backbone layers
    print("\n=== Step 4 — Phase 2: Fine-tuning ===")
    unfreeze_top_layers(base, num_layers=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p2 = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint("models/mask_detector_best.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir="logs/phase2"),
    ]
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        class_weight=class_weight,
        callbacks=callbacks_p2,
    )

    # 5. Evaluate
    print("\n=== Step 5 — Evaluation on test set ===")
    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc * 100:.2f}%  |  Test loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    plot_history(history, history_ft)
    plot_confusion_matrix(y_test, y_pred)

    # 6. Save final model
    model.save("models/mask_detector.h5")
    print("\n[train] Final model saved → models/mask_detector.h5")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train face mask detector")
    parser.add_argument("--data_dir",       default="data/",  help="Dataset root directory")
    parser.add_argument("--epochs",         type=int, default=20,  help="Phase 1 epochs")
    parser.add_argument("--finetune_epochs",type=int, default=10,  help="Phase 2 fine-tune epochs")
    parser.add_argument("--batch_size",     type=int, default=32,  help="Batch size")
    args = parser.parse_args()
    main(args)
