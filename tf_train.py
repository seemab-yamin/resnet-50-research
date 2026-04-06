# TensorFlow: model building, training, evaluation

from __future__ import annotations

import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

import config


def build_resnet50_model(num_classes: int) -> models.Model:
    """
    ResNet50 with frozen ImageNet backbone and a small dense head (256 → num_classes, softmax).
    Spatial size and channels follow ``config.IMG_SIZE`` and RGB.
    """
    h, w = config.IMG_SIZE
    in_shape = (int(h), int(w), 3)

    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=in_shape,
        pooling="avg",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=in_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="resnet50_transfer")


def configure_training() -> tuple[
    tf.keras.losses.Loss,
    tf.keras.optimizers.Optimizer,
    list[tf.keras.metrics.Metric | str],
]:
    """Loss, optimizer, and metrics aligned with the PyTorch notebook setup."""
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4,
    )
    metrics: list[tf.keras.metrics.Metric | str] = ["accuracy"]
    return loss, optimizer, metrics


class TrainingLogger(tf.keras.callbacks.Callback):
    """Writes running epoch history to JSON after each epoch."""

    def __init__(self, log_file: str | Path = "results/tf_history.json") -> None:
        super().__init__()
        self.log_file = Path(log_file)
        self.history: dict[str, list] = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "time": [],
        }
        self.epoch_start: float | None = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - (self.epoch_start or time.time())
        self.history["epoch"].append(epoch)
        self.history["loss"].append(logs.get("loss"))
        self.history["accuracy"].append(logs.get("accuracy"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["val_accuracy"].append(logs.get("val_accuracy"))
        self.history["time"].append(epoch_time)

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)


def get_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Checkpoint, schedule, early stop, and logging callbacks using ``config.PROJECT_ROOT_DIR``."""
    root = Path(config.PROJECT_ROOT_DIR)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    tb_dir = root / "results" / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath=str(results / "tf_best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        TensorBoard(log_dir=str(tb_dir), histogram_freq=1),
        CSVLogger(str(results / "tf_training_log.csv")),
        TrainingLogger(results / "tf_history.json"),
    ]
