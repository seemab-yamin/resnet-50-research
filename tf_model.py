import json
import os
import time
from datetime import datetime

# evaluate_and_save.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config
from base_loader import BaseDataLoader
from tf_dataset import TFImageDataset
from tf_train import build_resnet50_model, configure_training, get_callbacks

print(f"TensorFlow version: {tf.__version__}")
_build = tf.sysconfig.get_build_info()
_cuda = _build.get("cuda_version") or _build.get("cuda_version_number") or "n/a"
print(f"CUDA (TF build): {_cuda}")

_gpus = tf.config.list_physical_devices("GPU")
print(f"GPU available: {len(_gpus) > 0}")
for _i, _gpu in enumerate(_gpus):
    _name = _gpu.name
    try:
        _det = tf.config.experimental.get_device_details(_gpu)
        _name = _det.get("device_name") or _name
    except (AttributeError, TypeError, ValueError):
        pass
    print(f"GPU {_i} name: {_name}")

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(config.PROJECT_ROOT_DIR, "tf-results", run_timestamp)
os.makedirs(results_dir, exist_ok=True)

# fetch training data
train_base = BaseDataLoader(
    config.COVIDQU_PATH, split="Train", is_train_shuffle=True, seed=config.SEED
)

# Create TF dataset (no augmentation for validation; use augment=True for training)
train_dataset = TFImageDataset(
    train_base,
    img_size=config.IMG_SIZE,
    mean=config.MEAN,
    std=config.STD,
    batch_size=config.BATCH_SIZE,
    augment=True,
).build()

# Test one batch
for images, labels in train_dataset.take(1):
    print(f"Images shape: {images.shape}")  # Expected: (batch, 224, 224, 3)
    print(f"Labels: {labels.numpy()}")
    print(
        f"Image value range: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]"
    )

num_classes = len(train_base.class_to_idx)
model = build_resnet50_model(num_classes=num_classes)


# --- TensorFlow / Keras: layer-by-layer text dump (for side-by-side manual check) ---
def _keras_out_shape(layer):
    try:
        return tuple(layer.output.shape)
    except Exception:
        return None


def print_keras_stack(model, heading="Top-level model.layers"):
    print("=" * 110)
    print(heading)
    print("=" * 110)
    print(f"{'#':>4}  {'name':<38}  {'class':<30}  {'params':>12}  {'output_shape'}")
    print("-" * 110)
    for i, layer in enumerate(model.layers):
        params = layer.count_params()
        out = _keras_out_shape(layer)
        print(
            f"{i:4d}  {layer.name:<38}  {layer.__class__.__name__:<30}  {params:12,d}  {out}"
        )
    print("=" * 110)


def print_keras_nested(parent, heading=None, start=0, limit=800):
    subs = getattr(parent, "layers", None)
    if not subs or len(subs) <= 1:
        return
    h = heading or f"Nested under `{getattr(parent, 'name', type(parent).__name__)}`"
    print()
    print("=" * 110)
    print(h)
    print("=" * 110)
    print(f"{'#':>4}  {'name':<38}  {'class':<30}  {'params':>12}  {'output_shape'}")
    print("-" * 110)
    end = min(start + limit, len(subs))
    for j in range(start, end):
        sub = subs[j]
        params = sub.count_params()
        out = _keras_out_shape(sub)
        print(
            f"{j:4d}  {sub.name:<38}  {sub.__class__.__name__:<30}  {params:12,d}  {out}"
        )
    if len(subs) > limit:
        print(
            f"... ({len(subs) - limit} more sub-layers not printed; raise `limit` if needed)"
        )
    print("=" * 110)


print_keras_stack(model)
for lyr in model.layers:
    if lyr is model:
        continue
    if hasattr(lyr, "layers") and len(lyr.layers) > 5:
        print_keras_nested(lyr)

model.summary()


# Create base loaders for train and val
val_base = BaseDataLoader(
    config.COVIDQU_PATH, split="Val", is_train_shuffle=False, seed=config.SEED
)

# Create TF datasets
val_dataset = TFImageDataset(
    val_base,
    img_size=config.IMG_SIZE,
    mean=config.MEAN,
    std=config.STD,
    batch_size=config.BATCH_SIZE,
    augment=False,
).build()


def train_model(
    model,
    train_ds,
    val_ds,
    results_dir,
    epochs=config.EPOCHS,
):
    """Compile and train model"""

    loss, optimizer, metrics = configure_training()
    callbacks = get_callbacks(results_dir)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
    )

    return history


# Record start time
start_time = time.time()

# Train
history = train_model(model, train_dataset, val_dataset, results_dir)

# Record training time
train_time = time.time() - start_time

# Throughput: training images processed per wall-clock second (includes validation each epoch)
_dataset_size = len(train_base)
_epochs = len(history.history["loss"])
_throughput = (_dataset_size * _epochs) / train_time if train_time > 0 else 0.0


# Evaluate
test_base = BaseDataLoader(
    config.COVIDQU_PATH, split="Test", is_train_shuffle=False, seed=config.SEED
)
test_dataset = TFImageDataset(
    test_base,
    img_size=config.IMG_SIZE,
    mean=config.MEAN,
    std=config.STD,
    batch_size=config.BATCH_SIZE,
    augment=False,
).build()


# Save metrics
results = {
    "framework": "TensorFlow",
    "training_time_seconds": train_time,
    "throughput_images_per_sec": float(_throughput),
    "final_train_loss": float(history.history["loss"][-1]),
    "final_train_acc": float(history.history["accuracy"][-1]),
    "final_val_loss": float(history.history["val_loss"][-1]),
    "final_val_acc": float(history.history["val_accuracy"][-1]),
    "best_val_acc": float(np.max(history.history["val_accuracy"])),
    "epochs_completed": len(history.history["loss"]),
}
results_json = os.path.join(results_dir, "tf_training_summary.json")
with open(results_json, "w") as f:
    json.dump(results, f, indent=4)
print("Results saved to: ", results_json)


def evaluate_and_save_all(
    model, test_dataset, class_names, save_dir="results", framework="tf"
):
    """
    Complete evaluation: saves predictions, metrics, confusion matrix.
    Also returns test loss and accuracy.
    """

    # 1. Collect all predictions and compute test loss
    y_true = []
    y_pred = []
    y_pred_proba = []
    total_loss = 0.0
    num_batches = 0

    print("Collecting predictions and computing test loss...")
    for images, labels in test_dataset:
        # Get predictions
        pred_proba = model.predict(images, verbose=0)
        pred = np.argmax(pred_proba, axis=1)

        # Compute batch loss (using model's loss function)
        batch_loss = model.compute_loss(x=images, y=labels, y_pred=pred_proba)
        total_loss += batch_loss.numpy()
        num_batches += 1

        # Store predictions
        y_pred_proba.extend(pred_proba)
        y_pred.extend(pred)
        y_true.extend(labels.numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)

    # Calculate average test loss
    test_loss = total_loss / num_batches
    test_accuracy = accuracy_score(y_true, y_pred)

    # 2. Save all predictions
    np.savez(
        f"{save_dir}/{framework}_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
    )
    print(f"✅ Saved predictions to {save_dir}/{framework}_predictions.npz")

    # 3. Calculate full test metrics
    metrics = {
        "framework": framework.upper(),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "test_recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "num_samples": len(y_true),
        "class_names": class_names,
    }

    # Add per-class metrics
    per_class = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    metrics["per_class"] = per_class

    # Save metrics as JSON
    with open(f"{save_dir}/{framework}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Saved metrics to {save_dir}/{framework}_test_metrics.json")

    # 4. Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(f"{save_dir}/{framework}_confusion_matrix.npy", cm)
    print(f"✅ Saved confusion matrix to {save_dir}/{framework}_confusion_matrix.npy")

    # 5. Save readable classification report
    with open(f"{save_dir}/{framework}_classification_report.txt", "w") as f:
        f.write(f"Classification Report - {framework.upper()}\n")
        f.write("=" * 50 + "\n")
        f.write(
            classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0
            )
        )

    print(
        f"✅ Saved classification report to {save_dir}/{framework}_classification_report.txt"
    )

    # 6. Print summary
    print("\n" + "=" * 50)
    print(f"FINAL TEST METRICS - {framework.upper()}")
    print("=" * 50)
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"Precision: {metrics['test_precision_macro']:.4f} (macro)")
    print(f"Recall:    {metrics['test_recall_macro']:.4f} (macro)")
    print(f"F1-Score:  {metrics['test_f1_macro']:.4f} (macro)")
    print("=" * 50)

    return metrics, y_true, y_pred, y_pred_proba, test_loss, test_accuracy


# Run evaluation
class_names = list(test_base.class_to_idx.keys())
metrics, y_true, y_pred, y_pred_proba, test_loss, test_accuracy = evaluate_and_save_all(
    model, test_dataset, class_names, save_dir=results_dir
)
test_acc = metrics["test_accuracy"]

# ============================================
# QUICK ANALYSIS - Run immediately after evaluation
# ============================================

print("\n" + "=" * 60)
print("QUICK ANALYSIS - TensorFlow Run")
print("=" * 60)

# 1. Final Performance
print("\n📊 FINAL PERFORMANCE:")
print(f"   Test Accuracy:  {test_acc:.4f} ({test_acc * 100:.2f}%)")
print(f"   Test Loss:      {test_loss:.4f}")
print(f"   Best Val Acc:   {np.max(history.history['val_accuracy']):.4f}")
print(f"   Final Val Acc:  {history.history['val_accuracy'][-1]:.4f}")

# 2. Improvement Over Random Baseline
num_classes = len(test_base.class_to_idx)
random_baseline = 1 / num_classes
improvement = (test_acc - random_baseline) * 100
print("\n📈 IMPROVEMENT:")
print(f"   Random baseline: {random_baseline:.4f} ({random_baseline * 100:.1f}%)")
print(f"   Improvement:     +{improvement:.1f} percentage points")

# 3. Convergence Check
val_acc = history.history["val_accuracy"]

print("\n🎯 CONVERGENCE STATUS:")
if val_acc[-1] == max(val_acc):
    print("   ✅ Still improving (best at last epoch)")
elif val_acc[-1] >= 0.98 * max(val_acc):
    print("   ✅ Plateaued near peak (last epoch within 2% of best)")
else:
    print(
        f"   ⚠️ May need more epochs (last epoch {val_acc[-1]:.4f} vs best {max(val_acc):.4f})"
    )

# 4. Overfitting Check
train_acc = history.history["accuracy"][-1]
val_acc_final = history.history["val_accuracy"][-1]
gap = train_acc - val_acc_final

print("\n🔍 OVERFITTING CHECK:")
print(f"   Train accuracy:  {train_acc:.4f}")
print(f"   Val accuracy:    {val_acc_final:.4f}")
print(f"   Gap:             {gap:+.4f}")
if gap < 0.05:
    print("   ✅ No significant overfitting (gap < 5%)")
elif gap < 0.1:
    print("   ⚠️ Moderate overfitting (gap 5-10%)")
else:
    print("   ❌ Significant overfitting (gap > 10%)")

# 5. Training Efficiency
print("\n⚡ EFFICIENCY:")
print(f"   Total training:  {train_time:.1f} seconds ({train_time / 60:.2f} minutes)")
print(f"   Epochs:          {len(history.history['loss'])}")
print(f"   Avg per epoch:   {train_time / len(history.history['loss']):.1f} seconds")

# 6. Quick Verdict
print("\n✅ QUICK VERDICT:")
if test_acc >= 0.70:
    print(
        f"   Model achieved {test_acc * 100:.1f}% — Good for frozen transfer learning"
    )
elif test_acc >= 0.60:
    print(f"   Model achieved {test_acc * 100:.1f}% — Acceptable, could improve")
else:
    print(
        f"   Model achieved {test_acc * 100:.1f}% — Below expectations, check pipeline"
    )

# 7. Save quick summary to text file
quick_summary = f"""
========================================
QUICK SUMMARY - TensorFlow
========================================
Test Accuracy:     {test_acc:.4f} ({test_acc * 100:.2f}%)
Test Loss:         {test_loss:.4f}
Best Val Acc:      {np.max(history.history["val_accuracy"]):.4f}
Final Val Acc:     {history.history["val_accuracy"][-1]:.4f}
Random Baseline:   {random_baseline:.4f}
Improvement:       +{improvement:.1f}%

Training Time:     {train_time:.1f} seconds
Epochs:            {len(history.history["loss"])}
Gap (Train-Val):   {gap:+.4f}

Status:            {"✅ Good" if test_acc >= 0.70 else "⚠️ Acceptable" if test_acc >= 0.60 else "❌ Needs work"}
========================================
"""

with open(results_dir + "/tf_quick_summary.txt", "w") as f:
    f.write(quick_summary)

print("\n📁 Quick summary saved to:", results_dir + "/tf_quick_summary.txt")
print("=" * 60)
