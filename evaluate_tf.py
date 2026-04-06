# evaluate_tf.py
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import sys

common_modules_path = "/content/drive/MyDrive/ai-projects/2026-03-transfer-learning-resnet-50"
sys.path.append(common_modules_path)
print(f"Updated sys.path: {sys.path}")

def evaluate_and_save(model, test_dataset, class_names, save_dir='results/'):
    """Run full evaluation and save all metrics"""
    
    # Get all predictions
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'test_accuracy': float(accuracy_score(y_true, y_pred)),
        'test_precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'test_recall_macro': float(recall_score(y_true, y_pred, average='macro')),
        'test_f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'class_names': class_names,
        'num_samples': len(y_true)
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['per_class'] = report
    
    # Save metrics
    with open(f'{save_dir}/tf_test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save predictions (for later analysis)
    np.savez(f'{save_dir}/tf_predictions.npz',
             y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba)
    
    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(f'{save_dir}/tf_confusion_matrix.npy', cm)
    
    print(f"✅ Saved metrics, predictions, and confusion matrix to {save_dir}")
    return metrics

# Run evaluation
test_base = BaseDataLoader(config.COVIDQU_PATH, split="Test", is_train_shuffle=False, seed=config.SEED)
test_dataset = create_tf_dataset(test_base, batch_size=config.BATCH_SIZE, augment=False)
metrics = evaluate_and_save(model, test_dataset, class_names=['COVID-19', 'Normal', ])