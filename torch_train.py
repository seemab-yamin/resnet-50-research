# TensorFlow: model building, training, evaluation

import tensorflow as tf
from tensorflow.keras import layers, models
from base_loader import BaseDataLoader
from tf_dataset import create_tf_dataset
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
import config
import json
import time
import numpy as np

def build_resnet50_model(num_classes):
    """
    Builds ResNet50 model with frozen base + custom classification head.
    Matches PyTorch architecture exactly.
    """
    # Load pretrained ResNet50 without top (classification) layers
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', # weights learned on 1.2 million images
        include_top=False, # strip off original classification layer
        input_shape=(224, 224, 3), # input image W, H, Channel
        pooling='avg'  # Global average pooling — matches PyTorch's AdaptiveAvgPool2d
    )

    # Freeze base layers (feature extraction only — matches PyTorch freeze=True)
    base_model.trainable = False
    # Check if base model is frozen
    print(f"Base model trainable: {base_model.trainable}")  # Should be False

    # Add custom classification head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # add dense layers to classify specific classes
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def configure_training():
    """Setup loss, optimizer, and metrics matching PyTorch"""
    
    # Loss: CrossEntropyLoss (matches PyTorch)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # Optimizer: SGD with same parameters as PyTorch
    # PyTorch: SGD(lr=config.LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Metrics: Accuracy (matches PyTorch)
    metrics = ['accuracy']
    
    return loss, optimizer, metrics


class TrainingLogger(tf.keras.callbacks.Callback):
    """Custom callback to save epoch metrics to JSON"""
    def __init__(self, log_file='results/tf_history.json'):
        super().__init__()
        self.log_file = log_file
        self.history = {'epoch': [], 'loss': [], 'accuracy': [], 
                        'val_loss': [], 'val_accuracy': [], 'time': []}
        self.epoch_start = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        self.history['epoch'].append(epoch)
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['time'].append(epoch_time)
        
        # Save after each epoch (in case of crash)
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=4)

def get_callbacks():
    """Create callbacks for checkpointing and early stopping"""
    callbacks = [
        # Save best model
        ModelCheckpoint(
            filepath=config.PROJECT_ROOT_DIR + '/results/tf_best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # TensorBoard logging for comparison
        TensorBoard(
            log_dir=config.PROJECT_ROOT_DIR + '/results/tensorboard',
            histogram_freq=1
        ),
        CSVLogger(config.PROJECT_ROOT_DIR + '/results/tf_training_log.csv'),  # Simple CSV backup
        TrainingLogger(config.PROJECT_ROOT_DIR + '/results/tf_history.json')  # Custom JSON logger
    ]
    return callbacks