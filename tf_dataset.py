# TensorFlow: wraps base_loader + preprocessing

import tensorflow as tf

def create_tf_dataset(base_loader, IMG_SIZE, MEAN, STD, batch_size, augment=False):
    """
    Creates a tf.data.Dataset from your base loader.

    Args:
        base_loader: BaseDataLoader instance (already split-specific)
        batch_size: Number of samples per batch
        augment: Whether to apply data augmentation (training only)

    Returns:
        tf.data.Dataset yielding (image, label) batches
    """
    # Extract all paths and labels from base loader
    paths = [sample[0] for sample in base_loader.samples]
    labels = [sample[1] for sample in base_loader.samples]

    # Convert to tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    # Apply preprocessing
    dataset = dataset.map(
        lambda path, label: _load_and_preprocess(path, label, IMG_SIZE, MEAN, STD, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def _load_and_preprocess(path, label, IMG_SIZE, MEAN, STD, augment):
    """Load image, decode, resize, normalize, and optionally augment"""
    # Read and decode image
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Scale to [0,1]

    # Augmentation (training only)
    if augment:
        image = tf.image.random_flip_left_right(image)  # Same as PyTorch RandomHorizontalFlip

    # Normalize using same ImageNet stats as PyTorch
    mean = tf.constant(MEAN, dtype=tf.float32, shape=[1, 1, 3])
    std = tf.constant(STD, dtype=tf.float32, shape=[1, 1, 3])
    image = (image - mean) / std

    return image, label