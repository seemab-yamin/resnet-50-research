"""TensorFlow ``tf.data.Dataset`` from ``BaseDataLoader`` paths and labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import tensorflow as tf

if TYPE_CHECKING:
    from base_loader import BaseDataLoader


class TFImageDataset:
    """
    Build a batched dataset of ``(image, label)`` from a path-based ``BaseDataLoader``.

    Images are decoded as RGB, resized, scaled to ``[0, 1]``, then normalized with ``mean`` /
    ``std`` (e.g. from ``config``). Optional horizontal flip when ``augment`` is True.
    """

    def __init__(
        self,
        base_loader: BaseDataLoader,
        img_size: tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
        batch_size: int = 32,
        augment: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        base_loader
            Split-specific loader; uses ``samples`` as ``list[tuple[path, label]]``.
        img_size
            ``(height, width)`` for ``tf.image.resize``.
        mean, std
            Three-channel normalize values after scaling to ``[0, 1]``.
        batch_size
            Batch size after preprocessing.
        augment
            If True, apply random horizontal flip (training).
        """
        self.base_loader = base_loader
        self.img_size = img_size
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.batch_size = batch_size
        self.augment = augment

    def build(self) -> tf.data.Dataset:
        mean_tf = tf.constant(self.mean, dtype=tf.float32, shape=(1, 1, 3))
        std_tf = tf.constant(self.std, dtype=tf.float32, shape=(1, 1, 3))

        paths = [p for p, _ in self.base_loader.samples]
        labels = [int(y) for _, y in self.base_loader.samples]
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

        def load(path, label):
            return _load_and_preprocess(
                path, label, self.img_size, mean_tf, std_tf, self.augment
            )

        dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


def _load_and_preprocess(path, label, img_size, mean_tf, std_tf, augment):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    if augment:
        image = tf.image.random_flip_left_right(image)
    image = (image - mean_tf) / std_tf
    return image, tf.cast(label, tf.int32)
