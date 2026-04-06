import os
import random
from pathlib import Path

class BaseDataLoader:
    """
    Framework-agnostic data loader that provides image paths and labels.
    Both PyTorch and TensorFlow will use this as their source of truth.
    """

    def __init__(self, data_root, split="Train", seed=42, is_train_shuffle=True):
        """
        Args:
            data_root: Path to dataset root (e.g., "data/")
            split: "Train", "Val", or "Test"
            seed: Random seed for reproducibility
            is_train_shuffle: If True, shuffles samples for the "Train" split.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.seed = seed
        # Initialize random seed for reproducibility in operations like shuffling
        random.seed(seed)

        # Build index
        self.samples = []  # List of (image_path, label)
        self.class_to_idx = {}

        split_path = self.data_root / split
        if not split_path.exists():
            raise ValueError(f"Split path not found: {split_path}")

        # Assumes folder structure: split/class_name/image.jpg
        for class_name in sorted(os.listdir(split_path)):
            class_path = split_path / class_name
            if not class_path.is_dir():
                continue

            # Assign integer label
            label_idx = len(self.class_to_idx)
            self.class_to_idx[class_name] = label_idx

            # Add all images in this class
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_path / img_file
                    self.samples.append((str(img_path), label_idx))

        # Optionally shuffle (for training) based on is_train_shuffle argument
        if split == "Train" and is_train_shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (image_path, label)"""
        return self.samples[idx]

    def get_all_paths_and_labels(self):
        """Returns all (path, label) pairs"""
        return self.samples

    def get_class_mapping(self):
        """Returns dict: class_name -> label_idx"""
        return self.class_to_idx