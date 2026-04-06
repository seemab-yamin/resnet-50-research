"""Generic filesystem indexer for class-folder classification datasets."""

from __future__ import annotations

import os
import random
from pathlib import Path

# Filenames under each class directory must end with one of these (case-insensitive).
DEFAULT_FILE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class BaseDataLoader:
    """
    Index samples from a nested directory layout without binding to a specific ML framework.

    Expected layout::

        {data_root}/{split_name}/{class_name}/<files>

    Each immediate subfolder of ``split_name`` is a class; eligible files inside become
    one sample each as ``(absolute_path_str, integer_label)``.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "Train",
        seed: int = 42,
        is_train_shuffle: bool = True,
        *,
        file_extensions: tuple[str, ...] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        data_root
            Root directory containing per-split subdirectories.
        split
            Name of the split subdirectory under ``data_root`` (e.g. ``Train``, ``Val``, ``Test``).
        seed
            Seed for ``random`` (e.g. shuffling).
        is_train_shuffle
            If True, shuffle the sample list when ``split`` is ``"Train"``.
        file_extensions
            Allowed filename suffixes, lowercase with leading dot. Defaults to ``DEFAULT_FILE_EXTENSIONS``.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.seed = seed
        exts = (
            file_extensions if file_extensions is not None else DEFAULT_FILE_EXTENSIONS
        )
        self._extensions = tuple(e.lower() for e in exts)

        random.seed(seed)

        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}

        split_path = self.data_root / split
        if not split_path.exists():
            raise ValueError(f"Split path not found: {split_path}")

        for class_name in sorted(os.listdir(split_path)):
            class_path = split_path / class_name
            if not class_path.is_dir():
                continue

            label_idx = len(self.class_to_idx)
            self.class_to_idx[class_name] = label_idx

            for filename in os.listdir(class_path):
                if filename.lower().endswith(self._extensions):
                    file_path = class_path / filename
                    self.samples.append((str(file_path), label_idx))

        if split == "Train" and is_train_shuffle:
            random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Return the ``idx``-th ``(file_path, label_index)`` pair."""
        return self.samples[idx]

    def get_all_paths_and_labels(self) -> list[tuple[str, int]]:
        """Return all ``(file_path, label_index)`` pairs (same list as ``samples``)."""
        return self.samples

    def get_class_mapping(self) -> dict[str, int]:
        """Return ``class_name -> label_index`` (stable order from sorted class directory names)."""
        return self.class_to_idx
