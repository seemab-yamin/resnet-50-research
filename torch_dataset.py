"""PyTorch ``Dataset`` backed by ``BaseDataLoader`` paths + optional ``torchvision`` transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torchvision.io as io
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from base_loader import BaseDataLoader


class TorchImageDataset(Dataset):
    """
    Load RGB image paths from a ``BaseDataLoader`` and apply an optional tensor transform.

    Images are read as ``float32`` in ``[0, 1]`` (channels first). Single-channel inputs are
    repeated to three channels. Normalization and augmentation should be provided by ``transform``
    (for example ``transforms.Normalize`` with mean/std supplied by the caller, e.g. ``config``).
    """

    def __init__(
        self,
        base_loader: BaseDataLoader,
        transform: Callable[[Tensor], Tensor] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        base_loader
            Split-specific loader exposing ``__len__``, ``__getitem__`` as ``(path, label)``.
        transform
            Callable on image tensors ``(C, H, W)``. Use ``None`` for decode + scaling only.
        """
        self.base_loader = base_loader
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_loader)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path, label = self.base_loader[idx]
        image = io.read_image(img_path).float() / 255.0

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
