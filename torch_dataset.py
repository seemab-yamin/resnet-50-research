# PyTorch: wraps base_loader + transforms
import torch
from torch.utils.data import Dataset
import torchvision.io as io

class TorchMedicalDataset(Dataset):
    """PyTorch Dataset that uses your framework-agnostic base loader"""

    def __init__(self, base_loader, transform=None):
        """
        Args:
            base_loader: Your BaseDataLoader instance (already split-specific)
            transform: torchvision transforms to apply
        """
        self.base_loader = base_loader
        self.transform = transform

    def __len__(self):
        return len(self.base_loader)

    def __getitem__(self, idx):
        img_path, label = self.base_loader[idx]
        # Read image as tensor (C, H, W)
        image = io.read_image(img_path).float() / 255.0
    
        # Convert grayscale to RGB
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)