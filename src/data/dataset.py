"""Synthetic chest X-ray image tensor generator and PyTorch Dataset."""
import torch
from torch.utils.data import Dataset
import numpy as np

CLASSES = ["Normal", "Bacterial_Pneumonia", "Viral_Pneumonia", "Atelectasis"]

class SyntheticChestXRayDataset(Dataset):
    def __init__(self, num_samples: int = 400, img_size: int = 128, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        rng = np.random.default_rng(42)
        self.labels = rng.integers(0, len(CLASSES), size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic radiograph pattern (lung cavity simulation)
        x = np.linspace(-1, 1, self.img_size)
        y = np.linspace(-1, 1, self.img_size)
        xx, yy = np.meshgrid(x, y)
        base = np.exp(-(xx**2 + yy**2) * 1.5)
        noise = np.random.normal(0, 0.05, (self.img_size, self.img_size))
        img = np.clip(base + noise, 0, 1).astype(np.float32)
        tensor_img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)

        if self.transform:
            tensor_img = self.transform(tensor_img)

        return tensor_img, torch.tensor(self.labels[idx], dtype=torch.long)
