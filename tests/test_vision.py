import pytest
pytest.importorskip("torch")
"""Unit tests for dataset shapes, transforms, and neural network layers."""
import torch
from src.data.dataset import SyntheticChestXRayDataset
from src.models.baseline_cnn import BaselineCNN
from src.models.resnet_classifier import MedicalResNet

def test_dataset_tensor_shapes():
    ds = SyntheticChestXRayDataset(10, img_size=64)
    img, target = ds[0]
    assert img.shape == (3, 64, 64)
    assert 0 <= target.item() < 4

def test_baseline_cnn_forward():
    model = BaselineCNN(num_classes=4)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 4)

def test_resnet_forward():
    model = MedicalResNet(num_classes=4)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 4)
