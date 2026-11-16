"""Tests for Test-Time Augmentation."""
import torch
from src.models.baseline_cnn import BaselineCNN
from src.inference.tta_predictor import TTAPredictor

def test_tta_prediction_shape():
    model = BaselineCNN(num_classes=4)
    tta = TTAPredictor(model)
    dummy_scan = torch.randn(1, 3, 64, 64)
    probs = tta.predict_with_tta(dummy_scan)
    assert probs.shape == (1, 4)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)
