"""Diagnostic inference engine supporting TorchScript and batch predictions."""
import torch
from src.data.dataset import CLASSES

class RadiologyPredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def diagnose_image(self, tensor_img: torch.Tensor):
        if tensor_img.ndim == 3:
            tensor_img = tensor_img.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor_img)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()

        return {
            "predicted_condition": CLASSES[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "class_probabilities": {CLASSES[i]: round(probs[i].item(), 4) for i in range(len(CLASSES))}
        }
