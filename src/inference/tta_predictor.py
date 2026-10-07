"""Test-Time Augmentation (TTA) predictor for radiology diagnostics."""
import torch

class TTAPredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_with_tta(self, image_tensor: torch.Tensor, n_augmentations: int = 4):
        """Run multiple augmented forward passes and average logits."""
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        predictions = []
        with torch.no_grad():
            # Original
            predictions.append(torch.softmax(self.model(image_tensor), dim=-1))
            # Horizontal flip
            flipped = torch.flip(image_tensor, dims=[-1])
            predictions.append(torch.softmax(self.model(flipped), dim=-1))

        stacked = torch.stack(predictions)
        mean_prob = torch.mean(stacked, dim=0)
        return mean_prob
