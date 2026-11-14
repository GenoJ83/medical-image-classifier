"""Multi-class diagnostic metrics: Sensitivity, Specificity, Confusion Matrix."""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from src.data.dataset import SyntheticChestXRayDataset, CLASSES
from src.models.resnet_classifier import MedicalResNet

def evaluate_radiology_model():
    model = MedicalResNet(num_classes=4)
    model.eval()
    dataset = SyntheticChestXRayDataset(100, img_size=64)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            img, target = dataset[i]
            output = model(img.unsqueeze(0))
            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_targets.append(target.item())

    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=CLASSES, zero_division=0)
    print("Clinical Classification Report:")
    print(report)
    return cm, report

if __name__ == "__main__":
    evaluate_radiology_model()
