"""Training loop with learning rate scheduling and validation."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.dataset import SyntheticChestXRayDataset
from src.models.resnet_classifier import MedicalResNet

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, targets in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def run_training_experiment(epochs: int = 2):
    dataset = SyntheticChestXRayDataset(128, img_size=64)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = MedicalResNet(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion)
        print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.2%}")
    return model

if __name__ == "__main__":
    run_training_experiment()
