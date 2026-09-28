"""ResNet architecture with transfer learning classification head."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MedicalResNet(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
