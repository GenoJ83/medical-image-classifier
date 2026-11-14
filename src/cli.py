"""Command-line utility for chest radiograph classification."""
import argparse
import torch
from src.models.resnet_classifier import MedicalResNet
from src.inference.infer import RadiologyPredictor

def main():
    parser = argparse.ArgumentParser(description="Chest X-Ray Diagnostic CLI")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of simulated scans to analyze")
    args = parser.parse_args()

    model = MedicalResNet(num_classes=4)
    predictor = RadiologyPredictor(model)
    dummy_scan = torch.randn(args.batch_size, 3, 128, 128)
    
    print(f"Analyzing {args.batch_size} radiograph scan(s)...")
    for i in range(args.batch_size):
        result = predictor.diagnose_image(dummy_scan[i])
        print(f"Scan #{i+1}: {result['predicted_condition']} (Confidence: {result['confidence']:.2%})")

if __name__ == "__main__":
    main()
