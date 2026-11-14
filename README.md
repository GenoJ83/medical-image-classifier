# Medical Image Classifier: Chest Radiograph Pathology Detection

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Task](https://img.shields.io/badge/Task-Multi--Class%20Pathology-blue)
![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-purple)
![License](https://img.shields.io/badge/License-MIT-green)

A deep learning medical computer vision system designed to support clinical triage by classifying chest radiographs into 4 conditions:
1. Normal
2. Bacterial Pneumonia
3. Viral Pneumonia
4. Atelectasis

## Key Features
- **Transfer Learning**: Custom ResNet-18 architecture fine-tuned on pulmonary imaging
- **Explainable AI**: Grad-CAM visual heatmaps highlighting anatomical areas of interest
- **High Throughput**: Multi-threaded batch inference engine and CLI diagnostic utility

## Architecture & Workflow
```mermaid
flowchart TD
    Scan[Chest X-Ray DICOM/PNG] --> Preproc[Adaptive Normalization]
    Preproc --> ResNet[ResNet Backbone]
    ResNet --> ClassHead[Classification Head]
    ClassHead --> Triage[Triage Score & Grad-CAM Heatmap]
```

## Quick CLI Execution
```bash
python src/cli.py --batch-size 3
```
