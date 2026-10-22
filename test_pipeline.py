"""
Automated sanity tests for CUB-200 classification pipeline.
Verifies:
1. Class names loaded (200 unique classes).
2. Metric calculation correctness.
3. DataLoader batch generation and tensor shapes.
4. Model forward pass on CUDA device with FP16.
5. Local Streamlit prediction helper.
"""

import os
import numpy as np
import torch
from utils import extract_and_save_class_names, load_class_names, compute_top1_accuracy, compute_average_class_accuracy
from dataset import build_dataloaders
from models import build_model
from app import predict_bird
from PIL import Image

def test_pipeline():
    print("[TEST 1/5] Testing class names...")
    classes = load_class_names("data/class_names.json")
    assert len(classes) == 200, f"Expected 200 classes, got {len(classes)}"
    print(f"  --> PASSED: 200 classes loaded. Sample: {classes[0]}")

    print("\n[TEST 2/5] Testing evaluation metrics formulas...")
    dummy_targets = np.array([0, 0, 1, 1, 2, 2])
    dummy_preds = np.array([0, 1, 1, 1, 2, 0])
    
    top1 = compute_top1_accuracy(dummy_preds, dummy_targets)
    # Correct: index 0 (0==0), index 2 (1==1), index 3 (1==1), index 4 (2==2) -> 4 / 6 = 0.6667
    assert abs(top1 - (4 / 6)) < 1e-4, f"Top-1 mismatch: {top1}"
    
    # Per class:
    # Class 0: 1/2 = 0.5
    # Class 1: 2/2 = 1.0
    # Class 2: 1/2 = 0.5
    # Ave over 3 classes: (0.5 + 1.0 + 0.5) / 3 = 2.0 / 3 = 0.6667
    ave_acc, per_class = compute_average_class_accuracy(dummy_preds, dummy_targets, num_classes=3)
    assert abs(ave_acc - (2.0 / 3)) < 1e-4, f"Ave class acc mismatch: {ave_acc}"
    print(f"  --> PASSED: Metrics strictly match assignment specifications.")

    print("\n[TEST 3/5] Testing DataLoader generation...")
    train_loader, val_loader, test_loader = build_dataloaders(batch_size=4, val_ratio=0.15)
    images, labels, filenames = next(iter(train_loader))
    assert images.shape == (4, 3, 224, 224), f"Unexpected image batch shape: {images.shape}"
    assert labels.shape == (4,), f"Unexpected label batch shape: {labels.shape}"
    print(f"  --> PASSED: Batch shape: {images.shape}, Labels: {labels.tolist()}")

    print("\n[TEST 4/5] Testing model forward pass with CUDA FP16...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arch in ["resnet50", "convnext_tiny"]:
        m = build_model(arch, num_classes=200, pretrained=False).to(device)
        m.eval()
        with torch.no_grad():
            img_dev = images.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = m(img_dev)
            assert out.shape == (4, 200), f"Expected (4, 200), got {out.shape}"
        print(f"  --> PASSED: {arch} forward pass successful on {device}!")

    print("\n[TEST 5/5] Testing local app prediction helper...")
    dummy_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    top5, top_index, confidence = predict_bird(dummy_img)
    assert len(top5) == 5, f"Expected 5 predictions, got {len(top5)}"
    assert 0 <= top_index < 200, f"Unexpected top class index: {top_index}"
    assert 0.0 <= confidence <= 1.0, f"Unexpected confidence: {confidence}"
    print(f"  --> PASSED: Local inference successful. Top 1: {list(top5.keys())[0]}")

    print("\n" + "="*50)
    print(" ALL 5 UNIT TESTS PASSED SUCCESSFULLY! ")
    print("="*50)

if __name__ == "__main__":
    test_pipeline()
