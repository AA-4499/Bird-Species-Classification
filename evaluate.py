"""
Evaluation script for CUB-200 official test dataset.
Student: Andy Wei Jian CHU (ID: 104401283)
Subject: COS30082 Applied Machine Learning - Assignment 1

Evaluates:
1. Overall Top-1 Accuracy: (1 / N) * sum 1{argmax(y) == ground_truth}
2. Average Accuracy per Class: (1 / C) * sum T_i
3. Detailed Fine-Grained Error Analysis (Most confused species pairs, hardest classes)
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import parse_annotation_file, get_eval_transforms, CUB200Dataset
from models import build_model
from utils import (
    load_class_names,
    compute_top1_accuracy,
    compute_average_class_accuracy,
    find_top_confused_pairs
)


def evaluate_checkpoint(
    checkpoint_path: str,
    test_dir: str = "Test",
    test_txt: str = "test.txt",
    class_names_path: str = "data/class_names.json",
    batch_size: int = 32,
    device: str = "cuda",
    output_dir: str = "results",
    result_name: str = None,
):
    """Run full evaluation on official test dataset using a trained checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating checkpoint: {checkpoint_path} on {device}")

    # 1. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "convnext_tiny")
    training_config = checkpoint.get("training_config", {})
    fine_tune = training_config.get("mode", "fine_tuned") == "fine_tuned"
    
    # 2. Build Model & Load Weights
    model = build_model(model_name=model_name, num_classes=200, pretrained=False, fine_tune=fine_tune)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load Test Data
    test_samples = parse_annotation_file(test_txt)
    eval_transform = get_eval_transforms(img_size=224)
    test_dataset = CUB200Dataset(test_dir, test_samples, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    class_names = load_class_names(class_names_path)

    # 4. Inference
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Calculate Metrics
    top1_acc = compute_top1_accuracy(all_preds, all_targets)
    ave_class_acc, per_class_acc = compute_average_class_accuracy(all_preds, all_targets, num_classes=200)

    # Error analysis
    confused_pairs = find_top_confused_pairs(all_targets, all_preds, class_names, top_k=10)

    # Sort classes by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    lowest_5 = [(c, class_names.get(c, f"Class {c}"), per_class_acc[c]) for c, _ in sorted_classes[:5]]
    highest_5 = [(c, class_names.get(c, f"Class {c}"), per_class_acc[c]) for c, _ in sorted_classes[-5:]]

    # Display Report
    print("\n" + "="*75)
    print(f" EVALUATION REPORT: {model_name.upper()} (Checkpoint: {os.path.basename(checkpoint_path)})")
    print("="*75)
    print(f" Total Testing Images Evaluated: {len(all_targets)}")
    print(f" [METRIC 1] Overall Top-1 Accuracy:       {top1_acc*100:.2f}%")
    print(f" [METRIC 2] Average Accuracy Per Class:    {ave_class_acc*100:.2f}%")
    print("-" * 75)
    print(" Top 5 Highest Performing Classes:")
    for cid, name, acc in reversed(highest_5):
        print(f"   - Class {cid:03d} ({name}): {acc*100:.1f}%")
    print("-" * 75)
    print(" Top 5 Lowest Performing Classes (Most Confused):")
    for cid, name, acc in lowest_5:
        print(f"   - Class {cid:03d} ({name}): {acc*100:.1f}%")
    print("-" * 75)
    print(" Top 5 Misclassified Species Pairs:")
    for item in confused_pairs[:5]:
        print(f"   - True: {item['true_name']}  -->  Predicted: {item['pred_name']} ({item['error_count']} errors)")
    print("="*75 + "\n")

    # 6. Save JSON Results
    results_payload = {
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "seed": checkpoint.get("seed"),
        "training_config": checkpoint.get("training_config", {}),
        "mode": training_config.get("mode", "fine_tuned"),
        "total_test_samples": len(all_targets),
        "top1_accuracy_pct": float(round(top1_acc * 100, 2)),
        "average_accuracy_per_class_pct": float(round(ave_class_acc * 100, 2)),
        "top_confused_pairs": confused_pairs,
        "lowest_5_classes": [{"class_id": c, "name": n, "accuracy_pct": round(a*100, 2)} for c, n, a in lowest_5],
        "highest_5_classes": [{"class_id": c, "name": n, "accuracy_pct": round(a*100, 2)} for c, n, a in highest_5]
    }

    os.makedirs(output_dir, exist_ok=True)
    result_stem = result_name or model_name
    out_json = os.path.join(output_dir, f"test_metrics_{result_stem}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[INFO] Saved test metrics to {out_json}")

    # 7. Generate Per-Class Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    acc_values = [v * 100 for v in per_class_acc.values()]
    ax.hist(acc_values, bins=20, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax.set_title(f"Per-Class Accuracy Distribution - {model_name.upper()}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class Accuracy (%)", fontsize=11)
    ax.set_ylabel("Number of Species (out of 200)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    hist_plot_path = os.path.join(output_dir, f"class_acc_hist_{result_stem}.png")
    plt.savefig(hist_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved histogram to {hist_plot_path}")

    return results_payload


def main():
    parser = argparse.ArgumentParser(description="Evaluate CUB-200 Classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--test_dir", type=str, default="Test", help="Path to test image folder")
    parser.add_argument("--test_txt", type=str, default="test.txt", help="Path to test.txt annotation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for evaluation artifacts")
    parser.add_argument("--result_name", type=str, default=None,
                        help="Optional result filename stem for isolated experiment modes")
    args = parser.parse_args()

    evaluate_checkpoint(
        args.checkpoint,
        test_dir=args.test_dir,
        test_txt=args.test_txt,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        result_name=args.result_name,
    )


if __name__ == "__main__":
    main()
