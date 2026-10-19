"""
Utility functions for CUB-200 Bird Species Classification.
Student: Andy Wei Jian CHU (ID: 104401283)
Subject: COS30082 Applied Machine Learning - Assignment 1

Provides:
- Class name extraction and formatting
- Evaluation metrics: Top-1 Accuracy and Average Accuracy per Class
- Plotting functions for training history and error analysis
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt


def format_species_name(raw_name: str) -> str:
    """
    Format raw folder/prefix name (e.g., 'Black_footed_Albatross') into 
    a clean human-readable name ('Black-footed Albatross' or 'Black Footed Albatross').
    """
    return raw_name.replace("_", " ").title()


def extract_and_save_class_names(train_txt_path: str, output_json_path: str = "data/class_names.json") -> dict:
    """
    Parse train.txt to extract all 200 unique class IDs and their corresponding
    species names. Saves the mapping as a JSON dictionary: {class_id: species_name}.
    """
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    
    class_map = {}
    with open(train_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                img_name = parts[0]
                class_id = int(parts[1])
                # Image filenames follow: Species_Name_####_########.jpg
                # Strip the trailing index and hash numbers
                tokens = img_name.split("_")
                raw_species = "_".join(tokens[:-2]) if len(tokens) > 2 else tokens[0]
                clean_species = format_species_name(raw_species)
                
                if class_id not in class_map:
                    class_map[class_id] = clean_species

    # Ensure all class indices from 0 to max are present
    sorted_map = {str(k): class_map[k] for k in sorted(class_map.keys())}
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(sorted_map, f, indent=2)
        
    print(f"[INFO] Saved {len(sorted_map)} class names to {output_json_path}")
    return {int(k): v for k, v in sorted_map.items()}


def load_class_names(json_path: str = "data/class_names.json") -> dict:
    """Load class mapping dictionary {int(class_id): species_name}."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Class names file not found at {json_path}. Run extract_and_save_class_names first.")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    return {int(k): v for k, v in raw_map.items()}


def compute_top1_accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute Top-1 Accuracy:
    Top-1 accuracy = (1 / N) * sum_{k=1}^N 1{argmax(y) == ground_truth}
    
    Args:
        preds: Array or Tensor of predicted class indices (N,) or logits/probabilities (N, C).
        targets: Array or Tensor of ground truth class indices (N,).
        
    Returns:
        float: Accuracy value between 0.0 and 1.0 (or percentage when multiplied by 100).
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    if preds.ndim == 2:
        pred_labels = np.argmax(preds, axis=1)
    else:
        pred_labels = preds
        
    correct = np.sum(pred_labels == targets)
    total = len(targets)
    return float(correct / total) if total > 0 else 0.0


def compute_average_class_accuracy(preds: np.ndarray, targets: np.ndarray, num_classes: int = 200) -> tuple:
    """
    Compute Average Accuracy per Class:
    Ave = (1 / C) * sum_{i=1}^C T_i
    where T_i is the average accuracy for all test images belonging to class C_i.
    
    Args:
        preds: Array or Tensor of predicted class indices (N,) or logits/probabilities (N, C).
        targets: Array or Tensor of ground truth class indices (N,).
        num_classes: Total number of unique classes (default: 200).
        
    Returns:
        tuple: (average_accuracy_per_class: float, per_class_accuracies: dict)
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    if preds.ndim == 2:
        pred_labels = np.argmax(preds, axis=1)
    else:
        pred_labels = preds
        
    per_class_acc = {}
    valid_classes_acc = []
    
    for c in range(num_classes):
        class_mask = (targets == c)
        class_total = np.sum(class_mask)
        if class_total > 0:
            class_correct = np.sum(pred_labels[class_mask] == c)
            acc = float(class_correct / class_total)
            per_class_acc[c] = acc
            valid_classes_acc.append(acc)
        else:
            per_class_acc[c] = 0.0
            
    # The assignment specifies dividing by C (total number of classes):
    overall_ave_class_acc = float(np.sum(valid_classes_acc) / num_classes)
    return overall_ave_class_acc, per_class_acc


def plot_training_history(history: dict, model_name: str, save_path: str = None):
    """
    Plot and save training and validation loss and accuracy curves across epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title(f"{model_name}: Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (Label Smoothing CE)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["val_top1"]], "o-", label="Val Top-1 Acc (%)", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_ave_class"]], "s--", label="Val Ave Class Acc (%)", color="#9467bd", linewidth=2)
    ax2.set_title(f"{model_name}: Validation Accuracy Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved training curves to {save_path}")
    plt.close()


def find_top_confused_pairs(y_true: np.ndarray, y_pred: np.ndarray, class_names: dict, top_k: int = 10) -> list:
    """
    Identify the top K most frequently misclassified (true_class, predicted_class) pairs.
    Crucial for fine-grained visual categorization error analysis in the report.
    """
    misclassified = {}
    for true_c, pred_c in zip(y_true, y_pred):
        if true_c != pred_c:
            pair = (int(true_c), int(pred_c))
            misclassified[pair] = misclassified.get(pair, 0) + 1
            
    sorted_pairs = sorted(misclassified.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for (true_c, pred_c), count in sorted_pairs:
        results.append({
            "true_class_id": true_c,
            "true_name": class_names.get(true_c, f"Class {true_c}"),
            "pred_class_id": pred_c,
            "pred_name": class_names.get(pred_c, f"Class {pred_c}"),
            "error_count": count
        })
    return results
