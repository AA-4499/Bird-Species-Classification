"""
Training pipeline for CUB-200 Bird Species Classification with Overfitting Mitigation.
Student: Andy Wei Jian CHU (ID: 104401283)
Subject: COS30082 Applied Machine Learning - Assignment 1

Features:
- Mixed Precision (FP16 AMP) for fast execution on RTX 2050 (4GB VRAM)
- Regularization: Label Smoothing, Weight Decay (AdamW), Dropout
- Cosine Annealing Learning Rate Schedule with Warmup
- Continuous tracking of Top-1 Accuracy and Average Accuracy per Class
- Best model checkpointing based on Validation Average Class Accuracy
"""

import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from dataset import build_dataloaders
from models import build_model, get_parameter_groups
from utils import compute_top1_accuracy, compute_average_class_accuracy, plot_training_history


def set_seed(seed: int = 42):
    """Ensure experimental reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Execute one training epoch with Automatic Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        # Gradient clipping to prevent gradient explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_top1 = compute_top1_accuracy(np.array(all_preds), np.array(all_targets))
    return epoch_loss, epoch_top1


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, num_classes=200):
    """Evaluate model on validation or test dataset."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    top1 = compute_top1_accuracy(all_preds, all_targets)
    ave_class, per_class = compute_average_class_accuracy(all_preds, all_targets, num_classes=num_classes)
    return epoch_loss, top1, ave_class, per_class


def main():
    parser = argparse.ArgumentParser(description="Train CUB-200 Bird Classifier")
    parser.add_argument("--model", type=str, default="convnext_tiny", choices=["resnet50", "convnext_tiny"],
                        help="Model architecture to train (resnet50 or convnext_tiny)")
    parser.add_argument("--mode", choices=["frozen", "fine_tuned"], default="fine_tuned",
                        help="Train only the new head (frozen) or fine-tune the pretrained backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 fits comfortably in 4GB VRAM)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay for regularization")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--drop_rate", type=float, default=0.3, help="Dropout probability in classifier head")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation split proportion (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save training curves")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Build DataLoaders
    print(f"[INFO] Loading CUB-200 dataset with stratified {int((1-args.val_ratio)*100)}/{int(args.val_ratio*100)} split...")
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=0  # Safe for Windows subprocesses
    )

    # 2. Build Model
    model = build_model(
        model_name=args.model,
        num_classes=200,
        pretrained=True,
        drop_rate=args.drop_rate,
        fine_tune=args.mode == "fine_tuned",
    ).to(device)

    # 3. Setup Loss, Optimizer, and LR Scheduler
    # Label smoothing explicitly helps combat overconfident memorization on fine-grained birds
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if args.mode == "frozen":
        param_groups = [{"params": trainable_params, "lr": args.lr * 5.0}]
    else:
        param_groups = get_parameter_groups(model, model_name=args.model, base_lr=args.lr, head_lr_mult=5.0)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # 4. Training Loop
    history = {
        "train_loss": [],
        "train_top1": [],
        "val_loss": [],
        "val_top1": [],
        "val_ave_class": []
    }

    best_val_ave_acc = 0.0
    best_ckpt_path = os.path.join(args.save_dir, f"best_{args.model}.pth")
    latest_ckpt_path = os.path.join(args.save_dir, f"latest_{args.model}.pth")
    start_epoch = 1

    if args.resume:
        print(f"[INFO] Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_model = checkpoint.get("model_name", args.model)
        if checkpoint_model != args.model:
            raise ValueError(
                f"Checkpoint model '{checkpoint_model}' does not match requested model '{args.model}'."
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_ave_acc = checkpoint.get("best_val_ave_class", checkpoint.get("val_ave_class", 0.0))
        history = checkpoint.get("history", history)
        print(f"[INFO] Continuing from epoch {start_epoch} (best Val Ave Class: "
              f"{best_val_ave_acc*100:.2f}%)")

    print(f"\n{'='*70}")
    print(f" Starting Training: {args.model.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"{'='*70}\n")

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_top1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_top1, val_ave_acc, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_top1"].append(train_top1)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_ave_class"].append(val_ave_acc)

        epoch_duration = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s, lr={current_lr:.6f}) | "
              f"Train Loss: {train_loss:.4f}, Top-1: {train_top1*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Top-1: {val_top1*100:.2f}%, Val Ave Class: {val_ave_acc*100:.2f}%")

        improved = val_ave_acc > best_val_ave_acc
        if improved:
            best_val_ave_acc = val_ave_acc

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model,
            "seed": args.seed,
            "training_config": {
                "epochs": args.epochs,
                "mode": args.mode,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "drop_rate": args.drop_rate,
                "val_ratio": args.val_ratio,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_top1": val_top1,
            "val_ave_class": val_ave_acc,
            "best_val_ave_class": best_val_ave_acc,
            "history": history
        }

        # Save the latest state every epoch so interrupted training can resume.
        torch.save(checkpoint, latest_ckpt_path)
        print(f"  --> Saved latest checkpoint to {latest_ckpt_path}")

        # Save checkpoint if validation average class accuracy improved
        if improved:
            torch.save(checkpoint, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Ave Class Acc: {best_val_ave_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {total_time/60:.2f} minutes. Best Val Ave Class Acc: {best_val_ave_acc*100:.2f}%")

    # Save training history JSON
    history_path = os.path.join(args.save_dir, f"history_{args.model}.json")
    history_payload = {
        "model_name": args.model,
        "seed": args.seed,
        "training_config": checkpoint["training_config"],
        "history": history,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    # Plot training curves
    plot_training_history(
        history,
        model_name=args.model.upper(),
        save_path=os.path.join(args.results_dir, f"curves_{args.model}.png"),
    )


if __name__ == "__main__":
    main()
