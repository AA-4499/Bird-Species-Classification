"""
Dataset loading and data augmentation pipeline for CUB-200.
Student: Andy Wei Jian CHU (ID: 104401283)
Subject: COS30082 Applied Machine Learning - Assignment 1

Features:
- Stratified 85/15 train-validation split across all 200 bird species
- High-performance data augmentations (RandomResizedCrop, Flip, ColorJitter, RandomErasing)
- Clean validation and test preprocessing pipeline
"""

import os
from typing import Tuple, List, Dict
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CUB200Dataset(Dataset):
    """
    PyTorch Dataset for Caltech-UCSD Birds 200 (CUB-200).
    Loads images from directory and pairs them with their integer class labels (0..199).
    """
    def __init__(self, image_dir: str, samples: List[Tuple[str, int]], transform=None):
        """
        Args:
            image_dir: Directory containing image files (e.g. 'Train' or 'Test').
            samples: List of (image_filename, class_label) tuples.
            transform: Optional torchvision transform to be applied.
        """
        self.image_dir = image_dir
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        filename, label = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        
        # Load image and convert to RGB (some CUB images are grayscale or have alpha channels)
        image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, filename


def parse_annotation_file(txt_path: str) -> List[Tuple[str, int]]:
    """Parse text file in the format: <filename> <class_label>"""
    samples = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                samples.append((parts[0], int(parts[1])))
    return samples


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Heavy domain-specific augmentation for fine-grained bird species classification:
    - RandomResizedCrop handles scale differences and zooming into body parts (beaks/plumage)
    - RandomHorizontalFlip preserves bird morphology while doubling orientation variability
    - ColorJitter simulates varying outdoor daylight and lighting conditions
    - RandomErasing forces network to avoid over-relying on a single visual cue
    """
    resize_dim = int(img_size * 1.15)  # 256 for 224
    return transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """Standard evaluation pipeline: Resize larger, then CenterCrop to target dimension."""
    resize_dim = int(img_size * 1.15)
    return transforms.Compose([
        transforms.Resize((resize_dim, resize_dim)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def create_stratified_split(
    train_txt_path: str, 
    val_ratio: float = 0.15, 
    random_state: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Perform a stratified split on the training annotations to ensure every bird species
    is proportionately represented in both the training and validation partitions.
    """
    all_samples = parse_annotation_file(train_txt_path)
    filenames = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    train_files, val_files, train_lbls, val_lbls = train_test_split(
        filenames,
        labels,
        test_size=val_ratio,
        stratify=labels,
        random_state=random_state
    )

    train_samples = list(zip(train_files, train_lbls))
    val_samples = list(zip(val_files, val_lbls))

    print(f"[INFO] Stratified split: {len(train_samples)} training samples, {len(val_samples)} validation samples.")
    return train_samples, val_samples


def build_dataloaders(
    train_dir: str = "Train",
    test_dir: str = "Test",
    train_txt: str = "train.txt",
    test_txt: str = "test.txt",
    batch_size: int = 16,
    img_size: int = 224,
    val_ratio: float = 0.15,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct DataLoaders for Training, Validation, and Official Testing.
    Note: On Windows, num_workers=0 avoids multi-processing semaphore issues.
    """
    train_samples, val_samples = create_stratified_split(train_txt, val_ratio=val_ratio)
    test_samples = parse_annotation_file(test_txt)
    print(f"[INFO] Official test set: {len(test_samples)} samples.")

    train_transform = get_train_transforms(img_size=img_size)
    eval_transform = get_eval_transforms(img_size=img_size)

    train_dataset = CUB200Dataset(train_dir, train_samples, transform=train_transform)
    val_dataset = CUB200Dataset(train_dir, val_samples, transform=eval_transform)
    test_dataset = CUB200Dataset(test_dir, test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
