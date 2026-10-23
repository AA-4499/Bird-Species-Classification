"""
Neural Network Architectures for CUB-200 Bird Species Classification.
Student: Andy Wei Jian CHU (ID: 104401283)
Subject: COS30082 Applied Machine Learning - Assignment 1

Provides:
- Baseline Model: ResNet-50 (Deep Residual Learning)
- Advanced Model: ConvNeXt-Tiny (Modernized CNN with Vision Transformer design)
- Custom classification heads with dropout regularization for overfitting mitigation
"""

import torch
import torch.nn as nn
from torchvision import models as tv_models
import timm


def build_model(
    model_name: str = "convnext_tiny",
    num_classes: int = 200,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    fine_tune: bool = True,
) -> nn.Module:
    """
    Construct and initialize the model architecture.

    Args:
        model_name: 'resnet50' (Baseline) or 'convnext_tiny' (Advanced).
        num_classes: Number of bird species to predict (200 for CUB-200).
        pretrained: Whether to load ImageNet-1K pretrained weights.
        drop_rate: Dropout probability applied before final linear projection.
        fine_tune: If false, freeze the pretrained backbone and train only the
            newly initialized CUB-200 classifier head.

    Returns:
        nn.Module: Instantiated PyTorch neural network.
    """
    name = model_name.lower()

    if name == "resnet50":
        # Baseline Architecture: Deep Residual Network (He et al., 2016)
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        
        # Replace 1000-class ImageNet head with customized CUB-200 head
        model.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(in_features, num_classes)
        )
        print(f"[INFO] Initialized ResNet-50 (Pretrained={pretrained}, Dropout={drop_rate})")

    elif name == "convnext_tiny":
        # Advanced Architecture: ConvNeXt (Liu et al., CVPR 2022)
        # Modernized depthwise 7x7 convolutions, inverted bottleneck, GELU, and LayerNorm
        model = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
        print(f"[INFO] Initialized ConvNeXt-Tiny (Pretrained={pretrained}, Dropout={drop_rate})")

    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'. Choose 'resnet50' or 'convnext_tiny'.")

    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        if name == "resnet50":
            head = model.fc
        else:
            head = model.head
        for param in head.parameters():
            param.requires_grad = True
        print(f"[INFO] Frozen {name} backbone; training classification head only")

    return model


def get_parameter_groups(model: nn.Module, model_name: str, base_lr: float = 1e-4, head_lr_mult: float = 5.0):
    """
    Separate backbone and head parameters to apply differential learning rates.
    Fine-tuning a pretrained backbone requires a smaller learning rate, while
    the newly initialized classification head learns faster with a higher learning rate.
    """
    head_params = []
    backbone_params = []

    if model_name.lower() == "resnet50":
        head_names = ["fc"]
    else:  # convnext_tiny
        head_names = ["head"]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * head_lr_mult}
    ]
