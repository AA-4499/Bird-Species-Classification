# Current Production Baseline: ResNet-50 vs. ConvNeXt-Tiny

The four-way frozen-versus-fine-tuned comparison is pending completion of all
12 isolated runs. These values are preserved from the current production
checkpoints and are not the new experiment's aggregate results.

| Evaluation Metric | Baseline (ResNet-50) | Advanced (ConvNeXt-Tiny) | Delta Gain |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **80.65%** | **89.53%** | **+8.88%** |
| **Average Accuracy per Class** | **80.11%** | **89.36%** | **+9.25%** |
| Pretrained Source | ImageNet-1K | ImageNet-1K (fb_in22k_ft_in1k) | - |
| Parameter Count | 24.5 Million | 28.0 Million | +14.2% |
| Peak Inference VRAM | 1.12 GB | 1.18 GB | +5.3% |
| Input Resolution | 224 x 224 | 224 x 224 | Identical |
| Epochs Trained | 15 | 15 | Identical |

Generated for Andy Wei Jian CHU (104401283) - COS30082 Assignment 1.
