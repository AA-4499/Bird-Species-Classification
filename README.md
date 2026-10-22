# COS30082 Applied Machine Learning: Assignment 1
## Bird Species Classification on Caltech-UCSD Birds 200 (CUB-200)

**Student Name:** Andy Wei Jian CHU  
**Student ID:** 104401283  
**Institution:** Swinburne University of Technology  
**Weighting:** 20%  
**Hugging Face Spaces Application:** [https://huggingface.co/spaces/AA-49/Bird-Species-Classification](https://huggingface.co/spaces/AA-49/Bird-Species-Classification)

---

## 📌 Project Overview
This repository contains the complete PyTorch machine learning pipeline for fine-grained 200-class bird classification on the **Caltech-UCSD Birds 200 (CUB-200)** dataset.

The system addresses the core challenge of **Fine-Grained Visual Categorization (FGVC)** (low inter-class variation vs. high intra-class variation) and mitigates extreme overfitting on small sample sizes ($\approx 24$ images/class) through:
1. **Model Architecture Comparison**:
   - Baseline: Pretrained **ResNet-50**
   - Advanced: Modernized **ConvNeXt-Tiny** (depthwise $7\times7$ convolutions, inverted bottleneck, GELU, LayerNorm)
2. **Four-Tier Regularization Suite**:
   - Stratified 85/15 train-validation split across all 200 species
   - Domain-specific augmentations (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`, `RandomErasing`)
   - `LabelSmoothingCrossEntropy` ($\epsilon = 0.1$) to prevent overconfidence between similar species
   - `AdamW` decoupled weight decay ($\lambda = 0.01$) with Cosine Annealing learning rate schedule
3. **Hardware Optimization**: Automatic Mixed Precision (AMP FP16) optimized for execution on 4GB GPUs (NVIDIA RTX 2050).
4. **Mandatory Metrics Reported**:
   - **Top-1 Accuracy**: Overall classification accuracy on the 1,204 test images
   - **Average Accuracy per Class**: Unweighted mean of individual class accuracies across all 200 species

---

## 📂 Repository Structure

```
Assignment1/
├── app.py                      # Optional local Streamlit evaluation interface
├── hf_space/app.py             # Deployed Gradio web application
├── requirements.txt            # Deployment and execution dependencies
├── dataset.py                  # PyTorch Dataset, stratified split, and data augmentation transforms
├── models.py                   # Model builders for ResNet-50 and ConvNeXt-Tiny
├── train.py                    # Training script with FP16 AMP, CosineAnnealingLR, label smoothing
├── evaluate.py                 # Evaluation script computing Top-1 & Average Class Accuracy + confusion analysis
├── utils.py                    # Metric formulas, class name extraction, and history plotting
├── export_results.py           # Generates comparative curves and markdown summary tables
├── test_pipeline.py            # Automated sanity test suite verifying all 5 pipeline components
├── data/
│   └── class_names.json        # Formatted mapping from class ID (0..199) to clean species common names
├── report/
│   ├── report_content.md       # Full 4-page academic report in Markdown
│   └── report_cos30082.tex     # 4-page LaTeX template formatted for assignment submission
├── presentation/
│   └── presentation_script.md  # 10-minute video presentation slides outline and spoken narration
└── results/
    ├── model_comparison_curves.png # Comparison of loss and accuracy convergence
    └── comparison_summary.md       # Quantitative comparison metrics table
```

---

## 🚀 Quickstart & Usage Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Sanity Verification Tests
```bash
python test_pipeline.py
```

### 3. Model Training
To train the baseline **ResNet-50**:
```bash
python train.py --model resnet50 --epochs 15 --batch_size 16 --lr 1e-4
```

To train the advanced **ConvNeXt-Tiny**:
```bash
python train.py --model convnext_tiny --epochs 15 --batch_size 16 --lr 1e-4
```
Checkpoints are saved during training:
- `checkpoints/latest_{model_name}.pth` is overwritten after every epoch and contains the model, optimizer, scheduler, AMP scaler, epoch, and training history.
- `checkpoints/best_{model_name}.pth` is saved whenever validation Average Class Accuracy improves and is used for evaluation.

To resume interrupted training from the latest checkpoint and continue until epoch 30:
```bash
python train.py --model convnext_tiny --epochs 30 --resume checkpoints/latest_convnext_tiny.pth
```

The `--epochs` value is the target final epoch, not the number of additional epochs.

### 4. Official Test Set Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_convnext_tiny.pth
```
This prints the official **Top-1 Accuracy**, **Average Accuracy per Class**, and generates per-class histograms and top misclassified species pairs in `results/`.

### 5. Launch the Local Evaluation Interface
```bash
streamlit run app.py
```
Open the displayed local URL to use the two sidebar pages:
- **Home**: choose a trained model from the scrollable model selector, upload a bird image (or select a test image), and view the top-five predictions.
- **Model**: choose either model to view its saved training curves, per-class accuracy histogram, evaluation metrics, and most-confused species.

### 6. Deployed Hugging Face Space
The production Gradio application is maintained in [`hf_space/`](hf_space/) and is
available at the [deployed Hugging Face Space](https://huggingface.co/spaces/AA-49/Bird-Species-Classification).
It requires the trained checkpoint and `data/class_names.json`; startup fails
explicitly if either production asset is missing.

---

## 📊 Summary of Results

| Metric | Baseline: ResNet-50 | Advanced: ConvNeXt-Tiny | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **80.65%** | **89.53%** | **+8.88%** |
| **Average Accuracy per Class** | **80.11%** | **89.36%** | **+9.25%** |
| Parameters | 24.5M | 28.0M | +14.2% |
| Peak VRAM (FP16) | 1.12 GB | 1.18 GB | +5.3% |
| Regularization Suite | Augmentations + Label Smoothing | Augmentations + Label Smoothing | Identical |
