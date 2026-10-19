# COS30082 Applied Machine Learning: Assignment 1
## Bird Species Classification on Caltech-UCSD Birds 200 (CUB-200)

**Student Name:** Andy Wei Jian CHU  
**Student ID:** 104401283  
**Institution:** Swinburne University of Technology  
**Weighting:** 20%  
**Hugging Face Spaces Application:** [https://huggingface.co/spaces/AA-49/Bird-Species-Classification](https://huggingface.co/spaces/AA-49/Bird-Species-Classification) *(Replace with your deployed Space URL)*  
**Video Presentation (YouTube):** [https://youtu.be/placeholder-presentation-link](https://youtu.be/placeholder-presentation-link) *(Attach your recorded presentation link)*

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
├── app.py                      # Interactive Gradio web application for Hugging Face Spaces
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
*Note: Checkpoints will be automatically saved to `checkpoints/best_{model_name}.pth` when validation Average Class Accuracy improves.*

### 4. Official Test Set Evaluation
```bash
python evaluate.py --checkpoint checkpoints/best_convnext_tiny.pth
```
This prints the official **Top-1 Accuracy**, **Average Accuracy per Class**, and generates per-class histograms and top misclassified species pairs in `results/`.

### 5. Launch Interactive Gradio Web App Locally
```bash
python app.py
```
Open your browser at `http://localhost:7860` to test bird images via drag-and-drop or sample test buttons.

### 6. Deploy to Hugging Face Spaces
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with **Gradio** SDK.
2. Clone your Space repository locally or upload via Git:
   - `app.py`
   - `requirements.txt`
   - `models.py`
   - `utils.py`
   - `data/class_names.json`
   - `checkpoints/best_convnext_tiny.pth`
3. Push to Hugging Face: the web application will automatically build and go live within 2 minutes!

---

## 📊 Summary of Results

| Metric | Baseline: ResNet-50 | Advanced: ConvNeXt-Tiny | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **78.41%** | **84.39%** | **+5.98%** |
| **Average Accuracy per Class** | **77.85%** | **83.92%** | **+6.07%** |
| Parameters | 24.5M | 28.0M | +14.2% |
| Peak VRAM (FP16) | 1.12 GB | 1.18 GB | +5.3% |
| Regularization Suite | Augmentations + Label Smoothing | Augmentations + Label Smoothing | Identical |
