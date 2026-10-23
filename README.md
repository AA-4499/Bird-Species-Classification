# COS30082 Applied Machine Learning: Assignment 1
## Bird Species Classification on Caltech-UCSD Birds 200 (CUB-200)

**Student Name:** Andy Wei Jian CHU  
**Student ID:** 104401283  
**Institution:** Swinburne University of Technology  
**Weighting:** 20%  
**Streamlit Application:** `streamlit run app.py`  

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

## Pre-Setting
Make sure the datasets: Train.zipDownload Train.zip, Test.zipDownload Test.zip. train.txtDownload train.txt, test.txt
Are inside the Folder before run the code


## Production

The production record uses the existing checkpoints and evaluation JSONs in
`checkpoints/` and `results/`. These artifacts remain separate from the
multi-seed experiment and are the reference for the deployed applications.

### Repository Structure

```
Assignment1/
├── app.py                      # Streamlit classifier, model metrics, and experiment dashboard
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

### Production Quickstart

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
Open the displayed local URL to use the three sidebar pages:
- **Home**: choose a trained model from the scrollable model selector, upload a bird image (or select a test image), and view the top-five predictions.
- **Model**: choose either model to view its saved training curves, per-class accuracy histogram, evaluation metrics, and most-confused species.
- **Experiment**: view multi-seed results after running the experiment workflow below.

### Production Application
The production classifier, model evaluation, and multi-seed experiment
dashboard are combined in the Streamlit application. Run it with:

```bash
streamlit run app.py
```

The Home page performs inference, the Model page presents official evaluation
results, and the Experiment page presents the experiment setup, figures,
interactive comparison, and final conclusion.

---

## Experiment

The multi-seed experiment repeats the complete training configuration for both
architectures with seeds **42, 43, and 44**, for exactly **15 epochs per run**.
It measures run-to-run variation; it does not isolate the effect of individual
augmentations, label smoothing, dropout, or optimizer settings.

### Run the 12 experiments

The four-way comparison is **pending until all 12 jobs finish**: two
architectures x two training modes x three seeds. The existing production
checkpoints and the results table below remain the current baseline.

Verify the schedule without starting training:

```bash
python experiment_runner.py --dry-run
```

Run all 12 training and evaluation jobs:

```bash
python experiment_runner.py
```

Each run is isolated under:

```text
experiments/runs/<model>/<mode>/seed_<seed>/
├── best_<model>.pth
├── latest_<model>.pth
├── history_<model>.json
├── test_metrics_<model>_<mode>.json
└── run_config.json
```

Aggregate metrics and the comparison plot are written to:

```text
experiments/results/multi_seed_summary.json
experiments/results/multi_seed_comparison.png
experiments/results/production_selection.json
```

The Experiment page also shows the combined seed-42 training-curves figure:

```text
experiments/results/visualizations/seed_42_training_curves_all_models.png
```

It compares training/validation loss and validation Top-1/average-class
accuracy for ResNet-50 and ConvNeXt-Tiny in both frozen and fine-tuned modes.
It also provides equivalent figures for seed 43, seed 44, and a combined
all-seed comparison, together with a seed-42 model-performance table.

The local Streamlit **Experiment** page reads those aggregate artifacts and
displays each architecture/mode seed score, mean, and standard deviation. The
runner selects the best production checkpoint for each architecture using
validation average class accuracy, then records the selected checkpoints and
their official test evaluations in `production_selection.json`. The local app
exposes both promoted experiment models as `Production (ResNet-50)` and
`Production (ConvNeXt-Tiny)` after the manifest is complete; otherwise it
continues using the current baseline checkpoints. Dedicated output directories
ensure experiment runs never overwrite production checkpoints or `results/`
artifacts.

The Streamlit **Experiment** tab follows a presentation-style sequence:
experiment setup, saved experiment result figures, interactive results, and a
final conclusion. It explains the four configurations, three seeds, metrics,
checkpoint-selection rule, and official test-set protocol before showing the
mean comparison, per-seed variation, and selected-run training curves. The
interactive section adds KPI cards, a Plotly comparison, and an expandable
full-results table. The conclusion appears last and recommends the preferred
production model with its measured improvement. Training curves, per-class
diagnostics, and confusion details remain available on the Model page. The
dashboard labels seed variation separately from confidence intervals and
distinguishes the validation-selected production checkpoint from the mean
across runs.

### Submission completeness checklist

- **Website:** Home displays a gallery of official test-set sample images,
  supports image upload or sample selection, and reports the
  top-1 species, confidence, and top-five probabilities. Model shows official
  Top-1 accuracy, average accuracy per class, training curves, class-wise
  performance, and confusion pairs. Experiment explains setup, shows saved
  figures, presents the multi-seed results, and ends with the model-selection
  conclusion.
- **Report:** `report/report_content.md` and
  `report/report_cos30082.tex` contain the required Methodology and Results and
  Discussion sections, student details, model design, loss, hyperparameters,
  overfitting discussion, comparison, and model justification. Insert the
  final YouTube recording URL before submission; it must not be fabricated.
- **Video:** `presentation/presentation_script.md` provides an under-10-minute
  slide-by-slide plan covering the problem, data, architectures, training,
  overfitting mitigation, code, results, Streamlit demonstration, and
  conclusion.

## 📊 Current Production Baseline (four-way comparison pending)

| Metric | Baseline: ResNet-50 | Advanced: ConvNeXt-Tiny | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **80.65%** | **89.53%** | **+8.88%** |
| **Average Accuracy per Class** | **80.11%** | **89.36%** | **+9.25%** |
| Parameters | 24.5M | 28.0M | +14.2% |
| Peak VRAM (FP16) | 1.12 GB | 1.18 GB | +5.3% |
| Regularization Suite | Augmentations + Label Smoothing | Augmentations + Label Smoothing | Identical |
