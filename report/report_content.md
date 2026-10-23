# COS30082 Applied Machine Learning: Assignment 1
## Bird Species Classification on Caltech-UCSD Birds 200 (CUB-200)

**Student Name:** Andy Wei Jian CHU  
**Student ID:** 104401283  
**Institution:** Swinburne University of Technology  
**Hugging Face Spaces Application:** [https://huggingface.co/spaces/AA-49/Bird-Species-Classification](https://huggingface.co/spaces/AA-49/Bird-Species-Classification)  

---

## 1. Methodology

### 1.1 Problem Formulation & Challenges
The Caltech-UCSD Birds 200 (CUB-200) benchmark poses a classic **Fine-Grained Visual Categorization (FGVC)** problem containing 200 distinct bird species across 4,829 training images and 1,204 testing images. The dataset presents two fundamental challenges:
1. **Low Inter-Class Variance:** Morphologically adjacent species (such as the *Black-footed Albatross* vs. *Laysan Albatross*, or various subspecies of gulls and warblers) share identical silhouettes, differing only in minute regional cues such as bill tip coloration, eye rings, or covert feather patterns.
2. **High Intra-Class Variance & Severe Overfitting Risk:** Birds appear in diverse flight postures, perch angles, and occluding foliage under widely fluctuating outdoor illumination. With only approximately 24 training samples per species, conventional Deep Convolutional Neural Networks (CNNs) trained from scratch rapidly overfit by memorizing background artifacts (sky, tree branches, lake surfaces) rather than bird-specific diagnostic features.

To resolve these challenges, we design an end-to-end transfer learning framework featuring a rigorous comparative experiment between a canonical baseline (**ResNet-50**) and a modernized fine-grained architecture (**ConvNeXt-Tiny**), coupled with domain-specific regularization and data augmentation schemes.

---

### 1.2 Model Architectures

#### Architecture Comparison: ResNet-50 and ConvNeXt-Tiny
We compare two pretrained architectures (ResNet-50 and ConvNeXt-Tiny) under two training regimes: (1) a frozen feature extractor, where only the new 200-class classifier head is trained, and (2) end-to-end fine-tuning, where the backbone and head are trained with differential learning rates. This yields four architecture/regime combinations and supports both within-architecture comparisons (frozen vs. fine-tuned) and cross-architecture comparisons (ResNet-50 vs. ConvNeXt-Tiny at each regime). The existing production results below are fine-tuned model results; frozen-regime results must be populated from the new experiment before making claims about which regime performs best.

#### ResNet-50 (Residual Learning)
As a robust baseline, we employ **ResNet-50** (He et al., 2016), pre-trained on ImageNet-1K. ResNet utilizes identity skip connections to circumvent vanishing gradient phenomena across deep convolutional hierarchies:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
The network consists of four sequential residual stages with bottleneck building blocks ($1\times1$, $3\times3$, $1\times1$ convolutions), culminating in a 2048-dimensional feature representation following Global Average Pooling (GAP). To adapt the backbone for CUB-200, the original 1000-way linear classifier was detached and replaced with a fine-grained classification head:
$$\mathbf{z} = \mathbf{W}_{\text{head}} \cdot \text{Dropout}_{p=0.3}(\text{GAP}(\mathbf{F})) + \mathbf{b}_{\text{head}}$$
where $\mathbf{W}_{\text{head}} \in \mathbb{R}^{200 \times 2048}$. While ResNet-50 offers dependable representation capability, its standard $3\times3$ kernel structure exhibits a constrained effective receptive field early in the network, restricting its capacity to correlate distant bird attributes (e.g., matching bill shape with tail plumage).

#### ConvNeXt-Tiny (Modernized Pure Convolutional Network)
To surpass standard CNN limitations while avoiding the quadratic memory overhead of pure Vision Transformers, we introduce **ConvNeXt-Tiny** (Liu et al., CVPR 2022). ConvNeXt modernizes traditional ConvNets by incorporating pivotal architectural innovations from the Swin Transformer:
1. **Macro Design & Inverted Bottleneck:** Inversion of bottleneck channel dimensions (channel expansion ratio of 4 in intermediate layers) prevents information bottlenecks when processing intricate feather patterns.
2. **Large Receptive Field with Depthwise Convolutions:** By adopting $7\times7$ depthwise convolutions in place of $3\times3$ spatial filters, ConvNeXt dramatically expands its effective receptive field from the lowest stage, allowing simultaneous capture of macro body shape and subtle localized markings.
3. **Micro-level Regularization:** Standard ReLU activations and Batch Normalization layers are replaced with **GELU (Gaussian Error Linear Units)** and **Layer Normalization (LN)** applied sparingly. This stabilizes gradient updates across multi-channel feature maps without batch-size sensitivity.

---

### 1.3 Data Pipeline & Overfitting Mitigation Strategies

To rigorously quantify and eliminate overfitting on CUB-200, we implemented an integrated multi-tiered regularization pipeline:

#### 1. Stratified 85/15 Validation Split
Rather than validating against the test partition during training, the 4,829 training samples were partitioned into a **Stratified Training Set (4,104 images)** and a **Stratified Validation Set (725 images)** using fixed seed randomization ($\text{seed}=42$). This guarantees that each of the 200 bird species possesses proportional representation in both splits, enabling uncorrupted hyperparameter tuning and reliable early checkpointing.

#### 2. Domain-Specific Data Augmentation Suite
Given that birds appear at variable distances and orientations, training inputs undergo dynamic stochastic augmentation:
- **Random Resized Crop ($224\times224$, scale $0.5$ to $1.0$):** Forces scale invariance by simulating telephoto lens zooming and cropping into regional diagnostics (head, wingtips).
- **Random Horizontal Flip ($p=0.5$):** Enforces bilateral pose invariance.
- **Color Jitter (Brightness $0.2$, Contrast $0.2$, Saturation $0.2$, Hue $0.05$):** Emulates varying daylight temperatures and shaded tree canopies, neutralizing color shift memorization.
- **Random Erasing ($p=0.2$, scale $0.02$ to $0.2$):** Simulates natural occlusions by leaves and twigs, forcing the network to discover distributed visual evidence rather than a single cue.

#### 3. Regularization & Loss Function: Label Smoothing Cross-Entropy
Conventional hard-target Cross-Entropy forces output logits of the true class toward positive infinity, provoking extreme overconfidence between visual twins. We employ **Label Smoothing Cross-Entropy** ($\epsilon = 0.1$):
$$y_k^{\text{smooth}} = (1 - \epsilon)\delta_{k, y} + \frac{\epsilon}{C}$$
$$\mathcal{L}_{\text{LS}}(\mathbf{p}, \mathbf{y}) = - \sum_{k=1}^C y_k^{\text{smooth}} \log(p_k)$$
where $C=200$. This regularization exerts a soft penalty against overconfident predictions, preserving logit margin between visually similar candidate species.

#### 4. Optimization Scheme & Hardware Optimization
- **Optimizer:** **AdamW** with decoupled weight decay ($\lambda = 0.01$) to penalize large weight magnitudes.
- **Differential Learning Rates:** Pretrained backbone parameters are fine-tuned with a restrained base learning rate ($\eta_{\text{base}} = 1\times 10^{-4}$), while the newly initialized classification head learns at a higher rate ($\eta_{\text{head}} = 5\times 10^{-4}$).
- **Cosine Annealing Schedule:** A cosine decay schedule progressively anneals learning rates to $\eta_{\text{min}} = 1\times 10^{-6}$ over 15 epochs.
- **Automatic Mixed Precision (AMP FP16):** Implemented via `torch.cuda.amp.autocast()`, decreasing VRAM usage to $\sim 1.8$ GB and accelerating gradient passes on the local NVIDIA GeForce RTX 2050 (4GB VRAM).

---

## 2. Results and Discussion

### 2.1 Quantitative Evaluation on Official Test Dataset
All models were evaluated on the official test set (`Test/` directory, `test.txt`), comprising $N = 1,204$ images spanning all 200 species. In accordance with the assignment specifications, we report two formal metrics:

1. **Top-1 Accuracy:** Overall proportion of correct predictions across all testing samples:
$$\text{Top-1 Accuracy} = \frac{1}{N} \sum_{k=1}^N \mathbf{1}\{\text{argmax}(y) == \text{ground\_truth}\}$$

2. **Average Accuracy Per Class:** Unweighted arithmetic mean of individual class accuracies, evaluating balanced performance across rare and common classes:
$$\text{Ave} = \frac{1}{C} \sum_{i=1}^C T_i$$
where $T_i$ is the accuracy achieved across test samples belonging to class $C_i$, and $C = 200$.

#### Model Performance Comparison Table
| Metric | Baseline: ResNet-50 | Advanced: ConvNeXt-Tiny | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **80.65%** | **89.53%** | **+8.88%** |
| **Average Accuracy per Class** | **80.11%** | **89.36%** | **+9.25%** |
| Parameter Count (M) | 24.5M | 28.0M | +14.2% |
| Peak GPU VRAM (Inference) | 1.12 GB | 1.18 GB | +5.3% |
| Validation Loss (Final) | 1.542 | 1.318 | -14.5% |

---

### 2.2 Comparative Analysis & Justification of Chosen Architecture
The existing production baseline records an absolute improvement of **+8.88% in Top-1 Accuracy** ($89.53\%$ vs. $80.65\%$) and **+9.25% in Average Accuracy per Class** ($89.36\%$ vs. $80.11\%$) for ConvNeXt-Tiny over ResNet-50. The new four-way comparison is pending completion of all 12 isolated runs, so no frozen-versus-fine-tuned winner is claimed here.

Several architectural factors account for this marked performance disparity:
1. **Multi-Scale Spatial Context:** ConvNeXt's $7\times7$ depthwise convolutions effectively capture long-range spatial dependencies. For bird species classification, distinguishing a *Black-footed Albatross* from a *Laysan Albatross* requires establishing spatial relationships between bill curvature, facial masks, and torso coloration. ResNet-50's localized $3\times3$ convolutions fail to integrate these cues until the very final layers, diluting subtle spatial features.
2. **Channel Capacity & Feature Preservation:** ConvNeXt's inverted bottleneck preserves high-dimensional feature representations throughout the transformation pipeline, preventing fine-grained detail loss commonly encountered in ResNet's aggressive compression stages.
3. **Normalization Stability:** Layer Normalization applied across individual channel dimensions yields smoother optimization landscapes than Batch Normalization, resulting in faster and more monotonic loss convergence.

---

### 2.3 Fine-Grained Error Analysis & Confusion Pairs
The saved baseline ConvNeXt test evaluation identifies the following directed confusions. These counts are small (at most three errors per directed pair), so they indicate examples for qualitative review rather than broad conclusions about species families:

#### Top Misclassified Species Pairs:
1. **Common Tern (class 143) predicted as Arctic Tern (class 140):** 3 test errors in this direction.
2. **Chuck-will's-widow (class 21) predicted as Whip-poor-will (class 104):** 2 errors in this direction; the reverse direction also has 2 errors.
3. **Acadian Flycatcher (class 36) predicted as Least Flycatcher (class 38):** 2 errors in this direction; the reverse direction also has 2 errors.

The evaluation JSON contains predicted labels and aggregate counts, but it does not annotate image occlusion, pose, lighting, or plumage. Therefore, those visual explanations and the earlier estimate that 58% of errors involved occlusion are not claimed here; they would require a separately documented image-level review.

---

### 2.4 Overfitting Diagnosis & Mitigation Impact
The saved fine-tuned histories show that training accuracy continues to exceed validation accuracy. At the final recorded epoch, ResNet-50 reaches 99.42% training Top-1 versus 83.59% validation Top-1 (15.83 percentage points), while ConvNeXt-Tiny reaches 99.61% versus 88.41% (11.20 points). These histories document runs with the regularization configuration described above; they do not include a no-augmentation/no-smoothing ablation, so they cannot establish the causal effect of each regularizer or support an under-7% gap claim. The new frozen-versus-fine-tuned runs will add a regime comparison, not an ablation of individual regularization components.

---

## 3. Deployment & Application

### 3.1 Interactive Gradio Web Application
The current baseline checkpoints are packaged into an interactive **Gradio** web application (`app.py`). A new production checkpoint will only be promoted after the completed experiment selects it by validation average class accuracy and its official test evaluation is saved. Key capabilities include:
- **Real-Time Multi-Modal Input:** Accepts user-uploaded high-resolution bird images, clipboard pastes, and live webcam captures.
- **Top-5 Predicted Confidence Visualization:** Displays an interactive probability bar chart of the top 5 predicted bird species, mapped to clean, human-readable common names.
- **Instant Demonstration:** Pre-loaded test image examples enable one-click verification for evaluators without requiring external file preparation.
- **Hugging Face Spaces Deployment:** Packaged with `requirements.txt` and optimized for zero-GPU CPU inference or GPU acceleration on Hugging Face Spaces.

---

## 4. Conclusion
The current production baseline combines modern architectural design (**ConvNeXt-Tiny**) with strategic regularization (**Label Smoothing, Cosine Annealing, AdamW, and Random Resized Crops**) and achieves **89.53% Top-1 Accuracy** and **89.36% Average Class Accuracy**. The frozen-versus-fine-tuned experiment remains pending; its validation-selected checkpoint will be evaluated before any production promotion.
