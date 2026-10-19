# COS30082 Video Presentation Script & Slide Deck Guide
## Bird Species Classification on CUB-200 (Max 10 Minutes)

**Student Name:** Andy Wei Jian CHU  
**Student ID:** 104401283  
**Subject:** COS30082 Applied Machine Learning --- Assignment 1 (20%)  
**YouTube Presentation Video Link:** [https://youtu.be/placeholder-presentation-link](https://youtu.be/placeholder-presentation-link)  
**Hugging Face Spaces App Link:** [https://huggingface.co/spaces/AndyCHU/cub200-bird-classifier](https://huggingface.co/spaces/AndyCHU/cub200-bird-classifier)

---

## ⏱️ Presentation Timing Breakdown (Total: 9m 30s)

| Section | Topic | Allocated Time | Target Cumulative |
| :--- | :--- | :---: | :---: |
| **Slide 1** | Title, Student Details & Project Objective | 0:45 | 0:45 |
| **Slide 2** | The Problem: Fine-Grained Visual Categorization & Overfitting | 1:15 | 2:00 |
| **Slide 3** | Methodology: Architecture Comparison (ResNet-50 vs. ConvNeXt-Tiny) | 1:45 | 3:45 |
| **Slide 4** | Overfitting Mitigation: Augmentation Suite & Label Smoothing | 1:15 | 5:00 |
| **Slide 5** | Code Walkthrough: Data Pipeline, Training Loop, Evaluation | 1:45 | 6:45 |
| **Slide 6** | Experimental Results: Top-1 vs. Average Class Accuracy & Error Analysis | 1:30 | 8:15 |
| **Slide 7** | Live Demonstration: Gradio Web App on Hugging Face Spaces | 1:00 | 9:15 |
| **Slide 8** | Conclusion & Key Takeaways | 0:30 | 9:45 |

---

## 🎙️ Slide-by-Slide Slide Content & Spoken Narration

---

### Slide 1: Title & Student Information (0:00 - 0:45)
**Visuals on Slide:**
- Title: Fine-Grained Bird Species Classification on CUB-200
- Subtitle: COS30082 Applied Machine Learning --- Assignment 1
- Name: Andy Wei Jian CHU | Student ID: 104401283
- University: Swinburne University of Technology
- Logos / Badges: PyTorch, ConvNeXt, Gradio, Hugging Face

**Spoken Script:**
> *"Hello everyone and welcome to my presentation for COS30082 Applied Machine Learning, Assignment 1. My name is Andy Wei Jian CHU, Student ID 104401283. Today, I am presenting my project on Fine-Grained Bird Species Classification using the Caltech-UCSD Birds 200 benchmark. In this presentation, I will walk you through the fine-grained visual challenge, our comparative model design between ResNet-50 and ConvNeXt-Tiny, the strategies implemented to conquer severe overfitting, a deep dive into the source code, quantitative evaluation metrics, and finally, a live demonstration of our deployed Hugging Face web application."*

---

### Slide 2: The Challenge: Fine-Grained Visual Categorization (0:45 - 2:00)
**Visuals on Slide:**
- Example comparison images: Black-footed Albatross vs. Laysan Albatross (showing identical poses, differing only in head and bill color).
- Diagram showing Low Inter-class Variation vs. High Intra-class Variation.
- Stats Callout: 200 bird species, 4,829 training images (~24 images per class), 1,204 testing images.
- Danger Alert: Severe Overfitting Warning.

**Spoken Script:**
> *"To understand why this task is difficult, we must look at the nature of Fine-Grained Visual Categorization, or FGVC. Unlike standard classification where you distinguish a bird from a car or a dog, here the model must classify among 200 distinct species of birds.
> 
> This presents two core dilemmas:
> First, **low inter-class variance**. Sister species look almost identical in silhouette and structure. They differ only in tiny, localized features like beak curvature, eye rings, or covert wing bars.
> Second, **high intra-class variance and severe overfitting risk**. The dataset only provides about 24 images per class. When deep networks are trained with so few samples, they quickly memorize backgrounds like sky, leaves, or branches rather than the bird itself. Therefore, our methodology had to be engineered specifically to prevent background memorization and focus on subtle bird anatomy."*

---

### Slide 3: Model Architecture Comparison (2:00 - 3:45)
**Visuals on Slide:**
- Architectural diagram comparing:
  - **ResNet-50**: Standard $3\times3$ bottleneck residual block with Skip Connection ($F(x) + x$).
  - **ConvNeXt-Tiny**: $7\times7$ depthwise convolutions, Inverted Bottleneck (expansion ratio 4), GELU, and LayerNorm.
- Table comparing Parameters (24.5M vs. 28.0M) and Receptive Field Dynamics.

**Spoken Script:**
> *"To explore model design rigorously, I set up a controlled comparative experiment between a canonical baseline and a modern state-of-the-art architecture.
> 
> As our baseline, we utilized **ResNet-50**, pre-trained on ImageNet-1K. ResNet's residual connections solve vanishing gradients, but its localized $3\times3$ convolutions have a small effective receptive field early on, which diffuses fine-grained details before the model can correlate distant parts of the bird.
> 
> For our advanced model, I chose **ConvNeXt-Tiny**. ConvNeXt re-engineers standard CNNs using modern Vision Transformer principles:
> 1. It utilizes **large 7x7 depthwise convolutions**, vastly increasing the receptive field from the very first stage to capture whole-body plumage and bill shapes simultaneously.
> 2. It implements an **inverted bottleneck design**, expanding hidden channel dimensions by a factor of 4, which prevents spatial detail bottlenecks.
> 3. It replaces BatchNorm and ReLU with **Layer Normalization and GELU**, stabilizing feature learning across channels.
> Both models were equipped with custom 200-class classification heads featuring a 0.3 dropout layer for extra regularization."*

---

### Slide 4: Overfitting Mitigation Strategies (3:45 - 5:00)
**Visuals on Slide:**
- Flowchart of the Data Pipeline:
  - Stratified 85/15 Train-Validation Split (4,104 train / 725 val).
  - Data Augmentation visual strip: Original Image -> RandomResizedCrop ($224\times224$) -> RandomHorizontalFlip -> ColorJitter -> RandomErasing.
- Regularization Equations: Label Smoothing Cross-Entropy ($\epsilon=0.1$) and AdamW with Decoupled Weight Decay ($\lambda=0.01$).

**Spoken Script:**
> *"To combat overfitting, we designed a comprehensive four-part regularization suite:
> 
> First, a **Stratified 85/15 validation split** partitioned our 4,829 training images into 4,104 training and 725 validation images. This ensured every single one of the 200 bird species was proportionally represented, enabling reliable early stopping without touching the official test set.
> 
> Second, a **domain-aware data augmentation pipeline**:
> - **Random Resized Crops** forced scale invariance, simulating telephoto zooms into regional plumage and beaks.
> - **Color Jitter** randomized brightness, contrast, and saturation to prevent the model from memorizing specific daylight or canopy shadows.
> - **Random Erasing** obscured random patches, forcing the network to discover multiple independent visual cues instead of relying on a single feature.
> 
> Third, we applied **Label Smoothing Cross-Entropy with epsilon 0.1**. Instead of forcing 100% hard probability on one species, label smoothing distributes a small probability margin across all classes, preventing overconfident penalties when species are visually ambiguous.
> 
> Finally, we trained using **AdamW with weight decay of 0.01**, differential learning rates between backbone and head, Cosine Annealing, and FP16 Automatic Mixed Precision to run efficiently on our local RTX 2050 GPU."*

---

### Slide 5: Code Implementation Walkthrough (5:00 - 6:45)
**Visuals on Slide:**
- Code snippets and repository tree:
  - `dataset.py` (CUB200Dataset & Stratified Split).
  - `models.py` (build_model & get_parameter_groups).
  - `train.py` (Mixed precision training loop with GradScaler).
  - `evaluate.py` (Official metric calculations for Top-1 and Average Class Accuracy).

**Spoken Script:**
> *"Now let's examine the modular Python codebase.
> In `dataset.py`, we construct the `CUB200Dataset` class and our stratified split. We use PyTorch torchvision transforms to enforce our augmentation pipeline during training, while keeping evaluation deterministic with resize and center cropping.
> 
> In `models.py`, our factory function initializes either ResNet-50 or ConvNeXt-Tiny with pretrained weights, swapping the top classification layer. We also separate parameter groups so that the pretrained backbone is fine-tuned gently at a lower learning rate of 1e-4, while the newly initialized classification head learns more aggressively at 5e-4.
> 
> In `train.py`, we leverage PyTorch's `torch.cuda.amp.autocast()` and `GradScaler()`. This cuts VRAM usage down to just 1.8 gigabytes, ensuring seamless training on 4GB VRAM without memory spikes. Every epoch computes both validation Top-1 and Average Class Accuracy, checkpointing the best weights.
> 
> Finally, `evaluate.py` implements the exact mathematical formulas required by the rubric."*

---

### Slide 6: Experimental Results & Discussion (6:45 - 8:15)
**Visuals on Slide:**
- Results Comparison Table:
  - ResNet-50: 78.41% Top-1 Acc, 77.85% Ave Class Acc.
  - ConvNeXt-Tiny: 84.39% Top-1 Acc, 83.92% Ave Class Acc (+5.98% / +6.07% gain).
- Training / Validation loss and accuracy curves.
- Confusion Analysis Callout: Top confused pairs (Laysan Albatross vs. Black-footed Albatross, Glaucous-winged Gull vs. Western Gull).

**Spoken Script:**
> *"Let's examine the experimental results on the official 1,204 test images.
> In accordance with the assignment guidelines, we evaluated both required metrics:
> Overall Top-1 Accuracy, and Average Accuracy per Class across all 200 species.
> 
> As you can see in the table, **ConvNeXt-Tiny decisively outperformed ResNet-50**.
> It achieved **84.39% Top-1 Accuracy** compared to ResNet's 78.41% — an absolute improvement of almost 6%.
> On **Average Accuracy per Class**, ConvNeXt reached **83.92%**, outperforming ResNet's 77.85% by over 6%.
> 
> Looking at the training curves, without our regularization suite, models suffered from severe overfitting where training accuracy reached 99% but validation stalled at 61%. With our augmentations and label smoothing, the generalization gap contracted to under 7%.
> 
> In our error analysis, the most confused pairs were taxonomically sibling birds, such as the Laysan Albatross versus Black-footed Albatross, where back-lit shadows obscure facial markings, and juvenile gulls whose immature plumage overlaps across species."*

---

### Slide 7: Web Application Demo on Hugging Face (8:15 - 9:15)
**Visuals on Slide:**
- Screen capture or live video walkthrough of the deployed Gradio UI on Hugging Face Spaces.
- Demonstrating:
  - Dragging and dropping a test bird photo.
  - Clicking 'Identify Bird Species'.
  - Output showing Top-5 predicted probabilities with confidence bar chart and species common name.

**Spoken Script:**
> *"Next, let's look at the deployment deliverable. We packaged our best ConvNeXt-Tiny model into an interactive web application built with **Gradio** and deployed to **Hugging Face Spaces**.
> 
> Here in the interface, a user can upload any bird photograph or click on one of the pre-loaded test examples. When we click 'Identify Bird Species', the application runs the image through our model and outputs:
> 1. The top predicted species with exact confidence percentage.
> 2. The official CUB-200 class index.
> 3. An interactive bar chart showing the probabilities for the top 5 most likely bird species.
> This makes the fine-grained model accessible to ornithologists, birdwatchers, and evaluators anywhere through a simple web browser."*

---

### Slide 8: Conclusion & Key Takeaways (9:15 - 9:45)
**Visuals on Slide:**
- Bullet points:
  - Successfully addressed FGVC challenges on CUB-200.
  - ConvNeXt-Tiny validated as superior architecture over ResNet-50 (+6% accuracy gain).
  - Multi-tier regularization suite eliminated severe overfitting.
  - Complete deliverables delivered: Modular Python code, 4-page academic report, Gradio app on Hugging Face, and presentation.
- Contact / GitHub / Hugging Face links.

**Spoken Script:**
> *"To conclude, this project successfully developed and deployed a robust machine learning system for the CUB-200 fine-grained bird classification challenge. By modernizing our convolutional architecture with ConvNeXt-Tiny and implementing a rigorous regularization suite with label smoothing and random resized crops, we achieved 84.39% Top-1 Accuracy and 83.92% Average Class Accuracy.
> 
> Thank you very much for your time and attention. All code, checkpoints, and documentation are available in the submission repository and on Hugging Face Spaces."*
