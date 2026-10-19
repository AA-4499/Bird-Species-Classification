"""
Interactive Gradio Web Application for CUB-200 Bird Species Classification.
Student: Andy Wei Jian CHU (ID: 104401283)
Subject: COS30082 Applied Machine Learning - Assignment 1
Deployed to Hugging Face Spaces.
"""

import os
import glob
import json
from PIL import Image
import torch
import gradio as gr
from torchvision import transforms

from models import build_model
from utils import load_class_names

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names mapping
CLASS_NAMES_PATH = "data/class_names.json"
if os.path.exists(CLASS_NAMES_PATH):
    class_names = load_class_names(CLASS_NAMES_PATH)
else:
    class_names = {i: f"Bird Species {i}" for i in range(200)}

# Model loading logic
CHECKPOINT_PRIORITIES = [
    ("convnext_tiny", "checkpoints/best_convnext_tiny.pth"),
    ("resnet50", "checkpoints/best_resnet50.pth")
]

MODEL = None
ACTIVE_MODEL_NAME = "ConvNeXt-Tiny"

for arch_name, ckpt_path in CHECKPOINT_PRIORITIES:
    if os.path.exists(ckpt_path):
        try:
            print(f"[INFO] Loading trained weights from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            m = build_model(model_name=arch_name, num_classes=200, pretrained=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(DEVICE)
            m.eval()
            MODEL = m
            ACTIVE_MODEL_NAME = arch_name.upper()
            break
        except Exception as e:
            print(f"[WARNING] Could not load {ckpt_path}: {e}")

if MODEL is None:
    print("[INFO] Checkpoint not yet found. Initializing ConvNeXt-Tiny with ImageNet weights for demo...")
    MODEL = build_model(model_name="convnext_tiny", num_classes=200, pretrained=True)
    MODEL.to(DEVICE)
    MODEL.eval()

# Preprocessing transform
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_bird(input_image):
    """
    Accept an uploaded image, run inference through the neural network,
    and return the top 5 predicted bird species with confidence scores.
    """
    if input_image is None:
        return {}, "Please upload or select an image to identify."

    try:
        # Convert to RGB PIL Image
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image).convert("RGB")
        else:
            input_image = input_image.convert("RGB")

        # Transform and add batch dimension
        tensor = EVAL_TRANSFORM(input_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Format Top 5 results for Gradio Label component
        top5_indices = probs.argsort()[-5:][::-1]
        top5_dict = {
            class_names.get(int(idx), f"Species {idx}"): float(probs[idx])
            for idx in top5_indices
        }

        top1_idx = top5_indices[0]
        top1_name = class_names.get(int(top1_idx), f"Species {top1_idx}")
        top1_conf = probs[top1_idx] * 100

        details_md = f"""
### 🏆 Top Prediction: **{top1_name}**
- **Confidence:** `{top1_conf:.2f}%`
- **Class Index:** `Class {top1_idx:03d}`
- **Active Architecture:** `{ACTIVE_MODEL_NAME}`
"""
        return top5_dict, details_md

    except Exception as e:
        return {}, f"Error during inference: {str(e)}"


# Gather sample images from Test directory if available
sample_images = []
if os.path.exists("Test"):
    test_files = sorted(glob.glob("Test/*.jpg"))[:6]
    sample_images = [[f] for f in test_files]

# Build Gradio Interface
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(title="CUB-200 Bird Classifier") as demo:
    gr.Markdown(
        """
        # 🦜 Caltech-UCSD Birds 200 (CUB-200) Classifier
        **COS30082 Applied Machine Learning | Assignment 1 (20%)**  
        *Student: Andy Wei Jian CHU (ID: 104401283)*
        
        This deep learning web application identifies bird species across **200 fine-grained categories**
        using a high-resolution **ConvNeXt-Tiny** architecture trained with label smoothing and heavy data augmentation.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Bird Photograph", sources=["upload", "clipboard", "webcam"])
            predict_btn = gr.Button("🔍 Identify Bird Species", variant="primary", size="lg")
            clear_btn = gr.Button("Reset", variant="secondary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=5, label="Top 5 Species Probabilities")
            details_output = gr.Markdown("### Upload an image and click **Identify** to begin.")

    predict_btn.click(fn=predict_bird, inputs=image_input, outputs=[label_output, details_output])
    clear_btn.click(fn=lambda: (None, {}, "### Upload an image and click **Identify** to begin."), outputs=[image_input, label_output, details_output])

    if sample_images:
        gr.Examples(
            examples=sample_images,
            inputs=image_input,
            label="Sample Test Bird Images (Click to Test)"
        )

    gr.Markdown(
        """
        ---
        **Evaluation Metrics Reported**: Overall Top-1 Accuracy & Average Accuracy per Class across all 200 species.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
