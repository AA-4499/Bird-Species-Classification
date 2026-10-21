"""Streamlit interface for CUB-200 bird species classification."""

import glob
import os

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models import build_model
from utils import load_class_names


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES_PATH = "data/class_names.json"
class_names = load_class_names(CLASS_NAMES_PATH) if os.path.exists(CLASS_NAMES_PATH) else {
    i: f"Bird Species {i}" for i in range(200)
}

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@st.cache_resource
def load_model():
    for arch_name, checkpoint_path in [
        ("convnext_tiny", "checkpoints/best_convnext_tiny.pth"),
        ("resnet50", "checkpoints/best_resnet50.pth"),
    ]:
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                model = build_model(arch_name, num_classes=200, pretrained=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(DEVICE).eval()
                return model, arch_name.upper(), True
            except Exception as error:
                st.warning(f"Could not load {checkpoint_path}: {error}")

    model = build_model("convnext_tiny", num_classes=200, pretrained=True)
    model.to(DEVICE).eval()
    return model, "CONVNEXT_TINY (IMAGENET DEMO)", False


def predict_bird(input_image, model):
    image_tensor = EVAL_TRANSFORM(input_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0].cpu().numpy()

    top_indices = probabilities.argsort()[-5:][::-1]
    results = {
        class_names.get(int(index), f"Species {index}"): float(probabilities[index])
        for index in top_indices
    }
    return results, int(top_indices[0]), float(probabilities[top_indices[0]])


st.set_page_config(page_title="CUB-200 Bird Classifier", page_icon="🦜", layout="wide")
st.title("🦜 CUB-200 Bird Species Classifier")
st.write("Upload a bird photograph to see the five most likely species.")

model, model_name, has_checkpoint = load_model()
st.caption(f"Model: {model_name} | Device: {DEVICE.type}")
if not has_checkpoint:
    st.info("No trained checkpoint was found. The ImageNet demo model is loaded; add checkpoints/best_convnext_tiny.pth for trained predictions.")

sample_paths = sorted(glob.glob("Test/*.jpg") + glob.glob("Test/*.jpeg") + glob.glob("Test/*.png"))
sample_choice = st.selectbox("Or choose a test image", ["None"] + sample_paths)
uploaded_file = st.file_uploader("Upload a bird photograph", type=["jpg", "jpeg", "png"])

input_image = Image.open(uploaded_file) if uploaded_file is not None else None
if input_image is None and sample_choice != "None":
    input_image = Image.open(sample_choice)

if input_image is not None:
    st.image(input_image, caption="Selected image", use_container_width=True)
    if st.button("Identify bird species", type="primary"):
        try:
            results, top_index, confidence = predict_bird(input_image, model)
            left, right = st.columns(2)
            with left:
                st.subheader("Top prediction")
                st.metric("Species", class_names.get(top_index, f"Species {top_index}"))
                st.metric("Confidence", f"{confidence * 100:.2f}%")
            with right:
                st.subheader("Top 5 probabilities")
                st.bar_chart(results)
        except Exception as error:
            st.error(f"Prediction failed: {error}")
else:
    st.warning("Upload an image or choose a test image to begin.")
