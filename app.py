"""Streamlit interface for CUB-200 bird species classification."""

import glob
import json
import os
from typing import Dict, Optional, Tuple

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from models import build_model
from utils import load_class_names


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES_PATH = "data/class_names.json"
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Required class mapping is missing: {CLASS_NAMES_PATH}")
class_names = load_class_names(CLASS_NAMES_PATH)

MODEL_CONFIGS = {
    "ConvNeXt-Tiny": {
        "architecture": "convnext_tiny",
        "checkpoint": "checkpoints/best_convnext_tiny.pth",
        "metrics": "results/test_metrics_convnext_tiny.json",
        "curves": "results/curves_convnext_tiny.png",
        "histogram": "results/class_acc_hist_convnext_tiny.png",
    },
    "ResNet-50": {
        "architecture": "resnet50",
        "checkpoint": "checkpoints/best_resnet50.pth",
        "metrics": "results/test_metrics_resnet50.json",
        "curves": "results/curves_resnet50.png",
        "histogram": "results/class_acc_hist_resnet50.png",
    },
}
EXPERIMENT_SUMMARY_PATH = "experiments/results/multi_seed_summary.json"
EXPERIMENT_PLOT_PATH = "experiments/results/multi_seed_comparison.png"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model(model_label: str) -> torch.nn.Module:
    """Load the selected trained checkpoint."""
    config = MODEL_CONFIGS[model_label]
    checkpoint_path = config["checkpoint"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Required trained checkpoint is missing: {checkpoint_path}"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model = build_model(config["architecture"], num_classes=200, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError, OSError) as error:
        raise RuntimeError(f"Could not load trained checkpoint {checkpoint_path}: {error}") from error
    model.to(DEVICE).eval()
    return model


def predict_bird(
    input_image: Image.Image, model: Optional[torch.nn.Module] = None
) -> Tuple[Dict[str, float], int, float]:
    """Return the top-five species, top class index, and top confidence."""
    if model is None:
        model = load_model("ConvNeXt-Tiny")
    image_tensor = EVAL_TRANSFORM(input_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0].cpu().numpy()

    top_indices = probabilities.argsort()[-5:][::-1]
    results = {
        class_names.get(int(index), f"Species {index}"): float(probabilities[index])
        for index in top_indices
    }
    return results, int(top_indices[0]), float(probabilities[top_indices[0]])


def render_home() -> None:
    """Render the bird classifier page."""
    st.title("🦜 CUB-200 Bird Species Classifier")
    st.write("Upload a bird photograph to see the five most likely species.")

    model_label = st.selectbox(
        "Choose a model",
        list(MODEL_CONFIGS),
        help="Scroll through the list to select the model used for classification.",
    )
    model = load_model(model_label)
    st.caption(f"Model: {model_label} | Device: {DEVICE.type}")

    sample_paths = sorted(
        glob.glob("Test/*.jpg") + glob.glob("Test/*.jpeg") + glob.glob("Test/*.png")
    )
    sample_choice = st.selectbox("Or choose a test image", ["None"] + sample_paths)
    uploaded_file = st.file_uploader(
        "Upload a bird photograph", type=["jpg", "jpeg", "png"]
    )

    input_image: Optional[Image.Image] = (
        Image.open(uploaded_file) if uploaded_file is not None else None
    )
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
            except (RuntimeError, ValueError) as error:
                st.error(f"Prediction failed: {error}")
    else:
        st.warning("Upload an image or choose a test image to begin.")


def load_metrics(model_label: str) -> Optional[dict]:
    """Load saved evaluation results for a model when available."""
    metrics_path = MODEL_CONFIGS[model_label]["metrics"]
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read {metrics_path}: {error}")
        return None


def render_model_page() -> None:
    """Render saved performance images and evaluation results for each model."""
    st.title("📊 Model Performance")
    st.write("Compare the trained models using results from the official CUB-200 test set.")

    model_label = st.selectbox("Choose a model to inspect", list(MODEL_CONFIGS))
    config = MODEL_CONFIGS[model_label]
    metrics = load_metrics(model_label)

    if metrics is None:
        st.warning("No evaluation result is available for this model yet.")
        return

    metric_columns = st.columns(3)
    metric_columns[0].metric("Top-1 accuracy", f"{metrics['top1_accuracy_pct']:.2f}%")
    metric_columns[1].metric(
        "Average class accuracy",
        f"{metrics['average_accuracy_per_class_pct']:.2f}%",
    )
    metric_columns[2].metric("Test images", metrics["total_test_samples"])

    st.subheader(f"{model_label} training performance")
    if os.path.exists(config["curves"]):
        st.image(config["curves"], caption=f"{model_label} loss and accuracy curves")
    else:
        st.info("Training curve image is not available.")

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.subheader("Per-class accuracy distribution")
        if os.path.exists(config["histogram"]):
            st.image(config["histogram"], caption=f"{model_label} class accuracy histogram")
        else:
            st.info("Class accuracy histogram is not available.")

    with chart_columns[1]:
        st.subheader("Most confused species")
        confused_pairs = metrics.get("top_confused_pairs", [])
        if confused_pairs:
            st.dataframe(
                [
                    {
                        "True species": pair["true_name"],
                        "Predicted species": pair["pred_name"],
                        "Errors": pair["error_count"],
                    }
                    for pair in confused_pairs
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Confusion results are not available.")

    st.subheader("Best and hardest classes")
    best, hardest = st.columns(2)
    with best:
        st.write("Highest accuracy")
        st.dataframe(metrics.get("highest_5_classes", []), hide_index=True, use_container_width=True)
    with hardest:
        st.write("Lowest accuracy")
        st.dataframe(metrics.get("lowest_5_classes", []), hide_index=True, use_container_width=True)


def render_experiment_page() -> None:
    """Render aggregate results from the isolated multi-seed experiment."""
    st.title("🧪 Multi-seed Experiment")
    st.write(
        "Independent full-configuration runs measure run-to-run variation; "
        "they do not isolate individual training techniques."
    )
    if not os.path.exists(EXPERIMENT_SUMMARY_PATH):
        st.info(
            "Experiment results are not available yet. Run "
            "`python experiment_runner.py` to create the six runs."
        )
        return

    try:
        with open(EXPERIMENT_SUMMARY_PATH, "r", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read experiment results: {error}")
        return

    for model, model_results in summary.get("models", {}).items():
        st.subheader(model.replace("_", " ").title())
        metric_columns = st.columns(2)
        top1 = model_results["top1_accuracy_pct"]
        average_class = model_results["average_accuracy_per_class_pct"]
        metric_columns[0].metric(
            "Top-1 accuracy",
            f"{top1['mean']:.2f}% ± {top1['std']:.2f}%",
        )
        metric_columns[1].metric(
            "Average class accuracy",
            f"{average_class['mean']:.2f}% ± {average_class['std']:.2f}%",
        )
        st.dataframe(
            [
                {
                    "Seed": run["seed"],
                    "Top-1 accuracy": f"{run['top1_accuracy_pct']:.2f}%",
                    "Average class accuracy": f"{run['average_accuracy_per_class_pct']:.2f}%",
                }
                for run in model_results.get("runs", [])
            ],
            hide_index=True,
            use_container_width=True,
        )

    if os.path.exists(EXPERIMENT_PLOT_PATH):
        st.image(
            EXPERIMENT_PLOT_PATH,
            caption="Mean accuracy with standard deviation and individual seed scores",
            use_container_width=True,
        )
    else:
        st.info("The experiment comparison image is not available yet.")


st.set_page_config(page_title="CUB-200 Bird Classifier", page_icon="🦜", layout="wide")
with st.sidebar:
    st.markdown("### 🦜 Bird Classifier")
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"
    if st.button("Model", use_container_width=True):
        st.session_state.page = "Model"
    if st.button("Experiment", use_container_width=True):
        st.session_state.page = "Experiment"

    page = st.session_state.page

if page == "Home":
    render_home()
elif page == "Model":
    render_model_page()
else:
    render_experiment_page()
