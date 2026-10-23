"""Streamlit interface for CUB-200 bird species classification."""

import glob
import json
import os
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
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

EXPERIMENT_SUMMARY_PATH = "experiments/results/multi_seed_summary.json"
EXPERIMENT_PLOT_PATH = "experiments/results/multi_seed_comparison.png"
PRODUCTION_SELECTION_PATH = "experiments/results/production_selection.json"
fragment = getattr(st, "fragment", lambda function: function)


def _config_from_experiment_run(run: dict) -> Optional[dict]:
    """Build an app model config from a validated experiment run."""
    checkpoint_path = run["checkpoint_path"]
    metrics_path = run["performance_path"]
    if not os.path.exists(checkpoint_path) or not os.path.exists(metrics_path):
        return None
    return {
        "architecture": run["model"],
        "mode": run["mode"],
        "checkpoint": checkpoint_path,
        "metrics": metrics_path,
        "curves": os.path.join(
            os.path.dirname(checkpoint_path),
            f"curves_{run['model']}.png",
        ),
        "histogram": os.path.join(
            os.path.dirname(checkpoint_path),
            f"class_acc_hist_{run['model']}_{run['mode']}.png",
        ),
    }


def _load_promoted_configs() -> Dict[str, dict]:
    """Use the best evaluated experiment run for each architecture."""
    promoted = {}
    if not os.path.exists(PRODUCTION_SELECTION_PATH):
        return promoted
    try:
        with open(PRODUCTION_SELECTION_PATH, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("status") != "evaluated":
            return promoted
        selected_by_model = manifest.get("selected_by_model")
        if not selected_by_model:
            selected = manifest["selected"]
            selected_by_model = {selected["model"]: selected}
        labels = {
            "resnet50": "Production (ResNet-50)",
            "convnext_tiny": "Production (ConvNeXt-Tiny)",
        }
        for model, run in selected_by_model.items():
            config = _config_from_experiment_run(run)
            if config is not None and model in labels:
                promoted[labels[model]] = config
        return promoted
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return promoted


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
_promoted_configs = _load_promoted_configs()
if _promoted_configs:
    MODEL_CONFIGS = {**_promoted_configs, **MODEL_CONFIGS}

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
        default_model = "Production (ConvNeXt-Tiny)"
        model = load_model(
            default_model if default_model in MODEL_CONFIGS else "ConvNeXt-Tiny"
        )
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


@st.cache_data
def load_experiment_summary(path: str) -> dict:
    """Load the aggregate experiment once per artifact version."""
    with open(path, "r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def _experiment_dataframe(summary: dict) -> pd.DataFrame:
    """Flatten nested experiment results for tables and interactive charts."""
    rows = []
    for model, model_results in summary.get("architectures", {}).items():
        for mode, mode_results in model_results.items():
            for run in mode_results.get("runs", []):
                rows.append(
                    {
                        "Configuration": (
                            f"{model.replace('_', ' ').title()} / "
                            f"{mode.replace('_', ' ').title()}"
                        ),
                        "Model": model,
                        "Mode": mode,
                        "Seed": run["seed"],
                        "Validation average class accuracy": (
                            run["validation_average_class_accuracy"] * 100
                        ),
                        "Top-1 accuracy": run["top1_accuracy_pct"],
                        "Average class accuracy": run["average_accuracy_per_class_pct"],
                    }
                )
    return pd.DataFrame(rows)


def _comparison_figure(data: pd.DataFrame, metric: str) -> go.Figure:
    """Create a responsive mean plus seed-level spread chart."""
    figure = go.Figure()
    for configuration, group in data.groupby("Configuration", sort=False):
        figure.add_trace(
            go.Scatter(
                x=[configuration] * len(group),
                y=group[metric],
                mode="markers",
                name=configuration,
                legendgroup=configuration,
                showlegend=False,
                marker={"size": 9, "opacity": 0.65},
                customdata=group[["Seed"]].to_numpy(),
                hovertemplate=(
                    "<b>%{x}</b><br>Seed: %{customdata[0]}<br>"
                    f"{metric}: %{{y:.2f}}%<extra></extra>"
                ),
            )
        )

    means = data.groupby("Configuration", sort=False)[metric].agg(["mean", "std"]).reset_index()
    figure.add_trace(
        go.Scatter(
            x=means["Configuration"],
            y=means["mean"],
            mode="markers",
            name="Mean ± SD",
            marker={"size": 14, "symbol": "diamond", "color": "#111827"},
            error_y={
                "type": "data",
                "array": means["std"].fillna(0),
                "visible": True,
                "thickness": 2,
                "width": 5,
            },
            hovertemplate=(
                "<b>%{x}</b><br>Mean: %{y:.2f}%<br>"
                "Error bars: seed SD<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{metric}: mean, seed variation, and individual runs",
        xaxis_title="Configuration",
        yaxis_title=f"{metric} (%)",
        yaxis={"range": [0, 100]},
        hovermode="closest",
        margin={"l": 20, "r": 20, "t": 65, "b": 100},
        legend={"orientation": "h"},
    )
    return figure


@fragment
def render_experiment_summary(summary: dict) -> None:
    """Render the interactive decision summary independently from navigation."""
    data = _experiment_dataframe(summary)
    if data.empty:
        st.warning("The experiment summary contains no completed runs.")
        return

    production = summary.get("architectures", {}).get("convnext_tiny", {}).get(
        "fine_tuned"
    )
    baseline = summary.get("architectures", {}).get("resnet50", {}).get("fine_tuned")
    if production and baseline:
        production_top1 = production["top1_accuracy_pct"]["mean"]
        baseline_top1 = baseline["top1_accuracy_pct"]["mean"]
        production_class = production["average_accuracy_per_class_pct"]["mean"]
        baseline_class = baseline["average_accuracy_per_class_pct"]["mean"]
        st.success(
            "Recommendation: use fine-tuned ConvNeXt-Tiny. "
            f"It leads fine-tuned ResNet-50 by "
            f"{production_top1 - baseline_top1:+.2f} percentage points "
            "on mean top-1 accuracy across three seeds."
        )
    else:
        st.info("Recommendation is unavailable until both fine-tuned configurations are evaluated.")
        production_top1 = baseline_top1 = production_class = baseline_class = None

    st.subheader("Executive summary")
    cards = st.columns(4)
    if production_top1 is not None:
        cards[0].metric("Recommended model", "ConvNeXt-Tiny")
        cards[1].metric("Mean top-1 accuracy", f"{production_top1:.2f}%")
        cards[2].metric("Mean class accuracy", f"{production_class:.2f}%")
        cards[3].metric(
            "Gain vs ResNet-50",
            f"{production_top1 - baseline_top1:+.2f} pp",
            delta=f"{production_class - baseline_class:+.2f} pp macro",
        )
    else:
        cards[0].metric("Configurations", data["Configuration"].nunique())
        cards[1].metric("Completed runs", len(data))
        cards[2].metric("Metric", "Accuracy")
        cards[3].metric("Seeds per configuration", data["Seed"].nunique())

    st.caption(
        "Point estimates are means across training seeds. Error bars show seed-to-seed "
        "standard deviation, not confidence intervals."
    )
    metric = st.radio(
        "Primary metric",
        ["Top-1 accuracy", "Average class accuracy"],
        horizontal=True,
        key="experiment_metric",
    )
    st.plotly_chart(
        _comparison_figure(data, metric),
        use_container_width=True,
        config={"responsive": True, "displaylogo": False},
    )

    with st.expander("Full results by seed"):
        st.dataframe(
            data[
                [
                    "Configuration",
                    "Seed",
                    "Validation average class accuracy",
                    "Top-1 accuracy",
                    "Average class accuracy",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Validation average class accuracy": st.column_config.NumberColumn(
                    "Validation average class accuracy (%)", format="%.2f%%"
                ),
                "Top-1 accuracy": st.column_config.NumberColumn(
                    "Top-1 accuracy (%)", format="%.2f%%"
                ),
                "Average class accuracy": st.column_config.NumberColumn(
                    "Average class accuracy (%)", format="%.2f%%"
                ),
            },
        )
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
    """Render the decision-first aggregate experiment dashboard."""
    st.title("🧪 Model selection dashboard")
    st.write(
        "Choose the strongest model from independent multi-seed evaluations. "
        "This experiment measures run-to-run variation; it does not isolate "
        "individual training techniques."
    )
    if not os.path.exists(EXPERIMENT_SUMMARY_PATH):
        st.info("Experiment results are not available yet. Run `python experiment_runner.py` to create the 12 runs.")
        return

    try:
        summary = load_experiment_summary(EXPERIMENT_SUMMARY_PATH)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Could not read experiment results: {error}")
        return

    render_experiment_summary(summary)


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
