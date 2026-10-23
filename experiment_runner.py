"""Run and aggregate the isolated multi-seed model experiment."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


MODELS = ("resnet50", "convnext_tiny")
MODES = ("frozen", "fine_tuned")
SEEDS = (42, 43, 44)
EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_DROP_RATE = 0.3
DEFAULT_VAL_RATIO = 0.15


def experiment_schedule(
    models: Sequence[str] = MODELS,
    seeds: Sequence[int] = SEEDS,
    epochs: int = EPOCHS,
    modes: Sequence[str] = MODES,
) -> List[Dict[str, object]]:
    """Return the complete, deterministic model/seed schedule."""
    if epochs != EPOCHS:
        raise ValueError(f"Multi-seed runs must target exactly {EPOCHS} epochs.")
    invalid_models = set(models) - set(MODELS)
    if invalid_models:
        raise ValueError(f"Unsupported experiment models: {sorted(invalid_models)}")
    invalid_modes = set(modes) - set(MODES)
    if invalid_modes:
        raise ValueError(f"Unsupported training modes: {sorted(invalid_modes)}")
    return [
        {"model": model, "mode": mode, "seed": seed, "epochs": epochs}
        for model in models
        for mode in modes
        for seed in seeds
    ]


def run_directory(root: Path, model: str, mode: str, seed: int) -> Path:
    """Return the unique artifact directory for one model/seed run."""
    return root / "runs" / model / mode / f"seed_{seed}"


def _run_command(command: List[str]) -> None:
    print("[INFO]", " ".join(command))
    subprocess.run(command, check=True)


def run_experiment(
    root: Path = Path("experiments"),
    models: Sequence[str] = MODELS,
    seeds: Sequence[int] = SEEDS,
    epochs: int = EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    modes: Sequence[str] = MODES,
) -> None:
    """Train and evaluate every scheduled run using isolated artifact paths."""
    schedule = experiment_schedule(models=models, seeds=seeds, epochs=epochs, modes=modes)
    root.mkdir(parents=True, exist_ok=True)
    for item in schedule:
        model = str(item["model"])
        mode = str(item["mode"])
        seed = int(item["seed"])
        run_dir = run_directory(root, model, mode, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        train_command = [
            sys.executable,
            "train.py",
            "--model",
            model,
            "--mode",
            mode,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
            "--lr",
            str(lr),
            "--weight_decay",
            str(DEFAULT_WEIGHT_DECAY),
            "--label_smoothing",
            str(DEFAULT_LABEL_SMOOTHING),
            "--drop_rate",
            str(DEFAULT_DROP_RATE),
            "--val_ratio",
            str(DEFAULT_VAL_RATIO),
            "--seed",
            str(seed),
            "--save_dir",
            str(run_dir),
            "--results_dir",
            str(run_dir),
        ]
        _run_command(train_command)
        checkpoint_path = run_dir / f"best_{model}.pth"
        _run_command(
            [
                sys.executable,
                "evaluate.py",
                "--checkpoint",
                str(checkpoint_path),
                "--output_dir",
                str(run_dir),
                "--result_name",
                f"{model}_{mode}",
            ]
        )
        metadata = {
            "model": model,
            "mode": mode,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "label_smoothing": DEFAULT_LABEL_SMOOTHING,
            "drop_rate": DEFAULT_DROP_RATE,
            "val_ratio": DEFAULT_VAL_RATIO,
            "checkpoint": str(checkpoint_path),
            "history": str(run_dir / f"history_{model}.json"),
            "performance": str(run_dir / f"test_metrics_{model}_{mode}.json"),
        }
        (run_dir / "run_config.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    aggregate_results(root=root, models=models, seeds=seeds, modes=modes)


def aggregate_results(
    root: Path = Path("experiments"),
    models: Sequence[str] = MODELS,
    seeds: Sequence[int] = SEEDS,
    modes: Sequence[str] = MODES,
) -> Dict[str, object]:
    """Aggregate metrics, select by validation accuracy, and create the comparison plot."""
    aggregate: Dict[str, object] = {
        "architectures": {},
        "configurations": [],
        "seeds": list(seeds),
        "modes": list(modes),
        "selection_metric": "best validation average class accuracy",
    }
    plot_data = {}
    for model in models:
        aggregate["architectures"][model] = {}
        for mode in modes:
            runs = []
            for seed in seeds:
                run_dir = run_directory(root, model, mode, seed)
                metrics_path = run_dir / f"test_metrics_{model}_{mode}.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Missing experiment metrics: {metrics_path}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                checkpoint_path = run_dir / f"best_{model}.pth"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"Missing experiment checkpoint: {checkpoint_path}")
                history_path = run_dir / f"history_{model}.json"
                if not history_path.exists():
                    raise FileNotFoundError(f"Missing experiment history: {history_path}")
                history_payload = json.loads(history_path.read_text(encoding="utf-8"))
                checkpoint = _load_checkpoint_metadata(checkpoint_path)
                validation_history = history_payload.get("history", {}).get("val_ave_class", [])
                validation_average = float(
                    checkpoint.get(
                        "best_val_ave_class",
                        max(validation_history) if validation_history else 0.0,
                    )
                )
                run = {"seed": seed,
                    "model": model,
                    "mode": mode,
                    "validation_average_class_accuracy": validation_average,
                    "top1_accuracy_pct": metrics["top1_accuracy_pct"],
                    "average_accuracy_per_class_pct": metrics["average_accuracy_per_class_pct"],
                    "checkpoint_path": str(checkpoint_path),
                    "history_path": str(history_path),
                    "performance_path": str(metrics_path)}
                runs.append(run)
                aggregate["configurations"].append(run)
            top1 = [run["top1_accuracy_pct"] for run in runs]
            average_class = [run["average_accuracy_per_class_pct"] for run in runs]
            aggregate["architectures"][model][mode] = {
                "runs": runs,
                "top1_accuracy_pct": {"mean": float(np.mean(top1)), "std": float(np.std(top1))},
                "average_accuracy_per_class_pct": {"mean": float(np.mean(average_class)), "std": float(np.std(average_class))},
                "validation_average_class_accuracy": {
                    "mean": float(np.mean([run["validation_average_class_accuracy"] for run in runs])),
                    "std": float(np.std([run["validation_average_class_accuracy"] for run in runs])),
                },
            }
            plot_data[f"{model}_{mode}"] = (top1, average_class)

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = results_dir / "multi_seed_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _plot_results(plot_data, results_dir / "multi_seed_comparison.png")
    _write_production_selection(aggregate, results_dir)
    return aggregate


def _load_checkpoint_metadata(path: Path) -> Dict[str, object]:
    """Read checkpoint metadata without changing the production checkpoint set."""
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint metadata must be a dictionary: {path}")
    return checkpoint


def _write_production_selection(aggregate: Dict[str, object], results_dir: Path) -> Path:
    """Write a promotion manifest only after a validated checkpoint has been evaluated."""
    configurations = aggregate["configurations"]
    if not configurations:
        raise ValueError("Cannot select a production checkpoint from no configurations.")
    selected = max(
        configurations,
        key=lambda run: (
            run["validation_average_class_accuracy"],
            run["average_accuracy_per_class_pct"],
            run["top1_accuracy_pct"],
        ),
    )
    manifest = {
        "status": "evaluated",
        "selection_metric": aggregate["selection_metric"],
        "selected": selected,
        "official_test_metrics": selected["performance_path"],
        "baseline_preserved": True,
    }
    manifest_path = results_dir / "production_selection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _plot_results(data: Dict[str, tuple], path: Path) -> None:
    labels = ["Top-1 accuracy", "Average class accuracy"]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, (model, (top1, average_class)) in enumerate(data.items()):
        values = [top1, average_class]
        means = [np.mean(item) for item in values]
        spreads = [np.std(item) for item in values]
        positions = x + (offset - 1.5) * width
        ax.bar(
            positions,
            means,
            width,
            yerr=spreads,
            capsize=5,
            label=model.replace("_", " ").title(),
        )
        for metric_index, score_values in enumerate(values):
            ax.scatter(
                np.full(len(score_values), positions[metric_index]),
                score_values,
                color="black",
                s=28,
                zorder=3,
            )
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Multi-seed experiment: mean and run-to-run spread")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the twelve-run schedule without training")
    parser.add_argument("--root", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    schedule = experiment_schedule()
    if args.dry_run:
        print(json.dumps(schedule, indent=2))
        return
    run_experiment(root=args.root)


if __name__ == "__main__":
    main()
