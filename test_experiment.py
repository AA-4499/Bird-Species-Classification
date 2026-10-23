"""Acceptance tests for the isolated four-configuration experiment."""

import json
from pathlib import Path

import torch

from experiment_runner import (
    MODES,
    MODELS,
    SEEDS,
    aggregate_results,
    experiment_schedule,
    run_directory,
)
from models import build_model


def test_experiment_schedule_contains_twelve_isolated_jobs():
    schedule = experiment_schedule()

    assert len(schedule) == 12
    assert {(item["model"], item["mode"], item["seed"]) for item in schedule} == {
        (model, mode, seed)
        for model in MODELS
        for mode in MODES
        for seed in SEEDS
    }
    assert len(
        {
            run_directory(
                root=Path("experiments"),
                model=item["model"],
                mode=item["mode"],
                seed=item["seed"],
            )
            for item in schedule
        }
    ) == 12


def test_frozen_mode_only_trains_classifier_head():
    for model_name, head_name in (("resnet50", "fc"), ("convnext_tiny", "head")):
        model = build_model(model_name, num_classes=200, pretrained=False, fine_tune=False)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        assert trainable
        assert all(name.startswith(f"{head_name}.") for name in trainable)
        assert any(not parameter.requires_grad for parameter in model.parameters())


def test_fine_tuned_mode_trains_backbone_and_head():
    for model_name, head_name in (("resnet50", "fc"), ("convnext_tiny", "head")):
        model = build_model(model_name, num_classes=200, pretrained=False, fine_tune=True)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        assert any(name.startswith(f"{head_name}.") for name in trainable)
        assert any(not name.startswith(f"{head_name}.") for name in trainable)
        assert all(parameter.requires_grad for parameter in model.parameters())


def test_aggregate_selects_highest_validation_run_without_touching_baseline(tmp_path):
    for model in MODELS:
        for mode in MODES:
            for seed in SEEDS:
                run_dir = run_directory(tmp_path, model, mode, seed)
                run_dir.mkdir(parents=True)
                validation = (seed - 41) / 10 + (0.1 if mode == "fine_tuned" else 0)
                torch.save(
                    {
                        "model_name": model,
                        "seed": seed,
                        "best_val_ave_class": validation,
                    },
                    run_dir / f"best_{model}.pth",
                )
                (run_dir / f"history_{model}.json").write_text(
                    json.dumps({"history": {"val_ave_class": [validation]}}),
                    encoding="utf-8",
                )
                (run_dir / f"test_metrics_{model}_{mode}.json").write_text(
                    json.dumps(
                        {
                            "top1_accuracy_pct": 70 + seed,
                            "average_accuracy_per_class_pct": 60 + seed,
                        }
                    ),
                    encoding="utf-8",
                )

    baseline = tmp_path / "baseline-checkpoint.pth"
    baseline.write_bytes(b"baseline")
    aggregate = aggregate_results(root=tmp_path)
    manifest = json.loads(
        (tmp_path / "results" / "production_selection.json").read_text(encoding="utf-8")
    )

    assert set(aggregate["architectures"]) == set(MODELS)
    assert all(set(aggregate["architectures"][model]) == set(MODES) for model in MODELS)
    assert len(aggregate["configurations"]) == 12
    assert manifest["status"] == "evaluated"
    assert manifest["selection_metric"] == "best validation average class accuracy"
    assert manifest["selected"]["mode"] == "fine_tuned"
    assert manifest["selected"]["seed"] == 44
    assert set(manifest["selected_by_model"]) == set(MODELS)
    assert all(
        manifest["selected_by_model"][model]["mode"] == "fine_tuned"
        for model in MODELS
    )
    assert baseline.read_bytes() == b"baseline"
