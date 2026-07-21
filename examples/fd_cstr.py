"""Fault-detection smoke example on synthetic CSTR-like data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from joff import DataModule, FaultDetectionEvaluator, Trainer, build_model, seed_everything
from joff.artifacts import ArtifactStore
from joff.data.pipeline import Normalizer
from joff.evaluation import reconstruction_scores
from joff.plotting import FaultDetectionPlotter, TrainingPlotter
from _reporting import (
    print_artifacts,
    print_data_summary,
    print_history,
    print_kv,
    print_metrics,
    print_model_summary,
    print_section,
)


def main() -> None:
    """Train a DAE on normal CSTR data and evaluate reconstruction-based faults."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the smallest CPU smoke setup.")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    print_section("CSTR Fault Detection")
    seed = 31
    seed_everything(seed)
    rows = 72 if args.smoke else 180
    max_epochs = 1 if args.smoke else 5
    store = ArtifactStore(args.run_root, "fd_cstr_smoke" if args.smoke else "fd_cstr")
    config = {
        "seed": seed,
        "rows": rows,
        "model": {"type": "dae", "input_dim": 4, "latent_dim": 2, "hidden": [8], "act": ["r"]},
        "trainer": {"max_epochs": max_epochs, "optimizer": {"type": "adam", "lr": 0.01}},
        "fault_detection": {"expected_far": 0.05},
    }
    store.save_yaml("resolved_config.yaml", config)
    store.save_json("provenance.json", {"example": "fd_cstr", "data": "synthetic_cstr"})

    normal_train = _synthetic_cstr(rows, fault=False, seed=seed)
    normal_threshold = _synthetic_cstr(rows // 2, fault=False, seed=seed + 1)
    test_normal = _synthetic_cstr(rows // 2, fault=False, seed=seed + 2)
    test_fault = _synthetic_cstr(rows // 2, fault=True, seed=seed + 3)
    test = np.vstack([test_normal, test_fault])
    labels = np.concatenate([np.zeros(len(test_normal), dtype=int), np.ones(len(test_fault), dtype=int)])

    normalizer = Normalizer(method="standard").fit(normal_train)
    train_x = normalizer.transform(normal_train)
    threshold_x = normalizer.transform(normal_threshold)
    test_x = normalizer.transform(test)
    store.save_json("data/normalization_summary.json", normalizer.summary())

    data = DataModule.from_arrays(
        train_x,
        x_test=threshold_x,
        batch_size=12 if args.smoke else 24,
        shuffle=True,
    )
    model = build_model(config["model"])
    trainer = Trainer(
        max_epochs=max_epochs,
        device="cpu",
        optimizer=config["trainer"]["optimizer"],
        seed=seed,
        checkpoint_dir=store.path / "checkpoints",
    )
    print_kv("Resolved Config", {"seed": seed, "rows": rows, "max_epochs": max_epochs, "run_dir": store.path})
    print_data_summary(data, title="Normal Training Data")
    print_model_summary(model)
    training = trainer.fit(model, data)
    threshold_reconstruction = _reconstruct(model, threshold_x)
    test_reconstruction = _reconstruct(model, test_x)
    normal_scores = reconstruction_scores(threshold_x, threshold_reconstruction)
    test_scores = reconstruction_scores(test_x, test_reconstruction)
    report = FaultDetectionEvaluator(**config["fault_detection"]).fit_evaluate(
        normal_scores,
        test_scores,
        labels,
    )

    store.save_table("metrics/history.csv", training.history)
    store.save_json("metrics/fault_detection_report.json", report.to_dict())
    store.save_json("metrics/score_summary.json", _score_summary(normal_scores, test_scores))
    store.save_table(
        "metrics/test_scores.csv",
        [
            {"index": idx, "score": float(score), "label": int(label)}
            for idx, (score, label) in enumerate(zip(test_scores, labels, strict=True))
        ],
    )
    loss_figure = TrainingPlotter().loss_curve(training.history)
    store.save_figure("plots/loss.png", loss_figure)
    score_figure = _score_figure(test_scores, labels, report.threshold)
    store.save_figure("plots/fault_scores.png", score_figure)
    print_history(training.history)
    print_metrics(report.metrics, title="Fault Detection Metrics")
    print_kv(
        "Score Summary",
        {
            **_score_summary(normal_scores, test_scores),
            "threshold": report.threshold,
            "normal_rows": len(normal_scores),
            "test_rows": len(test_scores),
            "fault_rows": int(labels.sum()),
        },
    )
    print_artifacts(store.path)


def _synthetic_cstr(rows: int, *, fault: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, 8.0, rows)
    coolant = 0.6 + 0.08 * np.sin(time * 0.7) + 0.02 * rng.normal(size=rows)
    feed = 1.0 + 0.05 * np.cos(time * 0.5) + 0.015 * rng.normal(size=rows)
    concentration = 0.9 - 0.22 * coolant + 0.05 * np.sin(time) + 0.02 * rng.normal(size=rows)
    temperature = 1.2 + 0.28 * feed - 0.18 * coolant + 0.03 * np.cos(time * 1.3)
    temperature = temperature + 0.02 * rng.normal(size=rows)
    if fault:
        ramp = np.linspace(0.0, 1.0, rows)
        temperature = temperature + 0.9 * ramp
        concentration = concentration - 0.45 * ramp
        coolant = coolant - 0.25 * ramp
    return np.column_stack([feed, coolant, temperature, concentration]).astype(np.float32)


@torch.no_grad()
def _reconstruct(model: torch.nn.Module, data: np.ndarray) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(data, dtype=torch.float32)
    output = model(tensor)
    return output["reconstruction"].detach().cpu().numpy()


def _score_summary(normal_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, Any]:
    return {
        "normal_mean": float(np.mean(normal_scores)),
        "normal_max": float(np.max(normal_scores)),
        "test_mean": float(np.mean(test_scores)),
        "test_max": float(np.max(test_scores)),
    }


def _score_figure(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Any:
    return FaultDetectionPlotter().stat_curve(scores, threshold=threshold, labels=labels)


if __name__ == "__main__":
    main()
