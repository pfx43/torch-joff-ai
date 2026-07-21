from __future__ import annotations

import pandas as pd
import torch

from joff import DataModule, Experiment, ExperimentRunner
from joff.artifacts import ArtifactStore


def test_datamodule_from_arrays_pipeline_summaries(tmp_path) -> None:
    x = torch.linspace(0, 1, 30).unsqueeze(1).numpy()
    y = (x * 2.0)
    x[5, 0] = float("nan")
    data = DataModule.from_arrays(
        x,
        y,
        batch_size=8,
        missing={"strategy": "interpolate_then_drop"},
        outliers={"method": "mad", "feature_scope": "target", "max_removal_ratio": 0.2},
        normalization={"method": "standard"},
        split={"type": "sequential", "test_ratio": 0.25},
    )
    assert {"missing_summary", "outlier_summary", "normalization_summary", "split_summary"} <= set(
        data.summaries
    )
    result = Experiment.from_config(
        {
            "model": {"type": "mlp", "input_dim": 1, "output_dim": 1, "hidden": [4]},
            "artifacts": {"root": tmp_path, "name": "summaries_only"},
        }
    ).run()
    paths = data.save_summaries(ArtifactStore(tmp_path, "dm_summaries"))
    assert paths["split_summary"].exists()
    assert result.model is not None


def test_experiment_run_trains_and_saves_checkpoints_and_data_summaries(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": torch.linspace(0, 1, 40).numpy(),
            "x2": torch.linspace(1, 2, 40).numpy(),
            "y": torch.linspace(0, 1, 40).numpy() * 0.5,
        }
    )
    csv_path = tmp_path / "toy.csv"
    frame.to_csv(csv_path, index=False)

    experiment = Experiment.from_config(
        {
            "seed": 123,
            "data": {
                "path": csv_path,
                "batch_size": 8,
                "target_cols": -1,
                "missing": {"strategy": "interpolate_then_drop"},
                "normalization": {"method": "standard"},
                "split": {"type": "sequential", "test_ratio": 0.25},
            },
            "model": {"type": "mlp", "hidden": [8], "act": ["r"]},
            "trainer": {"max_epochs": 2, "optimizer": {"lr": 0.01}},
            "artifacts": {"root": tmp_path, "name": "experiment_train"},
        }
    )
    result = experiment.run()
    run_dir = result.run_dir
    assert result.history is not None and len(result.history) == 2
    assert result.metrics is not None and result.metrics["loss"] >= 0
    assert {"mse", "rmse", "mae", "r2"} <= set(result.metrics)
    assert (run_dir / "checkpoints" / "last.pt").exists()
    assert (run_dir / "checkpoints" / "best.pt").exists()
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert checkpoint["resolved_config"]["data"]["path"] == str(csv_path)
    assert checkpoint["resolved_config"]["model"]["input_dim"] == 2
    assert {"python", "numpy", "torch"} <= set(checkpoint["rng_state"])
    assert (run_dir / "metrics" / "history.csv").exists()
    assert (run_dir / "metrics" / "test_metrics.json").exists()
    assert (run_dir / "metrics" / "best_test_metrics.json").exists()
    assert (run_dir / "data" / "split_summary.json").exists()
    assert (run_dir / "data" / "split_summary.csv").exists()
    assert (run_dir / "data" / "normalization_summary.json").exists()
    assert (run_dir / "data" / "prepared_dataset_hash.json").exists()
    assert (run_dir / "logs" / "events.jsonl").exists()
    assert (run_dir / "plots" / "loss.png").exists()
    assert result.resolved_config["model"]["input_dim"] == 2
    assert result.resolved_config["model"]["output_dim"] == 1
    assert result.provenance["model.input_dim"][-1]["source"] == "derived:data_schema"
    assert result.provenance["model.output_dim"][-1]["source"] == "derived:data_schema"


def test_experiment_nkn_with_window_data_and_dynamic_split_artifacts(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "u": torch.linspace(0, 1, 48).numpy(),
            "y": torch.sin(torch.linspace(0, 3.14, 48)).numpy(),
        }
    )
    csv_path = tmp_path / "sequence.csv"
    frame.to_csv(csv_path, index=False)
    result = Experiment.from_config(
        {
            "seed": 5,
            "data": {
                "path": csv_path,
                "target_cols": -1,
                "batch_size": 4,
                "normalization": {"method": "standard"},
                "window": {"lookback": 2, "future_steps": 2},
                "split": {
                    "type": "dynamic_distribution",
                    "train_ratio": 0.75,
                    "test_ratio": 0.25,
                    "slice_length": 6,
                    "seed": 5,
                },
            },
            "model": {
                "type": "nkn",
                "hidden": [6],
                "act": ["r"],
                "coupling_layers": 2,
                "koopman": {"nice_loss_weight": 0.01},
            },
            "trainer": {"max_epochs": 1, "optimizer": {"lr": 0.005}},
            "artifacts": {"root": tmp_path, "name": "nkn_window"},
        }
    ).run()
    assert result.resolved_config["model"]["input_dim"] == 4
    assert result.resolved_config["model"]["output_dim"] == 4
    assert (result.run_dir / "checkpoints" / "best.pt").exists()
    assert (result.run_dir / "data" / "dynamic_slice_summary.csv").exists()
    assert (result.run_dir / "data" / "dynamic_split_summary.csv").exists()
    assert (result.run_dir / "data" / "dynamic_distribution_summary.csv").exists()
    assert (result.run_dir / "data" / "split_summary.csv").exists()
    assert (result.run_dir / "data" / "normalization_summary.json").exists()
    assert result.metrics is not None and result.metrics["loss"] >= 0


def test_experiment_run_from_dataset_card_preset_config(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "time": torch.arange(20).numpy(),
            "u": torch.linspace(0, 1, 20).numpy(),
            "y": torch.linspace(1, 2, 20).numpy(),
        }
    )
    test = pd.DataFrame(
        {
            "time": torch.arange(8).numpy(),
            "u": torch.linspace(1, 1.5, 8).numpy(),
            "y": torch.linspace(2, 2.5, 8).numpy(),
        }
    )
    train.to_csv(tmp_path / "preset_train.csv", index=False)
    test.to_csv(tmp_path / "preset_test.csv", index=False)
    card_path = tmp_path / "experiment_dataset_card.yaml"
    card_path.write_text(
        """
name: experiment_process
version: 1
files:
  root: .
  train: preset_train.csv
  test: preset_test.csv
schema:
  columns:
    - {name: time, role: time}
    - {name: u, role: control}
    - {name: y, role: output}
tasks:
  prediction:
    inputs: [control, output]
    targets: [output]
""".strip(),
        encoding="utf-8",
    )

    result = Experiment.from_config(
        {
            "data": {
                "preset": card_path,
                "task": "prediction",
                "batch_size": 4,
                "normalization": {"method": "standard"},
            },
            "model": {"type": "mlp", "hidden": [4], "act": ["r"]},
            "trainer": {"max_epochs": 1, "optimizer": {"lr": 0.01}},
            "artifacts": {"root": tmp_path, "name": "preset_experiment"},
        }
    ).run()
    assert result.resolved_config["data"]["preset"] == str(card_path)
    assert result.resolved_config["model"]["input_dim"] == 2
    assert result.resolved_config["model"]["output_dim"] == 1
    assert (result.run_dir / "data" / "schema.yaml").exists()
    assert (result.run_dir / "data" / "preset.yaml").exists()
    assert (result.run_dir / "data" / "pipeline.yaml").exists()


def test_experiment_run_with_imputation_mask_config(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x1": torch.linspace(0, 1, 24).numpy(),
            "x2": torch.linspace(1, 2, 24).numpy(),
        }
    )
    csv_path = tmp_path / "imputation.csv"
    frame.to_csv(csv_path, index=False)

    result = Experiment.from_config(
        {
            "seed": 21,
            "data": {
                "path": csv_path,
                "target_cols": None,
                "batch_size": 6,
                "normalization": {"method": "standard"},
                "split": {"type": "sequential", "test_ratio": 0.25},
                "mask": {"strategy": "random", "missing_rate": 0.25, "seed": 21},
            },
            "model": {"type": "mlp", "hidden": [6], "act": ["r"]},
            "trainer": {"max_epochs": 1, "optimizer": {"lr": 0.01}},
            "artifacts": {"root": tmp_path, "name": "imputation_experiment"},
        }
    ).run()
    assert result.resolved_config["model"]["input_dim"] == 4
    assert result.resolved_config["model"]["output_dim"] == 2
    assert (result.run_dir / "data" / "mask_summary.json").exists()


def test_experiment_data_pipeline_can_load_yaml_file(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x": torch.linspace(0, 1, 24).numpy(),
            "y": torch.linspace(1, 2, 24).numpy(),
        }
    )
    csv_path = tmp_path / "pipeline_source.csv"
    frame.to_csv(csv_path, index=False)
    pipeline_path = tmp_path / "data_pipeline.yaml"
    pipeline_path.write_text(
        """
pipeline:
  - split: {type: sequential, test_ratio: 0.25}
  - scaler: {method: standard}
""".strip(),
        encoding="utf-8",
    )

    result = Experiment.from_config(
        {
            "data": {
                "path": csv_path,
                "pipeline": pipeline_path,
                "batch_size": 6,
            },
            "model": {"type": "mlp", "hidden": [4], "act": ["r"]},
            "trainer": {"max_epochs": 1, "optimizer": {"lr": 0.01}},
            "artifacts": {"root": tmp_path, "name": "pipeline_file_experiment"},
        }
    ).run()
    assert result.data_summaries is not None
    assert result.data_summaries["normalization_summary"]["method"] == "standard"
    assert result.data_summaries["split_summary"]["type"] == "sequential"


def test_experiment_runner_runs_multiple_configs_and_writes_summary(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "x": torch.linspace(0, 1, 20).numpy(),
            "y": torch.linspace(0, 1, 20).numpy(),
        }
    )
    csv_path = tmp_path / "runner.csv"
    frame.to_csv(csv_path, index=False)
    base = {
        "data": {
            "path": csv_path,
            "target_cols": -1,
            "batch_size": 5,
            "split": {"type": "sequential", "test_ratio": 0.25},
        },
        "model": {"type": "mlp", "hidden": [4]},
        "trainer": {"max_epochs": 1, "optimizer": {"lr": 0.01}},
    }
    configs = []
    for idx, hidden in enumerate(([4], [6])):
        config = dict(base)
        config["model"] = {"type": "mlp", "hidden": hidden}
        config["artifacts"] = {"root": tmp_path, "name": f"runner_exp_{idx}"}
        configs.append(config)
    result = ExperimentRunner.from_config(
        {"name": "runner_summary", "artifacts": {"root": tmp_path}, "configs": configs}
    ).run()
    assert len(result.results) == 2
    assert len(result.summary) == 2
    assert result.run_dir is not None
    assert (result.run_dir / "summary" / "summary.csv").exists()
