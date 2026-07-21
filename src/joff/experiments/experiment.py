"""Minimal experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from joff.artifacts import ArtifactStore, RunLogger
from joff.core.config import ExperimentConfig, ModelConfig
from joff.core.factory import build_model
from joff.core.resolver import ConfigManager, ResolvedConfig
from joff.data import DataModule
from joff.plotting import TrainingPlotter
from joff.training import CheckpointManager, Trainer


@dataclass(frozen=True)
class ExperimentResult:
    """Result returned by :meth:`Experiment.run`."""

    run_dir: Path
    resolved_config: dict[str, Any]
    provenance: dict[str, Any]
    model: Any | None = None
    history: list[dict[str, float]] | None = None
    metrics: dict[str, float] | None = None
    checkpoint_paths: dict[str, Path] | None = None
    data_summaries: dict[str, Any] | None = None


class Experiment:
    """Compose config, model, and local artifacts for one experiment run."""

    def __init__(self, config: ExperimentConfig | ResolvedConfig) -> None:
        if isinstance(config, ResolvedConfig):
            self.resolved = config
        else:
            self.resolved = ResolvedConfig(
                config=config,
                raw_config=config.model_dump(mode="json"),
                provenance=ConfigManager().resolve(config.model_dump(mode="json")).provenance,
                config_hash="",
            )

    @classmethod
    def from_config(cls, source: str | Path | dict[str, Any]) -> "Experiment":
        """Build an experiment from a YAML path or mapping."""

        return cls(ConfigManager().resolve(source))

    def run(self, overrides: dict[str, Any] | None = None, **kwargs: Any) -> ExperimentResult:
        """Run a configured experiment and save config, metrics, summaries, and checkpoints."""

        resolved = self.resolved
        if overrides or kwargs:
            merged_overrides = dict(overrides or {})
            merged_overrides.update(kwargs)
            resolved = ConfigManager().resolve(
                resolved.raw_config,
                method_overrides=merged_overrides,
            )
        artifact_config = resolved.config.artifacts
        store = ArtifactStore(artifact_config.root, artifact_config.name or resolved.config_hash)
        logger = RunLogger(store.resolve("logs/events.jsonl"))
        logger.log_event("experiment_start", config_hash=resolved.config_hash)
        data = _build_data(resolved.config)
        model_config = _derive_model_config(resolved.config.model, data)
        run_config = resolved.config.model_copy(update={"model": model_config})
        resolved_dict = run_config.model_dump(mode="json")
        provenance_dict = resolved.provenance.to_dict()
        _record_derived_model_provenance(provenance_dict, resolved.config.model, model_config)
        store.save_yaml("resolved_config.yaml", resolved_dict)
        store.save_json("provenance.json", provenance_dict)
        model = build_model(model_config)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        store.save_json("model/parameter_count.json", {"parameters": parameter_count})
        logger.log_event("model_built", model_type=model_config.type, parameters=parameter_count)
        history: list[dict[str, float]] | None = None
        metrics: dict[str, float] | None = None
        checkpoint_paths: dict[str, Path] | None = None
        data_summaries: dict[str, Any] | None = None
        if data is not None:
            data.save_summaries(store)
            data_summaries = data.summaries
            trainer_config = run_config.trainer
            trainer = Trainer(
                max_epochs=trainer_config.max_epochs,
                device=run_config.device,
                optimizer=trainer_config.optimizer.model_dump(mode="json"),
                seed=run_config.seed,
                monitor=trainer_config.monitor,
                mode=trainer_config.mode,
                checkpoint_dir=store.path / "checkpoints",
                checkpoint_config=model_config.model_dump(mode="json"),
                resolved_config=resolved_dict,
            )
            training_result = trainer.fit(model, data)
            history = training_result.history
            checkpoint_paths = training_result.checkpoint_paths
            best_path = checkpoint_paths.get("best")
            if best_path is not None and best_path.exists():
                CheckpointManager(store.path / "checkpoints").load(
                    best_path,
                    model=model,
                    map_location=trainer.device,
                )
            metrics = trainer.evaluate(model, data)
            store.save_table("metrics/history.csv", history)
            store.save_json("metrics/test_metrics.json", metrics)
            loss_figure = TrainingPlotter().loss_curve(history)
            store.save_figure("plots/loss.png", loss_figure)
            if best_path is not None:
                store.save_json(
                    "metrics/best_test_metrics.json",
                    {"checkpoint": str(best_path), "metrics": metrics},
                )
            logger.log_event("experiment_evaluated", metrics=metrics)
        logger.log_event("experiment_end", run_dir=store.path)
        return ExperimentResult(
            run_dir=store.path,
            resolved_config=resolved_dict,
            provenance=provenance_dict,
            model=model,
            history=history,
            metrics=metrics,
            checkpoint_paths=checkpoint_paths,
            data_summaries=data_summaries,
        )


def _build_data(config: ExperimentConfig) -> DataModule | None:
    data_config = config.data
    if data_config.preset is not None:
        return DataModule.from_preset(
            data_config.preset,
            root=data_config.root,
            task=data_config.task,
            pipeline=data_config.pipeline,
            batch_size=data_config.batch_size,
            test_ratio=data_config.test_ratio,
            seed=data_config.seed,
            shuffle=data_config.shuffle,
            missing=data_config.missing,
            outliers=data_config.outliers,
            normalization=data_config.normalization,
            split=data_config.split,
            mask=data_config.mask,
            window=data_config.window,
            sequence=data_config.sequence,
            mpc_window=data_config.mpc_window,
        )
    if data_config.path is None:
        return None
    return DataModule.from_file(
        data_config.path,
        target_cols=data_config.target_cols,
        batch_size=data_config.batch_size,
        test_ratio=data_config.test_ratio,
        seed=data_config.seed,
        pipeline=data_config.pipeline,
        shuffle=data_config.shuffle,
        missing=data_config.missing,
        outliers=data_config.outliers,
        normalization=data_config.normalization,
        split=data_config.split,
        mask=data_config.mask,
        window=data_config.window,
        sequence=data_config.sequence,
        mpc_window=data_config.mpc_window,
    )


def _derive_model_config(config: ModelConfig, data: DataModule | None) -> ModelConfig:
    if data is None:
        return config
    model_type = config.type.strip().lower()
    sequence_like = model_type in {"sequence", "sequence_regressor", "rnn", "gru", "lstm"}
    input_dim, output_dim = _infer_dims_from_dataset(data.train_dataset, sequence_like=sequence_like)
    updates: dict[str, Any] = {}
    if config.input_dim is None:
        updates["input_dim"] = input_dim
    if model_type in {"mlp", "nkn", "sequence", "sequence_regressor", "rnn", "gru", "lstm"} and (
        config.output_dim is None and output_dim is not None
    ):
        updates["output_dim"] = output_dim
    return config.model_copy(update=updates) if updates else config


def _record_derived_model_provenance(
    provenance: dict[str, Any],
    original: ModelConfig,
    derived: ModelConfig,
) -> None:
    for field in ("input_dim", "output_dim"):
        old_value = getattr(original, field)
        new_value = getattr(derived, field)
        if old_value is None and new_value is not None:
            provenance.setdefault(f"model.{field}", []).append(
                {"source": "derived:data_schema", "value": new_value}
            )


def _infer_dims_from_dataset(dataset: Any, *, sequence_like: bool = False) -> tuple[int, int | None]:
    sample = dataset[0]
    x: torch.Tensor
    y: torch.Tensor | None = None
    if isinstance(sample, dict):
        x = _first_tensor(sample, ("x", "input", "inputs", "features", "history"))
        y = _optional_tensor(sample, ("y", "target", "targets", "future", "label", "labels"))
    elif isinstance(sample, (tuple, list)):
        x = sample[0]
        y = sample[1] if len(sample) > 1 else None
    else:
        x = sample
    if sequence_like:
        input_dim = int(x.shape[-1]) if x.ndim >= 2 else int(x.numel())
        output_dim = None if y is None else int(y.shape[-1]) if y.ndim >= 2 else int(y.numel())
        return input_dim, output_dim
    return int(x.numel()), None if y is None else int(y.numel())


def _first_tensor(sample: dict[str, Any], keys: tuple[str, ...]) -> torch.Tensor:
    tensor = _optional_tensor(sample, keys)
    if tensor is None:
        raise ValueError(f"Cannot infer input dimension from sample keys: {sorted(sample)}.")
    return tensor


def _optional_tensor(sample: dict[str, Any], keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        if key in sample:
            value = sample[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            return value
    return None
