"""PyTorch DataModule with lightweight pipeline integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from joff.artifacts import ArtifactStore
from joff.data.adapters import DATASET_REGISTRY, LEGACY_SPECIAL_PRESETS
from joff.data.adapters.base import CanonicalDataset
from joff.data.pipeline import (
    DataPipeline,
    DynamicDistributionSplitter,
    DynamicWindowDataset,
    ImputationMasker,
    MPCWindowDataset,
    MissingValueProcessor,
    Normalizer,
    OutlierConfig,
    OutlierProcessor,
    SequentialSplitter,
    SequenceDataset,
    TabularSeries,
    merge_pipeline_configs,
)
from joff.data.schema import DataSchema, TaskSchema
from joff.data.sources import read_source, split_source_xy
from joff.data.tasks import TaskView


class DataModule:
    """Small data container that returns PyTorch dataloaders and pipeline summaries."""

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset | None = None,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        summaries: dict[str, Any] | None = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.summaries = summaries or {}

    @classmethod
    def from_arrays(
        cls,
        x_train: Any,
        y_train: Any | None = None,
        x_test: Any | None = None,
        y_test: Any | None = None,
        *,
        batch_size: int = 32,
        shuffle: bool = True,
        test_ratio: float = 0.2,
        seed: int = 42,
        groups: Any | None = None,
        pipeline: dict[str, Any] | list[Any] | str | Path | DataPipeline | None = None,
        missing: dict[str, Any] | None = None,
        outliers: dict[str, Any] | None = None,
        normalization: dict[str, Any] | None = None,
        split: dict[str, Any] | None = None,
        mask: dict[str, Any] | None = None,
        window: dict[str, Any] | None = None,
        sequence: dict[str, Any] | None = None,
        mpc_window: dict[str, Any] | None = None,
    ) -> "DataModule":
        """Create a data module from array-like data and optional pipeline configs."""

        pipeline_config = _apply_explicit_pipeline_overrides(
            _normalize_pipeline_config(pipeline),
            missing=missing,
            outliers=outliers,
            normalization=normalization,
            split=split,
            mask=mask,
            window=window,
            sequence=sequence,
            mpc_window=mpc_window,
        )
        missing = pipeline_config.get("missing")
        outliers = pipeline_config.get("outliers")
        normalization = pipeline_config.get("normalization")
        split = _split_for_arrays(pipeline_config.get("split"))
        mask = pipeline_config.get("mask")
        window = pipeline_config.get("window")
        sequence = pipeline_config.get("sequence")
        if pipeline_config.get("mpc_window") is not None:
            raise ValueError(
                "pipeline.mpc_window requires schema role metadata. "
                "Use DataModule.from_preset with a dataset card or registered preset."
            )
        if mask is not None and y_train is not None:
            raise ValueError(
                "Imputation mask generation uses x as the target and requires y_train=None. "
                "For file inputs use target_cols=None when mask is enabled."
            )
        if sequence is not None:
            if mask is not None or window is not None:
                raise ValueError(
                    "sequence data cannot be combined with mask or window. Legal options are: "
                    "use sequence for RNN/GRU/LSTM data, window for flattened dynamic samples, "
                    "or mask for tabular imputation."
                )
            return cls._from_sequence_arrays(
                x_train,
                y_train,
                x_test,
                y_test,
                batch_size=batch_size,
                shuffle=_sequence_shuffle(sequence, fallback=shuffle),
                missing=missing,
                outliers=outliers,
                normalization=normalization,
                split=split,
                sequence=sequence,
                seed=seed,
                test_ratio=test_ratio,
                groups=groups,
            )
        if window is not None:
            if mask is not None:
                raise ValueError(
                    "mask with window data is not supported yet. Legal options are: "
                    "use mask for tabular imputation or window for dynamic prediction."
                )
            return cls._from_window_arrays(
                x_train,
                y_train,
                batch_size=batch_size,
                shuffle=shuffle,
                missing=missing,
                outliers=outliers,
                normalization=normalization,
                split=split,
                window=window,
                seed=seed,
                test_ratio=test_ratio,
            )

        if x_test is None:
            group_array = _optional_group_array(groups, expected_rows=np.asarray(x_train).shape[0])
            series = _clean_series(
                TabularSeries(x=np.asarray(x_train, dtype=float), y=_optional_array(y_train)),
                missing=missing,
                outliers=None,
            )
            split_groups = _align_groups_to_series(group_array, series.data)
            train_idx, test_idx, split_summary = _split_indices(
                series.data.row_count,
                split=split,
                test_ratio=test_ratio,
                seed=seed,
                labels=series.data.y,
                groups=split_groups,
            )
            train_series = _series_take(series.data, train_idx)
            test_series = _series_take(series.data, test_idx)
            summaries = dict(series.summaries)
            if outliers is not None:
                train_series, test_series, outlier_summaries = _apply_outliers_train_test(
                    train_series,
                    test_series,
                    outliers,
                )
                summaries.update(outlier_summaries)
                split_summary = {
                    **split_summary,
                    "train_samples_after_outliers": train_series.row_count,
                    "test_samples_after_outliers": test_series.row_count,
                }
            train_x = train_series.x
            train_y = train_series.y
            test_x = test_series.x
            test_y = test_series.y
            summaries["split_summary"] = split_summary
        else:
            train_series = _clean_series(
                TabularSeries(x=np.asarray(x_train, dtype=float), y=_optional_array(y_train)),
                missing=missing,
                outliers=None,
            )
            test_series = _clean_series(
                TabularSeries(x=np.asarray(x_test, dtype=float), y=_optional_array(y_test)),
                missing=missing,
                outliers=None,
            )
            summaries = dict(train_series.summaries)
            if test_series.summaries.get("missing_summary"):
                summaries["test_missing_summary"] = test_series.summaries["missing_summary"]
            train_data = train_series.data
            test_data = test_series.data
            if outliers is not None:
                train_data, test_data, outlier_summaries = _apply_outliers_train_test(
                    train_data,
                    test_data,
                    outliers,
                )
                summaries.update(outlier_summaries)
            train_x = train_data.x
            train_y = train_data.y
            test_x = test_data.x
            test_y = test_data.y
            summaries["split_summary"] = {
                "type": "provided",
                "train_samples": int(train_x.shape[0]),
                "test_samples": int(test_x.shape[0]),
            }

        if normalization is not None:
            normalizer = Normalizer(**normalization)
            train_x, test_x = normalizer.fit_transform_train_test(train_x, test_x)
            summaries["normalization_summary"] = normalizer.summary()

        if mask is not None:
            masker = ImputationMasker(**mask)
            train_result = masker.transform_split(train_x, split="train", seed_offset=0)
            test_result = masker.transform_split(test_x, split="test", seed_offset=10_000)
            train_dataset = train_result.dataset
            test_dataset = test_result.dataset
            summaries["mask_summary"] = {
                "train": train_result.summary,
                "test": test_result.summary,
            }
        else:
            train_dataset = _tensor_dataset(train_x, train_y)
            test_dataset = _tensor_dataset(test_x, test_y)
        return cls(
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            summaries=summaries,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        target_cols: int | list[int] | slice | None = -1,
        batch_size: int = 32,
        test_ratio: float = 0.2,
        seed: int = 42,
        groups: Any | None = None,
        pipeline: dict[str, Any] | list[Any] | str | Path | DataPipeline | None = None,
        shuffle: bool = True,
        missing: dict[str, Any] | None = None,
        outliers: dict[str, Any] | None = None,
        normalization: dict[str, Any] | None = None,
        split: dict[str, Any] | None = None,
        mask: dict[str, Any] | None = None,
        window: dict[str, Any] | None = None,
        sequence: dict[str, Any] | None = None,
        mpc_window: dict[str, Any] | None = None,
    ) -> "DataModule":
        """Load a supported source file and apply the lightweight pipeline."""

        source = read_source(path)
        x, y = split_source_xy(source, target_cols=target_cols)
        data = cls.from_arrays(
            x,
            y,
            batch_size=batch_size,
            shuffle=shuffle,
            test_ratio=test_ratio,
            seed=seed,
            groups=groups,
            pipeline=pipeline,
            missing=missing,
            outliers=outliers,
            normalization=normalization,
            split=split,
            mask=mask,
            window=window,
            sequence=sequence,
            mpc_window=mpc_window,
        )
        data.summaries["source_summary"] = source.summary()
        return data

    @classmethod
    def from_preset(
        cls,
        preset: str | Path,
        *,
        root: str | Path | None = None,
        task: str | None = None,
        pipeline: dict[str, Any] | list[Any] | str | Path | DataPipeline | None = None,
        batch_size: int | None = None,
        test_ratio: float = 0.2,
        seed: int = 42,
        shuffle: bool | None = None,
        missing: dict[str, Any] | None = None,
        outliers: dict[str, Any] | None = None,
        normalization: dict[str, Any] | None = None,
        split: dict[str, Any] | None = None,
        mask: dict[str, Any] | None = None,
        window: dict[str, Any] | None = None,
        sequence: dict[str, Any] | None = None,
        mpc_window: dict[str, Any] | None = None,
    ) -> "DataModule":
        """Create a data module from a registered preset or ``dataset_card.yaml`` path."""

        adapter = DATASET_REGISTRY.resolve(preset)
        task_name = _normalize_task_alias(task)
        task_schema = adapter.default_task(task_name)
        resolved_root = _resolve_dataset_root_alias(root)
        canonical = adapter.read(root=resolved_root, task=task_schema.name)
        schema = canonical.schema if canonical.schema.columns else adapter.schema()

        pipeline_config = _merge_pipeline_configs(
            adapter.default_pipeline(task_schema.name),
            _normalize_pipeline_config(pipeline),
        )
        pipeline_config = _apply_explicit_pipeline_overrides(
            pipeline_config,
            missing=missing,
            outliers=outliers,
            normalization=normalization,
            split=split,
            mask=mask,
            window=window,
            sequence=sequence,
            mpc_window=mpc_window,
        )
        to_torch = pipeline_config.get("to_torch", {})
        if not isinstance(to_torch, Mapping):
            raise ValueError("pipeline.to_torch must be a mapping when provided.")
        effective_batch_size = int(batch_size or to_torch.get("batch_size", 32))
        effective_shuffle = bool(shuffle if shuffle is not None else to_torch.get("shuffle_train", True))

        train_frame = _concat_split(canonical, "train")
        test_frame = _optional_concat_split(canonical, "test")
        task_view = TaskView.from_schema(train_frame, schema, task_schema)
        task_frames = {"train": train_frame}
        if test_frame is not None:
            task_frames["test"] = test_frame
        label_mapping = task_view.label_mapping(task_frames)
        label_mapping_summary = task_view.label_summary(task_frames, label_mapping=label_mapping)
        if pipeline_config.get("mpc_window") is not None:
            data = cls._from_mpc_frames(
                train_frame,
                test_frame,
                schema,
                task_schema,
                batch_size=effective_batch_size,
                shuffle=effective_shuffle,
                missing=pipeline_config.get("missing"),
                outliers=pipeline_config.get("outliers"),
                normalization=pipeline_config.get("normalization"),
                split=_split_for_arrays(pipeline_config.get("split")),
                mpc_window=pipeline_config["mpc_window"],
                seed=seed,
                test_ratio=test_ratio,
            )
        else:
            train_x, train_y = task_view.arrays(train_frame, label_mapping=label_mapping)
            train_groups = _frame_group_array(train_frame, schema)
            if test_frame is None:
                test_x = None
                test_y = None
            else:
                test_x, test_y = task_view.arrays(test_frame, label_mapping=label_mapping)

            data = cls.from_arrays(
                train_x,
                train_y,
                test_x,
                test_y,
                batch_size=effective_batch_size,
                shuffle=effective_shuffle,
                test_ratio=test_ratio,
                seed=seed,
                groups=train_groups if test_frame is None else None,
                missing=pipeline_config.get("missing"),
                outliers=pipeline_config.get("outliers"),
                normalization=pipeline_config.get("normalization"),
                split=_split_for_arrays(pipeline_config.get("split")),
                mask=pipeline_config.get("mask"),
                window=pipeline_config.get("window"),
                sequence=pipeline_config.get("sequence"),
            )
        preset_summary = adapter.summary(task_schema.name)
        source_summary = canonical.source_summary()
        preprocessing_summary = _source_preprocessing_summary(source_summary)
        reported_pipeline_steps = {
            "missing",
            "outliers",
            "normalization",
            "split",
            "mask",
            "window",
            "sequence",
            "mpc_window",
            "to_torch",
        }
        pipeline_summary = {
            key: _json_ready(value)
            for key, value in pipeline_config.items()
            if key in reported_pipeline_steps
        }
        data.summaries = {
            "schema_summary": schema.summary(),
            "preset_summary": preset_summary,
            "task_summary": task_schema.summary(),
            "task_view_summary": task_view.summary(train_frame),
            **({"label_mapping_summary": label_mapping_summary} if label_mapping_summary else {}),
            **({"preprocessing_summary": preprocessing_summary} if preprocessing_summary else {}),
            "source_summary": source_summary,
            "pipeline_summary": pipeline_summary,
            **data.summaries,
        }
        return data

    @classmethod
    def from_legacy_special(
        cls,
        special: str,
        **kwargs: Any,
    ) -> "DataModule":
        """Compatibility shim for old ``special=...`` dataset names."""

        if special not in LEGACY_SPECIAL_PRESETS:
            legal = ", ".join(sorted(LEGACY_SPECIAL_PRESETS))
            raise ValueError(
                f"Unknown legacy special dataset {special!r}. Legal options are: {legal}. "
                f"Current input was: {special!r}."
            )
        return cls.from_preset(LEGACY_SPECIAL_PRESETS[special], **kwargs)

    @classmethod
    def quick(
        cls,
        *,
        preset: str | Path | None = None,
        path: str | Path | None = None,
        task: str | None = None,
        scaler: str | None = None,
        history: int | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> "DataModule":
        """Small convenience constructor for common preset/file workflows."""

        normalization = kwargs.pop("normalization", None)
        if scaler is not None:
            normalization = {"method": scaler}
        window = kwargs.pop("window", None)
        if history is not None:
            window = dict(window or {})
            window.setdefault("lookback", history)
        if preset is not None:
            return cls.from_preset(
                preset,
                task=task,
                batch_size=batch_size,
                normalization=normalization,
                window=window,
                **kwargs,
            )
        if path is not None:
            return cls.from_file(
                path,
                batch_size=batch_size,
                normalization=normalization,
                window=window,
                **kwargs,
            )
        raise ValueError("DataModule.quick requires either 'preset' or 'path'.")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def test_dataloader(self) -> DataLoader | None:
        """Return the test dataloader if present."""

        if self.test_dataset is None:
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def prepare(self) -> "DataModule":
        """Return ``self`` for compatibility with prepare-style data APIs."""

        return self

    def loader(self, split: str) -> DataLoader:
        """Return a dataloader by split name."""

        normalized = split.strip().lower()
        if normalized == "train":
            return self.train_dataloader()
        if normalized == "test":
            loader = self.test_dataloader()
            if loader is None:
                raise ValueError("DataModule has no test split.")
            return loader
        raise ValueError(f"Unknown split {split!r}. Legal options are: train, test.")

    def save_summaries(self, store: ArtifactStore, *, prefix: str = "data") -> dict[str, Path]:
        """Save available pipeline summaries under an artifact store."""

        paths: dict[str, Path] = {}
        for name, summary in self.summaries.items():
            paths[name] = store.save_json(f"{prefix}/{name}.json", _json_ready(summary))
        paths["prepared_dataset_hash"] = store.save_json(
            f"{prefix}/prepared_dataset_hash.json",
            _prepared_dataset_hash_summary(self),
        )
        if "normalization_summary" in self.summaries:
            paths["scaler_summary"] = store.save_json(
                f"{prefix}/scaler_summary.json",
                _json_ready(self.summaries["normalization_summary"]),
            )
        if "schema_summary" in self.summaries:
            paths["schema_yaml"] = store.save_yaml(
                f"{prefix}/schema.yaml",
                _json_ready(self.summaries["schema_summary"]),
            )
        if "preset_summary" in self.summaries:
            paths["preset_yaml"] = store.save_yaml(
                f"{prefix}/preset.yaml",
                _json_ready(self.summaries["preset_summary"]),
            )
        if "pipeline_summary" in self.summaries:
            paths["pipeline_yaml"] = store.save_yaml(
                f"{prefix}/pipeline.yaml",
                _json_ready(self.summaries["pipeline_summary"]),
            )
        if "outlier_removed_indices" in self.summaries:
            paths["outlier_removed_indices_csv"] = store.save_table(
                f"{prefix}/outlier_removed_indices.csv",
                [{"index": item} for item in self.summaries["outlier_removed_indices"]],
            )
        summary_tables = (
            "missing_summary",
            "outlier_summary",
            "normalization_summary",
            "window_summary",
            "sequence_summary",
            "mask_summary",
            "mpc_window_summary",
            "label_mapping_summary",
            "preprocessing_summary",
        )
        for name in summary_tables:
            if name in self.summaries:
                paths[f"{name}_csv"] = store.save_table(
                    f"{prefix}/{name}.csv",
                    _summary_rows(self.summaries[name]),
                )
        if "split_summary" in self.summaries:
            split_summary = self.summaries["split_summary"]
            paths["split_summary_csv"] = store.save_table(
                f"{prefix}/split_summary.csv",
                [_flat_summary(split_summary, exclude={"slice_summary", "distribution_summary"})],
            )
            if isinstance(split_summary, dict) and "slice_summary" in split_summary:
                paths["dynamic_slice_summary_csv"] = store.save_table(
                    f"{prefix}/dynamic_slice_summary.csv",
                    split_summary["slice_summary"],
                )
                paths["dynamic_split_summary_csv"] = store.save_table(
                    f"{prefix}/dynamic_split_summary.csv",
                    [_flat_summary(split_summary, exclude={"slice_summary", "distribution_summary"})],
                )
            if isinstance(split_summary, dict) and "distribution_summary" in split_summary:
                paths["dynamic_distribution_summary_csv"] = store.save_table(
                    f"{prefix}/dynamic_distribution_summary.csv",
                    split_summary["distribution_summary"],
                )
        return paths

    @classmethod
    def _from_sequence_arrays(
        cls,
        x: Any,
        y: Any | None,
        x_test: Any | None,
        y_test: Any | None,
        *,
        batch_size: int,
        shuffle: bool,
        missing: dict[str, Any] | None,
        outliers: dict[str, Any] | None,
        normalization: dict[str, Any] | None,
        split: dict[str, Any] | None,
        sequence: dict[str, Any],
        seed: int,
        test_ratio: float,
        groups: Any | None,
    ) -> "DataModule":
        config = _resolve_sequence_config(sequence)
        if y is None:
            if not config["target_from_input"]:
                raise ValueError(
                    "sequence data requires y/target values. Legal options are: provide y_train "
                    "or set sequence.target_from_input=true for autoregressive targets."
                )
            y = x
        raw_groups = _optional_group_array(groups, expected_rows=np.asarray(x).shape[0])
        train_clean = _clean_series(
            TabularSeries(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), quality=np.asarray(y, dtype=float)),
            missing=missing,
            outliers=None,
        )
        summaries = dict(train_clean.summaries)
        if x_test is None:
            cleaned_groups = _groups_for_series(raw_groups, train_clean.data)
            train_idx, test_idx, split_summary = _sequence_split_indices(
                train_clean.data.row_count,
                split=split,
                test_ratio=test_ratio,
                seed=seed,
                groups=cleaned_groups,
            )
            train_series = _series_take(train_clean.data, train_idx)
            test_series = _series_take(train_clean.data, test_idx)
        else:
            if y_test is None:
                if not config["target_from_input"]:
                    raise ValueError(
                        "sequence data with x_test requires y_test. Legal options are: provide "
                        "y_test or set sequence.target_from_input=true."
                    )
                y_test = x_test
            test_clean = _clean_series(
                TabularSeries(
                    x=np.asarray(x_test, dtype=float),
                    y=np.asarray(y_test, dtype=float),
                    quality=np.asarray(y_test, dtype=float),
                ),
                missing=missing,
                outliers=None,
            )
            if test_clean.summaries.get("missing_summary"):
                summaries["test_missing_summary"] = test_clean.summaries["missing_summary"]
            train_series = train_clean.data
            test_series = test_clean.data
            split_summary = {
                "type": "provided",
                "train_rows": int(train_series.row_count),
                "test_rows": int(test_series.row_count),
            }

        if outliers is not None:
            train_series, test_series, outlier_summaries = _apply_outliers_train_test(
                train_series,
                test_series,
                outliers,
            )
            summaries.update(outlier_summaries)
            split_summary = {
                **split_summary,
                "train_rows_after_outliers": int(train_series.row_count),
                "test_rows_after_outliers": int(test_series.row_count),
            }
        if normalization is not None:
            train_series, test_series, normalization_summary = _normalize_sequence_train_test(
                train_series,
                test_series,
                normalization,
            )
            summaries["normalization_summary"] = normalization_summary

        train_groups = _groups_for_series(raw_groups, train_series)
        test_groups = _groups_for_series(raw_groups, test_series) if x_test is None else None
        train_dataset = _sequence_dataset_from_series(train_series, config, segment_ids=train_groups)
        test_dataset = _sequence_dataset_from_series(test_series, config, segment_ids=test_groups)
        _require_nonempty_sequence_dataset(train_dataset, split="train")
        _require_nonempty_sequence_dataset(test_dataset, split="test")
        summaries["split_summary"] = split_summary
        summaries["sequence_summary"] = {
            **train_dataset.summary(),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
        }
        return cls(
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            summaries=summaries,
        )

    @classmethod
    def _from_window_arrays(
        cls,
        x: Any,
        y: Any | None,
        *,
        batch_size: int,
        shuffle: bool,
        missing: dict[str, Any] | None,
        outliers: dict[str, Any] | None,
        normalization: dict[str, Any] | None,
        split: dict[str, Any] | None,
        window: dict[str, Any],
        seed: int,
        test_ratio: float,
    ) -> "DataModule":
        if y is None:
            raise ValueError("window data requires y/target values to construct future targets.")
        series = _clean_series(
            TabularSeries(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float), quality=np.asarray(y, dtype=float)),
            missing=missing,
            outliers=outliers,
        )
        lookback = int(window.get("lookback", window.get("history", window.get("p", 3))))
        future_steps = int(window.get("future_steps", window.get("horizon", 1)))
        source_x = series.data.x
        source_y = series.data.y
        dataset = DynamicWindowDataset(
            source_x,
            source_y,
            lookback=lookback,
            future_steps=future_steps,
            return_mode=window.get("return_mode", "dict"),
        )
        split_config = split or {"type": "sequential", "test_ratio": test_ratio, "seed": seed}
        split_type = _split_type(split_config, default="sequential")
        if split_type == "dynamic_distribution":
            splitter = DynamicDistributionSplitter(
                train_ratio=float(split_config.get("train_ratio", 1.0 - test_ratio)),
                eval_ratio=float(split_config.get("eval_ratio", 0.0)),
                test_ratio=float(split_config.get("test_ratio", test_ratio)),
                slice_length=int(split_config.get("slice_length", 64)),
                min_slice_length=split_config.get("min_slice_length"),
                seed=int(split_config.get("seed", seed)),
            )
        elif split_type == "sequential":
            splitter = SequentialSplitter(
                train_ratio=float(split_config.get("train_ratio", 1.0 - test_ratio)),
                eval_ratio=float(split_config.get("eval_ratio", 0.0)),
                test_ratio=float(split_config.get("test_ratio", test_ratio)),
            )
        else:
            raise ValueError(
                f"Unknown window split type {split_type!r}. Legal options are: sequential, dynamic_distribution."
            )
        result = splitter.split(dataset)
        summaries = dict(series.summaries)
        summaries["split_summary"] = result.summary
        summaries["window_summary"] = {
            "lookback": lookback,
            "future_steps": future_steps,
            "samples": len(dataset),
        }
        train_dataset = result.train
        test_dataset = result.test
        if normalization is not None:
            matrix = np.concatenate([source_x, source_y], axis=1)
            train_rows = _covered_dynamic_rows(dataset, result.split_indices["train"])
            normalizer = Normalizer(**normalization)
            normalizer.fit(matrix[train_rows])
            transformed = normalizer.transform(matrix)
            x_width = source_x.shape[1]
            normalized_dataset = DynamicWindowDataset(
                transformed[:, :x_width],
                transformed[:, x_width:],
                lookback=lookback,
                future_steps=future_steps,
                return_mode=window.get("return_mode", "dict"),
            )
            train_dataset = normalized_dataset.subset(result.split_indices["train"])
            test_dataset = normalized_dataset.subset(result.split_indices["test"])
            summaries["normalization_summary"] = normalizer.summary()
        return cls(
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            summaries=summaries,
        )

    @classmethod
    def _from_mpc_frames(
        cls,
        train_frame: pd.DataFrame,
        test_frame: pd.DataFrame | None,
        schema: DataSchema,
        task: TaskSchema,
        *,
        batch_size: int,
        shuffle: bool,
        missing: dict[str, Any] | None,
        outliers: dict[str, Any] | None,
        normalization: dict[str, Any] | None,
        split: dict[str, Any] | None,
        mpc_window: dict[str, Any],
        seed: int,
        test_ratio: float,
    ) -> "DataModule":
        role_columns = _mpc_role_columns(train_frame, schema, task)
        target_columns = _mpc_target_columns(train_frame, schema, task)
        selected_columns = _unique_columns(
            [
                *role_columns["state"],
                *role_columns["output"],
                *role_columns["control"],
                *role_columns["reference"],
                *target_columns,
            ]
        )
        if not selected_columns:
            raise ValueError("MPC task did not resolve any numeric schema columns.")

        if test_frame is None:
            groups = _frame_group_array(train_frame, schema)
            effective_split = split
            if effective_split is None:
                if groups is None:
                    effective_split = {"method": "sequential", "test_ratio": test_ratio, "seed": seed}
                else:
                    effective_split = {"method": "episode", "test_ratio": test_ratio, "seed": seed}
            train_idx, test_idx, split_summary = _split_indices(
                train_frame.shape[0],
                split=effective_split,
                test_ratio=test_ratio,
                seed=seed,
                groups=groups,
            )
            train_part = train_frame.iloc[train_idx].reset_index(drop=True)
            test_part = train_frame.iloc[test_idx].reset_index(drop=True)
        else:
            train_part = train_frame.reset_index(drop=True)
            test_part = test_frame.reset_index(drop=True)
            split_summary = {
                "type": "provided",
                "train_samples": int(train_part.shape[0]),
                "test_samples": int(test_part.shape[0]),
            }

        train_part, summaries = _clean_frame_columns(
            train_part,
            selected_columns,
            missing=missing,
            outliers=outliers,
        )
        test_part, test_summaries = _clean_frame_columns(
            test_part,
            selected_columns,
            missing=missing,
            outliers=None,
        )
        if test_summaries.get("missing_summary"):
            summaries["test_missing_summary"] = test_summaries["missing_summary"]
        summaries["split_summary"] = split_summary

        if normalization is not None:
            normalizer = Normalizer(**normalization)
            normalizer.fit(_numeric_frame(train_part, selected_columns, field_name="MPC normalization").to_numpy(float))
            train_part.loc[:, selected_columns] = normalizer.transform(
                _numeric_frame(train_part, selected_columns, field_name="MPC normalization").to_numpy(float)
            )
            test_part.loc[:, selected_columns] = normalizer.transform(
                _numeric_frame(test_part, selected_columns, field_name="MPC normalization").to_numpy(float)
            )
            summaries["normalization_summary"] = normalizer.summary()

        train_dataset = _mpc_dataset_from_frame(train_part, schema, task, mpc_window)
        test_dataset = _mpc_dataset_from_frame(test_part, schema, task, mpc_window)
        summaries["mpc_window_summary"] = {
            "past_horizon": int(mpc_window.get("past_horizon", 10)),
            "prediction_horizon": int(mpc_window.get("prediction_horizon", 20)),
            "control_horizon": int(mpc_window.get("control_horizon", 5)),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "role_columns": role_columns,
            "target_columns": target_columns,
        }
        return cls(
            train_dataset,
            test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            summaries=summaries,
        )


def _clean_series(
    series: TabularSeries,
    *,
    missing: dict[str, Any] | None,
    outliers: dict[str, Any] | None,
) -> "_CleanedSeries":
    summaries: dict[str, Any] = {}
    current = series
    if missing is not None:
        missing_result = MissingValueProcessor(**missing).fit_transform(current)
        current = missing_result.data
        summaries["missing_summary"] = missing_result.summary
    if outliers is not None:
        config = OutlierConfig(**outliers)
        outlier_result = OutlierProcessor(config).fit_transform(current)
        current = outlier_result.data
        summaries["outlier_summary"] = outlier_result.summary
        summaries["outlier_removed_indices"] = outlier_result.removed_indices.tolist()
    return _CleanedSeries(data=current, summaries=summaries)


def _apply_outliers_train_test(
    train: TabularSeries,
    test: TabularSeries,
    config: dict[str, Any],
) -> tuple[TabularSeries, TabularSeries, dict[str, Any]]:
    outlier_config = OutlierConfig(**config)
    processor = OutlierProcessor(outlier_config).fit(train)
    train_result = (
        processor.fit_transform(train)
        if outlier_config.remove_train
        else processor.transform(train, remove=False)
    )
    if outlier_config.remove_test:
        test_result = processor.transform(test, remove=True)
    else:
        test_result = processor.transform(test, remove=False)
    summaries: dict[str, Any] = {
        "outlier_summary": {
            "policy": {
                "fit_scope": "train",
                "remove_train": outlier_config.remove_train,
                "remove_test": outlier_config.remove_test,
                "mark_test": outlier_config.mark_test,
            },
            "train": train_result.summary,
            "test": test_result.summary,
        },
        "outlier_removed_indices": train_result.removed_indices.tolist(),
    }
    if outlier_config.remove_test:
        summaries["test_outlier_removed_indices"] = test_result.removed_indices.tolist()
    return train_result.data, test_result.data, summaries


def _series_take(series: TabularSeries, indices: np.ndarray) -> TabularSeries:
    mask = np.zeros(series.row_count, dtype=bool)
    mask[np.asarray(indices, dtype=int)] = True
    return series.select_rows(mask)


def _resolve_sequence_config(config: dict[str, Any]) -> dict[str, Any]:
    input_length = _first_int(
        config,
        ("input_length", "seq_len", "sequence_length", "lookback", "history", "n"),
        default=3,
    )
    target_length = _first_optional_int(
        config,
        ("target_length", "output_length", "future_steps", "horizon", "m"),
    )
    return {
        "input_length": input_length,
        "target_length": target_length,
        "task": str(config.get("task", config.get("mode", "n_to_1"))),
        "target_offset": int(config.get("target_offset", 0)),
        "stride": int(config.get("stride", 1)),
        "return_mode": str(config.get("return_mode", "dict")),
        "squeeze_single_target": bool(config.get("squeeze_single_target", True)),
        "target_from_input": bool(config.get("target_from_input", config.get("autoregressive", False))),
    }


def _sequence_shuffle(config: dict[str, Any], *, fallback: bool) -> bool:
    if "shuffle_train" in config:
        return bool(config["shuffle_train"])
    if "shuffle" in config:
        return bool(config["shuffle"])
    if bool(config.get("preserve_order", True)):
        return False
    return bool(fallback)


def _sequence_split_indices(
    row_count: int,
    *,
    split: dict[str, Any] | None,
    test_ratio: float,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = {"type": "sequential", "test_ratio": test_ratio, "seed": seed}
    config.update(split or {})
    split_type = _split_type(config, default="sequential")
    if split_type in {"random", "stratified"}:
        raise ValueError(
            f"sequence split type {split_type!r} would break temporal continuity. "
            "Legal options are: sequential, group, episode, segment."
        )
    return _split_indices(
        row_count,
        split=config,
        test_ratio=test_ratio,
        seed=seed,
        groups=groups,
    )


def _normalize_sequence_train_test(
    train: TabularSeries,
    test: TabularSeries,
    config: dict[str, Any],
) -> tuple[TabularSeries, TabularSeries, dict[str, Any]]:
    if train.y is None or test.y is None:
        raise ValueError("sequence normalization requires y to be present.")
    normalizer = Normalizer(**config)
    train_matrix = np.concatenate([train.x, train.y], axis=1)
    test_matrix = np.concatenate([test.x, test.y], axis=1)
    train_scaled, test_scaled = normalizer.fit_transform_train_test(train_matrix, test_matrix)
    x_width = train.x.shape[1]
    return (
        _series_with_xy(train, train_scaled[:, :x_width], train_scaled[:, x_width:]),
        _series_with_xy(test, test_scaled[:, :x_width], test_scaled[:, x_width:]),
        normalizer.summary(),
    )


def _series_with_xy(series: TabularSeries, x: np.ndarray, y: np.ndarray) -> TabularSeries:
    return TabularSeries(
        x=x,
        y=y,
        u=series.u,
        quality=y,
        labels=series.labels,
        column_names=series.column_names,
        index=series.index,
    )


def _sequence_dataset_from_series(
    series: TabularSeries,
    config: dict[str, Any],
    *,
    segment_ids: np.ndarray | None,
) -> SequenceDataset:
    if series.y is None:
        raise ValueError("sequence dataset requires y to be present.")
    return SequenceDataset(
        series.x,
        series.y,
        input_length=int(config["input_length"]),
        target_length=config["target_length"],
        task=str(config["task"]),
        target_offset=int(config["target_offset"]),
        stride=int(config["stride"]),
        segment_ids=segment_ids,
        return_mode=str(config["return_mode"]),
        squeeze_single_target=bool(config["squeeze_single_target"]),
    )


def _require_nonempty_sequence_dataset(dataset: SequenceDataset, *, split: str) -> None:
    if len(dataset) == 0:
        summary = dataset.summary()
        raise ValueError(
            f"sequence {split} split produced no samples. Current rows are insufficient for "
            f"input_length={summary['input_length']}, target_length={summary['target_length']}, "
            f"target_offset={summary['target_offset']}."
        )


def _groups_for_series(groups: np.ndarray | None, series: TabularSeries) -> np.ndarray | None:
    if groups is None:
        return None
    indices = np.asarray(series.index, dtype=int)
    return np.asarray(groups)[indices]


def _first_int(config: dict[str, Any], keys: tuple[str, ...], *, default: int) -> int:
    value = _first_value(config, keys, default=default)
    return int(value)


def _first_optional_int(config: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    value = _first_value(config, keys, default=None)
    return None if value is None else int(value)


def _first_value(config: dict[str, Any], keys: tuple[str, ...], *, default: Any) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return default


def _concat_split(canonical: CanonicalDataset, split: str) -> pd.DataFrame:
    frame = _optional_concat_split(canonical, split)
    if frame is None:
        legal = ", ".join(sorted(canonical.splits))
        raise ValueError(
            f"Dataset preset did not provide required split {split!r}. "
            f"Available splits are: {legal}."
        )
    return frame


def _optional_concat_split(canonical: CanonicalDataset, split: str) -> pd.DataFrame | None:
    segments = canonical.splits.get(split)
    if not segments:
        return None
    frames = [segment.frame for segment in segments]
    return pd.concat(frames, axis=0, ignore_index=True)


def _frame_to_task_arrays(
    frame: pd.DataFrame,
    schema: DataSchema,
    task: TaskSchema,
) -> tuple[np.ndarray, np.ndarray | None]:
    target_columns = _target_columns(frame, schema, task)
    input_columns = _input_columns(frame, schema, task, target_columns)
    if not input_columns:
        raise ValueError(
            f"Task {task.name!r} did not resolve any input columns. "
            f"Legal frame columns are: {', '.join(map(str, frame.columns))}."
        )
    x = _numeric_frame(frame, input_columns, field_name="input").to_numpy(dtype=float)
    if not target_columns:
        return x, None
    y = _numeric_frame(frame, target_columns, field_name="target").to_numpy(dtype=float)
    return x, y


def _frame_group_array(frame: pd.DataFrame, schema: DataSchema) -> np.ndarray | None:
    for role in ("group", "episode", "segment"):
        columns = [column for column in schema.role_columns(role) if column in frame.columns]
        if columns:
            return frame.loc[:, columns[0]].to_numpy()
    return None


def _mpc_role_columns(
    frame: pd.DataFrame,
    schema: DataSchema,
    task: TaskSchema,
) -> dict[str, list[str]]:
    role_columns = {
        role: [column for column in schema.role_columns(role) if column in frame.columns]
        for role in ("state", "output", "control", "reference")
    }
    if task.inputs:
        selected = set(_resolve_selectors(frame, schema, task.inputs))
        role_columns = {
            role: [column for column in columns if column in selected]
            for role, columns in role_columns.items()
        }
    if not any(role_columns.values()):
        raise ValueError(
            "MPC task requires at least one input column with role state, output, "
            "control, or reference."
        )
    return role_columns


def _mpc_target_columns(
    frame: pd.DataFrame,
    schema: DataSchema,
    task: TaskSchema,
) -> list[str]:
    if task.targets:
        return _resolve_selectors(frame, schema, task.targets)
    for role in ("target", "output", "state", "quality"):
        columns = [column for column in schema.role_columns(role) if column in frame.columns]
        if columns:
            return columns
    raise ValueError("MPC task requires target columns or output/state schema roles.")


def _mpc_dataset_from_frame(
    frame: pd.DataFrame,
    schema: DataSchema,
    task: TaskSchema,
    config: dict[str, Any],
) -> MPCWindowDataset:
    role_columns = _mpc_role_columns(frame, schema, task)
    target_columns = _mpc_target_columns(frame, schema, task)
    return MPCWindowDataset(
        state=_optional_numeric_frame(frame, role_columns["state"], field_name="MPC state"),
        output=_optional_numeric_frame(frame, role_columns["output"], field_name="MPC output"),
        control=_optional_numeric_frame(frame, role_columns["control"], field_name="MPC control"),
        reference=_optional_numeric_frame(frame, role_columns["reference"], field_name="MPC reference"),
        target=_optional_numeric_frame(frame, target_columns, field_name="MPC target"),
        episode_ids=_frame_group_array(frame, schema),
        past_horizon=int(config.get("past_horizon", 10)),
        prediction_horizon=int(config.get("prediction_horizon", 20)),
        control_horizon=int(config.get("control_horizon", 5)),
        return_mode=str(config.get("return_mode", "dict")),
    )


def _clean_frame_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    missing: dict[str, Any] | None,
    outliers: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    series = TabularSeries(
        x=_numeric_frame(frame, columns, field_name="MPC cleanup").to_numpy(dtype=float),
        index=np.arange(frame.shape[0]),
    )
    summaries: dict[str, Any] = {}
    current = series
    if missing is not None:
        missing_result = MissingValueProcessor(**missing).fit_transform(current)
        current = missing_result.data
        summaries["missing_summary"] = missing_result.summary
    if outliers is not None:
        outlier_result = OutlierProcessor(OutlierConfig(**outliers)).fit_transform(current)
        current = outlier_result.data
        summaries["outlier_summary"] = outlier_result.summary
        summaries["outlier_removed_indices"] = outlier_result.removed_indices.tolist()
    cleaned = frame.iloc[current.index.astype(int)].reset_index(drop=True).copy()
    cleaned.loc[:, columns] = current.x
    return cleaned, summaries


def _optional_numeric_frame(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    field_name: str,
) -> np.ndarray | None:
    if not columns:
        return None
    return _numeric_frame(frame, columns, field_name=field_name).to_numpy(dtype=float)


def _unique_columns(columns: list[str]) -> list[str]:
    output: list[str] = []
    for column in columns:
        if column not in output:
            output.append(column)
    return output


def _target_columns(frame: pd.DataFrame, schema: DataSchema, task: TaskSchema) -> list[str]:
    if task.label_column is not None:
        _require_columns(frame, [task.label_column], selector="label_column")
        return [task.label_column]
    columns = _resolve_selectors(frame, schema, task.targets)
    if columns:
        return columns
    for role in ("target", "fault_id", "label", "quality"):
        role_columns = [column for column in schema.role_columns(role) if column in frame.columns]
        if role_columns:
            return role_columns
    return []


def _input_columns(
    frame: pd.DataFrame,
    schema: DataSchema,
    task: TaskSchema,
    target_columns: list[str],
) -> list[str]:
    columns = _resolve_selectors(frame, schema, task.inputs)
    if columns:
        return columns
    excluded_roles = {"time", "group", "episode", "segment", "mask", "label", "fault_id", "target"}
    excluded = set(target_columns)
    for column in schema.columns:
        if column.role in excluded_roles:
            excluded.add(column.name)
    return [str(column) for column in frame.columns if str(column) not in excluded]


def _resolve_selectors(
    frame: pd.DataFrame,
    schema: DataSchema,
    selectors: tuple[str, ...],
) -> list[str]:
    resolved: list[str] = []
    if not selectors:
        return resolved
    frame_columns = {str(column) for column in frame.columns}
    roles = set(schema.roles)
    for selector in selectors:
        if selector in roles:
            role_candidates = list(schema.role_columns(selector))
            candidates = [column for column in role_candidates if column in frame_columns]
            if not candidates:
                _require_columns(frame, role_candidates, selector=selector)
        else:
            candidates = [selector]
            _require_columns(frame, candidates, selector=selector)
        for column in candidates:
            if column not in resolved:
                resolved.append(column)
    missing = [column for column in resolved if column not in frame_columns]
    if missing:
        raise ValueError(
            f"Resolved columns are missing from frame: {', '.join(missing)}. "
            f"Legal frame columns are: {', '.join(sorted(frame_columns))}."
        )
    return resolved


def _require_columns(frame: pd.DataFrame, columns: list[str], *, selector: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        legal = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Column selector {selector!r} resolved missing columns: {', '.join(missing)}. "
            f"Legal frame columns are: {legal}."
        )


def _numeric_frame(frame: pd.DataFrame, columns: list[str], *, field_name: str) -> pd.DataFrame:
    selected = frame.loc[:, columns]
    try:
        return selected.apply(pd.to_numeric)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Preset {field_name} columns must be numeric for tensorization. "
            f"Current columns: {', '.join(columns)}."
        ) from exc


def _normalize_pipeline_config(
    pipeline: dict[str, Any] | list[Any] | str | Path | DataPipeline | None,
) -> dict[str, Any]:
    return DataPipeline.from_config(pipeline).to_config()


_TASK_ALIASES = {
    "cls": "classification",
    "class": "classification",
    "classification": "classification",
    "fd": "fault_diagnosis",
    "fault": "fault_diagnosis",
    "fault_detection": "fault_diagnosis",
    "fault_diagnosis": "fault_diagnosis",
    "fe": "reconstruction",
    "fault_estimation": "reconstruction",
    "impute": "imputation",
    "imputation": "imputation",
    "mpc": "mpc",
    "pred": "prediction",
    "prd": "prediction",
    "prediction": "prediction",
    "recon": "reconstruction",
    "reconstruction": "reconstruction",
}


_DATASET_ROOT_ALIASES = {
    "cstr": "CSTR",
    "te": "TE",
    "tts": "TTS",
    "ne": "NE",
    "wpt": "WPT",
    "hy": "HY",
    "hy_prd": "HY_PRD",
    "hy-prd": "HY_PRD",
    "multiphase": "Multiphase_Flow_Facility",
    "multiphase_flow_facility": "Multiphase_Flow_Facility",
    "multiphase-flow-facility": "Multiphase_Flow_Facility",
    "mff": "Multiphase_Flow_Facility",
}


def _normalize_task_alias(task: str | None) -> str | None:
    if task is None:
        return None
    normalized = str(task).strip().lower().replace("-", "_").replace("/", "_")
    return _TASK_ALIASES.get(normalized, normalized)


def _resolve_dataset_root_alias(root: str | Path | None) -> Path | None:
    if root is None:
        return None
    raw = str(root).strip()
    if not raw:
        return Path(raw)
    access = "oa"
    lowered = raw.lower()
    for prefix, prefix_access in (
        ("private:", "private"),
        ("priv:", "private"),
        ("oa:", "oa"),
        ("open:", "oa"),
        ("public:", "oa"),
    ):
        if lowered.startswith(prefix):
            access = prefix_access
            raw = raw[len(prefix) :].strip()
            lowered = raw.lower()
            break
    if raw.startswith("*"):
        access = "private"
        raw = raw[1:].strip()
    elif raw.endswith("*"):
        access = "private"
        raw = raw[:-1].strip()

    direct = Path(raw).expanduser()
    if direct.exists() or direct.is_absolute() or raw.startswith((".", "~")):
        return direct

    normalized_parts = [part for part in raw.replace("\\", "/").split("/") if part]
    if not normalized_parts:
        return direct
    first = normalized_parts[0].strip().lower()
    mapped_first = _DATASET_ROOT_ALIASES.get(first)
    if mapped_first is None:
        return direct

    relative_root = Path(mapped_first, *normalized_parts[1:])
    candidates = [
        Path.cwd() / "datasets" / "raw" / access / relative_root,
        _project_root() / "datasets" / "raw" / access / relative_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _merge_pipeline_configs(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    return merge_pipeline_configs(base, override)


def _apply_explicit_pipeline_overrides(
    pipeline: dict[str, Any],
    *,
    missing: dict[str, Any] | None,
    outliers: dict[str, Any] | None,
    normalization: dict[str, Any] | None,
    split: dict[str, Any] | None,
    mask: dict[str, Any] | None,
    window: dict[str, Any] | None,
    sequence: dict[str, Any] | None,
    mpc_window: dict[str, Any] | None,
) -> dict[str, Any]:
    output = dict(pipeline)
    explicit = {
        "missing": missing,
        "outliers": outliers,
        "normalization": normalization,
        "split": split,
        "mask": mask,
        "window": window,
        "sequence": sequence,
        "mpc_window": mpc_window,
    }
    for key, value in explicit.items():
        if value is not None:
            output[key] = _json_ready(value)
    return output


def _split_for_arrays(split: Any) -> dict[str, Any] | None:
    if split is None:
        return None
    if not isinstance(split, Mapping):
        raise ValueError("pipeline.split must be a mapping when provided.")
    split_config = dict(split)
    if _split_type(split_config, default="").lower() == "official":
        return None
    return split_config


class _CleanedSeries:
    def __init__(self, data: TabularSeries, summaries: dict[str, Any]) -> None:
        self.data = data
        self.summaries = summaries


def _split_indices(
    row_count: int,
    *,
    split: dict[str, Any] | None,
    test_ratio: float,
    seed: int,
    labels: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = split or {}
    split_type = _split_type(config, default="random")
    ratio = float(config.get("test_ratio", test_ratio))
    if not 0 < ratio < 1:
        raise ValueError(f"test_ratio must be between 0 and 1. Current input: {ratio}.")
    if split_type == "stratified":
        train_idx, test_idx, class_counts = _stratified_split_indices(
            row_count,
            labels=labels,
            test_ratio=ratio,
            seed=int(config.get("seed", seed)),
        )
        return (
            train_idx,
            test_idx,
            {
                "type": split_type,
                "train_samples": int(train_idx.shape[0]),
                "test_samples": int(test_idx.shape[0]),
                "test_ratio": ratio,
                "class_counts": class_counts,
            },
        )
    if split_type in {"group", "groups", "episode", "segment"}:
        train_idx, test_idx, group_counts = _group_split_indices(
            row_count,
            groups=groups,
            test_ratio=ratio,
            seed=int(config.get("seed", seed)),
        )
        return (
            train_idx,
            test_idx,
            {
                "type": split_type,
                "train_samples": int(train_idx.shape[0]),
                "test_samples": int(test_idx.shape[0]),
                "test_ratio": ratio,
                "train_groups": int(sum(1 for item in group_counts.values() if item["split"] == "train")),
                "test_groups": int(sum(1 for item in group_counts.values() if item["split"] == "test")),
                "group_counts": group_counts,
            },
        )
    test_size = max(1, int(round(row_count * ratio)))
    train_size = row_count - test_size
    if train_size <= 0:
        raise ValueError(
            f"Not enough rows for split. Legal test_ratio must leave train rows. "
            f"Current rows={row_count}, test_ratio={ratio}."
        )
    if split_type == "random":
        rng = np.random.default_rng(int(config.get("seed", seed)))
        indices = rng.permutation(row_count)
        test_idx = np.sort(indices[:test_size])
        train_idx = np.sort(indices[test_size:])
    elif split_type == "sequential":
        train_idx = np.arange(train_size)
        test_idx = np.arange(train_size, row_count)
    else:
        raise ValueError(
            f"Unknown tabular split type {split_type!r}. Legal options are: random, sequential, "
            "stratified, group, episode, segment."
        )
    return (
        train_idx,
        test_idx,
        {
            "type": split_type,
            "train_samples": int(train_idx.shape[0]),
            "test_samples": int(test_idx.shape[0]),
            "test_ratio": ratio,
        },
    )


def _stratified_split_indices(
    row_count: int,
    *,
    labels: np.ndarray | None,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    if labels is None:
        raise ValueError("stratified split requires labels/y to be present.")
    label_array = np.asarray(labels)
    if label_array.ndim == 2 and label_array.shape[1] == 1:
        label_array = label_array[:, 0]
    if label_array.ndim != 1 or label_array.shape[0] != row_count:
        raise ValueError(
            "stratified split requires one label per row. "
            f"Current label shape: {np.asarray(labels).shape}, rows={row_count}."
        )
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    class_counts: dict[str, dict[str, int]] = {}
    for label in sorted(np.unique(label_array), key=lambda item: str(item)):
        class_idx = np.flatnonzero(label_array == label)
        shuffled = rng.permutation(class_idx)
        if shuffled.size < 2:
            raise ValueError(
                "stratified split requires at least two samples per class. "
                f"Class {label!r} has {shuffled.size} sample."
            )
        test_count = int(round(shuffled.size * test_ratio))
        test_count = min(max(1, test_count), shuffled.size - 1)
        test_part = np.sort(shuffled[:test_count])
        train_part = np.sort(shuffled[test_count:])
        test_parts.append(test_part)
        train_parts.append(train_part)
        class_counts[str(label)] = {
            "train": int(train_part.size),
            "test": int(test_part.size),
        }
    train_idx = np.sort(np.concatenate(train_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("stratified split produced an empty train or test split.")
    return train_idx, test_idx, class_counts


def _group_split_indices(
    row_count: int,
    *,
    groups: np.ndarray | None,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int | str]]]:
    if groups is None:
        raise ValueError("group/episode split requires one group value per row.")
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or group_array.shape[0] != row_count:
        raise ValueError(
            "group/episode split requires one group value per row. "
            f"Current group shape: {group_array.shape}, rows={row_count}."
        )
    unique_groups = np.unique(group_array)
    if unique_groups.shape[0] < 2:
        raise ValueError(
            "group/episode split requires at least two distinct groups. "
            f"Current group count: {unique_groups.shape[0]}."
        )
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    target_test_rows = max(1, int(round(row_count * test_ratio)))
    test_groups: list[Any] = []
    test_rows = 0
    for group in shuffled_groups[:-1]:
        test_groups.append(group)
        test_rows += int((group_array == group).sum())
        if test_rows >= target_test_rows:
            break
    if not test_groups:
        test_groups = [shuffled_groups[0]]
    test_group_set = set(test_groups)
    test_mask = np.asarray([group in test_group_set for group in group_array], dtype=bool)
    train_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("group/episode split produced an empty train or test split.")
    group_counts: dict[str, dict[str, int | str]] = {}
    for group in sorted(unique_groups, key=lambda item: str(item)):
        split_name = "test" if group in test_group_set else "train"
        group_counts[str(group)] = {
            "rows": int((group_array == group).sum()),
            "split": split_name,
        }
    return train_idx, test_idx, group_counts


def _optional_group_array(value: Any | None, *, expected_rows: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.shape[0] != expected_rows:
        raise ValueError(
            "groups must provide one value per input row. "
            f"Current shape: {array.shape}, expected rows={expected_rows}."
        )
    return array


def _align_groups_to_series(groups: np.ndarray | None, series: TabularSeries) -> np.ndarray | None:
    if groups is None:
        return None
    return groups[np.asarray(series.index, dtype=int)]


def _split_type(config: dict[str, Any], *, default: str) -> str:
    return str(config.get("type", config.get("method", default))).strip().lower()


def _tensor_dataset(x: Any, y: Any | None = None) -> TensorDataset:
    if x is None:
        raise ValueError("x cannot be None when building a TensorDataset.")
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    if y is None:
        return TensorDataset(x_tensor)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    return TensorDataset(x_tensor, y_tensor)


def _optional_array(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _covered_dynamic_rows(dataset: DynamicWindowDataset, sample_indices: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    span = dataset.lookback + dataset.future_steps
    for sample_idx in sample_indices:
        start = int(dataset.sample_starts[int(sample_idx)])
        rows.extend(range(start, start + span))
    if not rows:
        raise ValueError("Cannot fit normalization because the dynamic train split is empty.")
    return np.asarray(sorted(set(rows)), dtype=int)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _summary_rows(summary: Any) -> list[dict[str, Any]]:
    ready = _json_ready(summary)
    if isinstance(ready, dict):
        if ready and all(isinstance(value, dict) for value in ready.values()):
            return [
                {"section": key, **_flat_summary(value, exclude=set())}
                for key, value in ready.items()
            ]
        return [_flat_summary(ready, exclude=set())]
    if isinstance(ready, list):
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(ready):
            if isinstance(item, dict):
                rows.append({"row": idx, **_flat_summary(item, exclude=set())})
            else:
                rows.append({"row": idx, "value": item})
        return rows
    return [{"value": ready}]


def _source_preprocessing_summary(source_summary: dict[str, Any]) -> dict[str, Any]:
    segments = []
    for segment in source_summary.get("segments", []):
        if not isinstance(segment, dict):
            continue
        preprocessing = segment.get("preprocessing")
        if isinstance(preprocessing, dict) and preprocessing.get("configured"):
            segments.append(preprocessing)
    if not segments:
        return {}
    dropped_columns = sorted(
        {
            column
            for segment in segments
            for column in segment.get("dropped_columns", [])
        }
    )
    missing_drop_columns = sorted(
        {
            column
            for segment in segments
            for column in segment.get("missing_drop_columns", [])
        }
    )
    return {
        "segments": segments,
        "segment_count": len(segments),
        "dropped_rows": int(sum(int(segment.get("dropped_rows", 0)) for segment in segments)),
        "dropped_columns": dropped_columns,
        "missing_drop_columns": missing_drop_columns,
    }


def _prepared_dataset_hash_summary(data: DataModule) -> dict[str, Any]:
    payload = {
        "train": _dataset_signature(data.train_dataset),
        "test": None if data.test_dataset is None else _dataset_signature(data.test_dataset),
        "batch_size": data.batch_size,
        "shuffle": data.shuffle,
        "summaries": _json_ready(data.summaries),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return {
        "algorithm": "sha256",
        "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload_version": 1,
        "train_samples": len(data.train_dataset),
        "test_samples": None if data.test_dataset is None else len(data.test_dataset),
    }


def _dataset_signature(dataset: Dataset) -> dict[str, Any]:
    length = len(dataset)
    sample_indices = sorted({0, max(0, length // 2), max(0, length - 1)}) if length else []
    return {
        "type": dataset.__class__.__name__,
        "length": length,
        "samples": [_sample_signature(dataset[index]) for index in sample_indices],
    }


def _sample_signature(sample: Any) -> Any:
    if isinstance(sample, torch.Tensor):
        tensor = sample.detach().cpu()
        return {
            "type": "tensor",
            "shape": list(tensor.shape),
            "sum": float(tensor.sum().item()) if tensor.numel() else 0.0,
            "mean": float(tensor.float().mean().item()) if tensor.numel() else 0.0,
        }
    if isinstance(sample, dict):
        return {str(key): _sample_signature(value) for key, value in sorted(sample.items())}
    if isinstance(sample, (tuple, list)):
        return [_sample_signature(item) for item in sample]
    if isinstance(sample, np.ndarray):
        return _sample_signature(torch.as_tensor(sample))
    if isinstance(sample, np.generic):
        return sample.item()
    return sample


def _flat_summary(summary: Any, *, exclude: set[str]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {"value": summary}
    flattened: dict[str, Any] = {}
    for key, value in summary.items():
        if key in exclude:
            continue
        if isinstance(value, (dict, list, tuple, np.ndarray)):
            flattened[key] = str(_json_ready(value))
        else:
            flattened[key] = _json_ready(value)
    return flattened
