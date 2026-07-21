"""Dataset adapter protocol and dataset-card implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import glob
import numpy as np
import pandas as pd
import yaml

from joff.data.schema import DataSchema, SegmentInfo, TaskSchema
from joff.data.sources import read_source_frame


@dataclass(frozen=True)
class Segment:
    """One canonical dataset segment."""

    frame: pd.DataFrame
    meta: SegmentInfo

    def summary(self) -> dict[str, Any]:
        """Return a serializable segment summary."""

        return self.meta.summary()


@dataclass(frozen=True)
class CanonicalDataset:
    """Dataset split into canonical segments with schema and metadata."""

    splits: dict[str, tuple[Segment, ...]]
    schema: DataSchema
    metadata: dict[str, Any] = field(default_factory=dict)

    def split_rows(self) -> dict[str, int]:
        """Return row counts by split."""

        return {
            split: int(sum(segment.frame.shape[0] for segment in segments))
            for split, segments in self.splits.items()
        }

    def source_summary(self) -> dict[str, Any]:
        """Return a serializable provenance summary."""

        data = dict(self.metadata)
        data["split_rows"] = self.split_rows()
        data["segments"] = [
            segment.summary()
            for segments in self.splits.values()
            for segment in segments
        ]
        return data


class DatasetAdapter(Protocol):
    """Adapter interface for dataset-specific reading and metadata."""

    name: str
    version: str
    description: str

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        """Read raw files and return canonical split segments."""

    def schema(self) -> DataSchema:
        """Return the dataset schema."""

    def default_task(self, task: str | None = None) -> TaskSchema:
        """Return task metadata for ``task``."""

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """Return adapter-level default pipeline config."""

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """Return a serializable adapter summary."""


@dataclass(frozen=True)
class DatasetPreset:
    """Dataset-card content normalized for an adapter."""

    name: str
    version: str = "unknown"
    description: str = ""
    files: dict[str, Any] = field(default_factory=dict)
    schema: DataSchema = field(default_factory=DataSchema)
    tasks: dict[str, TaskSchema] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        default_name: str = "dataset",
    ) -> "DatasetPreset":
        """Build a normalized preset from dataset-card YAML data."""

        known = {
            "name",
            "version",
            "description",
            "files",
            "schema",
            "tasks",
            "pipeline",
            "preprocessing",
        }
        tasks_config = config.get("tasks", {}) or {}
        if not isinstance(tasks_config, Mapping):
            raise TypeError(
                "Dataset card key 'tasks' must be a mapping. "
                f"Current input type: {type(tasks_config).__name__}."
            )
        tasks = {
            str(name): TaskSchema.from_config(str(name), dict(value or {}))
            for name, value in tasks_config.items()
        }
        files = config.get("files", {}) or {}
        if not isinstance(files, Mapping):
            raise TypeError(
                "Dataset card key 'files' must be a mapping. "
                f"Current input type: {type(files).__name__}."
            )
        pipeline = config.get("pipeline", {}) or {}
        if not isinstance(pipeline, Mapping):
            raise TypeError(
                "Dataset card key 'pipeline' must be a mapping. "
                f"Current input type: {type(pipeline).__name__}."
            )
        preprocessing = config.get("preprocessing", {}) or {}
        if not isinstance(preprocessing, Mapping):
            raise TypeError(
                "Dataset card key 'preprocessing' must be a mapping. "
                f"Current input type: {type(preprocessing).__name__}."
            )
        return cls(
            name=str(config.get("name", default_name)),
            version=str(config.get("version", "unknown")),
            description=str(config.get("description", "")),
            files=dict(files),
            schema=DataSchema.from_config(config.get("schema")),
            tasks=tasks,
            pipeline=dict(pipeline),
            preprocessing=dict(preprocessing),
            metadata={str(key): value for key, value in config.items() if key not in known},
        )

    def task(self, task: str | None = None) -> TaskSchema:
        """Return the requested task, or a deterministic default task."""

        if not self.tasks:
            return TaskSchema(name=task or "default")
        if task is None:
            if "prediction" in self.tasks:
                return self.tasks["prediction"]
            if "fault_diagnosis" in self.tasks:
                return self.tasks["fault_diagnosis"]
            return self.tasks[sorted(self.tasks)[0]]
        if task not in self.tasks:
            legal = ", ".join(sorted(self.tasks))
            raise ValueError(
                f"Unknown task {task!r} for dataset preset {self.name!r}. "
                f"Legal tasks are: {legal}. Current input was: {task!r}."
            )
        return self.tasks[task]

    def task_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """Return preset-level pipeline merged with task-level pipeline."""

        merged = _deep_copy(self.pipeline)
        task_name = self.task(task).name
        raw_task = self.metadata.get("_raw_tasks", {}).get(task_name, {})
        if isinstance(raw_task, Mapping) and isinstance(raw_task.get("pipeline"), Mapping):
            _deep_merge(merged, dict(raw_task["pipeline"]))
        return merged

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """Return a serializable preset summary."""

        task_schema = self.task(task)
        summary = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "task": task_schema.summary(),
            "tasks": sorted(self.tasks),
            "files": _json_ready(self.files),
        }
        if self.metadata:
            summary["metadata"] = _json_ready(
                {key: value for key, value in self.metadata.items() if key != "_raw_tasks"}
            )
            if "access" in self.metadata:
                summary["access"] = _json_ready(self.metadata["access"])
        if self.preprocessing:
            summary["preprocessing"] = _json_ready(self.preprocessing)
        return summary


class DatasetCardAdapter:
    """Adapter backed by a local ``dataset_card.yaml`` file or mapping."""

    def __init__(
        self,
        preset: DatasetPreset,
        *,
        card_path: str | Path | None = None,
        raw_card: Mapping[str, Any] | None = None,
    ) -> None:
        self.preset = preset
        self.card_path = None if card_path is None else Path(card_path)
        self.raw_card = dict(raw_card or {})
        self.name = preset.name
        self.version = preset.version
        self.description = preset.description

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DatasetCardAdapter":
        """Load a dataset-card adapter from YAML."""

        card_path = Path(path)
        if not card_path.exists():
            raise FileNotFoundError(f"Dataset card does not exist: {card_path}")
        data = yaml.safe_load(card_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ValueError(
                f"Dataset card {card_path} must contain a YAML mapping. "
                f"Current input type: {type(data).__name__}."
            )
        raw = dict(data)
        preset = DatasetPreset.from_config(raw, default_name=card_path.stem)
        preset = _with_raw_tasks(preset, raw.get("tasks", {}) or {})
        return cls(preset, card_path=card_path, raw_card=raw)

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        card_path: str | Path | None = None,
    ) -> "DatasetCardAdapter":
        """Build a dataset-card adapter from a mapping."""

        preset = DatasetPreset.from_config(config)
        preset = _with_raw_tasks(preset, config.get("tasks", {}) or {})
        return cls(preset, card_path=card_path, raw_card=config)

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        """Read split files declared by the dataset card."""

        root_path = self._resolve_root(root)
        split_segments: dict[str, tuple[Segment, ...]] = {}
        for split in ("train", "eval", "test", "data"):
            patterns = self.preset.files.get(split)
            if patterns is None:
                continue
            paths = _resolve_patterns(root_path, patterns)
            segments = tuple(
                _read_segment(path, split, preprocessing=self.preset.preprocessing)
                for path in paths
            )
            if segments:
                target_split = "train" if split == "data" else split
                split_segments[target_split] = split_segments.get(target_split, ()) + segments
        if not split_segments:
            legal = ", ".join(k for k in ("train", "eval", "test", "data"))
            raise ValueError(
                f"Dataset preset {self.name!r} did not resolve any files under {root_path}. "
                f"Dataset-card 'files' must contain at least one of: {legal}."
            )
        return CanonicalDataset(
            splits=split_segments,
            schema=self.preset.schema,
            metadata={
                "source_type": "dataset_card",
                "preset": self.name,
                "version": self.version,
                "root": str(root_path),
                "card_path": None if self.card_path is None else str(self.card_path),
                "preprocessing": _json_ready(self.preset.preprocessing),
            },
        )

    def schema(self) -> DataSchema:
        """Return the dataset schema."""

        return self.preset.schema

    def default_task(self, task: str | None = None) -> TaskSchema:
        """Return the default task schema."""

        return self.preset.task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """Return default preprocessing pipeline config."""

        return self.preset.task_pipeline(task)

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """Return a serializable adapter summary."""

        return self.preset.summary(task)

    def _resolve_root(self, root: str | Path | None) -> Path:
        if root is not None:
            return Path(root)
        raw_root = self.preset.files.get("root")
        if raw_root is None:
            return self.card_path.parent if self.card_path is not None else Path(".")
        root_path = Path(raw_root)
        if root_path.is_absolute():
            return root_path
        if self.card_path is not None:
            return self.card_path.parent / root_path
        return root_path


def _resolve_patterns(root: Path, patterns: str | Path | list[str | Path]) -> list[Path]:
    raw_patterns: list[str | Path]
    if isinstance(patterns, (str, Path)):
        raw_patterns = [patterns]
    else:
        raw_patterns = list(patterns)
    resolved: list[Path] = []
    for raw_pattern in raw_patterns:
        pattern = Path(raw_pattern)
        search = str(pattern if pattern.is_absolute() else root / pattern)
        matches = sorted(Path(path) for path in glob.glob(search))
        if not matches and not any(marker in search for marker in ("*", "?", "[")):
            direct = Path(search)
            if direct.exists():
                matches = [direct]
        if not matches:
            raise FileNotFoundError(
                f"Dataset file pattern did not match any files: {search}. "
                "Legal dataset-card file patterns must resolve under the preset root."
            )
        resolved.extend(matches)
    return resolved


def _read_segment(path: Path, split: str, *, preprocessing: Mapping[str, Any] | None = None) -> Segment:
    frame = read_source_frame(path)
    frame, preprocessing_summary = _apply_preprocessing(frame, preprocessing or {}, split=split)
    metadata = {}
    if preprocessing_summary["configured"]:
        metadata["preprocessing"] = preprocessing_summary
    return Segment(
        frame=frame,
        meta=SegmentInfo(
            split=split,
            source=str(path),
            rows=int(frame.shape[0]),
            metadata=metadata,
        ),
    )


def _with_raw_tasks(preset: DatasetPreset, raw_tasks: Any) -> DatasetPreset:
    if not isinstance(raw_tasks, Mapping):
        return preset
    metadata = dict(preset.metadata)
    metadata["_raw_tasks"] = {str(key): dict(value or {}) for key, value in raw_tasks.items()}
    return DatasetPreset(
        name=preset.name,
        version=preset.version,
        description=preset.description,
        files=preset.files,
        schema=preset.schema,
        tasks=preset.tasks,
        pipeline=preset.pipeline,
        preprocessing=preset.preprocessing,
        metadata=metadata,
    )


def _apply_preprocessing(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    original_rows = int(output.shape[0])
    drop_columns, missing_columns = _resolve_drop_columns(output, config)
    if drop_columns:
        output = output.drop(columns=drop_columns)
    label_filters = _label_filters(config)
    label_filter_summaries: list[dict[str, Any]] = []
    total_dropped_rows = 0
    for column, values in label_filters.items():
        if column not in output.columns:
            label_filter_summaries.append(
                {
                    "column": column,
                    "values": _json_ready(values),
                    "missing_column": True,
                    "dropped_rows": 0,
                }
            )
            continue
        drop_mask = _category_mask(output[column], values)
        dropped_rows = int(drop_mask.sum())
        total_dropped_rows += dropped_rows
        if dropped_rows:
            output = output.loc[~drop_mask].reset_index(drop=True)
        label_filter_summaries.append(
            {
                "column": column,
                "values": _json_ready(values),
                "missing_column": False,
                "dropped_rows": dropped_rows,
            }
        )
    summary = {
        "configured": bool(config),
        "split": split,
        "original_rows": original_rows,
        "rows": int(output.shape[0]),
        "dropped_rows": total_dropped_rows,
        "dropped_columns": drop_columns,
        "missing_drop_columns": missing_columns,
        "label_filters": label_filter_summaries,
    }
    return output, summary


def _resolve_drop_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    raw = (
        config.get("drop_columns")
        or config.get("exclude_columns")
        or config.get("del_dim")
        or config.get("drop_dims")
        or ()
    )
    requested = _value_list(raw)
    columns: list[str] = []
    missing: list[str] = []
    frame_columns = list(frame.columns)
    for item in requested:
        if isinstance(item, int) and not isinstance(item, bool):
            if 0 <= item < len(frame_columns):
                column = str(frame_columns[item])
                if column not in columns:
                    columns.append(column)
            else:
                missing.append(str(item))
            continue
        column = str(item)
        if column in frame.columns:
            if column not in columns:
                columns.append(column)
        else:
            missing.append(column)
    return columns, missing


def _label_filters(config: Mapping[str, Any]) -> dict[str, list[Any]]:
    raw = (
        config.get("drop_label_values")
        or config.get("drop_categories")
        or config.get("drop_cate")
        or config.get("del_cate")
        or {}
    )
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            "Dataset-card preprocessing drop_label_values must be a mapping of column -> values. "
            f"Current input type: {type(raw).__name__}."
        )
    return {str(column): _value_list(values) for column, values in raw.items()}


def _category_mask(values: pd.Series, categories: list[Any]) -> pd.Series:
    string_values = {str(value) for value in categories}
    return values.isin(categories) | values.map(lambda item: str(item)).isin(string_values)


def _value_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        return [value]
    return list(value)


def _deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready(item) for key, item in value.items()}


def _deep_merge(target: dict[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in layer.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = _json_ready(value)
    return target


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
