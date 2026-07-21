"""Task-specific views over schema-backed tabular frames."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from joff.data.schema import DataSchema, TaskSchema


@dataclass(frozen=True)
class TaskView:
    """Resolved input/target columns for one task on one canonical dataset."""

    name: str
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...] = ()
    label_column: str | None = None
    target_policy: str = "resolved"
    normal_label: Any | None = None
    fault_switch: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_schema(
        cls,
        frame: pd.DataFrame,
        schema: DataSchema,
        task: TaskSchema,
    ) -> "TaskView":
        """Resolve task selectors against a frame and schema."""

        target_columns, target_policy = _target_columns(frame, schema, task)
        input_columns = _input_columns(frame, schema, task, target_columns)
        if not input_columns:
            raise ValueError(
                f"Task {task.name!r} did not resolve any input columns. "
                f"Legal frame columns are: {', '.join(map(str, frame.columns))}."
            )
        if _task_kind(task.name) == "reconstruction" and not target_columns:
            target_columns = input_columns
            target_policy = "reconstruction_input"
        return cls(
            name=task.name,
            input_columns=tuple(input_columns),
            target_columns=tuple(target_columns),
            label_column=task.label_column,
            target_policy=target_policy,
            normal_label=task.normal_label,
            fault_switch=task.fault_switch,
            metadata=dict(task.metadata),
        )

    def arrays(
        self,
        frame: pd.DataFrame,
        *,
        label_mapping: Mapping[str, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return numeric input and target arrays for this view."""

        x = _numeric_frame(frame, list(self.input_columns), field_name="input").to_numpy(dtype=float)
        if not self.target_columns:
            return x, None
        if label_mapping is not None:
            return x, _encoded_labels(frame, self.target_columns[0], label_mapping)
        y = _numeric_frame(frame, list(self.target_columns), field_name="target").to_numpy(dtype=float)
        return x, y

    def label_mapping(self, frames: Mapping[str, pd.DataFrame]) -> dict[str, int] | None:
        """Return a stable string-label mapping when this task needs encoding."""

        if not self._is_label_like() or len(self.target_columns) != 1:
            return None
        values = _concat_label_values(frames, self.target_columns[0])
        if _all_numeric(values):
            return None
        labels = sorted({str(value) for value in values.dropna().tolist()})
        if self.normal_label is not None:
            normal_key = str(self.normal_label)
            labels = [normal_key, *[label for label in labels if label != normal_key]]
        return {label: idx for idx, label in enumerate(labels)}

    def label_summary(
        self,
        frames: Mapping[str, pd.DataFrame],
        *,
        label_mapping: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        """Return label counts and optional string-to-index mapping."""

        if not self._is_label_like() or len(self.target_columns) != 1:
            return {}
        column = self.target_columns[0]
        data: dict[str, Any] = {
            "task": self.name,
            "column": column,
            "counts": {
                split: _label_counts(frame[column])
                for split, frame in frames.items()
                if column in frame.columns
            },
        }
        if label_mapping is not None:
            data["mapping"] = dict(label_mapping)
        return data

    def summary(self, frame: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return a serializable task-view summary."""

        data: dict[str, Any] = {
            "name": self.name,
            "kind": _task_kind(self.name),
            "input_columns": list(self.input_columns),
            "target_columns": list(self.target_columns),
            "target_policy": self.target_policy,
            "input_dim": len(self.input_columns),
            "target_dim": len(self.target_columns),
            "has_target": bool(self.target_columns),
        }
        if frame is not None:
            data["rows"] = int(frame.shape[0])
        if self.label_column is not None:
            data["label_column"] = self.label_column
        if self.normal_label is not None:
            data["normal_label"] = self.normal_label
        if self.fault_switch is not None:
            data["fault_switch"] = self.fault_switch
        data.update(self.metadata)
        return data

    def _is_label_like(self) -> bool:
        kind = _task_kind(self.name)
        return self.label_column is not None or kind in {"classification", "fault_diagnosis"}


def _target_columns(
    frame: pd.DataFrame,
    schema: DataSchema,
    task: TaskSchema,
) -> tuple[list[str], str]:
    if task.label_column is not None:
        _require_columns(frame, [task.label_column], selector="label_column")
        return [task.label_column], "label_column"
    columns = _resolve_selectors(frame, schema, task.targets)
    if columns:
        return columns, "explicit"
    kind = _task_kind(task.name)
    if kind == "mpc":
        fallback_roles = ("target", "output", "state", "quality")
    elif kind == "prediction":
        fallback_roles = ("target", "quality", "output")
    elif kind in {"imputation", "reconstruction"}:
        fallback_roles = ()
    elif kind in {"classification", "fault_diagnosis"}:
        fallback_roles = ("fault_id", "label", "target")
    else:
        fallback_roles = ("target", "fault_id", "label", "quality")
    for role in fallback_roles:
        role_columns = [column for column in schema.role_columns(role) if column in frame.columns]
        if role_columns:
            return role_columns, f"role:{role}"
    return [], "none"


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


def _encoded_labels(
    frame: pd.DataFrame,
    column: str,
    label_mapping: Mapping[str, int],
) -> np.ndarray:
    values = frame[column].map(lambda item: str(item))
    unknown = sorted(set(values.dropna().tolist()) - set(label_mapping))
    if unknown:
        legal = ", ".join(sorted(label_mapping))
        raise ValueError(
            f"Target column {column!r} contains labels not present in the label mapping: "
            f"{', '.join(unknown)}. Legal labels are: {legal}."
        )
    return values.map(lambda item: label_mapping[item]).to_numpy(dtype=float)[:, None]


def _concat_label_values(frames: Mapping[str, pd.DataFrame], column: str) -> pd.Series:
    values = [frame[column] for frame in frames.values() if column in frame.columns]
    if not values:
        return pd.Series([], dtype=object)
    return pd.concat(values, axis=0, ignore_index=True)


def _all_numeric(values: pd.Series) -> bool:
    if values.empty:
        return True
    numeric = pd.to_numeric(values.dropna(), errors="coerce")
    return bool(numeric.notna().all())


def _label_counts(values: pd.Series) -> dict[str, int]:
    counts = values.map(lambda item: str(item)).value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.sort_index().items()}


def _task_kind(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "cls": "classification",
        "fd": "fault_diagnosis",
        "fault_detection": "fault_diagnosis",
        "fault_diagnosis": "fault_diagnosis",
        "impute": "imputation",
        "mpc": "mpc",
        "prediction": "prediction",
        "reconstruction": "reconstruction",
    }
    return aliases.get(normalized, normalized)
