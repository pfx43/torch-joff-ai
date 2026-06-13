"""Dataset schema containers used by dataset adapters and presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ColumnSpec:
    """One dataset column and its semantic role."""

    name: str
    role: str = "input"
    dtype: str | None = None
    unit: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: str | dict[str, Any]) -> "ColumnSpec":
        """Build a column spec from a dataset-card entry."""

        if isinstance(config, str):
            return cls(name=config)
        if not isinstance(config, dict):
            raise TypeError(
                "ColumnSpec config must be a string or mapping. "
                f"Current input type: {type(config).__name__}."
            )
        if "name" not in config:
            raise ValueError("ColumnSpec config requires key 'name'.")
        known = {"name", "role", "dtype", "unit", "description"}
        metadata = {str(key): value for key, value in config.items() if key not in known}
        return cls(
            name=str(config["name"]),
            role=str(config.get("role", "input")).strip().lower(),
            dtype=None if config.get("dtype") is None else str(config["dtype"]),
            unit=None if config.get("unit") is None else str(config["unit"]),
            description=None
            if config.get("description") is None
            else str(config["description"]),
            metadata=metadata,
        )

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary."""

        data: dict[str, Any] = {"name": self.name, "role": self.role}
        if self.dtype is not None:
            data["dtype"] = self.dtype
        if self.unit is not None:
            data["unit"] = self.unit
        if self.description is not None:
            data["description"] = self.description
        data.update(self.metadata)
        return data


@dataclass(frozen=True)
class DataSchema:
    """Dataset-level schema with role-based column lookup."""

    columns: tuple[ColumnSpec, ...] = ()
    sample_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "DataSchema":
        """Build a data schema from a dataset-card mapping."""

        if config is None:
            return cls()
        if not isinstance(config, dict):
            raise TypeError(
                "DataSchema config must be a mapping. "
                f"Current input type: {type(config).__name__}."
            )
        raw_columns = config.get("columns", ())
        if isinstance(raw_columns, dict):
            columns = tuple(
                ColumnSpec.from_config({"name": name, "role": role})
                for name, role in raw_columns.items()
            )
        else:
            columns = tuple(ColumnSpec.from_config(item) for item in raw_columns)
        known = {"columns", "sample_rate"}
        metadata = {str(key): value for key, value in config.items() if key not in known}
        sample_rate = config.get("sample_rate")
        return cls(
            columns=columns,
            sample_rate=None if sample_rate is None else float(sample_rate),
            metadata=metadata,
        )

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return column names in schema order."""

        return tuple(column.name for column in self.columns)

    @property
    def roles(self) -> tuple[str, ...]:
        """Return known semantic roles."""

        return tuple(sorted({column.role for column in self.columns}))

    def role_columns(self, role: str) -> tuple[str, ...]:
        """Return column names matching ``role``."""

        normalized = role.strip().lower()
        return tuple(column.name for column in self.columns if column.role == normalized)

    def role_map(self) -> dict[str, list[str]]:
        """Return ``role -> columns`` mapping."""

        output: dict[str, list[str]] = {}
        for column in self.columns:
            output.setdefault(column.role, []).append(column.name)
        return output

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary."""

        data: dict[str, Any] = {
            "columns": [column.summary() for column in self.columns],
            "roles": self.role_map(),
        }
        if self.sample_rate is not None:
            data["sample_rate"] = self.sample_rate
        data.update(self.metadata)
        return data


@dataclass(frozen=True)
class TaskSchema:
    """Task-specific column selection and metadata."""

    name: str
    inputs: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    label_column: str | None = None
    normal_label: Any | None = None
    fault_switch: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any] | None) -> "TaskSchema":
        """Build task schema from a dataset-card task entry."""

        if config is None:
            return cls(name=name)
        if not isinstance(config, dict):
            raise TypeError(
                "TaskSchema config must be a mapping. "
                f"Current input type: {type(config).__name__}."
            )
        known = {"inputs", "targets", "label_column", "normal_label", "fault_switch", "pipeline"}
        metadata = {str(key): value for key, value in config.items() if key not in known}
        fault_switch = config.get("fault_switch")
        return cls(
            name=name,
            inputs=_string_tuple(config.get("inputs")),
            targets=_string_tuple(config.get("targets")),
            label_column=None
            if config.get("label_column") is None
            else str(config["label_column"]),
            normal_label=config.get("normal_label"),
            fault_switch=None if fault_switch is None else int(fault_switch),
            metadata=metadata,
        )

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary."""

        data: dict[str, Any] = {"name": self.name}
        if self.inputs:
            data["inputs"] = list(self.inputs)
        if self.targets:
            data["targets"] = list(self.targets)
        if self.label_column is not None:
            data["label_column"] = self.label_column
        if self.normal_label is not None:
            data["normal_label"] = self.normal_label
        if self.fault_switch is not None:
            data["fault_switch"] = self.fault_switch
        data.update(self.metadata)
        return data


@dataclass(frozen=True)
class SegmentInfo:
    """Provenance for one raw segment or file within a split."""

    split: str
    source: str
    rows: int
    segment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary."""

        data: dict[str, Any] = {
            "split": self.split,
            "source": self.source,
            "rows": self.rows,
        }
        if self.segment_id is not None:
            data["segment_id"] = self.segment_id
        data.update(self.metadata)
        return data


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
