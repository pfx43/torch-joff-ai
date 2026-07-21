"""Composable data pipeline configuration wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PIPELINE_STEP_ALIASES = {
    "scale": "normalization",
    "scaler": "normalization",
    "outlier": "outliers",
    "to_torch_dataset": "to_torch",
}

LEGAL_PIPELINE_STEPS = (
    "validate_schema",
    "missing",
    "outliers",
    "normalization",
    "split",
    "mask",
    "window",
    "sequence",
    "mpc_window",
    "to_torch",
)


@dataclass(frozen=True)
class PipelineStep:
    """One normalized data pipeline step."""

    name: str
    params: dict[str, Any]

    def to_config(self) -> dict[str, Any]:
        """Return this step as a mapping."""

        return {self.name: _json_ready(self.params)}


@dataclass(frozen=True)
class DataPipeline:
    """Serializable data pipeline config used by :class:`joff.data.DataModule`."""

    steps: tuple[PipelineStep, ...]

    @classmethod
    def from_config(cls, source: str | Path | Mapping[str, Any] | list[Any] | "DataPipeline" | None) -> "DataPipeline":
        """Build a pipeline from YAML path, mapping, list, another pipeline, or ``None``."""

        if isinstance(source, DataPipeline):
            return source
        loaded = _load_pipeline_source(source)
        config = normalize_pipeline_config(loaded)
        return cls(tuple(PipelineStep(name, dict(params)) for name, params in config.items()))

    def to_config(self) -> dict[str, Any]:
        """Return a canonical merged config mapping."""

        output: dict[str, Any] = {}
        for step in self.steps:
            if step.name in output and isinstance(output[step.name], dict):
                output[step.name].update(_json_ready(step.params))
            else:
                output[step.name] = _json_ready(step.params)
        return output

    def summary(self) -> dict[str, Any]:
        """Return a serializable summary."""

        return {
            "steps": [step.to_config() for step in self.steps],
            "config": self.to_config(),
        }

    def merge(self, other: str | Path | Mapping[str, Any] | list[Any] | "DataPipeline" | None) -> "DataPipeline":
        """Return a new pipeline with ``other`` overriding this config."""

        merged = merge_pipeline_configs(self.to_config(), DataPipeline.from_config(other).to_config())
        return DataPipeline.from_config(merged)


def normalize_pipeline_config(
    source: Mapping[str, Any] | list[Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize dict/list pipeline syntax into canonical step mappings."""

    if source is None:
        return {}
    if isinstance(source, Mapping):
        mapping = dict(source)
        if "pipeline" in mapping and len(mapping) == 1:
            return normalize_pipeline_config(mapping["pipeline"])
        return _canonical_pipeline_keys(mapping)
    if isinstance(source, list):
        output: dict[str, dict[str, Any]] = {}
        for item in source:
            if isinstance(item, str):
                step = {item: {}}
            elif isinstance(item, Mapping):
                step = dict(item)
            else:
                raise TypeError(
                    "Pipeline list entries must be strings or mappings. "
                    f"Current input type: {type(item).__name__}."
                )
            output = merge_pipeline_configs(output, _canonical_pipeline_keys(step))
        return output
    raise TypeError(
        "pipeline must be a mapping, a list of steps, a YAML path, DataPipeline, or None. "
        f"Current input type: {type(source).__name__}."
    )


def merge_pipeline_configs(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Merge normalized pipeline configs, with ``override`` taking precedence."""

    merged = normalize_pipeline_config(dict(base or {}))
    for key, value in normalize_pipeline_config(dict(override or {})).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(_json_ready(value))
            merged[key] = nested
        else:
            merged[key] = _json_ready(value)
    return merged


def _load_pipeline_source(source: str | Path | Mapping[str, Any] | list[Any] | None) -> Mapping[str, Any] | list[Any] | None:
    if source is None or isinstance(source, (Mapping, list)):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"DataPipeline config file does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, (Mapping, list)):
        raise ValueError(
            f"DataPipeline config file {path} must contain a mapping or list. "
            f"Current input type: {type(data).__name__}."
        )
    return data


def _canonical_pipeline_keys(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw_key, value in config.items():
        key = PIPELINE_STEP_ALIASES.get(str(raw_key), str(raw_key))
        if key not in LEGAL_PIPELINE_STEPS:
            legal = ", ".join(LEGAL_PIPELINE_STEPS)
            raise ValueError(
                f"Unknown data pipeline step {raw_key!r}. Legal options are: {legal}. "
                f"Current input was: {raw_key!r}."
            )
        if value is None:
            output[key] = {}
        elif isinstance(value, str) and key == "normalization":
            output[key] = {"method": value}
        elif isinstance(value, Mapping):
            output[key] = _json_ready(value)
        else:
            raise TypeError(
                f"Pipeline step {raw_key!r} must be configured with a mapping or null. "
                f"Current input type: {type(value).__name__}."
            )
    return output


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value
