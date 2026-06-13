"""Built-in dataset adapters used for smoke tests and legacy aliases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joff.data.schema import ColumnSpec, DataSchema, SegmentInfo, TaskSchema

from .base import CanonicalDataset, DatasetCardAdapter, Segment


class SyntheticCSTRFaultAdapter:
    """Small deterministic CSTR fault-diagnosis adapter.

    If ``root`` contains ``dataset_card.yaml`` the card is used. Otherwise this
    adapter returns a compact synthetic split so preset smoke tests do not need
    bundled external data files.
    """

    name = "cstr_fault_diagnosis"
    version = "synthetic-smoke-v1"
    description = "Deterministic CSTR-style fault diagnosis smoke dataset."

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        """Return dataset-card data from ``root`` or deterministic synthetic data."""

        if root is not None:
            root_path = Path(root)
            card_path = root_path / "dataset_card.yaml"
            if card_path.exists():
                return DatasetCardAdapter.from_yaml(card_path).read(root=root_path, task=task)
        train = _cstr_frame(n=72, fault_start=None, seed=13)
        test = _cstr_frame(n=36, fault_start=12, seed=17)
        return CanonicalDataset(
            splits={
                "train": (
                    Segment(
                        train,
                        SegmentInfo(
                            split="train",
                            source="synthetic:cstr_fault_diagnosis/train",
                            rows=int(train.shape[0]),
                            segment_id="train_normal",
                        ),
                    ),
                ),
                "test": (
                    Segment(
                        test,
                        SegmentInfo(
                            split="test",
                            source="synthetic:cstr_fault_diagnosis/test",
                            rows=int(test.shape[0]),
                            segment_id="test_fault",
                        ),
                    ),
                ),
            },
            schema=self.schema(),
            metadata={
                "source_type": "builtin_synthetic",
                "preset": self.name,
                "version": self.version,
            },
        )

    def schema(self) -> DataSchema:
        """Return the CSTR smoke schema."""

        return DataSchema(
            columns=(
                ColumnSpec("time", role="time"),
                ColumnSpec("u1", role="control"),
                ColumnSpec("u2", role="control"),
                ColumnSpec("u3", role="control"),
                ColumnSpec("y1", role="output"),
                ColumnSpec("y2", role="output"),
                ColumnSpec("fault_id", role="fault_id"),
            ),
            sample_rate=1.0,
            metadata={"domain": "process_control"},
        )

    def default_task(self, task: str | None = None) -> TaskSchema:
        """Return supported task metadata."""

        task_name = task or "fault_diagnosis"
        if task_name != "fault_diagnosis":
            raise ValueError(
                "Synthetic CSTR adapter supports only task 'fault_diagnosis'. "
                f"Current input was: {task_name!r}."
            )
        return TaskSchema(
            name="fault_diagnosis",
            inputs=("control", "output"),
            targets=("fault_id",),
            label_column="fault_id",
            normal_label=0,
            fault_switch=12,
        )

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """Return safe defaults for the smoke preset."""

        self.default_task(task)
        return {
            "split": {"type": "official"},
            "normalization": {"method": "standard"},
        }

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """Return a serializable adapter summary."""

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "task": self.default_task(task).summary(),
            "tasks": ["fault_diagnosis"],
            "files": {"source": "synthetic unless root/dataset_card.yaml exists"},
        }


@dataclass(frozen=True)
class SyntheticProcessAdapter:
    """Deterministic smoke adapter for legacy process-data presets."""

    name: str
    task_name: str
    description: str
    domain: str
    version: str = "synthetic-smoke-v1"

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        """Return dataset-card data from ``root`` or compact synthetic splits."""

        if root is not None:
            root_path = Path(root)
            card_path = root_path / "dataset_card.yaml"
            if card_path.exists():
                return DatasetCardAdapter.from_yaml(card_path).read(root=root_path, task=task)
        if self.task_name == "fault_diagnosis":
            train = _fault_frame(n=72, fault_start=None, seed=_seed_for(self.name, 3))
            test = _fault_frame(n=36, fault_start=12, seed=_seed_for(self.name, 7))
        elif self.task_name == "classification":
            train = _classification_frame(n=72, seed=_seed_for(self.name, 11))
            test = _classification_frame(n=36, seed=_seed_for(self.name, 13))
        elif self.task_name == "prediction":
            train = _prediction_frame(n=80, seed=_seed_for(self.name, 17))
            test = _prediction_frame(n=32, seed=_seed_for(self.name, 19))
        elif self.task_name == "mpc":
            train = _mpc_frame(episodes=4, rows_per_episode=10, seed=_seed_for(self.name, 23))
            test = _mpc_frame(episodes=2, rows_per_episode=10, seed=_seed_for(self.name, 29))
        else:
            raise ValueError(f"Unsupported synthetic task {self.task_name!r}.")
        return CanonicalDataset(
            splits={
                "train": (
                    _segment(
                        train,
                        split="train",
                        source=f"synthetic:{self.name}/train",
                        segment_id="train",
                    ),
                ),
                "test": (
                    _segment(
                        test,
                        split="test",
                        source=f"synthetic:{self.name}/test",
                        segment_id="test",
                    ),
                ),
            },
            schema=self.schema(),
            metadata={
                "source_type": "builtin_synthetic",
                "preset": self.name,
                "version": self.version,
            },
        )

    def schema(self) -> DataSchema:
        """Return task-specific smoke schema."""

        if self.task_name == "fault_diagnosis":
            columns = (
                ColumnSpec("time", role="time"),
                ColumnSpec("u1", role="control"),
                ColumnSpec("u2", role="control"),
                ColumnSpec("u3", role="control"),
                ColumnSpec("y1", role="output"),
                ColumnSpec("y2", role="output"),
                ColumnSpec("fault_id", role="fault_id"),
            )
        elif self.task_name == "classification":
            columns = (
                ColumnSpec("x1", role="input"),
                ColumnSpec("x2", role="input"),
                ColumnSpec("x3", role="input"),
                ColumnSpec("label", role="label"),
            )
        elif self.task_name == "prediction":
            columns = (
                ColumnSpec("time", role="time"),
                ColumnSpec("u1", role="control"),
                ColumnSpec("u2", role="control"),
                ColumnSpec("y1", role="output"),
                ColumnSpec("quality", role="quality"),
            )
        elif self.task_name == "mpc":
            columns = (
                ColumnSpec("episode", role="episode"),
                ColumnSpec("state1", role="state"),
                ColumnSpec("state2", role="state"),
                ColumnSpec("control1", role="control"),
                ColumnSpec("output1", role="output"),
                ColumnSpec("reference1", role="reference"),
            )
        else:
            columns = ()
        return DataSchema(
            columns=columns,
            sample_rate=1.0,
            metadata={"domain": self.domain},
        )

    def default_task(self, task: str | None = None) -> TaskSchema:
        """Return supported smoke task metadata."""

        task_name = task or self.task_name
        if self.task_name == "prediction" and task_name == "imputation":
            return TaskSchema(name="imputation", inputs=("control", "output", "quality"))
        if task_name != self.task_name:
            raise ValueError(
                f"Synthetic adapter {self.name!r} supports task {self.task_name!r}. "
                f"Current input was: {task_name!r}."
            )
        if task_name == "fault_diagnosis":
            return TaskSchema(
                name="fault_diagnosis",
                inputs=("control", "output"),
                targets=("fault_id",),
                label_column="fault_id",
                normal_label=0,
                fault_switch=12,
            )
        if task_name == "classification":
            return TaskSchema(
                name="classification",
                inputs=("input",),
                targets=("label",),
                label_column="label",
                normal_label=0,
            )
        if task_name == "prediction":
            return TaskSchema(name="prediction", inputs=("control", "output"), targets=("quality",))
        if task_name == "mpc":
            return TaskSchema(
                name="mpc",
                inputs=("state", "output", "control", "reference"),
                targets=("output",),
            )
        raise ValueError(f"Unsupported synthetic task {task_name!r}.")

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        """Return safe defaults for smoke data."""

        task_schema = self.default_task(task)
        pipeline: dict[str, Any] = {
            "split": {"type": "official"},
            "normalization": {"method": "standard"},
        }
        if task_schema.name == "classification":
            pipeline.pop("normalization")
        if task_schema.name == "imputation":
            pipeline["mask"] = {"strategy": "random", "missing_rate": 0.2, "seed": 42}
        return pipeline

    def summary(self, task: str | None = None) -> dict[str, Any]:
        """Return a serializable adapter summary."""

        task_schema = self.default_task(task)
        tasks = [self.task_name]
        if self.task_name == "prediction":
            tasks.append("imputation")
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "task": task_schema.summary(),
            "tasks": sorted(tasks),
            "files": {"source": "synthetic unless root/dataset_card.yaml exists"},
        }


def _cstr_frame(*, n: int, fault_start: int | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(n, dtype=float)
    u1 = np.sin(time / 8.0)
    u2 = np.cos(time / 9.0)
    u3 = 0.5 * np.sin(time / 5.0)
    fault = np.zeros(n, dtype=int)
    if fault_start is not None:
        fault[fault_start:] = 1
    drift = fault.astype(float) * np.linspace(0.0, 0.8, n)
    y1 = 0.45 * u1 - 0.15 * u2 + drift + rng.normal(0.0, 0.02, n)
    y2 = 0.25 * u2 + 0.2 * u3 + 0.5 * drift + rng.normal(0.0, 0.02, n)
    return pd.DataFrame(
        {
            "time": time,
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "y1": y1,
            "y2": y2,
            "fault_id": fault,
        }
    )


def _fault_frame(*, n: int, fault_start: int | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(n, dtype=float)
    u1 = np.sin(time / 9.0)
    u2 = np.cos(time / 7.0)
    u3 = np.sin(time / 5.0) * 0.4
    fault = np.zeros(n, dtype=int)
    if fault_start is not None:
        fault[fault_start:] = 1
    drift = fault.astype(float) * np.linspace(0.0, 1.0, n)
    y1 = 0.4 * u1 - 0.1 * u2 + drift + rng.normal(0.0, 0.015, n)
    y2 = -0.2 * u1 + 0.3 * u3 + 0.5 * drift + rng.normal(0.0, 0.015, n)
    return pd.DataFrame(
        {
            "time": time,
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "y1": y1,
            "y2": y2,
            "fault_id": fault,
        }
    )


def _classification_frame(*, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = np.arange(n) % 2
    x1 = rng.normal(loc=label * 1.2, scale=0.1, size=n)
    x2 = rng.normal(loc=1.0 - label * 0.6, scale=0.1, size=n)
    x3 = x1 - x2 + rng.normal(0.0, 0.05, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "label": label})


def _prediction_frame(*, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(n, dtype=float)
    u1 = np.sin(time / 11.0)
    u2 = np.cos(time / 13.0)
    y1 = 0.5 * u1 + 0.2 * u2 + rng.normal(0.0, 0.01, n)
    quality = 0.7 * y1 + 0.15 * u1 - 0.1 * u2 + rng.normal(0.0, 0.01, n)
    return pd.DataFrame({"time": time, "u1": u1, "u2": u2, "y1": y1, "quality": quality})


def _mpc_frame(*, episodes: int, rows_per_episode: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(episodes):
        time = np.arange(rows_per_episode, dtype=float)
        reference = 1.0 + 0.1 * episode + 0.05 * np.sin(time / 4.0)
        control = 0.2 * np.cos(time / 5.0) + 0.03 * episode
        state1 = reference + 0.1 * np.sin(time / 3.0) + rng.normal(0.0, 0.005, rows_per_episode)
        state2 = 0.5 * state1 + control
        output = state1 + 0.2 * control + rng.normal(0.0, 0.005, rows_per_episode)
        rows.append(
            pd.DataFrame(
                {
                    "episode": episode,
                    "state1": state1,
                    "state2": state2,
                    "control1": control,
                    "output1": output,
                    "reference1": reference,
                }
            )
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def _segment(frame: pd.DataFrame, *, split: str, source: str, segment_id: str) -> Segment:
    return Segment(
        frame,
        SegmentInfo(split=split, source=source, rows=int(frame.shape[0]), segment_id=segment_id),
    )


def _seed_for(name: str, offset: int) -> int:
    return sum(ord(char) for char in name) + offset
