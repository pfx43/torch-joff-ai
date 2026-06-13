"""Adapters for the real legacy process datasets copied into ``datasets/raw``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joff.data.schema import ColumnSpec, DataSchema, SegmentInfo, TaskSchema
from joff.data.sources import read_mat_arrays, read_npz_arrays

from .base import CanonicalDataset, Segment
from .builtin import SyntheticCSTRFaultAdapter, SyntheticProcessAdapter


_OA_ACCESS = {"tag": "oa", "disclosure": "open_access", "license": "to_verify"}
_PRIVATE_ACCESS = {
    "tag": "private",
    "disclosure": "non_public",
    "reason": "hydrocracking_proprietary",
}


class TEFaultDiagnosisAdapter:
    """Read the real Tennessee Eastman fault-diagnosis files."""

    name = "te_fault_diagnosis"
    version = "real-v1"
    description = "Tennessee Eastman fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic TE-style fault-diagnosis smoke dataset.",
            domain="tennessee_eastman",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "fd", required=("train/Mode1_Normal.mat",))
        train_array = _first_array(read_mat_arrays(root_path / "train" / "Mode1_Normal.mat"), "Normal")
        train = (
            _segment(
                _feature_frame(train_array, label=0, segment_id="Normal"),
                split="train",
                source=root_path / "train" / "Mode1_Normal.mat",
                segment_id="Normal",
                metadata={"fault_id": 0},
            ),
        )
        test_segments: list[Segment] = []
        npy_paths = sorted((root_path / "test").glob("Fault*.npy"))
        if npy_paths:
            for path in npy_paths:
                fault_id = _fault_id_from_name(path.stem)
                test_segments.append(
                    _segment(
                        _feature_frame(np.load(path), label=fault_id, segment_id=path.stem),
                        split="test",
                        source=path,
                        segment_id=path.stem,
                        metadata={"fault_id": fault_id},
                    )
                )
        else:
            for key, array in sorted(read_mat_arrays(root_path / "test" / "Mode1_Faulty.mat").items()):
                fault_id = _fault_id_from_name(key)
                test_segments.append(
                    _segment(
                        _feature_frame(array, label=fault_id, segment_id=key),
                        split="test",
                        source=root_path / "test" / "Mode1_Faulty.mat",
                        segment_id=key,
                        metadata={"fault_id": fault_id, "mat_key": key},
                    )
                )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train, "test": tuple(test_segments)},
            schema=_fault_schema(train_array.shape[1], domain="tennessee_eastman"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(33, domain="tennessee_eastman")

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "TE/fd"})


class TEClassificationAdapter:
    """Read the real Tennessee Eastman DAT classification files."""

    name = "te_classification"
    version = "real-v1"
    description = "Tennessee Eastman classification dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="classification",
            description="Deterministic TE-style classification smoke dataset.",
            domain="tennessee_eastman",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "cls", required=("train/d00.dat",))
        train_segments = tuple(_te_dat_segment(path, "train") for path in sorted((root_path / "train").glob("d*.dat")))
        test_segments = tuple(_te_dat_segment(path, "test") for path in sorted((root_path / "test").glob("d*_te.dat")))
        if not train_segments or not test_segments:
            raise FileNotFoundError(f"TE classification files were not found under {root_path}.")
        feature_count = int(train_segments[0].frame.filter(regex=r"^x").shape[1])
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=_classification_schema(feature_count, domain="tennessee_eastman"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _classification_schema(52, domain="tennessee_eastman")

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "classification"
        if task_name != "classification":
            raise ValueError(f"TE classification supports only task 'classification', got {task_name!r}.")
        return TaskSchema(
            name="classification",
            inputs=("input",),
            targets=("label",),
            label_column="label",
            normal_label=0,
        )

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "TE/cls"})


@dataclass(frozen=True)
class CSTRFaultAdapter:
    """Read CSTR fault-diagnosis MAT files."""

    name: str = "cstr_fault_diagnosis"
    subdir: str = "fd"
    feature_count: int = 10
    description: str = "Feedback-controlled CSTR fault-diagnosis dataset."
    version: str = "real-v1"

    def __post_init__(self) -> None:
        fallback = (
            SyntheticCSTRFaultAdapter()
            if self.name == "cstr_fault_diagnosis"
            else SyntheticProcessAdapter(
                name=self.name,
                task_name="fault_diagnosis",
                description="Deterministic closed-loop CSTR fault-diagnosis smoke dataset.",
                domain="process_control",
            )
        )
        object.__setattr__(self, "_fallback", fallback)

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, self.subdir, required=("train/model1[train].mat",))
        train_segments = _mat_fault_segments(
            root_path / "train" / "model1[train].mat",
            split="train",
            normal=True,
        )
        test_segments = _mat_fault_segments(
            root_path / "test" / "model1[test].mat",
            split="test",
            normal=False,
        )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=_fault_schema(self.feature_count, domain="process_control"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(self.feature_count, domain="process_control")

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": f"CSTR/{self.subdir}"})


class TTSFaultDiagnosisAdapter:
    """Read Three-Tank-System fault-diagnosis MAT files."""

    name = "tts_fault_diagnosis"
    version = "real-v1"
    description = "Three-tank-system fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic TTS-style fault-diagnosis smoke dataset.",
            domain="process_control",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "fd", required=("train/[train].mat",))
        train_segments = _mat_fault_segments(root_path / "train" / "[train].mat", split="train", normal=True)
        test_segments = _mat_fault_segments(root_path / "test" / "[test].mat", split="test", normal=False)
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=_fault_schema(7, domain="process_control"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(7, domain="process_control")

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "TTS/fd"})


@dataclass(frozen=True)
class NpyReconstructionAdapter:
    """Read Normal/Fault*.npy reconstruction datasets."""

    name: str
    subdir: str
    raw_folder: str
    description: str
    domain: str = "process_control"
    feature_count: int = 5
    version: str = "real-v1"

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return SyntheticProcessAdapter(
                name=self.name,
                task_name="prediction",
                description=self.description,
                domain=self.domain,
            ).read(root=None, task="prediction")
        root_path = _resolve_dataset_root(root, self.subdir, required=("train/Normal.npy",))
        train_segments = (
            _segment(
                _feature_frame(np.load(root_path / "train" / "Normal.npy"), segment_id="Normal"),
                split="train",
                source=root_path / "train" / "Normal.npy",
                segment_id="Normal",
            ),
        )
        test_segments = tuple(
            _segment(
                _feature_frame(np.load(path), segment_id=path.stem),
                split="test",
                source=path,
                segment_id=path.stem,
            )
            for path in sorted((root_path / "test").glob("Fault*.npy"))
        )
        if not test_segments:
            raise FileNotFoundError(f"No Fault*.npy files were found under {root_path / 'test'}.")
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": test_segments},
            schema=_reconstruction_schema(self.feature_count, domain=self.domain),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _reconstruction_schema(self.feature_count, domain=self.domain)

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "reconstruction"
        if task_name != "reconstruction":
            raise ValueError(f"{self.name} supports only task 'reconstruction', got {task_name!r}.")
        return TaskSchema(name="reconstruction")

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(
            self,
            self.default_task(task),
            access=_OA_ACCESS,
            files={"root": self.raw_folder},
        )


class MultiphaseFaultAdapter:
    """Read the Multiphase Flow Facility normal and faulty MAT files."""

    name = "multiphase_fd"
    version = "real-v1"
    description = "Multiphase Flow Facility fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic multiphase-flow fault-diagnosis smoke dataset.",
            domain="multiphase_flow",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = Path(root)
        if not (root_path / "train" / "Training.mat").exists():
            root_path = root_path / "Multiphase_Flow_Facility"
        _require_file(root_path / "train" / "Training.mat")
        train_segments = tuple(
            _segment(
                _feature_frame(array, label=0, segment_id=key),
                split="train",
                source=root_path / "train" / "Training.mat",
                segment_id=key,
                metadata={"fault_id": 0, "mat_key": key},
            )
            for key, array in sorted(read_mat_arrays(root_path / "train" / "Training.mat").items())
            if key.lower().startswith("normal")
        )
        test_segments: list[Segment] = []
        for path in sorted((root_path / "test").glob("FaultyCase*.mat")):
            arrays = read_mat_arrays(path)
            case_id = _fault_id_from_name(path.stem)
            for key, array in sorted(arrays.items()):
                if not key.startswith("Set"):
                    continue
                suffix = key.removeprefix("Set")
                label_key = f"EvoFault{suffix}"
                labels = arrays.get(label_key)
                if labels is None:
                    label = np.full(_as_2d(array).shape[0], case_id)
                else:
                    label = np.asarray(labels).reshape(-1)
                    label = np.where(label > 0, case_id, 0)
                segment_id = f"{path.stem}:{key}"
                test_segments.append(
                    _segment(
                        _feature_frame(array, label=label, segment_id=segment_id),
                        split="test",
                        source=path,
                        segment_id=segment_id,
                        metadata={"fault_case": case_id, "mat_key": key, "label_key": label_key},
                    )
                )
        if not train_segments or not test_segments:
            raise FileNotFoundError(f"Multiphase train/test segments were not found under {root_path}.")
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": tuple(test_segments)},
            schema=_fault_schema(24, domain="multiphase_flow"),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(24, domain="multiphase_flow")

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "Multiphase_Flow_Facility"})


class WPTMPCAdapter:
    """Read WPT PRBS input/output data for MPC windows."""

    name = "wpt_mpc"
    version = "real-v1"
    description = "WPT PRBS MPC dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="mpc",
            description="Deterministic WPT-style MPC smoke dataset.",
            domain="model_predictive_control",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = Path(root)
        if not (root_path / "PRBS_1024_u.mat").exists():
            root_path = root_path / "WPT"
        _require_file(root_path / "PRBS_1024_u.mat")
        u = _first_array(read_mat_arrays(root_path / "PRBS_1024_u.mat"), "uk")
        y = _first_array(read_mat_arrays(root_path / "PRBS_1024_y.mat"), "yk")
        frame = _wpt_frame(u, y)
        split_at = max(16, int(frame.shape[0] * 0.8))
        train_frame = frame.iloc[:split_at].reset_index(drop=True)
        test_frame = frame.iloc[split_at:].reset_index(drop=True)
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={
                "train": (
                    _segment(train_frame, split="train", source=root_path / "PRBS_1024_u.mat", segment_id="prbs_train"),
                ),
                "test": (
                    _segment(test_frame, split="test", source=root_path / "PRBS_1024_y.mat", segment_id="prbs_test"),
                ),
            },
            schema=self.schema(),
            access=_OA_ACCESS,
        )

    def schema(self) -> DataSchema:
        return DataSchema(
            columns=(
                ColumnSpec("episode", role="episode"),
                ColumnSpec("state1", role="state"),
                ColumnSpec("control1", role="control"),
                ColumnSpec("output1", role="output"),
                ColumnSpec("reference1", role="reference"),
            ),
            sample_rate=1.0,
            metadata={"domain": "model_predictive_control"},
        )

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "mpc"
        if task_name != "mpc":
            raise ValueError(f"WPT adapter supports only task 'mpc', got {task_name!r}.")
        return TaskSchema(name="mpc", inputs=("state", "output", "control", "reference"), targets=("output",))

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return {
            "split": {"type": "official"},
            "normalization": {"method": "standard"},
            "mpc_window": {
                "past_horizon": 2,
                "prediction_horizon": 2,
                "control_horizon": 2,
                "return_mode": "tuple",
            },
        }

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_OA_ACCESS, files={"root": "WPT"})


class HYFaultAdapter:
    """Read private hydrocracking NPZ fault-diagnosis splits."""

    name = "hy_fault_diagnosis"
    version = "real-v1"
    description = "Private hydrocracking fault-diagnosis dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="fault_diagnosis",
            description="Deterministic HY-style fault-diagnosis smoke dataset.",
            domain="hydrocracking",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = _resolve_dataset_root(root, "fd", required=("train/train_X.npz",))
        train_x = read_npz_arrays(root_path / "train" / "train_X.npz")
        test_x = read_npz_arrays(root_path / "test" / "test_X.npz")
        test_y = read_npz_arrays(root_path / "test" / "test_Y.npz")
        train_segments = tuple(
            _segment(
                _feature_frame(array, label=0, segment_id=key),
                split="train",
                source=root_path / "train" / "train_X.npz",
                segment_id=key,
                metadata={"fault_id": 0, "npz_key": key},
            )
            for key, array in sorted(train_x.items())
        )
        test_segments = []
        for key, array in sorted(test_x.items()):
            label_key = key.replace("_X", "_Y")
            labels = test_y.get(label_key)
            if labels is None:
                raise KeyError(f"Missing HY label key {label_key!r} for feature key {key!r}.")
            test_segments.append(
                _segment(
                    _feature_frame(array, label=np.asarray(labels).reshape(-1), segment_id=key),
                    split="test",
                    source=root_path / "test" / "test_X.npz",
                    segment_id=key,
                    metadata={"npz_key": key, "label_key": label_key},
                )
            )
        feature_count = int(next(iter(train_x.values())).shape[1])
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": train_segments, "test": tuple(test_segments)},
            schema=_fault_schema(feature_count, domain="hydrocracking", access=_PRIVATE_ACCESS),
            access=_PRIVATE_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _fault_schema(52, domain="hydrocracking", access=_PRIVATE_ACCESS)

    def default_task(self, task: str | None = None) -> TaskSchema:
        return _fault_task(task)

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        self.default_task(task)
        return _standard_official_pipeline()

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_PRIVATE_ACCESS, files={"root": "HY/fd"})


class HYQualityPredictionAdapter:
    """Read private hydrocracking product-quality CSV files."""

    name = "hy_quality_prediction"
    version = "real-v1"
    description = "Private hydrocracking product-quality prediction dataset."

    def __init__(self) -> None:
        self._fallback = SyntheticProcessAdapter(
            name=self.name,
            task_name="prediction",
            description="Deterministic HY quality-prediction smoke dataset.",
            domain="hydrocracking",
        )

    def read(self, *, root: str | Path | None = None, task: str | None = None) -> CanonicalDataset:
        if root is None:
            return self._fallback.read(root=None, task=task)
        root_path = Path(root)
        if not (root_path / "郑迪_prd" / "2017年后数据").exists():
            root_path = root_path / "HY_PRD"
        csv_paths = sorted((root_path / "郑迪_prd" / "2017年后数据").glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(f"No HY_PRD CSV files were found under {root_path}.")
        prepared = [_hy_prd_arrays(path) for path in csv_paths]
        feature_count = min(item[0].shape[1] for item in prepared)
        train_segments: list[Segment] = []
        test_segments: list[Segment] = []
        for path, (x, y, target_name) in zip(csv_paths, prepared, strict=True):
            x = x[:, :feature_count]
            split_at = max(1, int(x.shape[0] * 0.8))
            split_at = min(split_at, x.shape[0] - 1)
            train_segments.append(
                _segment(
                    _prediction_frame(x[:split_at], y[:split_at], segment_id=path.stem),
                    split="train",
                    source=path,
                    segment_id=path.stem,
                    metadata={"target_name": target_name},
                )
            )
            test_segments.append(
                _segment(
                    _prediction_frame(x[split_at:], y[split_at:], segment_id=path.stem),
                    split="test",
                    source=path,
                    segment_id=path.stem,
                    metadata={"target_name": target_name},
                )
            )
        return _canonical(
            name=self.name,
            version=self.version,
            root=root_path,
            splits={"train": tuple(train_segments), "test": tuple(test_segments)},
            schema=_prediction_schema(feature_count, domain="hydrocracking", access=_PRIVATE_ACCESS),
            access=_PRIVATE_ACCESS,
        )

    def schema(self) -> DataSchema:
        return _prediction_schema(42, domain="hydrocracking", access=_PRIVATE_ACCESS)

    def default_task(self, task: str | None = None) -> TaskSchema:
        task_name = task or "prediction"
        if task_name == "imputation":
            return TaskSchema(name="imputation")
        if task_name != "prediction":
            raise ValueError(f"HY_PRD adapter supports prediction/imputation, got {task_name!r}.")
        return TaskSchema(name="prediction", targets=("quality",))

    def default_pipeline(self, task: str | None = None) -> dict[str, Any]:
        task_schema = self.default_task(task)
        pipeline = _standard_official_pipeline()
        if task_schema.name == "imputation":
            pipeline["mask"] = {"strategy": "random", "missing_rate": 0.2, "seed": 42}
        return pipeline

    def summary(self, task: str | None = None) -> dict[str, Any]:
        return _summary(self, self.default_task(task), access=_PRIVATE_ACCESS, files={"root": "HY_PRD"})


def _canonical(
    *,
    name: str,
    version: str,
    root: Path,
    splits: dict[str, tuple[Segment, ...]],
    schema: DataSchema,
    access: dict[str, Any],
) -> CanonicalDataset:
    return CanonicalDataset(
        splits=splits,
        schema=schema,
        metadata={
            "source_type": "real_dataset",
            "preset": name,
            "version": version,
            "root": str(root),
            "access": dict(access),
        },
    )


def _summary(
    adapter: Any,
    task: TaskSchema,
    *,
    access: dict[str, Any],
    files: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": adapter.name,
        "version": adapter.version,
        "description": adapter.description,
        "task": task.summary(),
        "tasks": [task.name],
        "files": files,
        "access": dict(access),
        "source": "real dataset when root is supplied; synthetic smoke fallback when root is omitted",
    }


def _standard_official_pipeline() -> dict[str, Any]:
    return {"split": {"type": "official"}, "normalization": {"method": "standard"}}


def _fault_task(task: str | None) -> TaskSchema:
    task_name = task or "fault_diagnosis"
    if task_name != "fault_diagnosis":
        raise ValueError(f"Fault adapters support only task 'fault_diagnosis', got {task_name!r}.")
    return TaskSchema(
        name="fault_diagnosis",
        targets=("fault_id",),
        label_column="fault_id",
        normal_label=0,
    )


def _resolve_dataset_root(root: str | Path, child: str, *, required: tuple[str, ...]) -> Path:
    root_path = Path(root)
    if all((root_path / item).exists() for item in required):
        return root_path
    child_path = root_path / child
    if all((child_path / item).exists() for item in required):
        return child_path
    missing = ", ".join(required)
    raise FileNotFoundError(f"Dataset root {root_path} does not contain required files: {missing}.")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file does not exist: {path}")


def _mat_fault_segments(path: Path, *, split: str, normal: bool) -> tuple[Segment, ...]:
    arrays = read_mat_arrays(path)
    segments: list[Segment] = []
    for key, array in sorted(arrays.items()):
        fault_id = 0 if normal or "normal" in key.lower() else _fault_id_from_name(key)
        segments.append(
            _segment(
                _feature_frame(array, label=fault_id, segment_id=key),
                split=split,
                source=path,
                segment_id=key,
                metadata={"fault_id": fault_id, "mat_key": key},
            )
        )
    return tuple(segments)


def _te_dat_segment(path: Path, split: str) -> Segment:
    label = _fault_id_from_name(path.stem)
    array = pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=float)
    return _segment(
        _feature_frame(array, label=label, label_column="label", segment_id=path.stem),
        split=split,
        source=path,
        segment_id=path.stem,
        metadata={"label": label},
    )


def _feature_frame(
    array: Any,
    *,
    label: int | np.ndarray | None = None,
    label_column: str = "fault_id",
    segment_id: str | None = None,
) -> pd.DataFrame:
    values = _clean_array(array)
    frame = pd.DataFrame(values, columns=_feature_names(values.shape[1]))
    frame.insert(0, "time", np.arange(values.shape[0], dtype=float))
    if segment_id is not None:
        frame["segment"] = segment_id
    if label is not None:
        labels = np.asarray(label)
        if labels.ndim == 0:
            labels = np.full(values.shape[0], float(labels))
        labels = labels.reshape(-1)
        if labels.shape[0] != values.shape[0]:
            raise ValueError(
                f"Label length must match rows. Current labels={labels.shape[0]}, rows={values.shape[0]}."
            )
        frame[label_column] = labels.astype(float)
    return frame


def _prediction_frame(x: np.ndarray, y: np.ndarray, *, segment_id: str) -> pd.DataFrame:
    values = _clean_array(x)
    target = _as_2d(y)
    frame = pd.DataFrame(values, columns=_feature_names(values.shape[1]))
    frame.insert(0, "time", np.arange(values.shape[0], dtype=float))
    frame["segment"] = segment_id
    frame["quality"] = target.reshape(-1).astype(float)
    return frame


def _wpt_frame(u: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    u = _clean_array(u)
    y = _clean_array(y)
    rows = min(u.shape[0], y.shape[0])
    u = u[:rows]
    y = y[:rows]
    time = u[:, 0] if u.shape[1] > 1 else np.arange(rows, dtype=float)
    control = u[:, -1]
    output = y[:, -1]
    return pd.DataFrame(
        {
            "episode": np.zeros(rows, dtype=int),
            "time": time,
            "state1": output,
            "control1": control,
            "output1": output,
            "reference1": np.zeros(rows, dtype=float),
        }
    )


def _hy_prd_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(path)
    excluded = {"日期", "序号"}
    candidate_columns = [column for column in frame.columns if str(column) not in excluded]
    if len(candidate_columns) < 2:
        raise ValueError(f"HY_PRD CSV must contain target and feature columns: {path}")
    target_column = str(candidate_columns[0])
    feature_columns = [str(column) for column in candidate_columns[1:]]
    numeric = frame.loc[:, [target_column, *feature_columns]].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=0, how="any")
    if numeric.empty:
        raise ValueError(f"HY_PRD CSV produced no numeric rows after cleanup: {path}")
    y = numeric[target_column].to_numpy(dtype=float)[:, None]
    x = numeric.loc[:, feature_columns].to_numpy(dtype=float)
    return x, y, target_column


def _segment(
    frame: pd.DataFrame,
    *,
    split: str,
    source: str | Path,
    segment_id: str,
    metadata: dict[str, Any] | None = None,
) -> Segment:
    return Segment(
        frame=frame,
        meta=SegmentInfo(
            split=split,
            source=str(source),
            rows=int(frame.shape[0]),
            segment_id=segment_id,
            metadata=metadata or {},
        ),
    )


def _fault_schema(feature_count: int, *, domain: str, access: dict[str, Any] | None = None) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
            ColumnSpec("fault_id", role="fault_id"),
        ),
        sample_rate=1.0,
        metadata={"domain": domain, "access": access} if access else {"domain": domain},
    )


def _classification_schema(feature_count: int, *, domain: str) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
            ColumnSpec("label", role="label"),
        ),
        sample_rate=1.0,
        metadata={"domain": domain},
    )


def _reconstruction_schema(feature_count: int, *, domain: str) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
        ),
        sample_rate=1.0,
        metadata={"domain": domain},
    )


def _prediction_schema(feature_count: int, *, domain: str, access: dict[str, Any] | None = None) -> DataSchema:
    return DataSchema(
        columns=(
            ColumnSpec("time", role="time"),
            ColumnSpec("segment", role="segment"),
            *[ColumnSpec(name, role="input") for name in _feature_names(feature_count)],
            ColumnSpec("quality", role="quality"),
        ),
        sample_rate=1.0,
        metadata={"domain": domain, "access": access} if access else {"domain": domain},
    )


def _feature_names(feature_count: int) -> tuple[str, ...]:
    return tuple(f"x{idx:02d}" for idx in range(int(feature_count)))


def _fault_id_from_name(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits[-2:] if len(digits) >= 2 else digits or 0)


def _first_array(arrays: dict[str, np.ndarray], preferred: str) -> np.ndarray:
    if preferred in arrays:
        return arrays[preferred]
    if not arrays:
        raise ValueError("No numeric arrays were found in dataset file.")
    return arrays[sorted(arrays)[0]]


def _clean_array(array: Any) -> np.ndarray:
    values = _as_2d(array).astype(float)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _as_2d(array: Any) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 1:
        return values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Current shape: {values.shape}.")
    return values


__all__ = [
    "CSTRFaultAdapter",
    "HYFaultAdapter",
    "HYQualityPredictionAdapter",
    "MultiphaseFaultAdapter",
    "NpyReconstructionAdapter",
    "TEClassificationAdapter",
    "TEFaultDiagnosisAdapter",
    "TTSFaultDiagnosisAdapter",
    "WPTMPCAdapter",
]
