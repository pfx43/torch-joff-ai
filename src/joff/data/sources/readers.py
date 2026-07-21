"""Tabular and array file readers for data modules and dataset cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_SOURCE_SUFFIXES = (".csv", ".dat", ".mat", ".npy", ".npz", ".xlsx", ".xls")


@dataclass(frozen=True)
class SourceData:
    """Data loaded from one source file."""

    frame: pd.DataFrame | None = None
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    @property
    def rows(self) -> int:
        """Return row count."""

        if self.x is not None:
            return int(_as_2d(self.x).shape[0])
        if self.frame is not None:
            return int(self.frame.shape[0])
        return 0

    def to_frame(self) -> pd.DataFrame:
        """Return a frame representation, combining ``x`` and ``y`` arrays if needed."""

        if self.frame is not None:
            return self.frame
        if self.x is None:
            raise ValueError("SourceData has neither frame nor x array.")
        x_frame = _array_frame(self.x, prefix="x")
        if self.y is None:
            return x_frame
        y_frame = _array_frame(self.y, prefix="y")
        return pd.concat([x_frame, y_frame], axis=1)

    def summary(self) -> dict[str, Any]:
        """Return a serializable source summary."""

        data = dict(self.metadata or {})
        data["rows"] = self.rows
        if self.frame is not None:
            data["columns"] = [str(column) for column in self.frame.columns]
        if self.x is not None:
            data["x_shape"] = list(_as_2d(self.x).shape)
        if self.y is not None:
            data["y_shape"] = list(_as_2d(self.y).shape)
        return data


def read_source(path: str | Path, *, sheet_name: str | int | None = None) -> SourceData:
    """Read a supported data source file."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Data file does not exist: {source_path}")
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source_path)
        return _source(frame=frame, path=source_path, source_format="csv")
    if suffix == ".dat":
        frame = pd.read_csv(source_path, sep=r"\s+", header=None)
        return _source(frame=frame, path=source_path, source_format="dat")
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(source_path, sheet_name=sheet_name)
    if suffix == ".npz":
        return _read_npz(source_path)
    if suffix == ".npy":
        array = np.load(source_path)
        return _source(x=_as_2d(array), path=source_path, source_format="npy")
    if suffix == ".mat":
        return _read_mat(source_path)
    raise ValueError(
        f"Unsupported file suffix {suffix!r}. Legal options are: "
        f"{', '.join(SUPPORTED_SOURCE_SUFFIXES)}."
    )


def read_source_frame(path: str | Path, *, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read a source file and return a frame representation."""

    return read_source(path, sheet_name=sheet_name).to_frame()


def split_source_xy(
    source: SourceData,
    *,
    target_cols: int | list[int] | slice | None = -1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert loaded source data to ``x, y`` arrays."""

    if source.x is not None:
        return _as_2d(source.x), None if source.y is None else _as_2d(source.y)
    frame = source.to_frame()
    array = frame.to_numpy(dtype=np.float32)
    if target_cols is None:
        return array, None
    y = array[:, target_cols]
    if y.ndim == 1:
        y = y[:, None]
    x = np.delete(array, target_cols, axis=1)
    return x, y


def read_mat_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read all numeric 1D/2D arrays from a MAT file, including MATLAB v7.3 files."""

    source_path = Path(path)
    try:
        from scipy.io import loadmat

        loaded = loadmat(source_path)
        return {
            str(key): _as_2d(value)
            for key, value in loaded.items()
            if not str(key).startswith("__") and _is_numeric_1d_or_2d(value)
        }
    except NotImplementedError:
        return _read_hdf5_mat_arrays(source_path)
    except ValueError as exc:
        if "Unknown mat file type" in str(exc):
            return _read_hdf5_mat_arrays(source_path)
        raise


def read_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Read all arrays from an NPZ file as 2D numpy arrays."""

    loaded = np.load(Path(path))
    return {str(key): _as_2d(loaded[key]) for key in loaded.files}


def _read_excel(path: Path, *, sheet_name: str | int | None = None) -> SourceData:
    try:
        frame, used_sheet = _read_excel_nonempty(path, sheet_name=sheet_name)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel files requires the optional excel dependency. "
            "Install it with: pip install \"joff[excel]\"."
        ) from exc
    return _source(frame=frame, path=path, source_format="excel", sheet_name=used_sheet)


def _read_npz(path: Path) -> SourceData:
    arrays = read_npz_arrays(path)
    if "x" in arrays:
        x = arrays["x"]
        y = arrays["y"] if "y" in arrays else None
        return _source(x=x, y=y, path=path, source_format="npz")
    keys = sorted(str(key) for key in arrays)
    if len(keys) == 1:
        return _source(x=arrays[keys[0]], path=path, source_format="npz", variable=keys[0])
    raise ValueError(
        "NPZ file must contain key 'x' with optional key 'y', or exactly one array. "
        f"Current keys are: {', '.join(keys)}."
    )


def _read_mat(path: Path) -> SourceData:
    arrays = read_mat_arrays(path)
    if "x" in arrays:
        y = arrays["y"] if "y" in arrays else None
        return _source(x=arrays["x"], y=y, path=path, source_format="mat")
    if len(arrays) == 1:
        key, value = next(iter(arrays.items()))
        return _source(x=value, path=path, source_format="mat", variable=key)
    legal = ", ".join(sorted(arrays)) or "<none>"
    raise ValueError(
        "MAT file must contain array 'x' with optional 'y', or exactly one numeric 1D/2D array. "
        f"Current numeric arrays are: {legal}."
    )


def _source(
    *,
    path: Path,
    source_format: str,
    frame: pd.DataFrame | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    **metadata: Any,
) -> SourceData:
    base = {"path": str(path), "suffix": path.suffix.lower(), "format": source_format}
    base.update(metadata)
    return SourceData(frame=frame, x=x, y=y, metadata=base)


def _read_excel_nonempty(path: Path, *, sheet_name: str | int | None) -> tuple[pd.DataFrame, str | int]:
    if sheet_name is not None:
        return pd.read_excel(path, sheet_name=sheet_name), sheet_name
    book = pd.ExcelFile(path)
    if not book.sheet_names:
        return pd.read_excel(path), 0
    first = pd.read_excel(path, sheet_name=book.sheet_names[0])
    if not first.empty or len(book.sheet_names) == 1:
        return first, book.sheet_names[0]
    for candidate in book.sheet_names[1:]:
        frame = pd.read_excel(path, sheet_name=candidate)
        if not frame.empty:
            return frame, candidate
    return first, book.sheet_names[0]


def _read_hdf5_mat_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Reading MATLAB v7.3 files requires the optional hdf5 dependency. "
            "Install it with: pip install \"joff[hdf5]\"."
        ) from exc

    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        handle.visititems(lambda name, obj: _collect_hdf5_array(arrays, name, obj))
    return arrays


def _collect_hdf5_array(arrays: dict[str, np.ndarray], name: str, obj: Any) -> None:
    if not hasattr(obj, "shape") or not hasattr(obj, "dtype"):
        return
    value = np.asarray(obj)
    if _is_numeric_1d_or_2d(value):
        arrays[str(name)] = _as_2d(value)


def _array_frame(array: np.ndarray, *, prefix: str) -> pd.DataFrame:
    values = _as_2d(array)
    return pd.DataFrame(values, columns=[f"{prefix}{idx}" for idx in range(values.shape[1])])


def _as_2d(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Current shape: {array.shape}.")
    return array


def _is_numeric_1d_or_2d(value: Any) -> bool:
    array = np.asarray(value)
    return array.ndim in {1, 2} and np.issubdtype(array.dtype, np.number)
