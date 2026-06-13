"""Common data pipeline containers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TabularSeries:
    """Column-oriented process table used by pipeline processors."""

    x: np.ndarray
    y: np.ndarray | None = None
    u: np.ndarray | None = None
    quality: np.ndarray | None = None
    labels: np.ndarray | None = None
    column_names: list[str] | None = None
    index: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _as_2d_float(self.x))
        row_count = self.x.shape[0]
        for name in ("y", "u", "quality"):
            value = getattr(self, name)
            if value is not None:
                converted = _as_2d_float(value)
                if converted.shape[0] != row_count:
                    raise ValueError(
                        f"TabularSeries field {name!r} has {converted.shape[0]} rows, "
                        f"but x has {row_count}. Legal input must share row count."
                    )
                object.__setattr__(self, name, converted)
        if self.labels is not None:
            labels = np.asarray(self.labels)
            if labels.shape[0] != row_count:
                raise ValueError(
                    f"labels has {labels.shape[0]} rows, but x has {row_count}. "
                    "Legal input must share row count."
                )
            object.__setattr__(self, "labels", labels)
        if self.index is None:
            object.__setattr__(self, "index", np.arange(row_count))
        else:
            index = np.asarray(self.index)
            if index.shape[0] != row_count:
                raise ValueError(
                    f"index has {index.shape[0]} rows, but x has {row_count}. "
                    "Legal input must share row count."
                )
            object.__setattr__(self, "index", index)

    @property
    def row_count(self) -> int:
        """Number of rows."""

        return int(self.x.shape[0])

    def select_rows(self, mask: np.ndarray) -> "TabularSeries":
        """Return a new series containing rows selected by ``mask``."""

        mask = np.asarray(mask, dtype=bool)
        return replace(
            self,
            x=self.x[mask],
            y=None if self.y is None else self.y[mask],
            u=None if self.u is None else self.u[mask],
            quality=None if self.quality is None else self.quality[mask],
            labels=None if self.labels is None else self.labels[mask],
            index=self.index[mask],
        )

    def feature_matrix(self, scope: str = "input") -> np.ndarray:
        """Return a feature matrix for an outlier or scaling scope."""

        scope = scope.strip().lower()
        if scope in {"input", "x"}:
            return self.x
        if scope in {"target", "y"}:
            if self.y is None:
                raise ValueError("feature_scope='target' requires y to be present.")
            return self.y
        if scope == "quality":
            if self.quality is not None:
                return self.quality
            if self.y is not None:
                return self.y
            raise ValueError("feature_scope='quality' requires quality or y to be present.")
        if scope in {"u", "control"}:
            if self.u is None:
                raise ValueError("feature_scope='u' requires u to be present.")
            return self.u
        if scope == "input_target":
            if self.y is None:
                return self.x
            return np.concatenate([self.x, self.y], axis=1)
        raise ValueError(
            f"Unknown feature_scope {scope!r}. Legal options are: input, target, quality, u, "
            "input_target."
        )


def ensure_series(data: TabularSeries | np.ndarray | Any) -> TabularSeries:
    """Convert array-like input to :class:`TabularSeries`."""

    if isinstance(data, TabularSeries):
        return data
    return TabularSeries(x=np.asarray(data, dtype=float))


def replace_array(series: TabularSeries, field: str, value: np.ndarray) -> TabularSeries:
    """Return ``series`` with one array field replaced."""

    return replace(series, **{field: value})


def finite_row_mask(series: TabularSeries) -> np.ndarray:
    """Return rows where all numeric fields are finite."""

    mask = np.ones(series.row_count, dtype=bool)
    for name in ("x", "y", "u", "quality"):
        value = getattr(series, name)
        if value is not None:
            mask &= np.isfinite(value).all(axis=1)
    return mask


def _as_2d_float(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Current shape: {array.shape}.")
    return array

