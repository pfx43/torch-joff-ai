"""Train-only normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationSummary:
    """Summary of fitted scaler statistics."""

    method: str
    shape: tuple[int, int]
    constant_columns: int


class Normalizer:
    """Fit on train data and transform train/eval/test arrays without leakage."""

    def __init__(self, method: str = "standard", feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.method = method
        self.feature_range = feature_range
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.constant_mask_: np.ndarray | None = None
        self.summary_: NormalizationSummary | None = None

    def fit(self, train: np.ndarray) -> "Normalizer":
        """Fit scaler statistics from train data only."""

        array = _as_2d(train)
        method = self.method.strip().lower()
        if method == "none":
            self.center_ = np.zeros(array.shape[1], dtype=float)
            self.scale_ = np.ones(array.shape[1], dtype=float)
            self.constant_mask_ = np.zeros(array.shape[1], dtype=bool)
        elif method == "standard":
            self.center_ = np.nanmean(array, axis=0)
            raw_scale = np.nanstd(array, axis=0)
            self.constant_mask_ = raw_scale <= 1e-12
            self.scale_ = np.where(self.constant_mask_, 1.0, raw_scale)
        elif method == "minmax":
            self.min_ = np.nanmin(array, axis=0)
            self.max_ = np.nanmax(array, axis=0)
            raw_scale = self.max_ - self.min_
            self.constant_mask_ = raw_scale <= 1e-12
            self.scale_ = np.where(self.constant_mask_, 1.0, raw_scale)
            self.center_ = self.min_
        else:
            raise ValueError(
                f"Unknown normalization method {self.method!r}. Legal options are: none, standard, minmax."
            )
        self.summary_ = NormalizationSummary(
            method=method,
            shape=tuple(array.shape),
            constant_columns=int(self.constant_mask_.sum()),
        )
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data with fitted train statistics."""

        self._check_fitted()
        array = _as_2d(data)
        method = self.method.strip().lower()
        if method == "none":
            return array.copy()
        if method == "standard":
            return (array - self.center_) / self.scale_
        low, high = self.feature_range
        return ((array - self.center_) / self.scale_) * (high - low) + low

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Invert a previous transform."""

        self._check_fitted()
        array = _as_2d(data)
        method = self.method.strip().lower()
        if method == "none":
            return array.copy()
        if method == "standard":
            return array * self.scale_ + self.center_
        low, high = self.feature_range
        return ((array - low) / (high - low)) * self.scale_ + self.center_

    def fit_transform_train_test(
        self,
        train: np.ndarray,
        test: np.ndarray,
        eval_data: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit on train and transform train/test, plus optional eval data."""

        self.fit(train)
        train_out = self.transform(train)
        test_out = self.transform(test)
        if eval_data is None:
            return train_out, test_out
        return train_out, self.transform(eval_data), test_out

    def summary(self) -> dict[str, int | str | tuple[int, int]]:
        """Return a serializable fit summary."""

        self._check_fitted()
        assert self.summary_ is not None
        return {
            "method": self.summary_.method,
            "shape": self.summary_.shape,
            "constant_columns": self.summary_.constant_columns,
        }

    def _check_fitted(self) -> None:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Normalizer must be fit before transform or inverse_transform.")


def _as_2d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D data. Current shape: {array.shape}.")
    return array

