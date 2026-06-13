"""Global and local temporal outlier processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import TabularSeries, ensure_series


@dataclass(frozen=True)
class OutlierConfig:
    """Configuration for :class:`OutlierProcessor`."""

    enabled: bool = True
    feature_scope: str = "quality"
    scope: str | None = None
    method: str = "mad"
    detection_passes: int = 1
    iqr_multiplier: float = 1.5
    z_threshold: float = 3.0
    mad_threshold: float = 4.5
    tail_percent: float = 0.01
    tail_side: str = "both"
    use_local_temporal: bool = False
    local_window_radius: int = 8
    local_mad_threshold: float = 6.0
    local_min_abs_deviation: float = 0.0
    local_force_abs_deviation: float | None = None
    local_min_side_neighbors: int = 3
    local_require_both_sides: bool = True
    max_removal_ratio: float = 0.10
    min_remaining_rows: int = 0
    remove_train: bool = True
    remove_test: bool = False
    mark_test: bool = True

    def __post_init__(self) -> None:
        if self.scope is not None:
            object.__setattr__(self, "feature_scope", self.scope)


@dataclass(frozen=True)
class OutlierResult:
    """Result returned by :class:`OutlierProcessor`."""

    data: TabularSeries
    keep_mask: np.ndarray
    remove_mask: np.ndarray
    removed_indices: np.ndarray
    severity: np.ndarray
    summary: dict[str, int | float | str | bool]


class OutlierProcessor:
    """Detect and remove global and isolated temporal outliers."""

    def __init__(self, config: OutlierConfig | None = None, **kwargs) -> None:
        self.config = config or OutlierConfig(**kwargs)
        self._global_stats: dict[str, np.ndarray | str] | None = None

    def fit(self, data: TabularSeries | np.ndarray) -> "OutlierProcessor":
        """Fit global threshold statistics on one split."""

        series = ensure_series(data)
        if not self.config.enabled:
            self._global_stats = None
            return self
        self._global_stats = _fit_global_stats(series.feature_matrix(self.config.feature_scope), self.config)
        return self

    def transform(self, data: TabularSeries | np.ndarray, *, remove: bool = False) -> OutlierResult:
        """Mark or remove outliers using fitted threshold statistics."""

        series = ensure_series(data)
        if not self.config.enabled:
            keep_mask = np.ones(series.row_count, dtype=bool)
            return OutlierResult(
                data=series,
                keep_mask=keep_mask,
                remove_mask=~keep_mask,
                removed_indices=np.array([], dtype=int),
                severity=np.zeros(series.row_count, dtype=float),
                summary={
                    "enabled": False,
                    "policy": "remove" if remove else "mark",
                    "marked_rows": 0,
                    "removed_rows": 0,
                    "original_rows": series.row_count,
                    "remaining_rows": series.row_count,
                },
            )
        if self._global_stats is None:
            self.fit(series)
        features = series.feature_matrix(self.config.feature_scope)
        severity = _global_severity_from_stats(features, self.config, self._global_stats)
        if self.config.use_local_temporal:
            severity = np.maximum(severity, _local_temporal_severity(features, self.config))
        mark_mask = severity > 0
        remove_mask = _apply_removal_cap(mark_mask.copy(), severity, self.config) if remove else np.zeros(
            series.row_count,
            dtype=bool,
        )
        keep_mask = ~remove_mask
        if keep_mask.sum() < self.config.min_remaining_rows:
            raise ValueError(
                "Outlier removal would leave too few rows after max_removal_ratio cap. "
                f"min_remaining_rows={self.config.min_remaining_rows}, remaining={int(keep_mask.sum())}."
            )
        cleaned = series.select_rows(keep_mask)
        removed_indices = series.index[remove_mask]
        return OutlierResult(
            data=cleaned,
            keep_mask=keep_mask,
            remove_mask=remove_mask,
            removed_indices=removed_indices,
            severity=severity,
            summary={
                "enabled": True,
                "policy": "remove" if remove else "mark",
                "method": self.config.method,
                "feature_scope": self.config.feature_scope,
                "original_rows": series.row_count,
                "marked_rows": int(mark_mask.sum()),
                "removed_rows": int(remove_mask.sum()),
                "remaining_rows": cleaned.row_count,
                "max_removal_ratio": self.config.max_removal_ratio,
                "used_local_temporal": self.config.use_local_temporal,
            },
        )

    def fit_transform(self, data: TabularSeries | np.ndarray) -> OutlierResult:
        """Detect and remove outliers according to the configured policy."""

        series = ensure_series(data)
        if not self.config.enabled:
            keep_mask = np.ones(series.row_count, dtype=bool)
            return OutlierResult(
                data=series,
                keep_mask=keep_mask,
                remove_mask=~keep_mask,
                removed_indices=np.array([], dtype=int),
                severity=np.zeros(series.row_count, dtype=float),
                summary={"enabled": False, "removed_rows": 0, "original_rows": series.row_count},
            )

        self.fit(series)
        original_rows = series.row_count
        global_keep = np.ones(original_rows, dtype=bool)
        severity = np.zeros(original_rows, dtype=float)
        working = series
        for _ in range(max(1, self.config.detection_passes)):
            features = working.feature_matrix(self.config.feature_scope)
            local_severity = _global_severity(features, self.config)
            if self.config.use_local_temporal:
                local_severity = np.maximum(local_severity, _local_temporal_severity(features, self.config))
            pass_remove = local_severity > 0
            if not pass_remove.any():
                break
            original_positions = working.index.astype(int)
            severity[original_positions] = np.maximum(severity[original_positions], local_severity)
            remove_original = original_positions[pass_remove]
            global_keep[remove_original] = False
            working = series.select_rows(global_keep)
            if working.row_count <= self.config.min_remaining_rows:
                raise ValueError(
                    "Outlier removal would leave too few rows. "
                    f"min_remaining_rows={self.config.min_remaining_rows}, current={working.row_count}."
                )

        remove_mask = ~global_keep
        remove_mask = _apply_removal_cap(remove_mask, severity, self.config)
        keep_mask = ~remove_mask
        if keep_mask.sum() < self.config.min_remaining_rows:
            raise ValueError(
                "Outlier removal would leave too few rows after max_removal_ratio cap. "
                f"min_remaining_rows={self.config.min_remaining_rows}, remaining={int(keep_mask.sum())}."
            )
        cleaned = series.select_rows(keep_mask)
        removed_indices = series.index[remove_mask]
        return OutlierResult(
            data=cleaned,
            keep_mask=keep_mask,
            remove_mask=remove_mask,
            removed_indices=removed_indices,
            severity=severity,
            summary={
                "enabled": True,
                "method": self.config.method,
                "feature_scope": self.config.feature_scope,
                "original_rows": original_rows,
                "marked_rows": int((severity > 0).sum()),
                "removed_rows": int(remove_mask.sum()),
                "remaining_rows": cleaned.row_count,
                "max_removal_ratio": self.config.max_removal_ratio,
                "policy": "remove",
                "used_local_temporal": self.config.use_local_temporal,
            },
        )


def _fit_global_stats(features: np.ndarray, config: OutlierConfig) -> dict[str, np.ndarray | str]:
    method = config.method.strip().lower()
    if method == "iqr":
        q1 = np.nanpercentile(features, 25, axis=0)
        q3 = np.nanpercentile(features, 75, axis=0)
        return {"method": method, "q1": q1, "q3": q3, "iqr": np.maximum(q3 - q1, 1e-12)}
    if method == "zscore":
        return {
            "method": method,
            "mean": np.nanmean(features, axis=0),
            "std": np.maximum(np.nanstd(features, axis=0), 1e-12),
        }
    if method == "mad":
        median = np.nanmedian(features, axis=0)
        mad = np.nanmedian(np.abs(features - median), axis=0)
        return {"method": method, "median": median, "mad": np.maximum(mad, 1e-12)}
    if method == "tail_percent":
        pct = config.tail_percent * 100 if config.tail_percent <= 1 else config.tail_percent
        lower = np.nanpercentile(features, pct, axis=0)
        upper = np.nanpercentile(features, 100 - pct, axis=0)
        side = config.tail_side.strip().lower()
        if side not in {"min", "max", "both"}:
            raise ValueError("tail_side must be one of: min, max, both.")
        return {
            "method": method,
            "lower": lower,
            "upper": upper,
            "scale": np.maximum(upper - lower, 1e-12),
            "side": side,
        }
    raise ValueError(
        f"Unknown outlier method {config.method!r}. Legal options are: iqr, zscore, mad, tail_percent."
    )


def _global_severity_from_stats(
    features: np.ndarray,
    config: OutlierConfig,
    stats: dict[str, np.ndarray | str] | None,
) -> np.ndarray:
    if stats is None:
        return np.zeros(features.shape[0], dtype=float)
    method = str(stats["method"])
    if method == "iqr":
        q1 = np.asarray(stats["q1"])
        q3 = np.asarray(stats["q3"])
        iqr = np.asarray(stats["iqr"])
        score = np.maximum((q1 - features) / iqr, (features - q3) / iqr) - config.iqr_multiplier
    elif method == "zscore":
        score = np.abs((features - np.asarray(stats["mean"])) / np.asarray(stats["std"])) - config.z_threshold
    elif method == "mad":
        robust_z = 0.6745 * np.abs(features - np.asarray(stats["median"])) / np.asarray(stats["mad"])
        score = robust_z - config.mad_threshold
    elif method == "tail_percent":
        lower = np.asarray(stats["lower"])
        upper = np.asarray(stats["upper"])
        scale = np.asarray(stats["scale"])
        side = str(stats["side"])
        if side == "min":
            score = (lower - features) / scale
        elif side == "max":
            score = (features - upper) / scale
        elif side == "both":
            score = np.maximum((lower - features) / scale, (features - upper) / scale)
        else:
            raise ValueError("tail_side must be one of: min, max, both.")
    else:
        raise ValueError(
            f"Unknown outlier method {method!r}. Legal options are: iqr, zscore, mad, tail_percent."
        )
    return np.maximum(np.nanmax(score, axis=1), 0.0)


def _global_severity(features: np.ndarray, config: OutlierConfig) -> np.ndarray:
    method = config.method.strip().lower()
    if method == "iqr":
        q1 = np.nanpercentile(features, 25, axis=0)
        q3 = np.nanpercentile(features, 75, axis=0)
        iqr = np.maximum(q3 - q1, 1e-12)
        lower = q1 - config.iqr_multiplier * iqr
        upper = q3 + config.iqr_multiplier * iqr
        score = np.maximum((lower - features) / iqr, (features - upper) / iqr)
    elif method == "zscore":
        mean = np.nanmean(features, axis=0)
        std = np.maximum(np.nanstd(features, axis=0), 1e-12)
        score = np.abs((features - mean) / std) - config.z_threshold
    elif method == "mad":
        median = np.nanmedian(features, axis=0)
        mad = np.maximum(np.nanmedian(np.abs(features - median), axis=0), 1e-12)
        robust_z = 0.6745 * np.abs(features - median) / mad
        score = robust_z - config.mad_threshold
    elif method == "tail_percent":
        pct = config.tail_percent * 100 if config.tail_percent <= 1 else config.tail_percent
        lower = np.nanpercentile(features, pct, axis=0)
        upper = np.nanpercentile(features, 100 - pct, axis=0)
        scale = np.maximum(upper - lower, 1e-12)
        side = config.tail_side.strip().lower()
        if side == "min":
            score = (lower - features) / scale
        elif side == "max":
            score = (features - upper) / scale
        elif side == "both":
            score = np.maximum((lower - features) / scale, (features - upper) / scale)
        else:
            raise ValueError("tail_side must be one of: min, max, both.")
    else:
        raise ValueError(
            f"Unknown outlier method {config.method!r}. Legal options are: iqr, zscore, mad, tail_percent."
        )
    return np.maximum(np.nanmax(score, axis=1), 0.0)


def _local_temporal_severity(features: np.ndarray, config: OutlierConfig) -> np.ndarray:
    rows = features.shape[0]
    severity = np.zeros(rows, dtype=float)
    radius = max(1, config.local_window_radius)
    for row in range(rows):
        left = features[max(0, row - radius) : row]
        right = features[row + 1 : min(rows, row + radius + 1)]
        if config.local_require_both_sides and (
            left.shape[0] < config.local_min_side_neighbors
            or right.shape[0] < config.local_min_side_neighbors
        ):
            continue
        if not config.local_require_both_sides and (
            left.shape[0] + right.shape[0] < config.local_min_side_neighbors
        ):
            continue
        neighbors = np.concatenate([left, right], axis=0)
        center = features[row]
        median = np.nanmedian(neighbors, axis=0)
        abs_deviation = np.abs(center - median)
        mad = np.nanmedian(np.abs(neighbors - median), axis=0)
        scale = np.maximum(mad, 1e-12)
        score = 0.6745 * abs_deviation / scale
        candidate = (score > config.local_mad_threshold) & (
            abs_deviation >= config.local_min_abs_deviation
        )
        if config.local_force_abs_deviation is not None:
            candidate |= abs_deviation >= config.local_force_abs_deviation
        if candidate.any():
            severity[row] = float(np.nanmax(score[candidate]))
    return severity


def _apply_removal_cap(
    remove_mask: np.ndarray, severity: np.ndarray, config: OutlierConfig
) -> np.ndarray:
    max_remove = int(np.floor(remove_mask.shape[0] * config.max_removal_ratio))
    if config.max_removal_ratio > 0 and max_remove == 0 and remove_mask.any():
        max_remove = 1
    current = int(remove_mask.sum())
    if current <= max_remove:
        return remove_mask
    removable = np.flatnonzero(remove_mask)
    order = np.argsort(severity[removable])[::-1]
    keep_remove = removable[order[:max_remove]]
    capped = np.zeros_like(remove_mask, dtype=bool)
    capped[keep_remove] = True
    return capped
