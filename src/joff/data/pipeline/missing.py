"""Missing and non-finite value processing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import TabularSeries, ensure_series, finite_row_mask


@dataclass(frozen=True)
class MissingValueResult:
    """Result returned by :class:`MissingValueProcessor`."""

    data: TabularSeries
    keep_mask: np.ndarray
    summary: dict[str, int | str]


class MissingValueProcessor:
    """Handle NaN/Inf rows before model-specific data construction."""

    def __init__(self, strategy: str = "interpolate_then_drop", interpolation: str = "linear") -> None:
        self.strategy = strategy
        self.interpolation = interpolation

    def fit_transform(self, data: TabularSeries | np.ndarray) -> MissingValueResult:
        """Process missing values and return cleaned data plus summary."""

        series = ensure_series(data)
        original_rows = series.row_count
        if self.strategy == "none":
            keep_mask = finite_row_mask(series)
            return MissingValueResult(
                data=series,
                keep_mask=keep_mask,
                summary={
                    "strategy": self.strategy,
                    "original_rows": original_rows,
                    "removed_rows": 0,
                    "remaining_non_finite_rows": int((~keep_mask).sum()),
                },
            )
        if self.strategy in {"interpolate_then_drop", "ffill_bfill_then_drop"}:
            series = _interpolate_series(series, method=self.interpolation, ffill=self.strategy.startswith("ffill"))
        elif self.strategy != "drop":
            raise ValueError(
                f"Unknown missing value strategy {self.strategy!r}. Legal options are: "
                "none, drop, interpolate_then_drop, ffill_bfill_then_drop."
            )
        keep_mask = finite_row_mask(series)
        cleaned = series.select_rows(keep_mask)
        return MissingValueResult(
            data=cleaned,
            keep_mask=keep_mask,
            summary={
                "strategy": self.strategy,
                "original_rows": original_rows,
                "removed_rows": int(original_rows - cleaned.row_count),
                "remaining_non_finite_rows": int((~finite_row_mask(cleaned)).sum()),
            },
        )


def _interpolate_series(series: TabularSeries, *, method: str, ffill: bool) -> TabularSeries:
    updates = {}
    for field in ("x", "y", "u", "quality"):
        value = getattr(series, field)
        if value is None:
            continue
        frame = pd.DataFrame(value).replace([np.inf, -np.inf], np.nan)
        if ffill:
            frame = frame.ffill().bfill()
        else:
            frame = frame.interpolate(method=method, limit_direction="both").ffill().bfill()
        updates[field] = frame.to_numpy(dtype=float)
    return TabularSeries(
        x=updates.get("x", series.x),
        y=updates.get("y", series.y),
        u=updates.get("u", series.u),
        quality=updates.get("quality", series.quality),
        labels=series.labels,
        column_names=series.column_names,
        index=series.index,
    )

