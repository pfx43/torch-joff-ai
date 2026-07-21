"""Sequential and dynamic distribution splitters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .windowing import DynamicWindowDataset, DynamicWindowSubset


@dataclass(frozen=True)
class SplitResult:
    """Dataset split plus indices and summary."""

    train: DynamicWindowSubset
    eval: DynamicWindowSubset | None
    test: DynamicWindowSubset
    split_indices: dict[str, np.ndarray]
    summary: dict[str, object]


class SequentialSplitter:
    """Split dynamic samples in chronological order."""

    def __init__(
        self,
        train_ratio: float = 0.8,
        eval_ratio: float = 0.0,
        test_ratio: float | None = None,
    ) -> None:
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = 1.0 - train_ratio - eval_ratio if test_ratio is None else test_ratio

    def split(self, dataset: DynamicWindowDataset) -> SplitResult:
        """Split a dynamic window dataset without shuffling."""

        n = len(dataset)
        if n < 3:
            raise ValueError(
                f"Sequential split requires at least 3 dynamic samples. Current input: {n}."
            )
        _validate_ratios(self.train_ratio, self.eval_ratio, self.test_ratio)
        train_end = max(1, int(round(n * self.train_ratio)))
        eval_count = int(round(n * self.eval_ratio))
        test_start = min(n - 1, train_end + eval_count)
        train_idx = np.arange(0, train_end)
        eval_idx = np.arange(train_end, test_start)
        test_idx = np.arange(test_start, n)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Sequential split produced an empty train or test split. n={n}, "
                f"train_ratio={self.train_ratio}, eval_ratio={self.eval_ratio}."
            )
        return SplitResult(
            train=dataset.subset(train_idx),
            eval=dataset.subset(eval_idx) if len(eval_idx) else None,
            test=dataset.subset(test_idx),
            split_indices={"train": train_idx, "eval": eval_idx, "test": test_idx},
            summary={
                "type": "sequential",
                "n_samples": n,
                "train_samples": int(len(train_idx)),
                "eval_samples": int(len(eval_idx)),
                "test_samples": int(len(test_idx)),
            },
        )


class DynamicDistributionSplitter:
    """Split dynamic samples by contiguous slices while matching target distribution."""

    def __init__(
        self,
        train_ratio: float = 0.8,
        eval_ratio: float = 0.0,
        test_ratio: float = 0.2,
        slice_length: int = 64,
        min_slice_length: int | None = None,
        distribution_weight: float = 1.0,
        ratio_weight: float = 1.0,
        test_range_weight: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.slice_length = slice_length
        self.min_slice_length = min_slice_length
        self.distribution_weight = distribution_weight
        self.ratio_weight = ratio_weight
        self.test_range_weight = test_range_weight
        self.seed = seed

    def split(
        self,
        dataset: DynamicWindowDataset,
        target_values: np.ndarray | None = None,
    ) -> SplitResult:
        """Split dynamic samples by greedily selecting holdout slices."""

        n = len(dataset)
        if n < 3:
            raise ValueError(
                f"Dynamic distribution split requires at least 3 samples. Current input: {n}."
            )
        if self.slice_length <= 0:
            raise ValueError(f"slice_length must be positive. Current input: {self.slice_length}.")
        _validate_ratios(self.train_ratio, self.eval_ratio, self.test_ratio)
        target = dataset.target_values() if target_values is None else _as_2d(target_values)
        if target.shape[0] != n:
            raise ValueError(
                f"target_values must have one row per dynamic sample. Current rows: "
                f"{target.shape[0]}, expected: {n}."
            )
        slices = _make_slices(n, self.slice_length, self.min_slice_length)
        holdout_goal = max(1, int(round(n * (self.eval_ratio + self.test_ratio))))
        holdout_slices = _choose_holdout_slices(
            slices,
            target,
            holdout_goal=holdout_goal,
            distribution_weight=self.distribution_weight,
            ratio_weight=self.ratio_weight,
            range_weight=self.test_range_weight,
            seed=self.seed,
        )
        holdout_idx = np.concatenate([np.arange(s.start, s.stop) for s in holdout_slices])
        holdout_idx.sort()
        train_idx = np.setdiff1d(np.arange(n), holdout_idx)
        eval_count = int(round(n * self.eval_ratio))
        eval_idx = holdout_idx[:eval_count]
        test_idx = holdout_idx[eval_count:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                "Dynamic distribution split produced an empty train or test split. "
                f"n={n}, train={len(train_idx)}, test={len(test_idx)}."
            )
        slice_summary = [_slice_record(i, s, target) for i, s in enumerate(slices)]
        distribution_summary = _distribution_summary(
            target,
            split_indices={"train": train_idx, "eval": eval_idx, "test": test_idx},
        )
        return SplitResult(
            train=dataset.subset(train_idx),
            eval=dataset.subset(eval_idx) if len(eval_idx) else None,
            test=dataset.subset(test_idx),
            split_indices={"train": train_idx, "eval": eval_idx, "test": test_idx},
            summary={
                "type": "dynamic_distribution",
                "n_samples": n,
                "slice_length": self.slice_length,
                "n_slices": len(slices),
                "train_samples": int(len(train_idx)),
                "eval_samples": int(len(eval_idx)),
                "test_samples": int(len(test_idx)),
                "slice_summary": slice_summary,
                "distribution_summary": distribution_summary,
            },
        )


def _validate_ratios(train_ratio: float, eval_ratio: float, test_ratio: float) -> None:
    total = train_ratio + eval_ratio + test_ratio
    if min(train_ratio, eval_ratio, test_ratio) < 0 or not np.isclose(total, 1.0):
        raise ValueError(
            f"Split ratios must be non-negative and sum to 1. Current input: "
            f"train={train_ratio}, eval={eval_ratio}, test={test_ratio}, total={total}."
        )


def _make_slices(n: int, slice_length: int, min_slice_length: int | None) -> list[slice]:
    min_len = min_slice_length or max(1, slice_length // 2)
    slices: list[slice] = []
    start = 0
    while start < n:
        stop = min(n, start + slice_length)
        if stop - start < min_len and slices:
            previous = slices[-1]
            slices[-1] = slice(previous.start, stop)
        else:
            slices.append(slice(start, stop))
        start = stop
    return slices


def _choose_holdout_slices(
    slices: list[slice],
    target: np.ndarray,
    *,
    holdout_goal: int,
    distribution_weight: float,
    ratio_weight: float,
    range_weight: float,
    seed: int,
) -> list[slice]:
    rng = np.random.default_rng(seed)
    remaining = list(slices)
    rng.shuffle(remaining)
    selected: list[slice] = []
    while remaining and _slice_count(selected) < holdout_goal:
        best_slice = min(
            remaining,
            key=lambda candidate: _holdout_score(
                selected + [candidate],
                target,
                holdout_goal=holdout_goal,
                distribution_weight=distribution_weight,
                ratio_weight=ratio_weight,
                range_weight=range_weight,
            ),
        )
        selected.append(best_slice)
        remaining.remove(best_slice)
    return sorted(selected, key=lambda item: item.start)


def _holdout_score(
    selected: list[slice],
    target: np.ndarray,
    *,
    holdout_goal: int,
    distribution_weight: float,
    ratio_weight: float,
    range_weight: float,
) -> float:
    holdout_idx = np.concatenate([np.arange(s.start, s.stop) for s in selected])
    train_idx = np.setdiff1d(np.arange(target.shape[0]), holdout_idx)
    if train_idx.size == 0:
        return float("inf")
    holdout = target[holdout_idx]
    train = target[train_idx]
    distribution = np.linalg.norm(_descriptor(holdout) - _descriptor(train))
    ratio_error = abs(holdout_idx.size - holdout_goal) / max(holdout_goal, 1)
    full_range = np.maximum(np.nanmax(target, axis=0) - np.nanmin(target, axis=0), 1e-12)
    holdout_range = np.nanmax(holdout, axis=0) - np.nanmin(holdout, axis=0)
    range_penalty = float(np.mean(np.maximum(0.0, 1.0 - holdout_range / full_range)))
    return (
        distribution_weight * float(distribution)
        + ratio_weight * float(ratio_error)
        + range_weight * range_penalty
    )


def _descriptor(values: np.ndarray) -> np.ndarray:
    percentiles = np.nanpercentile(values, [10, 25, 50, 75, 90], axis=0).reshape(-1)
    stats = [
        np.nanmean(values, axis=0).reshape(-1),
        np.nanstd(values, axis=0).reshape(-1),
        np.nanmin(values, axis=0).reshape(-1),
        percentiles,
        np.nanmax(values, axis=0).reshape(-1),
    ]
    return np.concatenate(stats)


def _slice_count(slices: list[slice]) -> int:
    return int(sum(s.stop - s.start for s in slices))


def _slice_record(idx: int, item: slice, target: np.ndarray) -> dict[str, object]:
    values = target[item]
    return {
        "slice": idx,
        "start": item.start,
        "stop": item.stop,
        "count": item.stop - item.start,
        "mean": np.nanmean(values, axis=0).tolist(),
        "std": np.nanstd(values, axis=0).tolist(),
        "min": np.nanmin(values, axis=0).tolist(),
        "p50": np.nanpercentile(values, 50, axis=0).tolist(),
        "max": np.nanmax(values, axis=0).tolist(),
    }


def _distribution_summary(
    target: np.ndarray,
    *,
    split_indices: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, indices in split_indices.items():
        if indices.size == 0:
            rows.append({"split": split, "count": 0})
            continue
        values = target[indices]
        rows.append(
            {
                "split": split,
                "count": int(indices.size),
                "mean": np.nanmean(values, axis=0).tolist(),
                "std": np.nanstd(values, axis=0).tolist(),
                "min": np.nanmin(values, axis=0).tolist(),
                "p10": np.nanpercentile(values, 10, axis=0).tolist(),
                "p25": np.nanpercentile(values, 25, axis=0).tolist(),
                "p50": np.nanpercentile(values, 50, axis=0).tolist(),
                "p75": np.nanpercentile(values, 75, axis=0).tolist(),
                "p90": np.nanpercentile(values, 90, axis=0).tolist(),
                "max": np.nanmax(values, axis=0).tolist(),
            }
        )
    return rows


def _as_2d(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected 1D or 2D target_values. Current shape: {array.shape}.")
    return array
