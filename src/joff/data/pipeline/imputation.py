"""Imputation mask generation and dataset wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ImputationMaskConfig:
    """Configuration for artificial imputation masks."""

    strategy: str = "random"
    missing_rate: float = 0.2
    seed: int = 42
    fill_value: float = 0.0
    block_length: int = 4
    append_mask: bool = True


@dataclass(frozen=True)
class ImputationMaskResult:
    """Masked data arrays and summary."""

    dataset: "ImputationDataset"
    summary: dict[str, Any]


class ImputationMasker:
    """Generate artificial missing masks after data has been split."""

    def __init__(self, config: ImputationMaskConfig | None = None, **kwargs: Any) -> None:
        self.config = config or ImputationMaskConfig(**kwargs)
        if not 0.0 < self.config.missing_rate < 1.0:
            raise ValueError(
                "missing_rate must be between 0 and 1. "
                f"Current input: {self.config.missing_rate}."
            )
        if self.config.block_length <= 0:
            raise ValueError(
                f"block_length must be positive. Current input: {self.config.block_length}."
            )

    def transform_split(
        self,
        data: np.ndarray,
        *,
        split: str,
        seed_offset: int = 0,
    ) -> ImputationMaskResult:
        """Generate a split-local mask and return an imputation dataset."""

        array = _as_2d(data)
        rng = np.random.default_rng(self.config.seed + seed_offset)
        eval_mask = self._mask(array.shape, rng)
        observed_mask = ~eval_mask
        corrupted = array.copy()
        corrupted[eval_mask] = self.config.fill_value
        dataset = ImputationDataset(
            corrupted=corrupted,
            target=array,
            observed_mask=observed_mask,
            eval_mask=eval_mask,
            append_mask=self.config.append_mask,
        )
        summary = {
            "split": split,
            "strategy": self.config.strategy,
            "missing_rate": self.config.missing_rate,
            "seed": self.config.seed + seed_offset,
            "rows": int(array.shape[0]),
            "features": int(array.shape[1]),
            "masked_entries": int(eval_mask.sum()),
            "total_entries": int(eval_mask.size),
            "actual_missing_rate": float(eval_mask.mean()),
            "append_mask": self.config.append_mask,
        }
        return ImputationMaskResult(dataset=dataset, summary=summary)

    def _mask(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        strategy = self.config.strategy.strip().lower()
        if strategy in {"random", "mcar"}:
            mask = rng.random(shape) < self.config.missing_rate
        elif strategy in {"block", "block_missing"}:
            mask = _block_mask(
                shape,
                missing_rate=self.config.missing_rate,
                block_length=self.config.block_length,
                rng=rng,
            )
        else:
            raise ValueError(
                f"Unknown imputation mask strategy {self.config.strategy!r}. "
                "Legal options are: random, mcar, block, block_missing."
            )
        if not mask.any():
            row = int(rng.integers(0, shape[0]))
            col = int(rng.integers(0, shape[1]))
            mask[row, col] = True
        if mask.all():
            mask[0, 0] = False
        return mask


class ImputationDataset(Dataset):
    """Torch dataset for imputation tasks."""

    def __init__(
        self,
        *,
        corrupted: np.ndarray,
        target: np.ndarray,
        observed_mask: np.ndarray,
        eval_mask: np.ndarray,
        append_mask: bool = True,
    ) -> None:
        self.corrupted = _as_2d(corrupted).astype(np.float32)
        self.target = _as_2d(target).astype(np.float32)
        self.observed_mask = np.asarray(observed_mask, dtype=bool)
        self.eval_mask = np.asarray(eval_mask, dtype=bool)
        if self.corrupted.shape != self.target.shape:
            raise ValueError("corrupted and target must share shape.")
        if self.observed_mask.shape != self.target.shape or self.eval_mask.shape != self.target.shape:
            raise ValueError("observed_mask and eval_mask must match target shape.")
        self.append_mask = append_mask

    def __len__(self) -> int:
        """Return sample count."""

        return int(self.target.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one imputation sample."""

        corrupted = torch.as_tensor(self.corrupted[idx], dtype=torch.float32)
        observed_mask = torch.as_tensor(self.observed_mask[idx], dtype=torch.float32)
        features = torch.cat([corrupted, observed_mask], dim=0) if self.append_mask else corrupted
        return {
            "x": features,
            "target": torch.as_tensor(self.target[idx], dtype=torch.float32),
            "corrupted": corrupted,
            "observed_mask": observed_mask,
            "eval_mask": torch.as_tensor(self.eval_mask[idx], dtype=torch.float32),
        }


def _block_mask(
    shape: tuple[int, int],
    *,
    missing_rate: float,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows, cols = shape
    target_count = max(1, int(round(rows * cols * missing_rate)))
    mask = np.zeros(shape, dtype=bool)
    while int(mask.sum()) < target_count:
        col = int(rng.integers(0, cols))
        start = int(rng.integers(0, rows))
        stop = min(rows, start + block_length)
        mask[start:stop, col] = True
    return mask


def _as_2d(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D array. Current shape: {array.shape}.")
    return array
