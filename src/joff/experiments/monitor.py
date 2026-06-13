"""Persistent best-result monitor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class MonitorDecision:
    """Result returned by :meth:`BestResultMonitor.update_if_better`."""

    updated: bool
    record: dict[str, Any]


class BestResultMonitor:
    """Persist the best test result using RMSE then R2 tie-break."""

    def __init__(self, path: str | Path, *, eps: float = 1e-12) -> None:
        self.path = Path(path)
        self.eps = eps

    def load(self) -> dict[str, Any] | None:
        """Load current best record if it exists."""

        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update_if_better(
        self,
        metrics: dict[str, Any],
        config: dict[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        **extra: Any,
    ) -> MonitorDecision:
        """Update persistent best record if ``metrics`` are better."""

        current = self.load()
        candidate = {
            "metrics": dict(metrics),
            "config": config or {},
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        if current is None or self._is_better(candidate["metrics"], current.get("metrics", {})):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")
            return MonitorDecision(updated=True, record=candidate)
        return MonitorDecision(updated=False, record=current)

    def _is_better(self, candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        candidate_rmse = _metric(candidate, "rmse")
        current_rmse = _metric(current, "rmse")
        if candidate_rmse is None:
            return False
        if current_rmse is None:
            return True
        if candidate_rmse < current_rmse - self.eps:
            return True
        if abs(candidate_rmse - current_rmse) <= self.eps:
            candidate_r2 = _metric(candidate, "r2")
            current_r2 = _metric(current, "r2")
            if candidate_r2 is None:
                return False
            if current_r2 is None:
                return True
            return candidate_r2 > current_r2 + self.eps
        return False


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    for candidate_key in (key, key.upper(), key.capitalize()):
        if candidate_key in metrics and metrics[candidate_key] is not None:
            value = float(metrics[candidate_key])
            if value != value:
                return None
            return value
    return None
