"""Fault detection evaluator with thresholded statistic procedures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class FaultDetectionReport:
    """Fault detection metrics and threshold."""

    threshold: float
    metrics: dict[str, float]
    thresholds: dict[str, float] = field(default_factory=dict)
    procedure_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report."""

        return {
            "threshold": self.threshold,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "procedure_metrics": self.procedure_metrics,
        }


class FaultDetectionEvaluator:
    """Fit a normal threshold and compute FAR/MDR-style metrics."""

    def __init__(
        self,
        expected_far: float = 0.005,
        threshold: float | None = None,
        procedures: list[str] | tuple[str, ...] | None = None,
        covariance_method: str = "lstsq",
    ) -> None:
        if not 0 <= expected_far < 1:
            raise ValueError(
                f"expected_far must be in [0, 1). Current input: {expected_far}."
            )
        self.expected_far = expected_far
        self.threshold = threshold
        self.procedures = tuple(_parse_procedure(item) for item in (procedures or ()))
        self.covariance_method = covariance_method
        self.thresholds_: dict[str, float] = {}
        self.references_: dict[str, dict[str, np.ndarray]] = {}

    def fit(self, normal_scores: Any, **data: Any) -> "FaultDetectionEvaluator":
        """Fit threshold from normal scores."""

        if self.procedures:
            payload = _payload(normal_scores, **data)
            self.thresholds_ = {}
            self.references_ = {}
            for procedure in self.procedures:
                reference = _fit_reference(payload, procedure, self.covariance_method)
                scores = _procedure_scores(payload, procedure, reference)
                self.references_[procedure.name] = reference
                self.thresholds_[procedure.name] = _threshold(
                    scores,
                    expected_far=self.expected_far,
                    method=procedure.threshold,
                )
            self.threshold = next(iter(self.thresholds_.values())) if self.thresholds_ else None
            return self
        scores = _as_1d(normal_scores)
        percentile = 100.0 * (1.0 - self.expected_far)
        self.threshold = float(np.nanpercentile(scores, percentile))
        return self

    def evaluate(self, scores: Any, labels: Any, **data: Any) -> FaultDetectionReport:
        """Evaluate anomaly scores against binary labels where 1 means fault."""

        if self.procedures:
            if not self.thresholds_:
                raise RuntimeError("FaultDetectionEvaluator must be fit before evaluate.")
            payload = _payload(scores, **data)
            label_array = _as_1d(labels).astype(int)
            procedure_metrics: dict[str, dict[str, float]] = {}
            thresholds: dict[str, float] = {}
            for procedure in self.procedures:
                reference = self.references_[procedure.name]
                procedure_scores = _procedure_scores(payload, procedure, reference)
                if procedure_scores.shape[0] != label_array.shape[0]:
                    raise ValueError(
                        f"scores and labels must share length. Current lengths: "
                        f"{procedure_scores.shape[0]} and {label_array.shape[0]}."
                    )
                threshold = self.thresholds_[procedure.name]
                thresholds[procedure.name] = threshold
                procedure_metrics[procedure.name] = _metrics_from_scores(
                    procedure_scores,
                    label_array,
                    threshold,
                )
            aggregate = _aggregate_procedure_metrics(procedure_metrics)
            return FaultDetectionReport(
                threshold=float(next(iter(thresholds.values()))),
                thresholds=thresholds,
                metrics=aggregate,
                procedure_metrics=procedure_metrics,
            )
        if self.threshold is None:
            raise RuntimeError("FaultDetectionEvaluator must be fit before evaluate.")
        score_array = _as_1d(scores)
        label_array = _as_1d(labels).astype(int)
        if score_array.shape[0] != label_array.shape[0]:
            raise ValueError(
                f"scores and labels must share length. Current lengths: "
                f"{score_array.shape[0]} and {label_array.shape[0]}."
            )
        report = FaultDetectionReport(
            threshold=float(self.threshold),
            metrics=_metrics_from_scores(score_array, label_array, float(self.threshold)),
        )
        return report

    def fit_evaluate(self, normal_scores: Any, test_scores: Any, labels: Any) -> FaultDetectionReport:
        """Fit on normal scores and evaluate test scores."""

        return self.fit(normal_scores).evaluate(test_scores, labels)


def reconstruction_scores(x_true: Any, x_pred: Any) -> np.ndarray:
    """Compute per-row squared reconstruction error scores."""

    true = _as_2d(x_true)
    pred = _as_2d(x_pred)
    if true.shape != pred.shape:
        raise ValueError(
            f"x_true and x_pred must have the same shape. Current shapes: {true.shape} and {pred.shape}."
        )
    return np.mean((true - pred) ** 2, axis=1)


@dataclass(frozen=True)
class _Procedure:
    mode: str
    statistic: str
    threshold: str

    @property
    def name(self) -> str:
        return f"{self.mode}-{self.statistic}-{self.threshold}"


def _parse_procedure(value: str) -> _Procedure:
    parts = value.strip().lower().split("-")
    if len(parts) != 3:
        raise ValueError(
            f"Fault detection procedure {value!r} must have form mm-stat-threshold, "
            "for example 're-T2-kde'."
        )
    mode, statistic, threshold = parts
    if mode not in {"re", "lv", "custom"}:
        raise ValueError(f"Unknown FD mode {mode!r}. Legal options are: re, lv, custom.")
    if statistic not in {"t2", "q", "spe", "custom"}:
        raise ValueError(f"Unknown FD statistic {statistic!r}. Legal options are: T2, Q, SPE, custom.")
    if threshold not in {"kde", "ineq", "pdf"}:
        raise ValueError(f"Unknown FD threshold {threshold!r}. Legal options are: kde, ineq, pdf.")
    return _Procedure(mode=mode, statistic=statistic.upper(), threshold=threshold)


def _payload(primary: Any, **data: Any) -> dict[str, Any]:
    if isinstance(primary, dict):
        payload = dict(primary)
    elif isinstance(primary, (tuple, list)) and len(primary) == 2:
        payload = {"x_true": primary[0], "x_pred": primary[1]}
    else:
        payload = {"scores": primary}
    payload.update(data)
    return payload


def _fit_reference(
    payload: dict[str, Any],
    procedure: _Procedure,
    covariance_method: str,
) -> dict[str, np.ndarray]:
    values = _statistic_inputs(payload, procedure)
    reference: dict[str, np.ndarray] = {}
    if procedure.statistic == "T2":
        _validate_covariance_method(covariance_method)
        reference["center"] = np.mean(values, axis=0, keepdims=True)
        centered = values - reference["center"]
        variance = np.mean(centered**2, axis=0, keepdims=True)
        reference["inverse_variance"] = 1.0 / np.maximum(variance, 1e-8)
    return reference


def _procedure_scores(
    payload: dict[str, Any],
    procedure: _Procedure,
    reference: dict[str, np.ndarray],
) -> np.ndarray:
    if procedure.mode == "custom" or procedure.statistic == "CUSTOM":
        if "scores" not in payload:
            raise ValueError("custom fault detection procedures require explicit 'scores'.")
        return _as_1d(payload["scores"])
    values = _statistic_inputs(payload, procedure)
    if procedure.statistic == "T2":
        center = reference.get("center", np.mean(values, axis=0, keepdims=True))
        inverse_variance = reference["inverse_variance"]
        centered = values - center
        return np.sum(centered**2 * inverse_variance, axis=1)
    return np.mean(values**2, axis=1)


def _statistic_inputs(payload: dict[str, Any], procedure: _Procedure) -> np.ndarray:
    if procedure.mode == "re":
        true = _value(payload, "x_true", "true", "target")
        pred = _value(payload, "x_pred", "prediction", "reconstruction")
        return _as_2d(true) - _as_2d(pred)
    if procedure.mode == "lv":
        return _as_2d(_value(payload, "latent", "z", "lv"))
    return _as_2d(_value(payload, "scores"))


def _value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    raise ValueError(
        f"Fault detection payload is missing one of required keys: {', '.join(keys)}. "
        f"Current keys: {sorted(payload)}."
    )


def _validate_covariance_method(method: str) -> None:
    normalized = method.strip().lower()
    if normalized in {"diag", "diagonal", "lstsq", "least_squares", "pinv", "pseudo_inverse"}:
        return
    raise ValueError(
        f"Unknown covariance method {method!r}. Legal options are: diag, lstsq, pinv."
    )


def _threshold(scores: np.ndarray, *, expected_far: float, method: str) -> float:
    if method in {"kde", "pdf"}:
        return float(np.nanpercentile(scores, 100.0 * (1.0 - expected_far)))
    if method == "ineq":
        mean = float(np.nanmean(scores))
        std = float(np.nanstd(scores))
        factor = np.sqrt((1.0 - expected_far) / max(expected_far, 1e-12))
        return float(mean + factor * std)
    raise ValueError(f"Unknown threshold method {method!r}. Legal options are: kde, ineq, pdf.")


def _metrics_from_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    alarms = scores > threshold
    normal = labels == 0
    fault = labels == 1
    false_alarms = int(np.logical_and(alarms, normal).sum())
    missed = int(np.logical_and(~alarms, fault).sum())
    detected = int(np.logical_and(alarms, fault).sum())
    far = false_alarms / max(int(normal.sum()), 1)
    mdr = missed / max(int(fault.sum()), 1)
    fdr = detected / max(int(fault.sum()), 1)
    return {
        "FAR": float(far),
        "MDR": float(mdr),
        "FDR": float(fdr),
        "false_alarms": float(false_alarms),
        "missed_detections": float(missed),
        "detected_faults": float(detected),
    }


def _aggregate_procedure_metrics(procedure_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    if not procedure_metrics:
        return {}
    names = list(procedure_metrics)
    afar = float(np.mean([procedure_metrics[name]["FAR"] for name in names]))
    amdr = float(np.mean([procedure_metrics[name]["MDR"] for name in names]))
    afdr = float(np.mean([procedure_metrics[name]["FDR"] for name in names]))
    return {
        "AFAR": afar,
        "AMDR": amdr,
        "AFDR": afdr,
        "FAR": afar,
        "MDR": amdr,
        "FDR": afdr,
    }


def _as_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return np.asarray(array, dtype=float).reshape(-1)


def _as_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim >= 2:
        return array.reshape(array.shape[0], -1)
    raise ValueError(f"Expected at least 1D data. Current shape: {array.shape}.")
