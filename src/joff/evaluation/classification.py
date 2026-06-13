"""Classification evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ClassificationReport:
    """Classification metrics and confusion matrix."""

    overall: dict[str, float]
    per_class: list[dict[str, float | int]]
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable report."""

        return {
            "overall": self.overall,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


class ClassificationEvaluator:
    """Compute accuracy, confusion matrix, and per-class precision/recall/F1."""

    def evaluate(self, y_true: Any, y_pred: Any) -> ClassificationReport:
        """Evaluate predicted labels or logits."""

        target = _labels(y_true)
        prediction = _predicted_labels(y_pred)
        if target.shape[0] != prediction.shape[0]:
            raise ValueError(
                f"y_true and y_pred must share length. Current lengths: "
                f"{target.shape[0]} and {prediction.shape[0]}."
            )
        classes = sorted(set(target.tolist()) | set(prediction.tolist()))
        class_to_idx = {label: idx for idx, label in enumerate(classes)}
        matrix = np.zeros((len(classes), len(classes)), dtype=int)
        for true_label, pred_label in zip(target, prediction):
            matrix[class_to_idx[int(true_label)], class_to_idx[int(pred_label)]] += 1
        per_class = []
        for label in classes:
            idx = class_to_idx[label]
            tp = float(matrix[idx, idx])
            fp = float(matrix[:, idx].sum() - matrix[idx, idx])
            fn = float(matrix[idx, :].sum() - matrix[idx, idx])
            precision = tp / max(tp + fp, 1.0)
            recall = tp / max(tp + fn, 1.0)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            per_class.append(
                {
                    "Class": int(label),
                    "Precision": float(precision),
                    "Recall": float(recall),
                    "F1": float(f1),
                }
            )
        accuracy = float(np.mean(target == prediction))
        macro_f1 = float(np.mean([row["F1"] for row in per_class])) if per_class else float("nan")
        return ClassificationReport(
            overall={"Accuracy": accuracy, "MacroF1": macro_f1},
            per_class=per_class,
            confusion_matrix=matrix.tolist(),
        )


def _labels(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return np.asarray(array).reshape(-1).astype(int)


def _predicted_labels(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array)
    if array.ndim == 1:
        return array.reshape(-1).astype(int)
    return np.argmax(array, axis=1).astype(int)

