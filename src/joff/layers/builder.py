"""Dense network construction and width DSL parsing."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Sequence
from typing import Any

from torch import nn

from joff.core.config import DropoutConfig
from joff.core.errors import BuildError

from .activations import activation_changes_feature_dim, build_activation

_SPLIT_RE = re.compile(r"[,;，；、]+")
_SYMBOL_TRANSLATION = str.maketrans(
    {
        "＋": "+",
        "－": "-",
        "＊": "*",
        "×": "*",
        "／": "/",
        "÷": "/",
        "（": "(",
        "）": ")",
    }
)


def resolve_widths(
    input_dim: int,
    output_dim: int | None,
    spec: Sequence[int | str] | str | None,
) -> list[int]:
    """Resolve integer and expression width specs into positive widths.

    Supported expressions use only numbers, ``i``/``o``/``prev``, parentheses, and
    ``+ - * /``. Relative specs such as ``"*2"`` are interpreted against ``prev``.
    """

    if spec is None:
        return []
    normalized_spec = _normalize_width_spec(spec)
    widths: list[int] = []
    prev = input_dim
    for item in normalized_spec:
        width = _resolve_one_width(item, input_dim=input_dim, output_dim=output_dim, prev=prev)
        if width <= 0:
            raise BuildError(
                f"Resolved width must be positive. Current input {item!r} resolved to {width}."
            )
        widths.append(width)
        prev = width
    return widths


def dropout_rate_for_width(
    width: int,
    *,
    threshold: int = 100,
    scale: float = 100.0,
    max_rate: float = 0.5,
) -> float:
    """Return NKN-style automatic dropout rate for a layer width."""

    if threshold <= 0:
        raise BuildError(f"dropout threshold must be positive. Current input was: {threshold}.")
    if scale <= 0:
        raise BuildError(f"dropout scale must be positive. Current input was: {scale}.")
    if width <= threshold:
        return 0.0
    return float(min(max_rate, width / (threshold * scale)))


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden: Sequence[int | str] | str | None = None,
    *,
    act: Sequence[str] | str = ("r",),
    output_act: str = "a",
    dropout: str | float | DropoutConfig | dict[str, Any] = "none",
    dropout_threshold: int = 100,
    dropout_scale: float = 100.0,
    dropout_max_rate: float = 0.5,
    batch_norm: bool = False,
) -> nn.Sequential:
    """Build a dense ``nn.Sequential`` from dimensions and activation aliases."""

    hidden_widths = resolve_widths(input_dim=input_dim, output_dim=output_dim, spec=hidden)
    activation_names = _expand_activations(act, len(hidden_widths))
    modules: list[nn.Module] = []
    prev = input_dim
    dropout_policy = _normalize_dropout_policy(
        dropout,
        threshold=dropout_threshold,
        scale=dropout_scale,
        max_rate=dropout_max_rate,
    )

    for width, activation_name in zip(hidden_widths, activation_names):
        linear_out = width * 2 if activation_changes_feature_dim(activation_name) else width
        modules.append(nn.Linear(prev, linear_out))
        if batch_norm:
            modules.append(nn.BatchNorm1d(linear_out))
        modules.append(build_activation(activation_name, feature_dim=linear_out))
        rate = _dropout_rate(width, dropout_policy)
        if rate > 0:
            modules.append(nn.Dropout(rate))
        prev = width

    modules.append(nn.Linear(prev, output_dim))
    output_activation = build_activation(output_act, feature_dim=output_dim)
    if not isinstance(output_activation, nn.Identity):
        modules.append(output_activation)
    return nn.Sequential(*modules)


def _normalize_width_spec(spec: Sequence[int | str] | str) -> list[int | str]:
    if isinstance(spec, str):
        text = spec.translate(_SYMBOL_TRANSLATION).strip()
        if not text:
            return []
        return [part.strip() for part in _SPLIT_RE.split(text) if part.strip()]
    return [item.translate(_SYMBOL_TRANSLATION).strip() if isinstance(item, str) else item for item in spec]


def _resolve_one_width(
    item: int | str,
    *,
    input_dim: int,
    output_dim: int | None,
    prev: int,
) -> int:
    if isinstance(item, int):
        return item
    expr = item.strip().lower()
    if expr in {"i", "input", "input_dim"}:
        return input_dim
    if expr in {"o", "output", "output_dim"}:
        if output_dim is None:
            raise BuildError(
                f"Width expression {item!r} references output_dim, but output_dim is None."
            )
        return output_dim
    if expr in {"prev", "p"}:
        return prev
    if expr.startswith(("*", "/", "+", "-")):
        expr = f"prev{expr}"
    names = {"i": float(input_dim), "input": float(input_dim), "prev": float(prev), "p": float(prev)}
    if output_dim is not None:
        names.update({"o": float(output_dim), "output": float(output_dim)})
    value = _safe_eval_arithmetic(expr, names)
    if not math.isfinite(value):
        raise BuildError(f"Width expression {item!r} produced a non-finite value: {value}.")
    rounded = int(round(value))
    if abs(rounded - value) > 1e-8:
        raise BuildError(
            f"Width expression {item!r} must resolve to an integer. Current value: {value}."
        )
    return rounded


def _safe_eval_arithmetic(expr: str, names: dict[str, float]) -> float:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise BuildError(
            "Invalid width expression {!r}. Legal syntax allows numbers, i/o/prev, "
            "parentheses, and + - * /. Current input was: {!r}.".format(expr, expr)
        ) from exc
    return _eval_node(tree.body, names, expr)


def _eval_node(node: ast.AST, names: dict[str, float], expr: str) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            legal = ", ".join(sorted(names))
            raise BuildError(
                f"Unknown name {node.id!r} in width expression {expr!r}. "
                f"Legal names are: {legal}. Current input was: {expr!r}."
            )
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand, names, expr)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _eval_node(node.left, names, expr)
        right = _eval_node(node.right, names, expr)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise BuildError(f"Division by zero in width expression {expr!r}.")
        return left / right
    raise BuildError(
        f"Unsafe width expression {expr!r}. Legal syntax allows numbers, i/o/prev, "
        "parentheses, and + - * / only."
    )


def _expand_activations(act: Sequence[str] | str, count: int) -> list[str]:
    if count == 0:
        return []
    if isinstance(act, str):
        names = [part.strip() for part in _SPLIT_RE.split(act.translate(_SYMBOL_TRANSLATION)) if part.strip()]
    else:
        names = [str(part) for part in act]
    if not names:
        names = ["a"]
    if len(names) == 1:
        return names * count
    if len(names) < count:
        return names + [names[-1]] * (count - len(names))
    return names[:count]


def _normalize_dropout_policy(
    dropout: str | float | DropoutConfig | dict[str, Any],
    *,
    threshold: int,
    scale: float,
    max_rate: float,
) -> DropoutConfig:
    if isinstance(dropout, DropoutConfig):
        return dropout
    if isinstance(dropout, dict):
        return DropoutConfig.model_validate(dropout)
    if isinstance(dropout, (int, float)):
        return DropoutConfig(mode="fixed", rate=float(dropout))
    normalized = dropout.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return DropoutConfig(mode="none")
    if normalized == "auto":
        return DropoutConfig(mode="auto", threshold=threshold, scale=scale, max_rate=max_rate)
    raise BuildError(
        "Unknown dropout policy {!r}. Legal options are: 'none', 'auto', a fixed float, "
        "or DropoutConfig. Current input was: {!r}.".format(dropout, dropout)
    )


def _dropout_rate(width: int, config: DropoutConfig) -> float:
    if config.mode == "none":
        return 0.0
    if config.mode == "fixed":
        return float(config.rate)
    return dropout_rate_for_width(
        width,
        threshold=config.threshold,
        scale=config.scale,
        max_rate=config.max_rate,
    )

