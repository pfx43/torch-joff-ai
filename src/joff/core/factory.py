"""Factories for public build APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from .config import ModelConfig
from .errors import ConfigError
from .registry import EVALUATOR_REGISTRY, MODEL_REGISTRY


def register_model(
    key: str, model_cls: type | None = None, *, aliases: tuple[str, ...] = (), replace: bool = False
):
    """Register a model class in the global model registry."""

    return MODEL_REGISTRY.register(key, model_cls, aliases=aliases, replace=replace)


def register_evaluator(
    key: str,
    evaluator_cls: type | None = None,
    *,
    aliases: tuple[str, ...] = (),
    replace: bool = False,
):
    """Register an evaluator class in the global evaluator registry."""

    return EVALUATOR_REGISTRY.register(key, evaluator_cls, aliases=aliases, replace=replace)


def build_model(spec: ModelConfig | Mapping[str, Any]) -> Any:
    """Build a registered model from a :class:`ModelConfig` or plain mapping."""

    _ensure_builtin_models_registered()
    if isinstance(spec, ModelConfig):
        config = spec
    else:
        try:
            config = ModelConfig.model_validate(dict(spec))
        except ValidationError as exc:
            legal = ", ".join(ModelConfig.model_fields)
            raise ConfigError(
                f"Invalid model config. Legal model keys are: {legal}. "
                f"Current input was: {dict(spec)!r}. Details: {exc}"
            ) from exc
    model_cls = MODEL_REGISTRY.get(config.type)
    return model_cls(config)


def build_evaluator(spec: str | Mapping[str, Any]) -> Any:
    """Build a registered evaluator from a string or mapping with ``type``."""

    _ensure_builtin_evaluators_registered()
    if isinstance(spec, str):
        evaluator_type = spec
        kwargs: dict[str, Any] = {}
    else:
        raw = dict(spec)
        evaluator_type = str(raw.pop("type", "regression"))
        kwargs = raw
    evaluator_cls = EVALUATOR_REGISTRY.get(evaluator_type)
    return evaluator_cls(**kwargs)


def _ensure_builtin_models_registered() -> None:
    # Importing this module performs explicit registry registration. This happens only when
    # a user builds a model, not during core module import.
    import joff.models  # noqa: F401


def _ensure_builtin_evaluators_registered() -> None:
    import joff.evaluation  # noqa: F401
