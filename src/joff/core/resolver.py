"""Config loading, merging, validation, and provenance."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import hashlib
import json
import yaml
from pydantic import ValidationError

from .config import ExperimentConfig
from .defaults import DEFAULTS
from .errors import ConfigError
from .provenance import ConfigProvenance


@dataclass(frozen=True)
class ResolvedConfig:
    """Final validated config plus raw values and provenance."""

    config: ExperimentConfig
    raw_config: dict[str, Any]
    provenance: ConfigProvenance
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML serializable resolved config dictionary."""

        return self.config.model_dump(mode="json")


class ConfigManager:
    """Load, compose, override, validate, and resolve experiment configs."""

    LEGACY_ALIASES = {
        "struct": "model.struct",
        "act": "model.act",
        "b": "trainer.batch_size",
        "lr": "trainer.optimizer.lr",
        "opt": "trainer.optimizer.type",
        "l2_norm": "trainer.optimizer.weight_decay",
        "e": "trainer.max_epochs",
        "save_path": "artifacts.root",
    }

    def load(self, source: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
        """Load config from a mapping or YAML path."""

        if source is None:
            return {}
        if isinstance(source, Mapping):
            return _deep_copy_dict(dict(source))
        path = Path(source)
        if not path.exists():
            raise ConfigError(f"Config file does not exist: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ConfigError(
                f"Config file {path} must contain a YAML mapping. Current input type: "
                f"{type(data).__name__}."
            )
        return _deep_copy_dict(dict(data))

    def compose(self, *layers: Mapping[str, Any]) -> dict[str, Any]:
        """Deep-merge layers from lowest to highest precedence."""

        output: dict[str, Any] = {}
        for layer in layers:
            _deep_merge(output, dict(layer))
        return output

    def apply_overrides(
        self, cfg: Mapping[str, Any], overrides: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Apply dot-path overrides to a config mapping."""

        output = _deep_copy_dict(dict(cfg))
        for raw_key, value in (overrides or {}).items():
            key = self.LEGACY_ALIASES.get(raw_key, raw_key)
            _set_dot_path(output, key, value)
        return output

    def validate(self, cfg: Mapping[str, Any]) -> ExperimentConfig:
        """Validate a mapping as :class:`ExperimentConfig`."""

        try:
            return ExperimentConfig.model_validate(dict(cfg))
        except ValidationError as exc:
            raise ConfigError(_format_validation_error(exc)) from exc

    def resolve(
        self,
        source: str | Path | Mapping[str, Any] | None = None,
        *,
        trial_overrides: Mapping[str, Any] | None = None,
        api_kwargs: Mapping[str, Any] | None = None,
        method_overrides: Mapping[str, Any] | None = None,
        cli_overrides: Mapping[str, Any] | None = None,
    ) -> ResolvedConfig:
        """Resolve a single user config with the documented precedence ladder."""

        user_config = self.load(source)
        resolver = ConfigResolver()
        layers: list[tuple[str, Mapping[str, Any]]] = []
        if user_config:
            layers.append(("user_config", user_config))
        if trial_overrides:
            layers.append(("study_trial", self.apply_overrides({}, trial_overrides)))
        if api_kwargs:
            layers.append(("api_kwargs", self.apply_overrides({}, api_kwargs)))
        if method_overrides:
            layers.append(("method_call", self.apply_overrides({}, method_overrides)))
        if cli_overrides:
            layers.append(("cli_env", self.apply_overrides({}, cli_overrides)))
        return resolver.resolve(*layers)


class ConfigResolver:
    """Merge defaults and explicit layers into a frozen resolved config."""

    def resolve(self, *layers: tuple[str, Mapping[str, Any]]) -> ResolvedConfig:
        """Resolve config layers and track provenance for every leaf value."""

        cfg: dict[str, Any] = {}
        provenance = ConfigProvenance()
        self._apply_layer(cfg, DEFAULTS.get("package", "experiment"), "package_default", provenance)

        model_type = _find_effective_model_type(layers) or cfg.get("model", {}).get("type", "mlp")
        if model_type in DEFAULTS.list("model"):
            self._apply_layer(
                cfg,
                DEFAULTS.get("model", str(model_type)),
                f"model_default:{model_type}",
                provenance,
            )

        for source, layer in layers:
            self._apply_layer(cfg, dict(layer), source, provenance)

        manager = ConfigManager()
        validated = manager.validate(cfg)
        plain = validated.model_dump(mode="json")
        config_hash = hashlib.sha256(
            json.dumps(plain, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        return ResolvedConfig(
            config=validated,
            raw_config=_deep_copy_dict(cfg),
            provenance=provenance,
            config_hash=config_hash,
        )

    def _apply_layer(
        self,
        target: dict[str, Any],
        layer: Mapping[str, Any],
        source: str,
        provenance: ConfigProvenance,
    ) -> None:
        _deep_merge(target, dict(layer))
        for path, value in _iter_leaves(layer):
            provenance.record(path, source, value)


def _find_effective_model_type(layers: tuple[tuple[str, Mapping[str, Any]], ...]) -> str | None:
    model_type: str | None = None
    for _, layer in layers:
        current = layer.get("model", {})
        if isinstance(current, Mapping) and current.get("type") is not None:
            model_type = str(current["type"])
    return model_type


def _deep_merge(target: dict[str, Any], layer: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in layer.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = _deep_copy(value)
    return target


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(_plain_for_json(value), ensure_ascii=False))


def _deep_copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    return _deep_copy(value)


def _plain_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain_for_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain_for_json(v) for v in value]
    if isinstance(value, list):
        return [_plain_for_json(v) for v in value]
    return value


def _iter_leaves(layer: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []
    for key, value in layer.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            leaves.extend(_iter_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _set_dot_path(target: dict[str, Any], dot_path: str, value: Any) -> None:
    if not dot_path or ".." in dot_path:
        raise ConfigError(f"Invalid override path {dot_path!r}. Use dotted keys like 'trainer.max_epochs'.")
    parts = dot_path.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            raise ConfigError(
                f"Cannot set override {dot_path!r}: {part!r} is already a non-mapping value."
            )
    cursor[parts[-1]] = value


def _format_validation_error(exc: ValidationError) -> str:
    legal_top = ", ".join(ExperimentConfig.model_fields)
    messages: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
        if err.get("type") == "extra_forbidden":
            bad_key = str(err.get("loc", ("<unknown>",))[-1])
            parent = ".".join(str(part) for part in err.get("loc", ())[:-1])
            legal = _legal_fields_for_parent(parent)
            suggestion = get_close_matches(bad_key, legal, n=1)
            hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
            messages.append(
                f"Unknown config key {loc!r}.{hint} Legal options here are: "
                f"{', '.join(legal) or '<none>'}. Legal top-level keys are: {legal_top}."
            )
        else:
            messages.append(f"Invalid config value at {loc!r}: {err.get('msg')}.")
    return " ".join(messages)


def _legal_fields_for_parent(parent: str) -> list[str]:
    model = ExperimentConfig
    if not parent:
        return list(model.model_fields)
    for part in parent.split("."):
        field = model.model_fields.get(part)
        if field is None:
            return []
        annotation = field.annotation
        if hasattr(annotation, "model_fields"):
            model = annotation
        else:
            return []
    return list(model.model_fields)

