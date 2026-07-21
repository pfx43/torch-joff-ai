"""Dataset preset registry."""

from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter, DatasetCardAdapter


class DatasetRegistry:
    """Register and resolve dataset adapters by preset name."""

    def __init__(self) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        adapter: DatasetAdapter,
        *,
        aliases: tuple[str, ...] | list[str] = (),
        replace: bool = False,
    ) -> None:
        """Register an adapter and optional aliases."""

        key = _normalize(adapter.name)
        if not replace and key in self._adapters:
            raise ValueError(
                f"Dataset preset {adapter.name!r} is already registered. "
                f"Legal presets are: {', '.join(self.list())}."
            )
        self._adapters[key] = adapter
        for alias in aliases:
            self._aliases[_normalize(alias)] = key

    def get(self, name: str) -> DatasetAdapter:
        """Return an adapter by preset name or alias."""

        key = _normalize(name)
        resolved = self._aliases.get(key, key)
        if resolved not in self._adapters:
            legal = ", ".join(self.list()) or "<empty>"
            raise ValueError(
                f"Unknown dataset preset {name!r}. Legal presets are: {legal}. "
                f"Current input was: {name!r}."
            )
        return self._adapters[resolved]

    def resolve(self, preset: str | Path) -> DatasetAdapter:
        """Resolve a preset name or local dataset-card path."""

        path = Path(preset)
        if path.exists():
            return DatasetCardAdapter.from_yaml(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            raise FileNotFoundError(f"Dataset card does not exist: {path}")
        return self.get(str(preset))

    def list(self) -> tuple[str, ...]:
        """List registered preset names."""

        return tuple(sorted(adapter.name for adapter in self._adapters.values()))


def _normalize(name: str) -> str:
    return name.strip().lower()
