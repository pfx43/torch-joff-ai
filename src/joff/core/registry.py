"""Small explicit registries used instead of eval/exec."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Generic, TypeVar

from .errors import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable string names and aliases to buildable objects.

    Parameters
    ----------
    name:
        Human-readable registry name used in error messages.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        key: str,
        value: T | None = None,
        *,
        aliases: Iterable[str] = (),
        replace: bool = False,
    ) -> T | Callable[[T], T]:
        """Register a value under a canonical key and optional aliases."""

        normalized_key = self._normalize(key)

        def _decorator(obj: T) -> T:
            if not replace and normalized_key in self._items:
                raise RegistryError(
                    f"{self.name!r} registry already contains {normalized_key!r}. "
                    f"Legal options are: {', '.join(self.keys())}."
                )
            self._items[normalized_key] = obj
            for alias in aliases:
                normalized_alias = self._normalize(alias)
                if not replace and (
                    normalized_alias in self._aliases or normalized_alias in self._items
                ):
                    raise RegistryError(
                        f"{self.name!r} alias {normalized_alias!r} already exists. "
                        f"Legal options are: {', '.join(self.keys())}."
                    )
                self._aliases[normalized_alias] = normalized_key
            return obj

        if value is None:
            return _decorator
        return _decorator(value)

    def get(self, key: str) -> T:
        """Return the registered value for ``key`` or raise a helpful error."""

        normalized_key = self._normalize(key)
        canonical = self._aliases.get(normalized_key, normalized_key)
        if canonical not in self._items:
            legal = ", ".join(self.keys()) or "<empty>"
            raise RegistryError(
                f"Unknown {self.name} {key!r}. Legal options are: {legal}. "
                f"Current input was: {key!r}."
            )
        return self._items[canonical]

    def keys(self) -> tuple[str, ...]:
        """Return canonical keys and aliases in deterministic order."""

        return tuple(sorted((*self._items.keys(), *self._aliases.keys())))

    def canonical_keys(self) -> tuple[str, ...]:
        """Return only canonical registered keys."""

        return tuple(sorted(self._items.keys()))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        normalized_key = self._normalize(key)
        return normalized_key in self._items or normalized_key in self._aliases

    @staticmethod
    def _normalize(key: str) -> str:
        return key.strip().lower().replace("-", "_")


MODEL_REGISTRY: Registry[type] = Registry("model")
ACTIVATION_REGISTRY: Registry[type] = Registry("activation")
LOSS_REGISTRY: Registry[Callable[..., object]] = Registry("loss")
OPTIMIZER_REGISTRY: Registry[type] = Registry("optimizer")
EVALUATOR_REGISTRY: Registry[type] = Registry("evaluator")

