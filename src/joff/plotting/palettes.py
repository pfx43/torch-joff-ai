"""Original palettes and explicit user palette imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


_BUILT_INS: dict[str, list[str]] = {
    "joff_academic": ["#2F4858", "#33658A", "#86BBD8", "#F6AE2D", "#F26419"],
    "joff_colorblind": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"],
    "joff_fault": ["#2A6FBB", "#D1495B", "#EDA83A", "#4D4D4D"],
    "joff_quality": ["#1B998B", "#2D3047", "#FFFD82", "#FF9B71", "#E84855"],
    "joff_flow": ["#355070", "#6D597A", "#B56576", "#E56B6F", "#EAAC8B"],
    "joff_coolors_like": ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
}


@dataclass(frozen=True)
class Palette:
    """Named color palette with optional provenance metadata."""

    name: str
    colors: tuple[str, ...]
    source: str = "joff:built_in"
    source_url: str | None = None
    created_by: str = "joff"

    @classmethod
    def from_hex(
        cls,
        colors: Iterable[str],
        *,
        name: str,
        source: str = "user:hex",
        source_url: str | None = None,
    ) -> "Palette":
        """Create a palette from explicit HEX colors."""

        normalized = tuple(_normalize_hex(color) for color in colors)
        if not normalized:
            raise ValueError("Palette requires at least one HEX color. Current input was empty.")
        return cls(
            name=name,
            colors=normalized,
            source=source,
            source_url=source_url,
            created_by="user" if source.startswith("user") or "coolors" in source else "joff",
        )

    @classmethod
    def from_coolors_url(cls, url: str, *, name: str) -> "Palette":
        """Create a palette from a user-provided Coolors URL without network access."""

        colors = _parse_coolors_url(url)
        return cls.from_hex(
            colors,
            name=name,
            source="coolors:user_imported",
            source_url=url,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a serializable palette record."""

        return {
            "name": self.name,
            "colors": list(self.colors),
            "source": self.source,
            "source_url": self.source_url,
            "created_by": self.created_by,
        }


class PaletteRegistry:
    """In-memory registry for built-in and user-provided palettes."""

    def __init__(self) -> None:
        self._palettes = {
            name: Palette.from_hex(colors, name=name, source="joff:built_in")
            for name, colors in _BUILT_INS.items()
        }

    def register(self, name: str, palette: Palette, *, replace: bool = False) -> None:
        """Register a palette by name."""

        if name in self._palettes and not replace:
            raise ValueError(
                f"Palette {name!r} is already registered. Use replace=True to overwrite it."
            )
        self._palettes[name] = palette

    def get(self, name: str) -> Palette:
        """Return a registered palette."""

        try:
            return self._palettes[name]
        except KeyError as exc:
            legal = ", ".join(sorted(self._palettes))
            raise KeyError(
                f"Unknown palette {name!r}. Legal options are: {legal}. Current input: {name!r}."
            ) from exc

    def list(self) -> list[str]:
        """Return registered palette names."""

        return sorted(self._palettes)


def _parse_coolors_url(url: str) -> list[str]:
    token = url.strip().rstrip("/").split("/")[-1]
    if not token or token in {"coolors.co", "palettes", "trending"}:
        raise ValueError(
            "Coolors palette import requires a user-provided palette URL or HEX slug such as "
            "'https://coolors.co/264653-2a9d8f-e9c46a'. Trending pages are not imported."
        )
    parts = [part for part in token.split("-") if part]
    return [_normalize_hex(part) for part in parts]


def _normalize_hex(color: str) -> str:
    value = color.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) not in {3, 6} or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError(
            f"Invalid HEX color {color!r}. Legal inputs look like '#264653' or '264653'."
        )
    if len(value) == 3:
        value = "".join(character * 2 for character in value)
    return f"#{value.upper()}"
