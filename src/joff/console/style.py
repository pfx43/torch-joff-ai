"""Console styles and fallback themes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConsoleTheme:
    """Named console style mapping."""

    name: str = "joff"
    styles: dict[str, str] = field(
        default_factory=lambda: {
            "info": "cyan",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "good": "green",
            "bad": "red",
            "neutral": "white",
        }
    )

    @classmethod
    def from_name(cls, name: str) -> "ConsoleTheme":
        """Resolve a built-in console theme."""

        normalized = name.strip().lower()
        if normalized in {"joff", "default"}:
            return cls(name="joff")
        if normalized == "joff_dark":
            return cls(
                name="joff_dark",
                styles={
                    "info": "bright_cyan",
                    "success": "bright_green",
                    "warning": "bright_yellow",
                    "error": "bright_red",
                    "good": "bright_green",
                    "bad": "bright_red",
                    "neutral": "white",
                },
            )
        raise ValueError(
            f"Unknown console theme {name!r}. Legal options are: joff, joff_dark, default."
        )
