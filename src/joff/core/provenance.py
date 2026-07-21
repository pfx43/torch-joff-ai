"""Configuration provenance tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class ProvenanceEntry:
    """Single source contribution for a resolved configuration path."""

    source: str
    value: Any


@dataclass
class ConfigProvenance:
    """Record how each resolved config field was produced."""

    records: dict[str, list[ProvenanceEntry]] = field(default_factory=dict)

    def record(self, path: str, source: str, value: Any) -> None:
        """Append a source/value entry for ``path``."""

        self.records.setdefault(path, []).append(ProvenanceEntry(source=source, value=value))

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON/YAML serializable representation."""

        return {
            path: [{"source": entry.source, "value": _to_plain(entry.value)} for entry in entries]
            for path, entries in sorted(self.records.items())
        }

    def save_json(self, path: str | Path) -> Path:
        """Write provenance to ``path`` and return the final path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path


def _to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value

