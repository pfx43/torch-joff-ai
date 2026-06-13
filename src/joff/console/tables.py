"""Plain and Rich-compatible metric table helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricTable:
    """Normalized table data for console rendering."""

    headers: list[str]
    rows: list[list[str]]
    title: str | None = None

    @classmethod
    def from_data(cls, data: Any, *, title: str | None = None) -> "MetricTable":
        """Create a metric table from a mapping, records, or tabular object."""

        if isinstance(data, Mapping):
            return cls(
                headers=["name", "value"],
                rows=[[str(key), _format_value(value)] for key, value in data.items()],
                title=title,
            )
        records = _records_from_data(data)
        if not records:
            return cls(headers=["value"], rows=[], title=title)
        headers = _headers(records)
        rows = [[_format_value(record.get(header, "")) for header in headers] for record in records]
        return cls(headers=headers, rows=rows, title=title)

    def render_plain(self) -> str:
        """Render this table as plain text without ANSI codes."""

        widths = [len(header) for header in self.headers]
        for row in self.rows:
            for idx, value in enumerate(row):
                widths[idx] = max(widths[idx], len(value))
        lines: list[str] = []
        if self.title:
            lines.append(self.title)
        header = " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(self.headers))
        lines.append(header)
        lines.append("-+-".join("-" * width for width in widths))
        for row in self.rows:
            lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
        return "\n".join(lines)


class ConfigTable(MetricTable):
    """Table representation for nested config dictionaries."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, title: str | None = None) -> "ConfigTable":
        """Flatten a nested config mapping into path/value rows."""

        rows = [[key, _format_value(value)] for key, value in _flatten(config)]
        return cls(headers=["path", "value"], rows=rows, title=title)


def _records_from_data(data: Any) -> list[dict[str, Any]]:
    to_dict = getattr(data, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict("records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [dict(item) for item in records if isinstance(item, Mapping)]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        records = []
        for item in data:
            if isinstance(item, Mapping):
                records.append(dict(item))
            else:
                records.append({"value": item})
        return records
    return [{"value": data}]


def _headers(records: list[dict[str, Any]]) -> list[str]:
    headers: list[str] = []
    for record in records:
        for key in record:
            text = str(key)
            if text not in headers:
                headers.append(text)
    return headers


def _flatten(config: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.extend(_flatten(value, path))
        else:
            rows.append((path, value))
    return rows


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
