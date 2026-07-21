"""Console facade with Rich support and plain-text fallback."""

from __future__ import annotations

import sys
import traceback as traceback_module
from pathlib import Path
from typing import Any, TextIO

from .progress import ProgressReporter
from .style import ConsoleTheme
from .tables import MetricTable


class JoffConsole:
    """Human-readable console reporter with optional Rich rendering."""

    def __init__(
        self,
        *,
        theme: str | ConsoleTheme = "joff",
        quiet: bool = False,
        verbose: int = 1,
        file: TextIO | None = None,
        force_plain: bool = False,
    ) -> None:
        self.theme = ConsoleTheme.from_name(theme) if isinstance(theme, str) else theme
        self.quiet = quiet
        self.verbose = verbose
        self.file = file or sys.stdout
        self._rich_console = None if force_plain else _make_rich_console(self.file)

    def info(self, message: str, *, min_verbose: int = 1, **fields: Any) -> None:
        """Emit an informational message."""

        self._message("info", message, min_verbose=min_verbose, **fields)

    def success(self, message: str, path: str | Path | None = None, **fields: Any) -> None:
        """Emit a success message."""

        if path is not None:
            fields = {"path": path, **fields}
        self._message("success", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        """Emit a warning message."""

        self._message("warning", message, **fields)

    def error(self, message: str, exc: Exception | None = None, **fields: Any) -> None:
        """Emit an error message with optional exception detail."""

        if exc is not None:
            fields = {"error_type": exc.__class__.__name__, "message": str(exc), **fields}
        self._message("error", message, **fields)
        if exc is not None and self.verbose >= 2:
            self._emit_plain("".join(traceback_module.format_exception(exc)))

    def metric(self, name: str, value: float, style: str | None = None) -> None:
        """Emit one named metric."""

        selected = style or "neutral"
        text = f"{name}: {value:.6g}"
        self._emit(text, style=selected)

    def table(self, data: Any, title: str | None = None) -> None:
        """Render mapping, records, or tabular data."""

        if self._should_skip(1):
            return
        table = data if isinstance(data, MetricTable) else MetricTable.from_data(data, title=title)
        if self._rich_console is None:
            self._emit_plain(table.render_plain())
            return
        rich_table = _rich_table(table)
        if table.title:
            rich_table.title = table.title
        rich_printer = getattr(self._rich_console, "print")
        rich_printer(rich_table)

    def progress(self, total: int, description: str) -> ProgressReporter:
        """Create a progress reporter."""

        if total < 0:
            raise ValueError(f"progress total must be non-negative. Current input: {total}.")
        return ProgressReporter(console=self, total=total, description=description)

    def rule(self, title: str) -> None:
        """Render a section rule."""

        if self._should_skip(1):
            return
        if self._rich_console is not None:
            self._rich_console.rule(title)
        else:
            self._emit_plain(f"--- {title} ---")

    def _message(self, level: str, message: str, *, min_verbose: int = 1, **fields: Any) -> None:
        if self._should_skip(min_verbose):
            return
        field_text = _format_fields(fields)
        text = message if not field_text else f"{message} {field_text}"
        self._emit(text, style=level)

    def _emit(self, text: str, *, style: str) -> None:
        if self.quiet:
            return
        rich_style = self.theme.styles.get(style, style)
        if self._rich_console is None:
            self._emit_plain(text)
        else:
            rich_printer = getattr(self._rich_console, "print")
            rich_printer(text, style=rich_style)

    def _emit_plain(self, text: str) -> None:
        if self.quiet:
            return
        self.file.write(text)
        if not text.endswith("\n"):
            self.file.write("\n")
        flush = getattr(self.file, "flush", None)
        if callable(flush):
            flush()

    def _should_skip(self, min_verbose: int) -> bool:
        return self.quiet or self.verbose < min_verbose


def _make_rich_console(file: TextIO) -> Any | None:
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console(file=file, highlight=False, soft_wrap=True)


def _rich_table(table: MetricTable) -> Any:
    from rich.table import Table

    rich_table = Table(show_lines=False)
    for header in table.headers:
        rich_table.add_column(header)
    for row in table.rows:
        rich_table.add_row(*row)
    return rich_table


def _format_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    return " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())


def _format_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
