"""Progress reporting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .console import JoffConsole


@dataclass
class ProgressReporter:
    """Small progress reporter that routes output through ``JoffConsole``."""

    console: "JoffConsole"
    total: int
    description: str
    completed: int = 0
    closed: bool = False

    def __enter__(self) -> "ProgressReporter":
        """Start a progress context."""

        self.console.info(self.description, progress=f"{self.completed}/{self.total}", min_verbose=2)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close a progress context."""

        self.close()

    def advance(self, step: int = 1) -> None:
        """Advance progress by ``step``."""

        self.update(self.completed + step)

    def update(self, completed: int) -> None:
        """Set completed progress."""

        if self.closed:
            return
        self.completed = min(max(int(completed), 0), self.total)
        self.console.info(self.description, progress=f"{self.completed}/{self.total}", min_verbose=2)

    def close(self) -> None:
        """Mark progress as closed."""

        if not self.closed:
            self.closed = True
            self.console.info(self.description, progress=f"{self.completed}/{self.total}", min_verbose=2)
