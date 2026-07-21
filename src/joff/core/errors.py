"""Typed exceptions used by joff core components."""

from __future__ import annotations


class JoffError(Exception):
    """Base class for all public joff exceptions."""


class RegistryError(JoffError):
    """Raised when a registry lookup or registration fails."""


class ConfigError(JoffError):
    """Raised when user configuration cannot be loaded, merged, or validated."""


class BuildError(JoffError):
    """Raised when a model or layer cannot be built from a valid spec."""

