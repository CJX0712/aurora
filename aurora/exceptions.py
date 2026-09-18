"""Aurora-specific errors."""

from __future__ import annotations


class AuroraError(Exception):
    """Base class for all Aurora errors."""


class ToolError(AuroraError):
    """Raised when a tool fails irrecoverably."""


class ModelUnavailable(AuroraError):
    """Raised when no model backend can serve a request."""


class MaxStepsExceeded(AuroraError):
    """Raised when the agent hits its step budget without finishing."""
