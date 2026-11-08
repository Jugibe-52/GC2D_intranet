"""Array-compatible functions supplied to numerical algorithms."""

from collections.abc import Callable
from typing import Any, TypeAlias

VectorField: TypeAlias = Callable[[Any, Any], Any]

__all__ = ["VectorField"]
