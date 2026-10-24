"""Functions and derivative layouts supplied to numerical algorithms."""

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

VectorField: TypeAlias = Callable[[Any, Any], Any]
JacobianStructure: TypeAlias = Literal["dense", "particle"]

__all__ = ["VectorField", "JacobianStructure"]
