"""Scalar and array values carried by result diagnostics."""

from typing import TypeAlias
import numpy as np

DiagnosticValue: TypeAlias = np.ndarray | float | int | str | bool

__all__ = ["DiagnosticValue"]
