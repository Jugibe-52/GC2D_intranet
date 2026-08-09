"""Internal integration result passed from numerical methods to the runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


DiagnosticValue: TypeAlias = np.ndarray | float | int | str | bool


@dataclass(frozen=True, slots=True)
class IntegrationData:
	"""Physical time history and method/formulation diagnostics."""

	t: np.ndarray
	states: np.ndarray
	diagnostics: dict[str, DiagnosticValue]


__all__ = ["DiagnosticValue", "IntegrationData"]
