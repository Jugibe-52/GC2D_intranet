"""Independent snapshots of already evaluated Newton iterates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


@dataclass(frozen=True, slots=True)
class NewtonIteration:
	"""One evaluated iterate, including the initial guess and accepted root.

	``unknown`` and ``residual`` use the method's packed nonlinear coordinates.
	Projection events additionally expose the physical multiplier; SDIRK events
	identify their zero-based stage. Arrays are independent read-only snapshots.
	Callbacks observe advances, including off-grid output solves. Maps exposed
	by step observations omit callbacks so later diagnostic replay cannot alter
	the integration history. Observation adds no residual/field evaluations.
	"""

	time: float
	duration: float
	iteration: int
	unknown: np.ndarray
	residual: np.ndarray
	residual_norm: float
	tolerance: float
	multiplier: np.ndarray | None = None
	stage_index: int | None = None

	def __post_init__(self) -> None:
		"""Keep retained or modified observations independent of solver storage."""
		for name in ("unknown", "residual", "multiplier"):
			value = getattr(self, name)
			if value is not None:
				owned = np.asarray(value, dtype=float).copy()
				owned.setflags(write=False)
				object.__setattr__(self, name, owned)


NewtonObserver: TypeAlias = Callable[[NewtonIteration], None]

__all__ = ["NewtonIteration", "NewtonObserver"]
