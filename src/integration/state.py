"""One accepted internal state, independent of saved physical samples."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class IntegrationState:
	"""Own a stable internal vector and its time, in the formulation's units.

	The formulation interprets values; the controller replaces this snapshot
	after each accepted step. Shadow maps receive independent work arrays.
	"""

	time: float
	values: np.ndarray

	def __post_init__(self) -> None:
		value = np.array(self.values, copy=True)
		if not np.isfinite(self.time) or value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
			raise ValueError("An integration state needs a finite time and finite non-empty vector.")
		value.setflags(write=False)
		object.__setattr__(self, 'time', float(self.time))
		object.__setattr__(self, 'values', value)

	def __reduce__(self) -> tuple[type[IntegrationState], tuple[float, np.ndarray]]:
		return (IntegrationState, (self.time, self.values))
