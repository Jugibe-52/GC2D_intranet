"""One accepted physical state, independent of saved physical samples."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class IntegrationState:
	"""Own a stable physical vector and its time, in the physical model's units.

	Values contain one physical copy; the controller replaces this snapshot
	after each accepted step. Shadow maps receive independent work arrays.
	"""

	time: float
	values: np.ndarray
	momentum: np.ndarray | None = None

	def __post_init__(self) -> None:
		value: np.ndarray = np.array(self.values, copy=True)
		if not np.isfinite(self.time) or value.ndim != 1 or not value.size or not np.all(np.isfinite(value)):
			raise ValueError("An integration state needs a finite time and finite non-empty vector.")
		value.setflags(write=False)
		object.__setattr__(self, 'time', float(self.time))
		object.__setattr__(self, 'values', value)
		if self.momentum is not None:
			momentum: np.ndarray = np.array(self.momentum, copy=True)
			if momentum.ndim != 1 or not momentum.size or not np.all(np.isfinite(momentum)):
				raise ValueError('Momentum must be a finite non-empty vector.')
			momentum.setflags(write=False)
			object.__setattr__(self, 'momentum', momentum)

	def __reduce__(self) -> tuple[type[IntegrationState], tuple[float, np.ndarray, np.ndarray | None]]:
		return (IntegrationState, (self.time, self.values, self.momentum))
