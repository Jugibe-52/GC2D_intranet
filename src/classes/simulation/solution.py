"""Trajectory-like numerical result returned by every simulation method."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .configuration import InitialConfiguration


class Solution:
	"""Physical trajectory plus method-independent named diagnostics.

	``t`` is the explicit temporal dimension and ``states`` has shape
	``(physical_state_size, saved_times)``.  ``source`` is the initial
	configuration whose layout interprets every column.

	The legacy attributes ``y``, ``trajectory``, ``n_steps``, ``k`` and ``err``
	remain read/write aliases so existing notebooks can migrate independently.
	"""

	__slots__ = ("t", "states", "source", "diagnostics")

	def __init__(
		self,
		t: np.ndarray,
		y: np.ndarray | None = None,
		n_steps: int | None = None,
		k: np.ndarray | None = None,
		err: float | None = None,
		trajectory: InitialConfiguration | None = None,
		*,
		states: np.ndarray | None = None,
		source: InitialConfiguration | None = None,
		diagnostics: Mapping[str, Any] | None = None,
	) -> None:
		"""Create a canonical result while accepting the legacy positional order."""
		if states is not None and y is not None:
			raise TypeError("Provide either `states` or legacy `y`, not both.")
		if source is not None and trajectory is not None:
			raise TypeError("Provide either `source` or legacy `trajectory`, not both.")
		value = states if states is not None else y
		if value is None:
			raise TypeError("`states` is required.")
		times = np.asarray(t, dtype=float)
		state_history = np.asarray(value)
		if (
			times.ndim != 1
			or times.size < 2
			or state_history.ndim != 2
			or state_history.shape[1] != times.size
			or state_history.shape[0] == 0
		):
			raise ValueError(
				"`t` and `states` must have shapes (T,) and (state_size, T)."
			)
		self.t = times
		self.states = state_history
		self.source = source if source is not None else trajectory
		self.diagnostics = dict(diagnostics or {})
		if n_steps is not None:
			self.n_steps = n_steps
		if k is not None:
			self.k = k
		if err is not None:
			self.err = err

	@property
	def y(self) -> np.ndarray:
		"""Compatibility alias for :attr:`states`."""
		return self.states

	@y.setter
	def y(self, value: np.ndarray) -> None:
		state_history = np.asarray(value)
		if state_history.ndim != 2 or state_history.shape[1] != self.t.size:
			raise ValueError("`y` must have shape (state_size, saved_times).")
		self.states = state_history

	@property
	def trajectory(self) -> InitialConfiguration | None:
		"""Compatibility alias for the source initial configuration."""
		return self.source

	@trajectory.setter
	def trajectory(self, value: InitialConfiguration | None) -> None:
		self.source = value

	@property
	def n_steps(self) -> int:
		"""Compatibility view of the common step-count diagnostic."""
		return int(self.diagnostics.get("step_count", 0))

	@n_steps.setter
	def n_steps(self, value: int) -> None:
		if isinstance(value, (bool, np.bool_)) or int(value) < 0:
			raise ValueError("`n_steps` must be a non-negative integer.")
		self.diagnostics["step_count"] = int(value)

	@property
	def k(self) -> np.ndarray | None:
		"""Compatibility view of extended momentum."""
		value = self.diagnostics.get("extended_momentum")
		return None if value is None else np.asarray(value)

	@k.setter
	def k(self, value: np.ndarray | None) -> None:
		if value is None:
			self.diagnostics.pop("extended_momentum", None)
		else:
			self.diagnostics["extended_momentum"] = np.asarray(value)

	@property
	def err(self) -> float | None:
		"""Compatibility view of maximum generalized-energy drift."""
		value = self.diagnostics.get("energy_error")
		return None if value is None else float(value)

	@err.setter
	def err(self, value: float | None) -> None:
		if value is None:
			self.diagnostics.pop("energy_error", None)
		else:
			self.diagnostics["energy_error"] = float(value)

	def components(
		self,
		trajectory: InitialConfiguration | None = None,
	) -> tuple[np.ndarray, ...]:
		"""Return named physical components over the complete time history."""
		layout = self.source if trajectory is None else trajectory
		if layout is None:
			raise ValueError(
				"`trajectory` is required because this solution has no source layout."
			)
		if self.source is not None and layout.state_dimension != self.source.state_dimension:
			raise TypeError("The supplied trajectory is incompatible with this solution.")
		return layout.split(self.states)

	def positions(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return the time histories of both physical position components."""
		if self.source is None:
			raise ValueError("This solution has no source initial configuration.")
		return self.source.positions(self.states)


__all__ = ["Solution"]
