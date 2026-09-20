"""Fully extended BM4 arithmetic projection with ABBA observer conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynamics import GuidingCenterDynamics

from ...formulations import gc_coupling_matrix
from ...formulations.base import Projection
from ..abba.maps.extended import (
	_checked_extended_state, _flow_first, _flow_second, _synchronized_extended_time,
)
from ._core import _advance_composition



@dataclass(frozen=True, slots=True)
class _PreparedExtendedBM4:
	"""Direct/adjoint shears of two (x,y,t,k) copies, with spatial coupling."""

	dynamics: GuidingCenterDynamics
	coupling_frequency: float
	initial_internal_state: np.ndarray

	@property
	def dynamics_name(self) -> str:
		return type(self.dynamics).__name__

	def _couple(self, duration: float, state: np.ndarray) -> np.ndarray:
		"""Rotate the spatial copy difference, leaving both times and k unchanged."""
		value = _checked_extended_state(state, duplicated=True)
		indices = np.asarray([0, 1, 4, 5])
		value[indices] = gc_coupling_matrix(duration, self.coupling_frequency) @ value[indices]
		return value

	def direct_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Apply first and second shears, then the exact coupling flow."""
		value = _flow_first(self.dynamics, state, duration)
		value = _flow_second(self.dynamics, value, duration)
		return self._couple(duration, value)

	def adjoint_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Reverse the subflow ordering; time is intrinsic to each extended copy."""
		value = self._couple(duration, state)
		value = _flow_second(self.dynamics, value, duration)
		return _flow_first(self.dynamics, value, duration)

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Expose the physical mean for the prepared-formulation contract."""
		return np.asarray((internal_history[:2] + internal_history[4:6]) / 2.0), {}


def _midpoint_extended_bm4_step(
	prepared: _PreparedExtendedBM4, state: np.ndarray, step: float,
) -> tuple[np.ndarray, float]:
	"""Apply the shared complete BM4 cycle and average both full copies."""
	value = _checked_extended_state(state, duplicated=False)
	mapped = _advance_composition(
		prepared, float(value[2]), np.concatenate((value, value)), step,
		step_index=0, stage_observer=None,
		formulation_name="FullyExtendedBM4", method_name="BM4Midpoint",
	)
	accepted = _synchronized_extended_time(
		(mapped[:4] + mapped[4:]) / 2.0, float(value[2] + step),
		context="The fully extended BM4 midpoint map",
	)
	return accepted, float(np.linalg.norm(mapped[:4] - mapped[4:], ord=np.inf))


__all__: list[str] = []
