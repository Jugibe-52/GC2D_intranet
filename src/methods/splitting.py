"""Compose explicitly supplied exact cyclotron and electric-kick flows."""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any
import numpy as np
from contracts.dynamics.coordinates import StateLayout
from contracts.methods.maps import Projection


ExactFlow = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True, slots=True)
class FCSplitMaps:
	"""Signed direct/adjoint composition on physical component-major FC arrays."""

	cyclotron_flow: ExactFlow
	electric_kick: ExactFlow
	physical_size: int
	coordinates: StateLayout

	def __post_init__(self) -> None:
		if (self.coordinates.component_names != ('x', 'y', 'vx', 'vy')
				or self.coordinates.position_names != ('x', 'y')
				or self.coordinates.velocity_names != ('vx', 'vy')):
			raise TypeError('FC split maps require declared (x, y, vx, vy) coordinates.')
		if isinstance(self.physical_size, bool) or not isinstance(self.physical_size, Integral) or self.physical_size <= 0 or self.physical_size % 4:
			raise ValueError('FC maps require complete physical particle blocks.')
		if not callable(self.cyclotron_flow) or not callable(self.electric_kick):
			raise TypeError('Supply the exact cyclotron flow and electric kick as functions.')

	@property
	def particle_count(self) -> int:
		return self.physical_size // 4

	def _checked(self, state: np.ndarray) -> np.ndarray:
		value = np.asarray(state)
		if value.ndim == 0 or value.shape[0] != self.physical_size:
			raise ValueError('The FC map changed the physical state shape.')
		return value

	def direct_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Apply the cyclotron flow followed by the electric kick."""
		return self._checked(self.electric_kick(t, self.cyclotron_flow(t, self._checked(state), duration), duration))

	def adjoint_map(self, duration: float, t: float, state: np.ndarray) -> np.ndarray:
		"""Apply the electric kick followed by the cyclotron flow."""
		return self._checked(self.cyclotron_flow(t, self.electric_kick(t, self._checked(state), duration), duration))

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Expose physical samples without a single-copy formulation wrapper."""
		return self._checked(internal_history), {}


__all__ = ['FCSplitMaps']
