"""Full-cyclotron initial configurations and their state layout."""

from __future__ import annotations

import numpy as np

from contracts.state_layout import FCState, FCStateLayout
from .base import StateConfiguration


_FC_STATE_LAYOUT = FCStateLayout()


class FCInitialConfiguration(StateConfiguration):
	"""Full-cyclotron state in component-major ``[x, y, vx, vy]`` order.

	For ``N`` particles, the initial state contains all ``x`` values, then all
	``y``, ``vx`` and ``vy`` values. Physical ``rho`` and ``eta`` parameters are
	owned exclusively by the corresponding dynamics object.
	"""

	@property
	def layout(self) -> FCStateLayout:
		"""Return the shared stateless full-cyclotron layout."""
		return _FC_STATE_LAYOUT

	@classmethod
	def pack_components(cls, *components: np.ndarray) -> np.ndarray:
		"""Compatibility façade for the full-cyclotron layout builder."""
		return FCStateLayout.pack_components(*components)

	@classmethod
	def from_components(
		cls,
		*,
		x: np.ndarray,
		y: np.ndarray,
		vx: np.ndarray,
		vy: np.ndarray,
	) -> FCInitialConfiguration:
		"""Create an FC configuration without exposing packed state order."""
		state = FCStateLayout.pack_components(x, y, vx, vy)
		return cls(state)

	def velocities(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return normalized ``vx`` and ``vy`` coordinate blocks."""
		components = self.layout.split(state)
		return components.vx, components.vy

	def split(self, state: np.ndarray) -> FCState:
		"""Compatibility façade returning named physical component blocks."""
		return self.layout.split(state)

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Compatibility façade returning both position blocks."""
		return self.layout.positions(state)


__all__ = [
	"FCInitialConfiguration",
	"FCState",
	"FCStateLayout",
]
