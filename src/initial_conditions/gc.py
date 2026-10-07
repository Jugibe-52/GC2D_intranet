"""Guiding-centre initial configurations and their state layout."""

from __future__ import annotations

import numpy as np

from contracts.state_layout import GCState, GCStateLayout
from .base import StateConfiguration


_GC_STATE_LAYOUT = GCStateLayout()


class GCInitialConfiguration(StateConfiguration):
	"""Guiding-centre positions stored in component-major order ``[x, y]``.

	For ``N`` particles, the flat initial state is
	``[x_1, ..., x_N, y_1, ..., y_N]``.  No velocity block is needed because the
	GC equations determine coordinate rates directly from the effective field.
	"""

	@property
	def layout(self) -> GCStateLayout:
		"""Return the shared stateless guiding-centre layout."""
		return _GC_STATE_LAYOUT

	@classmethod
	def pack_components(cls, *components: np.ndarray) -> np.ndarray:
		"""Compatibility façade for the guiding-centre layout builder."""
		return GCStateLayout.pack_components(*components)

	@classmethod
	def from_components(
		cls,
		*,
		x: np.ndarray,
		y: np.ndarray,
	) -> GCInitialConfiguration:
		"""Create a GC configuration without exposing packed state order."""
		state = GCStateLayout.pack_components(x, y)
		return cls(state)

	def split(self, state: np.ndarray) -> GCState:
		"""Compatibility façade returning named coordinate blocks."""
		return self.layout.split(state)

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Compatibility façade returning both position blocks."""
		return self.layout.positions(state)


__all__ = [
	"GCInitialConfiguration",
	"GCState",
	"GCStateLayout",
]
