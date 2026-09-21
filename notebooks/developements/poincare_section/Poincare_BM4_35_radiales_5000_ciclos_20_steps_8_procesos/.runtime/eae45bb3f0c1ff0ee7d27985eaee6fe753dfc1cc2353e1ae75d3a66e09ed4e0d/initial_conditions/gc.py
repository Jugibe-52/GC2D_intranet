"""Guiding-centre initial configurations and their state layout."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .base import PackedStateLayout, StateConfiguration


class GCState(NamedTuple):
	"""Guiding-centre coordinates with matching particle/sample dimensions.

	Each field has shape ``(N, *sample_axes)``: axis zero identifies the
	particle (or contour vertex) and trailing axes identify solution samples.
	"""

	x: np.ndarray
	y: np.ndarray


class GCStateLayout(PackedStateLayout):
	"""Interpret packed guiding-centre coordinates in ``[x, y]`` order."""

	__slots__ = ()
	state_dimension = 2

	def split(self, state: np.ndarray) -> GCState:
		"""Return named ``x`` and ``y`` blocks, preserving sample axes."""
		x, y = super().split(state)
		return GCState(x, y)

	def positions(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return the two guiding-centre coordinate blocks."""
		components = self.split(state)
		return components.x, components.y


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


class TrajectoryGC(GCInitialConfiguration):
	"""Deprecated GC configuration carrying the former ``rho`` metadata."""

	def __init__(
		self,
		state: np.ndarray | None = None,
		*,
		rho: float = 0.0,
	) -> None:
		"""Create a legacy GC trajectory while validating ``rho`` metadata."""
		radius = float(rho)
		if not np.isfinite(radius) or radius < 0:
			raise ValueError("`rho` must be finite and non-negative.")
		self.rho = radius
		super().__init__(state)

	@classmethod
	def from_components(
		cls,
		*,
		x: np.ndarray,
		y: np.ndarray,
		rho: float = 0.0,
	) -> TrajectoryGC:
		"""Create a legacy GC trajectory from named coordinate blocks."""
		return cls(cls.pack_components(x, y), rho=rho)


__all__ = ["GCInitialConfiguration", "GCState", "GCStateLayout", "TrajectoryGC"]
