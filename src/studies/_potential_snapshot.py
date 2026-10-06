"""Shared processed HDF5 fields for spawned study workers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from potential.gc2d_h5 import GC2DH5Metadata
from potential.grid import Grid
from potential.potential import Potential


@dataclass(frozen=True, slots=True)
class _H5PotentialSnapshot:
	"""Pickle-safe processed HDF5 field used to initialize spawned workers."""

	grid: Grid
	mean: np.ndarray
	modes: np.ndarray
	frequencies: np.ndarray
	metadata: GC2DH5Metadata
	interpolation_order: int

	@classmethod
	def from_potential(
		cls,
		potential: Potential,
	) -> _H5PotentialSnapshot:
		"""Capture the selected and resampled fields without the source HDF5."""
		if not isinstance(potential.metadata, GC2DH5Metadata):
			raise TypeError("The potential must contain GC2D HDF5 metadata.")
		return cls(
			grid=potential.grid,
			mean=potential.mean,
			modes=potential.modes,
			frequencies=potential.frequencies,
			metadata=potential.metadata,
			interpolation_order=potential.interpolation_order,
		)

	def restore(self) -> Potential:
		"""Rebuild runtime splines once inside one worker process."""
		return Potential(
			self.grid,
			mean=self.mean,
			modes=self.modes,
			frequencies=self.frequencies,
			metadata=self.metadata,
			interpolation_order=self.interpolation_order,
		)
