"""Reproducible potential configurations shared by experiment notebooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from potential import Potential


@dataclass(frozen=True, slots=True)
class RandomPotentialConfig:
	"""Parameters for a reproducible random study potential.

	New studies use cubic reflected fields. Select direct periodic extension
	explicitly to reconstruct historical random-field experiments.
	"""

	amplitude: float = 0.7
	max_wave_number: int = 25
	nx: int = 64
	ny: int = 64
	seed: int = 27
	interpolation_order: int = 3
	spatial_extension: str = "reflected"

	def build(self) -> Potential:
		"""Construct the configured deterministic potential."""
		return Potential.random(
			A=self.amplitude,
			M=self.max_wave_number,
			nx=self.nx,
			ny=self.ny,
			seed=self.seed,
			interpolation_order=self.interpolation_order,
			spatial_extension=self.spatial_extension,
		)

	def metadata(self) -> dict[str, Any]:
		"""Return JSON-compatible parameters for persisted experiment metadata."""
		return asdict(self)


__all__ = ["RandomPotentialConfig"]
