"""Structural contracts consumed by visualization.energy."""

from __future__ import annotations

from typing import Protocol
import numpy as np


class GeneralizedEnergyRunView(Protocol):
	"""Step-dependent arrays consumed by generalized-energy plots."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def times(self) -> np.ndarray:
		"""Accepted-node time grid."""
		...

	@property
	def hamiltonian(self) -> np.ndarray:
		"""Physical Hamiltonian values."""
		...

	@property
	def kappa(self) -> np.ndarray:
		"""Normalized time-conjugate momentum values."""
		...

	@property
	def generalized_energy(self) -> np.ndarray:
		"""Generalized-energy values."""
		...

	@property
	def relative_errors(self) -> np.ndarray:
		"""Signed generalized-energy relative errors."""
		...

	@property
	def running_max_relative_errors(self) -> np.ndarray:
		"""Running maximum absolute relative errors."""
		...


class GeneralizedEnergySummaryView(Protocol):
	"""Scalar refinement fields consumed by the convergence plot."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def max_absolute_error(self) -> float:
		"""Maximum absolute generalized-energy drift."""
		...


class EnergyAccuracySeriesView(Protocol):
	"""Physical-Hamiltonian reference errors consumed by comparison plots."""

	@property
	def errors(self) -> np.ndarray:
		"""Return signed particle-time Hamiltonian errors."""
		...


class ExtendedSymplecticityRunView(Protocol):
	"""Step-dependent arrays consumed by the ``R^6`` defect plot."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def extended_symplecticity_times(self) -> np.ndarray:
		"""Accepted final times of the extended-map measurements."""
		...

	@property
	def extended_relative_defects(self) -> np.ndarray:
		"""Relative ``D Psi.T Omega D Psi - Omega`` defects."""
		...

	@property
	def extended_determinant_errors(self) -> np.ndarray:
		"""Absolute volume-determinant errors."""
		...


class ReducedExtendedSymplecticityRunView(Protocol):
	"""Arrays consumed by the complete projected ``R^4`` defect plot."""

	@property
	def actual_step(self) -> float:
		"""Accepted uniform integration step."""
		...

	@property
	def reduced_extended_symplecticity_times(self) -> np.ndarray:
		"""Accepted times of the projected physical map measurements."""
		...

	@property
	def reduced_extended_relative_defects(self) -> np.ndarray:
		"""Relative projected ``R^4`` symplecticity defects."""
		...

	@property
	def reduced_extended_determinant_errors(self) -> np.ndarray:
		"""Projected ``R^4`` determinant errors."""
		...


__all__ = ['GeneralizedEnergyRunView', 'GeneralizedEnergySummaryView', 'EnergyAccuracySeriesView', 'ExtendedSymplecticityRunView', 'ReducedExtendedSymplecticityRunView']
