"""Base composition of one potential and one trajectory."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from classes.potential import Potential
from classes.trajectory.trajectory import Trajectory

from .solution import Solution


class System(ABC):
	"""Hamiltonian dynamics created from independent domain entities."""

	def __init__(self, potential: Potential, trajectory: Trajectory) -> None:
		if not isinstance(potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(trajectory, Trajectory):
			raise TypeError("`trajectory` must be a Trajectory instance.")
		self.potential = potential
		self.trajectory = trajectory

	@abstractmethod
	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the equations of motion."""

	@abstractmethod
	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the Hamiltonian for every trajectory."""

	@abstractmethod
	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return minus the explicit time derivative of the Hamiltonian."""

	def simulate(
		self,
		*,
		step: float,
		t_span: tuple[float, float] = (0.0, 2 * np.pi),
		n_save_step: int = 361,
		check_energy: bool = False,
		progress: bool = False,
	) -> Solution:
		"""Integrate the trajectory with the fixed BM4 numerical path."""
		state = self.trajectory.state
		if state is None:
			raise ValueError("The trajectory has no initial state.")
		return self._integrate(
			state,
			step=step,
			t_span=t_span,
			n_save_step=n_save_step,
			check_energy=check_energy,
			progress=progress,
		)

	def _energy_error(self, solution: Solution) -> float:
		"""Return the maximum drift of physical or generalized energy."""
		energy = np.asarray(self.hamiltonian(solution.t, solution.y), dtype=float)
		if solution.k is not None:
			energy = energy + np.asarray(solution.k)
		if energy.ndim == 1:
			energy = energy[np.newaxis, :]
		return float(np.max(np.abs(energy - energy[:, :1])))

	@abstractmethod
	def _integrate(
		self,
		state: np.ndarray,
		*,
		step: float,
		t_span: tuple[float, float],
		n_save_step: int,
		check_energy: bool,
		progress: bool,
	) -> Solution:
		"""Run the concrete GC or FC integration path."""


__all__ = ["System"]
