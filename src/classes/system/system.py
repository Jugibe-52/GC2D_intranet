"""Base composition of one potential and one trajectory."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from classes.potential import Potential
from classes.trajectory.trajectory import Trajectory

from .solution import Solution
from .observation import StageObserver


class System(ABC):
	"""Hamiltonian dynamics created from independent domain entities.

	``potential`` supplies the scalar field and its derivatives. ``trajectory``
	defines both the physical parameters and the component-major state layout;
	the system combines them into equations of motion and an integrator.
	"""

	def __init__(self, potential: Potential, trajectory: Trajectory) -> None:
		"""Bind a physical potential to a compatible trajectory description."""
		if not isinstance(potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(trajectory, Trajectory):
			raise TypeError("`trajectory` must be a Trajectory instance.")
		self.potential = potential
		self.trajectory = trajectory

	@abstractmethod
	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the equations of motion with the same shape as ``state``."""

	@abstractmethod
	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate one Hamiltonian value per particle and optional saved time."""

	@abstractmethod
	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return one conjugate-momentum derivative per represented particle."""

	def simulate(
		self,
		*,
		step: float,
		t_span: tuple[float, float] = (0.0, 2 * np.pi),
		n_save_step: int = 361,
		check_energy: bool = False,
		progress: bool = False,
		stage_observer: StageObserver | None = None,
	) -> Solution:
		"""Integrate the stored initial state with the fixed BM4 path.

		``step`` bounds internal steps, whereas ``n_save_step`` controls only
		the uniformly spaced states exposed in the returned solution. ``t_span``
		contains the initial and final simulation times. ``check_energy`` augments
		the state with momentum conjugate to time; ``progress`` affects only the
		terminal display, never the numerical result. ``stage_observer`` receives
		independent diagnostic snapshots of every direct and adjoint BM4 stage; it is
		intended for research instrumentation and is disabled by default.
		"""
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
			stage_observer=stage_observer,
		)

	def _energy_error(self, solution: Solution) -> float:
		"""Return the maximum drift of physical or generalized energy.

		For a time-dependent Hamiltonian, ``k`` is the momentum conjugate to
		time and makes ``H + k`` the conserved extended-system quantity. The
		reduction compares every saved value with its particle's initial value and
		returns the largest absolute drift across particles and time.
		"""
		energy = np.asarray(self.hamiltonian(solution.t, solution.y), dtype=float)
		if solution.k is not None:
			energy = energy + np.asarray(solution.k)
		# ``energy`` is normally (particles, saved_times). A single particle may
		# omit that first axis, so insert it before reducing each time series.
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
		stage_observer: StageObserver | None,
	) -> Solution:
		"""Run the concrete GC or FC integration path."""


__all__ = ["System"]
