"""Full-cyclotron system."""

from __future__ import annotations

import numpy as np

from classes.dynamics import FullCyclotronDynamics
from classes.potential import Potential
from classes.simulation.formulations import FCSplitFormulation
from classes.simulation.methods import BM4Composition
from classes.simulation.problem import InitialValueProblem
from classes.simulation.request import SimulationRequest
from classes.simulation.runner import SimulationRunner
from classes.trajectory import TrajectoryFC

from .observation import StageObserver
from .solution import Solution
from .system import System


class SystemFC(System):
	"""Full-cyclotron dynamics over the physical potential.

	The trajectory supplies the component-major ``[x, y, vx, vy]`` initial
	configuration. :class:`FullCyclotronDynamics` owns the physical parameters
	and their derived scales. Unlike GC dynamics, FC uses the raw potential
	because the rapid cyclotron motion remains in the state itself.
	"""

	trajectory: TrajectoryFC
	dynamics: FullCyclotronDynamics

	def __init__(self, potential: Potential, trajectory: TrajectoryFC) -> None:
		"""Construct a full-cyclotron system from compatible domain objects."""
		if not isinstance(trajectory, TrajectoryFC):
			raise TypeError("SystemFC requires a TrajectoryFC instance.")
		super().__init__(potential, trajectory)
		self.dynamics = FullCyclotronDynamics(
			potential,
			rho=trajectory.rho,
			eta=trajectory.eta,
		)

	def electric_acceleration(
		self,
		t: float,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Return the electric contributions to ``dvx/dt`` and ``dvy/dt``.

		The returned arrays match the broadcast shape of ``x`` and ``y``. Their
		signed normalization is carried by the explicit dynamics object.
		"""
		return self.dynamics.electric_acceleration(t, x, y)

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate derivatives with the same ``[x, y, vx, vy]`` state layout."""
		return self.dynamics.vector_field(t, state)

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate kinetic plus scaled electrostatic energy per particle.

		For a saved solution the returned array keeps particle and time axes, so
		it can be combined directly with the extended momentum ``solution.k``.
		"""
		return self.dynamics.hamiltonian(t, state)

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the per-particle derivative of momentum conjugate to time."""
		return self.dynamics.extended_momentum_derivative(t, state)

	def _integrate(
		self,
		state: np.ndarray,
		*,
		step: float,
		t_span: tuple[float, float],
		n_output_samples: int,
		check_energy: bool,
		progress: bool,
		stage_observer: StageObserver | None,
	) -> Solution:
		"""Assemble the legacy BM4 façade from independent architecture objects."""
		stored_state = self.trajectory.state
		if stored_state is None or not np.array_equal(stored_state, state):
			raise ValueError("The supplied state must match the stored initial state.")
		problem = InitialValueProblem(self.dynamics, self.trajectory)
		request = SimulationRequest.uniform(
			t_span=t_span,
			max_step=step,
			sample_count=n_output_samples,
		)
		method = BM4Composition(
			FCSplitFormulation(),
			track_energy=check_energy,
			progress=progress,
			stage_observer=stage_observer,
		)
		return SimulationRunner().simulate(problem, method, request)


__all__ = ["SystemFC"]
