"""Composition root for potentials, trajectories and numerical simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from contracts import Array
from classes.potential import Potential
from classes.trajectory import Trajectory, TrajectoryFC, TrajectoryGC

if TYPE_CHECKING:
	from .solution import Solution


class System(ABC):
	"""A Hamiltonian system obtained by binding one potential and trajectory.

	``Potential`` and ``Trajectory`` remain independent entities.  This class is
	the only layer that knows both and the only domain object that invokes the
	numerical solver.
	"""

	time_dependent = True

	def __init__(self, potential: Potential, trajectory: Trajectory) -> None:
		if not isinstance(potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(trajectory, Trajectory):
			raise TypeError("`trajectory` must be a Trajectory instance.")
		self.potential = potential
		self.trajectory = trajectory
		self.physical_potential = self.potential
		self.effective_potential = self._build_effective_potential()
		# Optional workflow metadata is deliberately untyped here so the domain
		# layer does not depend on configuration or presentation modules.
		self.options: object | None = None
		self.modulo = False
		self.show_grid = False

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}(potential={self.potential!r}, "
			f"trajectory={self.trajectory!r})"
		)

	def __str__(self) -> str:
		return f"2D {self.__class__.__name__} ({self.trajectory.kind})"

	@property
	def kind(self) -> str:
		return self.trajectory.kind

	@property
	def degrees_of_freedom(self) -> int:
		return self.trajectory.degrees_of_freedom

	@property
	def grid(self) -> Any:
		return self.potential.grid

	@abstractmethod
	def _build_effective_potential(self) -> Potential:
		"""Create the potential view used by this trajectory model."""

	def phi(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the physical potential."""
		return self.physical_potential.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

	def psi(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		dx: int = 0,
		dy: int = 0,
		dt: int = 0,
	) -> Array:
		"""Evaluate the effective potential used by the equations of motion."""
		return self.effective_potential.field_at_time(t, x, y, dx=dx, dy=dy, dt=dt)

	def electric_field(
		self,
		t: float | Array,
		x: Array | None = None,
		y: Array | None = None,
		*,
		effective: bool = True,
	) -> tuple[Array, Array]:
		potential = self.effective_potential if effective else self.physical_potential
		return potential.electric_field(t, x, y)

	def get_positions(self, state: Array) -> tuple[Array, Array]:
		return self.trajectory.get_positions(state)

	def get_velocities(self, state: Array) -> tuple[Array, Array] | None:
		return self.trajectory.get_velocities(state)

	@abstractmethod
	def vector_field(self, t: float, state: Array) -> Array:
		"""Evaluate the equations of motion."""

	def y_dot(self, t: float, state: Array) -> Array:
		"""Conventional Hamiltonian alias for :meth:`vector_field`."""
		return self.vector_field(t, state)

	@abstractmethod
	def hamiltonian(self, t: float | Array, state: Array) -> Array:
		"""Evaluate the Hamiltonian for every trajectory."""

	@abstractmethod
	def extended_momentum_derivative(self, t: float, state: Array) -> Array:
		"""Return :math:`-\partial H/\partial t` for every trajectory."""

	def k_dot(self, t: float, state: Array) -> Array:
		return self.extended_momentum_derivative(t, state)

	def _spatial_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
		grid = self.grid
		if grid.period is None:
			return (grid.xmin, grid.xmax), (grid.ymin, grid.ymax)
		return (
			(grid.xmin, grid.xmin + grid.period),
			(grid.ymin, grid.ymin + grid.period),
		)

	def initial_state(self, **overrides: Any) -> Array:
		"""Generate an initial state using the trajectory's initialization policy."""
		x_bounds, y_bounds = self._spatial_bounds()
		return self.trajectory.initial_state(x_bounds, y_bounds, **overrides)

	def compute_energy(self, solution: Solution, *, max_error: bool = True) -> Array:
		"""Evaluate physical or extended energy along a solution."""
		values = np.asarray(self.hamiltonian(solution.t[np.newaxis], solution.y))
		if hasattr(solution, "k"):
			values = values + np.asarray(solution.k)
		if values.ndim == 1:
			values = values[np.newaxis, :]
		reference = values[:, :1]
		if max_error:
			return np.asarray(np.max(np.abs(values - reference)))
		return values

	def simulate(
		self,
		y0: Array | None = None,
		*,
		t_span: tuple[float, float] = (0.0, 2 * np.pi),
		step: float,
		t_eval: Array | None = None,
		save_step: float | None = None,
		n_save_step: int | None = None,
		method: str = "BM4",
		check_energy: bool = False,
		command: Callable[[float, Array], None] | None = None,
		progress: bool = False,
	) -> Solution:
		"""Integrate this system with the trajectory-appropriate symplectic path."""
		from .solver import solve_extended, solve_symplectic

		state = self.initial_state() if y0 is None else np.asarray(y0)
		# Validate the physical layout before adding any solver-owned variables.
		self.trajectory.split_state(state)
		if isinstance(self.trajectory, TrajectoryGC):
			return solve_extended(
				self,
				t_span,
				state,
				step,
				t_eval,
				method,
				command=command,
				check_energy=check_energy,
				save_step=save_step,
				n_save_step=n_save_step,
				progress=progress,
			)
		if not isinstance(self.trajectory, TrajectoryFC):
			raise TypeError(f"Unsupported system class: {type(self).__name__}.")
		flow = getattr(self, "flow", None)
		adjoint_flow = getattr(self, "adjoint_flow", None)
		if not callable(flow) or not callable(adjoint_flow):
			raise TypeError("A full-cyclotron system must provide flow and adjoint_flow.")

		n_trajectories = state.shape[0] // self.trajectory.state_dimension
		integrator_state = state
		if check_energy:
			integrator_state = np.concatenate((state, np.zeros(n_trajectories)))
		solution = solve_symplectic(
			lambda h, t, value: flow(h, t, value, check_energy=check_energy),
			lambda h, t, value: adjoint_flow(h, t, value, check_energy=check_energy),
			t_span,
			integrator_state,
			step,
			t_eval,
			method,
			command,
			save_step=save_step,
			n_save_step=n_save_step,
		)
		if check_energy:
			components = np.split(solution.y, 5, axis=0)
			solution.y = np.concatenate(components[:4], axis=0)
			solution.k = components[4]
			solution.err = self.compute_energy(solution)
		return solution


def create_system(potential: Potential, trajectory: Trajectory) -> System:
	"""Bind independent entities into the matching concrete system."""
	if isinstance(trajectory, TrajectoryGC):
		from .gc import SystemGC

		return SystemGC(potential, trajectory)
	if isinstance(trajectory, TrajectoryFC):
		from .fc import SystemFC

		return SystemFC(potential, trajectory)
	return _unsupported_trajectory(trajectory)


def _unsupported_trajectory(trajectory: Trajectory) -> System:
	raise TypeError(f"Unsupported trajectory class: {type(trajectory).__name__}.")


__all__ = ["System", "create_system"]
