"""Trajectory entities and state-layout rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import numpy as np

from contracts import Array, TrajectoryKind

InitializationKind = Literal["random", "fixed", "selected"]


class Trajectory(ABC):
	"""A trajectory model independent from potentials and numerical solvers."""

	kind: ClassVar[TrajectoryKind]
	state_dimension: ClassVar[int]
	degrees_of_freedom: ClassVar[int]

	def __init__(
		self,
		*,
		rho: float = 0.0,
		eta: float = 0.0,
		n_trajectories: int = 20,
		initialization: InitializationKind = "fixed",
		x0: Array | None = None,
		y0: Array | None = None,
		seed: int | None = 27,
	) -> None:
		self.rho = float(rho)
		self.eta = float(eta)
		if not np.isfinite(self.rho) or self.rho < 0:
			raise ValueError("`rho` must be a finite, non-negative number.")
		if not np.isfinite(self.eta):
			raise ValueError("`eta` must be finite.")
		if (
			isinstance(n_trajectories, (bool, np.bool_))
			or not isinstance(n_trajectories, (int, np.integer))
			or n_trajectories < 1
		):
			raise ValueError("`n_trajectories` must be a positive integer.")
		if initialization not in {"random", "fixed", "selected"}:
			raise ValueError("`initialization` must be 'random', 'fixed' or 'selected'.")
		self.n_trajectories = int(n_trajectories)
		self.initialization = initialization
		self.x0 = None if x0 is None else np.asarray(x0, dtype=float)
		self.y0 = None if y0 is None else np.asarray(y0, dtype=float)
		self.seed = None if seed is None else int(seed)

	def __repr__(self) -> str:
		return (
			f"{self.__class__.__name__}(rho={self.rho!r}, eta={self.eta!r}, "
			f"n_trajectories={self.n_trajectories!r}, "
			f"initialization={self.initialization!r})"
		)

	def split_state(self, state: Array, *, dimension: int | None = None) -> tuple[Array, ...]:
		"""Split a block-layout state along its first axis."""
		state_array = np.asarray(state)
		component_count = self.state_dimension if dimension is None else int(dimension)
		if state_array.ndim == 0 or state_array.shape[0] % component_count != 0:
			raise ValueError(
				f"The first state dimension must be divisible by {component_count} "
				f"for {self.__class__.__name__}."
			)
		return tuple(np.split(state_array, component_count, axis=0))

	def get_positions(self, state: Array) -> tuple[Array, Array]:
		x, y, *_ = self.split_state(state)
		return x, y

	@abstractmethod
	def get_velocities(self, state: Array) -> tuple[Array, Array] | None:
		"""Return independent velocity blocks when the model stores them."""

	def initial_state(
		self,
		x_bounds: tuple[float, float],
		y_bounds: tuple[float, float],
		*,
		n_trajectories: int | None = None,
		initialization: InitializationKind | None = None,
		rng: np.random.Generator | np.random.RandomState | None = None,
	) -> Array:
		"""Build a state using only spatial bounds and trajectory policy."""
		n = self.n_trajectories if n_trajectories is None else int(n_trajectories)
		if n < 1:
			raise ValueError("`n_trajectories` must be positive.")
		method = self.initialization if initialization is None else initialization
		if method not in {"random", "fixed", "selected"}:
			raise ValueError("`initialization` must be 'random', 'fixed' or 'selected'.")
		for bounds, name in ((x_bounds, "x_bounds"), (y_bounds, "y_bounds")):
			if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
				raise ValueError(f"`{name}` must contain two finite increasing values.")
		random = np.random.RandomState(self.seed) if rng is None else rng
		if method == "random":
			x = random.uniform(x_bounds[0], x_bounds[1], n)
			y = random.uniform(y_bounds[0], y_bounds[1], n)
		elif method == "fixed":
			points_per_axis = int(np.sqrt(n))
			x_axis = np.linspace(*x_bounds, points_per_axis, endpoint=False)
			y_axis = np.linspace(*y_bounds, points_per_axis, endpoint=False)
			x_mesh, y_mesh = np.meshgrid(x_axis, y_axis, indexing="ij")
			x, y = x_mesh.ravel(), y_mesh.ravel()
		else:
			if self.x0 is None or self.y0 is None:
				raise ValueError("Selected initialization requires both `x0` and `y0`.")
			if self.x0.shape != self.y0.shape:
				raise ValueError("`x0` and `y0` must have the same shape.")
			x, y = self.x0.ravel()[:n], self.y0.ravel()[:n]
			if x.size != n:
				raise ValueError(
					f"Selected initialization requested {n} trajectories but received {x.size}."
				)
		self.n_trajectories = int(x.size)
		return self._complete_initial_state(np.asarray(x), np.asarray(y), random)

	@abstractmethod
	def _complete_initial_state(
		self,
		x: Array,
		y: Array,
		rng: np.random.Generator | np.random.RandomState,
	) -> Array:
		"""Complete position blocks with model-specific state variables."""


__all__ = ["InitializationKind", "Trajectory"]
