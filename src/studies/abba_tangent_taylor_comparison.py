"""Compare tangent-Taylor ABBA trajectories with their generating base maps."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from dynamics import GuidingCenterDynamics
from potential import Potential
from simulation import (
	ABBA4Implicit1,
	ABBA4Implicit1TangentTaylor,
	ImplicitABBA1,
	ImplicitABBA1TangentTaylor,
	InitialConfiguration,
	InitialValueProblem,
	NONLINEAR_SOLVERS,
	NonlinearSolver,
	NumericalMethod,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import nonnegative_finite, positive_finite, positive_integer


@dataclass(frozen=True, slots=True)
class ABBATangentTaylorComparisonConfig:
	"""Shared physical grid and nonlinear controls for one paired comparison."""

	rho: float = 0.3
	t_span: tuple[float, float] = (0.0, 1.0)
	max_step: float = 0.01
	sample_count: int = 101
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False

	def __post_init__(self) -> None:
		"""Normalize every parameter that changes either paired trajectory."""
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		for name in (
			"max_step",
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)
		if (
			isinstance(self.sample_count, (bool, np.bool_))
			or not isinstance(self.sample_count, (int, np.integer))
			or self.sample_count < 2
		):
			raise ValueError("`sample_count` must be an integer of at least two.")
		object.__setattr__(self, "sample_count", int(self.sample_count))
		if self.nonlinear_solver not in NONLINEAR_SOLVERS:
			raise ValueError("Unknown nonlinear solver for the trajectory comparison.")
		object.__setattr__(self, "progress", bool(self.progress))


@dataclass(frozen=True, slots=True)
class ABBATangentTaylorComparisonResult:
	"""Aligned original and tangent-Taylor trajectories with periodic drift."""

	potential: Potential
	config: ABBATangentTaylorComparisonConfig
	base_method_name: str
	tangent_method_name: str
	base_solution: Solution
	tangent_solution: Solution
	base_runtime_seconds: float
	tangent_runtime_seconds: float

	def __post_init__(self) -> None:
		"""Require aligned finite trajectories and positive measured runtimes."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.config, ABBATangentTaylorComparisonConfig):
			raise TypeError("`config` must be an ABBATangentTaylorComparisonConfig.")
		if not self.base_method_name or not self.tangent_method_name:
			raise ValueError("Both method names must be non-empty.")
		if not isinstance(self.base_solution, Solution) or not isinstance(
			self.tangent_solution,
			Solution,
		):
			raise TypeError("Both trajectory comparison values must be Solutions.")
		if not np.array_equal(self.base_solution.t, self.tangent_solution.t):
			raise ValueError("The compared trajectories must share saved times.")
		if self.base_solution.states.shape != self.tangent_solution.states.shape:
			raise ValueError("The compared trajectories must share physical layout.")
		if not all(
			np.isfinite(value) and value > 0.0
			for value in (self.base_runtime_seconds, self.tangent_runtime_seconds)
		):
			raise ValueError("Both measured runtimes must be positive and finite.")

	@property
	def periodic_displacement_components(self) -> tuple[np.ndarray, np.ndarray]:
		"""Return minimum-image tangent-minus-base coordinate displacements."""
		base_x, base_y = self.base_solution.positions()
		tangent_x, tangent_y = self.tangent_solution.positions()
		period = float(self.potential.grid.period)
		delta_x = (tangent_x - base_x + period / 2.0) % period - period / 2.0
		delta_y = (tangent_y - base_y + period / 2.0) % period - period / 2.0
		return np.asarray(delta_x), np.asarray(delta_y)

	@property
	def particle_distances(self) -> np.ndarray:
		"""Return periodic Euclidean distance for every particle and saved time."""
		delta_x, delta_y = self.periodic_displacement_components
		return np.hypot(delta_x, delta_y)

	@property
	def rms_distance(self) -> np.ndarray:
		"""Return root-mean-square periodic particle distance at each saved time."""
		return np.asarray(
			np.sqrt(np.mean(self.particle_distances**2, axis=0)),
			dtype=float,
		)

	@property
	def max_distance(self) -> np.ndarray:
		"""Return maximum periodic particle distance at each saved time."""
		return np.asarray(np.max(self.particle_distances, axis=0), dtype=float)

	def print_summary(self) -> None:
		"""Print the paired method cost and aggregate trajectory separation."""
		print(f"base method: {self.base_method_name}")
		print(f"tangent-Taylor method: {self.tangent_method_name}")
		print(f"step count: {self.base_solution.n_steps}")
		print(f"base runtime: {self.base_runtime_seconds:.6f} s")
		print(f"tangent-Taylor runtime: {self.tangent_runtime_seconds:.6f} s")
		print(
			"runtime ratio tangent/base: "
			f"{self.tangent_runtime_seconds / self.base_runtime_seconds:.6f}"
		)
		print(f"maximum RMS periodic drift: {np.max(self.rms_distance):.8e}")
		print(f"final RMS periodic drift: {self.rms_distance[-1]:.8e}")
		print(f"maximum particle drift: {np.max(self.max_distance):.8e}")
		print(f"final maximum particle drift: {self.max_distance[-1]:.8e}")


def _run_comparison(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	config: ABBATangentTaylorComparisonConfig,
	use_abba4: bool,
) -> ABBATangentTaylorComparisonResult:
	"""Run one original/tangent pair on identical internal and saved grids."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError("`initial_configuration` must implement InitialConfiguration.")
	if not isinstance(config, ABBATangentTaylorComparisonConfig):
		raise TypeError("`config` must be an ABBATangentTaylorComparisonConfig.")
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.max_step,
		sample_count=config.sample_count,
	)
	base_method: NumericalMethod
	tangent_method: NumericalMethod
	if use_abba4:
		base_method = ABBA4Implicit1(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		)
		tangent_method = ABBA4Implicit1TangentTaylor(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		)
	else:
		base_method = ImplicitABBA1(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		)
		tangent_method = ImplicitABBA1TangentTaylor(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		)
	started = perf_counter()
	base_solution = simulate(problem, base_method, request)
	base_runtime = perf_counter() - started
	started = perf_counter()
	tangent_solution = simulate(problem, tangent_method, request)
	tangent_runtime = perf_counter() - started
	return ABBATangentTaylorComparisonResult(
		potential=potential,
		config=config,
		base_method_name=type(base_method).__name__,
		tangent_method_name=type(tangent_method).__name__,
		base_solution=base_solution,
		tangent_solution=tangent_solution,
		base_runtime_seconds=base_runtime,
		tangent_runtime_seconds=tangent_runtime,
	)


def run_implicit_abba1_tangent_taylor_comparison(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	config: ABBATangentTaylorComparisonConfig,
) -> ABBATangentTaylorComparisonResult:
	"""Compare `ImplicitABBA1TangentTaylor` with `ImplicitABBA1`."""
	return _run_comparison(
		potential,
		initial_configuration,
		config=config,
		use_abba4=False,
	)


def run_abba4_implicit1_tangent_taylor_comparison(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	config: ABBATangentTaylorComparisonConfig,
) -> ABBATangentTaylorComparisonResult:
	"""Compare `ABBA4Implicit1TangentTaylor` with `ABBA4Implicit1`."""
	return _run_comparison(
		potential,
		initial_configuration,
		config=config,
		use_abba4=True,
	)


__all__ = [
	"ABBATangentTaylorComparisonConfig",
	"ABBATangentTaylorComparisonResult",
	"run_abba4_implicit1_tangent_taylor_comparison",
	"run_implicit_abba1_tangent_taylor_comparison",
]
