"""Aligned trajectory and nonlinear-work comparison for four implicit methods."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dynamics import GuidingCenterDynamics
from potential import Potential
from simulation import (
	BM4Implicit1,
	BM4Implicit2,
	ImplicitABBA1,
	ImplicitABBA2,
	InitialConfiguration,
	InitialValueProblem,
	NonlinearSolver,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)


IMPLICIT_METHOD_NAMES: tuple[str, ...] = (
	"ImplicitABBA1",
	"ImplicitABBA2",
	"BM4Implicit1",
	"BM4Implicit2",
)


@dataclass(frozen=True, slots=True)
class ImplicitTrajectoryComparisonConfig:
	"""Shared physical grid and nonlinear controls for all four methods."""

	rho: float = 0.3
	coupling_frequency: float = float(np.pi / 8.0)
	t_span: tuple[float, float] = (0.0, 2.0)
	integration_step: float = 0.05
	nonlinear_solver: NonlinearSolver = "broyden"
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	progress: bool = False

	def __post_init__(self) -> None:
		"""Normalize every parameter that affects numerical reproducibility."""
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		for name in (
			"integration_step",
			"absolute_tolerance",
			"relative_tolerance",
			"newton_jacobian_relative_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(
			self,
			"max_iterations",
			positive_integer(self.max_iterations, "max_iterations"),
		)
		if self.nonlinear_solver not in ("newton", "broyden"):
			raise ValueError("Unknown nonlinear solver for the implicit comparison.")
		integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.integration_step,
			"duration / integration_step",
		)
		object.__setattr__(self, "progress", bool(self.progress))

	@property
	def step_count(self) -> int:
		"""Return the common number of complete integration steps."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.integration_step,
			"duration / integration_step",
		)

	@property
	def output_sample_count(self) -> int:
		"""Return one saved state for every common grid node."""
		return self.step_count + 1


@dataclass(frozen=True, slots=True)
class ImplicitTrajectoryDifferenceSummary:
	"""Aggregate periodic particle distances for one pair of methods."""

	first_method: str
	second_method: str
	global_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float


@dataclass(frozen=True, slots=True)
class ImplicitIterationSummary:
	"""Per-method nonlinear work over the common sequence of accepted steps."""

	method_name: str
	nonlinear_solver: str
	step_count: int
	minimum_iterations: int
	mean_iterations: float
	maximum_iterations: int
	total_iterations: int
	mean_residual_evaluations: float
	total_residual_evaluations: int
	maximum_residual_to_tolerance: float
	runtime_seconds: float


def _minimum_image_displacement(
	displacement: np.ndarray,
	period: float,
) -> np.ndarray:
	"""Map a coordinate difference to its nearest periodic representative."""
	return (np.asarray(displacement, dtype=float) + period / 2.0) % period - period / 2.0


@dataclass(frozen=True, slots=True)
class ImplicitTrajectoryComparisonResult:
	"""Four aligned solutions with trajectory and nonlinear-work summaries."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: InitialConfiguration
	config: ImplicitTrajectoryComparisonConfig
	solutions: Mapping[str, Solution]
	runtimes: Mapping[str, float]

	def __post_init__(self) -> None:
		"""Require complete method coverage and exactly aligned saved grids."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, InitialConfiguration):
			raise TypeError(
				"`initial_configuration` must implement InitialConfiguration."
			)
		if not isinstance(self.config, ImplicitTrajectoryComparisonConfig):
			raise TypeError("`config` must be an ImplicitTrajectoryComparisonConfig.")
		if tuple(self.solutions) != IMPLICIT_METHOD_NAMES:
			raise ValueError("The comparison must contain all four implicit methods.")
		if tuple(self.runtimes) != IMPLICIT_METHOD_NAMES:
			raise ValueError("The comparison must contain one runtime per method.")

		reference_times: np.ndarray | None = None
		for method_name in IMPLICIT_METHOD_NAMES:
			solution = self.solutions[method_name]
			if not isinstance(solution, Solution):
				raise TypeError("Every implicit comparison result must be a Solution.")
			if solution.source is not self.initial_configuration:
				raise ValueError("All solutions must share the initial configuration.")
			if int(solution.diagnostics.get("step_count", -1)) != self.config.step_count:
				raise ValueError("Every method must use the configured common step.")
			if solution.diagnostics.get("nonlinear_solver") != self.config.nonlinear_solver:
				raise ValueError("Every method must use the configured nonlinear solver.")
			candidate_times = np.asarray(solution.t, dtype=float)
			if reference_times is None:
				reference_times = candidate_times
			elif not np.array_equal(candidate_times, reference_times):
				raise ValueError("All implicit solutions must share the saved-time grid.")
			seconds = float(self.runtimes[method_name])
			if not np.isfinite(seconds) or seconds <= 0.0:
				raise ValueError("Every method runtime must be positive and finite.")

		object.__setattr__(self, "solutions", MappingProxyType(dict(self.solutions)))
		object.__setattr__(self, "runtimes", MappingProxyType(dict(self.runtimes)))

	@property
	def effective_potential(self) -> Potential:
		"""Return the gyroaveraged potential used by every trajectory."""
		return self.dynamics.effective_potential

	def trajectory_difference_summaries(
		self,
	) -> tuple[ImplicitTrajectoryDifferenceSummary, ...]:
		"""Compare every pair on the periodic cell over particles and time."""
		rows: list[ImplicitTrajectoryDifferenceSummary] = []
		period = self.potential.grid.period
		for first_method, second_method in combinations(IMPLICIT_METHOD_NAMES, 2):
			first_x, first_y = self.solutions[first_method].positions()
			second_x, second_y = self.solutions[second_method].positions()
			delta_x = _minimum_image_displacement(first_x - second_x, period)
			delta_y = _minimum_image_displacement(first_y - second_y, period)
			distances = np.hypot(delta_x, delta_y)
			rows.append(
				ImplicitTrajectoryDifferenceSummary(
					first_method=first_method,
					second_method=second_method,
					global_rms_distance=float(np.sqrt(np.mean(distances**2))),
					maximum_distance=float(np.max(distances)),
					final_rms_distance=float(
						np.sqrt(np.mean(distances[:, -1] ** 2))
					),
					final_maximum_distance=float(np.max(distances[:, -1])),
				)
			)
		return tuple(rows)

	def iteration_summaries(self) -> tuple[ImplicitIterationSummary, ...]:
		"""Summarize corrections, residual evaluations, convergence, and runtime."""
		rows: list[ImplicitIterationSummary] = []
		for method_name in IMPLICIT_METHOD_NAMES:
			solution = self.solutions[method_name]
			iterations = np.asarray(
				solution.diagnostics["nonlinear_iterations"], dtype=int
			)
			residual_evaluations = np.asarray(
				solution.diagnostics["residual_evaluations"], dtype=int
			)
			residuals = np.asarray(
				solution.diagnostics["nonlinear_residual_norms"], dtype=float
			)
			tolerances = np.asarray(
				solution.diagnostics["nonlinear_tolerances"], dtype=float
			)
			expected_shape = (self.config.step_count,)
			if any(
				value.shape != expected_shape
				for value in (
					iterations,
					residual_evaluations,
					residuals,
					tolerances,
				)
			):
				raise ValueError("Implicit per-step diagnostics are not aligned.")
			rows.append(
				ImplicitIterationSummary(
					method_name=method_name,
					nonlinear_solver=self.config.nonlinear_solver,
					step_count=self.config.step_count,
					minimum_iterations=int(np.min(iterations)),
					mean_iterations=float(np.mean(iterations)),
					maximum_iterations=int(np.max(iterations)),
					total_iterations=int(np.sum(iterations)),
					mean_residual_evaluations=float(
						np.mean(residual_evaluations)
					),
					total_residual_evaluations=int(np.sum(residual_evaluations)),
					maximum_residual_to_tolerance=float(
						np.max(residuals / tolerances)
					),
					runtime_seconds=float(self.runtimes[method_name]),
				)
			)
		return tuple(rows)


def run_implicit_trajectory_comparison(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	config: ImplicitTrajectoryComparisonConfig,
) -> ImplicitTrajectoryComparisonResult:
	"""Run all four methods once with a common problem, grid, and solver controls."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError(
			"`initial_configuration` must implement InitialConfiguration."
		)
	if not isinstance(config, ImplicitTrajectoryComparisonConfig):
		raise TypeError("`config` must be an ImplicitTrajectoryComparisonConfig.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	methods = (
		ImplicitABBA1(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		),
		ImplicitABBA2(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		),
		BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		),
		BM4Implicit2(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			nonlinear_solver=config.nonlinear_solver,
			progress=config.progress,
		),
	)
	solutions: dict[str, Solution] = {}
	runtimes: dict[str, float] = {}
	for method in methods:
		method_name = type(method).__name__
		started = perf_counter()
		solutions[method_name] = simulate(problem, method, request)
		runtimes[method_name] = perf_counter() - started

	return ImplicitTrajectoryComparisonResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		config=config,
		solutions=solutions,
		runtimes=runtimes,
	)


__all__ = [
	"IMPLICIT_METHOD_NAMES",
	"ImplicitIterationSummary",
	"ImplicitTrajectoryComparisonConfig",
	"ImplicitTrajectoryComparisonResult",
	"ImplicitTrajectoryDifferenceSummary",
	"run_implicit_trajectory_comparison",
]
