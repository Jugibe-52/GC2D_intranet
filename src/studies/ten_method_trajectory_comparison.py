"""Aligned trajectory comparison for two midpoint and eight implicit variants."""

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
	MidpointABBA,
	MidpointBM4,
	NonlinearSolver,
	NumericalMethod,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import integer_ratio, nonnegative_finite, positive_finite, positive_integer


@dataclass(frozen=True, slots=True)
class TrajectoryMethodVariant:
	"""Stable label and solver identity for one compared numerical variant."""

	label: str
	method_name: str
	family: str
	nonlinear_solver: NonlinearSolver | None


TEN_METHOD_VARIANTS: tuple[TrajectoryMethodVariant, ...] = (
	TrajectoryMethodVariant("Midpoint ABBA", "MidpointABBA", "midpoint", None),
	TrajectoryMethodVariant("Midpoint BM4", "MidpointBM4", "midpoint", None),
	TrajectoryMethodVariant("Implicit ABBA 1 (Newton)", "ImplicitABBA1", "abba", "newton"),
	TrajectoryMethodVariant("Implicit ABBA 1 (Broyden)", "ImplicitABBA1", "abba", "broyden"),
	TrajectoryMethodVariant("Implicit ABBA 2 (Newton)", "ImplicitABBA2", "abba", "newton"),
	TrajectoryMethodVariant("Implicit ABBA 2 (Broyden)", "ImplicitABBA2", "abba", "broyden"),
	TrajectoryMethodVariant("BM4 implicit 1 (Newton)", "BM4Implicit1", "bm4", "newton"),
	TrajectoryMethodVariant("BM4 implicit 1 (Broyden)", "BM4Implicit1", "bm4", "broyden"),
	TrajectoryMethodVariant("BM4 implicit 2 (Newton)", "BM4Implicit2", "bm4", "newton"),
	TrajectoryMethodVariant("BM4 implicit 2 (Broyden)", "BM4Implicit2", "bm4", "broyden"),
)
TEN_METHOD_LABELS: tuple[str, ...] = tuple(
	variant.label for variant in TEN_METHOD_VARIANTS
)


@dataclass(frozen=True, slots=True)
class TenMethodTrajectoryComparisonConfig:
	"""Common physical grid and nonlinear controls for all ten variants."""

	rho: float = 0.3
	coupling_frequency: float = float(np.pi / 8.0)
	t_span: tuple[float, float] = (0.0, 2.0)
	integration_step: float = 0.05
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
		"""Return one saved state for every common integration-grid node."""
		return self.step_count + 1


@dataclass(frozen=True, slots=True)
class TenMethodTrajectoryDifferenceSummary:
	"""Aggregate periodic particle distances for one pair of variants."""

	first_method: str
	second_method: str
	global_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float


@dataclass(frozen=True, slots=True)
class TenMethodRuntimeSummary:
	"""Wall-clock integration time for one numerical variant."""

	method_name: str
	family: str
	nonlinear_solver: str
	step_count: int
	runtime_seconds: float


@dataclass(frozen=True, slots=True)
class TenMethodNonlinearWorkSummary:
	"""Nonlinear work for one of the eight implicit variants."""

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


def _method_for_variant(
	variant: TrajectoryMethodVariant,
	config: TenMethodTrajectoryComparisonConfig,
) -> NumericalMethod:
	"""Build one configured method while keeping labels separate from class names."""
	if variant.method_name == "MidpointABBA":
		return MidpointABBA(progress=config.progress)
	if variant.method_name == "MidpointBM4":
		return MidpointBM4(progress=config.progress)
	if variant.nonlinear_solver is None:
		raise ValueError("Implicit variants require a nonlinear solver.")
	if variant.method_name == "ImplicitABBA1":
		return ImplicitABBA1(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver=variant.nonlinear_solver,
			progress=config.progress,
		)
	if variant.method_name == "ImplicitABBA2":
		return ImplicitABBA2(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver=variant.nonlinear_solver,
			progress=config.progress,
		)
	if variant.method_name == "BM4Implicit1":
		return BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			nonlinear_solver=variant.nonlinear_solver,
			progress=config.progress,
		)
	if variant.method_name == "BM4Implicit2":
		return BM4Implicit2(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			nonlinear_solver=variant.nonlinear_solver,
			progress=config.progress,
		)
	raise ValueError(f"Unknown trajectory method variant {variant.label!r}.")


@dataclass(frozen=True, slots=True)
class TenMethodTrajectoryComparisonResult:
	"""Ten aligned solutions with trajectory, runtime, and solver summaries."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: InitialConfiguration
	config: TenMethodTrajectoryComparisonConfig
	solutions: Mapping[str, Solution]
	runtimes: Mapping[str, float]

	def __post_init__(self) -> None:
		"""Require complete variant coverage and one exactly aligned output grid."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, InitialConfiguration):
			raise TypeError("`initial_configuration` must implement InitialConfiguration.")
		if not isinstance(self.config, TenMethodTrajectoryComparisonConfig):
			raise TypeError("`config` must be a TenMethodTrajectoryComparisonConfig.")
		if tuple(self.solutions) != TEN_METHOD_LABELS:
			raise ValueError("The comparison must contain all ten method variants.")
		if tuple(self.runtimes) != TEN_METHOD_LABELS:
			raise ValueError("The comparison must contain one runtime per variant.")

		reference_times: np.ndarray | None = None
		for variant in TEN_METHOD_VARIANTS:
			solution = self.solutions[variant.label]
			if not isinstance(solution, Solution):
				raise TypeError("Every trajectory comparison value must be a Solution.")
			if solution.source is not self.initial_configuration:
				raise ValueError("All solutions must share the initial configuration.")
			if int(solution.diagnostics.get("step_count", -1)) != self.config.step_count:
				raise ValueError("Every variant must use the configured common step.")
			if variant.nonlinear_solver is not None and solution.diagnostics.get(
				"nonlinear_solver"
			) != variant.nonlinear_solver:
				raise ValueError("An implicit solution used the wrong nonlinear solver.")
			candidate_times = np.asarray(solution.t, dtype=float)
			if reference_times is None:
				reference_times = candidate_times
			elif not np.array_equal(candidate_times, reference_times):
				raise ValueError("All ten solutions must share the saved-time grid.")
			seconds = float(self.runtimes[variant.label])
			if not np.isfinite(seconds) or seconds <= 0.0:
				raise ValueError("Every variant runtime must be positive and finite.")

		object.__setattr__(self, "solutions", MappingProxyType(dict(self.solutions)))
		object.__setattr__(self, "runtimes", MappingProxyType(dict(self.runtimes)))

	@property
	def effective_potential(self) -> Potential:
		"""Return the common gyroaveraged potential."""
		return self.dynamics.effective_potential

	@property
	def implicit_solutions(self) -> Mapping[str, Solution]:
		"""Return the eight variants with nonlinear diagnostics."""
		return MappingProxyType(
			{
				variant.label: self.solutions[variant.label]
				for variant in TEN_METHOD_VARIANTS
				if variant.nonlinear_solver is not None
			}
		)

	def trajectory_difference_summaries(
		self,
	) -> tuple[TenMethodTrajectoryDifferenceSummary, ...]:
		"""Compare all 45 pairs over the periodic cell, particles, and time."""
		rows: list[TenMethodTrajectoryDifferenceSummary] = []
		period = self.potential.grid.period
		for first_method, second_method in combinations(TEN_METHOD_LABELS, 2):
			first_x, first_y = self.solutions[first_method].positions()
			second_x, second_y = self.solutions[second_method].positions()
			delta_x = _minimum_image_displacement(first_x - second_x, period)
			delta_y = _minimum_image_displacement(first_y - second_y, period)
			distances = np.hypot(delta_x, delta_y)
			rows.append(
				TenMethodTrajectoryDifferenceSummary(
					first_method=first_method,
					second_method=second_method,
					global_rms_distance=float(np.sqrt(np.mean(distances**2))),
					maximum_distance=float(np.max(distances)),
					final_rms_distance=float(np.sqrt(np.mean(distances[:, -1] ** 2))),
					final_maximum_distance=float(np.max(distances[:, -1])),
				)
			)
		return tuple(rows)

	def runtime_summaries(self) -> tuple[TenMethodRuntimeSummary, ...]:
		"""Return wall-clock timings in the stable comparison order."""
		return tuple(
			TenMethodRuntimeSummary(
				method_name=variant.label,
				family=variant.family,
				nonlinear_solver=variant.nonlinear_solver or "explicit midpoint",
				step_count=self.config.step_count,
				runtime_seconds=float(self.runtimes[variant.label]),
			)
			for variant in TEN_METHOD_VARIANTS
		)

	def nonlinear_work_summaries(self) -> tuple[TenMethodNonlinearWorkSummary, ...]:
		"""Summarize corrections and residual evaluations for implicit variants."""
		rows: list[TenMethodNonlinearWorkSummary] = []
		expected_shape = (self.config.step_count,)
		for variant in TEN_METHOD_VARIANTS:
			if variant.nonlinear_solver is None:
				continue
			solution = self.solutions[variant.label]
			iterations = np.asarray(solution.diagnostics["nonlinear_iterations"], dtype=int)
			residual_evaluations = np.asarray(
				solution.diagnostics["residual_evaluations"], dtype=int
			)
			residuals = np.asarray(
				solution.diagnostics["nonlinear_residual_norms"], dtype=float
			)
			tolerances = np.asarray(
				solution.diagnostics["nonlinear_tolerances"], dtype=float
			)
			if any(
				value.shape != expected_shape
				for value in (iterations, residual_evaluations, residuals, tolerances)
			):
				raise ValueError("Implicit per-step diagnostics are not aligned.")
			rows.append(
				TenMethodNonlinearWorkSummary(
					method_name=variant.label,
					nonlinear_solver=variant.nonlinear_solver,
					step_count=self.config.step_count,
					minimum_iterations=int(np.min(iterations)),
					mean_iterations=float(np.mean(iterations)),
					maximum_iterations=int(np.max(iterations)),
					total_iterations=int(np.sum(iterations)),
					mean_residual_evaluations=float(np.mean(residual_evaluations)),
					total_residual_evaluations=int(np.sum(residual_evaluations)),
					maximum_residual_to_tolerance=float(np.max(residuals / tolerances)),
					runtime_seconds=float(self.runtimes[variant.label]),
				)
			)
		return tuple(rows)


def run_ten_method_trajectory_comparison(
	potential: Potential,
	initial_configuration: InitialConfiguration,
	*,
	config: TenMethodTrajectoryComparisonConfig,
) -> TenMethodTrajectoryComparisonResult:
	"""Run all ten variants once on one common problem and saved-time grid."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, InitialConfiguration):
		raise TypeError("`initial_configuration` must implement InitialConfiguration.")
	if not isinstance(config, TenMethodTrajectoryComparisonConfig):
		raise TypeError("`config` must be a TenMethodTrajectoryComparisonConfig.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	solutions: dict[str, Solution] = {}
	runtimes: dict[str, float] = {}
	for variant in TEN_METHOD_VARIANTS:
		method = _method_for_variant(variant, config)
		started = perf_counter()
		solutions[variant.label] = simulate(problem, method, request)
		runtimes[variant.label] = perf_counter() - started

	return TenMethodTrajectoryComparisonResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		config=config,
		solutions=solutions,
		runtimes=runtimes,
	)


__all__ = [
	"TEN_METHOD_LABELS",
	"TEN_METHOD_VARIANTS",
	"TenMethodNonlinearWorkSummary",
	"TenMethodRuntimeSummary",
	"TenMethodTrajectoryComparisonConfig",
	"TenMethodTrajectoryComparisonResult",
	"TenMethodTrajectoryDifferenceSummary",
	"TrajectoryMethodVariant",
	"run_ten_method_trajectory_comparison",
]
