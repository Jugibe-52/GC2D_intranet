"""Reference accuracy and Newton-work refinement for four implicit GC methods."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA4Implicit1,
	ABBA4SingleProjectionImplicit1,
	BM4Implicit1,
	ImplicitABBA1,
	InitialValueProblem,
	NumericalMethod,
	SimulationRequest,
	Solution,
	simulate,
)

from ._trajectory_accuracy import (
	TrajectoryAccuracySeries,
	accuracy_series,
	reference_distance_convention,
	reference_indices_for_times,
	validate_reference_identity,
	validated_refinement_steps,
)
from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)


IMPLICIT_ACCURACY_METHOD_NAMES: tuple[str, ...] = (
	"ImplicitABBA1",
	"ABBA4Implicit1",
	"ABBA4SingleProjectionImplicit1",
	"BM4Implicit1",
)
IMPLICIT_ACCURACY_METHOD_LABELS: Mapping[str, str] = MappingProxyType(
	{
		"ImplicitABBA1": "Implicit ABBA",
		"ABBA4Implicit1": "Implicit ABBA4 (three projections)",
		"ABBA4SingleProjectionImplicit1": "Implicit ABBA4 (single projection)",
		"BM4Implicit1": "Implicit BM4",
	}
)
IMPLICIT_ACCURACY_DESIGNED_ORDERS: Mapping[str, float] = MappingProxyType(
	{
		"ImplicitABBA1": 2.0,
		"ABBA4Implicit1": 4.0,
		"ABBA4SingleProjectionImplicit1": 4.0,
		"BM4Implicit1": 4.0,
	}
)


@dataclass(frozen=True, slots=True)
class ImplicitMethodAccuracyConfig:
	"""Common refinement grid, physical controls, and Newton tolerances."""

	integration_steps: tuple[float, ...] = (
		0.0025,
		0.00125,
		0.000625,
		0.0003125,
		0.00015625,
	)
	t_span: tuple[float, float] = (0.0, 4.0)
	save_interval: float = 0.01
	rho: float = 0.0
	coupling_frequency: float = float(np.pi / 8.0)
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	progress: bool = False

	def __post_init__(self) -> None:
		"""Require nested complete-step grids aligned with every saved time."""
		steps = validated_refinement_steps(self.integration_steps)
		object.__setattr__(self, "integration_steps", steps)
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(
			self,
			"save_interval",
			positive_finite(self.save_interval, "save_interval"),
		)
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		for name in (
			"absolute_tolerance",
			"relative_tolerance",
			"newton_jacobian_relative_step",
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		object.__setattr__(
			self,
			"max_iterations",
			positive_integer(self.max_iterations, "max_iterations"),
		)
		integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		)
		for step in steps:
			integer_ratio(
				self.t_span[1] - self.t_span[0],
				step,
				f"duration / integration step {step:g}",
			)
			integer_ratio(
				self.save_interval,
				step,
				f"save_interval / integration step {step:g}",
			)
		object.__setattr__(self, "progress", bool(self.progress))

	@property
	def output_sample_count(self) -> int:
		"""Return the shared saved-state count including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1

	def step_count(self, integration_step: float) -> int:
		"""Return the number of complete steps for one configured refinement."""
		step = positive_finite(integration_step, "integration_step")
		if step not in self.integration_steps:
			raise ValueError("`integration_step` must belong to the refinement grid.")
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			step,
			"duration / integration_step",
		)


@dataclass(frozen=True, slots=True)
class ImplicitMethodAccuracySummary:
	"""Accuracy, Newton convergence, work, and runtime at one step size."""

	method_name: str
	method_label: str
	designed_order: float
	integration_step: float
	step_count: int
	global_rms_distance: float
	time_integrated_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float
	reference_floor_ratio: float
	nonlinear_solves_per_step: int
	minimum_iterations_per_step: int
	mean_iterations_per_step: float
	maximum_iterations_per_step: int
	mean_iterations_per_solve: float
	percentile_95_iterations_per_solve: float
	maximum_iterations_per_solve: int
	zero_iteration_fraction: float
	one_iteration_fraction: float
	two_iteration_fraction: float
	three_or_more_iteration_fraction: float
	total_iterations: int
	mean_residual_evaluations_per_solve: float
	total_residual_evaluations: int
	mean_residual_to_tolerance: float
	maximum_residual_to_tolerance: float
	runtime_seconds: float


@dataclass(frozen=True, slots=True)
class ImplicitMethodAccuracyOrder:
	"""Observed accuracy order and Newton-work change for one refinement pair."""

	method_name: str
	method_label: str
	designed_order: float
	coarse_step: float
	fine_step: float
	time_integrated_rms_gain: float
	final_rms_gain: float
	time_integrated_rms_order: float
	final_rms_order: float
	time_integrated_order_deficit: float
	final_order_deficit: float
	time_integrated_resolved_above_reference_floor: bool
	final_resolved_above_reference_floor: bool
	accuracy_improved: bool
	mean_iterations_per_solve_change: float
	maximum_iterations_per_solve_change: int


def _frozen_nested_mapping(
	values: Mapping[str, Mapping[float, Any]],
) -> Mapping[str, Mapping[float, Any]]:
	"""Copy and freeze one method-by-step result mapping."""
	return MappingProxyType(
		{
			method_name: MappingProxyType(dict(step_values))
			for method_name, step_values in values.items()
		}
	)


def _configured_method(
	method_name: str,
	config: ImplicitMethodAccuracyConfig,
) -> NumericalMethod:
	"""Construct one Newton-solved method with the common tolerances."""
	if method_name == "ImplicitABBA1":
		return ImplicitABBA1(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	if method_name == "ABBA4Implicit1":
		return ABBA4Implicit1(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	if method_name == "ABBA4SingleProjectionImplicit1":
		return ABBA4SingleProjectionImplicit1(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	if method_name == "BM4Implicit1":
		return BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	assert False, f"Unknown implicit accuracy method {method_name!r}."


@dataclass(frozen=True, slots=True)
class ImplicitMethodAccuracyResult:
	"""Four aligned Newton refinements against one certified reference."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	reference: StoredReferenceTrajectory
	config: ImplicitMethodAccuracyConfig
	reference_sample_indices: np.ndarray
	solutions: Mapping[str, Mapping[float, Solution]]
	series: Mapping[str, Mapping[float, TrajectoryAccuracySeries]]
	runtimes: Mapping[str, Mapping[float, float]]

	def __post_init__(self) -> None:
		"""Freeze results and enforce exact method, step, and time alignment."""
		for values in (self.solutions, self.series, self.runtimes):
			if tuple(values) != IMPLICIT_ACCURACY_METHOD_NAMES:
				raise ValueError("Results must follow the implicit method order.")
		indices = np.array(self.reference_sample_indices, dtype=np.int64, copy=True)
		initial_state = self.initial_configuration.initial_state
		if initial_state is None:
			raise ValueError("The initial configuration must contain an initial state.")
		particle_count = self.initial_configuration.particle_count(initial_state)
		common_times: np.ndarray | None = None
		for method_name in IMPLICIT_ACCURACY_METHOD_NAMES:
			if any(
				tuple(values[method_name]) != self.config.integration_steps
				for values in (self.solutions, self.series, self.runtimes)
			):
				raise ValueError("Every method must follow the configured step order.")
			for step in self.config.integration_steps:
				solution = self.solutions[method_name][step]
				if not isinstance(solution, Solution):
					raise TypeError("Every implicit accuracy trajectory must be a Solution.")
				if solution.source is not self.initial_configuration:
					raise ValueError("Every solution must share one initial configuration.")
				if int(solution.diagnostics.get("step_count", -1)) != (
					self.config.step_count(step)
				):
					raise ValueError("A solution has an inconsistent complete-step count.")
				if solution.diagnostics.get("nonlinear_solver") != "newton":
					raise ValueError("Every implicit accuracy solve must use Newton.")
				if common_times is None:
					common_times = solution.t
				elif not np.array_equal(solution.t, common_times):
					raise ValueError("Every refinement must share one saved-time grid.")
				accuracy = self.series[method_name][step]
				if accuracy.method_name != method_name or accuracy.distances.shape != (
					particle_count,
					solution.t.size,
				):
					raise ValueError("An accuracy series is inconsistent with its solution.")
				runtime = float(self.runtimes[method_name][step])
				if not np.isfinite(runtime) or runtime <= 0.0:
					raise ValueError("Every runtime must be positive and finite.")
		assert common_times is not None
		if (
			indices.shape != common_times.shape
			or np.any(indices < 0)
			or np.any(indices >= self.reference.times.size)
			or not np.array_equal(self.reference.times[indices], common_times)
		):
			raise ValueError("Comparison samples do not align with the reference.")
		indices.setflags(write=False)
		object.__setattr__(self, "reference_sample_indices", indices)
		object.__setattr__(self, "solutions", _frozen_nested_mapping(self.solutions))
		object.__setattr__(self, "series", _frozen_nested_mapping(self.series))
		object.__setattr__(self, "runtimes", _frozen_nested_mapping(self.runtimes))

	@property
	def times(self) -> np.ndarray:
		"""Return the shared saved-time grid."""
		return self.solutions[IMPLICIT_ACCURACY_METHOD_NAMES[0]][
			self.config.integration_steps[0]
		].t

	@property
	def reference_floor(self) -> float:
		"""Return the time-integrated particle-RMS DOP853/Radau discrepancy."""
		distances = self.reference.audit_distances[:, self.reference_sample_indices]
		return float(
			np.sqrt(
				np.trapz(np.mean(distances**2, axis=0), self.times)
				/ float(self.times[-1] - self.times[0])
			)
		)

	@property
	def final_reference_floor(self) -> float:
		"""Return the final-time particle-RMS DOP853/Radau discrepancy."""
		values = self.reference.audit_distances[:, self.reference_sample_indices[-1]]
		return float(np.sqrt(np.mean(values**2)))

	@property
	def finest_series(self) -> Mapping[str, TrajectoryAccuracySeries]:
		"""Return every method's errors at the smallest configured step."""
		step = self.config.integration_steps[-1]
		return MappingProxyType(
			{method_name: self.series[method_name][step] for method_name in IMPLICIT_ACCURACY_METHOD_NAMES}
		)

	def iteration_frequencies(self, method_name: str, integration_step: float) -> Mapping[int, int]:
		"""Count Newton corrections over every nonlinear solve in one run."""
		if method_name not in IMPLICIT_ACCURACY_METHOD_NAMES:
			raise ValueError("Unknown implicit accuracy method.")
		if integration_step not in self.config.integration_steps:
			raise ValueError("Unknown implicit accuracy integration step.")
		diagnostics = self.solutions[method_name][integration_step].diagnostics
		step_iterations = np.asarray(diagnostics["nonlinear_iterations"], dtype=int)
		iterations = np.asarray(
			diagnostics.get(
				"substep_nonlinear_iterations",
				step_iterations[:, np.newaxis],
			),
			dtype=int,
		).ravel()
		values, counts = np.unique(iterations, return_counts=True)
		return MappingProxyType(
			{int(value): int(count) for value, count in zip(values, counts)}
		)

	def summaries(self) -> tuple[ImplicitMethodAccuracySummary, ...]:
		"""Return accuracy and Newton convergence metrics for every complete run."""
		rows: list[ImplicitMethodAccuracySummary] = []
		duration = float(self.times[-1] - self.times[0])
		floor = max(self.reference_floor, float(np.finfo(float).eps))
		for step in self.config.integration_steps:
			for method_name in IMPLICIT_ACCURACY_METHOD_NAMES:
				solution = self.solutions[method_name][step]
				accuracy = self.series[method_name][step]
				diagnostics = solution.diagnostics
				step_iterations = np.asarray(
					diagnostics["nonlinear_iterations"], dtype=int
				)
				step_residual_evaluations = np.asarray(
					diagnostics["residual_evaluations"], dtype=int
				)
				step_residuals = np.asarray(
					diagnostics["nonlinear_residual_norms"], dtype=float
				)
				step_tolerances = np.asarray(
					diagnostics["nonlinear_tolerances"], dtype=float
				)
				expected_shape = (self.config.step_count(step),)
				if any(
					value.shape != expected_shape
					for value in (
						step_iterations,
						step_residual_evaluations,
						step_residuals,
						step_tolerances,
					)
				):
					raise ValueError("Per-step Newton diagnostics are not aligned.")
				iterations = np.asarray(
					diagnostics.get(
						"substep_nonlinear_iterations",
						step_iterations[:, np.newaxis],
					),
					dtype=int,
				)
				residual_evaluations = np.asarray(
					diagnostics.get(
						"substep_residual_evaluations",
						step_residual_evaluations[:, np.newaxis],
					),
					dtype=int,
				)
				residuals = np.asarray(
					diagnostics.get(
						"substep_nonlinear_residual_norms",
						step_residuals[:, np.newaxis],
					),
					dtype=float,
				)
				tolerances = np.asarray(
					diagnostics.get(
						"substep_nonlinear_tolerances",
						step_tolerances[:, np.newaxis],
					),
					dtype=float,
				)
				if (
					iterations.ndim != 2
					or iterations.shape[0] != expected_shape[0]
					or residual_evaluations.shape != iterations.shape
					or residuals.shape != iterations.shape
					or tolerances.shape != iterations.shape
					or np.any(iterations < 0)
					or np.any(residual_evaluations < iterations + 1)
					or np.any(tolerances <= 0.0)
				):
					raise ValueError("Per-solve Newton diagnostics are inconsistent.")
				nonlinear_solves = int(
					diagnostics.get("nonlinear_solves_per_step", iterations.shape[1])
				)
				if nonlinear_solves != iterations.shape[1]:
					raise ValueError("The nonlinear-solve count is inconsistent.")
				flat_iterations = iterations.ravel()
				residual_ratios = residuals / tolerances
				time_rms = float(
					np.sqrt(
						np.trapz(accuracy.rms_distance**2, self.times) / duration
					)
				)
				rows.append(
					ImplicitMethodAccuracySummary(
						method_name=method_name,
						method_label=IMPLICIT_ACCURACY_METHOD_LABELS[method_name],
						designed_order=IMPLICIT_ACCURACY_DESIGNED_ORDERS[method_name],
						integration_step=step,
						step_count=expected_shape[0],
						global_rms_distance=float(
							np.sqrt(np.mean(accuracy.distances**2))
						),
						time_integrated_rms_distance=time_rms,
						maximum_distance=float(np.max(accuracy.distances)),
						final_rms_distance=float(accuracy.rms_distance[-1]),
						final_maximum_distance=float(accuracy.maximum_distance[-1]),
						reference_floor_ratio=time_rms / floor,
						nonlinear_solves_per_step=nonlinear_solves,
						minimum_iterations_per_step=int(np.min(step_iterations)),
						mean_iterations_per_step=float(np.mean(step_iterations)),
						maximum_iterations_per_step=int(np.max(step_iterations)),
						mean_iterations_per_solve=float(np.mean(flat_iterations)),
						percentile_95_iterations_per_solve=float(
							np.quantile(flat_iterations, 0.95)
						),
						maximum_iterations_per_solve=int(np.max(flat_iterations)),
						zero_iteration_fraction=float(np.mean(flat_iterations == 0)),
						one_iteration_fraction=float(np.mean(flat_iterations == 1)),
						two_iteration_fraction=float(np.mean(flat_iterations == 2)),
						three_or_more_iteration_fraction=float(
							np.mean(flat_iterations >= 3)
						),
						total_iterations=int(np.sum(step_iterations)),
						mean_residual_evaluations_per_solve=float(
							np.mean(residual_evaluations)
						),
						total_residual_evaluations=int(
							np.sum(step_residual_evaluations)
						),
						mean_residual_to_tolerance=float(np.mean(residual_ratios)),
						maximum_residual_to_tolerance=float(np.max(residual_ratios)),
						runtime_seconds=float(self.runtimes[method_name][step]),
					)
				)
		return tuple(rows)

	def convergence_orders(self) -> tuple[ImplicitMethodAccuracyOrder, ...]:
		"""Return adjacent observed orders and changes in Newton workload."""
		summaries = {
			(row.method_name, row.integration_step): row for row in self.summaries()
		}
		time_floor = max(self.reference_floor, float(np.finfo(float).eps))
		final_floor = max(self.final_reference_floor, float(np.finfo(float).eps))
		rows: list[ImplicitMethodAccuracyOrder] = []
		for coarse_step, fine_step in zip(
			self.config.integration_steps,
			self.config.integration_steps[1:],
		):
			step_ratio = coarse_step / fine_step
			for method_name in IMPLICIT_ACCURACY_METHOD_NAMES:
				coarse = summaries[(method_name, coarse_step)]
				fine = summaries[(method_name, fine_step)]
				time_gain = coarse.time_integrated_rms_distance / fine.time_integrated_rms_distance
				final_gain = coarse.final_rms_distance / fine.final_rms_distance
				time_resolved = (
					fine.time_integrated_rms_distance > 10.0 * time_floor
					and time_gain > 0.0
				)
				final_resolved = (
					fine.final_rms_distance > 10.0 * final_floor
					and final_gain > 0.0
				)
				time_order = (
					float(np.log(time_gain) / np.log(step_ratio))
					if time_resolved
					else float("nan")
				)
				final_order = (
					float(np.log(final_gain) / np.log(step_ratio))
					if final_resolved
					else float("nan")
				)
				designed_order = IMPLICIT_ACCURACY_DESIGNED_ORDERS[method_name]
				rows.append(
					ImplicitMethodAccuracyOrder(
						method_name=method_name,
						method_label=IMPLICIT_ACCURACY_METHOD_LABELS[method_name],
						designed_order=designed_order,
						coarse_step=coarse_step,
						fine_step=fine_step,
						time_integrated_rms_gain=float(time_gain),
						final_rms_gain=float(final_gain),
						time_integrated_rms_order=time_order,
						final_rms_order=final_order,
						time_integrated_order_deficit=(
							designed_order - time_order
							if np.isfinite(time_order)
							else float("nan")
						),
						final_order_deficit=(
							designed_order - final_order
							if np.isfinite(final_order)
							else float("nan")
						),
						time_integrated_resolved_above_reference_floor=time_resolved,
						final_resolved_above_reference_floor=final_resolved,
						accuracy_improved=time_gain > 1.0 and final_gain > 1.0,
						mean_iterations_per_solve_change=(
							fine.mean_iterations_per_solve
							- coarse.mean_iterations_per_solve
						),
						maximum_iterations_per_solve_change=(
							fine.maximum_iterations_per_solve
							- coarse.maximum_iterations_per_solve
						),
					)
				)
		return tuple(rows)


def run_implicit_method_accuracy_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: ImplicitMethodAccuracyConfig,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> ImplicitMethodAccuracyResult:
	"""Run four Newton methods on nested steps and compare with one reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if not isinstance(config, ImplicitMethodAccuracyConfig):
		raise TypeError("`config` must be ImplicitMethodAccuracyConfig.")
	validate_reference_identity(
		potential,
		initial_configuration,
		reference,
		config,
		potential_metadata=potential_metadata,
		initial_condition_metadata=initial_condition_metadata,
	)
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	distance_convention = reference_distance_convention(reference)
	solutions: dict[str, dict[float, Solution]] = {
		method_name: {} for method_name in IMPLICIT_ACCURACY_METHOD_NAMES
	}
	series: dict[str, dict[float, TrajectoryAccuracySeries]] = {
		method_name: {} for method_name in IMPLICIT_ACCURACY_METHOD_NAMES
	}
	runtimes: dict[str, dict[float, float]] = {
		method_name: {} for method_name in IMPLICIT_ACCURACY_METHOD_NAMES
	}
	reference_indices: np.ndarray | None = None
	for step in config.integration_steps:
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		for method_name in IMPLICIT_ACCURACY_METHOD_NAMES:
			started = perf_counter()
			solution = simulate(
				problem,
				_configured_method(method_name, config),
				request,
			)
			runtimes[method_name][step] = perf_counter() - started
			solutions[method_name][step] = solution
			indices = reference_indices_for_times(reference, solution.t)
			if reference_indices is None:
				reference_indices = indices
			elif not np.array_equal(indices, reference_indices):
				raise ValueError("Implicit refinements do not share reference samples.")
			series[method_name][step] = accuracy_series(
				method_name,
				solution.states,
				reference.states[:, indices],
				period=float(potential.grid.period),
				distance_convention=distance_convention,
			)
	assert reference_indices is not None
	return ImplicitMethodAccuracyResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		reference=reference,
		config=config,
		reference_sample_indices=reference_indices,
		solutions=solutions,
		series=series,
		runtimes=runtimes,
	)


__all__ = [
	"IMPLICIT_ACCURACY_DESIGNED_ORDERS",
	"IMPLICIT_ACCURACY_METHOD_LABELS",
	"IMPLICIT_ACCURACY_METHOD_NAMES",
	"ImplicitMethodAccuracyConfig",
	"ImplicitMethodAccuracyOrder",
	"ImplicitMethodAccuracyResult",
	"ImplicitMethodAccuracySummary",
	"run_implicit_method_accuracy_study",
]
