"""Accuracy, nonlinear-work, and timing comparison of two ABBA4 projections."""

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
	NONLINEAR_SOLVERS,
	InitialValueProblem,
	NonlinearSolver,
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


ABBA4_PROJECTION_METHOD_NAMES: tuple[str, ...] = (
	"ABBA4Implicit1",
	"ABBA4SingleProjectionImplicit1",
)
ABBA4_PROJECTION_METHOD_LABELS: Mapping[str, str] = MappingProxyType(
	{
		"ABBA4Implicit1": "ABBA4 (three projections)",
		"ABBA4SingleProjectionImplicit1": "SP-ABBA4 (single projection)",
	}
)


@dataclass(frozen=True, slots=True)
class ABBA4ProjectionComparisonConfig:
	"""Common refinement, nonlinear-solver, and timing controls."""

	integration_steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05)
	t_span: tuple[float, float] = (0.0, 2.0)
	save_interval: float = 0.4
	rho: float = 0.3
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	nonlinear_solver: NonlinearSolver = "newton"
	timing_warmups: int = 0
	timing_repeats: int = 1
	designed_order: float = 4.0
	order_reduction_threshold: float = 0.5
	progress: bool = False

	def __post_init__(self) -> None:
		"""Require nested fixed-step grids and reproducible timing controls."""
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
		for name in (
			"absolute_tolerance",
			"relative_tolerance",
			"designed_order",
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		object.__setattr__(
			self,
			"order_reduction_threshold",
			nonnegative_finite(
				self.order_reduction_threshold,
				"order_reduction_threshold",
			),
		)
		object.__setattr__(
			self,
			"max_iterations",
			positive_integer(self.max_iterations, "max_iterations"),
		)
		if self.nonlinear_solver not in NONLINEAR_SOLVERS:
			raise ValueError("`nonlinear_solver` must be 'newton' or 'broyden'.")
		if (
			isinstance(self.timing_warmups, (bool, np.bool_))
			or not isinstance(self.timing_warmups, (int, np.integer))
			or self.timing_warmups < 0
		):
			raise ValueError("`timing_warmups` must be a non-negative integer.")
		object.__setattr__(self, "timing_warmups", int(self.timing_warmups))
		object.__setattr__(
			self,
			"timing_repeats",
			positive_integer(self.timing_repeats, "timing_repeats"),
		)
		object.__setattr__(self, "progress", bool(self.progress))
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

	@property
	def output_sample_count(self) -> int:
		"""Return the common saved-state count including both endpoints."""
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
class ABBA4ProjectionComparisonSummary:
	"""Accuracy, nonlinear work, multiplier size, and timing at one step size."""

	method_name: str
	method_label: str
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
	maximum_iterations_per_solve: int
	total_iterations: int
	mean_residual_evaluations_per_step: float
	total_residual_evaluations: int
	mean_unprojected_abba_map_evaluations_per_step: float
	total_unprojected_abba_map_evaluations: int
	mean_newton_tangent_abba_map_evaluations_per_step: float
	total_newton_tangent_abba_map_evaluations: int
	mean_residual_to_tolerance: float
	maximum_residual_to_tolerance: float
	mean_projection_multiplier_norm: float
	maximum_projection_multiplier_norm: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	runtime_minimum_seconds: float
	runtime_maximum_seconds: float


@dataclass(frozen=True, slots=True)
class ABBA4ProjectionComparisonOrder:
	"""Observed accuracy and projection-multiplier orders under refinement."""

	method_name: str
	method_label: str
	coarse_step: float
	fine_step: float
	time_integrated_rms_gain: float
	final_rms_gain: float
	time_integrated_rms_order: float
	final_rms_order: float
	time_integrated_order_reduction: float
	final_order_reduction: float
	time_integrated_resolved_above_reference_floor: bool
	final_resolved_above_reference_floor: bool
	resolved_above_reference_floor: bool
	time_integrated_order_reduction_detected: bool
	final_order_reduction_detected: bool
	projection_multiplier_gain: float
	projection_multiplier_order: float


def _readonly_runtime_samples(value: np.ndarray) -> np.ndarray:
	"""Own and freeze one positive sequence of wall-clock measurements."""
	array = np.array(value, dtype=float, copy=True)
	if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
		raise ValueError("Runtime samples must be a non-empty finite vector.")
	if np.any(array <= 0.0):
		raise ValueError("Runtime samples must be strictly positive.")
	array.setflags(write=False)
	return array


def _frozen_nested_mapping(
	values: Mapping[str, Mapping[float, Any]],
) -> Mapping[str, Mapping[float, Any]]:
	"""Copy and freeze a method-by-step mapping without changing value identity."""
	return MappingProxyType(
		{
			method_name: MappingProxyType(dict(step_values))
			for method_name, step_values in values.items()
		}
	)


@dataclass(frozen=True, slots=True)
class ABBA4ProjectionComparisonResult:
	"""Two aligned ABBA4 refinements against one certified trajectory."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	reference: StoredReferenceTrajectory
	config: ABBA4ProjectionComparisonConfig
	reference_sample_indices: np.ndarray
	solutions: Mapping[str, Mapping[float, Solution]]
	series: Mapping[str, Mapping[float, TrajectoryAccuracySeries]]
	runtime_samples: Mapping[str, Mapping[float, np.ndarray]]

	def __post_init__(self) -> None:
		"""Freeze data and enforce method, refinement, and time-grid alignment."""
		if tuple(self.solutions) != ABBA4_PROJECTION_METHOD_NAMES:
			raise ValueError("Solutions must follow the ABBA4 comparison method order.")
		if tuple(self.series) != ABBA4_PROJECTION_METHOD_NAMES:
			raise ValueError("Accuracy series must follow the comparison method order.")
		if tuple(self.runtime_samples) != ABBA4_PROJECTION_METHOD_NAMES:
			raise ValueError("Runtime samples must follow the comparison method order.")
		indices = np.array(self.reference_sample_indices, dtype=np.int64, copy=True)
		initial_state = self.initial_configuration.initial_state
		if initial_state is None:
			raise ValueError("The initial configuration must contain an initial state.")
		particle_count = self.initial_configuration.particle_count(initial_state)
		common_times: np.ndarray | None = None
		frozen_runtimes: dict[str, Mapping[float, np.ndarray]] = {}
		for method_name in ABBA4_PROJECTION_METHOD_NAMES:
			if (
				tuple(self.solutions[method_name]) != self.config.integration_steps
				or tuple(self.series[method_name]) != self.config.integration_steps
				or tuple(self.runtime_samples[method_name])
				!= self.config.integration_steps
			):
				raise ValueError("Every method must follow the configured step order.")
			method_runtimes: dict[float, np.ndarray] = {}
			for step in self.config.integration_steps:
				solution = self.solutions[method_name][step]
				if not isinstance(solution, Solution):
					raise TypeError("Every comparison trajectory must be a Solution.")
				if solution.source is not self.initial_configuration:
					raise ValueError("Every solution must share one initial configuration.")
				if int(solution.diagnostics.get("step_count", -1)) != (
					self.config.step_count(step)
				):
					raise ValueError("A solution has an inconsistent complete-step count.")
				if solution.diagnostics.get("nonlinear_solver") != (
					self.config.nonlinear_solver
				):
					raise ValueError("A solution used the wrong nonlinear solver.")
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
				samples = _readonly_runtime_samples(
					self.runtime_samples[method_name][step]
				)
				if samples.size != self.config.timing_repeats:
					raise ValueError("Every run must contain the configured timing repeats.")
				method_runtimes[step] = samples
			frozen_runtimes[method_name] = MappingProxyType(method_runtimes)
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
		object.__setattr__(
			self,
			"runtime_samples",
			MappingProxyType(frozen_runtimes),
		)

	@property
	def times(self) -> np.ndarray:
		"""Return the common saved-time grid."""
		method_name = ABBA4_PROJECTION_METHOD_NAMES[0]
		return self.solutions[method_name][self.config.integration_steps[0]].t

	@property
	def reference_floor(self) -> float:
		"""Return the time-integrated particle-RMS DOP853/Radau discrepancy."""
		distances = self.reference.audit_distances[
			:, self.reference_sample_indices
		]
		mean_squared = np.mean(distances**2, axis=0)
		return float(
			np.sqrt(
				np.trapz(mean_squared, self.times)
				/ float(self.times[-1] - self.times[0])
			)
		)

	@property
	def final_reference_floor(self) -> float:
		"""Return the final-time particle-RMS DOP853/Radau discrepancy."""
		values = self.reference.audit_distances[
			:, self.reference_sample_indices[-1]
		]
		return float(np.sqrt(np.mean(values**2)))

	def summaries(self) -> tuple[ABBA4ProjectionComparisonSummary, ...]:
		"""Return comparable accuracy, Newton, multiplier, and timing rows."""
		rows: list[ABBA4ProjectionComparisonSummary] = []
		duration = float(self.times[-1] - self.times[0])
		floor = max(self.reference_floor, float(np.finfo(float).eps))
		for step in self.config.integration_steps:
			for method_name in ABBA4_PROJECTION_METHOD_NAMES:
				solution = self.solutions[method_name][step]
				accuracy = self.series[method_name][step]
				diagnostics = solution.diagnostics
				iterations = np.asarray(
					diagnostics["nonlinear_iterations"], dtype=int
				)
				residual_evaluations = np.asarray(
					diagnostics["residual_evaluations"], dtype=int
				)
				residuals = np.asarray(
					diagnostics["nonlinear_residual_norms"], dtype=float
				)
				tolerances = np.asarray(
					diagnostics["nonlinear_tolerances"], dtype=float
				)
				multipliers = np.asarray(
					diagnostics["projection_multiplier_norms"], dtype=float
				)
				expected_shape = (self.config.step_count(step),)
				if any(
					value.shape != expected_shape
					for value in (
						iterations,
						residual_evaluations,
						residuals,
						tolerances,
						multipliers,
					)
				):
					raise ValueError("Per-step nonlinear diagnostics are not aligned.")
				if np.any(tolerances <= 0.0):
					raise ValueError("Nonlinear tolerances must be strictly positive.")
				substep_iterations = np.asarray(
					diagnostics.get(
						"substep_nonlinear_iterations",
						iterations[:, np.newaxis],
					),
					dtype=int,
				)
				if (
					substep_iterations.ndim != 2
					or substep_iterations.shape[0] != expected_shape[0]
				):
					raise ValueError("Per-solve Newton diagnostics are not aligned.")
				nonlinear_solves = int(
					diagnostics.get(
						"nonlinear_solves_per_step",
						substep_iterations.shape[1],
					)
				)
				if nonlinear_solves != substep_iterations.shape[1]:
					raise ValueError("The reported nonlinear-solve count is inconsistent.")
				substep_residuals = np.asarray(
					diagnostics.get(
						"substep_nonlinear_residual_norms",
						residuals[:, np.newaxis],
					),
					dtype=float,
				)
				substep_tolerances = np.asarray(
					diagnostics.get(
						"substep_nonlinear_tolerances",
						tolerances[:, np.newaxis],
					),
					dtype=float,
				)
				if (
					substep_residuals.shape != substep_iterations.shape
					or substep_tolerances.shape != substep_iterations.shape
					or np.any(substep_tolerances <= 0.0)
				):
					raise ValueError("Per-solve residual diagnostics are not aligned.")
				residual_ratios = substep_residuals / substep_tolerances
				maps_per_residual = int(
					diagnostics.get(
						"unprojected_abba_maps_per_residual_evaluation",
						1,
					)
				)
				if maps_per_residual < 1:
					raise ValueError(
						"Every residual evaluation must traverse at least one ABBA map."
					)
				abba_map_evaluations = residual_evaluations * maps_per_residual
				newton_tangent_map_evaluations = (
					iterations * maps_per_residual
					if self.config.nonlinear_solver == "newton"
					else np.zeros_like(iterations)
				)
				time_rms = float(
					np.sqrt(
						np.trapz(accuracy.rms_distance**2, self.times) / duration
					)
				)
				runtime = self.runtime_samples[method_name][step]
				rows.append(
					ABBA4ProjectionComparisonSummary(
						method_name=method_name,
						method_label=ABBA4_PROJECTION_METHOD_LABELS[method_name],
						integration_step=step,
						step_count=expected_shape[0],
						global_rms_distance=float(
							np.sqrt(np.mean(accuracy.distances**2))
						),
						time_integrated_rms_distance=time_rms,
						maximum_distance=float(np.max(accuracy.distances)),
						final_rms_distance=float(accuracy.rms_distance[-1]),
						final_maximum_distance=float(
							accuracy.maximum_distance[-1]
						),
						reference_floor_ratio=time_rms / floor,
						nonlinear_solves_per_step=nonlinear_solves,
						minimum_iterations_per_step=int(np.min(iterations)),
						mean_iterations_per_step=float(np.mean(iterations)),
						maximum_iterations_per_step=int(np.max(iterations)),
						mean_iterations_per_solve=float(
							np.mean(substep_iterations)
						),
						maximum_iterations_per_solve=int(
							np.max(substep_iterations)
						),
						total_iterations=int(np.sum(iterations)),
						mean_residual_evaluations_per_step=float(
							np.mean(residual_evaluations)
						),
						total_residual_evaluations=int(
							np.sum(residual_evaluations)
						),
						mean_unprojected_abba_map_evaluations_per_step=(
							float(np.mean(abba_map_evaluations))
						),
						total_unprojected_abba_map_evaluations=int(
							np.sum(abba_map_evaluations)
						),
						mean_newton_tangent_abba_map_evaluations_per_step=(
							float(np.mean(newton_tangent_map_evaluations))
						),
						total_newton_tangent_abba_map_evaluations=int(
							np.sum(newton_tangent_map_evaluations)
						),
						mean_residual_to_tolerance=float(
							np.mean(residual_ratios)
						),
						maximum_residual_to_tolerance=float(
							np.max(residual_ratios)
						),
						mean_projection_multiplier_norm=float(
							np.mean(multipliers)
						),
						maximum_projection_multiplier_norm=float(
							np.max(multipliers)
						),
						runtime_seconds=float(np.median(runtime)),
						runtime_first_quartile_seconds=float(
							np.quantile(runtime, 0.25)
						),
						runtime_third_quartile_seconds=float(
							np.quantile(runtime, 0.75)
						),
						runtime_minimum_seconds=float(np.min(runtime)),
						runtime_maximum_seconds=float(np.max(runtime)),
					)
				)
		return tuple(rows)

	def convergence_orders(self) -> tuple[ABBA4ProjectionComparisonOrder, ...]:
		"""Return adjacent error orders, deficits, flags, and multiplier slopes."""
		summaries = {
			(row.method_name, row.integration_step): row for row in self.summaries()
		}
		time_floor = max(self.reference_floor, float(np.finfo(float).eps))
		final_floor = max(self.final_reference_floor, float(np.finfo(float).eps))
		rows: list[ABBA4ProjectionComparisonOrder] = []
		for coarse_step, fine_step in zip(
			self.config.integration_steps,
			self.config.integration_steps[1:],
		):
			step_ratio = coarse_step / fine_step
			for method_name in ABBA4_PROJECTION_METHOD_NAMES:
				coarse = summaries[(method_name, coarse_step)]
				fine = summaries[(method_name, fine_step)]
				time_resolved = (
					fine.time_integrated_rms_distance > 10.0 * time_floor
					and coarse.time_integrated_rms_distance > 0.0
					and fine.time_integrated_rms_distance > 0.0
				)
				final_resolved = (
					fine.final_rms_distance > 10.0 * final_floor
					and coarse.final_rms_distance > 0.0
					and fine.final_rms_distance > 0.0
				)
				time_gain = (
					coarse.time_integrated_rms_distance
					/ fine.time_integrated_rms_distance
					if fine.time_integrated_rms_distance > 0.0
					else float("nan")
				)
				final_gain = (
					coarse.final_rms_distance / fine.final_rms_distance
					if fine.final_rms_distance > 0.0
					else float("nan")
				)
				time_order = (
					float(np.log(time_gain) / np.log(step_ratio))
					if time_resolved and time_gain > 0.0
					else float("nan")
				)
				final_order = (
					float(np.log(final_gain) / np.log(step_ratio))
					if final_resolved and final_gain > 0.0
					else float("nan")
				)
				time_reduction = self.config.designed_order - time_order
				final_reduction = self.config.designed_order - final_order
				coarse_multiplier = coarse.maximum_projection_multiplier_norm
				fine_multiplier = fine.maximum_projection_multiplier_norm
				multiplier_gain = (
					coarse_multiplier / fine_multiplier
					if fine_multiplier > 0.0
					else float("nan")
				)
				multiplier_order = (
					float(np.log(multiplier_gain) / np.log(step_ratio))
					if multiplier_gain > 0.0
					else float("nan")
				)
				rows.append(
					ABBA4ProjectionComparisonOrder(
						method_name=method_name,
						method_label=ABBA4_PROJECTION_METHOD_LABELS[method_name],
						coarse_step=coarse_step,
						fine_step=fine_step,
						time_integrated_rms_gain=float(time_gain),
						final_rms_gain=float(final_gain),
						time_integrated_rms_order=time_order,
						final_rms_order=final_order,
						time_integrated_order_reduction=float(time_reduction),
						final_order_reduction=float(final_reduction),
						time_integrated_resolved_above_reference_floor=(
							time_resolved
						),
						final_resolved_above_reference_floor=final_resolved,
						resolved_above_reference_floor=(
							time_resolved and final_resolved
						),
						time_integrated_order_reduction_detected=(
							time_resolved
							and time_reduction > self.config.order_reduction_threshold
						),
						final_order_reduction_detected=(
							final_resolved
							and final_reduction > self.config.order_reduction_threshold
						),
						projection_multiplier_gain=float(multiplier_gain),
						projection_multiplier_order=multiplier_order,
					)
				)
		return tuple(rows)


def _configured_method(
	method_name: str,
	config: ABBA4ProjectionComparisonConfig,
) -> NumericalMethod:
	"""Construct either projection strategy with identical nonlinear controls."""
	method_type: type[ABBA4Implicit1] | type[ABBA4SingleProjectionImplicit1]
	if method_name == "ABBA4Implicit1":
		method_type = ABBA4Implicit1
	elif method_name == "ABBA4SingleProjectionImplicit1":
		method_type = ABBA4SingleProjectionImplicit1
	else:
		raise ValueError(f"Unknown ABBA4 projection method {method_name!r}.")
	return method_type(
		newton_absolute_tolerance=config.absolute_tolerance,
		newton_relative_tolerance=config.relative_tolerance,
		newton_max_iterations=config.max_iterations,
		nonlinear_solver=config.nonlinear_solver,
		progress=config.progress,
	)


def run_abba4_projection_comparison_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: ABBA4ProjectionComparisonConfig,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> ABBA4ProjectionComparisonResult:
	"""Refine both ABBA4 projection strategies against one stored reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if not isinstance(config, ABBA4ProjectionComparisonConfig):
		raise TypeError("`config` must be ABBA4ProjectionComparisonConfig.")
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
	solutions: dict[str, dict[float, Solution]] = {
		method_name: {} for method_name in ABBA4_PROJECTION_METHOD_NAMES
	}
	series: dict[str, dict[float, TrajectoryAccuracySeries]] = {
		method_name: {} for method_name in ABBA4_PROJECTION_METHOD_NAMES
	}
	runtimes: dict[str, dict[float, np.ndarray]] = {
		method_name: {} for method_name in ABBA4_PROJECTION_METHOD_NAMES
	}
	reference_indices: np.ndarray | None = None
	distance_convention = reference_distance_convention(reference)
	for step_index, step in enumerate(config.integration_steps):
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		methods = {
			method_name: _configured_method(method_name, config)
			for method_name in ABBA4_PROJECTION_METHOD_NAMES
		}
		for warmup_index in range(config.timing_warmups):
			warmup_order = (
				ABBA4_PROJECTION_METHOD_NAMES
				if (step_index + warmup_index) % 2 == 0
				else tuple(reversed(ABBA4_PROJECTION_METHOD_NAMES))
			)
			for method_name in warmup_order:
				simulate(problem, methods[method_name], request)
		measured: dict[str, list[float]] = {
			method_name: [] for method_name in ABBA4_PROJECTION_METHOD_NAMES
		}
		measured_solutions: dict[str, Solution] = {}
		for repetition in range(config.timing_repeats):
			measurement_order = (
				ABBA4_PROJECTION_METHOD_NAMES
				if (step_index + repetition) % 2 == 0
				else tuple(reversed(ABBA4_PROJECTION_METHOD_NAMES))
			)
			for method_name in measurement_order:
				started = perf_counter()
				candidate = simulate(problem, methods[method_name], request)
				measured[method_name].append(perf_counter() - started)
				measured_solutions.setdefault(method_name, candidate)
		for method_name in ABBA4_PROJECTION_METHOD_NAMES:
			solution = measured_solutions[method_name]
			solutions[method_name][step] = solution
			runtimes[method_name][step] = np.asarray(
				measured[method_name],
				dtype=float,
			)
			indices = reference_indices_for_times(reference, solution.t)
			if reference_indices is None:
				reference_indices = indices
			elif not np.array_equal(indices, reference_indices):
				raise ValueError("ABBA4 refinements do not share reference samples.")
			series[method_name][step] = accuracy_series(
				method_name,
				solution.states,
				reference.states[:, indices],
				period=float(potential.grid.period),
				distance_convention=distance_convention,
			)
	assert reference_indices is not None
	return ABBA4ProjectionComparisonResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		reference=reference,
		config=config,
		reference_sample_indices=reference_indices,
		solutions=solutions,
		series=series,
		runtime_samples=runtimes,
	)


__all__ = [
	"ABBA4_PROJECTION_METHOD_LABELS",
	"ABBA4_PROJECTION_METHOD_NAMES",
	"ABBA4ProjectionComparisonConfig",
	"ABBA4ProjectionComparisonOrder",
	"ABBA4ProjectionComparisonResult",
	"ABBA4ProjectionComparisonSummary",
	"run_abba4_projection_comparison_study",
]
