"""Reference accuracy and refinement study for fourth-order implicit ABBA."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

import numpy as np

from diagnostics import StoredReferenceTrajectory
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA4Implicit1,
	NONLINEAR_SOLVERS,
	InitialValueProblem,
	NonlinearSolver,
	SimulationRequest,
	Solution,
	simulate,
)

from ._trajectory_accuracy import (
	TrajectoryAccuracySeries,
	accuracy_series,
	reference_indices_for_times,
	reference_distance_convention,
	validate_reference_identity,
	validated_refinement_steps,
)
from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)


@dataclass(frozen=True, slots=True)
class ABBA4Implicit1AccuracyConfig:
	"""Physical, nonlinear, integration, and sampling controls."""

	integration_steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05)
	t_span: tuple[float, float] = (0.0, 2.0)
	save_interval: float = 0.4
	rho: float = 0.3
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	nonlinear_solver: NonlinearSolver = "newton"
	progress: bool = False

	def __post_init__(self) -> None:
		"""Require nested complete steps and one common main-grid cadence."""
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
		for name in ("absolute_tolerance", "relative_tolerance"):
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
		if self.nonlinear_solver not in NONLINEAR_SOLVERS:
			raise ValueError("`nonlinear_solver` must be 'newton' or 'broyden'.")
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
		"""Return the common number of saved states including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class ABBA4Implicit1AccuracySummary:
	"""Accuracy, nonlinear work, and runtime for one complete step size."""

	method_name: str
	integration_step: float
	step_count: int
	global_rms_distance: float
	time_integrated_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float
	normalized_global_rms_distance: float
	reference_floor_ratio: float
	runtime_seconds: float
	maximum_total_iterations_per_step: int
	mean_total_iterations_per_step: float
	maximum_substep_iterations: int
	total_residual_evaluations: int


@dataclass(frozen=True, slots=True)
class ABBA4Implicit1AccuracyOrder:
	"""Observed error gains and orders between adjacent nested steps."""

	coarse_step: float
	fine_step: float
	time_integrated_rms_gain: float
	final_rms_gain: float
	time_integrated_rms_order: float
	final_rms_order: float
	resolved_above_reference_floor: bool


@dataclass(frozen=True, slots=True)
class ABBA4Implicit1AccuracyResult:
	"""Aligned ABBA4 trajectories and errors across one nested refinement."""

	method_name: ClassVar[str] = "ABBA4Implicit1"
	summary_type: ClassVar[type[ABBA4Implicit1AccuracySummary]] = (
		ABBA4Implicit1AccuracySummary
	)
	order_type: ClassVar[type[ABBA4Implicit1AccuracyOrder]] = (
		ABBA4Implicit1AccuracyOrder
	)

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	reference: StoredReferenceTrajectory
	config: ABBA4Implicit1AccuracyConfig
	reference_sample_indices: np.ndarray
	solutions: Mapping[float, Solution]
	series: Mapping[float, TrajectoryAccuracySeries]
	runtimes: Mapping[float, float]

	def __post_init__(self) -> None:
		"""Freeze mappings and enforce the common saved-time grid."""
		steps = self.config.integration_steps
		if (
			tuple(self.solutions) != steps
			or tuple(self.series) != steps
			or tuple(self.runtimes) != steps
		):
			raise ValueError("ABBA4 accuracy results must follow configured step order.")
		indices = np.array(self.reference_sample_indices, dtype=np.int64, copy=True)
		initial_state = self.initial_configuration.initial_state
		if initial_state is None:
			raise ValueError("The initial configuration must contain an initial state.")
		particle_count = self.initial_configuration.particle_count(initial_state)
		reference_times: np.ndarray | None = None
		for step in steps:
			solution = self.solutions[step]
			if not isinstance(solution, Solution):
				raise TypeError("Every ABBA4 refinement value must be a Solution.")
			if solution.source is not self.initial_configuration:
				raise ValueError("Every ABBA4 solution must share one configuration.")
			if reference_times is None:
				reference_times = solution.t
			elif not np.array_equal(solution.t, reference_times):
				raise ValueError("Every ABBA4 solution must share one saved-time grid.")
			if self.series[step].distances.shape != (
				particle_count,
				solution.t.size,
			):
				raise ValueError("An ABBA4 accuracy series has an invalid shape.")
			if not np.isfinite(self.runtimes[step]) or self.runtimes[step] <= 0.0:
				raise ValueError("Every ABBA4 runtime must be positive and finite.")
		assert reference_times is not None
		if (
			indices.shape != reference_times.shape
			or np.any(indices < 0)
			or np.any(indices >= self.reference.times.size)
			or not np.array_equal(self.reference.times[indices], reference_times)
		):
			raise ValueError("ABBA4 samples do not align with the stored reference.")
		indices.setflags(write=False)
		object.__setattr__(self, "reference_sample_indices", indices)
		object.__setattr__(self, "solutions", MappingProxyType(dict(self.solutions)))
		object.__setattr__(self, "series", MappingProxyType(dict(self.series)))
		object.__setattr__(self, "runtimes", MappingProxyType(dict(self.runtimes)))

	@property
	def times(self) -> np.ndarray:
		"""Return the common saved-time grid."""
		return self.solutions[self.config.integration_steps[0]].t

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

	def summaries(self) -> tuple[ABBA4Implicit1AccuracySummary, ...]:
		"""Return error, work, and runtime metrics from coarse to fine."""
		rows: list[ABBA4Implicit1AccuracySummary] = []
		period = float(self.potential.grid.period)
		floor = max(self.reference_floor, float(np.finfo(float).eps))
		duration = float(self.times[-1] - self.times[0])
		for step in self.config.integration_steps:
			solution = self.solutions[step]
			values = self.series[step]
			iterations = np.asarray(
				solution.diagnostics["nonlinear_iterations"],
				dtype=int,
			)
			substep_iterations = np.asarray(
				solution.diagnostics["substep_nonlinear_iterations"],
				dtype=int,
			)
			residual_evaluations = np.asarray(
				solution.diagnostics["residual_evaluations"],
				dtype=int,
			)
			global_rms = float(np.sqrt(np.mean(values.distances**2)))
			time_rms = float(
				np.sqrt(
					np.trapz(values.rms_distance**2, self.times) / duration
				)
			)
			rows.append(
				self.summary_type(
					method_name=self.method_name,
					integration_step=step,
					step_count=int(solution.diagnostics["step_count"]),
					global_rms_distance=global_rms,
					time_integrated_rms_distance=time_rms,
					maximum_distance=float(np.max(values.distances)),
					final_rms_distance=float(values.rms_distance[-1]),
					final_maximum_distance=float(values.maximum_distance[-1]),
					normalized_global_rms_distance=global_rms / period,
					reference_floor_ratio=time_rms / floor,
					runtime_seconds=float(self.runtimes[step]),
					maximum_total_iterations_per_step=int(np.max(iterations)),
					mean_total_iterations_per_step=float(np.mean(iterations)),
					maximum_substep_iterations=int(np.max(substep_iterations)),
					total_residual_evaluations=int(np.sum(residual_evaluations)),
				)
			)
		return tuple(rows)

	def convergence_orders(self) -> tuple[ABBA4Implicit1AccuracyOrder, ...]:
		"""Estimate time-integrated and final RMS orders under step halving."""
		summaries = {row.integration_step: row for row in self.summaries()}
		time_floor = max(self.reference_floor, float(np.finfo(float).eps))
		final_floor = max(self.final_reference_floor, float(np.finfo(float).eps))
		rows: list[ABBA4Implicit1AccuracyOrder] = []
		for coarse_step, fine_step in zip(
			self.config.integration_steps,
			self.config.integration_steps[1:],
		):
			coarse = summaries[coarse_step]
			fine = summaries[fine_step]
			time_gain = (
				coarse.time_integrated_rms_distance
				/ fine.time_integrated_rms_distance
			)
			final_gain = coarse.final_rms_distance / fine.final_rms_distance
			step_ratio = coarse_step / fine_step
			time_resolved = fine.time_integrated_rms_distance > 10.0 * time_floor
			final_resolved = fine.final_rms_distance > 10.0 * final_floor
			rows.append(
				self.order_type(
					coarse_step=coarse_step,
					fine_step=fine_step,
					time_integrated_rms_gain=float(time_gain),
					final_rms_gain=float(final_gain),
					time_integrated_rms_order=(
						float(np.log(time_gain) / np.log(step_ratio))
						if time_resolved
						else float("nan")
					),
					final_rms_order=(
						float(np.log(final_gain) / np.log(step_ratio))
						if final_resolved
						else float("nan")
					),
					resolved_above_reference_floor=time_resolved and final_resolved,
				)
			)
		return tuple(rows)


def run_abba4_implicit_1_accuracy_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	reference: StoredReferenceTrajectory,
	*,
	config: ABBA4Implicit1AccuracyConfig,
	potential_metadata: Mapping[str, Any],
	initial_condition_metadata: Mapping[str, Any],
) -> ABBA4Implicit1AccuracyResult:
	"""Run ABBA4 on nested steps and compare every saved state to the reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(reference, StoredReferenceTrajectory):
		raise TypeError("`reference` must be a StoredReferenceTrajectory.")
	if not isinstance(config, ABBA4Implicit1AccuracyConfig):
		raise TypeError("`config` must be ABBA4Implicit1AccuracyConfig.")
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
	solutions: dict[float, Solution] = {}
	series: dict[float, TrajectoryAccuracySeries] = {}
	runtimes: dict[float, float] = {}
	reference_indices: np.ndarray | None = None
	distance_convention = reference_distance_convention(reference)
	for step in config.integration_steps:
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		started = perf_counter()
		solution = simulate(
			problem,
			ABBA4Implicit1(
				newton_absolute_tolerance=config.absolute_tolerance,
				newton_relative_tolerance=config.relative_tolerance,
				newton_max_iterations=config.max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
			),
			request,
		)
		runtimes[step] = perf_counter() - started
		solutions[step] = solution
		indices = reference_indices_for_times(reference, solution.t)
		if reference_indices is None:
			reference_indices = indices
		elif not np.array_equal(indices, reference_indices):
			raise ValueError("ABBA4 refinements do not share reference sample indices.")
		series[step] = accuracy_series(
			"ABBA4Implicit1",
			solution.states,
			reference.states[:, indices],
			period=float(potential.grid.period),
			distance_convention=distance_convention,
		)
	assert reference_indices is not None
	return ABBA4Implicit1AccuracyResult(
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
	"ABBA4Implicit1AccuracyConfig",
	"ABBA4Implicit1AccuracyOrder",
	"ABBA4Implicit1AccuracyResult",
	"ABBA4Implicit1AccuracySummary",
	"run_abba4_implicit_1_accuracy_study",
]
