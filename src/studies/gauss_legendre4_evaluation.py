"""Individual accuracy, geometry, energy, and runtime study for Gauss4."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from diagnostics import calculate_step_jacobian, central_difference_jacobian
from diagnostics.symplecticity import gc_physical_symplectic_form
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	GaussLegendre4,
	GaussLegendre4IntegrationStep,
	InitialValueProblem,
	IntegrationStep,
	SimulationRequest,
	Solution,
	simulate,
)

from ._gauss_legendre4_common import (
	AdaptiveReference,
	build_adaptive_reference,
	readonly_runtime_samples,
)
from ._trajectory_accuracy import TrajectoryAccuracySeries, accuracy_series
from ._validation import integer_ratio, nonnegative_finite, positive_finite, positive_integer
from ._trajectory_accuracy import validated_refinement_steps


@dataclass(frozen=True, slots=True)
class GaussLegendre4EvaluationConfig:
	"""Complete reproducible controls for one Gauss4 refinement study."""

	integration_steps: tuple[float, ...] = (0.2, 0.1, 0.05, 0.025)
	t_span: tuple[float, float] = (0.0, 2.0)
	save_interval: float = 0.2
	rho: float = 0.3
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	newton_audit_tolerance_factor: float = 0.1
	timing_warmups: int = 1
	timing_repeats: int = 5
	symplecticity_audit_stride: int = 10
	symplecticity_finite_difference_relative_step: float = float(
		np.cbrt(np.finfo(float).eps)
	)
	reference_relative_tolerance: float = 1e-13
	reference_absolute_tolerance: float = 1e-15
	reference_maximum_step: float = 0.0025
	audit_relative_tolerance: float = 1e-13
	audit_absolute_tolerance: float = 1e-15
	audit_maximum_step: float = 0.00125
	designed_order: float = 4.0
	order_reduction_threshold: float = 0.5
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate nested steps, common outputs, tolerances, and timing controls."""
		steps = validated_refinement_steps(self.integration_steps)
		object.__setattr__(self, "integration_steps", steps)
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		for name in (
			"save_interval",
			"absolute_tolerance",
			"relative_tolerance",
			"reference_relative_tolerance",
			"reference_absolute_tolerance",
			"reference_maximum_step",
			"audit_relative_tolerance",
			"audit_absolute_tolerance",
			"audit_maximum_step",
			"designed_order",
			"symplecticity_finite_difference_relative_step",
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		threshold = float(self.order_reduction_threshold)
		if not np.isfinite(threshold) or threshold < 0.0:
			raise ValueError("`order_reduction_threshold` must be finite and non-negative.")
		object.__setattr__(self, "order_reduction_threshold", threshold)
		audit_factor = float(self.newton_audit_tolerance_factor)
		if not np.isfinite(audit_factor) or not 0.0 < audit_factor < 1.0:
			raise ValueError(
				"`newton_audit_tolerance_factor` must be finite and lie in (0, 1)."
			)
		object.__setattr__(self, "newton_audit_tolerance_factor", audit_factor)
		for name in (
			"max_iterations",
			"timing_repeats",
			"symplecticity_audit_stride",
		):
			object.__setattr__(
				self,
				name,
				positive_integer(getattr(self, name), name),
			)
		if (
			isinstance(self.timing_warmups, (bool, np.bool_))
			or not isinstance(self.timing_warmups, (int, np.integer))
			or self.timing_warmups < 0
		):
			raise ValueError("`timing_warmups` must be a non-negative integer.")
		object.__setattr__(self, "timing_warmups", int(self.timing_warmups))
		if self.audit_maximum_step > self.reference_maximum_step:
			raise ValueError("The Radau audit step cannot exceed the DOP853 step.")
		if self.audit_relative_tolerance > self.reference_relative_tolerance:
			raise ValueError("The Radau audit tolerance cannot be looser than DOP853.")
		if self.audit_absolute_tolerance > self.reference_absolute_tolerance:
			raise ValueError("The Radau audit tolerance cannot be looser than DOP853.")
		duration = self.t_span[1] - self.t_span[0]
		integer_ratio(duration, self.save_interval, "duration / save_interval")
		for step in steps:
			integer_ratio(duration, step, "duration / integration_step")
			integer_ratio(self.save_interval, step, "save_interval / integration_step")

	@property
	def output_sample_count(self) -> int:
		"""Return the number of common saved states including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class GaussLegendre4SymplecticitySeries:
	"""Ideal-root and sparse finite-stopping-rule symplecticity diagnostics."""

	times: np.ndarray
	local_relative_defects: np.ndarray
	accumulated_relative_defects: np.ndarray
	accumulated_determinant_errors: np.ndarray
	audit_times: np.ndarray
	finite_difference_relative_defects: np.ndarray
	analytic_finite_difference_relative_differences: np.ndarray

	def __post_init__(self) -> None:
		"""Own immutable finite non-negative diagnostic arrays."""
		arrays = tuple(
			np.array(value, dtype=float, copy=True)
			for value in (
				self.times,
				self.local_relative_defects,
				self.accumulated_relative_defects,
				self.accumulated_determinant_errors,
				self.audit_times,
				self.finite_difference_relative_defects,
				self.analytic_finite_difference_relative_differences,
			)
		)
		if any(value.ndim != 1 or not np.all(np.isfinite(value)) for value in arrays):
			raise ValueError("Symplecticity diagnostics must be finite vectors.")
		if not (
			arrays[0].size
			== arrays[1].size
			== arrays[2].size
			== arrays[3].size
		):
			raise ValueError("Ideal-root symplecticity arrays must be aligned.")
		if not (arrays[4].size == arrays[5].size == arrays[6].size):
			raise ValueError("Finite-difference audit arrays must be aligned.")
		if any(np.any(value < 0.0) for value in (*arrays[1:4], *arrays[5:])):
			raise ValueError("Symplecticity defect values must be non-negative.")
		for value in arrays:
			value.setflags(write=False)
		for name, value in zip(
			(
				"times",
				"local_relative_defects",
				"accumulated_relative_defects",
				"accumulated_determinant_errors",
				"audit_times",
				"finite_difference_relative_defects",
				"analytic_finite_difference_relative_differences",
			),
			arrays,
			strict=True,
		):
			object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class GaussLegendre4NewtonAudit:
	"""Trajectory discrepancy after tightening both Newton tolerances."""

	time_integrated_rms_difference: float
	final_rms_difference: float
	maximum_distance: float

	def __post_init__(self) -> None:
		"""Require finite non-negative audit floors."""
		for name in (
			"time_integrated_rms_difference",
			"final_rms_difference",
			"maximum_distance",
		):
			value = float(getattr(self, name))
			if not np.isfinite(value) or value < 0.0:
				raise ValueError("Newton-audit differences must be finite and non-negative.")
			object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class GaussLegendre4EvaluationSummary:
	"""Accuracy, cost, energy, nonlinear work, and geometry at one step."""

	integration_step: float
	step_count: int
	time_integrated_rms_distance: float
	final_rms_distance: float
	maximum_distance: float
	reference_floor_ratio: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	maximum_generalized_energy_error: float
	maximum_relative_generalized_energy_error: float
	mean_newton_iterations: float
	maximum_newton_iterations: int
	mean_residual_evaluations: float
	maximum_residual_to_tolerance: float
	maximum_nonlinear_tolerance: float
	newton_audit_time_integrated_rms_difference: float
	newton_audit_final_rms_difference: float
	maximum_local_symplecticity_defect: float
	maximum_accumulated_symplecticity_defect: float
	maximum_accumulated_determinant_error: float
	maximum_finite_difference_symplecticity_defect: float
	maximum_analytic_finite_difference_difference: float


@dataclass(frozen=True, slots=True)
class GaussLegendre4ObservedOrder:
	"""Adjacent accuracy order and deficit from the fourth-order design."""

	coarse_step: float
	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float
	time_integrated_order_deficit: float
	final_order_deficit: float
	time_integrated_resolved: bool
	final_resolved: bool
	time_integrated_reduction_detected: bool
	final_reduction_detected: bool


@dataclass(frozen=True, slots=True)
class GaussLegendre4EvaluationResult:
	"""Complete individual Gauss4 refinement result on one GC problem."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	config: GaussLegendre4EvaluationConfig
	reference: AdaptiveReference
	solutions: Mapping[float, Solution]
	accuracy: Mapping[float, TrajectoryAccuracySeries]
	runtime_samples: Mapping[float, np.ndarray]
	newton_audits: Mapping[float, GaussLegendre4NewtonAudit]
	energy_times: Mapping[float, np.ndarray]
	generalized_energies: Mapping[float, np.ndarray]
	symplecticity: Mapping[float, GaussLegendre4SymplecticitySeries]

	@property
	def times(self) -> np.ndarray:
		"""Return the common saved-time grid."""
		return self.reference.times

	def summaries(self) -> tuple[GaussLegendre4EvaluationSummary, ...]:
		"""Return one scalar row per configured integration step."""
		rows: list[GaussLegendre4EvaluationSummary] = []
		duration = float(self.times[-1] - self.times[0])
		floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		for step in self.config.integration_steps:
			solution = self.solutions[step]
			series = self.accuracy[step]
			time_rms = float(
				np.sqrt(np.trapz(series.rms_distance**2, self.times) / duration)
			)
			runtimes = self.runtime_samples[step]
			energy = self.generalized_energies[step]
			energy_error = np.abs(energy - energy[:, :1])
			energy_scale = np.maximum(
				np.abs(energy[:, :1]),
				np.finfo(float).eps,
			)
			iterations = np.asarray(solution.diagnostics["nonlinear_iterations"])
			residual_evaluations = np.asarray(
				solution.diagnostics["residual_evaluations"]
			)
			residuals = np.asarray(
				solution.diagnostics["nonlinear_residual_norms"]
			)
			tolerances = np.asarray(solution.diagnostics["nonlinear_tolerances"])
			geometry = self.symplecticity[step]
			newton_audit = self.newton_audits[step]
			rows.append(
				GaussLegendre4EvaluationSummary(
					integration_step=step,
					step_count=int(solution.diagnostics["step_count"]),
					time_integrated_rms_distance=time_rms,
					final_rms_distance=float(series.rms_distance[-1]),
					maximum_distance=float(np.max(series.distances)),
					reference_floor_ratio=time_rms / floor,
					runtime_seconds=float(np.median(runtimes)),
					runtime_first_quartile_seconds=float(np.quantile(runtimes, 0.25)),
					runtime_third_quartile_seconds=float(np.quantile(runtimes, 0.75)),
					maximum_generalized_energy_error=float(np.max(energy_error)),
					maximum_relative_generalized_energy_error=float(
						np.max(energy_error / energy_scale)
					),
					mean_newton_iterations=float(np.mean(iterations)),
					maximum_newton_iterations=int(np.max(iterations)),
					mean_residual_evaluations=float(np.mean(residual_evaluations)),
					maximum_residual_to_tolerance=float(
						np.max(residuals / tolerances)
					),
					maximum_nonlinear_tolerance=float(np.max(tolerances)),
					newton_audit_time_integrated_rms_difference=(
						newton_audit.time_integrated_rms_difference
					),
					newton_audit_final_rms_difference=(
						newton_audit.final_rms_difference
					),
					maximum_local_symplecticity_defect=float(
						np.max(geometry.local_relative_defects)
					),
					maximum_accumulated_symplecticity_defect=float(
						np.max(geometry.accumulated_relative_defects)
					),
					maximum_accumulated_determinant_error=float(
						np.max(geometry.accumulated_determinant_errors)
					),
					maximum_finite_difference_symplecticity_defect=float(
						np.max(geometry.finite_difference_relative_defects)
					),
					maximum_analytic_finite_difference_difference=float(
						np.max(
							geometry.analytic_finite_difference_relative_differences
						)
					),
				)
			)
		return tuple(rows)

	def observed_orders(self) -> tuple[GaussLegendre4ObservedOrder, ...]:
		"""Estimate resolved adjacent orders and explicit deficits from order four."""
		summaries = {row.integration_step: row for row in self.summaries()}
		time_floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		final_floor = max(
			self.reference.final_rms_floor,
			float(np.finfo(float).eps),
		)
		rows: list[GaussLegendre4ObservedOrder] = []
		for coarse_step, fine_step in zip(
			self.config.integration_steps,
			self.config.integration_steps[1:],
		):
			coarse = summaries[coarse_step]
			fine = summaries[fine_step]
			ratio = coarse_step / fine_step
			time_resolved = fine.time_integrated_rms_distance > 10.0 * max(
				time_floor,
				fine.newton_audit_time_integrated_rms_difference,
			)
			final_resolved = fine.final_rms_distance > 10.0 * max(
				final_floor,
				fine.newton_audit_final_rms_difference,
			)
			time_order = (
				float(
					np.log(
						coarse.time_integrated_rms_distance
						/ fine.time_integrated_rms_distance
					)
					/ np.log(ratio)
				)
				if time_resolved and fine.time_integrated_rms_distance > 0.0
				else float("nan")
			)
			final_order = (
				float(np.log(coarse.final_rms_distance / fine.final_rms_distance) / np.log(ratio))
				if final_resolved and fine.final_rms_distance > 0.0
				else float("nan")
			)
			time_deficit = self.config.designed_order - time_order
			final_deficit = self.config.designed_order - final_order
			rows.append(
				GaussLegendre4ObservedOrder(
					coarse_step=coarse_step,
					fine_step=fine_step,
					time_integrated_rms_order=time_order,
					final_rms_order=final_order,
					time_integrated_order_deficit=time_deficit,
					final_order_deficit=final_deficit,
					time_integrated_resolved=time_resolved,
					final_resolved=final_resolved,
					time_integrated_reduction_detected=(
						time_resolved
						and time_deficit > self.config.order_reduction_threshold
					),
					final_reduction_detected=(
						final_resolved
						and final_deficit > self.config.order_reduction_threshold
					),
				)
			)
		return tuple(rows)

	def persistent_order_reduction(self) -> bool:
		"""Return true only for two adjacent resolved reductions in either norm."""
		orders = self.observed_orders()
		return any(
			(first.time_integrated_reduction_detected and second.time_integrated_reduction_detected)
			or (first.final_reduction_detected and second.final_reduction_detected)
			for first, second in zip(orders, orders[1:])
		)


class _SymplecticityCollector:
	"""Collect exact ideal-root tangents and sparse finite-stopping audits."""

	def __init__(
		self,
		*,
		particle_count: int,
		audit_stride: int,
		finite_difference_relative_step: float,
	) -> None:
		self.form = gc_physical_symplectic_form(particle_count)
		self.form_norm = float(np.linalg.norm(self.form, ord="fro"))
		self.accumulated = np.eye(2 * particle_count)
		self.audit_stride = audit_stride
		self.finite_difference_relative_step = finite_difference_relative_step
		self.times: list[float] = []
		self.local_defects: list[float] = []
		self.flow_defects: list[float] = []
		self.determinant_errors: list[float] = []
		self.audit_times: list[float] = []
		self.finite_difference_defects: list[float] = []
		self.jacobian_differences: list[float] = []

	def __call__(self, step: IntegrationStep) -> None:
		"""Advance the accumulated ideal tangent and selected map audits."""
		if not isinstance(step, GaussLegendre4IntegrationStep):
			raise TypeError(
				"Gauss4 symplecticity requires GaussLegendre4IntegrationStep data."
			)
		analytic = calculate_step_jacobian(step, method="implicit_function")
		local_defect = analytic.T @ self.form @ analytic - self.form
		self.accumulated = analytic @ self.accumulated
		flow_defect = self.accumulated.T @ self.form @ self.accumulated - self.form
		self.times.append(step.time)
		self.local_defects.append(
			float(np.linalg.norm(local_defect, ord="fro") / self.form_norm)
		)
		self.flow_defects.append(
			float(np.linalg.norm(flow_defect, ord="fro") / self.form_norm)
		)
		self.determinant_errors.append(
			abs(float(np.linalg.det(self.accumulated)) - 1.0)
		)
		if step.step_index % self.audit_stride:
			return
		numerical = central_difference_jacobian(
			step.map_state,
			step.state_before,
			relative_step=self.finite_difference_relative_step,
		)
		numerical_defect = numerical.T @ self.form @ numerical - self.form
		self.audit_times.append(step.time)
		self.finite_difference_defects.append(
			float(np.linalg.norm(numerical_defect, ord="fro") / self.form_norm)
		)
		self.jacobian_differences.append(
			float(
				np.linalg.norm(analytic - numerical, ord="fro")
				/ max(np.linalg.norm(analytic, ord="fro"), np.finfo(float).eps)
			)
		)

	def result(self) -> GaussLegendre4SymplecticitySeries:
		"""Freeze the collected scalar time series."""
		return GaussLegendre4SymplecticitySeries(
			times=np.asarray(self.times),
			local_relative_defects=np.asarray(self.local_defects),
			accumulated_relative_defects=np.asarray(self.flow_defects),
			accumulated_determinant_errors=np.asarray(self.determinant_errors),
			audit_times=np.asarray(self.audit_times),
			finite_difference_relative_defects=np.asarray(
				self.finite_difference_defects
			),
			analytic_finite_difference_relative_differences=np.asarray(
				self.jacobian_differences
			),
		)


def run_gauss_legendre4_evaluation(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: GaussLegendre4EvaluationConfig,
) -> GaussLegendre4EvaluationResult:
	"""Run one complete individual Gauss4 evaluation over nested refinements."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, GaussLegendre4EvaluationConfig):
		raise TypeError("`config` must be GaussLegendre4EvaluationConfig.")
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_steps[-1],
		sample_count=config.output_sample_count,
	)
	reference = build_adaptive_reference(
		dynamics,
		problem.initial_state,
		request.output_times,
		period=potential.grid.period,
		relative_tolerance=config.reference_relative_tolerance,
		absolute_tolerance=config.reference_absolute_tolerance,
		maximum_step=config.reference_maximum_step,
		audit_relative_tolerance=config.audit_relative_tolerance,
		audit_absolute_tolerance=config.audit_absolute_tolerance,
		audit_maximum_step=config.audit_maximum_step,
	)
	solutions: dict[float, Solution] = {}
	accuracy_by_step: dict[float, TrajectoryAccuracySeries] = {}
	runtimes_by_step: dict[float, np.ndarray] = {}
	newton_audits_by_step: dict[float, GaussLegendre4NewtonAudit] = {}
	energy_times_by_step: dict[float, np.ndarray] = {}
	energies_by_step: dict[float, np.ndarray] = {}
	geometry_by_step: dict[float, GaussLegendre4SymplecticitySeries] = {}
	particle_count = problem.particle_count

	for step in config.integration_steps:
		step_request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		collector = _SymplecticityCollector(
			particle_count=particle_count,
			audit_stride=config.symplecticity_audit_stride,
			finite_difference_relative_step=(
				config.symplecticity_finite_difference_relative_step
			),
		)
		method = GaussLegendre4(
			track_energy=False,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			progress=config.progress,
			step_observer=collector,
		)
		solution = simulate(problem, method, step_request)
		solutions[step] = solution
		accuracy_by_step[step] = accuracy_series(
			"GaussLegendre4",
			solution.states,
			reference.states,
			period=potential.grid.period,
			distance_convention="periodic",
		)
		energy_request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=(
				integer_ratio(
					config.t_span[1] - config.t_span[0],
					step,
					"duration / energy integration step",
				)
				+ 1
			),
		)
		energy_solution = simulate(
			problem,
			GaussLegendre4(
				track_energy=True,
				newton_absolute_tolerance=config.absolute_tolerance,
				newton_relative_tolerance=config.relative_tolerance,
				newton_max_iterations=config.max_iterations,
				newton_jacobian_method="analytic",
				progress=False,
			),
			energy_request,
		)
		momentum = energy_solution.k
		if momentum is None:
			raise RuntimeError("Gauss4 energy tracking did not return momentum.")
		energy_times_by_step[step] = energy_solution.t
		energies_by_step[step] = np.asarray(
			dynamics.hamiltonian(energy_solution.t, energy_solution.states) + momentum,
			dtype=float,
		)
		geometry_by_step[step] = collector.result()
		newton_audit_solution = simulate(
			problem,
			GaussLegendre4(
				track_energy=False,
				newton_absolute_tolerance=(
					config.absolute_tolerance
					* config.newton_audit_tolerance_factor
				),
				newton_relative_tolerance=(
					config.relative_tolerance
					* config.newton_audit_tolerance_factor
				),
				newton_max_iterations=config.max_iterations,
				newton_jacobian_method="analytic",
				progress=False,
			),
			step_request,
		)
		newton_audit_series = accuracy_series(
			"GaussLegendre4 tighter Newton audit",
			solution.states,
			newton_audit_solution.states,
			period=potential.grid.period,
			distance_convention="periodic",
		)
		duration = float(reference.times[-1] - reference.times[0])
		newton_audits_by_step[step] = GaussLegendre4NewtonAudit(
			time_integrated_rms_difference=float(
				np.sqrt(
					np.trapz(newton_audit_series.rms_distance**2, reference.times)
					/ duration
				)
			),
			final_rms_difference=float(newton_audit_series.rms_distance[-1]),
			maximum_distance=float(np.max(newton_audit_series.distances)),
		)

		timing_method = GaussLegendre4(
			track_energy=False,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			progress=False,
		)
		for _ in range(config.timing_warmups):
			simulate(problem, timing_method, step_request)
		timing_samples: list[float] = []
		for _ in range(config.timing_repeats):
			started = perf_counter()
			simulate(problem, timing_method, step_request)
			timing_samples.append(perf_counter() - started)
		runtimes_by_step[step] = readonly_runtime_samples(
			np.asarray(timing_samples)
		)

	return GaussLegendre4EvaluationResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		config=config,
		reference=reference,
		solutions=MappingProxyType(solutions),
		accuracy=MappingProxyType(accuracy_by_step),
		runtime_samples=MappingProxyType(runtimes_by_step),
		newton_audits=MappingProxyType(newton_audits_by_step),
		energy_times=MappingProxyType(energy_times_by_step),
		generalized_energies=MappingProxyType(energies_by_step),
		symplecticity=MappingProxyType(geometry_by_step),
	)


__all__ = [
	"GaussLegendre4EvaluationConfig",
	"GaussLegendre4EvaluationResult",
	"GaussLegendre4EvaluationSummary",
	"GaussLegendre4NewtonAudit",
	"GaussLegendre4ObservedOrder",
	"GaussLegendre4SymplecticitySeries",
	"run_gauss_legendre4_evaluation",
]
