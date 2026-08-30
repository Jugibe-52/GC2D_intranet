"""Accuracy and runtime comparison between Gauss4 and implicit BM4."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	BM4Implicit1,
	GaussLegendre4,
	InitialValueProblem,
	NumericalMethod,
	SimulationRequest,
	Solution,
	simulate,
)

from ._gauss_legendre4_common import (
	AdaptiveReference,
	build_adaptive_reference,
	readonly_runtime_samples,
)
from ._trajectory_accuracy import (
	TrajectoryAccuracySeries,
	accuracy_series,
	validated_refinement_steps,
)
from ._validation import integer_ratio, nonnegative_finite, positive_finite, positive_integer


GAUSS_BM4_METHODS: tuple[str, ...] = ("GaussLegendre4", "BM4Implicit1")
GAUSS_BM4_LABELS: Mapping[str, str] = MappingProxyType(
	{
		"GaussLegendre4": "Gauss--Legendre (2 stages, order 4)",
		"BM4Implicit1": "Implicit projected BM4",
	}
)


@dataclass(frozen=True, slots=True)
class GaussBM4ComparisonConfig:
	"""Shared refinement, Newton, reference, and timing controls."""

	integration_steps: tuple[float, ...] = (0.2, 0.1, 0.05, 0.025)
	t_span: tuple[float, float] = (0.0, 2.0)
	save_interval: float = 0.2
	rho: float = 0.3
	coupling_frequency: float = float(np.pi / 8.0)
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	timing_warmups: int = 1
	timing_repeats: int = 5
	reference_relative_tolerance: float = 1e-13
	reference_absolute_tolerance: float = 1e-15
	reference_maximum_step: float = 0.0025
	audit_relative_tolerance: float = 1e-13
	audit_absolute_tolerance: float = 1e-15
	audit_maximum_step: float = 0.00125
	designed_order: float = 4.0
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate one nested, aligned comparison configuration."""
		steps = validated_refinement_steps(self.integration_steps)
		object.__setattr__(self, "integration_steps", steps)
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
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
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		for name in ("max_iterations", "timing_repeats"):
			object.__setattr__(self, name, positive_integer(getattr(self, name), name))
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
		"""Return the common number of saved samples."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class GaussBM4ComparisonSummary:
	"""Accuracy and robust runtime metrics for one method and step."""

	method_name: str
	method_label: str
	integration_step: float
	step_count: int
	time_integrated_rms_distance: float
	final_rms_distance: float
	maximum_distance: float
	reference_floor_ratio: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	runtime_minimum_seconds: float
	runtime_maximum_seconds: float
	mean_newton_iterations: float
	mean_residual_evaluations: float
	maximum_residual_to_tolerance: float


@dataclass(frozen=True, slots=True)
class GaussBM4ObservedOrder:
	"""Resolved adjacent accuracy order for one comparison method."""

	method_name: str
	coarse_step: float
	fine_step: float
	time_integrated_rms_order: float
	final_rms_order: float
	resolved_above_reference_floor: bool


@dataclass(frozen=True, slots=True)
class GaussBM4EqualStepRatio:
	"""Direct Gauss/BM4 error and runtime ratios at one common step."""

	integration_step: float
	gauss_to_bm4_error_ratio: float
	bm4_to_gauss_runtime_ratio: float


@dataclass(frozen=True, slots=True)
class GaussBM4EqualAccuracyRatio:
	"""Interpolated runtime comparison at one common trajectory error."""

	target_time_integrated_rms_distance: float
	gauss_runtime_seconds: float
	bm4_runtime_seconds: float
	bm4_to_gauss_runtime_ratio: float


@dataclass(frozen=True, slots=True)
class GaussBM4ComparisonResult:
	"""Aligned Gauss4 and BM4 solutions across one refinement grid."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	config: GaussBM4ComparisonConfig
	reference: AdaptiveReference
	solutions: Mapping[str, Mapping[float, Solution]]
	accuracy: Mapping[str, Mapping[float, TrajectoryAccuracySeries]]
	runtime_samples: Mapping[str, Mapping[float, np.ndarray]]

	@property
	def times(self) -> np.ndarray:
		"""Return the common reference and solution output grid."""
		return self.reference.times

	def summaries(self) -> tuple[GaussBM4ComparisonSummary, ...]:
		"""Return step-major, method-minor accuracy and timing rows."""
		rows: list[GaussBM4ComparisonSummary] = []
		duration = float(self.times[-1] - self.times[0])
		floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		for step in self.config.integration_steps:
			for method_name in GAUSS_BM4_METHODS:
				solution = self.solutions[method_name][step]
				series = self.accuracy[method_name][step]
				time_rms = float(
					np.sqrt(np.trapz(series.rms_distance**2, self.times) / duration)
				)
				runtimes = self.runtime_samples[method_name][step]
				iterations = np.asarray(
					solution.diagnostics["nonlinear_iterations"], dtype=float
				)
				residual_evaluations = np.asarray(
					solution.diagnostics["residual_evaluations"], dtype=float
				)
				residuals = np.asarray(
					solution.diagnostics["nonlinear_residual_norms"], dtype=float
				)
				tolerances = np.asarray(
					solution.diagnostics["nonlinear_tolerances"], dtype=float
				)
				rows.append(
					GaussBM4ComparisonSummary(
						method_name=method_name,
						method_label=GAUSS_BM4_LABELS[method_name],
						integration_step=step,
						step_count=int(solution.diagnostics["step_count"]),
						time_integrated_rms_distance=time_rms,
						final_rms_distance=float(series.rms_distance[-1]),
						maximum_distance=float(np.max(series.distances)),
						reference_floor_ratio=time_rms / floor,
						runtime_seconds=float(np.median(runtimes)),
						runtime_first_quartile_seconds=float(
							np.quantile(runtimes, 0.25)
						),
						runtime_third_quartile_seconds=float(
							np.quantile(runtimes, 0.75)
						),
						runtime_minimum_seconds=float(np.min(runtimes)),
						runtime_maximum_seconds=float(np.max(runtimes)),
						mean_newton_iterations=float(np.mean(iterations)),
						mean_residual_evaluations=float(
							np.mean(residual_evaluations)
						),
						maximum_residual_to_tolerance=float(
							np.max(residuals / tolerances)
						),
					)
				)
		return tuple(rows)

	def observed_orders(self) -> tuple[GaussBM4ObservedOrder, ...]:
		"""Estimate adjacent orders only while both errors exceed audit floors."""
		summaries = {
			(row.method_name, row.integration_step): row for row in self.summaries()
		}
		time_floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		final_floor = max(
			self.reference.final_rms_floor,
			float(np.finfo(float).eps),
		)
		rows: list[GaussBM4ObservedOrder] = []
		for coarse_step, fine_step in zip(
			self.config.integration_steps,
			self.config.integration_steps[1:],
		):
			ratio = coarse_step / fine_step
			for method_name in GAUSS_BM4_METHODS:
				coarse = summaries[(method_name, coarse_step)]
				fine = summaries[(method_name, fine_step)]
				time_resolved = fine.time_integrated_rms_distance > 10.0 * time_floor
				final_resolved = fine.final_rms_distance > 10.0 * final_floor
				time_order = (
					float(
						np.log(
							coarse.time_integrated_rms_distance
							/ fine.time_integrated_rms_distance
						)
						/ np.log(ratio)
					)
					if time_resolved
					else float("nan")
				)
				final_order = (
					float(
						np.log(coarse.final_rms_distance / fine.final_rms_distance)
						/ np.log(ratio)
					)
					if final_resolved
					else float("nan")
				)
				rows.append(
					GaussBM4ObservedOrder(
						method_name=method_name,
						coarse_step=coarse_step,
						fine_step=fine_step,
						time_integrated_rms_order=time_order,
						final_rms_order=final_order,
						resolved_above_reference_floor=(
							time_resolved and final_resolved
						),
					)
				)
		return tuple(rows)

	def equal_step_ratios(self) -> tuple[GaussBM4EqualStepRatio, ...]:
		"""Return direct accuracy and runtime ratios at each shared step."""
		summaries = {
			(row.method_name, row.integration_step): row for row in self.summaries()
		}
		return tuple(
			GaussBM4EqualStepRatio(
				integration_step=step,
				gauss_to_bm4_error_ratio=(
					summaries[("GaussLegendre4", step)].time_integrated_rms_distance
					/ summaries[("BM4Implicit1", step)].time_integrated_rms_distance
				),
				bm4_to_gauss_runtime_ratio=(
					summaries[("BM4Implicit1", step)].runtime_seconds
					/ summaries[("GaussLegendre4", step)].runtime_seconds
				),
			)
			for step in self.config.integration_steps
		)

	def equal_accuracy_ratios(self) -> tuple[GaussBM4EqualAccuracyRatio, ...]:
		"""Interpolate both work--precision curves at three shared errors."""
		rows = self.summaries()
		curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
		for method_name in GAUSS_BM4_METHODS:
			method_rows = tuple(row for row in rows if row.method_name == method_name)
			positive_rows = tuple(
				row
				for row in method_rows
				if row.time_integrated_rms_distance > 0.0 and row.runtime_seconds > 0.0
			)
			if len(positive_rows) < 2:
				return ()
			sorted_rows = sorted(
				positive_rows,
				key=lambda row: row.time_integrated_rms_distance,
			)
			errors = np.asarray(
				[row.time_integrated_rms_distance for row in sorted_rows]
			)
			runtimes = np.asarray([row.runtime_seconds for row in sorted_rows])
			if np.any(np.diff(errors) <= 0.0):
				return ()
			curves[method_name] = errors, runtimes
		lower = max(float(np.min(curves[name][0])) for name in GAUSS_BM4_METHODS)
		upper = min(float(np.max(curves[name][0])) for name in GAUSS_BM4_METHODS)
		if lower >= upper:
			return ()
		targets = np.exp(np.linspace(np.log(lower), np.log(upper), 5)[1:-1])[::-1]

		def interpolated_runtime(method_name: str, target: float) -> float:
			errors, runtimes = curves[method_name]
			return float(
				np.exp(
					np.interp(
						np.log(target),
						np.log(errors),
						np.log(runtimes),
					)
				)
			)

		result: list[GaussBM4EqualAccuracyRatio] = []
		for target in targets:
			gauss_runtime = interpolated_runtime("GaussLegendre4", float(target))
			bm4_runtime = interpolated_runtime("BM4Implicit1", float(target))
			result.append(
				GaussBM4EqualAccuracyRatio(
					target_time_integrated_rms_distance=float(target),
					gauss_runtime_seconds=gauss_runtime,
					bm4_runtime_seconds=bm4_runtime,
					bm4_to_gauss_runtime_ratio=bm4_runtime / gauss_runtime,
				)
			)
		return tuple(result)


def _method(
	method_name: str,
	config: GaussBM4ComparisonConfig,
) -> NumericalMethod:
	"""Build one comparison method with aligned Newton controls."""
	if method_name == "GaussLegendre4":
		return GaussLegendre4(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			progress=config.progress,
		)
	if method_name == "BM4Implicit1":
		return BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			nonlinear_solver="newton",
			progress=config.progress,
		)
	raise ValueError(f"Unknown Gauss/BM4 comparison method {method_name!r}.")


def run_gauss_bm4_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: GaussBM4ComparisonConfig,
) -> GaussBM4ComparisonResult:
	"""Run aligned accuracy refinements and alternated robust timings."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, GaussBM4ComparisonConfig):
		raise TypeError("`config` must be GaussBM4ComparisonConfig.")
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	base_request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_steps[-1],
		sample_count=config.output_sample_count,
	)
	reference = build_adaptive_reference(
		dynamics,
		problem.initial_state,
		base_request.output_times,
		period=potential.grid.period,
		relative_tolerance=config.reference_relative_tolerance,
		absolute_tolerance=config.reference_absolute_tolerance,
		maximum_step=config.reference_maximum_step,
		audit_relative_tolerance=config.audit_relative_tolerance,
		audit_absolute_tolerance=config.audit_absolute_tolerance,
		audit_maximum_step=config.audit_maximum_step,
	)
	solutions: dict[str, dict[float, Solution]] = {
		name: {} for name in GAUSS_BM4_METHODS
	}
	accuracy_by_method: dict[str, dict[float, TrajectoryAccuracySeries]] = {
		name: {} for name in GAUSS_BM4_METHODS
	}
	runtimes: dict[str, dict[float, np.ndarray]] = {
		name: {} for name in GAUSS_BM4_METHODS
	}

	for step_index, step in enumerate(config.integration_steps):
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		for method_name in GAUSS_BM4_METHODS:
			solution = simulate(problem, _method(method_name, config), request)
			solutions[method_name][step] = solution
			accuracy_by_method[method_name][step] = accuracy_series(
				method_name,
				solution.states,
				reference.states,
				period=potential.grid.period,
				distance_convention="periodic",
			)
		timing_methods = {
			method_name: _method(method_name, config)
			for method_name in GAUSS_BM4_METHODS
		}
		for _ in range(config.timing_warmups):
			for method_name in GAUSS_BM4_METHODS:
				simulate(problem, timing_methods[method_name], request)
		samples: dict[str, list[float]] = {name: [] for name in GAUSS_BM4_METHODS}
		for repeat in range(config.timing_repeats):
			order = (
				GAUSS_BM4_METHODS
				if (step_index + repeat) % 2 == 0
				else tuple(reversed(GAUSS_BM4_METHODS))
			)
			for method_name in order:
				started = perf_counter()
				simulate(problem, timing_methods[method_name], request)
				samples[method_name].append(perf_counter() - started)
		for method_name in GAUSS_BM4_METHODS:
			runtimes[method_name][step] = readonly_runtime_samples(
				np.asarray(samples[method_name])
			)

	return GaussBM4ComparisonResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		config=config,
		reference=reference,
		solutions=MappingProxyType(
			{name: MappingProxyType(values) for name, values in solutions.items()}
		),
		accuracy=MappingProxyType(
			{
				name: MappingProxyType(values)
				for name, values in accuracy_by_method.items()
			}
		),
		runtime_samples=MappingProxyType(
			{name: MappingProxyType(values) for name, values in runtimes.items()}
		),
	)


__all__ = [
	"GAUSS_BM4_LABELS",
	"GAUSS_BM4_METHODS",
	"GaussBM4ComparisonConfig",
	"GaussBM4ComparisonResult",
	"GaussBM4ComparisonSummary",
	"GaussBM4EqualAccuracyRatio",
	"GaussBM4EqualStepRatio",
	"GaussBM4ObservedOrder",
	"run_gauss_bm4_comparison",
]
