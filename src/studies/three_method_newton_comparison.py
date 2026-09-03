"""Multi-trajectory comparison of three fourth-order Newton integrators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import GC2DH5Potential, Potential
from simulation import (
	ABBA4ImplicitSingleProjection,
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
from ._trajectory_accuracy import TrajectoryAccuracySeries, accuracy_series
from ._trajectory_distances import (
	DistanceConvention,
	normalized_distance_convention,
)
from ._validation import integer_ratio, nonnegative_finite, positive_finite, positive_integer


THREE_METHOD_NEWTON_METHODS: tuple[str, ...] = (
	"ABBA4ImplicitSingleProjection",
	"GaussLegendre4",
	"BM4Implicit1",
)
THREE_METHOD_NEWTON_LABELS: Mapping[str, str] = MappingProxyType(
	{
		"ABBA4ImplicitSingleProjection": "Single-projection implicit ABBA4",
		"GaussLegendre4": "Gauss--Legendre (2 stages, order 4)",
		"BM4Implicit1": "Single-projection implicit BM4",
	}
)


@dataclass(frozen=True, slots=True)
class ThreeMethodNewtonComparisonConfig:
	"""Common physical grid, Newton tolerances, and reference controls."""

	rho: float = 0.3
	coupling_frequency: float = float(np.pi / 8.0)
	t_span: tuple[float, float] = (0.0, 10.0)
	integration_step: float = 0.05
	save_interval: float | None = None
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 40
	jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	reference_relative_tolerance: float = 1e-13
	reference_absolute_tolerance: float = 1e-15
	reference_maximum_step: float = 0.01
	audit_relative_tolerance: float = 1e-13
	audit_absolute_tolerance: float = 1e-15
	audit_maximum_step: float = 0.005
	timing_warmups: int = 1
	timing_repeats: int = 3
	distance_convention: DistanceConvention = "periodic"
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate every parameter that affects reproducibility."""
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
			"integration_step",
			"absolute_tolerance",
			"relative_tolerance",
			"jacobian_relative_step",
			"reference_relative_tolerance",
			"reference_absolute_tolerance",
			"reference_maximum_step",
			"audit_relative_tolerance",
			"audit_absolute_tolerance",
			"audit_maximum_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(
			self,
			"max_iterations",
			positive_integer(self.max_iterations, "max_iterations"),
		)
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
		object.__setattr__(
			self,
			"distance_convention",
			normalized_distance_convention(self.distance_convention),
		)
		save_interval = (
			self.integration_step
			if self.save_interval is None
			else positive_finite(self.save_interval, "save_interval")
		)
		object.__setattr__(self, "save_interval", save_interval)
		if self.audit_maximum_step > self.reference_maximum_step:
			raise ValueError("The Radau audit step cannot exceed the DOP853 step.")
		if self.audit_relative_tolerance > self.reference_relative_tolerance:
			raise ValueError("The Radau audit tolerance cannot be looser than DOP853.")
		if self.audit_absolute_tolerance > self.reference_absolute_tolerance:
			raise ValueError("The Radau audit tolerance cannot be looser than DOP853.")
		duration = self.t_span[1] - self.t_span[0]
		integer_ratio(duration, self.integration_step, "duration / integration_step")
		integer_ratio(duration, save_interval, "duration / save_interval")
		integer_ratio(save_interval, self.integration_step, "save_interval / integration_step")
		object.__setattr__(self, "progress", bool(self.progress))

	@property
	def step_count(self) -> int:
		"""Return the number of complete fixed steps in every method run."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.integration_step,
			"duration / integration_step",
		)

	@property
	def output_sample_count(self) -> int:
		"""Return the number of common saved times, including both endpoints."""
		assert self.save_interval is not None
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class ThreeMethodNewtonSummary:
	"""Accuracy, runtime, and nonlinear work for one complete method run."""

	method_name: str
	method_label: str
	nonlinear_solver: str
	trajectory_count: int
	step_count: int
	global_rms_distance: float
	time_integrated_rms_distance: float
	final_rms_distance: float
	maximum_distance: float
	reference_floor_ratio: float
	time_integrated_rms_energy_error: float
	relative_time_integrated_rms_energy_error: float
	final_rms_energy_error: float
	maximum_absolute_energy_error: float
	energy_reference_floor_ratio: float
	runtime_seconds: float
	runtime_first_quartile_seconds: float
	runtime_third_quartile_seconds: float
	runtime_minimum_seconds: float
	runtime_maximum_seconds: float
	minimum_newton_iterations: int
	mean_newton_iterations: float
	maximum_newton_iterations: int
	total_newton_iterations: int
	mean_residual_evaluations: float
	total_residual_evaluations: int
	maximum_residual_to_tolerance: float


def _residual_evaluations(solution: Solution) -> np.ndarray:
	"""Read the common residual-work series from either public diagnostic schema."""
	key = (
		"residual_evaluations"
		if "residual_evaluations" in solution.diagnostics
		else "residual_evaluations_per_step"
	)
	return np.asarray(solution.diagnostics[key], dtype=int)


@dataclass(frozen=True, slots=True)
class EnergyAccuracySeries:
	"""Signed physical-Hamiltonian errors against DOP853 for every particle."""

	method_name: str
	errors: np.ndarray

	def __post_init__(self) -> None:
		"""Own one immutable finite ``(particles, samples)`` error history."""
		if not isinstance(self.method_name, str) or not self.method_name:
			raise ValueError("`method_name` must be a non-empty string.")
		errors = np.array(self.errors, dtype=float, copy=True)
		if errors.ndim != 2 or errors.size == 0 or not np.all(np.isfinite(errors)):
			raise ValueError("Energy errors must be one finite particle-time array.")
		errors.setflags(write=False)
		object.__setattr__(self, "errors", errors)

	@property
	def rms_error(self) -> np.ndarray:
		"""Return the particle-RMS physical-energy error at every saved time."""
		return np.asarray(np.sqrt(np.mean(self.errors**2, axis=0)), dtype=float)

	@property
	def maximum_absolute_error(self) -> np.ndarray:
		"""Return the worst absolute particle-energy error at every saved time."""
		return np.asarray(np.max(np.abs(self.errors), axis=0), dtype=float)

	@property
	def running_maximum_absolute_error(self) -> np.ndarray:
		"""Return the running worst particle-energy error through time."""
		return np.maximum.accumulate(self.maximum_absolute_error)


def _readonly_energy_history(
	values: np.ndarray,
	*,
	expected_shape: tuple[int, int],
) -> np.ndarray:
	"""Validate, own, and freeze one particle-energy history."""
	result = np.array(values, dtype=float, copy=True)
	if result.shape != expected_shape or not np.all(np.isfinite(result)):
		raise ValueError("Hamiltonian histories must share one finite particle-time grid.")
	result.setflags(write=False)
	return result


def _time_integrated_particle_rms(values: np.ndarray, times: np.ndarray) -> float:
	"""Return the time-normalized L2 norm of one particle-observable history."""
	return float(
		np.sqrt(
			np.trapz(np.mean(np.asarray(values, dtype=float) ** 2, axis=0), times)
			/ float(times[-1] - times[0])
		)
	)


@dataclass(frozen=True, slots=True)
class ThreeMethodNewtonComparisonResult:
	"""Reference and aligned solutions for the three Newton methods."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	config: ThreeMethodNewtonComparisonConfig
	reference: AdaptiveReference
	solutions: Mapping[str, Solution]
	accuracy: Mapping[str, TrajectoryAccuracySeries]
	reference_energies: np.ndarray
	audit_energies: np.ndarray
	energy_accuracy: Mapping[str, EnergyAccuracySeries]
	runtime_samples: Mapping[str, np.ndarray]
	wall_runtime_seconds: float

	def __post_init__(self) -> None:
		"""Require stable method coverage and aligned physical trajectories."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, GCInitialConfiguration):
			raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
		if not isinstance(self.config, ThreeMethodNewtonComparisonConfig):
			raise TypeError("`config` must be ThreeMethodNewtonComparisonConfig.")
		if tuple(self.solutions) != THREE_METHOD_NEWTON_METHODS:
			raise ValueError("The result must contain all three methods in stable order.")
		if tuple(self.accuracy) != THREE_METHOD_NEWTON_METHODS:
			raise ValueError("The result must contain three aligned accuracy series.")
		if tuple(self.energy_accuracy) != THREE_METHOD_NEWTON_METHODS:
			raise ValueError("The result must contain three aligned energy series.")
		if tuple(self.runtime_samples) != THREE_METHOD_NEWTON_METHODS:
			raise ValueError("The result must contain runtime samples for three methods.")
		particle_count = self.reference.states.shape[0] // 2
		energy_shape = (particle_count, self.reference.times.size)
		reference_energies = _readonly_energy_history(
			self.reference_energies,
			expected_shape=energy_shape,
		)
		audit_energies = _readonly_energy_history(
			self.audit_energies,
			expected_shape=energy_shape,
		)
		for method_name in THREE_METHOD_NEWTON_METHODS:
			solution = self.solutions[method_name]
			if not isinstance(solution, Solution):
				raise TypeError("Every comparison value must be a Solution.")
			if solution.source is not self.initial_configuration:
				raise ValueError("All methods must share one initial configuration.")
			if not np.array_equal(solution.t, self.reference.times):
				raise ValueError("Every method must share the reference output grid.")
			if int(solution.diagnostics.get("step_count", -1)) != self.config.step_count:
				raise ValueError("Every method must use the common complete step.")
			if solution.diagnostics.get("nonlinear_solver", "newton") != "newton":
				raise ValueError("Every compared method must use Newton.")
			series = self.accuracy[method_name]
			if series.method_name != method_name:
				raise ValueError("Accuracy labels must match their numerical methods.")
			if series.distances.shape[1] != self.reference.times.size:
				raise ValueError("Accuracy series must share the saved-time grid.")
			energy_series = self.energy_accuracy[method_name]
			if not isinstance(energy_series, EnergyAccuracySeries):
				raise TypeError("Every energy comparison must be EnergyAccuracySeries.")
			if energy_series.method_name != method_name:
				raise ValueError("Energy labels must match their numerical methods.")
			if energy_series.errors.shape != energy_shape:
				raise ValueError("Energy errors must share the particle-time grid.")
			samples = readonly_runtime_samples(self.runtime_samples[method_name])
			if samples.size != self.config.timing_repeats:
				raise ValueError("Every method must contain all measured timing repeats.")
		if not np.isfinite(self.wall_runtime_seconds) or self.wall_runtime_seconds <= 0.0:
			raise ValueError("The complete study runtime must be positive and finite.")
		object.__setattr__(self, "solutions", MappingProxyType(dict(self.solutions)))
		object.__setattr__(self, "accuracy", MappingProxyType(dict(self.accuracy)))
		object.__setattr__(self, "reference_energies", reference_energies)
		object.__setattr__(self, "audit_energies", audit_energies)
		object.__setattr__(
			self,
			"energy_accuracy",
			MappingProxyType(dict(self.energy_accuracy)),
		)
		object.__setattr__(
			self,
			"runtime_samples",
			MappingProxyType(
				{
					name: readonly_runtime_samples(values)
					for name, values in self.runtime_samples.items()
				}
			),
		)
		object.__setattr__(self, "wall_runtime_seconds", float(self.wall_runtime_seconds))

	@property
	def effective_potential(self) -> Potential:
		"""Return the gyroaveraged potential used by all three methods."""
		return self.dynamics.effective_potential

	@property
	def reference_energy_errors(self) -> np.ndarray:
		"""Return the signed DOP853-minus-Radau Hamiltonian discrepancy."""
		return np.asarray(self.reference_energies - self.audit_energies, dtype=float)

	@property
	def reference_energy_scale(self) -> float:
		"""Return the global particle-time RMS DOP853 Hamiltonian scale."""
		return max(
			_time_integrated_particle_rms(
				self.reference_energies,
				self.reference.times,
			),
			float(np.finfo(float).eps),
		)

	@property
	def energy_reference_time_integrated_rms_floor(self) -> float:
		"""Return the integrated DOP853/Radau Hamiltonian discrepancy."""
		return _time_integrated_particle_rms(
			self.reference_energy_errors,
			self.reference.times,
		)

	@property
	def energy_reference_final_rms_floor(self) -> float:
		"""Return the final particle-RMS DOP853/Radau Hamiltonian discrepancy."""
		return float(np.sqrt(np.mean(self.reference_energy_errors[:, -1] ** 2)))

	@property
	def runtimes(self) -> Mapping[str, float]:
		"""Return the median full-integration runtime for every method."""
		return MappingProxyType(
			{
				name: float(np.median(self.runtime_samples[name]))
				for name in THREE_METHOD_NEWTON_METHODS
			}
		)

	@property
	def total_method_runtime_seconds(self) -> float:
		"""Return the sum of the three measured integration calls."""
		return float(sum(self.runtimes.values()))

	@property
	def total_study_runtime_seconds(self) -> float:
		"""Return integration plus DOP853 and Radau reference time."""
		return self.wall_runtime_seconds

	def summaries(self) -> tuple[ThreeMethodNewtonSummary, ...]:
		"""Reduce each trajectory, timing, and per-step Newton diagnostic."""
		times = self.reference.times
		duration = float(times[-1] - times[0])
		floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		energy_floor = max(
			self.energy_reference_time_integrated_rms_floor,
			float(np.finfo(float).tiny),
		)
		initial_state = self.initial_configuration.initial_state
		assert initial_state is not None
		trajectory_count = self.initial_configuration.layout.particle_count(initial_state)
		rows: list[ThreeMethodNewtonSummary] = []
		for method_name in THREE_METHOD_NEWTON_METHODS:
			solution = self.solutions[method_name]
			series = self.accuracy[method_name]
			energy_series = self.energy_accuracy[method_name]
			runtime_samples = self.runtime_samples[method_name]
			iterations = np.asarray(
				solution.diagnostics["nonlinear_iterations"], dtype=int
			)
			residual_evaluations = _residual_evaluations(solution)
			residuals = np.asarray(
				solution.diagnostics["nonlinear_residual_norms"], dtype=float
			)
			tolerances = np.asarray(
				solution.diagnostics["nonlinear_tolerances"], dtype=float
			)
			expected_shape = (self.config.step_count,)
			if any(
				value.shape != expected_shape
				for value in (iterations, residual_evaluations, residuals, tolerances)
			):
				raise ValueError("Newton diagnostics must align with every complete step.")
			time_rms = float(
				np.sqrt(np.trapz(series.rms_distance**2, times) / duration)
			)
			energy_time_rms = _time_integrated_particle_rms(
				energy_series.errors,
				times,
			)
			rows.append(
				ThreeMethodNewtonSummary(
					method_name=method_name,
					method_label=THREE_METHOD_NEWTON_LABELS[method_name],
					nonlinear_solver="Newton",
					trajectory_count=trajectory_count,
					step_count=self.config.step_count,
					global_rms_distance=float(np.sqrt(np.mean(series.distances**2))),
					time_integrated_rms_distance=time_rms,
					final_rms_distance=float(series.rms_distance[-1]),
					maximum_distance=float(np.max(series.distances)),
					reference_floor_ratio=time_rms / floor,
					time_integrated_rms_energy_error=energy_time_rms,
					relative_time_integrated_rms_energy_error=(
						energy_time_rms / self.reference_energy_scale
					),
					final_rms_energy_error=float(energy_series.rms_error[-1]),
					maximum_absolute_energy_error=float(
						np.max(np.abs(energy_series.errors))
					),
					energy_reference_floor_ratio=energy_time_rms / energy_floor,
					runtime_seconds=float(np.median(runtime_samples)),
					runtime_first_quartile_seconds=float(
						np.quantile(runtime_samples, 0.25)
					),
					runtime_third_quartile_seconds=float(
						np.quantile(runtime_samples, 0.75)
					),
					runtime_minimum_seconds=float(np.min(runtime_samples)),
					runtime_maximum_seconds=float(np.max(runtime_samples)),
					minimum_newton_iterations=int(np.min(iterations)),
					mean_newton_iterations=float(np.mean(iterations)),
					maximum_newton_iterations=int(np.max(iterations)),
					total_newton_iterations=int(np.sum(iterations)),
					mean_residual_evaluations=float(np.mean(residual_evaluations)),
					total_residual_evaluations=int(np.sum(residual_evaluations)),
					maximum_residual_to_tolerance=float(
						np.max(residuals / tolerances)
					),
				)
			)
		return tuple(rows)


def _method(
	method_name: str,
	config: ThreeMethodNewtonComparisonConfig,
) -> NumericalMethod:
	"""Build one method with the common analytic-Newton controls."""
	if method_name == "ABBA4ImplicitSingleProjection":
		return ABBA4ImplicitSingleProjection(
			projection_formulation="reduced_multiplier",
			state_extension="physical",
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	if method_name == "GaussLegendre4":
		return GaussLegendre4(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			newton_jacobian_relative_step=config.jacobian_relative_step,
			progress=config.progress,
		)
	if method_name == "BM4Implicit1":
		return BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			newton_jacobian_relative_step=config.jacobian_relative_step,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	raise ValueError(f"Unknown three-method comparison method {method_name!r}.")


def run_three_method_newton_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: ThreeMethodNewtonComparisonConfig,
) -> ThreeMethodNewtonComparisonResult:
	"""Run three aligned Newton integrations and an independently audited reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, ThreeMethodNewtonComparisonConfig):
		raise TypeError("`config` must be ThreeMethodNewtonComparisonConfig.")
	if (
		isinstance(potential, GC2DH5Potential)
		and config.distance_convention != "euclidean"
	):
		raise ValueError(
			"GC2DH5Potential comparisons require Euclidean trajectory distances."
		)
	study_started = perf_counter()
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	reference = build_adaptive_reference(
		dynamics,
		problem.initial_state,
		request.output_times,
		period=(
			potential.grid.period
			if config.distance_convention == "periodic"
			else None
		),
		distance_convention=config.distance_convention,
		relative_tolerance=config.reference_relative_tolerance,
		absolute_tolerance=config.reference_absolute_tolerance,
		maximum_step=config.reference_maximum_step,
		audit_relative_tolerance=config.audit_relative_tolerance,
		audit_absolute_tolerance=config.audit_absolute_tolerance,
		audit_maximum_step=config.audit_maximum_step,
	)
	particle_count = problem.initial_state.size // 2
	energy_shape = (particle_count, request.output_times.size)
	reference_energies = _readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.states),
		expected_shape=energy_shape,
	)
	audit_energies = _readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.audit_states),
		expected_shape=energy_shape,
	)
	for _ in range(config.timing_warmups):
		for method_name in THREE_METHOD_NEWTON_METHODS:
			simulate(problem, _method(method_name, config), request)
	solutions: dict[str, Solution] = {}
	runtime_values: dict[str, list[float]] = {
		name: [] for name in THREE_METHOD_NEWTON_METHODS
	}
	for repeat in range(config.timing_repeats):
		order = (
			THREE_METHOD_NEWTON_METHODS
			if repeat % 2 == 0
			else tuple(reversed(THREE_METHOD_NEWTON_METHODS))
		)
		for method_name in order:
			started = perf_counter()
			solutions[method_name] = simulate(
				problem,
				_method(method_name, config),
				request,
			)
			runtime_values[method_name].append(perf_counter() - started)
	accuracy_by_method: dict[str, TrajectoryAccuracySeries] = {}
	energy_accuracy_by_method: dict[str, EnergyAccuracySeries] = {}
	for method_name in THREE_METHOD_NEWTON_METHODS:
		solution = solutions[method_name]
		accuracy_by_method[method_name] = accuracy_series(
			method_name,
			solution.states,
			reference.states,
			period=(
				potential.grid.period
				if config.distance_convention == "periodic"
				else None
			),
			distance_convention=config.distance_convention,
		)
		method_energies = _readonly_energy_history(
			dynamics.hamiltonian(request.output_times, solution.states),
			expected_shape=energy_shape,
		)
		energy_accuracy_by_method[method_name] = EnergyAccuracySeries(
			method_name=method_name,
			errors=method_energies - reference_energies,
		)
	return ThreeMethodNewtonComparisonResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		config=config,
		reference=reference,
		solutions=solutions,
		accuracy=accuracy_by_method,
		reference_energies=reference_energies,
		audit_energies=audit_energies,
		energy_accuracy=energy_accuracy_by_method,
		runtime_samples={
			name: np.asarray(runtime_values[name], dtype=float)
			for name in THREE_METHOD_NEWTON_METHODS
		},
		wall_runtime_seconds=perf_counter() - study_started,
	)


__all__ = [
	"THREE_METHOD_NEWTON_LABELS",
	"THREE_METHOD_NEWTON_METHODS",
	"EnergyAccuracySeries",
	"ThreeMethodNewtonComparisonConfig",
	"ThreeMethodNewtonComparisonResult",
	"ThreeMethodNewtonSummary",
	"run_three_method_newton_comparison",
]
