"""Long-time comparison of four implicit methods and classical explicit RK4."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sys
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	InitialValueProblem,
	NumericalMethod,
	RK4,
	SimulationRequest,
	Solution,
	simulate,
)

from ._gauss_legendre4_common import (
	AdaptiveReference,
	build_adaptive_reference,
	build_dop853_reference_with_reused_audit,
	readonly_runtime_samples,
)
from ._trajectory_accuracy import TrajectoryAccuracySeries, accuracy_series
from .four_method_sdirk_comparison import (
	FourMethodSDIRKComparisonConfig,
	_method as _implicit_method,
)
from .three_method_newton_comparison import (
	EnergyAccuracySeries,
	_readonly_energy_history,
	_residual_evaluations,
	_time_integrated_particle_rms,
)


FIVE_METHOD_COMPARISON_METHODS: tuple[str, ...] = (
	"ABBA4ImplicitSingleProjection",
	"GaussLegendre4",
	"BM4Implicit",
	"SDIRK4",
	"RK4",
)
FIVE_METHOD_IMPLICIT_METHODS = FIVE_METHOD_COMPARISON_METHODS[:-1]
FIVE_METHOD_COMPARISON_LABELS: Mapping[str, str] = MappingProxyType(
	{
		"ABBA4ImplicitSingleProjection": "Single-projection implicit ABBA4",
		"GaussLegendre4": "Gauss--Legendre (2 stages, order 4)",
		"BM4Implicit": "Single-projection implicit BM4",
		"SDIRK4": "SDIRK S54b (5 stages, order 4)",
		"RK4": "Classical explicit RK4",
	}
)


@dataclass(frozen=True, slots=True)
class FiveMethodComparisonConfig(FourMethodSDIRKComparisonConfig):
	"""Common physical grid, references, timings, and progress controls."""


@dataclass(frozen=True, slots=True)
class ExecutionLogEntry:
	"""One completed reference, warm-up, or timed execution."""

	phase: str
	method_name: str
	method_label: str
	repeat: int
	trajectory_count: int
	step_count: int
	runtime_seconds: float

	def __post_init__(self) -> None:
		"""Require one finite, self-describing completion record."""
		if self.phase not in {"reference", "warmup", "timing"}:
			raise ValueError("Unknown execution-log phase.")
		if not self.method_name or not self.method_label:
			raise ValueError("Execution-log method names must not be empty.")
		if self.repeat < 0 or self.trajectory_count < 1 or self.step_count < 0:
			raise ValueError("Execution-log counts must be non-negative and valid.")
		if not np.isfinite(self.runtime_seconds) or self.runtime_seconds <= 0.0:
			raise ValueError("Execution-log runtime must be positive and finite.")


@dataclass(frozen=True, slots=True)
class FiveMethodComparisonSummary:
	"""Accuracy, energy, and timing values shared by all five methods."""

	method_name: str
	method_label: str
	solver: str
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


@dataclass(frozen=True, slots=True)
class FiveMethodNonlinearWorkSummary:
	"""Newton work for one of the four implicit methods."""

	method_name: str
	method_label: str
	step_count: int
	nonlinear_solves_per_step: int
	minimum_newton_iterations: int
	mean_newton_iterations: float
	maximum_newton_iterations: int
	total_newton_iterations: int
	mean_residual_evaluations: float
	total_residual_evaluations: int
	maximum_residual_to_tolerance: float


@dataclass(frozen=True, slots=True)
class FiveMethodComparisonResult:
	"""Two references, selected aligned solutions, timings, and execution logs."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	config: FiveMethodComparisonConfig
	reference: AdaptiveReference
	solutions: Mapping[str, Solution]
	accuracy: Mapping[str, TrajectoryAccuracySeries]
	reference_energies: np.ndarray
	audit_energies: np.ndarray
	energy_accuracy: Mapping[str, EnergyAccuracySeries]
	runtime_samples: Mapping[str, np.ndarray]
	wall_runtime_seconds: float
	execution_log: tuple[ExecutionLogEntry, ...]

	def __post_init__(self) -> None:
		"""Require stable method coverage and aligned physical trajectories."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, GCInitialConfiguration):
			raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
		if not isinstance(self.config, FiveMethodComparisonConfig):
			raise TypeError("`config` must be FiveMethodComparisonConfig.")
		for values, description in (
			(self.solutions, "solutions"),
			(self.accuracy, "accuracy series"),
			(self.energy_accuracy, "energy series"),
			(self.runtime_samples, "runtime samples"),
		):
			if tuple(values) != tuple(self.solutions):
				raise ValueError(
					f"The result must contain matching {description} in stable order."
				)
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
		for method_name in self.solutions:
			solution = self.solutions[method_name]
			if not isinstance(solution, Solution):
				raise TypeError("Every comparison value must be a Solution.")
			if solution.source is not self.initial_configuration:
				raise ValueError("All methods must share one initial configuration.")
			if not np.array_equal(solution.t, self.reference.times):
				raise ValueError("Every method must share the reference output grid.")
			if int(solution.diagnostics.get("step_count", -1)) != self.config.step_count:
				raise ValueError("Every method must use the common complete step.")
			if method_name in FIVE_METHOD_IMPLICIT_METHODS:
				if solution.diagnostics.get("nonlinear_solver") != "newton":
					raise ValueError("Every implicit method must use Newton.")
			elif "nonlinear_solver" in solution.diagnostics:
				raise ValueError("Classical RK4 must not report a nonlinear solver.")
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
		log_values = tuple(self.execution_log)
		if not log_values or any(
			not isinstance(entry, ExecutionLogEntry) for entry in log_values
		):
			raise TypeError("`execution_log` must contain completion records.")
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
		object.__setattr__(self, "execution_log", log_values)
		object.__setattr__(self, "wall_runtime_seconds", float(self.wall_runtime_seconds))

	@property
	def effective_potential(self) -> Potential:
		"""Return the gyroaveraged potential used by all five methods."""
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
				for name in self.solutions
			}
		)

	@property
	def total_method_runtime_seconds(self) -> float:
		"""Return the sum of selected median integration times."""
		return float(sum(self.runtimes.values()))

	@property
	def total_study_runtime_seconds(self) -> float:
		"""Return integration plus DOP853 and Radau reference time."""
		return self.wall_runtime_seconds

	def summaries(self) -> tuple[FiveMethodComparisonSummary, ...]:
		"""Reduce the common accuracy, energy, and timing metrics."""
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
		rows: list[FiveMethodComparisonSummary] = []
		for method_name in self.solutions:
			series = self.accuracy[method_name]
			energy_series = self.energy_accuracy[method_name]
			runtime_samples = self.runtime_samples[method_name]
			time_rms = float(
				np.sqrt(np.trapz(series.rms_distance**2, times) / duration)
			)
			energy_time_rms = _time_integrated_particle_rms(
				energy_series.errors,
				times,
			)
			rows.append(
				FiveMethodComparisonSummary(
					method_name=method_name,
					method_label=FIVE_METHOD_COMPARISON_LABELS[method_name],
					solver=("Newton" if method_name in FIVE_METHOD_IMPLICIT_METHODS else "Explicit"),
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
				)
			)
		return tuple(rows)

	def nonlinear_work_summaries(
		self,
	) -> tuple[FiveMethodNonlinearWorkSummary, ...]:
		"""Reduce Newton diagnostics for the selected implicit methods."""
		rows: list[FiveMethodNonlinearWorkSummary] = []
		expected_shape = (self.config.step_count,)
		for method_name in self.solutions:
			if method_name not in FIVE_METHOD_IMPLICIT_METHODS:
				continue
			solution = self.solutions[method_name]
			iterations = np.asarray(
				solution.diagnostics["nonlinear_iterations"],
				dtype=int,
			)
			residual_evaluations = _residual_evaluations(solution)
			residuals = np.asarray(
				solution.diagnostics["nonlinear_residual_norms"],
				dtype=float,
			)
			tolerances = np.asarray(
				solution.diagnostics["nonlinear_tolerances"],
				dtype=float,
			)
			if any(
				value.shape != expected_shape
				for value in (iterations, residual_evaluations, residuals, tolerances)
			):
				raise ValueError("Newton diagnostics must align with every complete step.")
			rows.append(
				FiveMethodNonlinearWorkSummary(
					method_name=method_name,
					method_label=FIVE_METHOD_COMPARISON_LABELS[method_name],
					step_count=self.config.step_count,
					nonlinear_solves_per_step=int(
						solution.diagnostics.get("nonlinear_solves_per_step", 1)
					),
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
	config: FiveMethodComparisonConfig,
) -> NumericalMethod:
	"""Build one fixed-step method with the requested progress behavior."""
	if method_name == "RK4":
		return RK4(progress=config.progress)
	return _implicit_method(method_name, config)


def _print_progress(enabled: bool, message: str) -> None:
	"""Emit one immediately visible study-level progress record."""
	if enabled:
		print(f"[five-method study] {message}", file=sys.stderr, flush=True)


def run_five_method_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: FiveMethodComparisonConfig,
	method_names: tuple[str, ...] = FIVE_METHOD_COMPARISON_METHODS,
	reused_reference: AdaptiveReference | None = None,
	reused_audit_reference: AdaptiveReference | None = None,
	parallel_models: bool = False,
) -> FiveMethodComparisonResult:
	"""Run selected methods, optionally reusing references on the exact saved grid.

	The caller must verify physical parameters and potential provenance before
	passing a reused reference. Its times and initial states are checked here."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, FiveMethodComparisonConfig):
		raise TypeError("`config` must be FiveMethodComparisonConfig.")
	method_names = tuple(method_names)
	if not method_names or len(set(method_names)) != len(method_names) or any(
		name not in FIVE_METHOD_COMPARISON_METHODS for name in method_names
	):
		raise ValueError("Select distinct supported comparison methods.")
	study_started = perf_counter()
	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	request = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.integration_step,
		sample_count=config.output_sample_count,
	)
	trajectory_count = problem.particle_count
	execution_log: list[ExecutionLogEntry] = []

	if reused_reference is not None and reused_audit_reference is not None:
		raise ValueError("Reuse either the complete reference or only the Radau audit.")
	if reused_reference is None and reused_audit_reference is None:
		_print_progress(
			config.progress,
			"Starting DOP853 reference and tighter Radau audit for "
			f"{trajectory_count} trajectories on [{config.t_span[0]:g}, "
			f"{config.t_span[1]:g}].",
		)
		reference_started = perf_counter()
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
		reference_runtime = perf_counter() - reference_started
		execution_log.append(
			ExecutionLogEntry(
				phase="reference",
				method_name="DOP853+Radau",
				method_label="DOP853 reference + Radau audit",
				repeat=1,
				trajectory_count=trajectory_count,
				step_count=0,
				runtime_seconds=reference_runtime,
			)
		)
		_print_progress(
			config.progress,
			f"Completed both adaptive references in {reference_runtime:.2f} s.",
		)
	elif reused_reference is not None:
		reference = reused_reference
		if not np.array_equal(reference.times, request.output_times):
			raise ValueError("Reused reference must match the saved-time grid exactly.")
		for states in (reference.states, reference.audit_states):
			if states.shape != (problem.initial_state.size, request.output_times.size) or not np.all(np.isfinite(states)):
				raise ValueError("Reused reference has invalid states.")
			if not np.array_equal(states[:, 0], problem.initial_state):
				raise ValueError("Reused reference initial state differs.")
		_print_progress(config.progress, "Reusing saved DOP853 and Radau; no adaptive integration.")
	else:
		assert reused_audit_reference is not None
		_print_progress(
			config.progress,
			f"Recomputing DOP853 with maximum step {config.reference_maximum_step:g}; "
			"reusing the saved Radau audit.",
		)
		reference = build_dop853_reference_with_reused_audit(
			dynamics,
			problem.initial_state,
			request.output_times,
			audit_reference=reused_audit_reference,
			period=potential.grid.period if config.distance_convention == "periodic" else None,
			distance_convention=config.distance_convention,
			relative_tolerance=config.reference_relative_tolerance,
			absolute_tolerance=config.reference_absolute_tolerance,
			maximum_step=config.reference_maximum_step,
		)
		execution_log.append(
			ExecutionLogEntry(
				phase="reference",
				method_name="DOP853",
				method_label="DOP853 reference (saved Radau audit)",
				repeat=1,
				trajectory_count=trajectory_count,
				step_count=0,
				runtime_seconds=reference.dop853_runtime_seconds,
			)
		)

	energy_shape = (trajectory_count, request.output_times.size)
	reference_energies = _readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.states),
		expected_shape=energy_shape,
	)
	audit_energies = _readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.audit_states),
		expected_shape=energy_shape,
	)

	total_method_runs = len(method_names) * (
		config.timing_warmups + config.timing_repeats
	)
	completed_method_runs = 0
	method_campaign_started = perf_counter()
	progress_lock = Lock()

	def execute_method(
		method_name: str,
		*,
		phase: str,
		repeat: int,
		repeat_count: int,
	) -> tuple[Solution, float]:
		"""Execute and log one vectorized multi-trajectory integration."""
		nonlocal completed_method_runs
		label = FIVE_METHOD_COMPARISON_LABELS[method_name]
		_print_progress(
			config.progress,
			f"Starting {phase} {repeat}/{repeat_count}: {label}; "
			f"{trajectory_count} trajectories together, {config.step_count} steps.",
		)
		started = perf_counter()
		solution = simulate(problem, _method(method_name, config), request)
		runtime_seconds = perf_counter() - started
		with progress_lock:
			completed_method_runs += 1
			execution_log.append(
				ExecutionLogEntry(
					phase=phase,
					method_name=method_name,
					method_label=label,
					repeat=repeat,
					trajectory_count=trajectory_count,
					step_count=config.step_count,
					runtime_seconds=runtime_seconds,
				)
			)
			elapsed = perf_counter() - method_campaign_started
			remaining = total_method_runs - completed_method_runs
			eta = elapsed * remaining / completed_method_runs
			_print_progress(
				config.progress,
				f"Completed {phase} {repeat}/{repeat_count}: {label} in "
				f"{runtime_seconds:.2f} s; campaign "
				f"{completed_method_runs}/{total_method_runs} "
				f"({completed_method_runs / total_method_runs:.1%}), "
				f"elapsed {elapsed:.1f} s, ETA {eta:.1f} s.",
			)
		return solution, runtime_seconds

	def execute_model_campaign(method_name: str) -> tuple[Solution, list[float]]:
		"""Run all repetitions for one model in its dedicated worker."""
		for warmup in range(1, config.timing_warmups + 1):
			execute_method(
				method_name,
				phase="warmup",
				repeat=warmup,
				repeat_count=config.timing_warmups,
			)
		latest_solution: Solution | None = None
		runtimes: list[float] = []
		for repeat in range(1, config.timing_repeats + 1):
			latest_solution, runtime_seconds = execute_method(
				method_name,
				phase="timing",
				repeat=repeat,
				repeat_count=config.timing_repeats,
			)
			runtimes.append(runtime_seconds)
		assert latest_solution is not None
		return latest_solution, runtimes

	if parallel_models and len(method_names) > 1:
		_print_progress(config.progress, f"Running {len(method_names)} model campaigns in parallel.")
		with ThreadPoolExecutor(
			max_workers=len(method_names),
			thread_name_prefix="comparison-model",
		) as executor:
			campaigns = {
				name: executor.submit(execute_model_campaign, name)
				for name in method_names
			}
			campaign_results = {name: campaigns[name].result() for name in method_names}
	else:
		campaign_results = {
			name: execute_model_campaign(name)
			for name in method_names
		}
	solutions = {name: campaign_results[name][0] for name in method_names}
	runtime_values = {name: campaign_results[name][1] for name in method_names}

	accuracy_by_method: dict[str, TrajectoryAccuracySeries] = {}
	energy_accuracy_by_method: dict[str, EnergyAccuracySeries] = {}
	for method_name in method_names:
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
	_print_progress(
		config.progress,
		f"Completed study in {perf_counter() - study_started:.2f} s.",
	)
	return FiveMethodComparisonResult(
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
			for name in method_names
		},
		wall_runtime_seconds=perf_counter() - study_started,
		execution_log=tuple(execution_log),
	)


__all__ = [
	"FIVE_METHOD_COMPARISON_LABELS",
	"FIVE_METHOD_COMPARISON_METHODS",
	"FIVE_METHOD_IMPLICIT_METHODS",
	"ExecutionLogEntry",
	"FiveMethodComparisonConfig",
	"FiveMethodComparisonResult",
	"FiveMethodComparisonSummary",
	"FiveMethodNonlinearWorkSummary",
	"run_five_method_comparison",
]
