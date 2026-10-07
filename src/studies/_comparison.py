"""Shared operations for the three-, four-, and five-method comparison families."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from types import MappingProxyType
from typing import ClassVar, Mapping, TypeVar
import numpy as np

from contracts.comparison import (
	AdaptiveReference, EnergyAccuracySeries, TrajectoryAccuracySeries,
	FiveMethodComparisonSummary, FiveMethodNonlinearWorkSummary, ThreeMethodNewtonSummary,
)
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics.gc import GuidingCenterDynamics
from initial_conditions.gc import GCInitialConfiguration
from methods.base import NumericalMethod
from methods.extended.abba import ABBA4Implicit
from methods.extended.bm4 import BM4Implicit
from methods.classical.gauss_legendre import GaussLegendre4
from methods.classical.sdirk import SDIRK4
from methods.classical.rk4 import RK4
from potential.potential import Potential
from solution import Solution
from simulation.runner import simulate
from ._comparison_validation import comparison_runtime_samples, validate_comparison_solution, validate_trajectory_series, validate_energy_series
from ._trajectory_distances import DistanceConvention, normalized_distance_convention
from ._trajectory_accuracy import accuracy_series
from ._gauss_legendre4_common import build_adaptive_reference
from ._validation import finite_time_span, nonnegative_integer, integer_ratio, nonnegative_finite, positive_finite, positive_integer

COMPARISON_LABELS: Mapping[str, str] = MappingProxyType({
	"ABBA4Implicit": "Single-projection implicit ABBA4",
	"GaussLegendre4": "Gauss--Legendre (2 stages, order 4)",
	"BM4Implicit": "Single-projection implicit BM4",
	"SDIRK4": "SDIRK S54b (5 stages, order 4)",
	"RK4": "Classical explicit RK4",
})


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
	"""Common physical grid, Newton tolerances, and reference controls."""

	rho: float = 0.3
	coupling_frequency: float = 0.0
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
		object.__setattr__(
			self, "t_span",
			finite_time_span(self.t_span, message="`t_span` must contain two finite increasing times."),
		)
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
		object.__setattr__(self, "timing_warmups", nonnegative_integer(self.timing_warmups, "timing_warmups"))
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


def readonly_energy_history(
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


def time_integrated_particle_rms(values: np.ndarray, times: np.ndarray) -> float:
	"""Return the time-normalized L2 norm of one particle-observable history."""
	return float(
		np.sqrt(
			np.trapz(np.mean(np.asarray(values, dtype=float) ** 2, axis=0), times)
			/ float(times[-1] - times[0])
		)
	)


def solution_residual_evaluations(solution: Solution) -> np.ndarray:
	"""Read the canonical complete-step residual-work series."""
	return np.asarray(solution.diagnostics["residual_evaluations"], dtype=int)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
	"""Aligned comparison data and shared numerical invariants."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	config: ComparisonConfig
	reference: AdaptiveReference
	solutions: Mapping[str, Solution]
	accuracy: Mapping[str, TrajectoryAccuracySeries]
	reference_energies: np.ndarray
	audit_energies: np.ndarray
	energy_accuracy: Mapping[str, EnergyAccuracySeries]
	runtime_samples: Mapping[str, np.ndarray]
	wall_runtime_seconds: float

	config_type: ClassVar[type[ComparisonConfig]] = ComparisonConfig
	required_methods: ClassVar[tuple[str, ...] | None] = None
	allow_missing_solver: ClassVar[bool] = False

	def __post_init__(self) -> None:
		"""Validate alignment once for every comparison family and freeze owned data."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, GCInitialConfiguration):
			raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
		if not isinstance(self.config, self.config_type):
			raise TypeError(f"`config` must be {self.config_type.__name__}.")
		names = tuple(self.solutions)
		if not names or any(name not in COMPARISON_LABELS for name in names):
			raise ValueError("The result must contain supported numerical methods.")
		if self.required_methods is not None and names != self.required_methods:
			raise ValueError("The result must contain every configured method in stable order.")
		for values in (self.accuracy, self.energy_accuracy, self.runtime_samples):
			if tuple(values) != names:
				raise ValueError("Comparison histories must contain the same methods in stable order.")
		shape = (self.reference.states.shape[0] // 2, self.reference.times.size)
		reference_energies = readonly_energy_history(self.reference_energies, expected_shape=shape)
		audit_energies = readonly_energy_history(self.audit_energies, expected_shape=shape)
		runtimes: dict[str, np.ndarray] = {}
		for name, solution in self.solutions.items():
			validate_comparison_solution(solution, self.initial_configuration, self.reference.times, self.config.step_count)
			if name != "RK4":
				default = "newton" if self.allow_missing_solver else None
				if solution.diagnostics.get("nonlinear_solver", default) != "newton":
					raise ValueError("Every implicit method must use Newton.")
			elif "nonlinear_solver" in solution.diagnostics:
				raise ValueError("Classical RK4 must not report a nonlinear solver.")
			validate_trajectory_series(name, self.accuracy[name], self.reference.times.size)
			validate_energy_series(name, self.energy_accuracy[name], energy_type=EnergyAccuracySeries, energy_shape=shape)
			runtimes[name] = comparison_runtime_samples(self.runtime_samples[name], self.config.timing_repeats)
		if not np.isfinite(self.wall_runtime_seconds) or self.wall_runtime_seconds <= 0:
			raise ValueError("The complete study runtime must be positive and finite.")
		object.__setattr__(self, "solutions", MappingProxyType(dict(self.solutions)))
		object.__setattr__(self, "accuracy", MappingProxyType(dict(self.accuracy)))
		object.__setattr__(self, "energy_accuracy", MappingProxyType(dict(self.energy_accuracy)))
		object.__setattr__(self, "reference_energies", reference_energies)
		object.__setattr__(self, "audit_energies", audit_energies)
		object.__setattr__(self, "runtime_samples", MappingProxyType(runtimes))
		object.__setattr__(self, "wall_runtime_seconds", float(self.wall_runtime_seconds))

	@property
	def effective_potential(self) -> Potential:
		"""Return the gyroaveraged potential used by all compared methods."""
		return self.dynamics.effective_potential

	@property
	def reference_energy_errors(self) -> np.ndarray:
		"""Return the signed DOP853-minus-Radau Hamiltonian discrepancy."""
		return np.asarray(self.reference_energies - self.audit_energies, dtype=float)

	@property
	def reference_energy_scale(self) -> float:
		"""Return the global particle-time RMS DOP853 Hamiltonian scale."""
		return max(
			time_integrated_particle_rms(
				self.reference_energies,
				self.reference.times,
			),
			float(np.finfo(float).eps),
		)

	@property
	def energy_reference_time_integrated_rms_floor(self) -> float:
		"""Return the integrated DOP853/Radau Hamiltonian discrepancy."""
		return time_integrated_particle_rms(
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


def comparison_method(
	method_name: str,
	config: ComparisonConfig,
) -> NumericalMethod:
	"""Build one method with the common analytic-Newton controls."""
	if method_name == "ABBA4Implicit":
		return ABBA4Implicit(
			projection_placement="around_complete_composition",
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
	if method_name == "BM4Implicit":
		return BM4Implicit(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			newton_jacobian_relative_step=config.jacobian_relative_step,
			nonlinear_solver="newton",
			progress=config.progress,
		)
	if method_name == "SDIRK4":
		return SDIRK4(newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			newton_jacobian_relative_step=config.jacobian_relative_step,
			progress=config.progress)
	if method_name == "RK4":
		return RK4(progress=config.progress)
	raise ValueError(f"Unknown comparison method {method_name!r}.")


def accuracy_summaries(result: ComparisonResult) -> tuple[FiveMethodComparisonSummary, ...]:
	"""Reduce the common accuracy, energy, and timing metrics."""
	times = result.reference.times
	duration = float(times[-1] - times[0])
	floor = max(
		result.reference.time_integrated_rms_floor,
		float(np.finfo(float).eps),
	)
	energy_floor = max(
		result.energy_reference_time_integrated_rms_floor,
		float(np.finfo(float).tiny),
	)
	# Saved reference rows describe the completed campaign even if its source
	# configuration is later reused with different particles.
	trajectory_count = result.reference.states.shape[0] // 2
	rows: list[FiveMethodComparisonSummary] = []
	for method_name in result.solutions:
		series = result.accuracy[method_name]
		energy_series = result.energy_accuracy[method_name]
		runtime_samples = result.runtime_samples[method_name]
		time_rms = float(
			np.sqrt(np.trapz(series.rms_distance**2, times) / duration)
		)
		energy_time_rms = time_integrated_particle_rms(
			energy_series.errors,
			times,
		)
		rows.append(
			FiveMethodComparisonSummary(
				method_name=method_name,
				method_label=COMPARISON_LABELS[method_name],
				solver=("Newton" if method_name != "RK4" else "Explicit"),
				trajectory_count=trajectory_count,
				step_count=result.config.step_count,
				global_rms_distance=float(np.sqrt(np.mean(series.distances**2))),
				time_integrated_rms_distance=time_rms,
				final_rms_distance=float(series.rms_distance[-1]),
				maximum_distance=float(np.max(series.distances)),
				reference_floor_ratio=time_rms / floor,
				time_integrated_rms_energy_error=energy_time_rms,
				relative_time_integrated_rms_energy_error=(
					energy_time_rms / result.reference_energy_scale
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


def nonlinear_work_summaries(result: ComparisonResult) -> tuple[FiveMethodNonlinearWorkSummary, ...]:
	"""Reduce Newton diagnostics for the selected implicit methods."""
	rows: list[FiveMethodNonlinearWorkSummary] = []
	expected_shape = (result.config.step_count,)
	for method_name in result.solutions:
		if method_name == "RK4":
			continue
		solution = result.solutions[method_name]
		iterations = np.asarray(
			solution.diagnostics["nonlinear_iterations"],
			dtype=int,
		)
		residual_evaluations = solution_residual_evaluations(solution)
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
				method_label=COMPARISON_LABELS[method_name],
				step_count=result.config.step_count,
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


_ComparisonResultT = TypeVar("_ComparisonResultT", bound=ComparisonResult)


def run_alternating_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: ComparisonConfig,
	method_names: tuple[str, ...],
	result_type: type[_ComparisonResultT],
) -> _ComparisonResultT:
	"""Run three aligned Newton integrations and an independently audited reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, ComparisonConfig):
		raise TypeError("`config` must be ComparisonConfig.")
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
	reference_energies = readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.states),
		expected_shape=energy_shape,
	)
	audit_energies = readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.audit_states),
		expected_shape=energy_shape,
	)
	for _ in range(config.timing_warmups):
		for method_name in method_names:
			simulate(problem, comparison_method(method_name, config), request)
	solutions: dict[str, Solution] = {}
	runtime_values: dict[str, list[float]] = {
		name: [] for name in method_names
	}
	for repeat in range(config.timing_repeats):
		order = (
			method_names
			if repeat % 2 == 0
			else tuple(reversed(method_names))
		)
		for method_name in order:
			started = perf_counter()
			solutions[method_name] = simulate(
				problem,
				comparison_method(method_name, config),
				request,
			)
			runtime_values[method_name].append(perf_counter() - started)
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
		method_energies = readonly_energy_history(
			dynamics.hamiltonian(request.output_times, solution.states),
			expected_shape=energy_shape,
		)
		energy_accuracy_by_method[method_name] = EnergyAccuracySeries(
			method_name=method_name,
			errors=method_energies - reference_energies,
		)
	return result_type(
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
	)



_SummaryT = TypeVar("_SummaryT", bound=ThreeMethodNewtonSummary)


def implicit_summaries(
	result: ComparisonResult, summary_type: type[_SummaryT],
) -> tuple[_SummaryT, ...]:
	"""Join shared accuracy and work metrics into the implicit-family public record."""
	rows = []
	for accuracy, work in zip(accuracy_summaries(result), nonlinear_work_summaries(result), strict=True):
		values = asdict(accuracy)
		values["nonlinear_solver"] = values.pop("solver")
		values.update({name: value for name, value in asdict(work).items()
			if name not in {"method_name", "method_label", "step_count", "nonlinear_solves_per_step"}})
		rows.append(summary_type(**values))
	return tuple(rows)
