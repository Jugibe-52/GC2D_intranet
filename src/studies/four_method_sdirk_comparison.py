"""Long-time comparison of three geometric methods and fourth-order SDIRK."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from contracts.problem import InitialValueProblem
from methods.base import NumericalMethod
from methods.classical.sdirk import SDIRK4
from contracts.request import SimulationRequest
from solution import Solution
from simulation.runner import simulate

from ._gauss_legendre4_common import build_adaptive_reference, readonly_runtime_samples
from ._trajectory_accuracy import TrajectoryAccuracySeries, accuracy_series
from .three_method_newton_comparison import (
	EnergyAccuracySeries,
	ThreeMethodNewtonComparisonConfig,
	ThreeMethodNewtonComparisonResult,
	ThreeMethodNewtonSummary,
	_method as _baseline_method,
	_readonly_energy_history,
	_residual_evaluations,
	_time_integrated_particle_rms,
)


FOUR_METHOD_SDIRK_METHODS: tuple[str, ...] = (
	"ABBA4Implicit",
	"GaussLegendre4",
	"BM4Implicit",
	"SDIRK4",
)
FOUR_METHOD_SDIRK_LABELS: Mapping[str, str] = MappingProxyType(
	{
		"ABBA4Implicit": "Single-projection implicit ABBA4",
		"GaussLegendre4": "Gauss--Legendre (2 stages, order 4)",
		"BM4Implicit": "Single-projection implicit BM4",
		"SDIRK4": "SDIRK S54b (5 stages, order 4)",
	}
)


@dataclass(frozen=True, slots=True)
class FourMethodSDIRKComparisonConfig(ThreeMethodNewtonComparisonConfig):
	"""Common grid, Newton tolerances, references, and timing controls."""


@dataclass(frozen=True, slots=True)
class FourMethodSDIRKSummary(ThreeMethodNewtonSummary):
	"""Accuracy, runtime, and nonlinear work for one of the four methods."""


@dataclass(frozen=True, slots=True)
class FourMethodSDIRKComparisonResult(ThreeMethodNewtonComparisonResult):
	"""Audited reference and four aligned fourth-order implicit solutions."""

	config: FourMethodSDIRKComparisonConfig

	def __post_init__(self) -> None:
		"""Require stable method coverage and aligned physical trajectories."""
		if not isinstance(self.potential, Potential):
			raise TypeError("`potential` must be a Potential instance.")
		if not isinstance(self.dynamics, GuidingCenterDynamics):
			raise TypeError("`dynamics` must be GuidingCenterDynamics.")
		if not isinstance(self.initial_configuration, GCInitialConfiguration):
			raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
		if not isinstance(self.config, FourMethodSDIRKComparisonConfig):
			raise TypeError("`config` must be FourMethodSDIRKComparisonConfig.")
		if tuple(self.solutions) != FOUR_METHOD_SDIRK_METHODS:
			raise ValueError("The result must contain all four methods in stable order.")
		if tuple(self.accuracy) != FOUR_METHOD_SDIRK_METHODS:
			raise ValueError("The result must contain four aligned accuracy series.")
		if tuple(self.energy_accuracy) != FOUR_METHOD_SDIRK_METHODS:
			raise ValueError("The result must contain four aligned energy series.")
		if tuple(self.runtime_samples) != FOUR_METHOD_SDIRK_METHODS:
			raise ValueError("The result must contain runtime samples for four methods.")
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
		for method_name in FOUR_METHOD_SDIRK_METHODS:
			solution = self.solutions[method_name]
			if not isinstance(solution, Solution):
				raise TypeError("Every comparison value must be a Solution.")
			if solution.source is not self.initial_configuration:
				raise ValueError("All methods must share one initial configuration.")
			if not np.array_equal(solution.t, self.reference.times):
				raise ValueError("Every method must share the reference output grid.")
			if int(solution.diagnostics.get("step_count", -1)) != self.config.step_count:
				raise ValueError("Every method must use the common complete step.")
			if solution.diagnostics.get("nonlinear_solver") != "newton":
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
	def runtimes(self) -> Mapping[str, float]:
		"""Return the median full-integration runtime for every method."""
		return MappingProxyType(
			{
				name: float(np.median(self.runtime_samples[name]))
				for name in FOUR_METHOD_SDIRK_METHODS
			}
		)

	@property
	def total_method_runtime_seconds(self) -> float:
		"""Return the sum of the four median integration times."""
		return float(sum(self.runtimes.values()))

	def summaries(self) -> tuple[FourMethodSDIRKSummary, ...]:
		"""Reduce trajectory, timing, energy, and per-step Newton diagnostics."""
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
		rows: list[FourMethodSDIRKSummary] = []
		for method_name in FOUR_METHOD_SDIRK_METHODS:
			solution = self.solutions[method_name]
			series = self.accuracy[method_name]
			energy_series = self.energy_accuracy[method_name]
			runtime_samples = self.runtime_samples[method_name]
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
				FourMethodSDIRKSummary(
					method_name=method_name,
					method_label=FOUR_METHOD_SDIRK_LABELS[method_name],
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
	config: FourMethodSDIRKComparisonConfig,
) -> NumericalMethod:
	"""Build one method with common analytic-Newton controls."""
	if method_name == "SDIRK4":
		return SDIRK4(
			newton_absolute_tolerance=config.absolute_tolerance,
			newton_relative_tolerance=config.relative_tolerance,
			newton_max_iterations=config.max_iterations,
			newton_jacobian_method="analytic",
			newton_jacobian_relative_step=config.jacobian_relative_step,
			progress=config.progress,
		)
	return _baseline_method(method_name, config)


def run_four_method_sdirk_comparison(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: FourMethodSDIRKComparisonConfig,
) -> FourMethodSDIRKComparisonResult:
	"""Run four aligned Newton integrations and two adaptive references."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, FourMethodSDIRKComparisonConfig):
		raise TypeError("`config` must be FourMethodSDIRKComparisonConfig.")
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
		for method_name in FOUR_METHOD_SDIRK_METHODS:
			simulate(problem, _method(method_name, config), request)
	solutions: dict[str, Solution] = {}
	runtime_values: dict[str, list[float]] = {
		name: [] for name in FOUR_METHOD_SDIRK_METHODS
	}
	for repeat in range(config.timing_repeats):
		order = (
			FOUR_METHOD_SDIRK_METHODS
			if repeat % 2 == 0
			else tuple(reversed(FOUR_METHOD_SDIRK_METHODS))
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
	for method_name in FOUR_METHOD_SDIRK_METHODS:
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
	return FourMethodSDIRKComparisonResult(
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
			for name in FOUR_METHOD_SDIRK_METHODS
		},
		wall_runtime_seconds=perf_counter() - study_started,
	)


__all__ = [
	"FOUR_METHOD_SDIRK_LABELS",
	"FOUR_METHOD_SDIRK_METHODS",
	"FourMethodSDIRKComparisonConfig",
	"FourMethodSDIRKComparisonResult",
	"FourMethodSDIRKSummary",
	"run_four_method_sdirk_comparison",
]
