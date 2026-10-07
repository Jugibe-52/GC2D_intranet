"""Long-time comparison of four implicit methods and classical explicit RK4."""

from __future__ import annotations

from ._comparison import (
	COMPARISON_LABELS, ComparisonConfig, ComparisonResult, comparison_method,
	readonly_energy_history, accuracy_summaries, nonlinear_work_summaries,
)
from contracts.comparison import EnergyAccuracySeries

from contracts.comparison import (ExecutionLogEntry, FiveMethodComparisonSummary, FiveMethodNonlinearWorkSummary,)

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sys
from threading import Lock
from time import perf_counter
from types import MappingProxyType
from typing import ClassVar, Mapping

import numpy as np


from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from solution import Solution
from simulation.runner import simulate

from contracts.comparison import AdaptiveReference
from ._gauss_legendre4_common import (build_adaptive_reference, build_dop853_reference_with_reused_audit)
from contracts.comparison import TrajectoryAccuracySeries
from ._trajectory_accuracy import (accuracy_series)


FIVE_METHOD_COMPARISON_METHODS: tuple[str, ...] = (
	"ABBA4Implicit",
	"GaussLegendre4",
	"BM4Implicit",
	"SDIRK4",
	"RK4",
)
FIVE_METHOD_IMPLICIT_METHODS = FIVE_METHOD_COMPARISON_METHODS[:-1]
FIVE_METHOD_COMPARISON_LABELS: Mapping[str, str] = MappingProxyType(
	{name: COMPARISON_LABELS[name] for name in FIVE_METHOD_COMPARISON_METHODS}
)


@dataclass(frozen=True, slots=True)
class FiveMethodComparisonConfig(ComparisonConfig):
	"""Common physical grid, references, timings, and progress controls."""


@dataclass(frozen=True, slots=True)
class FiveMethodComparisonResult(ComparisonResult):
	"""Selected comparison methods and the completion log for their campaign."""

	config: FiveMethodComparisonConfig
	execution_log: tuple[ExecutionLogEntry, ...]
	config_type: ClassVar[type[ComparisonConfig]] = FiveMethodComparisonConfig

	def __post_init__(self) -> None:
		"""Add validated campaign completion records to the shared result contract."""
		ComparisonResult.__post_init__(self)
		entries = tuple(self.execution_log)
		if not entries or any(not isinstance(entry, ExecutionLogEntry) for entry in entries):
			raise TypeError("`execution_log` must contain completion records.")
		object.__setattr__(self, "execution_log", entries)

	def summaries(self) -> tuple[FiveMethodComparisonSummary, ...]:
		"""Reduce common accuracy, energy, and timing metrics."""
		return accuracy_summaries(self)

	def nonlinear_work_summaries(self) -> tuple[FiveMethodNonlinearWorkSummary, ...]:
		"""Reduce Newton diagnostics for the selected implicit methods."""
		return nonlinear_work_summaries(self)


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
	reference_energies = readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.states),
		expected_shape=energy_shape,
	)
	audit_energies = readonly_energy_history(
		dynamics.hamiltonian(request.output_times, reference.audit_states),
		expected_shape=energy_shape,
	)

	total_method_runs = len(method_names) * (
		config.timing_warmups + config.timing_repeats
	)
	completed_method_runs = 0
	method_campaign_started = perf_counter()
	progress_lock = Lock()

	def executecomparison_method(
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
		solution = simulate(problem, comparison_method(method_name, config), request)
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
			executecomparison_method(
				method_name,
				phase="warmup",
				repeat=warmup,
				repeat_count=config.timing_warmups,
			)
		latest_solution: Solution | None = None
		runtimes: list[float] = []
		for repeat in range(1, config.timing_repeats + 1):
			latest_solution, runtime_seconds = executecomparison_method(
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
		method_energies = readonly_energy_history(
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
