"""Shared physical guiding-centre symplecticity study infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, TypeVar

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import Area
from potential import Potential
from contracts.problem import InitialValueProblem
from contracts.observation import IntegrationStep, StepObserver
from methods.base import NumericalMethod
from contracts.request import SimulationRequest
from solution import Solution
from simulation.runner import simulate
from diagnostics.symplecticity import (
	GCAreaSymplecticityObserver,
	GCAreaSymplecticityRecord,
	StepJacobianMethod,
)

from ._gc_symplecticity_models import GCSymplecticityConfig, GCSymplecticitySummary
from ._validation import integer_ratio, resolve_rho
from .area_comparison import AreaStep


_SOLVER_DIAGNOSTICS = (
	"newton_iterations",
	"newton_residual_norms",
	"projection_multiplier_norms",
)


@dataclass(frozen=True, slots=True)
class _SolverSummary:
	"""Validated aggregate statistics for a nonlinear projection solve."""

	max_iterations: int
	mean_iterations: float
	max_residual_norm: float
	max_multiplier_norm: float


@dataclass(slots=True)
class _TimedStepObserver:
	"""Measure diagnostic callback and final persistence time."""

	observer: GCAreaSymplecticityObserver
	elapsed_seconds: float = 0.0

	def __enter__(self) -> _TimedStepObserver:
		"""Open the wrapped observer and return this callable timer."""
		self.observer.__enter__()
		return self

	def __exit__(self, *exception: object) -> None:
		"""Include final record construction and disk flushing in the timing."""
		started = perf_counter()
		try:
			self.observer.__exit__(*exception)
		finally:
			self.elapsed_seconds += perf_counter() - started

	def __call__(self, step: IntegrationStep) -> None:
		"""Forward one step and accumulate time spent inside the observer."""
		started = perf_counter()
		try:
			self.observer(step)
		finally:
			self.elapsed_seconds += perf_counter() - started


@dataclass(frozen=True, slots=True)
class _StepObserverFanout:
	"""Deliver each implicit step to several independent diagnostics."""

	observers: tuple[StepObserver, ...]

	def __call__(self, step: IntegrationStep) -> None:
		"""Forward the same step event in configured observer order."""
		for observer in self.observers:
			observer(step)


@dataclass(frozen=True, slots=True)
class GCSymplecticityResult:
	"""GC solutions, physical-flow Jacobians and diagnostic summaries."""

	dynamics: GuidingCenterDynamics
	area: Area
	steps: tuple[AreaStep, ...]
	solutions: Mapping[str, Solution]
	records: Mapping[str, tuple[GCAreaSymplecticityRecord, ...]]
	output_directories: Mapping[str, Path]
	simulation_runtime_seconds: Mapping[str, float]
	symplecticity_runtime_seconds: Mapping[str, float]
	jacobian_method: StepJacobianMethod = "finite_difference"

	method_name: ClassVar[str] = "GC numerical method"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = GCSymplecticitySummary

	@property
	def diagnostic_times(self) -> Mapping[str, np.ndarray]:
		"""Observation times aligned with the labeled GC solutions."""
		return {
			label: np.asarray([record.time for record in records])
			for label, records in self.records.items()
		}

	@property
	def relative_symplecticity_errors(self) -> Mapping[str, np.ndarray]:
		"""Accumulated physical-flow symplecticity defects."""
		return {
			label: np.asarray([record.relative_defect for record in records])
			for label, records in self.records.items()
		}

	def summaries(self) -> tuple[GCSymplecticitySummary, ...]:
		"""Return maximum diagnostics in configured step order."""
		rows: list[GCSymplecticitySummary] = []
		for step in self.steps:
			label = step.label
			solution = self.solutions[label]
			records = self.records[label]
			solver = _solver_summary(solution)
			rows.append(
				self.summary_type(
					label=label,
					step=step.value,
					step_count=int(solution.diagnostics["step_count"]),
					max_area_error=max(
						abs(record.relative_area_error) for record in records
					),
					max_local_defect=max(
						record.local_relative_defect for record in records
					),
					max_flow_defect=max(record.relative_defect for record in records),
					max_determinant_error=max(
						record.determinant_error for record in records
					),
					max_newton_iterations=(
						None if solver is None else solver.max_iterations
					),
					mean_newton_iterations=(
						None if solver is None else solver.mean_iterations
					),
					max_newton_residual_norm=(
						None if solver is None else solver.max_residual_norm
					),
					max_projection_multiplier_norm=(
						None if solver is None else solver.max_multiplier_norm
					),
				)
			)
		return tuple(rows)

	def print_summary(self) -> None:
		"""Print physical errors and optional nonlinear-solver statistics."""
		summaries = self.summaries()
		print(
			f"{'step':>22} {(self.method_name + ' steps'):>28} "
			f"{'max |area error|':>20} "
			f"{'max local defect':>20} {'max flow defect':>20} "
			f"{'max |det-1|':>16}"
		)
		for row in summaries:
			print(
				f"{row.label:>22} {row.step_count:28d} "
				f"{row.max_area_error:20.8e} {row.max_local_defect:20.8e} "
				f"{row.max_flow_defect:20.8e} "
				f"{row.max_determinant_error:16.8e}"
			)

		solver_rows = tuple(_summary_solver_values(row) for row in summaries)
		if all(values is None for values in solver_rows):
			return
		if any(values is None for values in solver_rows):
			raise ValueError(
				"Nonlinear-solver summaries must be available for every step or none."
			)
		print("\nNonlinear projection solve per main integration step:")
		for row, values in zip(summaries, solver_rows, strict=True):
			assert values is not None
			max_iterations, mean_iterations, max_residual, max_multiplier = values
			print(
				f"  {row.label}: max/mean iterations="
				f"{max_iterations}/{mean_iterations:.3f}, "
				f"max residual={max_residual:.3e}, "
				f"max |mu|_inf={max_multiplier:.3e}"
			)


def _optional_diagnostic(solution: Solution, name: str) -> np.ndarray | None:
	"""Return one non-empty finite numerical diagnostic when available."""
	value = solution.diagnostics.get(name)
	if value is None:
		return None
	result = np.asarray(value)
	if result.size == 0:
		return None
	if result.ndim != 1:
		raise ValueError(f"The `{name}` diagnostic must be one-dimensional.")
	if not np.all(np.isfinite(result)):
		raise ValueError(f"The `{name}` diagnostic contains non-finite values.")
	return result


def _solver_summary(solution: Solution) -> _SolverSummary | None:
	"""Validate all-or-none Newton diagnostics and aggregate their values."""
	arrays = tuple(_optional_diagnostic(solution, name) for name in _SOLVER_DIAGNOSTICS)
	if all(values is None for values in arrays):
		return None
	if any(values is None for values in arrays):
		raise ValueError(
			"Newton diagnostics must provide iterations, residuals and multipliers "
			"together."
		)
	iterations, residuals, multipliers = arrays
	assert iterations is not None
	assert residuals is not None
	assert multipliers is not None
	if not (iterations.size == residuals.size == multipliers.size):
		raise ValueError("Newton diagnostic arrays must have equal lengths.")
	step_count = int(solution.diagnostics.get("step_count", 0))
	if iterations.size != step_count:
		raise ValueError("Newton diagnostic arrays must contain one value per step.")
	if not np.all(np.equal(iterations, np.floor(iterations))) or np.any(iterations < 0):
		raise ValueError("`newton_iterations` must contain non-negative integers.")
	if np.any(residuals < 0) or np.any(multipliers < 0):
		raise ValueError("Newton residual and multiplier norms must be non-negative.")
	return _SolverSummary(
		max_iterations=int(np.max(iterations)),
		mean_iterations=float(np.mean(iterations)),
		max_residual_norm=float(np.max(residuals)),
		max_multiplier_norm=float(np.max(multipliers)),
	)


def _summary_solver_values(
	row: GCSymplecticitySummary,
) -> tuple[int, float, float, float] | None:
	"""Validate and return the optional solver fields of one summary row."""
	values = (
		row.max_newton_iterations,
		row.mean_newton_iterations,
		row.max_newton_residual_norm,
		row.max_projection_multiplier_norm,
	)
	if all(value is None for value in values):
		return None
	if any(value is None for value in values):
		raise ValueError("A solver summary must provide all Newton statistics together.")
	max_iterations, mean_iterations, max_residual, max_multiplier = values
	assert max_iterations is not None
	assert mean_iterations is not None
	assert max_residual is not None
	assert max_multiplier is not None
	return max_iterations, mean_iterations, max_residual, max_multiplier


_MethodFactory = Callable[[StepObserver], NumericalMethod]
_ResultT = TypeVar("_ResultT", bound=GCSymplecticityResult)


def _run_gc_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: GCSymplecticityConfig,
	method_factory: _MethodFactory,
	result_type: type[_ResultT],
	project_root: str | Path | None,
	metadata: Mapping[str, Any] | None,
	jacobian_method: StepJacobianMethod = "finite_difference",
) -> _ResultT:
	"""Run synchronized physical GC diagnostics for a step-observable method."""
	_validate_gc_study_inputs(potential, area, config)
	method_name = result_type.method_name
	if not isinstance(method_name, str) or not method_name.strip():
		raise ValueError("The result type must define a non-empty method name.")

	rho = resolve_rho(config.rho)
	dynamics = GuidingCenterDynamics(potential, rho=rho)
	problem = InitialValueProblem(dynamics, area)
	initial_state = area.initial_state
	assert initial_state is not None
	common_metadata = {
		**dict(metadata or {}),
		"method": method_name,
		"geometry": area.shape,
		"particle_count": area.layout.particle_count(initial_state),
		"rho": rho,
	}
	solutions: dict[str, Solution] = {}
	records_by_label: dict[str, tuple[GCAreaSymplecticityRecord, ...]] = {}
	output_directories: dict[str, Path] = {}
	simulation_runtimes: dict[str, float] = {}
	symplecticity_runtimes: dict[str, float] = {}

	for step in config.steps:
		record_every = integer_ratio(
			config.save_interval,
			step.value,
			f"save_interval / step for {step.label}",
		)
		step_tag = f"{step.value:.8f}".replace(".", "p")
		observer = GCAreaSymplecticityObserver(
			notebook_path=notebook_path,
			area=area,
			period=potential.grid.period,
			project_root=project_root,
			block_name=f"{config.block_prefix}_step_{step_tag}",
			record_every=record_every,
			chunk_size=config.chunk_size,
			relative_step=(
				config.finite_difference_relative_step
				if jacobian_method == "finite_difference"
				else None
			),
			jacobian_method=jacobian_method,
			verbose=False,
			metadata={
				**common_metadata,
				"integration_step": step.value,
			},
		)
		with _TimedStepObserver(observer) as timed_observer:
			request = SimulationRequest.uniform(
				t_span=config.t_span,
				max_step=step.value,
				sample_count=config.output_sample_count,
			)
			started = perf_counter()
			solution = simulate(
				problem,
				method_factory(timed_observer),
				request,
			)
			total_simulation_time = perf_counter() - started
			method_runtime = total_simulation_time - timed_observer.elapsed_seconds
			if method_runtime <= 0:
				raise RuntimeError(
					"Measured simulation time outside the symplecticity observer "
					"must be positive."
				)
		solutions[step.label] = solution
		records_by_label[step.label] = observer.records
		output_directories[step.label] = observer.output_directory
		simulation_runtimes[step.label] = method_runtime
		symplecticity_runtimes[step.label] = timed_observer.elapsed_seconds

	return result_type(
		dynamics=dynamics,
		area=area,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		records=MappingProxyType(records_by_label),
		output_directories=MappingProxyType(output_directories),
		simulation_runtime_seconds=MappingProxyType(simulation_runtimes),
		symplecticity_runtime_seconds=MappingProxyType(symplecticity_runtimes),
		jacobian_method=jacobian_method,
	)


def _run_gc_symplecticity_observers(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: GCSymplecticityConfig,
	method_factory: _MethodFactory,
	result_type: type[_ResultT],
	project_root: str | Path | None,
	metadata: Mapping[str, Any] | None,
	jacobian_methods: Mapping[str, StepJacobianMethod],
) -> Mapping[str, _ResultT]:
	"""Run one trajectory per step while fan-out observers use distinct Jacobians."""
	_validate_gc_study_inputs(potential, area, config)
	methods = dict(jacobian_methods)
	if not methods:
		raise ValueError("`jacobian_methods` must configure at least one observer.")
	if any(not label or not label.strip() for label in methods):
		raise ValueError("Jacobian observer labels must be non-empty strings.")
	method_name = result_type.method_name
	if not isinstance(method_name, str) or not method_name.strip():
		raise ValueError("The result type must define a non-empty method name.")

	rho = resolve_rho(config.rho)
	dynamics = GuidingCenterDynamics(potential, rho=rho)
	problem = InitialValueProblem(dynamics, area)
	initial_state = area.initial_state
	assert initial_state is not None
	common_metadata = {
		**dict(metadata or {}),
		"method": method_name,
		"geometry": area.shape,
		"particle_count": area.layout.particle_count(initial_state),
		"rho": rho,
	}
	solutions: dict[str, Solution] = {}
	simulation_runtimes: dict[str, float] = {}
	records: dict[
		str,
		dict[str, tuple[GCAreaSymplecticityRecord, ...]],
	] = {label: {} for label in methods}
	output_directories: dict[str, dict[str, Path]] = {
		label: {} for label in methods
	}
	observer_runtimes: dict[str, dict[str, float]] = {
		label: {} for label in methods
	}

	for step in config.steps:
		record_every = integer_ratio(
			config.save_interval,
			step.value,
			f"save_interval / step for {step.label}",
		)
		step_tag = f"{step.value:.8f}".replace(".", "p")
		observers: dict[str, GCAreaSymplecticityObserver] = {}
		timed_observers: dict[str, _TimedStepObserver] = {}
		with ExitStack() as stack:
			for label, jacobian_method in methods.items():
				observer = GCAreaSymplecticityObserver(
					notebook_path=notebook_path,
					area=area,
					period=potential.grid.period,
					project_root=project_root,
					block_name=f"{config.block_prefix}_{label}_step_{step_tag}",
					record_every=record_every,
					chunk_size=config.chunk_size,
					relative_step=(
						config.finite_difference_relative_step
						if jacobian_method == "finite_difference"
						else None
					),
					jacobian_method=jacobian_method,
					verbose=False,
					metadata={
						**common_metadata,
						"integration_step": step.value,
						"observer_label": label,
						"step_jacobian_method": jacobian_method,
						"step_jacobian_scope": (
							"emitted_finite_tolerance_solver_map"
							if jacobian_method == "finite_difference"
							else "ideal_converged_projection_root"
						),
					},
				)
				observers[label] = observer
				timed_observers[label] = stack.enter_context(
					_TimedStepObserver(observer)
				)

			request = SimulationRequest.uniform(
				t_span=config.t_span,
				max_step=step.value,
				sample_count=config.output_sample_count,
			)
			fanout = _StepObserverFanout(tuple(timed_observers.values()))
			started = perf_counter()
			solution = simulate(problem, method_factory(fanout), request)
			total_simulation_time = perf_counter() - started
			observer_time = sum(
				timed.elapsed_seconds for timed in timed_observers.values()
			)
			method_runtime = total_simulation_time - observer_time
			if method_runtime <= 0:
				raise RuntimeError(
					"Measured simulation time outside the symplecticity observers "
					"must be positive."
				)

		solutions[step.label] = solution
		simulation_runtimes[step.label] = method_runtime
		for label, observer in observers.items():
			records[label][step.label] = observer.records
			output_directories[label][step.label] = observer.output_directory
			observer_runtimes[label][step.label] = timed_observers[label].elapsed_seconds

	shared_solutions = MappingProxyType(solutions)
	shared_simulation_runtimes = MappingProxyType(simulation_runtimes)
	results = {
		label: result_type(
			dynamics=dynamics,
			area=area,
			steps=config.steps,
			solutions=shared_solutions,
			records=MappingProxyType(records[label]),
			output_directories=MappingProxyType(output_directories[label]),
			simulation_runtime_seconds=shared_simulation_runtimes,
			symplecticity_runtime_seconds=MappingProxyType(observer_runtimes[label]),
			jacobian_method=jacobian_method,
		)
		for label, jacobian_method in methods.items()
	}
	return MappingProxyType(results)


def _validate_gc_study_inputs(
	potential: Potential, area: Area, config: GCSymplecticityConfig,
) -> None:
	"""Check the shared physical, geometric, and study contracts of both runners."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(area, Area):
		raise TypeError("`area` must be an Area instance.")
	if not isinstance(config, GCSymplecticityConfig):
		raise TypeError("`config` must be a GCSymplecticityConfig instance.")


__all__: list[str] = []
