"""Step-refinement accuracy of tangent-Taylor ABBA methods and Euler."""

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
	ABBA4Implicit1TangentTaylor,
	ExplicitEuler,
	ImplicitABBA1TangentTaylor,
	InitialValueProblem,
	NONLINEAR_SOLVERS,
	NonlinearSolver,
	NumericalMethod,
	SimulationRequest,
	Solution,
	simulate,
)

from ._trajectory_accuracy import TrajectoryAccuracySeries, accuracy_series
from ._validation import nonnegative_finite, positive_finite, positive_integer
from .reference_trajectory import (
	AdaptiveReferenceSolveSummary,
	HighPrecisionReferenceConfig,
	ReferenceTrajectoryAuditSummary,
	_audit_summary,
	_output_times,
	_periodic_particle_distances,
	_solve_adaptive,
)


TANGENT_TAYLOR_EULER_METHOD_NAMES = (
	"ExplicitEuler",
	"ImplicitABBA1TangentTaylor",
	"ABBA4Implicit1TangentTaylor",
)


@dataclass(frozen=True, slots=True)
class TangentTaylorEulerAccuracyConfig:
	"""Physical, refinement, nonlinear, and certified-reference controls."""

	rho: float = 0.3
	t_span: tuple[float, float] = (0.0, 2.0 * np.pi)
	step_counts: tuple[int, ...] = (80, 160, 320, 640)
	newton_absolute_tolerance: float = 1e-14
	newton_relative_tolerance: float = 1e-14
	newton_max_iterations: int = 20
	nonlinear_solver: NonlinearSolver = "newton"
	reference_relative_tolerance: float = 1e-13
	reference_absolute_tolerance: float = 1e-15
	reference_maximum_step: float = 0.005
	audit_relative_tolerance: float = 1e-13
	audit_absolute_tolerance: float = 1e-15
	audit_maximum_step: float = 0.0025
	progress: bool = False

	def __post_init__(self) -> None:
		"""Normalize parameters and require nested complete-step grids."""
		span = np.asarray(self.t_span, dtype=float)
		if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
			raise ValueError("`t_span` must contain two finite, increasing times.")
		object.__setattr__(self, "t_span", (float(span[0]), float(span[1])))
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		counts = tuple(
			positive_integer(value, "step_counts") for value in self.step_counts
		)
		if len(counts) < 2 or any(
			coarse >= fine for coarse, fine in zip(counts, counts[1:])
		):
			raise ValueError("`step_counts` must increase from coarse to fine.")
		if any(count % counts[0] for count in counts[1:]):
			raise ValueError("Every refined grid must contain the coarsest saved grid.")
		for coarse, fine in zip(counts, counts[1:]):
			if fine % coarse:
				raise ValueError("Adjacent step grids must be nested integer refinements.")
		object.__setattr__(self, "step_counts", counts)
		for name in (
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
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
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)
		if self.nonlinear_solver not in NONLINEAR_SOLVERS:
			raise ValueError("Unknown nonlinear solver for the accuracy study.")
		if self.audit_maximum_step > self.reference_maximum_step:
			raise ValueError("The Radau audit step cannot exceed the DOP853 step.")
		if self.audit_relative_tolerance > self.reference_relative_tolerance:
			raise ValueError("The Radau audit relative tolerance cannot be looser.")
		if self.audit_absolute_tolerance > self.reference_absolute_tolerance:
			raise ValueError("The Radau audit absolute tolerance cannot be looser.")
		object.__setattr__(self, "progress", bool(self.progress))

	@property
	def integration_steps(self) -> tuple[float, ...]:
		"""Return exact complete-step sizes corresponding to ``step_counts``."""
		duration = self.t_span[1] - self.t_span[0]
		return tuple(duration / count for count in self.step_counts)

	@property
	def reference_config(self) -> HighPrecisionReferenceConfig:
		"""Build the DOP853/Radau configuration on the common coarse grid."""
		return HighPrecisionReferenceConfig(
			t_span=self.t_span,
			save_interval=self.integration_steps[0],
			rho=self.rho,
			relative_tolerance=self.reference_relative_tolerance,
			absolute_tolerance=self.reference_absolute_tolerance,
			maximum_step=self.reference_maximum_step,
			audit_relative_tolerance=self.audit_relative_tolerance,
			audit_absolute_tolerance=self.audit_absolute_tolerance,
			audit_maximum_step=self.audit_maximum_step,
		)


@dataclass(frozen=True, slots=True)
class TangentTaylorEulerReference:
	"""In-memory DOP853 reference and independent Radau audit."""

	times: np.ndarray
	states: np.ndarray
	audit_states: np.ndarray
	audit_periodic_distances: np.ndarray
	reference_solve: AdaptiveReferenceSolveSummary
	audit_solve: AdaptiveReferenceSolveSummary
	audit: ReferenceTrajectoryAuditSummary

	def __post_init__(self) -> None:
		"""Own immutable aligned reference arrays."""
		times = np.array(self.times, dtype=float, copy=True)
		states = np.array(self.states, dtype=float, copy=True)
		audit_states = np.array(self.audit_states, dtype=float, copy=True)
		distances = np.array(self.audit_periodic_distances, dtype=float, copy=True)
		if (
			times.ndim != 1
			or times.size < 2
			or np.any(np.diff(times) <= 0.0)
			or states.ndim != 2
			or states.shape != audit_states.shape
			or states.shape[1] != times.size
			or states.shape[0] % 2
			or distances.shape != (states.shape[0] // 2, times.size)
			or not all(
				np.all(np.isfinite(value))
				for value in (times, states, audit_states, distances)
			)
			or np.any(distances < 0.0)
		):
			raise ValueError("The in-memory reference arrays are inconsistent.")
		for value in (times, states, audit_states, distances):
			value.setflags(write=False)
		object.__setattr__(self, "times", times)
		object.__setattr__(self, "states", states)
		object.__setattr__(self, "audit_states", audit_states)
		object.__setattr__(self, "audit_periodic_distances", distances)

	@property
	def time_integrated_rms_floor(self) -> float:
		"""Return the DOP853/Radau RMS discrepancy over particles and time."""
		particle_rms_squared = np.mean(self.audit_periodic_distances**2, axis=0)
		return float(
			np.sqrt(
				np.trapz(particle_rms_squared, self.times)
				/ float(self.times[-1] - self.times[0])
			)
		)

	@property
	def final_rms_floor(self) -> float:
		"""Return the DOP853/Radau particle RMS discrepancy at final time."""
		return float(np.sqrt(np.mean(self.audit_periodic_distances[:, -1] ** 2)))


@dataclass(frozen=True, slots=True)
class TangentTaylorEulerAccuracyRun:
	"""One method trajectory and its periodic reference error."""

	method_name: str
	step_count: int
	integration_step: float
	solution: Solution
	accuracy: TrajectoryAccuracySeries
	runtime_seconds: float


@dataclass(frozen=True, slots=True)
class TangentTaylorEulerAccuracySummary:
	"""Scalar accuracy and runtime metrics for one method and step."""

	method_name: str
	step_count: int
	integration_step: float
	time_integrated_rms_distance: float
	global_rms_distance: float
	maximum_distance: float
	final_rms_distance: float
	final_maximum_distance: float
	runtime_seconds: float
	reference_floor_ratio: float


@dataclass(frozen=True, slots=True)
class TangentTaylorEulerAccuracyOrder:
	"""Observed error order between two adjacent nested refinements."""

	method_name: str
	coarse_step_count: int
	fine_step_count: int
	time_integrated_rms_order: float
	final_rms_order: float
	resolved_above_reference_floor: bool


@dataclass(frozen=True, slots=True)
class TangentTaylorEulerAccuracyResult:
	"""Certified-reference errors for all three methods on nested grids."""

	potential: Potential
	config: TangentTaylorEulerAccuracyConfig
	reference: TangentTaylorEulerReference
	runs: Mapping[int, Mapping[str, TangentTaylorEulerAccuracyRun]]

	def __post_init__(self) -> None:
		"""Require complete method coverage and exact common saved times."""
		if tuple(self.runs) != self.config.step_counts:
			raise ValueError("Accuracy runs must follow the configured step counts.")
		normalized: dict[int, Mapping[str, TangentTaylorEulerAccuracyRun]] = {}
		for step_count, runs in self.runs.items():
			if tuple(runs) != TANGENT_TAYLOR_EULER_METHOD_NAMES:
				raise ValueError("Every refinement must contain the three methods.")
			for method_name, run in runs.items():
				if (
					run.method_name != method_name
					or run.step_count != step_count
					or run.solution.n_steps != step_count
					or not np.array_equal(run.solution.t, self.reference.times)
					or run.accuracy.distances.shape[1] != self.reference.times.size
					or not np.isfinite(run.runtime_seconds)
					or run.runtime_seconds <= 0.0
				):
					raise ValueError("An accuracy run is inconsistent with its grid.")
			normalized[step_count] = MappingProxyType(dict(runs))
		object.__setattr__(self, "runs", MappingProxyType(normalized))

	@property
	def finest_runs(self) -> Mapping[str, TangentTaylorEulerAccuracyRun]:
		"""Return all methods at the smallest configured integration step."""
		return self.runs[self.config.step_counts[-1]]

	def summaries(self) -> tuple[TangentTaylorEulerAccuracySummary, ...]:
		"""Return scalar metrics in stable coarse-to-fine and method order."""
		rows: list[TangentTaylorEulerAccuracySummary] = []
		duration = float(self.reference.times[-1] - self.reference.times[0])
		floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		for step_count, step in zip(
			self.config.step_counts,
			self.config.integration_steps,
		):
			for method_name in TANGENT_TAYLOR_EULER_METHOD_NAMES:
				run = self.runs[step_count][method_name]
				series = run.accuracy
				time_rms = float(
					np.sqrt(np.trapz(series.rms_distance**2, self.reference.times) / duration)
				)
				rows.append(
					TangentTaylorEulerAccuracySummary(
						method_name=method_name,
						step_count=step_count,
						integration_step=step,
						time_integrated_rms_distance=time_rms,
						global_rms_distance=float(np.sqrt(np.mean(series.distances**2))),
						maximum_distance=float(np.max(series.distances)),
						final_rms_distance=float(series.rms_distance[-1]),
						final_maximum_distance=float(series.maximum_distance[-1]),
						runtime_seconds=run.runtime_seconds,
						reference_floor_ratio=time_rms / floor,
					)
				)
		return tuple(rows)

	def convergence_orders(self) -> tuple[TangentTaylorEulerAccuracyOrder, ...]:
		"""Estimate adjacent convergence orders when reference error is resolved."""
		summaries = {
			(row.step_count, row.method_name): row for row in self.summaries()
		}
		time_floor = max(
			self.reference.time_integrated_rms_floor,
			float(np.finfo(float).eps),
		)
		final_floor = max(
			self.reference.final_rms_floor,
			float(np.finfo(float).eps),
		)
		rows: list[TangentTaylorEulerAccuracyOrder] = []
		for coarse_count, fine_count in zip(
			self.config.step_counts,
			self.config.step_counts[1:],
		):
			step_ratio = fine_count / coarse_count
			for method_name in TANGENT_TAYLOR_EULER_METHOD_NAMES:
				coarse = summaries[(coarse_count, method_name)]
				fine = summaries[(fine_count, method_name)]
				resolved = (
					fine.time_integrated_rms_distance > 10.0 * time_floor
					and fine.final_rms_distance > 10.0 * final_floor
				)
				time_order = (
					float(
						np.log(
							coarse.time_integrated_rms_distance
							/ fine.time_integrated_rms_distance
						)
						/ np.log(step_ratio)
					)
					if resolved
					else float("nan")
				)
				final_order = (
					float(
						np.log(coarse.final_rms_distance / fine.final_rms_distance)
						/ np.log(step_ratio)
					)
					if resolved
					else float("nan")
				)
				rows.append(
					TangentTaylorEulerAccuracyOrder(
						method_name=method_name,
						coarse_step_count=coarse_count,
						fine_step_count=fine_count,
						time_integrated_rms_order=time_order,
						final_rms_order=final_order,
						resolved_above_reference_floor=resolved,
					)
				)
		return tuple(rows)

	def print_summary(self) -> None:
		"""Print reference certification and the finest-step comparison."""
		print(
			"DOP853/Radau time-integrated RMS floor: "
			f"{self.reference.time_integrated_rms_floor:.8e}"
		)
		print(f"DOP853/Radau final RMS floor: {self.reference.final_rms_floor:.8e}")
		finest_count = self.config.step_counts[-1]
		for row in self.summaries():
			if row.step_count == finest_count:
				print(
					f"{row.method_name}: h={row.integration_step:.8e}, "
					f"time RMS={row.time_integrated_rms_distance:.8e}, "
					f"final RMS={row.final_rms_distance:.8e}, "
					f"runtime={row.runtime_seconds:.6f} s"
				)


def run_tangent_taylor_euler_accuracy_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	config: TangentTaylorEulerAccuracyConfig,
) -> TangentTaylorEulerAccuracyResult:
	"""Run Euler and both tangent-Taylor methods against one audited reference."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be GCInitialConfiguration.")
	if not isinstance(config, TangentTaylorEulerAccuracyConfig):
		raise TypeError("`config` must be TangentTaylorEulerAccuracyConfig.")
	initial_state = initial_configuration.initial_state
	if initial_state is None:
		raise ValueError("The initial configuration has no state.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	reference_config = config.reference_config
	times = _output_times(reference_config)
	audit_states, audit_solve = _solve_adaptive(
		dynamics,
		initial_state,
		times,
		relative_tolerance=reference_config.audit_relative_tolerance,
		absolute_tolerance=reference_config.audit_absolute_tolerance,
		maximum_step=reference_config.audit_maximum_step,
		method="Radau",
	)
	reference_states, reference_solve = _solve_adaptive(
		dynamics,
		initial_state,
		times,
		relative_tolerance=reference_config.relative_tolerance,
		absolute_tolerance=reference_config.absolute_tolerance,
		maximum_step=reference_config.maximum_step,
		method="DOP853",
	)
	audit_distances = _periodic_particle_distances(
		reference_states,
		audit_states,
		period=potential.grid.period,
	)
	reference = TangentTaylorEulerReference(
		times=times,
		states=reference_states,
		audit_states=audit_states,
		audit_periodic_distances=audit_distances,
		reference_solve=reference_solve,
		audit_solve=audit_solve,
		audit=_audit_summary(audit_distances, reference_states, audit_states),
	)
	problem = InitialValueProblem(dynamics, initial_configuration)
	all_runs: dict[int, Mapping[str, TangentTaylorEulerAccuracyRun]] = {}
	for step_count, integration_step in zip(
		config.step_counts,
		config.integration_steps,
	):
		request = SimulationRequest(
			t_span=config.t_span,
			max_step=integration_step,
			output_times=times,
		)
		methods: tuple[NumericalMethod, ...] = (
			ExplicitEuler(progress=config.progress),
			ImplicitABBA1TangentTaylor(
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
			),
			ABBA4Implicit1TangentTaylor(
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				nonlinear_solver=config.nonlinear_solver,
				progress=config.progress,
			),
		)
		step_runs: dict[str, TangentTaylorEulerAccuracyRun] = {}
		for method in methods:
			method_name = type(method).__name__
			started = perf_counter()
			solution = simulate(problem, method, request)
			runtime = perf_counter() - started
			step_runs[method_name] = TangentTaylorEulerAccuracyRun(
				method_name=method_name,
				step_count=step_count,
				integration_step=integration_step,
				solution=solution,
				accuracy=accuracy_series(
					method_name,
					solution.states,
					reference.states,
					period=potential.grid.period,
				),
				runtime_seconds=runtime,
			)
		all_runs[step_count] = step_runs
	return TangentTaylorEulerAccuracyResult(
		potential=potential,
		config=config,
		reference=reference,
		runs=all_runs,
	)


__all__ = [
	"TANGENT_TAYLOR_EULER_METHOD_NAMES",
	"TangentTaylorEulerAccuracyConfig",
	"TangentTaylorEulerAccuracyOrder",
	"TangentTaylorEulerAccuracyResult",
	"TangentTaylorEulerAccuracyRun",
	"TangentTaylorEulerAccuracySummary",
	"TangentTaylorEulerReference",
	"run_tangent_taylor_euler_accuracy_study",
]
