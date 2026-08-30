"""Reproducible HBVM(4,2) evaluation and BM4 comparison studies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import DynamicalSystem, HamiltonianSystem
from initial_conditions import GCInitialConfiguration
from simulation import (
	BM4Composition,
	GCExtendedFormulation,
	HBVM42,
	HBVMJacobianMethod,
	InitialValueProblem,
	NumericalMethod,
	SimulationRequest,
	Solution,
	simulate,
)

from ._validation import integer_ratio, positive_finite, positive_integer


HBVM42_LABEL = "HBVM(4,2)"
BM4_LABEL = "BM4"


class QuarticOscillatorDynamics:
	"""Independent canonical oscillators with ``H=(p^2)/2+a(q^4)/4``.

	Packed states follow the project-wide component-major convention
	``[q_1, ..., q_N, p_1, ..., p_N]``. The quartic polynomial makes this a
	sharp energy-preservation test: HBVM(4,2) integrates its Hamiltonian line
	integral exactly apart from nonlinear-solver and floating-point errors.
	"""

	state_dimension = 2

	def __init__(self, *, quartic_strength: float = 1.0) -> None:
		"""Create oscillators with one positive quartic coefficient."""
		self.quartic_strength = positive_finite(
			quartic_strength,
			"quartic_strength",
		)

	@staticmethod
	def _split(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""Return component-major position and momentum blocks."""
		value = np.asarray(state, dtype=float)
		if value.ndim == 0 or value.shape[0] == 0 or value.shape[0] % 2:
			raise ValueError("Quartic oscillator states require equal q and p blocks.")
		particle_count = value.shape[0] // 2
		return value[:particle_count], value[particle_count:]

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate ``q'=p`` and ``p'=-a*q^3``."""
		del t
		position, momentum = self._split(state)
		return np.concatenate(
			(momentum, -self.quartic_strength * position**3),
			axis=0,
		)

	def particle_vector_field_jacobians(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return one exact canonical two-by-two field Jacobian per oscillator."""
		del t
		position, _ = self._split(state)
		result = np.zeros((position.shape[0], 2, 2), dtype=float)
		result[:, 0, 1] = 1.0
		result[:, 1, 0] = -3.0 * self.quartic_strength * position**2
		return result

	def hamiltonian(
		self,
		t: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate the autonomous quartic Hamiltonian for every oscillator."""
		del t
		position, momentum = self._split(state)
		return (
			0.5 * momentum**2
			+ 0.25 * self.quartic_strength * position**4
		)


def quartic_oscillator_configuration(
	*,
	position: float = 1.0,
	momentum: float = 0.0,
) -> GCInitialConfiguration:
	"""Build the one-oscillator configuration used by both HBVM notebooks."""
	values = np.asarray((position, momentum), dtype=float)
	if not np.all(np.isfinite(values)):
		raise ValueError("Initial position and momentum must be finite.")
	return GCInitialConfiguration.from_components(
		x=values[:1],
		y=values[1:],
	)


def _validated_span(t_span: tuple[float, float]) -> tuple[float, float]:
	"""Normalize one finite increasing integration interval."""
	values = np.asarray(t_span, dtype=float)
	if values.shape != (2,) or not np.all(np.isfinite(values)) or values[0] >= values[1]:
		raise ValueError("`t_span` must contain two finite, increasing times.")
	return float(values[0]), float(values[1])


def _validated_steps(
	steps: tuple[float, ...],
	*,
	duration: float,
) -> tuple[float, ...]:
	"""Return strictly decreasing complete step sizes spanning the interval."""
	values = tuple(positive_finite(step, "step") for step in steps)
	if len(values) < 2 or any(coarse <= fine for coarse, fine in zip(values, values[1:])):
		raise ValueError("`steps` must contain at least two strictly decreasing values.")
	for step in values:
		integer_ratio(duration, step, "duration / step")
	return values


def _validated_jacobian_method(value: str) -> HBVMJacobianMethod:
	"""Validate the public HBVM Jacobian selector at study construction time."""
	if value not in ("auto", "analytic", "finite_difference"):
		raise ValueError(
			"`jacobian_method` must be 'auto', 'analytic', or 'finite_difference'."
		)
	return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class HBVM42EvaluationConfig:
	"""Controls for individual accuracy, cost, energy, and geometry evaluation."""

	steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05)
	t_span: tuple[float, float] = (0.0, 8.0)
	save_interval: float = 0.4
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 12
	jacobian_method: HBVMJacobianMethod = "analytic"
	jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	symplecticity_relative_step: float = 1e-5
	reference_relative_tolerance: float = 2e-13
	reference_absolute_tolerance: float = 2e-15
	reference_maximum_step: float = 0.002
	runtime_warmups: int = 1
	runtime_repeats: int = 3
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate a nested fixed-grid study before any integration begins."""
		span = _validated_span(self.t_span)
		duration = span[1] - span[0]
		steps = _validated_steps(tuple(self.steps), duration=duration)
		save_interval = positive_finite(self.save_interval, "save_interval")
		integer_ratio(duration, save_interval, "duration / save_interval")
		for step in steps:
			integer_ratio(save_interval, step, "save_interval / step")
		object.__setattr__(self, "t_span", span)
		object.__setattr__(self, "steps", steps)
		object.__setattr__(self, "save_interval", save_interval)
		for name in (
			"absolute_tolerance",
			"relative_tolerance",
			"jacobian_relative_step",
			"symplecticity_relative_step",
			"reference_relative_tolerance",
			"reference_absolute_tolerance",
			"reference_maximum_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(self, "max_iterations", positive_integer(self.max_iterations, "max_iterations"))
		object.__setattr__(
			self,
			"jacobian_method",
			_validated_jacobian_method(self.jacobian_method),
		)
		if (
			isinstance(self.runtime_warmups, (bool, np.bool_))
			or not isinstance(self.runtime_warmups, (int, np.integer))
			or self.runtime_warmups < 0
		):
			raise ValueError("`runtime_warmups` must be a non-negative integer.")
		object.__setattr__(self, "runtime_warmups", int(self.runtime_warmups))
		object.__setattr__(self, "runtime_repeats", positive_integer(self.runtime_repeats, "runtime_repeats"))

	@property
	def output_sample_count(self) -> int:
		"""Number of common saved states including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class HBVM42EvaluationSummary:
	"""All individual HBVM metrics for one complete step size."""

	step: float
	step_count: int
	trajectory_rms_error: float
	final_error: float
	median_runtime_seconds: float
	minimum_runtime_seconds: float
	maximum_absolute_energy_error: float
	maximum_relative_energy_error: float
	local_symplecticity_defect: float
	flow_symplecticity_defect: float
	local_determinant_error: float
	flow_determinant_error: float
	maximum_nonlinear_iterations: int
	mean_nonlinear_iterations: float
	mean_vector_field_evaluations: float


@dataclass(frozen=True, slots=True)
class HBVM42OrderSummary:
	"""Observed global orders and reductions from the designed order four."""

	coarse_step: float
	fine_step: float
	trajectory_rms_order: float
	final_error_order: float
	trajectory_order_reduction: float
	final_order_reduction: float


@dataclass(frozen=True, slots=True)
class HBVM42EvaluationResult:
	"""Reference, HBVM solutions, and reduced metrics for one individual study."""

	config: HBVM42EvaluationConfig
	dynamics: DynamicalSystem
	initial_configuration: GCInitialConfiguration
	times: np.ndarray
	reference_states: np.ndarray
	solutions: Mapping[float, Solution]
	summary_rows: tuple[HBVM42EvaluationSummary, ...]
	order_rows: tuple[HBVM42OrderSummary, ...]

	def summaries(self) -> tuple[HBVM42EvaluationSummary, ...]:
		"""Return rows in configured coarse-to-fine order."""
		return self.summary_rows

	def convergence_orders(self) -> tuple[HBVM42OrderSummary, ...]:
		"""Return adjacent refinement slopes and order reductions."""
		return self.order_rows


@dataclass(frozen=True, slots=True)
class HBVM42BM4ComparisonConfig:
	"""Common accuracy and wall-clock controls for HBVM(4,2) versus BM4."""

	steps: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05)
	t_span: tuple[float, float] = (0.0, 8.0)
	absolute_tolerance: float = 1e-14
	relative_tolerance: float = 1e-13
	max_iterations: int = 12
	jacobian_method: HBVMJacobianMethod = "analytic"
	jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	coupling_frequency: float = float(np.pi / 8.0)
	reference_relative_tolerance: float = 2e-13
	reference_absolute_tolerance: float = 2e-15
	reference_maximum_step: float = 0.002
	runtime_warmups: int = 1
	runtime_repeats: int = 5
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate common grids, solver controls, and benchmark repetition counts."""
		span = _validated_span(self.t_span)
		steps = _validated_steps(tuple(self.steps), duration=span[1] - span[0])
		object.__setattr__(self, "t_span", span)
		object.__setattr__(self, "steps", steps)
		for name in (
			"absolute_tolerance",
			"relative_tolerance",
			"jacobian_relative_step",
			"reference_relative_tolerance",
			"reference_absolute_tolerance",
			"reference_maximum_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(self, "max_iterations", positive_integer(self.max_iterations, "max_iterations"))
		object.__setattr__(
			self,
			"jacobian_method",
			_validated_jacobian_method(self.jacobian_method),
		)
		frequency = float(self.coupling_frequency)
		if not np.isfinite(frequency) or frequency < 0.0:
			raise ValueError("`coupling_frequency` must be finite and non-negative.")
		object.__setattr__(self, "coupling_frequency", frequency)
		if (
			isinstance(self.runtime_warmups, (bool, np.bool_))
			or not isinstance(self.runtime_warmups, (int, np.integer))
			or self.runtime_warmups < 0
		):
			raise ValueError("`runtime_warmups` must be a non-negative integer.")
		object.__setattr__(self, "runtime_warmups", int(self.runtime_warmups))
		object.__setattr__(self, "runtime_repeats", positive_integer(self.runtime_repeats, "runtime_repeats"))


@dataclass(frozen=True, slots=True)
class HBVM42BM4Summary:
	"""Accuracy and timing of one method at one complete step size."""

	method: str
	step: float
	step_count: int
	final_error: float
	median_runtime_seconds: float
	minimum_runtime_seconds: float
	runtime_repeats: int


@dataclass(frozen=True, slots=True)
class HBVM42BM4ComparisonResult:
	"""Common reference and benchmark rows for HBVM(4,2) and BM4."""

	config: HBVM42BM4ComparisonConfig
	dynamics: DynamicalSystem
	initial_configuration: GCInitialConfiguration
	reference_final_state: np.ndarray
	summary_rows: tuple[HBVM42BM4Summary, ...]

	def summaries(self) -> tuple[HBVM42BM4Summary, ...]:
		"""Return method rows grouped by configured step order."""
		return self.summary_rows


def _validated_problem(
	dynamics: DynamicalSystem,
	configuration: GCInitialConfiguration,
	*,
	require_hamiltonian: bool,
) -> InitialValueProblem:
	"""Build the one-particle planar problem required by both studies."""
	if not isinstance(dynamics, DynamicalSystem):
		raise TypeError("`dynamics` must implement DynamicalSystem.")
	if require_hamiltonian and not isinstance(dynamics, HamiltonianSystem):
		raise TypeError("The individual HBVM study requires HamiltonianSystem dynamics.")
	if not isinstance(configuration, GCInitialConfiguration):
		raise TypeError("`configuration` must be a GCInitialConfiguration.")
	problem = InitialValueProblem(dynamics, configuration)
	if problem.particle_count != 1 or problem.initial_state.size != 2:
		raise ValueError("HBVM notebook studies require one planar trajectory.")
	return problem


def _reference_solution(
	problem: InitialValueProblem,
	times: np.ndarray,
	*,
	relative_tolerance: float,
	absolute_tolerance: float,
	maximum_step: float,
) -> np.ndarray:
	"""Compute an independent high-accuracy DOP853 reference on fixed times."""
	result = solve_ivp(
		fun=lambda time, state: problem.dynamics.vector_field(time, state),
		t_span=(float(times[0]), float(times[-1])),
		y0=problem.initial_state,
		method="DOP853",
		t_eval=times,
		rtol=relative_tolerance,
		atol=absolute_tolerance,
		max_step=maximum_step,
		dense_output=False,
		vectorized=False,
	)
	if not result.success:
		raise RuntimeError(f"DOP853 reference integration failed: {result.message}")
	states = np.asarray(result.y, dtype=float)
	if states.shape != (problem.initial_state.size, times.size):
		raise ValueError("The DOP853 reference returned an incompatible state history.")
	return states


def _hbvm_method(
	config: HBVM42EvaluationConfig | HBVM42BM4ComparisonConfig,
	*,
	track_energy: bool,
) -> HBVM42:
	"""Construct one identically configured HBVM instance."""
	return HBVM42(
		absolute_tolerance=config.absolute_tolerance,
		relative_tolerance=config.relative_tolerance,
		max_iterations=config.max_iterations,
		jacobian_method=config.jacobian_method,
		jacobian_relative_step=config.jacobian_relative_step,
		track_energy=track_energy,
		progress=config.progress,
	)


def _timed_simulations(
	problem: InitialValueProblem,
	method_factory: Callable[[], NumericalMethod],
	request: SimulationRequest,
	*,
	warmups: int,
	repeats: int,
) -> tuple[float, float, Solution]:
	"""Return median/minimum wall time and the last repeated solution."""
	for _ in range(warmups):
		simulate(problem, method_factory(), request)
	timings = np.empty(repeats, dtype=float)
	last_solution: Solution | None = None
	for index in range(repeats):
		started = perf_counter()
		last_solution = simulate(problem, method_factory(), request)
		timings[index] = perf_counter() - started
	assert last_solution is not None
	return float(np.median(timings)), float(np.min(timings)), last_solution


def _flow_jacobian(
	problem: InitialValueProblem,
	method_factory: Callable[[], NumericalMethod],
	*,
	t_span: tuple[float, float],
	step: float,
	relative_step: float,
) -> np.ndarray:
	"""Differentiate one complete numerical flow by centered initial states."""
	initial_state = problem.initial_state
	jacobian = np.empty((2, 2), dtype=float)
	request = SimulationRequest.uniform(
		t_span=t_span,
		max_step=step,
		sample_count=2,
	)
	for column in range(2):
		increment = relative_step * max(1.0, abs(float(initial_state[column])))
		endpoints: list[np.ndarray] = []
		for sign in (1.0, -1.0):
			perturbed = initial_state.copy()
			perturbed[column] += sign * increment
			perturbed_problem = InitialValueProblem(
				problem.dynamics,
				GCInitialConfiguration(perturbed),
			)
			endpoints.append(
				simulate(
					perturbed_problem,
					method_factory(),
					request,
				).states[:, -1]
			)
		jacobian[:, column] = (endpoints[0] - endpoints[1]) / (2.0 * increment)
	return jacobian


def _symplecticity_metrics(jacobian: np.ndarray) -> tuple[float, float]:
	"""Return relative canonical-form defect and absolute determinant error."""
	symplectic_form = np.asarray(((0.0, 1.0), (-1.0, 0.0)))
	defect = jacobian.T @ symplectic_form @ jacobian - symplectic_form
	return (
		float(np.linalg.norm(defect, ord="fro") / np.linalg.norm(symplectic_form, ord="fro")),
		abs(float(np.linalg.det(jacobian)) - 1.0),
	)


def _empirical_order(
	coarse_error: float,
	fine_error: float,
	coarse_step: float,
	fine_step: float,
) -> float:
	"""Return one refinement slope or NaN for a round-off-limited pair."""
	if coarse_error <= 0.0 or fine_error <= 0.0:
		return float("nan")
	return float(
		np.log(coarse_error / fine_error)
		/ np.log(coarse_step / fine_step)
	)


def run_hbvm42_evaluation(
	dynamics: DynamicalSystem,
	configuration: GCInitialConfiguration,
	*,
	config: HBVM42EvaluationConfig,
) -> HBVM42EvaluationResult:
	"""Evaluate HBVM(4,2) accuracy, cost, energy, geometry, and order."""
	if not isinstance(config, HBVM42EvaluationConfig):
		raise TypeError("`config` must be an HBVM42EvaluationConfig.")
	problem = _validated_problem(dynamics, configuration, require_hamiltonian=True)
	assert isinstance(dynamics, HamiltonianSystem)
	times = SimulationRequest.uniform(
		t_span=config.t_span,
		max_step=config.save_interval,
		sample_count=config.output_sample_count,
	).output_times
	reference_states = _reference_solution(
		problem,
		times,
		relative_tolerance=config.reference_relative_tolerance,
		absolute_tolerance=config.reference_absolute_tolerance,
		maximum_step=config.reference_maximum_step,
	)
	solutions: dict[float, Solution] = {}
	rows: list[HBVM42EvaluationSummary] = []
	for step in config.steps:
		request = SimulationRequest(
			t_span=config.t_span,
			max_step=step,
			output_times=times,
		)
		solution = simulate(problem, _hbvm_method(config, track_energy=True), request)
		solutions[step] = solution
		differences = solution.states - reference_states
		distances = np.linalg.norm(differences, axis=0)
		trajectory_rms_error = float(
			np.sqrt(
				np.trapz(distances**2, times)
				/ (config.t_span[1] - config.t_span[0])
			)
		)
		final_error = float(distances[-1])

		benchmark_request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=2,
		)
		median_runtime, minimum_runtime, _ = _timed_simulations(
			problem,
			lambda: _hbvm_method(config, track_energy=False),
			benchmark_request,
			warmups=config.runtime_warmups,
			repeats=config.runtime_repeats,
		)

		local_jacobian = _flow_jacobian(
			problem,
			lambda: _hbvm_method(config, track_energy=False),
			t_span=(config.t_span[0], config.t_span[0] + step),
			step=step,
			relative_step=config.symplecticity_relative_step,
		)
		flow_jacobian = _flow_jacobian(
			problem,
			lambda: _hbvm_method(config, track_energy=False),
			t_span=config.t_span,
			step=step,
			relative_step=config.symplecticity_relative_step,
		)
		local_defect, local_determinant_error = _symplecticity_metrics(local_jacobian)
		flow_defect, flow_determinant_error = _symplecticity_metrics(flow_jacobian)
		energies = np.asarray(solution.diagnostics["hamiltonian"], dtype=float)
		energy_scale = max(abs(float(energies[0, 0])), np.finfo(float).eps)
		iterations = np.asarray(solution.diagnostics["nonlinear_iterations"], dtype=float)
		field_evaluations = np.asarray(
			solution.diagnostics["vector_field_evaluations_per_step"],
			dtype=float,
		)
		rows.append(
			HBVM42EvaluationSummary(
				step=step,
				step_count=int(solution.diagnostics["step_count"]),
				trajectory_rms_error=trajectory_rms_error,
				final_error=final_error,
				median_runtime_seconds=median_runtime,
				minimum_runtime_seconds=minimum_runtime,
				maximum_absolute_energy_error=float(solution.diagnostics["energy_error"]),
				maximum_relative_energy_error=float(
					np.max(np.abs(energies - energies[:, :1])) / energy_scale
				),
				local_symplecticity_defect=local_defect,
				flow_symplecticity_defect=flow_defect,
				local_determinant_error=local_determinant_error,
				flow_determinant_error=flow_determinant_error,
				maximum_nonlinear_iterations=int(np.max(iterations)),
				mean_nonlinear_iterations=float(np.mean(iterations)),
				mean_vector_field_evaluations=float(np.mean(field_evaluations)),
			)
		)

	orders: list[HBVM42OrderSummary] = []
	for coarse, fine in zip(rows, rows[1:]):
		trajectory_order = _empirical_order(
			coarse.trajectory_rms_error,
			fine.trajectory_rms_error,
			coarse.step,
			fine.step,
		)
		final_order = _empirical_order(
			coarse.final_error,
			fine.final_error,
			coarse.step,
			fine.step,
		)
		orders.append(
			HBVM42OrderSummary(
				coarse_step=coarse.step,
				fine_step=fine.step,
				trajectory_rms_order=trajectory_order,
				final_error_order=final_order,
				trajectory_order_reduction=4.0 - trajectory_order,
				final_order_reduction=4.0 - final_order,
			)
		)
	return HBVM42EvaluationResult(
		config=config,
		dynamics=dynamics,
		initial_configuration=configuration,
		times=np.asarray(times),
		reference_states=reference_states,
		solutions=MappingProxyType(solutions),
		summary_rows=tuple(rows),
		order_rows=tuple(orders),
	)


def run_hbvm42_bm4_comparison(
	dynamics: DynamicalSystem,
	configuration: GCInitialConfiguration,
	*,
	config: HBVM42BM4ComparisonConfig,
) -> HBVM42BM4ComparisonResult:
	"""Compare fourth-order endpoint accuracy and runtime on identical grids."""
	if not isinstance(config, HBVM42BM4ComparisonConfig):
		raise TypeError("`config` must be an HBVM42BM4ComparisonConfig.")
	problem = _validated_problem(dynamics, configuration, require_hamiltonian=False)
	reference_times = np.asarray(config.t_span, dtype=float)
	reference_final_state = _reference_solution(
		problem,
		reference_times,
		relative_tolerance=config.reference_relative_tolerance,
		absolute_tolerance=config.reference_absolute_tolerance,
		maximum_step=config.reference_maximum_step,
	)[:, -1]
	method_factories: tuple[tuple[str, Callable[[], NumericalMethod]], ...] = (
		(HBVM42_LABEL, lambda: _hbvm_method(config, track_energy=False)),
		(
			BM4_LABEL,
			lambda: BM4Composition(
				GCExtendedFormulation(
					coupling_frequency=config.coupling_frequency,
				),
				progress=config.progress,
			),
		),
	)
	rows: list[HBVM42BM4Summary] = []
	for step in config.steps:
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=2,
		)
		for method_label, method_factory in method_factories:
			median_runtime, minimum_runtime, solution = _timed_simulations(
				problem,
				method_factory,
				request,
				warmups=config.runtime_warmups,
				repeats=config.runtime_repeats,
			)
			rows.append(
				HBVM42BM4Summary(
					method=method_label,
					step=step,
					step_count=int(solution.diagnostics["step_count"]),
					final_error=float(
						np.linalg.norm(solution.states[:, -1] - reference_final_state)
					),
					median_runtime_seconds=median_runtime,
					minimum_runtime_seconds=minimum_runtime,
					runtime_repeats=config.runtime_repeats,
				)
			)
	return HBVM42BM4ComparisonResult(
		config=config,
		dynamics=dynamics,
		initial_configuration=configuration,
		reference_final_state=reference_final_state,
		summary_rows=tuple(rows),
	)


__all__ = [
	"BM4_LABEL",
	"HBVM42_LABEL",
	"HBVM42BM4ComparisonConfig",
	"HBVM42BM4ComparisonResult",
	"HBVM42BM4Summary",
	"HBVM42EvaluationConfig",
	"HBVM42EvaluationResult",
	"HBVM42EvaluationSummary",
	"HBVM42OrderSummary",
	"QuarticOscillatorDynamics",
	"quartic_oscillator_configuration",
	"run_hbvm42_bm4_comparison",
	"run_hbvm42_evaluation",
]
