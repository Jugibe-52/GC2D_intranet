"""Generalized-energy refinement studies for projected implicit GC methods."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np

from diagnostics import (
	GCGeneralizedEnergyObserver,
	GCGeneralizedEnergyRecord,
	GCReducedTimeExtendedSymplecticityObserver,
	GCReducedTimeExtendedSymplecticityRecord,
	GCTimeExtendedSymplecticityObserver,
	GCTimeExtendedSymplecticityRecord,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA4Implicit1,
	BM4Implicit1,
	ImplicitABBA1,
	InitialValueProblem,
	NumericalMethod,
	SimulationRequest,
	Solution,
	IntegrationStep,
	simulate,
)

from ._validation import (
	nonnegative_finite,
	positive_finite,
	positive_integer,
	resolve_rho,
)


ImplicitEnergyMethod = Literal[
	"implicit_abba_1",
	"abba4_implicit_1",
	"bm4_implicit_1",
]
IMPLICIT_ENERGY_METHODS: tuple[ImplicitEnergyMethod, ...] = (
	"implicit_abba_1",
	"abba4_implicit_1",
	"bm4_implicit_1",
)
IMPLICIT_ENERGY_METHOD_LABELS: Mapping[ImplicitEnergyMethod, str] = MappingProxyType(
	{
		"implicit_abba_1": "ImplicitABBA1",
		"abba4_implicit_1": "ABBA4Implicit1",
		"bm4_implicit_1": "BM4Implicit1",
	}
)


def _validated_steps(steps: tuple[float, ...]) -> tuple[float, ...]:
	"""Normalize distinct positive steps ordered from coarsest to finest."""
	values = tuple(float(step) for step in steps)
	if not values or any(not np.isfinite(step) or step <= 0.0 for step in values):
		raise ValueError("`steps` must contain positive finite values.")
	if len(set(values)) != len(values):
		raise ValueError("`steps` must not contain duplicates.")
	if any(coarse <= fine for coarse, fine in zip(values, values[1:])):
		raise ValueError("`steps` must be ordered from coarsest to finest.")
	return values


def _validated_method(method: str) -> ImplicitEnergyMethod:
	"""Return one supported implicit energy-study method name."""
	if method not in IMPLICIT_ENERGY_METHODS:
		raise ValueError(
			"`method` must be one of "
			+ ", ".join(repr(value) for value in IMPLICIT_ENERGY_METHODS)
			+ "."
		)
	return method


@dataclass(frozen=True, slots=True)
class ImplicitGeneralizedEnergyConfig:
	"""Step refinement and nonlinear controls for one implicit GC method."""

	steps: tuple[float, ...]
	t_span: tuple[float, float]
	output_sample_count: int = 201
	rho: float | None = None
	coupling_frequency: float = float(np.pi / 8.0)
	newton_absolute_tolerance: float = 1e-14
	newton_relative_tolerance: float = 1e-13
	newton_max_iterations: int = 40
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	symplecticity_jacobian_relative_step: float = float(
		np.cbrt(np.finfo(float).eps)
	)
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate all controls before running the four refinements."""
		object.__setattr__(self, "steps", _validated_steps(tuple(self.steps)))
		try:
			start, stop = (float(value) for value in self.t_span)
		except (TypeError, ValueError) as exc:
			raise ValueError(
				"`t_span` must contain two finite increasing times."
			) from exc
		if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
			raise ValueError("`t_span` must contain two finite increasing times.")
		object.__setattr__(self, "t_span", (start, stop))
		object.__setattr__(
			self,
			"output_sample_count",
			positive_integer(self.output_sample_count, "output_sample_count"),
		)
		if self.output_sample_count < 2:
			raise ValueError("`output_sample_count` must be at least 2.")
		if self.rho is not None:
			object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		for name in (
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"newton_jacobian_relative_step",
			"symplecticity_jacobian_relative_step",
		):
			object.__setattr__(
				self,
				name,
				positive_finite(getattr(self, name), name),
			)
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)
		object.__setattr__(self, "progress", bool(self.progress))


@dataclass(frozen=True, slots=True)
class ImplicitGeneralizedEnergyRun:
	"""One complete step-size run and its accepted-node energy histories."""

	step: float
	actual_step: float
	solution: Solution
	records: tuple[GCGeneralizedEnergyRecord, ...]
	extended_symplecticity_records: tuple[
		GCTimeExtendedSymplecticityRecord,
		...,
	]
	reduced_extended_symplecticity_records: tuple[
		GCReducedTimeExtendedSymplecticityRecord,
		...,
	]

	@property
	def times(self) -> np.ndarray:
		"""Accepted main-grid times, including the initial node."""
		return np.asarray([record.time for record in self.records], dtype=float)

	@property
	def hamiltonian(self) -> np.ndarray:
		"""Physical non-autonomous Hamiltonian history ``h(t_n,z_n)``."""
		return np.asarray([record.hamiltonian for record in self.records], dtype=float)

	@property
	def kappa(self) -> np.ndarray:
		"""Normalized time-conjugate momentum history ``k/2``."""
		return np.asarray([record.kappa for record in self.records], dtype=float)

	@property
	def generalized_energy(self) -> np.ndarray:
		"""Autonomous extended-energy history ``K=h+kappa``."""
		return np.asarray(
			[record.generalized_energy for record in self.records],
			dtype=float,
		)

	@property
	def energy_errors(self) -> np.ndarray:
		"""Signed generalized-energy drift ``K_n-K_0``."""
		return np.asarray([record.energy_error for record in self.records], dtype=float)

	@property
	def relative_errors(self) -> np.ndarray:
		"""Signed generalized-energy drift normalized by ``|K_0|``."""
		return np.asarray([record.relative_error for record in self.records], dtype=float)

	@property
	def running_max_relative_errors(self) -> np.ndarray:
		"""Running envelope ``max_{j<=n}|epsilon_K(t_j)|``."""
		return np.maximum.accumulate(np.abs(self.relative_errors))

	@property
	def extended_symplecticity_times(self) -> np.ndarray:
		"""Accepted final times for the ``R^6`` splitting-map measurements."""
		return np.asarray(
			[record.time for record in self.extended_symplecticity_records],
			dtype=float,
		)

	@property
	def extended_relative_defects(self) -> np.ndarray:
		"""Maximum relative ``R^6`` symplectic defect per accepted step."""
		return np.asarray(
			[
				record.maximum_relative_defect
				for record in self.extended_symplecticity_records
			],
			dtype=float,
		)

	@property
	def extended_determinant_errors(self) -> np.ndarray:
		"""Maximum ``|det(D Psi)-1|`` per accepted splitting map group."""
		return np.asarray(
			[
				record.maximum_determinant_error
				for record in self.extended_symplecticity_records
			],
			dtype=float,
		)

	@property
	def reduced_extended_symplecticity_times(self) -> np.ndarray:
		"""Accepted times for the complete projected ``R^4`` measurements."""
		return np.asarray(
			[
				record.time
				for record in self.reduced_extended_symplecticity_records
			],
			dtype=float,
		)

	@property
	def reduced_extended_relative_defects(self) -> np.ndarray:
		"""Relative symplectic defects of the projected ``R^4`` map."""
		return np.asarray(
			[
				record.relative_defect
				for record in self.reduced_extended_symplecticity_records
			],
			dtype=float,
		)

	@property
	def reduced_extended_determinant_errors(self) -> np.ndarray:
		"""Determinant errors of the projected ``R^4`` map."""
		return np.asarray(
			[
				record.determinant_error
				for record in self.reduced_extended_symplecticity_records
			],
			dtype=float,
		)


@dataclass(frozen=True, slots=True)
class ImplicitGeneralizedEnergySummary:
	"""Energy drift, secular trend, and solver work for one step size."""

	method_name: str
	step: float
	actual_step: float
	step_count: int
	max_absolute_error: float
	final_absolute_error: float
	rms_absolute_error: float
	max_relative_error: float
	linear_drift_rate: float
	final_to_max_ratio: float
	max_newton_iterations: int
	mean_newton_iterations: float
	max_newton_residual_norm: float
	extended_base_map_count: int
	extended_symplecticity_scope: str
	max_extended_relative_defect: float
	rms_extended_relative_defect: float
	max_extended_determinant_error: float
	reduced_extended_symplecticity_scope: str
	max_reduced_extended_relative_defect: float
	rms_reduced_extended_relative_defect: float
	max_reduced_extended_determinant_error: float


@dataclass(frozen=True, slots=True)
class ImplicitGeneralizedEnergyOrder:
	"""Empirical max-error order between two adjacent refinements."""

	coarse_step: float
	fine_step: float
	maximum_error_order: float


def _empirical_order(
	coarse_error: float,
	fine_error: float,
	coarse_step: float,
	fine_step: float,
) -> float:
	"""Return a logarithmic refinement slope or NaN at an error floor."""
	if coarse_error <= 0.0 or fine_error <= 0.0:
		return float("nan")
	return float(
		np.log(coarse_error / fine_error) / np.log(coarse_step / fine_step)
	)


@dataclass(frozen=True, slots=True)
class ImplicitGeneralizedEnergyResult:
	"""Four-step energy refinement for one projected implicit GC method."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	method: ImplicitEnergyMethod
	runs: tuple[ImplicitGeneralizedEnergyRun, ...]

	@property
	def method_name(self) -> str:
		"""Public numerical method label."""
		return IMPLICIT_ENERGY_METHOD_LABELS[self.method]

	@property
	def steps(self) -> tuple[float, ...]:
		"""Configured maximum steps from coarsest to finest."""
		return tuple(run.step for run in self.runs)

	def summaries(self) -> tuple[ImplicitGeneralizedEnergySummary, ...]:
		"""Reduce energy histories and nonlinear diagnostics for each run."""
		rows: list[ImplicitGeneralizedEnergySummary] = []
		for run in self.runs:
			errors = run.energy_errors
			absolute_errors = np.abs(errors)
			max_absolute_error = float(np.max(absolute_errors))
			duration = run.times - run.times[0]
			linear_drift_rate = float(np.polyfit(duration, errors, deg=1)[0])
			iterations = np.asarray(
				run.solution.diagnostics["newton_iterations"],
				dtype=float,
			)
			residuals = np.asarray(
				run.solution.diagnostics["newton_residual_norms"],
				dtype=float,
			)
			extended_relative_defects = run.extended_relative_defects
			extended_determinant_errors = run.extended_determinant_errors
			first_extended_record = run.extended_symplecticity_records[0]
			reduced_defects = run.reduced_extended_relative_defects
			reduced_determinants = run.reduced_extended_determinant_errors
			first_reduced_record = run.reduced_extended_symplecticity_records[0]
			rows.append(
				ImplicitGeneralizedEnergySummary(
					method_name=self.method_name,
					step=run.step,
					actual_step=run.actual_step,
					step_count=run.solution.n_steps,
					max_absolute_error=max_absolute_error,
					final_absolute_error=float(errors[-1]),
					rms_absolute_error=float(np.sqrt(np.mean(errors**2))),
					max_relative_error=float(np.max(np.abs(run.relative_errors))),
					linear_drift_rate=linear_drift_rate,
					final_to_max_ratio=(
						abs(float(errors[-1])) / max_absolute_error
						if max_absolute_error > 0.0
						else 0.0
					),
					max_newton_iterations=int(np.max(iterations)),
					mean_newton_iterations=float(np.mean(iterations)),
					max_newton_residual_norm=float(np.max(residuals)),
					extended_base_map_count=first_extended_record.base_map_count,
					extended_symplecticity_scope=first_extended_record.scope,
					max_extended_relative_defect=float(
						np.max(extended_relative_defects)
					),
					rms_extended_relative_defect=float(
						np.sqrt(np.mean(extended_relative_defects**2))
					),
					max_extended_determinant_error=float(
						np.max(extended_determinant_errors)
					),
					reduced_extended_symplecticity_scope=first_reduced_record.scope,
					max_reduced_extended_relative_defect=float(
						np.max(reduced_defects)
					),
					rms_reduced_extended_relative_defect=float(
						np.sqrt(np.mean(reduced_defects**2))
					),
					max_reduced_extended_determinant_error=float(
						np.max(reduced_determinants)
					),
				)
			)
		return tuple(rows)

	def convergence_orders(self) -> tuple[ImplicitGeneralizedEnergyOrder, ...]:
		"""Return adjacent max-energy-error refinement orders."""
		summaries = self.summaries()
		return tuple(
			ImplicitGeneralizedEnergyOrder(
				coarse_step=coarse.actual_step,
				fine_step=fine.actual_step,
				maximum_error_order=_empirical_order(
					coarse.max_absolute_error,
					fine.max_absolute_error,
					coarse.actual_step,
					fine.actual_step,
				),
			)
			for coarse, fine in zip(summaries, summaries[1:])
		)

	def print_summary(self) -> None:
		"""Print one compact row per step and the adjacent refinement orders."""
		print(f"{self.method_name} generalized energy K=h+kappa")
		print(
			f"{'step':>9} {'steps':>7} {'max |dK|':>14} {'final dK':>14} "
			f"{'max |eps_K|':>14} {'drift/time':>14} {'ext defect':>14} "
			f"{'ext det':>12} {'R4 defect':>14} {'R4 det':>12} {'Newton':>8}"
		)
		for row in self.summaries():
			print(
				f"{row.actual_step:9.4g} {row.step_count:7d} "
				f"{row.max_absolute_error:14.6e} "
				f"{row.final_absolute_error:14.6e} "
				f"{row.max_relative_error:14.6e} "
				f"{row.linear_drift_rate:14.6e} "
				f"{row.max_extended_relative_defect:14.6e} "
				f"{row.max_extended_determinant_error:12.4e} "
				f"{row.max_reduced_extended_relative_defect:14.6e} "
				f"{row.max_reduced_extended_determinant_error:12.4e} "
				f"{row.mean_newton_iterations:8.3f}"
			)
		print("\nAdjacent max-error orders:")
		for order in self.convergence_orders():
			print(
				f"  h={order.coarse_step:g} -> {order.fine_step:g}: "
				f"p={order.maximum_error_order:.4f}"
			)


def _method_for_run(
	method: ImplicitEnergyMethod,
	config: ImplicitGeneralizedEnergyConfig,
	observer: Callable[[IntegrationStep], None],
) -> NumericalMethod:
	"""Construct one implicit method with one accepted-step observer."""
	if method == "implicit_abba_1":
		return ImplicitABBA1(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		)
	if method == "abba4_implicit_1":
		return ABBA4Implicit1(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		)
	return BM4Implicit1(
		coupling_frequency=config.coupling_frequency,
		newton_absolute_tolerance=config.newton_absolute_tolerance,
		newton_relative_tolerance=config.newton_relative_tolerance,
		newton_max_iterations=config.newton_max_iterations,
		newton_jacobian_relative_step=config.newton_jacobian_relative_step,
		progress=config.progress,
		step_observer=observer,
	)


def run_implicit_generalized_energy_study(
	potential: Potential,
	configuration: GCInitialConfiguration,
	*,
	method: ImplicitEnergyMethod,
	config: ImplicitGeneralizedEnergyConfig,
) -> ImplicitGeneralizedEnergyResult:
	"""Run accepted-stage generalized-energy reconstruction at several steps."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(configuration, GCInitialConfiguration):
		raise TypeError("`configuration` must be a GCInitialConfiguration instance.")
	if not isinstance(config, ImplicitGeneralizedEnergyConfig):
		raise TypeError("`config` must be ImplicitGeneralizedEnergyConfig.")
	method_name = _validated_method(method)
	initial_state = configuration.initial_state
	if initial_state is None or configuration.particle_count(initial_state) != 1:
		raise ValueError("The implicit energy study requires exactly one GC state.")

	dynamics = GuidingCenterDynamics(
		potential,
		rho=resolve_rho(config.rho, configuration),
	)
	problem = InitialValueProblem(dynamics, configuration)
	runs: list[ImplicitGeneralizedEnergyRun] = []
	for step in config.steps:
		energy_observer = GCGeneralizedEnergyObserver(
			dynamics,
			initial_time=config.t_span[0],
			initial_state=initial_state,
		)
		symplecticity_observer = GCTimeExtendedSymplecticityObserver(
			dynamics,
			relative_step=config.symplecticity_jacobian_relative_step,
		)

		def reduced_step_map(
			start_time: float,
			state: np.ndarray,
			duration: float,
		) -> tuple[np.ndarray, float]:
			candidate_state = np.asarray(state, dtype=float)
			candidate_configuration = GCInitialConfiguration.from_components(
				x=np.asarray([candidate_state[0]], dtype=float),
				y=np.asarray([candidate_state[1]], dtype=float),
			)
			candidate_problem = InitialValueProblem(dynamics, candidate_configuration)
			candidate_energy_observer = GCGeneralizedEnergyObserver(
				dynamics,
				initial_time=start_time,
				initial_state=candidate_state,
			)
			candidate_solution = simulate(
				candidate_problem,
				_method_for_run(
					method_name,
					config,
					candidate_energy_observer,
				),
				SimulationRequest.uniform(
					t_span=(start_time, start_time + duration),
					max_step=duration,
					sample_count=2,
				),
			)
			return (
				np.asarray(candidate_solution.states[:, -1], dtype=float),
				float(candidate_energy_observer.records[-1].kappa),
			)

		reduced_symplecticity_observer = (
			GCReducedTimeExtendedSymplecticityObserver(
				dynamics,
				step_map=reduced_step_map,
				relative_step=config.symplecticity_jacobian_relative_step,
			)
		)

		def observer(record: IntegrationStep) -> None:
			energy_observer(record)
			symplecticity_observer(record)
			reduced_symplecticity_observer(record)

		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step,
			sample_count=config.output_sample_count,
		)
		solution = simulate(
			problem,
			_method_for_run(method_name, config, observer),
			request,
		)
		records = energy_observer.records
		if len(records) != solution.n_steps + 1:
			raise RuntimeError("Energy records do not match the accepted step count.")
		extended_records = symplecticity_observer.records
		if len(extended_records) != solution.n_steps:
			raise RuntimeError(
				"Extended symplecticity records do not match the accepted step count."
			)
		reduced_records = reduced_symplecticity_observer.records
		if len(reduced_records) != solution.n_steps:
			raise RuntimeError(
				"Reduced extended records do not match the accepted step count."
			)
		actual_step = float(records[1].duration)
		runs.append(
			ImplicitGeneralizedEnergyRun(
				step=step,
				actual_step=actual_step,
				solution=solution,
				records=records,
				extended_symplecticity_records=extended_records,
				reduced_extended_symplecticity_records=reduced_records,
			)
		)
	return ImplicitGeneralizedEnergyResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=configuration,
		method=method_name,
		runs=tuple(runs),
	)


__all__ = [
	"IMPLICIT_ENERGY_METHOD_LABELS",
	"IMPLICIT_ENERGY_METHODS",
	"ImplicitEnergyMethod",
	"ImplicitGeneralizedEnergyConfig",
	"ImplicitGeneralizedEnergyOrder",
	"ImplicitGeneralizedEnergyResult",
	"ImplicitGeneralizedEnergyRun",
	"ImplicitGeneralizedEnergySummary",
	"run_implicit_generalized_energy_study",
]
