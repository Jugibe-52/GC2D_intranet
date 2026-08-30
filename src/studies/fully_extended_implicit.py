"""Energy and symplecticity studies for full ``(z,t,k)`` duplication."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np

from diagnostics import (
	GCFullyExtendedEnergyObserver,
	GCFullyExtendedEnergyRecord,
	GCFullyExtendedSymplecticityObserver,
	GCFullyExtendedSymplecticityRecord,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA2FullyExtendedImplicit,
	ABBA4FullyExtendedImplicit,
	BM4_implicit2,
	FullyExtendedImplicitIntegrationStep,
	IntegrationStep,
	InitialValueProblem,
	NumericalMethod,
	SimulationRequest,
	Solution,
	StepObserver,
	simulate,
)

from ._validation import (
	nonnegative_finite,
	positive_finite,
	positive_integer,
	resolve_rho,
)


FullyExtendedImplicitMethod = Literal[
	"abba2_fully_extended_implicit",
	"abba4_fully_extended_implicit",
	"bm4_implicit2",
]
FULLY_EXTENDED_IMPLICIT_METHODS: tuple[FullyExtendedImplicitMethod, ...] = (
	"abba2_fully_extended_implicit",
	"abba4_fully_extended_implicit",
	"bm4_implicit2",
)
FULLY_EXTENDED_IMPLICIT_LABELS: Mapping[FullyExtendedImplicitMethod, str] = (
	MappingProxyType(
		{
			"abba2_fully_extended_implicit": "ABBA2FullyExtendedImplicit",
			"abba4_fully_extended_implicit": "ABBA4FullyExtendedImplicit",
			"bm4_implicit2": "BM4_implicit2",
		}
	)
)


def _validated_steps(steps: tuple[float, ...]) -> tuple[float, ...]:
	"""Return distinct positive steps from coarsest to finest."""
	values = tuple(float(step) for step in steps)
	if not values or any(not np.isfinite(step) or step <= 0.0 for step in values):
		raise ValueError("`steps` must contain positive finite values.")
	if len(set(values)) != len(values):
		raise ValueError("`steps` must not contain duplicates.")
	if any(coarse <= fine for coarse, fine in zip(values, values[1:])):
		raise ValueError("`steps` must be ordered from coarsest to finest.")
	return values


def _validated_method(method: str) -> FullyExtendedImplicitMethod:
	"""Return one supported full-state method identifier."""
	if method not in FULLY_EXTENDED_IMPLICIT_METHODS:
		raise ValueError(
			"`method` must be one of "
			+ ", ".join(repr(item) for item in FULLY_EXTENDED_IMPLICIT_METHODS)
			+ "."
		)
	return method


@dataclass(frozen=True, slots=True)
class FullyExtendedImplicitConfig:
	"""Refinement, projection, and differentiation controls."""

	steps: tuple[float, ...]
	t_span: tuple[float, float]
	output_sample_count: int = 201
	rho: float | None = None
	coupling_frequency: float = float(np.pi / 8.0)
	newton_absolute_tolerance: float = 1e-14
	newton_relative_tolerance: float = 1e-13
	newton_max_iterations: int = 30
	symplecticity_jacobian_relative_step: float = float(
		np.cbrt(np.finfo(float).eps)
	)
	progress: bool = False

	def __post_init__(self) -> None:
		"""Validate all study controls before allocating simulations."""
		object.__setattr__(self, "steps", _validated_steps(tuple(self.steps)))
		try:
			start, stop = (float(value) for value in self.t_span)
		except (TypeError, ValueError) as exc:
			raise ValueError("`t_span` must contain two increasing times.") from exc
		if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
			raise ValueError("`t_span` must contain two increasing finite times.")
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
			"symplecticity_jacobian_relative_step",
		):
			object.__setattr__(self, name, positive_finite(getattr(self, name), name))
		object.__setattr__(
			self,
			"newton_max_iterations",
			positive_integer(self.newton_max_iterations, "newton_max_iterations"),
		)
		object.__setattr__(self, "progress", bool(self.progress))


@dataclass(frozen=True, slots=True)
class FullyExtendedImplicitRun:
	"""One complete step-size run with energy and form-defect histories."""

	step: float
	actual_step: float
	solution: Solution
	energy_records: tuple[GCFullyExtendedEnergyRecord, ...]
	symplecticity_records: tuple[GCFullyExtendedSymplecticityRecord, ...]

	@property
	def times(self) -> np.ndarray:
		"""Accepted-node times including the initial node."""
		return np.asarray([record.time for record in self.energy_records])

	@property
	def hamiltonian(self) -> np.ndarray:
		"""Physical non-autonomous Hamiltonian history."""
		return np.asarray([record.hamiltonian for record in self.energy_records])

	@property
	def k(self) -> np.ndarray:
		"""Directly integrated time-conjugate momentum history."""
		return np.asarray([record.momentum for record in self.energy_records])

	@property
	def kappa(self) -> np.ndarray:
		"""Return ``k`` through the common generalized-energy plotting protocol."""
		return self.k

	@property
	def generalized_energy(self) -> np.ndarray:
		"""Autonomous ``K=h+k`` history."""
		return np.asarray([record.generalized_energy for record in self.energy_records])

	@property
	def energy_errors(self) -> np.ndarray:
		"""Signed ``K_n-K_0`` history."""
		return np.asarray([record.energy_error for record in self.energy_records])

	@property
	def relative_errors(self) -> np.ndarray:
		"""Signed generalized-energy relative error."""
		return np.asarray([record.relative_error for record in self.energy_records])

	@property
	def running_max_relative_errors(self) -> np.ndarray:
		"""Running envelope of the generalized-energy relative error."""
		return np.maximum.accumulate(np.abs(self.relative_errors))

	@property
	def symplecticity_times(self) -> np.ndarray:
		"""Accepted final times of the form-defect measurements."""
		return np.asarray([record.time for record in self.symplecticity_records])

	@property
	def r8_relative_defects(self) -> np.ndarray:
		"""Maximum duplicated-base-map defect per accepted step."""
		return np.asarray(
			[record.maximum_r8_relative_defect for record in self.symplecticity_records]
		)

	@property
	def r8_determinant_errors(self) -> np.ndarray:
		"""Maximum duplicated-base-map determinant error per step."""
		return np.asarray(
			[
				record.maximum_r8_determinant_error
				for record in self.symplecticity_records
			]
		)

	@property
	def dpsi_jacobian_audit_errors(self) -> np.ndarray:
		"""Relative centered-difference audits of the analytic ``D Psi``."""
		return np.asarray(
			[
				record.maximum_dpsi_jacobian_audit_error
				for record in self.symplecticity_records
			]
		)

	@property
	def dr_jacobian_audit_errors(self) -> np.ndarray:
		"""Relative centered-difference audits of the analytic ``D R``."""
		return np.asarray(
			[
				record.maximum_dr_jacobian_audit_error
				for record in self.symplecticity_records
			]
		)

	@property
	def r4_relative_defects(self) -> np.ndarray:
		"""Complete projected physical-map defect per accepted step."""
		return np.asarray([record.r4_relative_defect for record in self.symplecticity_records])

	@property
	def r4_determinant_errors(self) -> np.ndarray:
		"""Complete projected physical-map determinant error per step."""
		return np.asarray([record.r4_determinant_error for record in self.symplecticity_records])

	@property
	def r4_jacobian_audit_errors(self) -> np.ndarray:
		"""Relative centered-difference audits of the analytic projected tangent."""
		return np.asarray(
			[record.r4_jacobian_audit_error for record in self.symplecticity_records]
		)


@dataclass(frozen=True, slots=True)
class FullyExtendedImplicitSummary:
	"""Energy, symplecticity, and nonlinear-work maxima for one step."""

	method_name: str
	step: float
	actual_step: float
	step_count: int
	max_absolute_error: float
	final_absolute_error: float
	max_relative_error: float
	linear_drift_rate: float
	max_r8_relative_defect: float
	max_r8_determinant_error: float
	max_r4_relative_defect: float
	max_r4_determinant_error: float
	max_dpsi_jacobian_audit_error: float
	max_dr_jacobian_audit_error: float
	max_r4_jacobian_audit_error: float
	max_newton_iterations: int
	mean_newton_iterations: float
	max_residual_evaluations: int
	mean_residual_evaluations: float
	max_newton_residual_norm: float
	max_projection_multiplier_norm: float


@dataclass(frozen=True, slots=True)
class FullyExtendedImplicitOrder:
	"""Adjacent generalized-energy refinement slope."""

	coarse_step: float
	fine_step: float
	maximum_error_order: float


@dataclass(frozen=True, slots=True)
class FullyExtendedImplicitResult:
	"""Four-step study result for one full-state implicit method."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	method: FullyExtendedImplicitMethod
	runs: tuple[FullyExtendedImplicitRun, ...]

	@property
	def method_name(self) -> str:
		"""Exact user-facing method identifier."""
		return FULLY_EXTENDED_IMPLICIT_LABELS[self.method]

	def summaries(self) -> tuple[FullyExtendedImplicitSummary, ...]:
		"""Reduce all histories to one scalar row per step size."""
		rows: list[FullyExtendedImplicitSummary] = []
		for run in self.runs:
			errors = run.energy_errors
			iterations = np.asarray(run.solution.diagnostics["newton_iterations"])
			evaluations = np.asarray(run.solution.diagnostics["residual_evaluations"])
			residuals = np.asarray(run.solution.diagnostics["newton_residual_norms"])
			multipliers = np.asarray(
				run.solution.diagnostics["projection_multiplier_norms"]
			)
			rows.append(
				FullyExtendedImplicitSummary(
					method_name=self.method_name,
					step=run.step,
					actual_step=run.actual_step,
					step_count=run.solution.n_steps,
					max_absolute_error=float(np.max(np.abs(errors))),
					final_absolute_error=float(errors[-1]),
					max_relative_error=float(np.max(np.abs(run.relative_errors))),
					linear_drift_rate=float(
						np.polyfit(run.times - run.times[0], errors, deg=1)[0]
					),
					max_r8_relative_defect=float(np.max(run.r8_relative_defects)),
					max_r8_determinant_error=float(
						np.max(run.r8_determinant_errors)
					),
					max_r4_relative_defect=float(np.max(run.r4_relative_defects)),
					max_r4_determinant_error=float(
						np.max(run.r4_determinant_errors)
					),
					max_dpsi_jacobian_audit_error=float(
						np.max(run.dpsi_jacobian_audit_errors)
					),
					max_dr_jacobian_audit_error=float(
						np.max(run.dr_jacobian_audit_errors)
					),
					max_r4_jacobian_audit_error=float(
						np.max(run.r4_jacobian_audit_errors)
					),
					max_newton_iterations=int(np.max(iterations)),
					mean_newton_iterations=float(np.mean(iterations)),
					max_residual_evaluations=int(np.max(evaluations)),
					mean_residual_evaluations=float(np.mean(evaluations)),
					max_newton_residual_norm=float(np.max(residuals)),
					max_projection_multiplier_norm=float(np.max(multipliers)),
				)
			)
		return tuple(rows)

	def convergence_orders(self) -> tuple[FullyExtendedImplicitOrder, ...]:
		"""Return adjacent maximum-energy-error refinement slopes."""
		rows = self.summaries()
		return tuple(
			FullyExtendedImplicitOrder(
				coarse_step=coarse.actual_step,
				fine_step=fine.actual_step,
				maximum_error_order=float(
					np.log(coarse.max_absolute_error / fine.max_absolute_error)
					/ np.log(coarse.actual_step / fine.actual_step)
				),
			)
			for coarse, fine in zip(rows, rows[1:])
		)

	def print_summary(self) -> None:
		"""Print energy, ``R^8``/``R^4`` defects, and nonlinear work."""
		print(f"{self.method_name}: fully duplicated (z,t,k) study")
		print(
			f"{'h':>8} {'steps':>7} {'max |dK|':>13} {'max R8':>12} "
			f"{'max R4':>12} {'Dpsi audit':>12} {'DR audit':>11} "
			f"{'Newton':>8} {'R eval':>8}"
		)
		for row in self.summaries():
			print(
				f"{row.actual_step:8.4g} {row.step_count:7d} "
				f"{row.max_absolute_error:13.5e} "
				f"{row.max_r8_relative_defect:12.4e} "
				f"{row.max_r4_relative_defect:12.4e} "
				f"{row.max_dpsi_jacobian_audit_error:12.3e} "
				f"{row.max_dr_jacobian_audit_error:11.3e} "
				f"{row.mean_newton_iterations:8.3f} "
				f"{row.mean_residual_evaluations:8.3f}"
			)
		print("\nAdjacent maximum-energy-error orders:")
		for order in self.convergence_orders():
			print(
				f"  h={order.coarse_step:g} -> {order.fine_step:g}: "
				f"p={order.maximum_error_order:.4f}"
			)


def _method_for_run(
	method: FullyExtendedImplicitMethod,
	config: FullyExtendedImplicitConfig,
	observer: StepObserver,
) -> NumericalMethod:
	"""Construct one requested method with the shared observer."""
	if method == "abba2_fully_extended_implicit":
		return ABBA2FullyExtendedImplicit(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		)
	if method == "abba4_fully_extended_implicit":
		return ABBA4FullyExtendedImplicit(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		)
	return BM4_implicit2(
		newton_absolute_tolerance=config.newton_absolute_tolerance,
		newton_relative_tolerance=config.newton_relative_tolerance,
		newton_max_iterations=config.newton_max_iterations,
		coupling_frequency=config.coupling_frequency,
		progress=config.progress,
		step_observer=observer,
	)


def run_fully_extended_implicit_study(
	potential: Potential,
	configuration: GCInitialConfiguration,
	*,
	method: FullyExtendedImplicitMethod,
	config: FullyExtendedImplicitConfig,
) -> FullyExtendedImplicitResult:
	"""Run one method over the configured energy/symplecticity refinement."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(configuration, GCInitialConfiguration):
		raise TypeError("`configuration` must be GCInitialConfiguration.")
	if not isinstance(config, FullyExtendedImplicitConfig):
		raise TypeError("`config` must be FullyExtendedImplicitConfig.")
	method_name = _validated_method(method)
	physical_initial = configuration.initial_state
	if physical_initial is None or configuration.layout.particle_count(physical_initial) != 1:
		raise ValueError("The fully extended study requires exactly one particle.")
	dynamics = GuidingCenterDynamics(
		potential,
		rho=resolve_rho(config.rho, configuration),
	)
	problem = InitialValueProblem(dynamics, configuration)
	initial_extended = np.concatenate(
		(physical_initial, (config.t_span[0], 0.0))
	)
	runs: list[FullyExtendedImplicitRun] = []
	for step in config.steps:
		energy_observer = GCFullyExtendedEnergyObserver(
			dynamics,
			initial_state=initial_extended,
		)
		symplecticity_observer = GCFullyExtendedSymplecticityObserver(
			dynamics,
			relative_step=config.symplecticity_jacobian_relative_step,
		)

		def observer(record: IntegrationStep) -> None:
			energy_observer(record)
			symplecticity_observer(record)

		solution = simulate(
			problem,
			_method_for_run(method_name, config, observer),
			SimulationRequest.uniform(
				t_span=config.t_span,
				max_step=step,
				sample_count=config.output_sample_count,
			),
		)
		energy_records = energy_observer.records
		symplecticity_records = symplecticity_observer.records
		if len(energy_records) != solution.n_steps + 1:
			raise RuntimeError("Energy records do not match the accepted step count.")
		if len(symplecticity_records) != solution.n_steps:
			raise RuntimeError(
				"Symplecticity records do not match the accepted step count."
			)
		runs.append(
			FullyExtendedImplicitRun(
				step=step,
				actual_step=float(energy_records[1].duration),
				solution=solution,
				energy_records=energy_records,
				symplecticity_records=symplecticity_records,
			)
		)
	return FullyExtendedImplicitResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=configuration,
		method=method_name,
		runs=tuple(runs),
	)


__all__ = [
	"FULLY_EXTENDED_IMPLICIT_LABELS",
	"FULLY_EXTENDED_IMPLICIT_METHODS",
	"FullyExtendedImplicitConfig",
	"FullyExtendedImplicitMethod",
	"FullyExtendedImplicitOrder",
	"FullyExtendedImplicitResult",
	"FullyExtendedImplicitRun",
	"FullyExtendedImplicitSummary",
	"run_fully_extended_implicit_study",
]
