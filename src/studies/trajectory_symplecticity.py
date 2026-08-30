"""Exact averaged symplecticity studies for independent GC trajectories."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics import (
	GCTrajectorySymplecticityObserver,
	TrajectoryJacobianCalculator,
	TrajectorySymplecticityRecord,
	abba4_implicit_step_particle_jacobians,
	bm4_implicit_1_step_particle_jacobians,
	abba2_implicit_step_particle_jacobians,
	abba2_midpoint_step_particle_jacobians,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA4Implicit,
	BM4Implicit1,
	ABBA2Implicit,
	InitialValueProblem,
	ABBA2Midpoint,
	NumericalMethod,
	SimulationRequest,
	Solution,
	StepObserver,
	simulate,
)
from visualization import (
	TrajectorySymplecticityRecordView,
	plot_gc_trajectory_points,
	plot_trajectory_symplecticity,
)

from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)
from .area_comparison import AreaStep


_BLOCK_PREFIX = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class TrajectorySymplecticityConfig:
	"""Shared physical grids and solver controls for trajectory studies."""

	steps: tuple[AreaStep, ...]
	t_span: tuple[float, float]
	save_interval: float
	rho: float = 0.0
	chunk_size: int = 64
	progress: bool = False
	block_prefix: str = "trajectory_symplecticity"
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 16
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	coupling_frequency: float = float(np.pi / 8.0)

	def __post_init__(self) -> None:
		"""Validate aligned output grids and nonlinear solver controls."""
		steps = tuple(self.steps)
		if not steps or any(not isinstance(step, AreaStep) for step in steps):
			raise ValueError("`steps` must contain at least one AreaStep value.")
		if len({step.label for step in steps}) != len(steps):
			raise ValueError("Integration-step labels must be unique.")
		if len({step.value for step in steps}) != len(steps):
			raise ValueError("Integration-step values must be unique.")
		if any(coarse.value <= fine.value for coarse, fine in zip(steps, steps[1:])):
			raise ValueError("Steps must be ordered from coarsest to finest.")
		object.__setattr__(self, "steps", steps)

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
			"save_interval",
			positive_finite(self.save_interval, "save_interval"),
		)
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"chunk_size",
			positive_integer(self.chunk_size, "chunk_size"),
		)
		for name in (
			"newton_absolute_tolerance",
			"newton_relative_tolerance",
			"newton_jacobian_relative_step",
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
		object.__setattr__(
			self,
			"coupling_frequency",
			nonnegative_finite(self.coupling_frequency, "coupling_frequency"),
		)
		if not isinstance(self.block_prefix, str) or not _BLOCK_PREFIX.fullmatch(
			self.block_prefix
		):
			raise ValueError(
				"`block_prefix` may contain only letters, numbers, '_' and '-'."
			)
		integer_ratio(stop - start, self.save_interval, "duration / save_interval")
		for step in steps:
			integer_ratio(
				self.save_interval,
				step.value,
				f"save_interval / step for {step.label}",
			)

	@property
	def output_sample_count(self) -> int:
		"""Number of common saved states including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class TrajectorySymplecticitySummary:
	"""Averaged geometric and optional nonlinear diagnostics for one step."""

	label: str
	step: float
	step_count: int
	trajectory_count: int
	max_mean_local_defect: float
	max_mean_accumulated_defect: float
	final_mean_accumulated_defect: float
	max_individual_accumulated_defect: float
	max_newton_iterations: int | None
	mean_newton_iterations: float | None
	max_newton_residual_norm: float | None


@dataclass(frozen=True, slots=True)
class TrajectoryDefectOrder:
	"""Empirical accumulated-defect slopes for one step refinement."""

	coarse_label: str
	fine_label: str
	maximum_accumulated_defect: float
	final_accumulated_defect: float


def _empirical_order(
	coarse_error: float,
	fine_error: float,
	coarse_step: float,
	fine_step: float,
) -> float:
	"""Return a refinement log slope or NaN when it is undefined."""
	if coarse_error <= 0.0 or fine_error <= 0.0:
		return float("nan")
	return float(
		np.log(coarse_error / fine_error)
		/ np.log(coarse_step / fine_step)
	)


def _solver_summary(
	solution: Solution,
) -> tuple[int | None, float | None, float | None]:
	"""Reduce optional Newton iteration and residual arrays."""
	iterations_value = solution.diagnostics.get("newton_iterations")
	residuals_value = solution.diagnostics.get("newton_residual_norms")
	if iterations_value is None and residuals_value is None:
		return None, None, None
	if iterations_value is None or residuals_value is None:
		raise ValueError("Newton iterations and residuals must be provided together.")
	iterations = np.asarray(iterations_value, dtype=float)
	residuals = np.asarray(residuals_value, dtype=float)
	if (
		iterations.ndim != 1
		or residuals.shape != iterations.shape
		or iterations.size != solution.n_steps
		or not np.all(np.isfinite(iterations))
		or not np.all(np.isfinite(residuals))
	):
		raise ValueError("Invalid Newton diagnostics in the numerical solution.")
	return (
		int(np.max(iterations)),
		float(np.mean(iterations)),
		float(np.max(residuals)),
	)


@dataclass(frozen=True, slots=True)
class TrajectorySymplecticityResult:
	"""Solutions and exact per-trajectory tangent diagnostics for one method."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	method_name: str
	jacobian_method: str
	steps: tuple[AreaStep, ...]
	solutions: Mapping[str, Solution]
	records: Mapping[str, tuple[TrajectorySymplecticityRecord, ...]]
	output_directories: Mapping[str, Path]

	def summaries(self) -> tuple[TrajectorySymplecticitySummary, ...]:
		"""Return geometric and nonlinear maxima in configured step order."""
		rows: list[TrajectorySymplecticitySummary] = []
		for step in self.steps:
			records = self.records[step.label]
			if not records:
				raise ValueError(f"No records exist for {step.label!r}.")
			solution = self.solutions[step.label]
			max_iterations, mean_iterations, max_residual = _solver_summary(solution)
			rows.append(
				TrajectorySymplecticitySummary(
					label=step.label,
					step=step.value,
					step_count=solution.n_steps,
					trajectory_count=records[0].particle_count,
					max_mean_local_defect=max(
						record.mean_local_relative_defect for record in records
					),
					max_mean_accumulated_defect=max(
						record.mean_accumulated_relative_defect for record in records
					),
					final_mean_accumulated_defect=(
						records[-1].mean_accumulated_relative_defect
					),
					max_individual_accumulated_defect=max(
						record.max_accumulated_relative_defect for record in records
					),
					max_newton_iterations=max_iterations,
					mean_newton_iterations=mean_iterations,
					max_newton_residual_norm=max_residual,
				)
			)
		return tuple(rows)

	def convergence_orders(self) -> tuple[TrajectoryDefectOrder, ...]:
		"""Estimate accumulated-defect slopes under step refinement."""
		rows = self.summaries()
		return tuple(
			TrajectoryDefectOrder(
				coarse_label=coarse.label,
				fine_label=fine.label,
				maximum_accumulated_defect=_empirical_order(
					coarse.max_mean_accumulated_defect,
					fine.max_mean_accumulated_defect,
					coarse.step,
					fine.step,
				),
				final_accumulated_defect=_empirical_order(
					coarse.final_mean_accumulated_defect,
					fine.final_mean_accumulated_defect,
					coarse.step,
					fine.step,
				),
			)
			for coarse, fine in zip(rows, rows[1:])
		)

	def print_summary(self) -> None:
		"""Print averaged errors, solver work, and refinement slopes."""
		print(f"{self.method_name} / {self.jacobian_method}")
		print(
			f"{'step':>12} {'steps':>8} {'paths':>7} "
			f"{'max mean local':>18} {'max mean flow':>18} "
			f"{'final mean flow':>18} {'max trajectory':>18}"
		)
		rows = self.summaries()
		for row in rows:
			print(
				f"{row.label:>12} {row.step_count:8d} {row.trajectory_count:7d} "
				f"{row.max_mean_local_defect:18.8e} "
				f"{row.max_mean_accumulated_defect:18.8e} "
				f"{row.final_mean_accumulated_defect:18.8e} "
				f"{row.max_individual_accumulated_defect:18.8e}"
			)
		if any(row.max_newton_iterations is not None for row in rows):
			print("\nNewton solve diagnostics:")
			for row in rows:
				print(
					f"  {row.label}: max/mean iterations="
					f"{row.max_newton_iterations}/{row.mean_newton_iterations:.3f}, "
					f"max residual={row.max_newton_residual_norm:.3e}"
				)
		print("\nEmpirical orders (maximum / final mean accumulated defect):")
		for order in self.convergence_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.maximum_accumulated_defect:.6f} / "
				f"{order.final_accumulated_defect:.6f}"
			)

	def plot_symplecticity(self) -> tuple[Figure, np.ndarray]:
		"""Plot arithmetic-mean local and accumulated errors over time."""
		return plot_trajectory_symplecticity(
			cast(
				Mapping[str, Sequence[TrajectorySymplecticityRecordView]],
				self.records,
			),
			labels=tuple(step.label for step in self.steps),
			method_name=self.method_name,
		)

	def plot_trajectories(
		self,
		*,
		label: str | None = None,
	) -> tuple[Figure, Axes]:
		"""Plot saved trajectory samples as unconnected points."""
		selected = (
			min(self.steps, key=lambda step: step.value).label
			if label is None
			else label
		)
		if selected not in self.solutions:
			raise ValueError(f"Unknown integration-step label {selected!r}.")
		return plot_gc_trajectory_points(
			self.solutions[selected],
			method_name=self.method_name,
			step_label=selected,
		)


_MethodFactory = Callable[[StepObserver], NumericalMethod]


def _run_trajectory_symplecticity_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: TrajectorySymplecticityConfig,
	method_name: str,
	method_slug: str,
	jacobian_method: str,
	jacobian_calculator: TrajectoryJacobianCalculator,
	method_factory: _MethodFactory,
	project_root: str | Path | None,
	metadata: Mapping[str, Any] | None,
) -> TrajectorySymplecticityResult:
	"""Run one method on the same initial trajectories for every step size."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError("`initial_configuration` must be a GC configuration.")
	if not isinstance(config, TrajectorySymplecticityConfig):
		raise TypeError("`config` must be a TrajectorySymplecticityConfig.")
	initial_state = initial_configuration.initial_state
	if initial_state is None:
		raise ValueError("The GC initial configuration has no initial state.")
	if potential.interpolation_order < 3:
		raise ValueError("Exact GC Jacobians require interpolation_order >= 3.")

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	trajectory_count = initial_configuration.layout.particle_count(initial_state)
	common_metadata = {
		**dict(metadata or {}),
		"method": method_name,
		"rho": config.rho,
		"trajectory_count": trajectory_count,
		"t_span": config.t_span,
		"save_interval": config.save_interval,
		"study_config": asdict(config),
		"jacobian_method": jacobian_method,
		"symplecticity_reduction": (
			"arithmetic_mean_of_per_trajectory_relative_defects"
		),
	}
	solutions: dict[str, Solution] = {}
	records: dict[str, tuple[TrajectorySymplecticityRecord, ...]] = {}
	output_directories: dict[str, Path] = {}
	for step in config.steps:
		record_every = integer_ratio(
			config.save_interval,
			step.value,
			f"save_interval / step for {step.label}",
		)
		step_tag = f"{step.value:.8f}".replace(".", "p")
		observer = GCTrajectorySymplecticityObserver(
			dynamics=dynamics,
			initial_configuration=initial_configuration,
			method_name=method_name,
			jacobian_method=jacobian_method,
			jacobian_calculator=jacobian_calculator,
			notebook_path=notebook_path,
			project_root=project_root,
			block_name=(
				f"{config.block_prefix}_{method_slug}_step_{step_tag}"
			),
			record_every=record_every,
			chunk_size=config.chunk_size,
			verbose=False,
			metadata={**common_metadata, "integration_step": step.value},
		)
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step.value,
			sample_count=config.output_sample_count,
		)
		with observer:
			solution = simulate(problem, method_factory(observer), request)
		solutions[step.label] = solution
		records[step.label] = observer.records
		output_directories[step.label] = observer.output_directory
	return TrajectorySymplecticityResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		method_name=method_name,
		jacobian_method=jacobian_method,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		records=MappingProxyType(records),
		output_directories=MappingProxyType(output_directories),
	)


def run_abba2_midpoint_trajectory_symplecticity_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: TrajectorySymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> TrajectorySymplecticityResult:
	"""Run midpoint ABBA with its explicit four-shear tangent."""
	return _run_trajectory_symplecticity_study(
		potential,
		initial_configuration,
		notebook_path=notebook_path,
		config=config,
		method_name="ABBA2Midpoint",
		method_slug="abba2_midpoint",
		jacobian_method="explicit_abba_stage_chain",
		jacobian_calculator=abba2_midpoint_step_particle_jacobians,
		method_factory=lambda observer: ABBA2Midpoint(
			progress=config.progress,
			step_observer=observer,
		),
		project_root=project_root,
		metadata={
			**dict(metadata or {}),
			"projection_kind": "arithmetic_mean",
			"projection_scope": "complete_abba_cycle",
		},
	)


def run_abba2_reduced_multiplier_trajectory_symplecticity_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: TrajectorySymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> TrajectorySymplecticityResult:
	"""Run reduced-multiplier ABBA2 with an exact ideal-root tangent."""
	return _run_trajectory_symplecticity_study(
		potential,
		initial_configuration,
		notebook_path=notebook_path,
		config=config,
		method_name="ABBA2Implicit",
		method_slug="abba2_implicit_reduced_multiplier",
		jacobian_method="explicit_implicit_function_theorem",
		jacobian_calculator=abba2_implicit_step_particle_jacobians,
		method_factory=lambda observer: ABBA2Implicit(
			projection_formulation="reduced_multiplier",
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		project_root=project_root,
		metadata={
			**dict(metadata or {}),
			"projection_formulation": "reduced_multiplier",
			"nonlinear_solver": "newton",
		},
	)


def run_abba4_implicit_trajectory_symplecticity_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: TrajectorySymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> TrajectorySymplecticityResult:
	"""Run reduced ABBA4 and compose its three exact physical tangents."""
	root_two = float(np.cbrt(2.0))
	gamma = 1.0 / (2.0 - root_two)
	delta = -root_two / (2.0 - root_two)
	return _run_trajectory_symplecticity_study(
		potential,
		initial_configuration,
		notebook_path=notebook_path,
		config=config,
		method_name="ABBA4Implicit",
		method_slug="abba4_implicit",
		jacobian_method="three_substep_explicit_implicit_function_theorem",
		jacobian_calculator=abba4_implicit_step_particle_jacobians,
		method_factory=lambda observer: ABBA4Implicit(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		project_root=project_root,
		metadata={
			**dict(metadata or {}),
			"projection_formulation": "reduced_multiplier",
			"composition_policy": "project_each_abba_substep",
			"nonlinear_solver": "newton",
			"composition_coefficients": (gamma, delta, gamma),
			"signed_substeps": True,
			"independent_multiplier_per_substep": True,
			"tangent_definition": "ideal_converged_projection_root",
		},
	)


def run_bm4_implicit_1_trajectory_symplecticity_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: TrajectorySymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> TrajectorySymplecticityResult:
	"""Run implicit BM4 formulation 1 with exact base-cycle differentiation."""
	return _run_trajectory_symplecticity_study(
		potential,
		initial_configuration,
		notebook_path=notebook_path,
		config=config,
		method_name="BM4Implicit1",
		method_slug="bm4_implicit_1",
		jacobian_method=(
			"explicit_coupled_bm4_stages_and_implicit_function_theorem"
		),
		jacobian_calculator=bm4_implicit_1_step_particle_jacobians,
		method_factory=lambda observer: BM4Implicit1(
			coupling_frequency=config.coupling_frequency,
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			newton_jacobian_relative_step=config.newton_jacobian_relative_step,
			progress=config.progress,
			step_observer=observer,
		),
		project_root=project_root,
		metadata={
			**dict(metadata or {}),
			"implicit_formulation": "bm4_implicit_1_reduced",
			"nonlinear_solver": "newton",
			"coupling_frequency": config.coupling_frequency,
		},
	)


__all__ = [
	"TrajectoryDefectOrder",
	"TrajectorySymplecticityConfig",
	"TrajectorySymplecticityResult",
	"TrajectorySymplecticitySummary",
	"run_abba4_implicit_trajectory_symplecticity_study",
	"run_bm4_implicit_1_trajectory_symplecticity_study",
	"run_abba2_reduced_multiplier_trajectory_symplecticity_study",
	"run_abba2_midpoint_trajectory_symplecticity_study",
]
