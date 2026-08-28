"""Averaged exact-Jacobian symplecticity studies for midpoint BM4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics import (
	MidpointBM4SymplecticityObserver,
	MidpointBM4SymplecticityRecord,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	InitialValueProblem,
	MidpointBM4,
	SimulationRequest,
	Solution,
	simulate,
)
from visualization import (
	plot_midpoint_bm4_symplecticity,
	plot_midpoint_bm4_trajectories,
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
class MidpointBM4SymplecticityConfig:
	"""Reproducible time grids for an exact midpoint-BM4 tangent study."""

	steps: tuple[AreaStep, ...]
	t_span: tuple[float, float]
	save_interval: float
	rho: float = 0.0
	chunk_size: int = 16
	progress: bool = False
	block_prefix: str = "bm4_midpoint_symplecticity"

	def __post_init__(self) -> None:
		"""Validate integration, sampling, and persistence parameters."""
		steps = tuple(self.steps)
		if not steps or any(not isinstance(step, AreaStep) for step in steps):
			raise ValueError("`steps` must contain at least one AreaStep value.")
		if len({step.label for step in steps}) != len(steps):
			raise ValueError("Midpoint-BM4 integration-step labels must be unique.")
		if len({step.value for step in steps}) != len(steps):
			raise ValueError("Midpoint-BM4 integration-step values must be unique.")
		if any(coarse.value <= fine.value for coarse, fine in zip(steps, steps[1:])):
			raise ValueError(
				"Midpoint-BM4 steps must be ordered from coarsest to finest."
			)
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

		save_interval = positive_finite(self.save_interval, "save_interval")
		object.__setattr__(self, "save_interval", save_interval)
		object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		object.__setattr__(
			self,
			"chunk_size",
			positive_integer(self.chunk_size, "chunk_size"),
		)
		if not isinstance(self.block_prefix, str) or not _BLOCK_PREFIX.fullmatch(
			self.block_prefix
		):
			raise ValueError(
				"`block_prefix` may contain only letters, numbers, '_' and '-'."
			)

		integer_ratio(stop - start, save_interval, "duration / save_interval")
		for step in steps:
			integer_ratio(
				save_interval,
				step.value,
				f"save_interval / step for {step.label}",
			)

	@property
	def output_sample_count(self) -> int:
		"""Number of uniformly saved physical states including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class MidpointBM4SymplecticitySummary:
	"""Averaged and worst-trajectory errors for one integration step."""

	label: str
	step: float
	step_count: int
	trajectory_count: int
	max_mean_local_defect: float
	max_mean_accumulated_defect: float
	final_mean_accumulated_defect: float
	max_individual_accumulated_defect: float


@dataclass(frozen=True, slots=True)
class MidpointBM4DefectOrder:
	"""Empirical averaged-defect orders between two consecutive step sizes."""

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
	"""Return a refinement slope or NaN when the logarithm is undefined."""
	if (
		coarse_error <= 0.0
		or fine_error <= 0.0
		or np.isclose(coarse_step, fine_step)
	):
		return float("nan")
	return float(
		np.log(coarse_error / fine_error)
		/ np.log(coarse_step / fine_step)
	)


@dataclass(frozen=True, slots=True)
class MidpointBM4SymplecticityResult:
	"""Trajectory solutions and exact tangent-flow diagnostics."""

	potential: Potential
	dynamics: GuidingCenterDynamics
	initial_configuration: GCInitialConfiguration
	steps: tuple[AreaStep, ...]
	solutions: Mapping[str, Solution]
	records: Mapping[str, tuple[MidpointBM4SymplecticityRecord, ...]]
	output_directories: Mapping[str, Path]

	@property
	def diagnostic_times(self) -> Mapping[str, np.ndarray]:
		"""Return record times for each configured integration step."""
		return MappingProxyType(
			{
				label: np.asarray([record.time for record in rows], dtype=float)
				for label, rows in self.records.items()
			}
		)

	@property
	def mean_symplecticity_errors(self) -> Mapping[str, np.ndarray]:
		"""Return arithmetic means of accumulated per-trajectory defects."""
		return MappingProxyType(
			{
				label: np.asarray(
					[
						record.mean_accumulated_relative_defect
						for record in rows
					],
					dtype=float,
				)
				for label, rows in self.records.items()
			}
		)

	def summaries(self) -> tuple[MidpointBM4SymplecticitySummary, ...]:
		"""Aggregate the averaged and largest individual measured errors."""
		initial_state = self.initial_configuration.initial_state
		assert initial_state is not None
		trajectory_count = self.initial_configuration.layout.particle_count(initial_state)
		rows: list[MidpointBM4SymplecticitySummary] = []
		for step in self.steps:
			records = self.records[step.label]
			if not records:
				raise ValueError(f"No symplecticity records exist for {step.label!r}.")
			rows.append(
				MidpointBM4SymplecticitySummary(
					label=step.label,
					step=step.value,
					step_count=int(
						self.solutions[step.label].diagnostics["step_count"]
					),
					trajectory_count=trajectory_count,
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
				)
			)
		return tuple(rows)

	def convergence_orders(self) -> tuple[MidpointBM4DefectOrder, ...]:
		"""Estimate averaged accumulated-defect slopes under refinement."""
		summaries = self.summaries()
		return tuple(
			MidpointBM4DefectOrder(
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
			for coarse, fine in zip(summaries, summaries[1:])
		)

	def print_summary(self) -> None:
		"""Print averaged errors and refinement slopes."""
		print(
			f"{'step':>14} {'steps':>9} {'trajectories':>13} "
			f"{'max mean local':>18} {'max mean flow':>18} "
			f"{'final mean flow':>18} {'max trajectory':>18}"
		)
		for row in self.summaries():
			print(
				f"{row.label:>14} {row.step_count:9d} "
				f"{row.trajectory_count:13d} {row.max_mean_local_defect:18.8e} "
				f"{row.max_mean_accumulated_defect:18.8e} "
				f"{row.final_mean_accumulated_defect:18.8e} "
				f"{row.max_individual_accumulated_defect:18.8e}"
			)
		print("\nEmpirical orders (maximum mean / final mean accumulated defect):")
		for order in self.convergence_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.maximum_accumulated_defect:.6f} / "
				f"{order.final_accumulated_defect:.6f}"
			)

	def plot_symplecticity(self) -> tuple[Figure, np.ndarray]:
		"""Plot averaged local and accumulated defects against time."""
		return plot_midpoint_bm4_symplecticity(
			self.records,
			labels=tuple(step.label for step in self.steps),
		)

	def plot_trajectories(
		self,
		*,
		label: str | None = None,
	) -> tuple[Figure, Axes]:
		"""Plot all paths from one step size, defaulting to the finest."""
		selected = (
			min(self.steps, key=lambda step: step.value).label
			if label is None
			else label
		)
		if selected not in self.solutions:
			raise ValueError(f"Unknown midpoint-BM4 step label {selected!r}.")
		return plot_midpoint_bm4_trajectories(
			self.solutions[selected],
			step_label=selected,
		)


def run_midpoint_bm4_symplecticity_study(
	potential: Potential,
	initial_configuration: GCInitialConfiguration,
	*,
	notebook_path: str | Path,
	config: MidpointBM4SymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> MidpointBM4SymplecticityResult:
	"""Run exact tangent diagnostics for the same trajectories at each step."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(initial_configuration, GCInitialConfiguration):
		raise TypeError(
			"`initial_configuration` must be a GCInitialConfiguration instance."
		)
	if not isinstance(config, MidpointBM4SymplecticityConfig):
		raise TypeError(
			"`config` must be a MidpointBM4SymplecticityConfig instance."
		)
	initial_state = initial_configuration.initial_state
	if initial_state is None:
		raise ValueError("The GC initial configuration has no initial state.")
	if potential.interpolation_order < 3:
		raise ValueError(
			"Explicit midpoint-BM4 Jacobians require interpolation_order >= 3."
		)

	dynamics = GuidingCenterDynamics(potential, rho=config.rho)
	problem = InitialValueProblem(dynamics, initial_configuration)
	trajectory_count = initial_configuration.layout.particle_count(initial_state)
	common_metadata = {
		**dict(metadata or {}),
		"method": "MidpointBM4",
		"rho": config.rho,
		"trajectory_count": trajectory_count,
		"t_span": config.t_span,
		"save_interval": config.save_interval,
		"study_config": asdict(config),
		"projection_kind": "arithmetic_mean",
		"projection_scope": "complete_bm4_cycle",
		"projections_per_step": 1,
		"bm4_stage_count": 12,
		"coupled": False,
		"jacobian_method": "explicit_uncoupled_stage_factorization",
		"symplecticity_reduction": (
			"arithmetic_mean_of_per_trajectory_relative_defects"
		),
	}
	solutions: dict[str, Solution] = {}
	records: dict[str, tuple[MidpointBM4SymplecticityRecord, ...]] = {}
	output_directories: dict[str, Path] = {}

	for step in config.steps:
		record_every = integer_ratio(
			config.save_interval,
			step.value,
			f"save_interval / step for {step.label}",
		)
		step_tag = f"{step.value:.8f}".replace(".", "p")
		observer = MidpointBM4SymplecticityObserver(
			dynamics=dynamics,
			notebook_path=notebook_path,
			project_root=project_root,
			block_name=f"{config.block_prefix}_step_{step_tag}",
			record_every=record_every,
			chunk_size=config.chunk_size,
			verbose=False,
			metadata={
				**common_metadata,
				"integration_step": step.value,
			},
		)
		request = SimulationRequest.uniform(
			t_span=config.t_span,
			max_step=step.value,
			sample_count=config.output_sample_count,
		)
		with observer:
			solution = simulate(
				problem,
				MidpointBM4(
					progress=config.progress,
					stage_observer=observer,
				),
				request,
			)
		solutions[step.label] = solution
		records[step.label] = observer.records
		output_directories[step.label] = observer.output_directory

	return MidpointBM4SymplecticityResult(
		potential=potential,
		dynamics=dynamics,
		initial_configuration=initial_configuration,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		records=MappingProxyType(records),
		output_directories=MappingProxyType(output_directories),
	)


__all__ = [
	"MidpointBM4DefectOrder",
	"MidpointBM4SymplecticityConfig",
	"MidpointBM4SymplecticityResult",
	"MidpointBM4SymplecticitySummary",
	"run_midpoint_bm4_symplecticity_study",
]
