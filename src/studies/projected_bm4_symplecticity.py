"""Physical symplecticity study for stage-projected BM4 composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from diagnostics.projection import (
	ProjectedAreaRecord,
	ProjectedSymplecticityAreaObserver,
)
from dynamics import GuidingCenterDynamics
from initial_conditions import Area
from potential import Potential
from simulation import (
	GCStageProjectedFormulation,
	InitialValueProblem,
	ProjectedBM4Composition,
	SimulationRequest,
	Solution,
	simulate,
)
from visualization import (
	animate_gc_area_comparison,
	plot_projected_bm4_symplecticity_convergence,
	plot_projected_bm4_symplecticity_diagnostics,
)

from ._gc_symplecticity_models import GCSymplecticityConfig
from ._validation import integer_ratio, resolve_rho
from .area_comparison import AreaStep


@dataclass(frozen=True, slots=True)
class ProjectedBM4SymplecticityConfig(GCSymplecticityConfig):
	"""Reproducible grids for a stage-projected BM4 symplecticity study."""

	block_prefix: str = "projected_bm4_symplecticity"


@dataclass(frozen=True, slots=True)
class ProjectedBM4SymplecticitySummary:
	"""Maximum physical-map defects for one complete BM4 step size."""

	label: str
	step: float
	step_count: int
	max_area_error: float
	max_local_defect: float
	max_flow_defect: float
	max_local_determinant_error: float
	max_flow_determinant_error: float
	max_relative_copy_separation: float


@dataclass(frozen=True, slots=True)
class ProjectedBM4DefectOrder:
	"""Empirical orders between two consecutive complete BM4 step sizes."""

	coarse_label: str
	fine_label: str
	local_defect: float
	flow_defect: float
	area_error: float


def _empirical_order(
	coarse_value: float,
	fine_value: float,
	coarse_step: float,
	fine_step: float,
) -> float:
	"""Return one log-ratio slope or NaN when it is not defined."""
	if (
		coarse_value <= 0.0
		or fine_value <= 0.0
		or np.isclose(coarse_step, fine_step)
	):
		return float("nan")
	return float(
		np.log(coarse_value / fine_value)
		/ np.log(coarse_step / fine_step)
	)


@dataclass(frozen=True, slots=True)
class ProjectedBM4SymplecticityResult:
	"""Stage-projected BM4 trajectories and physical symplecticity diagnostics."""

	dynamics: GuidingCenterDynamics
	area: Area
	steps: tuple[AreaStep, ...]
	solutions: Mapping[str, Solution]
	records: Mapping[str, tuple[ProjectedAreaRecord, ...]]
	output_directories: Mapping[str, Path]

	@property
	def diagnostic_times(self) -> Mapping[str, np.ndarray]:
		"""Return observation times aligned with each labeled solution."""
		return {
			label: np.asarray([record.time for record in records])
			for label, records in self.records.items()
		}

	@property
	def relative_symplecticity_errors(self) -> Mapping[str, np.ndarray]:
		"""Return accumulated physical-flow defects for animation."""
		return {
			label: np.asarray([record.relative_defect for record in records])
			for label, records in self.records.items()
		}

	def summaries(self) -> tuple[ProjectedBM4SymplecticitySummary, ...]:
		"""Aggregate local, accumulated, determinant, area, and copy defects."""
		rows: list[ProjectedBM4SymplecticitySummary] = []
		for step in self.steps:
			records = self.records[step.label]
			rows.append(
				ProjectedBM4SymplecticitySummary(
					label=step.label,
					step=step.value,
					step_count=int(
						self.solutions[step.label].diagnostics["step_count"]
					),
					max_area_error=max(
						abs(record.relative_area_error) for record in records
					),
					max_local_defect=max(
						record.local_relative_defect for record in records
					),
					max_flow_defect=max(
						record.relative_defect for record in records
					),
					max_local_determinant_error=max(
						record.local_determinant_error for record in records
					),
					max_flow_determinant_error=max(
						record.determinant_error for record in records
					),
					max_relative_copy_separation=max(
						record.relative_copy_separation for record in records
					),
				)
			)
		return tuple(rows)

	def convergence_orders(self) -> tuple[ProjectedBM4DefectOrder, ...]:
		"""Estimate refinement slopes for local, flow, and area errors."""
		rows = self.summaries()
		return tuple(
			ProjectedBM4DefectOrder(
				coarse_label=coarse.label,
				fine_label=fine.label,
				local_defect=_empirical_order(
					coarse.max_local_defect,
					fine.max_local_defect,
					coarse.step,
					fine.step,
				),
				flow_defect=_empirical_order(
					coarse.max_flow_defect,
					fine.max_flow_defect,
					coarse.step,
					fine.step,
				),
				area_error=_empirical_order(
					coarse.max_area_error,
					fine.max_area_error,
					coarse.step,
					fine.step,
				),
			)
			for coarse, fine in zip(rows, rows[1:])
		)

	def print_summary(self) -> None:
		"""Print compact maximum errors and empirical refinement slopes."""
		print(
			f"{'step':>18} {'steps':>8} {'area':>12} {'local':>12} "
			f"{'flow':>12} {'local det':>12} {'flow det':>12}"
		)
		for row in self.summaries():
			print(
				f"{row.label:>18} {row.step_count:8d} "
				f"{row.max_area_error:12.4e} {row.max_local_defect:12.4e} "
				f"{row.max_flow_defect:12.4e} "
				f"{row.max_local_determinant_error:12.4e} "
				f"{row.max_flow_determinant_error:12.4e}"
			)
		print("\nEmpirical orders (local / flow / area):")
		for order in self.convergence_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.local_defect:.4f} / {order.flow_defect:.4f} / "
				f"{order.area_error:.4f}"
			)

	def plot_diagnostics(self) -> tuple[Figure, np.ndarray]:
		"""Plot area, local, accumulated, and determinant diagnostics over time."""
		return plot_projected_bm4_symplecticity_diagnostics(self.records)

	def plot_convergence(self) -> tuple[Figure, Axes]:
		"""Plot maximum errors against the complete BM4 step size."""
		rows = self.summaries()
		return plot_projected_bm4_symplecticity_convergence(
			steps=np.asarray([row.step for row in rows]),
			local_defects=np.asarray([row.max_local_defect for row in rows]),
			flow_defects=np.asarray([row.max_flow_defect for row in rows]),
			area_errors=np.asarray([row.max_area_error for row in rows]),
		)

	def animate(
		self,
		*,
		frames: int | None = 120,
		interval: int = 50,
		repeat: bool = True,
	) -> FuncAnimation:
		"""Animate transported area and accumulated symplecticity defect."""
		return animate_gc_area_comparison(
			self.dynamics.effective_potential,
			self.area,
			self.solutions,
			diagnostic_times=self.diagnostic_times,
			relative_symplecticity_errors=self.relative_symplecticity_errors,
			frames=frames,
			interval=interval,
			repeat=repeat,
		)


def run_projected_bm4_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ProjectedBM4SymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ProjectedBM4SymplecticityResult:
	"""Run stage-projected BM4 and persist complete-step physical diagnostics."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(area, Area):
		raise TypeError("`area` must be an Area instance.")
	if not isinstance(config, ProjectedBM4SymplecticityConfig):
		raise TypeError("`config` must be a ProjectedBM4SymplecticityConfig.")

	rho = resolve_rho(config.rho, area)
	dynamics = GuidingCenterDynamics(potential, rho=rho)
	problem = InitialValueProblem(dynamics, area)
	initial_state = area.initial_state
	assert initial_state is not None
	common_metadata = {
		**dict(metadata or {}),
		"method": "ProjectedBM4Composition",
		"formulation": "GCStageProjectedFormulation",
		"projection": "arithmetic_mean_and_diagonal_reembedding",
		"projections_per_complete_step": 12,
		"particle_count": area.particle_count(initial_state),
		"rho": rho,
	}
	solutions: dict[str, Solution] = {}
	records_by_label: dict[str, tuple[ProjectedAreaRecord, ...]] = {}
	output_directories: dict[str, Path] = {}

	for step in config.steps:
		record_every = integer_ratio(
			config.save_interval,
			step.value,
			f"save_interval / step for {step.label}",
		)
		step_tag = f"{step.value:.8f}".replace(".", "p")
		with ProjectedSymplecticityAreaObserver(
			notebook_path=notebook_path,
			area=area,
			period=potential.grid.period,
			project_root=project_root,
			block_name=f"{config.block_prefix}_step_{step_tag}",
			record_every=record_every,
			chunk_size=config.chunk_size,
			relative_step=config.finite_difference_relative_step,
			verbose=False,
			metadata={**common_metadata, "integration_step": step.value},
		) as observer:
			request = SimulationRequest.uniform(
				t_span=config.t_span,
				max_step=step.value,
				sample_count=config.output_sample_count,
			)
			solution = simulate(
				problem,
				ProjectedBM4Composition(
					GCStageProjectedFormulation(),
					progress=config.progress,
					stage_observer=observer,
				),
				request,
			)
		solutions[step.label] = solution
		records_by_label[step.label] = observer.records
		output_directories[step.label] = observer.output_directory

	return ProjectedBM4SymplecticityResult(
		dynamics=dynamics,
		area=area,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		records=MappingProxyType(records_by_label),
		output_directories=MappingProxyType(output_directories),
	)


__all__ = [
	"ProjectedBM4DefectOrder",
	"ProjectedBM4SymplecticityConfig",
	"ProjectedBM4SymplecticityResult",
	"ProjectedBM4SymplecticitySummary",
	"run_projected_bm4_symplecticity_study",
]
