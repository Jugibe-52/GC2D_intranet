"""End-to-end transported-area comparisons for projected GC experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from matplotlib.animation import FuncAnimation

from dynamics import GuidingCenterDynamics
from initial_conditions import Area
from potential import Potential
from simulation import (
	BM4Implicit,
	InitialValueProblem,
	SimulationRequest,
	Solution,
	simulate,
)
from diagnostics.symplecticity import (
	GCAreaSymplecticityObserver,
	GCAreaSymplecticityRecord,
)

from ._validation import (
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
	resolve_rho,
)
from visualization import animate_gc_area_comparison


_BLOCK_PREFIX = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class AreaStep:
	"""One labeled integration step in an area comparison."""

	label: str
	value: float

	def __post_init__(self) -> None:
		"""Validate the display label and normalized BM4 step size."""
		if not isinstance(self.label, str) or not self.label.strip():
			raise ValueError("An area-step label must be a non-empty string.")
		object.__setattr__(self, "value", positive_finite(self.value, "value"))


def pi_area_steps(*denominators: int) -> tuple[AreaStep, ...]:
	"""Build labeled step sizes ``pi / denominator`` for comparison notebooks."""
	if len(denominators) < 2:
		raise ValueError("At least two step denominators are required.")
	steps: list[AreaStep] = []
	for denominator in denominators:
		value = positive_integer(denominator, "denominator")
		steps.append(
			AreaStep(
				label=rf"$\Delta t=\pi/{value}$",
				value=np.pi / value,
			)
		)
	return tuple(steps)


@dataclass(frozen=True, slots=True)
class AreaComparisonConfig:
	"""Numerical and persistence parameters for an implicit-BM4 area study."""

	steps: tuple[AreaStep, ...]
	t_span: tuple[float, float]
	save_interval: float
	rho: float | None = None
	coupling_frequency: float = 0.0
	newton_absolute_tolerance: float = 1e-13
	newton_relative_tolerance: float = 1e-12
	newton_max_iterations: int = 12
	newton_jacobian_relative_step: float = float(np.cbrt(np.finfo(float).eps))
	chunk_size: int = 16
	progress: bool = False
	block_prefix: str = "circle_comparison"

	def __post_init__(self) -> None:
		"""Validate synchronized step, output and diagnostic grids."""
		steps = tuple(self.steps)
		if len(steps) < 2 or any(not isinstance(step, AreaStep) for step in steps):
			raise ValueError("`steps` must contain at least two AreaStep values.")
		if len({step.label for step in steps}) != len(steps):
			raise ValueError("Area-step labels must be unique.")
		object.__setattr__(self, "steps", steps)

		try:
			start, stop = (float(value) for value in self.t_span)
		except (TypeError, ValueError) as exc:
			raise ValueError("`t_span` must contain two finite increasing times.") from exc
		if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
			raise ValueError("`t_span` must contain two finite increasing times.")
		object.__setattr__(self, "t_span", (start, stop))

		save_interval = positive_finite(self.save_interval, "save_interval")
		object.__setattr__(self, "save_interval", save_interval)
		if self.rho is not None:
			object.__setattr__(self, "rho", nonnegative_finite(self.rho, "rho"))
		integer_ratio(stop - start, save_interval, "duration / save_interval")
		for step in steps:
			integer_ratio(
				save_interval,
				step.value,
				f"save_interval / step for {step.label}",
			)

		frequency = float(self.coupling_frequency)
		if not np.isfinite(frequency) or frequency < 0:
			raise ValueError("`coupling_frequency` must be finite and non-negative.")
		object.__setattr__(self, "coupling_frequency", frequency)
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
			"chunk_size",
			positive_integer(self.chunk_size, "chunk_size"),
		)
		if not isinstance(self.block_prefix, str) or not _BLOCK_PREFIX.fullmatch(
			self.block_prefix
		):
			raise ValueError(
				"`block_prefix` may contain only letters, numbers, '_' and '-'."
			)

	@property
	def output_sample_count(self) -> int:
		"""Number of uniformly saved physical states, including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class AreaSummary:
	"""Maximum errors observed for one integration step."""

	label: str
	step: float
	step_count: int
	max_area_error: float
	max_local_symplectic_defect: float
	max_flow_symplectic_defect: float


@dataclass(frozen=True, slots=True)
class AreaComparisonResult:
	"""Solutions, projected observations and presentation for one comparison."""

	effective_potential: Potential
	area: Area
	steps: tuple[AreaStep, ...]
	solutions: Mapping[str, Solution]
	records: Mapping[str, tuple[GCAreaSymplecticityRecord, ...]]
	output_directories: Mapping[str, Path]

	@property
	def diagnostic_times(self) -> Mapping[str, np.ndarray]:
		"""Observation times aligned with the labeled solution mappings."""
		return {
			label: np.asarray([record.time for record in records])
			for label, records in self.records.items()
		}

	@property
	def relative_symplecticity_errors(self) -> Mapping[str, np.ndarray]:
		"""Projected relative symplectic defects for every integration step."""
		return {
			label: np.asarray([record.relative_defect for record in records])
			for label, records in self.records.items()
		}

	def summaries(self) -> tuple[AreaSummary, ...]:
		"""Return maximum diagnostics in configured step order."""
		rows: list[AreaSummary] = []
		for step in self.steps:
			records = self.records[step.label]
			rows.append(
				AreaSummary(
					label=step.label,
					step=step.value,
					step_count=int(
						self.solutions[step.label].diagnostics["step_count"]
					),
					max_area_error=max(
						abs(record.relative_area_error) for record in records
					),
					max_local_symplectic_defect=max(
						record.local_relative_defect for record in records
					),
					max_flow_symplectic_defect=max(
						record.relative_defect for record in records
					),
				)
			)
		return tuple(rows)

	def print_summary(self) -> None:
		"""Print one compact diagnostic row per configured integration step."""
		header = (
			"step",
			"integration steps",
			"max |area error|",
			"max local defect",
			"max flow defect",
		)
		print(
			f"{header[0]:>22} {header[1]:>12} {header[2]:>20} "
			f"{header[3]:>26} {header[4]:>26}"
		)
		for row in self.summaries():
			print(
				f"{row.label:>22} {row.step_count:12d} "
				f"{row.max_area_error:20.8e} "
				f"{row.max_local_symplectic_defect:26.8e} "
				f"{row.max_flow_symplectic_defect:26.8e}"
			)

	def animate(
		self,
		*,
		frames: int | None = None,
		interval: int = 200,
		repeat: bool = True,
	) -> FuncAnimation:
		"""Build the synchronized contour and projected-diagnostic animation."""
		return animate_gc_area_comparison(
			self.effective_potential,
			self.area,
			self.solutions,
			diagnostic_times=self.diagnostic_times,
			relative_symplecticity_errors=self.relative_symplecticity_errors,
			frames=frames,
			interval=interval,
			repeat=repeat,
		)


def run_area_comparison(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: AreaComparisonConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> AreaComparisonResult:
	"""Run all configured GC methods and persist projected observations."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(area, Area):
		raise TypeError("`area` must be an Area instance.")
	if not isinstance(config, AreaComparisonConfig):
		raise TypeError("`config` must be an AreaComparisonConfig instance.")

	rho = resolve_rho(config.rho, area)
	dynamics = GuidingCenterDynamics(potential, rho=rho)
	problem = InitialValueProblem(dynamics, area)
	initial_state = area.initial_state
	assert initial_state is not None
	solutions: dict[str, Solution] = {}
	records_by_label: dict[str, tuple[GCAreaSymplecticityRecord, ...]] = {}
	output_directories: dict[str, Path] = {}

	common_metadata = {
		**dict(metadata or {}),
		"geometry": area.shape,
		"particle_count": area.layout.particle_count(initial_state),
		"coupling_frequency": config.coupling_frequency,
		"method_name": "BM4Implicit",
		"projection_scope": "one_complete_twelve_stage_cycle",
		"projection_formulation": "reduced_multiplier",
		"rho": rho,
	}
	for step in config.steps:
		record_every = integer_ratio(
			config.save_interval,
			step.value,
			f"save_interval / step for {step.label}",
		)
		step_tag = f"{step.value:.8f}".replace(".", "p")
		with GCAreaSymplecticityObserver(
			notebook_path=notebook_path,
			area=area,
			period=potential.grid.period,
			project_root=project_root,
			block_name=f"{config.block_prefix}_step_{step_tag}",
			record_every=record_every,
			chunk_size=config.chunk_size,
			jacobian_method="finite_difference",
			verbose=False,
			metadata={
				**common_metadata,
				"integration_step": step.value,
			},
		) as observer:
			request = SimulationRequest.uniform(
				t_span=config.t_span,
				max_step=step.value,
				sample_count=config.output_sample_count,
			)
			method = BM4Implicit(
				coupling_frequency=config.coupling_frequency,
				newton_absolute_tolerance=config.newton_absolute_tolerance,
				newton_relative_tolerance=config.newton_relative_tolerance,
				newton_max_iterations=config.newton_max_iterations,
				newton_jacobian_relative_step=(
					config.newton_jacobian_relative_step
				),
				newton_jacobian_method="analytic",
				nonlinear_solver="newton",
				progress=config.progress,
				step_observer=observer,
			)
			solution = simulate(problem, method, request)
		solutions[step.label] = solution
		records_by_label[step.label] = observer.records
		output_directories[step.label] = observer.output_directory

	return AreaComparisonResult(
		effective_potential=dynamics.effective_potential,
		area=area,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		records=MappingProxyType(records_by_label),
		output_directories=MappingProxyType(output_directories),
	)


__all__ = [
	"AreaComparisonConfig",
	"AreaComparisonResult",
	"AreaStep",
	"AreaSummary",
	"pi_area_steps",
	"run_area_comparison",
]
