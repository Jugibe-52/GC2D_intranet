"""RK4 convergence studies for physical GC area and symplecticity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from classes import (
	Area,
	GuidingCenterDynamics,
	InitialValueProblem,
	Potential,
	RK4,
	SimulationRequest,
	Solution,
	simulate,
)
from research.symplecticity import (
	GCAreaSymplecticityObserver,
	GCAreaSymplecticityRecord,
)

from ._validation import integer_ratio, positive_finite, positive_integer
from .area_comparison import AreaStep
from .gc_visualization import animate_gc_area_comparison


_BLOCK_PREFIX = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class RK4SymplecticityConfig:
	"""Numerical, diagnostic and persistence parameters for an RK4 GC study."""

	steps: tuple[AreaStep, ...]
	t_span: tuple[float, float]
	save_interval: float
	chunk_size: int = 16
	progress: bool = False
	block_prefix: str = "rk4_symplecticity"
	finite_difference_relative_step: float | None = None

	def __post_init__(self) -> None:
		"""Validate synchronized integration and observation grids."""
		steps = tuple(self.steps)
		if len(steps) < 2 or any(not isinstance(step, AreaStep) for step in steps):
			raise ValueError("`steps` must contain at least two AreaStep values.")
		if len({step.label for step in steps}) != len(steps):
			raise ValueError("RK4 step labels must be unique.")
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
		integer_ratio(stop - start, save_interval, "duration / save_interval")
		for step in steps:
			integer_ratio(
				save_interval,
				step.value,
				f"save_interval / step for {step.label}",
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
		relative_step = self.finite_difference_relative_step
		if relative_step is not None:
			object.__setattr__(
				self,
				"finite_difference_relative_step",
				positive_finite(relative_step, "finite_difference_relative_step"),
			)

	@property
	def output_sample_count(self) -> int:
		"""Number of uniformly saved states, including both endpoints."""
		return integer_ratio(
			self.t_span[1] - self.t_span[0],
			self.save_interval,
			"duration / save_interval",
		) + 1


@dataclass(frozen=True, slots=True)
class RK4SymplecticitySummary:
	"""Maximum RK4 errors observed for one integration step."""

	label: str
	step: float
	step_count: int
	max_area_error: float
	max_local_defect: float
	max_flow_defect: float
	max_determinant_error: float


@dataclass(frozen=True, slots=True)
class RK4ConvergenceOrder:
	"""Empirical order between two consecutive RK4 step sizes."""

	coarse_label: str
	fine_label: str
	value: float


@dataclass(frozen=True, slots=True)
class RK4SymplecticityResult:
	"""RK4 solutions, physical-flow Jacobians and analysis helpers."""

	dynamics: GuidingCenterDynamics
	area: Area
	steps: tuple[AreaStep, ...]
	solutions: Mapping[str, Solution]
	records: Mapping[str, tuple[GCAreaSymplecticityRecord, ...]]
	output_directories: Mapping[str, Path]

	@property
	def diagnostic_times(self) -> Mapping[str, np.ndarray]:
		"""Observation times aligned with the labeled RK4 solutions."""
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

	def summaries(self) -> tuple[RK4SymplecticitySummary, ...]:
		"""Return maximum diagnostics in configured step order."""
		return tuple(
			RK4SymplecticitySummary(
				label=step.label,
				step=step.value,
				step_count=int(
					self.solutions[step.label].diagnostics["step_count"]
				),
				max_area_error=max(
					abs(record.relative_area_error)
					for record in self.records[step.label]
				),
				max_local_defect=max(
					record.local_relative_defect
					for record in self.records[step.label]
				),
				max_flow_defect=max(
					record.relative_defect for record in self.records[step.label]
				),
				max_determinant_error=max(
					record.determinant_error
					for record in self.records[step.label]
				),
			)
			for step in self.steps
		)

	def convergence_orders(self) -> tuple[RK4ConvergenceOrder, ...]:
		"""Estimate the order of the maximum accumulated symplecticity defect."""
		summaries = self.summaries()
		orders: list[RK4ConvergenceOrder] = []
		for coarse, fine in zip(summaries, summaries[1:]):
			if (
				coarse.max_flow_defect <= 0
				or fine.max_flow_defect <= 0
				or np.isclose(coarse.step, fine.step)
			):
				value = float("nan")
			else:
				value = float(
					np.log(coarse.max_flow_defect / fine.max_flow_defect)
					/ np.log(coarse.step / fine.step)
				)
			orders.append(
				RK4ConvergenceOrder(
					coarse_label=coarse.label,
					fine_label=fine.label,
					value=value,
				)
			)
		return tuple(orders)

	def print_summary(self) -> None:
		"""Print compact error and empirical-convergence tables."""
		print(
			f"{'step':>22} {'RK4 steps':>10} {'max |area error|':>20} "
			f"{'max local defect':>20} {'max flow defect':>20} "
			f"{'max |det-1|':>16}"
		)
		for row in self.summaries():
			print(
				f"{row.label:>22} {row.step_count:10d} "
				f"{row.max_area_error:20.8e} {row.max_local_defect:20.8e} "
				f"{row.max_flow_defect:20.8e} "
				f"{row.max_determinant_error:16.8e}"
			)
		print("\nEmpirical order of the maximum accumulated symplecticity defect:")
		for order in self.convergence_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.value:.6f}"
			)

	def plot_diagnostics(self) -> tuple[Figure, np.ndarray]:
		"""Plot area, local and accumulated symplecticity, and determinant drift."""
		figure, axes = plt.subplots(
			2,
			2,
			figsize=(13, 8),
			constrained_layout=True,
		)
		for step in self.steps:
			label = step.label
			records = self.records[label]
			times = np.asarray([record.time for record in records])
			area_errors = np.asarray(
				[record.relative_area_error for record in records]
			)
			local_defects = np.asarray(
				[record.local_relative_defect for record in records]
			)
			flow_defects = np.asarray(
				[record.relative_defect for record in records]
			)
			determinant_errors = np.asarray(
				[record.determinant_error for record in records]
			)
			axes[0, 0].plot(times, area_errors, label=label)
			axes[0, 1].semilogy(
				times[1:],
				_positive_for_log(local_defects[1:]),
				label=label,
			)
			axes[1, 0].semilogy(
				times[1:],
				_positive_for_log(flow_defects[1:]),
				label=label,
			)
			axes[1, 1].semilogy(
				times[1:],
				_positive_for_log(determinant_errors[1:]),
				label=label,
			)

		axes[0, 0].axhline(0.0, color="0.5", linestyle="--", linewidth=1)
		axes[0, 0].set(
			title="Relative transported-area error",
			xlabel="$t$",
			ylabel=r"$(A(t)-A(0))/|A(0)|$",
		)
		axes[0, 1].set(
			title="Local RK4-step symplecticity defect",
			xlabel="$t$",
			ylabel=r"$\|J_n^T\Omega J_n-\Omega\|_F/\|\Omega\|_F$",
		)
		axes[1, 0].set(
			title="Accumulated numerical-flow symplecticity defect",
			xlabel="$t$",
			ylabel=r"$\|DG_n^T\Omega DG_n-\Omega\|_F/\|\Omega\|_F$",
		)
		axes[1, 1].set(
			title="Accumulated numerical-flow determinant error",
			xlabel="$t$",
			ylabel=r"$|\det(DG_n)-1|$",
		)
		for axis in axes.flat:
			axis.grid(alpha=0.25)
			axis.legend()
		return figure, axes

	def plot_convergence(self) -> tuple[Figure, Axes]:
		"""Plot maximum symplecticity errors against the RK4 step size."""
		summaries = self.summaries()
		steps = np.asarray([row.step for row in summaries])
		local_errors = np.asarray([row.max_local_defect for row in summaries])
		flow_errors = np.asarray([row.max_flow_defect for row in summaries])
		figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
		axes.loglog(steps, local_errors, "o-", label="Maximum local-step defect")
		axes.loglog(steps, flow_errors, "s-", label="Maximum accumulated defect")
		axes.set(
			xlabel=r"RK4 step $\Delta t$",
			ylabel="Relative symplecticity defect",
			title="RK4 symplecticity-defect convergence",
		)
		axes.grid(which="both", alpha=0.25)
		axes.legend()
		axes.invert_xaxis()
		return figure, axes

	def animate(
		self,
		*,
		frames: int | None = 120,
		interval: int = 50,
		repeat: bool = True,
	) -> FuncAnimation:
		"""Animate RK4 contours, area errors and accumulated symplecticity."""
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


def _positive_for_log(values: np.ndarray) -> np.ndarray:
	"""Replace exact zeros by a local positive floor for logarithmic display."""
	result = np.asarray(values, dtype=float)
	positive = result[result > 0]
	floor = (
		float(np.min(positive)) / 10
		if positive.size
		else float(np.finfo(float).eps)
	)
	return np.maximum(result, floor)


def run_rk4_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: RK4SymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> RK4SymplecticityResult:
	"""Run synchronized RK4 steps and persist physical GC flow diagnostics."""
	if not isinstance(potential, Potential):
		raise TypeError("`potential` must be a Potential instance.")
	if not isinstance(area, Area):
		raise TypeError("`area` must be an Area instance.")
	if not isinstance(config, RK4SymplecticityConfig):
		raise TypeError("`config` must be an RK4SymplecticityConfig instance.")

	dynamics = GuidingCenterDynamics(potential, rho=area.rho)
	problem = InitialValueProblem(dynamics, area)
	initial_state = area.initial_state
	assert initial_state is not None
	common_metadata = {
		**dict(metadata or {}),
		"method": "RK4",
		"geometry": area.shape,
		"particle_count": area.particle_count(initial_state),
		"rho": area.rho,
	}
	solutions: dict[str, Solution] = {}
	records_by_label: dict[str, tuple[GCAreaSymplecticityRecord, ...]] = {}
	output_directories: dict[str, Path] = {}

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
			relative_step=config.finite_difference_relative_step,
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
			solution = simulate(
				problem,
				RK4(progress=config.progress, step_observer=observer),
				request,
			)
		solutions[step.label] = solution
		records_by_label[step.label] = observer.records
		output_directories[step.label] = observer.output_directory

	return RK4SymplecticityResult(
		dynamics=dynamics,
		area=area,
		steps=config.steps,
		solutions=MappingProxyType(solutions),
		records=MappingProxyType(records_by_label),
		output_directories=MappingProxyType(output_directories),
	)


__all__ = [
	"RK4ConvergenceOrder",
	"RK4SymplecticityConfig",
	"RK4SymplecticityResult",
	"RK4SymplecticitySummary",
	"run_rk4_symplecticity_study",
]
