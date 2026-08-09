"""Symplecticity studies for explicit ABBA with arithmetic projection."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from initial_conditions import Area
from potential import Potential
from simulation import (
	ExplicitABBA,
	Solution,
)

from ._gc_symplecticity import (
	GCSymplecticityConfig,
	GCSymplecticityResult,
	GCSymplecticitySummary,
	_run_gc_symplecticity_study,
)


@dataclass(frozen=True, slots=True)
class ExplicitABBASymplecticityConfig(GCSymplecticityConfig):
	"""Reproducible grids for an explicit averaged-ABBA GC study."""

	block_prefix: str = "explicit_abba_symplecticity"


@dataclass(frozen=True, slots=True)
class ExplicitABBASymplecticitySummary(GCSymplecticitySummary):
	"""Maximum geometric defects and copy separation for one step size."""

	max_copy_separation_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class ExplicitABBADefectOrder:
	"""Empirical local, accumulated and copy-separation orders."""

	coarse_label: str
	fine_label: str
	local_defect: float
	flow_defect: float
	copy_separation: float


def _maximum_copy_separation(solution: Solution) -> float:
	"""Validate and reduce the per-step off-diagonal copy separation."""
	values = np.asarray(solution.diagnostics.get("copy_separation_norms"))
	step_count = int(solution.diagnostics.get("step_count", 0))
	if values.shape != (step_count,):
		raise ValueError(
			"`copy_separation_norms` must contain one value per integration step."
		)
	if not np.all(np.isfinite(values)) or np.any(values < 0):
		raise ValueError("Copy-separation norms must be finite and non-negative.")
	return float(np.max(values))


def _empirical_order(
	coarse_value: float,
	fine_value: float,
	coarse_step: float,
	fine_step: float,
) -> float:
	"""Return a log-ratio order or NaN when the ratio is undefined."""
	if (
		coarse_value <= 0
		or fine_value <= 0
		or np.isclose(coarse_step, fine_step)
	):
		return float("nan")
	return float(
		np.log(coarse_value / fine_value)
		/ np.log(coarse_step / fine_step)
	)


@dataclass(frozen=True, slots=True)
class ExplicitABBASymplecticityResult(GCSymplecticityResult):
	"""Explicit ABBA solutions and physical-flow analysis helpers."""

	method_name: ClassVar[str] = "ExplicitABBA"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = (
		ExplicitABBASymplecticitySummary
	)

	def summaries(self) -> tuple[ExplicitABBASymplecticitySummary, ...]:
		"""Return geometric summaries augmented by the maximum copy separation."""
		base_rows = cast(
			tuple[ExplicitABBASymplecticitySummary, ...],
			GCSymplecticityResult.summaries(self),
		)
		return tuple(
			replace(
				row,
				max_copy_separation_norm=_maximum_copy_separation(
					self.solutions[row.label]
				),
			)
			for row in base_rows
		)

	def convergence_orders(self) -> tuple[ExplicitABBADefectOrder, ...]:
		"""Estimate defect and copy-separation orders between consecutive steps."""
		rows = self.summaries()
		orders: list[ExplicitABBADefectOrder] = []
		for coarse, fine in zip(rows, rows[1:]):
			orders.append(
				ExplicitABBADefectOrder(
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
					copy_separation=_empirical_order(
						coarse.max_copy_separation_norm,
						fine.max_copy_separation_norm,
						coarse.step,
						fine.step,
					),
				)
			)
		return tuple(orders)

	def print_summary(self) -> None:
		"""Print geometric errors, copy separation and empirical orders."""
		GCSymplecticityResult.print_summary(self)
		print("\nMaximum off-diagonal copy separation before averaging:")
		for row in self.summaries():
			print(f"  {row.label}: {row.max_copy_separation_norm:.8e}")
		print("\nEmpirical orders (local defect / flow defect / copy separation):")
		for order in self.convergence_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.local_defect:.6f} / {order.flow_defect:.6f} / "
				f"{order.copy_separation:.6f}"
			)

	def plot_convergence(self) -> tuple[Figure, Axes]:
		"""Plot defects and pre-projection copy separation against step size."""
		figure, axis = self._plot_step_defects(
			title="Explicit ABBA symplecticity-defect convergence",
			xlabel=r"Explicit ABBA step $\Delta t$",
		)
		rows = self.summaries()
		axis.loglog(
			[row.step for row in rows],
			[row.max_copy_separation_norm for row in rows],
			"^-",
			label="Maximum copy separation before averaging",
		)
		axis.set_ylabel("Relative defect or copy-separation norm")
		axis.legend()
		return figure, axis


def run_explicit_abba_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: ExplicitABBASymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> ExplicitABBASymplecticityResult:
	"""Run explicit ABBA steps and persist physical GC-flow diagnostics."""
	if not isinstance(config, ExplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be an ExplicitABBASymplecticityConfig instance."
		)
	study_metadata = {
		**dict(metadata or {}),
		"abba_stage_times": "t_n,t_n,t_n_plus_h,t_n_plus_h",
		"diagonal_embedding": "duplicate_physical_state_each_step",
		"projection": "arithmetic_mean",
		"projection_is_symplectic": False,
		"step_jacobian": "centered_difference_of_emitted_solver_map",
	}
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: ExplicitABBA(
			progress=config.progress,
			step_observer=observer,
		),
		result_type=ExplicitABBASymplecticityResult,
		project_root=project_root,
		metadata=study_metadata,
	)


__all__ = [
	"ExplicitABBADefectOrder",
	"ExplicitABBASymplecticityConfig",
	"ExplicitABBASymplecticityResult",
	"ExplicitABBASymplecticitySummary",
	"run_explicit_abba_symplecticity_study",
]
