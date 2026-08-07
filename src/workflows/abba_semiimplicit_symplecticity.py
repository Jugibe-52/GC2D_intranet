"""Exact-tangent symplecticity studies for semi-implicit ABBA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, cast

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from classes import Area, Potential, SemiImplicitABBA

from ._gc_symplecticity import (
	GCSymplecticityResult,
	GCSymplecticitySummary,
	_run_gc_symplecticity_study,
)
from .abba_symplecticity import (
	ABBAProjectionMultiplierOrder,
	ABBASymplecticityConfig,
	ABBASymplecticityResult,
	ABBASymplecticitySummary,
)


@dataclass(frozen=True, slots=True)
class SemiImplicitABBASymplecticityConfig(ABBASymplecticityConfig):
	"""Reproducible grids and Newton parameters for exact ABBA tangents."""

	block_prefix: str = "semiimplicit_abba_symplecticity"

	def __post_init__(self) -> None:
		"""Validate the ABBA solve and reject finite-difference parameters."""
		ABBASymplecticityConfig.__post_init__(self)
		if self.finite_difference_relative_step is not None:
			raise ValueError(
				"Semi-implicit ABBA uses exact step Jacobians; "
				"`finite_difference_relative_step` must be None."
			)


@dataclass(frozen=True, slots=True)
class SemiImplicitABBASymplecticitySummary(ABBASymplecticitySummary):
	"""Maximum exact-tangent defects and Newton statistics for one step size."""


@dataclass(frozen=True, slots=True)
class SemiImplicitABBASymplecticityResult(ABBASymplecticityResult):
	"""Semi-implicit ABBA solutions and exact physical tangent diagnostics."""

	method_name: ClassVar[str] = "SemiImplicitABBA"
	summary_type: ClassVar[type[GCSymplecticitySummary]] = (
		SemiImplicitABBASymplecticitySummary
	)

	def summaries(self) -> tuple[SemiImplicitABBASymplecticitySummary, ...]:
		"""Return exact-tangent summary values in configured step order."""
		return cast(
			tuple[SemiImplicitABBASymplecticitySummary, ...],
			GCSymplecticityResult.summaries(self),
		)

	def projection_multiplier_orders(
		self,
	) -> tuple[ABBAProjectionMultiplierOrder, ...]:
		"""Return multiplier scaling inherited from the projected physical step."""
		return ABBASymplecticityResult.projection_multiplier_orders(self)

	def print_summary(self) -> None:
		"""Print exact-tangent defects and nonlinear projection diagnostics."""
		GCSymplecticityResult.print_summary(self)
		print(
			"\nThe local matrices are exact implicit-function tangents evaluated "
			"at the converged ABBA stages. Reported defects therefore contain "
			"Newton-tolerance, potential-derivative, and floating-point effects, "
			"but no finite-difference differentiation floor."
		)
		print("\nEmpirical scaling of the maximum projection multiplier (expected ~3):")
		for order in self.projection_multiplier_orders():
			print(
				f"  {order.coarse_label} -> {order.fine_label}: "
				f"{order.value:.6f}"
			)

	def plot_defect_floor(self) -> tuple[Figure, Axes]:
		"""Plot exact-tangent defects across integration step sizes."""
		return self._plot_step_defects(
			title="Semi-implicit ABBA exact-Jacobian symplecticity floor",
			xlabel=r"ABBA step $\Delta t$",
		)


def run_semiimplicit_abba_symplecticity_study(
	potential: Potential,
	area: Area,
	*,
	notebook_path: str | Path,
	config: SemiImplicitABBASymplecticityConfig,
	project_root: str | Path | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> SemiImplicitABBASymplecticityResult:
	"""Run projected ABBA with exact local and accumulated physical tangents."""
	if not isinstance(config, SemiImplicitABBASymplecticityConfig):
		raise TypeError(
			"`config` must be a SemiImplicitABBASymplecticityConfig instance."
		)
	study_metadata = {
		**dict(metadata or {}),
		"newton_absolute_tolerance": config.newton_absolute_tolerance,
		"newton_relative_tolerance": config.newton_relative_tolerance,
		"newton_max_iterations": config.newton_max_iterations,
		"newton_initial_multiplier": "zero",
		"newton_residual_norm": "infinity",
		"abba_stage_times": "t_n,t_n,t_n_plus_h,t_n_plus_h",
		"step_jacobian": "exact_implicit_function_tangent",
		"tangent_formula": "P-Q*solve(K,L)",
	}
	return _run_gc_symplecticity_study(
		potential,
		area,
		notebook_path=notebook_path,
		config=config,
		method_factory=lambda observer: SemiImplicitABBA(
			newton_absolute_tolerance=config.newton_absolute_tolerance,
			newton_relative_tolerance=config.newton_relative_tolerance,
			newton_max_iterations=config.newton_max_iterations,
			progress=config.progress,
			step_observer=observer,
		),
		result_type=SemiImplicitABBASymplecticityResult,
		project_root=project_root,
		metadata=study_metadata,
		jacobian_source="exact",
	)


__all__ = [
	"SemiImplicitABBASymplecticityConfig",
	"SemiImplicitABBASymplecticityResult",
	"SemiImplicitABBASymplecticitySummary",
	"run_semiimplicit_abba_symplecticity_study",
]
