"""Validated configuration and summary models for GC symplecticity studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .area_comparison import _validated_area_steps, AreaStep

from ._validation import (
	unpacked_time_span,
	validate_block_prefix,
	integer_ratio,
	nonnegative_finite,
	positive_finite,
	positive_integer,
)





@dataclass(frozen=True, slots=True)
class GCSymplecticityConfig:
	"""Common numerical, diagnostic and persistence parameters for a GC study."""

	steps: tuple[AreaStep, ...]
	t_span: tuple[float, float]
	save_interval: float
	rho: float | None = None
	chunk_size: int = 16
	progress: bool = False
	block_prefix: str = "gc_symplecticity"
	finite_difference_relative_step: float | None = None

	def __post_init__(self) -> None:
		"""Validate synchronized integration and observation grids."""
		steps = _validated_area_steps(
			self.steps, minimum=1, label_message="GC integration-step labels must be unique.",
		)
		object.__setattr__(self, "steps", steps)

		start, stop = unpacked_time_span(self.t_span)
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
		object.__setattr__(
			self,
			"chunk_size",
			positive_integer(self.chunk_size, "chunk_size"),
		)
		validate_block_prefix(self.block_prefix)
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
class GCSymplecticitySummary:
	"""Maximum physical GC errors observed for one integration step."""

	label: str
	step: float
	step_count: int
	max_area_error: float
	max_local_defect: float
	max_flow_defect: float
	max_determinant_error: float
	max_newton_iterations: int | None = None
	mean_newton_iterations: float | None = None
	max_newton_residual_norm: float | None = None
	max_projection_multiplier_norm: float | None = None


@dataclass(frozen=True, slots=True)
class GCConvergenceOrder:
	"""Empirical diagnostic slope between two consecutive integration steps."""

	coarse_label: str
	fine_label: str
	value: float


__all__ = [
	"GCConvergenceOrder",
	"GCSymplecticityConfig",
	"GCSymplecticitySummary",
]
