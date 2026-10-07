"""Shared validation helpers for reproducible numerical studies."""

from __future__ import annotations

from contracts._study_validation import (
	positive_finite, nonnegative_finite, positive_integer, integer_ratio, finite_time_span,
)

from collections.abc import Iterable
import re

import numpy as np

from initial_conditions.gc import GCInitialConfiguration






def resolve_rho(explicit: float | None) -> float:
	"""Require physical gyro-radius in the study rather than its geometry."""
	if explicit is None:
		raise ValueError("`rho` must be explicit in the study configuration.")
	return nonnegative_finite(explicit, "rho")






def refinement_steps(steps: tuple[float, ...]) -> tuple[float, ...]:
	"""Normalize distinct positive steps ordered from coarsest to finest."""
	values = tuple(float(step) for step in steps)
	if not values or any(not np.isfinite(step) or step <= 0.0 for step in values):
		raise ValueError("`steps` must contain positive finite values.")
	if len(set(values)) != len(values):
		raise ValueError("`steps` must not contain duplicates.")
	if any(coarse <= fine for coarse, fine in zip(values, values[1:])):
		raise ValueError("`steps` must be ordered from coarsest to finest.")
	return values




def unpacked_time_span(values: tuple[float, float]) -> tuple[float, float]:
	"""Normalize an iterable pair, reporting conversion errors as span errors."""
	message = "`t_span` must contain two finite increasing times."
	try:
		start, stop = (float(value) for value in values)
	except (TypeError, ValueError) as exc:
		raise ValueError(message) from exc
	if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
		raise ValueError(message)
	return start, stop


def sample_count(value: int) -> int:
	"""Require at least two samples, retaining strict integer validation."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 2
	):
		raise ValueError("`sample_count` must be an integer of at least two.")
	return int(value)


def nonnegative_integer(value: int, name: str) -> int:
	"""Normalize a Python or NumPy integer while rejecting booleans."""
	if (
		isinstance(value, (bool, np.bool_))
		or not isinstance(value, (int, np.integer))
		or value < 0
	):
		raise ValueError(f"`{name}` must be a non-negative integer.")
	return int(value)


def validate_block_prefix(value: str) -> None:
	"""Require the existing filename-safe study block naming convention."""
	if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
		raise ValueError("`block_prefix` may contain only letters, numbers, '_' and '-'.")


def validate_sampling_grid(
	duration: float, save_interval: float, integration_steps: Iterable[float],
) -> None:
	"""Require saved outputs and every integration grid to divide the duration."""
	integer_ratio(duration, save_interval, "duration / save_interval")
	for step in integration_steps:
		integer_ratio(duration, step, "duration / integration_step")
		integer_ratio(save_interval, step, "save_interval / integration_step")


def single_gc_state(configuration: GCInitialConfiguration, *, message: str) -> np.ndarray:
	"""Return the one packed GC state required by single-trajectory studies."""
	initial_state = configuration.initial_state
	if initial_state is None or configuration.layout.particle_count(initial_state) != 1:
		raise ValueError(message)
	return initial_state


__all__ = [
	"finite_time_span",
	"sample_count",
	"single_gc_state",
	"unpacked_time_span",
	"nonnegative_integer",
	"validate_block_prefix",
	"validate_sampling_grid",
	"integer_ratio",
	"nonnegative_finite",
	"positive_finite",
	"positive_integer",
	"refinement_steps",
	"resolve_rho",
]
