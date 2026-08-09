"""Validation and axis-limit helpers for GC area visualizations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from initial_conditions import Area
from simulation.solution import Solution


AreaDiagnostics = tuple[np.ndarray, np.ndarray, np.ndarray | None]


def validated_solution_series(
	configuration: Area,
	solutions: Sequence[Solution],
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
	"""Validate solutions sampled at common times for direct comparison."""
	if not solutions:
		raise ValueError("At least one solution is required for an area animation.")
	initial_state = configuration.state
	assert initial_state is not None
	reference_times: np.ndarray | None = None
	state_series: list[np.ndarray] = []
	for solution in solutions:
		if not isinstance(solution, Solution):
			raise TypeError("Every animated solution must be a Solution instance.")
		times = np.asarray(solution.t, dtype=float)
		states = np.asarray(solution.y, dtype=float)
		if (
			times.ndim != 1
			or times.size < 2
			or states.ndim != 2
			or states.shape != (initial_state.size, times.size)
		):
			raise ValueError(
				"Each solution must contain at least two times and matching Area states."
			)
		if (
			not np.all(np.isfinite(times))
			or not np.all(np.isfinite(states))
			or np.any(np.diff(times) <= 0)
		):
			raise ValueError(
				"Solutions must contain finite states and strictly increasing times."
			)
		if reference_times is None:
			reference_times = times
		else:
			time_scale = max(1.0, float(np.max(np.abs(reference_times))))
			tolerance = float(32 * np.finfo(float).eps * time_scale)
			if times.shape != reference_times.shape or not np.allclose(
				times,
				reference_times,
				rtol=0.0,
				atol=tolerance,
			):
				raise ValueError(
					"Compared solutions must be saved at the same times."
				)
		state_series.append(states)
	assert reference_times is not None
	return reference_times, tuple(state_series)


def validated_labels(labels: Sequence[str], series_count: int) -> tuple[str, ...]:
	"""Require one distinct, non-empty display label per solution."""
	values = tuple(labels)
	if (
		len(values) != series_count
		or any(not isinstance(label, str) or not label.strip() for label in values)
		or len(set(values)) != len(values)
	):
		raise ValueError("Each solution requires a distinct, non-empty label.")
	return values


def validated_relative_diagnostics(
	diagnostic_times: np.ndarray | None,
	relative_symplecticity_errors: np.ndarray | None,
	relative_copy_separations: np.ndarray | None,
) -> AreaDiagnostics | None:
	"""Validate synchronized non-negative symplecticity and optional separation."""
	if (
		diagnostic_times is None
		and relative_symplecticity_errors is None
		and relative_copy_separations is None
	):
		return None
	if diagnostic_times is None or relative_symplecticity_errors is None:
		raise ValueError(
			"`diagnostic_times` and `relative_symplecticity_errors` "
			"must be provided together."
		)
	times = np.asarray(diagnostic_times, dtype=float)
	symplecticity_errors = np.asarray(relative_symplecticity_errors, dtype=float)
	copy_separations = (
		None
		if relative_copy_separations is None
		else np.asarray(relative_copy_separations, dtype=float)
	)
	if (
		times.ndim != 1
		or times.size < 1
		or symplecticity_errors.shape != times.shape
		or (copy_separations is not None and copy_separations.shape != times.shape)
	):
		raise ValueError(
			"Area diagnostics must be one-dimensional arrays of equal length."
		)
	if (
		not np.all(np.isfinite(times))
		or not np.all(np.isfinite(symplecticity_errors))
		or (copy_separations is not None and not np.all(np.isfinite(copy_separations)))
		or np.any(np.diff(times) <= 0)
	):
		raise ValueError(
			"Area diagnostics must be finite and have strictly increasing times."
		)
	if np.any(symplecticity_errors < 0) or (
		copy_separations is not None and np.any(copy_separations < 0)
	):
		raise ValueError("Relative area diagnostics must be non-negative.")
	return times, symplecticity_errors, copy_separations


def validated_diagnostic_series(
	diagnostic_times: Sequence[np.ndarray | None],
	relative_symplecticity_errors: Sequence[np.ndarray | None],
	relative_copy_separations: Sequence[np.ndarray | None],
	series_count: int,
) -> tuple[AreaDiagnostics, ...] | None:
	"""Validate either complete diagnostics for every series or none at all."""
	if not (
		len(diagnostic_times)
		== len(relative_symplecticity_errors)
		== len(relative_copy_separations)
		== series_count
	):
		raise ValueError("Diagnostic sequences must match the number of solutions.")
	validated = tuple(
		validated_relative_diagnostics(times, defects, separations)
		for times, defects, separations in zip(
			diagnostic_times,
			relative_symplecticity_errors,
			relative_copy_separations,
			strict=True,
		)
	)
	if all(item is None for item in validated):
		return None
	if any(item is None for item in validated):
		raise ValueError("Area diagnostics are required for every comparison.")
	result = tuple(item for item in validated if item is not None)
	if len({item[2] is None for item in result}) != 1:
		raise ValueError(
			"Copy-separation diagnostics must be present for every series or none."
		)
	return result


def positive_log_limits(values: Sequence[np.ndarray]) -> tuple[float, float]:
	"""Return stable global log limits, including diagnostics that begin at zero."""
	combined = np.concatenate(tuple(values))
	positive = combined[combined > 0]
	if positive.size:
		lower = max(float(np.min(positive)) / 2, float(np.finfo(float).tiny))
		upper = max(float(np.max(positive)) * 2, lower * 10)
	else:
		lower = float(np.finfo(float).eps)
		upper = 10 * lower
	return lower, upper


def linear_limits(values: Sequence[np.ndarray]) -> tuple[float, float]:
	"""Return padded global limits for several signed linear diagnostics."""
	lower = min(float(np.min(value)) for value in values)
	upper = max(float(np.max(value)) for value in values)
	span = upper - lower
	padding = float(
		0.05 * span
		if span > 0
		else 0.05 * abs(lower) or float(np.finfo(float).eps)
	)
	return lower - padding, upper + padding


__all__: list[str] = []
