"""Shared alignment contracts for multi-method trajectory comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from initial_conditions.gc import GCInitialConfiguration
from solution import Solution
from ._gauss_legendre4_common import readonly_runtime_samples
from ._trajectory_accuracy import TrajectoryAccuracySeries

if TYPE_CHECKING:
	from .three_method_newton_comparison import EnergyAccuracySeries


def validate_comparison_solution(
	solution: Solution, initial_configuration: GCInitialConfiguration,
	times: np.ndarray, step_count: int,
) -> None:
	"""Require one physical source and the exact common integration/output grid."""
	if not isinstance(solution, Solution):
		raise TypeError("Every comparison value must be a Solution.")
	if solution.source is not initial_configuration:
		raise ValueError("All methods must share one initial configuration.")
	if not np.array_equal(solution.t, times):
		raise ValueError("Every method must share the reference output grid.")
	if int(solution.diagnostics.get("step_count", -1)) != step_count:
		raise ValueError("Every method must use the common complete step.")


def validate_trajectory_series(
	method_name: str, series: TrajectoryAccuracySeries, sample_count: int,
) -> None:
	"""Require the trajectory label and saved-time dimension."""
	if series.method_name != method_name:
		raise ValueError("Accuracy labels must match their numerical methods.")
	if series.distances.shape[1] != sample_count:
		raise ValueError("Accuracy series must share the saved-time grid.")


def validate_energy_series(
	method_name: str, energy_series: EnergyAccuracySeries, *,
	energy_type: type[EnergyAccuracySeries], energy_shape: tuple[int, int],
) -> None:
	"""Require a typed energy series on the common particle-time grid."""
	if not isinstance(energy_series, energy_type):
		raise TypeError("Every energy comparison must be EnergyAccuracySeries.")
	if energy_series.method_name != method_name:
		raise ValueError("Energy labels must match their numerical methods.")
	if energy_series.errors.shape != energy_shape:
		raise ValueError("Energy errors must share the particle-time grid.")


def comparison_runtime_samples(values: np.ndarray, repeats: int) -> np.ndarray:
	"""Copy, validate, and freeze measured repeats exactly once per method."""
	samples = readonly_runtime_samples(values)
	if samples.size != repeats:
		raise ValueError("Every method must contain all measured timing repeats.")
	return samples


def freeze_reference_indices(
	values: np.ndarray, reference_times: np.ndarray, times: np.ndarray, *, message: str,
) -> np.ndarray:
	"""Validate and freeze an already-owned integer reference index array."""
	indices = values
	if (
		indices.shape != times.shape
		or np.any(indices < 0)
		or np.any(indices >= reference_times.size)
		or not np.array_equal(reference_times[indices], times)
	):
		raise ValueError(message)
	indices.setflags(write=False)
	return indices
