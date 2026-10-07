"""Shared reference, timing, and accuracy helpers for Gauss4 studies."""

from __future__ import annotations

from contracts.comparison import (AdaptiveReference,)


import numpy as np
from typing import Literal
from .reference_trajectory import _solve_adaptive

from dynamics import GuidingCenterDynamics

from ._trajectory_distances import DistanceConvention, particle_distances


def build_adaptive_reference(
	dynamics: GuidingCenterDynamics,
	initial_state: np.ndarray,
	times: np.ndarray,
	*,
	period: float | None,
	distance_convention: DistanceConvention = "periodic",
	relative_tolerance: float,
	absolute_tolerance: float,
	maximum_step: float,
	audit_relative_tolerance: float,
	audit_absolute_tolerance: float,
	audit_maximum_step: float,
) -> AdaptiveReference:
	"""Compute DOP853 and Radau histories on one prescribed output grid."""
	values = np.asarray(times, dtype=float)
	initial = np.asarray(initial_state, dtype=float)
	if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0.0):
		raise ValueError("Reference times must be a strictly increasing vector.")
	if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
		raise ValueError("The reference initial state must be a finite vector.")

	def solve(method: Literal["DOP853", "Radau"], *, rtol: float, atol: float, max_step: float) -> tuple[np.ndarray, float, int]:
		states, work = _solve_adaptive(dynamics, initial, values, method=method,
			relative_tolerance=rtol, absolute_tolerance=atol, maximum_step=max_step)
		return states, work.runtime_seconds, work.function_evaluations

	dop853, dop853_runtime, dop853_evaluations = solve(
		"DOP853",
		rtol=relative_tolerance,
		atol=absolute_tolerance,
		max_step=maximum_step,
	)
	radau, radau_runtime, radau_evaluations = solve(
		"Radau",
		rtol=audit_relative_tolerance,
		atol=audit_absolute_tolerance,
		max_step=audit_maximum_step,
	)
	distances = particle_distances(
		dop853,
		radau,
		distance_convention=distance_convention,
		period=period,
	)
	return AdaptiveReference(
		times=values,
		states=dop853,
		audit_states=radau,
		audit_distances=distances,
		dop853_runtime_seconds=dop853_runtime,
		radau_runtime_seconds=radau_runtime,
		dop853_function_evaluations=dop853_evaluations,
		radau_function_evaluations=radau_evaluations,
	)


def build_dop853_reference_with_reused_audit(
	dynamics: GuidingCenterDynamics,
	initial_state: np.ndarray,
	times: np.ndarray,
	*,
	audit_reference: AdaptiveReference,
	period: float | None,
	distance_convention: DistanceConvention = "periodic",
	relative_tolerance: float,
	absolute_tolerance: float,
	maximum_step: float,
) -> AdaptiveReference:
	"""Recompute DOP853 while retaining an aligned, previously computed Radau audit."""
	values = np.asarray(times, dtype=float)
	initial = np.asarray(initial_state, dtype=float)
	if not np.array_equal(audit_reference.times, values):
		raise ValueError("Reused Radau audit must match the reference output grid.")
	if (
		audit_reference.audit_states.shape != (initial.size, values.size)
		or not np.array_equal(audit_reference.audit_states[:, 0], initial)
	):
		raise ValueError("Reused Radau audit has a different initial state or shape.")

	states, work = _solve_adaptive(dynamics, initial, values, method="DOP853",
		relative_tolerance=relative_tolerance, absolute_tolerance=absolute_tolerance,
		maximum_step=maximum_step)
	runtime = work.runtime_seconds
	audit_states = audit_reference.audit_states
	distances = particle_distances(
		states,
		audit_states,
		distance_convention=distance_convention,
		period=period,
	)
	return AdaptiveReference(
		times=values,
		states=states,
		audit_states=audit_states,
		audit_distances=distances,
		dop853_runtime_seconds=float(runtime),
		radau_runtime_seconds=audit_reference.radau_runtime_seconds,
		dop853_function_evaluations=work.function_evaluations,
		radau_function_evaluations=audit_reference.radau_function_evaluations,
	)


def readonly_runtime_samples(values: np.ndarray) -> np.ndarray:
	"""Validate, copy, and freeze one positive timing sample vector."""
	result = np.array(values, dtype=float, copy=True)
	if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
		raise ValueError("Runtime samples must be a finite non-empty vector.")
	if np.any(result <= 0.0):
		raise ValueError("Runtime samples must be strictly positive.")
	result.setflags(write=False)
	return result


__all__: list[str] = []
