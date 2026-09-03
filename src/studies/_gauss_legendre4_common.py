"""Shared reference, timing, and accuracy helpers for Gauss4 studies."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from dynamics import GuidingCenterDynamics

from ._trajectory_distances import DistanceConvention, particle_distances


@dataclass(frozen=True, slots=True)
class AdaptiveReference:
	"""DOP853 trajectory with an independent Radau resolution audit."""

	times: np.ndarray
	states: np.ndarray
	audit_states: np.ndarray
	audit_distances: np.ndarray
	dop853_runtime_seconds: float
	radau_runtime_seconds: float
	dop853_function_evaluations: int
	radau_function_evaluations: int

	def __post_init__(self) -> None:
		"""Own immutable, aligned reference and audit arrays."""
		times = np.array(self.times, dtype=float, copy=True)
		states = np.array(self.states, dtype=float, copy=True)
		audit_states = np.array(self.audit_states, dtype=float, copy=True)
		distances = np.array(self.audit_distances, dtype=float, copy=True)
		if (
			times.ndim != 1
			or times.size < 2
			or states.ndim != 2
			or states.shape != audit_states.shape
			or states.shape[1] != times.size
			or distances.ndim != 2
			or distances.shape[1] != times.size
			or not all(
				np.all(np.isfinite(value))
				for value in (times, states, audit_states, distances)
			)
			or np.any(distances < 0.0)
		):
			raise ValueError("Adaptive reference arrays are invalid or misaligned.")
		for value in (times, states, audit_states, distances):
			value.setflags(write=False)
		object.__setattr__(self, "times", times)
		object.__setattr__(self, "states", states)
		object.__setattr__(self, "audit_states", audit_states)
		object.__setattr__(self, "audit_distances", distances)

	@property
	def time_integrated_rms_floor(self) -> float:
		"""Return the time-integrated particle-RMS DOP853/Radau discrepancy."""
		particle_rms_squared = np.mean(self.audit_distances**2, axis=0)
		return float(
			np.sqrt(
				np.trapz(particle_rms_squared, self.times)
				/ float(self.times[-1] - self.times[0])
			)
		)

	@property
	def final_rms_floor(self) -> float:
		"""Return the final-time particle-RMS DOP853/Radau discrepancy."""
		return float(np.sqrt(np.mean(self.audit_distances[:, -1] ** 2)))


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

	def solve(method: str, *, rtol: float, atol: float, max_step: float) -> tuple[np.ndarray, float, int]:
		started = perf_counter()
		result = solve_ivp(
			fun=lambda time, state: dynamics.vector_field(time, state),
			t_span=(float(values[0]), float(values[-1])),
			y0=initial,
			method=method,
			t_eval=values,
			rtol=rtol,
			atol=atol,
			max_step=max_step,
			dense_output=False,
			vectorized=False,
		)
		runtime = perf_counter() - started
		if not result.success:
			raise RuntimeError(f"{method} reference integration failed: {result.message}")
		states = np.asarray(result.y, dtype=float)
		if states.shape != (initial.size, values.size) or not np.all(
			np.isfinite(states)
		):
			raise ValueError(f"{method} returned an invalid reference history.")
		states[:, 0] = initial
		return states, float(runtime), int(result.nfev)

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
