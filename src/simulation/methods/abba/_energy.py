"""Optional conjugate-momentum tracking for physical ABBA trajectories."""

from __future__ import annotations

import numpy as np

from dynamics import ExtendedHamiltonianSystem

from ...formulations.base import generalized_energy_error
from ._core import _ABBAStages


def _validate_energy_tracking(
	dynamics: object,
	*,
	enabled: bool,
	method_name: str,
) -> None:
	"""Require the Hamiltonian capability only when tracking is requested."""
	if enabled and not isinstance(dynamics, ExtendedHamiltonianSystem):
		raise TypeError(f"{method_name} energy tracking requires ExtendedHamiltonianSystem.")


def _energy_tracking_initial_state(
	physical: np.ndarray,
	*,
	particle_count: int,
	enabled: bool,
) -> np.ndarray:
	"""Append one zero conjugate momentum per independent particle."""
	value = np.asarray(physical, dtype=float)
	if not enabled:
		return value
	return np.concatenate((value, np.zeros(particle_count, dtype=float)))


def _unpack_energy_tracking_state(
	state: np.ndarray,
	*,
	physical_size: int,
	particle_count: int,
	enabled: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
	"""Split the fixed-grid workspace into physical state and optional momentum."""
	value = np.asarray(state, dtype=float)
	expected_size = physical_size + (particle_count if enabled else 0)
	if (
		value.ndim != 1
		or value.size != expected_size
		or not np.all(np.isfinite(value))
	):
		raise ValueError(
			"The ABBA integration state changed shape or became non-finite."
		)
	physical = np.asarray(value[:physical_size])
	momentum = np.asarray(value[physical_size:]) if enabled else None
	return physical, momentum


def _conjugate_momentum_increment_from_stages(
	dynamics: ExtendedHamiltonianSystem,
	start_time: float,
	duration: float,
	stages: _ABBAStages,
	*,
	particle_count: int,
) -> np.ndarray:
	"""Integrate the projected momentum kappa through one ABBA map."""
	stop_time = start_time + duration

	def momentum_derivative(time: float, state: np.ndarray) -> np.ndarray:
		value = np.asarray(
			dynamics.extended_momentum_derivative(time, state),
			dtype=float,
		)
		if value.shape != (particle_count,) or not np.all(np.isfinite(value)):
			raise ValueError(
				"Energy tracking requires one finite momentum derivative "
				"per particle."
			)
		return value

	# The duplicated GC formulation projects its summed momentum back as
	# kappa=k/2, hence the additional factor one half after the four shears.
	with np.errstate(over="ignore", invalid="ignore"):
		doubled_increment = duration / 2.0 * (
			momentum_derivative(start_time, stages.v_initial)
			+ momentum_derivative(start_time, stages.u_first)
			+ momentum_derivative(stop_time, stages.u_first)
			+ momentum_derivative(stop_time, stages.v_final)
		)
		increment = np.asarray(doubled_increment / 2.0)
	if not np.all(np.isfinite(increment)):
		raise ValueError("The energy-tracking momentum increment became non-finite.")
	return increment


def _pack_energy_tracking_state(
	physical: np.ndarray,
	momentum: np.ndarray | None,
) -> np.ndarray:
	"""Return the next fixed-grid state without changing the physical map."""
	value = np.asarray(physical, dtype=float)
	if momentum is None:
		return value
	momentum_value = np.asarray(momentum, dtype=float)
	if momentum_value.ndim != 1 or not np.all(np.isfinite(momentum_value)):
		raise ValueError("The tracked conjugate momentum became non-finite.")
	return np.concatenate((value, momentum_value))


def _energy_tracking_diagnostics(
	times: np.ndarray,
	states: np.ndarray,
	momentum: np.ndarray | None,
	dynamics: object,
) -> dict[str, np.ndarray | float | str]:
	"""Return standard diagnostics for an optionally tracked physical run."""
	if momentum is None:
		return {}
	value = np.asarray(momentum, dtype=float)
	return {
		"extended_momentum": value,
		"extended_momentum_normalization": "kappa_equals_k_over_2",
		"energy_error": generalized_energy_error(times, states, value, dynamics),
	}


__all__: list[str] = []
