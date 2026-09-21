"""Bound workspace operations shared by the implicit ABBA family."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dynamics import ExtendedHamiltonianSystem, GuidingCenterDynamics

from ..._result import DiagnosticValue
from ._energy import (
	_conjugate_momentum_increment_from_stages,
	_energy_tracking_diagnostics,
	_energy_tracking_initial_state,
	_pack_energy_tracking_state,
	_unpack_energy_tracking_state,
)
from .maps.extended import _synchronized_extended_time
from .records import PhysicalProjectionTrace, ProjectedMapResult


@dataclass(frozen=True, slots=True)
class StatePolicy:
	"""Functions selected once for physical, tracked, or full-state evolution."""

	initialize: Callable[[np.ndarray, float], np.ndarray]
	unpack: Callable[[float, np.ndarray], np.ndarray]
	finish_step: Callable[
		[float, float, np.ndarray, tuple[ProjectedMapResult, ...]], np.ndarray
	]
	extract: Callable[
		[np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, DiagnosticValue]]
	]


def physical_state_policy(
	dynamics: object,
	*,
	physical_size: int,
	particle_count: int,
	track_energy: bool,
) -> StatePolicy:
	"""Bind physical workspace operations and the optional triangular update."""
	def initialize(z0: np.ndarray, t0: float) -> np.ndarray:
		return _energy_tracking_initial_state(
			z0, particle_count=particle_count, enabled=track_energy
		)

	def unpack(t: float, workspace: np.ndarray) -> np.ndarray:
		return _unpack_energy_tracking_state(
			workspace, physical_size=physical_size,
			particle_count=particle_count, enabled=track_energy,
		)[0]

	def finish_step(
		t: float,
		h: float,
		workspace: np.ndarray,
		projections: tuple[ProjectedMapResult, ...],
	) -> np.ndarray:
		_, momentum = _unpack_energy_tracking_state(
			workspace, physical_size=physical_size,
			particle_count=particle_count, enabled=track_energy,
		)
		if momentum is not None:
			assert isinstance(dynamics, ExtendedHamiltonianSystem)
			momentum = momentum.copy()
			for projection in projections:
				assert isinstance(projection.trace, PhysicalProjectionTrace)
				for base_map in projection.trace.maps:
					momentum += _conjugate_momentum_increment_from_stages(
						dynamics, base_map.start_time, base_map.duration,
						base_map.stages, particle_count=particle_count,
					)
		return _pack_energy_tracking_state(projections[-1].state, momentum)

	def extract(
		times: np.ndarray, history: np.ndarray
	) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
		states = np.asarray(history[:physical_size])
		momentum = np.asarray(history[physical_size:]) if track_energy else None
		return states, dict(_energy_tracking_diagnostics(times, states, momentum, dynamics))

	return StatePolicy(initialize, unpack, finish_step, extract)


def _fully_extended_energy_diagnostics(
	dynamics: GuidingCenterDynamics, extended_history: np.ndarray
) -> dict[str, DiagnosticValue]:
	"""Return direct-k energy histories without changing normalization."""
	physical_hamiltonian = np.asarray([
		float(np.asarray(dynamics.hamiltonian(
			float(extended_history[2, index]), extended_history[:2, index],
		)).reshape(-1)[0])
		for index in range(extended_history.shape[1])
	])
	generalized_energy = physical_hamiltonian + extended_history[3]
	generalized_error = generalized_energy - generalized_energy[0]
	return {
		"extended_time": np.asarray(extended_history[2]),
		"extended_momentum": np.asarray(extended_history[3]),
		"extended_momentum_normalization": "direct_k",
		"physical_hamiltonian": physical_hamiltonian,
		"generalized_energy": generalized_energy,
		"generalized_energy_error": generalized_error,
		"energy_error": float(np.max(np.abs(generalized_error))),
	}


def extended_state_policy(dynamics: GuidingCenterDynamics) -> StatePolicy:
	"""Bind intrinsic (x,y,t,k) evolution and its public physical extraction."""
	def initialize(z0: np.ndarray, t0: float) -> np.ndarray:
		return np.concatenate((z0, (float(t0), 0.0)))

	def unpack(t: float, workspace: np.ndarray) -> np.ndarray:
		return _synchronized_extended_time(workspace, t, context="The internal state")

	def finish_step(
		t: float,
		h: float,
		workspace: np.ndarray,
		projections: tuple[ProjectedMapResult, ...],
	) -> np.ndarray:
		return _synchronized_extended_time(
			projections[-1].state, t + h, context="The fully extended ABBA map"
		)

	def extract(
		times: np.ndarray, history: np.ndarray
	) -> tuple[np.ndarray, dict[str, DiagnosticValue]]:
		history[2] = times
		return np.asarray(history[:2]), _fully_extended_energy_diagnostics(dynamics, history)

	return StatePolicy(initialize, unpack, finish_step, extract)


__all__: list[str] = []
