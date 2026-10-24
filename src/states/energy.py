"""Assemble passive energy histories from supplied physical Hamiltonian values."""

import numpy as np
from contracts.states.history import DiagnosticValue


def energy_diagnostics(times: np.ndarray, physical: np.ndarray,
	auxiliary: dict[str, np.ndarray], *, energy: np.ndarray | None = None,
) -> dict[str, DiagnosticValue]:
	"""Assemble energy diagnostics from physical values evaluated by the caller.

	This output conversion evaluates no dynamics. The method supplies H on
	the saved states; this output helper only combines H with stored kappa.
	"""
	momentum: np.ndarray | None = auxiliary.get("extended_momentum")
	if momentum is None:
		return {}
	if energy is None:
		raise ValueError("Energy diagnostics require caller-supplied Hamiltonian values.")
	energy = np.asarray(energy, dtype=float)
	if energy.ndim == 1:
		energy = energy[np.newaxis, :]
	if energy.shape != momentum.shape or not np.all(np.isfinite(energy)):
		raise ValueError("Hamiltonian history must be finite with shape (particles, times).")
	balance: np.ndarray = energy + momentum
	error: np.ndarray = balance - balance[:, :1]
	return {
		"extended_time": np.broadcast_to(times, momentum.shape).copy(),
		"extended_momentum": momentum,
		"extended_momentum_normalization": "physical_kappa",
		"physical_hamiltonian": energy,
		"hamiltonian": energy,
		"energy_drift": energy - energy[:, :1],
		"generalized_energy": balance,
		"generalized_energy_error": error,
		"energy_error": float(np.max(np.abs(error))),
	}
