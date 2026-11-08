"""Physical Hamiltonian and analytic time derivatives at physical endpoints."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import numpy as np
from contracts.methods.step import StepInfo
from contracts.states.history import DiagnosticValue

EnergyFunction = Callable[[Any, Any], Any]


def endpoint_diagnostics(before: Any, after: Any, derivative_before: Any,
	derivative_after: Any) -> dict[str, Any]:
	"""Describe one accepted interval; derivatives are positive partial_t H."""
	return {"energy_start": before, "energy_end": after,
			"energy_change": after - before,
			"energy_time_derivative_start": derivative_before,
			"energy_time_derivative_end": derivative_after}


@dataclass(frozen=True, slots=True)
class EnergyObserver:
	"""Request endpoint H and analytic partial_t H, without auxiliary integration."""

	def new_run(self, hamiltonian: EnergyFunction, time_derivative: EnergyFunction,
		*, particle_count: int) -> PreparedEnergyObserver:
		"""Bind independent physical functions for one run."""
		return PreparedEnergyObserver(hamiltonian, time_derivative, particle_count)


class PreparedEnergyObserver:
	"""Validate endpoint values and expose saved physical-energy histories."""

	def __init__(self, hamiltonian: EnergyFunction, time_derivative: EnergyFunction,
		particle_count: int) -> None:
		self.hamiltonian: EnergyFunction = hamiltonian
		self.time_derivative: EnergyFunction = time_derivative
		self.particle_count: int = particle_count
		self._last_time: float | None = None
		self._last: tuple[np.ndarray, np.ndarray] | None = None

	def evaluate(self, time: Any, state: Any) -> tuple[np.ndarray, np.ndarray]:
		"""Evaluate the physical model, without differencing or stage access."""
		# Own endpoint snapshots even when a physical function reuses its buffer.
		energy: np.ndarray = np.array(self.hamiltonian(time, state), dtype=float, copy=True)
		derivative: np.ndarray = np.array(self.time_derivative(time, state), dtype=float, copy=True)
		expected: tuple[int, ...] = (self.particle_count, *np.shape(time))
		if energy.shape != expected or derivative.shape != expected:
			raise ValueError("Energy and its time derivative must contain one value per particle and time.")
		if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(derivative)):
			raise ValueError("Energy and its time derivative must be finite.")
		return energy, derivative

	def accept(self, info: StepInfo, state: np.ndarray) -> dict[str, Any]:
		"""Evaluate each accepted endpoint once and return interval observations."""
		before: np.ndarray
		rate_before: np.ndarray
		if self._last is not None and self._last_time == info.time:
			before, rate_before = self._last
		else:
			before, rate_before = self.evaluate(info.time, info.state_before)
		after: np.ndarray
		rate_after: np.ndarray
		after, rate_after = self.evaluate(info.end_time, state)
		self._last_time, self._last = info.end_time, (after, rate_after)
		return endpoint_diagnostics(before, after, rate_before, rate_after)

	def diagnose_history(self, times: np.ndarray, physical: np.ndarray,
		*, energy: np.ndarray | None = None,
		derivative: np.ndarray | None = None) -> dict[str, DiagnosticValue]:
		"""Report values at saved physical samples, with no conservation claim."""
		if energy is None or derivative is None:
			energy, derivative = self.evaluate(times, physical)
		expected: tuple[int, int] = (self.particle_count, times.size)
		if energy.shape != expected or derivative.shape != expected:
			raise ValueError("Energy histories must have shape (particles, saved times).")
		if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(derivative)):
			raise ValueError("Energy histories must be finite.")
		return {"physical_hamiltonian": energy, "hamiltonian": energy,
				"energy_drift": energy - energy[:, :1],
				"hamiltonian_time_derivative": derivative}


__all__ = ["EnergyObserver", "PreparedEnergyObserver"]
