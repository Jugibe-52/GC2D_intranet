"""Passive energy accumulation from method-owned accepted quadrature data."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.arrays import array_namespace
from contracts.methods.quadrature import InterpolantQuadrature, QuadratureData, StageQuadrature
from contracts.states.history import DiagnosticValue
from states.energy import energy_diagnostics


EnergyFunction = Callable[[Any, Any], Any]
_nodes, _weights = np.polynomial.legendre.leggauss(8)
_ENERGY_NODES = (_nodes + 1.) / 2.
_ENERGY_WEIGHTS = _weights / 2.


def momentum_increment(rate: EnergyFunction, quadrature: QuadratureData,
	particle_count: int, *, end_time: float | None = None,
) -> Any:
	"""Return a fresh physical increment; weighted stages remain JAX compatible.

	Adaptive output keeps the established eight-node Gauss quadrature and its
	arithmetic, evaluated along the accepted interpolant rather than saved data.
	"""
	if isinstance(quadrature, InterpolantQuadrature):
		stop = quadrature.end_time if end_time is None else end_time
		duration = stop - quadrature.start_time
		rates = [rate(quadrature.start_time + duration * node,
			quadrature.dense_state(quadrature.start_time + duration * node))
			for node in _ENERGY_NODES]
		return duration * sum(weight * value for weight, value in zip(_ENERGY_WEIGHTS, rates))
	xp = array_namespace(quadrature.times, quadrature.states, quadrature.weights)
	increment = xp.zeros(particle_count)
	if xp is not np and quadrature.weights:
		import jax
		# Trace the physical rate once even for long compositions. The ordered
		# scan retains the sequential accumulation used by the host observer.
		rates = jax.vmap(rate)(xp.stack(quadrature.times), xp.stack(quadrature.states))
		weighted = (xp.stack(quadrature.weights) * quadrature.reduction_scale)[:, None] * rates
		return jax.lax.scan(lambda total, value: (total + value, None), increment, weighted)[0] / quadrature.reduction_scale
	for time, state, weight in zip(quadrature.times, quadrature.states, quadrature.weights, strict=True):
		increment = increment + (weight * quadrature.reduction_scale) * rate(time, state)
	return increment / quadrature.reduction_scale


@dataclass(frozen=True, slots=True)
class EnergyObserver:
	"""Reusable passive-energy configuration; execution supplies physical functions."""

	initial_momentum: np.ndarray | None = None

	def __post_init__(self) -> None:
		if self.initial_momentum is not None:
			value = np.array(self.initial_momentum, dtype=float, copy=True)
			if value.ndim != 1 or not np.all(np.isfinite(value)):
				raise ValueError("Initial energy momentum must be a finite one-dimensional array.")
			value.setflags(write=False)
			object.__setattr__(self, "initial_momentum", value)

	def new_run(self, hamiltonian: EnergyFunction, momentum_rate: EnergyFunction, *,
		particle_count: int, initial_time: float,
	) -> "PreparedEnergyObserver":
		"""Start an independent accumulator without evaluating the physical functions."""
		momentum = np.zeros(particle_count) if self.initial_momentum is None else self.initial_momentum
		if momentum.shape != (particle_count,):
			raise ValueError("Initial energy momentum must contain one value per particle.")
		return PreparedEnergyObserver(hamiltonian, momentum_rate, initial_time, momentum)


class PreparedEnergyObserver:
	"""One accepted stream and its latest interval, with independent shadow samples."""

	def __init__(self, hamiltonian: EnergyFunction, momentum_rate: EnergyFunction,
		initial_time: float, initial_momentum: np.ndarray,
	) -> None:
		self.hamiltonian = hamiltonian
		self.momentum_rate = momentum_rate
		self._time = float(initial_time)
		self._momentum = initial_momentum.copy()
		self._left = self._momentum.copy()
		self._start = self._time
		self._index = -1
		self._quadrature: QuadratureData | None = None

	@property
	def momentum(self) -> np.ndarray:
		"""Return an independent snapshot of the accepted physical momentum."""
		return self._momentum.copy()

	def _checked_rate(self, time: Any, state: Any) -> np.ndarray:
		value = np.asarray(self.momentum_rate(time, state), dtype=float)
		if value.shape != self._momentum.shape or not np.all(np.isfinite(value)):
			raise ValueError("The momentum derivative must be finite with one value per particle.")
		return value

	def _increment(self, quadrature: QuadratureData, *, end_time: float | None = None) -> np.ndarray:
		with np.errstate(over="ignore", invalid="ignore"):
			value = np.asarray(momentum_increment(self._checked_rate, quadrature,
				self._momentum.size, end_time=end_time), dtype=float)
		if value.shape != self._momentum.shape or not np.all(np.isfinite(value)):
			raise ValueError("The energy-tracking momentum increment became non-finite.")
		return value

	def accept(self, step_index: int, start_time: float, end_time: float,
		quadrature: QuadratureData,
	) -> np.ndarray:
		"""Accumulate exactly once after acceptance; failed evaluations change no state."""
		tolerance = float(64 * np.finfo(float).eps) * max(1., abs(self._time), abs(start_time))
		if step_index != self._index + 1 or not np.isclose(start_time, self._time, rtol=0., atol=tolerance):
			raise ValueError("Energy steps must arrive once in sequential, continuous order.")
		if isinstance(quadrature, InterpolantQuadrature) and (
			quadrature.start_time != start_time or quadrature.end_time != end_time):
			raise ValueError("Adaptive energy quadrature must describe the accepted interval.")
		after = self._momentum + self._increment(quadrature)
		if not np.all(np.isfinite(after)):
			raise ValueError("The accumulated energy momentum became non-finite.")
		self._left, self._momentum = self._momentum, after
		self._start, self._time, self._index = start_time, end_time, step_index
		self._quadrature = quadrature
		return self.momentum

	def sample(self, time: float | np.ndarray, quadrature: StageQuadrature | None = None) -> np.ndarray:
		"""Evaluate from the latest left endpoint, never changing accepted momentum.

	Fixed interior queries require the shortened map's own quadrature. Adaptive
	queries integrate the accepted interpolant; both endpoint values are reused.
	"""
		queries = np.asarray(time, dtype=float)
		values: list[np.ndarray] = []
		for endpoint in queries.reshape(-1):
			if not min(self._start, self._time) <= endpoint <= max(self._start, self._time):
				raise ValueError("Energy samples must lie inside the current accepted interval.")
			if endpoint == self._time:
				values.append(self._momentum)
			elif endpoint == self._start:
				values.append(self._left)
			elif isinstance(self._quadrature, InterpolantQuadrature):
				values.append(self._left + self._increment(self._quadrature, end_time=float(endpoint)))
			elif quadrature is not None and queries.ndim == 0:
				values.append(self._left + self._increment(quadrature))
			else:
				raise ValueError("A fixed interior sample requires its independent shortened quadrature.")
		if not values:
			return np.empty((self._momentum.size, *queries.shape))
		return np.asarray(np.stack(values, axis=-1).reshape(self._momentum.size, *queries.shape))

	def diagnose_history(self, times: np.ndarray, physical: np.ndarray,
		auxiliary: dict[str, np.ndarray], *, energy: np.ndarray | None = None,
	) -> dict[str, DiagnosticValue]:
		"""Evaluate saved H once and preserve the established auxiliary result schema."""
		if "extended_momentum" not in auxiliary:
			return {}
		if energy is None:
			energy = np.asarray(self.hamiltonian(times, physical), dtype=float)
		return energy_diagnostics(times, physical, auxiliary, energy=energy)


__all__ = ["EnergyObserver", "PreparedEnergyObserver", "momentum_increment"]
