"""Bounded horizontal travel observation on accepted physical time intervals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from contracts.methods.step import StepInfo
from contracts.states.history import DiagnosticValue

PhysicalFunction = Callable[[Any, Any], Any]
PositionFunction = Callable[[Any], tuple[Any, Any]]
HORIZONTAL_METRIC_ARRAYS = frozenset(("horizontal_travel", "horizontal_min",
	"horizontal_max", "horizontal_width", "horizontal_transport_rate"))


@dataclass(frozen=True, slots=True)
class HorizontalTransportObserver:
	"""Integrate |dx/dt| with endpoint trapezoids; report travel/(cycles * width).

	Coordinates remain physical and unwrapped. Width includes the initial and
	every accepted full-step endpoint, not internal stages or shadow samples.
	"""

	cycle_duration: float = 1.0

	def __post_init__(self) -> None:
		if isinstance(self.cycle_duration, (bool, np.bool_)) or not np.isfinite(self.cycle_duration) or self.cycle_duration <= 0:
			raise ValueError("Horizontal observation requires a finite positive cycle duration.")

	def new_run(self, vector_field: PhysicalFunction, positions: PositionFunction,
		*, particle_count: int) -> PreparedHorizontalTransportObserver:
		"""Bind fresh independent accumulation to physical functions."""
		return PreparedHorizontalTransportObserver(vector_field, positions,
			particle_count, float(self.cycle_duration))


class PreparedHorizontalTransportObserver:
	"""Own only O(particles) endpoint accumulators for one execution."""

	def __init__(self, vector_field: PhysicalFunction, positions: PositionFunction,
		particle_count: int, cycle_duration: float) -> None:
		self.vector_field: PhysicalFunction = vector_field
		self.positions: PositionFunction = positions
		self.particle_count: int = particle_count
		self.cycle_duration: float = cycle_duration
		self._speed: np.ndarray | None = None
		self._travel: np.ndarray = np.zeros(particle_count)
		self._minimum: np.ndarray | None = None
		self._maximum: np.ndarray | None = None
		self._steps: int = 0
		self._duration: float = 0.0

	def _endpoint(self, time: float, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		x: np.ndarray = np.array(self.positions(state)[0], dtype=float, copy=True)
		speed: np.ndarray = np.abs(np.array(self.positions(self.vector_field(time, state))[0], dtype=float, copy=True))
		if x.shape != (self.particle_count,) or speed.shape != x.shape:
			raise ValueError("Horizontal positions and velocities must contain one value per particle.")
		if not np.all(np.isfinite(x)) or not np.all(np.isfinite(speed)):
			raise ValueError("Horizontal positions and velocities must be finite.")
		return x, speed

	def accept(self, info: StepInfo, state: np.ndarray) -> None:
		"""Observe physical endpoints once, without retaining a step history."""
		if self._speed is None:
			initial: np.ndarray
			initial, self._speed = self._endpoint(info.time, info.state_before)
			self._minimum, self._maximum = initial.copy(), initial.copy()
		x: np.ndarray
		speed: np.ndarray
		x, speed = self._endpoint(info.end_time, state)
		assert self._minimum is not None and self._maximum is not None
		self._travel += (0.5 * info.duration) * (self._speed + speed)
		self._minimum = np.minimum(self._minimum, x)
		self._maximum = np.maximum(self._maximum, x)
		self._speed = speed
		self._steps += 1
		self._duration += info.duration

	def diagnose(self, *, travel: np.ndarray | None = None, minimum: np.ndarray | None = None,
		maximum: np.ndarray | None = None, duration: float | None = None,
		steps: int | None = None) -> dict[str, DiagnosticValue]:
		"""Reduce host or compiled accumulators; zero width is explicitly undefined."""
		travel = self._travel if travel is None else travel
		minimum = self._minimum if minimum is None else minimum
		maximum = self._maximum if maximum is None else maximum
		duration = self._duration if duration is None else duration
		steps = self._steps if steps is None else steps
		if minimum is None or maximum is None or duration <= 0 or steps <= 0:
			raise ValueError("Horizontal observation requires accepted positive time intervals.")
		for value in (travel, minimum, maximum):
			if value.shape != (self.particle_count,) or not np.all(np.isfinite(value)):
				raise ValueError("Horizontal accumulators must be finite per-particle vectors.")
		width: np.ndarray = maximum - minimum
		if np.any(width < 0) or np.any(travel < 0):
			raise ValueError("Horizontal travel and width must be nonnegative.")
		cycles: float = duration / self.cycle_duration
		rate: np.ndarray = np.full(self.particle_count, np.nan)
		np.divide(travel, cycles * width, out=rate, where=width > 0)
		return {"horizontal_travel": travel, "horizontal_min": minimum,
			"horizontal_max": maximum, "horizontal_width": width,
			"horizontal_transport_rate": rate, "horizontal_observed_steps": steps,
			"horizontal_cycles": cycles, "horizontal_cycle_duration": self.cycle_duration,
			"horizontal_quadrature": "trapezoid_accepted_endpoints",
			"horizontal_zero_width": "undefined_nan",
			"horizontal_coordinates": "physical_unwrapped"}


__all__ = ["HorizontalTransportObserver", "PreparedHorizontalTransportObserver"]
