# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Private BM4 integration paths shared by GC and FC systems."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import sys
from typing import Callable, Protocol

import numpy as np

from classes.trajectory.fc import FCState, TrajectoryFC
from classes.trajectory.gc import GCState, TrajectoryGC

from .solution import Solution


Flow = Callable[[float, float, np.ndarray], np.ndarray]

_BM4_HALF_STAGES = np.asarray(
	[
		0.0792036964311957,
		0.1303114101821663,
		0.2228614958676077,
		-0.3667132690474257,
		0.3246481886897062,
		0.1096884778767498,
	],
	dtype=float,
)
_BM4_STAGES = np.concatenate((_BM4_HALF_STAGES, np.flip(_BM4_HALF_STAGES)))
_BM4_ORDERS = np.tile(np.asarray([1, 0], dtype=int), _BM4_HALF_STAGES.size)

_COUPLING_BASE = np.asarray(
	[[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],
	dtype=float,
)
_COUPLING_COS = np.asarray(
	[[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]],
	dtype=float,
)
_COUPLING_SIN = np.asarray(
	[[0, -1, 0, 1], [1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0]],
	dtype=float,
)


class _EnergySystem(Protocol):
	def _energy_error(self, solution: Solution) -> float:
		"""Return the maximum generalized-energy drift."""


class _GCSystem(_EnergySystem, Protocol):
	trajectory: TrajectoryGC

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the guiding-centre equations."""

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate minus the explicit time derivative of the Hamiltonian."""


class _FCSystem(_EnergySystem, Protocol):
	trajectory: TrajectoryFC

	def electric_acceleration(
		self,
		t: float,
		x: np.ndarray,
		y: np.ndarray,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Evaluate the electric acceleration."""

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate minus the explicit time derivative of the Hamiltonian."""


@dataclass(frozen=True, slots=True)
class _GCExtendedState:
	"""Two GC copies and the optional extended momentum used by BM4."""

	first: np.ndarray
	second: np.ndarray
	momentum: np.ndarray | None = None

	@classmethod
	def unpack(
		cls,
		value: np.ndarray,
		*,
		physical_size: int,
		particle_count: int,
		track_momentum: bool,
	) -> _GCExtendedState:
		expected_size = 2 * physical_size + (particle_count if track_momentum else 0)
		if value.ndim == 0 or value.shape[0] != expected_size:
			raise ValueError("The GC extended flow changed the state shape.")
		momentum = value[2 * physical_size :] if track_momentum else None
		return cls(
			first=value[:physical_size],
			second=value[physical_size : 2 * physical_size],
			momentum=momentum,
		)

	def pack(self) -> np.ndarray:
		parts = (self.first, self.second)
		return np.concatenate(parts if self.momentum is None else (*parts, self.momentum))


@dataclass(frozen=True, slots=True)
class _FCExtendedState:
	"""One physical FC state and the optional extended momentum used by BM4."""

	physical: FCState
	momentum: np.ndarray | None = None

	@classmethod
	def unpack(
		cls,
		value: np.ndarray,
		*,
		trajectory: TrajectoryFC,
		physical_size: int,
		particle_count: int,
		track_momentum: bool,
	) -> _FCExtendedState:
		expected_size = physical_size + (particle_count if track_momentum else 0)
		if value.ndim == 0 or value.shape[0] != expected_size:
			raise ValueError("The FC extended flow changed the state shape.")
		physical = trajectory.split(value[:physical_size])
		momentum = value[physical_size:] if track_momentum else None
		return cls(physical=physical, momentum=momentum)

	def pack(self, trajectory: TrajectoryFC) -> np.ndarray:
		physical = trajectory.pack(*self.physical)
		return (
			physical
			if self.momentum is None
			else np.concatenate((physical, self.momentum))
		)


class _Progress:
	def __init__(self, label: str, total: int) -> None:
		self.label = label
		self.total = max(total, 1)
		self.every = max(self.total // 100, 1)
		self.steps = 0

	def update(self, t: float) -> None:
		self.steps += 1
		if self.steps % self.every and self.steps < self.total:
			return
		fraction = min(self.steps / self.total, 1.0)
		width = 30
		filled = int(width * fraction)
		bar = "=" * filled
		if filled < width:
			bar += ">" + " " * (width - filled - 1)
		print(
			f"\r{self.label} [{bar}] {fraction:6.1%} "
			f"({self.steps}/{self.total}, t={t:.6g})",
			end="",
			file=sys.stderr,
			flush=True,
		)

	def close(self) -> None:
		print(file=sys.stderr, flush=True)


def _step_count(duration: float, maximum_step: float) -> int:
	ratio = duration / maximum_step
	return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _validate_inputs(
	t_span: tuple[float, float],
	state: np.ndarray,
	step: float,
	n_save_step: int,
) -> tuple[float, float, np.ndarray, float, np.ndarray]:
	span = np.asarray(t_span, dtype=float)
	if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
		raise ValueError("`t_span` must contain two finite, increasing times.")
	maximum_step = float(step)
	if (
		isinstance(step, (bool, np.bool_))
		or not np.isfinite(maximum_step)
		or maximum_step <= 0
	):
		raise ValueError("`step` must be positive and finite.")
	if (
		isinstance(n_save_step, (bool, np.bool_))
		or not isinstance(n_save_step, (int, np.integer))
		or n_save_step < 2
	):
		raise ValueError("`n_save_step` must be an integer of at least 2.")
	value = np.asarray(state, dtype=float)
	if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
		raise ValueError("The initial state must be a finite, non-empty vector.")
	t0, tf = float(span[0]), float(span[1])
	times = np.linspace(t0, tf, int(n_save_step), dtype=float)
	return t0, tf, value, maximum_step, times


def _checked_flow(
	flow: Flow,
	h: float,
	t: float,
	state: np.ndarray,
) -> np.ndarray:
	result = np.asarray(flow(h, t, state))
	if result.shape != state.shape:
		raise ValueError("An integration flow changed the state shape.")
	return result


def _advance(
	flow: Flow,
	adjoint_flow: Flow,
	t: float,
	state: np.ndarray,
	step: float,
) -> tuple[float, np.ndarray]:
	for coefficient, order in zip(_BM4_STAGES, _BM4_ORDERS, strict=True):
		stage = float(coefficient * step)
		if order == 0:
			state = _checked_flow(flow, stage, t + stage, state)
		else:
			state = _checked_flow(adjoint_flow, stage, t, state)
		t += stage
	return t, state


def _planned_steps(times: np.ndarray, maximum_step: float) -> int:
	return sum(
		_step_count(float(stop - start), maximum_step)
		for start, stop in zip(times[:-1], times[1:], strict=True)
		if stop > start
	)


def _solve_composed(
	flow: Flow,
	adjoint_flow: Flow,
	t_span: tuple[float, float],
	state: np.ndarray,
	step: float,
	n_save_step: int,
	*,
	progress: bool,
	label: str,
) -> Solution:
	t, _tf, value, maximum_step, times = _validate_inputs(
		t_span,
		state,
		step,
		n_save_step,
	)
	result = np.empty(value.shape + (times.size,), dtype=value.dtype)
	result[:, 0] = value
	n_steps = 0
	progress_bar = _Progress(label, _planned_steps(times, maximum_step)) if progress else None

	try:
		for output_index, target in enumerate(times[1:], start=1):
			duration = float(target) - t
			count = _step_count(duration, maximum_step)
			internal_step = duration / count
			segment_start = t
			for index in range(count):
				t, value = _advance(flow, adjoint_flow, t, value, internal_step)
				t = segment_start + (index + 1) * internal_step
				n_steps += 1
				if progress_bar is not None:
					progress_bar.update(t)
			result[:, output_index] = value
	finally:
		if progress_bar is not None:
			progress_bar.close()

	return Solution(t=times, y=result, n_steps=n_steps)


@lru_cache(maxsize=256)
def _gc_coupling(step: float) -> np.ndarray:
	frequency = 10.0
	return np.asarray(
		(
			_COUPLING_BASE
			+ np.cos(2 * frequency * step) * _COUPLING_COS
			+ np.sin(2 * frequency * step) * _COUPLING_SIN
		)
		/ 2
	)


def _couple_gc_state(
	step: float,
	state: _GCExtendedState,
	trajectory: TrajectoryGC,
) -> _GCExtendedState:
	first = trajectory.split(state.first)
	second = trajectory.split(state.second)
	blocks = np.stack((*first, *second), axis=0)
	coupled = np.asarray(np.einsum("ij,j...->i...", _gc_coupling(step), blocks))
	return _GCExtendedState(
		first=trajectory.pack(coupled[0], coupled[1]),
		second=trajectory.pack(coupled[2], coupled[3]),
		momentum=state.momentum,
	)


def _updated_momentum(
	system: _GCSystem | _FCSystem,
	momentum: np.ndarray | None,
	step: float,
	t: float,
	physical_state: np.ndarray,
) -> np.ndarray | None:
	if momentum is None:
		return None
	derivative: np.ndarray = np.asarray(
		system.extended_momentum_derivative(t, physical_state)
	)
	if derivative.shape != momentum.shape:
		raise ValueError("The extended-momentum derivative changed its shape.")
	return np.asarray(momentum + step * derivative)


def solve_gc(
	system: _GCSystem,
	state: np.ndarray,
	*,
	t_span: tuple[float, float],
	step: float,
	n_save_step: int,
	check_energy: bool,
	progress: bool,
) -> Solution:
	"""Integrate one GC physical state in doubled extended phase space."""
	trajectory = system.trajectory
	physical_state = np.asarray(state, dtype=float)
	trajectory.split(physical_state)
	physical_size = physical_state.size
	particle_count = trajectory.particle_count(physical_state)
	track_momentum = bool(check_energy)
	extended = _GCExtendedState(
		first=physical_state,
		second=physical_state,
		momentum=np.zeros(particle_count) if track_momentum else None,
	)

	def unpack(value: np.ndarray) -> _GCExtendedState:
		return _GCExtendedState.unpack(
			value,
			physical_size=physical_size,
			particle_count=particle_count,
			track_momentum=track_momentum,
		)

	def vector_field(t: float, value: np.ndarray) -> np.ndarray:
		derivative = np.asarray(system.vector_field(t, value))
		if derivative.shape != value.shape:
			raise ValueError("The GC vector field changed the physical state shape.")
		return derivative

	def flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		current = unpack(value)
		second = current.second + h * vector_field(t, current.first)
		momentum = _updated_momentum(system, current.momentum, h, t, current.first)
		first = current.first + h * vector_field(t, second)
		momentum = _updated_momentum(system, momentum, h, t, second)
		return _couple_gc_state(
			h,
			_GCExtendedState(first, second, momentum),
			trajectory,
		).pack()

	def adjoint_flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		current = _couple_gc_state(h, unpack(value), trajectory)
		first = current.first + h * vector_field(t, current.second)
		momentum = _updated_momentum(system, current.momentum, h, t, current.second)
		second = current.second + h * vector_field(t, first)
		momentum = _updated_momentum(system, momentum, h, t, first)
		return _GCExtendedState(first, second, momentum).pack()

	solution = _solve_composed(
		flow,
		adjoint_flow,
		t_span,
		extended.pack(),
		step,
		n_save_step,
		progress=progress,
		label="SystemGC",
	)
	final_state = unpack(solution.y)
	first = trajectory.split(final_state.first)
	second = trajectory.split(final_state.second)
	solution.y = trajectory.pack(
		(first.x + second.x) / 2,
		(first.y + second.y) / 2,
	)
	if final_state.momentum is not None:
		solution.k = final_state.momentum / 2
		solution.err = system._energy_error(solution)
	return solution


def _real_imaginary(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	return np.asarray(value.real), np.asarray(value.imag)


def _cyclotron_step(
	trajectory: TrajectoryFC,
	state: FCState,
	step: float,
) -> FCState:
	rotation = np.exp(-1j * step * trajectory.larmor_frequency)
	x, y = _real_imaginary(
		state.x
		+ 1j * state.y
		+ 1j
		* trajectory.rho
		* np.sign(trajectory.eta)
		* (rotation - 1)
		* (state.vx + 1j * state.vy)
	)
	vx, vy = _real_imaginary(rotation * (state.vx + 1j * state.vy))
	return FCState(x, y, vx, vy)


def solve_fc(
	system: _FCSystem,
	state: np.ndarray,
	*,
	t_span: tuple[float, float],
	step: float,
	n_save_step: int,
	check_energy: bool,
	progress: bool,
) -> Solution:
	"""Integrate one FC physical state with private direct and adjoint flows."""
	trajectory = system.trajectory
	physical_state = np.asarray(state, dtype=float)
	physical = trajectory.split(physical_state)
	physical_size = physical_state.size
	particle_count = trajectory.particle_count(physical_state)
	track_momentum = bool(check_energy)
	extended = _FCExtendedState(
		physical=physical,
		momentum=np.zeros(particle_count) if track_momentum else None,
	)

	def unpack(value: np.ndarray) -> _FCExtendedState:
		return _FCExtendedState.unpack(
			value,
			trajectory=trajectory,
			physical_size=physical_size,
			particle_count=particle_count,
			track_momentum=track_momentum,
		)

	def flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		current = unpack(value)
		physical = _cyclotron_step(trajectory, current.physical, h)
		acceleration_x, acceleration_y = system.electric_acceleration(
			t,
			physical.x,
			physical.y,
		)
		physical = FCState(
			physical.x,
			physical.y,
			physical.vx + h * acceleration_x,
			physical.vy + h * acceleration_y,
		)
		physical_array = trajectory.pack(*physical)
		momentum = _updated_momentum(
			system,
			current.momentum,
			h,
			t,
			physical_array,
		)
		return _FCExtendedState(physical, momentum).pack(trajectory)

	def adjoint_flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		current = unpack(value)
		acceleration_x, acceleration_y = system.electric_acceleration(
			t,
			current.physical.x,
			current.physical.y,
		)
		physical = FCState(
			current.physical.x,
			current.physical.y,
			current.physical.vx + h * acceleration_x,
			current.physical.vy + h * acceleration_y,
		)
		physical_array = trajectory.pack(*physical)
		momentum = _updated_momentum(
			system,
			current.momentum,
			h,
			t,
			physical_array,
		)
		physical = _cyclotron_step(trajectory, physical, h)
		return _FCExtendedState(physical, momentum).pack(trajectory)

	solution = _solve_composed(
		flow,
		adjoint_flow,
		t_span,
		extended.pack(trajectory),
		step,
		n_save_step,
		progress=progress,
		label="SystemFC",
	)
	final_state = unpack(solution.y)
	solution.y = trajectory.pack(*final_state.physical)
	if final_state.momentum is not None:
		solution.k = final_state.momentum
		solution.err = system._energy_error(solution)
	return solution


__all__: list[str] = []
