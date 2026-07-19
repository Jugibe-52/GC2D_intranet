# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Private BM4 integration paths shared by GC and FC systems."""

from __future__ import annotations

from functools import lru_cache
import math
import sys
from typing import Callable, Protocol

import numpy as np

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
	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Evaluate the guiding-centre equations."""

	def extended_momentum_derivative(
		self,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Evaluate minus the explicit time derivative of the Hamiltonian."""


class _FCSystem(_EnergySystem, Protocol):
	def _flow(
		self,
		h: float,
		t: float,
		state: np.ndarray,
		*,
		check_energy: bool,
	) -> np.ndarray:
		"""Apply the direct FC split flow."""

	def _adjoint_flow(
		self,
		h: float,
		t: float,
		state: np.ndarray,
		*,
		check_energy: bool,
	) -> np.ndarray:
		"""Apply the adjoint FC split flow."""


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
	"""Integrate one GC block state in doubled extended phase space."""
	physical_state = np.asarray(state, dtype=float)
	if physical_state.ndim != 1 or physical_state.size % 2:
		raise ValueError("A GC state must contain equally sized x and y blocks.")
	state_size = physical_state.size
	trajectory_count = state_size // 2
	track_momentum = bool(check_energy)
	combined_state = np.concatenate((physical_state, physical_state))
	if track_momentum:
		combined_state = np.concatenate((combined_state, np.zeros(trajectory_count)))
	expected_size = combined_state.size

	def split_copies(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
		if value.shape[0] != expected_size:
			raise ValueError("The GC extended flow changed the state shape.")
		first = value[:state_size]
		second = value[state_size : 2 * state_size]
		momentum = value[2 * state_size :] if track_momentum else None
		return first, second, momentum

	def split_variables(value: np.ndarray) -> tuple[np.ndarray, ...]:
		first, second, momentum = split_copies(value)
		variables: tuple[np.ndarray, ...] = (
			first[:trajectory_count],
			first[trajectory_count:],
			second[:trajectory_count],
			second[trajectory_count:],
		)
		return variables if momentum is None else (*variables, momentum)

	def vector_field(t: float, value: np.ndarray) -> np.ndarray:
		derivative = np.asarray(system.vector_field(t, value))
		if derivative.shape != value.shape:
			raise ValueError("The GC vector field changed the physical state shape.")
		return derivative

	def update_momentum(
		momentum: np.ndarray | None,
		h: float,
		t: float,
		value: np.ndarray,
	) -> None:
		if momentum is not None:
			momentum += h * np.asarray(system.extended_momentum_derivative(t, value))

	@lru_cache(maxsize=256)
	def coupling(h: float) -> np.ndarray:
		frequency = 10.0
		return np.asarray(
			(
				_COUPLING_BASE
				+ np.cos(2 * frequency * h) * _COUPLING_COS
				+ np.sin(2 * frequency * h) * _COUPLING_SIN
			)
			/ 2
		)

	def coupled_state(h: float, blocks: tuple[np.ndarray, ...]) -> np.ndarray:
		return np.asarray(
			np.einsum("ij,j...->i...", coupling(h), np.stack(blocks, axis=0))
		).flatten()

	def flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		first, second, momentum = split_copies(value)
		second += h * vector_field(t, first)
		update_momentum(momentum, h, t, first)
		first += h * vector_field(t, second)
		update_momentum(momentum, h, t, second)
		result = coupled_state(h, tuple(np.split(np.concatenate((first, second)), 4)))
		return result if momentum is None else np.concatenate((result, momentum))

	def adjoint_flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		variables = split_variables(value)
		physical_variables = variables[:-1] if track_momentum else variables
		result = coupled_state(h, physical_variables)
		if track_momentum:
			result = np.concatenate((result, variables[-1]))
		first, second, momentum = split_copies(result)
		first += h * vector_field(t, second)
		update_momentum(momentum, h, t, second)
		second += h * vector_field(t, first)
		update_momentum(momentum, h, t, first)
		parts = (first, second) if momentum is None else (first, second, momentum)
		return np.concatenate(parts)

	solution = _solve_composed(
		flow,
		adjoint_flow,
		t_span,
		combined_state,
		step,
		n_save_step,
		progress=progress,
		label="SystemGC",
	)
	variables = split_variables(solution.y)
	solution.y = np.concatenate(
		(
			(variables[0] + variables[2]) / 2,
			(variables[1] + variables[3]) / 2,
		),
		axis=0,
	)
	if track_momentum:
		solution.k = variables[-1] / 2
		solution.err = system._energy_error(solution)
	return solution


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
	"""Integrate one FC block state with its explicit split flows."""
	physical_state = np.asarray(state, dtype=float)
	if physical_state.ndim != 1 or physical_state.size % 4:
		raise ValueError("An FC state must contain equally sized x, y, vx and vy blocks.")
	trajectory_count = physical_state.size // 4
	combined_state = physical_state
	if check_energy:
		combined_state = np.concatenate((physical_state, np.zeros(trajectory_count)))

	solution = _solve_composed(
		lambda h, t, value: system._flow(
			h,
			t,
			value,
			check_energy=check_energy,
		),
		lambda h, t, value: system._adjoint_flow(
			h,
			t,
			value,
			check_energy=check_energy,
		),
		t_span,
		combined_state,
		step,
		n_save_step,
		progress=progress,
		label="SystemFC",
	)
	if check_energy:
		components = np.split(solution.y, 5, axis=0)
		solution.y = np.concatenate(components[:4], axis=0)
		solution.k = components[4]
		solution.err = system._energy_error(solution)
	return solution


__all__: list[str] = []
