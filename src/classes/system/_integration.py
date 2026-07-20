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


# A split flow receives a (possibly signed) stage duration, the time at which
# that subflow is evaluated, and a packed state. It must return the same layout.
Flow = Callable[[float, float, np.ndarray], np.ndarray]

# These dimensionless BM4 coefficients are signed fractions of one internal
# step. The second half mirrors the first, producing twelve palindromic stages;
# ``_BM4_ORDERS`` selects adjoint (1) and direct (0) subflows alternately. This
# symmetry cancels the leading splitting errors and yields fourth-order accuracy.
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

# The GC integrator duplicates the physical state. These 4x4 matrices act on
# coordinate blocks ordered as ``(x_first, y_first, x_second, y_second)`` and
# rotate both copies around their common diagonal. Splitting the matrix into
# constant, cosine, and sine terms makes it cheap to evaluate for any step size.
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
	"""Minimal interface required by the optional energy diagnostic."""

	def _energy_error(self, solution: Solution) -> float:
		"""Return the maximum generalized-energy drift."""


class _GCSystem(_EnergySystem, Protocol):
	"""Operations that the private GC integrator consumes from a system."""

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
	"""Operations that the private FC integrator consumes from a system."""

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
	"""Two GC copies and the optional extended momentum used by BM4.

	``first`` and ``second`` both have leading size ``2 * particle_count`` in
	component-major ``[x, y]`` order. ``momentum`` has one value per particle;
	all three arrays may also carry matching trailing saved-time axes.
	"""

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
		"""Split the leading packed axis into the two copies and momentum.

		``physical_size`` is the flattened leading size of one GC copy, whereas
		``particle_count`` is the size of each individual coordinate block.
		"""
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
		"""Pack the two physical copies and optional momentum for a flow."""
		parts = (self.first, self.second)
		return np.concatenate(parts if self.momentum is None else (*parts, self.momentum))


@dataclass(frozen=True, slots=True)
class _FCExtendedState:
	"""One physical FC state and the optional extended momentum used by BM4.

	The four blocks in ``physical`` each contain ``particle_count`` values in
	``[x, y, vx, vy]`` order. ``momentum`` adds one conjugate-to-time value per
	particle when the energy diagnostic is active. Trailing time axes are kept.
	"""

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
		"""Validate and split the packed FC state along its leading axis.

		``physical_size`` counts all four physical blocks; ``particle_count``
		counts the entries in one block and therefore also in ``momentum``.
		"""
		expected_size = physical_size + (particle_count if track_momentum else 0)
		if value.ndim == 0 or value.shape[0] != expected_size:
			raise ValueError("The FC extended flow changed the state shape.")
		physical = trajectory.split(value[:physical_size])
		momentum = value[physical_size:] if track_momentum else None
		return cls(physical=physical, momentum=momentum)

	def pack(self, trajectory: TrajectoryFC) -> np.ndarray:
		"""Restore the flat layout expected by the composition engine."""
		physical = trajectory.pack_components(*self.physical)
		return self.pack_array(physical)

	def pack_array(self, physical: np.ndarray) -> np.ndarray:
		"""Attach optional momentum to an already packed physical state."""
		return (
			physical
			if self.momentum is None
			else np.concatenate((physical, self.momentum))
		)


class _Progress:
	"""Small stderr progress indicator for long fixed-step integrations.

	``total`` and ``steps`` count complete BM4 steps, not the twelve internal
	composition stages. ``every`` controls how often the display is refreshed.
	"""

	def __init__(self, label: str, total: int) -> None:
		"""Prepare updates at approximately one-percent intervals."""
		self.label = label
		self.total = max(total, 1)
		self.every = max(self.total // 100, 1)
		self.steps = 0

	def update(self, t: float) -> None:
		"""Advance the counter and occasionally redraw the same terminal line."""
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
		"""Finish the in-place progress line."""
		print(file=sys.stderr, flush=True)


def _step_count(duration: float, maximum_step: float) -> int:
	"""Return the fewest steps that do not exceed ``maximum_step``.

	Both arguments are time increments in the simulation's time convention.
	``nextafter`` prevents round-off just above an exact integer ratio from
	introducing a redundant final step.
	"""
	ratio = duration / maximum_step
	return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


def _validate_inputs(
	t_span: tuple[float, float],
	state: np.ndarray,
	step: float,
	n_save_step: int,
) -> tuple[float, float, np.ndarray, float, np.ndarray]:
	"""Validate inputs and construct the one-dimensional saved-time grid.

	``step`` is an upper bound for internal BM4 steps, while ``n_save_step`` is
	the number of externally visible samples and includes both ends of
	``t_span``. ``state`` is a flat component-major initial condition.
	"""
	span = np.asarray(t_span, dtype=float)
	if span.shape != (2,) or not np.all(np.isfinite(span)) or span[0] >= span[1]:
		raise ValueError("`t_span` must contain two finite, increasing times.")
	maximum_step = float(step)
	# Booleans are numeric in Python, but accepting them here would silently turn
	# ``True`` into a physically meaningless unit step.
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
	# Every trajectory owns a flat physical layout. Higher dimensions are used
	# only internally after saved states have been assembled over time.
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
	"""Apply one flow while enforcing the integrator's shape invariant."""
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
	"""Apply one complete fourth-order BM4 composition.

	``step`` is the complete internal advance; ``stage`` below is its signed
	fraction for one direct or adjoint subflow. ``t`` follows every signed stage
	so that time-dependent potentials are evaluated at the correct endpoint.
	"""
	for coefficient, order in zip(_BM4_STAGES, _BM4_ORDERS, strict=True):
		stage = float(coefficient * step)
		# The adjoint is evaluated at the incoming time and the direct map at
		# the outgoing time so the non-autonomous composition remains symmetric.
		if order == 0:
			state = _checked_flow(flow, stage, t + stage, state)
		else:
			state = _checked_flow(adjoint_flow, stage, t, state)
		t += stage
	return t, state


def _planned_steps(times: np.ndarray, maximum_step: float) -> int:
	"""Count complete BM4 steps across the saved-time grid for progress output."""
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
	"""Integrate between requested output times with a composed fixed step.

	Each saved interval is subdivided independently. This makes every requested
	output time exact even when ``maximum_step`` does not divide the interval.
	The returned ``y`` has shape ``(packed_state_size, n_save_step)`` and
	``n_steps`` counts complete BM4 compositions rather than individual stages.
	"""
	t, _tf, value, maximum_step, times = _validate_inputs(
		t_span,
		state,
		step,
		n_save_step,
	)
	# Append the saved-time axis to the packed state; during stepping ``value``
	# itself remains the one-dimensional instantaneous state.
	result = np.empty(value.shape + (times.size,), dtype=value.dtype)
	result[:, 0] = value
	n_steps = 0
	progress_bar = _Progress(label, _planned_steps(times, maximum_step)) if progress else None

	try:
		for output_index, target in enumerate(times[1:], start=1):
			duration = float(target) - t
			count = _step_count(duration, maximum_step)
			# This interval-local step reaches ``target`` exactly and is guaranteed
			# not to exceed the public ``maximum_step`` bound.
			internal_step = duration / count
			segment_start = t
			for index in range(count):
				t, value = _advance(flow, adjoint_flow, t, value, internal_step)
				# Reconstructing time from the segment origin avoids accumulating
				# floating-point drift over many composition stages.
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
	"""Return the 4x4 exact harmonic mixing matrix for the two GC copies."""
	# This is an algorithmic binding frequency for the duplicated states, not
	# the physical Larmor frequency owned by an FC trajectory.
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
	"""Mix duplicated GC coordinates while leaving time momentum unchanged."""
	first = trajectory.split(state.first)
	second = trajectory.split(state.second)
	# ``blocks`` has leading order (x_first, y_first, x_second, y_second); all
	# particle and optional saved-time axes are vectorized by the ellipsis.
	blocks = np.stack((*first, *second), axis=0)
	coupled = np.asarray(np.einsum("ij,j...->i...", _gc_coupling(step), blocks))
	return _GCExtendedState(
		# Both slices already have explicit (component, particle, *sample) axes.
		# Flattening them is normally a view and postpones the sole required copy
		# until the extended state is packed for the composition engine.
		first=trajectory.from_blocks(coupled[:2]),
		second=trajectory.from_blocks(coupled[2:]),
		momentum=state.momentum,
	)


def _updated_momentum(
	system: _GCSystem | _FCSystem,
	momentum: np.ndarray | None,
	step: float,
	t: float,
	physical_state: np.ndarray,
) -> np.ndarray | None:
	"""Advance the per-particle momentum conjugate to time.

	``momentum`` and its derivative have shape ``(particle_count, ...)``; a
	``None`` value means the optional generalized-energy diagnostic is disabled.
	"""
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
	"""Integrate one GC physical state in doubled extended phase space.

	The two copies make each triangular subflow explicit. Their average is
	projected back to the physical state after the BM4 integration. The input is
	a flat ``[x, y]`` state; the returned physical solution has shape
	``(2 * particle_count, n_save_step)``.
	"""
	trajectory = system.trajectory
	physical_state = np.asarray(state, dtype=float)
	trajectory.split(physical_state)
	# ``physical_size`` counts both coordinate blocks, whereas
	# ``particle_count`` is the number of values in either x or y alone.
	physical_size = physical_state.size
	particle_count = trajectory.particle_count(physical_state)
	# Momentum exists solely to diagnose conservation in extended phase space.
	track_momentum = bool(check_energy)
	extended = _GCExtendedState(
		# Both copies start on the physical diagonal; coupling keeps them close
		# as the alternating explicit updates move them apart.
		first=physical_state,
		second=physical_state,
		momentum=np.zeros(particle_count) if track_momentum else None,
	)

	def unpack(value: np.ndarray) -> _GCExtendedState:
		"""Interpret a packed value produced by an internal GC flow."""
		return _GCExtendedState.unpack(
			value,
			physical_size=physical_size,
			particle_count=particle_count,
			track_momentum=track_momentum,
		)

	def vector_field(t: float, value: np.ndarray) -> np.ndarray:
		"""Evaluate and shape-check the physical GC vector field."""
		derivative = np.asarray(system.vector_field(t, value))
		if derivative.shape != value.shape:
			raise ValueError("The GC vector field changed the physical state shape.")
		return derivative

	def flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		"""Apply the explicit triangular GC map followed by copy coupling."""
		current = unpack(value)
		# Each copy supplies the frozen argument needed to update the other one
		# explicitly; the same states determine the time-momentum increments.
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
		"""Apply the reverse-ordered counterpart of the direct GC map."""
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
	# Here ``unpack`` preserves the saved-time axis, so each copy has shape
	# ``(physical_size, n_save_step)`` rather than being an instantaneous vector.
	final_state = unpack(solution.y)
	first = trajectory.split(final_state.first)
	second = trajectory.split(final_state.second)
	# Project the doubled coordinates back onto the physical diagonal.
	solution.y = trajectory.pack_components(
		(first.x + second.x) / 2,
		(first.y + second.y) / 2,
	)
	solution.trajectory = trajectory
	if final_state.momentum is not None:
		# The extended Hamiltonian contains one contribution from each physical
		# copy, so its conjugate momentum is twice the physical value.
		solution.k = final_state.momentum / 2
		solution.err = system._energy_error(solution)
	return solution


def _real_imaginary(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Expose complex planar calculations again as real coordinate arrays."""
	return np.asarray(value.real), np.asarray(value.imag)


def _cyclotron_step(
	trajectory: TrajectoryFC,
	state: FCState,
	step: float,
) -> FCState:
	"""Advance exactly under the field-free cyclotron sub-Hamiltonian.

	Complex notation applies the velocity rotation and its analytically
	integrated position displacement in one vectorized operation. Thus ``x + i*y``
	represents planar position and ``vx + i*vy`` planar velocity; ``rotation``
	is the unit complex phase accumulated during ``step``.
	"""
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
	"""Integrate one FC physical state with direct and adjoint split flows.

	The direct map applies cyclotron motion before the electric kick; its
	adjoint reverses that order. BM4 composes both maps symmetrically. The input
	is flat ``[x, y, vx, vy]`` and the returned solution has shape
	``(4 * particle_count, n_save_step)``.
	"""
	trajectory = system.trajectory
	physical_state = np.asarray(state, dtype=float)
	physical = trajectory.split(physical_state)
	# ``physical_size`` spans all four state blocks; ``particle_count`` is the
	# common leading size of each individual position or velocity block.
	physical_size = physical_state.size
	particle_count = trajectory.particle_count(physical_state)
	# The optional momentum extends each particle by one diagnostic variable.
	track_momentum = bool(check_energy)
	extended = _FCExtendedState(
		physical=physical,
		momentum=np.zeros(particle_count) if track_momentum else None,
	)

	def unpack(value: np.ndarray) -> _FCExtendedState:
		"""Interpret a packed value produced by an internal FC flow."""
		return _FCExtendedState.unpack(
			value,
			trajectory=trajectory,
			physical_size=physical_size,
			particle_count=particle_count,
			track_momentum=track_momentum,
		)

	def flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		"""Apply exact cyclotron motion followed by an electric kick."""
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
		physical_array = trajectory.pack_components(*physical)
		momentum = _updated_momentum(
			system,
			current.momentum,
			h,
			t,
			physical_array,
		)
		return _FCExtendedState(physical, momentum).pack_array(physical_array)

	def adjoint_flow(h: float, t: float, value: np.ndarray) -> np.ndarray:
		"""Apply the electric kick before exact cyclotron motion."""
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
		momentum = current.momentum
		if momentum is not None:
			# This pre-cyclotron representation is needed only for the optional
			# generalized-energy diagnostic. Avoid allocating it in the default path.
			physical_array = trajectory.pack_components(*physical)
			momentum = _updated_momentum(
				system,
				momentum,
				h,
				t,
				physical_array,
			)
		physical = _cyclotron_step(trajectory, physical, h)
		# The cyclotron map changes the physical arrays after the momentum update,
		# so pack its result once; unlike the direct flow, the pre-map packed state
		# cannot also be returned to the composition engine.
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
	# Unpacking the saved result retains its final time axis in every FC block.
	final_state = unpack(solution.y)
	solution.y = trajectory.pack_components(*final_state.physical)
	solution.trajectory = trajectory
	if final_state.momentum is not None:
		solution.k = final_state.momentum
		solution.err = system._energy_error(solution)
	return solution


__all__: list[str] = []
