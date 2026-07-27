# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Full-cyclotron split formulation for direct/adjoint compositions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from classes.dynamics import CyclotronSplitSystem, ExtendedHamiltonianSystem
from classes.trajectory import FCState, TrajectoryFC

from ..problem import InitialValueProblem
from .base import PreparedDirectAdjointFormulation, Projection


@dataclass(frozen=True, slots=True)
class _FCExtendedState:
	"""One physical FC state and optional time-conjugate momentum."""

	physical: FCState
	momentum: np.ndarray | None = None

	def pack(self, configuration: TrajectoryFC) -> np.ndarray:
		physical = configuration.pack_components(*self.physical)
		return self.pack_array(physical)

	def pack_array(self, physical: np.ndarray) -> np.ndarray:
		return (
			physical
			if self.momentum is None
			else np.concatenate((physical, self.momentum))
		)


def _real_imaginary(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Expose complex planar calculations as real coordinate arrays."""
	return np.asarray(value.real), np.asarray(value.imag)


@dataclass(frozen=True, slots=True)
class _PreparedFC:
	"""Immutable FC split maps bound to one problem."""

	dynamics: CyclotronSplitSystem
	configuration: TrajectoryFC
	physical_size: int
	particle_count: int
	track_energy: bool
	dynamics_name: str
	initial_internal_state: np.ndarray

	def _unpack(self, value: np.ndarray) -> _FCExtendedState:
		expected = self.physical_size + (
			self.particle_count if self.track_energy else 0
		)
		if value.ndim == 0 or value.shape[0] != expected:
			raise ValueError("The FC split map changed the internal state shape.")
		physical = self.configuration.split(value[: self.physical_size])
		momentum = value[self.physical_size :] if self.track_energy else None
		return _FCExtendedState(physical=physical, momentum=momentum)

	def _cyclotron_step(self, state: FCState, duration: float) -> FCState:
		"""Advance the exactly solvable field-free cyclotron subflow."""
		# Multiplication by the precomputed frequency preserves the established
		# floating-point path as well as the analytical rotation.
		rotation = np.exp(-1j * duration * self.dynamics.larmor_frequency)
		x, y = _real_imaginary(
			state.x
			+ 1j * state.y
			+ 1j
			* self.dynamics.rho
			* np.sign(self.dynamics.eta)
			* (rotation - 1)
			* (state.vx + 1j * state.vy)
		)
		vx, vy = _real_imaginary(rotation * (state.vx + 1j * state.vy))
		return FCState(x, y, vx, vy)

	def _updated_momentum(
		self,
		momentum: np.ndarray | None,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray | None:
		if momentum is None:
			return None
		assert isinstance(self.dynamics, ExtendedHamiltonianSystem)
		derivative = np.asarray(
			self.dynamics.extended_momentum_derivative(t, state)
		)
		if derivative.shape != momentum.shape:
			raise ValueError("The extended-momentum derivative changed its shape.")
		return np.asarray(momentum + duration * derivative)

	def direct_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply exact cyclotron motion followed by an electric kick."""
		current = self._unpack(state)
		physical = self._cyclotron_step(current.physical, duration)
		acceleration_x, acceleration_y = self.dynamics.electric_acceleration(
			t,
			physical.x,
			physical.y,
		)
		physical = FCState(
			physical.x,
			physical.y,
			physical.vx + duration * acceleration_x,
			physical.vy + duration * acceleration_y,
		)
		physical_array = self.configuration.pack_components(*physical)
		momentum = self._updated_momentum(
			current.momentum,
			duration,
			t,
			physical_array,
		)
		return _FCExtendedState(physical, momentum).pack_array(physical_array)

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply an electric kick followed by exact cyclotron motion."""
		current = self._unpack(state)
		acceleration_x, acceleration_y = self.dynamics.electric_acceleration(
			t,
			current.physical.x,
			current.physical.y,
		)
		physical = FCState(
			current.physical.x,
			current.physical.y,
			current.physical.vx + duration * acceleration_x,
			current.physical.vy + duration * acceleration_y,
		)
		momentum = current.momentum
		if momentum is not None:
			pre_cyclotron = self.configuration.pack_components(*physical)
			momentum = self._updated_momentum(
				momentum,
				duration,
				t,
				pre_cyclotron,
			)
		physical = self._cyclotron_step(physical, duration)
		return _FCExtendedState(physical, momentum).pack(self.configuration)

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Strip optional extended momentum from the physical FC history."""
		final_state = self._unpack(internal_history)
		states = self.configuration.pack_components(*final_state.physical)
		diagnostics: dict[str, np.ndarray | float | int | str | bool] = {}
		if final_state.momentum is not None:
			diagnostics["extended_momentum"] = final_state.momentum
		return states, diagnostics


@dataclass(frozen=True, slots=True)
class FCSplitFormulation:
	"""Reusable exact-cyclotron/electric-kick formulation."""

	def prepare(
		self,
		problem: InitialValueProblem,
		*,
		track_energy: bool,
	) -> PreparedDirectAdjointFormulation:
		"""Bind immutable FC maps to one compatible problem."""
		configuration = problem.initial_configuration
		if not isinstance(configuration, TrajectoryFC):
			raise TypeError("FCSplitFormulation requires an FC configuration.")
		if not isinstance(problem.dynamics, CyclotronSplitSystem):
			raise TypeError(
				"FCSplitFormulation requires CyclotronSplitSystem dynamics."
			)
		if track_energy and not isinstance(
			problem.dynamics,
			ExtendedHamiltonianSystem,
		):
			raise TypeError("Energy tracking requires ExtendedHamiltonianSystem.")
		physical = problem.initial_state
		particle_count = configuration.particle_count(physical)
		initial = _FCExtendedState(
			physical=configuration.split(physical),
			momentum=np.zeros(particle_count) if track_energy else None,
		)
		initial_internal_state = initial.pack(configuration)
		initial_internal_state.setflags(write=False)
		return _PreparedFC(
			dynamics=problem.dynamics,
			configuration=configuration,
			physical_size=physical.size,
			particle_count=particle_count,
			track_energy=bool(track_energy),
			dynamics_name=type(problem.dynamics).__name__,
			initial_internal_state=initial_internal_state,
		)


__all__ = ["FCSplitFormulation"]
