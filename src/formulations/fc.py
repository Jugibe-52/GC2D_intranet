# Copyright (c) 2023, Cristel Chandre
# SPDX-License-Identifier: BSD-2-Clause

"""Full-cyclotron split formulation for direct/adjoint compositions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contracts.dynamics.protocols import CyclotronSplitSystem, HamiltonianSystem
from contracts.formulations.layout import StateLayout
from formulations.layout import FCState, FCStateLayout
from formulations.state import SingleFormulation

from typing import TYPE_CHECKING


if TYPE_CHECKING:
	from problem import InitialValueProblem
	from typing import Any


from formulations.base import _updated_momentum
from contracts.formulations.base import PreparedDirectAdjointFormulation, Projection


@dataclass(frozen=True, slots=True)
class _FCExtendedState:
	"""One physical FC state and optional time-conjugate momentum."""

	physical: FCState
	momentum: np.ndarray | None = None

	def pack(self, representation: SingleFormulation) -> np.ndarray:
		physical: np.ndarray = FCStateLayout.pack_components(*self.physical)
		return self.pack_array(physical, representation)

	def pack_array(self, physical: np.ndarray, representation: SingleFormulation) -> np.ndarray:
		"""Use the common physical representation before adding map momentum."""
		spatial: np.ndarray = representation.lift(physical)
		return (
			spatial
			if self.momentum is None
			else np.concatenate((spatial, self.momentum))
		)


def _real_imaginary(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Expose complex planar calculations as real coordinate arrays."""
	return np.asarray(value.real), np.asarray(value.imag)


@dataclass(frozen=True, slots=True)
class _PreparedFC:
	"""Immutable FC split maps bound to one problem."""

	dynamics: CyclotronSplitSystem
	configuration: np.ndarray
	layout: StateLayout
	state_formulation: SingleFormulation
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
		spatial: np.ndarray = self.state_formulation.split_copies(value[:self.physical_size])[0]
		physical: FCState = FCState(*self.layout.split(spatial))
		momentum = value[self.physical_size :] if self.track_energy else None
		return _FCExtendedState(physical=physical, momentum=momentum)

	def _cyclotron_step(self, state: FCState, duration: float) -> FCState:
		"""Advance the exactly solvable field-free cyclotron subflow."""
		# Multiplication by the precomputed frequency preserves the established
		# floating-point path as well as the analytical rotation.
		rotation: Any = np.exp(-1j * duration * self.dynamics.larmor_frequency)
		x: np.ndarray
		y: np.ndarray
		x, y = _real_imaginary(
			state.x
			+ 1j * state.y
			+ 1j
			* self.dynamics.rho
			* np.sign(self.dynamics.eta)
			* (rotation - 1)
			* (state.vx + 1j * state.vy)
		)
		vx: np.ndarray
		vy: np.ndarray
		vx, vy = _real_imaginary(rotation * (state.vx + 1j * state.vy))
		return FCState(x, y, vx, vy)

	def direct_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply exact cyclotron motion followed by an electric kick."""
		current: _FCExtendedState = self._unpack(state)
		physical: FCState = self._cyclotron_step(current.physical, duration)
		acceleration_x: Any
		acceleration_y: Any
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
		physical_array: np.ndarray = FCStateLayout.pack_components(*physical)
		momentum: np.ndarray | None = _updated_momentum(
			self.dynamics,
			current.momentum,
			duration,
			t,
			physical_array,
		)
		return _FCExtendedState(physical, momentum).pack_array(physical_array, self.state_formulation)

	def adjoint_map(
		self,
		duration: float,
		t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Apply an electric kick followed by exact cyclotron motion."""
		current: _FCExtendedState = self._unpack(state)
		acceleration_x: Any
		acceleration_y: Any
		acceleration_x, acceleration_y = self.dynamics.electric_acceleration(
			t,
			current.physical.x,
			current.physical.y,
		)
		physical: FCState = FCState(
			current.physical.x,
			current.physical.y,
			current.physical.vx + duration * acceleration_x,
			current.physical.vy + duration * acceleration_y,
		)
		momentum: np.ndarray | None = current.momentum
		if momentum is not None:
			pre_cyclotron: np.ndarray = FCStateLayout.pack_components(*physical)
			momentum = _updated_momentum(
				self.dynamics,
				momentum,
				duration,
				t,
				pre_cyclotron,
			)
		physical = self._cyclotron_step(physical, duration)
		return _FCExtendedState(physical, momentum).pack(self.state_formulation)

	def project(self, internal_history: np.ndarray) -> Projection:
		"""Strip optional extended momentum from the physical FC history."""
		final_state: _FCExtendedState = self._unpack(internal_history)
		states: np.ndarray = FCStateLayout.pack_components(*final_state.physical)
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
		layout = problem.layout
		if layout.state_dimension != 4:
			raise TypeError("FCSplitFormulation requires a four-component FC layout.")
		if not isinstance(problem.dynamics, CyclotronSplitSystem):
			raise TypeError(
				"FCSplitFormulation requires CyclotronSplitSystem dynamics."
			)
		if track_energy and not isinstance(
			problem.dynamics,
			HamiltonianSystem,
		):
			raise TypeError("Energy tracking requires HamiltonianSystem.")
		physical = problem.initial_state
		particle_count = problem.particle_count
		representation: SingleFormulation = SingleFormulation(layout, particle_count)
		initial: _FCExtendedState = _FCExtendedState(
			physical=FCState(*layout.split(physical)),
			momentum=np.zeros(particle_count) if track_energy else None,
		)
		initial_internal_state: np.ndarray = initial.pack(representation)
		initial_internal_state.setflags(write=False)
		return _PreparedFC(
			dynamics=problem.dynamics,
			configuration=problem.initial_state,
			layout=layout,
			state_formulation=representation,
			physical_size=physical.size,
			particle_count=particle_count,
			track_energy=bool(track_energy),
			dynamics_name=type(problem.dynamics).__name__,
			initial_internal_state=initial_internal_state,
		)


__all__ = ["FCSplitFormulation"]
