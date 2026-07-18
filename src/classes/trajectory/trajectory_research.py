"""Stability diagnostics for trajectory dynamics."""

from __future__ import annotations

import numpy as np

from ..potential import Array
from .trajectory import Trajectory
from .trajectory_fc import TrajectoryFC
from .trajectory_gc import TrajectoryGC


class TrajectoryResearch:
	"""Analyse the stability of an existing guiding-centre or full-cyclotron trajectory."""

	def __init__(self, trajectory: Trajectory) -> None:
		if not isinstance(trajectory, Trajectory):
			raise TypeError("`trajectory` must be a Trajectory instance.")
		self.trajectory = trajectory

	@staticmethod
	def _split_augmented_state(state: Array, state_dimension: int) -> tuple[tuple[Array, ...], Array]:
		"""Separate a phase-space state from its tangent-map matrix."""
		state_array = np.asarray(state)
		component_count = state_dimension + state_dimension**2
		if state_array.ndim == 0 or state_array.shape[0] % component_count != 0:
			raise ValueError(
				f"The first state dimension must be divisible by {component_count} "
				f"for a {state_dimension}-dimensional tangent map."
			)
		components = tuple(np.split(state_array, component_count, axis=0))
		phase_state = components[:state_dimension]
		jacobian = np.asarray(components[state_dimension:]).reshape((state_dimension, state_dimension, -1))
		return phase_state, jacobian

	def y_dot_lyap(self, t: float, state: Array) -> Array:
		r"""Return the trajectory and tangent-map derivatives.

		The augmented state stores the phase-space coordinates followed by the
		flattened tangent matrix :math:`J`.  This method returns
		:math:`(\dot z, \dot J)` with :math:`\dot J = (\partial f/\partial z)J`.
		"""
		if isinstance(self.trajectory, TrajectoryGC):
			return self._guiding_center_y_dot_lyap(t, state)
		if isinstance(self.trajectory, TrajectoryFC):
			return self._full_cyclotron_y_dot_lyap(t, state)
		raise TypeError(f"Unsupported trajectory class: {type(self.trajectory).__name__}.")

	def _guiding_center_y_dot_lyap(self, t: float, state: Array) -> Array:
		(x, y), jacobian = self._split_augmented_state(state, state_dimension=2)
		trajectory = self.trajectory
		phase_state = np.concatenate((x, y))

		d2psi_dx2 = trajectory.psi(t, x, y, dx=2)
		d2psi_dxdy = trajectory.psi(t, x, y, dx=1, dy=1)
		d2psi_dy2 = trajectory.psi(t, x, y, dy=2)
		linearization = np.zeros_like(jacobian)
		linearization[0, 0], linearization[0, 1] = -d2psi_dxdy, -d2psi_dy2
		linearization[1, 0], linearization[1, 1] = d2psi_dx2, d2psi_dxdy
		jacobian_dot = np.einsum("ijm,jkm->ikm", linearization, jacobian)
		return np.concatenate((trajectory.y_dot(t, phase_state), jacobian_dot.reshape(-1)))

	def _full_cyclotron_y_dot_lyap(self, t: float, state: Array) -> Array:
		(x, y, vx, vy), jacobian = self._split_augmented_state(state, state_dimension=4)
		trajectory = self.trajectory
		phase_state = np.concatenate((x, y, vx, vy))

		d2phi_dx2 = -trajectory.electric_scale * trajectory.phi(t, x, y, dx=2)
		d2phi_dxdy = -trajectory.electric_scale * trajectory.phi(t, x, y, dx=1, dy=1)
		d2phi_dy2 = -trajectory.electric_scale * trajectory.phi(t, x, y, dy=2)
		linearization = np.zeros_like(jacobian)
		linearization[0, 2] = trajectory.velocity_scale
		linearization[1, 3] = trajectory.velocity_scale
		linearization[2, 3] = trajectory.larmor_frequency
		linearization[3, 2] = -trajectory.larmor_frequency
		linearization[2, 0], linearization[2, 1] = d2phi_dx2, d2phi_dxdy
		linearization[3, 0], linearization[3, 1] = d2phi_dxdy, d2phi_dy2
		jacobian_dot = np.einsum("ijm,jkm->ikm", linearization, jacobian)
		return np.concatenate((trajectory.y_dot(t, phase_state), jacobian_dot.reshape(-1)))


# Keep the spelling used by existing research helpers.
trajectory_researche = TrajectoryResearch

__all__ = ["TrajectoryResearch", "trajectory_researche"]
