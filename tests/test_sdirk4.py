"""Contracts for the five-stage, fourth-order S54b SDIRK method."""

from __future__ import annotations

import unittest

import numpy as np

from initial_conditions import GCInitialConfiguration
from simulation import (
	SDIRK4_TABLEAU_A,
	SDIRK4_TABLEAU_B,
	SDIRK4_TABLEAU_C,
	SDIRK_JACOBIAN_METHODS,
	InitialValueProblem,
	SDIRK4,
	SimulationRequest,
	simulate,
)


class _RotationHamiltonian:
	"""Independent unit-frequency planar Hamiltonian oscillators."""

	state_dimension = 2

	def vector_field(self, _time: float, state: np.ndarray) -> np.ndarray:
		particle_count = state.size // 2
		x = state[:particle_count]
		y = state[particle_count:]
		return np.concatenate((-y, x))

	def particle_vector_field_jacobians(
		self,
		_time: float,
		state: np.ndarray,
	) -> np.ndarray:
		particle_count = state.size // 2
		return np.broadcast_to(
			np.asarray(((0.0, -1.0), (1.0, 0.0))),
			(particle_count, 2, 2),
		).copy()

	def hamiltonian(
		self,
		_time: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		particle_count = state.shape[0] // 2
		x = state[:particle_count]
		y = state[particle_count:]
		return 0.5 * (x**2 + y**2)

	def extended_momentum_derivative(
		self,
		_time: float,
		state: np.ndarray,
	) -> np.ndarray:
		return np.zeros(state.shape[0] // 2)


class _CubicTimeField:
	"""Non-autonomous field whose integral is a degree-four polynomial."""

	state_dimension = 2

	def vector_field(self, time: float, _state: np.ndarray) -> np.ndarray:
		return np.asarray((time**3, 0.0))


def _rotation_problem(*, particles: int = 1) -> InitialValueProblem:
	"""Return packed oscillators with distinct initial phases."""
	angles = np.linspace(0.0, 0.4, particles)
	state = np.concatenate((np.cos(angles), np.sin(angles)))
	return InitialValueProblem(
		_RotationHamiltonian(),
		GCInitialConfiguration(state),
	)


class SDIRK4Tests(unittest.TestCase):
	"""Verify order, tableau geometry, diagnostics, and ODE fallback."""

	def test_s54b_tableau_has_order_four_but_is_not_geometric(self) -> None:
		A = SDIRK4_TABLEAU_A
		b = SDIRK4_TABLEAU_B
		c = SDIRK4_TABLEAU_C
		self.assertEqual(A.shape, (5, 5))
		self.assertEqual(b.shape, c.shape, (5,))
		np.testing.assert_allclose(np.diag(A), 0.25, rtol=0.0, atol=0.0)
		np.testing.assert_allclose(np.sum(A, axis=1), c, rtol=0.0, atol=0.0)
		np.testing.assert_allclose(A[-1], b, rtol=0.0, atol=0.0)

		order_values = np.asarray(
			(
				b.sum(),
				b @ c,
				b @ (c**2),
				b @ A @ c,
				b @ (c**3),
				b @ (c * (A @ c)),
				b @ A @ (c**2),
				b @ A @ A @ c,
			)
		)
		np.testing.assert_allclose(
			order_values,
			(1.0, 0.5, 1.0 / 3.0, 1.0 / 6.0, 0.25, 0.125, 1.0 / 12.0, 1.0 / 24.0),
			rtol=0.0,
			atol=3e-16,
		)

		symplecticity_conditions = (
			b[:, None] * A
			+ b[None, :] * A.T
			- b[:, None] * b[None, :]
		)
		symmetry_conditions = A + A[::-1, ::-1] - b[::-1][None, :]
		self.assertGreater(float(np.max(np.abs(symplecticity_conditions))), 0.1)
		self.assertGreater(float(np.max(np.abs(symmetry_conditions))), 0.1)

	def test_public_configuration_and_validation(self) -> None:
		self.assertEqual(
			SDIRK_JACOBIAN_METHODS,
			("auto", "analytic", "finite_difference"),
		)
		with self.assertRaises(ValueError):
			SDIRK4(newton_absolute_tolerance=0.0)
		with self.assertRaises(ValueError):
			SDIRK4(newton_max_iterations=0)
		with self.assertRaises(ValueError):
			SDIRK4(newton_jacobian_method="complex_step")  # type: ignore[arg-type]

	def test_fourth_order_rotation_convergence_and_diagnostics(self) -> None:
		problem = _rotation_problem(particles=2)
		errors: list[float] = []
		finest = None
		for step in (0.2, 0.1, 0.05):
			finest = simulate(
				problem,
				SDIRK4(
					newton_absolute_tolerance=1e-15,
					newton_relative_tolerance=1e-14,
					newton_jacobian_method="analytic",
				),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=step,
					sample_count=int(round(1.0 / step)) + 1,
				),
			)
			angles = np.linspace(0.0, 0.4, 2) + 1.0
			expected = np.concatenate((np.cos(angles), np.sin(angles)))
			errors.append(float(np.linalg.norm(finest.states[:, -1] - expected)))
		for coarse, fine in zip(errors, errors[1:]):
			self.assertGreater(coarse / fine, 13.0)
			self.assertLess(coarse / fine, 19.0)
		assert finest is not None
		diagnostics = finest.diagnostics
		self.assertEqual(diagnostics["stage_count"], 5)
		self.assertEqual(diagnostics["designed_order"], 4)
		self.assertEqual(diagnostics["tableau_name"], "Skvortsov S54b")
		self.assertTrue(diagnostics["stiffly_accurate"])
		self.assertFalse(diagnostics["symmetric"])
		self.assertFalse(diagnostics["symplectic"])
		self.assertEqual(
			np.asarray(diagnostics["stage_nonlinear_iterations"]).shape,
			(20, 5),
		)
		self.assertEqual(
			np.asarray(diagnostics["nonlinear_iterations"]).shape,
			(20,),
		)

	def test_generic_time_dependent_field_uses_sdirk_stage_times(self) -> None:
		problem = InitialValueProblem(
			_CubicTimeField(),
			GCInitialConfiguration(np.zeros(2)),
		)
		solution = simulate(
			problem,
			SDIRK4(newton_jacobian_method="auto"),
			SimulationRequest.uniform(
				t_span=(0.0, 1.0),
				max_step=1.0,
				sample_count=2,
			),
		)
		np.testing.assert_allclose(solution.states[:, -1], (0.25, 0.0), atol=2e-14)
		self.assertEqual(
			solution.diagnostics["newton_jacobian_method"],
			"finite_difference",
		)
		with self.assertRaises(TypeError):
			simulate(
				problem,
				SDIRK4(newton_jacobian_method="analytic"),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=1.0,
					sample_count=2,
				),
			)

	def test_energy_tracking_preserves_physical_states(self) -> None:
		problem = _rotation_problem(particles=2)
		request = SimulationRequest.uniform(
			t_span=(0.0, 1.0),
			max_step=0.1,
			sample_count=11,
		)
		plain = simulate(problem, SDIRK4(), request)
		tracked = simulate(problem, SDIRK4(track_energy=True), request)
		np.testing.assert_array_equal(plain.states, tracked.states)
		assert tracked.k is not None
		np.testing.assert_array_equal(tracked.k, 0.0)


if __name__ == "__main__":
	unittest.main()
