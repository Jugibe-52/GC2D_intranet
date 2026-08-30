"""Contracts for the two-stage fourth-order Gauss--Legendre method."""

from __future__ import annotations

import unittest

import numpy as np

from diagnostics import (
	calculate_step_jacobian,
	central_difference_jacobian,
)
from diagnostics.symplecticity import gc_physical_symplectic_form
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	GAUSS_JACOBIAN_METHODS,
	GaussLegendre4,
	GaussLegendre4IntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)


class _RotationHamiltonian:
	"""Independent unit-frequency planar Hamiltonian oscillators."""

	state_dimension = 2

	def vector_field(self, _time: float, state: np.ndarray) -> np.ndarray:
		particle_count = state.shape[0] // 2
		x = state[:particle_count]
		y = state[particle_count:]
		return np.concatenate((-y, x))

	def particle_vector_field_jacobians(
		self,
		_time: float,
		state: np.ndarray,
	) -> np.ndarray:
		particle_count = state.shape[0] // 2
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
	"""Generic non-autonomous field integrated exactly by two-point quadrature."""

	state_dimension = 2

	def vector_field(self, time: float, _state: np.ndarray) -> np.ndarray:
		return np.asarray((time**3, 0.0))


class _ExponentialRotationHamiltonian:
	"""Nonautonomous oscillator with a closed physical and momentum solution."""

	state_dimension = 2

	def vector_field(self, time: float, state: np.ndarray) -> np.ndarray:
		frequency = float(np.exp(time))
		particle_count = state.shape[0] // 2
		x = state[:particle_count]
		y = state[particle_count:]
		return frequency * np.concatenate((-y, x))

	def particle_vector_field_jacobians(
		self,
		time: float,
		state: np.ndarray,
	) -> np.ndarray:
		particle_count = state.shape[0] // 2
		return np.broadcast_to(
			np.exp(time) * np.asarray(((0.0, -1.0), (1.0, 0.0))),
			(particle_count, 2, 2),
		).copy()

	def hamiltonian(
		self,
		time: float | np.ndarray,
		state: np.ndarray,
	) -> np.ndarray:
		particle_count = state.shape[0] // 2
		x = state[:particle_count]
		y = state[particle_count:]
		return 0.5 * np.exp(time) * (x**2 + y**2)

	def extended_momentum_derivative(
		self,
		time: float,
		state: np.ndarray,
	) -> np.ndarray:
		return -np.asarray(self.hamiltonian(time, state), dtype=float)


def _rotation_problem(*, particles: int = 1) -> InitialValueProblem:
	"""Return packed independent oscillators with distinct initial phases."""
	angles = np.linspace(0.0, 0.4, particles)
	state = np.concatenate((np.cos(angles), np.sin(angles)))
	return InitialValueProblem(
		_RotationHamiltonian(),
		GCInitialConfiguration(state),
	)


class GaussLegendre4Tests(unittest.TestCase):
	"""Verify accuracy, geometry, diagnostics, and general ODE fallback."""

	def test_public_configuration_and_validation(self) -> None:
		self.assertEqual(
			GAUSS_JACOBIAN_METHODS,
			("auto", "analytic", "finite_difference"),
		)
		with self.assertRaises(ValueError):
			GaussLegendre4(newton_absolute_tolerance=0.0)
		with self.assertRaises(ValueError):
			GaussLegendre4(newton_max_iterations=0)
		with self.assertRaises(ValueError):
			GaussLegendre4(newton_jacobian_method="complex_step")  # type: ignore[arg-type]

	def test_fourth_order_rotation_convergence(self) -> None:
		problem = _rotation_problem()
		errors: list[float] = []
		for step in (0.2, 0.1, 0.05):
			solution = simulate(
				problem,
				GaussLegendre4(newton_jacobian_method="analytic"),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=step,
					sample_count=2,
				),
			)
			expected = np.asarray((np.cos(1.0), np.sin(1.0)))
			errors.append(float(np.linalg.norm(solution.states[:, -1] - expected)))
		for coarse, fine in zip(errors, errors[1:]):
			self.assertGreater(coarse / fine, 14.0)
			self.assertLess(coarse / fine, 18.0)

	def test_generic_time_dependent_field_uses_gauss_stage_times(self) -> None:
		problem = InitialValueProblem(
			_CubicTimeField(),
			GCInitialConfiguration(np.zeros(2)),
		)
		solution = simulate(
			problem,
			GaussLegendre4(newton_jacobian_method="auto"),
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
				GaussLegendre4(newton_jacobian_method="analytic"),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=1.0,
					sample_count=2,
				),
			)

	def test_auto_falls_back_when_gc_hessians_are_unavailable(self) -> None:
		potential = Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=2,
		)
		problem = InitialValueProblem(
			GuidingCenterDynamics(potential, rho=0.05),
			GCInitialConfiguration(np.asarray((1.0, 1.2))),
		)
		request = SimulationRequest.uniform(
			t_span=(0.0, 0.02),
			max_step=0.01,
			sample_count=3,
		)
		solution = simulate(problem, GaussLegendre4(), request)
		self.assertTrue(np.all(np.isfinite(solution.states)))
		self.assertEqual(
			solution.diagnostics["newton_jacobian_method"],
			"finite_difference",
		)
		with self.assertRaises(ValueError):
			simulate(
				problem,
				GaussLegendre4(newton_jacobian_method="analytic"),
				request,
			)

	def test_exact_step_jacobian_is_symplectic_and_matches_map_audit(self) -> None:
		events: list[GaussLegendre4IntegrationStep] = []
		simulate(
			_rotation_problem(particles=2),
			GaussLegendre4(
				newton_jacobian_method="analytic",
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.1),
				max_step=0.1,
				sample_count=2,
			),
		)
		self.assertEqual(len(events), 1)
		step = events[0]
		analytic = calculate_step_jacobian(step, method="implicit_function")
		numerical = central_difference_jacobian(step.map_state, step.state_before)
		form = gc_physical_symplectic_form(2)
		defect = analytic.T @ form @ analytic - form
		self.assertLess(float(np.linalg.norm(defect, ord="fro")), 2e-14)
		self.assertLess(abs(float(np.linalg.det(analytic)) - 1.0), 2e-14)
		np.testing.assert_allclose(analytic, numerical, rtol=2e-8, atol=2e-9)

	def test_nonlinear_gc_tangent_matches_the_implemented_map(self) -> None:
		potential = Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		)
		problem = InitialValueProblem(
			GuidingCenterDynamics(potential, rho=0.05),
			GCInitialConfiguration(np.asarray((1.0, 2.0, 1.2, 2.1))),
		)
		events: list[GaussLegendre4IntegrationStep] = []
		simulate(
			problem,
			GaussLegendre4(
				newton_absolute_tolerance=1e-15,
				newton_relative_tolerance=1e-14,
				newton_jacobian_method="analytic",
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.03),
				max_step=0.03,
				sample_count=2,
			),
		)
		analytic = calculate_step_jacobian(events[0], method="implicit_function")
		numerical = central_difference_jacobian(
			events[0].map_state,
			events[0].state_before,
		)
		relative_difference = (
			np.linalg.norm(analytic - numerical, ord="fro")
			/ np.linalg.norm(analytic, ord="fro")
		)
		self.assertLess(float(relative_difference), 2e-8)

	def test_energy_tracking_is_triangular_and_preserves_quadratic_energy(self) -> None:
		problem = _rotation_problem(particles=2)
		request = SimulationRequest.uniform(
			t_span=(0.0, 1.0),
			max_step=0.1,
			sample_count=11,
		)
		plain = simulate(problem, GaussLegendre4(), request)
		tracked = simulate(problem, GaussLegendre4(track_energy=True), request)
		np.testing.assert_array_equal(plain.states, tracked.states)
		assert tracked.k is not None
		self.assertEqual(tracked.k.shape, (2, 11))
		np.testing.assert_array_equal(tracked.k, 0.0)
		assert tracked.err is not None
		self.assertLess(tracked.err, 5e-15)

	def test_nonautonomous_generalized_energy_converges_at_order_four(self) -> None:
		initial_state = np.asarray((1.0, 0.0))
		problem = InitialValueProblem(
			_ExponentialRotationHamiltonian(),
			GCInitialConfiguration(initial_state),
		)
		errors: list[float] = []
		finest = None
		for step in (0.2, 0.1, 0.05):
			finest = simulate(
				problem,
				GaussLegendre4(
					track_energy=True,
					newton_jacobian_method="analytic",
				),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=step,
					sample_count=int(round(1.0 / step)) + 1,
				),
			)
			assert finest.err is not None
			errors.append(float(finest.err))
		for coarse, fine in zip(errors, errors[1:]):
			self.assertGreater(coarse / fine, 14.0)
			self.assertLess(coarse / fine, 18.0)
		assert finest is not None and finest.k is not None
		angle = float(np.e - 1.0)
		np.testing.assert_allclose(
			finest.states[:, -1],
			(np.cos(angle), np.sin(angle)),
			atol=1e-6,
		)
		np.testing.assert_allclose(
			finest.k[:, -1],
			(-0.5 * (np.e - 1.0),),
			atol=1e-6,
		)

	def test_observations_and_diagnostics_exclude_shadow_steps(self) -> None:
		potential = Potential.random(
			A=0.08,
			M=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=3,
		)
		problem = InitialValueProblem(
			GuidingCenterDynamics(potential, rho=0.05),
			GCInitialConfiguration(np.asarray((1.0, 1.2))),
		)
		events: list[GaussLegendre4IntegrationStep] = []
		solution = simulate(
			problem,
			GaussLegendre4(step_observer=events.append),
			SimulationRequest.uniform(
				t_span=(0.0, 0.02),
				max_step=0.01,
				sample_count=5,
			),
		)
		self.assertEqual(solution.n_steps, 2)
		self.assertEqual(len(events), 2)
		for name in (
			"nonlinear_iterations",
			"residual_evaluations",
			"nonlinear_residual_norms",
			"nonlinear_tolerances",
		):
			self.assertEqual(np.asarray(solution.diagnostics[name]).shape, (2,))
		for event in events:
			np.testing.assert_allclose(
				event.map_state(event.state_before),
				event.state_after,
			)

	def test_output_density_does_not_change_main_grid_path(self) -> None:
		problem = _rotation_problem()
		method = GaussLegendre4()
		sparse = simulate(
			problem,
			method,
			SimulationRequest.uniform(
				t_span=(0.0, 0.2), max_step=0.1, sample_count=3
			),
		)
		dense = simulate(
			problem,
			method,
			SimulationRequest.uniform(
				t_span=(0.0, 0.2), max_step=0.1, sample_count=5
			),
		)
		np.testing.assert_array_equal(sparse.states, dense.states[:, ::2])


if __name__ == "__main__":
	unittest.main()
