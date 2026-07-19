from __future__ import annotations

import unittest

import numpy as np

from classes import (
	FourierPotential,
	System,
	SystemFC,
	SystemGC,
	TrajectoryFC,
	TrajectoryGC,
	create_system,
)
from workflows import initialize_trajectory


def single_mode_potential(*, coefficient: complex = 1.0) -> FourierPotential:
	coefficients = np.zeros((2, 2), dtype=np.complex128)
	coefficients[1, 1] = coefficient
	return FourierPotential(1.0, 1, coefficients=coefficients)


class SystemCompositionTests(unittest.TestCase):
	def test_system_is_an_abstract_composition_contract(self) -> None:
		with self.assertRaises(TypeError):
			System(single_mode_potential(), TrajectoryGC())  # type: ignore[abstract]

	def test_gc_composition_preserves_entity_identity_and_separation(self) -> None:
		potential = single_mode_potential()
		trajectory = TrajectoryGC(rho=0.1, eta=0.2, n_trajectories=4)

		system = create_system(potential, trajectory)

		self.assertIsInstance(system, SystemGC)
		self.assertIsInstance(system, System)
		self.assertIs(system.potential, potential)
		self.assertIs(system.trajectory, trajectory)
		self.assertFalse(hasattr(potential, "trajectory"))
		self.assertFalse(hasattr(trajectory, "potential"))

	def test_factory_selects_fc_from_the_trajectory_entity(self) -> None:
		potential = single_mode_potential()
		trajectory = TrajectoryFC(rho=0.2, eta=0.1, n_trajectories=4)

		system = create_system(potential, trajectory)

		self.assertIsInstance(system, SystemFC)
		self.assertIs(system.potential, potential)
		self.assertIs(system.trajectory, trajectory)

	def test_gc_and_fc_own_distinct_numerical_paths(self) -> None:
		self.assertIn("_integrate", SystemGC.__dict__)
		self.assertIn("_integrate", SystemFC.__dict__)
		self.assertNotEqual(SystemGC._integrate, SystemFC._integrate)
		self.assertNotIn("flow", SystemGC.__dict__)
		self.assertIn("flow", SystemFC.__dict__)

	def test_concrete_systems_reject_the_wrong_trajectory(self) -> None:
		potential = single_mode_potential()

		with self.assertRaisesRegex(TypeError, "TrajectoryGC"):
			SystemGC(potential, TrajectoryFC(rho=0.2, eta=0.1))  # type: ignore[arg-type]
		with self.assertRaisesRegex(TypeError, "TrajectoryFC"):
			SystemFC(potential, TrajectoryGC())  # type: ignore[arg-type]


class SystemEquationTests(unittest.TestCase):
	def test_gc_equations_hamiltonian_and_k_dot_match_one_analytic_mode(self) -> None:
		system = create_system(single_mode_potential(), TrajectoryGC(rho=0.0))
		x = np.array([0.2, 0.7, 1.1])
		y = np.array([0.4, 0.3, -0.2])
		state = np.concatenate((x, y))
		t = 0.25
		phase = x + y - t

		derivative = system.vector_field(t, state)
		hamiltonian = system.hamiltonian(t, state)
		k_dot = system.extended_momentum_derivative(t, state)

		np.testing.assert_allclose(derivative, np.concatenate((-np.cos(phase), np.cos(phase))))
		np.testing.assert_allclose(hamiltonian, np.sin(phase))
		np.testing.assert_allclose(k_dot, np.cos(phase))
		self.assertEqual(k_dot.shape, (3,))

	def test_fc_equations_hamiltonian_and_k_dot_match_one_analytic_mode(self) -> None:
		trajectory = TrajectoryFC(rho=0.2, eta=0.1)
		system = create_system(single_mode_potential(), trajectory)
		x = np.array([0.2, 0.7])
		y = np.array([0.4, 0.3])
		vx = np.array([1.0, -0.5])
		vy = np.array([0.25, 0.75])
		state = np.concatenate((x, y, vx, vy))
		t = 0.25
		phase = x + y - t
		gradient = np.cos(phase)

		derivative = system.vector_field(t, state)
		hamiltonian = system.hamiltonian(t, state)
		k_dot = system.extended_momentum_derivative(t, state)

		expected_derivative = np.concatenate((
			trajectory.velocity_scale * vx,
			trajectory.velocity_scale * vy,
			-trajectory.electric_scale * gradient + trajectory.larmor_frequency * vy,
			-trajectory.electric_scale * gradient - trajectory.larmor_frequency * vx,
		))
		expected_hamiltonian = (
			trajectory.rho / (4 * abs(trajectory.eta)) * (vx**2 + vy**2)
			+ trajectory.electric_scale * np.sin(phase)
		)
		np.testing.assert_allclose(derivative, expected_derivative)
		np.testing.assert_allclose(hamiltonian, expected_hamiltonian)
		np.testing.assert_allclose(k_dot, trajectory.electric_scale * gradient)
		self.assertEqual(k_dot.shape, (2,))

	def test_hamiltonian_preserves_trajectory_and_time_axes(self) -> None:
		system = create_system(single_mode_potential(), TrajectoryGC())
		states = np.array([
			[1.0, 1.2, 1.4],
			[2.0, 2.2, 2.4],
			[3.0, 3.2, 3.4],
			[4.0, 4.2, 4.4],
		])

		energy = system.hamiltonian(np.array([0.0, 0.5, 1.0]), states)

		self.assertEqual(energy.shape, (2, 3))

	def test_fc_flows_preserve_physical_and_energy_state_shapes(self) -> None:
		potential = single_mode_potential()
		trajectory = TrajectoryFC(rho=0.2, eta=0.1, n_trajectories=4)
		state = initialize_trajectory(trajectory, potential.grid)
		system = create_system(potential, trajectory)
		state_with_k = np.concatenate((state, np.zeros(4)))

		self.assertEqual(system.flow(0.01, 0.0, state).shape, state.shape)
		self.assertEqual(system.adjoint_flow(0.01, 0.0, state).shape, state.shape)
		self.assertEqual(
			system.flow(0.01, 0.0, state_with_k, check_energy=True).shape,
			state_with_k.shape,
		)
		self.assertEqual(
			system.adjoint_flow(0.01, 0.0, state_with_k, check_energy=True).shape,
			state_with_k.shape,
		)


class SystemSimulationTests(unittest.TestCase):
	def test_gc_simulation_smoke_uses_extended_path(self) -> None:
		potential = single_mode_potential(coefficient=0.0)
		trajectory = TrajectoryGC(n_trajectories=4, initialization="fixed")
		initial = initialize_trajectory(trajectory, potential.grid)
		system = create_system(potential, trajectory)

		solution = system.simulate(
			t_span=(0.0, 0.04),
			step=0.01,
			n_save_step=3,
			method="Verlet",
			check_energy=True,
		)

		expected = np.repeat(initial[:, np.newaxis], solution.t.size, axis=1)
		np.testing.assert_allclose(solution.y, expected)
		self.assertEqual(solution.y.shape, (8, 3))
		self.assertEqual(solution.k.shape, (4, 3))
		self.assertTrue(np.isfinite(solution.err))

	def test_fc_energy_simulation_keeps_4n_physical_rows_and_n_k_rows(self) -> None:
		trajectory = TrajectoryFC(rho=0.2, eta=0.1, n_trajectories=2)
		initial = np.array([
			0.2,
			0.7,
			0.4,
			0.3,
			1.0,
			-0.5,
			0.25,
			0.75,
		])

		trajectory.set_initial_state(initial)
		system = create_system(single_mode_potential(), trajectory)

		solution = system.simulate(
			t_span=(0.0, 0.04),
			step=0.01,
			n_save_step=3,
			method="Verlet",
			check_energy=True,
		)

		self.assertEqual(solution.y.shape, (8, 3))
		self.assertEqual(solution.k.shape, (2, 3))
		self.assertTrue(np.all(np.isfinite(solution.y)))
		self.assertTrue(np.all(np.isfinite(solution.k)))
		self.assertTrue(np.isfinite(solution.err))

	def test_simulate_rejects_an_uninitialized_trajectory(self) -> None:
		system = create_system(
			single_mode_potential(coefficient=0.0),
			TrajectoryGC(n_trajectories=4, initialization="fixed"),
		)

		with self.assertRaisesRegex(ValueError, "no initial state"):
			system.simulate(
				t_span=(0.0, 0.02),
				step=0.01,
				n_save_step=2,
				method="Verlet",
			)

	def test_simulate_uses_the_state_owned_by_trajectory(self) -> None:
		trajectory = TrajectoryGC(n_trajectories=2)
		initial = np.array([0.2, 0.7, 0.4, 0.3])
		trajectory.set_initial_state(initial)
		system = create_system(single_mode_potential(coefficient=0.0), trajectory)

		solution = system.simulate(
			t_span=(0.0, 0.02),
			step=0.01,
			n_save_step=2,
			method="Verlet",
		)

		np.testing.assert_allclose(solution.y[:, 0], initial)
		np.testing.assert_allclose(solution.y[:, 1], initial)


if __name__ == "__main__":
	unittest.main()
