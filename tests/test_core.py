"""Fast contract tests for the notebook-facing simulation core."""

from __future__ import annotations

import unittest

import numpy as np

from classes import Potential, SystemFC, SystemGC, TrajectoryFC, TrajectoryGC


def random_potential(*, interpolation_order: int = 3) -> Potential:
	return Potential.random(
		A=0.08,
		M=3,
		nx=16,
		ny=16,
		seed=27,
		interpolation_order=interpolation_order,
	)


class PotentialTests(unittest.TestCase):
	def test_random_is_deterministic_and_derivatives_are_finite(self) -> None:
		first = random_potential()
		second = random_potential()

		np.testing.assert_allclose(first.evaluate(0.3), second.evaluate(0.3))
		times = np.asarray([0.0, 0.2, 0.5])
		fields = first.evaluate(times)
		self.assertEqual(fields.shape, first.grid.shape + (times.size,))
		for index, time in enumerate(times):
			np.testing.assert_allclose(fields[..., index], first.evaluate(time))

		x = np.asarray([0.7, 1.4, 2.1])
		y = np.asarray([0.9, 1.7, 2.5])
		for derivative in (
			first.evaluate(0.3, x, y, dx=1),
			first.evaluate(0.3, x, y, dy=1),
			first.evaluate(0.3, x, y, dt=1),
		):
			self.assertTrue(np.all(np.isfinite(derivative)))

		ex, ey = first.electric_field(0.3, x, y)
		np.testing.assert_allclose(ex, -first.evaluate(0.3, x, y, dx=1))
		np.testing.assert_allclose(ey, -first.evaluate(0.3, x, y, dy=1))

	def test_gyroaverage_preserves_the_original_potential(self) -> None:
		potential = random_potential(interpolation_order=5)
		original = potential.evaluate(0.2).copy()

		averaged = potential.gyroaverage(0.1)

		self.assertIsInstance(averaged, Potential)
		self.assertTrue(np.all(np.isfinite(averaged.evaluate(0.2))))
		np.testing.assert_allclose(potential.evaluate(0.2), original)
		self.assertTrue(callable(potential.plot))
		self.assertTrue(callable(potential.animate))

		with self.assertRaises(ValueError):
			Potential.random(A=0.1, M=2, nx=0, ny=8, interpolation_order=3)
		with self.assertRaises(ValueError):
			Potential.random(A=0.1, M=2, nx=8, ny=8, interpolation_order=1)


class TrajectoryTests(unittest.TestCase):
	def test_gc_state_layout_and_copy(self) -> None:
		trajectory = TrajectoryGC(rho=0.2)
		state = np.asarray([1.0, 2.0, 3.0, 4.0])
		trajectory.set_initial_state(state)
		state[0] = -10.0

		stored = trajectory.state
		self.assertIsNotNone(stored)
		assert stored is not None
		np.testing.assert_allclose(stored, [1.0, 2.0, 3.0, 4.0])
		x, y = trajectory.positions(stored)
		np.testing.assert_allclose(x, [1.0, 2.0])
		np.testing.assert_allclose(y, [3.0, 4.0])

		stored[0] = -20.0
		np.testing.assert_allclose(trajectory.state, [1.0, 2.0, 3.0, 4.0])

	def test_fc_state_layout_and_scales(self) -> None:
		trajectory = TrajectoryFC(rho=0.4, eta=-0.2)
		state = np.asarray([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, -0.5, -0.6])
		trajectory.set_initial_state(state)

		x, y = trajectory.positions(state)
		vx, vy = trajectory.velocities(state)
		np.testing.assert_allclose(x, [1.0, 2.0])
		np.testing.assert_allclose(y, [3.0, 4.0])
		np.testing.assert_allclose(vx, [0.5, 0.6])
		np.testing.assert_allclose(vy, [-0.5, -0.6])
		self.assertAlmostEqual(trajectory.velocity_scale, 1.0)
		self.assertAlmostEqual(trajectory.electric_scale, -2.5)
		self.assertAlmostEqual(trajectory.larmor_frequency, -2.5)


class SystemTests(unittest.TestCase):
	def test_simulate_requires_an_initial_state(self) -> None:
		system = SystemGC(random_potential(), TrajectoryGC(rho=0.05))

		with self.assertRaises(ValueError):
			system.simulate(
				step=0.01,
				t_span=(0.0, 0.02),
				n_save_step=3,
				check_energy=False,
			)

		trajectory = TrajectoryGC(rho=0.05)
		trajectory.set_initial_state(np.asarray([1.0, 1.2]))
		with self.assertRaises(ValueError):
			SystemGC(random_potential(), trajectory).simulate(
				step=0.01,
				t_span=(1.0, 1.0),
				n_save_step=3,
			)

	def test_gc_bm4_simulation_tracks_generalized_energy(self) -> None:
		trajectory = TrajectoryGC(rho=0.05)
		trajectory.set_initial_state(np.asarray([1.0, 1.2]))
		system = SystemGC(random_potential(), trajectory)

		solution = system.simulate(
			step=0.01,
			t_span=(0.0, 0.04),
			n_save_step=5,
			check_energy=True,
			progress=False,
		)

		np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.04, 5))
		self.assertEqual(solution.y.shape, (2, 5))
		self.assertEqual(np.asarray(solution.k).shape, (1, 5))
		self.assertGreater(solution.n_steps, 0)
		self.assertTrue(np.all(np.isfinite(np.asarray(solution.err))))
		self.assertTrue(np.all(np.isfinite(system.hamiltonian(solution.t, solution.y))))

	def test_fc_bm4_simulation_tracks_generalized_energy(self) -> None:
		trajectory = TrajectoryFC(rho=0.2, eta=0.1)
		trajectory.set_initial_state(np.asarray([1.0, 1.2, 0.4, -0.3]))
		system = SystemFC(random_potential(), trajectory)

		solution = system.simulate(
			step=0.01,
			t_span=(0.0, 0.04),
			n_save_step=5,
			check_energy=True,
			progress=False,
		)

		np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.04, 5))
		self.assertEqual(solution.y.shape, (4, 5))
		self.assertEqual(np.asarray(solution.k).shape, (1, 5))
		self.assertGreater(solution.n_steps, 0)
		self.assertTrue(np.all(np.isfinite(np.asarray(solution.err))))
		self.assertTrue(np.all(np.isfinite(system.hamiltonian(solution.t, solution.y))))


if __name__ == "__main__":
	unittest.main()
