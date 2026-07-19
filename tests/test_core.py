"""Fast contract tests for the notebook-facing simulation core."""

from __future__ import annotations

import unittest

import numpy as np

from classes import Area, Potential, SystemFC, SystemGC, TrajectoryFC, TrajectoryGC


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
		state = np.asarray([1.0, 2.0, 3.0, 4.0])
		trajectory = TrajectoryGC(state, rho=0.2)
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
		state = np.asarray([1.0, 2.0, 3.0, 4.0, 0.5, 0.6, -0.5, -0.6])
		trajectory = TrajectoryFC(state, rho=0.4, eta=-0.2)

		x, y = trajectory.positions(state)
		vx, vy = trajectory.velocities(state)
		np.testing.assert_allclose(x, [1.0, 2.0])
		np.testing.assert_allclose(y, [3.0, 4.0])
		np.testing.assert_allclose(vx, [0.5, 0.6])
		np.testing.assert_allclose(vy, [-0.5, -0.6])
		self.assertAlmostEqual(trajectory.velocity_scale, 1.0)
		self.assertAlmostEqual(trajectory.electric_scale, -2.5)
		self.assertAlmostEqual(trajectory.larmor_frequency, -2.5)

	def test_area_constructors_and_area_calculation(self) -> None:
		square = Area.square(
			center=(2 * np.pi - 0.25, 2 * np.pi - 0.25),
			side=1.0,
			points_per_side=4,
			rho=0.2,
		)
		self.assertIsInstance(square, TrajectoryGC)
		self.assertEqual(square.shape, "square")
		self.assertAlmostEqual(float(square.calculate_area()), 1.0)

		square_state = square.state
		assert square_state is not None
		x, y = square.positions(square_state)
		wrapped = np.concatenate((x % (2 * np.pi), y % (2 * np.pi)))
		self.assertAlmostEqual(
			float(square.calculate_area(wrapped, period=2 * np.pi)),
			1.0,
		)

		circle = Area.circle(center=(1.0, 2.0), radius=0.5, points=128)
		self.assertEqual(circle.shape, "circle")
		expected_polygon_area = 128 * 0.5**2 * np.sin(2 * np.pi / 128) / 2
		self.assertAlmostEqual(
			float(circle.calculate_area()),
			expected_polygon_area,
		)

		circle_state = circle.state
		assert circle_state is not None
		time_series = np.column_stack((circle_state, circle_state + 0.1))
		areas = circle.calculate_area(time_series)
		self.assertEqual(areas.shape, (2,))
		np.testing.assert_allclose(areas, expected_polygon_area)

		system = SystemGC(random_potential(), square)
		solution = system.simulate(
			step=0.01,
			t_span=(0.0, 0.02),
			n_save_step=3,
			check_energy=False,
			progress=False,
		)
		transported_area = square.calculate_area(solution.y, period=2 * np.pi)
		self.assertEqual(transported_area.shape, (3,))
		self.assertTrue(np.all(np.isfinite(transported_area)))


class SystemTests(unittest.TestCase):
	def test_gc_area_animation_tracks_relative_error(self) -> None:
		area = Area.square(
			center=(np.pi, np.pi),
			side=1.0,
			points_per_side=4,
			rho=0.05,
		)
		system = SystemGC(random_potential(), area)
		solution = system.simulate(
			step=0.01,
			t_span=(0.0, 0.02),
			n_save_step=3,
			check_energy=False,
			progress=False,
		)

		animation = system.animate_area(solution, frames=3, interval=20)
		artists = animation._func(2)
		self.assertEqual(len(artists), 6)
		self.assertEqual(artists[1].__class__.__name__, "Quiver")
		expected_area = area.calculate_area(solution.y, period=2 * np.pi)
		expected_error = (expected_area - expected_area[0]) / abs(expected_area[0])
		np.testing.assert_allclose(artists[3].get_xdata(), solution.t)
		np.testing.assert_allclose(artists[3].get_ydata(), expected_error)
		self.assertIn("varepsilon_A", artists[5].get_text())
		animation._draw_was_started = True

		plain_trajectory = TrajectoryGC(np.asarray([1.0, 1.2]), rho=0.05)
		plain_system = SystemGC(random_potential(), plain_trajectory)
		with self.assertRaises(TypeError):
			plain_system.animate_area(solution)
		with self.assertRaises(ValueError):
			system.animate_area(solution, frames=1)

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
