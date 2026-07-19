from __future__ import annotations

import unittest

import numpy as np

from classes import Trajectory, TrajectoryFC, TrajectoryGC


class TrajectoryTests(unittest.TestCase):
	def test_trajectory_is_an_abstract_contract(self) -> None:
		with self.assertRaises(TypeError):
			Trajectory()  # type: ignore[abstract]

	def test_gc_can_be_built_without_a_potential(self) -> None:
		trajectory = TrajectoryGC(rho=0.2, eta=0.1, n_trajectories=4)

		self.assertEqual(trajectory.kind, "gc")
		self.assertFalse(hasattr(trajectory, "potential"))
		self.assertFalse(hasattr(trajectory, "system"))

	def test_fixed_gc_state_uses_block_layout_and_square_count(self) -> None:
		trajectory = TrajectoryGC(n_trajectories=5, initialization="fixed")

		state = trajectory.initial_state((0.0, 2 * np.pi), (0.0, 2 * np.pi))
		x, y = trajectory.get_positions(state)

		self.assertEqual(state.shape, (8,))
		self.assertEqual(trajectory.n_trajectories, 4)
		np.testing.assert_allclose(np.unique(x), [0.0, np.pi])
		np.testing.assert_allclose(np.unique(y), [0.0, np.pi])
		self.assertIsNone(trajectory.get_velocities(state))

	def test_random_initialization_is_seeded_and_uses_full_bounds(self) -> None:
		first = TrajectoryGC(n_trajectories=6, initialization="random", seed=42)
		second = TrajectoryGC(n_trajectories=6, initialization="random", seed=42)

		first_state = first.initial_state((-1.0, 3.0), (2.0, 8.0))
		second_state = second.initial_state((-1.0, 3.0), (2.0, 8.0))
		x, y = first.get_positions(first_state)

		np.testing.assert_array_equal(first_state, second_state)
		self.assertTrue(np.all((-1.0 <= x) & (x < 3.0)))
		self.assertTrue(np.all((2.0 <= y) & (y < 8.0)))

	def test_selected_initialization_requires_matching_coordinates(self) -> None:
		trajectory = TrajectoryGC(
			n_trajectories=2,
			initialization="selected",
			x0=np.array([0.1, 0.2]),
			y0=np.array([0.3]),
		)

		with self.assertRaisesRegex(ValueError, "same shape"):
			trajectory.initial_state((0.0, 1.0), (0.0, 1.0))

	def test_fc_adds_unit_velocity_blocks(self) -> None:
		trajectory = TrajectoryFC(
			rho=0.2,
			eta=-0.1,
			n_trajectories=4,
			initialization="fixed",
			seed=9,
		)

		state = trajectory.initial_state((0.0, 2 * np.pi), (0.0, 2 * np.pi))
		vx, vy = trajectory.get_velocities(state)

		self.assertEqual(trajectory.kind, "fc")
		self.assertEqual(state.shape, (16,))
		np.testing.assert_allclose(vx**2 + vy**2, 1.0)
		self.assertAlmostEqual(trajectory.velocity_scale, 1.0)
		self.assertAlmostEqual(trajectory.electric_scale, -5.0)
		self.assertAlmostEqual(trajectory.larmor_frequency, -5.0)

	def test_fc_requires_nonzero_rho_and_eta(self) -> None:
		for rho, eta in ((0.0, 0.1), (0.1, 0.0)):
			with self.subTest(rho=rho, eta=eta):
				with self.assertRaisesRegex(ValueError, "non-zero"):
					TrajectoryFC(rho=rho, eta=eta)


if __name__ == "__main__":
	unittest.main()
