from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from classes import FourierPotential, SystemResearch, TrajectoryGC, create_system
from workflows import initialize_guiding_center_square, initialize_trajectory


class WorkflowTrajectoryInitializationTests(unittest.TestCase):
	def test_general_initializer_assigns_state_to_trajectory(self) -> None:
		potential = FourierPotential(0.0, 1)
		trajectory = TrajectoryGC(n_trajectories=4, initialization="fixed")

		state = initialize_trajectory(trajectory, potential.grid, n_trajectories=4)

		np.testing.assert_allclose(trajectory.state, state)
		self.assertEqual(trajectory.n_trajectories, 4)

	def test_square_initializer_is_compatible_with_system_research(self) -> None:
		potential = FourierPotential(0.0, 1)
		trajectory = TrajectoryGC()

		state = initialize_guiding_center_square(
			trajectory,
			potential.grid,
			side=1.0,
			lower_left=(1.0, 1.0),
			points_per_side=1,
		)
		system = create_system(potential, trajectory)
		research = SystemResearch(system)
		solution = SimpleNamespace(t=np.array([0.0]), y=state[:, np.newaxis])

		np.testing.assert_allclose(trajectory.state, state)
		np.testing.assert_allclose(research.guiding_center_polygon_area(solution), 1.0)

	def test_square_initializer_can_sample_each_edge(self) -> None:
		potential = FourierPotential(0.0, 1)
		trajectory = TrajectoryGC()

		state = initialize_guiding_center_square(
			trajectory,
			potential.grid,
			points_per_side=3,
		)
		x, _ = trajectory.get_positions(state)

		self.assertEqual(x.size, 12)


if __name__ == "__main__":
	unittest.main()
