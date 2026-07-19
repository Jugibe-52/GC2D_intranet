from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import numpy as np
from matplotlib.animation import FuncAnimation

from classes import (
	FourierPotential,
	Grid,
	GridPotential,
	PotentialFields,
	SystemGC,
	SystemResearch,
	TrajectoryFC,
	TrajectoryGC,
	create_system,
)


def single_mode_potential() -> FourierPotential:
	coefficients = np.zeros((2, 2), dtype=np.complex128)
	coefficients[1, 1] = 1.0
	return FourierPotential(1.0, 1, coefficients=coefficients)


class SystemResearchTangentTests(unittest.TestCase):
	def test_guiding_center_tangent_dynamics_preserve_augmented_shape(self) -> None:
		system = create_system(single_mode_potential(), TrajectoryGC())
		state = np.concatenate((np.array([1.0, 2.0]), np.eye(2).reshape(-1)))

		derivative = SystemResearch(system).y_dot_lyap(0.0, state)

		self.assertEqual(derivative.shape, state.shape)
		np.testing.assert_allclose(derivative[:2], system.vector_field(0.0, state[:2]))

	def test_full_cyclotron_tangent_dynamics_preserve_augmented_shape(self) -> None:
		system = create_system(
			single_mode_potential(),
			TrajectoryFC(rho=0.1, eta=0.2),
		)
		state = np.concatenate((
			np.array([1.0, 2.0, 0.5, -0.5]),
			np.eye(4).reshape(-1),
		))

		derivative = SystemResearch(system).y_dot_lyap(0.0, state)

		self.assertEqual(derivative.shape, state.shape)
		np.testing.assert_allclose(derivative[:4], system.vector_field(0.0, state[:4]))


class GuidingCenterAreaTests(unittest.TestCase):
	@staticmethod
	def make_system(*, periodic: bool = False) -> SystemGC:
		period = 2 * np.pi
		coordinates = np.linspace(0.0, period, 8, endpoint=not periodic)
		potential = GridPotential(
			Grid.from_axes(coordinates, coordinates, period=period if periodic else None),
			PotentialFields(mean=np.zeros((8, 8))),
			k=3,
		)
		return cast(SystemGC, create_system(potential, TrajectoryGC()))

	def test_square_initial_conditions_are_centred_and_counter_clockwise(self) -> None:
		system = self.make_system()
		research = SystemResearch(system)

		initial = research.guiding_center_square_initial_conditions(side=1.0)
		x, y = system.get_positions(initial)

		centre_x = (system.grid.xmin + system.grid.xmax) / 2
		centre_y = (system.grid.ymin + system.grid.ymax) / 2
		np.testing.assert_allclose(x, [centre_x, centre_x + 1, centre_x + 1, centre_x])
		np.testing.assert_allclose(y, [centre_y, centre_y, centre_y + 1, centre_y + 1])

	def test_area_element_is_preserved_by_shear(self) -> None:
		research = SystemResearch(self.make_system())
		solution = SimpleNamespace(
			t=np.array([0.0, 0.5, 1.0]),
			y=np.array([
				[1.0, 1.0, 1.0],
				[1.1, 1.1, 1.1],
				[1.0, 1.1, 1.2],
				[1.0, 1.0, 1.0],
				[1.0, 1.0, 1.0],
				[1.2, 1.2, 1.2],
			]),
		)

		area = research.guiding_center_area_element(solution)

		np.testing.assert_allclose(area, 0.02)

	def test_area_element_uses_minimum_periodic_displacement(self) -> None:
		system = self.make_system(periodic=True)
		self.assertIsNotNone(system.grid.period)
		period = cast(float, system.grid.period)
		solution = SimpleNamespace(
			t=np.array([0.0, 1.0]),
			y=np.array([
				[period - 0.05, period - 0.05],
				[0.05, 0.05],
				[period - 0.05, period - 0.05],
				[period - 0.05, period - 0.05],
				[period - 0.05, period - 0.05],
				[0.15, 0.15],
			]),
		)

		area = SystemResearch(system).guiding_center_area_element(solution)

		np.testing.assert_allclose(area, 0.02)

	def test_polygon_area_is_preserved_by_shear(self) -> None:
		research = SystemResearch(self.make_system())
		solution = SimpleNamespace(
			t=np.array([0.0, 0.5, 1.0]),
			y=np.array([
				[1.0, 1.0, 1.0],
				[2.0, 2.0, 2.0],
				[2.0, 2.25, 2.5],
				[1.0, 1.25, 1.5],
				[1.0, 1.0, 1.0],
				[1.0, 1.0, 1.0],
				[2.0, 2.0, 2.0],
				[2.0, 2.0, 2.0],
			]),
		)

		area = research.guiding_center_polygon_area(solution)

		np.testing.assert_allclose(area, 1.0)

	def test_square_boundary_can_use_more_than_four_points(self) -> None:
		system = self.make_system()
		initial = SystemResearch(system).guiding_center_square_initial_conditions(
			side=1.0,
			lower_left=(1.0, 1.0),
			points_per_side=3,
		)
		x, _ = system.get_positions(initial)

		self.assertEqual(x.size, 12)

	def test_area_animation_is_created(self) -> None:
		research = SystemResearch(self.make_system())
		solution = SimpleNamespace(
			t=np.array([0.0, 0.5, 1.0]),
			y=np.array([
				[1.0, 1.0, 1.0],
				[1.1, 1.1, 1.1],
				[1.0, 1.1, 1.2],
				[1.0, 1.0, 1.0],
				[1.0, 1.0, 1.0],
				[1.2, 1.2, 1.2],
			]),
		)

		animation = research.animate_electric_psi_area_conservation(
			solution,
			frame_stride=2,
			step=1,
			repeat=False,
		)
		setattr(animation, "_draw_was_started", True)

		self.assertIsInstance(animation, FuncAnimation)

	def test_area_animation_rejects_zero_initial_area(self) -> None:
		research = SystemResearch(self.make_system())
		solution = SimpleNamespace(
			t=np.array([0.0, 1.0]),
			y=np.array([
				[1.0, 1.0],
				[1.1, 1.1],
				[1.2, 1.2],
				[1.0, 1.0],
				[1.0, 1.0],
				[1.0, 1.0],
			]),
		)

		with self.assertRaisesRegex(ValueError, "must be non-zero"):
			research.animate_electric_psi_area_conservation(solution)


if __name__ == "__main__":
	unittest.main()
