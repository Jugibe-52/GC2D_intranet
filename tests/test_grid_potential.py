from __future__ import annotations

import unittest

import numpy as np

from classes import Grid, Potential


class GridTests(unittest.TestCase):
	def test_grid_exposes_validated_geometry(self) -> None:
		grid = Grid(np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 3.0, 7))

		self.assertEqual(grid.shape, (5, 7))
		self.assertAlmostEqual(grid.dx, 0.5)
		self.assertAlmostEqual(grid.dy, 0.5)
		self.assertFalse(grid.x.flags.writeable)
		self.assertFalse(grid.y.flags.writeable)

	def test_periodic_grid_requires_a_complete_period(self) -> None:
		with self.assertRaisesRegex(ValueError, r"nx \* dx"):
			Grid(np.linspace(0.0, 1.0, 8), np.linspace(0.0, 1.0, 8), period=2.0)

	def test_periodic_resize_preserves_domain(self) -> None:
		period = 2 * np.pi
		grid = Grid(
			np.linspace(0.0, period, 8, endpoint=False),
			np.linspace(0.0, period, 10, endpoint=False),
			period=period,
		)
		resized = grid.resized(12, 14)

		self.assertEqual(resized.shape, (12, 14))
		self.assertAlmostEqual(resized.nx * resized.dx, period)
		self.assertAlmostEqual(resized.ny * resized.dy, period)


class PotentialGridTests(unittest.TestCase):
	def test_potential_uses_grid_without_flat_geometry_aliases(self) -> None:
		period = 2 * np.pi
		grid = Grid(
			np.linspace(0.0, period, 12, endpoint=False),
			np.linspace(0.0, period, 10, endpoint=False),
			period=period,
		)
		X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
		potential = Potential(grid, [None, [np.sin(X) + 1j * np.cos(Y)]], [1.0])

		self.assertIs(potential.grid, grid)
		self.assertFalse(hasattr(potential, "x"))
		self.assertFalse(hasattr(potential, "nx"))

		resampled = potential.resample(grid.resized(16, 14))
		self.assertEqual(resampled.grid.shape, (16, 14))
		fluctuations = resampled.fields[1]
		self.assertIsNotNone(fluctuations)
		assert fluctuations is not None
		self.assertEqual(fluctuations[0].shape, (16, 14))


if __name__ == "__main__":
	unittest.main()
