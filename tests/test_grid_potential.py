from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from classes import Grid, Potential, PotentialFields, PotentialMode
from workflows.potentials import extract_potential


class GridTests(unittest.TestCase):
	def test_grid_exposes_validated_geometry(self) -> None:
		grid = Grid(x0=-1.0, y0=0.0, dx=0.5, dy=0.5, nx=5, ny=7)

		self.assertEqual(grid.shape, (5, 7))
		self.assertAlmostEqual(grid.dx, 0.5)
		self.assertAlmostEqual(grid.dy, 0.5)
		self.assertFalse(grid.x.flags.writeable)
		self.assertFalse(grid.y.flags.writeable)

	def test_periodic_grid_requires_a_complete_period(self) -> None:
		with self.assertRaisesRegex(ValueError, r"nx \* dx"):
			Grid(x0=0.0, y0=0.0, dx=1 / 7, dy=1 / 7, nx=8, ny=8, period=2.0)

	def test_from_axes_reduces_hdf5_coordinates_to_parameters(self) -> None:
		x = np.linspace(-2.0, 2.0, 9)
		y = np.linspace(1.0, 4.0, 7)

		grid = Grid.from_axes(x, y)

		self.assertEqual((grid.x0, grid.y0), (-2.0, 1.0))
		self.assertAlmostEqual(grid.dx, 0.5)
		self.assertAlmostEqual(grid.dy, 0.5)
		np.testing.assert_allclose(grid.x, x)
		np.testing.assert_allclose(grid.y, y)

	def test_periodic_resize_preserves_domain(self) -> None:
		period = 2 * np.pi
		grid = Grid.from_bounds(0.0, period, 0.0, period, 8, 10, periodic=True)
		resized = grid.resized(12, 14)

		self.assertEqual(resized.shape, (12, 14))
		self.assertAlmostEqual(resized.nx * resized.dx, period)
		self.assertAlmostEqual(resized.ny * resized.dy, period)


class PotentialGridTests(unittest.TestCase):
	def test_field_adapter_rejects_unpaired_frequencies(self) -> None:
		with self.assertRaisesRegex(ValueError, "mode coefficients"):
			PotentialFields.from_arrays(None, [np.zeros((4, 4))], [])

	def test_potential_uses_grid_without_flat_geometry_aliases(self) -> None:
		period = 2 * np.pi
		grid = Grid.from_bounds(0.0, period, 0.0, period, 12, 10, periodic=True)
		X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
		fields = PotentialFields(modes=(PotentialMode(np.sin(X) + 1j * np.cos(Y), 1.0),))
		potential = Potential(grid, fields)

		self.assertIs(potential.grid, grid)
		self.assertFalse(hasattr(potential, "x"))
		self.assertFalse(hasattr(potential, "nx"))

		resampled = potential.resample(grid.resized(16, 14))
		self.assertEqual(resampled.grid.shape, (16, 14))
		self.assertEqual(resampled.fields.modes[0].coefficient.shape, (16, 14))
		self.assertEqual(resampled.fields.modes[0].frequency, 1.0)

	def test_hdf5_axes_are_adapted_and_can_be_resampled(self) -> None:
		x = np.linspace(-2.0, 2.0, 5)
		y = np.linspace(1.0, 4.0, 4)
		fields = np.zeros((2, y.size, x.size), dtype=np.complex128)
		fields[1] = np.add.outer(y, x)

		with TemporaryDirectory() as directory:
			filename = Path(directory) / "potential.h5"
			with h5py.File(filename, "w") as file:
				file["Rcells"] = x
				file["Zcells"] = y
				file["freqs"] = np.array([0.0, 1.0])
				file["fields"] = fields

			potential = extract_potential(filename, target_shape=(7, 6))

		self.assertEqual(potential.grid.shape, (7, 6))
		self.assertEqual(potential.grid.xmin, x[0])
		self.assertEqual(potential.grid.xmax, x[-1])
		self.assertEqual(potential.grid.ymin, y[0])
		self.assertEqual(potential.grid.ymax, y[-1])


if __name__ == "__main__":
	unittest.main()
