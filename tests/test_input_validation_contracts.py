"""Public input policies retained when preparing validated potential data."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from initial_conditions.star import radial_star, ranked_radial_star
from potential.grid import Grid
from potential.potential import Potential


class InputValidationContractsTests(unittest.TestCase):
	"""Protect conversion, error precedence, and geometry at public boundaries."""

	def test_grid_constructor_and_factory_preserve_distinct_type_errors(self) -> None:
		for invalid in (True, np.bool_(True), 4.0):
			with self.subTest(value=invalid):
				with self.assertRaisesRegex(TypeError, "`nx` must be an integer"):
					Grid(0.0, 0.0, 0.25, 0.25, invalid, 4, 1.0)
				with self.assertRaisesRegex(ValueError, "`nx` must be an integer of at least 2"):
					Grid.periodic(invalid, 4, 1.0)
		grid = Grid.periodic(np.int64(4), np.int64(4), 1.0)
		self.assertEqual(grid.shape, (4, 4))
		self.assertIs(type(grid.nx), int)
		self.assertFalse(grid.x.flags.writeable)

	def test_random_amplitude_retains_numeric_conversion_but_counts_reject_bool(self) -> None:
		boolean_amplitude = Potential.random(A=True, M=1, nx=8, ny=8, seed=np.int64(3))
		numeric_amplitude = Potential.random(A=1.0, M=1, nx=8, ny=8, seed=3)
		np.testing.assert_array_equal(boolean_amplitude.modes, numeric_amplitude.modes)
		with self.assertRaisesRegex(ValueError, "`M` must be a positive integer"):
			Potential.random(A=1.0, M=True, nx=0, ny=0, seed=False)
		with self.assertRaisesRegex(TypeError, "`seed` must be an integer"):
			Potential.random(A=1.0, M=1, nx=0, ny=0, seed=False)

	def test_import_controls_are_checked_before_opening_the_file(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			missing = Path(directory) / "missing.h5"
			with self.assertRaisesRegex(ValueError, "`B` must be finite and non-zero"):
				Potential.load(missing, B=0.0, sigma=-1.0)
			for sigma in (-1.0, np.nan, np.inf, -np.inf):
				with self.subTest(sigma=sigma):
					with self.assertRaisesRegex(ValueError, "`sigma` must be finite and non-negative"):
						Potential.load(missing, B=1.5, sigma=sigma)

	def test_field_selection_retains_integer_conversion_and_frozen_provenance(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "field.h5"
			with h5py.File(path, "w") as stream:
				stream["Rcells"] = np.arange(8) * 0.01
				stream["Zcells"] = np.arange(8) * 0.01
				stream["freqs"] = np.asarray([0.0, 2.0])
				stream["fields"] = np.stack((np.ones((8, 8)), np.eye(8))).astype(complex)
				stream.attrs["scale"] = np.asarray([2.0])
			converted = Potential.load(path, B=1.5, indx=[0.9, 1.9])
			integer = Potential.load(path, B=1.5, indx=[0, 1])
		np.testing.assert_array_equal(converted.mean, integer.mean)
		np.testing.assert_array_equal(converted.modes, integer.modes)
		self.assertFalse(converted.metadata.source_field_indices.flags.writeable)
		self.assertFalse(converted.metadata.attributes["scale"].flags.writeable)

	def test_star_validation_keeps_count_precedence_and_particle_order(self) -> None:
		with self.assertRaisesRegex(ValueError, "arms must be a positive integer"):
			radial_star(center=(float("nan"), 0.0), arm_length=1.0, arms=True, particles_per_arm=2)
		with self.assertRaisesRegex(ValueError, "equal count per arm"):
			ranked_radial_star(center=(float("nan"), 0.0), outer_radius=1.0, particles=3, arms=2)
		star = ranked_radial_star(center=(0.0, 0.0), outer_radius=1.0, particles=4, arms=2)
		state = star.initial_state
		assert state is not None
		x, y = star.positions(state)
		np.testing.assert_allclose(np.hypot(x, y), np.linspace(0.0, 1.0, 4))


if __name__ == "__main__":
	unittest.main()
