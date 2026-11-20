"""Study identities distinguish reflected dynamics from historical direct wrapping."""

from __future__ import annotations

import unittest

import numpy as np

from potential import Grid, Potential
from studies._trajectory_accuracy import _validate_reference_dynamics, _validate_reference_provenance
from studies.historical_h5 import sample_study_grid
from studies.potentials import RandomPotentialConfig
from studies.reference_trajectory import _potential_grid_metadata, potential_fingerprint


class SpatialExtensionStudyTests(unittest.TestCase):
	def test_new_random_studies_default_to_cubic_reflection(self) -> None:
		config = RandomPotentialConfig(nx=8, ny=8, max_wave_number=2)
		potential = config.build()
		self.assertEqual(potential.spatial_extension, "reflected")
		self.assertEqual(potential.interpolation_order, 3)
		self.assertEqual(config.metadata()["spatial_extension"], "reflected")

	def test_identical_zero_fields_have_distinct_spatial_identities(self) -> None:
		grid = Grid.periodic(8, 8)
		periodic = Potential(grid, spatial_extension="periodic")
		reflected = Potential(grid, spatial_extension="reflected")
		np.testing.assert_array_equal(periodic.evaluate_grid(0.), reflected.evaluate_grid(0.))
		self.assertNotEqual(potential_fingerprint(periodic), potential_fingerprint(reflected))

	def test_untagged_reference_grid_is_only_compatible_with_periodic_physics(self) -> None:
		grid = Grid.periodic(8, 8)
		periodic = Potential(grid, spatial_extension="periodic")
		grid_metadata = _potential_grid_metadata(periodic)
		grid_metadata.pop("spatial_extension")
		grid_metadata.pop("spatial_period")
		metadata = {"potential_grid": grid_metadata,
			"dynamics_fingerprint_algorithm": "gc2d-sampled-potential-v1-sha256",
			"dynamics_fingerprint_sha256": potential_fingerprint(periodic)}
		_validate_reference_dynamics(periodic, metadata, 0.)
		with self.assertRaisesRegex(ValueError, "potential grid"):
			_validate_reference_dynamics(Potential(grid), metadata, 0.)

	def test_reflected_fingerprint_includes_independent_upper_samples(self) -> None:
		grid = Grid.periodic(8, 8)
		original = Potential(grid)
		closed = np.array(original.reflected_samples, copy=True)
		closed[0, -1, 4] = .01
		changed = Potential(grid, reflected_samples=closed)
		np.testing.assert_array_equal(original.mean, changed.mean)
		self.assertNotEqual(potential_fingerprint(original), potential_fingerprint(changed))

	def test_historical_resampling_refuses_reflected_reinterpretation(self) -> None:
		with self.assertRaisesRegex(ValueError, "direct-periodic"):
			sample_study_grid(Potential(Grid.periodic(8, 8)), 6, 6)

	def test_explicit_periodic_recipe_matches_untagged_reference_provenance(self) -> None:
		current = RandomPotentialConfig(spatial_extension="periodic").metadata()
		historical = dict(current)
		historical.pop("spatial_extension")
		metadata = {"potential": historical, "initial_conditions": {}}
		_validate_reference_provenance(metadata, potential_metadata=current,
			initial_condition_metadata={})
		with self.assertRaisesRegex(ValueError, "Potential metadata"):
			_validate_reference_provenance(metadata,
				potential_metadata={**current, "spatial_extension": "reflected"},
				initial_condition_metadata={})


if __name__ == "__main__":
	unittest.main()
