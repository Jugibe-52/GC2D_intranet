"""Direct source selections and reconstruction of historical HDF5 studies."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from potential.potential import Potential
from studies.h5_provenance import prepare_verified_h5_field, restore_h5_selection
from studies.poincare_rho_sweep import RhoStarConfig, rho_config_from_metadata


class H5StudySelectionTests(unittest.TestCase):
	"""Keep old rank selectors out of the new source-index loading contract."""

	def setUp(self) -> None:
		self.directory = tempfile.TemporaryDirectory()
		self.path = Path(self.directory.name) / "field.h5"
		frequencies = np.zeros(16)
		frequencies[15] = 7.0
		fields = np.zeros((16, 8, 8), dtype=complex)
		fields[0] = np.eye(8)
		fields[15] = 0.3j * np.eye(8)
		with h5py.File(self.path, "w") as h5:
			h5["freqs"] = frequencies
			h5["fields"] = fields
			h5["Rcells"] = np.arange(8) * 0.01
			h5["Zcells"] = np.arange(8) * 0.01

	def tearDown(self) -> None:
		self.directory.cleanup()

	def test_verified_field_uses_original_indices_in_historical_provenance(self) -> None:
		"""The old (0, 1) rank label does not change the reconstructed field."""
		potential = Potential.load(self.path)
		metadata = potential.metadata
		with self.path.open("rb") as stream:
			digest = hashlib.file_digest(stream, "sha256").hexdigest()
		original = dict(
			B_tesla=1.5, characteristic_length_m=0.06,
			source_selection=[0, 1], source_field_indices=[15], interpolation_order=3,
			source_hdf5_sha256=digest, denoising=False, resampling=False,
			characteristic_period_s=metadata.characteristic_period,
			normalization_factor=metadata.normalization_factor,
			grid=asdict(potential.grid), source_origin_m=[0.0, 0.0],
		)
		provenance_path = self.path.with_suffix(".json")
		provenance_path.write_text(json.dumps(original))
		restored, _ = prepare_verified_h5_field(
			self.path, magnetic_field=1.5, characteristic_length=0.06,
			source_selection=(15,), interpolation_order=3,
			expected_source_sha256=digest, original_provenance=provenance_path,
		)
		np.testing.assert_array_equal(restored.mean, potential.mean)
		np.testing.assert_array_equal(restored.modes, potential.modes)
		np.testing.assert_array_equal(restored.frequencies, potential.frequencies)
		self.assertEqual(json.loads(provenance_path.read_text()), original)

	def test_saved_rho_config_recovers_recorded_source_indices(self) -> None:
		config = asdict(RhoStarConfig(rho_hat=0.3))
		self.assertEqual(config["source_selection"], (15,))
		config["source_selection"] = [0, 1]
		record = dict(config=config, source_field_indices=[15])
		self.assertEqual(rho_config_from_metadata(record).source_selection, (15,))
		self.assertEqual(config["source_selection"], [0, 1])
		with self.assertRaisesRegex(ValueError, "recorded source_field_indices"):
			rho_config_from_metadata(dict(config=config))

	def test_old_notebook_can_recover_a_unique_positive_mode(self) -> None:
		self.assertEqual(restore_h5_selection(self.path, {"mode_selection": [0, 1]}), (15,))
		self.assertEqual(restore_h5_selection(self.path, {"mode_selection": [15]}), (15,))
		self.assertEqual(restore_h5_selection(self.path, {"mode_selection": [0]}), ())
		with h5py.File(self.path, "r+") as h5:
			h5["freqs"][3] = 2.0
		with self.assertRaisesRegex(ValueError, "recorded source_field_indices"):
			restore_h5_selection(self.path, {"mode_selection": [0, 1]})
		self.assertEqual(restore_h5_selection(self.path, {
			"mode_selection": [0, 1], "source_field_indices": [15],
		}), (15,))


if __name__ == "__main__":
	unittest.main()
