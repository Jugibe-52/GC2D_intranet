"""Protect peak sampling, time-grid semantics and complete HDF5 persistence."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from diagnostics.gc_energy_bound import load_energy_bound_result, save_energy_bound_result
from studies.gc_energy_bound import GCEnergyBoundConfig, GCEnergyBoundResult, envelope_statistics, time_rms


class GCEnergyBoundTests(unittest.TestCase):
    def test_envelope_keeps_interior_peak_and_signed_excursions(self) -> None:
        times = np.arange(9, dtype=float) / 4
        errors = np.array([0, 1, -9, 2, 1, 0, -3, 0, 2.])
        records = envelope_statistics(times, errors, (1., 2.))
        self.assertEqual([row["maximum"] for row in records], [9., 9.])
        self.assertEqual(records[-1]["growth_ratio"], 1.)
        self.assertIsNone(envelope_statistics(times, np.zeros(9), (1., 2.))[-1]["growth_ratio"])

    def test_off_grid_horizon_and_incompatible_refinement_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            envelope_statistics(np.arange(5.)/4, np.zeros(5), (.3,))
        with self.assertRaises(ValueError):
            GCEnergyBoundConfig(steps=(.11, .05, .025))
        with self.assertRaises(ValueError):
            GCEnergyBoundConfig(horizons=(1., 201.))
        with self.assertRaises(ValueError):
            GCEnergyBoundConfig(timing_repeats=1)

    def test_time_rms_and_hdf5_roundtrip_preserve_small_signals(self) -> None:
        times = np.array([0., .1, .2, .3])
        self.assertAlmostEqual(time_rms(np.full(4, -2.), times), 2.)
        arrays = {"reference/times": times, "reference/states": np.arange(8.).reshape(2, 4),
                  "h0/BM4Implicit/K_error": np.array([0., -1e-15, 2e-15, -3e-15])}
        result = GCEnergyBoundResult(arrays, {"config": {"steps": (.1,)}, "field_sha256": "test"},
                                     [{"method": "BM4Implicit", "step": .1}], [], [], [])
        with TemporaryDirectory() as directory:
            path = save_energy_bound_result(Path(directory)/"result.h5", result)
            restored = load_energy_bound_result(path)
            self.assertEqual(set(restored.arrays), set(arrays))
            for name, values in arrays.items():
                np.testing.assert_array_equal(restored.arrays[name], values)
            self.assertEqual(restored.metadata["field_sha256"], "test")
            self.assertEqual(restored.summary, result.summary)
            with self.assertRaises(FileExistsError):
                save_energy_bound_result(path, result)


if __name__ == "__main__":
    unittest.main()
