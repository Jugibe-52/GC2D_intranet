"""Explicit Poincare seeds preserve physical inputs and independent archives."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from diagnostics.persistence import load_solution
from execution.execution import Execution
from studies.poincare_gap_probes import (
    GapProbeSeeds, build_gap_probes, validate_gap_solution, validate_saved_gap_probes,
)
from studies.poincare_rho_sweep import (
    RhoStarConfig, build_rho_star, run_and_save_rho_star, run_rho_star,
)
from test_poincare_rho_sweep import field


def seeds():
    """Two deterministic probes in each of four cell regions."""
    return GapProbeSeeds(
        fractions=((.43, .58), (.44, .57), (.59, .58), (.60, .57),
                   (.43, .39), (.44, .38), (.57, .39), (.58, .38)),
        particle_ids=tuple(range(41, 49)),
        gap_names=("NW", "NW", "NE", "NE", "SW", "SW", "SE", "SE"),
    )


class GapProbeTests(unittest.TestCase):
    def test_geometry_is_explicit_and_scaled_with_unchanged_physics(self):
        config = RhoStarConfig(rho_hat=.3, particles=8, arms=8, cycles=1,
                              steps_per_cycle=4, hamiltonian_convention="radial")
        for normalization in ("none", "characteristic_length"):
            with self.subTest(normalization=normalization):
                config = replace(config, spatial_normalization=normalization)
                prepared = build_gap_probes(field(), config, source_sha256="synthetic", seeds=seeds())
                background = build_rho_star(field(), replace(config, particles=40), source_sha256="synthetic")
                xy = -.09 + .18 * np.asarray(seeds().fractions)
                np.testing.assert_array_equal(prepared.metadata["initial_positions_m"], xy)
                if normalization == "characteristic_length":
                    xy = (xy + .09) * (2*np.pi/.06)
                np.testing.assert_array_equal(prepared.problem.initial_state, xy.T.ravel())
                np.testing.assert_array_equal(prepared.request.output_times, background.request.output_times)
                np.testing.assert_array_equal(prepared.problem.dynamics.effective_potential.mean,
                                              background.problem.dynamics.effective_potential.mean)
                self.assertEqual(prepared.metadata["study"], "poincare_gap_probes")
                for key in ("arm_id", "star_center", "initial_radii", "initial_radius_fraction"):
                    self.assertNotIn(key, prepared.metadata)
                self.assertEqual(prepared.metadata["probe_config"], seeds().to_metadata())

    def test_saved_probes_match_background_and_reuse_requires_exact_seed_contract(self):
        config = RhoStarConfig(rho_hat=.3, particles=8, arms=8, cycles=1,
                              steps_per_cycle=4, hamiltonian_convention="radial", method="RK4")
        prepared = build_gap_probes(field(), config, source_sha256="synthetic", seeds=seeds())
        base_prepared = build_rho_star(field(), replace(config, particles=40), source_sha256="synthetic")
        background = run_rho_star(base_prepared, executor=Execution(), options=ExecutionOptions())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probes"
            stored = run_and_save_rho_star(prepared, path, executor=Execution(),
                                          options=ExecutionOptions(), save_folded_returns=True)
            loaded = load_solution(path)
            validate_gap_solution(loaded.solution, prepared)
            self.assertEqual(validate_saved_gap_probes(loaded, background=background), seeds())
            self.assertEqual(loaded.solution.states.shape, (16, 2))
            np.testing.assert_array_equal(stored.solution.states, loaded.solution.states)
            with patch("studies.poincare_rho_sweep.run_rho_star", side_effect=AssertionError("must not run")):
                again = run_and_save_rho_star(prepared, path, executor=Execution(), options=ExecutionOptions())
                np.testing.assert_array_equal(again.solution.states, loaded.solution.states)
                # Labels are provenance too: equal positions do not authorize
                # reuse after changing the selected-region interpretation.
                changed = replace(seeds(), gap_names=("different",) + seeds().gap_names[1:])
                different = build_gap_probes(field(), config, source_sha256="synthetic", seeds=changed)
                with self.assertRaisesRegex(ValueError, "different scientific inputs"):
                    run_and_save_rho_star(different, path, executor=Execution(), options=ExecutionOptions())
            loaded.metadata["initial_positions"][0][0] += 1e-3
            with self.assertRaisesRegex(ValueError, "initial_positions"):
                validate_saved_gap_probes(loaded, background=background)

    def test_saved_background_mismatches_and_id_collisions_are_rejected(self):
        config = RhoStarConfig(rho_hat=.3, particles=8, arms=8, cycles=1, steps_per_cycle=2, method="RK4")
        prepared = build_gap_probes(field(), config, source_sha256="synthetic", seeds=seeds())
        stored = run_rho_star(prepared, executor=Execution(), options=ExecutionOptions())
        background = run_rho_star(build_rho_star(field(), replace(config, particles=40), source_sha256="synthetic"),
                                 executor=Execution(), options=ExecutionOptions())
        for name, changed in (("source_sha256", "wrong"), ("time_unit_seconds", 2.)):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                wrong = replace(background, metadata={**background.metadata, name: changed})
                validate_saved_gap_probes(stored, background=wrong)
        with self.assertRaisesRegex(ValueError, "different scientific inputs"):
            wrong = replace(background, metadata={**background.metadata,
                "config": {**background.metadata["config"], "rho_hat": .2}})
            validate_saved_gap_probes(stored, background=wrong)
        with self.assertRaisesRegex(ValueError, "overlap"):
            wrong = replace(background, metadata={**background.metadata, "particle_id": [41]})
            validate_saved_gap_probes(stored, background=wrong)

    def test_seed_inputs_are_immutable_validated_and_counted(self):
        original = seeds()
        self.assertEqual(GapProbeSeeds(**original.to_metadata()), original)
        for changes in ({"fractions": ((1., .5), (.2, .3))},
                        {"fractions": ((.2, .3), (.2, .3))},
                        {"particle_ids": (41,) * 8},
                        {"particle_ids": (True,) + original.particle_ids[1:]},
                        {"gap_names": ("",) + original.gap_names[1:]},
                        {"selection_rho": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(original, **changes)
        with self.assertRaisesRegex(ValueError, "probe count"):
            build_gap_probes(field(), RhoStarConfig(rho_hat=.3), source_sha256="synthetic", seeds=original)


if __name__ == "__main__":
    unittest.main()
