"""Cycle-time scaling, retained radial geometry and archive independence."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from dynamics.gc import GuidingCenterDynamics
from execution.execution import Execution
from potential import Grid, Potential
from studies.dimensional_h5_midpoint import DimensionalH5Field
from studies.poincare_radial_cycle_time import (
    RadialCycleConfig, build_radial_cycle, run_and_save_radial_cycle,
)
from studies.poincare_rho_sweep import RhoStarConfig, build_rho_star, run_rho_star
from visualization.poincare_radial_cycle_time import render_radial_cycle


class RadialCycleTests(unittest.TestCase):
    def setUp(self):
        n, length = 16, .18
        s = 2 * np.pi / .06
        grid = Grid(0., 0., length/n, length/n, n, n, length)
        xx, yy = np.meshgrid(np.arange(n)*2*np.pi/n, np.arange(n)*2*np.pi/n, indexing="ij")
        raw = Potential(grid, mean=1e-5*np.cos(xx)*np.cos(yy),
                        modes=np.asarray([1e-6*np.exp(1j*xx)]), frequencies=np.array([1.]),
                        interpolation_order=3)
        self.field = DimensionalH5Field(raw, raw, 1., 1.5, (15,))
        self.canonical = Potential(Grid(0., 0., grid.dx*s, grid.dy*s, n, n, length*s),
                                   mean=raw.mean*s*s/(2*np.pi), modes=raw.modes*s*s/(2*np.pi),
                                   frequencies=raw.frequencies, interpolation_order=3)
        self.config = RadialCycleConfig(tuple(np.linspace(0, .49, 48)), cycles=2)
        self.baseline = dict(
            run_id="original", particle_ids=list(range(1, 49)),
            colours={str(i): "#123456" for i in range(1, 49)},
            radial_fractions=list(self.config.radial_fractions), rho=.3,
            coupling_frequency=np.pi/8, radial_angle_rad=0.,
            field_provenance=dict(B_tesla=1.5, characteristic_length_m=.06,
                source_selection=[0, 1], source_field_indices=[15], interpolation_order=3, source_hdf5_sha256="synthetic",
                grid=dict(x0=0., y0=0., period=length*s), source_origin_m=[0., 0.],
                characteristic_period_s=1.),
        )
        self.prepared = build_radial_cycle(self.canonical, self.config, self.baseline,
                                           source_sha256="synthetic", baseline_sha256="baseline")

    def test_same_vector_field_and_cycle_returns_as_A(self):
        star = build_rho_star(self.field, RhoStarConfig(
            rho_hat=.3, particles=48, arms=1, outer_radius_fraction=.98,
            cycles=2, steps_per_cycle=20, coupling_frequency=np.pi/8,
            spatial_normalization="characteristic_length"), source_sha256="synthetic")
        p = self.prepared.problem
        np.testing.assert_allclose(p.initial_state, star.problem.initial_state, atol=1e-14)
        legacy = GuidingCenterDynamics(self.canonical, rho=.3)
        for t in (0., .173, .637):
            velocity = p.dynamics.vector_field(t, p.initial_state)
            np.testing.assert_allclose(velocity, 2*np.pi*legacy.vector_field(t, p.initial_state), atol=1e-14)
            np.testing.assert_allclose(velocity, star.problem.dynamics.vector_field(t, p.initial_state), atol=1e-14)
        with tempfile.TemporaryDirectory() as tmp:
            saved = run_and_save_radial_cycle(self.prepared, Path(tmp)/"result")
            reference = run_rho_star(star, executor=Execution(), options=ExecutionOptions())
            np.testing.assert_allclose(saved.solution.states[:, ::20], reference.solution.states, atol=1e-12)
            self.assertEqual(saved.solution.states.shape, (96, 41))
            with patch("studies.poincare_radial_cycle_time.simulate", side_effect=AssertionError("must reuse")):
                again = run_and_save_radial_cycle(self.prepared, Path(tmp)/"result")
            np.testing.assert_array_equal(again.solution.states, saved.solution.states)
            legacy_config = {**saved.metadata["config"], "source_selection": [0, 1]}
            legacy = replace(saved, metadata={**saved.metadata, "config": legacy_config})
            with patch("studies.poincare_radial_cycle_time.load_solution", return_value=legacy), \
                    patch("studies.poincare_radial_cycle_time.simulate", side_effect=AssertionError("must reuse")):
                self.assertIs(run_and_save_radial_cycle(self.prepared, Path(tmp)/"result"), legacy)
            self.assertEqual(legacy.metadata["config"]["source_selection"], [0, 1])
            changed = replace(self.prepared, metadata={**self.prepared.metadata, "baseline_metadata_sha256": "changed"})
            with self.assertRaisesRegex(ValueError, "different experiment"):
                run_and_save_radial_cycle(changed, Path(tmp)/"result")
            html = render_radial_cycle(again, Path(tmp)/"figures")
            self.assertTrue(html.is_file())
            self.assertTrue(html.with_name("diagnostics.png").is_file())

    def test_physical_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "Physical inputs"):
            build_radial_cycle(self.canonical, replace(self.config, rho=.4), self.baseline,
                               source_sha256="synthetic", baseline_sha256="baseline")


if __name__ == "__main__":
    unittest.main()
