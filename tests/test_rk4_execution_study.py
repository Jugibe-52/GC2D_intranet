"""Persistence and measured-record contracts for the RK4 execution notebooks."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from studies.rk4_execution import (
    RK4ExecutionConfig, run_rk4_execution_comparison,
    save_rk4_execution_comparison, load_rk4_execution_comparison,
)


@unittest.skipUnless(importlib.util.find_spec("jax"), "Optional JAX is not installed")
class RK4ExecutionStudyTests(unittest.TestCase):
    def test_equivalence_timings_and_archive_roundtrip_without_reintegration(self):
        import jax
        previous = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        config = RK4ExecutionConfig(
            amplitude=.1, maximum_wave_number=3, nx=16, ny=16, potential_seed=27,
            interpolation_order=3, particle_count=8, particle_seed=2026, rho=.3,
            t_span=(.1, .14), max_step=.01, sample_count=7, track_energy=True,
            repetitions=3, equivalence_rtol=1e-10, equivalence_atol=1e-11,
        )
        executions = (ExecutionOptions(), ExecutionOptions(backend="jax"))
        try:
            with self.assertRaisesRegex(ValueError, "three"):
                run_rk4_execution_comparison(replace(config, repetitions=2), executions=executions)
            comparison = run_rk4_execution_comparison(config, executions=executions)
            self.assertEqual(comparison.metadata["timing_orders"][0], ["scipy_cpu_0", "jax_cpu_0"])
            self.assertEqual(comparison.metadata["timing_orders"][1], ["jax_cpu_0", "scipy_cpu_0"])
            for record in comparison.metadata["runs"].values():
                self.assertEqual(len(record["runtime_seconds"]), 3)
                self.assertEqual(record["median_seconds"], np.median(record["runtime_seconds"]))
                self.assertLess(record["maximum_periodic_discrepancy"], 1e-10)
            with tempfile.TemporaryDirectory() as root:
                destination = Path(root) / "comparison"
                save_rk4_execution_comparison(comparison, destination)
                with patch("studies.rk4_execution.simulate", side_effect=AssertionError("Unexpected integration")):
                    loaded = load_rk4_execution_comparison(destination)
                for key, solution in comparison.solutions.items():
                    np.testing.assert_array_equal(solution.states, loaded.solutions[key].states)
                self.assertEqual(loaded.metadata["runs"], comparison.metadata["runs"])
                with self.assertRaises(FileExistsError):
                    save_rk4_execution_comparison(comparison, destination)
        finally:
            jax.config.update("jax_enable_x64", previous)


if __name__ == "__main__":
    unittest.main()
