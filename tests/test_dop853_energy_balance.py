"""Energy reconstruction, sampling and persistence tests for the new study."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from potential import Potential
from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from simulation import BM4Implicit, InitialValueProblem, SimulationRequest, simulate
from diagnostics import GCGeneralizedEnergyObserver
from diagnostics.bm4_energy_balance import BM4EnergyBalanceObserver
from diagnostics.dop853_energy_balance_npz import save_energy_balance, load_energy_balance
from studies.dop853_energy_balance import horizontal_configuration, run_energy_balance, METHODS
from visualization.dop853_energy_balance import summary_rows, trajectory_player


class EnergyBalanceTests(unittest.TestCase):
    def setUp(self):
        self.potential = Potential.random(A=.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5)
        self.configuration = horizontal_configuration(self.potential, [.25, .5, .75], .5)
        self.config = dict(cycles=1, cycle_duration=.02, steps_per_cycle=4, saves_per_cycle=2,
                           rho=.05, coupling_frequency=float(np.pi/8), newton_atol=1e-13,
                           newton_rtol=1e-12, newton_max_iterations=40,
                           jacobian_relative_step=float(np.cbrt(np.finfo(float).eps)),
                           reference_rtol=1e-10, reference_atol=1e-12)

    def test_bm4_batch_matches_independent_particle_energy(self):
        """Audit packed particle/stage slicing against existing scalar observers."""
        dynamics = GuidingCenterDynamics(self.potential, rho=.05)
        request = SimulationRequest.uniform(t_span=(0., .02), max_step=.005, sample_count=3)
        observer = BM4EnergyBalanceObserver(dynamics, 3, 2)
        simulate(InitialValueProblem(dynamics, self.configuration),
                 BM4Implicit(step_observer=observer), request)
        for p in range(3):
            z = self.configuration.initial_state[[p, p+3]]
            scalar = GCGeneralizedEnergyObserver(dynamics, initial_time=0., initial_state=z)
            simulate(InitialValueProblem(dynamics, GCInitialConfiguration(z)),
                     BM4Implicit(step_observer=scalar), request)
            np.testing.assert_allclose(np.asarray(observer.momenta)[:, p],
                                       [r.kappa for r in scalar.records][::2], atol=2e-14, rtol=1e-10)
        self.assertEqual(observer.step_count, 4)
        self.assertFalse(observer.block)
        self.assertTrue(np.all(np.asarray(observer.mu_max) >= np.asarray(observer.mu_endpoint)))

    def test_roundtrip_and_diagnostics(self):
        arrays = run_energy_balance(self.potential, self.configuration, self.config)
        for name in METHODS:
            np.testing.assert_allclose(arrays[name+'.balance'][:, 0], 0, atol=1e-15)
            np.testing.assert_allclose(arrays[name+'.states'][:, 0], self.configuration.initial_state)
        np.testing.assert_array_equal(arrays['DOP853.distance'], 0)
        self.assertLess(np.max(np.abs(arrays['DOP853.balance'])), 1e-9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'results.npz'
            save_energy_balance(path, arrays, {'schema_version':1, 'config':self.config})
            loaded, metadata = load_energy_balance(path)
            for key in arrays:
                np.testing.assert_array_equal(arrays[key], loaded[key])
        self.assertEqual(len(summary_rows(loaded)), 4)
        html = trajectory_player(loaded).data
        for name in METHODS:
            self.assertIn(name, html)
        self.assertIn('checkbox', html)
        self.assertIn('Play', html)

    def test_invalid_horizontal_geometry(self):
        with self.assertRaises(ValueError):
            horizontal_configuration(self.potential, [.25, .5, .8], .5)


if __name__ == '__main__':
    unittest.main()
