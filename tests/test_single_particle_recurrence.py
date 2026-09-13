"""Independent analytical checks for recurrence detection and cache identity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from diagnostics.single_particle_recurrence import load_recurrence, save_recurrence
from studies.single_particle_recurrence import (
    SingleParticleRecurrenceConfig, distance_to_initial, locate_returns,
    minimum_image, threshold_windows,
)


class Circle:
    """Exact unit-radius circle with period one and initial state (0, 0)."""

    @staticmethod
    def state(t):
        return np.asarray([np.sin(2 * np.pi * t), 1 - np.cos(2 * np.pi * t)])

    @staticmethod
    def vector_field(t, state):
        return 2 * np.pi * np.asarray([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])


class RecurrenceTests(unittest.TestCase):
    def test_exact_returns_between_saved_times(self):
        times = np.linspace(0.2, 2.2, 68)
        roots = locate_returns(Circle.state, Circle(), np.zeros(2), 10., times, root_xtol=1e-12)
        np.testing.assert_allclose(roots, [1., 2.], rtol=0, atol=2e-12)
        np.testing.assert_allclose(distance_to_initial(Circle.state(roots), np.zeros(2), 10),
                                   0, rtol=0, atol=2e-11)

    def test_narrow_threshold_visits_and_censoring(self):
        times = np.linspace(0.2, 2.2, 68)
        radius = 1e-4
        windows = threshold_windows(Circle.state, np.zeros(2), 10., times,
                                    np.array([1., 2.]), radius, root_xtol=1e-12)
        width = np.arcsin(radius / 2) / np.pi
        self.assertEqual(len(windows), 2)
        for n, window in enumerate(windows, 1):
            self.assertAlmostEqual(window['entry'], n - width, delta=2e-12)
            self.assertAlmostEqual(window['exit'], n + width, delta=2e-12)
            self.assertFalse(window['entry_censored'])
        clipped = threshold_windows(Circle.state, np.zeros(2), 10., np.array([1., 1.1]),
                                    np.array([1.]), radius, root_xtol=1e-12)
        self.assertTrue(clipped[0]['entry_censored'])

    def test_periodic_distance_across_cell_edge(self):
        np.testing.assert_allclose(minimum_image(np.array([9.9, -9.9]), 10), [-.1, .1])
        self.assertAlmostEqual(float(distance_to_initial(np.array([.05, 2.]),
                                                         np.array([9.95, 2.]), 10)), .1)

    def test_invalid_controls(self):
        for kwargs in ({'final_time': .5}, {'samples_per_cycle': 0},
                       {'search_start': 50}, {'refined_rtol': 1e-8}):
            with self.assertRaises(ValueError):
                SingleParticleRecurrenceConfig(**kwargs)

    def test_cache_rejects_changed_physics(self):
        arrays = {'times': np.array([0., 1.]), 'initial_state': np.zeros(2)}
        for label in ('DOP853', 'DOP853_refined', 'Radau'):
            arrays[label + '.states'] = np.zeros((2, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'result.npz'
            save_recurrence(path, arrays, {'description': 'test'}, {'rho': .3})
            loaded, _ = load_recurrence(path, {'rho': .3})
            np.testing.assert_array_equal(loaded['times'], arrays['times'])
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                load_recurrence(path, {'rho': .4})


if __name__ == '__main__':
    unittest.main()
