"""Ensure progress observation preserves adaptive integration and logs failures."""

from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from diagnostics.reference_progress import reference_progress_log
from studies.reference_trajectory import _solve_adaptive


class Oscillator:
    """Small independent ODE with a known smooth solution."""

    state_dimension = 2

    def vector_field(self, time, state):
        return np.array([state[1], -state[0]])


class ReferenceProgressTests(unittest.TestCase):
    def test_observation_preserves_both_solvers(self):
        for method in ("DOP853", "Radau"):
            with self.subTest(method=method):
                kwargs = dict(relative_tolerance=1e-10, absolute_tolerance=1e-12,
                              maximum_step=0.025, method=method)
                times = np.linspace(0, 0.2, 11)
                baseline, work = _solve_adaptive(Oscillator(), np.array([1., 0.]), times, **kwargs)
                events = []
                observed, observed_work = _solve_adaptive(
                    Oscillator(), np.array([1., 0.]), times,
                    progress_callback=lambda *args: events.append(args), **kwargs)
                np.testing.assert_array_equal(observed, baseline)
                self.assertEqual(work.function_evaluations, observed_work.function_evaluations)
                self.assertEqual(events[0], (method, 0., 0, "started"))
                self.assertEqual(events[-1][1], times[-1])
                self.assertEqual(events[-1][3], "finished")
                self.assertTrue(np.all(np.diff([event[1] for event in events]) > 0))

    def test_log_is_live_and_reports_interruption(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            path = Path(directory) / "progress.log"
            with self.assertRaises(KeyboardInterrupt):
                with reference_progress_log(path, t_span=(0., 200.)) as report:
                    report("Radau", 0., 0, "started")
                    report("Radau", 100., 10, "finished")
                    content = path.read_text()
                    self.assertIn("50.00%", content)
                    self.assertIn("remaining simulated time=100", content)
                    self.assertIn("ETA this solver", content)
                    raise KeyboardInterrupt()
            self.assertIn("interrupted or failed: KeyboardInterrupt", path.read_text())
            self.assertNotIn("persistence completed", path.read_text())


if __name__ == "__main__":
    unittest.main()
