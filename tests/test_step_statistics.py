"""Optional histories must not change trajectories or weaken solve failures."""
from dataclasses import replace
import unittest
import pickle
import numpy as np
import jax

from dynamics.gc import GuidingCenterDynamics
from execution.execution import Execution
from execution.options import ExecutionOptions
from methods.palindromic.bm4_implicit import BM4Implicit
from methods.palindromic.bm4_midpoint import BM4Midpoint
from potential.potential import Potential
from problem import InitialValueProblem
from request import SimulationRequest
from solvers.nonlinear import NewtonSolver


class StepStatisticsTests(unittest.TestCase):
    def test_older_modal_options_restore_with_default_recording(self):
        old = object.__new__(ExecutionOptions)
        old.__setstate__(['jax', 'cpu', 0, False])
        self.assertEqual(old, ExecutionOptions(backend='jax'))
        lean = replace(old, record_step_statistics=False)
        self.assertEqual(pickle.loads(pickle.dumps(lean)), lean)

    def setUp(self):
        jax.config.update('jax_enable_x64', True)
        field = Potential.random(A=.05, M=3, nx=12, ny=12, seed=71)
        self.problem = InitialValueProblem(GuidingCenterDynamics(field, rho=.015),
            np.array([.4, .6, .5, .7]), layout=GuidingCenterDynamics.coordinates,
            request=SimulationRequest((0., .1), np.array([0., .025, .1])))

    def test_both_bm4_methods_keep_states_without_any_history_arrays(self):
        for backend in ('scipy', 'jax'):
            for method in (BM4Midpoint(step=.025), BM4Implicit(step=.025)):
                with self.subTest(backend=backend, method=type(method).__name__):
                    options = ExecutionOptions(backend=backend)
                    full = Execution().run(self.problem, method, options=options)
                    lean = Execution().run(self.problem, method,
                        options=replace(options, record_step_statistics=False))
                    np.testing.assert_array_equal(lean.states, full.states)
                    np.testing.assert_array_equal(lean.t, full.t)
                    self.assertEqual(lean.n_steps, full.n_steps)
                    self.assertEqual(lean.diagnostics['step_count'], 4)
                    self.assertFalse(any(isinstance(v, np.ndarray) for v in lean.diagnostics.values()))
                    self.assertIn('step_times', full.diagnostics)

    def test_disabling_history_does_not_hide_nonconvergence(self):
        method = BM4Implicit(step=.1, solver=NewtonSolver(
            absolute_tolerance=1e-30, relative_tolerance=1e-30, max_iterations=1))
        for backend in ('scipy', 'jax'):
            with self.subTest(backend=backend), self.assertRaises(RuntimeError):
                Execution().run(self.problem, method,
                    options=ExecutionOptions(backend=backend, record_step_statistics=False))


if __name__ == '__main__':
    unittest.main()
