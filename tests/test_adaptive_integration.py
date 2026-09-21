"""Adaptive methods share collection while preserving SciPy numerical sessions."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from diagnostics import AdaptiveTrajectoryObserver
from initial_conditions import GCInitialConfiguration
from simulation import (
    DOP853, Radau, ABBA2Implicit, ABBA2Midpoint, ABBA4Implicit, ABBA6Implicit,
    BM4Implicit, BM4Midpoint, ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
    InitialValueProblem, SimulationRequest, simulate,
)
from simulation.integration import IntegrationMethod, integrate_method
from tests.test_method_integration import _problem


class TimeDependentRotation:
    """Rotation with exact angle (t squared minus t0 squared) / 2."""

    state_dimension = 2

    def vector_field(self, time, state):
        x, y = np.split(state, 2)
        return time * np.concatenate((-y, x))


class AdaptiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.problem = InitialValueProblem(TimeDependentRotation(),
            GCInitialConfiguration.from_components(x=[1., .4, -.2], y=[0., .2, .5]))
        self.request = SimulationRequest.uniform(t_span=(.3, .8), max_step=.08, sample_count=31)

    def test_every_method_uses_the_same_public_integrate(self):
        for cls in (DOP853, Radau, ABBA2Implicit, ABBA2Midpoint, ABBA4Implicit,
                    ABBA6Implicit, BM4Implicit, BM4Midpoint, ExplicitEuler, RK4,
                    GaussLegendre4, SDIRK4, HBVM42):
            self.assertIs(cls.integrate, IntegrationMethod.integrate)

    def test_states_work_and_accepted_intervals_match_scipy(self):
        for cls in (DOP853, Radau):
            for count in (2, 31):
                with self.subTest(method=cls.__name__, outputs=count):
                    request = replace(self.request, output_times=np.linspace(.3, .8, count))
                    reference = solve_ivp(self.problem.dynamics.vector_field, request.t_span,
                        self.problem.initial_state, method=cls.__name__, t_eval=request.output_times,
                        max_step=request.max_step, rtol=1e-10, atol=1e-12)
                    events = []
                    plain = simulate(self.problem, cls(), request)
                    observed = simulate(self.problem, cls(step_observer=events.append), request)
                    np.testing.assert_array_equal(plain.states, reference.y)
                    np.testing.assert_array_equal(observed.states, plain.states)
                    for key, total in zip(('function_evaluations', 'jacobian_evaluations', 'lu_decompositions'),
                                          (reference.nfev, reference.njev, reference.nlu)):
                        self.assertEqual(int(np.sum(plain.diagnostics[key])), total)
                        np.testing.assert_array_equal(plain.diagnostics[key], observed.diagnostics[key])
                    self.assertEqual(len(events), plain.n_steps)
                    self.assertTrue(np.all(np.asarray(plain.diagnostics['step_sizes']) > 0))
                    self.assertTrue(np.all(np.asarray(plain.diagnostics['step_sizes']) <= .08 + 1e-16))
                    self.assertEqual(events[-1].time, .8)
                    self.assertFalse(hasattr(events[0], 'map_state'))
                    angle = (request.output_times**2 - .3**2) / 2
                    x, y = np.split(self.problem.initial_state, 2)
                    exact = np.vstack((x[:, None] * np.cos(angle) - y[:, None] * np.sin(angle),
                                       y[:, None] * np.cos(angle) + x[:, None] * np.sin(angle)))
                    np.testing.assert_allclose(plain.states, exact, rtol=0, atol=2e-11)

    def test_run_runs_are_fresh_and_snapshots_are_independent(self):
        for cls in (DOP853, Radau):
            events = []
            def mutate(event):
                time = (event.start_time + event.time) / 2
                expected = event.dense_state(time)
                event.state_before[:] = 999
                event.state_after[:] = -999
                event.dense_state(time)[:] = 999
                np.testing.assert_array_equal(event.dense_state(time), expected)
                events.append(event)
            run = cls(step_observer=mutate).new_run(self.problem, self.request)
            self.assertEqual(events, [])
            plain = simulate(self.problem, cls(), self.request)
            first = integrate_method(run)
            count = len(events)
            second = integrate_method(run.new_run(self.problem, self.request))
            self.assertEqual(len(events), 2 * count)
            np.testing.assert_array_equal(first.states, plain.states)
            np.testing.assert_array_equal(second.states, first.states)

    def test_dense_retention_matches_scipy_at_unscheduled_times(self):
        for cls in (DOP853, Radau):
            observer = AdaptiveTrajectoryObserver()
            configured = cls(dense_output=True, step_observer=observer)
            actual = simulate(self.problem, configured, self.request)
            reference = solve_ivp(self.problem.dynamics.vector_field, self.request.t_span,
                self.problem.initial_state, method=cls.__name__, max_step=.08,
                rtol=1e-10, atol=1e-12, dense_output=True)
            times = np.r_[.8, .3, .321, .644, .566, .799]
            np.testing.assert_array_equal(observer.evaluate(times), reference.sol(times))
            np.testing.assert_array_equal(observer.evaluate(.515), reference.sol(.515))
            self.assertEqual(np.sum(actual.diagnostics['function_evaluations']), reference.nfev)
            self.assertEqual(observer.evaluate(np.array([])).shape, (6, 0))
            with self.assertRaisesRegex(ValueError, 'within'):
                observer.evaluate(.9)

    def test_energy_is_passive_and_matches_an_independent_augmented_reference(self):
        problem = _problem()
        for cls in (DOP853, Radau):
            def derivative(t, z):
                return np.r_[problem.dynamics.vector_field(t, z[:2]),
                             problem.dynamics.extended_momentum_derivative(t, z[:2])]
            request = SimulationRequest.uniform(t_span=(.3, .34), max_step=.02, sample_count=5)
            reference = solve_ivp(derivative, request.t_span, np.r_[problem.initial_state, 0.],
                method='DOP853', rtol=1e-13, atol=1e-15, max_step=.001, t_eval=request.output_times)
            actual = simulate(problem, cls(track_energy=True), request)
            plain = simulate(problem, cls(), request)
            np.testing.assert_array_equal(actual.states, plain.states)
            for key in ('step_times', 'function_evaluations', 'jacobian_evaluations', 'lu_decompositions'):
                np.testing.assert_array_equal(actual.diagnostics[key], plain.diagnostics[key])
            np.testing.assert_allclose(actual.diagnostics['extended_momentum'], reference.y[2:], rtol=1e-9, atol=1e-13)
            np.testing.assert_array_equal(actual.diagnostics['extended_time'], actual.t[None, :])

    def test_advance_keeps_one_live_solver_and_independent_runs(self):
        for cls in (DOP853, Radau):
            first = cls(dense_output=True).new_run(self.problem, self.request)
            second = cls(dense_output=True).new_run(self.problem, self.request)
            self.assertIsNot(first.solver, second.solver)
            solver = first.solver
            reference = cls.solver_type(self.problem.dynamics.vector_field, .3,
                self.problem.initial_state, .8, max_step=.08, rtol=1e-10, atol=1e-12)
            for _ in range(2):
                start = float(solver.t)
                before = solver.y.copy()
                actual = first.advance(start, before, .08)
                reference.step()
                expected_dense = reference.dense_output()
                self.assertIs(first.solver, solver)
                self.assertEqual(solver.t, reference.t)
                np.testing.assert_array_equal(actual.state, reference.y)
                queries = np.linspace(start, reference.t, 7)
                np.testing.assert_array_equal(actual.details.dense_state(queries), expected_dense(queries))
            self.assertEqual(second.solver.t, .3)
            self.assertEqual(second.accepted_steps, 0)
            with self.assertRaisesRegex(ValueError, 'live solver state'):
                first.advance(.3, self.problem.initial_state, .08)
            with self.assertRaisesRegex(ValueError, 'finite and positive'):
                first.advance(float(solver.t), solver.y.copy(), -1.)

    def test_invalid_controls_and_failed_steps_are_reported(self):
        for cls in (DOP853, Radau):
            for options in ({'relative_tolerance': 0}, {'absolute_tolerance': np.nan}, {'first_step': -1}):
                with self.assertRaises(ValueError):
                    cls(**options)
            with self.assertRaises(TypeError):
                cls(track_energy=True).new_run(self.problem, self.request)
            def fail(solver):
                solver.status = 'failed'
                return 'injected failure'
            with patch.object(cls.solver_type, 'step', fail):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    simulate(self.problem, cls(), self.request)

    def test_first_step_and_optional_radau_jacobian(self):
        problem = InitialValueProblem(TimeDependentRotation(), GCInitialConfiguration(np.array([1., 0.])))
        method = Radau(first_step=.01, jacobian=lambda t, z: t * np.array([[0., -1.], [1., 0.]]))
        actual = simulate(problem, method, self.request)
        reference = solve_ivp(problem.dynamics.vector_field, self.request.t_span, problem.initial_state,
            method='Radau', rtol=1e-10, atol=1e-12, max_step=.08, first_step=.01,
            jac=method.jacobian, t_eval=self.request.output_times)
        np.testing.assert_array_equal(actual.states, reference.y)
        self.assertEqual(actual.diagnostics['first_step'], .01)


if __name__ == '__main__':
    unittest.main()
