"""Complete-job execution, subclass dispatch and result validation contracts."""


from dynamics.gc import GuidingCenterDynamics

import unittest
from unittest.mock import patch

import numpy as np

from execution.options import ExecutionOptions
from problem import InitialValueProblem
from request import SimulationRequest
from states.history import States
from execution import Execution
from execution.execution import Execution as CanonicalExecution

from methods.classical.rk4 import RK4
from simulation.runner import simulate


class RotationDynamics:
    """Unit angular frequency, using component-major planar coordinates."""

    state_dimension = 2

    def vector_field(self, time, state):
        x, y = np.split(state, 2)
        return np.concatenate((y, -x))


class ReturnedExecution(Execution):
    """Represent an executor that obtains its result outside the local method."""

    def __init__(self, data):
        self.data = data
        self.jobs = []

    def run(self, problem, method, *, options=None, observers=()):
        request = problem.request
        self.jobs.append((problem, method, request, options))
        return self.data


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.problem = InitialValueProblem(RotationDynamics(), GuidingCenterDynamics.coordinates.pack_components(np.array([1.]), np.array([0.])), layout=GuidingCenterDynamics.coordinates)
        self.method = RK4(step=.01)
        self.request = SimulationRequest.uniform(
            t_span=(0., .2),  sample_count=3,
        )
        self.problem = self.problem.with_request(self.request)

    def test_local_executor_returns_complete_integration_and_can_be_reused(self):
        self.assertIs(Execution, CanonicalExecution)
        executor = Execution()
        first = executor.run(self.problem.with_request(self.request), self.method)
        second = executor.run(self.problem.with_request(self.request), self.method)
        self.assertIsInstance(first, States)
        np.testing.assert_array_equal(first.t, self.request.output_times)
        expected = np.array([np.cos(first.t), -np.sin(first.t)])
        np.testing.assert_allclose(first.states.T, expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(first.states, second.states)
        self.assertIsNot(first.states, second.states)
        self.assertEqual(self.method._status, "configuration")

    def test_default_and_explicit_executor_produce_the_same_solution(self):
        default = simulate(self.problem.with_request(self.request), self.method)
        explicit = simulate(self.problem.with_request(self.request), self.method, execution=Execution())
        np.testing.assert_array_equal(default.states, explicit.states)
        self.assertFalse(explicit.states.flags.writeable)

    def test_simulation_dispatches_one_complete_job_to_the_subclass(self):
        data = Execution().run(self.problem.with_request(self.request), self.method)
        executor = ReturnedExecution(data)
        options = ExecutionOptions(backend="jax")
        with patch.object(RK4, "new_run", side_effect=AssertionError("Unexpected local calculation")):
            solution = simulate(self.problem.with_request(self.request), self.method,
                                execution=executor, options=options)
        self.assertEqual(len(executor.jobs), 1)
        for received, expected in zip(executor.jobs[0],
                                      (self.problem, self.method, self.request, options)):
            self.assertIs(received, expected)
        np.testing.assert_array_equal(solution.states, data.states)
        self.assertIs(solution, data)
        with self.assertRaises(ValueError):
            data.states[-1, 0] = 99.

    def test_subclass_results_still_undergo_simulation_validation(self):
        data = Execution().run(self.problem.with_request(self.request), self.method)
        bad_initial = data.states.T.copy()
        bad_initial[0, 0] += 1.
        nonfinite = data.states.T.copy()
        nonfinite[0, -1] = np.nan
        for times, values in ((data.t+1., data.states.T), (data.t, bad_initial), (data.t, nonfinite)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                invalid = States(times, values.T, layout=self.problem.layout, initial_state=self.problem.initial_state)
                simulate(self.problem.with_request(self.request), self.method, execution=ReturnedExecution(invalid))

    def test_executor_cannot_relabel_a_valid_physical_history(self):
        from dynamics.fc import FullCyclotronDynamics
        problem = InitialValueProblem(RotationDynamics(), np.array([1., 0., 2., 0.]), layout=GuidingCenterDynamics.coordinates)
        # Two GC particles and one FC particle share four packed coordinates.
        values = np.repeat(problem.initial_state[:, None], self.request.output_times.size, axis=1)
        for layout, initial in ((FullCyclotronDynamics.coordinates, problem.initial_state),
                                (GuidingCenterDynamics.coordinates, None),
                                (GuidingCenterDynamics.coordinates, problem.initial_state + 1)):
            invalid = States(self.request.output_times, values.T, layout=layout, initial_state=initial)
            with self.subTest(layout=layout, initial=initial), self.assertRaises(ValueError):
                simulate(problem.with_request(self.request), self.method, execution=ReturnedExecution(invalid))

    def test_local_executor_owns_preparation_and_drives_the_controller(self):
        from execution._host import FixedStepController
        run = self.method.new_run(self.problem.dynamics.vector_field, self.problem.initial_conditions, self.request)
        with patch.object(RK4, "new_run", return_value=run) as prepare:
            with patch.object(run, "controller", wraps=run.controller) as controller:
                result = Execution().run(self.problem.with_request(self.request), self.method)
        prepare.assert_called_once_with(self.problem.dynamics.vector_field, self.problem.initial_conditions, self.request,
                                        jacobian=None, jacobian_structure="dense", observe=False, capture_quadrature=False)
        controller.assert_called_once_with()
        self.assertEqual(run._status, "finished")
        self.assertEqual(result.diagnostics["execution_backend"], "scipy")
        np.testing.assert_allclose(result.values[-1], [np.cos(.2), -np.sin(.2)], atol=1e-10)

    def test_external_provider_prepares_a_run_without_an_integration_entry_point(self):
        calls = []

        class ExternalMethod:

            def new_run(self, f, initial_conditions, request, **kwargs):
                calls.append((f, initial_conditions, request))
                return RK4(step=.01).new_run(f, initial_conditions, request, **kwargs)

        for options in (None, ExecutionOptions()):
            with self.subTest(options=options):
                result = simulate(self.problem.with_request(self.request), ExternalMethod(), options=options)
                np.testing.assert_allclose(result.values[-1], [np.cos(.2), -np.sin(.2)], atol=1e-10)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], self.problem.dynamics.vector_field)

    def test_retired_integrate_only_provider_is_rejected(self):
        class RetiredMethod:
            def integrate(self, problem, request):
                raise AssertionError("Retired entry point must not be used")

        with self.assertRaisesRegex(TypeError, "NumericalMethod"):
            simulate(self.problem.with_request(self.request), RetiredMethod())

    def test_controller_preparation_failure_consumes_only_the_fresh_run(self):
        run = self.method.new_run(self.problem.dynamics.vector_field, self.problem.initial_conditions, self.request)
        with patch.object(RK4, "new_run", return_value=run):
            with patch.object(run, "controller", side_effect=RuntimeError("Controller unavailable")):
                with self.assertRaisesRegex(RuntimeError, "Controller unavailable"):
                    Execution().run(self.problem.with_request(self.request), self.method)
        self.assertEqual(run._status, "finished")
        self.assertEqual(self.method._status, "configuration")
        self.assertTrue(Execution().run(self.problem.with_request(self.request), self.method).complete)

    def test_result_preparation_and_finalization_failures_consume_only_the_run(self):
        for operation in ("prepare", "finalize"):
            run = self.method.new_run(self.problem.dynamics.vector_field, self.problem.initial_conditions, self.request)
            with self.subTest(operation=operation), patch.object(RK4, "new_run", return_value=run):
                with patch.object(States, operation, side_effect=RuntimeError("Result failure")):
                    with self.assertRaisesRegex(RuntimeError, "Result failure"):
                        Execution().run(self.problem.with_request(self.request), self.method)
                self.assertEqual(run._status, "finished")
                with self.assertRaisesRegex(TypeError, "fresh"):
                    Execution().run(self.problem.with_request(self.request), self.method)
            self.assertEqual(self.method._status, "configuration")
        self.assertTrue(Execution().run(self.problem.with_request(self.request), self.method).complete)

    def test_host_result_is_finalized_once_while_the_run_is_active(self):
        run = self.method.new_run(self.problem.dynamics.vector_field, self.problem.initial_conditions, self.request)
        finalize = States.finalize
        finished = []
        def finish(result, *args, **kwargs):
            self.assertEqual(run._status, "running")
            self.assertFalse(result.complete)
            finished.append(result)
            return finalize(result, *args, **kwargs)
        with patch.object(RK4, "new_run", return_value=run), patch.object(States, "finalize", finish):
            result = Execution().run(self.problem.with_request(self.request), self.method)
        self.assertEqual(len(finished), 1)
        self.assertIs(finished[0], result)
        self.assertEqual(run._status, "finished")

    def test_wrong_executor_and_options_are_rejected_before_local_calculation(self):
        with patch.object(RK4, "new_run", side_effect=AssertionError("Unexpected calculation")):
            with self.assertRaisesRegex(TypeError, "Execution instance"):
                simulate(self.problem.with_request(self.request), self.method, execution=ExecutionOptions())
            with self.assertRaisesRegex(TypeError, "ExecutionOptions instance"):
                simulate(self.problem.with_request(self.request), self.method, options=Execution())

    def test_executor_failure_propagates_without_local_fallback(self):
        class FailedExecution(Execution):
            def run(self, problem, method, *, options=None, observers=()):
                request = problem.request
                raise RuntimeError("Execution failed")

        with patch.object(RK4, "new_run", side_effect=AssertionError("Unexpected fallback")):
            with self.assertRaisesRegex(RuntimeError, "Execution failed"):
                simulate(self.problem.with_request(self.request), self.method, execution=FailedExecution())


if __name__ == "__main__":
    unittest.main()
