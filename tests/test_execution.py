"""Complete-job execution, subclass dispatch and result validation contracts."""

import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from contracts.result import IntegrationData
from execution import Execution
from execution.execution import Execution as CanonicalExecution
from initial_conditions.gc import GCInitialConfiguration
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

    def run(self, problem, method, request, *, options=None):
        self.jobs.append((problem, method, request, options))
        return self.data


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.problem = InitialValueProblem(
            RotationDynamics(),
            GCInitialConfiguration.from_components(x=np.array([1.]), y=np.array([0.])),
        )
        self.method = RK4()
        self.request = SimulationRequest.uniform(
            t_span=(0., .2), max_step=.01, sample_count=3,
        )

    def test_local_executor_returns_complete_integration_and_can_be_reused(self):
        self.assertIs(Execution, CanonicalExecution)
        executor = Execution()
        first = executor.run(self.problem, self.method, self.request)
        second = executor.run(self.problem, self.method, self.request)
        self.assertIsInstance(first, IntegrationData)
        np.testing.assert_array_equal(first.t, self.request.output_times)
        expected = np.array([np.cos(first.t), -np.sin(first.t)])
        np.testing.assert_allclose(first.states, expected, rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(first.states, second.states)
        self.assertIsNot(first.states, second.states)
        self.assertEqual(self.method._status, "configuration")

    def test_default_and_explicit_executor_produce_the_same_solution(self):
        default = simulate(self.problem, self.method, self.request)
        explicit = simulate(self.problem, self.method, self.request, execution=Execution())
        np.testing.assert_array_equal(default.states, explicit.states)
        self.assertFalse(explicit.states.flags.writeable)

    def test_simulation_dispatches_one_complete_job_to_the_subclass(self):
        data = self.method.integrate(self.problem, self.request)
        executor = ReturnedExecution(data)
        options = ExecutionOptions(backend="jax")
        with patch.object(RK4, "integrate", side_effect=AssertionError("Unexpected local calculation")):
            solution = simulate(self.problem, self.method, self.request,
                                execution=executor, options=options)
        self.assertEqual(len(executor.jobs), 1)
        for received, expected in zip(executor.jobs[0],
                                      (self.problem, self.method, self.request, options)):
            self.assertIs(received, expected)
        np.testing.assert_array_equal(solution.states, data.states)
        data.states[0, -1] = 99.
        self.assertNotEqual(solution.states[0, -1], 99.)

    def test_subclass_results_still_undergo_simulation_validation(self):
        data = self.method.integrate(self.problem, self.request)
        bad_initial = data.states.copy()
        bad_initial[0, 0] += 1.
        nonfinite = data.states.copy()
        nonfinite[0, -1] = np.nan
        cases = (
            IntegrationData(data.t + 1., data.states, {}),
            IntegrationData(data.t, bad_initial, {}),
            IntegrationData(data.t, nonfinite, {}),
        )
        for invalid in cases:
            with self.subTest(data=invalid), self.assertRaises(ValueError):
                simulate(self.problem, self.method, self.request,
                         execution=ReturnedExecution(invalid))

    def test_local_executor_passes_explicit_backend_options_to_the_method(self):
        options = ExecutionOptions(backend="jax")
        data = self.method.integrate(self.problem, self.request)
        with patch.object(RK4, "integrate", return_value=data) as integrate:
            result = Execution().run(self.problem, self.method, self.request, options=options)
        integrate.assert_called_once_with(self.problem, self.request, execution=options)
        self.assertIs(result, data)

    def test_external_cpu_method_keeps_its_two_argument_interface(self):
        data = self.method.integrate(self.problem, self.request)

        class ExternalMethod:
            def integrate(self, problem, request):
                return data

        for options in (None, ExecutionOptions()):
            with self.subTest(options=options):
                result = simulate(self.problem, ExternalMethod(), self.request,
                                  execution=Execution(), options=options)
                np.testing.assert_array_equal(result.states, data.states)

    def test_wrong_executor_and_options_are_rejected_before_local_calculation(self):
        with patch.object(RK4, "integrate", side_effect=AssertionError("Unexpected calculation")):
            with self.assertRaisesRegex(TypeError, "Execution instance"):
                simulate(self.problem, self.method, self.request, execution=ExecutionOptions())
            with self.assertRaisesRegex(TypeError, "ExecutionOptions instance"):
                simulate(self.problem, self.method, self.request, options=Execution())

    def test_executor_failure_propagates_without_local_fallback(self):
        class FailedExecution(Execution):
            def run(self, problem, method, request, *, options=None):
                raise RuntimeError("Execution failed")

        with patch.object(RK4, "integrate", side_effect=AssertionError("Unexpected fallback")):
            with self.assertRaisesRegex(RuntimeError, "Execution failed"):
                simulate(self.problem, self.method, self.request, execution=FailedExecution())


if __name__ == "__main__":
    unittest.main()
