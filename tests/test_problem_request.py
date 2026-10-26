"""Complete problem ownership and the two-input simulation boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import pickle
import unittest

import numpy as np

from dynamics.coordinates import GCCoordinates
from execution.execution import Execution
from execution._modal_worker import decode_job, encode_job, execute_payload
from methods.classical.rk4 import RK4
from problem import InitialValueProblem
from request import SimulationRequest
from simulation.runner import simulate
from states.initial import InitialConditions


class TimeDrivenDynamics:
	"""Integrate z' = t in both components to expose a wrong initial clock."""

	state_dimension = 2
	coordinates = GCCoordinates()

	def vector_field(self, time: float, state: np.ndarray) -> np.ndarray:
		return np.full_like(state, time)


class ProblemRequestTests(unittest.TestCase):
	"""Preserve physical inputs while composing and transporting temporal requests."""

	def setUp(self) -> None:
		self.initial = InitialConditions.gc(x=[1.0], y=[-2.0])
		self.dynamics = TimeDrivenDynamics()
		self.request = SimulationRequest.uniform(t_span=(2.0, 2.3), sample_count=5)
		self.problem = InitialValueProblem(self.dynamics, self.initial, request=self.request)

	def test_explicit_and_default_requests_keep_existing_constructors(self) -> None:
		self.assertIs(self.problem.request, self.request)
		self.assertIs(self.problem.initial_conditions, self.initial)
		first = InitialValueProblem(self.dynamics, self.initial)
		second = InitialValueProblem(self.dynamics, self.initial)
		self.assertEqual(first.request.t_span, (0.0, 1.0))
		np.testing.assert_array_equal(first.request.output_times, np.linspace(0.0, 1.0, 11))
		self.assertIsNot(first.request, second.request)
		self.assertFalse(first.request.output_times.flags.writeable)
		with self.assertRaisesRegex(TypeError, 'SimulationRequest'):
			InitialValueProblem(self.dynamics, self.initial, request='invalid')

	def test_with_request_reuses_physical_owners_without_mutation(self) -> None:
		other_request = SimulationRequest.uniform(t_span=(3.0, 3.4), sample_count=3)
		other = self.problem.with_request(other_request)
		self.assertIs(other.dynamics, self.problem.dynamics)
		self.assertIs(other.initial_conditions, self.problem.initial_conditions)
		self.assertIs(other.request, other_request)
		self.assertIs(self.problem.request, self.request)
		self.assertIs(self.problem.with_request(self.request), self.problem)
		with self.assertRaises(FrozenInstanceError):
			self.problem.request = other_request
		with self.assertRaisesRegex(TypeError, 'SimulationRequest'):
			self.problem.with_request(None)

	def test_simulation_and_executor_use_the_problem_clock_and_output_times(self) -> None:
		method = RK4(step=0.07)
		for problem in (self.problem, self.problem.with_request(
			SimulationRequest.uniform(t_span=(3.0, 3.4), sample_count=3),
		)):
			with self.subTest(span=problem.request.t_span):
				result = simulate(problem, method)
				direct = Execution().run(problem, method)
				times = problem.request.output_times
				expected = self.initial.values[:, None] + (times**2 - times[0]**2) / 2
				np.testing.assert_array_equal(result.t, times)
				np.testing.assert_allclose(result.states, expected, rtol=0, atol=3e-15)
				np.testing.assert_array_equal(result.states, direct.states)
		self.assertEqual(method.step, 0.07)

	def test_separate_simulation_request_argument_is_rejected(self) -> None:
		with self.assertRaises(TypeError):
			simulate(self.problem, RK4(), self.request)
		with self.assertRaises(TypeError):
			Execution().run(self.problem, RK4(), self.request)

	def test_worker_transports_one_composed_problem_and_rejects_old_payloads(self) -> None:
		method = RK4(step=0.07)
		payload = encode_job(self.problem, method, None)
		wire = pickle.loads(payload)
		self.assertEqual(wire[0], 3)
		self.assertEqual(len(wire), 5)
		problem, restored_method, options, observers = decode_job(payload)
		self.assertIsInstance(problem.request, SimulationRequest)
		self.assertEqual(problem.request.t_span, self.request.t_span)
		actual = execute_payload(payload)[2]
		expected = simulate(self.problem, method)
		np.testing.assert_array_equal(actual.t, expected.t)
		np.testing.assert_array_equal(actual.states, expected.states)
		with self.assertRaisesRegex(ValueError, 'version'):
			decode_job(pickle.dumps((2, self.problem, method, self.request, options, observers)))
