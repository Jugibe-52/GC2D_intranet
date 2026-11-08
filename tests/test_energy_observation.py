"""Endpoint-only H and analytic partial_t H, independent of numerical stages."""

import unittest
from unittest.mock import patch

import numpy as np

from contracts.methods.step import StepInfo
from diagnostics.methods.energy import EnergyObserver
from dynamics.coordinates import GCCoordinates
from methods import RK4, ABBA4Implicit
from problem import InitialValueProblem
from request import SimulationRequest
from simulation.runner import simulate
from tests.test_method_integration import _problem


class CubicTimeEnergy:
	"""A stationary physical state with exactly known nonautonomous energy."""
	state_dimension = 2
	coordinates = GCCoordinates()

	def __init__(self):
		self.calls = []

	def vector_field(self, t, state):
		return np.zeros_like(state)

	def particle_vector_field_jacobians(self, t, state):
		return np.zeros((state.size // 2, 2, 2))

	def hamiltonian(self, t, state):
		self.calls.append(('H', np.asarray(t).copy()))
		return np.ones_like(state[:state.shape[0] // 2]) * np.asarray(t)**3

	def hamiltonian_time_derivative(self, t, state):
		self.calls.append(('rate', np.asarray(t).copy()))
		return np.ones_like(state[:state.shape[0] // 2]) * 3*np.asarray(t)**2


class EnergyObservationTests(unittest.TestCase):
	def test_only_accepted_endpoints_and_requested_samples_are_evaluated(self):
		for cls in (RK4, ABBA4Implicit):
			dynamics = CubicTimeEnergy()
			problem = InitialValueProblem(dynamics, np.array([.2, .4, .3, .5]),
				request=SimulationRequest((.2, .5), np.array([.2, .23, .5])))
			# Signed composition stages extend outside the interval; energy must not visit them.
			result = simulate(problem, cls(step=.1), observers=(EnergyObserver(),))
			for name in ('H', 'rate'):
				scalar_times = [float(t) for kind, t in dynamics.calls if kind == name and t.ndim == 0]
				np.testing.assert_allclose(scalar_times, [.2, .3, .4, .5], rtol=0., atol=1e-15)
				arrays = [t for kind, t in dynamics.calls if kind == name and t.ndim == 1]
				self.assertEqual(len(arrays), 1)
				np.testing.assert_array_equal(arrays[0], result.t)
			d = result.diagnostics
			np.testing.assert_allclose(d['hamiltonian_time_derivative'], np.broadcast_to(3*result.t**2, (2, 3)))
			np.testing.assert_allclose(d['energy_time_derivative_start'], np.broadcast_to(3*np.array([.2,.3,.4])[:,None]**2,(3,2)))
			np.testing.assert_allclose(d['energy_change'], np.broadcast_to(np.diff(np.array([.2,.3,.4,.5])**3)[:,None], (3,2)))
			self.assertNotIn('extended_momentum', d)
			self.assertNotIn('energy_error', d)
			self.assertNotIn('generalized_energy', d)
			# A secant of H would not equal either endpoint derivative for cubic time dependence.
			self.assertFalse(np.allclose(d['energy_change'][0] / .1, d['energy_time_derivative_end'][0]))

	def test_configuration_has_no_momentum_and_preparation_has_no_physical_work(self):
		configuration = EnergyObserver()
		dynamics = CubicTimeEnergy()
		first = configuration.new_run(dynamics.hamiltonian, dynamics.hamiltonian_time_derivative, particle_count=1)
		second = configuration.new_run(dynamics.hamiltonian, dynamics.hamiltonian_time_derivative, particle_count=1)
		self.assertEqual(dynamics.calls, [])
		info = StepInfo(0, .2, .1, .3, np.array([1., 2.]))
		first.accept(info, info.state_before)
		self.assertIsNone(second._last)
		self.assertFalse(hasattr(first, 'momentum'))
		with self.assertRaises(TypeError):
			EnergyObserver(initial_momentum=np.ones(1))

	def test_bad_energy_shape_or_nonfinite_derivative_is_rejected(self):
		for value in (np.array([np.nan]), np.ones(2), np.array([[1.]])):
			for bad_h in (False, True):
				h = (lambda t, z: value) if bad_h else (lambda t, z: np.ones(1))
				rate = (lambda t, z: np.ones(1)) if bad_h else (lambda t, z: value)
				run = EnergyObserver().new_run(h, rate, particle_count=1)
				with self.assertRaises(ValueError):
					run.accept(StepInfo(0, 0., .1, .1, np.zeros(2)), np.zeros(2))
				self.assertIsNone(run._last)

	def test_sparse_outputs_do_not_change_accepted_energy_observations(self):
		problem = _problem()
		dense_request = SimulationRequest((.3,.4),np.array([.3,.301,.32,.35,.4]))
		sparse_request = SimulationRequest((.3,.4),np.array([.3,.4]))
		observer = EnergyObserver()
		dense = simulate(problem.with_request(dense_request), RK4(step=.04), observers=(observer,))
		sparse = simulate(problem.with_request(sparse_request), RK4(step=.04), observers=(observer,))
		for name in ('energy_start','energy_end','energy_change','energy_time_derivative_start','energy_time_derivative_end'):
			np.testing.assert_array_equal(dense.diagnostics[name], sparse.diagnostics[name])
		np.testing.assert_array_equal(dense.states[[0,-1]], sparse.states)

	def test_endpoint_snapshots_survive_reused_physical_buffers(self):
		buffer = np.zeros(1)
		def energy(time, state):
			buffer[:] = time**3
			return buffer
		def rate(time, state):
			buffer[:] = 3*time**2
			return buffer
		run = EnergyObserver().new_run(energy, rate, particle_count=1)
		state = np.zeros(2)
		first = run.accept(StepInfo(0, 1., 1., 2., state), state)
		run.accept(StepInfo(1, 2., 1., 3., state), state)
		for name, expected in (("energy_start", 1.), ("energy_end", 8.),
			("energy_change", 7.), ("energy_time_derivative_start", 3.),
			("energy_time_derivative_end", 12.)):
			np.testing.assert_array_equal(first[name], [expected])

	def test_gc_partial_derivative_uses_normalized_potential_time_differentiation(self):
		problem = _problem()
		dynamics = problem.dynamics
		potential = dynamics.hamiltonian_potential
		t = .31; z = problem.initial_state; x,y = np.split(z,2)
		expected = potential.evaluate(t,x,y,dt=1)
		with patch.object(type(potential), 'evaluate', autospec=True, side_effect=type(potential).evaluate) as evaluate:
			np.testing.assert_array_equal(dynamics.hamiltonian_time_derivative(t,z),expected)
			self.assertEqual(evaluate.call_count,1)
			self.assertEqual(evaluate.call_args.kwargs['dt'],1)
