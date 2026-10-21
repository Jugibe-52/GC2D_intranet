"""BM4 excludes C while the independent exact harmonic flow remains usable."""

import unittest
from unittest.mock import patch

import numpy as np

from formulations import HarmonicCoupling
from formulations.gc import GCDoubledMaps
from methods.extended.core.composition import BM4, compose
from methods.extended.core.projection import solve_projection
from solvers.nonlinear import SolverOptions
from simulation import BM4Implicit, BM4Midpoint, SimulationRequest, simulate
from test_extended_family import problem


class UncoupledBM4Tests(unittest.TestCase):
	"""Check the removed option and the absence of hidden identity coupling."""

	def test_both_methods_skip_c_and_match_zero_coupling_reference(self) -> None:
		p = problem(3)
		h, t = .15, .3
		request = SimulationRequest.uniform(t_span=(t, t + h), max_step=h, sample_count=2)
		old_maps = GCDoubledMaps(p, coupling_frequency=0.)
		trace = compose(old_maps, BM4, t, np.tile(p.initial_state, 2), h)
		size = p.initial_state.size
		references = (
			solve_projection(old_maps, BM4, SolverOptions('newton', 1e-13, 1e-12, 12),
				'reduced_multiplier', t, p.initial_state, h).state,
			(trace.state[:size] + trace.state[size:]) / 2,
		)
		for constructor, expected in zip((BM4Implicit, BM4Midpoint), references):
			with self.subTest(method=constructor.__name__):
				with self.assertRaises(TypeError):
					constructor(coupling_frequency=0.)
				with patch.object(HarmonicCoupling, 'apply', side_effect=AssertionError('BM4 invoked C')):
					run = constructor(track_energy=True).new_run(p, request)
					self.assertIsNone(run.formulation.coupling)
					actual = simulate(p, constructor(track_energy=True), request)
				np.testing.assert_allclose(actual.states[:, -1], expected, rtol=0., atol=3e-15)
				self.assertEqual(actual.diagnostics['coupling_frequency'], 0.)


class HarmonicCouplingTests(unittest.TestCase):
	"""Validate C independently of BM4 or any projection."""

	def test_exact_rotation_mean_group_law_and_input_ownership(self) -> None:
		# Two planar copies, each containing two particles in component order.
		state = np.array([2., 3., 1., 4., 0., -1., -1., 2.])
		before = state.copy()
		flow = HarmonicCoupling(.7)
		angle = .37
		duration = angle / (2 * flow.frequency)
		mean = (state[:4] + state[4:]) / 2
		difference = (state[:4] - state[4:]).reshape(2, 2)
		rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
		rotated = (rotation @ difference).reshape(-1)
		actual = flow.apply(state, duration)
		np.testing.assert_allclose(actual, np.concatenate((mean + rotated / 2, mean - rotated / 2)), atol=1e-15)
		np.testing.assert_array_equal(state, before)
		self.assertFalse(np.shares_memory(actual, state))
		np.testing.assert_allclose(flow.apply(actual, -duration), state, atol=2e-15)
		np.testing.assert_allclose(flow.apply(flow.apply(state, .1), .2), flow.apply(state, .3), atol=2e-15)
		np.testing.assert_array_equal(HarmonicCoupling(0.).apply(state, .8), state)

	def test_invalid_rate_state_and_duration(self) -> None:
		for rate in (-1., np.nan, np.inf):
			with self.assertRaises(ValueError):
				HarmonicCoupling(rate)
		flow = HarmonicCoupling(.5)
		for state in (np.zeros(0), np.zeros(3), np.zeros((4, 1)), np.full(4, np.nan)):
			with self.assertRaises(ValueError):
				flow.apply(state, .1)
		with self.assertRaises(ValueError):
			flow.apply(np.zeros(4), np.inf)


if __name__ == '__main__':
	unittest.main()
