"""Function-only doubled methods, dense tangents and scoped solver observations."""

import unittest
import numpy as np

from contracts.methods.step import StepInfo
from methods import ABBA2Implicit, ABBA4Implicit, BM4Implicit, ABBA2Midpoint, BM4Midpoint
from methods.palindromic.core.composition import compose
from methods.palindromic.core.operations import A, B, T, C
from methods.palindromic.core.jacobians import composition_jacobian
from methods.palindromic.maps import GCDoubledMaps
from request import SimulationRequest
from solvers import NewtonSolver
from solvers.derivatives import centered_difference, packed_particle_jacobian
from states import InitialConditions


class FunctionPalindromesTests(unittest.TestCase):
	def setUp(self) -> None:
		self.initial = InitialConditions.gc(x=[.2, .4], y=[.3, .5])
		self.request = SimulationRequest.uniform(t_span=(.2, .3), sample_count=3)
		self.blocks = np.array([[[0., 1.], [-1., 0.]], [[0., .7], [-.4, 0.]]])
		self.matrix = packed_particle_jacobian(self.blocks)
		self.f = lambda t, z: self.matrix @ z
		self.jacobian = lambda t, z: self.blocks

	def test_all_presets_accept_plain_functions_and_own_only_physical_state(self) -> None:
		for kind in (ABBA2Implicit, ABBA4Implicit,  BM4Implicit, ABBA2Midpoint, BM4Midpoint):
			with self.subTest(method=kind.__name__):
				run = kind().new_run(self.f, self.initial, jacobian=self.jacobian)
				step = run.advance(.2, self.initial.values, .03)
				self.assertEqual(step.state.shape, (4,))
				for name in ('dynamics', 'problem', 'state_formulation', 'track_energy', 'initial_momentum'):
					self.assertFalse(hasattr(run, name))
				self.assertIs(run.f, self.f)
				self.assertIsNotNone(step.quadrature)
				self.assertAlmostEqual(sum(step.quadrature.weights), .03, places=15)
				event = run.build_observation(StepInfo(0, .2, .03, .23, self.initial.values), step)
				self.assertIsNone(event.dynamics)
				np.testing.assert_array_equal(event.map_state(self.initial.values), step.state)


	def test_dense_and_independent_particle_derivatives_give_same_projection(self) -> None:
		blocks = np.array([[[0., 1.], [-1., 0.]], [[0., .7], [-.4, 0.]]])
		matrix = packed_particle_jacobian(blocks)
		f = lambda t, z: matrix @ z
		for projection in ('reduced_multiplier', 'simultaneous_state_multiplier'):
			method = ABBA4Implicit(projection_formulation=projection)
			dense = ABBA4Implicit(projection_formulation=projection, solver=NewtonSolver(jacobian_method="finite_difference")).new_run(f, self.initial)
			particle = method.new_run(f, self.initial,
				jacobian=lambda t, z: blocks)
			np.testing.assert_allclose(dense.advance(.2, self.initial.values, .03).state,
				particle.advance(.2, self.initial.values, .03).state, rtol=0, atol=2e-12)

	def test_iteration_observation_is_suppressed_only_for_diagnostic_replay(self) -> None:
		events = []
		method = BM4Implicit(solver=NewtonSolver(jacobian_method='analytic', iteration_observer=events.append))
		run = method.new_run(self.f, self.initial, jacobian=self.jacobian)
		step = run.advance(.2, self.initial.values, .03)
		self.assertTrue(events)
		self.assertEqual(events[0].duration, .03)
		count = len(events)
		event = run.build_observation(StepInfo(0, .2, .03, .23, self.initial.values), step)
		event.map_state(self.initial.values)
		run.advance(.2, self.initial.values, .03, observe=False)
		self.assertEqual(len(events), count)
		run.advance(.2, self.initial.values, .03)
		self.assertEqual(len(events), 2 * count)

	def test_stage_quadrature_retains_signed_accepted_sources(self) -> None:
		method = ABBA4Implicit()
		plain = method.new_run(self.f, self.initial, jacobian=self.jacobian)
		tracked = method.new_run(self.f, self.initial, jacobian=self.jacobian)
		left = plain.advance(.2, self.initial.values, -.03)
		right = tracked.advance(.2, self.initial.values, -.03)
		np.testing.assert_array_equal(left.state, right.state)
		self.assertIsNotNone(left.quadrature)
		self.assertTrue(any(w > 0 for w in right.quadrature.weights))
		self.assertTrue(any(w < 0 for w in right.quadrature.weights))
		for source, stage in zip(right.quadrature.states, (s for s in right.details.trace.operations if s.source is not None)):
			self.assertIs(source, stage.source)


if __name__ == '__main__':
	unittest.main()
