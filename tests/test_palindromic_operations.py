"""Independent numerical checks for explicit time/spatial palindromes."""

from dynamics import GuidingCenterDynamics
from diagnostics.methods.energy import EnergyObserver
from solvers import BroydenSolver, NewtonSolver
from dataclasses import dataclass, replace
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np

from execution.options import ExecutionOptions
from methods.extended.maps import HarmonicCoupling
from methods.extended.maps import GCDoubledMaps
from methods.extended.core.composition import compose
from methods.extended.core.jacobians import composition_jacobian
from solvers.derivatives import centered_difference, packed_particle_jacobian
from simulation import (
	A, B, T, C, Operation, PalindromicMethod, PalindromicImplicit, PalindromicMidpoint, ABBA2Implicit,
	ABBA4Implicit, ABBA2Midpoint, BM4Implicit, BM4Midpoint, InitialValueProblem, SimulationRequest,
	simulate,
)
from tests.test_extended_family import problem

EDGES = (T(.5), A(.5), B(.5), B(.5), A(.5), T(.5))
COUPLED = (T(.5), C(.5), A(.5), B(.5), B(.5), A(.5), C(.5), T(.5))


class TimeRotation:
	"""Exact nonautonomous rotation with an independently known energy balance."""

	state_dimension = 2

	def vector_field(self, t, state):
		x, y = np.split(state, 2)
		return (1 + t + t*t) * np.concatenate((-y, x))

	def particle_vector_field_jacobians(self, t, state):
		return np.broadcast_to((1+t+t*t)*np.array([[0., -1.], [1., 0.]]), (state.size//2, 2, 2)).copy()

	def hamiltonian(self, t, state):
		x, y = np.split(state, 2)
		return .5 * (1+t+t*t) * (x*x+y*y)

	def extended_momentum_derivative(self, t, state):
		x, y = np.split(state, 2)
		return -.5 * (1+2*t) * (x*x+y*y)


class PalindromicOperationTests(unittest.TestCase):
	def test_tuple_recipes_are_validated_when_preparing_a_run(self):
		self.assertEqual(tuple(op.name for op in EDGES), tuple('TABBAT'))
		self.assertIsInstance(ABBA2Implicit.recipe, tuple)
		self.assertEqual(tuple(op.name for op in ABBA2Implicit.recipe), tuple('ABTBA'))
		p = problem()
		request = SimulationRequest((.3, .34), np.array([.3, .34]))
		for operations in ((), list(ABBA2Implicit.recipe), ('A',), (A(1.), B(1.), T(1.)),
			(A(.2), B(.5), T(1.), B(.5), A(.2))):
			for method in (PalindromicMidpoint, PalindromicImplicit):
				with self.assertRaises(ValueError):
					method(operations).new_run(p.dynamics.vector_field, p.initial_conditions, request)
		for coefficient in (np.nan, np.inf, True):
			with self.assertRaises(ValueError): A(coefficient)
		with self.assertRaises(ValueError): Operation('Z', 1.)

	def test_spatial_operations_have_no_hidden_half_or_time_advance(self):
		p = problem(2)
		maps = GCDoubledMaps(p.dynamics.vector_field, p.initial_state.size, coordinates=p.layout, coupling_frequency=None, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None))
		u, v = p.initial_state+.01, p.initial_state-.02
		packed = np.concatenate((u, v))
		for operation, expected in (
			('A', np.concatenate((u+.3*p.dynamics.vector_field(.4, v), v))),
			('B', np.concatenate((u, v+.3*p.dynamics.vector_field(.4, u))))):
			actual = maps.apply_spatial(operation, .3, .4, packed)
			np.testing.assert_array_equal(actual, expected)
		trace = compose(maps, EDGES, .4, packed, -.3)
		self.assertAlmostEqual(trace.time, .1)
		for op in trace.operations:
			if op.operation == 'T':
				self.assertIs(op.state_after, op.state_before)
				self.assertIsNone(op.source)
			else:
				self.assertEqual(op.time, op.end_time)
				self.assertAlmostEqual(op.time, .25)

	def test_end_clock_recipe_matches_independent_midtime_shears(self):
		p = problem(3); z = p.initial_state
		maps = GCDoubledMaps(p.dynamics.vector_field, p.initial_state.size, coordinates=p.layout, coupling_frequency=None, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None))
		for h in (.2, -.2):
			u, v = z+.003, z-.004
			first = u+h/2*p.dynamics.vector_field(.3+h/2, v)
			middle = v+h/2*p.dynamics.vector_field(.3+h/2, first)
			last = middle+h/2*p.dynamics.vector_field(.3+h/2, first)
			final = first+h/2*p.dynamics.vector_field(.3+h/2, last)
			actual = compose(maps, EDGES, .3, np.concatenate((u, v)), h)
			np.testing.assert_allclose(actual.state, np.concatenate((final, last)), rtol=0, atol=2e-15)

	def test_signed_reversal_and_exact_tangents_with_explicit_c(self):
		p = problem(3)
		initial = np.concatenate((p.initial_state+.01, p.initial_state-.02))
		for recipe, rate in ((EDGES, None), (COUPLED, .7)):
			maps = GCDoubledMaps(p.dynamics.vector_field, p.initial_state.size, coordinates=p.layout, coupling_frequency=rate, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None))
			for h in (.15, -.15):
				trace = compose(maps, recipe, .3, initial, h)
				reverse = compose(maps, recipe, .3+h, trace.state, -h)
				np.testing.assert_allclose(reverse.state, initial, rtol=0, atol=3e-15)
				analytic = packed_particle_jacobian(composition_jacobian(maps, trace))
				numeric = centered_difference(lambda z: compose(maps, recipe, .3, z, h).state,
					initial, relative_step=float(np.cbrt(np.finfo(float).eps)))
				np.testing.assert_allclose(analytic, numeric, rtol=2e-8, atol=2e-9)

	def test_every_preset_inherits_common_execution_and_retains_its_recipe(self):
		p = problem(); request = SimulationRequest((.3, .34),  np.array([.3, .34]))
		for cls, recipe in ((ABBA2Implicit, ABBA2Implicit.recipe), (ABBA4Implicit, ABBA4Implicit.recipe),
			(BM4Implicit, BM4Midpoint.recipe), (ABBA2Midpoint, ABBA2Implicit.recipe), (BM4Midpoint, BM4Midpoint.recipe)):
			method = cls()
			self.assertIsInstance(method, PalindromicMethod)
			self.assertIs(method.recipe, recipe)
			generic = PalindromicMidpoint if cls in (ABBA2Midpoint, BM4Midpoint) else PalindromicImplicit
			self.assertIs(cls.advance, generic.advance)
			actual = simulate(p.with_request(request), replace(method, step=0.02), observers=(EnergyObserver(),))
			self.assertEqual(json.loads(actual.diagnostics['operation_coefficients']),
				[op.coefficient for op in recipe])
			expected = simulate(p.with_request(request), replace(generic(recipe), step=0.02), observers=(EnergyObserver(),))
			np.testing.assert_array_equal(actual.states, expected.states)
			np.testing.assert_array_equal(actual.diagnostics['extended_momentum'], expected.diagnostics['extended_momentum'])

	def test_recipe_only_subclass_and_explicit_c_configuration(self):
		@dataclass
		class EndTimeMidpoint(ABBA2Midpoint):
			recipe = EDGES
		p = problem(); request = SimulationRequest((.3, .34),  np.array([.3, .34]))
		np.testing.assert_array_equal(simulate(p.with_request(request), replace(EndTimeMidpoint(), step=0.02)).states,
			simulate(p.with_request(request), PalindromicMidpoint(EDGES, step=0.02)).states)
		for method in (PalindromicMidpoint(COUPLED), PalindromicImplicit(COUPLED),
			PalindromicMidpoint(EDGES, coupling=HarmonicCoupling(.7))):
			with self.assertRaisesRegex(ValueError, 'configured together'):
				replace(method, step=0.02).new_run(p.dynamics.vector_field, p.initial_conditions, request, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None), jacobian_structure='particle', observe=True)
		with patch.object(HarmonicCoupling, 'apply', wraps=HarmonicCoupling(.7).apply) as operation:
			run = PalindromicMidpoint(COUPLED, step=0.02, coupling=HarmonicCoupling(0.7)).new_run(p.dynamics.vector_field, p.initial_conditions, request, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None), jacobian_structure='particle', observe=True)
			run.advance(.3, run.initial_state, .02)
			self.assertEqual(operation.call_count, 2)

	def test_both_time_placements_have_second_order_and_passive_balance(self):
		p = InitialValueProblem(TimeRotation(), np.array([1., .2]), layout=GuidingCenterDynamics.coordinates)
		end = .8
		angle = end+end**2/2+end**3/3
		exact = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ p.initial_state
		coarse = []
		for cls in (PalindromicImplicit, PalindromicMidpoint):
			for recipe in (ABBA2Implicit.recipe, EDGES):
				errors, balances = [], []
				for h in (.1, .05, .025):
					r = SimulationRequest((0., end),  np.array([0., end]))
					s = simulate(p.with_request(r), replace(cls(recipe), step=h), observers=(EnergyObserver(),))
					errors.append(np.max(abs(s.states[:, -1]-exact)))
					balances.append(np.max(abs(s.diagnostics['generalized_energy_error'][:, -1])))
					if h == .1: coarse.append(s.states[:, -1])
				for gains in (np.array(errors[:-1])/errors[1:], np.array(balances[:-1])/balances[1:]):
					self.assertTrue(np.all((gains>3.5) & (gains<4.5)), gains)
				print(f'{cls.__name__} {"center" if recipe is ABBA2Implicit.recipe else "edges"}: errors={errors}, balance={balances}')
		self.assertGreater(np.max(abs(coarse[0]-coarse[1])), 1e-5)

	@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
	def test_generic_recipes_agree_on_cpu_and_jax(self):
		import jax
		p = problem(3); request = SimulationRequest((.3, .34),  np.array([.3, .305, .34]))
		methods = (
			PalindromicMidpoint(EDGES),
			PalindromicMidpoint(COUPLED, coupling=HarmonicCoupling(0.7)),
			PalindromicImplicit(EDGES),
			PalindromicImplicit(COUPLED, coupling=HarmonicCoupling(0.7)),
			PalindromicImplicit(EDGES, projection_formulation='simultaneous_state_multiplier'),
			PalindromicImplicit(EDGES, solver=NewtonSolver(jacobian_method='finite_difference')),
			PalindromicImplicit(EDGES, solver=BroydenSolver(jacobian_method='analytic')),
		)
		with jax.experimental.enable_x64():
			for index, method in enumerate(methods):
				observers = (EnergyObserver(),) if index < 4 else ()
				cpu = simulate(p.with_request(request), replace(method, step=0.02), observers=observers)
				compiled = simulate(p.with_request(request), replace(method, step=0.02), options=ExecutionOptions(backend='jax'), observers=observers)
				np.testing.assert_allclose(compiled.states, cpu.states, rtol=0, atol=3e-12)
				if index < 4:
					np.testing.assert_allclose(compiled.diagnostics['extended_momentum'],
						cpu.diagnostics['extended_momentum'], rtol=0, atol=3e-12)


if __name__ == '__main__':
	unittest.main()
