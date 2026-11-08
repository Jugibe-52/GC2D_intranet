"""Request-free preparation and deterministic differentiation across families."""

import unittest

import numpy as np

from dynamics.coordinates import GCCoordinates
from methods import (
    ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
    ABBA2Implicit, ABBA4Implicit, BM4Implicit, ABBA2Midpoint, BM4Midpoint,
)
from request import SimulationRequest
from solvers.nonlinear import NewtonSolver, BroydenSolver
from states.initial import InitialConditions


IMPLICIT = (GaussLegendre4, SDIRK4, HBVM42, ABBA2Implicit, ABBA4Implicit, BM4Implicit)


class TemporalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.initial = InitialConditions([1., .4], layout=GCCoordinates())
        self.matrix = np.array([[0., -1.], [1., 0.]])

    def field(self, time, state):
        return (1 + time) * (self.matrix @ state)

    def jacobian(self, time, state):
        return (1 + time) * self.matrix[None, :, :]

    def test_preparation_evaluates_no_physical_function_and_keeps_no_request(self):
        def forbidden(time, state):
            raise AssertionError('Preparation must not evaluate physical functions.')

        for cls in (*IMPLICIT, ExplicitEuler, RK4, ABBA2Midpoint, BM4Midpoint):
            with self.subTest(method=cls.__name__):
                method = cls(solver=NewtonSolver(jacobian_method='auto')) if cls in IMPLICIT else cls()
                run = method.new_run(forbidden, self.initial, jacobian=forbidden)
                self.assertFalse(hasattr(run, 'request'))
                self.assertEqual(run._status, 'ready')
                self.assertIsNot(run, method)
                self.assertFalse(hasattr(run, 'initial_state'))

    def test_auto_propagates_a_supplied_derivative_failure_at_use(self):
        def invalid(time, state):
            raise ValueError('Supplied derivative failed.')

        for cls in IMPLICIT:
            with self.subTest(method=cls.__name__):
                run = cls(solver=NewtonSolver(jacobian_method='auto')).new_run(
                    self.field, self.initial, jacobian=invalid)
                with self.assertRaisesRegex(ValueError, 'Supplied derivative failed'):
                    run.advance(.7, self.initial.values, .1)

    def test_auto_without_derivative_and_explicit_approximation_agree(self):
        def forbidden(time, state):
            raise AssertionError('Explicit finite differences must ignore this derivative.')

        for cls in IMPLICIT:
            with self.subTest(method=cls.__name__):
                automatic = cls(solver=NewtonSolver(jacobian_method='auto')).new_run(self.field, self.initial)
                approximate = cls(solver=NewtonSolver(jacobian_method='finite_difference')).new_run(
                    self.field, self.initial, jacobian=forbidden)
                analytic = cls(solver=NewtonSolver(jacobian_method='analytic')).new_run(
                    self.field, self.initial, jacobian=self.jacobian)
                expected = analytic.advance(.7, self.initial.values, .1).state
                actual = automatic.advance(.7, self.initial.values, .1).state
                np.testing.assert_array_equal(actual, approximate.advance(.7, self.initial.values, .1).state)
                np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)

    def test_bad_derivative_values_are_validated_when_used(self):
        for cls in IMPLICIT:
            for value in (np.ones((3, 3)), np.full((2, 2), np.nan)):
                with self.subTest(method=cls.__name__, shape=value.shape):
                    run = cls(solver=NewtonSolver(jacobian_method='auto')).new_run(
                        self.field, self.initial, jacobian=lambda t, z: value)
                    with self.assertRaises((ValueError, FloatingPointError)):
                        run.advance(.7, self.initial.values, .1)

    def test_broyden_uses_its_supplied_projection_matrix_without_field_derivatives(self):
        for cls in (ABBA2Implicit, ABBA4Implicit, BM4Implicit):
            with self.subTest(method=cls.__name__):
                run = cls(solver=BroydenSolver(jacobian_method='analytic')).new_run(self.field, self.initial)
                result = run.advance(.7, self.initial.values, .05)
                self.assertTrue(np.all(np.isfinite(result.state)))
                self.assertTrue(result.converged)



if __name__ == '__main__':
    unittest.main()
