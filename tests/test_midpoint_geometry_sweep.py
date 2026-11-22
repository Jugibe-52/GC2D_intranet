"""Independent accumulated-map, horizon and nonautonomous-clock checks."""
import unittest
import numpy as np
from dynamics.coordinates import GCCoordinates
from methods.palindromic.core.midpoint import midpoint_step
from methods.palindromic.maps import GCDoubledMaps
from studies.midpoint_geometry import J, RECIPES
from studies.midpoint_geometry_sweep import integrate_case
from diagnostics.studies.midpoint_geometry_sweep import validate_case, matrix_products


class AccumulatedSweepTests(unittest.TestCase):
    def test_product_order_with_noncommuting_shears(self):
        jacobians = np.array([[[[1., 1.], [0., 1.]]], [[[1., 0.], [2., 1.]]]])
        result = matrix_products(jacobians, np.array([1, 2]))
        expected = jacobians[1, 0] @ jacobians[0, 0]
        np.testing.assert_array_equal(result['cumulative_jacobians_decimal'][-1, 0].astype(float), expected)
        self.assertFalse(np.array_equal(expected, jacobians[0, 0] @ jacobians[1, 0]))
        np.testing.assert_array_equal(result['area_defect'], 0)

    def test_decimal_product_retains_area_despite_large_matrix_entries(self):
        shear = float(2**30)
        jacobians = np.array([[[[1., shear], [0., 1.]]], [[[1., 0.], [shear, 1.]]]])
        result = matrix_products(jacobians, np.array([2]))
        np.testing.assert_array_equal(result['area_defect'], 0)
        self.assertEqual(float(result['cumulative_determinants_decimal'][0, 0]), 1.)
        rounded = jacobians[1, 0] @ jacobians[0, 0]
        self.assertEqual(np.linalg.det(rounded), 0.)

    def test_oscillator_closed_form_over_step_horizon_grid(self):
        initial = np.array([.7, -.2, .3, .8])
        maps = GCDoubledMaps(lambda t, y: (J@y.reshape(2, -1)).ravel(), 4,
            coordinates=GCCoordinates(), jacobian=lambda t, y: np.broadcast_to(J, (2, 2, 2)),
            coupling_frequency=None)
        for count in (4, 8):
            a = integrate_case(maps, RECIPES[0], initial, 1., count, (1, 2, 4))
            expected = np.expm1(np.array([1, 2, 4])*count*np.log1p((1/count)**6/64))
            np.testing.assert_allclose(a['area_defect'], np.broadcast_to(expected[:, None], (3, 2)), rtol=2e-11)
            np.testing.assert_allclose(a['return_norm'], expected[:, None]*np.linalg.norm(initial.reshape(2, -1), axis=0), rtol=2e-7, atol=3e-15)
            validate_case(a, count, [1, 2, 4], initial.reshape(2, -1).T)

    def test_non_autonomous_reversal_and_independent_composed_jacobian(self):
        def field(t, z):
            x, y = z
            return (1+.2*t)*np.array([-y*(1+.5*x*x), x*(1+.5*y*y)])
        def derivative(t, z):
            x, y = z
            return ((1+.2*t)*np.array([[-x*y, -1-.5*x*x], [1+.5*y*y, x*y]]))[None]
        maps = GCDoubledMaps(field, 2, coordinates=GCCoordinates(), jacobian=derivative, coupling_frequency=None)
        initial = np.array([.7, .3])
        for recipe in RECIPES:
            a = integrate_case(maps, recipe, initial, 1., 10, (1, 2))
            def forward(z):
                for index in range(20):
                    z = midpoint_step(maps, recipe, index/10, z, .1).state
                return z
            delta = 1e-5
            tangent = np.column_stack([(forward(initial+delta*axis)-forward(initial-delta*axis))/(2*delta) for axis in np.eye(2)])
            accumulated = a['cumulative_jacobians_decimal'][-1, 0].astype(float)
            np.testing.assert_allclose(accumulated, tangent, rtol=0, atol=3e-9)
            self.assertAlmostEqual(abs(np.linalg.det(tangent)-1), float(a['area_defect'][-1, 0]), delta=3e-9)
            validate_case(a, 10, [1, 2], initial[None])
            wrong = a['cycle_states'][2].copy()
            for index in range(20, 0, -1):
                wrong = midpoint_step(maps, recipe, (index-1)/10, wrong, -.1).state
            self.assertGreater(np.linalg.norm(wrong-a['returned_states'][1]), 1e-5)
            np.testing.assert_array_equal(a['cycle_states'][0], initial)

    def test_invalid_horizons_rejected(self):
        maps = GCDoubledMaps(lambda t,y:J@y, 2, coordinates=GCCoordinates(), jacobian=lambda t,y:J[None], coupling_frequency=None)
        for horizons in ((), (0,), (2, 1), (1, 1)):
            with self.assertRaises(ValueError):
                integrate_case(maps, RECIPES[0], np.array([.7, .3]), 1., 20, horizons)


if __name__ == '__main__':
    unittest.main()
