"""Analytic ratio, floor and nonmonotone-branch checks."""

import unittest
import numpy as np
from studies.equal_accuracy_steps import equal_accuracy_steps


class EqualAccuracyStepsTests(unittest.TestCase):
    def test_fourth_order_constants_and_orientation(self):
        h = np.array([0.4, 0.2, 0.1, 0.05])
        rows = equal_accuracy_steps(h, h**4, 16 * h**4, reference_floor=0)
        self.assertTrue(rows)
        np.testing.assert_allclose([r.step_ratio for r in rows], 0.5)
        np.testing.assert_allclose([r.bm4_local_order for r in rows], 4)

    def test_floor_and_no_extrapolation(self):
        self.assertEqual(equal_accuracy_steps([1, 2], [1, 16], [1, 16], reference_floor=2), ())
        self.assertEqual(equal_accuracy_steps([1, 2], [1, 16], [1, 16], reference_floor=0, targets=[0.5, 20]), ())

    def test_ambiguous_nonmonotone_curve(self):
        self.assertEqual(equal_accuracy_steps([1, 2, 3, 4], [1, 4, 2, 8], [1, 4, 2, 8], reference_floor=0, targets=[3]), ())

    def test_different_orders_do_not_force_constant_ratio(self):
        h = np.array([0.4, 0.2, 0.1, 0.05])
        rows = equal_accuracy_steps(h, h**4, h**2, reference_floor=0)
        self.assertTrue(rows)
        np.testing.assert_allclose([r.step_ratio for r in rows], [r.target_error**0.25 for r in rows])


if __name__ == "__main__":
    unittest.main()
