"""Analytic geometry checks for transverse Poincare seed placement."""
import unittest
import numpy as np
from studies.poincare_parallel import normal_geometry, inside_polygon, OFFSETS


class ParallelGeometryTests(unittest.TestCase):
    def test_circle_normal_offsets_have_correct_signed_radii(self):
        theta=np.arange(5001)*2*np.pi*(np.sqrt(5)-1)/2
        points=.5+.1*np.column_stack((np.cos(theta),np.sin(theta)))
        g=normal_geometry(points)
        np.testing.assert_allclose(g['normal'],[1,0],atol=2e-4)
        self.assertAlmostEqual(g['width_fraction'],.2,places=5)
        radii=np.linalg.norm(np.array(g['fractions'])-.5,axis=1)
        np.testing.assert_allclose(radii,.1+.2*np.array(OFFSETS),atol=2e-6)
        self.assertEqual(g['inside'],[True,True,False,False])

    def test_ellipse_offsets_scale_and_rotate_with_the_curve(self):
        theta=np.arange(5001)*2*np.pi*(np.sqrt(5)-1)/2
        local=np.column_stack((.08*np.cos(theta),.02*np.sin(theta)))
        rotation=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
        first=normal_geometry(.5+local)
        transformed=normal_geometry(.5+.3*local@rotation.T)
        np.testing.assert_allclose(transformed['normal'],rotation@first['normal'],atol=1e-10)
        self.assertAlmostEqual(transformed['width_fraction'],.3*first['width_fraction'],places=12)
        np.testing.assert_allclose(np.array(transformed['fractions'])-.5,
            .3*(np.array(first['fractions'])-.5)@rotation.T,atol=1e-12)

    def test_seven_islands_use_initial_component_width(self):
        n=np.arange(7001); centers=.5+.1*np.column_stack((np.cos(2*np.pi*(n%7)/7),np.sin(2*np.pi*(n%7)/7)))
        theta=(n//7)*2*np.pi*(np.sqrt(5)-1)/2
        points=centers+.005*np.column_stack((np.cos(theta),np.sin(theta)))
        g=normal_geometry(points,component_stride=7)
        self.assertAlmostEqual(g['width_fraction'],.01,places=5)
        self.assertEqual(g['inside'],[True,True,False,False])

    def test_rejects_unresolved_curve_and_classifies_nonconvex_polygon(self):
        with self.assertRaises(ValueError):normal_geometry(np.ones((20,2)))
        polygon=np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]])
        self.assertTrue(inside_polygon(np.array([.5,1.5]),polygon))
        self.assertFalse(inside_polygon(np.array([1.5,1.5]),polygon))


if __name__=='__main__':unittest.main()
