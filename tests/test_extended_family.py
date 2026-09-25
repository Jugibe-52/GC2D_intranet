"""Structural and numerical contracts of the common extended-space engine."""
from __future__ import annotations

from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch
import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint,
                        BM4Implicit, BM4Midpoint, InitialValueProblem, SimulationRequest, simulate)
from formulations.gc import GCDoubledMaps
from methods.extended.composition import ABBA2, ABBA4, BM4, Composition, compose
from methods.extended.jacobians import central_difference_jacobian, packed_jacobian, particle_jacobians
from methods.extended.projection import solve_projection


def problem(count: int = 3) -> InitialValueProblem:
    """Small reproducible nonautonomous field with analytic spatial derivatives."""
    return InitialValueProblem(
        GuidingCenterDynamics(Potential.random(A=.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5), rho=.05),
        GCInitialConfiguration.from_components(x=np.linspace(1, 1.2, count), y=np.linspace(1.2, 1.4, count)),
    )


class ExtendedFamilyTests(unittest.TestCase):
    def test_public_exports_identify_one_canonical_implementation(self) -> None:
        family = importlib.import_module('methods.extended')
        for name in family.__all__:
            for package in ('methods', 'simulation'):
                self.assertIs(getattr(importlib.import_module(package), name), getattr(family, name))
        for name in ('ABBA2Implicit', 'ABBA4Implicit', 'ABBA6Implicit',
                     'ABBA2Midpoint', 'BM4Implicit', 'BM4Midpoint'):
            method = getattr(family, name)
            self.assertTrue(method.__module__.startswith('methods.extended.'))
            self.assertIs(getattr(importlib.import_module(method.__module__), name), method)

    def test_presets_bind_one_projection_engine_and_preserve_placement(self) -> None:
        p = problem()
        request = SimulationRequest.uniform(t_span=(.3, .34), max_step=(.34 - .3) / 2, sample_count=3)
        for cls, recipe, solves in ((ABBA2Implicit, ABBA2, 1), (ABBA4Implicit, ABBA4, 1),
                                   (ABBA6Implicit, ABBA2, 7), (BM4Implicit, BM4, 1)):
            run = cls().new_run(p, request)
            self.assertIs(run.project.func, solve_projection)
            self.assertIs(run.project.args[1], recipe)
            self.assertEqual(run.metadata['nonlinear_solves_per_step'], solves)

    def test_recipes_reject_invalid_palindromes_and_total_durations(self) -> None:
        for weights in ((1.,), (.2, .8), (.2, .2), (float('nan'), float('nan'))):
            with self.assertRaises(ValueError):
                Composition('invalid', weights)
        self.assertEqual([len(r.coefficients) for r in (ABBA2, ABBA4, BM4)], [2, 6, 12])

    def test_abba_pairs_match_independent_endpoint_shear_equations(self) -> None:
        p = problem()
        maps = GCDoubledMaps(p, coupling_frequency=None)
        z = p.initial_state
        for h in (.2, -.2):
            u, v = z + .003, z - .003
            a = u + h / 2 * p.dynamics.vector_field(.3, v)
            b = v + h / 2 * p.dynamics.vector_field(.3, a)
            c = b + h / 2 * p.dynamics.vector_field(.3 + h, a)
            d = a + h / 2 * p.dynamics.vector_field(.3 + h, c)
            actual = compose(maps, ABBA2, .3, np.concatenate((u, v)), h)
            np.testing.assert_allclose(actual.state, np.concatenate((d, c)), rtol=0, atol=2e-15)

    def test_signed_nonauto_compositions_reverse_and_have_correct_tangents(self) -> None:
        p = problem()
        initial = np.concatenate((p.initial_state + .003, p.initial_state - .003))
        for recipe, coupling in ((ABBA2, None), (ABBA4, None), (BM4, 0.), (BM4, .4)):
            maps = GCDoubledMaps(p, coupling_frequency=coupling)
            for h in (.15, -.15):
                with self.subTest(recipe=recipe.name, coupling=coupling, h=h):
                    trace = compose(maps, recipe, .3, initial, h)
                    reverse = compose(maps, recipe, .3 + h, trace.state, -h)
                    np.testing.assert_allclose(reverse.state, initial, rtol=0, atol=5e-15)
                    exact = packed_jacobian(particle_jacobians(maps, trace))
                    numeric = central_difference_jacobian(
                        lambda z: compose(maps, recipe, .3, z, h).state, initial,
                        relative_step=float(np.cbrt(np.finfo(float).eps)))
                    np.testing.assert_allclose(exact, numeric, rtol=2e-8, atol=2e-9)
                    clock = .3
                    for stage in trace.stages:
                        self.assertEqual(stage.time, clock + stage.duration if stage.direct else clock)
                        clock += stage.duration
                    self.assertAlmostEqual(clock, .3 + h)

    def test_tracking_and_observers_add_no_spatial_work_or_broyden_jacobians(self) -> None:
        p = problem()
        request = SimulationRequest.uniform(t_span=(.3, .34), max_step=(.34 - .3) / 2, sample_count=3)
        for method in (ABBA2Implicit(nonlinear_solver='broyden'), ABBA4Implicit(nonlinear_solver='broyden'),
                       ABBA6Implicit(nonlinear_solver='broyden'), BM4Implicit(nonlinear_solver='broyden', coupling_frequency=.4),
                       ABBA2Midpoint(), BM4Midpoint(coupling_frequency=.4)):
            results, counts = [], []
            for enabled in (False, True):
                observed = []
                with patch.object(p.dynamics, 'vector_field', wraps=p.dynamics.vector_field) as field, \
                     patch.object(p.dynamics, 'particle_vector_field_jacobians', side_effect=AssertionError('Unexpected Jacobian')):
                    results.append(simulate(p, replace(method, track_energy=enabled,
                                                      step_observer=observed.append if enabled else None), request))
                    counts.append(field.call_count)
                if enabled:
                    self.assertEqual(len(observed), 2)
            np.testing.assert_array_equal(results[0].states, results[1].states)
            self.assertEqual(counts[0], counts[1])


if __name__ == '__main__':
    unittest.main()
