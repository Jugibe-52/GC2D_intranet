"""Structural and numerical contracts of the common extended-space engine."""
from __future__ import annotations

from methods.extended.abba import _ABBA4_COEFFICIENTS
from diagnostics.methods.energy import EnergyObserver
from solvers import BroydenSolver

from dataclasses import replace
import importlib
import unittest
from unittest.mock import patch
import numpy as np

from dynamics import GuidingCenterDynamics

from potential import Potential
from simulation import ABBA2Implicit, ABBA4Implicit, ABBA2Midpoint, BM4Implicit, BM4Midpoint, InitialValueProblem, SimulationRequest, simulate
from methods.extended.maps import GCDoubledMaps
from methods.extended.palindromic import ProjectedPalindromicMethod
from methods.extended.core.composition import compose, validate_composition
from methods.extended.core.operations import A, B, T, C
from methods.extended.core.jacobians import composition_jacobian
from solvers.derivatives import centered_difference, packed_particle_jacobian
from methods.extended.core.projection import solve_projection


def problem(count: int = 3) -> InitialValueProblem:
    """Small reproducible nonautonomous field with analytic spatial derivatives."""
    return InitialValueProblem(GuidingCenterDynamics(Potential.random(A=.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5), rho=.05), GuidingCenterDynamics.coordinates.pack_components(np.linspace(1, 1.2, count), np.linspace(1.2, 1.4, count)), layout=GuidingCenterDynamics.coordinates)


_BM4_HALF = (.0792036964311957, .1303114101821663, .2228614958676077,
             -.3667132690474257, .3246481886897062, .1096884778767498)


def unmerged_recipe(weights, coupled=False):
    """Reproduce the former direct/adjoint clock schedule independently."""
    operations = []
    for index, weight in enumerate(weights):
        group = ((C(weight),) if coupled else ()) + (A(weight), B(weight), T(weight))
        operations.extend(group if index % 2 == 0 else group[::-1])
    return tuple(operations)


class ExtendedFamilyTests(unittest.TestCase):
    def test_public_exports_identify_one_canonical_implementation(self) -> None:
        family = importlib.import_module('methods.extended')
        for name in family.__all__:
            for package in ('methods', 'simulation'):
                self.assertIs(getattr(importlib.import_module(package), name), getattr(family, name))
        for name in ('ABBA2Implicit', 'ABBA4Implicit',
                     'ABBA2Midpoint', 'BM4Implicit', 'BM4Midpoint'):
            method = getattr(family, name)
            owner = 'abba' if name.startswith('ABBA') else 'bm4'
            self.assertEqual(method.__module__, f'methods.extended.{owner}')
            self.assertIs(getattr(importlib.import_module(method.__module__), name), method)

    def test_implicit_presets_share_direct_preparation_and_advance(self) -> None:
        for cls in (ABBA2Implicit, ABBA4Implicit, BM4Implicit):
            self.assertEqual(cls.__bases__, (ProjectedPalindromicMethod,))
            self.assertIs(cls.initialize, ProjectedPalindromicMethod.initialize)
            self.assertIs(cls.advance, ProjectedPalindromicMethod.advance)
        abba = importlib.import_module('methods.extended.abba')
        self.assertFalse(hasattr(abba, '_ABBAImplicitMethod'))
        self.assertFalse(hasattr(abba, '_ComposedABBAImplicit'))

    def test_presets_bind_one_projection_engine_and_preserve_placement(self) -> None:
        p = problem()
        request = SimulationRequest.uniform(t_span=(.3, .34),  sample_count=3)
        for cls, recipe, solves in ((ABBA2Implicit, ABBA2Implicit.recipe, 1), (ABBA4Implicit, ABBA4Implicit.recipe, 1),
                                    (BM4Implicit, BM4Midpoint.recipe, 1)):
            run = replace(cls(), step=(0.34 - 0.3) / 2).new_run(p.dynamics.vector_field, p.initial_conditions, request, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None), jacobian_structure='particle', observe=True)
            self.assertIs(run.project.func, solve_projection)
            self.assertIs(run.project.args[1], recipe)
            self.assertEqual(run.metadata['nonlinear_solves_per_step'], solves)

    def test_presets_keep_spatial_work_and_combine_consecutive_clocks(self) -> None:
        for recipe, count, shears in ((ABBA2Implicit.recipe, 5, 4), (ABBA4Implicit.recipe, 15, 12), (BM4Midpoint.recipe, 30, 24)):
            validate_composition(recipe)
            self.assertEqual(len(recipe), count)
            self.assertEqual(sum(op.name in ('A', 'B') for op in recipe), shears)
            self.assertFalse(any(a.name == b.name == 'T' for a, b in zip(recipe, recipe[1:])))

    def test_merged_clocks_preserve_signed_states_tangents_and_energy_sources(self) -> None:
        p = problem()
        maps = GCDoubledMaps(p.dynamics.vector_field, p.initial_state.size, coordinates=p.layout,
            jacobian=p.dynamics.particle_vector_field_jacobians, jacobian_structure='particle')
        initial = np.concatenate((p.initial_state+.003, p.initial_state-.004))
        for recipe, weights in ((ABBA2Implicit.recipe, (.5, .5)),
                (ABBA4Implicit.recipe, tuple(c/2 for c in _ABBA4_COEFFICIENTS for _ in range(2))),
                (BM4Midpoint.recipe, _BM4_HALF + _BM4_HALF[::-1])):
            for h in (.15, -.15):
                expected = compose(maps, unmerged_recipe(weights), .3, initial, h)
                actual = compose(maps, recipe, .3, initial, h)
                np.testing.assert_allclose(actual.state, expected.state, rtol=0, atol=3e-15)
                np.testing.assert_allclose(composition_jacobian(maps, actual),
                    composition_jacobian(maps, expected), rtol=0, atol=3e-15)
                self.assertAlmostEqual(actual.time, expected.time, places=14)
                for a, b in zip(actual.quadrature_points, expected.quadrature_points, strict=True):
                    self.assertAlmostEqual(a[0], b[0], places=14)
                    self.assertEqual(a[1], b[1])
                    np.testing.assert_allclose(a[2], b[2], rtol=0, atol=3e-15)

    def test_abba_pairs_match_independent_endpoint_shear_equations(self) -> None:
        p = problem()
        maps = GCDoubledMaps(p.dynamics.vector_field, p.initial_state.size, coordinates=p.layout, coupling_frequency=None, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None))
        z = p.initial_state
        for h in (.2, -.2):
            u, v = z + .003, z - .003
            a = u + h / 2 * p.dynamics.vector_field(.3, v)
            b = v + h / 2 * p.dynamics.vector_field(.3, a)
            c = b + h / 2 * p.dynamics.vector_field(.3 + h, a)
            d = a + h / 2 * p.dynamics.vector_field(.3 + h, c)
            actual = compose(maps, ABBA2Implicit.recipe, .3, np.concatenate((u, v)), h)
            np.testing.assert_allclose(actual.state, np.concatenate((d, c)), rtol=0, atol=2e-15)

    def test_signed_nonauto_compositions_reverse_and_have_correct_tangents(self) -> None:
        p = problem()
        initial = np.concatenate((p.initial_state + .003, p.initial_state - .003))
        for recipe, coupling in ((ABBA2Implicit.recipe, None), (ABBA4Implicit.recipe, None), (BM4Midpoint.recipe, 0.), (unmerged_recipe(_BM4_HALF + _BM4_HALF[::-1], coupled=True), .4)):
            maps = GCDoubledMaps(p.dynamics.vector_field, p.initial_state.size, coordinates=p.layout, coupling_frequency=coupling, jacobian=getattr(p.dynamics, 'particle_vector_field_jacobians', None))
            for h in (.15, -.15):
                with self.subTest(recipe=recipe, coupling=coupling, h=h):
                    trace = compose(maps, recipe, .3, initial, h)
                    reverse = compose(maps, recipe, .3 + h, trace.state, -h)
                    np.testing.assert_allclose(reverse.state, initial, rtol=0, atol=5e-15)
                    exact = packed_particle_jacobian(composition_jacobian(maps, trace))
                    numeric = centered_difference(
                        lambda z: compose(maps, recipe, .3, z, h).state, initial,
                        relative_step=float(np.cbrt(np.finfo(float).eps)))
                    np.testing.assert_allclose(exact, numeric, rtol=2e-8, atol=2e-9)
                    clock = .3
                    for stage in trace.operations:
                        self.assertEqual(stage.time, clock)
                        if stage.operation == 'T':
                            clock += stage.duration
                    self.assertAlmostEqual(clock, .3 + h)

    def test_tracking_and_observers_add_no_spatial_work_or_broyden_jacobians(self) -> None:
        p = problem()
        request = SimulationRequest.uniform(t_span=(.3, .34),  sample_count=3)
        for method in (ABBA2Implicit(solver=BroydenSolver(jacobian_method='analytic')), ABBA4Implicit(solver=BroydenSolver(jacobian_method='analytic')),
                        BM4Implicit(solver=BroydenSolver(jacobian_method='analytic')),
                       ABBA2Midpoint(), BM4Midpoint()):
            results, counts = [], []
            for enabled in (False, True):
                observed = []
                with patch.object(type(p.dynamics), 'vector_field', autospec=True, side_effect=type(p.dynamics).vector_field) as field, \
                     patch.object(type(p.dynamics), 'particle_vector_field_jacobians', autospec=True, side_effect=AssertionError('Unexpected Jacobian')):
                    results.append(simulate(p.with_request(request), replace(replace(method), step=(0.34 - 0.3) / 2), observers=((EnergyObserver(),) if enabled else ()) + tuple((o for o in (observed.append if enabled else None,) if o is not None))))
                    counts.append(field.call_count)
                if enabled:
                    self.assertEqual(len(observed), 2)
            np.testing.assert_array_equal(results[0].states, results[1].states)
            self.assertEqual(counts[0], counts[1])


if __name__ == '__main__':
    unittest.main()
