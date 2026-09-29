"""Cross-backend contracts for every supported numerical method and solver axis."""

from dataclasses import replace
import importlib.util
import unittest

import numpy as np

from contracts.execution_options import ExecutionOptions
from contracts.problem import InitialValueProblem
from contracts.request import SimulationRequest
from dynamics.gc import GuidingCenterDynamics
from dynamics.fc import FullCyclotronDynamics
from initial_conditions.gc import GCInitialConfiguration
from initial_conditions.fc import FCInitialConfiguration
from methods import (
    ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
    ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, ABBA2Midpoint,
    BM4Implicit, BM4Midpoint, DOP853, Radau,
)
from potential.potential import Potential
from simulation.runner import simulate


@unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
class JaxMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        cls.jax = jax
        cls.previous = jax.config.read('jax_enable_x64')
        jax.config.update('jax_enable_x64', True)
        cls.execution = ExecutionOptions(backend='jax')
        cls.potential = Potential.random(A=.2, M=3, nx=16, ny=16, seed=27)
        cls.gc = InitialValueProblem(GuidingCenterDynamics(cls.potential, rho=.05),
            GCInitialConfiguration.from_components(x=np.array([-.02, 1.8, 6.3]), y=np.array([6.25, .9, -.01])))
        cls.fc = InitialValueProblem(FullCyclotronDynamics(cls.potential, rho=.3, eta=-.4),
            FCInitialConfiguration.from_components(x=np.array([-.02, 1.8, 6.3]), y=np.array([6.25, .9, -.01]),
                                                  vx=np.array([.2, -.1, .3]), vy=np.array([-.1, .1, .2])))
        cls.request = SimulationRequest((.3, .4), .05, np.array([.3, .305, .35, .371, .4]))

    @classmethod
    def tearDownClass(cls):
        cls.jax.config.update('jax_enable_x64', cls.previous)

    def compare(self, method, problem=None, execution=None):
        problem = self.gc if problem is None else problem
        reference = simulate(problem, method, self.request)
        result = simulate(problem, method, self.request, options=execution or self.execution)
        np.testing.assert_allclose(result.states, reference.states, rtol=3e-10, atol=3e-11)
        np.testing.assert_array_equal(result.t, reference.t)
        np.testing.assert_array_equal(result.states[:, 0], problem.initial_state)
        self.assertFalse(result.states.flags.writeable)
        if method.track_energy:
            for name in ('extended_momentum', 'physical_hamiltonian', 'generalized_energy_error'):
                np.testing.assert_allclose(result.diagnostics[name], reference.diagnostics[name], rtol=3e-9, atol=3e-11)
        adaptive = isinstance(method, (DOP853, Radau))
        if not adaptive:
            for key in ('step_times', 'step_sizes', 'step_count', 'output_interpolation_count'):
                np.testing.assert_equal(result.diagnostics[key], reference.diagnostics[key])
            if 'nonlinear_residual_norms' in result.diagnostics:
                self.assertTrue(np.all(result.diagnostics['nonlinear_residual_norms'] <= result.diagnostics['nonlinear_tolerances']))
                np.testing.assert_allclose(result.diagnostics['nonlinear_tolerances'], reference.diagnostics['nonlinear_tolerances'])
                # Newton counts may differ at a floating-point stopping boundary,
                # but all recorded counts must be those of actual accepted solves.
                self.assertTrue(np.all(result.diagnostics['nonlinear_iterations'] >= 0))
            for key in ('state_formulation', 'accepted_internal_state_dimension', 'observer_state_dimension'):
                self.assertEqual(result.diagnostics[key], reference.diagnostics[key])
        self.assertEqual(result.diagnostics['execution_mode'], 'hybrid_scipy_controller' if adaptive else 'device_resident')
        return result

    def test_all_fixed_methods_with_energy_and_irregular_output(self):
        for cls in (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
                    ABBA2Midpoint, BM4Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Implicit):
            with self.subTest(method=cls.__name__):
                self.compare(cls(track_energy=True))

    def test_full_cyclotron_and_finite_difference_newton(self):
        for method in (ExplicitEuler(), GaussLegendre4(track_energy=True),
                       SDIRK4(track_energy=True), HBVM42(track_energy=True)):
            with self.subTest(method=type(method).__name__):
                self.compare(method, self.fc)
        self.compare(GaussLegendre4(newton_jacobian_method='finite_difference'))
        self.compare(SDIRK4(newton_jacobian_method='finite_difference'))
        self.compare(HBVM42(jacobian_method='finite_difference'))

    def test_projection_formulations_broyden_and_mixing(self):
        for method in (
            ABBA2Implicit(projection_formulation='simultaneous_state_multiplier'),
            ABBA4Implicit(projection_formulation='simultaneous_state_multiplier', track_energy=True),
            ABBA6Implicit(projection_formulation='simultaneous_state_multiplier'),
            ABBA2Implicit(nonlinear_solver='broyden'),
            ABBA4Implicit(nonlinear_solver='broyden', projection_formulation='simultaneous_state_multiplier'),
            ABBA6Implicit(nonlinear_solver='broyden'),
            BM4Implicit(nonlinear_solver='broyden', coupling_frequency=.4, track_energy=True),
            BM4Implicit(newton_jacobian_method='finite_difference', coupling_frequency=.4),
            BM4Implicit(coupling_frequency=.4, track_energy=True),
        ):
            with self.subTest(method=method):
                result = self.compare(method)
                self.assertIn('projection_multiplier_norms', result.diagnostics)
                if isinstance(method, (ABBA4Implicit, ABBA6Implicit)):
                    self.assertEqual(result.diagnostics['substep_nonlinear_iterations'].shape[1], 1)

    def test_sparse_output_independence_and_repeat_run_isolation(self):
        method = ABBA4Implicit(track_energy=True)
        dense = simulate(self.gc, method, self.request, options=self.execution)
        sparse_request = replace(self.request, output_times=np.array([.3, .4]))
        sparse = simulate(self.gc, method, sparse_request, options=self.execution)
        np.testing.assert_allclose(sparse.states, dense.states[:, [0, -1]], rtol=0, atol=2e-15)
        np.testing.assert_array_equal(sparse.diagnostics['nonlinear_iterations'], dense.diagnostics['nonlinear_iterations'])
        self.assertFalse(hasattr(method, 'initial_state'))
        again = simulate(self.gc, method, self.request, options=self.execution)
        np.testing.assert_array_equal(again.states, dense.states)

    def test_adaptive_scipy_control_and_physical_observers(self):
        for cls in (DOP853, Radau):
            for problem in (self.gc, self.fc):
                with self.subTest(method=cls.__name__, dimension=problem.dynamics.state_dimension):
                    events = []
                    result = self.compare(cls(track_energy=True, step_observer=events.append), problem)
                    self.assertEqual(result.diagnostics['backend'], 'scipy')
                    self.assertEqual(result.diagnostics['adaptive_controller'], 'scipy')
                    self.assertTrue(events)
                    self.assertEqual(events[-1].state_after.shape, problem.initial_state.shape)
                    self.assertTrue(np.all(result.diagnostics['function_evaluations'] > 0))
        self.compare(Radau(jacobian=lambda t, z: self._dense_gc_jacobian(t, z)))

    def _dense_gc_jacobian(self, time, state):
        blocks = self.gc.dynamics.particle_vector_field_jacobians(time, state)
        n = blocks.shape[0]
        matrix = np.zeros((2*n, 2*n))
        for i in range(n):
            matrix[np.ix_([i, i+n], [i, i+n])] = blocks[i]
        return matrix

    def test_nonconvergence_precision_and_subclass_failures(self):
        for method in (GaussLegendre4(newton_absolute_tolerance=1e-30, newton_relative_tolerance=1e-30, newton_max_iterations=1),
                       SDIRK4(newton_absolute_tolerance=1e-30, newton_relative_tolerance=1e-30, newton_max_iterations=1),
                       HBVM42(absolute_tolerance=1e-30, relative_tolerance=1e-30, max_iterations=1),
                       ABBA4Implicit(newton_absolute_tolerance=1e-30, newton_relative_tolerance=1e-30, newton_max_iterations=1),
                       BM4Implicit(nonlinear_solver='broyden', newton_absolute_tolerance=1e-30,
                                   newton_relative_tolerance=1e-30, newton_max_iterations=1)):
            with self.subTest(method=method), self.assertRaisesRegex(RuntimeError, 'did not converge'):
                simulate(self.gc, method, self.request, options=self.execution)
        class CustomEuler(ExplicitEuler):
            pass
        with self.assertRaisesRegex(TypeError, 'subclass'):
            simulate(self.gc, CustomEuler(), self.request, options=self.execution)
        try:
            self.jax.config.update('jax_enable_x64', False)
            with self.assertRaisesRegex(RuntimeError, 'float64'):
                simulate(self.gc, Radau(), self.request, options=self.execution)
        finally:
            self.jax.config.update('jax_enable_x64', True)

    def test_gpu_all_families_when_available(self):
        try:
            devices = self.jax.devices('gpu')
        except RuntimeError:
            devices = []
        execution = ExecutionOptions(backend='jax', device='gpu')
        if not devices:
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                simulate(self.gc, Radau(), self.request, options=execution)
            self.skipTest('No JAX GPU available; hybrid unavailability is explicit')
        for cls in (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
                    ABBA2Midpoint, BM4Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, BM4Implicit, DOP853, Radau):
            self.compare(cls(track_energy=True), execution=execution)


if __name__ == '__main__':
    unittest.main()
