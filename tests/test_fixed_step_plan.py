"""Shared temporal decisions preserve sampling and execution ownership."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from contracts.execution.plan import SAMPLE_START, SAMPLE_END, SAMPLE_SHADOW
from dynamics.gc import GuidingCenterDynamics
from execution import Execution
from execution._plan import build_fixed_step_plan
from execution.options import ExecutionOptions
from methods import RK4
from potential.potential import Potential
from problem import InitialValueProblem
from request import SimulationRequest
from states.history import States


class FixedStepPlanTests(unittest.TestCase):
    def test_irregular_samples_and_clipped_last_step(self):
        request = SimulationRequest((.3, .53), np.array([.3, .301, .4, .47, .5, .53]))
        plan = build_fixed_step_plan(request, .1)
        np.testing.assert_array_equal(plan.step_starts, [.3, .4, .5])
        np.testing.assert_array_equal(plan.step_ends, [.4, .5, .53])
        np.testing.assert_array_equal(plan.step_sizes, [.1, .1, .53 - .5])
        np.testing.assert_array_equal(plan.sample_offsets, [1, 3, 5, 6])
        np.testing.assert_array_equal(plan.sample_modes, [SAMPLE_START, SAMPLE_SHADOW, SAMPLE_END, SAMPLE_SHADOW, SAMPLE_END, SAMPLE_END])
        self.assertEqual(plan.interior_sample_count, 2)
        self.assertEqual(plan.sample_durations[3], .47 - .4)
        self.assertFalse(np.shares_memory(plan.output_times, request.output_times))
        for array in vars(plan).values():
            self.assertFalse(array.flags.writeable)

    def test_near_boundaries_use_end_priority_and_preserve_requested_times(self):
        eps = np.finfo(float).eps
        times = np.array([0., eps, .1 - eps, .1, .1 + eps, .15, .2])
        plan = build_fixed_step_plan(SimulationRequest((0., .2), times), .1)
        np.testing.assert_array_equal(plan.output_times, times)
        np.testing.assert_array_equal(plan.sample_offsets, [1, 5, 7])
        np.testing.assert_array_equal(plan.sample_modes, [0, 0, 1, 1, 1, 2, 1])
        # Both ends are within tolerance on this tiny interval. The requested
        # final sample must retain the right endpoint, not the initial state.
        tiny = build_fixed_step_plan(SimulationRequest((0., eps), np.array([0., eps])), eps)
        np.testing.assert_array_equal(tiny.sample_modes, [0, 1])

    def test_sparse_outputs_leave_empty_sample_groups_and_allow_one_step(self):
        plan = build_fixed_step_plan(SimulationRequest((0., 1.), np.array([0., 1.])), .25)
        np.testing.assert_array_equal(plan.sample_offsets, [1, 1, 1, 1, 2])
        single = build_fixed_step_plan(SimulationRequest((.3, .35), np.array([.3, .31, .35])), 1.)
        np.testing.assert_array_equal(single.step_sizes, [.35 - .3])
        np.testing.assert_array_equal(single.sample_modes, [0, 2, 1])

    def problem(self, times):
        potential = Potential.random(A=.1, M=2, nx=12, ny=12, seed=74)
        return InitialValueProblem(GuidingCenterDynamics(potential), np.array([.2, .3]),
                                   request=SimulationRequest((.3, .53), np.asarray(times)))

    def test_host_plans_once_before_allocation_and_preserves_observations(self):
        problem = self.problem([.3, .301, .4, .47, .5, .53])
        events = []
        plans = []
        def build(request, step):
            plan = build_fixed_step_plan(request, step)
            plans.append(plan)
            self.assertEqual(events, [])
            return plan
        allocate = States.prepare
        def allocate_after_plan(*args, **kwargs):
            self.assertEqual(len(plans), 1)
            return allocate(*args, **kwargs)
        with patch('execution.execution.build_fixed_step_plan', side_effect=build) as builder:
            with patch('execution._host.build_fixed_step_plan', side_effect=AssertionError('Repeated planning')):
                with patch.object(States, 'prepare', side_effect=allocate_after_plan):
                    result = Execution().run(problem, RK4(step=.1), observers=(events.append,))
        builder.assert_called_once_with(problem.request, .1)
        self.assertEqual(len(events), 3)
        self.assertEqual(result.diagnostics['output_interpolation_count'], 2)
        sparse = Execution().run(problem.with_request(SimulationRequest((.3, .53), np.array([.3, .53]))), RK4(step=.1))
        np.testing.assert_array_equal(result.states[-1], sparse.states[-1])

    def test_plan_failure_finishes_run_before_result_allocation(self):
        problem = self.problem([.3, .53])
        method = RK4(step=.1)
        run = method.new_run(problem.dynamics.vector_field, problem.initial_conditions, problem.request)
        with patch.object(RK4, 'new_run', return_value=run):
            with patch('execution.execution.build_fixed_step_plan', side_effect=ValueError('Invalid plan')):
                with patch.object(States, 'prepare') as allocate:
                    with self.assertRaisesRegex(ValueError, 'Invalid plan'):
                        Execution().run(problem, method)
                    allocate.assert_not_called()
        self.assertEqual(run._status, 'finished')
        self.assertEqual(method._status, 'configuration')

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'Optional JAX is not installed')
    def test_jax_receives_the_plan_before_preparation_and_matches_host_boundaries(self):
        import jax
        from execution import _jax_fixed, _jax_step
        previous = jax.config.read('jax_enable_x64')
        jax.config.update('jax_enable_x64', True)
        self.addCleanup(jax.config.update, 'jax_enable_x64', previous)
        eps = np.finfo(float).eps
        problem = self.problem([.3, .3 + eps, .4 - eps, .4, .4 + eps, .47, .53])
        plans = []
        prepare = _jax_fixed.prepare_integration
        step_prepare = _jax_step.prepare_step
        def build(request, step):
            plan = build_fixed_step_plan(request, step)
            plans.append(plan)
            return plan
        def prepare_step(*args, **kwargs):
            self.assertEqual(len(plans), 1)
            return step_prepare(*args, **kwargs)
        def prepare_integration(run, kernel, plan, **kwargs):
            self.assertIs(plan, plans[0])
            return prepare(run, kernel, plan, **kwargs)
        with patch('execution.execution.build_fixed_step_plan', side_effect=build) as builder:
            with patch.object(_jax_step, 'prepare_step', side_effect=prepare_step):
                with patch.object(_jax_fixed, 'prepare_integration', side_effect=prepare_integration):
                    result = Execution().run(problem, RK4(step=.1), options=ExecutionOptions(backend='jax'))
        builder.assert_called_once()
        reference = Execution().run(problem, RK4(step=.1))
        np.testing.assert_array_equal(result.t, reference.t)
        np.testing.assert_allclose(result.states, reference.states, rtol=2e-12, atol=2e-12)
        np.testing.assert_array_equal(result.diagnostics['step_sizes'], reference.diagnostics['step_sizes'])
        self.assertEqual(result.diagnostics['output_interpolation_count'], reference.diagnostics['output_interpolation_count'])


if __name__ == '__main__':
    unittest.main()
