"""Behavioral contracts of the common method-instance integration lifecycle."""

from __future__ import annotations

from dynamics.gc import GuidingCenterDynamics

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from dynamics import GuidingCenterDynamics

from potential import Potential
from simulation import (
	ABBA2Implicit, ABBA2Midpoint, ABBA4Implicit, BM4Implicit, BM4Midpoint, ExplicitEuler, RK4,
	GaussLegendre4, SDIRK4, HBVM42, InitialValueProblem, SimulationRequest, simulate,
)
from methods.base import IntegrationMethod
from contracts.methods.step import StepInfo, StepResult
from execution._host import FixedStepController
from execution import Execution
from diagnostics.methods.energy import EnergyObserver


def _execute_prepared(run, problem, *, controller=None, observers=()):
	"""Inspect a specific prepared run through the complete execution boundary."""
	with patch.object(type(run), "new_run", return_value=run):
		if controller is None:
			return Execution().run(problem.with_request(run.request), run, observers=observers)
		with patch.object(run, "controller", return_value=controller):
			return Execution().run(problem.with_request(run.request), run, observers=observers)


def _problem() -> InitialValueProblem:
	"""One smooth, non-autonomous particle exercises all internal state modes."""
	return InitialValueProblem(GuidingCenterDynamics(Potential.random(
			A=0.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5,
		), rho=0.05), GuidingCenterDynamics.coordinates.pack_components([1.0], [1.2]), layout=GuidingCenterDynamics.coordinates)


def _request(dense: bool = False) -> SimulationRequest:
	return SimulationRequest(
		t_span=(0.3, 0.34),
		output_times=np.asarray([0.3, 0.305, 0.32, 0.333, 0.34] if dense else [0.3, 0.32, 0.34]),
	)


def _methods():
	"""Cover all public fixed methods with and without independent energy state."""
	for cls in (ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
		BM4Implicit, BM4Midpoint, ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit,):
		for tracking in (False, True):
			yield cls(), tracking


def _prepare(method, problem, request, *, tracking=False, observe=False):
	return method.new_run(problem.dynamics.vector_field, problem.initial_conditions, request,
		jacobian=problem.dynamics.particle_vector_field_jacobians, jacobian_structure='particle',
		capture_quadrature=tracking, observe=observe)


class MethodIntegrationTests(unittest.TestCase):
	"""Protect shared orchestration without replacing model-level accuracy tests."""

	def test_all_methods_prepare_without_advancing_or_owning_integration(self) -> None:
		problem = _problem()
		for method, tracking in _methods():
			with self.subTest(method=method):
				self.assertFalse(hasattr(method, "integrate"))
				self.assertFalse(hasattr(method, "_integrate_jax"))
				events = []
				run = _prepare(replace(method, step=(0.34 - 0.3) / 2), problem, _request(), tracking=tracking, observe=True)
				self.assertIs(type(run), type(method))
				self.assertIsNot(run, method)
				self.assertEqual(events, [])
				self.assertFalse(run.initial_state.flags.writeable)
		with patch("methods.extended.palindromic.solve_projection", side_effect=AssertionError("Unexpected solve")):
			_prepare(BM4Implicit(step=(0.34 - 0.3) / 2), problem, _request())

	def test_run_advances_and_shadow_samples_do_not_emit_or_accumulate(self) -> None:
		problem = _problem()
		for method, tracking in _methods():
			with self.subTest(method=method):
				events = []
				run = _prepare(replace(method, step=(0.34 - 0.3) / 2), problem, _request(True), tracking=tracking, observe=True)
				initial = run.initial_state.copy()
				first = run.advance(0.3, initial, 0.02)
				run.advance(0.3, initial, 0.005)
				self.assertEqual(events, [])
				np.testing.assert_array_equal(initial, run.initial_state)
				np.testing.assert_array_equal(first.state, run.advance(0.3, initial, 0.02).state)
				observers = ((EnergyObserver(),) if tracking else ()) + (events.append,)
				dense = _execute_prepared(run, problem, observers=observers)
				self.assertEqual(len(events), 2)
				events.clear()
				again = _execute_prepared(_prepare(replace(method, step=(0.34 - 0.3) / 2), problem, _request(True), tracking=tracking, observe=True), problem, observers=observers)
				self.assertEqual(len(events), 2)
				for key in dense.diagnostics:
					np.testing.assert_equal(dense.diagnostics[key], again.diagnostics[key])
				sparse = simulate(problem.with_request(_request()), replace(method, step=(0.34 - 0.3) / 2), observers=(EnergyObserver(),) if tracking else ())
				np.testing.assert_array_equal(dense.states[:, [0, 2, 4]], sparse.states)
				for key in ("step_times", "step_start_times", "step_sizes"):
					np.testing.assert_array_equal(dense.diagnostics[key], sparse.diagnostics[key])
				self.assertEqual(dense.diagnostics["output_interpolation_count"], 2)
				self.assertEqual(len(dense.diagnostics["step_times"]), 2)

	def test_observation_snapshots_cannot_change_samples_or_metrics(self) -> None:
		problem = _problem()
		def corrupt(event):
			event.state_before[:] = -999
			event.state_after[:] = 999
			if hasattr(event, "multiplier"):
				event.multiplier[:] = 888
			for stage in getattr(event, "base_stages", ()):
				stage.state_before[:] = 777
				stage.state_after[:] = 777
		for method, tracking in _methods():
			with self.subTest(method=method):
				observers = (EnergyObserver(),) if tracking else ()
				plain = simulate(problem.with_request(_request(True)), replace(method, step=(0.34 - 0.3) / 2), observers=observers)
				observed = simulate(problem.with_request(_request(True)), replace(method, step=(0.34 - 0.3) / 2), observers=observers + (corrupt,))
				np.testing.assert_array_equal(plain.states, observed.states)
				for key in plain.diagnostics:
					np.testing.assert_equal(plain.diagnostics[key], observed.diagnostics[key])

	def test_unobserved_bm4_does_not_replay_stage_events(self) -> None:
		with patch("methods.extended.bm4.stage_events", side_effect=AssertionError("Unexpected event construction")):
			simulate(_problem().with_request(_request(True)), BM4Implicit(step=(0.34 - 0.3) / 2))

	def test_controller_can_supply_nonuniform_accepted_steps(self) -> None:
		"""The coordinator uses accepted intervals, independent of step policy."""
		class PrescribedController(FixedStepController):
			def steps(self, run, request):
				state = run.initial_state.copy()
				grid = (0.3, 0.31, 0.33, 0.34)
				for index, (t, end) in enumerate(zip(grid[:-1], grid[1:])):
					info = StepInfo(index, t, end - t, end, state.copy())
					result = run.advance(t, state, end - t)
					yield info, result
					state = result.state
		problem = _problem()
		run = _prepare(BM4Implicit(step=(0.34 - 0.3) / 2), problem, _request(True))
		data = _execute_prepared(run, problem, controller=PrescribedController())
		state = run.initial_state.copy()
		for t, end in zip((0.3, 0.31, 0.33), (0.31, 0.33, 0.34)):
			state = run.advance(t, state, end - t).state
		np.testing.assert_array_equal(data.states[:, -1], state)
		np.testing.assert_array_equal(data.diagnostics["step_times"], [0.31, 0.33, 0.34])
		self.assertEqual(data.diagnostics["step_count"], 3)

	def test_failed_step_is_not_observed_or_accepted(self) -> None:
		events = []
		problem = _problem()
		run = _prepare(BM4Implicit(step=(0.34 - 0.3) / 2), problem, _request(), observe=True)
		def fail(t, state, h, *, observe=True):
			raise RuntimeError("Nonlinear solve failed")
		with patch.object(run, "advance", side_effect=fail):
			with self.assertRaisesRegex(RuntimeError, "Nonlinear solve failed"):
				_execute_prepared(run, problem, observers=(events.append,))
		self.assertEqual(events, [])

	def test_run_resources_are_isolated_and_a_completed_run_cannot_be_reused(self) -> None:
		problem = _problem()
		other = InitialValueProblem(problem.dynamics, GuidingCenterDynamics.coordinates.pack_components([1.1], [1.3]), layout=problem.layout)
		for method, tracking in _methods():
			with self.subTest(method=method):
				first = _prepare(replace(method, step=(0.34 - 0.3) / 2), problem, _request())
				second = _prepare(replace(method, step=(0.34 - 0.3) / 2), other, _request(True))
				self.assertFalse(hasattr(first, "problem"))
				self.assertFalse(hasattr(second, "dynamics"))
				self.assertFalse(np.shares_memory(first.initial_state, second.initial_state))
				self.assertFalse(hasattr(method, "initial_state"))
				result = _execute_prepared(first, problem)
				_execute_prepared(second, other)
				np.testing.assert_array_equal(result.states, simulate(problem.with_request(_request()), replace(method, step=(0.34 - 0.3) / 2)).states)
				with self.assertRaisesRegex(TypeError, "fresh"):
					_execute_prepared(first, problem)


if __name__ == "__main__":
	unittest.main()
