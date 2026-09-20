"""Behavioral contracts of the common method-instance integration lifecycle."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import (
	ABBA2Implicit, ABBA2Midpoint, ABBA4Implicit, ABBA6Implicit,
	BM4Implicit, BM4Midpoint, ExplicitEuler, RK4, GaussLegendre4, SDIRK4, HBVM42,
	InitialValueProblem, SimulationRequest, simulate,
)
from simulation.integration import (
	FixedStepController, IntegrationMethod, StepInfo, StepResult,
	integrate_method,
)


def _problem() -> InitialValueProblem:
	"""One smooth, non-autonomous particle exercises all internal state modes."""
	return InitialValueProblem(
		GuidingCenterDynamics(Potential.random(
			A=0.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5,
		), rho=0.05),
		GCInitialConfiguration.from_components(x=[1.0], y=[1.2]),
	)


def _request(dense: bool = False) -> SimulationRequest:
	return SimulationRequest(
		t_span=(0.3, 0.34), max_step=(0.34 - 0.3) / 2,
		output_times=np.asarray([0.3, 0.305, 0.32, 0.333, 0.34] if dense else [0.3, 0.32, 0.34]),
	)


def _methods():
	"""Cover all public methods, internal domains, and explicit ABBA4 compatibility settings."""
	yield ExplicitEuler()
	for method in (RK4, GaussLegendre4, SDIRK4, HBVM42):
		for tracking in (False, True):
			yield method(track_energy=tracking)
	yield BM4Implicit(coupling_frequency=0.2)
	for method in (BM4Midpoint, ABBA2Midpoint, ABBA2Implicit, ABBA4Implicit, ABBA6Implicit):
		for extension, tracking in (("physical", False), ("physical", True), ("fully_extended", True)):
			yield method(state_extension=extension, track_energy=tracking)
	yield ABBA4Implicit(projection_placement="around_complete_composition")
	yield ABBA4Implicit(state_extension="fully_extended", projection_placement="around_complete_composition")


class MethodIntegrationTests(unittest.TestCase):
	"""Protect shared orchestration without replacing model-level accuracy tests."""

	def test_all_methods_share_integrate_and_preparation_does_not_advance(self) -> None:
		problem = _problem()
		for method in _methods():
			with self.subTest(method=method):
				self.assertIs(type(method).integrate, IntegrationMethod.integrate)
				events = []
				run = replace(method, step_observer=events.append).new_run(problem, _request())
				self.assertIs(type(run), type(method))
				self.assertIsNot(run, method)
				self.assertEqual(events, [])
				self.assertFalse(run.initial_state.flags.writeable)
		with patch("simulation.methods.bm4.implicit._solve_reduced_projected_bm4_step", side_effect=AssertionError("Unexpected solve")):
			BM4Implicit().new_run(problem, _request())

	def test_run_advances_and_shadow_samples_do_not_emit_or_accumulate(self) -> None:
		problem = _problem()
		for method in _methods():
			with self.subTest(method=method):
				events = []
				run = replace(method, step_observer=events.append).new_run(problem, _request(True))
				initial = run.initial_state.copy()
				first = run.advance(0.3, initial, 0.02)
				run.advance(0.3, initial, 0.005)
				self.assertEqual(events, [])
				np.testing.assert_array_equal(initial, run.initial_state)
				np.testing.assert_array_equal(first.state, run.advance(0.3, initial, 0.02).state)
				dense = integrate_method(run)
				self.assertEqual(len(events), 2)
				events.clear()
				again = integrate_method(run.new_run(problem, _request(True)))
				self.assertEqual(len(events), 2)
				for key in dense.diagnostics:
					np.testing.assert_equal(dense.diagnostics[key], again.diagnostics[key])
				sparse = simulate(problem, method, _request())
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
		for method in _methods():
			with self.subTest(method=method):
				plain = simulate(problem, method, _request(True))
				observed = simulate(problem, replace(method, step_observer=corrupt), _request(True))
				np.testing.assert_array_equal(plain.states, observed.states)
				for key in plain.diagnostics:
					np.testing.assert_equal(plain.diagnostics[key], observed.diagnostics[key])

	def test_unobserved_bm4_does_not_replay_stage_events(self) -> None:
		from simulation.methods.bm4 import implicit
		original = implicit._advance_composition
		def checked(*args, **kwargs):
			self.assertIsNone(kwargs["stage_observer"])
			return original(*args, **kwargs)
		with patch.object(implicit, "_advance_composition", side_effect=checked):
			simulate(_problem(), BM4Implicit(), _request(True))

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
		run = BM4Implicit().new_run(_problem(), _request(True))
		data = integrate_method(run, controller=PrescribedController())
		state = run.initial_state.copy()
		for t, end in zip((0.3, 0.31, 0.33), (0.31, 0.33, 0.34)):
			state = run.advance(t, state, end - t).state
		np.testing.assert_array_equal(data.states[:, -1], state)
		np.testing.assert_array_equal(data.diagnostics["step_times"], [0.31, 0.33, 0.34])
		self.assertEqual(data.diagnostics["step_count"], 3)

	def test_failed_step_is_not_observed_or_accepted(self) -> None:
		events = []
		run = BM4Implicit(step_observer=events.append).new_run(_problem(), _request())
		def fail(t, state, h):
			raise RuntimeError("Nonlinear solve failed")
		with patch.object(run, "advance", side_effect=fail):
			with self.assertRaisesRegex(RuntimeError, "Nonlinear solve failed"):
				integrate_method(run)
		self.assertEqual(events, [])

	def test_run_resources_are_isolated_and_a_completed_run_cannot_be_reused(self) -> None:
		problem = _problem()
		other = replace(problem, initial_configuration=GCInitialConfiguration.from_components(x=[1.1], y=[1.3]))
		for method in _methods():
			with self.subTest(method=method):
				first = method.new_run(problem, _request())
				second = method.new_run(other, _request(True))
				self.assertIs(first.problem, problem)
				self.assertIs(second.problem, other)
				self.assertFalse(np.shares_memory(first.initial_state, second.initial_state))
				self.assertFalse(hasattr(method, "initial_state"))
				result = integrate_method(first)
				integrate_method(second)
				np.testing.assert_array_equal(result.states, simulate(problem, method, _request()).states)
				with self.assertRaisesRegex(RuntimeError, "fresh"):
					integrate_method(first)


if __name__ == "__main__":
	unittest.main()
