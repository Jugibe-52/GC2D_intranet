"""Behavioral safeguards for the common prepared ABBA runtime."""

from __future__ import annotations

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import ABBA2Implicit, ABBA4Implicit, ABBA6Implicit, InitialValueProblem, SimulationRequest, simulate
from methods._nonlinear import _solve_newton
from methods.extended.core.records import CompositionTrace, ProjectedMapResult
from contracts.step import StepInfo


def _problem() -> InitialValueProblem:
	"""One inexpensive smooth non-autonomous particle."""
	potential = Potential.random(A=0.08, M=3, nx=16, ny=16, seed=27, interpolation_order=5)
	return InitialValueProblem(
		GuidingCenterDynamics(potential, rho=0.05),
		GCInitialConfiguration.from_components(x=[1.0], y=[1.2]),
	)


def _request(times: list[float]) -> SimulationRequest:
	return SimulationRequest(t_span=(0.0, 0.04), max_step=0.02, output_times=np.asarray(times))


class SharedABBARuntimeTests(unittest.TestCase):
	"""Protect sampling, optional observation, and projection record meaning."""

	def test_prepared_records_distinguish_projection_count_from_base_map_count(self) -> None:
		problem = _problem()
		request = _request([0.0, 0.02, 0.04])
		for formulation, solver, track_energy, (method_type, maps) in product(
			("reduced_multiplier", "simultaneous_state_multiplier"),
			("newton", "broyden"),
			(False, True),
			(
				(ABBA2Implicit, 1),
				(ABBA4Implicit, 3),
				(ABBA6Implicit, 7),
			),
		):
			with self.subTest(method=method_type.__name__, formulation=formulation,
				solver=solver, track_energy=track_energy):
				method = method_type(projection_formulation=formulation,
					nonlinear_solver=solver, track_energy=track_energy,
					newton_max_iterations=50, step_observer=lambda event: None)
				prepared = method.new_run(problem, request)
				state = prepared.state_formulation.physical(prepared.initial_state)
				step = prepared.advance(0.0, prepared.initial_state, 0.02)
				result = step.details
				self.assertIsInstance(result, ProjectedMapResult)
				self.assertIsInstance(result.trace, CompositionTrace)
				self.assertEqual(len(result.trace.stages), 2 * maps)
				self.assertLessEqual(result.stats.residual_norm, result.stats.tolerance)
				np.testing.assert_array_equal(result.state_before, state)
				np.testing.assert_array_equal(
					result.state, prepared.state_formulation.physical(step.state))
				event = prepared.build_observation(
					StepInfo(0, 0.0, 0.02, 0.02, prepared.initial_state), step)
				for metric, observed in (
					("nonlinear_iterations", event.newton_iterations),
					("residual_evaluations", event.residual_evaluations),
					("nonlinear_residual_norms", event.newton_residual_norm),
					("nonlinear_tolerances", event.newton_tolerance),
					("projection_multiplier_norms", event.projection_multiplier_norm),
				):
					self.assertEqual(step.statistics[metric], observed)
					if maps > 1:
						np.testing.assert_array_equal(step.statistics[f"substep_{metric}"], [observed])
					else:
						self.assertNotIn(f"substep_{metric}", step.statistics)
				np.testing.assert_array_equal(event.state_after, result.state)
				np.testing.assert_allclose(event.map_state(state), result.state, rtol=0.0, atol=0.0)
				if maps > 1:
					self.assertEqual(len(event.substeps), maps)
				self.assertFalse(prepared.initial_state.flags.writeable)
				with self.assertRaises(TypeError):
					prepared.metadata["step_count"] = 3  # type: ignore[index]

	def test_shadow_samples_preserve_main_trajectory_metrics_and_events(self) -> None:
		problem = _problem()
		for extension, solver, placement in product(
			("physical",), ("newton", "broyden"),
			("around_complete_composition",),
		):
			with self.subTest(extension=extension, solver=solver, placement=placement):
				options = dict(state_extension=extension, nonlinear_solver=solver,
					projection_placement=placement, track_energy=True, newton_max_iterations=50)
				main_events, dense_events = [], []
				main = simulate(problem, ABBA4Implicit(**options, step_observer=main_events.append),
					_request([0.0, 0.02, 0.04]))
				dense = simulate(problem, ABBA4Implicit(**options, step_observer=dense_events.append),
					_request([0.0, 0.009, 0.02, 0.031, 0.04]))
				np.testing.assert_array_equal(main.states, dense.states[:, [0, 2, 4]])
				for key in ("nonlinear_iterations", "residual_evaluations", "nonlinear_residual_norms",
					"nonlinear_tolerances", "projection_multiplier_norms"):
					np.testing.assert_array_equal(main.diagnostics[key], dense.diagnostics[key])
				self.assertEqual(len(main_events), 2)
				self.assertEqual(len(dense_events), 2)
				for a, b in zip(main_events, dense_events, strict=True):
					np.testing.assert_array_equal(a.state_after, b.state_after)
				np.testing.assert_array_equal(
					main.diagnostics["extended_momentum"],
					np.asarray(dense.diagnostics["extended_momentum"])[..., [0, 2, 4]],
				)

	def test_unobserved_compositions_do_not_construct_events(self) -> None:
		with patch("methods.extended.observations.ABBA2ImplicitIntegrationStep",
			side_effect=AssertionError("Unexpected observer allocation")):
			result = simulate(_problem(), ABBA4Implicit(), _request([0.0, 0.04]))
		self.assertEqual(result.diagnostics["nonlinear_solves_per_step"], 1)
		self.assertEqual(np.asarray(result.diagnostics["substep_nonlinear_iterations"]).shape, (2, 1))

	def test_observer_snapshots_cannot_mutate_state_or_tracked_energy(self) -> None:
		problem = _problem()
		request = _request([0.0, 0.02, 0.04])
		expected = simulate(problem, ABBA4Implicit(track_energy=True), request)

		def mutate_snapshot(event):
			event.state_after[:] = 999.0
			event.state_before[:] = -999.0
			for substep in event.substeps:
				substep.u_first[:] = 888.0
				substep.u_initial[:] = 888.0

		actual = simulate(problem, ABBA4Implicit(track_energy=True, step_observer=mutate_snapshot), request)
		np.testing.assert_array_equal(expected.states, actual.states)
		np.testing.assert_array_equal(
			expected.diagnostics["extended_momentum"], actual.diagnostics["extended_momentum"],
		)


class SharedNewtonTests(unittest.TestCase):
	"""Convergence bookkeeping is independent of a projection formulation."""

	def test_cached_root_counts_one_evaluation_without_a_correction(self) -> None:
		def unexpected(*args):
			raise AssertionError("A cached root must need no new numerical work")
		result = _solve_newton(
			unexpected, np.asarray([2.0]), unexpected,
			tolerance=1e-12, max_iterations=3, context="cached root",
			initial_evaluation=(np.zeros(1), "accepted"),
		)
		self.assertEqual(result.iterations, 0)
		self.assertEqual(result.residual_evaluations, 1)
		self.assertEqual(result.payload, "accepted")

	def test_nonlinear_root_and_failure_are_explicit(self) -> None:
		def residual(x):
			return x * x - 2.0, None
		def update(x, value, payload):
			return x - value / (2.0 * x)
		result = _solve_newton(residual, np.ones(1), update,
			tolerance=1e-12, max_iterations=8, context="square root")
		self.assertAlmostEqual(result.unknown[0], np.sqrt(2.0), places=12)
		self.assertEqual(result.residual_evaluations, result.iterations + 1)
		with self.assertRaisesRegex(RuntimeError, "did not converge.*square root"):
			_solve_newton(residual, np.ones(1), update,
				tolerance=1e-14, max_iterations=1, context="square root")


if __name__ == "__main__":
	unittest.main()
