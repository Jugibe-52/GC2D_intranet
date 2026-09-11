"""Behavioral safeguards for the common prepared ABBA runtime."""

from __future__ import annotations

from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from dynamics import GuidingCenterDynamics
from initial_conditions import GCInitialConfiguration
from potential import Potential
from simulation import ABBA2Implicit, ABBA4Implicit, InitialValueProblem, SimulationRequest, simulate
from simulation.methods._nonlinear import _solve_newton
from simulation.methods.abba.preparation import prepare_abba
from simulation.methods.abba.records import ExtendedProjectionTrace, PhysicalProjectionTrace


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
		for extension, (method_type, order, placement, solves, maps) in product(
			("physical", "fully_extended"),
			(
				(ABBA2Implicit, 2, "after_each_abba_map", 1, 1),
				(ABBA4Implicit, 4, "after_each_abba_map", 3, 1),
				(ABBA4Implicit, 4, "around_complete_composition", 1, 3),
			),
		):
			with self.subTest(extension=extension, order=order, placement=placement):
				method = method_type(state_extension=extension)
				prepared = prepare_abba(problem, method, request, order=order, projection_placement=placement)
				state = prepared.state_ops.unpack(0.0, prepared.initial_workspace)
				results = prepared.solve_step(0.0, state, 0.02)
				self.assertIsInstance(results, tuple)
				self.assertEqual(len(results), solves)
				for result in results:
					if isinstance(result.trace, PhysicalProjectionTrace):
						self.assertEqual(len(result.trace.maps), maps)
					else:
						self.assertIsInstance(result.trace, ExtendedProjectionTrace)
						self.assertEqual(len(result.trace.coefficients), maps)
					self.assertLessEqual(result.stats.residual_norm, result.stats.tolerance)
				self.assertFalse(prepared.initial_workspace.flags.writeable)
				with self.assertRaises(TypeError):
					prepared.method_metadata["step_count"] = 3  # type: ignore[index]

	def test_shadow_samples_preserve_main_trajectory_metrics_and_events(self) -> None:
		problem = _problem()
		for extension, solver, placement in product(
			("physical", "fully_extended"), ("newton", "broyden"),
			("after_each_abba_map", "around_complete_composition"),
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
		with patch("simulation.methods.abba.observations.ABBA2ImplicitIntegrationStep",
			side_effect=AssertionError("Unexpected observer allocation")):
			result = simulate(_problem(), ABBA4Implicit(), _request([0.0, 0.04]))
		self.assertEqual(result.diagnostics["nonlinear_solves_per_step"], 3)
		self.assertEqual(np.asarray(result.diagnostics["substep_nonlinear_iterations"]).shape, (2, 3))

	def test_observer_snapshots_cannot_mutate_state_or_tracked_energy(self) -> None:
		problem = _problem()
		request = _request([0.0, 0.02, 0.04])
		expected = simulate(problem, ABBA4Implicit(track_energy=True), request)

		def mutate_snapshot(event):
			event.state_after[:] = 999.0
			event.state_before[:] = -999.0
			for substep in event.substeps:
				substep.u_first[:] = 888.0
				substep.multiplier[:] = 888.0

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
