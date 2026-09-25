"""Contracts for the sixth-order symmetric ABBA composition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from diagnostics import (
	ImplicitABBAReversibilityObserver, abba6_implicit_step_particle_jacobians,
	central_difference_jacobian,
)

from initial_conditions import GCInitialConfiguration
from simulation import (
	ABBA6Implicit,
	ABBA6ImplicitIntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from methods.extended.core.composition import _ABBA6_COEFFICIENTS
from studies import (
	ABBA6AccuracyConfig,
	HighPrecisionReferenceConfig,
	RandomPotentialConfig,
	random_gc_configuration,
	run_abba6_accuracy_study,
	run_high_precision_reference_trajectory,
)

from tests.test_abba4_implicit import _LinearRotationDynamics, _rotation_problem
from tests.test_abba4_single_projection import (
	_NonlinearHamiltonianDynamics, _problem, _dense_particle_blocks,
)


class _SixthDegreeTimeDynamics:
	"""Non-autonomous canonical field ``(x, y)' = (t**6, 0)``."""

	state_dimension = 2

	def vector_field(self, t: float, state: np.ndarray) -> np.ndarray:
		"""Return the degree-six time polynomial in packed layout."""
		particle_count = state.size // 2
		return np.concatenate(
			(
				np.full(particle_count, float(t) ** 6),
				np.zeros(particle_count),
			)
		)

	def particle_vector_field_jacobians(
		self,
		_t: float,
		state: np.ndarray,
	) -> np.ndarray:
		"""Return the zero spatial Jacobian for every particle."""
		return np.zeros((state.size // 2, 2, 2), dtype=float)


class ABBA6MethodTests(unittest.TestCase):
	"""Verify coefficients, sixth order, signed times, and reversibility."""

	def test_nonlinear_non_autonomous_order_for_both_roots_and_solvers(self) -> None:
		problem = _problem(_NonlinearHamiltonianDynamics(),
			x=np.asarray([1.0, 0.7]), y=np.asarray([0.2, -0.3]))
		reference = solve_ivp(problem.dynamics.vector_field, (0.0, 0.8), problem.initial_state,
			method="DOP853", rtol=3e-14, atol=1e-15, max_step=0.002)
		self.assertTrue(reference.success)
		for formulation in ("reduced_multiplier", "simultaneous_state_multiplier"):
			for solver in ("newton", "broyden"):
				with self.subTest(formulation=formulation, solver=solver):
					errors = []
					for h in (0.2, 0.1, 0.05):
						result = simulate(problem, ABBA6Implicit(
							projection_formulation=formulation, nonlinear_solver=solver,
							newton_absolute_tolerance=1e-14, newton_relative_tolerance=1e-14,
							newton_max_iterations=40), SimulationRequest.uniform(
								t_span=(0.0, 0.8), max_step=h, sample_count=2))
						errors.append(np.linalg.norm(result.states[:, -1] - reference.y[:, -1]))
						self.assertEqual(result.diagnostics["nonlinear_solves_per_step"], 1)
						self.assertTrue(np.all(result.diagnostics["nonlinear_residual_norms"]
							<= result.diagnostics["nonlinear_tolerances"]))
					for gain in np.asarray(errors[:-1]) / errors[1:]:
						self.assertGreater(gain, 60.0)
						self.assertLess(gain, 68.0)

	def test_outer_projection_tangent_symplecticity_and_reverse_observer(self) -> None:
		problem = _problem(_NonlinearHamiltonianDynamics(),
			x=np.asarray([1.0, 0.7]), y=np.asarray([0.2, -0.3]))
		request = SimulationRequest.uniform(t_span=(0.3, 0.5), max_step=0.2, sample_count=2)
		for formulation in ("reduced_multiplier", "simultaneous_state_multiplier"):
			for solver in ("newton", "broyden"):
				with self.subTest(formulation=formulation, solver=solver):
					events = []
					simulate(problem, ABBA6Implicit(projection_formulation=formulation,
						nonlinear_solver=solver, newton_absolute_tolerance=1e-14,
						newton_relative_tolerance=1e-14, newton_max_iterations=40,
						step_observer=events.append), request)
					event = events[0]
					blocks = abba6_implicit_step_particle_jacobians(event)
					numeric = central_difference_jacobian(event.map_state, event.state_before)
					np.testing.assert_allclose(_dense_particle_blocks(blocks), numeric,
						rtol=2e-8, atol=2e-9)
					omega = np.asarray([[0., -1.], [1., 0.]])
					for block in blocks:
						np.testing.assert_allclose(block.T @ omega @ block, omega, rtol=0, atol=5e-13)
					observer = ImplicitABBAReversibilityObserver(nonlinear_solver=solver,
						newton_absolute_tolerance=1e-14, newton_relative_tolerance=1e-14,
						newton_max_iterations=40)
					observer(event)
					self.assertLess(observer.samples[0].backward_state_error_norm, 2e-13)
					self.assertLess(observer.samples[0].jacobian_composition_defect_norm, 2e-12)

	def test_intermediate_projection_selector_is_rejected(self) -> None:
		with self.assertRaisesRegex(ValueError, "around_complete_composition"):
			ABBA6Implicit(projection_placement="after_each_abba_map")

	def test_observation_contains_seven_continuous_signed_substeps(self) -> None:
		coefficients = _ABBA6_COEFFICIENTS
		np.testing.assert_array_equal(coefficients, coefficients[::-1])
		self.assertAlmostEqual(float(np.sum(coefficients)), 1.0, places=15)
		self.assertAlmostEqual(float(np.sum(coefficients**3)), 0.0, places=15)
		self.assertAlmostEqual(float(np.sum(coefficients**5)), 0.0, places=15)
		events = []
		solution = simulate(
			_rotation_problem(),
			ABBA6Implicit(
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
				step_observer=events.append,
			),
			SimulationRequest.uniform(
				t_span=(0.0, 0.2),
				max_step=0.1,
				sample_count=3,
			),
		)
		self.assertEqual(len(events), 2)
		step = events[0]
		self.assertIsInstance(step, ABBA6ImplicitIntegrationStep)
		self.assertEqual(len(step.substeps), 7)
		np.testing.assert_allclose(
			[substep.duration for substep in step.substeps],
			coefficients * step.duration,
			rtol=0.0,
			atol=2e-16,
		)
		expected_starts = step.start_time + np.concatenate(
			(
				np.asarray([0.0]),
				np.cumsum(coefficients[:-1]) * step.duration,
			)
		)
		np.testing.assert_allclose(
			[substep.start_time for substep in step.substeps],
			expected_starts,
			rtol=0.0,
			atol=3e-16,
		)
		for first, second in zip(step.substeps, step.substeps[1:]):
			np.testing.assert_array_equal(first.u_final, second.u_initial)
			np.testing.assert_array_equal(first.v_final, second.v_initial)
		np.testing.assert_array_equal(step.substeps[0].u_initial, step.state_before + step.multiplier)
		np.testing.assert_array_equal(step.substeps[0].v_initial, step.state_before - step.multiplier)
		last = step.substeps[-1]
		np.testing.assert_allclose(last.u_final - last.v_final + 2 * step.multiplier, 0., atol=2e-14)
		np.testing.assert_allclose((last.u_final + last.v_final) / 2, step.state_after, atol=2e-14)
		self.assertEqual(solution.diagnostics["unprojected_abba_maps_per_step"], 7)
		self.assertEqual(solution.diagnostics["composition_stage_count"], 14)
		self.assertEqual(solution.diagnostics["projection_placement"], "around_complete_composition")
		self.assertEqual(solution.diagnostics["base_composition"], "unprojected_abba6_yoshida")
		self.assertEqual(solution.diagnostics["nonlinear_solves_per_step"], 1)
		self.assertEqual(
			solution.diagnostics["substep_nonlinear_iterations"].shape,
			(2, 1),
		)

	def test_method_is_sixth_order_and_reversible(self) -> None:
		problem = _rotation_problem()
		exact = np.asarray([np.cos(0.8), np.sin(0.8)])
		errors = []
		for step in (0.2, 0.1, 0.05):
			solution = simulate(
				problem,
				ABBA6Implicit(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 0.8),
					max_step=step,
					sample_count=2,
				),
			)
			errors.append(float(np.linalg.norm(solution.states[:, -1] - exact)))
		for gain in np.asarray(errors[:-1]) / np.asarray(errors[1:]):
			self.assertGreater(float(gain), 60.0)
			self.assertLess(float(gain), 68.0)

		state = np.asarray([1.0, 0.2])
		run = ABBA6Implicit(newton_absolute_tolerance=1e-14, newton_relative_tolerance=1e-14).new_run(
			problem, SimulationRequest.uniform(t_span=(0.3, 0.5), max_step=0.2, sample_count=2))
		forward = run.project(0.3, state, 0.2)
		backward = run.project(0.5, forward.state, -0.2)

		np.testing.assert_allclose(backward.state, state, rtol=0.0, atol=5e-15)

	def test_sixth_order_composition_uses_signed_non_autonomous_times(self) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([0.0]),
			y=np.asarray([0.0]),
		)
		problem = InitialValueProblem(_SixthDegreeTimeDynamics(), configuration)
		errors = []
		for step in (0.25, 0.125, 0.0625):
			solution = simulate(
				problem,
				ABBA6Implicit(
					newton_absolute_tolerance=1e-14,
					newton_relative_tolerance=1e-14,
				),
				SimulationRequest.uniform(
					t_span=(0.0, 1.0),
					max_step=step,
					sample_count=2,
				),
			)
			errors.append(abs(float(solution.states[0, -1]) - 1.0 / 7.0))
		for gain in np.asarray(errors[:-1]) / np.asarray(errors[1:]):
			self.assertGreater(float(gain), 63.9)
			self.assertLess(float(gain), 64.1)

	def test_short_reference_accuracy_study(self) -> None:
		potential_config = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		)
		potential = potential_config.build()
		configuration = random_gc_configuration(
			potential,
			particle_count=2,
			seed=41,
		)
		initial_metadata = {
			"particle_count": 2,
			"seed": 41,
			"sampling": "uniform_full_periodic_cell",
		}
		with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
			root = Path(temporary)
			(root / "pyproject.toml").write_text("[project]\nname='test'\n")
			reference_result = run_high_precision_reference_trajectory(
				potential,
				configuration,
				notebook_path=(
					root / "notebooks/developements/accuracy/reference.ipynb"
				),
				config=HighPrecisionReferenceConfig(
					t_span=(0.0, 0.2),
					save_interval=0.025,
					rho=0.05,
					relative_tolerance=1e-11,
					absolute_tolerance=1e-13,
					maximum_step=0.005,
					audit_relative_tolerance=1e-11,
					audit_absolute_tolerance=1e-13,
					audit_maximum_step=0.0025,
				),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
				project_root=root,
			)
			accuracy = run_abba6_accuracy_study(
				potential,
				configuration,
				reference_result.trajectory,
				config=ABBA6AccuracyConfig(
					integration_steps=(0.1, 0.05),
					t_span=(0.0, 0.2),
					save_interval=0.1,
					rho=0.05,
					absolute_tolerance=1e-14,
					relative_tolerance=1e-14,
				),
				potential_metadata=potential_config.metadata(),
				initial_condition_metadata=initial_metadata,
			)
		self.assertEqual(len(accuracy.summaries()), 2)
		self.assertEqual(len(accuracy.convergence_orders()), 1)
		for summary in accuracy.summaries():
			self.assertEqual(summary.method_name, "ABBA6Implicit")
		for solution in accuracy.solutions.values():
			self.assertEqual(solution.diagnostics["nonlinear_solves_per_step"], 1)
		for values in accuracy.series.values():
			np.testing.assert_array_equal(values.distances[:, 0], 0.0)


if __name__ == "__main__":
	unittest.main()
