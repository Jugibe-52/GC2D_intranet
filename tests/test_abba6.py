"""Contracts for the sixth-order symmetric ABBA composition."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from initial_conditions import GCInitialConfiguration
from simulation import (
	ABBA6Implicit,
	ABBA6ImplicitIntegrationStep,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from simulation.methods.abba.order6_implicit import _ABBA6_COEFFICIENTS, _solve_abba6_step
from studies import (
	ABBA6AccuracyConfig,
	HighPrecisionReferenceConfig,
	RandomPotentialConfig,
	random_gc_configuration,
	run_abba6_accuracy_study,
	run_high_precision_reference_trajectory,
)

from tests.test_abba4_implicit import _LinearRotationDynamics, _rotation_problem


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
			np.testing.assert_array_equal(first.state_after, second.state_before)
		np.testing.assert_array_equal(step.substeps[-1].state_after, step.state_after)
		self.assertEqual(solution.diagnostics["nonlinear_solves_per_step"], 7)
		self.assertEqual(
			solution.diagnostics["substep_nonlinear_iterations"].shape,
			(2, 7),
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
		solver = {
			"absolute_tolerance": 1e-14,
			"relative_tolerance": 1e-14,
			"max_iterations": 20,
			"nonlinear_solver": "newton",
		}
		forward = _solve_abba6_step(
			_LinearRotationDynamics(),
			0.3,
			state,
			0.2,
			**solver,
		)
		backward = _solve_abba6_step(
			_LinearRotationDynamics(),
			0.5,
			forward.state,
			-0.2,
			**solver,
		)
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
			self.assertEqual(solution.diagnostics["nonlinear_solves_per_step"], 7)
		for values in accuracy.series.values():
			np.testing.assert_array_equal(values.distances[:, 0], 0.0)


if __name__ == "__main__":
	unittest.main()
