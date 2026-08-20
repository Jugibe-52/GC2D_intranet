"""Forward/backward tangent and proposed-increment diagnostics for implicit ABBA."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnostics import ImplicitABBAReversibilityObserver
from initial_conditions import GCInitialConfiguration
from simulation import (
	ABBA4Implicit1,
	ImplicitABBA1,
	ImplicitABBA2,
	InitialValueProblem,
	SimulationRequest,
	simulate,
)
from studies import (
	ImplicitABBAReversibilityStudyConfig,
	RandomPotentialConfig,
	run_implicit_abba_reversibility_study,
)
from tests.test_abba import gc_dynamics
from visualization import (
	plot_implicit_abba_reversibility_diagnostics,
	plot_implicit_abba_transport_components,
)


class ImplicitABBAReversibilityObserverTests(unittest.TestCase):
	"""Verify independently solved reverse tangents and requested vector formulas."""

	def test_both_formulations_record_the_requested_quantities(self) -> None:
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([1.2]),
		)
		problem = InitialValueProblem(gc_dynamics(), configuration)
		request = SimulationRequest.uniform(
			t_span=(0.2, 0.34),
			max_step=0.07,
			sample_count=3,
		)
		solver = {
			"newton_absolute_tolerance": 1e-14,
			"newton_relative_tolerance": 1e-14,
			"newton_max_iterations": 12,
		}
		for method_type in (ImplicitABBA1, ImplicitABBA2, ABBA4Implicit1):
			observer = ImplicitABBAReversibilityObserver(**solver)
			solution = simulate(
				problem,
				method_type(step_observer=observer, **solver),
				request,
			)
			self.assertEqual(len(observer.samples), solution.n_steps)
			for sample in observer.samples:
				identity = np.eye(2)
				np.testing.assert_allclose(
					sample.jacobian_composition_defect,
					sample.backward_jacobian @ sample.forward_jacobian - identity,
				)
				np.testing.assert_allclose(
					sample.forward_action_on_initial_velocity,
					sample.forward_jacobian @ sample.velocity_before,
				)
				np.testing.assert_allclose(
					sample.backward_action_on_final_velocity,
					sample.backward_jacobian @ sample.velocity_after,
				)
				np.testing.assert_allclose(
					sample.endpoint_velocity_action_difference,
					sample.forward_jacobian @ sample.velocity_before
					- sample.backward_jacobian @ sample.velocity_after,
				)
				expected_forward = (
					sample.duration * sample.velocity_before
					+ 0.5
					* sample.duration**2
					* sample.forward_action_on_initial_velocity
				)
				expected_backward = (
					-sample.duration * sample.velocity_after
					+ 0.5
					* sample.duration**2
					* sample.backward_action_on_final_velocity
				)
				np.testing.assert_allclose(sample.forward_increment, expected_forward)
				np.testing.assert_allclose(sample.backward_increment, expected_backward)
				np.testing.assert_allclose(
					sample.increment_closure,
					expected_forward + expected_backward,
				)
				expected_scale = max(
					np.linalg.norm(expected_forward),
					np.linalg.norm(expected_backward),
				)
				self.assertAlmostEqual(
					sample.increment_closure_scale,
					expected_scale,
				)
				self.assertAlmostEqual(
					sample.normalized_increment_closure,
					np.linalg.norm(expected_forward + expected_backward)
					/ expected_scale,
				)
				self.assertLess(sample.backward_state_error_norm, 5e-13)
				self.assertLess(sample.jacobian_composition_defect_norm, 5e-12)

	def test_study_and_plots_run_for_one_particle(self) -> None:
		potential = RandomPotentialConfig(
			amplitude=0.08,
			max_wave_number=3,
			nx=16,
			ny=16,
			seed=27,
			interpolation_order=5,
		).build()
		configuration = GCInitialConfiguration.from_components(
			x=np.asarray([1.0]),
			y=np.asarray([1.2]),
		)
		result = run_implicit_abba_reversibility_study(
			potential,
			configuration,
			config=ImplicitABBAReversibilityStudyConfig(
				formulation="abba4_implicit_1",
				rho=0.05,
				t_span=(0.0, 0.04),
				max_step=0.02,
				sample_count=3,
				newton_absolute_tolerance=1e-14,
				newton_relative_tolerance=1e-14,
			),
		)
		self.assertEqual(len(result.samples), 2)
		figure, axes = plot_implicit_abba_reversibility_diagnostics(result.samples)
		self.assertEqual(axes.shape, (2,))
		figure.canvas.draw()
		plt.close(figure)
		figure, axes = plot_implicit_abba_transport_components(result.samples)
		self.assertEqual(axes.shape, (2,))
		figure.canvas.draw()
		plt.close(figure)


if __name__ == "__main__":
	unittest.main()
